"""
============================================================================
INCREMENTAL REFRESH — Script Server Entry Point
============================================================================
Every 10 minutes:
  1. Check pipeline version — if bumped, trigger a full rebuild automatically
  2. Check if baseline is >24h old — if so, trigger full rebuild
  3. Otherwise run delta extraction, upsert into baseline, re-render

Schedule: Every 10 minutes
============================================================================
"""

import os
import sys
import time
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pytz

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
STATE_DIR    = SCRIPT_DIR / "dashboard_state"
WWW_DIR      = SCRIPT_DIR.parent / "www"
BASELINE     = STATE_DIR / "baseline.parquet"
LAST_SYNC    = STATE_DIR / "last_sync.txt"
VERSION_FILE = STATE_DIR / "pipeline_version.txt"
HTML_OUT     = WWW_DIR / "dna_sc_dashboard.html"

LOG_DIR      = SCRIPT_DIR / "logs"

STATE_DIR.mkdir(parents=True, exist_ok=True)
WWW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

# ── Imports ───────────────────────────────────────────────────────────────────
from dnasc import run_pipeline, PipelineConfig
from dnasc.pipeline import _assign_lsp_roots, _finalize_metadata
from dnasc.renderer import render_dashboard

# ── Pipeline version — always in sync with config.py ─────────────────────────
PIPELINE_VERSION = PipelineConfig.PIPELINE_VERSION

# ── Overlap window: catch late-arriving BQ rows ──────────────────────────────
OVERLAP_MINUTES = 30

# ─────────────────────────────────────────────────────────────────────────────
def needs_full_rebuild() -> tuple[bool, str]:
    """Return (True, reason) if a full rebuild is required."""
    if not BASELINE.exists():
        return True, "No baseline found"

    if not VERSION_FILE.exists():
        return True, "No version file found"

    stored = VERSION_FILE.read_text().strip()
    if stored != PIPELINE_VERSION:
        return True, f"Version mismatch: stored={stored}, current={PIPELINE_VERSION}"

    age = datetime.now() - datetime.fromtimestamp(BASELINE.stat().st_mtime)
    if age > timedelta(hours=24):
        return True, f"Baseline is {age.total_seconds()/3600:.1f}h old (>24h)"

    return False, ""


def get_last_sync() -> datetime:
    """Return the last sync timestamp (UTC), defaulting to 2h ago."""
    if LAST_SYNC.exists():
        try:
            ts = float(LAST_SYNC.read_text().strip())
            return datetime.fromtimestamp(ts, tz=pytz.UTC)  # fixed deprecation warning
        except Exception:
            pass
    return datetime.now(pytz.UTC) - timedelta(hours=2)


def save_last_sync():
    LAST_SYNC.write_text(str(datetime.now(pytz.UTC).timestamp()))


def run_incremental() -> pd.DataFrame | None:
    """
    Pull only rows updated since last_sync - OVERLAP_MINUTES.
    Returns the delta DataFrame, or None if nothing changed.
    """
    since = get_last_sync() - timedelta(minutes=OVERLAP_MINUTES)
    since_str = since.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  📥 Fetching delta since {since_str} UTC...")

    # Override pipeline to incremental mode
    PipelineConfig.INCREMENTAL_MODE = True
    PipelineConfig.DATE_FILTER = since_str

    try:
        delta_df = run_pipeline()
    finally:
        # Always reset back to full mode so next import is clean
        PipelineConfig.INCREMENTAL_MODE = False

    if delta_df is None or delta_df.empty:
        print("  ✅ No changes detected — skipping render")
        return None

    print(f"  ✅ Delta: {len(delta_df):,} rows")
    return delta_df


def upsert_baseline(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Merge delta into baseline, preferring new rows on workorder_id.

    For OpTracker-sourced columns (queue data), if the delta row has no ops
    but the baseline row had ops, restore the baseline ops.  The incremental
    pipeline only fetches recent OpTracker events, so queue data for workorders
    touched in BIOS (status update) but not in OpTracker (no new ops) would
    otherwise be silently erased.
    """
    import numpy as np

    # Columns populated entirely from OpTracker — never overwrite with None/empty
    OPTRACKER_COLS = [
        "protocol_name", "operation_state", "operation_start",
        "operation_ready", "job_id", "well_location",
        "op_batch_id", "lsp_batch_id_from_optracker",
    ]

    def _is_empty(v) -> bool:
        if v is None:
            return True
        if isinstance(v, float) and pd.isna(v):
            return True
        if isinstance(v, (list, np.ndarray)) and len(v) == 0:
            return True
        return False

    baseline_df = pd.read_parquet(BASELINE)
    print(f"  📦 Baseline: {len(baseline_df):,} rows  +  delta: {len(delta_df):,} rows")

    # Standard upsert: delta wins
    combined = pd.concat([baseline_df, delta_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["workorder_id"], keep="last")

    # For workorders in both baseline and delta, restore OpTracker columns
    # where delta has None/empty but baseline had data.
    overlap_ids = set(baseline_df["workorder_id"]) & set(delta_df["workorder_id"])
    if overlap_ids:
        ot_cols_present = [c for c in OPTRACKER_COLS if c in baseline_df.columns and c in combined.columns]
        if ot_cols_present:
            base_ot = (
                baseline_df[baseline_df["workorder_id"].isin(overlap_ids)]
                [["workorder_id"] + ot_cols_present]
                .set_index("workorder_id")
            )
            overlap_mask = combined["workorder_id"].isin(overlap_ids)
            for col in ot_cols_present:
                empty_in_delta = combined[col].apply(_is_empty)
                needs_fill = overlap_mask & empty_in_delta
                if needs_fill.any():
                    fill_vals = combined.loc[needs_fill, "workorder_id"].map(base_ot[col])
                    combined.loc[needs_fill, col] = fill_vals

    print(f"  ✅ Merged: {len(combined):,} rows")
    return combined


def fix_mixed_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix columns where numpy.datetime64 scalars are mixed with list-of-timestamps.
    operation_start / operation_ready store lists per row — bare scalars from
    the delta need to be wrapped in a list so the column is uniformly typed.
    """
    import pyarrow as pa
    import numpy as np

    # Columns that should always contain lists of timestamps
    LIST_TIMESTAMP_COLS = {'operation_start', 'operation_ready'}

    def _norm_ts(item):
        """Normalize a single timestamp element to UTC-aware pd.Timestamp."""
        if item is None:
            return None
        try:
            if isinstance(item, float) and pd.isna(item):
                return None
        except Exception:
            pass
        if isinstance(item, pd.Timestamp):
            return item if item.tzinfo else item.tz_localize('UTC')
        try:
            ts = pd.Timestamp(item)
            return ts if ts.tzinfo else ts.tz_localize('UTC')
        except Exception:
            return None

    def normalize_list_col(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return []
        if isinstance(v, (np.ndarray, list)):
            return [_norm_ts(x) for x in v]
        if isinstance(v, (np.datetime64, pd.Timestamp)):
            return [_norm_ts(v)]
        return []

    def normalize_scalar(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NaT
        if isinstance(v, list):
            return v
        if isinstance(v, np.datetime64):
            return pd.Timestamp(v, tz='UTC')
        return v

    # Always normalize list-timestamp cols — pyarrow detection is too permissive
    for col in LIST_TIMESTAMP_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(normalize_list_col)

    for col in df.columns:
        if col in LIST_TIMESTAMP_COLS or df[col].dtype != object:
            continue
        try:
            pa.Array.from_pandas(df[col])
        except pa.lib.ArrowInvalid:
            print(f"  ⚠️  Fixing column: {col}")
            df[col] = df[col].map(normalize_scalar)

    return df


# ─────────────────────────────────────────────────────────────────────────────
def main():
    start = time.time()
    print("=" * 70)
    print(f"⚡ INCREMENTAL REFRESH  |  version={PIPELINE_VERSION}  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    rebuild, reason = needs_full_rebuild()

    if rebuild:
        print(f"\n🔄 Full rebuild triggered: {reason}")
        print("   → Delegating to full_refresh.py logic...\n")
        from full_refresh import main as full_main
        full_main()
        return

    # ── Incremental path ──
    print("\n⏱  Running incremental extraction...")
    delta_df = run_incremental()

    if delta_df is None:
        save_last_sync()
        return

    # Upsert into baseline
    final_df = upsert_baseline(delta_df)

    # Re-run root assignment on the full merged df — incremental delta only
    # contains new/updated rows, so _assign_lsp_roots can't resolve source
    # transformations that are already in the baseline. Running it here fixes
    # LSP root pointers and the strain fill in _finalize_metadata.
    print("\n🔧 Re-running root assignment and metadata fill on merged baseline...")
    final_df = _assign_lsp_roots(final_df)
    final_df = _finalize_metadata(final_df)

    # Fix mixed-type columns before saving
    print("\n🔧 Fixing column types for parquet serialization...")
    final_df = fix_mixed_type_columns(final_df)

    # Save updated baseline
    final_df.to_parquet(BASELINE, index=False)
    print(f"  💾 Baseline updated")

    # Save sync timestamp immediately after parquet — before render
    # so a render crash doesn't freeze the sync window
    save_last_sync()

    # Re-render
    print("\n🎨 Re-rendering dashboard...")
    html = render_dashboard(final_df)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"  ✅ Dashboard written → {HTML_OUT}")

    elapsed = time.time() - start
    print(f"\n🎉 Incremental refresh complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
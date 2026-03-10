"""
dnasc/pipeline.py
──────────────────
Top-level pipeline orchestrator.
Coordinates all extractors and transformers in the correct order.
This is the only file that knows the full execution sequence.
"""

from __future__ import annotations
import time
import concurrent.futures

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger
from dnasc.extractors import (
    BIOSExtractor,
    LSPExtractor,
    LIMSExtractor,
    OpTrackerExtractor,
)
from dnasc.transformers import (
    LineageTransformer,
    ProcessingTransformer,
    RepairTransformer,
    ValidationTransformer,
)
from dnasc.transformers.repair import (
    populate_synthetic_optracker_batch,
    resolve_downstream_plates,
    resolve_lims_streakouts,
    resolve_optracker_streakouts,
)

log = get_logger(__name__)


def run_pipeline() -> pd.DataFrame:
    """
    Execute the full DNA SC data pipeline.

    Steps
    ─────
    1.  Parallel extraction  (BIOS, LSP workorders, LSP aliquots, OpTracker)
    2.  LSP merge & orphan recovery
    3.  Lineage bridging
    4.  Synthetic streakout creation
    5.  Core processing (JSON parsing, status enrichment, yield calc)
    6.  LSP root assignment
    7.  OpTracker aggregation
    8.  LIMS colony extraction
    9.  Final merges
    10. Synthetic OpTracker population
    11. Root repair & metadata backfill
    12. Smart filtering & UI enrichment

    Returns
    ───────
    pd.DataFrame  — fully enriched, render-ready dataset
    """
    pipeline_start = time.time()
    log.info("=" * 70)
    log.info("PIPELINE START  version=%s", PipelineConfig.PIPELINE_VERSION)
    log.info("=" * 70)

    # ── STEP 1: Parallel extraction ───────────────────────────────────────────
    log.info("STEP 1 — Parallel extraction")
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        f_bios  = pool.submit(BIOSExtractor.get_bios_workorders)
        f_lsp   = pool.submit(LSPExtractor.get_lsp_workorders)
        f_aliq  = pool.submit(LSPExtractor.get_lsp_aliquots)
        f_op    = pool.submit(OpTrackerExtractor.get_optracker_operations)

        bios_df      = f_bios.result()
        lsp_df       = f_lsp.result()
        aliq_df      = f_aliq.result()
        optracker_raw = f_op.result()
    log.info("Extraction complete in %.2fs", time.time() - t)

    # ── STEP 2: LSP merge & orphan recovery ───────────────────────────────────
    log.info("STEP 2 — LSP merge & orphan recovery")
    t = time.time()
    lsp_full = _merge_lsp(lsp_df, aliq_df)
    log.info("LSP merge complete in %.2fs", time.time() - t)

    # ── STEP 3–5: Lineage, synthetics, processing ─────────────────────────────
    log.info("STEP 3 — Lineage bridging")
    workorder_data = LineageTransformer.bridge_lsp_lineage(bios_df, lsp_full)

    log.info("STEP 4 — Synthetic streakout creation")
    workorder_data = RepairTransformer.create_synthetic_streakouts(workorder_data)

    log.info("STEP 5 — Core processing")
    processed = ProcessingTransformer.process_workorder_data(workorder_data)

    # ── STEP 6: LSP root assignment ───────────────────────────────────────────
    log.info("STEP 6 — LSP root assignment")
    processed = _assign_lsp_roots(processed)

    # ── STEP 7: OpTracker aggregation ─────────────────────────────────────────
    log.info("STEP 7 — OpTracker aggregation")
    t = time.time()
    optracker_raw["process_id"] = (
        optracker_raw["process_id"].astype(str).str.strip('"').str.lower()
    )
    op_agg = (
        optracker_raw.groupby("process_id")
        .agg({
            "protocol_name":    list,
            "operation_state":  list,
            "operation_start":  list,
            "operation_ready":  list,
            "job_id":           list,
            "well_location":    list,
        })
        .reset_index()
    )

    # Secondary aggregation keyed by LSP batch ID (e.g. "LSP-8403").
    # Synthetic LSP rows have no BIOS workorder UUID so they never match
    # op_agg; but OpTracker operations carry an "LSP Batch" parameter that
    # maps directly to the batch ID — use that as a fallback.
    _lsp_op_rows = optracker_raw[optracker_raw["lsp_batch_id_from_optracker"].notna()].copy()
    _lsp_op_rows["_lsp_key"] = _lsp_op_rows["lsp_batch_id_from_optracker"].str.strip().str.upper()
    lsp_op_agg = (
        _lsp_op_rows.groupby("_lsp_key")
        .agg({
            "protocol_name":    list,
            "operation_state":  list,
            "operation_start":  list,
            "operation_ready":  list,
            "job_id":           list,
            "well_location":    list,
        })
        .reset_index()
        .rename(columns={"_lsp_key": "lsp_batch_key"})
    )
    log.info("OpTracker aggregated in %.2fs", time.time() - t)

    # ── STEP 7b: LIMS streakout resolution ────────────────────────────────────
    # Must run before Step 8 so colony extraction includes the new synthetic rows.
    log.info("STEP 7b — LIMS streakout resolution")
    t = time.time()
    processed = resolve_lims_streakouts(processed)
    log.info("LIMS streakout resolution complete in %.2fs", time.time() - t)

    # ── STEP 8: LIMS colony extraction ────────────────────────────────────────
    log.info("STEP 8 — LIMS colony extraction")
    t = time.time()
    colony_data = LIMSExtractor.get_colony_data(processed["workorder_id"].unique().tolist())
    log.info("Colony data extracted in %.2fs", time.time() - t)

    # ── STEP 9: Final merges ──────────────────────────────────────────────────
    log.info("STEP 9 — Final merges")
    t = time.time()
    final_df = processed.merge(colony_data, on="workorder_id", how="left")

    def _optracker_key(row):
        return str(row.get("workorder_id", "")).replace("STBL3_", "").replace("stbl3_", "").lower()

    final_df["join_key"] = final_df.apply(_optracker_key, axis=1)
    final_df = final_df.merge(
        op_agg,
        left_on="join_key", right_on="process_id",
        how="left", suffixes=("", "_raw_op"),
    )
    # For LSP rows that got no queue data from the primary merge,
    # fill from the lsp_op_agg keyed by lsp_batch_id_from_optracker.
    # Pass 1: synthetic LSP rows (workorder_id = "LSP-XXXX")
    # Pass 2: real LSP workorders (UUID workorder_id) using bios_batch_id
    _op_cols = ["protocol_name", "operation_state", "operation_start", "operation_ready", "job_id", "well_location"]
    _no_ops = ~final_df["protocol_name"].apply(lambda x: isinstance(x, list) and len(x) > 0)

    _syn_lsp_empty = (
        final_df["workorder_id"].astype(str).str.upper().str.startswith("LSP-") & _no_ops
    )
    if _syn_lsp_empty.any() and not lsp_op_agg.empty:
        _fill_idx  = final_df[_syn_lsp_empty].index
        _fill_keys = final_df.loc[_fill_idx, "workorder_id"].astype(str).str.upper()
        for _col in _op_cols:
            _col_map = lsp_op_agg.set_index("lsp_batch_key")[_col].to_dict()
            final_df.loc[_fill_idx, _col] = _fill_keys.map(_col_map)
        log.info(
            "Filled OpTracker queue data for %d synthetic LSPs via lsp_batch_id",
            _syn_lsp_empty.sum(),
        )

    # Pass 2: real LSP workorders (UUID) with bios_batch_id set but still no queue data
    if "bios_batch_id" in final_df.columns and not lsp_op_agg.empty:
        _no_ops2 = ~final_df["protocol_name"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        _real_lsp_empty = (
            (final_df["type"] == "lsp_workorder") &
            ~final_df["workorder_id"].astype(str).str.upper().str.startswith("LSP-") &
            final_df["bios_batch_id"].notna() &
            _no_ops2
        )
        if _real_lsp_empty.any():
            _fill_idx2  = final_df[_real_lsp_empty].index
            _fill_keys2 = final_df.loc[_fill_idx2, "bios_batch_id"].astype(str).str.upper()
            for _col in _op_cols:
                _col_map = lsp_op_agg.set_index("lsp_batch_key")[_col].to_dict()
                final_df.loc[_fill_idx2, _col] = _fill_keys2.map(_col_map)
            log.info(
                "Filled OpTracker queue data for %d real LSPs via bios_batch_id",
                _real_lsp_empty.sum(),
            )
    log.info("Final merges complete in %.2fs", time.time() - t)

    # Deduplicate columns introduced by the Step 9 merge before any concat steps
    if final_df.columns.duplicated().any():
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # ── STEP 9b: Resolve OpTracker-only streakouts ────────────────────────────
    log.info("STEP 9b — OpTracker streakout resolution")
    t = time.time()
    final_df = resolve_optracker_streakouts(final_df, optracker_raw)
    log.info("OpTracker streakout resolution complete in %.2fs", time.time() - t)

    # ── STEP 10: Synthetic OpTracker population ───────────────────────────────
    log.info("STEP 10 — Synthetic OpTracker population")
    t = time.time()
    final_df = populate_synthetic_optracker_batch(final_df)
    log.info("Synthetic OpTracker populated in %.2fs", time.time() - t)

    # ── STEP 10b: Resolve downstream plates (Rearray/Quant/NGS) ──────────────  ← ADD HERE
    log.info("STEP 10b — Downstream plate resolution")
    t = time.time()
    final_df = resolve_downstream_plates(final_df)                              
    log.info("Downstream plate resolution complete in %.2fs", time.time() - t) 

    # ── STEP 11: Root repair & metadata backfill ──────────────────────────────
    log.info("STEP 11 — Root repair & metadata backfill")
    t = time.time()
    final_df = RepairTransformer.repair_data(final_df)
    # Re-run LSP root assignment after repair — Step 6 ran before Step 11
    # resolved STREAK synthetic row roots, so legacy LSPs sourced via
    # lsp_process_id=STREAK_well* were left self-rooted. Now that repair has
    # filled STREAK roots, re-resolve so those LSPs inherit the correct
    # root_work_order_id, experiment_name, and cloning_strain.
    final_df = _assign_lsp_roots(final_df)
    final_df = _finalize_metadata(final_df)
    log.info("Repair complete in %.2fs", time.time() - t)

    # ── STEP 12: Smart filtering & UI enrichment ──────────────────────────────
    log.info("STEP 12 — Smart filtering & UI enrichment")
    t = time.time()
    final_df = _filter_and_enrich(final_df)
    log.info("Filtering complete in %.2fs", time.time() - t)

    # ── Dedup columns from merges ─────────────────────────────────────────────
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # ── Sanitize any cells that were corrupted by duplicate-column assignments ─
    # When a df has duplicate column names and a groupby/fillna is applied, some
    # cells may end up containing pd.Series objects instead of scalar values.
    # Extract the first non-null value from any such cells before serialization.
    import pandas as _pd
    for _col in final_df.select_dtypes(include="object").columns:
        _has_series = final_df[_col].map(lambda x: isinstance(x, (_pd.Series, _pd.DataFrame))).any()
        if _has_series:
            log.warning("Sanitizing Series-valued cells in column %s", _col)
            final_df[_col] = final_df[_col].map(
                lambda x: (x.dropna().iloc[0] if not x.dropna().empty else None)
                if isinstance(x, (_pd.Series, _pd.DataFrame)) else x
            )

    elapsed = time.time() - pipeline_start
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE  %.1fs  |  %d rows  |  %d experiments  |  %d requests",
             elapsed, len(final_df),
             final_df["experiment_name"].nunique(),
             final_df["req_id"].nunique())
    log.info("=" * 70)
    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers  (logic extracted from the old monolith run_pipeline())
# ─────────────────────────────────────────────────────────────────────────────

def _merge_lsp(lsp_df: pd.DataFrame, aliq_df: pd.DataFrame) -> pd.DataFrame:
    """Primary + secondary LSP merge with orphan recovery."""
    import re
    import pandas as pd

    cutoff = pd.Timestamp(PipelineConfig.LSP_CUTOFF_DATE, tz="UTC")
    aliq_df_copy = aliq_df.copy()
    aliq_df_copy["batch_created_at"] = pd.to_datetime(
        aliq_df_copy["batch_created_at"], errors="coerce"
    )

    lsp_full = lsp_df.merge(
        aliq_df_copy, how="outer",
        left_on="workorder_id", right_on="lsp_process_id",
        indicator=True,
    )
    lsp_full["batch_created_at"] = pd.to_datetime(lsp_full["batch_created_at"], errors="coerce")

    # Secondary pass — identity recovery for unlinked batches that map to a known BIOS workorder.
    # Covers: (a) post-cutoff batches where process_id mismatch prevents primary join,
    #         (b) pre-cutoff batches that still have a BIOS workorder (e.g. CANCELED/FAILED).
    batch_lookup = (
        lsp_df.dropna(subset=["bios_batch_id"])
        .set_index("bios_batch_id")["workorder_id"]
        .to_dict()
    )
    right_only_mask = lsp_full["_merge"] == "right_only"
    failed_mask = right_only_mask & (
        (lsp_full["batch_created_at"] >= cutoff) |
        lsp_full["lsp_batch_id"].isin(set(batch_lookup.keys()))
    )
    if failed_mask.any():
        log.info("Identity recovery for %d unlinked LSP rows", failed_mask.sum())
        lsp_full.loc[failed_mask, "workorder_id"] = (
            lsp_full.loc[failed_mask, "lsp_batch_id"].map(batch_lookup)
        )
        lsp_full.loc[failed_mask, "data_source"] = "LSP"
        lsp_full.loc[failed_mask, "type"] = "lsp_workorder"
        success_mask = failed_mask & lsp_full["workorder_id"].notna()
        lsp_full.loc[success_mask, "_merge"] = "both"

        # Copy wo_status from the matched lsp_df row so that CANCELED workorders
        # don't appear as IN_PROGRESS on the recovered aliquot row.
        # (The right_only row has wo_status=NaN; without this copy it bypasses
        # the canceled_no_work filter downstream.)
        for _status_col in ["wo_status", "request_id"]:
            if _status_col in lsp_df.columns:
                _col_lookup = (
                    lsp_df.dropna(subset=["bios_batch_id"])
                    .drop_duplicates("bios_batch_id")
                    .set_index("bios_batch_id")[_status_col]
                    .to_dict()
                )
                lsp_full.loc[success_mask, _status_col] = (
                    lsp_full.loc[success_mask, "lsp_batch_id"].map(_col_lookup)
                )

    # Recombine
    matched    = lsp_full[lsp_full["_merge"] == "both"].copy()
    pre_aliq   = lsp_full[lsp_full["_merge"] == "left_only"].copy()
    orphaned   = lsp_full[
        (lsp_full["_merge"] == "right_only") &
        (lsp_full["batch_created_at"] < cutoff)
    ].copy()

    if not orphaned.empty:
        orphaned["workorder_id"] = orphaned["lsp_batch_id"]
        orphaned["type"]         = "lsp_workorder"
        orphaned["data_source"]  = "SYNTHETIC_LSP"
        orphaned["wo_status"]    = "UNKNOWN"
        orphaned["wo_created_at"]= orphaned["batch_created_at"]
        orphaned["root_work_order_id"] = None

        # bios_batch_id metadata snap
        bios_lookup = (
            lsp_df.dropna(subset=["bios_batch_id"])
            .drop_duplicates("bios_batch_id")
            .set_index("bios_batch_id")
        )
        meta_cols = [
            "workorder_id", "wo_status", "req_id", "request_status",
            "request_created_at", "priority", "submitter_email",
            "construct_name", "delivery_format", "for_partner",
            "experiment_name", "root_work_order_id",
            "source_lsp_process_id", "source_workorder_id",
        ]
        for col in meta_cols:
            if col in bios_lookup.columns and col in orphaned.columns:
                orphaned[col] = orphaned[col].fillna(
                    orphaned["lsp_batch_id"].map(bios_lookup[col])
                )

        recovered = orphaned["workorder_id"].notna() & (orphaned["data_source"] == "SYNTHETIC_LSP")
        orphaned.loc[recovered, "data_source"] = "LSP"
        orphaned.loc[recovered, "type"]        = "lsp_workorder"

        # Strip SCALEUP_ prefix
        def _clean_scaleup(val):
            if pd.isna(val) or str(val) == "nan":
                return val
            cleaned = re.sub(r"^SCALEUP_", "", str(val), flags=re.IGNORECASE)
            uuid_re = r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
            return cleaned if re.match(uuid_re, cleaned, re.IGNORECASE) else val

        if "lsp_process_id" in orphaned.columns:
            cleaned = orphaned["lsp_process_id"].apply(_clean_scaleup)
            changed = cleaned != orphaned["lsp_process_id"]
            orphaned.loc[changed, "source_lsp_process_id"] = cleaned[changed]
            log.debug("Stripped SCALEUP_ prefix from %d orphaned LSPs", changed.sum())

    result = pd.concat([matched, pre_aliq, orphaned], ignore_index=True)
    result = result.drop("_merge", axis=1, errors="ignore")

    # Consolidate duplicate workorder_ids produced by identity recovery.
    # Identity recovery converts right_only aliq rows (yield data, null BIOS cols)
    # to "matched", but the original left_only pre_aliq row (BIOS cols, null yield)
    # still exists.  Sort so rows with lsp_batch_id (yield-bearing) come first,
    # then groupby.first() coalesces: yield from matched + BIOS from pre_aliq.
    if result["workorder_id"].duplicated().any():
        n_before = len(result)
        result = (
            result
            .sort_values("lsp_batch_id", ascending=True, na_position="last")
            .groupby("workorder_id", sort=False, as_index=False)
            .first()
        )
        log.info("Consolidated %d duplicate LSP rows", n_before - len(result))

    log.info("LSP merge result: %d rows", len(result))
    return result


def _assign_lsp_roots(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve root_work_order_id for LSP workorders via source link columns."""
    import pandas as pd

    id_to_root = (
        df.dropna(subset=["root_work_order_id"])
        .set_index("workorder_id")["root_work_order_id"]
        .astype(str)
        .to_dict()
    )

    def _resolve(row):
        if row["type"] != "lsp_workorder":
            return row["root_work_order_id"]
        for col in ["source_lsp_process_id", "lsp_process_id", "middle_root", "source_workorder_id"]:
            val = row.get(col)
            if pd.isna(val):
                continue
            val = str(val).strip()
            if val in ("nan", "None", "", str(row["workorder_id"])):
                continue
            if val.upper().startswith("LSP-"):
                continue
            if val in id_to_root:
                root = id_to_root[val]
                if pd.notna(root) and str(root) not in ("nan", "None", ""):
                    return root
            if len(val) > 20:
                return val
        return row.get("root_work_order_id") or row["workorder_id"]

    df["root_work_order_id"] = df.apply(_resolve, axis=1)

    # Backfill source_lsp_process_id for orphans
    lsp_mask     = df["type"] == "lsp_workorder"
    source_empty = lsp_mask & df["source_lsp_process_id"].isna()
    for col in ["lsp_process_id", "source_workorder_id", "middle_root"]:
        if col in df.columns:
            still = source_empty & df["source_lsp_process_id"].isna()
            df.loc[still, "source_lsp_process_id"] = df.loc[still, col]

    self_ref = df["source_lsp_process_id"].astype(str) == df["workorder_id"].astype(str)
    df.loc[self_ref, "source_lsp_process_id"] = None
    log.debug("LSP root assignment complete")
    return df


def _finalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Final metadata gap-fill after RepairTransformer."""
    # Temporal disqualifier: if a workorder was created BEFORE its assigned
    # request, the link is spurious (e.g. an LSP batch from wave5 reused by a
    # wave6 request that didn't exist yet).  Nullify req_id/experiment_name so
    # the root-based fill below can assign the correct experiment instead.
    if "wo_created_at" in df.columns and "request_created_at" in df.columns:
        wo_ts  = pd.to_datetime(df["wo_created_at"],       utc=True, errors="coerce")
        req_ts = pd.to_datetime(df["request_created_at"],  utc=True, errors="coerce")
        # Restrict to LSP workorders only — Gibson/Transformation rows are always
        # created FOR their request, not reused across experiments like LSP batches.
        temporal_mismatch = (
            wo_ts.notna() & req_ts.notna() & (wo_ts < req_ts) &
            (df["type"] == "lsp_workorder")
        )
        df.loc[temporal_mismatch, "req_id"]          = None
        df.loc[temporal_mismatch, "experiment_name"] = None

    # For req_id and experiment_name, fill only from the root row
    # (workorder_id == root_work_order_id), not from any sibling in the group.
    # Using transform("first") can pull a sibling LSP's req_id from a
    # different experiment when multiple requests share the same root
    # (e.g. a wave5 and wave6 LSP both sourced from the same Gibson clone).
    root_rows = df[df["workorder_id"] == df["root_work_order_id"]]
    for col in ["req_id", "experiment_name"]:
        if col in df.columns:
            root_map = root_rows.set_index("root_work_order_id")[col].to_dict()
            df[col] = df[col].fillna(df["root_work_order_id"].map(root_map))

    # SYNTHETIC_LSP orphans have no real request of their own — always force
    # their req_id/experiment_name from the root, overriding whatever bios_lookup
    # may have set (e.g. a wave5 orphan incorrectly inheriting a wave6 req_id).
    syn_mask = df["data_source"] == "SYNTHETIC_LSP"
    if syn_mask.any():
        for col in ["req_id", "experiment_name"]:
            if col in df.columns:
                df.loc[syn_mask, col] = df.loc[syn_mask, "root_work_order_id"].map(root_map)

    # Other metadata cols are safe to fill from any group member
    cols = ["request_status", "priority", "construct_name", "for_partner"]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(
                df.groupby("root_work_order_id")[col].transform("first")
            )
    orphan_mask = (df["data_source"] == "SYNTHETIC_LSP") & df["req_id"].isna()
    df.loc[orphan_mask, "req_id"]         = "ORPHAN_LEGACY"
    df.loc[orphan_mask, "request_status"] = "SUCCEEDED"

    wip_mask = (df["data_source"] == "LSP") & df["req_id"].isna()
    df.loc[wip_mask, "req_id"]         = "ACTIVE_WIP"
    df.loc[wip_mask, "request_status"] = "IN_PROGRESS"

    # Recompute STOCK_ID from final root assignments.
    # root_STOCK_ID is computed in Step 5 (processing.py) before Step 6
    # (_assign_lsp_roots) corrects LSP root pointers, so it may be stale.
    #
    # For LSP rows, fill STOCK_ID from plasmid_id FIRST (the aliquot batch's
    # specific plasmid_id from LIMS).  This prevents a sibling LSP's STOCK_ID
    # from polluting other rows when the group-fill below runs — e.g. two LSPs
    # for the same request sharing root_work_order_id, one with pAI-X and one
    # with a null STOCK_ID that incorrectly inherits pAI-X.
    if "STOCK_ID" in df.columns and "plasmid_id" in df.columns:
        lsp_mask = df["type"] == "lsp_workorder"
        df.loc[lsp_mask, "STOCK_ID"] = df.loc[lsp_mask, "STOCK_ID"].fillna(
            df.loc[lsp_mask, "plasmid_id"]
        )

    # Propagate STOCK_ID from the root workorder down to all family members.
    if "STOCK_ID" in df.columns and "root_work_order_id" in df.columns:
        df["STOCK_ID"] = df["STOCK_ID"].fillna(
            df.groupby("root_work_order_id")["STOCK_ID"].transform("first")
        )

    # Copy cloning_strain from source transformation to LSP rows where null.
    # LSPs sourced from internal transformations don't inherit strain via the
    # root group (they're in a different assembly group), so fill directly.
    if "cloning_strain" in df.columns and "source_lsp_process_id" in df.columns:
        src_strain = (
            df.dropna(subset=["cloning_strain"])
            .set_index("workorder_id")["cloning_strain"]
            .to_dict()
        )
        lsp_null = (df["type"] == "lsp_workorder") & df["cloning_strain"].isna()
        df.loc[lsp_null, "cloning_strain"] = df.loc[lsp_null, "source_lsp_process_id"].map(src_strain)

    # Backfill source_lsp_process_id and lsp_input_well from sibling LSP rows
    # sharing the same input_well_id. Legacy workorders (LSP-XXXX format) predate
    # BIOS and have these fields null even when newer workorders for the same
    # source well have them populated.
    if "input_well_id" in df.columns and "source_lsp_process_id" in df.columns:
        lsp_well_mask = (df["type"] == "lsp_workorder") & df["input_well_id"].notna()
        if lsp_well_mask.any():
            for col in ["source_lsp_process_id", "lsp_input_well"]:
                if col not in df.columns:
                    continue
                filled = (
                    df[lsp_well_mask & df[col].notna()]
                    .drop_duplicates(subset=["input_well_id"])
                    .set_index("input_well_id")[col]
                )
                null_mask = lsp_well_mask & (df[col].isna() | (df[col].astype(str) == "None"))
                if null_mask.any():
                    df.loc[null_mask, col] = df.loc[null_mask, "input_well_id"].map(filled)

    return df


def _filter_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Smart filtering, status bridging, and UI enrichment."""
    import pandas as pd

    blacklist = set(PipelineConfig.LSP_BLACKLIST)
    active_statuses = {
        "NEW", "PLANNED", "IN_PROGRESS", "ACTIVE_WIP",
        "ORPHAN_LEGACY", "SUCCEEDED", "FULFILLED",
    }

    def _keep(row):
        if str(row.get("lsp_batch_id", "")) in blacklist:
            return False
        req_id = str(row.get("req_id", ""))
        status = str(row.get("request_status", ""))
        if pd.notna(row.get("req_id")) and (
            status in active_statuses or "ORPHAN" in req_id or "ACTIVE" in req_id
        ):
            return True
        if str(row.get("lsp_batch_id", "")).startswith("LSP-10"):
            return True
        if float(row.get("total_volume_ul", 0) or 0) > 1.0:
            return True
        if isinstance(row.get("protocol_name"), list) and len(row["protocol_name"]) > 0:
            return True
        return False

    df = df[df.apply(_keep, axis=1)].copy()

    # Remove LSP rows whose location is a test/fake placeholder
    if "location" in df.columns:
        _test_loc = df["location"].astype(str).str.upper()
        _fake_mask = (
            (df["type"] == "lsp_workorder") &
            _test_loc.str.contains(r"\bTEST\b|\bFAKE\b|TESTFAKE", na=False)
        )
        if _fake_mask.any():
            log.info("Removing %d LSP rows with test/fake locations", _fake_mask.sum())
            df = df[~_fake_mask].copy()

    # Remove canceled LSPs with no OpTracker work
    canceled_no_work = (
        (df["type"] == "lsp_workorder") &
        (df["wo_status"].astype(str).str.upper() == "CANCELED") &
        (~df["protocol_name"].apply(lambda x: isinstance(x, list) and len(x) > 0))
    )
    df = df[~canceled_no_work].copy()
    log.info("Removed %d canceled LSPs with no OpTracker work", canceled_no_work.sum())

    # Lineage glue
    df["is_visible"] = True
    if "parent_id" not in df.columns:
        df["parent_id"] = None
    df["parent_id"] = df["parent_id"].fillna(df["source_lsp_process_id"])
    df["parent_id"] = df["parent_id"].fillna(df["root_work_order_id"])
    df.loc[df["workorder_id"] == df["parent_id"], "parent_id"] = None

    all_parents = set(df["root_work_order_id"].dropna().unique())
    df["is_leaf"] = (df["type"] == "lsp_workorder") | (~df["workorder_id"].isin(all_parents))

    # Dedup columns
    df = df.loc[:, ~df.columns.duplicated()]

    # STOCK_ID fallback
    if "STOCK_ID" in df.columns and "root_STOCK_ID" in df.columns:
        df["STOCK_ID"] = df["STOCK_ID"].fillna(df["root_STOCK_ID"])

    # Status bridge
    def _bridge_status(row):
        # CANCELED wo_status always wins — ops may have FA states but the
        # workorder was explicitly canceled, so don't let ops override it.
        wo = str(row.get("wo_status", "")).strip().upper()
        if wo == "CANCELED":
            return "CANCELED"
        states = row.get("operation_state")
        if isinstance(states, list) and states:
            # Active states take priority regardless of list order.
            # An LSP Order in READY beats a prior Glycerol Stocking SC.
            if "RU" in states: return "RUNNING"
            if "RD" in states: return "READY"
            for s in reversed(states):
                if s == "SC": return "SUCCEEDED"
                if s == "FA": return "FAILED"
        wo = str(row.get("wo_status", "")).strip().upper()
        if wo and wo not in ("NAN", "NONE", "", "UNKNOWN"):
            return wo
        # Fall back to request_status before assuming IN_PROGRESS.
        # BIOS workorders linked to a canceled LSP request often have
        # wo_status=NULL (never updated in BIOS) but request_status=CANCELED.
        req = str(row.get("request_status", "")).strip().upper()
        if req and req not in ("NAN", "NONE", "", "UNKNOWN"):
            return req
        if row.get("data_source") == "SYNTHETIC_LSP":
            return "UNKNOWN"
        return "IN_PROGRESS"

    df["visual_status"] = df.apply(_bridge_status, axis=1)

    nan_mask = df["wo_status"].isna() | df["wo_status"].astype(str).str.upper().isin(["NAN", "NONE", ""])
    df.loc[nan_mask, "wo_status"] = df.loc[nan_mask, "visual_status"]

    log.info("Filtering & enrichment complete: %d rows ready for render", len(df))
    return df

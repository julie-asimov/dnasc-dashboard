"""
============================================================================
FULL REFRESH — Script Server Entry Point
============================================================================
Runs the complete pipeline from scratch, saves baseline.parquet,
then renders and writes the dashboard HTML to www/.

Schedule: Once daily (or on deploy / version bump)
============================================================================
"""

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
import pytz

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
STATE_DIR   = SCRIPT_DIR / "dashboard_state"
WWW_DIR     = SCRIPT_DIR.parent / "www"
BASELINE     = STATE_DIR / "baseline.parquet"
VERSION_FILE = STATE_DIR / "pipeline_version.txt"
LAST_SYNC    = STATE_DIR / "last_sync.txt"
HTML_OUT     = WWW_DIR / "dna_sc_dashboard.html"

STATE_DIR.mkdir(parents=True, exist_ok=True)
WWW_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

# ── Imports ───────────────────────────────────────────────────────────────────
from dnasc import run_pipeline, render_dashboard, PipelineConfig
from dnasc.extractors.sheets import fetch_due_dates

# ── Pipeline version (bump this string when you push new code) ───────────────
PIPELINE_VERSION = PipelineConfig.PIPELINE_VERSION

def main():
    start = time.time()
    print("=" * 70)
    print(f"🚀 FULL REFRESH  |  version={PIPELINE_VERSION}  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # 1. Run the full pipeline
    print("\n⏱  Running full pipeline...")
    final_df = run_pipeline()

    # 2. Save baseline parquet + stamp version + reset sync timestamp
    print(f"\n💾 Saving baseline → {BASELINE}")
    final_df.to_parquet(BASELINE, index=False)
    VERSION_FILE.write_text(PIPELINE_VERSION)
    LAST_SYNC.write_text(str(datetime.now(pytz.UTC).timestamp()))
    print(f"   ✅ Baseline saved ({len(final_df):,} rows)")

    # 3. Fetch due dates from Google Sheet (or CSV fallback)
    print("\n📅 Fetching experiment due dates...")
    fetch_due_dates()

    # 4. Render HTML
    print("\n🎨 Rendering dashboard...")
    html = render_dashboard(final_df)

    # 4. Write to www/
    HTML_OUT.write_text(html, encoding="utf-8")
    VERSION_TS = WWW_DIR / "dnasc_version.txt"
    VERSION_TS.write_text(str(int(time.time())))
    print(f"   ✅ Dashboard written → {HTML_OUT}")

    elapsed = time.time() - start
    print(f"\n🎉 Full refresh complete in {elapsed:.1f}s")

if __name__ == "__main__":
    main()

"""
render_only.py
──────────────
Re-render the dashboard HTML from the existing baseline.parquet
without re-running the full pipeline (~9 min).

Use this when:
  - The renderer crashed after the pipeline saved parquet
  - You patched renderer code and want to preview the change
  - The pipeline succeeded but HTML was not written

Usage:
    ! /opt/anaconda3/bin/python3 /Users/juliehachey/scripts/render_only.py
"""

import sys
import time
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent.resolve()
STATE_DIR   = SCRIPT_DIR / "dashboard_state"
WWW_DIR     = SCRIPT_DIR.parent / "www"
BASELINE    = STATE_DIR / "baseline.parquet"
HTML_OUT    = WWW_DIR / "dna_sc_dashboard.html"

sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
from dnasc import render_dashboard
from dnasc.extractors.bios import BIOSExtractor
from dnasc.extractors.sheets import fetch_due_dates

def main():
    if not BASELINE.exists():
        print(f"ERROR: baseline.parquet not found at {BASELINE}")
        sys.exit(1)

    start = time.time()
    print(f"Reading {BASELINE}...")
    df = pd.read_parquet(BASELINE)
    print(f"  {len(df):,} rows loaded")

    print("Fetching due dates...")
    fetch_due_dates()

    print("Fetching experiment active map...")
    exp_active_map = BIOSExtractor.get_experiment_active_map()

    print("Rendering dashboard...")
    html = render_dashboard(df, experiment_active_map=exp_active_map)

    WWW_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"  Written → {HTML_OUT}  ({time.time() - start:.1f}s)")

if __name__ == "__main__":
    main()

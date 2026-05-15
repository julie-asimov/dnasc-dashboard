"""
Re-render the dashboard HTML from the existing baseline parquet.
Use this to test renderer-only changes without re-running the full pipeline.

Usage:
    ! /opt/anaconda3/bin/python3 /Users/juliehachey/scripts/rerender.py
"""

import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
BASELINE   = SCRIPT_DIR / "dashboard_state" / "baseline.parquet"
WWW_DIR    = SCRIPT_DIR.parent / "www"
HTML_OUT   = WWW_DIR / "dna_sc_dashboard.html"

sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
from dnasc import render_dashboard, PipelineConfig

def main():
    if not BASELINE.exists():
        print(f"ERROR: baseline parquet not found at {BASELINE}")
        sys.exit(1)

    print(f"Reading parquet → {BASELINE}")
    df = pd.read_parquet(BASELINE)
    print(f"  {len(df):,} rows loaded")

    print("Rendering dashboard...")
    t0 = time.time()
    html = render_dashboard(df)
    print(f"  Done in {time.time()-t0:.1f}s")

    HTML_OUT.write_text(html, encoding="utf-8")
    VERSION_TS = WWW_DIR / "dnasc_version.txt"
    VERSION_TS.write_text(str(int(time.time())))
    print(f"  Written → {HTML_OUT}")
    print(f"\nVersion: {PipelineConfig.PIPELINE_VERSION}")

if __name__ == "__main__":
    main()

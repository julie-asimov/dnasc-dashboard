"""Re-render dashboard HTML from saved baseline.parquet (no pipeline run)."""
import pandas as pd
from dnasc.renderer import render_dashboard
from pathlib import Path

BASELINE = Path(__file__).parent / "dashboard_state" / "baseline.parquet"
HTML_OUT = Path(__file__).parent.parent / "www" / "dna_sc_dashboard.html"

df = pd.read_parquet(BASELINE)
print(f"Loaded {len(df)} rows from {BASELINE}")
html = render_dashboard(df)
HTML_OUT.write_text(html, encoding="utf-8")
print(f"Done → {HTML_OUT}")

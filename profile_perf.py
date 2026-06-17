"""
profile_perf.py — performance harness for the dnasc dashboard pipeline.

Run this after adding a tab / changing a render path to catch the hot-loop
anti-patterns (per-item full-DataFrame scans, iterrows, unsubsetted to_dict,
object-dtype == comparisons) BEFORE they compound. See PERF.md for the patterns.

Uses the existing baseline parquet (final pipeline df) so render/enrichment can
be profiled in isolation, no BigQuery needed.

Usage (always use the anaconda python — it has pandas/pyarrow):
  /opt/anaconda3/bin/python3 profile_perf.py render        # profile the full dashboard render
  /opt/anaconda3/bin/python3 profile_perf.py enrich        # profile compute_request_enrichment
  /opt/anaconda3/bin/python3 profile_perf.py render --top 40

What to look for in the output:
  - High-`tottime` pandas internals (comp_method_OBJECT_ARRAY, isin, to_dict,
    Series.__init__) usually mean a per-item loop doing full-frame work.
  - A dnasc/ function with very high `ncalls` (~3300 = per-request, ~48740 =
    per-row, ~20000 = per-root) is the loop to inspect — the fix is almost always
    "pre-compute a lookup dict ONCE before the loop" or "subset columns before
    to_dict" or "iterate .to_numpy() instead of iterrows".
"""
import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
BASELINE = SCRIPT_DIR / "dashboard_state" / "baseline.parquet"


def _load():
    import pandas as pd
    if not BASELINE.exists():
        sys.exit(f"baseline parquet not found at {BASELINE} — run full_refresh.py first")
    return pd.read_parquet(BASELINE)


def _profile(fn, label, top):
    """Run fn() once for wall-clock, once under cProfile, print top hot spots."""
    t = time.time()
    fn()
    wall = time.time() - t
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    print(f"\n{'='*72}\n{label}: {wall:.1f}s wall-clock\n{'='*72}")

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top)
    print("── Top by self-time (tottime) ──")
    print(s.getvalue())

    # Heuristic: dnasc/ functions with the highest call counts = the hot loops.
    print("── dnasc/ functions by call count (the loops to inspect) ──")
    s2 = io.StringIO()
    st = pstats.Stats(pr, stream=s2)
    rows = []
    for func, (cc, nc, tt, ct, _callers) in st.stats.items():
        fname, lineno, funcname = func
        if "/dnasc/" in fname:
            rows.append((nc, tt, ct, f"{Path(fname).name}:{lineno}({funcname})"))
    for nc, tt, ct, where in sorted(rows, reverse=True)[:15]:
        print(f"  ncalls={nc:>8}  tottime={tt:7.2f}s  cumtime={ct:7.2f}s  {where}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["render", "enrich"], help="what to profile")
    ap.add_argument("--top", type=int, default=25, help="how many hot spots to show")
    args = ap.parse_args()

    df = _load()
    print(f"loaded baseline: {len(df):,} rows / {df['req_id'].nunique():,} requests")

    if args.target == "render":
        from dnasc.renderer.dashboard import render_dashboard
        _profile(lambda: render_dashboard(df), "render_dashboard", args.top)
    elif args.target == "enrich":
        from dnasc.transformers.enrichment import EnrichmentTransformer
        _profile(lambda: EnrichmentTransformer.compute_request_enrichment(df), "compute_request_enrichment", args.top)


if __name__ == "__main__":
    main()

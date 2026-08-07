"""
============================================================================
FULL REFRESH — Script Server Entry Point
============================================================================
Runs the complete pipeline from scratch, saves baseline.parquet, refreshes the
Parts tab data, then renders and writes the dashboard HTML to www/.

The Parts tab is a SEPARATE BigQuery pull (gen_parts_pkl) that also has its own
cron on the server, so the two halves of the dashboard can age independently. That
split meant a manual full refresh left the Parts tab stale — so it is refreshed
here too, before the render, but ONLY IF the dedicated parts cron did not just run
(PipelineConfig.PARTS_MAX_AGE_MINUTES, measured from when THIS refresh started —
not from when it reaches the parts step, which is ~11 min later once the pipeline
has run). On the server the parts cron fires 10 min before this, so it is a no-op;
locally there is no such cron, so one command covers both halves. If the cron
failed, the pkl is yesterday's and this picks up the slack.

    --skip-parts    never touch the parts pkl
    --force-parts   pull it even if it is fresh

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
from dnasc.extractors.bios import BIOSExtractor
from dnasc.extractors.sheets import fetch_due_dates, append_experiment_names

MISSING_DUE_FILE = STATE_DIR / "missing_asana_dates.json"

# ── Pipeline version (bump this string when you push new code) ───────────────
PIPELINE_VERSION = PipelineConfig.PIPELINE_VERSION

# The parts pull is ~250s on top of a ~640s refresh, so it runs only when the pkl is stale.
SKIP_PARTS  = "--skip-parts" in sys.argv
FORCE_PARTS = "--force-parts" in sys.argv
PARTS_PKL   = STATE_DIR / "parts_result.pkl"


def _parts_age_minutes(ref_ts):
    """Age of parts_result.pkl in minutes AS OF ref_ts, or None if it does not exist.

    ref_ts is the refresh start, not now: the pipeline runs first, so measuring at the parts
    step would add ~11 minutes to every reading and make a just-run cron look stale.
    """
    try:
        return (ref_ts - PARTS_PKL.stat().st_mtime) / 60.0
    except FileNotFoundError:
        return None

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

    # 3b. Refresh the Parts tab data — MUST run before the render, since the render
    #     reads parts_result.pkl. Non-fatal by design: gen_parts_pkl writes atomically,
    #     so a failure leaves the previous good pkl in place and the tab simply renders
    #     a little stale rather than taking the whole refresh down with it.
    _age = _parts_age_minutes(start)          # `start` = when this refresh began
    _limit = PipelineConfig.PARTS_MAX_AGE_MINUTES
    if SKIP_PARTS:
        print("\n🧬 Parts data: skipped (--skip-parts)")
    elif not FORCE_PARTS and _age is not None and _age <= _limit:
        print(f"\n🧬 Parts data: parts cron ran {_age:.1f} min before this refresh "
              f"(limit {_limit}) — skipping the pull")
    else:
        why = ("missing" if _age is None else
               "forced" if FORCE_PARTS else
               f"{_age:.1f} min old at refresh start (limit {_limit})")
        print(f"\n🧬 Refreshing Parts tab data — {why}...")
        _pt = time.time()
        try:
            import gen_parts_pkl
            gen_parts_pkl.main()
            print(f"   ✅ Parts data refreshed in {time.time() - _pt:.1f}s")
        except Exception as e:
            print(f"   ⚠️  Parts pull failed ({e}) — keeping the previous parts_result.pkl")

    # 4. Render HTML
    print("\n🎨 Rendering dashboard...")
    exp_active_map = BIOSExtractor.get_experiment_active_map()
    # Stream the HTML straight to disk (out_path) so the full ~160 MB document
    # is never held in memory — this is what keeps render from OOM-killing.
    render_dashboard(final_df, experiment_active_map=exp_active_map, out_path=HTML_OUT)
    VERSION_TS = WWW_DIR / "dnasc_version.txt"
    VERSION_TS.write_text(str(int(time.time())))
    print(f"   ✅ Dashboard written → {HTML_OUT}")

    # 5. Append any active partner experiments missing an Asana due date to the
    #    sheet (render wrote the list). Safe no-op if the service account lacks
    #    Editor access — logs a warning, never blocks the refresh.
    print("\n📤 Syncing missing partner experiments to the due-date sheet...")
    try:
        import json
        names = json.loads(MISSING_DUE_FILE.read_text()) if MISSING_DUE_FILE.exists() else []
        if names:
            res = append_experiment_names(names)
            if res["ok"] and res["appended"]:
                print(f"   ✅ Added {len(res['appended'])} new name(s): {', '.join(res['appended'])}")
            elif res["ok"]:
                print("   ✅ Nothing to add (all already in the sheet).")
            else:
                print(f"   ⚠️  Could not write to sheet ({res['error']}) — names flagged on dashboard only.")
        else:
            print("   ✅ No partner experiments missing an Asana date.")
    except Exception as e:
        print(f"   ⚠️  Sync skipped: {e}")

    elapsed = time.time() - start
    print(f"\n🎉 Full refresh complete in {elapsed:.1f}s")

if __name__ == "__main__":
    main()

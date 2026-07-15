"""Refresh the parts-preview data cache from a fresh BigQuery pull.

Standalone pull — runs on its own cron, independent of the dashboard build. Writes
parts_result.pkl ATOMICALLY (temp file + os.replace) so the dashboard render never
reads a half-written file: it sees either the previous good pkl or the complete new one.
"""
import sys, os, pickle, tempfile, traceback
from datetime import datetime, timezone

# resolve paths relative to THIS file so it works on both the Mac and the server
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parts_inventory as P

OUT = os.path.join(HERE, "dashboard_state", "parts_result.pkl")

def main():
    result = P.run_parts_inventory()
    # atomic write: dump to a temp file in the same dir, then os.replace() (atomic on same fs)
    d = os.path.dirname(OUT)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".parts_result.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(result, fh)
        os.replace(tmp, OUT)   # atomic swap into place
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"cached fresh parts_result.pkl @ {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        # never leave a torn file; surface the error for the cron log and exit non-zero
        traceback.print_exc()
        sys.exit(1)

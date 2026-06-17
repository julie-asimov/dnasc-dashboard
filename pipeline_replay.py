"""
pipeline_replay.py — record/replay harness to verify pipeline changes are
output-equivalent WITHOUT a live BigQuery run each time.

Why: the slow mid-pipeline steps (repair.py: populate_synthetic_optracker_batch,
resolve_downstream_plates, etc.) query BigQuery internally and can't be isolated
on baseline.parquet. And re-running the full pipeline twice (old vs new) drifts:
live BQ data changes + synthetic rows stamp pd.Timestamp.now(). This harness
removes both: it CACHES every BigQuery result keyed by the SQL text, so a replay
runs the exact same data deterministically and offline.

Workflow to verify a PYTHON-logic optimization (the repair.py steps):
  1. /opt/anaconda3/bin/python3 pipeline_replay.py record      # ~once, hits BQ, caches everything + saves golden ref
  2. <make your code change>
  3. /opt/anaconda3/bin/python3 pipeline_replay.py replay      # fast, offline — same cached BQ data
  4. /opt/anaconda3/bin/python3 pipeline_replay.py compare     # diffs ref vs new per-column, ignoring now()-stamped fields

A column-level "0 mismatches" on the logic columns (visual_status, chain_status,
root_work_order_id, req_id, stage, ...) means the change is output-equivalent.

For a BigQuery QUERY rewrite (changes the SQL), the SQL key changes -> cache miss
-> it falls through to live BQ. Verify those with an in-BQ FULL OUTER JOIN diff
instead (see PERF.md / the well-mapping fix), not this harness.
"""
import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
CACHE_DIR = SCRIPT_DIR / "replay_cache"
REF_PARQUET = CACHE_DIR / "_output_ref.pkl"
NEW_PARQUET = CACHE_DIR / "_output_new.pkl"

# Columns that are stamped with pd.Timestamp.now()/datetime.now() on synthetic
# rows (so they differ every run) — excluded from the equivalence comparison.
VOLATILE_COLS = {"wo_created_at", "wo_updated_at", "batch_created_at"}


def _key(sql: str) -> str:
    return hashlib.md5(" ".join(str(sql).split()).encode()).hexdigest()


def _install_bq_cache(mode: str):
    """Monkeypatch pandas.read_gbq and bigquery.Client.query to cache by SQL.

    mode='record': miss -> real call -> cache. mode='replay': miss -> real call
    (logged) so a changed query still works, but same-SQL hits are deterministic.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    from google.cloud import bigquery

    stats = {"hit": 0, "miss": 0}

    def _load_or_run(sql, run_real):
        f = CACHE_DIR / f"{_key(sql)}.pkl"
        if f.exists():
            stats["hit"] += 1
            with open(f, "rb") as fh:
                return pickle.load(fh)
        stats["miss"] += 1
        if mode == "replay":
            print(f"  [replay] cache MISS (changed/new query) -> live BQ: {_key(sql)[:8]}")
        df = run_real()
        with open(f, "wb") as fh:
            pickle.dump(df, fh)
        return df

    # --- patch pandas.read_gbq ---
    _real_read_gbq = pd.read_gbq
    def _cached_read_gbq(query, *a, **k):
        return _load_or_run(query, lambda: _real_read_gbq(query, *a, **k))
    pd.read_gbq = _cached_read_gbq

    # --- patch bigquery.Client.query -> job whose .to_dataframe() is cached ---
    _real_query = bigquery.Client.query
    class _CachedJob:
        def __init__(self, client, sql, args, kwargs):
            self._client, self._sql, self._a, self._k = client, sql, args, kwargs
            self._job = None
        def to_dataframe(self, *a, **k):
            return _load_or_run(self._sql, lambda: _real_query(self._client, self._sql, *self._a, **self._k).to_dataframe(*a, **k))
        def __getattr__(self, name):
            # any other access (.result(), .slot_millis, ...) -> run the real job
            if self._job is None:
                self._job = _real_query(self._client, self._sql, *self._a, **self._k)
            return getattr(self._job, name)
    def _cached_query(self, query, *a, **k):
        return _CachedJob(self, query, a, k)
    bigquery.Client.query = _cached_query

    return stats


def cmd_record():
    print("RECORD: running full pipeline, caching every BQ result + saving golden ref...")
    stats = _install_bq_cache("record")
    from dnasc import run_pipeline
    df = run_pipeline()
    with open(REF_PARQUET, "wb") as fh:
        pickle.dump(df, fh)
    print(f"  cached {stats['miss']} queries; golden ref = {len(df):,} rows -> {REF_PARQUET}")


def cmd_replay():
    if not REF_PARQUET.exists():
        sys.exit("no golden ref — run `record` first")
    print("REPLAY: running full pipeline from cached BQ data (offline, deterministic)...")
    stats = _install_bq_cache("replay")
    from dnasc import run_pipeline
    df = run_pipeline()
    with open(NEW_PARQUET, "wb") as fh:
        pickle.dump(df, fh)
    print(f"  cache hits={stats['hit']} miss={stats['miss']}; new output = {len(df):,} rows -> {NEW_PARQUET}")
    if stats["miss"]:
        print("  NOTE: misses mean the new code issued different SQL (expected only for query rewrites).")


def _norm_cell(v):
    import numpy as np, pandas as pd
    if isinstance(v, (list, np.ndarray)):
        return "|".join("" if pd.isna(x) else str(x) for x in list(v))
    try:
        if pd.isna(v):
            return ""
    except (ValueError, TypeError):
        pass
    return str(v)


def cmd_compare(key="workorder_id", show=8):
    import pandas as pd
    if not (REF_PARQUET.exists() and NEW_PARQUET.exists()):
        sys.exit("need both ref (record) and new (replay) outputs")
    with open(REF_PARQUET, "rb") as fh: ref = pickle.load(fh)
    with open(NEW_PARQUET, "rb") as fh: new = pickle.load(fh)
    print(f"COMPARE: ref={len(ref):,} rows  new={len(new):,} rows")

    # row-set equality on the key
    rk, nk = set(ref[key].astype(str)), set(new[key].astype(str))
    if rk != nk:
        print(f"  ROW-SET DIFF: only-in-ref={len(rk - nk)}  only-in-new={len(nk - rk)}")
        for x in list(rk - nk)[:show]: print(f"    - missing in new: {x}")
        for x in list(nk - rk)[:show]: print(f"    + added in new:   {x}")
    else:
        print(f"  row set identical ({len(rk):,} keys) ✓")

    # per-column comparison on shared keys (aligned by key), ignoring volatile cols
    common = sorted(rk & nk)
    r = ref[ref[key].astype(str).isin(common)].set_index(ref[key].astype(str)).sort_index()
    n = new[new[key].astype(str).isin(common)].set_index(new[key].astype(str)).sort_index()
    cols = [c for c in ref.columns if c in new.columns and c not in VOLATILE_COLS]
    print(f"  comparing {len(cols)} columns over {len(common):,} shared rows (ignoring {sorted(VOLATILE_COLS)}):")
    any_diff = False
    for c in cols:
        rv = r[c].map(_norm_cell).values
        nv = n[c].map(_norm_cell).values
        mism = int((rv != nv).sum())
        if mism:
            any_diff = True
            print(f"    ✗ {c}: {mism} mismatches")
            idx = [common[i] for i in range(len(common)) if rv[i] != nv[i]][:show]
            for k_ in idx:
                print(f"        {k_}: ref={_norm_cell(r.loc[k_, c])!r}  new={_norm_cell(n.loc[k_, c])!r}")
    if not any_diff:
        print("  ALL non-volatile columns identical ✓  — change is output-equivalent")


def main():
    # Determinism: the pipeline builds query IN-lists and some root/req assignments
    # from set iteration, which Python randomizes per process. Pin the hash seed so
    # record and replay produce identical SQL (-> cache hits) and identical output.
    # Must be set before the interpreter starts, so re-exec ourselves once.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["record", "replay", "compare"])
    args = ap.parse_args()
    {"record": cmd_record, "replay": cmd_replay, "compare": cmd_compare}[args.cmd]()


if __name__ == "__main__":
    main()

# Performance conventions — dnasc dashboard

The pipeline + render are **CPU-bound pandas** over a large frame (~48k rows /
3,330 requests / 20,175 roots). As tabs are added, the same handful of
anti-patterns keep creeping back into per-item render loops. This doc is the
playbook to keep new code fast and to verify optimizations safely.

## Profile first — `profile_perf.py`

After adding a tab or changing a render/transform path, run the harness (it uses
the existing `dashboard_state/baseline.parquet`, no BigQuery needed):

```
/opt/anaconda3/bin/python3 profile_perf.py render     # full dashboard render
/opt/anaconda3/bin/python3 profile_perf.py enrich     # compute_request_enrichment
```

Read the output: a `dnasc/` function with a huge **ncalls** (~3,300 = per-request,
~48,740 = per-row, ~20,000 = per-root) is the loop to inspect. High `tottime` in
pandas internals (`comp_method_OBJECT_ARRAY`, `isin`, `to_dict`, `Series.__init__`)
means a per-item loop is doing full-frame work.

The pipeline already logs a per-step `STEP TIMING BREAKDOWN` — read it to see which
of the 20 steps regressed.

## The anti-patterns (and the fix)

1. **Full-DataFrame scan inside a per-item loop** → O(n²). The worst offender.
   ```python
   for root_id in roots:                       # ~20k iterations
       hits = df[(df['root_work_order_id'] == root_id) & df['type'].isin(parts)]   # full 48k scan EACH time
   ```
   **Fix:** pre-group ONCE before the loop, then dict-lookup.
   ```python
   by_root = {k: g for k, g in df[df['type'].isin(parts)].groupby('root_work_order_id')}
   for root_id in roots:
       hits = by_root.get(root_id)             # O(1)
   ```
   (Fixed in `enrichment.py` and the renderer's cross-request parts fan-in.)

2. **Unsubsetted `.to_dict('records')`** on a wide frame when only a few columns
   are read downstream — converts all ~60 columns per call.
   **Fix:** `grp[['type','vendor']].to_dict('records')`. (Fixed in `enrichment.py`,
   cut that step ~45%.) Exception: the renderer's `row_map` legitimately needs all
   columns — don't subset that one.

3. **`iterrows()`** builds a pandas Series per row in a hot loop.
   **Fix:** iterate raw arrays — `zip(df['a'].to_numpy(), df['b'].to_numpy())`.
   (Fixed in `enrichment.py`.)

4. **Repeated `.astype(str)` / object-dtype `==`** inside a loop.
   **Fix:** hoist the conversion out of the loop, or convert the column to
   `category` once up front (the renderer does this for `type`/`wo_status`/
   `visual_status`/`data_source` so inner-loop `.isin()`/`==` use integer codes).

5. **`df.copy()` is load-bearing in `compute_request_enrichment`** — it
   defragments the frame before the per-request loop. Do NOT switch it to
   `copy(deep=False)` (tried; it regressed ~80s and re-fragmented). See
   `memory/perf_pipeline_2026-06.md`.

6. **BigQuery: recursive CTEs re-run their base join every level** (BQ inlines
   CTEs). **Fix:** `CREATE TEMP TABLE` the base once, recurse against it.
   (Fixed `_fetch_well_mapping`: 119s → 57s.) Also watch for non-sargable joins
   (a function on the join key) and `SELECT *` pulling unused wide columns.

## Verifying an optimization is output-identical

The render embeds live timestamps/ages, so **byte-diffing two renders is useless**
(a same-code render differs by ~430 time-dependent lines — that's expected, not a
bug). Verify by **comparing data, not HTML**:

- **Semantic check (preferred):** reproduce the old computation and the new one on
  `baseline.parquet`, compare the resulting sets/values directly. Example: the
  fan-in fix was verified by checking the dict returned the identical
  `workorder_id` set as the old scan for all 20,175 roots (0 mismatches).
- **Transformer functions** (enrichment, etc.) are pure functions of the df —
  run old vs new on `baseline.parquet` and assert the output columns are equal.
- **BigQuery query rewrites:** verify with an in-BQ `FULL OUTER JOIN` diff of old
  vs new on the key column (well-mapping: 0 mismatches over 699k rows).

Always bump `config.py PIPELINE_VERSION` on a shipped change. Run `pytest tests/`
(294 tests) before committing.

## Known remaining debt (measure before touching)

- `render`: `batch_to_est`/`to_est` timezone conversion (~15s, called 100k+ times),
  and `row_map` `to_dict` (~37s, needs all columns).
- `repair.py`: `populate_synthetic_optracker_batch` (~75–100s) and
  `resolve_downstream_plates` (~70–85s) — partly BQ-bound, hard to isolate; need a
  full-pipeline run to verify.
- Extraction queries (`lims.py` colony ~210s, `optracker.py`/`bios.py` ~160–250s) —
  network-bound; optimize the SQL with the in-BQ diff verification above.

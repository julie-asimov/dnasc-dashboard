"""
verify_fixes.py
---------------
Compare baseline.parquet (pre-fix) vs a freshly-generated parquet (post-fix)
on the metrics most affected by the bug fixes. Run after full_refresh.py.

Usage:
    # 1. Make a backup of current (pre-fix) parquet
    #    cp dashboard_state/baseline.parquet dashboard_state/baseline_prefx.parquet
    # 2. Run full refresh
    #    /opt/anaconda3/bin/python3 full_refresh.py
    # 3. Run this script
    #    /opt/anaconda3/bin/python3 verify_fixes.py
"""

import pandas as pd
import numpy as np
import sys

PRE  = "dashboard_state/baseline_prefx.parquet"
POST = "dashboard_state/baseline.parquet"

def load(path):
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        print(f"ERROR: {path} not found")
        sys.exit(1)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def row(label, before, after, fmt=str):
    delta = ""
    try:
        d = after - before
        delta = f"  ({'+' if d >= 0 else ''}{fmt(d)})"
    except Exception:
        pass
    print(f"  {label:<40} {fmt(before):>10} → {fmt(after):>10}{delta}")

pre  = load(PRE)
post = load(POST)

# ── 1. Row counts ─────────────────────────────────────────────────────────────
section("Row counts")
row("Total rows", len(pre), len(post))
for t in sorted(pre["type"].dropna().unique()):
    row(f"  {t}", (pre["type"] == t).sum(), (post["type"] == t).sum())

# ── 2. Status distribution ────────────────────────────────────────────────────
section("Visual status distribution")
for s in sorted(pre["visual_status"].dropna().unique()):
    row(f"  {s}", (pre["visual_status"] == s).sum(), (post["visual_status"] == s).sum())

# ── 3. LSP rows (fan-out fix) ─────────────────────────────────────────────────
section("LSP workorder dedup (fan-out fix)")
pre_lsp  = pre[pre["type"] == "lsp_workorder"]
post_lsp = post[post["type"] == "lsp_workorder"]
row("LSP rows total",            len(pre_lsp), len(post_lsp))
row("LSP unique workorder_ids",  pre_lsp["workorder_id"].nunique(), post_lsp["workorder_id"].nunique())
pre_dup  = len(pre_lsp)  - pre_lsp["workorder_id"].nunique()
post_dup = len(post_lsp) - post_lsp["workorder_id"].nunique()
row("LSP duplicate rows",        pre_dup, post_dup)

# ── 4. Colony counts (lims dedup fix) ────────────────────────────────────────
section("Colony counts (lims dedup fix)")
for col in ["total_colonies", "available_colonies", "seq_confirmed"]:
    if col in pre.columns and col in post.columns:
        pre_v  = pre[pre[col].notna()][col].astype(float)
        post_v = post[post[col].notna()][col].astype(float)
        print(f"  {col}")
        print(f"    pre  — mean: {pre_v.mean():.2f}  median: {pre_v.median():.0f}  max: {pre_v.max():.0f}")
        print(f"    post — mean: {post_v.mean():.2f}  median: {post_v.median():.0f}  max: {post_v.max():.0f}")

# ── 5. req_id assignments (lineage fix) ───────────────────────────────────────
section("req_id assignment (lineage fix)")
pre_lsp_req  = pre_lsp[pre_lsp["req_id"].notna()]["workorder_id"].nunique()
post_lsp_req = post_lsp[post_lsp["req_id"].notna()]["workorder_id"].nunique()
row("LSP rows with req_id", pre_lsp_req, post_lsp_req)

# LSPs that changed req_id
if "workorder_id" in pre_lsp.columns and "req_id" in pre_lsp.columns:
    pre_map  = pre_lsp.drop_duplicates("workorder_id").set_index("workorder_id")["req_id"]
    post_map = post_lsp.drop_duplicates("workorder_id").set_index("workorder_id")["req_id"]
    common   = pre_map.index.intersection(post_map.index)
    changed  = (pre_map[common] != post_map[common]).sum()
    row("LSP workorders with changed req_id", changed, 0)

# ── 6. Canceled LSP rows (ndarray fix) ───────────────────────────────────────
section("Canceled LSP rows kept (ndarray fix)")
def has_ops(x):
    if isinstance(x, (list, np.ndarray)):
        return len(x) > 0
    return False

pre_can  = pre_lsp[pre_lsp["wo_status"].astype(str).str.upper() == "CANCELED"]
post_can = post_lsp[post_lsp["wo_status"].astype(str).str.upper() == "CANCELED"]
row("Canceled LSP rows total",       len(pre_can),  len(post_can))
pre_can_ops  = pre_can["protocol_name"].apply(has_ops).sum()
post_can_ops = post_can["protocol_name"].apply(has_ops).sum()
row("Canceled LSP with ops (kept)",  pre_can_ops, post_can_ops)

# ── 7. BIOS_DRAFT in enrichment (enrichment fix) ─────────────────────────────
section("BIOS_DRAFT rows")
if "data_source" in pre.columns:
    row("BIOS_DRAFT rows",
        (pre["data_source"] == "BIOS_DRAFT").sum(),
        (post["data_source"] == "BIOS_DRAFT").sum() if "data_source" in post.columns else 0)

# ── 8. req_phase coverage (enrichment produces this) ─────────────────────────
section("req_phase / req_operation coverage")
for col in ["req_phase", "req_operation"]:
    if col in pre.columns and col in post.columns:
        pre_filled  = pre[col].notna() & (pre[col].astype(str) != "")
        post_filled = post[col].notna() & (post[col].astype(str) != "")
        row(f"{col} non-empty rows", pre_filled.sum(), post_filled.sum())

print("\nDone.")

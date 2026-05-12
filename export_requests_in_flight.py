#!/usr/bin/env /opt/anaconda3/bin/python3
"""
export_requests_in_flight.py
----------------------------
Exports one row per active request (non-canceled requests in active experiments)
with pipeline stage, OpTracker step, submitter, and Last Feasible Cycle milestone dates.

Usage:
    /opt/anaconda3/bin/python3 export_requests_in_flight.py
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import os

PARQUET = "/Users/juliehachey/scripts/dashboard_state/baseline.parquet"
OUT_DIR = os.path.expanduser("~/Downloads")


def to_list(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, np.ndarray)):
        return [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return [v]


# Priority = position in lab workflow (higher = further along)
PROTO_MAP = {
    ('Synthesis Order',                      'RD'): (5,  'Synthesis Order Ready'),
    ('Synthesis Order',                      'RU'): (6,  'Synthesis Order Running'),
    ('Receive SynPart Synthesis',            'RD'): (7,  'Receive SynPart Ready'),
    ('Receive SynPart Synthesis',            'RU'): (8,  'Receive SynPart Running'),
    ('Receive Plasmid Synthesis',            'RD'): (8,  'Receive Plasmid Synthesis Ready'),
    ('Receive Plasmid Synthesis',            'RU'): (8,  'Receive Plasmid Synthesis Running'),
    ('Fragment Analyzer',                    'RD'): (9,  'Fragment Analyzer Ready'),
    ('Fragment Analyzer',                    'RU'): (9,  'Fragment Analyzer Running'),
    ('Golden Gate Assembly',                 'RD'): (10, 'GG Assembly Ready'),
    ('Golden Gate Assembly',                 'RU'): (11, 'GG Assembly Running'),
    ('Gibson Assembly',                      'RD'): (10, 'Gibson Assembly Ready'),
    ('Gibson Assembly',                      'RU'): (11, 'Gibson Assembly Running'),
    ('STAR Transformation',                  'RD'): (20, 'Transformation Ready'),
    ('STAR Transformation',                  'RU'): (21, 'Transformation Running'),
    ('Create Minipreps and Glycerol Stocks', 'RD'): (30, 'Miniprep Ready'),
    ('Create Minipreps and Glycerol Stocks', 'RU'): (31, 'Miniprep Running'),
    ('Rearray 96 to 384',                    'RD'): (35, 'Rearray Ready'),
    ('Rearray 96 to 384',                    'RU'): (36, 'Rearray Running'),
    ('DNA Quantification',                   'RD'): (40, 'DNA Quant Ready'),
    ('DNA Quantification',                   'RU'): (41, 'DNA Quant Running'),
    ('NGS Sequence Confirmation',            'RD'): (50, 'NGS Ready'),
    ('NGS Sequence Confirmation',            'RU'): (51, 'NGS Running'),
    ('LSP Order',                            'RD'): (60, 'LSP Order Ready'),
    ('LSP Order',                            'RU'): (61, 'LSP Order Running'),
    ('LSP Receiving',                        'RD'): (65, 'LSP Receiving Ready'),
    ('LSP Receiving',                        'RU'): (66, 'LSP Receiving Running'),
    ('Glycerol Stocking Scinomix',           'RD'): (70, 'Glycerol Stocking Ready'),
    ('Glycerol Stocking Scinomix',           'RU'): (71, 'Glycerol Stocking Running'),
    ('LSP Reviewing',                        'RD'): (80, 'LSP Reviewing Ready'),
    ('LSP Reviewing',                        'RU'): (81, 'LSP Reviewing Running'),
    ('LSP Releasing',                        'RD'): (90, 'LSP Releasing Ready'),
    ('LSP Releasing',                        'RU'): (91, 'LSP Releasing Running'),
}


def build_ot_stage_map(df):
    """
    For each req_id, find the highest-priority active (RD/RU) OpTracker step.
    Scans all workorders for the request EXCEPT ones with a different pAI STOCK_ID
    (those are foreign constructs that happen to share the same request row).
    """
    root_rows = df[df['workorder_id'] == df['root_work_order_id']].dropna(subset=['req_id'])
    req_root_stock = root_rows.drop_duplicates('req_id').set_index('req_id')['STOCK_ID'].to_dict()

    LSP_PROTOCOLS = {
        'LSP Order', 'LSP Receiving', 'Glycerol Stocking Scinomix',
        'LSP Reviewing', 'LSP Releasing',
    }

    ot_map = {}
    req_df = df[df['req_id'].notna() & (df['req_id'].astype(str) != '')].copy()
    for req_id, grp in req_df.groupby('req_id'):
        root_stock = req_root_stock.get(req_id, '')

        def is_relevant(row):
            sid = str(row['STOCK_ID']) if row['STOCK_ID'] else ''
            # Always include LSP-stage workorders regardless of STOCK_ID
            protos = set(str(p) for p in to_list(row['protocol_name']))
            if protos & LSP_PROTOCOLS:
                return True
            # Exclude rows whose STOCK_ID is a *different* pAI construct
            if sid.startswith('pAI-') and sid != root_stock:
                return False
            return True

        scan = grp[grp.apply(is_relevant, axis=1)]
        if scan.empty:
            scan = grp

        best = (-1, '')
        for _, row in scan.iterrows():
            for p, s in zip(to_list(row['protocol_name']), to_list(row['operation_state'])):
                key = (str(p), str(s))
                if key in PROTO_MAP:
                    pri, label = PROTO_MAP[key]
                    if pri > best[0]:
                        best = (pri, label)
        ot_map[req_id] = best[1]
    return ot_map


def last_ngs_before(dt):
    for i in range(7):
        d = dt - timedelta(days=i)
        if d.weekday() in (0, 3):  # Monday=0, Thursday=3
            return d
    return dt


def milestones(created_at, for_partner):
    try:
        cd = pd.Timestamp(created_at).date()
    except Exception:
        return {}
    weeks = 5 if for_partner else 6
    due = cd + timedelta(weeks=weeks)
    ngs = last_ngs_before(due - timedelta(days=1))
    scaleup_offset = -5 if ngs.weekday() == 0 else -3
    recv_offset    = -3 if ngs.weekday() == 0 else -1
    return {
        'due_date':     due,
        'assembly':     ngs - timedelta(days=13),
        'asm_ngs':      ngs - timedelta(days=6),
        'lsp_scaleup':  ngs + timedelta(days=scaleup_offset),
        'lsp_received': ngs + timedelta(days=recv_offset),
        'lsp_ngs':      ngs,
        'release':      ngs + timedelta(days=1),
    }


def main():
    print(f"Reading {PARQUET} ...")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df):,} rows loaded")

    print("Building OpTracker stage map...")
    ot_map = build_ot_stage_map(df)
    print(f"  {sum(1 for v in ot_map.values() if v)} / {len(ot_map)} requests have active stage")

    # Root workorders only (fulfills_request rows)
    roots = df[df['workorder_id'] == df['root_work_order_id']].copy()
    roots = roots[roots['req_id'].notna() & (roots['req_id'].astype(str) != '')].copy()
    roots = roots[roots['request_status'] != 'CANCELED'].copy()

    # Active experiments = any experiment with >= 1 IN_PROGRESS request
    active_exps = set(
        roots[roots['request_status'] == 'IN_PROGRESS']['experiment_name'].dropna().unique()
    )
    roots = roots[roots['experiment_name'].isin(active_exps)].copy()
    req_rows = roots.sort_values('request_created_at').drop_duplicates(subset='req_id', keep='first')
    print(f"  {len(req_rows)} requests in active experiments")

    records = []
    for _, row in req_rows.iterrows():
        req_id = row['req_id']
        fp = str(row.get('for_partner', '')).lower() == 'true'
        ms = milestones(row.get('request_created_at'), fp)
        records.append({
            'experiment':         row.get('experiment_name', ''),
            'construct_name':     row.get('construct_name', ''),
            'pAI':                row.get('STOCK_ID', ''),
            'for_partner':        fp,
            'customer':           row.get('customer', ''),
            'submitter_email':    row.get('submitter_email', ''),
            'request_status':     row.get('request_status', ''),
            'stage':              row.get('stage', ''),
            'optracker_stage':    ot_map.get(req_id, ''),
            'req_id':             req_id,
            'request_created_at': str(row.get('request_created_at', ''))[:10],
            'due_date':           str(ms.get('due_date', '')),
            'assembly':           str(ms.get('assembly', '')),
            'asm_ngs':            str(ms.get('asm_ngs', '')),
            'lsp_scaleup':        str(ms.get('lsp_scaleup', '')),
            'lsp_received':       str(ms.get('lsp_received', '')),
            'lsp_ngs':            str(ms.get('lsp_ngs', '')),
            'release':            str(ms.get('release', '')),
        })

    out = pd.DataFrame(records).sort_values(['experiment', 'assembly'])

    today = date.today().strftime('%Y%m%d')
    out_path = os.path.join(OUT_DIR, f"requests_in_flight_{today}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows → {out_path}")

    blank = out[(out['request_status'] == 'IN_PROGRESS') & (out['optracker_stage'].isna() | (out['optracker_stage'] == ''))]
    if len(blank):
        print(f"\nIN_PROGRESS with no active OpTracker step ({len(blank)} rows — these have no running/ready ops):")
        print(blank[['construct_name', 'stage']].to_string())


if __name__ == '__main__':
    main()

"""
dnasc/renderer/inflight.py
──────────────────────────
Requests In Flight tab.
Python builds the data payload (records list + metadata) and emits it as JSON
into the iframe HTML. The table is rendered entirely client-side by window.ifRender()
so filtering and sorting work without round-trips.
"""

from __future__ import annotations
import json
from datetime import date, timedelta

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc import protocols as proto


# ── Colony Tracking rollup ─────────────────────────────────────────────────────
# Strain display labels (cloning_strain → header label). Raw values are kept
# elsewhere in the codebase (dashboard tracking tab shows them verbatim); we only
# shorten NEB_STABLE for the condensed colony view.
_STRAIN_LABEL = {'NEB_STABLE': 'NEB_STBL', 'EPI400': 'EPI400', 'STBL3': 'STBL3'}
# Workorder type → display label.
_DTYPE_LABEL = {
    'golden_gate_workorder':            'Golden Gate',
    'gibson_workorder':                 'Gibson',
    'transformation_workorder':         'Transformation',
    'transformation_offline_operation': 'Transformation',
    'lsp_workorder':                    'LSP',
    'streakout_operation':              'Streakout',
    'pcr_workorder':                    'PCR',
    'oligo_synthesis_workorder':        'Oligo',
    'syn_part_synthesis_workorder':     'Syn Part',
    'plasmid_synthesis_workorder':      'Plasmid Syn',
}
# Assembly workorder types (carry attempt_anchor_id; define the design).
_ASM_TYPES = frozenset({'golden_gate_workorder', 'gibson_workorder'})
# Colony-picking workorder types shown at L3 (assembly + strain transformations).
# LSP (downstream scale-up) and parts/inputs are excluded — LSP's total/seq just
# re-count the already-confirmed clone, and parts carry no colony data.
_L3_TYPES = frozenset(_ASM_TYPES | {'transformation_workorder', 'transformation_offline_operation'})
# L3 row ordering: assembly attempts → transformations → lsp → parts/inputs.
_KIND_RANK = {'assembly': 0, 'transformation': 1, 'lsp': 2, 'parts': 3}
# Design-status priority (lower wins) for rolling attempt statuses to a design verdict.
_STATUS_RANK = {'SUCCEEDED': 0, 'RUNNING': 1, 'READY': 1, 'WAITING': 2,
                'IN_PROGRESS': 2, 'FAILED': 3, 'CANCELED': 4, 'DRAFT': 5}
_COLONY_METRICS = ['imaged_colonies', 'pickable_colonies', 'picked_colonies',
                   'total_colonies', 'seq_confirmed']


def _kind(wtype: str) -> str:
    if wtype in _ASM_TYPES:                          return 'assembly'
    if wtype in ('transformation_workorder', 'transformation_offline_operation'): return 'transformation'
    if wtype == 'lsp_workorder':                     return 'lsp'
    return 'parts'


def _pp(s) -> str:
    """Compact a 'd8004:True, d8073:True' backbone/parts string to 'd8004, d8073'.
    Drops parts flagged ':False' (declared but not actually used in the build)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ''
    out = []
    for tok in str(s).split(','):
        tok = tok.strip()
        if not tok:
            continue
        pid, _, flag = tok.partition(':')
        if flag.strip().lower() == 'false':
            continue
        out.append(pid.strip())
    return ', '.join(out)


def _i(v) -> int:
    """Coalesce a possibly-null colony metric to int 0."""
    try:
        return int(v) if pd.notna(v) else 0
    except Exception:
        return 0


def _op_list(v) -> list:
    """Normalize a per-workorder op array (ndarray/list/None/scalar) to a list."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if hasattr(v, 'tolist'):          # numpy ndarray
        return list(v.tolist())
    return []


def _star_date(row) -> str:
    """Date (YYYY-MM-DD, no time) of the 'STAR Transformation' op for this workorder, if any."""
    names  = _op_list(row.get('protocol_name'))
    starts = _op_list(row.get('operation_start'))
    for i, n in enumerate(names):
        if str(n) == proto.STAR_TRANSF and i < len(starts) and starts[i]:
            try:
                return str(pd.Timestamp(starts[i]).date())
            except Exception:
                return ''
    return ''


def _well_alpha(pos, count) -> str:
    """Alphanumeric well (e.g. B4) from a 0-indexed LIMS position + plate well-count.
    Column-major like the dashboard maps: 96→8 rows, 384→16 rows, agar→2 rows."""
    try:
        idx = int(pos)
    except Exception:
        return ''
    if idx < 0:
        return ''
    try:
        cnt = int(count)
    except Exception:
        cnt = 0
    nrows = 16 if cnt == 384 else (8 if cnt == 96 else 2)
    return f"{chr(ord('A') + (idx % nrows))}{idx // nrows + 1}"


def _agar(row) -> tuple[str, str]:
    cpid = row.get('colony_plate_id')
    if pd.isna(cpid) or str(cpid) in ('nan', 'None', ''):
        return '', ''
    try:
        pid = int(cpid)
    except Exception:
        return '', ''
    well = _well_alpha(row.get('colony_well_position'), row.get('colony_plate_well_count'))
    label = f'Plate {pid}' + (f' · {well}' if well else '')
    return f'https://bios.asimov.io/inventory/plates/{pid}', label


_STALL_STRAIN = PipelineConfig.MIN_PICKABLE_COLONIES


def _low_pick(raw_pickable) -> bool:
    """A colony-picking row is low-pickable only when picking data exists and is below threshold.
    Rows with no picking-count data (LSP, etc.) coalesce to null → never flagged."""
    return bool(pd.notna(raw_pickable) and raw_pickable < PipelineConfig.MIN_PICKABLE_COLONIES)


def _has_colony(row) -> bool:
    """True when the workorder carries any colony-stage data (imaged/pickable/picked/total/seq).
    Parts/inputs (PCR, Oligo, Syn Part, Plasmid Syn) never do → excluded from L3."""
    return any(pd.notna(row.get(c)) for c in _COLONY_METRICS)


def _by_strain(rows: list) -> list:
    """Per-strain rollup (pickable/picked/seq/tot) over a list of colony-bearing row dicts."""
    agg: dict = {}
    for r in rows:
        s = r.get('strain') or '—'
        a = agg.setdefault(s, {'strain': s, 'pickable': 0, 'picked': 0, 'seq': 0, 'tot': 0})
        a['pickable'] += r['pickable']; a['picked'] += r['picked']
        a['seq']      += r['seq'];      a['tot']    += r['totc']
    return [agg[k] for k in sorted(agg)]


def _build_colony_rollup(base: pd.DataFrame, today: date, req_ids: set | None = None) -> dict:
    """
    Build the design-first 3-level Colony Tracking structure per req_id, from ALL
    workorder rows of the request (so assembly attempts, strain transformations and
    parts/inputs are all present — mirroring the tracking tab, condensed).

      L1 REQUEST = sum across every design/attempt/strain
      L2 DESIGN  = one per attempt_anchor_id (distinct backbone+parts). Retries of the
                   SAME design share an anchor and fold together. Transformation/parts
                   rows resolve to a design via root_work_order_id → anchor.
      L3 ROW     = one workorder within the design (deduped by workorder_id):
                   assembly attempt(s) → strain transformations → parts/inputs.

      { req_id: {
          'col':     {imaged,pickable,picked,seq,tot,has_winner,cflags:[...]},
          'designs': [
             {anchor,dtype,backbone,parts,status,n_attempts,strains:[...],low_pick,
              has_winner,imaged,pickable,picked,seq,tot,
              rows:[ {kind,attempt_n,strain,dtype,wid,status,created,star_date,noc,
                      low_pick,imaged,pickable,picked,seq,totc,agar_url,agar_label} ]}
          ]
      } }
    """
    rows = base[base['req_id'].notna()].copy()
    if req_ids is not None:
        rows = rows[rows['req_id'].isin(req_ids)]
    if rows.empty or 'attempt_anchor_id' not in rows.columns:
        return {}

    out: dict = {}
    for req_id, g in rows.groupby('req_id'):
        g = g.copy()
        # root_work_order_id → design anchor, taken from the assembly (Gibson/GG) row.
        asm = g[g['type'].isin(_ASM_TYPES)]
        root2anchor = {}
        for _, ar in asm.iterrows():
            wid = str(ar.get('workorder_id') or '')
            anc = ar.get('attempt_anchor_id')
            root2anchor[wid] = str(anc) if (anc is not None and pd.notna(anc)) else wid

        def _design(row):
            if row.get('type') in _ASM_TYPES:
                anc = row.get('attempt_anchor_id')
                return str(anc) if (anc is not None and pd.notna(anc)) else str(row.get('workorder_id') or '')
            root = str(row.get('root_work_order_id') or '')
            if root in root2anchor:
                return root2anchor[root]
            return root or str(row.get('workorder_id') or '')

        g['_design'] = g.apply(_design, axis=1)

        # ── L1 request sums (colony-picking rows only: assembly + transformation) ──
        gc = g[g['type'].isin(_L3_TYPES)]
        r_imaged = sum(_i(v) for v in gc.get('imaged_colonies',   []))
        r_pick   = sum(_i(v) for v in gc.get('pickable_colonies', []))
        r_picked = sum(_i(v) for v in gc.get('picked_colonies',   []))
        r_seq    = sum(_i(v) for v in gc.get('seq_confirmed',     []))
        r_tot    = sum(_i(v) for v in gc.get('total_colonies',    []))
        has_winner = bool(g['has_seq_winner'].any()) if 'has_seq_winner' in g else False

        last_op = None
        if 'wo_updated_at' in g.columns:
            _lo = g['wo_updated_at'].dropna()
            if not _lo.empty:
                last_op = _lo.max()

        # ── L2 DESIGN (by attempt_anchor_id) → L3 ATTEMPT (by root) → L4 workorders ──
        designs = []
        for anchor, dg in g.groupby('_design'):
            asm_all = dg[dg['type'].isin(_ASM_TYPES)]          # all assembly attempts (for anchor/verdict)

            # Build one attempt per assembly root_work_order_id; transformations rooted
            # to that Gibson nest underneath it.
            atts = []
            for root, ag in dg.groupby('root_work_order_id'):
                ag = ag[ag['type'].isin(_L3_TYPES)]            # assembly + transformation only
                if ag.empty:
                    continue
                gib = ag[ag['type'].isin(_ASM_TYPES)]
                att_num = None
                if not gib.empty and pd.notna(gib.iloc[0].get('attempt_number')):
                    att_num = int(gib.iloc[0]['attempt_number'])
                seen, wo_rows = set(), []
                for _, ar in ag.iterrows():
                    wid = str(ar.get('workorder_id') or '')
                    if wid in seen:
                        continue
                    seen.add(wid)
                    kind = _kind(ar.get('type'))
                    au, al = _agar(ar)
                    cr = ar.get('wo_created_at')
                    raw_pick = ar.get('pickable_colonies')
                    wo_rows.append({
                        'kind':    kind,
                        'is_child':kind == 'transformation',
                        'strain':  _STRAIN_LABEL.get(ar.get('cloning_strain'), str(ar.get('cloning_strain') or '')),
                        'dtype':   _DTYPE_LABEL.get(ar.get('type'), str(ar.get('type', '') or '')),
                        'wid':     wid,
                        'status':  str(ar.get('visual_status', '') or ''),
                        'created': str(pd.Timestamp(cr).date()) if pd.notna(cr) else '',
                        'star_date': _star_date(ar),
                        'hascol':  _has_colony(ar),
                        'imaged':  _i(ar.get('imaged_colonies')),
                        'pickable':_i(raw_pick),
                        'picked':  _i(ar.get('picked_colonies')),
                        'seq':     _i(ar.get('seq_confirmed')),
                        'totc':    _i(ar.get('total_colonies')),
                        'low_pick':_low_pick(raw_pick),
                        '_csort':  pd.Timestamp(cr) if pd.notna(cr) else pd.Timestamp.max,
                        'agar_url':au, 'agar_label': al,
                    })
                # Gibson first, then its transformations by creation
                wo_rows.sort(key=lambda r: (0 if r['kind'] == 'assembly' else 1, r['_csort']))
                csort = min((r['_csort'] for r in wo_rows), default=pd.Timestamp.max)
                for r in wo_rows:
                    r.pop('_csort', None)
                # only keep an attempt that produced colony data somewhere
                col = [r for r in wo_rows if r['hascol']]
                if not col:
                    continue
                # Attempt verdict = the assembly's pipeline-computed chain_status
                # (single source of truth shared with the tracking tab): a CANCELED
                # Gibson with a SUCCEEDED transformation reads SUCCEEDED. Fall back to
                # the best wo-row status for parquet snapshots predating the column.
                _chain = str(gib.iloc[0].get('chain_status') or '') if not gib.empty else ''
                att_status = _chain if _chain else min(
                    (r['status'] for r in wo_rows), key=lambda s: _STATUS_RANK.get(s, 8))
                atts.append({
                    'root':    str(root),
                    'att_num': att_num,
                    'status':  att_status,
                    '_csort':  csort,
                    'date':    (wo_rows[0]['created'] if wo_rows else ''),  # assembly date (wo_rows is assembly-first)
                    'strains': sorted({r['strain'] for r in col if r['strain']}),
                    'imaged':  sum(r['imaged']   for r in col),
                    'pickable':sum(r['pickable'] for r in col),
                    'picked':  sum(r['picked']   for r in col),
                    'seq':     sum(r['seq']      for r in col),
                    'tot':     sum(r['totc']     for r in col),
                    'low_pick':any(r['low_pick'] for r in col),
                    'by_strain': _by_strain(col),
                    'rows':    wo_rows,
                })
            # number only the attempts that actually did work (colony-producing),
            # chronologically — CANCELED no-work resubmissions are not counted.
            atts.sort(key=lambda a: (a['att_num'] if a['att_num'] is not None else 99, a['_csort']))
            n_att = len(atts)
            for i, a in enumerate(atts, 1):
                a['n']     = i
                a['tot_n'] = n_att
                a.pop('_csort', None)

            all_col = [r for a in atts for r in a['rows'] if r['hascol']]
            anchor_row = asm_all[asm_all['workorder_id'].astype(str) == str(anchor)]
            src = anchor_row.iloc[0] if not anchor_row.empty else (asm_all.iloc[0] if not asm_all.empty else dg.iloc[0])
            # Design verdict = best status across its colony-producing attempts. Each
            # attempt status is the assembly's pipeline-computed chain_status (rolls up
            # the Gibson/GG + its transformations), so a design whose anchor assembly is
            # FAILED/CANCELED but which produced a seq-confirmed clone downstream reads
            # SUCCEEDED — the same single-source value the tracking tab uses (e.g.
            # pAI-21725 6e4af7eb: GG FAILED, transformation 5f5ae4b1 SUCCEEDED → SUCCEEDED).
            # Falls back to the anchor's own chain_status, then visual_status, when no
            # attempt produced colony data.
            if atts:
                d_status = min((a['status'] for a in atts), key=lambda s: _STATUS_RANK.get(s, 8))
            else:
                _sc = src.get('chain_status')
                _sc = str(_sc) if (_sc is not None and not (isinstance(_sc, float) and pd.isna(_sc))) else ''
                d_status = _sc or str(src.get('visual_status', '') or '')
            # Skip DRAFT designs — unsubmitted BIOS draft plans (data_source='BIOS_DRAFT')
            # have no colonies to track, so they're noise in the Colony Tracking view.
            if d_status == 'DRAFT':
                continue
            cr_min = dg['wo_created_at'].min()
            designs.append({
                'anchor':   str(anchor),
                'dtype':    _DTYPE_LABEL.get(src.get('type'), ''),
                'backbone': _pp(src.get('backbone')),
                'parts':    _pp(src.get('parts')),
                'status':   d_status,
                'n_attempts': n_att,
                '_csort':   pd.Timestamp(cr_min) if pd.notna(cr_min) else pd.Timestamp.max,
                'strains':  sorted({r['strain'] for r in all_col if r['strain']}),
                'has_winner': any(r['seq'] > 0 for r in all_col),
                'imaged':   sum(r['imaged']   for r in all_col),
                'pickable': sum(r['pickable'] for r in all_col),
                'picked':   sum(r['picked']   for r in all_col),
                'seq':      sum(r['seq']      for r in all_col),
                'tot':      sum(r['totc']     for r in all_col),
                'low_pick': any(r['low_pick'] for r in all_col),
                'by_strain': _by_strain(all_col),
                'attempts': atts,
            })
        # Match the tracking tab's section order (dashboard.py:1419):
        # winner first → status rank → newest first.
        designs.sort(key=lambda d: (not d['has_winner'],
                                    _STATUS_RANK.get(d['status'], 8),
                                    -d['_csort'].value))
        for d in designs:
            d.pop('_csort', None)

        # ── L1 flags ──
        cflags: list = []
        if any(d['low_pick'] for d in designs):
            cflags.append('LOW_PICKABLE')
        if r_picked > 0 and r_seq == 0 and last_op is not None:
            try:
                stale_days = (pd.Timestamp(today, tz='UTC') - last_op).days
            except Exception:
                stale_days = 0
            if stale_days > PipelineConfig.SEQ_STALL_DAYS:
                cflags.append('SEQ_STALLED')

        out[req_id] = {
            'col': {
                'imaged': r_imaged, 'pickable': r_pick, 'picked': r_picked,
                'seq': r_seq, 'tot': r_tot, 'has_winner': has_winner,
                'cflags': cflags,
            },
            'designs': designs,
        }
    return out


# ── Milestone math ────────────────────────────────────────────────────────────

def _last_ngs_before(dt: date) -> date:
    for i in range(7):
        d = dt - timedelta(days=i)
        if d.weekday() in (0, 3):
            return d
    return dt


def _milestones(created_at, for_partner: bool) -> dict:
    try:
        cd = pd.Timestamp(created_at).date()
    except Exception:
        return {}
    weeks = 5 if for_partner else 6
    # Find the last NGS run (Mon or Thu) that falls within the delivery window
    deadline = cd + timedelta(weeks=weeks)
    ngs = _last_ngs_before(deadline - timedelta(days=1))
    return {
        'assembly':     ngs - timedelta(days=13),
        'asm_ngs':      ngs - timedelta(days=6),
        'lsp_scaleup':  ngs - timedelta(days=5),
        'lsp_received': ngs - timedelta(days=3),
        'lsp_ngs':      ngs,
        'due_date':     ngs + timedelta(days=1),  # LFC release = day after LSP NGS
    }


_DEFAULT_EXCLUDED_EXP  = frozenset()
_DEFAULT_HIDDEN_STATUS = frozenset(['FULFILLED', 'CANCELED'])
_PINNED_EXPS           = frozenset(['LSP Refill Requests', 'A469-Build DNASC CHO Destination Vectors'])

# ── Main renderer ─────────────────────────────────────────────────────────────

def render_inflight_tab(df: pd.DataFrame) -> str:
    today = date.today()

    pai_map: dict = {}
    if 'fulfills_request' in df.columns:
        fulfills = df[
            (df['fulfills_request'] == True) &
            ~df['STOCK_ID'].fillna('').str.startswith('#') &
            df['STOCK_ID'].notna()
        ]
        for req_id, grp in fulfills.groupby('req_id'):
            stocks = grp['STOCK_ID'].dropna().unique().tolist()
            pai_map[req_id] = ', '.join(stocks) if stocks else ''

    base = df[df['req_id'].notna() & ~df['req_id'].isin(['ACTIVE_WIP', 'ORPHAN_LEGACY'])].copy()
    fr_col = base.get('fulfills_request', pd.Series(False, index=base.index))
    req_rows = pd.concat([base[fr_col == True], base]).drop_duplicates(subset='req_id', keep='first')

    active_exps = set(
        req_rows[req_rows['request_status'] == 'IN_PROGRESS']['experiment_name'].dropna().unique()
    )
    req_rows = req_rows[req_rows['experiment_name'].isin(active_exps)].copy()

    # Colony Tracking rollup — built from ALL workorder rows of the active requests
    # so the 3-level (request → design → workorder) structure survives.
    colony_roll = _build_colony_rollup(base, today, req_ids=set(req_rows['req_id'].dropna()))
    _EMPTY_COL = {'imaged': 0, 'pickable': 0, 'picked': 0, 'seq': 0, 'tot': 0,
                  'has_winner': False, 'cflags': []}

    records = []
    for _, row in req_rows.iterrows():
        fp     = str(row.get('for_partner', '')).lower() == 'true'
        ms     = _milestones(row.get('request_created_at'), fp)
        req_id = str(row.get('req_id', ''))
        phase  = str(row.get('req_phase', '') or '')
        op     = str(row.get('req_operation', '') or '')
        status = str(row.get('request_status', '') or '')
        due        = ms.get('due_date')
        asm        = ms.get('assembly')
        lsp_scaleup = ms.get('lsp_scaleup')
        is_stalled = bool(row.get('is_stalled', False))
        flags: list = []
        if status not in ('FULFILLED', 'CANCELED'):
            if due and due < today:
                flags.append('PAST_DUE')
            elif phase in ('ASM', 'PARTS') and (
                (asm and asm < today) or (lsp_scaleup and lsp_scaleup < today)
            ):
                flags.append('AT_RISK')
        if is_stalled and status == 'IN_PROGRESS':
            flags.append('STALLED')
        op_display = '' if is_stalled else op
        _cr = colony_roll.get(req_id, {})
        records.append({
            'exp':       str(row.get('experiment_name', '') or ''),
            'construct': str(row.get('construct_name', '') or ''),
            'pAI':       pai_map.get(req_id, ''),
            'fp':        fp,
            'customer':  str(row.get('customer', '') or ''),
            'submitter': str(row.get('submitter_email', '') or ''),
            'status':    status,
            'phase':     phase,
            'operation': op_display,
            'flags':     flags,
            'req_id':    req_id,
            'due_date':   str(due or ''),
            'assembly':   str(asm or ''),
            'lsp_scaleup':str(lsp_scaleup or ''),
            'pinned':    str(row.get('experiment_name', '') or '') in _PINNED_EXPS,
            'col':       _cr.get('col', _EMPTY_COL),
            'designs':   _cr.get('designs', []),
        })

    # Sort: pinned last; then by due date (soonest first). Experiments are ordered
    # by their earliest due date and within an experiment rows are ordered by due
    # date. exp-level due date and exp come BEFORE the per-row due date so all rows
    # of an experiment stay contiguous (ifRender relies on this). Blank due dates
    # sort last via the '9999-99-99' sentinel.
    _DUE_LAST = '9999-99-99'
    _exp_due = {}
    for r in records:
        due = r['due_date'] or _DUE_LAST
        _exp_due[r['exp']] = min(_exp_due.get(r['exp'], _DUE_LAST), due)
    records.sort(key=lambda r: (
        1 if r['pinned'] else 0,
        _exp_due.get(r['exp'], _DUE_LAST),
        r['exp'],
        r['due_date'] or _DUE_LAST,
        r['assembly'],
    ))

    _ip = [r for r in records if r['status'] == 'IN_PROGRESS']
    in_prog  = len(_ip)
    flagged  = sum(1 for r in _ip if r['flags'])
    past_due = sum(1 for r in _ip if 'PAST_DUE' in r['flags'])
    at_risk  = sum(1 for r in _ip if 'AT_RISK'  in r['flags'])
    stalled  = sum(1 for r in _ip if 'STALLED'  in r['flags'])

    data_json          = json.dumps(records, ensure_ascii=False)
    excl_exp_json      = json.dumps(sorted(_DEFAULT_EXCLUDED_EXP))

    all_exps      = sorted(set(r['exp']       for r in records))
    all_statuses  = sorted(set(r['status']    for r in records if r['status']))
    all_phases    = sorted(set(r['phase']     for r in records if r['phase']))
    all_customers = sorted(set(r['customer']  for r in records if r['customer']))
    all_submitters= sorted(set(r['submitter'] for r in records if r['submitter']))

    btn_s = 'font-size:9px;padding:2px 8px;border-radius:4px;border:1px solid #d1d5db;background:#fff;cursor:pointer;'

    return f"""<style>
.iff-active{{outline:2px solid #374151;}}
.if-vbtn.if-vactive{{background:#374151 !important;color:#fff !important;border-color:#374151 !important;}}
.if-caret{{display:inline-block;width:11px;color:#9ca3af;font-size:9px;transition:transform .1s;cursor:pointer;}}
.if-caret.open{{transform:rotate(90deg);color:#534AB7;}}
.if-att-row{{background:#fafafa;font-size:11px;}}
.if-att-row:hover{{background:#f3f2fb;}}
.if-strain-row{{background:#fafafa;font-size:11px;}}
.if-strain-row:hover{{background:#f3f2fb;}}
.if-cnum{{font-variant-numeric:tabular-nums;text-align:right;font-size:11px;color:#1a1a1a;font-weight:500;}}
.if-cz{{color:#cbd5e1;}}
#inflight-table td:first-child{{padding-left:14px;}}
.if-plate-link{{color:#185FA5;text-decoration:none;}}
.if-plate-link:hover{{text-decoration:underline;}}
/* construct "cards": light gap between constructs + top border on each card */
.if-cardgap td{{height:8px;padding:0 !important;background:#f1f0f7;border:none !important;}}
.if-cardtop td{{border-top:1px solid #e0e0e0;background:#fff;}}
.if-cardtop td:first-child{{border-left:1px solid #e0e0e0;}}
.if-cardtop td:last-child{{border-right:1px solid #e0e0e0;}}
</style>
<div style="padding:12px 16px;background:#fff;min-height:100%;">

  <!-- Summary bar -->
  <div style="display:flex;gap:14px;align-items:center;margin-bottom:10px;flex-wrap:wrap;font-size:10px;color:#6b7280;">
    <span style="font-weight:700;color:#374151;">IN PROGRESS: <span style="color:#1d4ed8;">{in_prog}</span></span>
    <span>Flagged: <b style="color:#b45309;">{flagged}</b></span>
    <span>Past Due: <b style="color:#991b1b;">{past_due}</b></span>
    <span>At Risk: <b style="color:#713f12;">{at_risk}</b></span>
    <span>Stalled: <b style="color:#dc2626;">{stalled}</b></span>
  </div>

  <!-- View toggle -->
  <div style="display:flex;gap:6px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
    <span style="font-size:10px;color:#6b7280;font-weight:600;">View:</span>
    <button onclick="ifSetView('standard')" id="if-v-standard" class="if-vbtn if-vactive" style="{btn_s}">Standard View</button>
    <button onclick="ifSetView('colony')"   id="if-v-colony"   class="if-vbtn"            style="{btn_s}">Colony Tracking View</button>
    <span id="if-colony-hint" style="display:none;font-size:9px;color:#9ca3af;">Click a request to expand designs → workorders</span>
  </div>

  <!-- Flag filter bar -->
  <div style="display:flex;gap:6px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
    <span style="font-size:10px;color:#6b7280;font-weight:600;">Show:</span>
    <button onclick="ifFlagFilter('all')"      id="iff-all"      class="iff-fbtn iff-active" style="{btn_s}font-weight:700;">All</button>
    <button onclick="ifFlagFilter('ip')"       id="iff-ip"       class="iff-fbtn"            style="{btn_s}">IN PROGRESS</button>
    <button onclick="ifFlagFilter('flagged')"  id="iff-flagged"  class="iff-fbtn"            style="{btn_s}">All Flags</button>
    <button onclick="ifFlagFilter('PAST_DUE')" id="iff-PAST_DUE" class="iff-fbtn"            style="{btn_s}background:#fee2e2;color:#991b1b;border-color:#fca5a5;">Past Due</button>
    <button onclick="ifFlagFilter('AT_RISK')"  id="iff-AT_RISK"  class="iff-fbtn"            style="{btn_s}background:#fef9c3;color:#713f12;border-color:#fde047;">At Risk</button>
    <button onclick="ifFlagFilter('STALLED')"  id="iff-STALLED"  class="iff-fbtn"            style="{btn_s}background:#fef2f2;color:#dc2626;border-color:#fca5a5;">Stalled</button>
  </div>

  <!-- Table -->
  <div style="overflow-x:auto;">
    <table id="inflight-table" style="width:100%;border-collapse:collapse;">
      <thead id="inflight-thead"></thead>
      <tbody id="inflight-tbody"><tr><td colspan="14" style="padding:20px;color:#6b7280;font-size:11px;">Loading…</td></tr></tbody>
    </table>
  </div>
</div>

<!-- Shared column-filter dropdown (position:fixed, avoids overflow clipping) -->
<div id="if-col-dd" style="display:none;position:fixed;background:#fff;border:1px solid #d1d5db;
     border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,.15);padding:8px 10px;z-index:9999;
     min-width:200px;max-height:320px;overflow-y:auto;"></div>

<script>
(function() {{
  var _TODAY   = '{today.isoformat()}';
  var _IFD     = {data_json};
  var _ALL_EXP  = {json.dumps(all_exps)};
  var _ALL_ST   = {json.dumps(all_statuses)};
  var _ALL_PH   = {json.dumps(all_phases)};
  var _ALL_CUST = {json.dumps(all_customers)};
  var _ALL_SUBM = {json.dumps(all_submitters)};
  var _EXCL_EXP = {excl_exp_json};

  // ── Filter state — ifRender NEVER reads from DOM ──────────────────────────
  var _flt = {{
    status:    new Set(_ALL_ST),
    exp:       (function(){{ var s=new Set(_ALL_EXP); _EXCL_EXP.forEach(function(e){{s.delete(e);}}); return s; }})(),
    phase:     null,   // null = show all
    fp:        null,   // null = show all; true/false = filter
    flag:      'all',  // 'all','flagged','ip','PAST_DUE','AT_RISK','STALLED'
    // text search (lowercase, empty = no filter)
    construct: '', pAI: '', customer: '', submitter: '', operation: '', req_id: '',
  }};

  // ── Color maps ────────────────────────────────────────────────────────────
  // Reverted to the old Requests-tab palette (matches dashboard.py .status-*).
  var _ST_GRAY  ='background:#f5f5f7;color:#6b7280;border:1px solid #d1d5db;';  // fallback / canceled / unknown
  var S_ST = {{
    'IN_PROGRESS':'background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;',
    'RUNNING':    'background:#f5f3ff;color:#6d28d9;border:1px solid #ddd6fe;',
    'LSP_RUNNING':'background:#f5f3ff;color:#6d28d9;border:1px solid #ddd6fe;',
    'READY':      'background:#fff7ed;color:#c2410c;border:1px solid #fed7aa;',
    'SUCCEEDED':  'background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;',
    'FULFILLED':  'background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;',
    'FAILED':     'background:#fff1f5;color:#be185d;border:1px solid #fecdd3;',
    'BLOCKED':    'background:#be185d;color:white;border:none;',
    'CANCELED':   'background:#f5f5f7;color:#6b7280;border:1px solid #d1d5db;',
    'DRAFT':      'background:#f1f5f9;color:#64748b;border:1px solid #cbd5e1;',
    'WAITING':    'background:#fffbeb;color:#d97706;border:1px solid #fde68a;',
    'PLANNED':    'background:#f5f3ff;color:#6d28d9;border:1px solid #ddd6fe;',
    'UNKNOWN':    'background:#f5f5f7;color:#6b7280;border:1px solid #d1d5db;',
  }};
  var P_ST = {{
    'LSP':  'background:#059669;color:#fff;border:1px solid #047857;',
    'ASM':  'background:#2563eb;color:#fff;border:1px solid #1d4ed8;',
    'PARTS':'background:#ea580c;color:#fff;border:1px solid #c2410c;',
  }};
  var F_ST = {{
    'PAST_DUE':'background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;',
    'AT_RISK': 'background:#ffedd5;color:#7c2d12;border:1px solid #fdba74;',
    'STALLED': 'background:#fef2f2;color:#dc2626;border:1px solid #fca5a5;',
  }};
  var F_BG = {{}};
  var BDG  = 'display:inline-block;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:700;white-space:nowrap;margin:1px 1px;';
  var PILL = BDG;
  var TD   = 'padding:6px 14px;border-bottom:0.5px solid #eeecf6;vertical-align:top;font-size:10px;';

  function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}}
  function bdg(t,st){{return '<span style="'+BDG+st+'">'+esc(t)+'</span>';}}
  // status pill (12px, rounded) — shared by both views so they match.
  function statusBdg(s){{return '<span style="'+PILL+(S_ST[s]||_ST_GRAY)+'">'+esc(String(s).replace(/_/g,' '))+'</span>';}}
  var PAI_STY   ='display:inline-block;background:#ede9fe;color:#6d28d9;border:1px solid #c4b5fd;padding:1px 4px;border-radius:2px;font-family:monospace;font-weight:700;font-size:9px;white-space:nowrap;margin:1px 1px;';
  var PAI_STY_RD='display:inline-block;background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;padding:1px 4px;border-radius:2px;font-family:monospace;font-weight:700;font-size:9px;white-space:nowrap;margin:1px 1px;';
  function paiBadges(s,cust){{if(!s)return'';var st=cust==='R_D'?PAI_STY_RD:PAI_STY;return s.split(',').map(function(p){{p=p.trim();return p?'<span style="'+st+'">'+esc(p)+'</span>':'';}}).join('');}}
  var CUST_MAP={{'R_D':['R&D','#f0fdf4','#166534'],'INTERNAL_CLD':['CLD','#dbeafe','#1d4ed8'],'TECH_OUT':['Tech Out','#ffedd5','#c2410c'],'EXTERNAL_TECH_OUT':['Ext TechOut','#fce7f3','#be185d']}};
  function custBadge(s,fp){{var m=CUST_MAP[s]||['—','#f3f4f6','#6b7280'];return'<span style="padding:2px 6px;border-radius:3px;font-size:10px;background:'+m[1]+';color:'+m[2]+';">'+m[0]+'</span>';}}
  var _DPILL='display:inline-block;padding:0px 5px;border-radius:3px;font-size:9px;font-weight:600;white-space:nowrap;margin-top:2px;';
  function fmtDate(s){{if(!s)return'';var diff=Math.round((new Date(s)-new Date(_TODAY))/(864e5));var bg,clr,lbl;if(diff<0){{bg='#fee2e2';clr='#991b1b';lbl=Math.abs(diff)+'d ago';}}else if(diff===0){{bg='#fef3c7';clr='#92400e';lbl='today';}}else if(diff<=7){{bg='#fef9c3';clr='#713f12';lbl='in '+diff+'d';}}else{{bg='#f3f4f6';clr='#6b7280';lbl='in '+diff+'d';}}return'<span style="color:#374151;">'+esc(s)+'</span><br><span style="background:'+bg+';color:'+clr+';'+_DPILL+'">'+lbl+'</span>';}}
  function fmtSubmitter(s){{if(!s||s.indexOf('@')===-1)return esc(s);var parts=s.split('@');var local=parts[0];var domain=parts[1];var org=domain.split('.')[0];org=org.charAt(0).toUpperCase()+org.slice(1);var name=local.split('.').map(function(p){{return p.charAt(0).toUpperCase()+p.slice(1);}}).join(' ');var ext=!domain.toLowerCase().startsWith('asimov.');var orgSty=ext?'display:inline-block;font-size:9px;font-weight:600;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;border-radius:3px;padding:1px 5px;margin-top:1px;':'display:block;color:#9ca3af;font-size:9px;';return'<span style="display:block;">'+esc(name)+'</span><span style="'+orgSty+'">'+esc(org)+'</span>';}}

  // ── Colony Tracking view state + helpers ──────────────────────────────────
  var _view = 'standard';                 // 'standard' | 'colony'
  var _expR = {{}};                        // expanded request ids   {{req_id: true}}
  var _expA = {{}};                        // expanded attempts      {{req_id|n: true}}
  try {{ if (localStorage.getItem('if_view') === 'colony') _view = 'colony'; }} catch(e) {{}}

  // Competent-cell / strain chips (item 7) — muted outlined.
  var STRAIN_STY = {{
    'NEBV':    'background:#EAF3DE;color:#3B6D11;border:0.5px solid #97C459;',
    'NEB_STBL':'background:#EAF3DE;color:#3B6D11;border:0.5px solid #97C459;',
    'EPI400':  'background:#E6F1FB;color:#185FA5;border:0.5px solid #85B7EB;',
    'STBL3':   'background:#FAEEDA;color:#633806;border:0.5px solid #EF9F27;',
  }};
  var STRAIN_CHIP='display:inline-block;font-size:11px;padding:2px 7px;border-radius:5px;font-weight:500;white-space:nowrap;margin:1px 1px;';
  function strainBdg(s){{var st=STRAIN_STY[s]||'background:#F1EFE8;color:#5F5E5A;border:0.5px solid #D3D1C7;';return '<span style="'+STRAIN_CHIP+st+'">'+esc(s)+'</span>';}}
  // Flag chips (item 8) — muted, thin border.
  var CF_ST = {{
    'LOW_PICKABLE':'background:#FAEEDA;color:#633806;border:0.5px solid #EF9F27;',
    'SEQ_STALLED': 'background:#FAEEDA;color:#633806;border:0.5px solid #EF9F27;',
    'PAST_DUE':    'background:#FCEBEB;color:#A32D2D;border:0.5px solid #E24B4A;',
    'AT_RISK':     'background:#FAEEDA;color:#633806;border:0.5px solid #EF9F27;',
  }};
  var CF_LBL = {{'LOW_PICKABLE':'LOW COLONIES','SEQ_STALLED':'SEQ STALLED','PAST_DUE':'PAST DUE'}};
  // L1 colony flags = colony flags + PAST_DUE inherited from the request flags
  function colFlags(r){{var f=(r.col.cflags||[]).slice();if(r.flags.indexOf('PAST_DUE')!==-1)f.push('PAST_DUE');return f;}}
  // Numeric cell — neutral dark, right-aligned, no conditional red (item 9).
  function num(n,low){{n=n||0;var c=(n===0)?'#c8c6bf':'#1a1a1a';return '<span style="display:block;text-align:right;font-variant-numeric:tabular-nums;font-size:11px;font-weight:500;color:'+c+';">'+n+'</span>';}}
  // Passing-ratio pill (item 4) — colored by seq performance.
  var SEQPILL='display:inline-block;font-size:11px;padding:3px 8px;border-radius:20px;font-weight:500;white-space:nowrap;';
  function seqBdg(seq,tot,winner,status,picked){{
    seq=seq||0;tot=tot||0;picked=picked||0;
    // Sequencing not done yet: 0 confirmations while the design/attempt is still
    // active is PENDING, not a 0/N failure (which would read as a dead run).
    var term=(status==='FAILED'||status==='CANCELED'||status==='SUCCEEDED'||status==='FULFILLED');
    if(seq===0 && !term){{
      // Nothing picked yet → nothing is awaiting sequencing. PENDING only once
      // colonies have actually been picked; otherwise show a neutral dash.
      if(picked<=0) return '<span style="color:#cbd5e1;">&mdash;</span>';
      return '<span style="'+SEQPILL+'background:#F0FDF4;color:#166534;border:0.5px solid #BBF7D0;">PENDING</span>';
    }}
    var pct=tot>0?(seq/tot):0, sty;
    if(seq===0)        sty='background:#FCEBEB;color:#A32D2D;';
    else if(pct<0.20)  sty='background:#FAEEDA;color:#633806;';
    else               sty='background:#EAF3DE;color:#3B6D11;';
    var b='<span style="'+SEQPILL+sty+'">'+seq+'/'+tot+'</span>';
    if(winner) b+='<span style="'+BDG+'background:#EAF3DE;color:#3B6D11;border:0.5px solid #97C459;">&#10003; clone</span>';
    return b;
  }}
  // MM/DD/YYYY (no time) for date columns
  function fmtMDY(s){{if(!s)return'<span class="if-cz">—</span>';var p=s.split('-');return p.length===3?(p[1]+'/'+p[2]+'/'+p[0]):esc(s);}}
  function agarLink(u,l){{return u?'<a href="'+esc(u)+'" target="_blank" class="if-plate-link" style="font-size:10px;">'+esc(l)+'</a>':'';}}

  // ── Row filter ────────────────────────────────────────────────────────────
  function _pass(r) {{
    if (!_flt.status.has(r.status))           return false;
    if (_flt.exp && !_flt.exp.has(r.exp))     return false;
    if (_flt.phase !== null && !_flt.phase.has(r.phase)) return false;
    if (_flt.fp !== null && r.fp !== _flt.fp) return false;
    var ff = _flt.flag;
    if      (ff === 'ip')      {{ if (r.status !== 'IN_PROGRESS') return false; }}
    else if (ff === 'flagged') {{ if (!r.flags.length)             return false; }}
    else if (ff !== 'all')     {{ if (r.flags.indexOf(ff) === -1)  return false; }}
    var q;
    if ((q=_flt.construct) && r.construct.toLowerCase().indexOf(q)===-1) return false;
    if ((q=_flt.pAI)       && r.pAI.toLowerCase().indexOf(q)===-1)       return false;
    if ((q=_flt.customer)  && r.customer.toLowerCase().indexOf(q)===-1)  return false;
    if ((q=_flt.submitter) && r.submitter.toLowerCase().indexOf(q)===-1) return false;
    if ((q=_flt.operation) && r.operation.toLowerCase().indexOf(q)===-1) return false;
    if ((q=_flt.req_id)    && r.req_id.toLowerCase().indexOf(q)===-1)    return false;
    return true;
  }}

  // ── Render (rebuilds tbody from _IFD using _flt) ──────────────────────────
  // Buckets passing rows by experiment before rendering so column sorts never
  // produce duplicate experiment headers or scattered rows.
  function _rowHtml(r) {{
    var bg='';
    for(var fi=0;fi<r.flags.length;fi++){{if(F_BG[r.flags[fi]]){{bg='background:'+F_BG[r.flags[fi]]+';';break;}}}}
    var fps = r.fp ? 'color:#7c3aed;font-weight:700;' : '';
    var st  = statusBdg(r.status);
    var ph  = (r.phase && P_ST[r.phase]) ? bdg(r.phase, P_ST[r.phase]) : '';
    var fl  = r.flags.map(function(f){{return bdg(f.replace(/_/g,' '),F_ST[f]||F_ST['STALLED']);}}).join('');
    return '<tr style="'+bg+'">'
          + '<td style="'+TD+fps+'">'+(r.fp?'★':'')+'</td>'
          + '<td style="'+TD+'max-width:160px;overflow-wrap:break-word;word-break:break-word;">'+esc(r.exp)+'</td>'
          + '<td style="'+TD+'max-width:180px;overflow-wrap:break-word;word-break:break-word;">'+esc(r.construct)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+paiBadges(r.pAI,r.customer)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+custBadge(r.customer,r.fp)+'</td>'
          + '<td style="'+TD+'max-width:110px;">'+fmtSubmitter(r.submitter)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+st+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+ph+'</td>'
          + '<td style="'+TD+'max-width:160px;overflow-wrap:break-word;word-break:break-word;">'+esc(r.operation)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fl+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.assembly)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.lsp_scaleup)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.due_date)+'</td>'
          + '<td style="'+TD+'font-family:monospace;font-size:9px;color:#9ca3af;overflow-wrap:anywhere;">'+esc(r.req_id)+'</td>'
          + '</tr>';
  }}
  // ── Colony Tracking row builders (L1 request → L2 design → L3 attempt → L4 wo) ──
  function _dash(){{ return '<span style="display:block;text-align:right;color:#cbd5e1;font-size:11px;">&mdash;</span>'; }}
  // Per-strain breakdown line (so the summed totals can be split by strain at a glance).
  function strainSummary(bs) {{
    if (!bs || !bs.length) return '';
    return '<div style="margin-top:3px;">' + bs.map(function(s) {{
      return '<span style="display:inline-block;margin-right:12px;font-size:9px;white-space:nowrap;">'
        + strainBdg(s.strain)
        + ' <span style="color:#475569;font-variant-numeric:tabular-nums;">'+s.pickable+'pk &middot; '+s.picked+'pkd &middot; '+s.seq+'/'+s.tot+'</span></span>';
    }}).join('') + '</div>';
  }}
  function _colReqRow(r) {{
    var open = !!_expR[r.req_id], c = r.col;
    var fps = r.fp ? 'color:#7c3aed;font-weight:700;' : '';
    var ph  = (r.phase && P_ST[r.phase]) ? bdg(r.phase, P_ST[r.phase]) : '';
    return '<tr class="if-cardtop" style="cursor:pointer;font-weight:600;" onclick="ifToggleReq(\\''+r.req_id+'\\')">'
      + '<td style="'+TD+'"><span class="if-caret'+(open?' open':'')+'">&#9654;</span></td>'
      + '<td style="'+TD+fps+'">'+(r.fp?'★':'')+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+paiBadges(r.pAI,r.customer)+'</td>'
      + '<td style="'+TD+'max-width:180px;overflow-wrap:break-word;word-break:break-word;">'+esc(r.construct)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+custBadge(r.customer,r.fp)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(r.status)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+ph+'</td>'
      + '<td style="'+TD+'">'+num(c.pickable)+'</td>'
      + '<td style="'+TD+'">'+num(c.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(c.seq,c.tot,c.has_winner,r.status,c.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;color:#64748b;">'+fmtMDY(r.assembly)+'</td>'
      + '</tr>';
  }}
  // L2 — DESIGN (the triangle): one per attempt_anchor_id = distinct backbone+parts.
  function _colDesignRow(r, d) {{
    var hasAtt = (d.attempts||[]).length > 0;
    var open  = hasAtt && !!_expA[r.req_id+'|'+d.anchor];
    var natt  = d.n_attempts>1 ? ' <span style="font-size:9px;color:#64748b;">&times;'+d.n_attempts+' attempts</span>' : '';
    var bp    = [d.backbone, d.parts].filter(Boolean).join(', ');
    var parts = bp ? '<div style="font-size:8px;font-family:monospace;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:300px;">'+esc(bp)+'</div>' : '';
    // ✓ flags a seq-confirmed downstream clone — redundant when the design itself
    // already reads SUCCEEDED/FULFILLED, so only show it for FAILED/CANCELED designs.
    var win   = (d.has_winner && d.status!=='SUCCEEDED' && d.status!=='FULFILLED') ? '<span style="'+BDG+'background:#dcfce7;color:#15803d;border:1px solid #86efac;">&#10003;</span>' : '';
    var caret = hasAtt ? '<span class="if-caret'+(open?' open':'')+'">&#9654;</span>' : '<span style="color:#e5e7eb;">&bull;</span>';
    var click = hasAtt ? ' style="cursor:pointer;" onclick="ifToggleDesign(\\''+r.req_id+'\\',\\''+d.anchor+'\\')"' : '';
    return '<tr class="if-att-row"'+click+'>'
      + '<td style="'+TD+'padding-left:20px;">'+caret+'</td>'
      + '<td style="'+TD+'" colspan="4"><span style="font-size:10px;font-weight:700;color:#334155;">'+esc(d.dtype||'Design')+'</span>'+natt+parts+strainSummary(d.by_strain)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(d.status)+win+'</td>'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'">'+num(d.pickable)+'</td>'
      + '<td style="'+TD+'">'+num(d.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(d.seq,d.tot,false,d.status,d.picked)+'</td>'
      + '<td style="'+TD+'"></td>'
      + '</tr>';
  }}
  // L3 — ATTEMPT banner ("Gibson — Attempt N of M" + verdict). Only when >1 attempt.
  function _attBanner(a) {{
    return '<tr class="if-strain-row" style="background:#f8fafc;">'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'padding-left:38px;" colspan="4"><span style="font-size:9px;font-weight:700;color:#475569;">Gibson &mdash; Attempt '+a.n+' of '+a.tot_n+'</span></td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(a.status)+'</td>'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'">'+num(a.pickable)+'</td>'
      + '<td style="'+TD+'">'+num(a.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(a.seq,a.tot,false,a.status,a.picked)+'</td>'
      + '<td style="'+TD+'"></td>'
      + '</tr>';
  }}
  // L4 — workorder within an attempt: Gibson row, then its &#9492;&#9472; transformations.
  // Phase column shows the agar plate &middot; well coordinate.
  function _colWoRow(w) {{
    var pad = w.is_child ? 66 : 52;
    var pre = w.is_child ? '<span style="color:#cbd5e1;">&#9492;&#9472; </span>' : '';
    var lbl = pre + (w.strain?strainBdg(w.strain)+' ':'') + '<span style="font-size:9px;color:#64748b;">'+esc(w.dtype)+'</span>'
            + ' <span style="font-size:8px;font-family:monospace;color:#cbd5e1;">'+esc(String(w.wid).slice(0,8))+'</span>';
    var c8,c9,c10;
    if (!w.hascol) {{ c8=_dash(); c9=_dash(); c10='<span style="color:#cbd5e1;">&mdash;</span>'; }}
    else {{ c8=num(w.pickable); c9=num(w.picked); c10=seqBdg(w.seq,w.totc,false,w.status,w.picked); }}
    return '<tr class="if-strain-row">'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'padding-left:'+pad+'px;" colspan="4">'+lbl+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(w.status)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;">'+agarLink(w.agar_url,w.agar_label)+'</td>'
      + '<td style="'+TD+'">'+c8+'</td>'
      + '<td style="'+TD+'">'+c9+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+c10+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;color:#64748b;">'+fmtMDY(w.star_date)+'</td>'
      + '</tr>';
  }}
  // Strain cards for one attempt: one card per strain — tag, agar plate link, and
  // Pkl/Pkd (always shown, including 0|0). The assembly (Gibson/GG) row carries a strain
  // (e.g. NEB_STBL) + its own colonies, so include it alongside the transformation
  // rows (EPI400/STBL3) — not just is_child.
  function strainCards(a){{
    var rows=(a.rows||[]).filter(function(w){{return w.strain;}});
    if(!rows.length) rows=(a.rows||[]);
    if(!rows.length) return '';
    return '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:5px;">'+rows.map(function(w){{
      var metrics = '<span style="font-size:10px;color:#64748b;">Pkl: <strong style="color:#1a1a1a;">'+(w.pickable||0)+'</strong> | Pkd: <strong style="color:#1a1a1a;">'+(w.picked||0)+'</strong></span>';
      return '<div style="display:flex;flex-direction:column;gap:3px;min-width:130px;background:#fdfdfd;border:1px solid #f1f5f9;padding:6px;border-radius:4px;">'
        + (w.strain?strainBdg(w.strain):'')
        + (w.agar_label?agarLink(w.agar_url,w.agar_label):'<span style="font-size:10px;color:#cbd5e1;">&mdash;</span>')
        + metrics
        + '</div>';
    }}).join('')+'</div>';
  }}
  // One row per attempt (mock layout): "<Method> — Attempt N of M" + strain cards,
  // with attempt-level status / pickable / picked / seq / date in the columns.
  function _colAttemptRow(a, dtype){{
    var lbl = '<span style="font-size:10px;font-weight:700;color:#334155;">'+esc(dtype||'Assembly')+' &mdash; Attempt '+a.n+' of '+a.tot_n+'</span>';
    return '<tr class="if-att-row">'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'padding-left:38px;" colspan="4">'+lbl+strainCards(a)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(a.status)+'</td>'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'">'+num(a.pickable)+'</td>'
      + '<td style="'+TD+'">'+num(a.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(a.seq,a.tot,false,a.status,a.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;color:#64748b;">'+fmtMDY(a.date)+'</td>'
      + '</tr>';
  }}
  function _colonyRows(r) {{
    var html = '<tr class="if-cardgap"><td colspan="11"></td></tr>' + _colReqRow(r);
    if (_expR[r.req_id]) {{
      (r.designs||[]).forEach(function(d) {{
        html += _colDesignRow(r, d);
        if (_expA[r.req_id+'|'+d.anchor]) {{
          var atts = d.attempts||[];
          atts.forEach(function(a){{
            html += _colAttemptRow(a, d.dtype);
          }});
        }}
      }});
      if (!(r.designs||[]).length)
        html += '<tr class="if-att-row"><td style="'+TD+'"></td><td colspan="10" style="'+TD+'color:#9ca3af;font-style:italic;">No colony data yet.</td></tr>';
    }}
    return html;
  }}

  window.ifToggleReq = function(id) {{ if (_expR[id]) delete _expR[id]; else _expR[id] = true; window.ifRender(); }};
  window.ifToggleDesign = function(id, anchor) {{ var k = id+'|'+anchor; if (_expA[k]) delete _expA[k]; else _expA[k] = true; window.ifRender(); }};
  window.ifSetView = function(v) {{
    _view = v;
    try {{ localStorage.setItem('if_view', v); }} catch(e) {{}}
    var s = document.getElementById('if-v-standard'), c = document.getElementById('if-v-colony');
    if (s) s.classList.toggle('if-vactive', v === 'standard');
    if (c) c.classList.toggle('if-vactive', v === 'colony');
    var h = document.getElementById('if-colony-hint'); if (h) h.style.display = v === 'colony' ? 'inline' : 'none';
    window.ifBuildHead();
    window.ifRender();
  }};

  window.ifRender = function() {{
    window.ifBuildHead();
    var tbody = document.getElementById('inflight-tbody');
    if (!tbody) return;
    var COLONY = _view === 'colony', NCOL = COLONY ? 11 : 14;
    var expOrder = [], buckets = {{}};
    _IFD.forEach(function(r) {{
      if (!_pass(r)) return;
      if (!buckets.hasOwnProperty(r.exp)) {{ expOrder.push(r.exp); buckets[r.exp] = {{rows:[], fp:r.fp, pinned:r.pinned}}; }}
      buckets[r.exp].rows.push(r);
    }});
    var html = '';
    expOrder.forEach(function(exp) {{
      var g = buckets[exp];
      var grpSt = 'background:#F8F7FF;border-left:3px solid #7F77DD;font-weight:600;color:#3C3489;';
      html += '<tr class="if-grp"><td colspan="'+NCOL+'" style="padding:10px 14px;font-size:11px;'+grpSt+'">'
            + esc(exp) + (g.fp ? ' ★' : '') + '</td></tr>';
      g.rows.forEach(function(r) {{ html += COLONY ? _colonyRows(r) : _rowHtml(r); }});
    }});
    if (!html) html = '<tr><td colspan="'+NCOL+'" style="padding:20px;color:#6b7280;font-size:11px;text-align:center;">No matching requests.</td></tr>';
    tbody.innerHTML = html;
  }};

  // ── Sort — sorts _IFD then re-renders (group headers rebuild correctly) ───
  var _sortKey = null, _sortDir = 1;
  // numeric accessor for the colony-sum columns
  function _cval(r, k) {{
    if (k === 'c_pickable') return r.col.pickable;
    if (k === 'c_picked')   return r.col.picked;
    if (k === 'c_seq')      return r.col.tot ? (r.col.seq / r.col.tot) : -1;
    return 0;
  }}
  function _ifSort(k) {{
    if (k === '_caret') return;
    _sortDir = (_sortKey === k) ? _sortDir * -1 : 1;
    _sortKey = k;
    var numeric = k.indexOf('c_') === 0;
    _IFD.sort(function(a,b) {{
      if (a.pinned !== b.pinned) return a.pinned ? 1 : -1;
      if (numeric) return _sortDir * ((_cval(a,k)||0) - (_cval(b,k)||0));
      var va = Array.isArray(a[k]) ? a[k].join(',') : String(a[k]||'');
      var vb = Array.isArray(b[k]) ? b[k].join(',') : String(b[k]||'');
      return _sortDir * va.localeCompare(vb);
    }});
    window.ifRender();
    // update sort indicator in headers
    document.querySelectorAll('#inflight-thead th').forEach(function(th) {{
      var lbl = th.getAttribute('data-col');
      var ind = th.querySelector('.if-sort-ind');
      if (ind) ind.textContent = (lbl === k) ? (_sortDir===1 ? ' ↑' : ' ↓') : ' ↕';
    }});
  }}

  // ── Column-filter dropdown ────────────────────────────────────────────────
  var _ddCol = null;
  function _openColDD(colKey, thEl) {{
    var dd = document.getElementById('if-col-dd');
    if (!dd) return;
    if (_ddCol === colKey && dd.style.display !== 'none') {{
      dd.style.display = 'none'; _ddCol = null; return;
    }}
    _ddCol = colKey;
    dd.innerHTML = _ddContent(colKey);
    var rect = thEl.getBoundingClientRect();
    dd.style.left = Math.max(4, Math.min(rect.left, window.innerWidth - 224)) + 'px';
    dd.style.top  = (rect.bottom + 2) + 'px';
    dd.style.display = 'block';
    var inp = dd.querySelector('input[type=text]');
    if (inp) setTimeout(function(){{inp.focus();}}, 0);
  }}
  function _closeDD() {{
    var d = document.getElementById('if-col-dd');
    if (d) d.style.display = 'none';
    _ddCol = null;
  }}
  document.addEventListener('click', function(e) {{
    var d = document.getElementById('if-col-dd');
    if (!d || d.style.display === 'none') return;
    if (!d.contains(e.target) && !e.target.classList.contains('if-fi')) _closeDD();
  }});

  function _ddContent(k) {{
    var bs = 'width:100%;font-size:9px;padding:3px 6px;border:1px solid #d1d5db;border-radius:4px;margin-bottom:4px;box-sizing:border-box;';
    var hd = '<div style="font-size:9px;font-weight:700;margin-bottom:6px;color:#374151;">Filter: ' + esc(k) + '</div>';

    // Text search columns
    if (['construct','pAI','customer','submitter','operation','req_id'].indexOf(k) !== -1) {{
      var cur = _flt[k] || '';
      return hd
        + '<input type="text" style="'+bs+'" placeholder="Search..." value="'+esc(cur)+'"'
        + ' oninput="_ifSetTxt(\\''+k+'\\',this.value)">'
        + (cur ? '<button onclick="_ifSetTxt(\\''+k+'\\',\\'\\');document.querySelector(\\'#if-col-dd input\\').value=\\'\\'" style="font-size:9px;color:#6b7280;background:none;border:none;cursor:pointer;padding:2px 0;">✕ Clear</button>' : '');
    }}

    // Partner filter
    if (k === 'fp') {{
      return hd
        + _radio('fp','null',  _flt.fp===null,  'All')
        + _radio('fp','true',  _flt.fp===true,  '★ Partner only')
        + _radio('fp','false', _flt.fp===false, 'Non-partner only');
    }}

    // Set-based filters
    var vals, fset, allVals;
    if (k === 'status')   {{ vals = _ALL_ST;   fset = _flt.status; allVals = _ALL_ST; }}
    else if (k === 'exp') {{ vals = _ALL_EXP;  fset = _flt.exp;    allVals = _ALL_EXP; }}
    else if (k === 'phase'){{ vals = _ALL_PH;  fset = _flt.phase;  allVals = _ALL_PH; }}
    else return '';

    var h = hd
      + '<div style="display:flex;gap:4px;margin-bottom:6px;">'
      + '<button onclick="_ifSetAll(\\''+k+'\\',true)"  style="font-size:8px;padding:1px 6px;border:1px solid #d1d5db;border-radius:3px;cursor:pointer;background:#fff;">All</button>'
      + '<button onclick="_ifSetAll(\\''+k+'\\',false)" style="font-size:8px;padding:1px 6px;border:1px solid #d1d5db;border-radius:3px;cursor:pointer;background:#fff;">None</button>'
      + '</div>';
    vals.forEach(function(v) {{
      var chk = (fset === null) || fset.has(v);
      h += '<label style="display:flex;align-items:center;gap:5px;font-size:9px;padding:2px 0;cursor:pointer;">'
         + '<input type="checkbox" class="if-cbx" data-col="'+k+'" data-val="'+esc(v)+'" '+(chk?'checked ':'')
         + 'onchange="_ifToggle(this.dataset.col,this.dataset.val,this.checked)">'
         + esc(v) + '</label>';
    }});
    return h;
  }}

  function _radio(name, val, checked, lbl) {{
    return '<label style="display:flex;align-items:center;gap:5px;font-size:9px;padding:2px 0;cursor:pointer;">'
      + '<input type="radio" name="if-'+name+'" value="'+val+'" '+(checked?'checked ':'')
      + 'onchange="window._ifFpChange(this.value)"> ' + esc(lbl) + '</label>';
  }}
  window._ifFpChange = function(v) {{
    _flt.fp = v==='null' ? null : v==='true';
    window.ifRender();
  }};

  window._ifSetTxt = function(k, v) {{
    _flt[k] = v.toLowerCase();
    window.ifRender();
  }};
  window._ifToggle = function(k, v, checked) {{
    if (k === 'status') {{ if (checked) _flt.status.add(v); else _flt.status.delete(v); }}
    else if (k === 'exp') {{ if (_flt.exp) {{ if (checked) _flt.exp.add(v); else _flt.exp.delete(v); }} }}
    else if (k === 'phase') {{
      if (_flt.phase === null) _flt.phase = new Set(_ALL_PH);
      if (checked) _flt.phase.add(v); else _flt.phase.delete(v);
    }}
    window.ifRender();
  }};
  window._ifSetAll = function(k, v) {{
    if      (k === 'status') _flt.status = v ? new Set(_ALL_ST)  : new Set();
    else if (k === 'exp')    _flt.exp    = v ? new Set(_ALL_EXP) : new Set();
    else if (k === 'phase')  _flt.phase  = v ? null : new Set();
    document.querySelectorAll('#if-col-dd .if-cbx').forEach(function(cb){{cb.checked=v;}});
    window.ifRender();
  }};
  window.ifFlagFilter = function(f) {{
    _flt.flag = f;
    // "All" = show every status; "IN PROGRESS" = narrow to just IN_PROGRESS
    if (f === 'all') {{
      _flt.status = new Set(_ALL_ST);
    }} else if (f === 'ip') {{
      _flt.status = new Set(['IN_PROGRESS']);
      _flt.flag   = 'all';   // flag filter is irrelevant once status is narrowed
    }}
    document.querySelectorAll('.iff-fbtn').forEach(function(b){{b.classList.remove('iff-active');b.style.fontWeight='';}});
    var btn = document.getElementById('iff-'+(f==='ip'?'ip':f));
    if (btn) {{ btn.classList.add('iff-active'); btn.style.fontWeight='700'; }}
    window.ifRender();
  }};

  // ── Build column headers (rebuilds when the view changes) ─────────────────
  var _headView = null;
  var _COLS_STD = [
      {{k:'fp',        lbl:'★',         filter:true}},
      {{k:'exp',       lbl:'Experiment', filter:true}},
      {{k:'construct', lbl:'Construct',  filter:true}},
      {{k:'pAI',       lbl:'pAI',        filter:true}},
      {{k:'customer',  lbl:'Customer',   filter:true}},
      {{k:'submitter', lbl:'Submitter',  filter:true}},
      {{k:'status',    lbl:'Status',     filter:true}},
      {{k:'phase',     lbl:'Phase',      filter:true}},
      {{k:'operation', lbl:'Operation',  filter:true}},
      {{k:'flags',     lbl:'Flags',      filter:false}},
      {{k:'assembly',   lbl:'Assembly',    filter:false}},
      {{k:'lsp_scaleup',lbl:'LSP Scale-up',filter:false}},
      {{k:'due_date',   lbl:'Due Date',   filter:false}},
      {{k:'req_id',    lbl:'Req ID',     filter:true}},
  ];
  var _COLS_COL = [
      {{k:'_caret',    lbl:'',              filter:false}},
      {{k:'fp',        lbl:'★',             filter:true}},
      {{k:'pAI',       lbl:'pAI',           filter:true}},
      {{k:'construct', lbl:'Construct',     filter:true}},
      {{k:'customer',  lbl:'Customer',      filter:true}},
      {{k:'status',    lbl:'Status',        filter:true}},
      {{k:'phase',     lbl:'Phase / Agar',  filter:true}},
      {{k:'c_pickable',lbl:'Pickable',      filter:false}},
      {{k:'c_picked',  lbl:'Picked',        filter:false}},
      {{k:'c_seq',     lbl:'Seq Conf',      filter:false}},
      {{k:'assembly',  lbl:'Assembly',      filter:false}},
  ];
  window.ifBuildHead = function() {{
    if (_headView === _view) return;
    _headView = _view;
    var thead = document.getElementById('inflight-thead');
    if (!thead) return;
    thead.innerHTML = '';
    var TH = 'padding:5px 6px;text-align:left;border-bottom:2px solid #d1d5db;font-size:9px;color:#374151;'
           + 'font-weight:700;text-transform:uppercase;background:#f9fafb;position:sticky;top:0;z-index:2;'
           + 'white-space:nowrap;cursor:pointer;user-select:none;';
    var COLS = (_view === 'colony') ? _COLS_COL : _COLS_STD;
    var tr = document.createElement('tr');
    COLS.forEach(function(col) {{
      var th = document.createElement('th');
      th.style.cssText = TH;
      th.setAttribute('data-col', col.k);
      th.addEventListener('click', function(e) {{
        if (e.target.classList.contains('if-fi')) return;
        _ifSort(col.k);
      }});
      var lbl = document.createElement('span');
      lbl.textContent = col.lbl;
      th.appendChild(lbl);
      var ind = document.createElement('span');
      ind.className = 'if-sort-ind';
      ind.style.cssText = 'color:#9ca3af;font-size:9px;';
      ind.textContent = ' ↕';
      th.appendChild(ind);
      if (col.filter) {{
        var fi = document.createElement('span');
        fi.className = 'if-fi';
        fi.title = 'Filter';
        fi.textContent = ' ▾';
        fi.style.cssText = 'color:#9ca3af;font-size:10px;cursor:pointer;padding:0 2px;';
        fi.addEventListener('click', function(e) {{
          e.stopPropagation();
          _openColDD(col.k, th);
        }});
        th.appendChild(fi);
      }}
      tr.appendChild(th);
    }});
    thead.appendChild(tr);
  }};

  // Reflect a restored (localStorage) view in the toggle buttons on load.
  (function() {{
    var s = document.getElementById('if-v-standard'), c = document.getElementById('if-v-colony');
    if (s) s.classList.toggle('if-vactive', _view === 'standard');
    if (c) c.classList.toggle('if-vactive', _view === 'colony');
    var h = document.getElementById('if-colony-hint'); if (h && _view === 'colony') h.style.display = 'inline';
  }})();

}})();
</script>
"""

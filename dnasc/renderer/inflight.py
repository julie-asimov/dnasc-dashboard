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
from datetime import date, datetime, timedelta

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc import protocols as proto
from dnasc.renderer import tokens as tok


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
    return _milestones_from_ngs(ngs, ngs + timedelta(days=1))


def _milestones_from_ngs(ngs: date, due: date) -> dict:
    """Build the milestone chain off a known LSP-NGS date and due date."""
    return {
        'assembly':     ngs - timedelta(days=13),
        'asm_ngs':      ngs - timedelta(days=6),
        'lsp_scaleup':  ngs - timedelta(days=5),
        'lsp_received': ngs - timedelta(days=3),
        'lsp_ngs':      ngs,
        'due_date':     due,
    }


def _milestones_from_due(due: date) -> dict:
    """
    Milestone chain anchored on a curated override due date (mirrors the tracking
    tab): NGS = last Mon/Thu before the due date, assembly = NGS − 13 days. The
    override due date is authoritative — not re-derived as NGS + 1.
    """
    ngs = _last_ngs_before(due - timedelta(days=1))
    return _milestones_from_ngs(ngs, due)


def _parse_override_due(raw) -> date | None:
    """Normalize a due_dates.json entry (str | dict | list) to a date, or None."""
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw
    elif isinstance(raw, dict):
        s = raw.get('due_date', '')
    elif isinstance(raw, list):
        s = raw[0].get('due_date', '') if raw and isinstance(raw[0], dict) else ''
    else:
        s = ''
    s = str(s or '').strip()
    if not s or s in ('nan', 'None'):
        return None
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except Exception:
        return None


_DEFAULT_EXCLUDED_EXP  = frozenset()
_DEFAULT_HIDDEN_STATUS = frozenset(['FULFILLED', 'CANCELED'])
_PINNED_EXPS           = frozenset(['LSP Refill Requests', 'A469-Build DNASC CHO Destination Vectors'])

# ── Main renderer ─────────────────────────────────────────────────────────────

def render_inflight_tab(df: pd.DataFrame) -> str:
    today = date.today()

    # Curated due-date overrides (experiment_name → {due_date, ...}), same source
    # the tracking tab uses. When present, the due date and the assembly milestone
    # are taken from the override instead of the created_at + N-weeks formula.
    try:
        from dnasc.extractors.sheets import load_due_dates
        _due_date_map = load_due_dates()
    except Exception:
        _due_date_map = {}

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
        exp_name = str(row.get('experiment_name', '') or '')
        # Override due date (if curated) re-anchors due + assembly; created_at is
        # left untouched and still drives the fallback when there's no override.
        _ov_due = _parse_override_due(_due_date_map.get(exp_name))
        ms     = _milestones_from_due(_ov_due) if _ov_due else _milestones(row.get('request_created_at'), fp)
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

    btn_s = 'font-size:12px;padding:4px 11px;border-radius:6px;border:1px solid #e5e7eb;background:#fff;color:#374151;font-weight:500;cursor:pointer;'

    # ── Design tokens → JS maps (single source of truth: renderer/tokens.py) ──
    def _tint(triple):
        bg, txt, bd = triple
        return f"background:{bg};color:{txt};border:1px solid {bd};"

    def _solid(pair):
        return f"background:{pair[0]};color:{pair[1]};"

    def _geo(key):
        g = tok.GEOM[key]
        s = (f"display:inline-block;padding:{g['pad']};border-radius:{g['radius']};"
             f"font-size:{g['size']};font-weight:{g['weight']};white-space:nowrap;margin:1px 1px;")
        if g["upper"]:
            s += "text-transform:uppercase;"
        return s

    _status_map = dict(tok.STATUS)
    _status_map["PLANNED"] = tok.STATUS["RUNNING"]   # legacy state -> brand purple
    JS_S_ST    = "{" + ",".join(f"'{k}':'{_tint(v)}'" for k, v in _status_map.items()) + "}"
    JS_S_ICON  = "{" + ",".join(f"'{k}':'{ic}'" for k, ic in tok.STATUS_ICON.items()) + "}"
    JS_ST_GRAY = _tint(tok.STATUS["CANCELED"])
    JS_P_ST    = "{" + ",".join(f"'{k}':'{_tint(v)}'" for k, v in tok.PHASE.items()) + "}"
    JS_F_ST    = "{" + ",".join(f"'{k}':'{_tint(v)}'" for k, v in tok.FLAG.items()) + "}"
    _cf_keys   = ["LOW_PICKABLE", "SEQ_STALLED", "PAST_DUE", "AT_RISK"]
    JS_CF_ST   = "{" + ",".join(f"'{k}':'{_tint(tok.FLAG[k])}'" for k in _cf_keys) + "}"
    JS_CUST    = "{" + ",".join(f"'{k}':['{lbl}','{bg}','{txt}']"
                                for k, (lbl, bg, txt) in tok.CUSTOMER.items()) + "}"
    GEO_STATUS = _geo("status")
    GEO_PHASE  = _geo("phase")
    GEO_CUST   = _geo("customer")
    _pg = tok.GEOM["pai"]
    PAI_STYLE = (f"display:inline-block;background:{tok.PURPLE_BG_2};color:{tok.PURPLE};"
                 f"border:1px solid {tok.PURPLE_BORDER_2};padding:{_pg['pad']};"
                 f"border-radius:{_pg['radius']};font-family:monospace;font-weight:{_pg['weight']};"
                 f"font-size:{_pg['size']};white-space:nowrap;margin:1px 1px;")
    # R&D pAI badge = blue (master convention), so R&D reads distinct from the
    # purple Partner/other badges. Same geometry as PAI_STYLE.
    PAI_STYLE_RD = (f"display:inline-block;background:#dbeafe;color:#1d4ed8;"
                    f"border:1px solid #93c5fd;padding:{_pg['pad']};"
                    f"border-radius:{_pg['radius']};font-family:monospace;font-weight:{_pg['weight']};"
                    f"font-size:{_pg['size']};white-space:nowrap;margin:1px 1px;")
    CUST_DOT = tok.CUSTOMER_DOT

    return f"""<style>
.iff-active{{background:#eff4ff !important;border-color:#bcd0fb !important;color:#1d4ed8 !important;}}
.if-vbtn.if-vactive{{background:#2563eb !important;color:#fff !important;border-color:#2563eb !important;}}
.if-caret{{display:inline-block;width:13px;color:#9ca3af;font-size:11px;transition:transform .1s;cursor:pointer;}}
.if-caret.open{{transform:rotate(90deg);color:#2563eb;}}
/* Kernel metadata-cloud pill + workbench primitives */
.kpill{{display:inline-flex;align-items:center;gap:5px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:6px;padding:3px 9px;font-size:12px;line-height:1.3;color:#374151;font-weight:500;white-space:nowrap;}}
.kpill .kk{{color:#6b7280;font-weight:500;}}
.kpill b{{font-weight:700;color:#111827;}}
.kbtn:hover{{background:#f9fafb;}}
.iff-fbtn:hover,.if-vbtn:hover{{background:#f9fafb;}}
/* Kernel dropdown overlay rows: muted grey hover, no heavy blue highlight. */
#if-col-dd .if-dd-row:hover{{background:#f3f4f6;}}
.if-att-row{{background:#fafafa;font-size:11px;}}
.if-att-row:hover{{background:#f5f3ff;box-shadow:inset 2px 0 0 #7c3aed;}}
/* Attempt header: tinted band + divider line on top so each attempt reads as a
   distinct group rather than blurring into the strain rows beneath it. */
.if-attempt{{background:#eceef5;font-size:11px;border-top:2px solid #c7cbe0;}}
.if-attempt:hover{{background:#e4e7f1;}}
.if-attempt .if-cnum{{font-weight:700;color:#111;}}
/* Strain rows recede to white so they nest visually under their attempt header. */
.if-strain-row{{background:#fff;font-size:11px;}}
.if-strain-row:hover{{background:#f5f3ff;box-shadow:inset 2px 0 0 #7c3aed;}}
.if-cnum{{display:block;font-variant-numeric:tabular-nums;text-align:right;font-size:11px;color:#1a1a1a;font-weight:500;}}
.if-cnum.if-cz{{color:#c8c6bf;font-weight:500;}}
.if-cz{{color:#cbd5e1;}}
#inflight-table td:first-child{{padding-left:14px;}}
.if-plate-link{{color:#185FA5;text-decoration:none;}}
.if-plate-link:hover{{text-decoration:underline;}}
/* construct "cards": light gap between constructs + top border on each card */
.if-cardgap td{{height:8px;padding:0 !important;background:#f1f0f7;border:none !important;}}
/* pAI anchor row: neutral grey rail (purple is reserved for the project/experiment
   header) + faint tint so the construct stays in focus without looking like a project. */
.if-cardtop td{{border-top:1px solid #ececf1;background:#fbfbfd;}}
.if-cardtop td:first-child{{border-left:4px solid #b9bdc9;}}
.if-cardtop td:last-child{{border-right:1px solid #ececf1;}}
/* Left rail: nested rows share a faint grey spine that ties them back to the
   pAI anchor above, so you don't lose track of which construct you're inside. */
.if-att-row td:first-child,
.if-attempt td:first-child,
.if-strain-row td:first-child{{border-left:4px solid #e3e5ea;}}
</style>
<div style="padding:12px 16px;background:#fff;min-height:100%;">

  <!-- Metadata clouds (Kernel-style) -->
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap;">
    <span class="kpill"><span class="kk">In progress</span><b style="color:#1d4ed8;">{in_prog}</b></span>
    <span class="kpill"><span class="kk">Flagged</span><b id="if-flagged-ct">{flagged}</b></span>
    <span class="kpill"><span class="kk">Past due</span><b style="color:#991b1b;">{past_due}</b></span>
    <span class="kpill"><span class="kk">At risk</span><b style="color:#854d0e;">{at_risk}</b></span>
    <span class="kpill"><span class="kk">Stalled</span><b style="color:#dc2626;">{stalled}</b></span>
    <span class="kpill"><span class="kk">Colony risk</span><b id="if-colrisk-ct" style="color:#991b1b;">0</b></span>
  </div>

  <!-- View toggle -->
  <div style="display:flex;gap:6px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
    <span style="font-size:10px;color:#6b7280;font-weight:600;">View:</span>
    <button onclick="ifSetView('standard')" id="if-v-standard" class="if-vbtn if-vactive" style="{btn_s}">Standard View</button>
    <button onclick="ifSetView('colony')"   id="if-v-colony"   class="if-vbtn"            style="{btn_s}">Colony Tracking View</button>
    <span id="if-colony-hint" style="display:none;font-size:9px;color:#9ca3af;">Click a request to expand designs → workorders</span>
    <span id="if-band-legend" style="display:none;align-items:center;gap:6px;font-size:9px;color:#9ca3af;margin-left:6px;border-left:1px solid #e5e7eb;padding-left:8px;">
      <span style="font-weight:600;color:#6b7280;">Pickable band:</span>
      <span title="0–7 pickable — below the median; bottom half of all workorders" style="display:inline-block;font-size:8px;font-weight:700;padding:0 4px;border-radius:3px;background:#FDE2E2;color:#B42318;border:0.5px solid #F5A3A3;">LOW 0–7</span>
      <span title="8–22 pickable — median up to the 75th percentile; typical / healthy" style="display:inline-block;font-size:8px;font-weight:700;padding:0 4px;border-radius:3px;background:#FEF3C7;color:#92400E;border:0.5px solid #FCD34D;">MED 8–22</span>
      <span title="23+ pickable — top quartile (38+ is top 10%)" style="display:inline-block;font-size:8px;font-weight:700;padding:0 4px;border-radius:3px;background:#DCFCE7;color:#15803D;border:0.5px solid #86EFAC;">HIGH 23+</span>
    </span>
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
    <button onclick="ifFlagFilter('COLONY_RISK')" id="iff-COLONY_RISK" class="iff-fbtn"      style="{btn_s}background:#fee2e2;color:#991b1b;border-color:#fca5a5;">Colony Risk</button>
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
<div id="if-col-dd" style="display:none;position:fixed;background:#fff;border:1px solid #e5e7eb;
     border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.05);padding:6px;z-index:9999;
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

  // ── Color maps — generated from renderer/tokens.py (single source of truth) ─
  var _ST_GRAY  ='{JS_ST_GRAY}';        // fallback (CANCELED gray)
  var S_ST   = {JS_S_ST};               // status tint fragments
  var S_ICON = {JS_S_ICON};             // colorblind icon per status
  var P_ST   = {JS_P_ST};               // phase solid fragments (brand sweep)
  var F_ST   = {JS_F_ST};               // flag tint fragments
  var F_BG = {{}};
  var BDG  = 'display:inline-block;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:700;white-space:nowrap;margin:1px 1px;';
  var GEO_STATUS = '{GEO_STATUS}';
  var GEO_PHASE  = '{GEO_PHASE}';
  var GEO_CUST   = '{GEO_CUST}';
  var PILL = GEO_STATUS;
  var TD   = 'padding:6px 14px;border-bottom:0.5px solid #eeecf6;vertical-align:top;font-size:10px;';

  function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}}
  function bdg(t,st){{return '<span style="'+BDG+st+'">'+esc(t)+'</span>';}}
  // ── Inline Lucide SVG icons (1.5px stroke, currentColor) — no icon package ──
  function lucide(p,sz){{var s=sz||12;return '<svg width="'+s+'" height="'+s+'" viewBox="0 0 24 24" fill="none" '
    +'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" '
    +'style="display:inline-block;vertical-align:-2px;flex-shrink:0;">'+p+'</svg>';}}
  var LU = {{
    check:'<path d="M20 6 9 17l-5-5"/>',
    x:'<path d="M18 6 6 18M6 6l12 12"/>',
    refresh:'<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 4v5h-5"/>',
    clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    hourglass:'<path d="M5 22h14M5 2h14M17 22v-4.2a2 2 0 0 0-.6-1.4L12 12l-4.4 4.4a2 2 0 0 0-.6 1.4V22M7 2v4.2a2 2 0 0 0 .6 1.4L12 12l4.4-4.4A2 2 0 0 0 17 6.2V2"/>',
    ban:'<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>',
    slash:'<circle cx="12" cy="12" r="9"/><path d="m15 9-6 6"/>',
    pencil:'<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    star:'<path d="M12 2l3 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.9 21l1.2-6.8-5-4.9 6.9-1z"/>',
    ext:'<path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h6"/>',
    chevron:'<path d="m9 18 6-6-6-6"/>'
  }};
  var STATUS_LU = {{SUCCEEDED:'check',FULFILLED:'star',FAILED:'x',RUNNING:'refresh',LSP_RUNNING:'refresh',
    REPICK:'refresh',READY:'clock',WAITING:'hourglass',BLOCKED:'ban',CANCELED:'slash',DRAFT:'pencil'}};
  // status badge: Lucide icon (colorblind cue, inherits text color) + label + tint.
  function statusBdg(s){{var k=STATUS_LU[s];var ic=k?lucide(LU[k])+'&nbsp;':'';
    return '<span style="'+PILL+'display:inline-flex;align-items:center;'+(S_ST[s]||_ST_GRAY)+'">'+ic+esc(String(s).replace(/_/g,' '))+'</span>';}}
  // phase pill: solid brand-sweep fill, own geometry.
  function phaseBdg(p){{return P_ST[p]?'<span style="'+GEO_PHASE+P_ST[p]+'">'+esc(p)+'</span>':'';}}
  var PAI_STY   ='{PAI_STYLE}';
  var PAI_STY_RD='{PAI_STYLE_RD}';
  function paiBadges(s,cust){{if(!s)return'';var st=cust==='R_D'?PAI_STY_RD:PAI_STY;return s.split(',').map(function(p){{p=p.trim();return p?'<span style="'+st+'">'+esc(p)+'</span>':'';}}).join('');}}
  var CUST_MAP={JS_CUST};
  // customer badge: optional leading marker (CUST_DOT, from tokens) + label + tint.
  function custBadge(s,fp){{var m=CUST_MAP[s]||['—','#f3f4f6','#6b7280'];return'<span style="'+GEO_CUST+'background:'+m[1]+';color:'+m[2]+';">{CUST_DOT}'+m[0]+'</span>';}}
  var _DPILL='display:inline-block;padding:0px 5px;border-radius:3px;font-size:9px;font-weight:600;white-space:nowrap;margin-top:2px;';
  function fmtDate(s){{if(!s)return'';var diff=Math.round((new Date(s)-new Date(_TODAY))/(864e5));var bg,clr,lbl;if(diff<0){{bg='#fee2e2';clr='#991b1b';lbl=Math.abs(diff)+'d ago';}}else if(diff===0){{bg='#fef3c7';clr='#92400e';lbl='today';}}else if(diff<=7){{bg='#fef9c3';clr='#713f12';lbl='in '+diff+'d';}}else{{bg='#f3f4f6';clr='#6b7280';lbl='in '+diff+'d';}}return'<span style="color:#374151;">'+esc(s)+'</span><br><span style="background:'+bg+';color:'+clr+';'+_DPILL+'">'+lbl+'</span>';}}
  function fmtSubmitter(s){{if(!s||s.indexOf('@')===-1)return esc(s);var parts=s.split('@');var local=parts[0];var domain=parts[1];var org=domain.split('.')[0];org=org.charAt(0).toUpperCase()+org.slice(1);var name=local.split('.').map(function(p){{return p.charAt(0).toUpperCase()+p.slice(1);}}).join(' ');var ext=!domain.toLowerCase().startsWith('asimov.');var orgSty=ext?'display:inline-block;font-size:9px;font-weight:600;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;border-radius:3px;padding:1px 5px;margin-top:1px;':'display:block;color:#9ca3af;font-size:9px;';return'<span style="display:block;">'+esc(name)+'</span><span style="'+orgSty+'">'+esc(org)+'</span>';}}

  // ── Colony Tracking view state + helpers ──────────────────────────────────
  var _view = 'standard';                 // 'standard' | 'colony'
  var _expR = {{}};                        // expanded request ids   {{req_id: true}}
  var _expA = {{}};                        // expanded attempts      {{req_id|n: true}}
  try {{ if (localStorage.getItem('if_view') === 'colony') _view = 'colony'; }} catch(e) {{}}

  // Competent-cell / strain chips (item 7) — distinct, saturated hues so NEB vs EPI
  // are separable at a glance (were both pale pastels that read alike).
  var STRAIN_STY = {{
    'NEBV':    'background:#FBE0EB;color:#A82A5E;border:0.5px solid #ED90B5;',
    'NEB_STBL':'background:#FBE0EB;color:#A82A5E;border:0.5px solid #ED90B5;',
    'EPI400':  'background:#D6F0F2;color:#0E6E7A;border:0.5px solid #57BFCB;',
    'STBL3':   'background:#FAE6C8;color:#8A4B05;border:0.5px solid #E8961B;',
  }};
  var STRAIN_CHIP='display:inline-block;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:600;white-space:nowrap;margin:0 1px;';
  function strainBdg(s){{var st=STRAIN_STY[s]||'background:#F1EFE8;color:#5F5E5A;border:0.5px solid #D3D1C7;';return '<span style="'+STRAIN_CHIP+st+'">'+esc(s)+'</span>';}}
  // Flag chips (item 8) — muted, thin border.
  var CF_ST = {JS_CF_ST};
  var CF_LBL = {{'LOW_PICKABLE':'LOW COLONIES','SEQ_STALLED':'SEQ STALLED','PAST_DUE':'PAST DUE'}};
  // L1 colony flags = colony flags + PAST_DUE inherited from the request flags
  function colFlags(r){{var f=(r.col.cflags||[]).slice();if(r.flags.indexOf('PAST_DUE')!==-1)f.push('PAST_DUE');return f;}}
  // Numeric cell — neutral dark, right-aligned, no conditional red (item 9).
  function num(n,low){{n=n||0;return '<span class="if-cnum'+(n===0?' if-cz':'')+'">'+n+'</span>';}}
  // Pickable risk band for an attempt's pickable count (see legend in the toolbar):
  //   Low 0–7 (below median), Medium 8–22 (median→75th pct), High 23+ (top quartile).
  var PICK_LOW_MAX=7, PICK_MED_MAX=22;
  function pickBand(n){{
    n=n||0;
    var lbl, st;
    if(n<=PICK_LOW_MAX){{lbl='LOW'; st='background:#FDE2E2;color:#B42318;border:0.5px solid #F5A3A3;';}}
    else if(n<=PICK_MED_MAX){{lbl='MED'; st='background:#FEF3C7;color:#92400E;border:0.5px solid #FCD34D;';}}
    else{{lbl='HIGH'; st='background:#DCFCE7;color:#15803D;border:0.5px solid #86EFAC;';}}
    return '<span title="'+n+' pickable — '+lbl+' band" style="display:inline-block;font-size:8px;font-weight:700;padding:0 4px;border-radius:3px;white-space:nowrap;margin-left:6px;vertical-align:middle;'+st+'">'+lbl+'</span>';
  }}
  // Risk level for a design, by the BEST attempt available (its pickable ceiling):
  //   best still LOW (0–PICK_LOW_MAX)      → HIGH RISK (only low options)
  //   best MED (PICK_LOW_MAX–PICK_MED_MAX) → MED RISK  (no strong attempt yet)
  //   best HIGH (>PICK_MED_MAX)            → healthy, no badge
  // Designs with a sequence-confirmed winner / already succeeded are never flagged.
  function designRisk(d){{
    var atts=d.attempts||[];
    if(!atts.length) return '';
    if(d.has_winner || d.status==='SUCCEEDED' || d.status==='FULFILLED') return '';
    var best=0; atts.forEach(function(a){{var p=a.pickable||0; if(p>best) best=p;}});
    if(best<=PICK_LOW_MAX) return 'HIGH';
    if(best<=PICK_MED_MAX) return 'MED';
    return '';
  }}
  function riskBadge(level){{
    if(level!=='HIGH' && level!=='MED') return '';
    var st = level==='HIGH'
      ? 'background:#FEE2E2;color:#991B1B;border:0.5px solid #FCA5A5;'
      : 'background:#FEF3C7;color:#92400E;border:0.5px solid #FCD34D;';
    var tip = level==='HIGH'
      ? 'Every attempt is in the LOW pickable band (0–'+PICK_LOW_MAX+') with no sequence-confirmed colony — at risk of running out of viable picks.'
      : 'Best attempt is only MEDIUM (≤'+PICK_MED_MAX+' pickable) with no sequence-confirmed colony — watch this one.';
    return '<span title="'+tip+'" style="display:inline-block;font-size:8px;font-weight:700;padding:0 4px;border-radius:3px;white-space:nowrap;margin-left:6px;vertical-align:middle;'+st+'">&#9888; '+level+' RISK</span>';
  }}
  // Worst colony risk across a request's designs + the pickable count driving it.
  function reqColRisk(r){{
    var lv='', pk=0;
    (r.designs||[]).forEach(function(d){{
      var rk=designRisk(d);
      if(rk==='HIGH' && lv!=='HIGH'){{ lv='HIGH'; pk=d.pickable||0; }}
      else if(rk==='MED' && lv===''){{ lv='MED'; pk=d.pickable||0; }}
    }});
    return {{level:lv, pick:pk}};
  }}
  // Colony-risk flag badge (for the standard-view Flags column) — shows severity AND
  // the pickable colony count so the standard view carries the colony info too.
  function colRiskFlag(level,pick){{
    if(level!=='HIGH' && level!=='MED') return '';
    var st = level==='HIGH' ? 'background:#FEE2E2;color:#991B1B;border:1px solid #FCA5A5;'
                            : 'background:#FEF3C7;color:#92400E;border:1px solid #FCD34D;';
    var tip = level==='HIGH'
      ? 'Colony at risk: every attempt LOW (0–'+PICK_LOW_MAX+' pickable), no seq-confirmed clone. '+pick+' pickable.'
      : 'Colony watch: best attempt only MEDIUM (≤'+PICK_MED_MAX+' pickable), no seq-confirmed clone. '+pick+' pickable.';
    return '<span title="'+tip+'" style="'+BDG+st+'">'+level+' RISK &middot; '+pick+'pk</span>';
  }}
  // One-time: fold colony risk into each record's flags so it filters/sorts like the
  // other flags (and "All Flags" includes it). Idempotent via the indexOf guard.
  (function(){{
    var _crCt = 0;
    _IFD.forEach(function(r){{
      r.flags = r.flags || [];
      // Colony risk only applies while the request is in assembly (ASM). Past that
      // (LSP/PARTS/etc.) the colony picture is no longer the actionable signal.
      var cr = (r.phase === 'ASM') ? reqColRisk(r) : {{level:'', pick:0}};
      r._colRisk = cr.level; r._colPick = cr.pick;
      if(cr.level && r.flags.indexOf('COLONY_RISK')===-1) {{ r.flags.push('COLONY_RISK'); _crCt++; }}
    }});
    var _el = document.getElementById('if-colrisk-ct');
    if(_el) _el.textContent = _crCt;
    // "Flagged" total now includes colony risk — recompute as any-flag count.
    var _fl = document.getElementById('if-flagged-ct');
    if(_fl) _fl.textContent = _IFD.filter(function(r){{return r.flags.length;}}).length;
  }})();
  // Passing-ratio pill (item 4) — colored by seq performance. Shares the status
  // badge geometry (GEO_STATUS) so it sits in scale beside the Status column
  // instead of being an oversized rounded pill.
  var SEQPILL=GEO_STATUS;
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
    var ph  = phaseBdg(r.phase);
    var fl  = r.flags.map(function(f){{
      if(f==='COLONY_RISK') return colRiskFlag(r._colRisk, r._colPick);
      return bdg(f.replace(/_/g,' '),F_ST[f]||F_ST['STALLED']);
    }}).join('');
    return '<tr style="'+bg+'">'
          + '<td style="'+TD+fps+'">'+(r.fp?'★':'')+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+paiBadges(r.pAI,r.customer)+'</td>'
          + '<td title="'+esc(r.construct)+'" style="'+TD+'max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'+esc(r.construct)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+custBadge(r.customer,r.fp)+'</td>'
          + '<td style="'+TD+'max-width:110px;">'+fmtSubmitter(r.submitter)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+st+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+ph+'</td>'
          + '<td style="'+TD+'max-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'+esc(r.operation)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fl+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.assembly)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.lsp_scaleup)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.due_date)+'</td>'
          + '<td style="'+TD+'font-family:monospace;font-size:9px;color:#9ca3af;overflow-wrap:anywhere;">'+esc(r.req_id)+'</td>'
          + '</tr>';
  }}
  // ── Colony Tracking row builders (L1 request → L2 design → L3 attempt → L4 wo) ──
  function _dash(){{ return '<span style="display:block;text-align:right;color:#cbd5e1;font-size:11px;">&mdash;</span>'; }}
  function _colReqRow(r) {{
    var open = !!_expR[r.req_id], c = r.col;
    var fps = r.fp ? 'color:#7c3aed;font-weight:700;' : '';
    var ph  = phaseBdg(r.phase);
    var _lv = (r.designs||[]).map(designRisk);
    var rwarn = riskBadge(_lv.indexOf('HIGH')!==-1 ? 'HIGH' : (_lv.indexOf('MED')!==-1 ? 'MED' : ''));
    return '<tr class="if-cardtop" style="cursor:pointer;font-weight:600;" onclick="ifToggleReq(\\''+r.req_id+'\\')">'
      + '<td style="'+TD+'"><span class="if-caret'+(open?' open':'')+'">&#9654;</span></td>'
      + '<td style="'+TD+fps+'">'+(r.fp?'★':'')+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+paiBadges(r.pAI,r.customer)+'</td>'
      + '<td style="'+TD+'max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'+esc(r.construct)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+custBadge(r.customer,r.fp)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+ph+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+rwarn+'</td>'
      + '<td style="'+TD+'">'+num(c.pickable)+'</td>'
      + '<td style="'+TD+'">'+num(c.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(c.seq,c.tot,c.has_winner,r.status,c.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(r.status)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;color:#64748b;">'+fmtMDY(r.assembly)+'</td>'
      + '</tr>';
  }}
  // L2 — DESIGN (the triangle): one per attempt_anchor_id = distinct backbone+parts.
  function _colDesignRow(r, d) {{
    var hasAtt = (d.attempts||[]).length > 0;
    var open  = hasAtt && !!_expA[r.req_id+'|'+d.anchor];
    var natt  = ' <span style="font-size:9px;color:#64748b;font-weight:600;">&middot; '+d.n_attempts+' attempt'+(d.n_attempts==1?'':'s')+'</span>';
    var bp    = [d.backbone, d.parts].filter(Boolean).join(', ');
    var parts = bp ? '<div style="font-size:8px;font-family:monospace;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:300px;">'+esc(bp)+'</div>' : '';
    // ✓ flags a seq-confirmed downstream clone — redundant when the design itself
    // already reads SUCCEEDED/FULFILLED, so only show it for FAILED/CANCELED designs.
    var win   = (d.has_winner && d.status!=='SUCCEEDED' && d.status!=='FULFILLED') ? '<span style="'+BDG+'background:#dcfce7;color:#15803d;border:1px solid #86efac;">&#10003;</span>' : '';
    var caret = hasAtt ? '<span class="if-caret'+(open?' open':'')+'">&#9654;</span>' : '<span style="color:#e5e7eb;">&bull;</span>';
    var click = hasAtt ? ' style="cursor:pointer;" onclick="ifToggleDesign(\\''+r.req_id+'\\',\\''+d.anchor+'\\')"' : '';
    // Single-attempt design: the design row IS that attempt, so band it here. Multi-
    // attempt designs band each attempt row instead (the design total is a sum).
    var band = ((d.attempts||[]).length <= 1) ? pickBand(d.pickable) : '';
    var warn = riskBadge(designRisk(d));
    return '<tr class="if-att-row"'+click+'>'
      + '<td style="'+TD+'padding-left:20px;">'+caret+'</td>'
      + '<td style="'+TD+'" colspan="4"><span style="font-size:10px;font-weight:700;color:#334155;">'+esc(d.dtype||'Design')+'</span>'+natt+parts+'</td>'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'white-space:nowrap;">'+(warn||band)+'</td>'
      + '<td style="'+TD+'">'+num(d.pickable)+'</td>'
      + '<td style="'+TD+'">'+num(d.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(d.seq,d.tot,false,d.status,d.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(d.status)+win+'</td>'
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
    var c8,c9,c10,rb;
    if (!w.hascol) {{ c8=_dash(); c9=_dash(); c10='<span style="color:#cbd5e1;">&mdash;</span>'; rb=''; }}
    else {{ c8=num(w.pickable); c9=num(w.picked); c10=seqBdg(w.seq,w.totc,false,w.status,w.picked); rb=pickBand(w.pickable); }}
    return '<tr class="if-strain-row">'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'padding-left:'+pad+'px;" colspan="4">'+lbl+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;">'+agarLink(w.agar_url,w.agar_label)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+rb+'</td>'
      + '<td style="'+TD+'">'+c8+'</td>'
      + '<td style="'+TD+'">'+c9+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+c10+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(w.status)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;color:#64748b;">'+fmtMDY(w.star_date)+'</td>'
      + '</tr>';
  }}
  // Attempt header row: "<Method> — Attempt N of M" + attempt-level totals in the
  // right columns. The per-strain breakdown is emitted as aligned sub-rows below
  // (see strainRows) rather than floated as cards, so the data lines up under the
  // PICKABLE / PICKED / SEQ headers instead of leaving the middle of the row empty.
  function _colAttemptRow(a, dtype){{
    var lbl = '<span style="font-size:10px;font-weight:700;color:#334155;">'+esc(dtype||'Assembly')+' &mdash; Attempt '+a.n+' of '+a.tot_n+'</span>';
    return '<tr class="if-attempt">'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'padding-left:38px;" colspan="4">'+lbl+'</td>'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'white-space:nowrap;">'+pickBand(a.pickable)+'</td>'
      + '<td style="'+TD+'">'+num(a.pickable)+'</td>'
      + '<td style="'+TD+'">'+num(a.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(a.seq,a.tot,false,a.status,a.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(a.status)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;color:#64748b;">'+fmtMDY(a.date)+'</td>'
      + '</tr>';
  }}
  // Per-strain sub-rows for one attempt, rendered as aligned table rows via _colWoRow
  // (strain tag + agar plate&middot;well in the Phase column + Pickable/Picked/Seq under
  // their headers). Mirrors strainCards' row selection: strain rows, else all rows.
  function strainRows(a){{
    var rows=(a.rows||[]).filter(function(w){{return w.strain;}});
    if(!rows.length) rows=(a.rows||[]);
    return rows.map(_colWoRow).join('');
  }}
  function _colonyRows(r) {{
    var html = '<tr class="if-cardgap"><td colspan="12"></td></tr>' + _colReqRow(r);
    if (_expR[r.req_id]) {{
      (r.designs||[]).forEach(function(d) {{
        html += _colDesignRow(r, d);
        if (_expA[r.req_id+'|'+d.anchor]) {{
          var atts = d.attempts||[];
          // Single attempt: the design row already carries the totals + "· 1 attempt",
          // so skip the redundant "Attempt 1 of 1" header and show strain rows directly.
          var multi = atts.length > 1;
          atts.forEach(function(a){{
            html += (multi ? _colAttemptRow(a, d.dtype) : '') + strainRows(a);
          }});
        }}
      }});
      if (!(r.designs||[]).length)
        html += '<tr class="if-att-row"><td style="'+TD+'"></td><td colspan="11" style="'+TD+'color:#9ca3af;font-style:italic;">No colony data yet.</td></tr>';
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
    var bl = document.getElementById('if-band-legend'); if (bl) bl.style.display = v === 'colony' ? 'inline-flex' : 'none';
    window.ifBuildHead();
    window.ifRender();
  }};

  window.ifRender = function() {{
    window.ifBuildHead();
    var tbody = document.getElementById('inflight-tbody');
    if (!tbody) return;
    var COLONY = _view === 'colony', NCOL = COLONY ? 12 : 13;
    var expOrder = [], buckets = {{}};
    _IFD.forEach(function(r) {{
      if (!_pass(r)) return;
      if (!buckets.hasOwnProperty(r.exp)) {{ expOrder.push(r.exp); buckets[r.exp] = {{rows:[], fp:r.fp, pinned:r.pinned, customer:r.customer}}; }}
      buckets[r.exp].fp = buckets[r.exp].fp || r.fp;   // partner if ANY request is for_partner
      buckets[r.exp].rows.push(r);
    }});
    var html = '';
    expOrder.forEach(function(exp) {{
      var g = buckets[exp];
      // Project (experiment) header — neutralized into the flat Kernel system:
      // white surface, bold dark title, hairline separator, Partner as a quiet pill.
      var partnerPill = g.fp
        ? ' <span style="font-weight:600;font-size:10px;color:#6d28d9;background:#f5f3ff;'
          + 'border:1px solid #ddd6fe;border-radius:5px;padding:1px 7px;margin-left:8px;">Partner</span>'
        : '';
      html += '<tr class="if-grp"><td colspan="'+NCOL+'" style="padding:16px 14px 7px;font-size:13px;'
            + 'font-weight:700;color:#111827;background:#fff;border-top:1px solid #e5e7eb;">'
            + esc(exp) + partnerPill + '</td></tr>';
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
    var bs = 'width:100%;font-size:9px;padding:5px 8px;border:1px solid #e5e7eb;border-radius:6px;margin-bottom:4px;box-sizing:border-box;';
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
      + '<button onclick="_ifSetAll(\\''+k+'\\',true)"  style="font-size:8px;padding:1px 6px;border:1px solid #e5e7eb;border-radius:3px;cursor:pointer;background:#fff;">All</button>'
      + '<button onclick="_ifSetAll(\\''+k+'\\',false)" style="font-size:8px;padding:1px 6px;border:1px solid #e5e7eb;border-radius:3px;cursor:pointer;background:#fff;">None</button>'
      + '</div>';
    vals.forEach(function(v) {{
      var chk = (fset === null) || fset.has(v);
      h += '<label class="if-dd-row" style="display:flex;align-items:center;gap:5px;font-size:9px;padding:4px 6px;border-radius:5px;cursor:pointer;">'
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
      {{k:'pAI',       lbl:'pAI',        filter:true}},
      {{k:'construct', lbl:'Construct',  filter:true}},
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
      {{k:'phase',     lbl:'Phase / Agar',  filter:true}},
      {{k:'risk',      lbl:'Risk',          filter:false}},
      {{k:'c_pickable',lbl:'Pickable',      filter:false}},
      {{k:'c_picked',  lbl:'Picked',        filter:false}},
      {{k:'c_seq',     lbl:'Seq Conf',      filter:false}},
      {{k:'status',    lbl:'Status',        filter:true}},
      {{k:'assembly',  lbl:'Assembly',      filter:false}},
  ];
  window.ifBuildHead = function() {{
    if (_headView === _view) return;
    _headView = _view;
    var thead = document.getElementById('inflight-thead');
    if (!thead) return;
    thead.innerHTML = '';
    var TH = 'padding:7px 8px;text-align:left;border-bottom:1px solid #cbd5e1;font-size:9px;color:#0f172a;letter-spacing:0.04em;'
           + 'font-weight:700;text-transform:uppercase;background:#f3f4f6;position:sticky;top:0;z-index:2;'
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
    var bl = document.getElementById('if-band-legend'); if (bl && _view === 'colony') bl.style.display = 'inline-flex';
  }})();

}})();
</script>
"""

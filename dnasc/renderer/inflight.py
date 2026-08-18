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
import re
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
# In-flight visual statuses — an attempt with any of these is still "live" and is
# surfaced even before it has colony data (e.g. a resubmitted assembly whose
# transformations haven't imaged yet). Dead no-work resubmissions (all CANCELED/
# FAILED, no colonies) are still dropped.
_ACTIVE_VS = frozenset({'RUNNING', 'READY', 'IN_PROGRESS', 'WAITING', 'BLOCKED', 'LSP_RUNNING'})
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
    """Compact a 'd8004:True, d8073:False' backbone/parts string to 'd8004, d8073'.
    The ':True/:False' is the BIOS `available` (inventory) flag — it does NOT define
    what the design is, so EVERY declared component is shown (a design is the design
    whether or not its parts are on hand). Only strips the flag suffix."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ''
    out = []
    for tok in str(s).split(','):
        tok = tok.strip()
        if not tok:
            continue
        pid = tok.partition(':')[0].strip()
        if pid:
            out.append(pid)
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


# Furthest OpTracker protocol an attempt has reached. protocol_name is a LIST column
# (one row can carry several protocols), so membership must be tested per element —
# comparing the stringified array to a name silently matches nothing.
#
# Colony counts land partway through MINIPREP ("Create Minipreps and Glycerol Stocks"),
# so this is what distinguishes "no count because it has not got there yet" from
# "no count because nobody entered it": 91 of 4449 attempts reached miniprep with the
# counts still missing.
_STAGE_PROTOS = [('miniprep',       (proto.MINIPREP, proto.REPICK)),
                 ('transformation', (proto.STAR_TRANSF, proto.TRANSFORMATION)),
                 ('assembly',       (proto.GOLDEN_GATE, proto.GIBSON))]


def _attempt_stage(ag) -> str:
    """'miniprep' | 'transformation' | 'assembly' | '' (nothing started) for an attempt."""
    names = set()
    for v in ag.get('protocol_name', []):
        if v is None:
            continue
        try:
            names.update(str(x) for x in v)
        except TypeError:
            continue
    for label, protos in _STAGE_PROTOS:      # ordered furthest-first
        if names & set(protos):
            return label
    return ''


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
                # Keep an attempt that produced colony data OR is still actively in
                # flight (a resubmitted assembly whose transformations haven't imaged
                # yet). Drop only dead no-work resubmissions (no colonies, nothing live).
                col = [r for r in wo_rows if r['hascol']]
                if not col and not any(r['status'] in _ACTIVE_VS for r in wo_rows):
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
                    'stage_p': _attempt_stage(ag),   # furthest protocol reached
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
# Request-level statuses that count as "in flight" — everything that is not
# FULFILLED/CANCELED. Lifecycle (lsp_capacity._STATUS_RANK): NEW (actively being
# designed) -> PLANNED (design done, no work yet) -> IN_PROGRESS (work started)
# -> REMEDIATION. All four count as in progress.
_ACTIVE_REQ_STATUS = frozenset(['NEW', 'PLANNED', 'IN_PROGRESS', 'REMEDIATION'])
_PINNED_EXPS           = PipelineConfig.PINNED_INFRA_EXPERIMENTS
# Experiments where a trailing _vN construct suffix marks a redo variant that
# should group under its original. `_v2` is overloaded elsewhere (e.g. dep_rep
# and other uses), so this grouping is opt-in per experiment — match is a
# substring of experiment_name. Remove an entry (or empty the tuple) to turn the
# original+v2 grouping off for that experiment.
_VARIANT_GROUP_EXPS = ('VRT002',)

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
        req_rows[req_rows['request_status'].isin(_ACTIVE_REQ_STATUS)]['experiment_name'].dropna().unique()
    )
    req_rows = req_rows[req_rows['experiment_name'].isin(active_exps)].copy()
    # For the pinned reference projects (LSP Refill / DV / A385 RD) only, drop
    # terminal requests so their large FULFILLED/CANCELED history doesn't clutter
    # the in-flight tab. Other experiments still show all their requests.
    req_rows = req_rows[~(
        req_rows['experiment_name'].isin(_PINNED_EXPS)
        & req_rows['request_status'].isin(_DEFAULT_HIDDEN_STATUS)
    )].copy()

    # Colony Tracking rollup — built from ALL workorder rows of the active requests
    # so the 3-level (request → design → workorder) structure survives.
    colony_roll = _build_colony_rollup(base, today, req_ids=set(req_rows['req_id'].dropna()))
    _EMPTY_COL = {'imaged': 0, 'pickable': 0, 'picked': 0, 'seq': 0, 'tot': 0,
                  'has_winner': False, 'cflags': []}

    # Outsourced-LSP detection: when the active LSP prep is out at an external
    # vendor (Aldevron / Azenta), the wait is outside dnasc's control, so we don't
    # count it as PAST DUE. The vendor shows in vendor_order_id (Batch_..._Aldevron_.../
    # ..._Azenta_...) and prep_method ("Aldevron 10mg" / "Genewiz ..." = Azenta) on the
    # active (non-terminal) lsp_workorder row. Genewiz is Azenta's sequencing brand.
    req_vendor_out: dict = {}
    _vout_cols = [c for c in ('vendor_order_id', 'prep_method', 'location',
                              'batch_comments', 'deposited_by', 'digest_note')
                  if c in base.columns]
    if _vout_cols:
        _lsp_active = base[
            (base['type'] == 'lsp_workorder')
            & ~base['visual_status'].isin(('SUCCEEDED', 'FULFILLED', 'FAILED', 'CANCELED'))
        ]
        for _rid, _grp in _lsp_active.groupby('req_id'):
            _blob = ' '.join(_grp[_vout_cols].fillna('').astype(str).values.ravel()).lower()
            if 'aldevron' in _blob:
                req_vendor_out[str(_rid)] = 'Aldevron'
            elif 'azenta' in _blob or 'genewiz' in _blob:
                req_vendor_out[str(_rid)] = 'Azenta'

    # Requests with a BLOCKED part/assembly workorder. Surfaced as its own flag so
    # a build waiting on a stuck part shows BLOCKED alongside its running/ready op
    # (e.g. pAI-22328: 2 PCRs RUNNING + 1 BLOCKED → op "PCR: RUNNING" + BLOCKED badge).
    req_blocked = set(base.loc[base['visual_status'] == 'BLOCKED', 'req_id'].astype(str))

    # A build BLOCKED on a missing part is not idle when that part is already mid-refill — the
    # batch is the only thing that will unblock it, and today that fact lives on the Parts tab
    # while the request here shows an empty Operation next to BLOCKED/STALLED. Same reader as the
    # Parts tab (parts_result.pkl), keyed by the product the blocked workorder makes = the pAI.
    # Optional by design: a missing/failed parts pull just leaves the column as it was.
    try:
        from dnasc.renderer.parts import blocking_refill_progress
        _part_batches = blocking_refill_progress()
    except Exception:
        _part_batches = {}

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
        vendor_out = req_vendor_out.get(req_id, '')
        flags: list = []
        if status not in ('FULFILLED', 'CANCELED'):
            if due and due < today:
                # Out at an external vendor → the delay isn't ours: badge it AT VENDOR
                # instead of PAST DUE so it doesn't count against our TAT.
                flags.append('AT_VENDOR' if vendor_out else 'PAST_DUE')
            elif phase in ('ASM', 'PARTS') and (
                (asm and asm < today) or (lsp_scaleup and lsp_scaleup < today)
            ):
                flags.append('AT_RISK')
            # Separate, additive flag: a stuck part/assembly. Coexists with the op
            # label so "PCR: RUNNING" + BLOCKED can both show.
            if phase in ('ASM', 'PARTS') and req_id in req_blocked:
                flags.append('BLOCKED')
        if is_stalled and status == 'IN_PROGRESS':
            flags.append('STALLED')
        # A finished request has no active operation/phase — blank both so
        # FULFILLED/SUCCEEDED/CANCELED rows don't read as "still in ASM".
        _terminal = status in ('FULFILLED', 'SUCCEEDED', 'CANCELED')
        op_display = '' if (is_stalled or _terminal) else op
        phase_display = '' if _terminal else phase
        _cr = colony_roll.get(req_id, {})
        # Variant grouping: a redo of the same design carries a trailing `_vN`
        # suffix on the construct name (e.g. "...(CO 1.5)" vs "...(CO 1.5)_v2").
        # Strip it to a shared base so the original + its v2/v3 group together —
        # but ONLY for opt-in experiments (_VARIANT_GROUP_EXPS), since `_v2` is
        # used for other things elsewhere.
        construct = str(row.get('construct_name', '') or '')
        _grp = any(tag in exp_name for tag in _VARIANT_GROUP_EXPS)
        _vm  = re.search(r'_(v\d+)$', construct) if _grp else None
        # Refill batches running for the part(s) this request's build is blocked on. A request can
        # carry several pAIs, and one build can be short more than one part, so this is a list.
        _batches: list = []
        for _p in (s.strip() for s in str(pai_map.get(req_id, '')).split(',')):
            for _b in _part_batches.get(_p, []):
                if _b not in _batches: _batches.append(_b)
        records.append({
            'exp':       str(row.get('experiment_name', '') or ''),
            'construct': construct,
            'base':      construct[:_vm.start()] if _vm else construct,
            'variant':   _vm.group(1) if _vm else '',
            'pAI':       pai_map.get(req_id, ''),
            'fp':        fp,
            'customer':  str(row.get('customer', '') or ''),
            'submitter': str(row.get('submitter_email', '') or ''),
            'status':    status,
            'phase':     phase_display,
            'operation': op_display,
            'batches':   _batches,
            'flags':     flags,
            'vendor_out': vendor_out,
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
    _base_due = {}
    _base_cnt = {}
    for r in records:
        due = r['due_date'] or _DUE_LAST
        _exp_due[r['exp']] = min(_exp_due.get(r['exp'], _DUE_LAST), due)
        _bk = (r['exp'], r['base'])
        _base_due[_bk] = min(_base_due.get(_bk, _DUE_LAST), due)
        _base_cnt[_bk] = _base_cnt.get(_bk, 0) + 1

    def _vrank(v):
        try:
            return int(v[1:]) if v else 0
        except Exception:
            return 99

    # Within each experiment, cluster the multi-variant groups (original + v2/v3)
    # first so the lone, ungrouped constructs don't interleave between them; then
    # keep each base-construct group contiguous (by the group's earliest due) and
    # order the original before its v2/v3.
    records.sort(key=lambda r: (
        1 if r['pinned'] else 0,
        _exp_due.get(r['exp'], _DUE_LAST),
        r['exp'],
        0 if _base_cnt.get((r['exp'], r['base']), 0) > 1 else 1,
        _base_due.get((r['exp'], r['base']), _DUE_LAST),
        r['base'],
        _vrank(r['variant']),
        r['due_date'] or _DUE_LAST,
        r['assembly'],
    ))

    _ip = [r for r in records if r['status'] in _ACTIVE_REQ_STATUS]
    in_prog  = len(_ip)
    flagged  = sum(1 for r in _ip if r['flags'])
    past_due  = sum(1 for r in _ip if 'PAST_DUE'  in r['flags'])
    at_vendor = sum(1 for r in _ip if 'AT_VENDOR' in r['flags'])
    at_risk   = sum(1 for r in _ip if 'AT_RISK'   in r['flags'])
    blocked   = sum(1 for r in _ip if 'BLOCKED'   in r['flags'])
    stalled   = sum(1 for r in _ip if 'STALLED'   in r['flags'])

    # Where the in-flight work actually sits, plus what's already done. Static for
    # the whole tab (deliberately NOT filter-reactive like Flagged / Colony risk) —
    # this row is the standing shape of the queue, not a read-out of the current view.
    # Phase is read off _ip because active statuses are never terminal, so their
    # phase label survives the terminal-row blanking above. NEW requests have no
    # phase yet (nothing has been planned), which is why the three phases don't
    # sum to `in_prog` — the remainder is still in design.
    ph_asm       = sum(1 for r in _ip if r['phase'] == 'ASM')
    ph_lsp       = sum(1 for r in _ip if r['phase'] == 'LSP')
    ph_parts     = sum(1 for r in _ip if r['phase'] == 'PARTS')
    ph_design    = in_prog - ph_asm - ph_lsp - ph_parts
    fulfilled_ct = sum(1 for r in records if r['status'] == 'FULFILLED')
    # Total deliberately EXCLUDES CANCELED so the row reconciles: in-flight +
    # fulfilled = total. Canceled requests are still listed in the table (and
    # reachable from the Status filter) — they're just not work we ever owed, so
    # counting them here left an unexplained gap between the tiles.
    total_ct     = sum(1 for r in records if r['status'] != 'CANCELED')

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

    def _tint_nb(triple):
        # Borderless soft tint — for the flag pills (Behind / Colony / etc.) so they
        # read as calm chips, not hard-outlined stickers. Static badges keep _tint.
        bg, txt, _bd = triple
        return f"background:{bg};color:{txt};border:1px solid transparent;"

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
    _status_map["PLANNED"] = tok.STATUS["RUNNING"]   # design done, no work -> in-progress purple
    _status_map["NEW"]     = ("#f1f5f9", "#475569", "#cbd5e1")  # NEW = "In Design" — slate (matches dashboard In Design stage)
    JS_S_ST    = "{" + ",".join(f"'{k}':'{_tint(v)}'" for k, v in _status_map.items()) + "}"
    JS_LU      = json.dumps(tok.LUCIDE_PATHS)            # icon name -> SVG path (single source)
    JS_STAT_LU = json.dumps(tok.STATUS_LUCIDE)           # status -> icon name (single source)
    JS_ST_GRAY = _tint(tok.STATUS["CANCELED"])
    JS_P_ST    = "{" + ",".join(f"'{k}':'{_tint(v)}'" for k, v in tok.PHASE.items()) + "}"
    JS_F_ST    = "{" + ",".join(f"'{k}':'{_tint_nb(v)}'" for k, v in tok.FLAG.items()) + "}"
    _cf_keys   = ["LOW_PICKABLE", "PAST_DUE", "AT_RISK"]
    JS_CF_ST   = "{" + ",".join(f"'{k}':'{_tint_nb(tok.FLAG[k])}'" for k in _cf_keys) + "}"
    JS_CUST    = "{" + ",".join(f"'{k}':['{lbl}','{bg}','{txt}']"
                                for k, (lbl, bg, txt) in tok.CUSTOMER.items()) + "}"
    GEO_STATUS = _geo("status")
    GEO_PHASE  = _geo("phase")
    # For the per-experiment phase split on the experiment header row: the active
    # statuses (single source = _ACTIVE_REQ_STATUS) and the DESIGN chip style,
    # which reuses the NEW status tint exactly like the tab-wide pill does.
    IF_ACT_JSON = json.dumps({s: 1 for s in sorted(_ACTIVE_REQ_STATUS)})
    DESIGN_CHIP = GEO_PHASE + _tint(_status_map['NEW'])
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

    # Colony band + risk thresholds are config, not literals in the JS — see PipelineConfig for
    # how they were calibrated and why the descriptive band and the risk trigger differ.
    PICK_LOW_MAX  = PipelineConfig.PICK_BAND_LOW_MAX
    PICK_MED_MAX  = PipelineConfig.PICK_BAND_MED_MAX
    RISK_HIGH_MAX = PipelineConfig.COLONY_RISK_HIGH_MAX
    RISK_MED_MAX  = PipelineConfig.COLONY_RISK_MED_MAX

    return f"""<style>
.iff-active{{background:#eff4ff !important;border-color:#bcd0fb !important;color:#1d4ed8 !important;}}
.if-vbtn.if-vactive{{background:#2563eb !important;color:#fff !important;border-color:#2563eb !important;}}
.if-caret{{display:inline-block;width:13px;color:#9ca3af;font-size:11px;transition:transform .1s;cursor:pointer;}}
.if-caret.open{{transform:rotate(90deg);color:#2563eb;}}
/* Kernel metadata-cloud pill + workbench primitives */
.kpill{{display:inline-flex;align-items:center;gap:5px;background:#f1f5f9;border:1px solid #e5e7eb;border-radius:6px;padding:3px 9px;font-size:12px;line-height:1.3;color:#374151;font-weight:500;white-space:nowrap;}}
.kpill .kk{{color:#6b7280;font-weight:500;}}
.kpill b{{font-weight:700;color:#111827;}}
.kbtn:hover{{background:#f9fafb;}}
.iff-fbtn:hover,.if-vbtn:hover{{background:#f9fafb;}}
/* Kernel dropdown overlay rows: muted grey hover, no heavy blue highlight. */
#if-col-dd .if-dd-row:hover{{background:#f1f5f9;}}
.if-att-row{{background:#f8fafc;font-size:11px;}}
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
.if-cardtop td{{border-top:1px solid #e0e0e0;background:#fbfbfd;}}
.if-cardtop td:first-child{{border-left:4px solid #b9bdc9;}}
.if-cardtop td:last-child{{border-right:1px solid #e0e0e0;}}
/* Left rail: nested rows share a faint grey spine that ties them back to the
   pAI anchor above, so you don't lose track of which construct you're inside. */
.if-att-row td:first-child,
.if-attempt td:first-child,
.if-strain-row td:first-child{{border-left:4px solid #e3e5ea;}}
/* Variant group: original + its v2/v3 redo share one construct header and a
   common spine so they read as a single grouped section — kept neutral/quiet. */
.if-cgrp td{{background:#f8fafc;border-top:1px solid #ededed;}}
.if-cgrp td:first-child{{border-left:3px solid #cbd5e1;}}
.if-cgrp-mem td:first-child{{border-left:3px solid #e8e8ee;}}
</style>
<div style="padding:12px 16px;background:#e9ecf2;min-height:100%;">

  <!-- Metadata clouds (Kernel-style) -->
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap;">
    <span class="kpill"><span class="kk">In progress</span><b style="color:#1d4ed8;">{in_prog}</b></span>
    <span class="kpill" title="Requests carrying at least one flag. A request can carry several, so the individual flag counts sum higher than this."><span class="kk">Flagged</span><b id="if-flagged-ct">{flagged}</b></span>
    <span class="kpill"><span class="kk">Past due</span><b style="color:#991b1b;">{past_due}</b></span>
    <span class="kpill"><span class="kk">At vendor</span><b style="color:#3730a3;">{at_vendor}</b></span>
    <span class="kpill"><span class="kk">Behind schedule</span><b style="color:#854d0e;">{at_risk}</b></span>
    <span class="kpill"><span class="kk">Blocked</span><b style="color:#b91c1c;">{blocked}</b></span>
    <span class="kpill"><span class="kk">Stalled</span><b style="color:#dc2626;">{stalled}</b></span>
    <span class="kpill"><span class="kk">Colony risk</span><b id="if-colrisk-ct" style="color:#991b1b;">0</b></span>
  </div>

  <!-- Phase split of the in-flight work + fulfilled / tab total. Static for the whole
       tab (does not follow the filter bar) — see the Python comment above. -->
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap;">
    <span class="kpill" style="gap:9px;">
      <span class="kk">In phase</span>
      <span style="display:inline-flex;align-items:center;gap:4px;"><span style="{GEO_PHASE}{_tint(tok.PHASE['ASM'])}">ASM</span><b>{ph_asm}</b></span>
      <span style="display:inline-flex;align-items:center;gap:4px;"><span style="{GEO_PHASE}{_tint(tok.PHASE['LSP'])}">LSP</span><b>{ph_lsp}</b></span>
      <span style="display:inline-flex;align-items:center;gap:4px;"><span style="{GEO_PHASE}{_tint(tok.PHASE['PARTS'])}">PARTS</span><b>{ph_parts}</b></span>
      <span style="display:inline-flex;align-items:center;gap:4px;" title="NEW requests — no phase yet, still in design"><span style="{GEO_PHASE}{_tint(_status_map['NEW'])}">DESIGN</span><b>{ph_design}</b></span>
    </span>
    <span class="kpill" title="Requests in this tab already delivered"><span class="kk">Fulfilled</span><b style="color:#15803d;">{fulfilled_ct}</b></span>
    <span class="kpill" title="In-flight + fulfilled requests in this tab. Canceled requests are listed in the table but not counted here."><span class="kk">Total</span><b>{total_ct}</b></span>
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
    <button onclick="ifFlagFilter('AT_VENDOR')" id="iff-AT_VENDOR" class="iff-fbtn"          style="{btn_s}background:#eef2ff;color:#3730a3;border-color:#c7d2fe;">At Vendor</button>
    <button onclick="ifFlagFilter('AT_RISK')"  id="iff-AT_RISK"  class="iff-fbtn"            style="{btn_s}background:#fef9c3;color:#713f12;border-color:#fde047;">Behind Schedule</button>
    <button onclick="ifFlagFilter('BLOCKED')"  id="iff-BLOCKED"  class="iff-fbtn"            style="{btn_s}background:#fee2e2;color:#b91c1c;border-color:#fca5a5;">Blocked</button>
    <button onclick="ifFlagFilter('STALLED')"  id="iff-STALLED"  class="iff-fbtn"            style="{btn_s}background:#fef2f2;color:#dc2626;border-color:#fca5a5;">Stalled</button>
    <button onclick="ifFlagFilter('COLONY_RISK')" id="iff-COLONY_RISK" class="iff-fbtn"      style="{btn_s}background:#fee2e2;color:#991b1b;border-color:#fca5a5;">Colony Risk</button>
  </div>

  <!-- Table — soft off-white surface on the grey tab body, matching the Tracking tab cards -->
  <div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;box-shadow:0 1px 3px rgba(15,23,42,0.06);overflow:hidden;">
  <div style="overflow-x:auto;">
    <table id="inflight-table" style="width:100%;border-collapse:collapse;background:#f8fafc;">
      <thead id="inflight-thead"></thead>
      <tbody id="inflight-tbody"><tr><td colspan="14" style="padding:20px;color:#6b7280;font-size:11px;">Loading…</td></tr></tbody>
    </table>
  </div>
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
  var _ACTIVE_ST = {json.dumps(sorted(_ACTIVE_REQ_STATUS))};   // in-flight: NEW/PLANNED/IN_PROGRESS/REMEDIATION

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
  // Lucide paths + status->icon map come from renderer/tokens.py (single source
  // shared with the Tracking tab).
  var LU = {JS_LU};
  var STATUS_LU = {JS_STAT_LU};
  // status badge: Lucide icon (colorblind cue, inherits text color) + label + tint.
  function statusBdg(s){{var k=STATUS_LU[s];var ic=k?lucide(LU[k])+'&nbsp;':'';
    var lbl=(s==='NEW')?'In Design':String(s).replace(/_/g,' ');   // NEW request = actively being designed
    return '<span style="'+PILL+'display:inline-flex;align-items:center;'+(S_ST[s]||_ST_GRAY)+'">'+ic+esc(lbl)+'</span>';}}
  // phase pill: solid brand-sweep fill, own geometry.
  function phaseBdg(p){{return P_ST[p]?'<span style="'+GEO_PHASE+P_ST[p]+'">'+esc(p)+'</span>':'';}}
  // Per-experiment summary for the experiment header row — the same three pills
  // the tab-wide bar carries at the top (In phase split, Fulfilled, Total), one
  // line per project. It is an OVERVIEW: static, tallied off every row of the
  // experiment, so the filter bar never moves it — same contract as the top bar
  // (see the Python comment on ph_asm/ph_lsp). The phase split uses the identical
  // rule as the top pill: anything active that is not ASM/LSP/PARTS is DESIGN (a
  // NEW request has no phase yet). Fulfilled is status == FULFILLED, Total is
  // every row regardless of status.
  var _IF_ACT={IF_ACT_JSON};                   // mirror of _ACTIVE_REQ_STATUS
  var _DESIGN_ST='{DESIGN_CHIP}';
  // Same chip vocabulary as the top bar: one .kpill holding the "In phase" split,
  // then a .kpill each for Fulfilled and Total. Reusing the class (rather than
  // restyling here) is what keeps the two rows reading as the same system, and
  // gives the counts their own surface so they stop blending into the Partner pill.
  function _expKpill(lbl,n,col){{
    return '<span class="kpill" style="margin-left:6px;"><span class="kk">'+lbl+'</span>'
         + '<b'+(col?' style="color:'+col+';"':'')+'>'+n+'</b></span>';
  }}
  // Tallied ONCE off the full record set, not off the filtered rows — these are the
  // standing shape of each experiment, exactly like the tab-wide bar at the top, so
  // clicking IN PROGRESS or any flag filter never moves them. Keyed by experiment.
  var _EXP_STAT=(function(){{
    var m={{}};
    _IFD.forEach(function(r){{
      var s=m[r.exp]||(m[r.exp]={{ASM:0,LSP:0,PARTS:0,DESIGN:0,ful:0,total:0,ip:0}});
      s.total++;
      if(r.status==='FULFILLED') s.ful++;
      if(_IF_ACT[r.status]){{
        s[(r.phase==='ASM'||r.phase==='LSP'||r.phase==='PARTS')?r.phase:'DESIGN']++;
        s.ip++;
      }}
    }});
    return m;
  }})();
  function _expSummary(exp){{
    var s=_EXP_STAT[exp];
    if(!s) return '';
    var out='';
    if(s.ip){{
      var chips='';
      ['ASM','LSP','PARTS','DESIGN'].forEach(function(k){{
        if(!s[k]) return;
        var sty=(k==='DESIGN')?_DESIGN_ST:(GEO_PHASE+P_ST[k]);
        chips+='<span style="display:inline-flex;align-items:center;gap:4px;">'
             + '<span style="'+sty+'">'+k+'</span><b>'+s[k]+'</b></span>';
      }});
      out+='<span class="kpill" style="gap:9px;margin-left:10px;"><span class="kk">In phase</span>'
         + chips + '</span>';
    }}
    out += _expKpill('Fulfilled', s.ful, '#15803d') + _expKpill('Total', s.total, '');
    return '<span title="This experiment: in-progress requests by phase, plus fulfilled and total.'
         + ' Static — does not change with the filters, same as the bar at the top.">'+out+'</span>';
  }}
  var PAI_STY   ='{PAI_STYLE}';
  var PAI_STY_RD='{PAI_STYLE_RD}';
  function paiBadges(s,cust){{if(!s)return'';var st=cust==='R_D'?PAI_STY_RD:PAI_STY;return s.split(',').map(function(p){{p=p.trim();return p?'<span style="'+st+'">'+esc(p)+'</span>':'';}}).join('');}}
  var CUST_MAP={JS_CUST};
  // customer badge: optional leading marker (CUST_DOT, from tokens) + label + tint.
  function custBadge(s,fp){{var m=CUST_MAP[s]||['—','#f1f5f9','#6b7280'];return'<span style="'+GEO_CUST+'background:'+m[1]+';color:'+m[2]+';">{CUST_DOT}'+m[0]+'</span>';}}
  var _DPILL='display:inline-block;padding:0px 5px;border-radius:3px;font-size:9px;font-weight:600;white-space:nowrap;margin-top:2px;';
  function fmtDate(s){{if(!s)return'';var diff=Math.round((new Date(s)-new Date(_TODAY))/(864e5));var bg,clr,lbl;if(diff<0){{bg='#fee2e2';clr='#991b1b';lbl=Math.abs(diff)+'d ago';}}else if(diff===0){{bg='#fef3c7';clr='#92400e';lbl='today';}}else if(diff<=7){{bg='#fef9c3';clr='#713f12';lbl='in '+diff+'d';}}else{{bg='#f1f5f9';clr='#6b7280';lbl='in '+diff+'d';}}return'<span style="color:#374151;">'+esc(s)+'</span><br><span style="background:'+bg+';color:'+clr+';'+_DPILL+'">'+lbl+'</span>';}}
  function fmtSubmitter(s){{if(!s||s.indexOf('@')===-1)return esc(s);var parts=s.split('@');var local=parts[0];var domain=parts[1];var org=domain.split('.')[0];org=org.charAt(0).toUpperCase()+org.slice(1);var name=local.split('.').map(function(p){{return p.charAt(0).toUpperCase()+p.slice(1);}}).join(' ');var ext=!domain.toLowerCase().startsWith('asimov.');var orgSty=ext?'display:inline-block;font-size:9px;font-weight:600;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;border-radius:3px;padding:1px 5px;margin-top:1px;':'display:block;color:#9ca3af;font-size:9px;';return'<span style="display:block;">'+esc(name)+'</span><span style="'+orgSty+'">'+esc(org)+'</span>';}}

  // ── Colony Tracking view state + helpers ──────────────────────────────────
  var _view = 'standard';                 // 'standard' | 'colony' — always open in Standard View
  var _expR = {{}};                        // expanded request ids   {{req_id: true}}
  var _expA = {{}};                        // expanded attempts      {{req_id|n: true}}

  // Competent-cell / strain chips (item 7) — distinct, saturated hues so NEB vs EPI
  // are separable at a glance (were both pale pastels that read alike).
  // Categorical (non-status) hues: strains are labels, not outcomes, so they must
  // stay out of the red/amber/green lane used by status, risk band, and seq-conf.
  var STRAIN_STY = {{
    'NEB_STBL':'background:#E7E9FD;color:#3730A3;border:0.5px solid #A5B4FC;',
    'EPI400':  'background:#E0F2FE;color:#075985;border:0.5px solid #7DD3FC;',
    'STBL3':   'background:#EEF2F7;color:#334155;border:0.5px solid #94A3B8;',
  }};
  var STRAIN_CHIP='display:inline-block;font-size:9px;padding:1px 5px;border-radius:4px;font-weight:600;white-space:nowrap;margin:0 1px;';
  function strainBdg(s){{var st=STRAIN_STY[s]||'background:#F1F5F9;color:#475569;border:0.5px solid #CBD5E1;';return '<span style="'+STRAIN_CHIP+st+'">'+esc(s)+'</span>';}}
  // Flag chips (item 8) — muted, thin border.
  var CF_ST = {JS_CF_ST};
  var CF_LBL = {{'LOW_PICKABLE':'LOW COLONIES','PAST_DUE':'PAST DUE'}};
  // L1 colony flags = colony flags + PAST_DUE inherited from the request flags
  function colFlags(r){{var f=(r.col.cflags||[]).slice();if(r.flags.indexOf('PAST_DUE')!==-1)f.push('PAST_DUE');return f;}}
  // Numeric cell — neutral dark, right-aligned, no conditional red (item 9).
  function num(n){{n=n||0;return '<span class="if-cnum">'+n+'</span>';}}
  // Real counts vs "not counted yet": a colony metric is a genuine count only when
  // SOME metric is positive — a non-null 0 placeholder is NOT a count. Uncounted →
  // "—"; counted → the real number, never greyed (even a genuine 0).
  function _counted(o){{return ((o.imaged||0)+(o.pickable||0)+(o.picked||0)+(o.seq||0)+(o.tot||0)+(o.totc||0))>0;}}
  function ccell(n,o){{return _counted(o)?num(n):_dash();}}
  // Pickable risk band for an attempt's pickable count (see legend in the toolbar):
  //   Low 0–7 (below median), Medium 8–22 (median→75th pct), High 23+ (top quartile).
  var PICK_LOW_MAX={PICK_LOW_MAX}, PICK_MED_MAX={PICK_MED_MAX};
  var RISK_HIGH_MAX={RISK_HIGH_MAX}, RISK_MED_MAX={RISK_MED_MAX};
  // Bands are PER STRAIN. An attempt sums its strain transformations, so judging a 1-strain
  // attempt against the same number as a 2-strain one penalised it for having one pool:
  // 1-strain median 12 vs 2-strain 26, ~1.9x more likely to read LOW. Per-strain yield is
  // near-identical (12 vs 13), so dividing by the strain count compares like with like.
  function _nstrain(a){{
    var n=(a.by_strain||[]).filter(_counted).length;
    if(!n) n=(a.strains||[]).length;
    return n||1;
  }}
  function _perStrain(a){{ return Math.round((a.pickable||0)/_nstrain(a)); }}
  function pickBand(n){{
    n=n||0;
    var lbl, st;
    if(n<=PICK_LOW_MAX){{lbl='LOW'; st='background:#FDE2E2;color:#B42318;border:0.5px solid #F5A3A3;';}}
    else if(n<=PICK_MED_MAX){{lbl='MED'; st='background:#FEF3C7;color:#92400E;border:0.5px solid #FCD34D;';}}
    else{{lbl='HIGH'; st='background:#DCFCE7;color:#15803D;border:0.5px solid #86EFAC;';}}
    return '<span title="'+n+' pickable per strain — '+lbl+' band (per-strain median '+PICK_LOW_MAX+', p75 '+PICK_MED_MAX+')" style="display:inline-block;font-size:8px;font-weight:700;padding:0 4px;border-radius:3px;white-space:nowrap;margin-left:6px;vertical-align:middle;'+st+'">'+lbl+'</span>';
  }}
  // Risk level for a design, by the BEST attempt available (its pickable ceiling):
  //   per-strain <= RISK_HIGH_MAX (p25) → HIGH RISK (about to run out of viable picks)
  //   per-strain <= RISK_MED_MAX (median) → MED RISK  (watch it)
  //   above that                          → healthy, no badge
  // Deliberately NOT the descriptive band: that one describes where a count sits in the
  // distribution, this one asks whether the work is about to stall. Reusing one number for
  // both made every below-median attempt an alarm.
  // Designs with a sequence-confirmed winner / already succeeded are never flagged.
  // Colony risk is a statement about work that is STILL LIVE. A design or attempt that
  // has already failed or been canceled is over — its colony counts are history, and
  // reading them as current risk flags requests that have nothing at risk right now.
  var _DEAD_ST={{FAILED:1,CANCELED:1}};
  function _dead(o){{ return !!_DEAD_ST[o.status]; }}
  // A retry already in flight: an attempt with no colony counts yet whose workorder is
  // queued or running. Low colonies matter far less when the next attempt is already
  // moving, and 56% of HIGH flags are in exactly that position — so the badge has to
  // say it, or it overstates how much of this needs a human today.
  var _PEND_RANK={{RUNNING:4,READY:3,WAITING:2,NEW:1}};
  // Returns {{st, stage}} for the furthest-along uncounted attempt. `stage` is the furthest
  // OpTracker protocol it reached, which answers "is this before the colony-count step?" —
  // counts land partway through miniprep, so 'miniprep' means a count is imminent while
  // 'assembly' / '' means it is nowhere near one yet.
  var _STAGE_TXT={{miniprep:'at miniprep — counts pending',transformation:'at transformation',
                  assembly:'at assembly'}};
  function _pending(d){{
    var st='', rank=0, stage='';
    (d.attempts||[]).forEach(function(a){{
      if(_counted(a)) return;                       // already has colonies — not a retry
      var rk=_PEND_RANK[a.status]||0;
      if(rk>rank){{ rank=rk; st=a.status; stage=a.stage_p||''; }}
    }});
    return {{st:st, stage:stage}};
  }}
  // Returns {{level, cur, pend}} — `cur` is the pickable count that drove the verdict, so
  // the badge can show its own reason instead of looking like it contradicts the rows below.
  function designRisk(d){{
    var atts=d.attempts||[];
    if(!atts.length) return {{level:'',cur:0,pend:'',pstage:''}};
    if(d.has_winner || d.status==='SUCCEEDED' || d.status==='FULFILLED') return {{level:'',cur:0,pend:'',pstage:''}};
    if(_dead(d)) return {{level:'',cur:0,pend:'',pstage:''}};   // dead design: not currently at risk
    // Only assess pickable-band risk over attempts that have ACTUAL colony counts.
    // An uncounted attempt (nothing imaged yet) has pickable 0, which is NOT "low
    // pickable" — it just hasn't been counted, so it must not read as HIGH RISK.
    // Failed attempts drop out for the same reason as failed designs: those colonies
    // are gone, so they are not options you could still pick from.
    var counted=atts.filter(function(a){{ return _counted(a) && !_dead(a); }});
    if(!counted.length) return {{level:'',cur:0,pend:'',pstage:''}};
    // The NEWEST counted attempt is the verdict — not the best one. An earlier attempt
    // can sit in transformation/miniprep for weeks, and taking the max let its old
    // colony count keep the design looking healthy after the lab had already moved on
    // to a fresh attempt. Once the new attempt is counted, that count is the state of
    // the work: 40 colonies last month does not help if today's attempt yielded 3.
    // `n` is assigned chronologically when attempts are built, so highest n = newest.
    var cur=counted[0];
    counted.forEach(function(a){{ if((a.n||0) > (cur.n||0)) cur=a; }});
    var pick=_perStrain(cur);          // per strain, so 1- and 2-strain attempts compare fairly
    var pend=_pending(d);
    if(pick<=RISK_HIGH_MAX) return {{level:'HIGH',cur:pick,pend:pend.st,pstage:pend.stage}};
    if(pick<=RISK_MED_MAX) return {{level:'MED',cur:pick,pend:pend.st,pstage:pend.stage}};
    return {{level:'',cur:pick,pend:pend.st,pstage:pend.stage}};
  }}
  // level is the RISK (High = bad); the LOW/MED/HIGH on the rows below is the pickable
  // COUNT band. Those two scales run opposite ways, so name the driver in the badge.
  function riskBadge(level, cur, pend, pstage){{
    if(level!=='HIGH' && level!=='MED') return '';
    var st = level==='HIGH'
      ? 'background:#FEE2E2;color:#991B1B;border:1px solid transparent;'
      : 'background:#FEF3C7;color:#92400E;border:1px solid transparent;';
    var tip = level==='HIGH'
      ? 'The newest counted attempt is at or below '+RISK_HIGH_MAX+' pickable per strain (25th percentile) with no sequence-confirmed colony — at risk of running out of viable picks. Failed/canceled attempts are excluded.'
      : 'The newest counted attempt is at or below '+RISK_MED_MAX+' pickable per strain (the median) with no sequence-confirmed colony — watch this one. Failed/canceled attempts are excluded.';
    var L = level.charAt(0)+level.slice(1).toLowerCase();
    var drv = (cur||cur===0) ? ' &middot; latest attempt '+(cur||0)+' pk/strain' : '';
    // A queued/running retry is the single biggest thing that changes how urgent this is.
    var _sg = pstage ? (_STAGE_TXT[pstage]||pstage) : 'not started';
    var rt  = pend ? ' &middot; retry '+pend.toLowerCase()+' &middot; '+_sg : '';
    if(pend) tip += ' A further attempt is already '+pend.toLowerCase()+' ('+_sg+') with no colonies counted yet.';
    return '<span title="'+tip+'" style="display:inline-block;font-size:8px;font-weight:700;padding:0 4px;border-radius:3px;white-space:nowrap;margin-left:6px;vertical-align:middle;'+st+'">&#9888; Colony risk: '+L+drv+rt+'</span>';
  }}
  // Worst colony risk across a request's designs + the pickable/picked counts driving it.
  function reqColRisk(r){{
    var lv='', pk=0, pd=0, cu=0, pn='', ps='';
    (r.designs||[]).forEach(function(d){{
      var dr=designRisk(d), rk=dr.level;
      if(rk==='HIGH' && lv!=='HIGH'){{ lv='HIGH'; pk=d.pickable||0; pd=d.picked||0; cu=dr.cur; pn=dr.pend; ps=dr.pstage; }}
      else if(rk==='MED' && lv===''){{ lv='MED'; pk=d.pickable||0; pd=d.picked||0; cu=dr.cur; pn=dr.pend; ps=dr.pstage; }}
    }});
    return {{level:lv, pick:pk, picked:pd, cur:cu, pend:pn, pstage:ps}};
  }}
  // Colony-risk flag badge (for the standard-view Flags column) — shows severity AND
  // the pickable + total-picked colony counts so the standard view carries the colony info too.
  function colRiskFlag(level,pick,picked,pend,pstage){{
    if(level!=='HIGH' && level!=='MED') return '';
    picked=picked||0;
    var st = level==='HIGH' ? 'background:#FEE2E2;color:#991B1B;border:1px solid transparent;'
                            : 'background:#FEF3C7;color:#92400E;border:1px solid transparent;';
    var tip = level==='HIGH'
      ? 'Colony at risk: the newest counted attempt is at or below '+RISK_HIGH_MAX+' pickable per strain, no seq-confirmed clone. '+pick+' pickable, '+picked+' picked across the design. Failed/canceled attempts excluded.'
      : 'Colony watch: the newest counted attempt is at or below '+RISK_MED_MAX+' pickable per strain, no seq-confirmed clone. '+pick+' pickable, '+picked+' picked across the design. Failed/canceled attempts excluded.';
    var L = level.charAt(0)+level.slice(1).toLowerCase();
    // Same retry qualifier the Colony Tracking badge carries — the standard view was
    // showing the alarm without the one fact that says whether it needs you today.
    var _sg = pstage ? (_STAGE_TXT[pstage]||pstage) : 'not started';
    var rt = pend ? ' &middot; retry '+pend.toLowerCase()+' &middot; '+_sg : '';
    if(pend) tip += ' A further attempt is already '+pend.toLowerCase()+' ('+_sg+') with no colonies counted yet.';
    return '<span title="'+tip+'" style="'+BDG+st+'">Colony: '+L+' &middot; '+pick+'pk, '+picked+' picked'+rt+'</span>';
  }}
  // One-time: fold colony risk into each record's flags so it filters/sorts like the
  // other flags (and "All Flags" includes it). Idempotent via the indexOf guard.
  (function(){{
    var _crCt = 0;
    _IFD.forEach(function(r){{
      r.flags = r.flags || [];
      // Colony risk only applies while the request is in assembly (ASM). Past that
      // (LSP/PARTS/etc.) the colony picture is no longer the actionable signal.
      var cr = (r.phase === 'ASM') ? reqColRisk(r) : {{level:'', pick:0, picked:0, pend:'', pstage:''}};
      r._colRisk = cr.level; r._colPick = cr.pick; r._colPicked = cr.picked;
      r._colPend = cr.pend; r._colPStage = cr.pstage;
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
    // Terminal (CANCELED/FAILED) with no colonies ever sequenced: "0/0" is meaningless
    // — the design never produced sequencing data, so show a neutral dash.
    if(seq===0 && tot===0) return '<span style="color:#cbd5e1;">&mdash;</span>';
    var pct=tot>0?(seq/tot):0, sty;
    if(seq===0)        sty='background:#FCEBEB;color:#A32D2D;';
    else if(pct<0.20)  sty='background:#FAEEDA;color:#633806;';
    else               sty='background:#EAF3DE;color:#3B6D11;';
    var b='<span style="'+SEQPILL+sty+'">'+seq+'/'+tot+'</span>';
    if(winner) b+='<span style="'+BDG+'background:#EAF3DE;color:#3B6D11;border:0.5px solid #97C459;">&#10003; clone</span>';
    return b;
  }}
  // Request-level seq status: the request row sums colony numbers across ALL designs
  // but the overall request status (e.g. IN_PROGRESS, kept alive by still-WAITING
  // designs) doesn't reflect the seq outcome of the designs that actually produced
  // those colonies. If every design that picked colonies is terminal and none are
  // seq-confirmed, the picked colonies FAILED sequencing — not PENDING. Only report a
  // non-terminal (PENDING-eligible) status when a colony-producing design is still
  // active and could yet yield a seq result.
  function reqSeqStatus(r){{
    var term=function(s){{return s==='FAILED'||s==='CANCELED'||s==='SUCCEEDED'||s==='FULFILLED';}};
    var anyActivePicked=(r.designs||[]).some(function(d){{return (d.picked||0)>0 && !term(d.status);}});
    return anyActivePicked ? r.status : 'FAILED';
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
    if      (ff === 'ip')      {{ if (_ACTIVE_ST.indexOf(r.status) === -1) return false; }}
    else if (ff === 'flagged') {{ if (!r.flags.length)             return false; }}
    else if (ff !== 'all')     {{ if (r.flags.indexOf(ff) === -1)  return false; }}
    var q;
    if ((q=_flt.construct) && r.construct.toLowerCase().indexOf(q)===-1) return false;
    if ((q=_flt.pAI)       && r.pAI.toLowerCase().indexOf(q)===-1)       return false;
    if ((q=_flt.customer)  && r.customer.toLowerCase().indexOf(q)===-1)  return false;
    if ((q=_flt.submitter) && r.submitter.toLowerCase().indexOf(q)===-1) return false;
    // The refill-batch lines live in this column too, so the Operation filter has to see them —
    // "19132" or "stalled" should find the requests those batches are holding up.
    if ((q=_flt.operation) && opText(r).toLowerCase().indexOf(q)===-1) return false;
    if ((q=_flt.req_id)    && r.req_id.toLowerCase().indexOf(q)===-1)    return false;
    return true;
  }}

  // ── Render (rebuilds tbody from _IFD using _flt) ──────────────────────────
  // Buckets passing rows by experiment before rendering so column sorts never
  // produce duplicate experiment headers or scattered rows.
  // Variant pill: 'orig' (no suffix) vs 'v2'/'v3' redo — shown in the construct
  // cell of a grouped member, since the full construct name lives in the header.
  function variantPill(v){{
    var lbl=v?v:'orig';
    var st=v?'background:#eef1f6;color:#566077;border:1px solid #dde2ec;'
            :'background:#f4f4f5;color:#6b7280;border:1px solid #e6e6e9;';
    return '<span style="display:inline-block;font-size:8px;font-weight:600;padding:1px 6px;border-radius:3px;'+st+'">'+esc(lbl)+'</span>';
  }}
  // Construct cell — wraps the full name (no ellipsis truncation). For a grouped
  // variant member, the base name is in the header, so show just the variant pill.
  function constructCell(r, grouped){{
    if(grouped) return '<td title="'+esc(r.construct)+'" style="'+TD+'max-width:240px;">'+variantPill(r.variant)+'</td>';
    return '<td title="'+esc(r.construct)+'" style="'+TD+'max-width:260px;white-space:normal;word-break:break-word;">'+esc(r.construct)+'</td>';
  }}
  // Construct-group header: one full base-construct name spanning the row, shown
  // above the original + v2/v3 members that share it.
  function _constructHeader(base, ncol){{
    return '<tr class="if-cgrp"><td colspan="'+ncol+'" style="padding:3px 14px 3px 20px;font-size:10px;'
         + 'font-weight:600;color:#475569;white-space:normal;word-break:break-word;line-height:1.3;">'
         + esc(base) + '</td></tr>';
  }}
  // ── Operation cell ────────────────────────────────────────────────────────
  // Below the OpTracker operation, the refill batches running for the part(s) this build is
  // blocked on. The flags say a request is stuck; this says what is being done about it — and
  // for a BLOCKED/STALLED row (operation blank by design) it is the only movement there is.
  function opText(r){{
    var t=r.operation||'';
    (r.batches||[]).forEach(function(b){{
      t+=' '+b.part+' '+b.stage+' '+b.proc+(b.stalled?' stalled':' refill running');
    }});
    return t;
  }}
  function batchLines(r){{
    var b=r.batches||[]; if(!b.length) return '';
    var out='';
    for(var i=0;i<b.length;i++){{
      var x=b[i];
      var tip=x.part+' — furthest stage '+x.stage+', last activity '+x.age+'d ago · '+x.proc
            + (x.sequencing?' · its NGS job is still open — the result lands when that job closes':'');
      out+='<div title="'+esc(tip)+'" style="font-size:9px;line-height:1.35;margin-top:2px;color:'
         + (x.stalled?'#b91c1c':'#15803d')+';font-weight:600;">'
         + (x.stalled?'⚠ refill stalled':'⟳ refill running')
         + ' <span style="font-family:monospace">'+esc(x.part)+'</span>'
         + '<span style="color:#6b7280;font-weight:400;"> · '+esc(x.stage)+' · '+x.age+'d ago'
         + (x.sequencing?' · NGS open':'')+'</span></div>';
    }}
    return out;
  }}
  function opCell(r){{
    // Wider than it used to be (was a 160px nowrap ellipsis): the room was going to Flags, and
    // the batch lines need to read as sentences, not as truncated fragments.
    return '<td style="'+TD+'min-width:180px;max-width:300px;white-space:normal;word-break:break-word;">'
         + '<span title="'+esc(r.operation)+'">'+esc(r.operation)+'</span>'+batchLines(r)+'</td>';
  }}
  function _rowHtml(r, grouped) {{
    var bg='';
    for(var fi=0;fi<r.flags.length;fi++){{if(F_BG[r.flags[fi]]){{bg='background:'+F_BG[r.flags[fi]]+';';break;}}}}
    var fps = r.fp ? 'color:#7c3aed;font-weight:700;' : '';
    var st  = statusBdg(r.status);
    var ph  = phaseBdg(r.phase);
    var fl  = r.flags.map(function(f){{
      if(f==='COLONY_RISK') return colRiskFlag(r._colRisk, r._colPick, r._colPicked, r._colPend, r._colPStage);
      if(f==='AT_VENDOR') return bdg('AT VENDOR'+(r.vendor_out?' · '+r.vendor_out:''),F_ST['AT_VENDOR']);
      if(f==='AT_RISK') return '<span title="Behind the internal milestone schedule needed to hit the committed due date — the assembly or LSP scale-up milestone has already passed." style="'+BDG+F_ST['AT_RISK']+'">'+esc('Behind')+'</span>';
      return bdg(f.replace(/_/g,' '),F_ST[f]||F_ST['STALLED']);
    }}).join('');
    return '<tr class="'+(grouped?'if-cgrp-mem':'')+'" style="'+bg+'">'
          + '<td style="'+TD+fps+'">'+(r.fp?'★':'')+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+paiBadges(r.pAI,r.customer)+'</td>'
          + constructCell(r, grouped)
          + '<td style="'+TD+'white-space:nowrap;">'+custBadge(r.customer,r.fp)+'</td>'
          + '<td style="'+TD+'max-width:110px;">'+fmtSubmitter(r.submitter)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+st+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+ph+'</td>'
          + opCell(r)
          + '<td style="'+TD+'white-space:nowrap;">'+fl+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.assembly)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.lsp_scaleup)+'</td>'
          + '<td style="'+TD+'white-space:nowrap;">'+fmtDate(r.due_date)+'</td>'
          + '<td style="'+TD+'font-family:monospace;font-size:9px;color:#9ca3af;overflow-wrap:anywhere;">'+esc(r.req_id)+'</td>'
          + '</tr>';
  }}
  // ── Colony Tracking row builders (L1 request → L2 design → L3 attempt → L4 wo) ──
  function _dash(){{ return '<span style="display:block;text-align:right;color:#cbd5e1;font-size:11px;">&mdash;</span>'; }}
  function _colReqRow(r, grouped) {{
    var open = !!_expR[r.req_id], c = r.col;
    var fps = r.fp ? 'color:#7c3aed;font-weight:700;' : '';
    var ph  = phaseBdg(r.phase);
    // Worst live design drives the request badge; carry that design's best-attempt count
    // through so the request row explains itself the same way the design rows do.
    var _rk = reqColRisk(r);
    var rwarn = riskBadge(_rk.level, _rk.cur, _rk.pend, _rk.pstage);
    return '<tr class="if-cardtop'+(grouped?' if-cgrp-mem':'')+'" data-tk="'+esc(r.req_id)+'" style="cursor:pointer;font-weight:600;" onclick="ifToggleReq(\\''+r.req_id+'\\')">'
      + '<td style="'+TD+'"><span class="if-caret'+(open?' open':'')+'">&#9654;</span></td>'
      + '<td style="'+TD+fps+'">'+(r.fp?'★':'')+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+paiBadges(r.pAI,r.customer)+'</td>'
      + constructCell(r, grouped)
      + '<td style="'+TD+'white-space:nowrap;">'+custBadge(r.customer,r.fp)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+ph+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+rwarn+'</td>'
      + '<td style="'+TD+'">'+ccell(c.pickable,c)+'</td>'
      + '<td style="'+TD+'">'+ccell(c.picked,c)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(c.seq,c.tot,c.has_winner,reqSeqStatus(r),c.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(r.status)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;font-size:9px;color:#64748b;">'+fmtMDY(r.assembly)+'</td>'
      + '</tr>';
  }}
  // L2 — DESIGN (the triangle): one per attempt_anchor_id = distinct backbone+parts.
  function _colDesignRow(r, d, di) {{
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
    // Zero-attempt designs (WAITING/CANCELED with no colony work) have no pickable
    // data to band — pickBand(0) would falsely read "LOW", so suppress it.
    var band = ((d.attempts||[]).length === 1 && _counted(d))
             ? pickBand(_perStrain((d.attempts||[])[0] || d)) : '';
    var _dr = designRisk(d);
    var warn = riskBadge(_dr.level, _dr.cur, _dr.pend, _dr.pstage);
    return '<tr class="if-att-row" data-tk="'+esc(r.req_id+'|'+d.anchor)+'"'+click+'>'
      + '<td style="'+TD+'padding-left:20px;">'+caret+'</td>'
      + '<td style="'+TD+'" colspan="4"><span style="font-size:10px;font-weight:700;color:#334155;">Design '+(di+1)+' &middot; '+esc(d.dtype||'Design')+'</span>'+natt+parts+'</td>'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'white-space:nowrap;">'+(warn||band)+'</td>'
      + '<td style="'+TD+'">'+ccell(d.pickable,d)+'</td>'
      + '<td style="'+TD+'">'+ccell(d.picked,d)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+seqBdg(d.seq,d.tot,false,d.status,d.picked)+'</td>'
      + '<td style="'+TD+'white-space:nowrap;">'+statusBdg(d.status)+win+'</td>'
      + '<td style="'+TD+'"></td>'
      + '</tr>';
  }}
  // L4 — workorder within an attempt: Gibson row, then its &#9492;&#9472; transformations.
  // Phase column shows the agar plate &middot; well coordinate.
  function _colWoRow(w) {{
    // Strain rows stack flush under the design (no &#9492;&#9472; tree nesting): the strain chip
    // is centered under the CONSTRUCT column, and the agar plate&middot;well is centered across
    // the CUSTOMER+PHASE span (between them). The top request row keeps a plain Phase cell.
    var strainCell = (w.strain?strainBdg(w.strain):'');
    var agarCell = agarLink(w.agar_url,w.agar_label);
    // Full process id (un-truncated) on the left, under the PAI / Golden Gate column,
    // tab-indented so it reads as nested under the attempt.
    var pidCell = '<span style="font-size:8px;font-family:monospace;color:#94a3b8;">'+esc(String(w.wid))+'</span>';
    var c8,c9,c10,rb;
    if (!_counted(w)) {{ c8=_dash(); c9=_dash(); c10='<span style="color:#cbd5e1;">&mdash;</span>'; rb=''; }}
    else {{ c8=num(w.pickable); c9=num(w.picked); c10=seqBdg(w.seq,w.totc,false,w.status,w.picked); rb=pickBand(w.pickable); }}
    return '<tr class="if-strain-row">'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'"></td>'
      + '<td style="'+TD+'padding-left:24px;white-space:nowrap;">'+pidCell+'</td>'
      + '<td style="'+TD+'text-align:center;white-space:nowrap;">'+strainCell+'</td>'
      + '<td style="'+TD+'text-align:center;white-space:nowrap;font-size:9px;" colspan="2">'+agarCell+'</td>'
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
      + '<td style="'+TD+'white-space:nowrap;">'+(_counted(a)?pickBand(_perStrain(a)):'')+'</td>'
      + '<td style="'+TD+'">'+ccell(a.pickable,a)+'</td>'
      + '<td style="'+TD+'">'+ccell(a.picked,a)+'</td>'
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
  function _colonyRows(r, grouped, suppressGap) {{
    var html = (suppressGap ? '' : '<tr class="if-cardgap"><td colspan="12"></td></tr>') + _colReqRow(r, grouped);
    if (_expR[r.req_id]) {{
      (r.designs||[]).forEach(function(d, di) {{
        html += _colDesignRow(r, d, di);
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

  // Keep the clicked row visually still across the rebuild. Restoring a raw scrollTop
  // is not enough: expanding/collapsing changes the container's total height, so the
  // browser clamps the old offset and the view lurches. Anchoring to the clicked row
  // holds it at the same screen position no matter how the height changes.
  var _anchorKey = null, _anchorOff = 0;
  function _markAnchor(key) {{
    var sc = document.getElementById('tab-inflight');
    var el = document.querySelector('#inflight-tbody [data-tk="'+key+'"]');
    if (sc && el) {{ _anchorKey = key; _anchorOff = el.getBoundingClientRect().top - sc.getBoundingClientRect().top; }}
    else {{ _anchorKey = null; }}
  }}
  window.ifToggleReq = function(id) {{ _markAnchor(id); if (_expR[id]) delete _expR[id]; else _expR[id] = true; window.ifRender(); }};
  window.ifToggleDesign = function(id, anchor) {{ var k = id+'|'+anchor; _markAnchor(k); if (_expA[k]) delete _expA[k]; else _expA[k] = true; window.ifRender(); }};
  window.ifSetView = function(v) {{
    _view = v;
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
    // Preserve scroll position: rebuilding tbody.innerHTML resets the scroll
    // container to the top, so toggling a dropdown would jump the page. Capture the
    // scroll offset here and restore it after the rebuild.
    var _scEl = document.getElementById('tab-inflight');
    var _scTop = _scEl ? _scEl.scrollTop : 0;
    var _winY  = window.scrollY || document.documentElement.scrollTop || 0;
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
            + esc(exp) + partnerPill + _expSummary(exp) + '</td></tr>';
      // Sub-group rows by base construct so an original + its v2/v3 redo render as
      // one section (shared header). Grouping is by base regardless of adjacency,
      // so it survives column re-sorts; singletons render exactly as before.
      var vgroups = [], vidx = {{}};
      g.rows.forEach(function(r) {{
        var b = r.base || r.construct;
        if (!vidx.hasOwnProperty(b)) {{ vidx[b] = vgroups.length; vgroups.push([]); }}
        vgroups[vidx[b]].push(r);
      }});
      vgroups.forEach(function(rows) {{
        if (rows.length > 1) {{
          html += _constructHeader(rows[0].base || rows[0].construct, NCOL);
          rows.forEach(function(r) {{ html += COLONY ? _colonyRows(r, true, true) : _rowHtml(r, true); }});
        }} else {{
          html += COLONY ? _colonyRows(rows[0]) : _rowHtml(rows[0]);
        }}
      }});
    }});
    if (!html) html = '<tr><td colspan="'+NCOL+'" style="padding:20px;color:#6b7280;font-size:11px;text-align:center;">No matching requests.</td></tr>';
    tbody.innerHTML = html;
    // Anchored restore first (survives height changes); fall back to the raw offset.
    var _done = false;
    if (_anchorKey && _scEl) {{
      var _el2 = document.querySelector('#inflight-tbody [data-tk="'+_anchorKey+'"]');
      if (_el2) {{
        _scEl.scrollTop += (_el2.getBoundingClientRect().top - _scEl.getBoundingClientRect().top) - _anchorOff;
        _done = true;
      }}
    }}
    _anchorKey = null;
    if (!_done && _scEl) _scEl.scrollTop = _scTop;
    // The tab is the scroll container, but restore the page offset too in case the
    // window is also scrolled (short viewport) — otherwise the page itself lurches.
    if (_winY) window.scrollTo(0, _winY);
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
    // "All" = show every status; "IN PROGRESS" = narrow to in-flight statuses
    // (NEW/PLANNED/IN_PROGRESS/REMEDIATION — everything not fulfilled/canceled).
    if (f === 'all') {{
      _flt.status = new Set(_ALL_ST);
    }} else if (f === 'ip') {{
      _flt.status = new Set(_ACTIVE_ST);
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
      {{k:'phase',     lbl:'Phase',  filter:true}},
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
           + 'font-weight:700;text-transform:uppercase;background:#f1f5f9;position:sticky;top:0;z-index:2;'
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

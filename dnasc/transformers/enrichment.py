"""
dnasc/transformers/enrichment.py
─────────────────────────────────
Request-level enrichment: pipeline stage, stall detection, and status ranking.
All fields are fully deterministic from df columns — no external data or current-time
dependencies. Must run after _bridge_status so visual_status is already set.

Adds these columns to the parquet:
  status_rank   — numeric sort priority (0=urgent … 5=canceled)
  stage         — pipeline stage per request: In Design / PCR / Vendor Parts /
                  DV/PL1 Build / Assembly / Assembly QC / LSP / LSP QC /
                  Reviewing / Releasing / Stalled / Fulfilled / Canceled
  is_stalled    — bool: request has no active work and isn't finished
  is_asm_review — bool: a winning assembly succeeded but another is still open
  is_finished   — bool: request_status in FULFILLED/SUCCEEDED
  is_blocked    — bool: any row in the request has visual_status=BLOCKED
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from dnasc.logger import get_logger
from dnasc import protocols as proto

log = get_logger(__name__)

_STATUS_PRIORITY = {
    'RUNNING': 0, 'IN_PROGRESS': 0,
    'BLOCKED': 1,
    'WAITING': 2, 'READY': 2, 'DRAFT': 2,
    'SUCCEEDED': 3, 'FAILED': 4, 'CANCELED': 5,
}
_ROOT_CHAIN_TYPES = frozenset({
    'gibson_workorder', 'golden_gate_workorder', 'transformation_workorder',
    'transformation_offline_operation', 'streakout_operation', 'lsp_workorder',
})
_ASM_TYPES   = frozenset({'golden_gate_workorder', 'gibson_workorder'})
_PARTS_TYPES = frozenset({
    'pcr_workorder', 'syn_part_synthesis_workorder',
    'oligo_synthesis_workorder', 'plasmid_synthesis_workorder',
})
_ACTIVE_STATUSES = frozenset({'RUNNING', 'READY', 'IN_PROGRESS', 'WAITING', 'BLOCKED', 'LSP_RUNNING'})
_PHASE_ACTIVE    = frozenset({'RUNNING', 'READY', 'IN_PROGRESS', 'WAITING', 'BLOCKED'})

# Maps (protocol_name, operation_state) → (priority, display_label)
# Higher priority = further along in the pipeline — used to pick the single
# most-advanced active step to surface as req_operation.
_PROTO_MAP: dict[tuple[str, str], tuple[int, str]] = {
    (proto.SYNTHESIS_ORDER,  'RD'): (5,  'SYNTHESIS ORDER: READY'),
    (proto.SYNTHESIS_ORDER,  'RU'): (6,  'SYNTHESIS ORDER: RUNNING'),
    (proto.ORDER_OLIGOS,     'RD'): (5,  'ORDER OLIGOS: READY'),
    (proto.ORDER_OLIGOS,     'RU'): (6,  'ORDER OLIGOS: RUNNING'),
    (proto.RECEIVE_SYNPART,  'RD'): (7,  'RECEIVE SYNPART SYNTHESIS: READY'),
    (proto.RECEIVE_SYNPART,  'RU'): (8,  'RECEIVE SYNPART SYNTHESIS: RUNNING'),
    (proto.RECEIVE_PLASMID,  'RD'): (8,  'RECEIVE PLASMID SYNTHESIS: READY'),
    (proto.RECEIVE_PLASMID,  'RU'): (8,  'RECEIVE PLASMID SYNTHESIS: RUNNING'),
    (proto.PCR,              'RD'): (8,  'PCR: READY'),
    (proto.PCR,              'RU'): (9,  'PCR: RUNNING'),
    (proto.FRAGMENT_ANALYZER,'RD'): (9,  'FRAGMENT ANALYZER: READY'),
    (proto.FRAGMENT_ANALYZER,'RU'): (9,  'FRAGMENT ANALYZER: RUNNING'),
    (proto.GOLDEN_GATE,      'RD'): (10, 'GOLDEN GATE ASSEMBLY: READY'),
    (proto.GOLDEN_GATE,      'RU'): (11, 'GOLDEN GATE ASSEMBLY: RUNNING'),
    (proto.GIBSON,           'RD'): (10, 'GIBSON ASSEMBLY: READY'),
    (proto.GIBSON,           'RU'): (11, 'GIBSON ASSEMBLY: RUNNING'),
    (proto.STAR_TRANSF,      'RD'): (20, 'TRANSFORMATION: READY'),
    (proto.STAR_TRANSF,      'RU'): (21, 'TRANSFORMATION: RUNNING'),
    (proto.MINIPREP,         'RD'): (30, 'MINIPREP: READY'),
    (proto.MINIPREP,         'RU'): (31, 'MINIPREP: RUNNING'),
    (proto.REPICK,           'RD'): (31, 'REPICK MINIPREP: READY'),
    (proto.REPICK,           'RU'): (32, 'REPICK MINIPREP: RUNNING'),
    (proto.REARRAY,          'RD'): (35, 'REARRAY: READY'),
    (proto.REARRAY,          'RU'): (36, 'REARRAY: RUNNING'),
    (proto.DNA_QUANT,        'RD'): (40, 'DNA QUANT: READY'),
    (proto.DNA_QUANT,        'RU'): (41, 'DNA QUANT: RUNNING'),
    (proto.NGS,              'RD'): (50, 'NGS: READY'),
    (proto.NGS,              'RU'): (51, 'NGS: RUNNING'),
    (proto.LSP_ORDER,        'RD'): (60, 'LSP ORDER: READY'),
    (proto.LSP_ORDER,        'RU'): (61, 'LSP ORDER: RUNNING'),
    (proto.LSP_RECEIVING,    'RD'): (65, 'LSP RECEIVING: READY'),
    (proto.LSP_RECEIVING,    'RU'): (66, 'LSP RECEIVING: RUNNING'),
    (proto.GLYCEROL_STOCKING,'RD'): (70, 'GLYCEROL STOCKING: READY'),
    (proto.GLYCEROL_STOCKING,'RU'): (71, 'GLYCEROL STOCKING: RUNNING'),
    (proto.LSP_REVIEWING,    'RD'): (80, 'LSP REVIEWING: READY'),
    (proto.LSP_REVIEWING,    'RU'): (81, 'LSP REVIEWING: RUNNING'),
    (proto.LSP_RELEASING,    'RD'): (90, 'LSP RELEASING: READY'),
    (proto.LSP_RELEASING,    'RU'): (91, 'LSP RELEASING: RUNNING'),
}


def _active_protocols_raw(pn, ps) -> set:
    """Active-protocol set from raw protocol_name / operation_state cell values."""
    if isinstance(pn, np.ndarray): pn = pn.tolist()
    if isinstance(ps, np.ndarray): ps = ps.tolist()
    if not isinstance(pn, list) or not isinstance(ps, list):
        return set()
    return {p for p, s in zip(pn, ps) if s in ('RD', 'RU')}


def _active_protocols(row) -> set:
    """Return set of protocol names whose operation_state is RD or RU."""
    return _active_protocols_raw(row.get('protocol_name'), row.get('operation_state'))


def _stage_from_parts(prt_df: pd.DataFrame) -> str:
    if not prt_df[prt_df['type'] == 'pcr_workorder'].empty:
        return 'PCR'
    if not prt_df[prt_df['type'].isin({'syn_part_synthesis_workorder', 'oligo_synthesis_workorder'})].empty:
        return 'Vendor Parts'
    psw = prt_df[prt_df['type'] == 'plasmid_synthesis_workorder']
    if not psw.empty:
        v = str(psw.iloc[0].get('vendor') or '')
        return 'Vendor Parts' if v not in ('', 'nan', 'None') else 'DV/PL1 Build'
    return 'Vendor Parts'


def _infer_stage(
    r_df: pd.DataFrame,
    active_rows: pd.DataFrame,
    is_stalled: bool,
    has_real_workorders: bool,
    status: str,
    is_finished: bool,
    global_parts_by_stock: dict,
    stock_to_req: dict,
) -> str:
    if is_finished:   return 'Fulfilled'
    if status == 'CANCELED': return 'Canceled'
    if not has_real_workorders: return 'In Design'
    if is_stalled:    return 'Stalled'

    # Build root stock set (STOCK_IDs of root workorders in this request)
    rsm: dict = {}
    for rid2, rdf2 in r_df.groupby('root_work_order_id', dropna=False):
        rw2 = rdf2[rdf2['workorder_id'] == rid2]['STOCK_ID']
        rs2_val = rw2.iloc[0] if not rw2.empty else rdf2['STOCK_ID'].iloc[0]
        rs2 = str(rs2_val) if pd.notna(rs2_val) else 'nan'
        if rs2 not in ('nan', 'None', 'N/A', ''):
            rsm[rid2] = rs2
    all_root_stocks = set(rsm.values())

    eff   = active_rows[active_rows['visual_status'].isin(_ACTIVE_STATUSES)]
    lsp_s = eff[eff['type'] == 'lsp_workorder']
    asm_s = eff[(eff['type'] != 'lsp_workorder') & eff['STOCK_ID'].astype(str).isin(all_root_stocks)]
    prt_s = eff[(eff['type'] != 'lsp_workorder') & ~eff['STOCK_ID'].astype(str).isin(all_root_stocks)]

    # ── LSP phase ────────────────────────────────────────────────────
    if not lsp_s.empty:
        p = _active_protocols(lsp_s.iloc[0])
        if proto.LSP_RELEASING in p:                                                return 'Releasing'
        if proto.LSP_REVIEWING in p:                                                return 'Reviewing'
        if p & {proto.DNA_QUANT, proto.NGS, proto.FRAGMENT_ANALYZER}:              return 'LSP QC'
        return 'LSP'

    # ── Assembly phase ───────────────────────────────────────────────
    if not asm_s.empty:
        asm_prog = asm_s[asm_s['visual_status'].isin({'RUNNING', 'READY', 'IN_PROGRESS', 'BLOCKED'})]
        if not asm_prog.empty:
            asm_prog = asm_prog.copy()
            asm_prog['_r'] = asm_prog['visual_status'].map(
                {'RUNNING': 0, 'READY': 1, 'IN_PROGRESS': 2, 'BLOCKED': 3}
            ).fillna(99)
            p = _active_protocols(asm_prog.sort_values('_r').iloc[0])
            if p & {proto.DNA_QUANT, proto.NGS}:                                    return 'Assembly QC'
            return 'Assembly'

        # All ASM are WAITING — fall through to parts
        if not prt_s.empty:
            return _stage_from_parts(prt_s)

        # WAITING GG with no visible parts — walk backbone/parts token lists
        # against the global parts lookup, then fall back to backbone stock_to_req
        bb_stage   = 'Vendor Parts'
        determined = False
        for _, aw in asm_s[asm_s['visual_status'] == 'WAITING'].iterrows():
            for fld in ('parts', 'backbone'):
                raw = str(aw.get(fld) or '')
                for tok in raw.split(','):
                    psid = tok.split(':')[0].strip()
                    if not psid or psid in ('nan', 'None', ''):
                        continue
                    for gpr in global_parts_by_stock.get(psid, []):
                        gwt = str(gpr.get('type', '') or '')
                        if 'syn_part' in gwt or 'oligo' in gwt:
                            bb_stage = 'Vendor Parts'; determined = True; break
                        elif 'pcr' in gwt:
                            bb_stage = 'PCR';          determined = True; break
                        elif 'plasmid_synthesis' in gwt:
                            v = str(gpr.get('vendor') or '')
                            bb_stage = 'Vendor Parts' if v not in ('', 'nan', 'None') else 'DV/PL1 Build'
                            determined = True; break
                    if determined: break
                if determined: break
            if determined: break
            # Secondary: backbone STOCK_ID lookup via stock_to_req
            bb_sid = str(aw.get('backbone') or '').split(':')[0].strip()
            if bb_sid and bb_sid not in ('nan', 'None', ''):
                bb_info = stock_to_req.get(bb_sid)
                if bb_info:
                    wt = bb_info.get('wo_type', '')
                    if 'syn_part' in wt or 'oligo' in wt:
                        bb_stage = 'Vendor Parts'
                    elif 'plasmid_synthesis' in wt:
                        has_vendor = any(
                            str(r.get('vendor') or '') not in ('', 'nan', 'None')
                            for r in global_parts_by_stock.get(bb_sid, [])
                        )
                        bb_stage = 'Vendor Parts' if has_vendor else 'DV/PL1 Build'
                    else:
                        bb_stage = 'DV/PL1 Build'
                break  # backbone found — take result for this aw
        return bb_stage

    # ── Parts-only phase ─────────────────────────────────────────────
    if not prt_s.empty:
        return _stage_from_parts(prt_s)

    return 'Stalled'


class EnrichmentTransformer:

    @staticmethod
    def compute_request_enrichment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds status_rank, stage, is_stalled, is_asm_review, is_finished, is_blocked
        to every row. All are deterministic from existing df columns.
        """
        log.info("Computing request enrichment fields...")
        df = df.copy()

        # ── status_rank (per row) ─────────────────────────────────────
        df['status_rank'] = df['visual_status'].map(_STATUS_PRIORITY).fillna(99).astype(int)

        # ── Global lookup: STOCK_ID → list of part-type rows ─────────
        parts_mask = df['type'].isin(_PARTS_TYPES) & df['STOCK_ID'].notna()
        # Only `type` and `vendor` are ever read off these records (see
        # _infer_stage), so subset before to_dict — converting the full ~50-col
        # group per stock was the dominant cost of this step.
        global_parts_by_stock: dict[str, list] = {}
        _parts_cols = [c for c in ('type', 'vendor') if c in df.columns]
        for sid, grp in df.loc[parts_mask, _parts_cols + ['STOCK_ID']].groupby('STOCK_ID'):
            global_parts_by_stock[str(sid)] = grp[_parts_cols].to_dict('records')

        # ── stock_to_req: first active STOCK_ID → {req_id, wo_type} ──
        active_mask = (
            df['visual_status'].isin(_ACTIVE_STATUSES)
            & (df['wo_status'].astype(str) != 'CANCELED')
            & df['STOCK_ID'].notna()
        )
        stock_to_req: dict[str, dict] = {}
        for _, row in df[active_mask].iterrows():
            sid = str(row['STOCK_ID'])
            if sid not in ('nan', 'None', 'N/A') and sid not in stock_to_req:
                stock_to_req[sid] = {'req_id': row.get('req_id'), 'wo_type': row.get('type'), 'exp_name': str(row.get('experiment_name') or '')}

        # ── Per-request computation ───────────────────────────────────
        req_stage             : dict[str, str]  = {}
        req_phase             : dict[str, str]  = {}
        req_operation         : dict[str, str]  = {}
        req_op_status         : dict[str, str]  = {}
        req_is_stalled        : dict[str, bool] = {}
        req_is_asm_review     : dict[str, bool] = {}
        req_is_finished       : dict[str, bool] = {}
        req_is_blocked        : dict[str, bool] = {}
        req_has_seq_winner    : dict[str, bool] = {}
        req_has_order_pending : dict[str, bool] = {}

        _ORDER_PROTOCOLS = proto.ORDER_PROTOS
        _ORDER_PARTS_TYPES = frozenset({
            'syn_part_synthesis_workorder',
            'oligo_synthesis_workorder',
            'plasmid_synthesis_workorder',
        })
        _ORDER_THRESHOLD = timedelta(hours=4)
        _now = datetime.now(timezone.utc)

        # Group once and reuse — both this loop and the PARTS-phase BB-fill loop
        # below need per-request slices.  Re-running df[df['req_id'] == rid] per
        # request rescans the full frame each time (O(requests × rows)).
        req_groups: dict = dict(tuple(df.groupby('req_id', dropna=True)))

        for req_id, r_df in req_groups.items():
            status = str(
                r_df['request_status'].dropna().iloc[0]
                if 'request_status' in r_df.columns and not r_df['request_status'].isna().all()
                else 'NEW'
            ).upper()
            is_finished = status in ('FULFILLED', 'SUCCEEDED')
            req_is_finished[req_id] = is_finished

            active_rows = r_df[r_df['wo_status'].astype(str) != 'CANCELED']
            req_is_blocked[req_id] = 'BLOCKED' in active_rows['visual_status'].values

            _draft_mask = (
                r_df['data_source'].eq('BIOS_DRAFT')
                if 'data_source' in r_df.columns
                else pd.Series(False, index=r_df.index)
            )
            has_real_workorders = not r_df[
                r_df['workorder_id'].notna()
                & ~r_df['workorder_id'].astype(str).str.startswith('REQ-')
                & ~_draft_mask
            ].empty

            has_life = not active_rows[
                active_rows['visual_status'].isin(
                    ['RUNNING', 'READY', 'IN_PROGRESS', 'LSP_RUNNING', 'WAITING']
                )
            ].empty

            _data_src = active_rows['data_source'] if 'data_source' in active_rows.columns else pd.Series('', index=active_rows.index)
            rc_rows     = active_rows[
                active_rows['type'].isin(_ROOT_CHAIN_TYPES) &
                (_data_src != 'BIOS_DRAFT')
            ]
            rc_exists   = not rc_rows.empty
            rc_finished = rc_rows['visual_status'].isin(['SUCCEEDED', 'FAILED', 'CANCELED']).all()

            asm_stuck = (
                rc_exists
                and rc_rows['visual_status'].isin(['BLOCKED']).any()
                and not rc_rows['visual_status'].isin(['RUNNING', 'READY', 'IN_PROGRESS']).any()
                and not active_rows[active_rows['type'] == 'lsp_workorder']['visual_status']
                    .isin(['RUNNING', 'READY', 'IN_PROGRESS']).any()
            )

            lsp_in_chain = rc_rows[rc_rows['type'] == 'lsp_workorder']
            lsp_done     = (not lsp_in_chain.empty) and (lsp_in_chain['visual_status'] == 'SUCCEEDED').any()

            is_stalled = (
                has_real_workorders
                and not is_finished
                and status != 'CANCELED'
                and not lsp_done
                and (not has_life or (rc_exists and rc_finished) or asm_stuck)
            )
            req_is_stalled[req_id] = is_stalled

            asm_rows_act = active_rows[active_rows['type'].isin(_ASM_TYPES)]
            req_is_asm_review[req_id] = (
                has_real_workorders
                and not is_finished
                and status != 'CANCELED'
                and asm_rows_act['visual_status'].eq('SUCCEEDED').any()
                and asm_rows_act['visual_status'].isin(['READY', 'WAITING']).any()
            )

            # fulfills_request stocks — reused for phase, seq_winner
            _root_stocks = set(
                r_df[(r_df['fulfills_request'] == True) & ~r_df['STOCK_ID'].fillna('').str.startswith('#')]['STOCK_ID'].dropna()
            ) if 'fulfills_request' in r_df.columns else set()

            # ── phase label (PARTS / ASM / LSP) ──────────────────────────
            _phase_rows  = r_df[r_df['visual_status'].isin(_PHASE_ACTIVE) & (r_df['wo_status'].astype(str) != 'CANCELED')]
            _lsp_ph      = _phase_rows[_phase_rows['type'] == 'lsp_workorder']
            _asm_ph      = _phase_rows[_phase_rows['type'].isin(_ASM_TYPES) & _phase_rows['STOCK_ID'].astype(str).isin(_root_stocks)]
            _parts_ph    = _phase_rows[(_phase_rows['type'] != 'lsp_workorder') & ~_phase_rows['STOCK_ID'].astype(str).isin(_root_stocks)]
            _asm_progressing = _asm_ph[_asm_ph['visual_status'].isin({'RUNNING', 'READY', 'IN_PROGRESS', 'BLOCKED'})]
            if not _lsp_ph.empty:
                req_phase[req_id] = 'LSP'
            elif not _asm_progressing.empty:
                req_phase[req_id] = 'ASM'
            elif not _asm_ph.empty or not _parts_ph.empty:
                req_phase[req_id] = 'PARTS'
            elif is_stalled:
                # Fallback for stalled requests: infer from highest-priority non-canceled WO type
                _nc = r_df[r_df['wo_status'].astype(str) != 'CANCELED']
                if not _nc[_nc['type'] == 'lsp_workorder'].empty:
                    req_phase[req_id] = 'LSP'
                elif not _nc[_nc['type'].isin(_ASM_TYPES) & _nc['STOCK_ID'].astype(str).isin(_root_stocks)].empty:
                    req_phase[req_id] = 'ASM'
                else:
                    req_phase[req_id] = 'PARTS'
            else:
                req_phase[req_id] = ''

            # ── active operation (highest-priority RD/RU step) ───────────
            # Scoped to root/fulfills_request stocks + LSP rows to avoid foreign
            # backbone constructs (e.g. pAI-21680) polluting the operation label.
            _nc = r_df[r_df['wo_status'].astype(str) != 'CANCELED']
            _op_rows = _nc[
                _nc['STOCK_ID'].isin(_root_stocks) |
                (_nc['type'] == 'lsp_workorder') |
                (_nc['type'].isin(_PARTS_TYPES))
            ] if _root_stocks else _nc
            _best_pri, _best_label, _best_state = -1, '', ''
            # Iterate raw cell arrays instead of iterrows() — avoids building a
            # pandas Series per row (the dominant cost of this step).
            _op_pn = _op_rows['protocol_name'].to_numpy()
            _op_st = _op_rows['operation_state'].to_numpy()
            for _pn_cell, _st_cell in zip(_op_pn, _op_st):
                if not isinstance(_pn_cell, (list, np.ndarray)) or not isinstance(_st_cell, (list, np.ndarray)):
                    continue
                for _p, _s in zip(_pn_cell, _st_cell):
                    _key = (str(_p), str(_s))
                    if _key in _PROTO_MAP:
                        _pri, _lbl = _PROTO_MAP[_key]
                        if _pri > _best_pri:
                            _best_pri, _best_label, _best_state = _pri, _lbl, _s
            req_operation[req_id] = _best_label
            _phase = req_phase.get(req_id, '')
            if _best_state == 'RD':
                req_op_status[req_id] = 'READY'
            elif _best_state == 'RU':
                req_op_status[req_id] = 'LSP_RUNNING' if _phase == 'LSP' else 'RUNNING'
            else:
                req_op_status[req_id] = 'WAITING' if _phase == 'PARTS' else _phase or ''

            # seq winner: deliverable constructs have ≥1 seq-confirmed colony,
            # no LSP workorder exists yet — winner in hand, not yet acted on.
            # Restricted to fulfills_request=True stocks to avoid foreign-construct
            # inputs (e.g. backbone Gibson pAI-21680) triggering the flag.
            # Draft placeholder stocks (#-prefixed) are excluded.
            _root_rows = r_df[r_df['STOCK_ID'].isin(_root_stocks)] if _root_stocks else r_df
            _seq_col = _root_rows['seq_confirmed'] if 'seq_confirmed' in _root_rows.columns else None
            _has_lsp = 'lsp_workorder' in active_rows['type'].values
            req_has_seq_winner[req_id] = (
                has_real_workorders
                and not is_finished
                and status != 'CANCELED'
                and not _has_lsp
                and _seq_col is not None
                and pd.to_numeric(_seq_col, errors='coerce').fillna(0).gt(0).any()
            )

            # order pending: any ordering-step part (synpart/plasmid/oligo) stuck in
            # Synthesis Order or Order Oligos (RD/RU) for > 4 hours
            _order_pending = False
            if has_real_workorders and not is_finished and status != 'CANCELED':
                parts_rows = active_rows[active_rows['type'].isin(_ORDER_PARTS_TYPES)]
                _pr_pn = parts_rows['protocol_name'].to_numpy()
                _pr_st = parts_rows['operation_state'].to_numpy()
                _pr_created = (
                    parts_rows['wo_created_at'].to_numpy()
                    if 'wo_created_at' in parts_rows.columns
                    else np.full(len(parts_rows), None)
                )
                for _pn_cell, _st_cell, _created in zip(_pr_pn, _pr_st, _pr_created):
                    _active = _active_protocols_raw(_pn_cell, _st_cell)
                    if _active & _ORDER_PROTOCOLS:
                        try:
                            _created = pd.Timestamp(_created)
                            if _created.tzinfo is None:
                                _created = _created.tz_localize('UTC')
                            if (_now - _created.to_pydatetime()) > _ORDER_THRESHOLD:
                                _order_pending = True
                                break
                        except Exception:
                            pass
            req_has_order_pending[req_id] = _order_pending

            req_stage[req_id] = _infer_stage(
                r_df, active_rows, is_stalled, has_real_workorders,
                status, is_finished, global_parts_by_stock, stock_to_req,
            )

        # ── PARTS-phase: fill blank req_operation with BB source info ────
        _asm_types_set = {'golden_gate_workorder', 'gibson_workorder'}
        for _rid, _rph in req_phase.items():
            if _rph != 'PARTS' or req_operation.get(_rid):
                continue
            _rdf = req_groups.get(_rid)
            if _rdf is None:
                continue
            _asm_waiting = _rdf[
                _rdf['type'].isin(_asm_types_set) &
                (_rdf['visual_status'] == 'WAITING')
            ]
            if _asm_waiting.empty:
                continue
            _bb_raw = str(_asm_waiting.iloc[0].get('backbone', '') or '').split(':')[0].strip()
            if not _bb_raw or _bb_raw in ('nan', 'None', ''):
                continue
            _inf = stock_to_req.get(_bb_raw)
            if _inf and _inf.get('req_id') and _inf['req_id'] != _rid:
                _exp = _inf.get('exp_name', '')
                _exp_s = (_exp[:22] + '…') if len(_exp) > 22 else _exp
                req_operation[_rid] = f"BB: {_bb_raw} · {_exp_s}"
                req_op_status[_rid] = 'WAITING'

        # ── Broadcast back to all rows ────────────────────────────────
        df['stage']               = df['req_id'].map(req_stage)
        df['req_phase']           = df['req_id'].map(req_phase).fillna('')
        df['req_operation']       = df['req_id'].map(req_operation).fillna('')
        df['req_op_status']       = df['req_id'].map(req_op_status).fillna('')
        df['is_stalled']          = df['req_id'].map(req_is_stalled).fillna(False)
        df['is_asm_review']       = df['req_id'].map(req_is_asm_review).fillna(False)
        df['is_finished']         = df['req_id'].map(req_is_finished).fillna(False)
        df['is_blocked']          = df['req_id'].map(req_is_blocked).fillna(False)
        df['has_seq_winner']      = df['req_id'].map(req_has_seq_winner).fillna(False)
        df['has_order_pending']   = df['req_id'].map(req_has_order_pending).fillna(False)

        log.info(
            "Enrichment complete: %d requests → %d stalled, %d asm-review, %d seq-winner, %d order-pending",
            len(req_stage), sum(req_is_stalled.values()), sum(req_is_asm_review.values()),
            sum(req_has_seq_winner.values()), sum(req_has_order_pending.values()),
        )
        return df

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

from dnasc.logger import get_logger

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


def _active_protocols(row) -> set:
    """Return set of protocol names whose operation_state is RD or RU."""
    pn = row.get('protocol_name')
    ps = row.get('operation_state')
    if isinstance(pn, np.ndarray): pn = pn.tolist()
    if isinstance(ps, np.ndarray): ps = ps.tolist()
    if not isinstance(pn, list) or not isinstance(ps, list):
        return set()
    return {p for p, s in zip(pn, ps) if s in ('RD', 'RU')}


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
        if 'LSP Releasing' in p:                                                    return 'Releasing'
        if 'LSP Reviewing' in p:                                                    return 'Reviewing'
        if p & {'DNA Quantification', 'NGS Sequence Confirmation', 'Fragment Analyzer'}: return 'LSP QC'
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
            if p & {'DNA Quantification', 'NGS Sequence Confirmation'}:             return 'Assembly QC'
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
        global_parts_by_stock: dict[str, list] = {}
        for sid, grp in df[parts_mask].groupby('STOCK_ID'):
            global_parts_by_stock[str(sid)] = grp.to_dict('records')

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
                stock_to_req[sid] = {'req_id': row.get('req_id'), 'wo_type': row.get('type')}

        # ── Per-request computation ───────────────────────────────────
        req_stage       : dict[str, str]  = {}
        req_is_stalled  : dict[str, bool] = {}
        req_is_asm_review: dict[str, bool] = {}
        req_is_finished : dict[str, bool] = {}
        req_is_blocked  : dict[str, bool] = {}

        for req_id, r_df in df.groupby('req_id', dropna=True):
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

            rc_rows     = active_rows[active_rows['type'].isin(_ROOT_CHAIN_TYPES)]
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

            req_stage[req_id] = _infer_stage(
                r_df, active_rows, is_stalled, has_real_workorders,
                status, is_finished, global_parts_by_stock, stock_to_req,
            )

        # ── Broadcast back to all rows ────────────────────────────────
        df['stage']         = df['req_id'].map(req_stage)
        df['is_stalled']    = df['req_id'].map(req_is_stalled).fillna(False)
        df['is_asm_review'] = df['req_id'].map(req_is_asm_review).fillna(False)
        df['is_finished']   = df['req_id'].map(req_is_finished).fillna(False)
        df['is_blocked']    = df['req_id'].map(req_is_blocked).fillna(False)

        log.info(
            "Enrichment complete: %d requests → %d stalled, %d asm-review",
            len(req_stage), sum(req_is_stalled.values()), sum(req_is_asm_review.values()),
        )
        return df

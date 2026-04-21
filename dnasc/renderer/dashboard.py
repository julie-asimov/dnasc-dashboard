"""
dnasc/renderer/dashboard.py
────────────────────────────
Loads logo assets from the scripts/ directory, then delegates to
render_all_projects_dashboard() from your existing renderer logic.

Public API:
    from dnasc.renderer import render_dashboard
    html = render_dashboard(final_df)
"""

from __future__ import annotations
import base64
import warnings
from pathlib import Path

import re
import json
import random
import hashlib
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from dnasc.logger import get_logger
try:
    from dnasc.renderer.lsp_capacity import render_lsp_capacity_tab
except ImportError:
    def render_lsp_capacity_tab(_df):
        return "<p style='color:#6b7280;padding:1rem;'>LSP capacity view not available.</p>"

log = get_logger(__name__)

# ── Asset resolution ──────────────────────────────────────────────────────────
# Assets live in scripts/ (two levels up from this file: renderer/ → dnasc/ → scripts/)
_SCRIPTS_DIR = Path(__file__).parent.parent.parent.resolve()

_ASSET_FILES = {
    "logo":     "dnasc_logo.png",
    "tracking": "tracking_icon.png",
    "metrics":  "metrics_icon.png",
    "cost":     "cost_icon.png",
}


def _load_b64(filename: str) -> str:
    path = _SCRIPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Asset not found: {path}\n"
            f"Place '{filename}' in: {_SCRIPTS_DIR}"
        )
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# Pre-load at import time — fast for repeated render calls
try:
    _ASSETS = {k: _load_b64(v) for k, v in _ASSET_FILES.items()}
except FileNotFoundError as e:
    warnings.warn(str(e))
    _ASSETS = {k: "" for k in _ASSET_FILES}


# ── Paste render_all_projects_dashboard here ──────────────────────────────────
# Copy the full function body from your Colab renderer.py unchanged.
# Only the function signature + the call at the bottom of this file matter here.

def render_all_projects_dashboard(
    df: pd.DataFrame,
    logo_b64: str = "",
    tracking_icon_b64: str = "",
    metrics_icon_b64: str = "",
    cost_icon_b64: str = "",
    generated_at: str = "",
) -> str:

    if df.empty:
        return "<h3>No data found.</h3>"

    # =========================================================================
    # 1. HELPERS
    # =========================================================================
    def to_est(dt_val):
        if pd.isna(dt_val) or dt_val == '': return None
        try:
            dt = pd.to_datetime(dt_val)
            if dt.tz is None: dt = dt.tz_localize('UTC')
            return dt.tz_convert('US/Eastern')
        except: return None

    def clean_plate_id(val):
        if pd.isna(val): return None
        match = re.search(r'(\d+)', str(val))
        return match.group(1) if match else None

    def get_pai_sort_key(req_df):
        stock_ids = req_df['STOCK_ID'].dropna().astype(str).values
        min_pai = 99999999
        found = False
        for s in stock_ids:
            match = re.search(r'pAI-(\d+)', s, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if val < min_pai:
                    min_pai = val
                    found = True
        if not found: return 99999999
        return min_pai

    def parse_pipeline_operations(protocol_names, operation_states, operation_starts, job_ids, well_locations_list, operation_ready_times, ngs_run_numbers=None):
        # --- STEP 0: TYPE PROTECTION ---
        # Force all inputs to be lists. If they are NaN or None, they become empty lists.
        def ensure_list(x):
            if isinstance(x, (list, np.ndarray)): return list(x)
            return []

        protocol_names = ensure_list(protocol_names)
        operation_states = ensure_list(operation_states)
        operation_starts = ensure_list(operation_starts)
        job_ids = ensure_list(job_ids)
        well_locations_list = ensure_list(well_locations_list)
        operation_ready_times = ensure_list(operation_ready_times)
        ngs_run_numbers = ensure_list(ngs_run_numbers)

        if not protocol_names:
            return []

        # --- STEP 1: BUILD RAW OPERATIONS ---
        raw_ops = []
        for i in range(len(protocol_names)):
            # Safe access using index checks
            r_time = to_est(operation_ready_times[i]) if i < len(operation_ready_times) else None
            s_time = to_est(operation_starts[i]) if i < len(operation_starts) else None
            run_num = ngs_run_numbers[i] if i < len(ngs_run_numbers) else None

            raw_ops.append({
                'protocol': protocol_names[i],
                'state': operation_states[i] if i < len(operation_states) else 'Unknown',
                'start_time': s_time,
                'ready_time': r_time,
                'job_id': job_ids[i] if i < len(job_ids) else None,
                'well_location': well_locations_list[i] if i < len(well_locations_list) else None,
                'run_number': run_num if (run_num is not None and pd.notna(run_num)) else None,
            })

        # --- STEP 2: SORTING ---
        # Sort by ready_time then start_time
        raw_ops.sort(key=lambda x: (
            0 if pd.notna(x['ready_time']) else 1,
            x['ready_time'] if pd.notna(x['ready_time']) else pd.Timestamp.min,
            0 if pd.notna(x['start_time']) else 1,
            x['start_time'] if pd.notna(x['start_time']) else pd.Timestamp.min
        ))

        # --- STEP 2b: DEDUP same-protocol same-state ops (e.g. Rearray from OpTracker + downstream) ---
        # Keep the op that has a job_id; merge wells from both sources.
        seen: dict = {}
        deduped: list = []
        for op in raw_ops:
            key = (op['protocol'], op['state'])
            if key not in seen:
                seen[key] = len(deduped)
                deduped.append(op)
            else:
                existing = deduped[seen[key]]
                if op['job_id'] is not None and pd.notna(op['job_id']) and (existing['job_id'] is None or pd.isna(existing['job_id'])):
                    existing['job_id'] = op['job_id']
                if op['start_time'] and (not existing['start_time'] or op['start_time'] < existing['start_time']):
                    existing['start_time'] = op['start_time']
        raw_ops = deduped

        # --- STEP 3: SOFT FAIL & GROUPING ---
        protocol_success = {op['protocol'] for op in raw_ops if op['state'] == 'SC'}
        groupable_protocols = {
            'DNA Quantification', 'Rearray 96 to 384', 'NGS Sequence Confirmation',
            'Fragment Analyzer', 'Sanger Sequencing'
        }

        result = []
        current_group = []

        for op in raw_ops:
            protocol = op['protocol']
            state = op['state']

            # Skip Fails and Cancels if a Success exists for the same protocol
            if state in ('FA', 'CA') and protocol in protocol_success:
                continue

            state_map = {
                'SC': {'state': 'Completed', 'class': 'succeeded'},
                'FA': {'state': 'Failed', 'class': 'failed'},
                'RU': {'state': 'Running', 'class': 'running'},
                'RD': {'state': 'Ready', 'class': 'ready'},
                'CA': {'state': 'Canceled', 'class': 'canceled'}
            }
            state_info = state_map.get(state, {'state': 'Unknown', 'class': 'pending'})

            clean_op = {
                'queue': protocol,
                'state': state_info['state'],
                'class': state_info['class'],
                'start_time': op['start_time'],
                'ready_time': op['ready_time'],
                'job_id': op['job_id'],
                'wells': [op['well_location']] if pd.notna(op['well_location']) else [],
                'run_numbers': [op['run_number']] if op['run_number'] is not None else [],
            }

            # Grouping Engine
            def _job_null(j):
                return j is None or (isinstance(j, float) and pd.isna(j))
            if not current_group:
                current_group.append(clean_op)
            else:
                prev_op = current_group[-1]
                same_job = (_job_null(clean_op['job_id']) and _job_null(prev_op['job_id'])) or \
                           (not _job_null(clean_op['job_id']) and not _job_null(prev_op['job_id']) and clean_op['job_id'] == prev_op['job_id'])
                if protocol == prev_op['queue'] and protocol in groupable_protocols and same_job:
                    # Merge wells and run numbers; update state if necessary
                    prev_op['wells'].extend(clean_op['wells'])
                    prev_op['run_numbers'].extend(clean_op['run_numbers'])
                    if clean_op['class'] == 'running':
                        prev_op['state'] = 'Running'
                        prev_op['class'] = 'running'
                else:
                    result.append(current_group[0])
                    current_group = [clean_op]

        if current_group:
            result.append(current_group[0])

        return result

    def get_active_step_info(row):
        queue_data = parse_pipeline_operations(
            row.get('protocol_name', []), row.get('operation_state', []), row.get('operation_start', []),
            row.get('job_id', []), row.get('well_location', []), row.get('operation_ready', []),
            row.get('ngs_run_number', [])
        )
        if not queue_data: return None
        for op in queue_data:
            if op['state'] == 'Ready': return f"{op['queue']}: Ready"
            if op['state'] == 'Running': return f"{op['queue']}: Running"
        return None

    def format_type_label(type_str):
        """Format workorder type with proper capitalization"""
        label = type_str.replace('_workorder', '').replace('_', ' ').title()
        # Fix specific capitalizations
        label = label.replace('Lsp', 'LSP').replace('Pcr', 'PCR').replace('Ngs', 'NGS')
        return label

    # =========================================================================
    # 2. LOGIC & LOOKUPS
    # =========================================================================
    parent_details = {}
    well_to_root = {}

    for _, row in df.iterrows():
        root = row.get('root_work_order_id')
        if pd.notna(root):
            search_text = str(row.get('all_locations', '')) + " " + str(row.get('operation_well_locations', ''))
            found_wells = re.findall(r'\b(\d{5,8})\b', search_text)
            for w in found_wells: well_to_root[w] = root

    # Global parts lookup: stock_id → list of parts row dicts across ALL requests.
    # Lets GG/Gibson sections fan in syn parts / PCR workorders that belong to a
    # different request but are referenced in the GG's parts_json / backbone_json.
    # Also includes fulfills_request=False GG/Gibson workorders (intermediate plasmids
    # assembled as inputs) so they can be fanned in to the main assembly's Parts section.
    _global_parts_rows_by_stock: dict = {}
    _global_parts_types = {'oligo_synthesis_workorder', 'pcr_workorder',
                           'plasmid_synthesis_workorder', 'syn_part_synthesis_workorder'}
    _global_asm_types = {'gibson_workorder', 'golden_gate_workorder'}
    for _, _gpr in df[df['type'].isin(_global_parts_types | _global_asm_types)].iterrows():
        if _gpr.get('type') in _global_asm_types and _gpr.get('fulfills_request') != False:
            continue
        _gsid = str(_gpr.get('STOCK_ID', '') or '').strip()
        if _gsid and _gsid not in ('nan', 'None', ''):
            _global_parts_rows_by_stock.setdefault(_gsid, []).append(_gpr.to_dict())

    # Lookup: workorder_id → assembly_plan_id (used to validate cross-request fan-in)
    _wid_to_assembly_plan_id: dict = {}
    for _, _r in df[df['assembly_plan_id'].notna()].iterrows():
        _wid_to_assembly_plan_id[str(_r['workorder_id'])] = str(_r['assembly_plan_id'])

    for _, row in df.iterrows():
        wid = str(row['workorder_id'])
        _jid = row.get('job_id')
        if isinstance(_jid, np.ndarray): _jid = _jid.tolist()
        job_id = _jid[0] if isinstance(_jid, list) and len(_jid) > 0 else None
        completion_time = None
        op_states, op_starts = row.get('operation_state'), row.get('operation_start')
        op_protocols = row.get('protocol_name')
        if isinstance(op_protocols, np.ndarray): op_protocols = op_protocols.tolist()
        if isinstance(op_states, np.ndarray): op_states = op_states.tolist()
        if isinstance(op_starts, np.ndarray): op_starts = op_starts.tolist()
        if isinstance(op_states, list) and isinstance(op_starts, list):
            # For GG/Gibson, use the assembly step time (not the last op which may be NGS)
            _asm_proto_map = {'golden_gate_workorder': 'Golden Gate Assembly', 'gibson_workorder': 'Gibson Assembly'}
            _target_proto = _asm_proto_map.get(str(row.get('type', '')))
            if _target_proto and isinstance(op_protocols, list):
                for proto, state, start in zip(op_protocols, op_states, op_starts):
                    if proto == _target_proto and state in ('SC', 'FA') and pd.notna(start):
                        completion_time = to_est(start); break
            if not completion_time:
                valid_times = [to_est(start) for state, start in zip(op_states, op_starts) if state in ('SC', 'FA') and pd.notna(start)]
                if valid_times: completion_time = max(valid_times)
        if not completion_time: completion_time = to_est(row['wo_created_at'])
        plate_id = None
        json_str = row.get('all_protocol_plates', '{}')
        if pd.notna(json_str) and json_str != '{}':
            try:
                data = json.loads(json_str)
                for proto in ['Golden Gate Assembly', 'Gibson Assembly', 'PCR', 'DNA Quantification']:
                    if proto in data:
                        raw = str(data[proto]).split(',')[0]; clean = clean_plate_id(raw)
                        if clean: plate_id = clean; break
            except: pass
        if not plate_id:
            locs = str(row.get('all_locations', '')) + " " + str(row.get('colony_plates', ''))
            match = re.search(r'Plate(\d+)', locs)
            if match: plate_id = match.group(1)
        parent_details[wid] = {
            'type': format_type_label(row['type']),
            'job': job_id, 'plate': plate_id,
            'completion_time': completion_time,
            'completion_str': completion_time.strftime('%m/%d/%Y %H:%M') if completion_time else ""
        }

    id_to_root = df.set_index('workorder_id')['root_work_order_id'].astype(str).to_dict()
    valid_ids = set(df['workorder_id'].astype(str).values)

    def get_visual_status(row):
        raw = row.get('wo_status')
        original = str(raw).strip().upper() if pd.notna(raw) else ''

        # NaN/UNKNOWN guard
        if original in ('', 'NAN', 'NONE'):
            pre = row.get('visual_status')
            if pd.notna(pre) and str(pre).upper() not in ('NAN', 'NONE', ''):
                return str(pre).upper(), False
            return 'IN_PROGRESS', False

        if original == 'UNKNOWN':
            return 'UNKNOWN', False

        # Colony-based overrides (only for assembly/transformation types)
        colony_types = ['gibson_workorder', 'golden_gate_workorder', 'transformation_workorder',
                        'transformation_offline_operation', 'streakout_operation']

        if row['type'] not in colony_types:
            # For non-colony types, trust _bridge_status (visual_status) over raw wo_status.
            # wo_status can be stale (e.g. RUNNING) while _bridge_status correctly shows SUCCEEDED.
            pre = row.get('visual_status')
            if pd.notna(pre) and str(pre).upper() not in ('NAN', 'NONE', ''):
                return str(pre).upper(), False
            return original, False

        if row['type'] in colony_types:
            tot = int(row.get('total_colonies')) if pd.notna(row.get('total_colonies')) else 0
            seq = int(row.get('seq_confirmed')) if pd.notna(row.get('seq_confirmed')) else 0

            # Software Fail: BIOS=FAILED but colonies passed seq → override to SUCCEEDED
            if original == 'FAILED' and seq > 0:
                return 'SUCCEEDED', True

            # Status lag: BIOS still RUNNING/IN_PROGRESS but sequencing confirmed → SUCCEEDED
            # (e.g. engineer canceled OpTracker ops after completion, BIOS never auto-closed)
            if original in ('RUNNING', 'IN_PROGRESS') and seq > 0:
                return 'SUCCEEDED', False

            # Zero Colony: BIOS=SUCCEEDED but no colonies in LIMS → always a failure
            if original == 'SUCCEEDED':
                if tot == 0:
                    return 'FAILED', False

                protocols = row.get('protocol_name')
                if isinstance(protocols, np.ndarray):
                    protocols = list(protocols)
                if isinstance(protocols, list) and len(protocols) > 0:
                    # Protocols that indicate colonies were actually processed
                    colony_progress_protocols = {
                        'Rearray 96 to 384', 'DNA Quantification',
                        'NGS Sequence Confirmation', 'Fragment Analyzer',
                        'Sanger Sequencing'
                    }
                    states = row.get('operation_state')
                    if isinstance(states, np.ndarray):
                        states = list(states)
                    if isinstance(states, list):
                        # Check if ANY colony-progress protocol completed successfully
                        has_colony_progress = False
                        for proto, state in zip(protocols, states):
                            if proto in colony_progress_protocols and state == 'SC':
                                has_colony_progress = True
                                break

                        if not has_colony_progress:
                            # Confirm miniprep completed (pipeline did run past transformation)
                            has_miniprep = any(
                                p == 'Create Minipreps and Glycerol Stocks' and s == 'SC'
                                for p, s in zip(protocols, states)
                            )
                            if has_miniprep:
                                return 'FAILED', False
                        else:
                            # seq=0 after colony progress — only FAILED if a sequencing
                            # protocol actually ran (NGS/Sanger/Fragment Analyzer SC).
                            # Rearray or Quant done but NGS not yet scheduled = still running.
                            if tot > 0 and seq == 0:
                                _seq_protocols = {'NGS Sequence Confirmation', 'Fragment Analyzer', 'Sanger Sequencing'}
                                if any(p in _seq_protocols and s == 'SC' for p, s in zip(protocols, states)):
                                    return 'FAILED', False
                                # Use real OpTracker state if an op is active
                                if any(s == 'RU' for s in states):
                                    return 'RUNNING', False
                                if any(s == 'RD' for s in states):
                                    return 'READY', False
                                return 'IN_PROGRESS', False
                else:
                    # No OpTracker protocol data (e.g. synthetic LIMS-only streakout).
                    # Colonies were picked but sequencing not yet done → in progress.
                    if tot > 0 and seq == 0:
                        return 'RUNNING', False

        return original, False

    df[['visual_status', 'is_software_fail']] = df.apply(lambda row: pd.Series(get_visual_status(row)), axis=1)

    status_priority = {'RUNNING': 0, 'IN_PROGRESS': 0, 'BLOCKED': 1, 'WAITING': 2, 'READY': 2, 'DRAFT': 2, 'SUCCEEDED': 3, 'FAILED': 4, 'CANCELED': 5}
    df['status_rank'] = df['visual_status'].map(status_priority).fillna(99)
    df['req_rank'] = df.groupby('req_id')['status_rank'].transform('min')
    df['group_rank'] = df.groupby('root_work_order_id')['status_rank'].transform('min')

    def is_visible(row):
        if row['wo_status'] != 'CANCELED': return True
        pnames = row.get('protocol_name')
        if isinstance(pnames, list) and len(pnames) > 0 and pd.notna(pnames[0]): return True
        return False
    df['is_visible'] = df.apply(is_visible, axis=1)

    if 'experiment_name' not in df.columns: df['experiment_name'] = "Unknown Project"
    if 'experiment_created_at' in df.columns:
        df = df.sort_values(by=['experiment_created_at', 'req_rank', 'req_id', 'group_rank'], ascending=[False, True, True, True])
    else:
        df = df.sort_values(by=['req_rank', 'req_id', 'group_rank'])

    # =========================================================================
    # HTML CSS WITH TABS
    # =========================================================================

    # Part 1: CSS and JS (no f-string needed - no variables)
    html = """
    <style>
    /* SCALE */
    :root {
      --text-xs:   7px;   /* timestamps, monospace IDs */
      --text-sm:   9px;   /* table cells, badges, labels */
      --text-base: 11px;  /* detail content */
      --text-md:   12px;  /* req names, section headers */
      --text-lg:   13px;  /* dashboard title */
    }
    /* BASE */
    * { -webkit-font-smoothing: antialiased; box-sizing: border-box; }
    body { background: #e8e8ed; padding: 6px; margin: 0; }
    .dashboard-container { max-width: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 11px; }

    /* LOGO HEADER */
    .dashboard-header { display: flex; align-items: center; padding: 8px 12px; background: white; border-bottom: 1px solid #e5e5e7; gap: 12px; border-radius: 5px 5px 0 0; }
    .dashboard-logo { height: 32px; width: auto; }
    .dashboard-title { font-size: 13px; font-weight: 800; color: #1d1d1f; }
    .dashboard-updated { font-size: 10px; color: #8e8e93; margin-left: auto; }
    .tab-icon-img { height: 36px; width: 36px; border-radius: 6px; }

    /* TABS */
    .tab-container { margin-bottom: 0; }
    .tab-nav { display: flex; gap: 0; border-bottom: none; background: linear-gradient(135deg, #7c3aed 0%, #be185d 100%); border-radius: 0; overflow: hidden; }
    .tab-btn { padding: 10px 16px; font-size: 11px; font-weight: 700; color: rgba(255,255,255,0.9); background: transparent; border: none; cursor: pointer; border-right: 1px solid rgba(255,255,255,0.2); display: flex; align-items: center; gap: 6px; }
    .tab-btn:last-child { border-right: none; }
    .tab-btn:hover { background: rgba(255,255,255,0.1); }
    .tab-btn.active { background: rgba(255,255,255,0.2); border-bottom: 3px solid #fff; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .tab-icon-img { height: 24px; width: 24px; border-radius: 4px; }
    .tab-btn .tab-text { display: none; }
    .tab-btn:hover .tab-text { display: inline; }

    /* UNDER CONSTRUCTION */
    .under-construction { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; text-align: center; }
    .uc-icon { font-size: 40px; margin-bottom: 10px; }
    .uc-title { font-size: 18px; font-weight: 800; color: #1d1d1f; margin-bottom: 5px; }
    .uc-subtitle { font-size: 11px; color: #86868b; max-width: 300px; line-height: 1.4; }
    .uc-badge { background: linear-gradient(135deg, #7c3aed, #be185d); color: white; padding: 4px 10px; border-radius: 10px; font-size: 9px; font-weight: 700; text-transform: uppercase; margin-top: 10px; }

    /* CONTROLS */
    .controls-container { display: flex; align-items: center; gap: 10px; padding: 6px 10px; background: #f5f5f7; border-bottom: 1px solid #d1d1d6; }
    .toggle-wrapper { display: flex; align-items: center; gap: 5px; }
    .toggle-label { font-size: 10px; font-weight: 600; color: #86868b; white-space: nowrap; }
    .switch { position: relative; display: inline-block; width: 28px; height: 16px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #d1d1d6; border-radius: 16px; }
    .slider:before { position: absolute; content: ""; height: 12px; width: 12px; left: 2px; bottom: 2px; background-color: white; border-radius: 50%; }
    input:checked + .slider { background: linear-gradient(135deg, #7c3aed, #be185d); }
    input:checked + .slider:before { left: 14px; }

    /* PROJECT WRAPPER */
    .project-wrapper { margin-bottom: 6px; border-bottom: 1px solid #d1d1d6; padding-bottom: 6px; }
    .header-banner { color: white; padding: 6px 8px; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-bottom: 4px; cursor: pointer; }
    .header-banner:hover { opacity: 0.95; }
    .header-title { font-size: 12px; font-weight: 700; white-space: nowrap; }
    .header-main-stat { font-size: 10px; font-weight: 700; background: rgba(255,255,255,0.2); padding: 1px 5px; border-radius: 3px; white-space: nowrap; }
    .stat-item { background: rgba(255,255,255,0.2); padding: 1px 4px; border-radius: 2px; font-size: 8px; font-weight: 600; white-space: nowrap; }
    .stat-label { font-weight: 800; color: rgba(255,255,255,0.9); margin-right: 2px; font-size: 9px; }

    /* REQUEST CARDS */
    .req-card { border: 1px solid #d1d1d6; background: white; margin-top: 3px; border-radius: 3px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .req-title-bar { padding: 3px 6px; border-bottom: 1px solid #e5e5e7; display: flex; justify-content: space-between; align-items: center; border-left-width: 3px; border-left-style: solid; gap: 4px; min-height: 20px; }
    .req-name { font-size: 10px; font-weight: 700; color: #1d1d1f; white-space: nowrap; }
    .req-meta { font-size: 7px; color: #86868b; margin-top: 0px; }

    /* ASSEMBLY SECTIONS */
    .assembly-section { margin: 3px 6px; border: 1px solid #e5e5e7; border-radius: 3px; }
    .assembly-section.dimmed { opacity: 0.5; }
    .dropdown-btn { width: 100%; background: #fafafa; border: none; padding: 3px 6px; text-align: left; cursor: pointer; display: flex; align-items: center; font-size: 9px; }
    .dropdown-btn:hover { background: #f0f0f2; }
    .dropdown-btn.active-header { background: #e5e5e7; }
    .dropdown-icon { color: #86868b; margin-right: 5px; font-size: 8px; }
    .dropdown-icon.open { transform: rotate(90deg); }
    .assembly-info { flex-grow: 1; display: flex; align-items: center; gap: 6px; white-space: nowrap; overflow: hidden; }
    .assembly-type { font-weight: 800; color: #4b5563; font-size: 10px; }
    .assembly-counts { font-weight: 600; color: #86868b; font-size: 9px; }

    /* BADGES */
    .badge { padding: 1px 4px; border-radius: 2px; font-size: 8px; font-weight: 700; text-transform: uppercase; white-space: nowrap; }
    .status-SUCCEEDED, .status-FULFILLED { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; padding: 1px 7px; }
    .status-FAILED    { background: #fff1f5; color: #be185d; border: 1px solid #fecdd3; }
    .status-CANCELED  { background: #f5f5f7; color: #6b7280; border: 1px solid #d1d5db; }
    .status-RUNNING   { background: #f5f3ff; color: #7c3aed; border: 1px solid #ddd6fe; }
    .status-LSP_RUNNING { background: #fdf4ff; color: #a21caf; border: 1px solid #f0abfc; }
    .status-IN_PROGRESS { background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe; }
    .status-READY     { background: #f0fdfa; color: #0d9488; border: 1px solid #99f6e4; }
    .status-WAITING   { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
    .status-DRAFT     { background: #f1f5f9; color: #64748b; border: 1px solid #cbd5e1; }
    .status-UNKNOWN   { background: #f5f5f7; color: #6b7280; border: 1px solid #d1d5db; }
    .status-BLOCKED   { background: #be185d; color: white; border: none; }

    /* STOCK TAG */
    .stock-tag { background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db; padding: 1px 4px; border-radius: 2px; font-family: monospace; font-weight: 700; font-size: 9px; white-space: nowrap; }
    .dropdown-btn .stock-tag { font-size: 8px; padding: 1px 3px; }
    .stock-id-badge { background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db; padding: 1px 4px; border-radius: 2px; font-family: monospace; font-weight: 700; font-size: 9px; white-space: nowrap; }
    .stock-id-badge.matches-root { background: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; }
    .stock-id-badge.secondary-root { background: #ddd6fe; color: #4c1d95; border: 1px solid #7c3aed; }
    .wo-id-tag { background: none; color: #374151; padding: 1px 3px; font-family: monospace; font-size: 7px; }

    /* TABLE */
    .content-pane { display: none; padding: 0; background: #fff; }
    .wo-table { width: 100%; border-collapse: collapse; font-size: 9px !important; }
    .wo-table th { text-align: left; color: #86868b; padding: 3px 5px; border-bottom: 1px solid #e5e5e7; font-size: 8px; font-weight: 700; background: #fafafa; text-transform: uppercase; white-space: nowrap; }
    .wo-table td { padding: 3px 5px; border-bottom: 1px solid #f5f5f7; vertical-align: top; font-weight: 600; color: #1d1d1f; }
    .wo-table .stock-tag { font-size: 8px; padding: 1px 3px; }

    /* TREE ROWS */
    .tree-row-0 { border-left: 3px solid #d1d1d6 !important; background: #ffffff !important; }
    .tree-row-1 { border-left: 3px solid #d1d1d6 !important; background: #f9f9fb !important; }
    .tree-row-2 { border-left: 3px solid #d1d1d6 !important; background: #f4f4f6 !important; }
    .tree-line-icon { color: #d1d1d6; margin-right: 2px; font-family: monospace; font-size: 9px; }
    .source-badge { font-size: 7px; padding: 0px 2px; border-radius: 2px; background: #f0f0f2; color: #86868b; margin-left: 2px; }

    /* TIMELINE */
    .timeline-container { position: relative; padding-left: 2px; }
    .timeline-row { display: flex; align-items: flex-start; margin-bottom: 2px; position: relative; min-height: 14px; }
    .timeline-row:not(:last-child):after { content: ''; position: absolute; left: 2px; top: 8px; bottom: -4px; width: 1px; background: #e5e5e7; z-index: 1; }
    .t-dot { width: 5px; height: 5px; border-radius: 50%; margin-top: 3px; margin-right: 4px; flex-shrink: 0; z-index: 2; }
    .t-dot.succeeded { background: #10b981; }
    .t-dot.failed { background: #be185d; }
    .t-dot.running { background: #7c3aed; animation: pulse 2s infinite; }
    .t-dot.ready { background: #f59e0b; }
    .t-dot.pending { background: #e5e5e7; }
    .t-dot.source { background: #6366f1; }
    .t-dot.canceled { background: #9ca3af; }
    .t-content { flex-grow: 1; font-size: 8px; }
    .t-header { display: flex; justify-content: space-between; align-items: center; }
    .t-name { font-weight: 700; color: #1d1d1f; white-space: nowrap; font-size: 9px; }
    .t-time { font-family: monospace; color: #86868b; font-size: 8px; white-space: nowrap; }
    .t-details { font-size: 7px; color: #86868b; margin-top: 0px; display: flex; flex-wrap: wrap; gap: 2px; }
    .t-pill { background: #f0f0f2; color: #86868b; border: 1px solid #d1d1d6; padding: 0px 2px; border-radius: 2px; font-family: monospace; font-size: 8px; text-decoration: none; }
    .t-pill:hover { background: #e5e5e7; }

    /* PLATE HOVER */
    .plate-hover-container { position: relative; display: inline-block; }
    .plate-trigger { cursor: pointer; background: #f0f0f2; color: #86868b; padding: 0px 2px; border-radius: 2px; font-family: monospace; font-size: 7px; border: 1px solid #d1d1d6; text-decoration: none; }
    .plate-trigger:hover { background: #e5e5e7; }
    .plate-popover { display: none; position: absolute; top: 100%; left: 0; background: white; border: 1px solid #d1d1d6; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 4px 6px; border-radius: 3px; z-index: 9999; min-width: 150px; margin-top: 2px; }
    .plate-hover-container:hover .plate-popover { display: block; }
    .popover-title { font-weight: 700; font-size: 7px; color: #86868b; text-transform: uppercase; margin-bottom: 2px; font-family: monospace; }
    .popover-link { font-size: 7px; color: #86868b; text-decoration: none; padding: 0px 2px; font-family: monospace; }
    .popover-link:hover { text-decoration: underline; }

    /* PART TAGS */
    .part-tag { background: #f0f0f2; color: #86868b; padding: 0px 2px; border-radius: 2px; font-family: monospace; font-size: 8px; margin-right: 2px; border: 1px solid #e5e5e7; }
    .part-tag.in-production { background: #fffbeb; color: #d97706; border: 1px dashed #fcd34d; }
    .part-tag.missing { background: #fdf2f8; color: #be185d; border: 1px solid #f9a8d4; font-weight: 700; cursor: default; }
    .missing-tip-wrap { position: relative; display: inline-block; }
    .missing-tip { display: none; position: absolute; bottom: calc(100% + 4px); left: 0; background: #1e293b;
        border: 1px solid #334155; border-radius: 4px; padding: 5px 8px; z-index: 9999; min-width: 220px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4); pointer-events: none; }
    .missing-tip-wrap:hover .missing-tip { display: block; }
    .missing-tip-req { font-family: monospace; font-size: 8px; color: #94a3b8; margin-bottom: 2px; }
    .missing-tip-exp { font-size: 9px; font-weight: 700; color: #e2e8f0; }
    .missing-tip-status { font-size: 8px; color: #be185d; font-weight: 600; margin-top: 2px; }
    .colony-badge { font-size: 7px; padding: 1px 3px; border-radius: 2px; font-weight: 600; }
    .tat-cell { font-family: monospace; color: #86868b; font-size: 8px; white-space: nowrap; }

    /* GROUP HEADERS */
    .group-header { padding: 5px 8px; font-size: 10px; font-weight: 700; background: #f0f0f2; color: #1d1d1f; border: 1px solid #d1d1d6; cursor: pointer; display: flex; align-items: center; border-radius: 3px; margin: 6px 0 3px 0; }
    .group-header:hover { background: #e5e5e7; }
    .group-header.in-progress { background: linear-gradient(135deg, #fdf4ff, #fef3c7); border-color: #d946ef; color: #a21caf; }
    .group-header.new { background: linear-gradient(135deg, #f5f3ff, #ede9fe); border-color: #7c3aed; color: #6d28d9; }
    .group-header.fulfilled { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-color: #10b981; color: #059669; }
    .group-header.canceled { background: linear-gradient(135deg, #f5f5f7, #e5e5e7); border-color: #9ca3af; color: #6b7280; }
    .group-arrow { margin-right: 5px; font-size: 8px; }
    details[open] .group-arrow { transform: rotate(90deg); }
    details > summary { list-style: none; }

    /* ANIMATION */
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    /* WARNING & SEARCH */
    .warning-note { background: #fdf2f8; border: 1px solid #f9a8d4; color: #be185d; padding: 2px 5px; margin: 3px 6px; border-radius: 2px; font-weight: 600; font-size: 8px; }
    #search_box { width: 280px; padding: 4px 8px; border: 1px solid #d1d1d6; border-radius: 4px; font-size: 10px; background: white; }
    #search_box:focus { outline: none; border-color: #7c3aed; box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.1); }
    .search-match td { background: #fef9c3 !important; }
    .search-match-section { outline: 2px solid #f59e0b !important; outline-offset: 1px; border-radius: 3px; }
    </style>

    <script>
    function switchTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.querySelector('[data-tab="' + tabName + '"]').classList.add('active');
        document.getElementById('tab-' + tabName).classList.add('active');
    }
    function toggleBucketView(expId) {
        var tl  = document.getElementById('timeline_' + expId);
        var bk  = document.getElementById('bucket_'   + expId);
        var btn = document.getElementById('bucket_btn_' + expId);
        if (!tl || !bk || !btn) return;
        if (bk.style.display === 'none') {
            bk.style.display = 'block'; tl.style.display = 'none'; btn.textContent = 'Timeline';
        } else {
            bk.style.display = 'none'; tl.style.display = 'block'; btn.textContent = 'Stage View';
        }
    }
    function toggleSection(id) {
        var el = document.getElementById(id); var icon = document.getElementById(id + '_icon'); var btn = document.getElementById(id + '_btn');
        if (el.style.display === "block") { el.style.display = "none"; if(icon) icon.classList.remove('open'); if(btn) btn.classList.remove('active-header'); }
        else { el.style.display = "block"; if(icon) icon.classList.add('open'); if(btn) btn.classList.add('active-header'); }
    }
    var _sortedByDue = false;
    var _originalOrder = null;
    function sortByDueDate() {
        var container = document.querySelector('#tab-tracking > div[style*="padding"]');
        if (!container) return;
        var cards = Array.from(container.querySelectorAll(':scope > .project-wrapper'));
        if (!_sortedByDue) {
            // Save original order
            _originalOrder = cards.slice();
            // Sort ascending by data-due-date (no date = end)
            cards.sort(function(a, b) {
                var da = a.getAttribute('data-due-date') || '9999-99-99';
                var db = b.getAttribute('data-due-date') || '9999-99-99';
                return da < db ? -1 : da > db ? 1 : 0;
            });
            cards.forEach(function(c) { container.appendChild(c); });
            _sortedByDue = true;
            document.getElementById('sort_due_btn').textContent = 'Sort: Default';
            document.getElementById('sort_due_btn').style.background = '#e8f4fd';
            document.getElementById('sort_due_btn').style.borderColor = '#0891b2';
            document.getElementById('sort_due_btn').style.color = '#0891b2';
        } else {
            // Restore original order
            if (_originalOrder) { _originalOrder.forEach(function(c) { container.appendChild(c); }); }
            _sortedByDue = false;
            document.getElementById('sort_due_btn').textContent = 'Sort: Due Date';
            document.getElementById('sort_due_btn').style.background = '#fff';
            document.getElementById('sort_due_btn').style.borderColor = '#d1d1d6';
            document.getElementById('sort_due_btn').style.color = '#1d1d1f';
        }
    }
    var _filterTimer = null;
    function filterDashboardDebounced() {
        clearTimeout(_filterTimer);
        _filterTimer = setTimeout(filterDashboard, 400);
    }
    function filterDashboard() {
        var searchTerm = document.getElementById('search_box').value.toLowerCase().trim();
        var activeOnly = document.getElementById('active_toggle').checked;
        // Don't search on a single character — too many matches causes DOM freeze
        if (searchTerm.length === 1) { return; }
        try { localStorage.setItem('dash_activeOnly', activeOnly ? '1' : '0'); } catch(e) {}
        try { localStorage.setItem('dash_search', document.getElementById('search_box').value); } catch(e) {}
        document.querySelectorAll('.search-match').forEach(function(el) { el.classList.remove('search-match'); });
        document.querySelectorAll('.search-match-section').forEach(function(el) { el.classList.remove('search-match-section'); });
        document.querySelectorAll('.req-card').forEach(function(el) { el.style.display = ''; });
        var projects = document.getElementsByClassName('project-wrapper');
        var firstTarget = null;
        for (var i = 0; i < projects.length; i++) {
            var project = projects[i];
            var isActive = project.getAttribute('data-active') === 'true';
            if (activeOnly && !isActive) { project.style.display = 'none'; continue; }
            if (searchTerm) {
                if (!project.textContent.toLowerCase().includes(searchTerm)) { project.style.display = 'none'; continue; }
                project.style.display = 'block';
                project.querySelectorAll('.req-card').forEach(function(reqCard) {
                    if (!reqCard.textContent.toLowerCase().includes(searchTerm)) {
                        reqCard.style.display = 'none'; return;
                    }
                    reqCard.style.display = 'block';
                    // Open the parent <details> so the card is actually visible.
                    var parentDetails = reqCard.closest('details');
                    if (parentDetails) parentDetails.open = true;
                    // Highlight the title bar — don't touch the content pane (that would open the dropdown).
                    var titleBar = reqCard.querySelector('.req-title-bar');
                    if (titleBar) {
                        titleBar.classList.add('search-match-section');
                        if (!firstTarget) firstTarget = reqCard;
                    }
                    // Pre-mark matching rows inside so they're highlighted when opened manually.
                    reqCard.querySelectorAll('tr').forEach(function(row) {
                        if (row.textContent.toLowerCase().includes(searchTerm)) {
                            row.classList.add('search-match');
                        }
                    });
                });
            } else {
                project.style.display = 'block';
                project.querySelectorAll('.req-card').forEach(function(reqCard) {
                    reqCard.style.display = 'block';
                });
            }
        }
        if (firstTarget && searchTerm) {
            setTimeout(function() { firstTarget.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 50);
        }
    }
    document.addEventListener('DOMContentLoaded', function() {
        try {
            var savedActive = localStorage.getItem('dash_activeOnly');
            var savedSearch = localStorage.getItem('dash_search');
            if (savedActive === '1') { document.getElementById('active_toggle').checked = true; }
            if (savedSearch) { document.getElementById('search_box').value = savedSearch; }
            if (savedActive === '1' || savedSearch) { filterDashboard(); }
        } catch(e) {}
        var firstExp = document.querySelector('.exp-content');
        var firstExpIcon = document.querySelector('.exp-toggle-icon');
        if (firstExp) { firstExp.style.display = 'block'; }
        if (firstExpIcon) { firstExpIcon.classList.add('open'); }
        document.addEventListener('click', function(e) {
            var trigger = e.target.closest('.colony-badge, .plate-trigger');
            if (trigger) {
                e.stopPropagation();
                var container = trigger.closest('.plate-hover-container');
                var popover = container.querySelector('.plate-popover');
                if (popover.classList.contains('sticky')) { popover.classList.remove('sticky'); }
                else { document.querySelectorAll('.plate-popover.sticky').forEach(function(p) { p.classList.remove('sticky'); }); popover.classList.add('sticky'); }
                return;
            }
            if (e.target.closest('.plate-popover')) { e.stopPropagation(); return; }
            document.querySelectorAll('.plate-popover.sticky').forEach(function(p) { p.classList.remove('sticky'); });
        });
    });
    </script>
    """

    # LSP Capacity tab (generated once, injected into template)
    lsp_capacity_html = render_lsp_capacity_tab(df)

    # Part 2: HTML with variables (f-string)
    html += f"""
    <div class="dashboard-container">
        <!-- HEADER WITH LOGO -->
        <div class="dashboard-header">
            <img src="data:image/jpeg;base64,{logo_b64}" class="dashboard-logo" alt="DNASC">
            <span class="dashboard-title">DNA Strain & Construction</span>
            <span class="dashboard-updated">Data pulled: {generated_at}</span>
        </div>
        <!-- TAB NAVIGATION -->
        <div class="tab-container">
            <div class="tab-nav">
                <button class="tab-btn active" data-tab="tracking" onclick="switchTab('tracking')">
                    <img src="data:image/png;base64,{tracking_icon_b64}" class="tab-icon-img" alt="Tracking">
                    <span class="tab-text">Tracking</span>
                </button>
                <button class="tab-btn" data-tab="metrics" onclick="switchTab('metrics')">
                    <img src="data:image/png;base64,{metrics_icon_b64}" class="tab-icon-img" alt="Metrics">
                    <span class="tab-text">Metrics</span>
                </button>
                <button class="tab-btn" data-tab="costs" onclick="switchTab('costs')">
                    <img src="data:image/png;base64,{cost_icon_b64}" class="tab-icon-img" alt="Costs">
                    <span class="tab-text">Costs</span>
                </button>
                <button class="tab-btn" data-tab="capacity" onclick="switchTab('capacity')">
                    <span style="font-size:16px;">📈</span>
                    <span class="tab-text">LSP Capacity</span>
                </button>
            </div>
            <!-- TRACKING TAB -->
            <div id="tab-tracking" class="tab-content active">
                <div class="controls-container">
                    <div class="toggle-wrapper" style="margin-right: auto;">
                        <input type="text" id="search_box" placeholder="Search Stock ID, Experiment, or Construct..." oninput="filterDashboardDebounced()">
                    </div>
                    <div class="toggle-wrapper">
                        <span class="toggle-label">Active Projects Only</span>
                        <label class="switch">
                            <input type="checkbox" id="active_toggle" onclick="filterDashboard()">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div class="toggle-wrapper">
                        <button id="sort_due_btn" onclick="sortByDueDate()" style="font-size:10px;font-weight:700;padding:4px 10px;border-radius:4px;border:1px solid #d1d1d6;background:#fff;color:#1d1d1f;cursor:pointer;white-space:nowrap;">Sort: Due Date</button>
                    </div>
                </div>
                <div style="padding: 10px;">
    """

    # =========================================================================
    # 3. HELPER: RENDER SINGLE REQUEST
    # =========================================================================
    def render_single_request_html(req_id, req_df, is_stalled=False, is_asm_review=False):
        html = ""
        construct = req_df['construct_name'].iloc[0] or "Unknown Construct"
        req_status = req_df['request_status'].iloc[0] if 'request_status' in req_df.columns else "Unknown"
        req_priority = req_df['priority'].iloc[0] if 'priority' in req_df.columns else ""
        is_partner_req = False
        if 'for_partner' in req_df.columns:
            vals = req_df['for_partner'].astype(str).str.lower()
            if vals.str.contains('true').any(): is_partner_req = True
        if is_partner_req:
            req_bg_color = "#f9fafb"; req_border_left = "#7c3aed"
        else:
            req_bg_color = "#f9fafb"; req_border_left = "#6b7280"

        # Collect all root stock IDs across all chains — used for header tags and parts filtering
        _root_stock_map = {}
        for _rid, _rdf in req_df.groupby('root_work_order_id'):
            _rw = _rdf[_rdf['workorder_id'] == _rid]['STOCK_ID']
            _rstock = str(_rw.iloc[0]) if not _rw.empty and pd.notna(_rw.iloc[0]) else str(_rdf['STOCK_ID'].iloc[0])
            if _rstock not in ('nan', 'None', 'N/A'):
                _root_stock_map[_rid] = _rstock
        all_root_stocks = set(_root_stock_map.values())
        target_pais = all_root_stocks
        pai_pattern = re.compile(r'pAI-\d+', re.IGNORECASE)
        final_list = []
        for s in target_pais:
            match = pai_pattern.search(s)
            if match: final_list.append(match.group())
        final_list = sorted(list(set(final_list)), key=lambda x: int(re.search(r'\d+', x).group()))
        # Split root stocks into primary (lowest pAI) and secondary sets for badge styling
        _primary_pai_str = final_list[0] if final_list else None
        _primary_root_stocks = {s for s in all_root_stocks if _primary_pai_str and pai_pattern.search(s) and pai_pattern.search(s).group() == _primary_pai_str}
        _secondary_root_stocks = all_root_stocks - _primary_root_stocks

        # Lowest pAI number = primary (solid purple); others = secondary (deeper purple)
        def _pai_tag(pid, is_primary):
            if is_primary:
                return f"<span class='stock-tag' style='font-size:9px; padding:3px 8px; vertical-align:middle; background:#ede9fe; color:#6d28d9; border: 1px solid #c4b5fd;'>{pid}</span>"
            return f"<span class='stock-tag' style='font-size:9px; padding:3px 8px; vertical-align:middle; background:#ddd6fe; color:#4c1d95; border: 1px solid #7c3aed;'>{pid}</span>"
        pais_display = " ".join([_pai_tag(pid, i == 0) for i, pid in enumerate(final_list)])
        partner_badge = '<span class="badge" style="background:#ede9fe;color:#7c3aed;margin-right:8px;border:1px solid #c4b5fd;">PARTNER</span>' if is_partner_req else ""

        _customer_styles = {
            'R_D':             ('R&D',          '#cffafe', '#0e7490', '#a5f3fc'),
            'INTERNAL_CLD':    ('CLD',          '#dbeafe', '#1d4ed8', '#93c5fd'),
            'TECH_OUT':        ('TECH OUT',     '#ffedd5', '#c2410c', '#fdba74'),
            'EXTERNAL_TECH_OUT':('EXT TECH OUT','#fce7f3', '#be185d', '#f9a8d4'),
        }
        _cust_raw = str(req_df['customer'].iloc[0]) if 'customer' in req_df.columns and pd.notna(req_df['customer'].iloc[0]) else None
        if _cust_raw and _cust_raw not in ('nan', 'None', ''):
            _clabel, _cbg, _cfg, _cborder = _customer_styles.get(_cust_raw, (_cust_raw.replace('_', ' '), '#f3f4f6', '#374151', '#d1d5db'))
            customer_badge = f'<span class="badge" style="background:{_cbg};color:{_cfg};border:1px solid {_cborder};margin-right:4px;">{_clabel}</span>'
        else:
            customer_badge = ""

        # --- UPDATED CONTEXT-AWARE STATUS SELECTION ---
        # Compute render-time effective status for each row so colony-type overrides
        # (e.g. streakout with colonies but no seq → RUNNING) are reflected in phase
        # detection, not just in the per-row display.
        req_df = req_df.copy()
        req_df['_eff_status'] = req_df.apply(lambda r: get_visual_status(r)[0], axis=1)
        active_rows = req_df[req_df['_eff_status'].isin(['RUNNING', 'READY', 'WAITING', 'BLOCKED', 'IN_PROGRESS'])]
        target_row = None
        status_badge_html = ""

        if not active_rows.empty:
            # Phase detection is stock-ID-based: root stocks = ASM, non-root stocks = PARTS
            lsp_active  = active_rows[active_rows['type'] == 'lsp_workorder']
            asm_active  = active_rows[
                (active_rows['type'] != 'lsp_workorder') &
                active_rows['STOCK_ID'].astype(str).isin(all_root_stocks)
            ]
            parts_active = active_rows[
                (active_rows['type'] != 'lsp_workorder') &
                ~active_rows['STOCK_ID'].astype(str).isin(all_root_stocks)
            ]
            _parts_priority = {'BLOCKED': 0, 'RUNNING': 1, 'READY': 2, 'IN_PROGRESS': 3, 'WAITING': 4}

            phase_label = ""
            phase_bg = "#f5f5f7"
            phase_color = "#6b7280"
            phase_border = "#d1d5db"

            if not lsp_active.empty:
                target_row = lsp_active.iloc[0]
                phase_label = "LSP"
                phase_bg = "#cffafe"
                phase_color = "#0e7490"
                phase_border = "#a5f3fc"
            elif not asm_active.empty:
                _progressing = {'RUNNING', 'READY', 'IN_PROGRESS', 'BLOCKED'}
                asm_progressing = asm_active[asm_active['_eff_status'].isin(_progressing)]
                if not asm_progressing.empty:
                    asm_priority = {'RUNNING': 0, 'READY': 1, 'IN_PROGRESS': 2, 'WAITING': 3, 'BLOCKED': 4}
                    asm_active = asm_active.copy()
                    asm_active['_rank'] = asm_active['_eff_status'].map(asm_priority).fillna(99)
                    target_row = asm_active.sort_values('_rank').iloc[0]
                    phase_label = "ASM"
                    phase_bg = "#dbeafe"
                    phase_color = "#1d4ed8"
                    phase_border = "#bfdbfe"
                else:
                    # All ASM WAITING — show most urgent part, fall back to waiting ASM
                    if not parts_active.empty:
                        parts_active = parts_active.copy()
                        parts_active['_rank'] = parts_active['_eff_status'].map(_parts_priority).fillna(99)
                        target_row = parts_active.sort_values('_rank').iloc[0]
                    else:
                        # parts_active empty — check cross-request fanned-in parts via
                        # global lookup (parts belonging to other req_ids but referenced
                        # in this GG's backbone/parts_json).
                        _xreq_candidates = []
                        for _aw in asm_active.itertuples():
                            for _fld in ('backbone', 'parts'):
                                _raw = getattr(_aw, _fld, None) or ''
                                if pd.isna(_raw): continue
                                for _tok in str(_raw).split(','):
                                    _sname = _tok.split(':')[0].strip()
                                    for _gpr in _global_parts_rows_by_stock.get(_sname, []):
                                        _xreq_candidates.append(_gpr)
                        if _xreq_candidates:
                            _xdf = pd.DataFrame(_xreq_candidates)
                            _xdf['_eff_status'] = _xdf.apply(lambda r: get_visual_status(r)[0], axis=1)
                            _xdf_active = _xdf[_xdf['_eff_status'].isin(_parts_priority)]
                            if not _xdf_active.empty:
                                _xdf_active = _xdf_active.copy()
                                _xdf_active['_rank'] = _xdf_active['_eff_status'].map(_parts_priority).fillna(99)
                                target_row = _xdf_active.sort_values('_rank').iloc[0].to_dict()
                        if target_row is None:
                            target_row = asm_active.iloc[0]
                    phase_label = "PARTS"
                    phase_bg = "#ffedd5"
                    phase_color = "#c2410c"
                    phase_border = "#fed7aa"
            elif not parts_active.empty:
                parts_active = parts_active.copy()
                parts_active['_rank'] = parts_active['_eff_status'].map(_parts_priority).fillna(99)
                target_row = parts_active.sort_values('_rank').iloc[0]
                phase_label = "PARTS"
                phase_bg = "#ffedd5"
                phase_color = "#c2410c"
                phase_border = "#fed7aa"
            else:
                target_row = active_rows.iloc[0]

            if target_row is not None:
                status = target_row['_eff_status']
                active_info = get_active_step_info(target_row)

                if active_info and 'Ready' in active_info:
                    status = 'READY'

                phase_html = f'''
                    <span style="
                        background: {phase_bg}; color: {phase_color};
                        border: 1px solid {phase_border};
                        padding: 0 5px; border-radius: 3px;
                        margin-right: 8px; font-weight: 800; font-size: 8.5px;
                        display: inline-flex; align-items: center; height: 14px;
                    ">{phase_label}</span>
                '''

                # Build display text: use active queue step if available, else status
                if active_info:
                    display_text = active_info.upper()
                else:
                    # Fallback: find the most recent running/ready protocol name
                    p_names = target_row.get('protocol_name', [])
                    p_states = target_row.get('operation_state', [])
                    running_step = None
                    if isinstance(p_names, list) and isinstance(p_states, list):
                        for pname, pstate in zip(reversed(p_names), reversed(p_states)):
                            if pstate in ('RU', 'RD', 'SC'):
                                running_step = pname
                                break
                    if running_step:
                        display_text = f"{running_step}: {status}".upper()
                    else:
                        _type = str(target_row.get('type', '')).lower()
                        if 'streak' in _type:
                            display_text = f"STREAKOUT: {status}"
                        else:
                            display_text = status.upper()

                badge_class = f"status-{status}"
                if target_row['type'] == 'lsp_workorder' and status == 'RUNNING':
                    badge_class = "status-LSP_RUNNING"

                status_badge_html = f'''
                    <span class="badge {badge_class}" style="
                        font-size: 9px; padding: 0 7px; height: 17px;
                        display: inline-flex; align-items: center; line-height: 1;
                        box-sizing: border-box; vertical-align: middle;
                        white-space: nowrap; font-weight: 700; border-radius: 4px;
                    ">
                        {phase_html} {display_text}
                    </span>
                '''
        req_created = to_est(req_df['request_created_at'].iloc[0])
        submitted_str = req_created.strftime('%Y-%m-%d') if req_created else "N/A"
        _req_email_raw = req_df['submitter_email'].dropna().iloc[0] if 'submitter_email' in req_df.columns and not req_df['submitter_email'].dropna().empty else ''
        _req_email = str(_req_email_raw).strip() if _req_email_raw else ''
        now = datetime.now(pytz.timezone('US/Eastern'))
        is_done = str(req_status).upper() in ['FULFILLED', 'SUCCEEDED', 'CANCELED']

        stalled_badge = '<span class="badge" style="background:#be185d; color:white; border:2px solid #9f1239; font-size:12px; padding:4px 12px; font-weight:800;">⚠️ STALLED</span>' if is_stalled else ""
        asm_review_badge = '<span class="badge" style="background:#d97706; color:white; border:2px solid #b45309; font-size:12px; padding:4px 12px; font-weight:800;">🔬 ASM REVIEW</span>' if is_asm_review else ""

        ready_to_ship_time = None
        final_release_time = None
        lsp_rows = req_df[req_df['type'] == 'lsp_workorder']
        if not lsp_rows.empty:
            for _, lrow in lsp_rows.iterrows():
                p_names = lrow.get('protocol_name', [])
                p_states = lrow.get('operation_state', [])
                p_starts = lrow.get('operation_start', [])
                if isinstance(p_names, np.ndarray): p_names = p_names.tolist()
                if isinstance(p_states, np.ndarray): p_states = p_states.tolist()
                if isinstance(p_starts, np.ndarray): p_starts = p_starts.tolist()
                if isinstance(p_names, list):
                    for name, state, start in zip(p_names, p_states, p_starts):
                        if name == 'LSP Reviewing' and state == 'SC': ready_to_ship_time = to_est(start)
                        if name == 'LSP Releasing' and state == 'SC': final_release_time = to_est(start)

        time_badges_html = ""
        shared_time_style = "display: flex; align-items: center; background: #f3f4f6; border: 1px solid #d1d5db; padding: 1px 4px; border-radius: 2px; gap: 3px; height: 16px;"
        if req_created:
            if is_done:
                production_end = ready_to_ship_time if ready_to_ship_time else now
                production_days = (production_end - req_created).days
                pw, pday = production_days // 7, production_days % 7
                production_str = f"{pw}w {pday}d" if pw > 0 else f"{pday}d"
                total_end = final_release_time if final_release_time else production_end
                total_days = (total_end - req_created).days
                tw, tday = total_days // 7, total_days % 7
                total_str = f"{tw}w {tday}d" if tw > 0 else f"{tday}d"
                time_badges_html = f"""
                  <div style="display: flex; gap: 8px; align-items: center;">
                      <div style="{shared_time_style}">
                          <span style="font-size: 9px; color: #6b7280; font-weight: 700; text-transform: uppercase;">Production:</span>
                          <span style="font-size: 10px; font-weight: 700; color: #4b5563; font-family: monospace;">{production_str}</span>
                      </div>
                      <div style="{shared_time_style}">
                          <span style="font-size: 9px; color: #6b7280; font-weight: 700; text-transform: uppercase;">Total:</span>
                          <span style="font-size: 10px; font-weight: 700; color: #4b5563; font-family: monospace;">{total_str}</span>
                      </div>
                  </div>"""
            else:
                running_days = (now - req_created).days
                rw, rday = running_days // 7, running_days % 7
                running_str = f"{rw}w {rday}d" if rw > 0 else f"{rday}d"
                time_badges_html = f"""
                <div style="{shared_time_style}">
                    <span style="font-size: 8px; color: #6b7280; font-weight: 700; text-transform: uppercase;">Running:</span>
                    <span style="font-size: 9px; font-weight: 700; color: #4b5563; font-family: monospace;">{running_str}</span>
                </div>"""
        html += f"""
        <div class="req-card">
            <div class="req-title-bar" style="background: {req_bg_color}; border-left-color: {req_border_left};">
                <div style="flex-grow: 1; min-width: 0;">
                    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: nowrap;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            {pais_display}
                        </div>
                        <span class="req-name">{construct}</span>
                        <div style="display: flex; align-items: center; gap: 8px;">

                            <div style="display: flex; align-items: center; background: #f3f4f6; border: 1px solid #d1d5db; padding: 1px 4px; border-radius: 2px; gap: 3px; height: 16px;">
                                <span style="font-size: 8px; color: #6b7280; font-weight: 700;">CREATED:</span>
                                <span style="font-size: 9px; font-weight: 700; color: #4b5563; font-family: monospace;">{submitted_str}</span>
                            </div>
                            <div style="font-size: 1px;">{time_badges_html}</div>
                        </div>
                    </div>
                    <div style="margin-top: 4px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <span style="color: #94a3b8; font-size: 10px; font-family: monospace; letter-spacing: -0.2px;">REQ ID: {req_id}</span>
                        {(f'<span style="color: #1e40af; font-size: 10px; font-family: monospace; font-weight: 600;">{_req_email}</span>') if _req_email else ''}
                    </div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    {customer_badge}{partner_badge}
                    <span class="badge status-{str(req_status).replace(" ", "_")}" style="font-size: 10px; padding: 2px 8px;">{req_status}</span>
                    {status_badge_html}
                    {stalled_badge}
                    {asm_review_badge}
                </div>
            </div>
        """

        html += f"""
        <div style="padding: 3px 6px; background: #fafafa; border-bottom: 1px solid #e5e5e7; cursor: pointer;" onclick="toggleSection('req_{req_id.replace("-", "_")}')">
            <span id="req_{req_id.replace("-", "_")}_icon" class="dropdown-icon">▶</span>
            <span style="font-size: 9px; font-weight: 600; color: #86868b;">Workorder Details</span>
        </div>
        <div id="req_{req_id.replace("-", "_")}" style="display: none;">
        """

        is_req_fulfilled = str(req_status).upper() in ['FULFILLED', 'SUCCEEDED']
        root_status_map = {}
        has_winner = False
        _req_wid_set = set(req_df['workorder_id'].astype(str))
        for root_id, r_df in req_df.groupby('root_work_order_id'):
            # Skip roots whose root workorder belongs to a different request.
            # These are cross-request synthesis parts (e.g. synparts ordered for
            # another GG) — they'll appear fanned in under that GG's section.
            # Exempt LSP workorders: their root may be a source Gibson from a
            # different request (or a LIMS batch ID), but they belong here.
            has_lsp = r_df['type'].eq('lsp_workorder').any()
            if not has_lsp and str(root_id) not in _req_wid_set:
                continue
            is_winner = False
            if is_req_fulfilled:
                if r_df['fulfills_request'].any() or r_df['wo_status'].isin(['SUCCEEDED', 'FULFILLED']).any():
                    is_winner = True; has_winner = True
            status_priority = {'RUNNING': 0, 'IN_PROGRESS': 0, 'BLOCKED': 1, 'WAITING': 2, 'READY': 2, 'DRAFT': 2, 'SUCCEEDED': 3, 'FAILED': 4, 'CANCELED': 5}
            r_df['local_rank'] = r_df['visual_status'].map(status_priority).fillna(99)
            min_rank = r_df['local_rank'].min()
            root_status_map[root_id] = {'is_winner': is_winner, 'rank': min_rank}
        sorted_roots = sorted(root_status_map.keys(), key=lambda r: (not root_status_map[r]['is_winner'], root_status_map[r]['rank']))

        # Pre-compute request-level attempt numbering for assembly roots.
        # With self-rooting each Gibson/GG is its own root section, so per-section
        # attempt counting always gives len=1 and never fires. We need to count
        # across all root sections in the request.
        _asm_types_req = {'golden_gate_workorder', 'gibson_workorder'}
        _asm_types_dfs = _asm_types_req
        _parts_types_dfs = {'oligo_synthesis_workorder', 'pcr_workorder',
                            'plasmid_synthesis_workorder', 'syn_part_synthesis_workorder'}

        # Pre-compute parts fan-out: parts with an assembly_plan_id shared by GG/Gibson
        # sections should display under EVERY assembly section in the request, not just one.
        _plan_to_asm_root_ids: dict = {}   # plan_id → set of root_work_order_ids that contain GG rows
        _plan_to_part_rows: dict   = {}    # plan_id → list of parts row dicts (from any root)
        _suppress_part_root_ids: set = set()  # standalone parts-only root_ids to skip
        if 'assembly_plan_id' in req_df.columns:
            for _, _fan_r in req_df.iterrows():
                _fan_pid  = _fan_r.get('assembly_plan_id')
                _fan_type = _fan_r.get('type', '')
                _fan_rwid = str(_fan_r.get('root_work_order_id', ''))
                if pd.isna(_fan_pid) or not _fan_pid:
                    continue
                if _fan_type in _asm_types_dfs:
                    _plan_to_asm_root_ids.setdefault(_fan_pid, set()).add(_fan_rwid)
                elif _fan_type in _parts_types_dfs:
                    _plan_to_part_rows.setdefault(_fan_pid, []).append(_fan_r.to_dict())
            # Plans that have both GG sections and parts: parts outside an asm root get suppressed
            for _fan_pid in set(_plan_to_asm_root_ids) & set(_plan_to_part_rows):
                for _fan_prow in _plan_to_part_rows[_fan_pid]:
                    _fan_prwid = str(_fan_prow.get('root_work_order_id', ''))
                    if _fan_prwid not in _plan_to_asm_root_ids.get(_fan_pid, set()):
                        _suppress_part_root_ids.add(_fan_prwid)

        # Pre-compute request-level attempt map for self-rooted GG/Gibson sections.
        # Pass 1: same-plan RETRY grouping — same-plan GGs collapse into one section via
        # plan_attempt_roots so each plan has at most one self-rooted anchor here.
        # Pass 2: cross-plan MANUAL grouping — catches manual retries where a new assembly
        # plan is created for the same construct. Groups by (STOCK_ID, backbone_json,
        # parts_json) so only identical-design GGs are grouped; different designs that
        # happen to share a STOCK_ID are NOT grouped. CANCELED already excluded below.
        _req_asm_attempt_map: dict = {}  # root_id → (attempt_number, attempt_total, attempt_kind)
        _req_asm_roots_info = []  # (root_id, plan, wo_created_at, stock_id, backbone, parts)
        for _r in sorted_roots:
            _rrows = req_df[req_df['workorder_id'] == _r]
            if _rrows.empty:
                continue
            _rrow = _rrows.iloc[0]
            if _rrow.get('type', '') not in _asm_types_req:
                continue
            if str(_rrow.get('wo_status', '') or '') in ('DRAFT', 'CANCELED'):
                continue
            # Only count sections where the root GG/Gibson is self-rooted
            if str(_rrow.get('root_work_order_id', '')) != str(_r):
                continue
            _rplan = str(_rrow.get('assembly_plan_id', '') or '')
            if not _rplan or _rplan in ('nan', 'None', ''):
                continue
            _rts = _rrow.get('wo_created_at') or pd.Timestamp.min
            _rstock = str(_rrow.get('STOCK_ID', '') or '')
            if _rstock in ('nan', 'None'): _rstock = ''
            _rbb = str(_rrow.get('backbone', '') or '')
            if _rbb in ('nan', 'None'): _rbb = ''
            _rparts = str(_rrow.get('parts', '') or '')
            if _rparts in ('nan', 'None'): _rparts = ''
            _req_asm_roots_info.append((_r, _rplan, _rts, _rstock, _rbb, _rparts))
        # Pass 1: same-plan RETRY — must be same design (backbone + parts), not just same plan
        _req_asm_by_plan: dict = {}
        for _r, _rplan, _rts, _rstock, _rbb, _rparts in _req_asm_roots_info:
            _req_asm_by_plan.setdefault((_rplan, _rbb, _rparts), []).append((_r, _rts))
        for (_plan, _bb, _pts), _ars in _req_asm_by_plan.items():
            if len(_ars) > 1:
                _ars.sort(key=lambda x: x[1])
                for _ai, (_r, _) in enumerate(_ars, 1):
                    _kind = '' if _ai == 1 else 'RETRY'
                    _req_asm_attempt_map[_r] = (_ai, len(_ars), _kind)
        # Pass 2: cross-plan MANUAL — group by (STOCK_ID, backbone, parts)
        _req_asm_by_design_xplan: dict = {}
        for _r, _rplan, _rts, _rstock, _rbb, _rparts in _req_asm_roots_info:
            if not _rstock or _r in _req_asm_attempt_map:
                continue
            _design_key = (_rstock, _rbb, _rparts)
            _req_asm_by_design_xplan.setdefault(_design_key, []).append((_r, _rplan, _rts))
        # Cross-plan sub-roots: fold into primary section (same as same-plan retries).
        # _sec_asm_by_stock within the merged section handles attempt numbering.
        _cross_plan_sub_roots: set = set()
        _cross_plan_sub_roots_for: dict = {}  # primary_root → [sub_roots]
        for _design_key, _ars in _req_asm_by_design_xplan.items():
            if len(_ars) > 1:
                _ars.sort(key=lambda x: x[2])
                _primary = _ars[0][0]
                for _r, _, _ in _ars[1:]:
                    _cross_plan_sub_roots.add(_r)
                    _cross_plan_sub_roots_for.setdefault(_primary, []).append(_r)
                # Remove from attempt map — within-section numbering takes over
                for _r, _, _ in _ars:
                    _req_asm_attempt_map.pop(_r, None)

        for root_id in sorted_roots:
            if root_id in _cross_plan_sub_roots:
                continue
            root_df = req_df[req_df['root_work_order_id'] == root_id]
            # Fold cross-plan same-design sub-roots into this primary section
            for _sub in _cross_plan_sub_roots_for.get(root_id, []):
                _sub_df = req_df[req_df['root_work_order_id'] == _sub]
                if not _sub_df.empty:
                    root_df = pd.concat([root_df, _sub_df]).drop_duplicates(subset='workorder_id')
            # Fan in cross-request parts (synthesis types) routed here by ADWOA.
            # These have root_work_order_id == root_id but belong to a different req_id.
            # Only fan in when root_id belongs to this request — foreign-root sections
            # (e.g. LSP sourced from another request's Gibson) must not pull in that
            # Gibson's unrelated parts from other requests.
            if str(root_id) in _req_wid_set:
                _xreq_parts = df[
                    (df['root_work_order_id'] == root_id) &
                    ~df['workorder_id'].isin(req_df['workorder_id']) &
                    df['type'].isin(_parts_types_dfs)
                ]
                if not _xreq_parts.empty:
                    root_df = pd.concat([root_df, _xreq_parts]).drop_duplicates(subset='workorder_id')
            # Skip parts-only roots whose rows will appear fanned under assembly sections
            if root_id in _suppress_part_root_ids:
                if not root_df['type'].isin(_asm_types_dfs | {'transformation_workorder', 'lsp_workorder'}).any():
                    continue
            is_this_winner = root_status_map[root_id]['is_winner']
            section_class = "assembly-section"
            if is_req_fulfilled and has_winner and not is_this_winner: section_class += " dimmed"
            row_map = {row['workorder_id']: row.to_dict() for _, row in root_df.iterrows()}
            adj = defaultdict(list)
            roots_in_view = []
            # Fan in parts rows from other roots that share the same assembly plan
            _fanned_wids: set = set()
            if 'assembly_plan_id' in req_df.columns:
                _sec_plan_ids = {
                    str(rrow.get('assembly_plan_id'))
                    for rrow in row_map.values()
                    if pd.notna(rrow.get('assembly_plan_id'))
                }
                for _fan_pid in _sec_plan_ids:
                    if root_id in _plan_to_asm_root_ids.get(_fan_pid, set()):
                        for _fan_prow in _plan_to_part_rows.get(_fan_pid, []):
                            _fan_pwid  = str(_fan_prow.get('workorder_id', ''))
                            _fan_prwid = str(_fan_prow.get('root_work_order_id', ''))
                            # Only fan in parts already rooted to an asm section in
                            # this request (retry case). Cross-request batch parts
                            # that belong to other requests' GG roots are excluded
                            # here and handled by the stock-ID fan-in below.
                            if _fan_prwid not in _plan_to_asm_root_ids.get(_fan_pid, set()):
                                continue
                            if _fan_pwid not in row_map:
                                row_map[_fan_pwid] = dict(_fan_prow)
                                _fanned_wids.add(_fan_pwid)

            # Cross-request STOCK_ID fan-in: for active assemblies, pull in parts
            # referenced by STOCK_ID that have no direct ADWOA root here. Validate
            # each candidate by checking that its root GG shares the same
            # assembly_plan_id as the current GG — confirming they are part of the
            # same design, not just a coincidental stock name match.
            for _gg_wid, _gg_row in list(row_map.items()):
                if _gg_row.get('type') not in _asm_types_dfs:
                    continue
                # Skip completed assemblies — parts already consumed.
                if str(_gg_row.get('wo_status', '') or '') in ('SUCCEEDED', 'FAILED', 'CANCELED'):
                    continue
                _gg_plan_id = str(_gg_row.get('assembly_plan_id', '') or '')
                if not _gg_plan_id or _gg_plan_id in ('nan', 'None', ''):
                    continue
                _needed_stocks: set = set()
                _bb = _gg_row.get('backbone', '')
                if _bb and pd.notna(_bb):
                    _bb_name = str(_bb).split(':')[0].strip()
                    if _bb_name:
                        _needed_stocks.add(_bb_name)
                _pts = _gg_row.get('parts', '')
                if _pts and pd.notna(_pts):
                    for _pt in str(_pts).split(','):
                        _pt_name = _pt.split(':')[0].strip()
                        if _pt_name:
                            _needed_stocks.add(_pt_name)
                for _ns in _needed_stocks:
                    _candidates = _global_parts_rows_by_stock.get(_ns, [])
                    if not _candidates:
                        continue
                    if any(str(_c.get('workorder_id', '')) in row_map for _c in _candidates):
                        continue
                    # Only include candidates whose root GG shares the same assembly_plan_id
                    _valid = []
                    for _c in _candidates:
                        _c_root = str(_c.get('root_work_order_id', '') or '')
                        _c_plan = _wid_to_assembly_plan_id.get(_c_root, '') or str(_c.get('assembly_plan_id', '') or '')
                        if _c_plan == _gg_plan_id:
                            _valid.append(_c)
                    if not _valid:
                        continue
                    # Among valid candidates pick the most active, then most recent
                    _status_rank = {'RUNNING': 0, 'READY': 1, 'WAITING': 2,
                                    'IN_PROGRESS': 3, 'SUCCEEDED': 4, 'FAILED': 5, 'CANCELED': 6}
                    def _fan_rank(r):
                        st = str(r.get('visual_status') or r.get('wo_status') or '')
                        ts = r.get('wo_created_at')
                        try: neg_ts = -ts.timestamp()
                        except Exception: neg_ts = 0
                        return (_status_rank.get(st, 99), neg_ts)
                    _best = min(_valid, key=_fan_rank)
                    _gpwid = str(_best.get('workorder_id', ''))
                    if _gpwid not in row_map:
                        row_map[_gpwid] = dict(_best)
                        _fanned_wids.add(_gpwid)

            for _, row in root_df.iterrows():
                wid = row['workorder_id']
                parent = None; label_suffix = ""
                if row['type'] == 'transformation_workorder': parent = row.get('source_asm_process_id')
                elif row['type'] == 'lsp_workorder':
                    for pc_col in ['source_lsp_process_id', 'source_workorder_id', 'lsp_process_id', 'middle_root']:
                        pc = row.get(pc_col)
                        if pd.notna(pc) and str(pc).lower() not in ('nan', 'none', '') \
                          and str(pc) != wid and not str(pc).upper().startswith('LSP-'):
                            parent = str(pc).strip()
                            break
                elif row['type'] == 'transformation_offline_operation': parent = row.get('source_asm_process_id')
                elif row['type'] == 'streakout_operation': parent = row.get('source_asm_process_id')
                if parent and isinstance(parent, str):
                    if parent in row_map: pass
                    else:
                        match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', parent, re.IGNORECASE)
                        if match:
                            uuid_val = match.group(1)
                            if uuid_val in row_map: parent = uuid_val
                            else: parent = uuid_val
                    parent_raw = str(row.get('source_lsp_process_id') or row.get('source_workorder_id'))
                    if 'stbl3' in parent_raw.lower(): label_suffix = " (STBL3)"
                    elif 'epi400' in parent_raw.lower(): label_suffix = " (EPI400)"
                    elif 'streakout' in parent_raw.lower(): label_suffix = " (Streakout)"
                    if 'well' in parent.lower() and parent not in row_map:
                        well_match = re.search(r'well(\d+)', parent, re.IGNORECASE)
                        if well_match and well_match.group(1) in well_to_root: parent = root_id; label_suffix = " (Streakout)"
                if parent and parent in row_map:
                    adj[parent].append(wid)
                    if label_suffix: row_map[wid]['visual_suffix'] = label_suffix
                else:
                    roots_in_view.append(wid)
                    if label_suffix: row_map[wid]['visual_suffix'] = label_suffix
                    if parent and parent not in row_map:
                        try:
                            row_map[wid]['visual_suffix'] = f" (from {str(parent)[:20]}...)"
                        except:
                            pass
            roots_in_view.sort(key=lambda x: 0 if x == root_id else 1)
            # Fanned-in parts have no parent in this section — add them as roots
            for _fan_pwid in _fanned_wids:
                if _fan_pwid not in adj:
                    roots_in_view.append(_fan_pwid)

            # Separate assembly roots from vendor parts roots so they render in distinct sections
            def _root_will_render(wid):
                """Returns True if this root row will not be skipped by the CANCELED+no-ops filter."""
                r = row_map.get(wid, {})
                if r.get('wo_status') != 'CANCELED': return True
                _pn = r.get('protocol_name')
                if hasattr(_pn, 'tolist'): _pn = _pn.tolist()
                return isinstance(_pn, list) and len(_pn) > 0

            # fulfills_request=False GG/Gibson are intermediate plasmids assembled as
            # inputs to the main assembly — treat them as parts, not as assembly sections.
            asm_roots_list = [r for r in roots_in_view if r in row_map
                              and row_map[r].get('type') in _asm_types_dfs
                              and row_map[r].get('fulfills_request', True) != False]
            parts_roots_list = [r for r in roots_in_view if r in row_map
                                and (
                                    (row_map[r].get('type') in _parts_types_dfs and row_map[r].get('wo_status') != 'DRAFT')
                                    or (row_map[r].get('type') in _asm_types_dfs and row_map[r].get('fulfills_request', True) == False)
                                )]
            other_roots_list = [r for r in roots_in_view if r not in asm_roots_list and r not in parts_roots_list and not (row_map.get(r, {}).get('type') in _parts_types_dfs and row_map.get(r, {}).get('wo_status') == 'DRAFT')]

            # Sort assembly roots by created_at and assign attempt numbers when there are retries.
            # Only count roots that will actually render (not CANCELED with no queue data).
            asm_roots_list.sort(key=lambda x: (row_map[x].get('wo_created_at') or pd.Timestamp.min))
            visible_asm_roots = [r for r in asm_roots_list if _root_will_render(r) and row_map[r].get('wo_status') != 'DRAFT']
            # Compute best downstream status for each assembly root's chain
            _chain_status_rank = {'SUCCEEDED': 0, 'READY': 1, 'RUNNING': 2, 'IN_PROGRESS': 3, 'WAITING': 4, 'BLOCKED': 5, 'FAILED': 6, 'CANCELED': 7}
            def _subtree_best_status(node_id):
                best = row_map[node_id].get('visual_status', '') if node_id in row_map else ''
                for child in adj.get(node_id, []):
                    child_best = _subtree_best_status(child)
                    if _chain_status_rank.get(child_best, 99) < _chain_status_rank.get(best, 99):
                        best = child_best
                return best

            # Section-level attempt numbering: only group roots with the same STOCK_ID + design.
            # Fanned-in Gibsons from other requests may appear in visible_asm_roots but
            # have different STOCK_IDs — they must NOT be numbered as attempts of each other.
            # Different backbone/parts = different design, not a retry — don't assign attempt numbers.
            if len(visible_asm_roots) > 1:
                _sec_asm_by_stock: dict = {}
                for _ar in visible_asm_roots:
                    _arstock = str(row_map[_ar].get('STOCK_ID', '') or '')
                    _arbb = str(row_map[_ar].get('backbone', '') or '').strip()
                    if _arbb in ('nan', 'None'): _arbb = ''
                    _arpts = str(row_map[_ar].get('parts', '') or '').strip()
                    if _arpts in ('nan', 'None'): _arpts = ''
                    _sec_asm_by_stock.setdefault((_arstock, _arbb, _arpts), []).append(_ar)
                for (_arstock, _, _), _ars in _sec_asm_by_stock.items():
                    if len(_ars) > 1:
                        for _ai, _ar in enumerate(_ars, 1):
                            row_map[_ar]['_attempt_number'] = _ai
                            row_map[_ar]['_attempt_total'] = len(_ars)
                            if _ai > 1:
                                _ar_resubmit = row_map[_ar].get('resubmit_count')
                                _ar_is_retry = (isinstance(_ar_resubmit, (int, float))
                                                and not pd.isna(_ar_resubmit)
                                                and int(_ar_resubmit) > 0)
                                row_map[_ar]['_attempt_kind'] = 'RETRY' if _ar_is_retry else 'MANUAL'
            # Always assign chain status for assembly roots (even single-attempt, used in banner)
            for _ar in visible_asm_roots:
                row_map[_ar]['_attempt_chain_status'] = _subtree_best_status(_ar)
            # Apply request-level attempt numbers to the section root (self-rooted GGs only).
            # Multiple GGs within one section are already handled by _sec_asm_by_stock above.
            if root_id in _req_asm_attempt_map and root_id in row_map:
                row_map[root_id]['_attempt_number'] = _req_asm_attempt_map[root_id][0]
                row_map[root_id]['_attempt_total'] = _req_asm_attempt_map[root_id][1]
                if _req_asm_attempt_map[root_id][2]:
                    row_map[root_id]['_attempt_kind'] = _req_asm_attempt_map[root_id][2]

            # Assign attempt numbers within parts: group by (type, STOCK_ID), sort by wo_created_at.
            # A retry = same part ordered again for a DIFFERENT root (GG retry attempt).
            # Multiple parts with same STOCK_ID under the SAME root = parallel scale-up, not retries.
            from collections import defaultdict as _dd
            _parts_by_key = _dd(list)
            for _pr in parts_roots_list:
                _prow = row_map[_pr]
                _key = (_prow.get('type', ''), str(_prow.get('STOCK_ID', '') or ''))
                _parts_by_key[_key].append(_pr)
            for _key, _prs in _parts_by_key.items():
                if len(_prs) > 1 and _key[0] == 'pcr_workorder':
                    # Use resubmit_count from BIOS as the authoritative retry signal.
                    # resubmit_count > 0 means BIOS created this workorder as a resubmission.
                    # Only label the group when at least one workorder was resubmitted.
                    _has_resubmit = any(
                        (row_map[x].get('resubmit_count') or 0) > 0
                        for x in _prs
                    )
                    if not _has_resubmit:
                        continue
                    _prs.sort(key=lambda x: (row_map[x].get('wo_created_at') or pd.Timestamp.min))
                    for _pi, _pr in enumerate(_prs, 1):
                        row_map[_pr]['_attempt_number'] = _pi
                        row_map[_pr]['_attempt_total'] = len(_prs)
            # Reorder: single-attempt (first-time-right) parts first, retry groups at the end.
            # Within retries keep same-key groups together sorted by created_at.
            _single_parts = [r for r in parts_roots_list if not isinstance(row_map[r].get('_attempt_number'), int)]
            _retry_parts  = [r for r in parts_roots_list if isinstance(row_map[r].get('_attempt_number'), int)]
            _retry_parts.sort(key=lambda x: (
                (row_map[x].get('type', ''), str(row_map[x].get('STOCK_ID', '') or '')),
                row_map[x].get('wo_created_at') or pd.Timestamp.min
            ))
            parts_roots_list = _single_parts + _retry_parts

            # Filter parts to only those whose STOCK_ID is explicitly referenced
            # in a GG/Gibson row's backbone or parts columns.  Batch assembly plans
            # order syn parts for many constructs under one plan root; without this
            # filter all batch parts pile up under the single winning GG.
            _needed_stocks: set = set()
            for _nrow in row_map.values():
                if _nrow.get('type') not in _asm_types_dfs:
                    continue
                _bb = _nrow.get('backbone', '')
                if _bb and pd.notna(_bb):
                    _bn = str(_bb).split(':')[0].strip()
                    if _bn: _needed_stocks.add(_bn)
                _pts = _nrow.get('parts', '')
                if _pts and pd.notna(_pts):
                    for _pt in str(_pts).split(','):
                        _pn = _pt.split(':')[0].strip()
                        if _pn: _needed_stocks.add(_pn)
            if _needed_stocks:
                # Only filter fanned-in parts (cross-request/plan) by stock match.
                # Native parts already rooted here (not in _fanned_wids) always show —
                # oligo primers etc. aren't in the GG's parts/backbone columns.
                parts_roots_list = [r for r in parts_roots_list
                                    if r not in _fanned_wids
                                    or str(row_map[r].get('STOCK_ID', '') or '') in _needed_stocks]

            ordered_data = []
            parts_ordered = []
            def dfs(node_id, depth):
                if node_id in row_map:
                    row_data = row_map[node_id]; row_data['tree_depth'] = depth; ordered_data.append(row_data)
                children = sorted(adj[node_id], key=lambda x: (0 if row_map[x]['type'] == 'transformation_workorder' else 1))
                for child in children: dfs(child, depth + 1)
            def dfs_p(node_id, depth):
                if node_id in row_map:
                    row_data = row_map[node_id]; row_data['tree_depth'] = depth; parts_ordered.append(row_data)
                for child in adj.get(node_id, []):
                    dfs_p(child, depth + 1)
            for r in asm_roots_list + other_roots_list: dfs(r, 0)
            for r in parts_roots_list: dfs_p(r, 0)
            sorted_root_df = pd.DataFrame(ordered_data + parts_ordered)

            # Skip root groups where every row is CANCELED with no queue data (empty dropdown)
            def _row_is_visible(r):
                if r.get('wo_status') != 'CANCELED': return True
                _pn = r.get('protocol_name')
                if hasattr(_pn, 'tolist'): _pn = _pn.tolist()
                return isinstance(_pn, list) and len(_pn) > 0
            if sorted_root_df.empty or not any(_row_is_visible(r) for _, r in sorted_root_df.iterrows()):
                continue

            badges_html = ""; header_extra_info = ""; target_row = None
            lsp_rows = root_df[root_df['type'] == 'lsp_workorder']
            if not lsp_rows.empty:
                active_lsps = lsp_rows[lsp_rows['visual_status'].isin(['RUNNING', 'READY', 'IN_PROGRESS', 'WAITING', 'BLOCKED'])]
                completed_lsps = lsp_rows[lsp_rows['visual_status'].isin(['SUCCEEDED', 'FULFILLED'])]
                failed_lsps = lsp_rows[lsp_rows['visual_status'] == 'FAILED']
                if not active_lsps.empty: target_row = active_lsps.iloc[0]
                elif not completed_lsps.empty: target_row = completed_lsps.iloc[0]
                elif not failed_lsps.empty: target_row = failed_lsps.iloc[0]
                else: target_row = lsp_rows.sort_values('wo_created_at', ascending=False).iloc[0]
            if target_row is None:
                if asm_roots_list:
                    # Multiple assembly attempts may exist (FAILED + RUNNING). Pick the most active.
                    _asr = {'RUNNING': 0, 'IN_PROGRESS': 0, 'READY': 1, 'WAITING': 2, 'BLOCKED': 3, 'SUCCEEDED': 4, 'FAILED': 5, 'CANCELED': 6}
                    _best_asm = min(asm_roots_list, key=lambda r: (
                        _asr.get(str(row_map[r].get('visual_status') or row_map[r].get('wo_status') or ''), 99),
                        -(row_map[r].get('wo_created_at').timestamp() if hasattr(row_map[r].get('wo_created_at'), 'timestamp') else 0)
                    ))
                    target_row = row_map[_best_asm]
                elif not sorted_root_df.empty:
                    target_row = sorted_root_df.iloc[0]
                else:
                    target_row = root_df.iloc[0]
            status = target_row['visual_status']
            # If the root assembly failed but a downstream streakout recovered
            # colonies, show SUCCEEDED in the header.  Only check streakout_operation
            # rows — PCR/Oligo/Syn are upstream inputs, not recovery outcomes, so
            # their SUCCEEDED status must not override the assembly result.
            # Also never applies when the header target is an LSP.
            if status == 'FAILED' and target_row['type'] != 'lsp_workorder':
                recovery_types = {'streakout_operation', 'transformation_offline_operation'}
                recovery_rows = root_df[root_df['type'].isin(recovery_types)]
                if not recovery_rows.empty:
                    running_recovery = recovery_rows[recovery_rows['visual_status'].isin(['RUNNING', 'READY', 'IN_PROGRESS', 'WAITING'])]
                    if not running_recovery.empty:
                        target_row = running_recovery.iloc[0]
                        status = target_row['visual_status']
                    elif (recovery_rows['visual_status'] == 'SUCCEEDED').any():
                        status = 'SUCCEEDED'
            if str(status).upper() in ('NAN', 'NONE', '', 'UNKNOWN'):
                active_info = get_active_step_info(target_row)
                if active_info:
                    status = 'READY' if 'Ready' in active_info else 'RUNNING'
                else:
                    status = 'IN_PROGRESS'
            active_info = get_active_step_info(target_row)
            f_type = format_type_label(target_row['type'])
            _parts_types_set = {'oligo_synthesis_workorder', 'pcr_workorder', 'plasmid_synthesis_workorder', 'syn_part_synthesis_workorder'}
            if target_row['type'] == 'lsp_workorder':
                if active_info:
                    if 'Ready' in active_info and status == 'RUNNING': status = 'READY'
                    header_extra_info = f"<div style='font-size:10px; font-weight:800; margin-top:3px; color:#0e7490; text-align:center;'>{active_info}</div>"
                s_class = f"status-LSP_{status}" if status == 'RUNNING' else f"status-{status}"
            else:
                if active_info:
                    if 'Ready' in active_info and status == 'RUNNING': status = 'READY'
                    _ei_color = '#c2410c' if target_row['type'] in _parts_types_set else '#1d4ed8'
                    header_extra_info = f"<div style='font-size:10px; font-weight:800; margin-top:3px; color:{_ei_color}; text-align:center;'>{active_info}</div>"
                s_class = f"status-{status}"
            badges_html += f'<div style="text-align:right"><span class="badge {s_class}"><b>{f_type}: {status}</b></span>{header_extra_info}</div>'
            root_workorder_row = root_df[root_df['workorder_id'] == root_id]
            if not root_workorder_row.empty: root_stock = root_workorder_row['STOCK_ID'].iloc[0]
            else: root_stock = root_df['STOCK_ID'].iloc[0]
            if pd.isna(root_stock) or str(root_stock).startswith('#'): root_stock = "N/A"
            assembly_types = [format_type_label(t) for t in ['golden_gate_workorder', 'gibson_workorder'] if (root_df['type'] == t).any()]
            assembly_label_text = ' + '.join(assembly_types) if assembly_types else 'Workflow'
            assembly_label = f"<b>{assembly_label_text}</b>"
            div_id = f"group_{req_id.replace('-', '_')}_{root_id.replace('-', '_')}"
            _countable_df = root_df[~((root_df['wo_status'] == 'DRAFT') & root_df['type'].isin(_parts_types_dfs))]
            type_counts = _countable_df['type'].value_counts()
            count_str = ", ".join([f"{count} {format_type_label(k).split()[0].upper()}" for k, count in type_counts.items()])

            html += f"""<div class="{section_class}"><button id="{div_id}_btn" class="dropdown-btn" onclick="toggleSection('{div_id}')"><span id="{div_id}_icon" class="dropdown-icon">▶</span><div class="assembly-info"><span class="assembly-type">{assembly_label}</span><span class="stock-tag" style="font-size:9px; padding:3px 8px; background:#ede9fe; color:#6d28d9; border: 1px solid #c4b5fd;">{root_stock}</span><span class="assembly-counts" style="font-weight: 600;">{count_str}</span><span class="wo-id-tag">Root: {root_id}</span></div><div class="status-badges">{badges_html}</div></button><div id="{div_id}" class="content-pane"><table class="wo-table"><thead><tr><th>Type</th><th>Workorder ID</th><th>Status</th><th>Stock ID</th><th>Created</th><th>TAT</th><th>Details</th><th style="width: 350px;">Queue</th></tr></thead><tbody>"""

            _emitted_parts_header = False
            _emitted_retry_header = False
            # Pre-check: does this section have any non-CANCELED rows other than the root?
            # Used to show a CANCELED root assembly that has no lab data of its own but
            # whose inputs DID have work done (e.g. Gibson canceled after parts were built).
            _section_inputs_have_work = any(
                r.get('wo_status') != 'CANCELED'
                for _, r in sorted_root_df.iterrows()
                if str(r.get('workorder_id', '')) != str(root_id)
            )
            for _, row in sorted_root_df.iterrows():
                # Skip CANCELED workorders that never ran (no queue data) —
                # these are abandoned attempts that clutter the timeline.
                # Exception: show a CANCELED fulfills_request root when its input parts
                # had real work done (the root is needed for context).
                if row.get('wo_status') == 'CANCELED':
                    _pn = row.get('protocol_name')
                    if hasattr(_pn, 'tolist'):
                        _pn = _pn.tolist()
                    if not isinstance(_pn, list):
                        _pn = []
                    if not _pn:
                        if (row.get('fulfills_request') == True
                                and row.get('type') in ('gibson_workorder', 'golden_gate_workorder')
                                and _section_inputs_have_work):
                            pass  # show it — parts had work done
                        else:
                            continue
                _is_parts_row = (row.get('type') in _parts_types_dfs
                                 or (row.get('type') in _asm_types_dfs
                                     and row.get('fulfills_request') == False))
                _attempt_num = row.get('_attempt_number')
                _has_attempt = isinstance(_attempt_num, (int, float)) and not (isinstance(_attempt_num, float) and pd.isna(_attempt_num))
                _status_colors = {'SUCCEEDED': '#15803d', 'FAILED': '#be185d', 'RUNNING': '#0891b2', 'WAITING': '#d97706', 'READY': '#0d9488', 'DRAFT': '#64748b'}

                # "Parts / Inputs" section header — emitted once before the first single-attempt parts row
                if _is_parts_row and not _emitted_parts_header:
                    _emitted_parts_header = True
                    html += f"""<tr><td colspan="8" style="padding:5px 10px 4px; background:#f8fafc; border-top:2px solid #e2e8f0; border-bottom:1px solid #e2e8f0;"><span style="font-size:10px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em;">Parts / Inputs</span></td></tr>"""

                if _is_parts_row and _has_attempt and row.get('tree_depth', 0) == 0:
                    # "Retried Parts" divider — emitted once before the first retry attempt group
                    if not _emitted_retry_header:
                        _emitted_retry_header = True
                        html += f"""<tr><td colspan="8" style="padding:5px 10px 4px; background:#fef9ec; border-top:2px solid #fde68a; border-bottom:1px solid #fde68a;"><span style="font-size:10px; font-weight:700; color:#92400e; text-transform:uppercase; letter-spacing:0.05em;">Retried Parts</span></td></tr>"""
                    _atype = row.get('type', '')
                    _alabel = format_type_label(_atype)
                    _attempt_total = int(row.get('_attempt_total', 1))
                    _sid_raw = row.get('STOCK_ID')
                    _sid_lbl = '' if (_sid_raw is None or (isinstance(_sid_raw, float) and pd.isna(_sid_raw)) or str(_sid_raw).lower() in ('nan', 'none', '') or str(_sid_raw).startswith('#')) else str(_sid_raw).strip()
                    _vstatus = row.get('visual_status', '') or ''
                    _vstatus = '' if (isinstance(_vstatus, float) and pd.isna(_vstatus)) else str(_vstatus)
                    _vcolor = _status_colors.get(_vstatus, '#64748b')
                    _vicon = '✓ ' if _vstatus == 'SUCCEEDED' else '✗ ' if _vstatus == 'FAILED' else ''
                    html += f"""<tr><td colspan="8" style="padding:4px 10px; background:#f1f5f9; border-top:1px solid #cbd5e1; border-bottom:1px solid #e2e8f0;"><span style="font-size:10px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.04em;">{(_sid_lbl + ' — ') if _sid_lbl else ''}Attempt {int(_attempt_num)} of {_attempt_total}</span><span style="margin-left:8px; font-size:10px; font-weight:600; color:{_vcolor};">{_vicon}{_vstatus}</span></td></tr>"""

                elif not _is_parts_row and _has_attempt and row.get('tree_depth', 0) == 0:
                    # Assembly attempt banner (GG/Gibson) — uses best chain status
                    _atype = row.get('type', '')
                    _alabel = format_type_label(_atype)
                    _attempt_total = int(row.get('_attempt_total', 1))
                    _cs_raw = row.get('_attempt_chain_status')
                    _astatus = row.get('visual_status', '') if (_cs_raw is None or (isinstance(_cs_raw, float) and pd.isna(_cs_raw))) else str(_cs_raw)
                    _astatus_color = _status_colors.get(_astatus, '#64748b')
                    _status_icon = '✓ ' if _astatus == 'SUCCEEDED' else '✗ ' if _astatus == 'FAILED' else ''
                    html += f"""<tr><td colspan="8" style="padding:4px 10px; background:#f1f5f9; border-top:2px solid #cbd5e1; border-bottom:1px solid #e2e8f0;"><span style="font-size:10px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.04em;">{_alabel} — Attempt {int(_attempt_num)} of {_attempt_total}</span><span style="margin-left:8px; font-size:10px; font-weight:600; color:{_astatus_color};">{_status_icon}{_astatus}</span></td></tr>"""
                depth = row.get('tree_depth', 0)
                row_class = f"tree-row-{min(depth, 2)}"  # CSS only has 0, 1, 2
                spacer = ""
                if depth >= 1:
                    indent = 20 * (depth - 1)  # 0px for depth 1, 20px for depth 2, 40px for depth 3, etc.
                    arrows = "└" + "─" * depth
                    spacer = f'<span class="tree-line-icon" style="margin-left:{indent}px;">{arrows}</span>'
                type_text = format_type_label(row['type'])
                suffix = str(row.get('visual_suffix', ''));
                if suffix == 'nan': suffix = ''
                extra_tag = ""
                if row['type'] == 'streakout_operation': extra_tag = '<span class="source-badge">Offline Streakout</span>'
                elif row['type'] == 'transformation_offline_operation':
                    if 'STBL3' in suffix: extra_tag = '<span class="source-badge">STBL3 Offline</span>'
                    elif 'EPI400' in suffix: extra_tag = '<span class="source-badge">EPI400 Offline</span>'
                    else: extra_tag = '<span class="source-badge">Offline Trans</span>'
                elif row['type'] in ['lsp_workorder', 'transformation_workorder']:
                    if row['type'] == 'lsp_workorder' and 'Streakout' in suffix: extra_tag = '<span class="source-badge">from Streakout</span>'
                    elif row['type'] == 'lsp_workorder' and 'STBL3' in suffix: extra_tag = '<span class="source-badge">from STBL3</span>'
                    elif row['type'] == 'lsp_workorder' and 'EPI400' in suffix: extra_tag = '<span class="source-badge">from EPI400</span>'
                type_display = f"{spacer}{type_text}{extra_tag}"
                queue_data = parse_pipeline_operations(row.get('protocol_name', []), row.get('operation_state', []), row.get('operation_start', []), row.get('job_id', []), row.get('well_location', []), row.get('operation_ready', []), row.get('ngs_run_number', []))
                effective_status = row['visual_status']
                if queue_data and effective_status == 'RUNNING':
                    for op in queue_data:
                        if op['state'] == 'Ready': effective_status = 'READY'; break
                        if op['state'] == 'Running': break
                badge_class = f"status-{effective_status}"
                if row['type'] == 'lsp_workorder' and effective_status == 'RUNNING': badge_class = "status-LSP_RUNNING"
                tat_display = ""
                wo_start = to_est(row['wo_created_at'])
                if wo_start:
                    now = datetime.now(pytz.timezone('US/Eastern')); end_time = now
                    if effective_status in ['SUCCEEDED', 'FAILED', 'CANCELED', 'UNKNOWN']:
                        if pd.notna(row.get('op_batch_id')) and row['workorder_id'] in parent_details and parent_details[row['workorder_id']]['completion_time']:
                            end_time = parent_details[row['workorder_id']]['completion_time']
                        elif queue_data:
                            for op in reversed(queue_data):
                                if op['state'] == 'Completed' and pd.notna(op['start_time']): end_time = to_est(op['start_time']); break
                    duration = end_time - wo_start
                    if duration.days < 0:  # bad end_time lookup — fall back to now
                        duration = now - wo_start
                    _tat_h = duration.seconds // 3600
                    if effective_status in ['RUNNING', 'IN_PROGRESS', 'WAITING', 'BLOCKED', 'READY']:
                        tat_display = f"Running: {duration.days}d {_tat_h}h"
                    else:
                        tat_display = f"Total: {duration.days}d {_tat_h}h"
                # --- DETAILS & PLATES ---
                lims_plate_map = {}
                json_str = row.get('all_protocol_plates', '{}')
                if pd.notna(json_str) and json_str.strip() != '{}':
                    try:
                        data = json.loads(json_str)
                        for proto, plate_str in data.items():
                            if plate_str:
                                pids = [clean_plate_id(p) for p in str(plate_str).split(',') if clean_plate_id(p)]
                                if proto not in lims_plate_map: lims_plate_map[proto] = []
                                lims_plate_map[proto].extend(pids)
                    except: pass
                for col in ['colony_plates', 'all_locations']:
                    val = row.get(col, '')
                    if pd.notna(val):
                        for entry in str(val).split(' | '):
                            match = re.search(r'Plate(\d+)\s*\(([^)]+)\)', entry)
                            if match:
                                pid, proto = match.group(1), match.group(2).strip()
                                if proto not in lims_plate_map: lims_plate_map[proto] = []
                                if pid not in lims_plate_map[proto]: lims_plate_map[proto].append(pid)
                step_keywords = {'Golden Gate Assembly': ['Golden Gate'], 'Gibson Assembly': ['Gibson'], 'STAR Transformation': ['Transformation', 'Agar'], 'Create Minipreps and Glycerol Stocks': ['Overnight', 'Miniprep', 'Glycerol'], 'Rearray 96 to 384': ['Rearray'], 'DNA Quantification': ['Quant', 'DNA'], 'NGS Sequence Confirmation': ['NGS', 'Sequence'], 'PCR': ['PCR'], 'LSP Receiving': ['LSP Receiving'], 'Manual: Miniprep/Glycerol/Media created': ['Overnight', 'Miniprep', 'Glycerol']}

                pipeline_html = '<div class="timeline-container">'
                if row['type'] == 'transformation_workorder':
                    src_id = row.get('source_asm_process_id')
                    if src_id and src_id in parent_details:
                        parent = parent_details[src_id]
                        src_name = parent['type'].replace(" Assembly", "")
                        det_pills = f'<span class="t-pill">Linked</span>'
                        if pd.notna(parent['job']): det_pills += f'<a href="https://op-tracker.asimov.io/job/{int(parent["job"])}/group/0/step/0/" target="_blank" class="t-pill">Job {int(parent["job"])}</a>'
                        if pd.notna(parent['plate']): det_pills += f'<a href="https://bios.asimov.io/inventory/plates/{clean_plate_id(parent["plate"])}" target="_blank" class="t-pill">Plate{clean_plate_id(parent["plate"])}</a>'
                        pipeline_html += f"""<div class="timeline-row"><div class="t-dot source"></div><div class="t-content"><div class="t-header"><span class="t-name" style="color:#1e3a5f">Source: {src_name}</span><span class="t-time">{parent["completion_str"]}</span></div><div class="t-details">{det_pills}</div></div></div>"""
                if queue_data:
                    for item in queue_data:
                        is_ready = item['state'] == 'Ready'
                        time_str = item["ready_time"].strftime("%m/%d/%Y %H:%M") + " (Ready)" if is_ready and pd.notna(item["ready_time"]) else (item["start_time"].strftime("%m/%d/%Y %H:%M") if pd.notna(item["start_time"]) else "")
                        tooltip_groups = {}
                        keywords = step_keywords.get(item['queue'], [])
                        for lims_proto, pids in lims_plate_map.items():
                            match = False
                            for kw in keywords:
                                if kw.lower() in lims_proto.lower():
                                    if item['queue'] == 'Create Minipreps and Glycerol Stocks' and 'scinomix' in lims_proto.lower(): continue
                                    if item['queue'] == 'LSP Receiving' and 'scinomix' in lims_proto.lower(): continue
                                    match = True; break
                            if match:
                                if lims_proto not in tooltip_groups: tooltip_groups[lims_proto] = set()
                                tooltip_groups[lims_proto].update(pids)
                        if item['queue'] != 'Create Minipreps and Glycerol Stocks' and item['wells']:
                            for w in item['wells']:
                                pid = clean_plate_id(w)
                                if pid:
                                    if item['queue'] not in tooltip_groups: tooltip_groups[item['queue']] = set()
                                    tooltip_groups[item['queue']].add(pid)
                        details_pills = ""
                        if pd.notna(item['job_id']): details_pills += f'<a href="https://op-tracker.asimov.io/job/{int(item["job_id"])}/group/0/step/0/" target="_blank" class="t-pill">Job {int(item["job_id"])}</a> '
                        _run_nums = list(dict.fromkeys(r for r in item.get('run_numbers', []) if r))
                        if _run_nums: details_pills += f'<span class="t-pill">Run {" / ".join(str(r) for r in _run_nums)}</span> '
                        unique_plates = set()
                        for p_set in tooltip_groups.values(): unique_plates.update(p_set)
                        total_plates = len(unique_plates)
                        if total_plates > 0:
                            tooltip_html = ""
                            for proto_name, p_set in tooltip_groups.items():
                                if not p_set: continue
                                tooltip_html += f'<div class="popover-group"><div class="popover-title">{proto_name}</div><div style="margin-left: 8px;">'
                                sorted_plates = sorted(list(p_set), key=lambda x: int(x) if x.isdigit() else 0)
                                for i, pid in enumerate(sorted_plates):
                                    tooltip_html += f'<a href="https://bios.asimov.io/inventory/plates/{pid}" target="_blank" class="popover-link">Plate {pid}</a>'
                                    if (i + 1) % 3 == 0 and i < len(sorted_plates) - 1: tooltip_html += '<br>'
                                tooltip_html += '</div></div>'
                            details_pills += f"""<div class="plate-hover-container"><span class="plate-trigger">{total_plates} Plates</span><div class="plate-popover">{tooltip_html}</div></div>"""
                        pipeline_html += f"""<div class="timeline-row"><div class="t-dot {item['class']}"></div><div class="t-content"><div class="t-header"><span class="t-name">{item['queue']}</span><span class="t-time">{time_str}</span></div><div class="t-details">{details_pills}</div></div></div>"""
                else:
                    if lims_plate_map:
                        # Group all LIMS plates under a single manual entry with the same
                        # plate-hover popover format used for normal OpTracker timeline rows.
                        tooltip_html = ""
                        all_pids = set()
                        for proto_name, pids in lims_plate_map.items():
                            pids_sorted = sorted(pids, key=lambda x: int(x) if x.isdigit() else 0)
                            all_pids.update(pids_sorted)
                            tooltip_html += f'<div class="popover-group"><div class="popover-title">{proto_name}</div><div style="margin-left: 8px;">'
                            for i, pid in enumerate(pids_sorted):
                                tooltip_html += f'<a href="https://bios.asimov.io/inventory/plates/{pid}" target="_blank" class="popover-link">Plate {pid}</a>'
                                if (i + 1) % 3 == 0 and i < len(pids_sorted) - 1: tooltip_html += '<br>'
                            tooltip_html += '</div></div>'
                        lims_pills = f'<div class="plate-hover-container"><span class="plate-trigger">{len(all_pids)} Plates</span><div class="plate-popover">{tooltip_html}</div></div>'
                        _fallback_labels = {
                            'pcr_workorder': 'PCR',
                            'oligo_synthesis_workorder': 'Oligo Synthesis',
                            'plasmid_synthesis_workorder': 'Plasmid Synthesis',
                            'syn_part_synthesis_workorder': 'Syn Part Synthesis',
                        }
                        _fallback_label = _fallback_labels.get(row['type'], 'Manual: Miniprep/Glycerol/Media created')
                        pipeline_html += f"""<div class="timeline-row"><div class="t-dot succeeded"></div><div class="t-content"><div class="t-header"><span class="t-name">{_fallback_label}</span><span class="t-time"></span></div><div class="t-details">{lims_pills}</div></div></div>"""
                    else:
                        pipeline_html += '<span style="color: #9ca3af; font-size: 11px;">No queue data</span>'
                pipeline_html += '</div>'

                # --- DETAILS INFO ---
                details_info = ""
                waiting_items = set()
                if row['wo_status'] in ['WAITING', 'BLOCKED'] and pd.notna(row.get('Waiting')):
                    waiting_items = set([x.strip() for x in str(row['Waiting']).split(',') if x.strip()])
                tree_stock_ids = set(sorted_root_df['STOCK_ID'].dropna().astype(str).values)
                def render_part_tag(part_name, label_prefix=""):
                    clean_name = part_name.split(':')[0].strip()
                    if not clean_name: return ""
                    if clean_name in waiting_items:
                        if clean_name in tree_stock_ids:
                            return f'<span class="part-tag in-production" title="Being made in this workflow">{label_prefix}{clean_name}</span>'
                        else:
                            _tip_info = _stock_to_req.get(clean_name)
                            if _tip_info:
                                _tip_req = _tip_info['req_id'][:8] + '…' if len(_tip_info['req_id']) > 8 else _tip_info['req_id']
                                _tip_html = (
                                    f'<div class="missing-tip">'
                                    f'<div class="missing-tip-req">REQ: {_tip_info["req_id"]}</div>'
                                    f'<div class="missing-tip-exp">{_tip_info["exp_name"]}</div>'
                                    f'<div class="missing-tip-status">In Progress · Part Waiting</div>'
                                    f'</div>'
                                )
                                return f'<span class="missing-tip-wrap"><span class="part-tag missing">{label_prefix}{clean_name}</span>{_tip_html}</span>'
                            else:
                                return f'<span class="part-tag missing" title="Being built outside this workflow">{label_prefix}{clean_name}</span>'
                    else:
                        return f'<span class="part-tag">{label_prefix}{clean_name}</span>'
                inputs_html = '<div class="parts-container">'
                bb = row.get('backbone', '')
                if pd.notna(bb) and ':' in bb: inputs_html += render_part_tag(bb, "BB: ")
                parts_raw = row.get('parts', '')
                if pd.notna(parts_raw):
                    for p in [p for p in str(parts_raw).split(', ') if ':' in p]: inputs_html += render_part_tag(p)
                pcr_info = row.get('pcr_info', '')
                if pd.notna(pcr_info):
                    for p in [p for p in str(pcr_info).split(', ') if ':' in p]: inputs_html += render_part_tag(p)
                inputs_html += '</div>'
                if 'part-tag' in inputs_html: details_info += inputs_html

                if row['type'] == 'lsp_workorder':
                    lims_id = row.get('lsp_batch_id')
                    op_id = row.get('lsp_batch_id_from_optracker')
                    bios_id = row.get('bios_batch_id')
                    if pd.notna(lims_id) and str(lims_id).lower() != 'nan': display_lsp_id = lims_id
                    elif pd.notna(op_id) and str(op_id).lower() != 'nan': display_lsp_id = op_id
                    elif pd.notna(bios_id) and str(bios_id).lower() != 'nan': display_lsp_id = bios_id
                    else: display_lsp_id = f"WO-{str(row['workorder_id'])[:8]}"
                    _batch_num = str(display_lsp_id).replace("LSP-", "").strip()
                    if _batch_num.isdigit():
                        _batch_href = f"https://bios.asimov.io/inventory/lsp-batches/{_batch_num}"
                        _batch_label = f'<a href="{_batch_href}" target="_blank" style="color:#7c3aed;text-decoration:underline;font-weight:700;">{display_lsp_id}</a>'
                    else:
                        _batch_label = f'<span style="color:#4b5563;font-weight:700;">{display_lsp_id}</span>'
                    lsp_parts = [f'<div style="font-size: 10px; font-weight: 700; margin-bottom: 4px;">{_batch_label}</div>']


                    # Source Material Popover
                    source_raw = str(row.get('source_material_link', ''))
                    exp_name = str(row.get('experiment_name', 'N/A'))
                    construct_name, proc_id = "N/A", "N/A"
                    # Use source_lsp_process_id as fallback for proc_id
                    fallback_proc = row.get('source_lsp_process_id')
                    if pd.notna(fallback_proc) and str(fallback_proc) != 'nan':
                        proc_id = str(fallback_proc)
                    if source_raw and source_raw != 'nan':
                        if ":" in source_raw:
                            parts = source_raw.split(':', 1)
                            exp_name = parts[0].strip()
                            if len(parts) > 1:
                                remainder = parts[1].strip()
                                if "(" in remainder:
                                    construct_part, id_part = remainder.rsplit("(", 1)
                                    construct_name = construct_part.strip()
                                    proc_id = id_part.replace(")", "").strip()
                                else: construct_name = remainder
                        else:
                            # Orphaned LSP format: "construct_name (process_id)"
                            if "(" in source_raw:
                                construct_part, id_part = source_raw.rsplit("(", 1)
                                construct_name = construct_part.strip()
                                proc_id = id_part.replace(")", "").strip()
                            else:
                                construct_name = source_raw
                    # Fall back to the row's own construct_name column if parsing didn't find one
                    if construct_name in ("N/A", "") or construct_name.startswith("Source:"):
                        own_cn = row.get('construct_name')
                        if pd.notna(own_cn) and str(own_cn) not in ('nan', ''):
                            construct_name = str(own_cn)

                    # Fallback: if construct_name looks like a UUID, source_material_link was unpopulated
                    # — look up the source row directly from req_df
                    if re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', construct_name):
                        source_row = req_df[req_df['workorder_id'] == proc_id]
                        if not source_row.empty:
                            resolved_construct = source_row['construct_name'].iloc[0]
                            if pd.notna(resolved_construct) and str(resolved_construct) not in ('nan', ''):
                                construct_name = str(resolved_construct)
                            resolved_exp = source_row['experiment_name'].iloc[0] if 'experiment_name' in source_row.columns else None
                            if pd.notna(resolved_exp) and str(resolved_exp) not in ('nan', ''):
                                exp_name = str(resolved_exp)
                            job_val = source_row['job_id'].iloc[0]
                            if isinstance(job_val, list) and len(job_val) > 0 and pd.notna(job_val[0]):
                                proc_id = f"job__{int(job_val[0])}"

                    pai_val = row.get('STOCK_ID', 'N/A')

                    # Look up source workorder for full experiment name, req_id, request_status
                    src_req_id, src_req_status, src_exp_name = "N/A", "", exp_name
                    if proc_id and proc_id != "N/A" and not proc_id.startswith("job__"):
                        _src = df[df["workorder_id"] == proc_id]
                        if not _src.empty:
                            _r = _src.iloc[0]
                            _e = str(_r.get("experiment_name") or "")
                            if _e and _e not in ("nan", ""):
                                src_exp_name = _e
                            src_req_id = str(_r.get("req_id") or "N/A")
                            src_req_status = str(_r.get("request_status") or "")

                    # Status badge for source request
                    _st = src_req_status.upper()
                    if _st == "CANCELED":
                        src_badge = '<span style="background:#fee2e2;color:#b91c1c;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;border:1px solid #fca5a5;margin-left:6px;">CANCELED</span>'
                    elif _st in ("FULFILLED", "COMPLETED"):
                        src_badge = f'<span style="background:#dcfce7;color:#15803d;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;border:1px solid #86efac;margin-left:6px;">{src_req_status}</span>'
                    elif _st and _st not in ("", "NAN", "N/A"):
                        src_badge = f'<span style="background:#f3f4f6;color:#6b7280;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;border:1px solid #d1d5db;margin-left:6px;">{src_req_status}</span>'
                    else:
                        src_badge = ""

                    # BIOS link for process ID
                    if proc_id.startswith("job__"):
                        proc_href = f"https://op-tracker.asimov.io/job/{proc_id.replace('job__', '')}/group/0/step/0/"
                        proc_label = f"Job {proc_id.replace('job__', '')}"
                    else:
                        proc_href = f"https://bios.asimov.io/inbox/work-orders?filter_-l=%5B%7B%22id%22%3A%22workOrderId%22%2C%22value%22%3A%22{proc_id}%22%7D%5D"
                        proc_label = proc_id

                    # Resolve input well ID for Source Info popover
                    _input_well_raw = row.get('lsp_input_well')
                    if pd.isna(_input_well_raw) or str(_input_well_raw) == 'nan':
                        _input_well_raw = row.get('input_well_id')
                    _input_well_html = ""
                    if pd.notna(_input_well_raw) and str(_input_well_raw) != 'nan':
                        _wm = re.search(r'"id":\s*(\d+)', str(_input_well_raw))
                        _fid = _wm.group(1) if _wm else str(_input_well_raw)
                        _input_well_html = f"""
                                    <span style="color:#9ca3af;font-size:10px;font-weight:600;text-transform:uppercase;padding:4px 0;border-bottom:1px solid #f3f4f6;">Input:</span>
                                    <span style="font-size:11px;padding:4px 0;border-bottom:1px solid #f3f4f6;"><a href="https://bios.asimov.io/inventory/wells/{_fid}" target="_blank" style="color:#7c3aed;text-decoration:underline;font-weight:700;">well{_fid}</a></span>"""

                    lsp_parts.append(f"""
                        <div class="plate-hover-container" style="display: inline-block; margin-bottom: 6px;">
                            <span class="plate-trigger" style="background: #e5e7eb; color: #4b5563; cursor: pointer; font-size: 9px; font-weight: 600; padding: 2px 6px; border-radius: 3px; border: 1px solid #d1d5db;">
                                Source Info
                            </span>
                            <div class="plate-popover" style="width: 460px; white-space: normal; padding: 15px; border-top: 4px solid #6b7280; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                                <div style="border-bottom: 1px solid #e5e7eb; margin-bottom: 10px; padding-bottom: 5px; font-weight: 800; color: #4b5563; text-transform: uppercase; font-size: 11px;">
                                    Source Material Context
                                </div>
                                <div style="display: grid; grid-template-columns: 115px 1fr; gap: 0; font-size: 12px; line-height: 1.5; color: #1f2937;">
                                    <span style="color:#9ca3af;font-size:10px;font-weight:600;text-transform:uppercase;padding:4px 0;border-bottom:1px solid #f3f4f6;">Experiment:</span>
                                    <span style="padding:4px 0;border-bottom:1px solid #f3f4f6;">{src_exp_name}</span>
                                    <span style="color:#9ca3af;font-size:10px;font-weight:600;text-transform:uppercase;padding:4px 0;border-bottom:1px solid #f3f4f6;">Process ID:</span>
                                    <div style="padding:4px 0;border-bottom:1px solid #f3f4f6;">
                                        <a href="{proc_href}" target="_blank" style="font-family: monospace; font-size: 11px; color: #4b5563; background: #f3f4f6; padding: 2px 6px; border-radius: 3px; text-decoration: underline; border: 1px solid #d1d5db;">{proc_label}</a>
                                        <div style="font-size: 9px; color: #9ca3af; margin-top: 4px; padding-left: 2px;">{construct_name} &nbsp;·&nbsp; <span style="font-family: monospace; color: #6b7280;">{pai_val}</span></div>
                                    </div>
                                    <span style="color:#9ca3af;font-size:10px;font-weight:600;text-transform:uppercase;padding:4px 0;border-bottom:1px solid #f3f4f6;">Request ID:</span>
                                    <span style="font-family: monospace; font-size: 11px; color: #4b5563; padding:4px 0;border-bottom:1px solid #f3f4f6;">{src_req_id}{src_badge}</span>
                                    {_input_well_html}
                                </div>
                            </div>
                        </div>""")
                    # ── QC Details popover ────────────────────────────────────
                    _qc_fields = [
                        ("QC Status",            row.get("qc_status")),
                        ("NGS Status",           row.get("ngs_status")),
                        ("Concentration Status", row.get("concentration_status")),
                        ("Yield Status",         row.get("yield_status")),
                        ("Digest",               row.get("digest")),
                        ("Available",            "Yes" if row.get("available") is True or str(row.get("available","")).lower() == "true" else ("No" if row.get("available") is False or str(row.get("available","")).lower() == "false" else None)),
                        ("Comment",              row.get("batch_comments")),
                    ]
                    # Filter to fields that have a real value
                    _qc_rows = [(lbl, str(val)) for lbl, val in _qc_fields if val is not None and str(val) not in ("nan","None","")]

                    # Button color: red if any Fail, green if all status fields Pass, grey otherwise
                    _status_vals = [str(row.get(f) or "") for f in ["qc_status","ngs_status","concentration_status","yield_status","digest"]]
                    _status_vals = [v for v in _status_vals if v and v not in ("nan","None","")]
                    if any(v.lower() == "fail" for v in _status_vals):
                        _qc_btn_style = "background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;"
                    elif _status_vals and all(v.lower() == "pass" for v in _status_vals):
                        _qc_btn_style = "background:#dcfce7;color:#15803d;border:1px solid #86efac;"
                    else:
                        _qc_btn_style = "background:#e5e7eb;color:#4b5563;border:1px solid #d1d5db;"

                    def _qc_dot(val):
                        v = str(val).lower()
                        if v == "pass" or v == "yes":
                            return '<span style="color:#16a34a;font-size:11px;margin-right:4px;">●</span>'
                        elif v == "fail" or v == "no":
                            return '<span style="color:#dc2626;font-size:11px;margin-right:4px;">●</span>'
                        else:
                            return '<span style="color:#9ca3af;font-size:11px;margin-right:4px;">○</span>'

                    if _qc_rows:
                        _qc_grid = "".join(
                            f'<span style="color:#9ca3af;font-size:10px;font-weight:600;text-transform:uppercase;padding:4px 0;border-bottom:1px solid #f3f4f6;">{lbl}:</span>'
                            f'<span style="font-size:11px;color:#1f2937;padding:4px 0;border-bottom:1px solid #f3f4f6;">{"" if lbl == "Comment" else _qc_dot(val)}{val}</span>'
                            for lbl, val in _qc_rows
                        )
                        lsp_parts.append(f"""
                        <div class="plate-hover-container" style="display: inline-block; margin-bottom: 6px; margin-left: 4px;">
                            <span class="plate-trigger" style="{_qc_btn_style} cursor: pointer; font-size: 9px; font-weight: 600; padding: 2px 6px; border-radius: 3px;">
                                QC Details
                            </span>
                            <div class="plate-popover" style="width: 340px; white-space: normal; padding: 15px; border-top: 4px solid #6b7280; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                                <div style="border-bottom: 1px solid #e5e7eb; margin-bottom: 10px; padding-bottom: 5px; font-weight: 800; color: #4b5563; text-transform: uppercase; font-size: 11px;">
                                    QC Details
                                </div>
                                <div style="display: grid; grid-template-columns: 160px 1fr; gap: 0; line-height: 1.6;">
                                    {_qc_grid}
                                </div>
                            </div>
                        </div>""")

                    metrics_list = []
                    strain = row.get('comp_cell') or row.get('cloning_strain')
                    conc = row.get('qubit_concentration_ngul')
                    yld = row.get('qubit_yield')
                    dl_fmt = row.get('delivery_format')
                    if pd.notna(dl_fmt) and str(dl_fmt) not in ('nan', ''): metrics_list.append(("Format", str(dl_fmt)))
                    if pd.notna(strain) and str(strain) != 'nan': metrics_list.append(("Strain", str(strain)))
                    if pd.notna(conc) and str(conc) != 'nan': metrics_list.append(("Qubit", f"{conc} ng/µL"))
                    if pd.notna(yld) and str(yld) != 'nan': metrics_list.append(("Yield", f"{yld} µg"))
                    loc = row.get('location')
                    if pd.notna(loc) and str(loc) != 'nan' and str(loc).strip():
                        metrics_list.append(("Location", str(loc)))
                    if metrics_list:
                        grid_cells = "".join(
                            f'<span style="color:#6b7280;font-size:9px;font-weight:700;text-transform:uppercase;">{lbl}</span>'
                            f'<span style="font-size:10px;color:#1e293b;">{val}</span>'
                            for lbl, val in metrics_list
                        )
                        lsp_parts.append(
                            f'<div style="display:grid;grid-template-columns:52px 1fr;gap:2px 6px;margin-bottom:4px;">{grid_cells}</div>'
                        )
                    details_info += "".join(lsp_parts)

                elif row['type'] in ['oligo_synthesis_workorder', 'pcr_workorder', 'plasmid_synthesis_workorder', 'syn_part_synthesis_workorder']:
                    _vendor = row.get('vendor')
                    if pd.notna(_vendor) and str(_vendor).strip() not in ('', 'nan', 'None'):
                        details_info += f"<div style='font-size:10px;color:#64748b;margin-top:2px;'>Vendor: {str(_vendor).strip()}</div>"
                    if row['type'] == 'pcr_workorder':
                        _wc = row.get('well_comments')
                        _wc_clean = str(_wc).strip().strip(';').strip() if _wc is not None and not (isinstance(_wc, float) and pd.isna(_wc)) else ''
                        if _wc_clean and _wc_clean not in ('nan', 'None'):
                            details_info += f"<div style='font-size:10px;color:#b45309;background:#fffbeb;border:1px solid #fcd34d;border-radius:3px;padding:2px 5px;margin-top:3px;'>&#9888; {_wc_clean}</div>"

                elif row['type'] in ['golden_gate_workorder', 'gibson_workorder', 'transformation_workorder', 'transformation_offline_operation', 'streakout_operation']:
                    strain = row.get('cloning_strain')
                    if pd.notna(strain): details_info += f"<div style='font-size:10px;color:#64748b;margin-top:2px;'>Strain: {strain}</div>"
                    _imaged   = row.get('imaged_colonies')
                    _pickable = row.get('pickable_colonies')
                    _picked   = row.get('picked_colonies')
                    if any(pd.notna(v) for v in [_imaged, _pickable, _picked]):
                        # Colony counts (plain, no link wrapping)
                        details_info += (
                            f"<div style='display:grid;grid-template-columns:58px 1fr;gap:1px 4px;margin-top:3px;'>"
                            f"<span style='font-size:9px;font-weight:700;text-transform:uppercase;color:#6b7280;'>Imaged</span><span style='font-size:10px;color:#1e293b;'>{int(_imaged)}</span>"
                            f"<span style='font-size:9px;font-weight:700;text-transform:uppercase;color:#6b7280;'>Pickable</span><span style='font-size:10px;color:#1e293b;'>{int(_pickable) if pd.notna(_pickable) else 0}</span>"
                            f"<span style='font-size:9px;font-weight:700;text-transform:uppercase;color:#6b7280;'>Picked</span><span style='font-size:10px;color:#1e293b;'>{int(_picked) if pd.notna(_picked) else 0}</span>"
                            f"</div>"
                        )
                        # Agar plate link — plate_id + alphanumeric well from colonypickingcounts
                        _cpid   = row.get('colony_plate_id')
                        _cwpos  = row.get('colony_well_position')
                        _cwcnt  = row.get('colony_plate_well_count')
                        if _cpid and pd.notna(_cpid) and str(_cpid) not in ('nan', 'None', ''):
                            _plate_url  = f'https://bios.asimov.io/inventory/plates/{int(_cpid)}'
                            _well_alpha = ''
                            try:
                                if pd.notna(_cwpos) and pd.notna(_cwcnt):
                                    _ncols = 24 if int(_cwcnt) == 384 else 12
                                    _pos   = int(_cwpos)
                                    _row_i = (_pos - 1) // _ncols
                                    _col_i = (_pos - 1) %  _ncols + 1
                                    _well_alpha = chr(65 + _row_i) + str(_col_i)
                            except Exception:
                                pass
                            _well_label = f' · {_well_alpha}' if _well_alpha else ''
                            details_info += (
                                f"<div style='margin-top:4px;margin-bottom:4px;'>"
                                f"<span style='font-size:9px;font-weight:700;text-transform:uppercase;color:#6b7280;margin-right:4px;'>Agar</span>"
                                f"<a href='{_plate_url}' target='_blank' "
                                f"style='font-size:9px;font-family:monospace;"
                                f"color:#0369a1;text-decoration:underline dotted;'>"
                                f"Plate {int(_cpid)}{_well_label}</a>"
                                f"</div>"
                            )
                    if row['visual_status'] == 'FAILED' and str(row['wo_status']).upper() == 'SUCCEEDED' and not row['is_software_fail']:
                        _tot = row.get('total_colonies')
                        if pd.isna(_tot) or int(_tot) == 0:
                            details_info += f'<br><span class="colony-badge" style="background:#fce7f3;color:#be185d;">0 colonies</span>'
                    if row['visual_status'] in ['SUCCEEDED', 'FAILED'] or row['wo_status'] == 'FAILED':
                        tot = row.get('total_colonies'); seq = row.get('seq_confirmed')
                        if pd.notna(tot) and tot > 0:
                            tot = int(tot); seq = int(seq) if pd.notna(seq) else 0
                            color, bg = ("#0e7490", "#cffafe") if seq > 0 else ("#be185d", "#fce7f3")
                            seq_conf_list = row.get('seq_confirmed_colonies', '')
                            selected_col = row.get('selected_colony', 'None'); selected_col_num = None
                            if selected_col != 'None' and ':' in str(selected_col):
                                clean = re.sub(r'\[.*?\]', '', str(selected_col)).strip()
                                _, selected_col_num = clean.split(':')
                            protocol_wells = {}
                            if pd.notna(seq_conf_list) and seq_conf_list:
                                for entry in str(seq_conf_list).split(','):
                                    entry = entry.strip()
                                    match = re.match(r'(\d+):(\d+)\[([^\]]+)\]', entry)
                                    if match:
                                        well_id, col_num, protocol_name = match.groups()
                                        if protocol_name.strip() not in protocol_wells: protocol_wells[protocol_name.strip()] = {}
                                        protocol_wells[protocol_name.strip()][col_num] = well_id
                            protocol_order = ['LSP Receiving', 'Miniprep', 'Bank Overnights', 'Rearray 96 to 384', 'Glycerol Stocking Scinomix', 'Glycerol'] if row['type'] == 'lsp_workorder' else ['Miniprep', 'Bank Overnights', 'Rearray 96 to 384', 'Glycerol Stocking Scinomix']
                            popover_content = ""
                            for protocol_name in protocol_order:
                                matching_wells = {}
                                for proto, wells in protocol_wells.items():
                                    if protocol_name.lower() in proto.lower(): matching_wells.update(wells)
                                if matching_wells:
                                    links = ""
                                    for col_num in sorted(matching_wells.keys(), key=lambda x: int(x)):
                                        w_id = matching_wells[col_num]; url = f"https://bios.asimov.io/inventory/wells/{w_id}"; label = f"well{w_id}_col{col_num}"
                                        if col_num == selected_col_num: links += f'<a href="{url}" target="_blank" class="popover-link" style="color:#0891b2;">★ {label}</a>'
                                        else: links += f'<a href="{url}" target="_blank" class="popover-link">{label}</a>'
                                    popover_content += f'<div class="popover-group"><div class="popover-title">{protocol_name}</div>{links}</div>'
                            if popover_content: details_info += f'<br><div class="plate-hover-container"><span class="colony-badge" style="background: {bg}; color: {color}; cursor:pointer;">{seq}/{tot} colonies seq confirmed</span><div class="plate-popover">{popover_content}</div></div>'
                            else: details_info += f'<br><span class="colony-badge" style="background: {bg}; color: {color};">{seq}/{tot} colonies seq confirmed</span>'
                _sid = row.get("STOCK_ID", "N/A")
                _sid_str = str(_sid)
                if _sid_str.startswith('#'):  # placeholder, not a real stock ID
                    _sid_str = ""; _sid = ""
                if _sid_str not in ("N/A", "nan", "None") and _sid_str in _primary_root_stocks:
                    _sid_class = "stock-id-badge matches-root"
                elif _sid_str not in ("N/A", "nan", "None") and _sid_str in _secondary_root_stocks:
                    _sid_class = "stock-id-badge secondary-root"
                else:
                    _sid_class = "stock-id-badge"
                html += f"""<tr class="{row_class}"><td><span class="type-label">{type_display}</span></td><td><code class="wo-id-tag" title="{row['workorder_id']}">{row['workorder_id']}</code></td><td><span class="badge {badge_class}">{effective_status}</span></td><td><span class="{_sid_class}">{_sid}</span></td><td><div class="date-tag">{pd.to_datetime(row['wo_created_at']).strftime('%Y-%m-%d') if pd.notna(row['wo_created_at']) else ''}</div></td><td class="tat-cell">{tat_display}</td><td class="details-cell">{details_info}</td><td>{pipeline_html}</td></tr>"""
            html += "</tbody></table></div></div>"
        html += "</div></div>"
        return html

    # =========================================================================
    # 4. MAIN RENDER LOOP
    # =========================================================================

    _BUCKET_STAGES = [
        ('In Design',    '#cbd5e1'),
        ('Vendor Parts', '#94a3b8'),
        ('DV/PL1 Build', '#60a5fa'),
        ('PCR',         '#38bdf8'),
        ('Assembly',    '#a78bfa'),
        ('Assembly QC', '#818cf8'),
        ('LSP',         '#34d399'),
        ('LSP QC',      '#2dd4bf'),
        ('Reviewing',   '#fbbf24'),
        ('Releasing',   '#f97316'),
        ('Stalled',     '#f87171'),
    ]

    # Shared age-color palette — used by both timeline dots and stage view bars
    _BLUE_RAMP   = ['#bae6fd','#7dd3fc','#38bdf8','#0ea5e9','#0284c7','#0369a1','#075985','#0c4a6e']
    _PURPLE_RAMP = ['#f5d0fe','#e879f9','#d946ef','#c026d3','#a21caf','#86198f','#701a75','#4a044e']
    _WARN_COLOR = '#b45309'   # amber-700 — muted warm gold, easier on eyes
    _OVER_COLOR = '#dc2626'   # red-600

    def _age_color(age_weeks, yellow_limit, red_limit, ramp=None):
        """Return hex color for a request based on its age and per-experiment thresholds."""
        if ramp is None: ramp = _BLUE_RAMP
        if age_weeks >= red_limit:
            return _OVER_COLOR
        if age_weeks >= yellow_limit:
            return _WARN_COLOR
        max_idx = max(yellow_limit - 1, 1)
        idx = min(int(age_weeks / max_idx * (len(ramp) - 1)), len(ramp) - 1)
        return ramp[idx]

    def _render_bucket_chart(stage_counts, fulfilled_week_counts, yellow_limit, red_limit, stage_items=None, ramp=None):
        """Render a horizontal bar chart showing all pipeline stages.
        On-track buckets use a light→dark gradient (colorblind-safe).
        Warning uses amber-gold. Overdue uses solid red.
        When no active requests remain, shows fulfilled TAT distribution instead."""
        def _seg_color(bucket):
            return _age_color(bucket, yellow_limit, red_limit, ramp=ramp)

        def _txt_color(bucket):
            # Lightest two blues need dark text for contrast; everything else white
            if bucket < yellow_limit and bucket < 2:
                return 'rgba(15,23,42,0.9)'
            return 'rgba(255,255,255,0.95)'

        total = sum(sum(wc.values()) for wc in stage_counts.values()) if stage_counts else 0

        # When all requests are fulfilled, show TAT distribution instead of stage rows
        if total == 0 and fulfilled_week_counts:
            f_total = sum(fulfilled_week_counts.values())
            f_max   = max(fulfilled_week_counts.values(), default=1)
            segs = ''
            for bucket in range(9):
                bc = fulfilled_week_counts.get(bucket, 0)
                if bc == 0: continue
                seg_w = max(22, int((bc / f_total) * 400))
                color = _age_color(bucket, yellow_limit, red_limit)
                tc = 'rgba(15,23,42,0.9)' if bucket < yellow_limit and bucket < 2 else 'rgba(255,255,255,0.95)'
                age_label = f'{bucket}w' if bucket < 8 else '8w+'
                segs += (
                    f'<div style="width:{seg_w}px;height:24px;background:{color};flex-shrink:0;'
                    f'display:flex;flex-direction:column;align-items:center;justify-content:center;'
                    f'overflow:hidden;cursor:default;" title="Fulfilled in {age_label}: {bc}">'
                    f'<span style="font-size:10px;font-weight:700;color:{tc};line-height:1;">{bc}</span>'
                    f'<span style="font-size:7px;color:{tc};opacity:0.8;line-height:1;">{age_label}</span>'
                    f'</div>'
                )
            return f'''<div style="padding:10px 14px 8px 14px;">
                <div style="font-size:9px;color:rgba(255,255,255,0.45);margin-bottom:8px;font-family:monospace;">{f_total} fulfilled — production TAT distribution</div>
                <div style="display:flex;border-radius:4px;overflow:hidden;gap:1px;">{segs}</div>
                <div style="font-size:8px;color:rgba(255,255,255,0.35);margin-top:6px;">weeks from request creation to LSP ready-to-ship</div>
            </div>'''

        max_c = max((sum(wc.values()) for wc in stage_counts.values()), default=1)
        rows  = ''
        for stage_key, _ in _BUCKET_STAGES:
            wc  = stage_counts.get(stage_key, {})
            cnt = sum(wc.values())
            label_style = "color:rgba(255,255,255,0.8);font-weight:600;" if cnt > 0 else "color:rgba(255,255,255,0.3);font-weight:400;"
            if cnt == 0:
                bar_html = '<div style="flex:1;height:20px;background:rgba(255,255,255,0.07);border-radius:3px;border:1px dashed rgba(255,255,255,0.15);"></div>'
            else:
                bar_total_w = max(20, int((cnt / max_c) * 200))
                segs = ''
                for bucket in range(9):
                    bc = wc.get(bucket, 0)
                    if bc == 0: continue
                    seg_w = max(22, int((bc / cnt) * bar_total_w))
                    color = _seg_color(bucket)
                    tc    = _txt_color(bucket)
                    age_label = f'{bucket}w' if bucket < 8 else '8w+'
                    _items = (stage_items or {}).get(stage_key, {}).get(bucket, [])
                    if _items:
                        _MAX = 12
                        _shown = _items[:_MAX]
                        _rest = len(_items) - _MAX
                        _tip = f"{age_label}: {bc}\n" + "\n".join(_shown)
                        if _rest > 0: _tip += f"\n… and {_rest} more"
                    else:
                        _tip = f"{age_label}: {bc}"
                    segs += (
                        f'<div style="width:{seg_w}px;height:20px;background:{color};flex-shrink:0;'
                        f'display:flex;flex-direction:column;align-items:center;justify-content:center;'
                        f'overflow:hidden;cursor:default;" title="{_tip}">'
                        f'<span style="font-size:9px;font-weight:700;color:{tc};line-height:1;">{bc}</span>'
                        f'<span style="font-size:7px;color:{tc};opacity:0.8;line-height:1;">{age_label}</span>'
                        f'</div>'
                    )
                bar_html = f'<div style="display:flex;border-radius:3px;overflow:hidden;gap:1px;">{segs}</div>'
            count_html = f'<span style="font-size:10px;color:white;font-weight:700;font-family:monospace;margin-left:4px;">{cnt}</span>' if cnt > 0 else ''
            rows += f'''<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                <div style="width:88px;font-size:9px;{label_style}text-align:right;white-space:nowrap;font-family:monospace;">{stage_key}</div>
                <div style="flex:1;display:flex;align-items:center;">{bar_html}{count_html}</div>
            </div>'''
        total_str = f'{total} active request{"s" if total != 1 else ""}' if total > 0 else 'No active requests'
        _mid_blue = _BLUE_RAMP[len(_BLUE_RAMP) // 2]
        legend = f'''<div style="display:flex;gap:12px;margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.1);">
            <span style="font-size:8px;color:rgba(255,255,255,0.5);display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:{_BLUE_RAMP[0]};border:1px solid rgba(255,255,255,0.25);border-radius:2px;display:inline-block;"></span>Newer</span>
            <span style="font-size:8px;color:rgba(255,255,255,0.5);display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:{_mid_blue};border-radius:2px;display:inline-block;"></span>On Track</span>
            <span style="font-size:8px;color:rgba(255,255,255,0.5);display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:{_WARN_COLOR};border-radius:2px;display:inline-block;"></span>Warning</span>
            <span style="font-size:8px;color:rgba(255,255,255,0.5);display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:{_OVER_COLOR};border-radius:2px;display:inline-block;"></span>Overdue</span>
        </div>'''
        return f'''<div style="padding:10px 14px 8px 14px;">
            <div style="font-size:9px;color:rgba(255,255,255,0.45);margin-bottom:8px;font-family:monospace;">{total_str}</div>
            {rows}{legend}
        </div>'''

    # Build STOCK_ID → request info lookup for missing-part tooltips.
    # For each active (non-finished, non-canceled) workorder, map its STOCK_ID
    # to the req_id and experiment_name that is building it.
    _ACTIVE_VS = {'WAITING', 'RUNNING', 'IN_PROGRESS', 'READY', 'BLOCKED', 'LSP_RUNNING'}
    _active_wo = df[df['visual_status'].isin(_ACTIVE_VS) & df['STOCK_ID'].notna()]
    _stock_to_req: dict = {}
    for _, _wr in _active_wo.iterrows():
        _sid = str(_wr.get('STOCK_ID', '') or '').strip()
        if not _sid or _sid in ('nan', 'None'): continue
        if _sid in _stock_to_req: continue  # keep first hit
        _stock_to_req[_sid] = {
            'req_id':   str(_wr.get('req_id') or _wr.get('request_id') or ''),
            'exp_name': str(_wr.get('experiment_name') or ''),
            'status':   str(_wr.get('visual_status') or ''),
            'wo_type':  str(_wr.get('type') or ''),
        }

    # Load experiment due dates (written by fetch_due_dates() before render)
    _due_date_map: dict[str, str] = {}
    try:
        from dnasc.extractors.sheets import load_due_dates
        _due_date_map = load_due_dates()
    except Exception:
        pass

    # Pre-compute canonical experiment for each req_id so that requests whose parts
    # belong to a different experiment (cross-experiment fanning) are only rendered
    # once — in the experiment that owns the root GG/Gibson workorder.
    # Without this, a req_id appears in multiple project groups → duplicate div IDs
    # → the second card's toggle never works (getElementById finds the first).
    _asm_types_canon = {'golden_gate_workorder', 'gibson_workorder'}
    _req_canon_exp: dict = {}
    for _rc_rid, _rc_df in df.groupby('req_id'):
        _gg = _rc_df[_rc_df['type'].isin(_asm_types_canon) & (_rc_df['fulfills_request'] == True)]
        _exp_vals = _gg['experiment_name'].dropna() if not _gg.empty else _rc_df['experiment_name'].dropna()
        if not _exp_vals.empty:
            _req_canon_exp[_rc_rid] = _exp_vals.mode().iloc[0]

    for experiment_name, project_df in df.groupby('experiment_name', sort=False):
        # FIX: Filter out Canceled requests that have no real workorders
        # (This prevents 'Ghost' requests from inflating the count)
        project_df = project_df[~(
            (project_df['request_status'].str.upper() == 'CANCELED') &
            (project_df['workorder_id'].astype(str).str.startswith('REQ-'))
        )]
        safe_exp_id = "exp_" + hashlib.md5(experiment_name.encode()).hexdigest()
        now = datetime.now(pytz.timezone('US/Eastern'))
        count_fulfilled = 0; count_canceled = 0; count_new = 0; count_planned = 0
        count_blocked = 0; count_stalled = 0; count_in_lsp = 0; count_asm_review = 0
        count_in_assembly = 0; count_active_waiting = 0; count_ship_ready = 0
        new_req_list = []; active_req_list = []; fulfilled_req_list = []; canceled_req_list = []
        stalled_reqs = set(); asm_review_reqs = set(); production_tats = []; total_tats = []
        has_ptr = project_df['for_partner'].astype(str).str.lower().str.contains('true').any()
        _exp_customer_styles = {
            'R_D':              ('R&D',           '#cffafe', '#0e7490', '#a5f3fc'),
            'INTERNAL_CLD':     ('CLD',           '#dbeafe', '#1d4ed8', '#93c5fd'),
            'TECH_OUT':         ('TECH OUT',      '#ffedd5', '#c2410c', '#fdba74'),
            'EXTERNAL_TECH_OUT':('EXT TECH OUT',  '#fce7f3', '#be185d', '#f9a8d4'),
        }
        _exp_customers = []
        if 'customer' in project_df.columns:
            for _cv in project_df['customer'].dropna().unique():
                _cv = str(_cv)
                if _cv not in ('nan', 'None', '') and _cv not in [c for c, *_ in _exp_customers]:
                    _cl, _cbg, _cfg, _cborder = _exp_customer_styles.get(_cv, (_cv.replace('_', ' '), 'rgba(255,255,255,0.15)', 'rgba(255,255,255,0.9)', 'rgba(255,255,255,0.3)'))
                    _exp_customers.append((_cv, _cl, _cbg, _cfg, _cborder))
        exp_customer_tags = " ".join(
            f'<span style="font-size:9px;font-weight:700;padding:2px 7px;border-radius:3px;background:{bg};color:{fg};border:1px solid {bd};">{lbl}</span>'
            for _, lbl, bg, fg, bd in _exp_customers
        )
        dots_html = ""; stage_counts = {}; stage_items = {}; fulfilled_week_counts = {}
        # Sort requests: newest first, but group base+variant construct names together.
        # Strip trailing _identifier suffix to find the base construct name, then use
        # the newest request_created_at in the base group as the anchor date so variants
        # stay adjacent to their base construct rather than floating by their own date.
        def _base_construct(r_df):
            cn = str(r_df['construct_name'].iloc[0] if len(r_df) > 0 else '')
            return re.sub(r'_[^_()]+$', '', cn).strip()
        def _req_date(r_df):
            d = r_df['request_created_at'].iloc[0] if 'request_created_at' in r_df.columns else None
            return pd.Timestamp(d) if d and pd.notna(d) else pd.Timestamp.min
        _raw_groups = [
            (rid, rdf) for rid, rdf in project_df.groupby('req_id')
            if _req_canon_exp.get(rid, experiment_name) == experiment_name
        ]
        _base_anchor = {}
        for _rid, _rdf in _raw_groups:
            _bc = _base_construct(_rdf)
            _dt = _req_date(_rdf)
            if _bc not in _base_anchor or _dt > _base_anchor[_bc]:
                _base_anchor[_bc] = _dt
        req_groups = sorted(_raw_groups, key=lambda x: (
            -_base_anchor.get(_base_construct(x[1]), pd.Timestamp.min).timestamp(),
            _base_construct(x[1]),
            str(x[1]['construct_name'].iloc[0] if len(x[1]) > 0 else '')
        ))

        for rid, r_df in req_groups:
            r_created = to_est(r_df['request_created_at'].iloc[0]) or to_est(r_df['wo_created_at'].min())
            if not r_created: continue
            status = str(r_df.get('request_status', ['NEW']).iloc[0]).upper()
            is_partner = 'true' in str(r_df.get('for_partner', ['false']).iloc[0]).lower()
            if is_partner: yellow_limit, red_limit = 4.0, 5.0
            else: yellow_limit, red_limit = 5.0, 6.0

            ready_to_ship_time = None; final_release_time = None
            lsp_rows = r_df[r_df['type'] == 'lsp_workorder']
            if not lsp_rows.empty:
                for _, lrow in lsp_rows.iterrows():
                    pn, ps, pt = lrow.get('protocol_name', []), lrow.get('operation_state', []), lrow.get('operation_start', [])
                    if isinstance(pn, np.ndarray): pn = pn.tolist()
                    if isinstance(ps, np.ndarray): ps = ps.tolist()
                    if isinstance(pt, np.ndarray): pt = pt.tolist()
                    if isinstance(pn, list):
                        for name, state, start in zip(pn, ps, pt):
                            if name == 'LSP Reviewing' and state == 'SC': ready_to_ship_time = to_est(start)
                            if name == 'LSP Releasing' and state == 'SC': final_release_time = to_est(start)

            is_finished = status in ['FULFILLED', 'SUCCEEDED']
            if is_finished:
                production_end = ready_to_ship_time or now
                total_end = final_release_time or production_end
                age_weeks = (production_end - r_created).days / 7
                production_tats.append((production_end - r_created).days)
                total_tats.append((total_end - r_created).days)
                dot_color = _age_color(age_weeks, yellow_limit, red_limit, ramp=_BLUE_RAMP if has_ptr else _PURPLE_RAMP)
            else:
                age_weeks = (now - r_created).days / 7
                dot_color = _age_color(age_weeks, yellow_limit, red_limit, ramp=_BLUE_RAMP if has_ptr else _PURPLE_RAMP)
            pos = max(0, min(100, (age_weeks / 8) * 100))

            active_rows = r_df[r_df['wo_status'] != 'CANCELED']
            is_blocked = 'BLOCKED' in active_rows['visual_status'].values
            _draft_mask = r_df['data_source'].eq('BIOS_DRAFT') if 'data_source' in r_df.columns else pd.Series(False, index=r_df.index)
            real_wos = r_df[r_df['workorder_id'].notna() & ~r_df['workorder_id'].astype(str).str.startswith('REQ-') & ~_draft_mask]
            has_real_workorders = not real_wos.empty
            has_life = not active_rows[active_rows['visual_status'].isin(['RUNNING', 'READY', 'IN_PROGRESS', 'LSP_RUNNING', 'WAITING'])].empty
            # After calculating has_life, add root chain check:
            root_chain_types = ['gibson_workorder', 'golden_gate_workorder', 'transformation_workorder',
                                'transformation_offline_operation', 'streakout_operation', 'lsp_workorder']

            root_chain_rows = active_rows[active_rows['type'].isin(root_chain_types)]
            root_chain_finished = root_chain_rows['visual_status'].isin(['SUCCEEDED', 'FAILED', 'CANCELED']).all()
            root_chain_exists = not root_chain_rows.empty

            asm_blocked_and_stuck = (
                root_chain_exists
                and root_chain_rows['visual_status'].isin(['BLOCKED']).any()
                and not active_rows[active_rows['type'] == 'lsp_workorder']['visual_status']
                    .isin(['RUNNING', 'READY', 'IN_PROGRESS']).any()
            )

            # A SUCCEEDED LSP means physical delivery is done — not stalled, just pending BIOS closure.
            # The stall check (root_chain_finished) fires for Gibson→done-but-no-LSP AND for
            # LSP-done-but-no-BIOS-fulfillment; the latter is a false positive.
            lsp_succeeded = (
                not root_chain_rows[root_chain_rows['type'] == 'lsp_workorder'].empty
                and (root_chain_rows[root_chain_rows['type'] == 'lsp_workorder']['visual_status'] == 'SUCCEEDED').any()
            )

            is_stalled = (
                has_real_workorders
                and not is_finished
                and status != 'CANCELED'
                and not lsp_succeeded  # LSP delivered — BIOS just needs to mark FULFILLED
                and (
                    not has_life  # Original condition: no activity anywhere
                    or (root_chain_exists and root_chain_finished)  # root chain dead even if parts running
                    or asm_blocked_and_stuck  # BLOCKED ASM with no active LSP downstream
                )
            )

            # ASM REVIEW: a winner assembly already succeeded for this root, but another
            # GG or Gibson is still READY or WAITING — likely should be canceled.
            _asm_types_flag = {'golden_gate_workorder', 'gibson_workorder'}
            _asm_rows = active_rows[active_rows['type'].isin(_asm_types_flag)]
            _has_succeeded_asm = _asm_rows['visual_status'].eq('SUCCEEDED').any()
            _has_ready_asm     = _asm_rows['visual_status'].isin(['READY', 'WAITING']).any()
            is_asm_review = (
                has_real_workorders
                and not is_finished
                and status != 'CANCELED'
                and _has_succeeded_asm
                and _has_ready_asm
            )

            if not is_finished and status != 'CANCELED':
                _week_bucket = min(int(age_weeks), 8)
                # Build tooltip label: "STOCK_ID: construct_name" for this request
                _tip_cn = str(r_df['construct_name'].iloc[0] if 'construct_name' in r_df.columns and not r_df['construct_name'].dropna().empty else '') or ''
                _tip_sids = r_df[r_df['STOCK_ID'].notna()]['STOCK_ID'].dropna().unique().tolist()
                _tip_sid = str(_tip_sids[0]) if _tip_sids else ''
                _tip_label = f"{_tip_sid}: {_tip_cn}".strip(': ') if (_tip_sid or _tip_cn) else str(rid)[:8]
                if not has_real_workorders:
                    _s = 'In Design'
                    stage_counts.setdefault(_s, {})
                    stage_counts[_s][_week_bucket] = stage_counts[_s].get(_week_bucket, 0) + 1
                    stage_items.setdefault(_s, {}).setdefault(_week_bucket, []).append(_tip_label)
                else:
                    # Use same phase detection as the request badge (all_root_stocks-based)
                    _rsm = {}
                    for _rid2, _rdf2 in r_df.groupby('root_work_order_id'):
                        _rw2 = _rdf2[_rdf2['workorder_id'] == _rid2]['STOCK_ID']
                        _rs2 = str(_rw2.iloc[0]) if not _rw2.empty and pd.notna(_rw2.iloc[0]) else str(_rdf2['STOCK_ID'].iloc[0])
                        if _rs2 not in ('nan', 'None', 'N/A'): _rsm[_rid2] = _rs2
                    _ars = set(_rsm.values())
                    _eff_a = active_rows[active_rows['visual_status'].isin({'RUNNING','READY','WAITING','BLOCKED','IN_PROGRESS','LSP_RUNNING'})]
                    _lsp_s = _eff_a[_eff_a['type'] == 'lsp_workorder']
                    _asm_s = _eff_a[(_eff_a['type'] != 'lsp_workorder') & _eff_a['STOCK_ID'].astype(str).isin(_ars)]
                    _prt_s = _eff_a[(_eff_a['type'] != 'lsp_workorder') & ~_eff_a['STOCK_ID'].astype(str).isin(_ars)]
                    def _ap(row):
                        def _to_list(v):
                            if isinstance(v, (list, np.ndarray)): return list(v)
                            if isinstance(v, pd.Series): return list(v.dropna())
                            if v is None or (isinstance(v, float) and pd.isna(v)): return []
                            return list(v) if v else []
                        pn = _to_list(row.get('protocol_name'))
                        ps = _to_list(row.get('operation_state'))
                        return {p for p, s in zip(pn, ps) if s in ('RD', 'RU')}
                    if is_stalled:
                        _stage = 'Stalled'
                    elif not _lsp_s.empty:
                        _tr = _lsp_s.iloc[0]; _p = _ap(_tr)
                        if 'LSP Releasing' in _p: _stage = 'Releasing'
                        elif 'LSP Reviewing' in _p: _stage = 'Reviewing'
                        elif _p & {'DNA Quantification','NGS Sequence Confirmation','Fragment Analyzer'}: _stage = 'LSP QC'
                        else: _stage = 'LSP'
                    elif not _asm_s.empty:
                        _asm_prog = _asm_s[_asm_s['visual_status'].isin({'RUNNING','READY','IN_PROGRESS','BLOCKED'})]
                        if not _asm_prog.empty:
                            _asm_prog = _asm_prog.copy()
                            _asm_prog['_r'] = _asm_prog['visual_status'].map({'RUNNING':0,'READY':1,'IN_PROGRESS':2,'BLOCKED':3}).fillna(99)
                            _tr = _asm_prog.sort_values('_r').iloc[0]; _p = _ap(_tr)
                            if _p & {'DNA Quantification','NGS Sequence Confirmation'}: _stage = 'Assembly QC'
                            else: _stage = 'Assembly'
                        else:
                            if not _prt_s.empty:
                                _tr = _prt_s.iloc[0]
                                if _tr['type'] == 'pcr_workorder': _stage = 'PCR'
                                elif _tr['type'] == 'plasmid_synthesis_workorder' and not (_tr.get('vendor') and str(_tr.get('vendor')) not in ('nan','None','')): _stage = 'DV/PL1 Build'
                                else: _stage = 'Vendor Parts'
                            else:
                                # WAITING GG, no parts in tree — check global parts lookup
                                # then fall back to backbone lookup in full df.
                                _bb_stage = 'Vendor Parts'
                                _determined = False
                                for _, _aw in _asm_s[_asm_s['visual_status'] == 'WAITING'].iterrows():
                                    # Check parts list first against global lookup
                                    for _fld in ('parts', 'backbone'):
                                        _raw = str(_aw.get(_fld) or '')
                                        for _tok in _raw.split(','):
                                            _psid = _tok.split(':')[0].strip()
                                            if not _psid or _psid in ('nan', 'None', ''): continue
                                            for _gpr in _global_parts_rows_by_stock.get(_psid, []):
                                                _gwt = str(_gpr.get('type', '') or '')
                                                if 'syn_part' in _gwt or 'oligo' in _gwt:
                                                    _bb_stage = 'Vendor Parts'; _determined = True; break
                                                elif 'pcr' in _gwt:
                                                    _bb_stage = 'PCR'; _determined = True; break
                                                elif 'plasmid_synthesis' in _gwt:
                                                    _v = str(_gpr.get('vendor') or '')
                                                    _bb_stage = 'Vendor Parts' if _v not in ('', 'nan', 'None') else 'DV/PL1 Build'
                                                    _determined = True; break
                                            if _determined: break
                                        if _determined: break
                                    if _determined: break
                                    # Backbone lookup via _stock_to_req as secondary check
                                    _bb = str(_aw.get('backbone') or '')
                                    _bb_sid = _bb.split(':')[0].strip()
                                    if _bb_sid and _bb_sid not in ('nan', 'None', ''):
                                        _bb_info = _stock_to_req.get(_bb_sid)
                                        if _bb_info:
                                            _wt = _bb_info.get('wo_type', '')
                                            if 'syn_part' in _wt or 'oligo' in _wt: _bb_stage = 'Vendor Parts'
                                            elif 'plasmid_synthesis' in _wt:
                                                _bb_rows = _global_parts_rows_by_stock.get(_bb_sid, [])
                                                _has_vendor = any(str(_r.get('vendor') or '') not in ('', 'nan', 'None') for _r in _bb_rows)
                                                _bb_stage = 'Vendor Parts' if _has_vendor else 'DV/PL1 Build'
                                            else: _bb_stage = 'DV/PL1 Build'
                                        # else: backbone not being built → default Vendor Parts
                                        break
                                _stage = _bb_stage
                    elif not _prt_s.empty:
                        _tr = _prt_s.iloc[0]
                        if _tr['type'] == 'pcr_workorder': _stage = 'PCR'
                        elif _tr['type'] == 'plasmid_synthesis_workorder' and not (_tr.get('vendor') and str(_tr.get('vendor')) not in ('nan','None','')): _stage = 'DV/PL1 Build'
                        else: _stage = 'Vendor Parts'
                    else: _stage = 'Stalled'
                    stage_counts.setdefault(_stage, {})
                    stage_counts[_stage][_week_bucket] = stage_counts[_stage].get(_week_bucket, 0) + 1
                    stage_items.setdefault(_stage, {}).setdefault(_week_bucket, []).append(_tip_label)

            if is_finished:
                count_fulfilled += 1; fulfilled_req_list.append((rid, r_df))
                _wb = min(int(age_weeks), 8)
                fulfilled_week_counts[_wb] = fulfilled_week_counts.get(_wb, 0) + 1
            elif status == 'CANCELED':
                if has_real_workorders: count_canceled += 1; canceled_req_list.append((rid, r_df))
            elif has_real_workorders or status == 'PLANNED':
                active_req_list.append((rid, r_df))
                if not has_real_workorders: count_planned += 1
                elif is_stalled: count_stalled += 1; stalled_reqs.add(rid)
                if is_asm_review: count_asm_review += 1; asm_review_reqs.add(rid)
                elif is_blocked: count_blocked += 1
                else:
                    is_ship_ready = False
                    lsp_active = active_rows[(active_rows['type'] == 'lsp_workorder') & (active_rows['visual_status'].isin(['RUNNING', 'READY', 'IN_PROGRESS']))]
                    for _, lsp_row in lsp_active.iterrows():
                        pnames = lsp_row.get('protocol_name', []); pstates = lsp_row.get('operation_state', [])
                        if isinstance(pnames, list) and isinstance(pstates, list):
                            for name, state in zip(pnames, pstates):
                                if name == 'LSP Releasing' and state == 'RD': is_ship_ready = True; break
                        if is_ship_ready: break
                    if is_ship_ready: count_ship_ready += 1
                    elif not lsp_active.empty: count_in_lsp += 1
                    elif not active_rows[(active_rows['type'].isin(['golden_gate_workorder', 'gibson_workorder', 'transformation_workorder', 'transformation_offline_operation', 'streakout_operation', 'pcr_workorder'])) & (active_rows['visual_status'].isin(['RUNNING', 'READY', 'IN_PROGRESS']))].empty: count_in_assembly += 1
                    else: count_active_waiting += 1
            else: count_new += 1; new_req_list.append((rid, r_df))

            if is_finished:
                shape_css = "border-radius: 2px; transform: translate(-50%, -50%) rotate(45deg);"
                dot_size, dot_opacity, z_idx, border_css = "12px", "1.0", 20, "border: 2px solid #1e3a5f;"
            elif status == 'CANCELED':
                dot_color = "transparent"
                shape_css = "background: linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.4) 40%, rgba(255,255,255,0.4) 60%, transparent 60%), linear-gradient(-45deg, transparent 40%, rgba(255,255,255,0.4) 40%, rgba(255,255,255,0.4) 60%, transparent 60%); transform: translate(-50%, -50%);"
                dot_size, dot_opacity, z_idx, border_css = "11px", "0.8", 5, "border: none;"
            else:
                shape_css = "border-radius: 50%; transform: translate(-50%, -50%);"
                dot_size, dot_opacity, z_idx, border_css = "14px", "1.0", 30, "border: 2px solid #1e3a5f;"
            v_jitter = random.uniform(-12, 12)
            dots_html += f'''<div style="position:absolute; left:{pos}%; top:{v_jitter}px; width:{dot_size}; height:{dot_size}; background:{dot_color}; {shape_css} {border_css} z-index:{z_idx}; opacity:{dot_opacity};"></div>'''


        avg_tat_html = ""
        if production_tats or total_tats:
            tat_parts = []
            if production_tats:
                avg_f = sum(production_tats) / len(production_tats); weeks, days = int(avg_f//7), int(avg_f%7)
                tat_parts.append(f"<span style='background:rgba(255,255,255,0.2); color:white; padding:2px 6px; border-radius:3px; font-size:10px; white-space:nowrap;'>Avg Production: <span style='color:#67e8f9; font-weight:700;'>{weeks}w {days}d</span></span>")
            if total_tats:
                avg_t = sum(total_tats) / len(total_tats); weeks, days = int(avg_t//7), int(avg_t%7)
                tat_parts.append(f"<span style='background:rgba(255,255,255,0.2); color:white; padding:2px 6px; border-radius:3px; font-size:10px; white-space:nowrap;'>Avg Total: <span style='color:#67e8f9; font-weight:700;'>{weeks}w {days}d</span></span>")
            avg_tat_html = f'''<div style="display:flex; gap:10px; font-weight:700;">{" ".join(tat_parts)}</div>'''

        # ── Experiment creation date (needed for due date marker) ────────────
        exp_created_str = "N/A"
        exp_created_dt = None
        if 'experiment_created_at' in project_df.columns:
            exp_created_raw = project_df['experiment_created_at'].iloc[0]
            exp_created_dt = to_est(exp_created_raw)
            if exp_created_dt:
                exp_created_str = exp_created_dt.strftime('%Y-%m-%d')

        # ── Due date (from Google Sheet / CSV override) ───────────────────────
        _NO_TIMELINE_MARKERS = {"LSP Refill Requests", "A469-Build DNASC CHO Destination Vectors"}
        _is_infra_exp = experiment_name in _NO_TIMELINE_MARKERS
        _due_raw         = None if _is_infra_exp else _due_date_map.get(experiment_name)
        _due_badge_html  = ""
        _due_marker_html = ""
        _sort_due_date   = "9999-99-99"   # default: no due date → sorts to end

        # Normalize due entry — use the first (or only) entry from the sheet
        if _due_raw is None:
            _due_entry_data = None
        elif isinstance(_due_raw, str):
            _due_entry_data = {"due_date": _due_raw, "date_in_cld_gnatt": ""}
        elif isinstance(_due_raw, dict):
            _due_entry_data = _due_raw
        else:
            _due_entry_data = _due_raw[0] if _due_raw else None

        _due_entry = bool(_due_entry_data)   # truthy flag used by bracket/sort logic

        if _due_entry_data:
            try:
                _now_utc      = datetime.now(pytz.UTC)
                _8w_days      = 56.0
                _ec_date      = exp_created_dt.date() if exp_created_dt else None
                def _pos(dt):
                    if not _ec_date: return 0
                    d = dt.date() if hasattr(dt, 'date') else dt
                    return min(100, max(0, (d - _ec_date).days / _8w_days * 100))

                _due_date_str = _due_entry_data.get("due_date", "")
                _gantt_str    = _due_entry_data.get("date_in_cld_gnatt", "")
                if _due_date_str:
                    _due_dt   = datetime.strptime(_due_date_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                    _sort_due_date = _due_date_str
                    _days_remaining = (_due_dt.date() - _now_utc.date()).days
                    if _days_remaining < 0:
                        _dbg, _dfg = "#be185d", "white"
                        _dlabel = f"Overdue by {abs(_days_remaining)}d"
                    elif _days_remaining <= 7:
                        _dbg, _dfg = "#d97706", "white"
                        _dlabel = f"Due in {_days_remaining}d"
                    elif _days_remaining <= 14:
                        _dbg, _dfg = "#0891b2", "white"
                        _dlabel = f"Due in {_days_remaining}d"
                    else:
                        _dbg, _dfg = "rgba(255,255,255,0.18)", "rgba(255,255,255,0.95)"
                        _dlabel = f"Due {_due_dt.strftime('%b %-d')}"
                    _due_badge_html = (
                        f'<span style="font-size:9px;font-weight:700;padding:2px 8px;border-radius:3px;'
                        f'background:{_dbg};color:{_dfg};border:1px solid rgba(255,255,255,0.25);white-space:nowrap;">'
                        f'{_dlabel}</span>'
                    )

                    if exp_created_dt:
                        _ngs_dt    = _due_dt - pd.Timedelta(days=1)
                        _gantt_dt  = datetime.strptime(_gantt_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC) if _gantt_str else None
                        _ngs_pos   = _pos(_ngs_dt)
                        _due_pos   = _pos(_due_dt)
                        _gantt_pos = _pos(_gantt_dt) if _gantt_dt else _due_pos
                        if 0 < _due_pos <= 100:
                            _dr            = (_due_dt.date() - _now_utc.date()).days
                            _urgency_color = "#f87171" if _dr < 0 else "#fcd34d" if _dr <= 7 else "#6ee7b7"
                            _range_width   = max(0.5, _gantt_pos - _ngs_pos)
                            _ngs_label     = _ngs_dt.strftime("%a %b %-d")
                            _due_label     = _due_dt.strftime("%a %b %-d")
                            _gantt_label   = _gantt_dt.strftime("%a %-m/%-d") if _gantt_dt else ""
                            _pop_id        = f"duepop_{safe_exp_id}"
                            _pill_text     = f"DUE {_due_dt.strftime('%a %-m/%-d')}"
                            _due_marker_html = (
                                # Range bar
                                f'<div style="position:absolute;left:{_ngs_pos:.2f}%;top:0;'
                                f'width:{_range_width:.2f}%;height:100%;'
                                f'background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.4);'
                                f'border-radius:3px;z-index:1;pointer-events:none;"></div>'
                                # Vertical line
                                f'<div style="position:absolute;left:{_due_pos:.2f}%;top:-3px;'
                                f'width:4px;height:28px;background:white;border-radius:1px;'
                                f'box-shadow:0 0 8px rgba(0,0,0,0.5);z-index:4;transform:translateX(-50%);pointer-events:none;"></div>'
                                # Hover wrapper + pill
                                f'<div style="position:absolute;left:{_due_pos:.2f}%;top:-74px;'
                                f'width:90px;height:72px;transform:translateX(-50%);z-index:25;cursor:pointer;"'
                                f' onmouseenter="document.getElementById(\'{_pop_id}\').style.display=\'block\'"'
                                f' onmouseleave="document.getElementById(\'{_pop_id}\').style.display=\'none\'">'
                                f'<div style="position:absolute;left:50%;top:0;transform:translateX(-50%);'
                                f'background:white;color:#1e1b4b;font-size:9px;font-weight:700;'
                                f'padding:2px 5px;border-radius:3px;white-space:nowrap;letter-spacing:0.02em;'
                                f'box-shadow:0 1px 5px rgba(0,0,0,0.4);">{_pill_text}</div>'
                                # Popover
                                f'<div id="{_pop_id}" style="display:none;position:absolute;left:38px;top:8px;'
                                f'background:#1e1b4b;color:white;font-size:10px;padding:8px 11px;border-radius:5px;'
                                f'white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,0.7);z-index:100;border:1px solid rgba(255,255,255,0.3);">'
                                f'<div style="font-weight:800;color:white;margin-bottom:5px;font-size:11px;">Due Date</div>'
                                f'<div style="display:grid;grid-template-columns:72px 1fr;gap:2px 8px;">'
                                f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Last NGS</span>'
                                f'<span style="font-weight:600;">{_ngs_label}</span>'
                                f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Due</span>'
                                f'<span style="font-weight:700;color:{_urgency_color};">{_due_label}</span>'
                                + (f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">CLD Gantt</span>'
                                   f'<span style="font-weight:600;">{_gantt_label}</span>' if _gantt_dt else '') +
                                f'</div></div></div>'
                            )
            except Exception:
                pass

        # ── Default NGS window bracket + light-purple DUE flag (no CLD override)
        # Bracket: white semi-transparent, Last NGS → red threshold.
        # Purple DUE pill at day-after-last-NGS (aligned with TAT threshold, based on exp_created_dt).
        _default_bracket_html = ""
        if not _due_entry and exp_created_dt and not _is_infra_exp:
            try:
                from datetime import timedelta as _td2
                _tat_weeks = 5 if has_ptr else 6
                _red_thresh_dt = exp_created_dt + _td2(weeks=_tat_weeks)
                def _last_ngs_b(dt):
                    for i in range(7):
                        c = dt - _td2(days=i)
                        if c.weekday() in (0, 3): return c
                    return dt
                _def_ngs_dt  = _last_ngs_b(_red_thresh_dt - _td2(days=1))
                _def_due_dt  = _def_ngs_dt + _td2(days=1)   # day after last NGS = standard due
                _sort_due_date = _def_due_dt.strftime("%Y-%m-%d")  # ISO for sort
                _8w = 56.0
                _ec_date2 = exp_created_dt.date()
                def _pos2(dt):
                    d = dt.date() if hasattr(dt, 'date') else dt
                    return min(100, max(0, (d - _ec_date2).days / _8w * 100))
                _def_ngs_pos = _pos2(_def_ngs_dt)
                _def_red_pos = _pos2(_red_thresh_dt)
                _def_due_pos = _pos2(_def_due_dt)
                _def_width   = max(0.5, _def_red_pos - _def_ngs_pos)
                _def_pop_id  = f"defduepop_{safe_exp_id}"
                _def_ngs_str = _def_ngs_dt.strftime("%a %b %-d")
                _def_due_str = _def_due_dt.strftime("%a %-m/%-d")
                _def_red_str = _red_thresh_dt.strftime("%a %b %-d")
                _default_bracket_html = (
                    # White semi-transparent bracket: NGS → red threshold
                    f'<div style="position:absolute;left:{_def_ngs_pos:.2f}%;top:0;'
                    f'width:{_def_width:.2f}%;height:100%;'
                    f'background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.45);'
                    f'border-radius:3px;z-index:1;pointer-events:none;"></div>'
                    + (
                    # Light-purple vertical line at day-after-NGS
                    f'<div style="position:absolute;left:{_def_due_pos:.2f}%;top:-3px;'
                    f'width:4px;height:28px;background:#c4b5fd;border-radius:1px;'
                    f'box-shadow:0 0 8px rgba(167,139,250,0.7);z-index:4;transform:translateX(-50%);pointer-events:none;"></div>'
                    # Hover wrapper: pill + popover
                    f'<div style="position:absolute;left:{_def_due_pos:.2f}%;top:-74px;'
                    f'width:80px;height:72px;transform:translateX(-50%);z-index:25;cursor:pointer;"'
                    f' onmouseenter="document.getElementById(\'{_def_pop_id}\').style.display=\'block\'"'
                    f' onmouseleave="document.getElementById(\'{_def_pop_id}\').style.display=\'none\'">'
                    # Light-purple DUE pill
                    f'<div style="position:absolute;left:50%;top:0;transform:translateX(-50%);'
                    f'background:#7c3aed;color:white;font-size:9px;font-weight:700;'
                    f'padding:2px 5px;border-radius:3px;white-space:nowrap;letter-spacing:0.02em;'
                    f'box-shadow:0 1px 5px rgba(0,0,0,0.4);border:1px solid #c4b5fd;">'
                    f'DUE {_def_due_str}</div>'
                    # Popover
                    f'<div id="{_def_pop_id}" style="display:none;position:absolute;left:38px;top:8px;'
                    f'background:#1e1b4b;color:white;font-size:10px;padding:8px 11px;border-radius:5px;'
                    f'white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,0.7);z-index:100;border:1px solid #7c3aed;">'
                    f'<div style="font-weight:800;color:white;margin-bottom:5px;font-size:11px;">Standard TAT</div>'
                    f'<div style="display:grid;grid-template-columns:80px 1fr;gap:2px 8px;">'
                    f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Last NGS</span>'
                    f'<span style="font-weight:600;">{_def_ngs_str}</span>'
                    f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Due</span>'
                    f'<span style="font-weight:600;color:#c4b5fd;">{_def_due_str}</span>'
                    f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Threshold</span>'
                    f'<span style="font-weight:600;color:#f87171;">{_def_red_str}</span>'
                    f'</div></div></div>'
                )
                )
            except Exception:
                pass

        orange_week = 4 if has_ptr else 5; red_week = 5 if has_ptr else 6
        from datetime import timedelta as _td
        def _last_ngs_before(dt):
            """Most recent Monday (0) or Thursday (3) on or before dt."""
            for i in range(7):
                cand = dt - _td(days=i)
                if cand.weekday() in (0, 3):
                    return cand
            return dt
        _is_refill = _is_infra_exp
        def _threshold_html(week, color, glow, pop_id, show_popover=True, label_top=-52):
            if not exp_created_dt or _is_refill:
                return ""
            _thresh_dt = exp_created_dt + _td(weeks=week)
            _left      = f"{(week/8)*100}%"
            _thresh_str = _thresh_dt.strftime("%a %-m/%-d")
            _bar = (f'<div style="position:absolute;left:{_left};width:2px;height:28px;background:{color};'
                    f'top:-3px;border-radius:1px;box-shadow:0 0 6px {glow};z-index:2;"></div>')
            if show_popover:
                _ngs_dt    = _last_ngs_before(_thresh_dt - _td(days=1))
                _ngs_str   = _ngs_dt.strftime("%a %b %-d")
                _thresh_full = _thresh_dt.strftime("%a %b %-d")
                _label = (
                    f'<div style="position:absolute;left:{_left};top:{label_top}px;transform:translateX(-50%);'
                    f'background:{color};color:white;font-size:9px;font-weight:700;'
                    f'padding:2px 5px;border-radius:3px;white-space:nowrap;letter-spacing:0.02em;'
                    f'box-shadow:0 1px 5px rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.6);'
                    f'z-index:20;cursor:pointer;"'
                    f' onmouseenter="document.getElementById(\'{pop_id}\').style.display=\'block\'"'
                    f' onmouseleave="document.getElementById(\'{pop_id}\').style.display=\'none\'">'
                    f'{_thresh_str}'
                    f'<div id="{pop_id}" style="display:none;position:absolute;left:50%;top:22px;transform:translateX(-50%);'
                    f'background:#1e1b4b;color:white;font-size:10px;padding:8px 11px;border-radius:5px;'
                    f'white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,0.7);z-index:100;border:1px solid {color};font-weight:400;">'
                    f'<div style="font-weight:800;color:white;margin-bottom:5px;font-size:11px;">{int(week)}w Threshold</div>'
                    f'<div style="display:grid;grid-template-columns:80px 1fr;gap:2px 8px;">'
                    f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Last NGS</span>'
                    f'<span style="font-weight:600;">{_ngs_str}</span>'
                    f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Threshold</span>'
                    f'<span style="font-weight:600;color:{color};">{_thresh_full}</span>'
                    f'</div></div></div>'
                )
            else:
                # No popover — plain pill only
                _label = (
                    f'<div style="position:absolute;left:{_left};top:{label_top}px;transform:translateX(-50%);'
                    f'background:{color};color:white;font-size:9px;font-weight:700;'
                    f'padding:2px 5px;border-radius:3px;white-space:nowrap;letter-spacing:0.02em;'
                    f'box-shadow:0 1px 5px rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.6);z-index:20;">'
                    f'{_thresh_str}</div>'
                )
            return _label + _bar
        _orange_html = _threshold_html(orange_week, "#f97316", "rgba(249,115,22,0.6)", f"tpop_o_{safe_exp_id}", show_popover=False)
        _red_html    = _threshold_html(red_week,    "#be185d", "rgba(190,24,93,0.6)",  f"tpop_r_{safe_exp_id}", show_popover=True)
        if _is_refill:
            _default_bracket_html = ""
            _due_marker_html = ""
        timeline_bar = f"""<div style="margin: 10px 12px 8px 12px; padding: 10px; background: rgba(0,0,0,0.15); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);"><div style="display:flex; justify-content:space-between; font-size:11px; color:rgba(255,255,255,1); margin-bottom:8px; font-family:monospace; font-weight:900; letter-spacing:1px; text-shadow: 0 1px 3px rgba(0,0,0,0.4);"><span>START</span><span>1w</span><span>2w</span><span>3w</span><span>4w</span><span>5w</span><span>6w</span><span>7w</span><span>8w+</span></div><div style="position:relative; width:100%; height:22px; background:rgba(255,255,255,0.15); border-radius:11px; box-shadow: inset 0 1px 4px rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.2);">{" ".join([f'<div style="position:absolute; left:{(w/8)*100}%; width:1px; height:100%; background:rgba(255,255,255,0.1); z-index:1;"></div>' for w in range(1,8)])}{_orange_html}{_red_html}<div style="position:absolute; width:100%; height:100%; top:50%; left:0; z-index:10;">{dots_html}</div>{_default_bracket_html}{_due_marker_html}</div>
            <div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap; margin-top:10px; padding: 8px 12px; background: rgba(0,0,0,0.2); border-radius: 6px;">
              <div style="display:flex; align-items:center; gap:8px; color:white; font-size:9px; font-weight:600;">
                  <span style="color:rgba(255,255,255,0.7);">IN PROGRESS:</span>
                  <span style="display:flex; align-items:center; gap:4px;"><span style="width:8px; height:8px; background:#0891b2; border-radius:50%; border:1px solid #1e3a5f;"></span> On Track</span>
                  <span style="display:flex; align-items:center; gap:4px;"><span style="width:8px; height:8px; background:#f97316; border-radius:50%; border:1px solid #1e3a5f;"></span> Warning</span>
                  <span style="display:flex; align-items:center; gap:4px;"><span style="width:8px; height:8px; background:#be185d; border-radius:50%; border:1px solid #1e3a5f;"></span> Overdue</span>
              </div>
              <div style="width:1px; background:rgba(255,255,255,0.3);"></div>
              <div style="display:flex; align-items:center; gap:8px; color:white; font-size:9px; font-weight:600;">
                  <span style="color:rgba(255,255,255,0.7);">FULFILLED:</span>
                  <span style="display:flex; align-items:center; gap:4px;"><span style="width:8px; height:8px; background:#0891b2; border-radius:2px; transform:rotate(45deg); border:1px solid #1e3a5f;"></span> On Time</span>
                  <span style="display:flex; align-items:center; gap:4px;"><span style="width:8px; height:8px; background:#f97316; border-radius:2px; transform:rotate(45deg); border:1px solid #1e3a5f;"></span> Warning</span>
                  <span style="display:flex; align-items:center; gap:4px;"><span style="width:8px; height:8px; background:#be185d; border-radius:2px; transform:rotate(45deg); border:1px solid #1e3a5f;"></span> Late</span>
              </div>
          </div>
        </div>"""

        db_active = str(project_df.get('experiment_active', ['true']).iloc[0]).lower() in ['true', '1', 't']
        exp_header_gradient = "linear-gradient(135deg, #7c3aed 0%, #be185d 100%)" if has_ptr else "linear-gradient(135deg, #1e3a5f 0%, #0891b2 100%)"

        _exp_emails_raw = [str(e).strip() for e in project_df['submitter_email'].dropna().unique() if str(e).strip() not in ('', 'nan', 'none', 'None')] if 'submitter_email' in project_df.columns else []
        _exp_email_str = ' / '.join(_exp_emails_raw[:2]) if 1 <= len(_exp_emails_raw) <= 2 else ''

        html += f"""
            <div class="project-wrapper" data-active="{"true" if db_active else "false"}" data-due-date="{_sort_due_date}">
                <div class="header-banner" style="background: {exp_header_gradient}; min-height: auto; padding: 12px 18px;" onclick="toggleSection('{safe_exp_id}')">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                        <button id="bucket_btn_{safe_exp_id}" onclick="event.stopPropagation();toggleBucketView('{safe_exp_id}')" style="margin-left:auto;order:99;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.35);color:white;font-size:9px;padding:3px 9px;border-radius:4px;cursor:pointer;font-family:monospace;font-weight:600;letter-spacing:0.5px;white-space:nowrap;flex-shrink:0;">Stage View</button>
                        <div>
                            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                                <div class="header-title" style="margin-bottom: 0; white-space: nowrap;">{experiment_name}</div>
                                {exp_customer_tags}
                            </div>
                            <div style="font-size: 11px; color: rgba(255,255,255,0.75); font-weight: 400; margin-top: 3px; font-family: inherit; letter-spacing: 0;">
                                <span style="font-size: 10px; color: rgba(255,255,255,0.55);">Created: {exp_created_str}</span>{(f' &nbsp;<span style="color:#e0e7ff; font-weight:600; font-size:13px;">' + _exp_email_str + '</span>') if _exp_email_str else ''}
                            </div>
                        </div>
                        <div class="header-main-stat" style="margin-bottom: 0; font-size: 11px;">
                            <span style="color:#67e8f9; font-weight:700;">{len(req_groups)}</span> Requests:
                            <span style="color:#67e8f9; font-weight:700;">{count_fulfilled}</span> Fulfilled
                        </div>
                        <div style="font-size: 1px;">{avg_tat_html}</div>
                    </div>
                    <div style="margin-bottom:10px; margin-top:52px;">
                        <div id="timeline_{safe_exp_id}">{timeline_bar}</div>
                        <div id="bucket_{safe_exp_id}" style="display:none;background:rgba(0,0,0,0.15);border-radius:8px;border:1px solid rgba(255,255,255,0.1);">{_render_bucket_chart(stage_counts, fulfilled_week_counts, orange_week, red_week, stage_items, ramp=_BLUE_RAMP if has_ptr else _PURPLE_RAMP)}</div>
                    </div>
                    <div class="header-stats" style="margin-top: 0; display: flex; gap: 6px; flex-wrap: wrap;">
                        {f'<span class="stat-item" style="background:rgba(217,119,6,0.6); border:1px solid rgba(255,255,255,0.4);"><span class="stat-label" style="font-size:11px;">{count_new}</span> <span style="font-size:10px;">New</span></span>' if count_new > 0 else ''}
                        {f'<span class="stat-item" style="background:rgba(34,197,94,0.4); border:1px solid rgba(255,255,255,0.3);"><span class="stat-label" style="font-size:11px;">{count_ship_ready}</span> <span style="font-size:10px;">Ship Ready</span></span>' if count_ship_ready > 0 else ''}
                        {f'<span class="stat-item" style="background:rgba(8,145,178,0.4); border:1px solid rgba(255,255,255,0.3);"><span class="stat-label" style="font-size:11px;">{count_in_lsp}</span> <span style="font-size:10px;">In LSP</span></span>' if count_in_lsp > 0 else ''}
                        {f'<span class="stat-item" style="background:rgba(124,58,237,0.4); border:1px solid rgba(255,255,255,0.3);"><span class="stat-label" style="font-size:11px;">{count_in_assembly}</span> <span style="font-size:10px;">In Assembly</span></span>' if count_in_assembly > 0 else ''}
                        {f'<span class="stat-item" style="background:rgba(249,115,22,0.4); border:1px solid rgba(255,255,255,0.3);"><span class="stat-label" style="font-size:11px;">{count_active_waiting}</span> <span style="font-size:10px;">Waiting</span></span>' if count_active_waiting > 0 else ''}
                        {f'<span class="stat-item" style="background:rgba(190,24,93,0.5); border:1px solid rgba(255,255,255,0.3);"><span class="stat-label" style="font-size:11px;">⚠️ {count_stalled}</span> <span style="font-size:10px;">Stalled</span></span>' if count_stalled > 0 else ''}
                        {f'<span class="stat-item" style="background:rgba(190,24,93,0.5); border:1px solid rgba(255,255,255,0.3);"><span class="stat-label" style="font-size:11px;">{count_blocked}</span> <span style="font-size:10px;">Blocked</span></span>' if count_blocked > 0 else ''}
                        {f'<span class="stat-item" style="background:rgba(100,116,139,0.4); border:1px solid rgba(255,255,255,0.3);"><span class="stat-label" style="font-size:11px;">{count_canceled}</span> <span style="font-size:10px;">Canceled</span></span>' if count_canceled > 0 else ''}
                    </div>
                </div>"""

        if active_req_list:
            html += f"""
                <details>
                    <summary class="group-header in-progress">
                        <span class="group-arrow">▶</span> Planned / In-Progress ({len(active_req_list)}{f' - ⚠️ {count_stalled} Stalled' if count_stalled > 0 else ''}{f' - 🔬 {count_asm_review} ASM Review' if count_asm_review > 0 else ''})
                    </summary>"""
            for rid, r_df in active_req_list:
                is_stalled_req = rid in stalled_reqs
                is_asm_review_req = rid in asm_review_reqs
                req_html = render_single_request_html(rid, r_df, is_stalled_req, is_asm_review_req)
                html += req_html
            html += "</details>"
        if new_req_list:
            html += f'<details><summary class="group-header new"><span class="group-arrow">▶</span> New ({len(new_req_list)})</summary>'
            for rid, r_df in new_req_list: html += render_single_request_html(rid, r_df, False)
            html += "</details>"
        if fulfilled_req_list:
            html += f'<details><summary class="group-header fulfilled"><span class="group-arrow">▶</span> Fulfilled ({len(fulfilled_req_list)})</summary>'
            for rid, r_df in fulfilled_req_list: html += render_single_request_html(rid, r_df, False)
            html += "</details>"
        if canceled_req_list:
            html += f'<details><summary class="group-header canceled"><span class="group-arrow">▶</span> Canceled ({len(canceled_req_list)})</summary>'
            for rid, r_df in canceled_req_list:
                html += render_single_request_html(rid, r_df, False)
            html += "</details>"

        html += "</div>"

    html += """
                </div>
            </div>

            <!-- METRICS TAB -->
            <div id="tab-metrics" class="tab-content">
                <div class="under-construction">
                    <div class="uc-icon">📊</div>
                    <div class="uc-title">Metrics Dashboard</div>
                    <div class="uc-subtitle">
                        Advanced analytics and performance metrics are coming soon.
                        Track throughput, success rates, cycle times, and more.
                    </div>
                    <div class="uc-badge">🚧 Under Construction</div>
                </div>
            </div>

            <!-- COSTS TAB -->
            <div id="tab-costs" class="tab-content">
                <div class="under-construction">
                    <div class="uc-icon">💰</div>
                    <div class="uc-title">Cost Analysis</div>
                    <div class="uc-subtitle">
                        Detailed cost breakdowns and financial analytics are coming soon.
                        Track reagent costs, labor hours, and project budgets.
                    </div>
                    <div class="uc-badge">🚧 Under Construction</div>
                </div>
            </div>

            <!-- LSP CAPACITY TAB -->
            <div id="tab-capacity" class="tab-content">
                __LSP_CAPACITY_TAB_CONTENT__
            </div>

        </div>
    </div>
    """
    html = html.replace("__LSP_CAPACITY_TAB_CONTENT__", lsp_capacity_html)
    return html


# ── Public API ────────────────────────────────────────────────────────────────

def render_dashboard(df: pd.DataFrame) -> str:
    """
    Render the full dashboard HTML for `df`.
    Assets are loaded automatically from the scripts/ directory.
    Injects a 10-minute browser auto-refresh meta tag.
    """
    if not all(_ASSETS.values()):
        raise FileNotFoundError(
            "One or more dashboard assets are missing. "
            f"Ensure all PNG files exist in: {_SCRIPTS_DIR}"
        )

    log.info("Rendering dashboard for %d rows...", len(df))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = render_all_projects_dashboard(
        df,
        logo_b64          = _ASSETS["logo"],
        tracking_icon_b64 = _ASSETS["tracking"],
        metrics_icon_b64  = _ASSETS["metrics"],
        cost_icon_b64     = _ASSETS["cost"],
        generated_at      = generated_at,
    )

    # Wrap in full HTML document with UTF-8 charset + auto-refresh
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>DNA SC Dashboard</title>
    <script>
    // Poll dnasc_version.txt every 2 minutes. When the cron job updates the
    // dashboard the timestamp changes and we hard-reload automatically — no
    // manual refresh needed.
    (function() {{
        var _loadedTs = null;
        function _checkVersion() {{
            fetch('dnasc_version.txt?_=' + Date.now(), {{cache: 'no-store'}})
                .then(function(r) {{ return r.text(); }})
                .then(function(ts) {{
                    ts = ts.trim();
                    if (_loadedTs === null) {{ _loadedTs = ts; }}
                    else if (ts !== _loadedTs) {{
                        // reload(true) is ignored by modern browsers — navigate
                        // with the new timestamp as a cache-busting query param
                        window.location.href = window.location.pathname + '?v=' + ts;
                    }}
                }})
                .catch(function() {{}});
        }}
        setInterval(_checkVersion, 120000);
        _checkVersion();
    }})();
    </script>
</head>
<body>
{html}
</body>
</html>"""

    log.info("Dashboard rendered successfully")
    return html

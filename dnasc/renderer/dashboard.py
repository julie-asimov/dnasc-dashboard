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

    def parse_pipeline_operations(protocol_names, operation_states, operation_starts, job_ids, well_locations_list, operation_ready_times):
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

        if not protocol_names:
            return []

        # --- STEP 1: BUILD RAW OPERATIONS ---
        raw_ops = []
        for i in range(len(protocol_names)):
            # Safe access using index checks
            r_time = to_est(operation_ready_times[i]) if i < len(operation_ready_times) else None
            s_time = to_est(operation_starts[i]) if i < len(operation_starts) else None

            raw_ops.append({
                'protocol': protocol_names[i],
                'state': operation_states[i] if i < len(operation_states) else 'Unknown',
                'start_time': s_time,
                'ready_time': r_time,
                'job_id': job_ids[i] if i < len(job_ids) else None,
                'well_location': well_locations_list[i] if i < len(well_locations_list) else None
            })

        # --- STEP 2: SORTING ---
        # Sort by ready_time then start_time
        raw_ops.sort(key=lambda x: (
            0 if pd.notna(x['ready_time']) else 1,
            x['ready_time'] if pd.notna(x['ready_time']) else pd.Timestamp.min,
            0 if pd.notna(x['start_time']) else 1,
            x['start_time'] if pd.notna(x['start_time']) else pd.Timestamp.min
        ))

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

            # Skip Fails if a Success exists for the same protocol
            if state == 'FA' and protocol in protocol_success:
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
                'wells': [op['well_location']] if pd.notna(op['well_location']) else []
            }

            # Grouping Engine
            if not current_group:
                current_group.append(clean_op)
            else:
                prev_op = current_group[-1]
                if protocol == prev_op['queue'] and protocol in groupable_protocols:
                    # Merge wells and update state if necessary
                    prev_op['wells'].extend(clean_op['wells'])
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
            row.get('job_id', []), row.get('well_location', []), row.get('operation_ready', [])
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

    for _, row in df.iterrows():
        wid = str(row['workorder_id'])
        _jid = row.get('job_id')
        if isinstance(_jid, np.ndarray): _jid = _jid.tolist()
        job_id = _jid[0] if isinstance(_jid, list) and len(_jid) > 0 else None
        completion_time = None
        op_states, op_starts = row.get('operation_state'), row.get('operation_start')
        if isinstance(op_states, np.ndarray): op_states = op_states.tolist()
        if isinstance(op_starts, np.ndarray): op_starts = op_starts.tolist()
        if isinstance(op_states, list) and isinstance(op_starts, list):
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
                            # Colony progress ran but LIMS shows 0 seq confirmed → failure
                            # Use LIMS data (seq_confirmed) directly — don't require NGS in OpTracker
                            if tot > 0 and seq == 0:
                                return 'FAILED', False

        return original, False

    df[['visual_status', 'is_software_fail']] = df.apply(lambda row: pd.Series(get_visual_status(row)), axis=1)

    status_priority = {'RUNNING': 0, 'IN_PROGRESS': 0, 'BLOCKED': 1, 'WAITING': 2, 'READY': 2, 'SUCCEEDED': 3, 'FAILED': 4, 'CANCELED': 5}
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
    .tab-btn.active { background: rgba(255,255,255,0.2); }
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
    .status-SUCCEEDED, .status-FULFILLED { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    .status-FAILED    { background: #fff1f5; color: #be185d; border: 1px solid #fecdd3; }
    .status-CANCELED  { background: #f5f5f7; color: #6b7280; border: 1px solid #d1d5db; }
    .status-RUNNING   { background: #f5f3ff; color: #7c3aed; border: 1px solid #ddd6fe; }
    .status-LSP_RUNNING { background: #fdf4ff; color: #a21caf; border: 1px solid #f0abfc; }
    .status-IN_PROGRESS { background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe; }
    .status-READY     { background: #f0fdfa; color: #0d9488; border: 1px solid #99f6e4; }
    .status-WAITING   { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
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
    .part-tag.missing { background: #fdf2f8; color: #be185d; border: 1px solid #f9a8d4; font-weight: 700; }
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
    </style>

    <script>
    function switchTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.querySelector('[data-tab="' + tabName + '"]').classList.add('active');
        document.getElementById('tab-' + tabName).classList.add('active');
    }
    function toggleSection(id) {
        var el = document.getElementById(id); var icon = document.getElementById(id + '_icon'); var btn = document.getElementById(id + '_btn');
        if (el.style.display === "block") { el.style.display = "none"; if(icon) icon.classList.remove('open'); if(btn) btn.classList.remove('active-header'); }
        else { el.style.display = "block"; if(icon) icon.classList.add('open'); if(btn) btn.classList.add('active-header'); }
    }
    function filterDashboard() {
        var searchTerm = document.getElementById('search_box').value.toLowerCase();
        var activeOnly = document.getElementById('active_toggle').checked;
        try { localStorage.setItem('dash_activeOnly', activeOnly ? '1' : '0'); } catch(e) {}
        try { localStorage.setItem('dash_search', document.getElementById('search_box').value); } catch(e) {}
        var projects = document.getElementsByClassName('project-wrapper');
        for (var i = 0; i < projects.length; i++) {
            var project = projects[i];
            var isActive = project.getAttribute('data-active') === 'true';
            if (activeOnly && !isActive) { project.style.display = 'none'; continue; }
            if (searchTerm) {
                var projectText = project.textContent.toLowerCase();
                var reqCards = project.getElementsByClassName('req-card');
                var hasMatch = false;
                if (projectText.includes(searchTerm)) { hasMatch = true; }
                if (!hasMatch && reqCards.length > 0) {
                    for (var j = 0; j < reqCards.length; j++) {
                        var cardText = reqCards[j].textContent.toLowerCase();
                        if (cardText.includes(searchTerm)) { hasMatch = true; break; }
                    }
                }
                project.style.display = hasMatch ? 'block' : 'none';
            } else { project.style.display = 'block'; }
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
            </div>
            <!-- TRACKING TAB -->
            <div id="tab-tracking" class="tab-content active">
                <div class="controls-container">
                    <div class="toggle-wrapper" style="margin-right: auto;">
                        <input type="text" id="search_box" placeholder="Search Stock ID, Experiment, or Construct..." oninput="filterDashboard()">
                    </div>
                    <div class="toggle-wrapper">
                        <span class="toggle-label">Active Projects Only</span>
                        <label class="switch">
                            <input type="checkbox" id="active_toggle" onclick="filterDashboard()">
                            <span class="slider"></span>
                        </label>
                    </div>
                </div>
                <div style="padding: 10px;">
    """

    # =========================================================================
    # 3. HELPER: RENDER SINGLE REQUEST
    # =========================================================================
    def render_single_request_html(req_id, req_df, is_stalled=False):
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

        # --- UPDATED CONTEXT-AWARE STATUS SELECTION ---
        active_rows = req_df[req_df['visual_status'].isin(['RUNNING', 'READY', 'WAITING', 'BLOCKED', 'IN_PROGRESS'])]
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
                asm_progressing = asm_active[asm_active['visual_status'].isin(_progressing)]
                if not asm_progressing.empty:
                    asm_priority = {'RUNNING': 0, 'READY': 1, 'IN_PROGRESS': 2, 'WAITING': 3, 'BLOCKED': 4}
                    asm_active = asm_active.copy()
                    asm_active['_rank'] = asm_active['visual_status'].map(asm_priority).fillna(99)
                    target_row = asm_active.sort_values('_rank').iloc[0]
                    phase_label = "ASM"
                    phase_bg = "#dbeafe"
                    phase_color = "#1d4ed8"
                    phase_border = "#bfdbfe"
                else:
                    # All ASM WAITING — show most urgent part, fall back to waiting ASM
                    if not parts_active.empty:
                        parts_active = parts_active.copy()
                        parts_active['_rank'] = parts_active['visual_status'].map(_parts_priority).fillna(99)
                        target_row = parts_active.sort_values('_rank').iloc[0]
                    else:
                        target_row = asm_active.iloc[0]
                    phase_label = "PARTS"
                    phase_bg = "#ffedd5"
                    phase_color = "#c2410c"
                    phase_border = "#fed7aa"
            elif not parts_active.empty:
                parts_active = parts_active.copy()
                parts_active['_rank'] = parts_active['visual_status'].map(_parts_priority).fillna(99)
                target_row = parts_active.sort_values('_rank').iloc[0]
                phase_label = "PARTS"
                phase_bg = "#ffedd5"
                phase_color = "#c2410c"
                phase_border = "#fed7aa"
            else:
                target_row = active_rows.iloc[0]

            if target_row is not None:
                status = target_row['visual_status']
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
                    display_text = f"{running_step}: {status}".upper() if running_step else status.upper()

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
        now = datetime.now(pytz.timezone('US/Eastern'))
        is_done = str(req_status).upper() in ['FULFILLED', 'SUCCEEDED', 'CANCELED']

        stalled_badge = '<span class="badge" style="background:#be185d; color:white; border:2px solid #9f1239; font-size:12px; padding:4px 12px; font-weight:800;">⚠️ STALLED</span>' if is_stalled else ""

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
                    <div style="margin-top: 4px; color: #94a3b8; font-size: 10px; font-family: monospace; letter-spacing: -0.2px;">REQ ID: {req_id}</div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    {partner_badge}
                    <span class="badge status-{str(req_status).replace(" ", "_")}" style="font-size: 10px; padding: 2px 8px;">{req_status}</span>
                    {status_badge_html}
                    {stalled_badge}
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
        for root_id, r_df in req_df.groupby('root_work_order_id'):
            is_winner = False
            if is_req_fulfilled:
                if r_df['fulfills_request'].any() or r_df['wo_status'].isin(['SUCCEEDED', 'FULFILLED']).any():
                    is_winner = True; has_winner = True
            status_priority = {'RUNNING': 0, 'IN_PROGRESS': 0, 'BLOCKED': 1, 'WAITING': 2, 'READY': 2, 'SUCCEEDED': 3, 'FAILED': 4, 'CANCELED': 5}
            r_df['local_rank'] = r_df['visual_status'].map(status_priority).fillna(99)
            min_rank = r_df['local_rank'].min()
            root_status_map[root_id] = {'is_winner': is_winner, 'rank': min_rank}
        sorted_roots = sorted(root_status_map.keys(), key=lambda r: (not root_status_map[r]['is_winner'], root_status_map[r]['rank']))

        for root_id in sorted_roots:
            root_df = req_df[req_df['root_work_order_id'] == root_id]
            is_this_winner = root_status_map[root_id]['is_winner']
            section_class = "assembly-section"
            if is_req_fulfilled and has_winner and not is_this_winner: section_class += " dimmed"
            row_map = {row['workorder_id']: row for _, row in root_df.iterrows()}
            adj = defaultdict(list)
            roots_in_view = []

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
            ordered_data = []
            def dfs(node_id, depth):
                if node_id in row_map:
                    row_data = row_map[node_id]; row_data['tree_depth'] = depth; ordered_data.append(row_data)
                children = sorted(adj[node_id], key=lambda x: (0 if row_map[x]['type'] == 'transformation_workorder' else 1))
                for child in children: dfs(child, depth + 1)
            for r in roots_in_view: dfs(r, 0)
            sorted_root_df = pd.DataFrame(ordered_data)

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
            if target_row is None: target_row = sorted_root_df.iloc[0] if not sorted_root_df.empty else root_df.iloc[0]
            status = target_row['visual_status']
            # If the root assembly failed but a downstream streakout recovered
            # colonies, show SUCCEEDED in the header.  Only check streakout_operation
            # rows — PCR/Oligo/Syn are upstream inputs, not recovery outcomes, so
            # their SUCCEEDED status must not override the assembly result.
            # Also never applies when the header target is an LSP.
            if status == 'FAILED' and target_row['type'] != 'lsp_workorder':
                recovery_types = {'streakout_operation', 'transformation_offline_operation'}
                recovery_rows = root_df[root_df['type'].isin(recovery_types)]
                if not recovery_rows.empty and (recovery_rows['visual_status'] == 'SUCCEEDED').any():
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
            if pd.isna(root_stock): root_stock = "N/A"
            assembly_types = [format_type_label(t) for t in ['golden_gate_workorder', 'gibson_workorder'] if (root_df['type'] == t).any()]
            assembly_label_text = ' + '.join(assembly_types) if assembly_types else 'Workflow'
            assembly_label = f"<b>{assembly_label_text}</b>"
            div_id = f"group_{req_id.replace('-', '_')}_{root_id.replace('-', '_')}"
            type_counts = root_df['type'].value_counts()
            count_str = ", ".join([f"{count} {format_type_label(k).split()[0].upper()}" for k, count in type_counts.items()])

            html += f"""<div class="{section_class}"><button id="{div_id}_btn" class="dropdown-btn" onclick="toggleSection('{div_id}')"><span id="{div_id}_icon" class="dropdown-icon">▶</span><div class="assembly-info"><span class="assembly-type">{assembly_label}</span><span class="stock-tag" style="font-size:9px; padding:3px 8px; background:#ede9fe; color:#6d28d9; border: 1px solid #c4b5fd;">{root_stock}</span><span class="assembly-counts" style="font-weight: 600;">{count_str}</span><span class="wo-id-tag">Root: {root_id}</span></div><div class="status-badges">{badges_html}</div></button><div id="{div_id}" class="content-pane"><table class="wo-table"><thead><tr><th>Type</th><th>Workorder ID</th><th>Status</th><th>Stock ID</th><th>Created</th><th>TAT</th><th>Details</th><th style="width: 350px;">Queue</th></tr></thead><tbody>"""

            for _, row in sorted_root_df.iterrows():
                # Skip CANCELED workorders that never ran (no queue data) —
                # these are abandoned attempts that clutter the timeline.
                if row.get('wo_status') == 'CANCELED':
                    _pn = row.get('protocol_name')
                    if hasattr(_pn, 'tolist'):
                        _pn = _pn.tolist()
                    if not isinstance(_pn, list):
                        _pn = []
                    if not _pn:
                        continue
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
                queue_data = parse_pipeline_operations(row.get('protocol_name', []), row.get('operation_state', []), row.get('operation_start', []), row.get('job_id', []), row.get('well_location', []), row.get('operation_ready', []))
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
                else: pipeline_html += '<span style="color: #9ca3af; font-size: 11px;">No queue data</span>'
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
                        if clean_name in tree_stock_ids: return f'<span class="part-tag in-production" title="Being made in this workflow">{label_prefix}{clean_name}</span>'
                        else: return f'<span class="part-tag missing" title="Missing material">{label_prefix}{clean_name}</span>'
                    else: return f'<span class="part-tag">{label_prefix}{clean_name}</span>'
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
                    lsp_parts = [f'<div style="color: #4b5563; font-size: 10px; font-weight: 700; margin-bottom: 4px;">{display_lsp_id}</div>']


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
                    lsp_parts.append(f"""
                        <div class="plate-hover-container" style="display: inline-block; margin-bottom: 6px;">
                            <span class="plate-trigger" style="background: #e5e7eb; color: #4b5563; cursor: pointer; font-size: 9px; font-weight: 600; padding: 2px 6px; border-radius: 3px; border: 1px solid #d1d5db;">
                                Source Info
                            </span>
                            <div class="plate-popover" style="width: 450px; white-space: normal; padding: 15px; border-top: 4px solid #6b7280; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                                <div style="border-bottom: 1px solid #e5e7eb; margin-bottom: 10px; padding-bottom: 5px; font-weight: 800; color: #4b5563; text-transform: uppercase; font-size: 11px;">
                                    Source Material Context
                                </div>
                                <div style="display: grid; grid-template-columns: 100px 1fr; gap: 8px; font-size: 12px; line-height: 1.5; color: #1f2937;">
                                    <span style="color: #6b7280; font-size: 10px; font-weight: 700; text-transform: uppercase;">Experiment:</span>
                                    <span>{exp_name}</span>
                                    <span style="color: #6b7280; font-size: 10px; font-weight: 700; text-transform: uppercase;">Construct:</span>
                                    <span>{construct_name}</span>
                                    <span style="color: #6b7280; font-size: 10px; font-weight: 700; text-transform: uppercase;">Stock ID:</span>
                                    <span style="font-family: monospace; color: #4b5563; font-weight: 700;">{pai_val}</span>
                                    <span style="color: #6b7280; font-size: 10px; font-weight: 700; text-transform: uppercase;">Process ID:</span>
                                    <a href="{'https://op-tracker.asimov.io/job/' + proc_id.replace('job__', '') + '/group/0/step/0/' if proc_id.startswith('job__') else 'https://op-tracker.asimov.io/workorder/' + proc_id}" target="_blank" style="font-family: monospace; font-size: 11px; color: #4b5563; background: #f3f4f6; padding: 2px 6px; border-radius: 3px; text-decoration: underline; border: 1px solid #d1d5db;">
                                        {'Job ' + proc_id.replace('job__', '') if proc_id.startswith('job__') else proc_id}
                                    </a>
                                </div>
                            </div>
                        </div>""")
                    metrics_list = []
                    strain = row.get('comp_cell') or row.get('cloning_strain')
                    conc = row.get('qubit_concentration_ngul')
                    yld = row.get('qubit_yield')
                    if pd.notna(strain) and str(strain) != 'nan': metrics_list.append(("Cell", str(strain)))
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
                    input_well = row.get('lsp_input_well')
                    if pd.isna(input_well) or str(input_well) == 'nan':
                        input_well = row.get('input_well_id')
                    if pd.notna(input_well) and str(input_well) != 'nan':
                        well_id_match = re.search(r'"id":\s*(\d+)', str(input_well))
                        final_id = well_id_match.group(1) if well_id_match else str(input_well)
                        lsp_parts.append(f'<div style="font-size: 11px; margin-top: 4px;"><b style="color: #475569;">Input:</b> <a href="https://bios.asimov.io/inventory/wells/{final_id}" target="_blank" style="color: #7c3aed; text-decoration: underline; font-weight: 700;">well{final_id}</a></div>')
                    details_info += "".join(lsp_parts)

                elif row['type'] in ['golden_gate_workorder', 'gibson_workorder', 'transformation_workorder', 'transformation_offline_operation', 'streakout_operation']:
                    strain = row.get('cloning_strain')
                    if pd.notna(strain): details_info += f"<div style='font-size:10px;color:#64748b;margin-top:2px;'>Strain: {strain}</div>"
                    if row['is_software_fail']: details_info += f'<br><div style="font-size:9px;color:#be185d;margin-top:2px;font-weight:bold;">Software: FAILED</div>'
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
        count_blocked = 0; count_stalled = 0; count_in_lsp = 0
        count_in_assembly = 0; count_active_waiting = 0; count_ship_ready = 0
        new_req_list = []; active_req_list = []; fulfilled_req_list = []; canceled_req_list = []
        stalled_reqs = set(); production_tats = []; total_tats = []
        has_ptr = project_df['for_partner'].astype(str).str.lower().str.contains('true').any()
        dots_html = ""; req_groups = list(project_df.groupby('req_id'))

        for rid, r_df in req_groups:
            r_created = to_est(r_df['request_created_at'].iloc[0]) or to_est(r_df['wo_created_at'].min())
            if not r_created: continue
            status = str(r_df.get('request_status', ['NEW']).iloc[0]).upper()
            is_partner = 'true' in str(r_df.get('for_partner', ['false']).iloc[0]).lower()
            if is_partner: yellow_limit, red_limit = 5.0, 6.0
            else: yellow_limit, red_limit = 6.0, 7.0

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
                if age_weeks < yellow_limit: dot_color = "#0891b2"
                elif age_weeks < red_limit: dot_color = "#f97316"
                else: dot_color = "#be185d"
            else:
                age_weeks = (now - r_created).days / 7
                if age_weeks >= red_limit: dot_color = "#be185d"
                elif age_weeks >= yellow_limit: dot_color = "#f97316"
                else: dot_color = "#0891b2"
            pos = max(0, min(100, (age_weeks / 8) * 100))

            active_rows = r_df[r_df['wo_status'] != 'CANCELED']
            is_blocked = 'BLOCKED' in active_rows['visual_status'].values
            real_wos = r_df[r_df['workorder_id'].notna() & ~r_df['workorder_id'].astype(str).str.startswith('REQ-')]
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

            is_stalled = (
                has_real_workorders
                and not is_finished
                and status != 'CANCELED'
                and (
                    not has_life  # Original condition: no activity anywhere
                    or (root_chain_exists and root_chain_finished)  # root chain dead even if parts running
                    or asm_blocked_and_stuck  # BLOCKED ASM with no active LSP downstream
                )
            )

            if is_finished: count_fulfilled += 1; fulfilled_req_list.append((rid, r_df))
            elif status == 'CANCELED':
                if has_real_workorders: count_canceled += 1; canceled_req_list.append((rid, r_df))
            elif has_real_workorders or status == 'PLANNED':
                active_req_list.append((rid, r_df))
                if not has_real_workorders: count_planned += 1
                elif is_stalled: count_stalled += 1; stalled_reqs.add(rid)
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

        orange_week = 5 if has_ptr else 6; red_week = 6 if has_ptr else 7
        timeline_bar = f"""<div style="margin: 10px 12px 8px 12px; padding: 10px; background: rgba(0,0,0,0.15); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);"><div style="display:flex; justify-content:space-between; font-size:11px; color:rgba(255,255,255,1); margin-bottom:8px; font-family:monospace; font-weight:900; letter-spacing:1px; text-shadow: 0 1px 3px rgba(0,0,0,0.4);"><span>START</span><span>1w</span><span>2w</span><span>3w</span><span>4w</span><span>5w</span><span>6w</span><span>7w</span><span>8w+</span></div><div style="position:relative; width:100%; height:22px; background:rgba(255,255,255,0.15); border-radius:11px; box-shadow: inset 0 1px 4px rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.2);">{" ".join([f'<div style="position:absolute; left:{(w/8)*100}%; width:1px; height:100%; background:rgba(255,255,255,0.1); z-index:1;"></div>' for w in range(1,8)])}<div style="position:absolute; left:{(orange_week/8)*100}%; width:2px; height:28px; background:#f97316; top:-3px; border-radius:1px; box-shadow: 0 0 6px rgba(249,115,22,0.6); z-index:2;"></div><div style="position:absolute; left:{(red_week/8)*100}%; width:2px; height:28px; background:#be185d; top:-3px; border-radius:1px; box-shadow: 0 0 6px rgba(190,24,93,0.6); z-index:2;"></div><div style="position:absolute; width:100%; height:100%; top:50%; left:0; z-index:10;">{dots_html}</div></div>
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

        exp_created_str = "N/A"
        if 'experiment_created_at' in project_df.columns:
            exp_created_raw = project_df['experiment_created_at'].iloc[0]
            exp_created_dt = to_est(exp_created_raw)
            if exp_created_dt:
                exp_created_str = exp_created_dt.strftime('%Y-%m-%d')

        html += f"""
            <div class="project-wrapper" data-active="{"true" if db_active else "false"}">
                <div class="header-banner" style="background: {exp_header_gradient}; min-height: auto; padding: 12px 18px;" onclick="toggleSection('{safe_exp_id}')">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                        <div>
                            <div class="header-title" style="margin-bottom: 0; white-space: nowrap;">{experiment_name}</div>
                            <div style="font-size: 9px; color: rgba(255,255,255,0.6); font-weight: 500; margin-top: 2px; font-family: monospace;">
                                Created: {exp_created_str}
                            </div>
                        </div>
                        <div class="header-main-stat" style="margin-bottom: 0; font-size: 11px;">
                            <span style="color:#67e8f9; font-weight:700;">{len(req_groups)}</span> Requests:
                            <span style="color:#67e8f9; font-weight:700;">{count_fulfilled}</span> Fulfilled
                        </div>
                        <div style="font-size: 1px;">{avg_tat_html}</div>
                    </div>
                    <div style="margin-bottom: 10px;">{timeline_bar}</div>
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
                        <span class="group-arrow">▶</span> Planned / In-Progress ({len(active_req_list)}{f' - ⚠️ {count_stalled} Stalled' if count_stalled > 0 else ''})
                    </summary>"""
            for rid, r_df in active_req_list:
                is_stalled_req = rid in stalled_reqs
                req_html = render_single_request_html(rid, r_df, is_stalled_req)
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

        </div>
    </div>
    """
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
    <meta http-equiv="refresh" content="3600">
    <title>DNA SC Dashboard</title>
</head>
<body>
{html}
</body>
</html>"""

    log.info("Dashboard rendered successfully")
    return html

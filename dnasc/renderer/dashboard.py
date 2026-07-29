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
import io
import os
import time
import warnings
from pathlib import Path

import re
import json
import random
import hashlib
import urllib.parse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger
from dnasc import protocols as proto
from dnasc.renderer import tokens as tok

# Omni plate maps: LIMS position is 0-indexed; add 1 to get the 1-indexed key.
# All formats are column-major (positions fill down each column before moving right).
_WELL_MAP_96 = {
    '1':'A1','2':'B1','3':'C1','4':'D1','5':'E1','6':'F1','7':'G1','8':'H1',
    '9':'A2','10':'B2','11':'C2','12':'D2','13':'E2','14':'F2','15':'G2','16':'H2',
    '17':'A3','18':'B3','19':'C3','20':'D3','21':'E3','22':'F3','23':'G3','24':'H3',
    '25':'A4','26':'B4','27':'C4','28':'D4','29':'E4','30':'F4','31':'G4','32':'H4',
    '33':'A5','34':'B5','35':'C5','36':'D5','37':'E5','38':'F5','39':'G5','40':'H5',
    '41':'A6','42':'B6','43':'C6','44':'D6','45':'E6','46':'F6','47':'G6','48':'H6',
    '49':'A7','50':'B7','51':'C7','52':'D7','53':'E7','54':'F7','55':'G7','56':'H7',
    '57':'A8','58':'B8','59':'C8','60':'D8','61':'E8','62':'F8','63':'G8','64':'H8',
    '65':'A9','66':'B9','67':'C9','68':'D9','69':'E9','70':'F9','71':'G9','72':'H9',
    '73':'A10','74':'B10','75':'C10','76':'D10','77':'E10','78':'F10','79':'G10','80':'H10',
    '81':'A11','82':'B11','83':'C11','84':'D11','85':'E11','86':'F11','87':'G11','88':'H11',
    '89':'A12','90':'B12','91':'C12','92':'D12','93':'E12','94':'F12','95':'G12','96':'H12',
}
_WELL_MAP_384 = {
    '1':'A1','2':'B1','3':'C1','4':'D1','5':'E1','6':'F1','7':'G1','8':'H1',
    '9':'I1','10':'J1','11':'K1','12':'L1','13':'M1','14':'N1','15':'O1','16':'P1',
    '17':'A2','18':'B2','19':'C2','20':'D2','21':'E2','22':'F2','23':'G2','24':'H2',
    '25':'I2','26':'J2','27':'K2','28':'L2','29':'M2','30':'N2','31':'O2','32':'P2',
    '33':'A3','34':'B3','35':'C3','36':'D3','37':'E3','38':'F3','39':'G3','40':'H3',
    '41':'I3','42':'J3','43':'K3','44':'L3','45':'M3','46':'N3','47':'O3','48':'P3',
    '49':'A4','50':'B4','51':'C4','52':'D4','53':'E4','54':'F4','55':'G4','56':'H4',
    '57':'I4','58':'J4','59':'K4','60':'L4','61':'M4','62':'N4','63':'O4','64':'P4',
    '65':'A5','66':'B5','67':'C5','68':'D5','69':'E5','70':'F5','71':'G5','72':'H5',
    '73':'I5','74':'J5','75':'K5','76':'L5','77':'M5','78':'N5','79':'O5','80':'P5',
    '81':'A6','82':'B6','83':'C6','84':'D6','85':'E6','86':'F6','87':'G6','88':'H6',
    '89':'I6','90':'J6','91':'K6','92':'L6','93':'M6','94':'N6','95':'O6','96':'P6',
    '97':'A7','98':'B7','99':'C7','100':'D7','101':'E7','102':'F7','103':'G7','104':'H7',
    '105':'I7','106':'J7','107':'K7','108':'L7','109':'M7','110':'N7','111':'O7','112':'P7',
    '113':'A8','114':'B8','115':'C8','116':'D8','117':'E8','118':'F8','119':'G8','120':'H8',
    '121':'I8','122':'J8','123':'K8','124':'L8','125':'M8','126':'N8','127':'O8','128':'P8',
    '129':'A9','130':'B9','131':'C9','132':'D9','133':'E9','134':'F9','135':'G9','136':'H9',
    '137':'I9','138':'J9','139':'K9','140':'L9','141':'M9','142':'N9','143':'O9','144':'P9',
    '145':'A10','146':'B10','147':'C10','148':'D10','149':'E10','150':'F10','151':'G10','152':'H10',
    '153':'I10','154':'J10','155':'K10','156':'L10','157':'M10','158':'N10','159':'O10','160':'P10',
    '161':'A11','162':'B11','163':'C11','164':'D11','165':'E11','166':'F11','167':'G11','168':'H11',
    '169':'I11','170':'J11','171':'K11','172':'L11','173':'M11','174':'N11','175':'O11','176':'P11',
    '177':'A12','178':'B12','179':'C12','180':'D12','181':'E12','182':'F12','183':'G12','184':'H12',
    '185':'I12','186':'J12','187':'K12','188':'L12','189':'M12','190':'N12','191':'O12','192':'P12',
    '193':'A13','194':'B13','195':'C13','196':'D13','197':'E13','198':'F13','199':'G13','200':'H13',
    '201':'I13','202':'J13','203':'K13','204':'L13','205':'M13','206':'N13','207':'O13','208':'P13',
    '209':'A14','210':'B14','211':'C14','212':'D14','213':'E14','214':'F14','215':'G14','216':'H14',
    '217':'I14','218':'J14','219':'K14','220':'L14','221':'M14','222':'N14','223':'O14','224':'P14',
    '225':'A15','226':'B15','227':'C15','228':'D15','229':'E15','230':'F15','231':'G15','232':'H15',
    '233':'I15','234':'J15','235':'K15','236':'L15','237':'M15','238':'N15','239':'O15','240':'P15',
    '241':'A16','242':'B16','243':'C16','244':'D16','245':'E16','246':'F16','247':'G16','248':'H16',
    '249':'I16','250':'J16','251':'K16','252':'L16','253':'M16','254':'N16','255':'O16','256':'P16',
    '257':'A17','258':'B17','259':'C17','260':'D17','261':'E17','262':'F17','263':'G17','264':'H17',
    '265':'I17','266':'J17','267':'K17','268':'L17','269':'M17','270':'N17','271':'O17','272':'P17',
    '273':'A18','274':'B18','275':'C18','276':'D18','277':'E18','278':'F18','279':'G18','280':'H18',
    '281':'I18','282':'J18','283':'K18','284':'L18','285':'M18','286':'N18','287':'O18','288':'P18',
    '289':'A19','290':'B19','291':'C19','292':'D19','293':'E19','294':'F19','295':'G19','296':'H19',
    '297':'I19','298':'J19','299':'K19','300':'L19','301':'M19','302':'N19','303':'O19','304':'P19',
    '305':'A20','306':'B20','307':'C20','308':'D20','309':'E20','310':'F20','311':'G20','312':'H20',
    '313':'I20','314':'J20','315':'K20','316':'L20','317':'M20','318':'N20','319':'O20','320':'P20',
    '321':'A21','322':'B21','323':'C21','324':'D21','325':'E21','326':'F21','327':'G21','328':'H21',
    '329':'I21','330':'J21','331':'K21','332':'L21','333':'M21','334':'N21','335':'O21','336':'P21',
    '337':'A22','338':'B22','339':'C22','340':'D22','341':'E22','342':'F22','343':'G22','344':'H22',
    '345':'I22','346':'J22','347':'K22','348':'L22','349':'M22','350':'N22','351':'O22','352':'P22',
    '353':'A23','354':'B23','355':'C23','356':'D23','357':'E23','358':'F23','359':'G23','360':'H23',
    '361':'I23','362':'J23','363':'K23','364':'L23','365':'M23','366':'N23','367':'O23','368':'P23',
    '369':'A24','370':'B24','371':'C24','372':'D24','373':'E24','374':'F24','375':'G24','376':'H24',
    '377':'I24','378':'J24','379':'K24','380':'L24','381':'M24','382':'N24','383':'O24','384':'P24',
}
_WELL_MAP_AGAR = {'1':'A1','2':'B1','3':'A2','4':'B2','5':'A3','6':'B3','7':'A4','8':'B4'}
try:
    from dnasc.renderer.lsp_capacity import render_lsp_capacity_tab
except (ImportError, SyntaxError):
    # SyntaxError too: lsp_capacity.py is an optional, separately-developed tab.
    # A syntax error there (e.g. a 3.12-only f-string on the 3.9 server) must
    # degrade to the stub, not crash the whole dashboard render.
    def render_lsp_capacity_tab(_df):
        return "<p style='color:#6b7280;padding:1rem;'>LSP capacity view not available.</p>"

try:
    from dnasc.renderer.inflight import render_inflight_tab
except ImportError:
    def render_inflight_tab(_df):
        return "<p style='color:#6b7280;padding:1rem;'>Requests in flight view not available.</p>"

try:
    from dnasc.renderer.parts import render_parts_tab
except (ImportError, SyntaxError):
    # Optional, separately-developed tab with its own data pull (parts_result.pkl).
    # Degrade to a stub rather than crash the whole dashboard render.
    def render_parts_tab():
        return "<p style='color:#6b7280;padding:1rem;'>Parts inventory view not available.</p>"

log = get_logger(__name__)

# ── Asset resolution ──────────────────────────────────────────────────────────
# Assets live in scripts/ (two levels up from this file: renderer/ → dnasc/ → scripts/)
_SCRIPTS_DIR = Path(__file__).parent.parent.parent.resolve()

_ASSET_FILES = {
    # Full-res source images — drop in any resolution; _load_b64 downsizes to
    # _ASSET_MAX_PX at build time (icons display at 24-36px, logo at 32px).
    "logo":     "dnasc_logo.png",
    "tracking": "tracking_icon.png",
    "metrics":  "metrics_icon.png",
    "cost":     "cost_icon.png",
}

# Tab icons display at 24-36px and the logo at 32px; ~128px covers 3-4x retina.
# Source art can be any size — it's downsized here so we never embed oversized PNGs.
_ASSET_MAX_PX = 128

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None


def _load_b64(filename: str) -> str:
    path = _SCRIPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Asset not found: {path}\n"
            f"Place '{filename}' in: {_SCRIPTS_DIR}"
        )
    # Downsize to display resolution and emit PNG (with alpha). Falls back to the
    # raw bytes if Pillow is unavailable or the image can't be processed.
    if _PILImage is not None:
        try:
            im = _PILImage.open(path).convert("RGBA")
            im.thumbnail((_ASSET_MAX_PX, _ASSET_MAX_PX), _PILImage.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:  # pragma: no cover - defensive fallback
            warnings.warn(f"Could not downsize asset {filename}: {e}; embedding raw bytes")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# Pre-load at import time — fast for repeated render calls
try:
    _ASSETS = {k: _load_b64(v) for k, v in _ASSET_FILES.items()}
except FileNotFoundError as e:
    warnings.warn(str(e))
    _ASSETS = {k: "" for k in _ASSET_FILES}


# ── Module-level helpers (extracted from render_all_projects_dashboard) ───────

def to_est(dt_val):
    if pd.isna(dt_val) or dt_val == '': return None
    try:
        dt = pd.to_datetime(dt_val)
        if dt.tz is None: dt = dt.tz_localize('UTC')
        return dt.tz_convert('US/Eastern')
    except: return None


# Render-wide timestamp cache: raw value (str-normalized) → US/Eastern Timestamp.
# Primed once per render by prime_est_cache() over every op-time cell in the frame,
# so batch_to_est becomes O(1) dict lookups instead of constructing a DatetimeIndex
# per call (the renderer makes ~110K such calls — the dominant render-time cost).
_EST_CACHE: dict = {}
_EST_MISS = object()


def _est_key(t) -> str:
    """Canonical cache key. Op timestamps reach batch_to_est as numpy.datetime64
    (str uses 'T') from raw cells AND as python datetime (str uses ' ') from
    .tolist()'d cells — normalize the separator so both forms collide."""
    return str(t).replace('T', ' ')


def prime_est_cache(df: pd.DataFrame,
                    columns=("operation_start", "operation_ready")) -> None:
    """Convert every distinct timestamp in the given list-valued columns once and
    store it str-keyed in _EST_CACHE. Subsequent batch_to_est() calls are lookups."""
    _EST_CACHE.clear()
    vals: set = set()
    for col in columns:
        if col not in df.columns:
            continue
        for cell in df[col].values:
            if cell is None:
                continue
            if isinstance(cell, np.ndarray):
                it = cell.tolist()
            elif isinstance(cell, (list, tuple)):
                it = cell
            else:
                it = (cell,)
            for x in it:
                if x is None:
                    continue
                try:
                    if pd.isna(x):
                        continue
                except (TypeError, ValueError):
                    pass
                vals.add(_est_key(x))
    if not vals:
        return
    keys = list(vals)
    converted = pd.to_datetime(keys, errors="coerce", utc=True).tz_convert("US/Eastern")
    for k, v in zip(keys, converted):
        _EST_CACHE[k] = (v if not pd.isnull(v) else None)


def batch_to_est(times: list) -> list:
    """List-of-timestamps → US/Eastern. Uses the primed _EST_CACHE when available
    (O(1) lookups); otherwise falls back to a single vectorized conversion."""
    if not times:
        return []
    if not _EST_CACHE:
        # Cache not primed (e.g. called outside a full render) — original path.
        try:
            converted = pd.to_datetime(times, errors='coerce', utc=True).tz_convert('US/Eastern')
            return [x if not pd.isnull(x) else None for x in converted]
        except Exception:
            return [to_est(t) for t in times]
    out = []
    for t in times:
        if t is None:
            out.append(None)
            continue
        k = _est_key(t)
        v = _EST_CACHE.get(k, _EST_MISS)
        if v is _EST_MISS:
            # Value not seen during priming — convert once and memoize (self-heals).
            v = to_est(t)
            _EST_CACHE[k] = v
        out.append(v)
    return out


def parse_pipeline_operations(protocol_names, operation_states, operation_starts, job_ids, well_locations_list, operation_ready_times, ngs_run_numbers=None):
    # --- STEP 0: TYPE PROTECTION ---
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
    n = len(protocol_names)
    _s_times = batch_to_est(operation_starts[:n])
    _r_times = batch_to_est(operation_ready_times[:n])
    raw_ops = []
    for i in range(n):
        r_time = _r_times[i] if i < len(_r_times) else None
        s_time = _s_times[i] if i < len(_s_times) else None
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
    raw_ops.sort(key=lambda x: (
        0 if pd.notna(x['ready_time']) else 1,
        x['ready_time'] if pd.notna(x['ready_time']) else pd.Timestamp.min,
        0 if pd.notna(x['start_time']) else 1,
        x['start_time'] if pd.notna(x['start_time']) else pd.Timestamp.min
    ))

    # --- STEP 2b: DEDUP same-protocol same-state same-job ops ---
    def _safe_job_key(j):
        try:
            return None if (j is None or pd.isna(j)) else int(j)
        except (TypeError, ValueError):
            return None

    seen: dict = {}
    deduped: list = []
    for op in raw_ops:
        key = (op['protocol'], op['state'], _safe_job_key(op['job_id']))
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(op)
        else:
            existing = deduped[seen[key]]
            if op['start_time'] and (not existing['start_time'] or op['start_time'] < existing['start_time']):
                existing['start_time'] = op['start_time']
    raw_ops = deduped

    # --- STEP 2c: PROPAGATE run_number across states for same (protocol, job) ---
    _run_by_job: dict = {}
    for op in raw_ops:
        key = (op['protocol'], _safe_job_key(op['job_id']))
        if op.get('run_number') is not None and key not in _run_by_job:
            _run_by_job[key] = op['run_number']
    for op in raw_ops:
        if op.get('run_number') is None:
            op['run_number'] = _run_by_job.get((op['protocol'], _safe_job_key(op['job_id'])))

    # --- STEP 3: SOFT FAIL & GROUPING ---
    protocol_success = {op['protocol'] for op in raw_ops if op['state'] == 'SC'}
    groupable_protocols = {
        proto.DNA_QUANT, proto.REARRAY, proto.NGS, proto.FRAGMENT_ANALYZER,
    }

    result = []
    current_group = []

    for op in raw_ops:
        protocol = op['protocol']
        state = op['state']

        if state in ('FA', 'CA') and protocol in protocol_success:
            continue

        state_map = {
            'SC': {'state': 'Completed', 'class': 'succeeded'},
            'FA': {'state': 'Failed', 'class': 'failed'},
            'RU': {'state': 'Running', 'class': 'running'},
            'RD': {'state': 'Ready', 'class': 'ready'},
            'CA': {'state': 'Canceled', 'class': 'canceled'},
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

        def _job_null(j):
            try: return j is None or bool(pd.isna(j))
            except TypeError: return False
        if not current_group:
            current_group.append(clean_op)
        else:
            prev_op = current_group[-1]
            same_job = (_job_null(clean_op['job_id']) and _job_null(prev_op['job_id'])) or \
                       (not _job_null(clean_op['job_id']) and not _job_null(prev_op['job_id']) and clean_op['job_id'] == prev_op['job_id'])
            if protocol == prev_op['queue'] and protocol in groupable_protocols and same_job:
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
    experiment_active_map: dict | None = None,
    out_fh=None,
) -> str:

    if df.empty:
        if out_fh is not None:
            out_fh.write("<h3>No data found.</h3>")
            return ""
        return "<h3>No data found.</h3>"

    # Convert every op timestamp in the frame to US/Eastern once up front so the
    # ~110K downstream batch_to_est() calls are O(1) cache lookups, not per-call
    # DatetimeIndex construction.
    prime_est_cache(df)

    # =========================================================================
    # 1. HELPERS
    # =========================================================================
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
    _wtr_df = df[df['root_work_order_id'].notna()].copy()
    _wtr_df['_search'] = (
        _wtr_df.get('all_locations', pd.Series('', index=_wtr_df.index)).fillna('').astype(str) + ' ' +
        _wtr_df.get('operation_well_locations', pd.Series('', index=_wtr_df.index)).fillna('').astype(str)
    )
    _wtr_df['_wells'] = _wtr_df['_search'].str.findall(r'\b(\d{5,8})\b')
    _wtr_exploded = _wtr_df[_wtr_df['_wells'].str.len() > 0][['_wells', 'root_work_order_id']].explode('_wells')
    well_to_root = dict(zip(_wtr_exploded['_wells'], _wtr_exploded['root_work_order_id']))

    _global_parts_types = {'oligo_synthesis_workorder', 'pcr_workorder',
                           'plasmid_synthesis_workorder', 'syn_part_synthesis_workorder'}
    _global_asm_types = {'gibson_workorder', 'golden_gate_workorder'}

    # Stock-keyed lookup for stage-summary type/vendor checks.
    _global_parts_rows_by_stock: dict = {}
    for _gpr in df[df['type'].isin(_global_parts_types | _global_asm_types)].to_dict('records'):
        if _gpr.get('type') in _global_asm_types and _gpr.get('fulfills_request') != False:
            continue
        _gsid = str(_gpr.get('STOCK_ID', '') or '').strip()
        if _gsid and _gsid not in ('nan', 'None', ''):
            _global_parts_rows_by_stock.setdefault(_gsid, []).append(_gpr)

    _ASM_PROTO_MAP = {'golden_gate_workorder': proto.GOLDEN_GATE, 'gibson_workorder': proto.GIBSON}
    for row in df.to_dict('records'):
        wid = str(row['workorder_id'])
        _jid = row.get('job_id')
        if isinstance(_jid, np.ndarray): _jid = _jid.tolist()
        job_id = None
        completion_time = None
        op_states = row.get('operation_state')
        op_starts = row.get('operation_start')
        op_protocols = row.get('protocol_name')
        if isinstance(op_protocols, np.ndarray): op_protocols = op_protocols.tolist()
        if isinstance(_jid, list) and isinstance(op_protocols, list):
            _asm_proto = _ASM_PROTO_MAP.get(str(row.get('type', '')))
            if _asm_proto:
                job_id = next((j for proto, j in zip(op_protocols, _jid) if proto == _asm_proto and j is not None and pd.notna(j)), None)
            if job_id is None:
                job_id = next((j for j in _jid if j is not None and pd.notna(j)), None)
        if isinstance(op_states, np.ndarray): op_states = op_states.tolist()
        if isinstance(op_starts, np.ndarray): op_starts = op_starts.tolist()
        if isinstance(op_states, list) and isinstance(op_starts, list):
            _target_proto = _ASM_PROTO_MAP.get(str(row.get('type', '')))
            if _target_proto and isinstance(op_protocols, list):
                sc_fa_starts = next(
                    ([start] for proto, state, start in zip(op_protocols, op_states, op_starts)
                     if proto == _target_proto and state in ('SC', 'FA') and pd.notna(start)),
                    [s for st, s in zip(op_states, op_starts) if st in ('SC', 'FA') and pd.notna(s)]
                )
            else:
                sc_fa_starts = [s for st, s in zip(op_states, op_starts) if st in ('SC', 'FA') and pd.notna(s)]
            valid_times = [t for t in batch_to_est(sc_fa_starts) if t is not None]
            if valid_times: completion_time = max(valid_times)
        if not completion_time: completion_time = to_est(row['wo_created_at'])
        plate_id = None
        json_str = row.get('all_protocol_plates', '{}')
        if pd.notna(json_str) and json_str != '{}':
            try:
                data = json.loads(json_str)
                for _pn in [proto.GOLDEN_GATE, proto.GIBSON, proto.PCR, proto.DNA_QUANT]:
                    if _pn in data:
                        raw = str(data[_pn]).split(',')[0]; clean = clean_plate_id(raw)
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

    # visual_status and is_software_fail are pre-computed by the pipeline
    # (_bridge_status + _apply_colony_status_overrides) — read directly from parquet.
    if 'is_software_fail' not in df.columns:
        df['is_software_fail'] = False

    # status_rank / req_rank / group_rank pre-computed by pipeline enrichment step;
    # recompute here only as a fallback for parquets written before v1.8.47.
    if 'status_rank' not in df.columns:
        status_priority = {'RUNNING': 0, 'IN_PROGRESS': 0, 'BLOCKED': 1, 'WAITING': 2, 'READY': 2, 'DRAFT': 2, 'SUCCEEDED': 3, 'FAILED': 4, 'CANCELED': 5}
        df['status_rank'] = df['visual_status'].map(status_priority).fillna(99)
        df['req_rank']    = df.groupby('req_id')['status_rank'].transform('min')
        df['group_rank']  = df.groupby('root_work_order_id')['status_rank'].transform('min')

    # is_visible: CANCELED rows with no protocol data are hidden
    canceled = df['wo_status'] == 'CANCELED'
    has_protocol = df['protocol_name'].map(
        lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0 and pd.notna(v[0] if isinstance(v, list) else v.flat[0])
    )
    df['is_visible'] = ~canceled | has_protocol

    if 'experiment_name' not in df.columns: df['experiment_name'] = "Unknown Project"
    if 'experiment_created_at' in df.columns:
        df = df.sort_values(by=['experiment_created_at', 'req_rank', 'req_id', 'group_rank'], ascending=[False, True, True, True])
    else:
        df = df.sort_values(by=['req_rank', 'req_id', 'group_rank'])

    # Convert high-cardinality string columns to categoricals so that inner-loop
    # .isin() and == comparisons use integer encoding instead of string matching.
    for _cat_col in ('type', 'wo_status', 'visual_status', 'data_source'):
        if _cat_col in df.columns:
            df[_cat_col] = df[_cat_col].astype('category')

    # Pre-group parts-type rows by root_work_order_id once, AFTER df is final
    # (sorted, all columns added, categoricals applied) so these slices match
    # exactly what a live df[...] scan would return.  render_single_request_html
    # fans cross-request parts into each root section; doing that via a full-df
    # scan per root per request (df[df['root_work_order_id']==root_id & ...]) was
    # an O(requests × roots × rows) hot spot (object== + isin over all 48k rows).
    _parts_by_root_dfs = {
        rid: grp
        for rid, grp in df[df['type'].isin(_global_parts_types)].groupby('root_work_order_id')
    }

    # First-match workorder_id → source fields, built over the FINAL (sorted) df so
    # it returns exactly what a live `df[df['workorder_id']==x].iloc[0]` scan would.
    # Replaces that full-48K-row boolean mask in the per-row section build loop.
    _global_src_lookup: dict = {}
    _gsl_wids = df['workorder_id'].tolist()
    _gsl_exp = df['experiment_name'].tolist() if 'experiment_name' in df.columns else [None] * len(_gsl_wids)
    _gsl_req = df['req_id'].tolist() if 'req_id' in df.columns else [None] * len(_gsl_wids)
    _gsl_rst = df['request_status'].tolist() if 'request_status' in df.columns else [None] * len(_gsl_wids)
    for _gi in range(len(_gsl_wids)):
        _gw = str(_gsl_wids[_gi])
        if _gw not in _global_src_lookup:
            _global_src_lookup[_gw] = {
                'experiment_name': _gsl_exp[_gi],
                'req_id': _gsl_req[_gi],
                'request_status': _gsl_rst[_gi],
            }

    # DataFrame index → full row record, built once via a SINGLE to_dict over the
    # final sorted df (~2s). Per-section row_map was rebuilt with root_df.to_dict
    # ('records') — ~3980 small to_dict calls whose per-call block-manager overhead
    # dominated (~35s). df.index is unique, and every section slice (req_df, fan-in
    # parts, cross-plan subs) is a slice of df, so its rows resolve by index here —
    # a shallow dict() copy per row keeps per-section mutations (visual_suffix,
    # _attempt_*) isolated, exactly as fresh to_dict records were.
    _rec_by_index = dict(zip(df.index, df.to_dict('records')))

    # Plate-popover dedup pool. The ~88k plate/colony hover popovers are ~91%
    # duplicates and were inlined in full (~21 MB). Intern each unique popover
    # body once; build sites emit an EMPTY <div class="plate-popover" data-pop="N">
    # and the pool (window.PLATE_POP) is emitted once in the tail. JS fills each
    # popover from the pool on first hover/click — CSS hover + click-pin unchanged.
    _popover_pool: dict[str, int] = {}
    def _intern_popover(content: str) -> int:
        idx = _popover_pool.get(content)
        if idx is None:
            idx = len(_popover_pool)
            _popover_pool[content] = idx
        return idx

    # =========================================================================
    # HTML CSS WITH TABS
    # =========================================================================

    # ── Design tokens → CSS (single source of truth: renderer/tokens.py) ──────
    _g = tok.GEOM["status"]
    _status_css = (
        f".badge {{ padding:{_g['pad']}; border-radius:{_g['radius']}; "
        f"font-size:{_g['size']}; font-weight:{_g['weight']}; text-transform:uppercase; "
        f"white-space:nowrap; }}\n"
    )
    # Status icon = Lucide SVG (tokens.LUCIDE_PATHS), baked into the ::before as a
    # data-URI with the stroke colored to match the status text. Same paths the
    # Requests-In-Flight tab renders inline, so icons are identical across tabs.
    def _status_icon_uri(_path: str, _color: str) -> str:
        _svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='11' height='11' "
            "viewBox='0 0 24 24' fill='none' stroke='" + _color + "' "
            "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            + _path + "</svg>"
        )
        return "data:image/svg+xml," + urllib.parse.quote(_svg, safe="")
    for _k, (_bg, _txt, _bd) in tok.STATUS.items():
        _border = "none" if _k == "BLOCKED" and _bd is None else f"1px solid {_bd}"
        _status_css += f"    .status-{_k} {{ background:{_bg}; color:{_txt}; border:{_border}; }}\n"
        _icname = tok.STATUS_LUCIDE.get(_k)
        if _icname:
            _uri = _status_icon_uri(tok.LUCIDE_PATHS[_icname], _txt)
            _status_css += (
                f"    .status-{_k}::before {{ content:url(\"{_uri}\") '\\00a0'; "
                f"vertical-align:-1.5px; }}\n"
            )
    _tdot_css = ""
    for _state, _color in tok.TIMELINE_DOT.items():
        _anim = " animation: pulse 2s infinite;" if _state == "running" else ""
        _tdot_css += f"    .t-dot.{_state} {{ background:{_color};{_anim} }}\n"

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
      /* NEUTRAL SCALE — single cool-gray (slate) family for all chrome */
      --c-border:        #e5e7eb;  /* hairline borders */
      --c-border-strong: #cbd5e1;  /* emphasized dividers */
      --c-surface:       #ffffff;
      --c-surface-1:     #f8fafc;  /* subtle fill */
      --c-surface-2:     #f1f5f9;  /* muted fill / code pills */
      --c-accent:        #2563eb;
      /* RADII */
      --r-sm: 4px;
      --r-md: 6px;
      --r-lg: 8px;
      /* ELEVATION */
      --shadow-sm: 0 1px 2px rgba(15,23,42,0.05);
      --shadow-md: 0 1px 3px rgba(15,23,42,0.06);
      --shadow-lg: 0 4px 14px rgba(15,23,42,0.10);
    }
    /* BASE */
    * { -webkit-font-smoothing: antialiased; box-sizing: border-box; }
    body { background: #e9ecf2; padding: 10px; margin: 0; }
    /* RESPONSIVE — tighten chrome padding as the viewport narrows so dense
       content keeps room before it has to wrap (laptop vs. external monitor). */
    @media (max-width: 1400px) {
      body { padding: 7px; }
      .project-wrapper { padding: 10px 11px; }
    }
    @media (max-width: 1100px) {
      body { padding: 5px; }
      .project-wrapper { padding: 8px; margin-bottom: 8px; }
      .req-title-bar { padding: 3px 5px; }
      .controls-container { gap: 7px; padding: 5px 8px; }
    }
    .dashboard-container { max-width: 100%; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 11px; line-height: 1.45; color: #1d1d1f; letter-spacing: 0.05px; }

    /* LOGO HEADER */
    .dashboard-header { display: flex; align-items: center; padding: 8px 12px; background: white; border-bottom: 1px solid #e5e7eb; gap: 12px; border-radius: 5px 5px 0 0; }
    .dashboard-logo { height: 32px; width: auto; }
    .dashboard-title { font-size: 13px; font-weight: 800; color: #1d1d1f; }
    .dashboard-updated { margin-left: auto; display: inline-flex; align-items: baseline; gap: 5px;
        background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 3px 10px;
        font-size: 10px; font-weight: 600; color: #1e40af; letter-spacing: .02em; }
    .dashboard-updated .du-label { color: #60a5fa; text-transform: uppercase; font-size: 9px; letter-spacing: .06em; }
    .dashboard-updated .du-time { font-size: 12px; font-weight: 700; color: #1d4ed8; }
    .tab-icon-img { height: 36px; width: 36px; border-radius: 6px; }

    /* TABS */
    .tab-container { margin-bottom: 0; }
    /* Kernel nav track: white, text tabs, blue active underline, hairline under it. */
    .tab-nav { display: flex; gap: 4px; background: #fff; border-bottom: 1px solid #e5e7eb; border-radius: 0; padding: 0 8px; }
    .tab-btn { padding: 11px 14px; font-size: 13px; font-weight: 600; color: #6b7280; background: transparent; border: none; border-bottom: 2px solid transparent; cursor: pointer; display: flex; align-items: center; gap: 6px; margin-bottom: -1px; }
    .tab-btn:last-child { border-right: none; }
    .tab-btn:hover { color: #374151; }
    .tab-btn.active { color: #1d4ed8; border-bottom: 2px solid #2563eb; background: transparent; }
    /* Hide inactive tabs with content-visibility (not display:none) so their
       render state is cached. Switching back to a large tab (Tracking) becomes a
       repaint instead of a full render-tree rebuild + style recalc + reflow of
       ~1.37M nodes. content-visibility:hidden ALSO drops the offscreen tab from
       painting/compositing entirely — visibility:hidden still kept the giant
       Tracking layer painted, which checkerboarded (grey-on-scroll) the active
       tab. position:absolute keeps hidden tabs out of flow so the active tab
       isn't pushed down. */
    .tab-content { position: absolute; content-visibility: hidden; }
    .tab-content.active { position: static; content-visibility: visible; }
    .tab-icon-img { height: 24px; width: 24px; border-radius: 4px; }
    /* Kernel text tabs: hide the leading icon (img or emoji), show the text label. */
    .tab-btn > :first-child { display: none; }
    .tab-btn .tab-text { display: inline; }

    /* UNDER CONSTRUCTION */
    .under-construction { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; text-align: center; }
    .uc-icon { font-size: 40px; margin-bottom: 10px; }
    .uc-title { font-size: 18px; font-weight: 800; color: #1d1d1f; margin-bottom: 5px; }
    .uc-subtitle { font-size: 11px; color: #86868b; max-width: 300px; line-height: 1.4; }
    .uc-badge { background: #eff4ff; color: #1d4ed8; border: 1px solid #bcd0fb; padding: 4px 11px; border-radius: 6px; font-size: 11px; font-weight: 600; margin-top: 10px; }

    /* CONTROLS */
    .controls-container { display: flex; align-items: center; gap: 10px; padding: 6px 10px; background: #f8fafc; border-bottom: 1px solid #e5e7eb; }
    .toggle-wrapper { display: flex; align-items: center; gap: 5px; }
    .toggle-label { font-size: 10px; font-weight: 600; color: #86868b; white-space: nowrap; }
    .switch { position: relative; display: inline-block; width: 28px; height: 16px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #e5e7eb; border-radius: 16px; }
    .slider:before { position: absolute; content: ""; height: 12px; width: 12px; left: 2px; bottom: 2px; background-color: white; border-radius: 50%; }
    input:checked + .slider { background: #2563eb; }
    input:checked + .slider:before { left: 14px; }

    /* PROJECT WRAPPER — rigid, bounded horizontal lanes (engineering grid). */
    .project-wrapper { margin: 0 0 12px 0; padding: 12px 14px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; box-shadow: 0 1px 3px rgba(15,23,42,0.06); }
    .header-banner { color: #0f172a; border: none; border-radius: 8px; box-shadow: none; margin-bottom: 4px; cursor: pointer; }
    .header-banner:hover { filter: brightness(0.98); }
    .header-title { font-size: 14px; font-weight: 700; color: #0f172a; white-space: nowrap; padding-right: 16px; }
    /* Kernel metadata pill (shared with inflight tab) */
    .kpill { display:inline-flex; align-items:center; gap:5px; background:#f1f5f9; border:1px solid #e5e7eb; border-radius:6px; padding:3px 9px; font-size:11px; color:#1f2937; font-weight:500; white-space:nowrap; }
    .kpill .kk { color:#6b7280; font-weight:500; }
    .kpill b { font-weight:700; color:#1e2937; }
    .header-main-stat { font-size: 10px; font-weight: 700; color: #374151; background: #f1f5f9; border: 1px solid #e5e7eb; padding: 2px 8px; border-radius: 6px; white-space: nowrap; }
    .stat-item { background: #f1f5f9; color: #374151; border: 1px solid #e5e7eb; padding: 2px 7px; border-radius: 6px; font-size: 8px; font-weight: 600; white-space: nowrap; }
    .stat-label { font-weight: 800; color: #111827; margin-right: 3px; font-size: 9px; }

    /* REQUEST CARDS */
    .req-card { border: 1px solid var(--c-border); background: var(--c-surface); margin-top: 4px; border-radius: var(--r-lg); box-shadow: var(--shadow-md); }
    .req-title-bar { padding: 3px 6px; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; border-left-width: 3px; border-left-style: solid; gap: 4px; min-height: 20px; }
    .req-name { font-size: 10px; font-weight: 700; color: #1d1d1f; white-space: nowrap; }
    .req-meta { font-size: 7px; color: #86868b; margin-top: 0px; }

    /* ASSEMBLY SECTIONS */
    .assembly-section { margin: 5px 6px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
    .assembly-section.dimmed { opacity: 0.5; }
    .dropdown-btn { width: 100%; background: var(--c-surface-1); border: none; padding: 5px 8px; text-align: left; cursor: pointer; display: flex; align-items: center; font-size: 9px; }
    .dropdown-btn:hover { background: var(--c-surface-2); }
    .dropdown-btn.active-header { background: #e5e7eb; }
    .dropdown-icon { color: #86868b; margin-right: 5px; font-size: 8px; }
    .dropdown-icon.open { transform: rotate(90deg); }
    .assembly-info { flex-grow: 1; display: flex; align-items: center; gap: 6px; white-space: nowrap; overflow: hidden; }
    .assembly-type { font-weight: 800; color: #4b5563; font-size: 10px; }
    .assembly-counts { font-weight: 600; color: #86868b; font-size: 9px; }

    /* BADGES */
    .bios-override-label { font-size: 7px; color: #86868b; margin-top: 3px; }
    /*__STATUS_CSS__*/

    /* STOCK TAG */
    .stock-tag { background: #f1f5f9; color: #4b5563; border: 1px solid #cbd5e1; padding: 1px 4px; border-radius: 2px; font-family: monospace; font-weight: 700; font-size: 9px; white-space: nowrap; }
    .dropdown-btn .stock-tag { font-size: 8px; padding: 1px 3px; }
    .stock-id-badge { background: #f1f5f9; color: #4b5563; border: 1px solid #cbd5e1; padding: 1px 4px; border-radius: 2px; font-family: monospace; font-weight: 700; font-size: 9px; white-space: nowrap; }
    .stock-id-badge.matches-root { background: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; }
    .stock-id-badge.secondary-root { background: #ddd6fe; color: #4c1d95; border: 1px solid #7c3aed; }
    .wo-id-tag { background: none; color: #374151; padding: 1px 3px; font-family: monospace; font-size: 7px; }

    /* TABLE */
    .content-pane { display: none; padding: 0; background: #fff; overflow-x: auto; }
    .wo-table { width: 100%; border-collapse: collapse; font-size: 9px !important; }
    .wo-table th { text-align: left; color: #0f172a; padding: 5px 8px; border-bottom: 1px solid #cbd5e1; font-size: 8px; font-weight: 700; background: #f1f5f9; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
    .wo-table td { padding: 4px 8px; border-bottom: 1px solid #f1f5f9; vertical-align: top; font-weight: 600; color: #1d1d1f; line-height: 1.4; }
    .wo-table tbody tr:hover { background: #f5f3ff; box-shadow: inset 2px 0 0 #7c3aed; }
    .wo-table .stock-tag { font-size: 8px; padding: 1px 3px; }

    /* TREE ROWS */
    .tree-row-0 { border-left: 3px solid #e5e7eb !important; background: #ffffff !important; }
    .tree-row-1 { border-left: 3px solid #e5e7eb !important; background: #f8fafc !important; }
    .tree-row-2 { border-left: 3px solid #e5e7eb !important; background: #f1f5f9 !important; }
    .tree-line-icon { color: #e5e7eb; margin-right: 2px; font-family: monospace; font-size: 9px; }
    .source-badge { font-size: 7px; padding: 0px 2px; border-radius: 2px; background: #f1f5f9; color: #86868b; margin-left: 2px; }

    /* Hoisted high-frequency styles (was ~16MB of repeated inline style="" attrs) */
    .u1 { color:#9ca3af;font-size:10px;font-weight:600;text-transform:uppercase;padding:4px 0;border-bottom:1px solid #f1f5f9; }
    .u2 { margin-left: 8px; }
    .u3 { color:#6b7280;font-size:9px;font-weight:700;text-transform:uppercase; }
    .u4 { font-size:11px;color:#1f2937;padding:4px 0;border-bottom:1px solid #f1f5f9; }
    .u5 { display: flex; align-items: center; background: #f1f5f9; border: 1px solid #cbd5e1; padding: 1px 4px; border-radius: 2px; gap: 3px; height: 16px; }
    .u6 { border-bottom: 1px solid #e5e7eb; margin-bottom: 10px; padding-bottom: 5px; font-weight: 800; color: #4b5563; text-transform: uppercase; font-size: 11px; }
    .u7 { font-size:10px;color:#1e293b; }
    .u8 { font-family: monospace; font-size: 11px; color: #4b5563; background: #f1f5f9; padding: 2px 6px; border-radius: 3px; text-decoration: underline; border: 1px solid #cbd5e1; }
    .u9 { font-size: 9px; color: #6b7280; font-weight: 700; text-transform: uppercase; }
    .u10 { font-size: 10px; font-weight: 700; color: #4b5563; font-family: monospace; }
    .u11 { color:#16a34a;font-size:11px;margin-right:4px; }
    .u12 { color:#7c3aed;text-decoration:underline;font-weight:700; }
    .u13 { font-weight:600;color:#7c3aed; }
    .u14 { padding:5px 10px 4px; background:#f8fafc; border-top:2px solid #e2e8f0; border-bottom:1px solid #e2e8f0; }

    /* TIMELINE */
    .timeline-container { position: relative; padding-left: 2px; }
    .timeline-row { display: flex; align-items: flex-start; margin-bottom: 1px; position: relative; min-height: 11px; }
    .timeline-row:not(:last-child):after { content: ''; position: absolute; left: 2px; top: 6px; bottom: -3px; width: 1px; background: #e5e5e7; z-index: 1; }
    .t-dot { width: 5px; height: 5px; border-radius: 50%; margin-top: 2px; margin-right: 4px; flex-shrink: 0; z-index: 2; }
    /*__TDOT_CSS__*/
    .t-content { flex-grow: 1; font-size: 8px; }
    .t-header { display: flex; justify-content: space-between; align-items: center; }
    .t-name { font-weight: 700; color: #1d1d1f; white-space: nowrap; font-size: 9px; }
    .t-time { font-family: monospace; color: #86868b; font-size: 8px; white-space: nowrap; }
    .t-details { font-size: 7px; color: #86868b; margin-top: 0px; display: flex; flex-wrap: wrap; gap: 2px; }
    .t-pill { background: #f1f5f9; color: #86868b; border: 1px solid #e5e7eb; padding: 0px 2px; border-radius: 2px; font-family: monospace; font-size: 8px; text-decoration: none; }
    .t-pill:hover { background: #e5e7eb; }

    /* PLATE HOVER */
    .plate-hover-container { position: relative; display: inline-block; }
    .plate-trigger { cursor: pointer; background: #f1f5f9; color: #86868b; padding: 0px 2px; border-radius: 2px; font-family: monospace; font-size: 7px; border: 1px solid #e5e7eb; text-decoration: none; }
    .plate-trigger:hover { background: #e5e7eb; }
    .plate-popover { display: none; position: absolute; top: 100%; left: 0; background: white; border: 1px solid #e5e7eb; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 4px 6px; border-radius: 3px; z-index: 9999; min-width: 150px; margin-top: 2px; }
    .plate-hover-container:hover .plate-popover { display: block; }
    .popover-title { font-weight: 700; font-size: 7px; color: #86868b; text-transform: uppercase; margin-bottom: 2px; font-family: monospace; }
    .popover-link { font-size: 7px; color: #86868b; text-decoration: none; padding: 0px 2px; font-family: monospace; }
    .popover-link:hover { text-decoration: underline; }

    /* PART TAGS */
    .part-tag { background: #f1f5f9; color: #86868b; padding: 0px 2px; border-radius: 2px; font-family: monospace; font-size: 8px; margin-right: 2px; border: 1px solid #e5e7eb; }
    .part-tag.in-production { background: #fffbeb; color: #d97706; border: 1px dashed #fcd34d; }
    .part-tag.missing { background: #fdf2f8; color: #be185d; border: 1px solid #f9a8d4; font-weight: 700; cursor: default; }
    .ci-wrap { position: relative; display: inline-block; cursor: pointer; }
    .ci-tip { display: none; position: absolute; bottom: calc(100% + 4px); left: 0; background: #1e293b;
        border: 1px solid #334155; border-radius: 4px; padding: 6px 8px; z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4); white-space: nowrap; }
    .ci-tip.ci-open { display: block; }
    .ci-tip.ci-flip { bottom: auto; top: calc(100% + 4px); }
    .ci-tip-header { font-family: monospace; font-size: 8px; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; padding-bottom: 3px;
        border-bottom: 1px solid #334155; }
    .ci-tip-row { display: flex; gap: 10px; font-family: monospace; font-size: 9px; color: #e2e8f0; line-height: 1.8; }
    .ci-tip-stock { font-weight: 700; color: #a78bfa; min-width: 80px; }
    .ci-tip-plate { color: #38bdf8; text-decoration: underline; }
    .ci-tip-pid { color: #475569; }
    .missing-tip-wrap { position: relative; display: inline-block; }
    .missing-tip { display: none; position: absolute; bottom: calc(100% + 4px); left: 0; background: #1e293b;
        border: 1px solid #334155; border-radius: 4px; padding: 5px 8px; z-index: 9999; min-width: 220px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4); pointer-events: none; }
    .missing-tip-wrap:hover .missing-tip { display: block; }
    .missing-tip.mt-flip { bottom: auto; top: calc(100% + 4px); }
    .badge-tip-wrap { position: relative; display: inline-block; }
    .badge-tip { display: none; position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%);
        background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 4px; padding: 5px 10px;
        font-size: 11px; font-weight: 500; white-space: nowrap; z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4); pointer-events: none; }
    .badge-tip-wrap:hover .badge-tip { display: block; }
    .missing-tip-req { font-family: monospace; font-size: 8px; color: #94a3b8; margin-bottom: 2px; }
    .missing-tip-exp { font-size: 9px; font-weight: 700; color: #e2e8f0; }
    .missing-tip-status { font-size: 8px; color: #be185d; font-weight: 600; margin-top: 2px; }
    .wait-ops { font-size: 11px; }
    .wait-ops-hd { font-weight: 700; color: #b45309; margin-bottom: 4px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.03em; }
    .wait-ops-row { display: flex; gap: 8px; align-items: baseline; line-height: 1.65; }
    .wait-part { font-family: monospace; font-weight: 700; color: #7c3aed; min-width: 78px; }
    .wait-loc { color: #64748b; font-size: 10px; }
    .wait-loc.elsewhere { color: #be185d; font-weight: 600; }
    .wait-loc.here { color: #0d9488; }
    .wait-loc.none { color: #9ca3af; font-style: italic; }
    .colony-badge { font-size: 7px; padding: 1px 3px; border-radius: 2px; font-weight: 600; }
    .tat-cell { font-family: monospace; color: #86868b; font-size: 8px; white-space: nowrap; }

    /* GROUP HEADERS */
    .group-header { padding: 5px 8px; font-size: 10px; font-weight: 700; background: #f1f5f9; color: #1d1d1f; border: 1px solid #e5e7eb; cursor: pointer; display: flex; align-items: center; border-radius: 3px; margin: 6px 0 3px 0; }
    .group-header:hover { background: #e5e7eb; }
    .group-header.in-progress { background: #f9fafb; border-color: #e5e7eb; border-left: 3px solid #2563eb; color: #374151; }
    .group-header.new { background: #f9fafb; border-color: #e5e7eb; border-left: 3px solid #7c3aed; color: #374151; }
    .group-header.fulfilled { background: #f9fafb; border-color: #e5e7eb; border-left: 3px solid #16a34a; color: #374151; }
    .group-header.canceled { background: #f9fafb; border-color: #e5e7eb; border-left: 3px solid #9ca3af; color: #374151; }
    .group-arrow { margin-right: 5px; font-size: 8px; }
    details[open] .group-arrow { transform: rotate(90deg); }
    details > summary { list-style: none; }

    /* ANIMATION */
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    /* WARNING & SEARCH */
    .warning-note { background: #fdf2f8; border: 1px solid #f9a8d4; color: #be185d; padding: 2px 5px; margin: 3px 6px; border-radius: 2px; font-weight: 600; font-size: 8px; }
    #search_box { width: 280px; padding: 4px 8px; border: 1px solid #e5e7eb; border-radius: 4px; font-size: 10px; background: white; }
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
        try { localStorage.setItem('dash_activeTab', tabName); } catch(e) {}
        if (tabName === 'capacity' && typeof window.lspInitChart === 'function') {
            setTimeout(window.lspInitChart, 0);
        }
        if (tabName === 'inflight' && !window._ifBuilt) {
            if (typeof window.ifBuildHead === 'function') { window.ifBuildHead(); window.ifRender(); window._ifBuilt = true; }
        }
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
    // Lazily build the heavy Queue/timeline cells the first time an assembly
    // content-pane is opened. Until then they live in inert <template>s (out of
    // the render tree), so the initial DOM is ~60% smaller.
    function _buildQueues(el) {
        if (!el || !el.classList || !el.classList.contains('content-pane')) return;
        var cells = el.querySelectorAll('td.queue-cell:not([data-built])');
        for (var i = 0; i < cells.length; i++) {
            var td = cells[i];
            var tpl = td.querySelector('template.qtpl');
            if (tpl) { td.appendChild(tpl.content.cloneNode(true)); }
            td.setAttribute('data-built', '1');
        }
    }
    function toggleSection(id) {
        var el = document.getElementById(id); var icon = document.getElementById(id + '_icon'); var btn = document.getElementById(id + '_btn');
        if (el.style.display === "block") { el.style.display = "none"; if(icon) icon.classList.remove('open'); if(btn) btn.classList.remove('active-header'); }
        else { _buildQueues(el); el.style.display = "block"; if(icon) icon.classList.add('open'); if(btn) btn.classList.add('active-header'); }
    }
    var _sortedByDue = false;
    function sortByDueDate() {
        var container = document.getElementById('projects-container');
        if (!container) return;
        var cards = Array.from(container.querySelectorAll(':scope > .project-wrapper'));
        if (!_sortedByDue) {
            cards.sort(function(a, b) {
                var da = a.getAttribute('data-due-date') || '9999-99-99';
                var db = b.getAttribute('data-due-date') || '9999-99-99';
                return da < db ? -1 : da > db ? 1 : 0;
            });
            cards.forEach(function(c, i) { c.style.order = i; });
            _sortedByDue = true;
            document.getElementById('sort_due_btn').textContent = 'Sort: Default';
            document.getElementById('sort_due_btn').style.background = '#e8f4fd';
            document.getElementById('sort_due_btn').style.borderColor = '#0891b2';
            document.getElementById('sort_due_btn').style.color = '#0891b2';
        } else {
            cards.forEach(function(c) { c.style.order = ''; });
            _sortedByDue = false;
            document.getElementById('sort_due_btn').textContent = 'Sort: Due Date';
            document.getElementById('sort_due_btn').style.background = '#fff';
            document.getElementById('sort_due_btn').style.borderColor = '#e5e7eb';
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
                var headerBanner = project.querySelector('.header-banner');
                var headerMatches = headerBanner && headerBanner.textContent.toLowerCase().includes(searchTerm);
                project.querySelectorAll('.req-card').forEach(function(reqCard) {
                    if (!headerMatches && !reqCard.textContent.toLowerCase().includes(searchTerm)) {
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
        var countEl = document.getElementById('search_count');
        if (countEl) {
            if (searchTerm) {
                var typeLabels = {
                    'golden_gate_workorder': 'Golden Gate',
                    'gibson_workorder': 'Gibson',
                    'lsp_workorder': 'LSP',
                    'pcr_workorder': 'PCR',
                    'oligo_synthesis_workorder': 'Oligo',
                    'plasmid_synthesis_workorder': 'Synthesis',
                    'transformation_workorder': 'Transformation',
                    'streak_workorder': 'Streakout'
                };
                var sourceCounts = {}, inputCounts = {}, sourceReqs = 0, inputReqs = 0;
                var seenWoIds = {};
                document.querySelectorAll('.req-card').forEach(function(card) {
                    var rows = card.querySelectorAll('tr.search-match[data-wo-type]');
                    var hasSource = false, hasInput = false;
                    rows.forEach(function(row) {
                        var t = row.getAttribute('data-wo-type');
                        if (!typeLabels[t]) return;
                        var woId = row.getAttribute('data-wo-id') || '';
                        var stock = row.getAttribute('data-wo-stock') || '';
                        var isSource = stock && stock.includes(searchTerm);
                        var bucket = isSource ? 'src' : 'inp';
                        var key = bucket + '|' + woId;
                        if (woId && seenWoIds[key]) return;
                        if (woId) seenWoIds[key] = true;
                        if (isSource) { sourceCounts[t] = (sourceCounts[t] || 0) + 1; hasSource = true; }
                        else { inputCounts[t] = (inputCounts[t] || 0) + 1; hasInput = true; }
                    });
                    if (hasSource) sourceReqs++;
                    if (hasInput) inputReqs++;
                });
                var srcTypeParts = Object.keys(sourceCounts).map(function(t) {
                    return sourceCounts[t] + ' ' + typeLabels[t];
                });
                var inTypeParts = Object.keys(inputCounts).map(function(t) {
                    return inputCounts[t] + ' ' + typeLabels[t] + ' workorder' + (inputCounts[t] === 1 ? '' : 's');
                });
                var lines = [];
                if (srcTypeParts.length) lines.push('Product: ' + sourceReqs + ' request' + (sourceReqs===1?'':'s') + ' (' + srcTypeParts.join(', ') + ')');
                if (inTypeParts.length) lines.push('Input to: ' + inTypeParts.join(', ') + ' · ' + inputReqs + ' request' + (inputReqs===1?'':'s'));
                countEl.textContent = lines.join('  |  ');
                var dlBtn = document.getElementById('search_download');
                if (dlBtn) dlBtn.style.display = lines.length ? 'inline' : 'none';
                var listBtn = document.getElementById('search_list_btn');
                if (listBtn) listBtn.style.display = lines.length ? 'inline' : 'none';
                countEl.style.display = 'inline';
            } else {
                countEl.style.display = 'none';
                var dlBtn2 = document.getElementById('search_download');
                if (dlBtn2) dlBtn2.style.display = 'none';
                var listBtn2 = document.getElementById('search_list_btn');
                if (listBtn2) listBtn2.style.display = 'none';
                var pop = document.getElementById('search_list_popup');
                if (pop) pop.style.display = 'none';
            }
        }
    }
    function _searchMatchRows() {
        var searchTerm = document.getElementById('search_box').value.trim().toLowerCase();
        var typeLabels = {
            'golden_gate_workorder': 'Golden Gate', 'gibson_workorder': 'Gibson',
            'lsp_workorder': 'LSP', 'pcr_workorder': 'PCR',
            'oligo_synthesis_workorder': 'Oligo', 'plasmid_synthesis_workorder': 'Synthesis',
            'transformation_workorder': 'Transformation', 'streak_workorder': 'Streakout'
        };
        var results = [];
        var seen = {};
        document.querySelectorAll('tr.search-match[data-wo-type]').forEach(function(tr) {
            var woId = tr.getAttribute('data-wo-id') || '';
            var t = tr.getAttribute('data-wo-type') || '';
            var stock = (tr.getAttribute('data-wo-stock') || '').toLowerCase();
            var isSource = stock && searchTerm && stock.includes(searchTerm);
            var key = (isSource ? 'src' : 'inp') + '|' + woId;
            if (woId && seen[key]) return;
            if (woId) seen[key] = true;
            var cells = tr.querySelectorAll('td');
            var status = cells[2] ? cells[2].textContent.trim() : '';
            var created = cells[4] ? cells[4].textContent.trim() : '';
            var stockDisplay = cells[3] ? cells[3].textContent.trim() : (tr.getAttribute('data-wo-stock') || '');
            var projEl = tr.closest('.project-wrapper');
            var expName = projEl ? (projEl.querySelector('.header-title') || {textContent: ''}).textContent.trim() : '';
            var partner = tr.getAttribute('data-wo-partner') === '1' ? 'Yes' : 'No';
            var role = isSource ? 'Product' : 'Input to';
            var typeLbl = typeLabels[t] || t;
            results.push({role: role, type: typeLbl, woId: woId, stock: stockDisplay, status: status, created: created, experiment: expName, partner: partner});
        });
        return results;
    }
    function downloadSearchCSV() {
        var searchTerm = document.getElementById('search_box').value.trim();
        var results = _searchMatchRows();
        var rows = ['"Role","Type","Workorder ID","Stock ID","Status","Created","Experiment","For Partner"'];
        results.forEach(function(r) {
            rows.push([r.role, r.type, r.woId, r.stock, r.status, r.created, r.experiment, r.partner].map(function(v) {
                return '"' + String(v).replace(/"/g, '""') + '"';
            }).join(','));
        });
        var csv = rows.join('\\n');
        var blob = new Blob([csv], {type: 'text/csv'});
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'search_' + searchTerm.replace(/[^a-zA-Z0-9_-]/g, '_') + '.csv';
        a.click();
        URL.revokeObjectURL(a.href);
    }
    var _slSortState = {};
    function _slApplyFilter(filterVal) {
        var term = filterVal.toLowerCase().trim();
        document.querySelectorAll('#search_list_popup tbody tr').forEach(function(tr) {
            var text = tr.textContent.toLowerCase();
            tr.style.display = (!term || text.includes(term)) ? '' : 'none';
        });
    }
    function _slSortTable(table, colIdx) {
        var key = table.id + ':' + colIdx;
        var asc = _slSortState[key] !== true;
        _slSortState[key] = asc;
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort(function(a, b) {
            var av = (a.cells[colIdx] ? a.cells[colIdx].textContent.trim() : '').toLowerCase();
            var bv = (b.cells[colIdx] ? b.cells[colIdx].textContent.trim() : '').toLowerCase();
            if (av < bv) return asc ? -1 : 1;
            if (av > bv) return asc ? 1 : -1;
            return 0;
        });
        rows.forEach(function(r) { tbody.appendChild(r); });
        table.querySelectorAll('thead th').forEach(function(th, i) {
            var base = th.getAttribute('data-label') || th.textContent.replace(/[ ↑↓]/g,'').trim();
            th.setAttribute('data-label', base);
            th.textContent = base + (i === colIdx ? (asc ? ' ↑' : ' ↓') : '');
        });
    }
    function _buildListSection(title, rows) {
        if (!rows.length) return '';
        var cols = ['Type','Workorder ID','Stock ID','Status','Created','Partner','Experiment'];
        var thStyle = 'padding:2px 8px;text-align:left;white-space:nowrap;cursor:pointer;user-select:none;';
        thStyle += 'border-bottom:2px solid #cbd5e1;font-size:10px;color:#374151;';
        var html = '<div style="font-size:10px;font-weight:700;color:#374151;margin:8px 0 3px;">' + title + '</div>';
        var tid = 'sl_' + title.toLowerCase().replace(/[^a-z]/g,'_');
        html += '<table id="' + tid + '" style="width:100%;border-collapse:collapse;font-size:10px;">';
        html += '<thead><tr style="background:#f1f5f9;">';
        cols.forEach(function(c) { html += '<th data-label="' + c + '" style="' + thStyle + '">' + c + '</th>'; });
        html += '</tr></thead><tbody>';
        rows.forEach(function(r) {
            var vals = [r.type, r.woId, r.stock, r.status, r.created, r.partner, r.experiment];
            html += '<tr style="border-bottom:1px solid #f0f0f0;">';
            vals.forEach(function(v, i) {
                var s = 'padding:2px 8px;';
                if (i === 1) s += 'font-family:monospace;font-size:9px;';
                if (i === 6) s += 'max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
                html += '<td style="' + s + '" title="' + String(v).replace(/"/g,'&quot;') + '">' + v + '</td>';
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        return {html: html, tid: tid};
    }
    function toggleSearchList() {
        var pop = document.getElementById('search_list_popup');
        if (!pop) return;
        if (pop.style.display !== 'none' && pop.style.display !== '') { pop.style.display = 'none'; return; }
        var results = _searchMatchRows();
        var srcRows = results.filter(function(r){ return r.role === 'Product'; });
        var inpRows = results.filter(function(r){ return r.role === 'Input to'; });
        var src = _buildListSection('Product', srcRows);
        var inp = _buildListSection('Input to', inpRows);
        var closeBtn = document.createElement('div');
        closeBtn.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;';
        closeBtn.innerHTML = '<span style="font-size:10px;font-weight:700;color:#1e293b;">Search Results</span>';
        var xBtn = document.createElement('button');
        xBtn.textContent = '×';
        xBtn.style.cssText = 'background:none;border:none;font-size:14px;cursor:pointer;color:#64748b;line-height:1;';
        xBtn.onclick = function() { pop.style.display = 'none'; };
        closeBtn.appendChild(xBtn);
        var filterRow = document.createElement('div');
        filterRow.style.cssText = 'margin-bottom:6px;';
        var filterInput = document.createElement('input');
        filterInput.type = 'text';
        filterInput.placeholder = 'Filter rows…';
        filterInput.style.cssText = 'width:100%;box-sizing:border-box;padding:3px 7px;font-size:10px;border:1px solid #cbd5e1;border-radius:3px;';
        filterInput.oninput = function() { _slApplyFilter(this.value); };
        filterRow.appendChild(filterInput);
        pop.innerHTML = (src ? src.html : '') + (inp ? inp.html : '');
        pop.insertBefore(filterRow, pop.firstChild);
        pop.insertBefore(closeBtn, pop.firstChild);
        var tids = [];
        if (src) tids.push(src.tid);
        if (inp) tids.push(inp.tid);
        tids.forEach(function(tid) {
            var table = document.getElementById(tid);
            if (!table) return;
            table.querySelectorAll('thead th').forEach(function(th, i) {
                th.addEventListener('click', function() { _slSortTable(table, i); });
            });
        });
        pop.style.display = 'block';
    }
    document.addEventListener('DOMContentLoaded', function() {
        var _overlay = document.getElementById('_loading_overlay');
        function _hideOverlay() {
            if (!_overlay) return;
            _overlay.classList.add('fade-out');
            setTimeout(function() { _overlay.style.display = 'none'; }, 260);
        }
        try {
            var savedTab = localStorage.getItem('dash_activeTab');
            var savedActive = localStorage.getItem('dash_activeOnly');
            var eh = document.getElementById('_earlyhide'); if (eh) eh.remove();
            if (savedTab && savedTab !== 'tracking' && document.querySelector('[data-tab="' + savedTab + '"]')) { switchTab(savedTab); }
            if (savedActive === '1') { document.getElementById('active_toggle').checked = true; filterDashboard(); }
        } catch(e) {}
        var firstExp = document.querySelector('.exp-content');
        var firstExpIcon = document.querySelector('.exp-toggle-icon');
        if (firstExp) { firstExp.style.display = 'block'; }
        if (firstExpIcon) { firstExpIcon.classList.add('open'); }
        setTimeout(function() { _hideOverlay(); }, 0);
        // Lazy-fill deduped plate popovers from window.PLATE_POP on first hover/click.
        function _fillPop(p) {
            if (p && p.dataset.pop !== undefined && !p._filled) {
                p.innerHTML = (window.PLATE_POP && window.PLATE_POP[+p.dataset.pop]) || '';
                p._filled = 1;
            }
        }
        document.addEventListener('mouseover', function(e) {
            var c = e.target.closest('.plate-hover-container');
            if (c) { _fillPop(c.querySelector('.plate-popover')); }
            // Missing-part tooltip opens upward by default; flip it downward when the
            // top would be clipped by a scrolling/overflow ancestor (.content-pane).
            // Measure the trigger (stable regardless of current flip state), not the tip.
            var mw = e.target.closest('.missing-tip-wrap');
            if (mw) {
                var mt = mw.querySelector('.missing-tip');
                if (mt) {
                    var clipTop = 0, anc = mw.parentElement;
                    while (anc) {
                        var oy = getComputedStyle(anc).overflowY;
                        if (oy === 'auto' || oy === 'scroll' || oy === 'hidden') { clipTop = anc.getBoundingClientRect().top; break; }
                        anc = anc.parentElement;
                    }
                    var wr = mw.getBoundingClientRect();
                    if (wr.top - mt.offsetHeight - 4 < clipTop) mt.classList.add('mt-flip');
                    else mt.classList.remove('mt-flip');
                }
            }
        });
        document.addEventListener('click', function(e) {
            var trigger = e.target.closest('.colony-badge, .plate-trigger');
            if (trigger) {
                e.stopPropagation();
                var container = trigger.closest('.plate-hover-container');
                var popover = container.querySelector('.plate-popover');
                _fillPop(popover);
                if (popover.classList.contains('sticky')) { popover.classList.remove('sticky'); }
                else { document.querySelectorAll('.plate-popover.sticky').forEach(function(p) { p.classList.remove('sticky'); }); popover.classList.add('sticky'); }
                return;
            }
            if (e.target.closest('.plate-popover')) { e.stopPropagation(); return; }
            document.querySelectorAll('.plate-popover.sticky').forEach(function(p) { p.classList.remove('sticky'); });
            if (e.target.closest('.ci-tip')) { e.stopPropagation(); return; }
            if (e.target.closest('.ci-wrap')) {
                e.stopPropagation();
                var tip = e.target.closest('.ci-wrap').querySelector('.ci-tip');
                var isOpen = tip.classList.contains('ci-open');
                document.querySelectorAll('.ci-tip.ci-open').forEach(function(t) { t.classList.remove('ci-open'); t.classList.remove('ci-flip'); });
                if (!isOpen) {
                    tip.classList.add('ci-open');
                    // Default opens upward; flip downward if the top would be clipped
                    // by a scrolling/overflow ancestor (e.g. .content-pane overflow-x:auto).
                    var clipTop = 0, anc = tip.parentElement;
                    while (anc) {
                        var oy = getComputedStyle(anc).overflowY;
                        if (oy === 'auto' || oy === 'scroll' || oy === 'hidden') { clipTop = anc.getBoundingClientRect().top; break; }
                        anc = anc.parentElement;
                    }
                    if (tip.getBoundingClientRect().top < clipTop) tip.classList.add('ci-flip');
                }
                return;
            }
            document.querySelectorAll('.ci-tip.ci-open').forEach(function(t) { t.classList.remove('ci-open'); });
        });
    });
    </script>
    """

    # Splice token-generated status + timeline-dot CSS into the static block.
    html = html.replace("/*__STATUS_CSS__*/", _status_css).replace("/*__TDOT_CSS__*/", _tdot_css)

    # LSP Capacity tab (generated once, injected into template)
    lsp_capacity_html = render_lsp_capacity_tab(df)

    # Requests In Flight tab — rendered as a plain HTML fragment, injected directly
    _inflight_fragment = render_inflight_tab(df)

    # Parts inventory tab — plain HTML fragment; reads its own parts_result.pkl (separate pull)
    _parts_fragment = render_parts_tab()

    # Part 2: HTML with variables (f-string)
    html += f"""
    <div class="dashboard-container">
        <!-- HEADER WITH LOGO -->
        <div class="dashboard-header">
            <img src="data:image/png;base64,{logo_b64}" class="dashboard-logo" alt="DNASC">
            <span class="dashboard-title">DNA Strain & Construction</span>
            <span class="dashboard-updated"><span class="du-label">Data pulled</span><span class="du-time">{generated_at}</span></span>
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
                <button class="tab-btn" data-tab="inflight" onclick="switchTab('inflight')">
                    <span style="font-size:16px;">🗂️</span>
                    <span class="tab-text">Requests In Flight</span>
                </button>
                <button class="tab-btn" data-tab="parts" onclick="switchTab('parts')">
                    <span style="font-size:16px;">🧬</span>
                    <span class="tab-text">Parts</span>
                </button>
            </div>
            <script>(function(){{try{{var t=localStorage.getItem('dash_activeTab');if(t&&t!=='tracking'){{document.querySelector('[data-tab="tracking"]').classList.remove('active');var b=document.querySelector('[data-tab="'+t+'"]');if(b)b.classList.add('active');var s=document.createElement('style');s.id='_earlyhide';s.textContent='#tab-tracking{{display:none!important}}';document.head.appendChild(s);}}}}catch(e){{}}}}());</script>
            <!-- TRACKING TAB -->
            <div id="tab-tracking" class="tab-content active">
                <div class="controls-container">
                    <div class="toggle-wrapper" style="margin-right: auto; flex-direction: column; align-items: flex-start; gap: 2px; position: relative;">
                        <input type="text" id="search_box" placeholder="Search Stock ID, Experiment, or Construct..." oninput="filterDashboardDebounced()">
                        <span id="search_count" style="font-size:10px;color:#64748b;white-space:nowrap;display:none;padding-left:2px;"></span>
                        <div style="display:flex;gap:4px;align-items:center;">
                            <button id="search_download" onclick="downloadSearchCSV()" style="display:none;font-size:9px;padding:2px 6px;border-radius:3px;border:1px solid #e5e7eb;background:#fff;color:#374151;cursor:pointer;white-space:nowrap;">&#8595; CSV</button>
                            <button id="search_list_btn" onclick="toggleSearchList()" style="display:none;font-size:9px;padding:2px 6px;border-radius:3px;border:1px solid #e5e7eb;background:#fff;color:#374151;cursor:pointer;white-space:nowrap;">&#9776; List</button>
                        </div>
                        <div id="search_list_popup" style="display:none;position:absolute;top:100%;left:0;margin-top:4px;background:#fff;border:1px solid #e5e7eb;border-radius:6px;box-shadow:0 4px 16px rgba(0,0,0,0.12);padding:10px 12px;z-index:999;min-width:700px;max-width:90vw;max-height:60vh;overflow-y:auto;font-family:inherit;"></div>
                    </div>
                    <div class="toggle-wrapper">
                        <span class="toggle-label">Active Projects Only</span>
                        <label class="switch">
                            <input type="checkbox" id="active_toggle" onclick="filterDashboard()">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div class="toggle-wrapper">
                        <button id="sort_due_btn" onclick="sortByDueDate()" style="font-size:10px;font-weight:700;padding:4px 10px;border-radius:4px;border:1px solid #e5e7eb;background:#fff;color:#1d1d1f;cursor:pointer;white-space:nowrap;">Sort: Due Date</button>
                    </div>
                </div>
                <div id="projects-container" style="padding: 10px; display: flex; flex-direction: column;">
                  <div style="display:flex; gap:16px; align-items:center; justify-content:center; flex-wrap:wrap; padding:9px 14px; margin:0 0 12px 0; background:#fff; border:1px solid #e5e7eb; border-radius:8px; box-shadow:0 1px 3px rgba(15,23,42,0.06); font-size:10px; font-weight:600; color:#374151;"><span style="color:#6b7280;">IN PROGRESS:</span><span style="display:flex;align-items:center;gap:4px;"><span style="width:9px;height:9px;background:#0891b2;border-radius:50%;"></span>On Track</span><span style="display:flex;align-items:center;gap:4px;"><span style="width:9px;height:9px;background:#f97316;border-radius:50%;"></span>Warning</span><span style="display:flex;align-items:center;gap:4px;"><span style="width:9px;height:9px;background:#be185d;border-radius:50%;"></span>Overdue</span><span style="width:1px;height:12px;background:#cbd5e1;"></span><span style="color:#6b7280;">FULFILLED:</span><span style="display:flex;align-items:center;gap:4px;"><span style="width:9px;height:9px;background:#0891b2;border-radius:2px;transform:rotate(45deg);"></span>On Time</span><span style="display:flex;align-items:center;gap:4px;"><span style="width:9px;height:9px;background:#be185d;border-radius:2px;transform:rotate(45deg);"></span>Late</span><span style="width:1px;height:12px;background:#cbd5e1;"></span><span style="display:flex;align-items:center;gap:5px;"><span style="width:16px;height:9px;border-radius:3px;background:linear-gradient(90deg,#7461b8,#a05f8a);"></span>Partner</span><span style="display:flex;align-items:center;gap:5px;"><span style="width:16px;height:9px;border-radius:3px;background:linear-gradient(90deg,#3a5c7a,#3d8aa2);"></span>R&amp;D</span><span style="width:1px;height:12px;background:#cbd5e1;"></span><span style="color:#6b7280;">TIMELINE:</span><span style="display:flex;align-items:center;gap:4px;"><span style="width:16px;height:9px;background:repeating-linear-gradient(45deg,#64748b,#64748b 3px,#cbd5e1 3px,#cbd5e1 6px);border:1px solid #cbd5e1;"></span>upstream seq&rarr;dnasc (not in TAT)</span><span style="display:flex;align-items:center;gap:4px;"><span style="width:3px;height:11px;background:#334155;display:inline-block;"></span>0w = dnasc created (entry)</span><span style="display:flex;align-items:center;gap:4px;"><span style="width:9px;height:9px;background:#10b981;border-radius:50%;"></span>now</span></div>
    """

    # =========================================================================
    # 3. HELPER: RENDER SINGLE REQUEST
    # =========================================================================
    def render_single_request_html(req_id, req_df, is_stalled=False, is_asm_review=False, has_seq_winner=False, has_order_pending=False, has_antibiotic_mismatch=False, has_dual_antibiotic=False):
        html = []
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
        # Also include fulfills_request=True stocks (catches LSP STOCK_IDs distinct from the GG root)
        if 'fulfills_request' in req_df.columns:
            _fr_stocks = set(
                req_df[
                    (req_df['fulfills_request'] == True) &
                    req_df['STOCK_ID'].notna() &
                    ~req_df['STOCK_ID'].fillna('').str.startswith('#')
                ]['STOCK_ID'].astype(str)
            ) - {'nan', 'None', 'N/A', ''}
            all_root_stocks = all_root_stocks | _fr_stocks
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
        partner_badge = '<span class="badge" style="background:#ede9fe;color:#7c3aed;margin-left:12px;margin-right:8px;border:1px solid #c4b5fd;">PARTNER</span>' if is_partner_req else ""

        # Customer badge — green R&D + leading dot (shape cue), canonical geometry
        # (mixed-case, no .badge class). Sourced from renderer/tokens.py.
        _cg = tok.GEOM['customer']
        _cust_geom = (f"display:inline-block;font-size:{_cg['size']};font-weight:{_cg['weight']};"
                      f"padding:{_cg['pad']};border-radius:{_cg['radius']};margin-right:4px;white-space:nowrap;")
        _cust_raw = str(req_df['customer'].iloc[0]) if 'customer' in req_df.columns and pd.notna(req_df['customer'].iloc[0]) else None
        if _cust_raw and _cust_raw not in ('nan', 'None', ''):
            _clabel, _cbg, _cfg = tok.CUSTOMER.get(_cust_raw, (_cust_raw.replace('_', ' '),) + tok.CUSTOMER_FALLBACK[1:])
            customer_badge = f'<span style="{_cust_geom}background:{_cbg};color:{_cfg};">{tok.CUSTOMER_DOT}{_clabel}</span>'
        else:
            customer_badge = ""

        # --- UPDATED CONTEXT-AWARE STATUS SELECTION ---
        # Compute render-time effective status for each row so colony-type overrides
        # (e.g. streakout with colonies but no seq → RUNNING) are reflected in phase
        # detection, not just in the per-row display.
        req_df = req_df.copy()

        # First-match workorder_id → row dict (only the columns read by scalar
        # lookups below), built once via columnar zip. Replaces repeated
        # `req_df[req_df['workorder_id']==x].iloc[0]` boolean-mask scans.
        _RBW_COLS = ('construct_name', 'experiment_name', 'job_id', 'type',
                     'wo_status', 'root_work_order_id', 'assembly_plan_id',
                     'wo_created_at', 'STOCK_ID', 'backbone', 'parts',
                     'attempt_anchor_id', 'attempt_number')
        _rbw_cols = [c for c in _RBW_COLS if c in req_df.columns]
        _rbw_wids = req_df['workorder_id'].tolist()
        _rbw_data = {c: req_df[c].tolist() for c in _rbw_cols}
        record_by_wid: dict = {}
        for _i in range(len(_rbw_wids)):
            _w = _rbw_wids[_i]
            if _w not in record_by_wid:
                record_by_wid[_w] = {c: _rbw_data[c][_i] for c in _rbw_cols}

        status_badge_html = ""

        phase_label  = str(req_df['req_phase'].iloc[0]     or '') if 'req_phase'     in req_df.columns else ''
        display_text = str(req_df['req_operation'].iloc[0]  or '') if 'req_operation'  in req_df.columns else ''
        op_status    = str(req_df['req_op_status'].iloc[0]  or '') if 'req_op_status'  in req_df.columns else ''

        if phase_label:
            # PARTS light variant: no WAITING GG → only synthesis workorders exist yet
            _asm_types_set = {'golden_gate_workorder', 'gibson_workorder'}
            _has_waiting_gg = not req_df[
                req_df['type'].isin(_asm_types_set) & (req_df['visual_status'] == 'WAITING')
            ].empty

            # Phase pills — Kernel palette (LSP=blue, ASM=orange, PARTS=teal),
            # AA-safe tint + accent border, 9px. Sourced from renderer/tokens.py.
            phase_bg, phase_color, phase_border = tok.PHASE.get(phase_label, ('#f8fafc', '#6b7280', '#e5e7eb'))
            _pg = tok.GEOM['phase']

            phase_html = f'''
                <span style="
                    background: {phase_bg}; color: {phase_color}; border: 1px solid {phase_border};
                    padding: {_pg['pad']}; border-radius: {_pg['radius']};
                    margin-left: 6px; font-weight: {_pg['weight']}; font-size: {_pg['size']};
                    text-transform: uppercase;
                    display: inline-flex; align-items: center;
                ">{phase_label}</span>
            '''

            if display_text:
                if not op_status:
                    op_status = 'READY' if display_text.endswith('READY') else ('RUNNING' if display_text.endswith('RUNNING') else 'WAITING')
                status_badge_html = f'''
                    <span class="badge status-{op_status}" style="
                        font-size: 8px; padding: 1px 4px;
                        display: inline-flex; align-items: center; line-height: 1;
                        box-sizing: border-box; vertical-align: middle;
                        white-space: nowrap; font-weight: 700; border-radius: 2px;
                    ">{display_text}</span>{phase_html}
                '''
            else:
                status_badge_html = phase_html
        req_created = to_est(req_df['request_created_at'].iloc[0])
        submitted_str = req_created.strftime('%Y-%m-%d') if req_created else "N/A"
        _req_email_raw = req_df['submitter_email'].dropna().iloc[0] if 'submitter_email' in req_df.columns and not req_df['submitter_email'].dropna().empty else ''
        _req_email = str(_req_email_raw).strip() if _req_email_raw else ''
        now = datetime.now(pytz.timezone('US/Eastern'))
        is_done = str(req_status).upper() in ['FULFILLED', 'SUCCEEDED', 'CANCELED']
        # A finished request has no active operation — drop the phase pill (ASM/PARTS/LSP)
        # and the op badge so FULFILLED/SUCCEEDED/CANCELED rows don't read as "still in ASM".
        if is_done:
            status_badge_html = ""

        stalled_badge = '<div class="badge-tip-wrap"><span class="badge" style="background:#dc2626; color:white; border:2px solid #b91c1c; font-size:12px; padding:4px 12px; font-weight:800;">⚠️ STALLED</span><div class="badge-tip">No pipeline progress detected — may need intervention</div></div>' if is_stalled else ""
        asm_review_badge = '<div class="badge-tip-wrap"><span class="badge" style="background:#d97706; color:white; border:2px solid #b45309; font-size:12px; padding:4px 12px; font-weight:800;">🔬 ASM REVIEW</span><div class="badge-tip">Assembly needs review before proceeding</div></div>' if is_asm_review else ""
        seq_winner_badge = '<div class="badge-tip-wrap"><span class="badge" style="background:#059669; color:white; border:2px solid #047857; font-size:12px; padding:4px 12px; font-weight:800;">🏆 SEQ WINNER</span><div class="badge-tip">A sequencing winner has been identified — ready for LSP</div></div>' if has_seq_winner else ""
        order_pending_badge = '<div class="badge-tip-wrap"><span class="badge" style="background:#7c3aed; color:white; border:2px solid #6d28d9; font-size:12px; padding:4px 12px; font-weight:800;">⏳ ORDER PENDING</span><div class="badge-tip">Parts order submitted to synthesis vendor — waiting on delivery</div></div>' if has_order_pending else ""
        antibiotic_mismatch_badge = '<div class="badge-tip-wrap"><span class="badge" style="background:#dc2626; color:white; border:2px solid #b91c1c; font-size:12px; padding:4px 12px; font-weight:800;">🚨 ANTIBIOTIC MISMATCH</span><div class="badge-tip">An active workorder has an antibiotic that does not match LIMS — check and correct before proceeding</div></div>' if has_antibiotic_mismatch else ""
        dual_antibiotic_badge = '<div class="badge-tip-wrap"><span class="badge" style="background:#fef3c7; color:#92400e; border:2px solid #f59e0b; font-size:12px; padding:4px 12px; font-weight:800;">⚠️ DUAL ANTIBIOTIC (LIMS)</span><div class="badge-tip">LIMS lists two bacterial antibiotics on this plasmid — often the NeoR mammalian marker mis-flagged as Kan. The correct one is selected, but the BIOS/LIMS record should be corrected.</div></div>' if has_dual_antibiotic else ""

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
                        if name == proto.LSP_REVIEWING and state == 'SC': ready_to_ship_time = to_est(start)
                        if name == proto.LSP_RELEASING and state == 'SC': final_release_time = to_est(start)

        time_badges_html = ""
        shared_time_style = "display: flex; align-items: center; background: #f1f5f9; border: 1px solid #cbd5e1; padding: 1px 4px; border-radius: 2px; gap: 3px; height: 16px;"
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
                          <span class="u9">Production:</span>
                          <span class="u10">{production_str}</span>
                      </div>
                      <div style="{shared_time_style}">
                          <span class="u9">Total:</span>
                          <span class="u10">{total_str}</span>
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
        html.append(f"""
        <div class="req-card">
            <div class="req-title-bar" style="background: {req_bg_color}; border-left-color: {req_border_left};">
                <div style="flex-grow: 1;">
                    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: nowrap;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            {pais_display}
                        </div>
                        <span class="req-name">{construct}</span>
                        <div style="display: flex; align-items: center; gap: 8px;">

                            <div class="u5">
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
                <div style="display: flex; gap: 8px; align-items: center; flex-shrink: 0; margin-left: auto;">
                    {customer_badge}{partner_badge}
                    <span class="badge status-{str(req_status).replace(" ", "_")}">{req_status}</span>
                    {status_badge_html}
                    {stalled_badge}
                    {asm_review_badge}
                    {seq_winner_badge}
                    {order_pending_badge}
                    {antibiotic_mismatch_badge}
                    {dual_antibiotic_badge}
                </div>
            </div>
        """)

        html.append(f"""
        <div style="padding: 3px 6px; background: #f8fafc; border-bottom: 1px solid #e5e7eb; cursor: pointer;" onclick="toggleSection('req_{req_id.replace("-", "_")}')">
            <span id="req_{req_id.replace("-", "_")}_icon" class="dropdown-icon">▶</span>
            <span style="font-size: 9px; font-weight: 600; color: #86868b;">Workorder Details</span>
        </div>
        <div id="req_{req_id.replace("-", "_")}" style="display: none;">
        """)

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
            # For assembly GG/Gibson roots, rank by the root workorder's own status,
            # not the minimum across all rows (sub-assembly BLOCKED parts would otherwise
            # pull the whole section's rank up, hiding newer active attempts).
            _root_own_rows = r_df[r_df['workorder_id'] == root_id]
            _root_own_type = str(_root_own_rows['type'].iloc[0]) if not _root_own_rows.empty else ''
            if _root_own_type in ('golden_gate_workorder', 'gibson_workorder') and not _root_own_rows.empty:
                _root_own_status = str(_root_own_rows['visual_status'].iloc[0])
                root_rank = status_priority.get(_root_own_status, 99)
            else:
                root_rank = min_rank
            _root_ts = _root_own_rows['wo_created_at'].iloc[0] if not _root_own_rows.empty else None
            try: _root_ts_val = _root_ts.timestamp() if _root_ts is not None and pd.notna(_root_ts) else 0
            except Exception: _root_ts_val = 0
            root_status_map[root_id] = {'is_winner': is_winner, 'rank': root_rank, 'ts': _root_ts_val}
        sorted_roots = sorted(root_status_map.keys(), key=lambda r: (
            not root_status_map[r]['is_winner'],
            root_status_map[r]['rank'],
            -root_status_map[r].get('ts', 0)   # newest first within same rank
        ))

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
            _rrow = record_by_wid.get(_r)
            if _rrow is None:
                continue
            if _rrow.get('type', '') not in _asm_types_req:
                continue
            if str(_rrow.get('wo_status', '') or '') in ('DRAFT',):
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
        # Build anchor → sub-roots map from attempt_anchor_id (pre-computed in BQ).
        # Falls back to renderer-side design-key grouping for old parquets.
        _cross_plan_sub_roots: set = set()
        _cross_plan_sub_roots_for: dict = {}  # anchor_root → [sub_roots sorted by attempt_number]

        _has_anchor_col = "attempt_anchor_id" in req_df.columns

        if _has_anchor_col:
            # New path: read BQ-pre-computed grouping — single pass, no design-key logic
            _anchor_rows: dict = {}
            for _r, _rplan, _rts, _rstock, _rbb, _rparts in _req_asm_roots_info:
                _rrow = record_by_wid.get(_r)
                if _rrow is None: continue
                _anchor = str(_rrow.get("attempt_anchor_id") or _r)
                if _anchor in ("nan", "None", ""): _anchor = _r
                _anum = _rrow.get("attempt_number")
                _anum = int(_anum) if pd.notna(_anum) else 1
                _anchor_rows.setdefault(_anchor, []).append((_anum, _r))
            for _anchor, _pairs in _anchor_rows.items():
                if len(_pairs) > 1:
                    _pairs.sort(key=lambda x: x[0])
                    for _, _r in _pairs[1:]:
                        _cross_plan_sub_roots.add(_r)
                        _cross_plan_sub_roots_for.setdefault(_anchor, []).append(_r)
        else:
            # Fallback: renderer-side design-key grouping (same-plan then cross-plan)
            _req_asm_by_plan: dict = {}
            for _r, _rplan, _rts, _rstock, _rbb, _rparts in _req_asm_roots_info:
                _req_asm_by_plan.setdefault((_rplan, _rbb, _rparts), []).append((_r, _rts))
            for (_plan, _bb, _pts), _ars in _req_asm_by_plan.items():
                if len(_ars) > 1:
                    _ars.sort(key=lambda x: x[1])
                    _primary = _ars[0][0]
                    for _r, _ in _ars[1:]:
                        _cross_plan_sub_roots.add(_r)
                        _cross_plan_sub_roots_for.setdefault(_primary, []).append(_r)
            _req_asm_by_design_xplan: dict = {}
            for _r, _rplan, _rts, _rstock, _rbb, _rparts in _req_asm_roots_info:
                if not _rstock or _r in _cross_plan_sub_roots: continue
                _req_asm_by_design_xplan.setdefault((_rstock, _rbb, _rparts), []).append((_r, _rplan, _rts))
            for _design_key, _ars in _req_asm_by_design_xplan.items():
                if len(_ars) > 1:
                    _ars.sort(key=lambda x: x[2])
                    _primary = _ars[0][0]
                    for _r, _, _ in _ars[1:]:
                        _cross_plan_sub_roots.add(_r)
                        _cross_plan_sub_roots_for.setdefault(_primary, []).append(_r)
                    for _r, _, _ in _ars:
                        _req_asm_attempt_map.pop(_r, None)

        for root_id in sorted_roots:
            if root_id in _cross_plan_sub_roots:
                continue
            # Hide DRAFT GG/Gibson sections only when every row in the section is also
            # DRAFT — meaning no real workorder was ever executed for any input.
            # If even one input has any non-DRAFT status (WAITING, READY, RUNNING,
            # SUCCEEDED, CANCELED, etc.) a real workorder ran and we keep the section.
            _root_own = req_df[req_df['workorder_id'] == root_id]
            if not _root_own.empty:
                _root_type = str(_root_own['type'].iloc[0])
                _root_ws   = str(_root_own['wo_status'].iloc[0])
                if _root_type in ('golden_gate_workorder', 'gibson_workorder') and _root_ws == 'DRAFT':
                    _section_rows = req_df[req_df['root_work_order_id'] == root_id]
                    _all_draft = _section_rows['wo_status'].fillna('DRAFT').str.upper().eq('DRAFT').all()
                    if _all_draft:
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
                _cand_parts = _parts_by_root_dfs.get(root_id)
                if _cand_parts is not None:
                    _xreq_parts = _cand_parts[
                        ~_cand_parts['workorder_id'].isin(req_df['workorder_id'])
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
            # Serve each row from the prebuilt index→record map (shallow-copied so
            # later per-section mutations stay isolated). Iterating root_df.index in
            # order preserves the dict-comprehension's last-wins-per-wid semantics.
            try:
                row_map = {}
                for _ridx in root_df.index:
                    _rec = dict(_rec_by_index[_ridx])
                    row_map[_rec['workorder_id']] = _rec
            except KeyError:
                # Any row not sourced from df (shouldn't happen) — exact fallback.
                row_map = {row['workorder_id']: row for row in root_df.to_dict('records')}
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

            # Assign attempt numbers from BQ-pre-computed columns when available,
            # otherwise fall back to design-key grouping within the section.
            if len(visible_asm_roots) > 1:
                if _has_anchor_col:
                    # Group visible roots by attempt_anchor_id and re-number sequentially.
                    # BQ attempt_total counts all anchored roots including CANCELED+no-work
                    # ones excluded from visible_asm_roots, so we recompute from visible count.
                    _anchor_groups: dict = {}
                    for _ar in visible_asm_roots:
                        _anum_bq = row_map[_ar].get('attempt_number')
                        _atot_bq = row_map[_ar].get('attempt_total')
                        if pd.notna(_anum_bq) and pd.notna(_atot_bq) and int(_atot_bq or 1) > 1:
                            _anchor = row_map[_ar].get('attempt_anchor_id') or _ar
                            _anchor_groups.setdefault(_anchor, []).append((_anum_bq, _ar))
                    for _anchor_roots in _anchor_groups.values():
                        _anchor_roots.sort()  # sort by BQ attempt_number → temporal order
                        _vis_total = len(_anchor_roots)
                        for _ai, (_, _ar) in enumerate(_anchor_roots, 1):
                            row_map[_ar]['_attempt_number'] = _ai
                            row_map[_ar]['_attempt_total']  = _vis_total
                            if _ai > 1:
                                _ar_resubmit = row_map[_ar].get('resubmit_count')
                                _ar_is_retry = (isinstance(_ar_resubmit, (int, float))
                                                and not pd.isna(_ar_resubmit)
                                                and int(_ar_resubmit) > 0)
                                row_map[_ar]['_attempt_kind'] = 'RETRY' if _ar_is_retry else 'MANUAL'
                else:
                    # Fallback: derive from design-key grouping within section
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
            # Always assign chain status for assembly roots (even single-attempt, used
            # in banner). Prefer the pipeline-computed `chain_status` column (single
            # source of truth shared with the colony tab); fall back to the render-time
            # subtree walk for parquet snapshots predating that column.
            for _ar in visible_asm_roots:
                _cs = row_map[_ar].get('chain_status')
                if _cs is not None and not (isinstance(_cs, float) and pd.isna(_cs)) and str(_cs):
                    row_map[_ar]['_attempt_chain_status'] = str(_cs)
                else:
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
                    # Group as attempts when: BIOS auto-retry (resubmit_count > 0) OR
                    # manual retry (FAILED/CANCELED PCR followed by a new one).
                    # Multiple all-SUCCEEDED PCRs for the same stock = parallel scale-up, not retries.
                    _has_resubmit = any(
                        (row_map[x].get('resubmit_count') or 0) > 0
                        for x in _prs
                    )
                    _has_failure = any(
                        row_map[x].get('visual_status', '') in ('FAILED', 'CANCELED')
                        for x in _prs
                    )
                    if not _has_resubmit and not _has_failure:
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
                # Filter fanned-in parts by stock match.
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
            sorted_records = ordered_data + parts_ordered

            # Skip root groups where every row is CANCELED with no queue data (empty dropdown)
            def _row_is_visible(r):
                if r.get('wo_status') != 'CANCELED': return True
                _pn = r.get('protocol_name')
                if hasattr(_pn, 'tolist'): _pn = _pn.tolist()
                return isinstance(_pn, list) and len(_pn) > 0
            if not sorted_records or not any(_row_is_visible(r) for r in sorted_records):
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
                elif sorted_records:
                    target_row = sorted_records[0]
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
            # An assembly attempt whose own workorder is FAILED/CANCELED can still
            # have succeeded downstream — e.g. a child transformation produced a
            # seq-confirmed colony (a1763876: Gibson CANCELED, transformation
            # 0ba2439c SUCCEEDED). The attempt banners already reflect this via
            # _attempt_chain_status; mirror it in the section-header badge so the
            # roll-up shows SUCCEEDED when any attempt's chain succeeded, instead
            # of the assembly root's own FAILED/CANCELED status.
            if status in ('FAILED', 'CANCELED') and target_row['type'] != 'lsp_workorder':
                _succ_root = next(
                    (r for r in visible_asm_roots
                     if row_map[r].get('_attempt_chain_status') == 'SUCCEEDED'),
                    None,
                )
                if _succ_root is not None:
                    target_row = row_map[_succ_root]
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
                    header_extra_info = f"<div style='font-size:10px; font-weight:800; margin-top:4px; color:#059669; text-align:center;'>{active_info}</div>"
                s_class = f"status-LSP_{status}" if status == 'RUNNING' else f"status-{status}"
            else:
                if active_info:
                    if 'Ready' in active_info and status == 'RUNNING': status = 'READY'
                    _ei_color = '#ea580c' if target_row['type'] in _parts_types_set else '#2563eb'
                    header_extra_info = f"<div style='font-size:10px; font-weight:800; margin-top:4px; color:{_ei_color}; text-align:center;'>{active_info}</div>"
                s_class = f"status-{status}"
            badges_html += f'<div style="text-align:right"><span class="badge {s_class}"><b>{f_type}: {status}</b></span>{header_extra_info}</div>'
            if 'antibiotic_mismatch' in root_df.columns:
                _ab_mis = root_df[
                    root_df['type'].isin(['golden_gate_workorder', 'gibson_workorder', 'transformation_workorder'])
                    & root_df['antibiotic_mismatch'].eq(True)
                    & ~root_df['visual_status'].isin(['SUCCEEDED', 'FAILED', 'CANCELED'])
                ]
                if not _ab_mis.empty:
                    badges_html += '<div style="text-align:right;margin-top:2px;"><span style="font-size:10px;font-weight:800;color:#b91c1c;background:#fee2e2;border:1px solid #fca5a5;border-radius:3px;padding:1px 6px;">🚨 ANTIBIOTIC MISMATCH</span></div>'
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

            # Extra deliverable stocks from LSP workorders in this section (e.g. pAI-21929 from a child LSP)
            _lsp_del = set(
                root_df[
                    (root_df['type'] == 'lsp_workorder') &
                    (root_df['fulfills_request'] == True) &
                    root_df['STOCK_ID'].notna() &
                    ~root_df['STOCK_ID'].fillna('').str.startswith('#')
                ]['STOCK_ID'].astype(str)
            ) - {'nan', 'None', 'N/A', '', str(root_stock)}
            _extra_stock_tags = "".join(
                f'<span class="stock-tag" style="font-size:9px; padding:3px 8px; background:#ddd6fe; color:#4c1d95; border: 1px solid #7c3aed;">{s}</span>'
                for s in sorted(_lsp_del)
            )

            html.append(f"""<div class="{section_class}"><button id="{div_id}_btn" class="dropdown-btn" onclick="toggleSection('{div_id}')"><span id="{div_id}_icon" class="dropdown-icon">▶</span><div class="assembly-info"><span class="assembly-type">{assembly_label}</span><span class="stock-tag" style="font-size:9px; padding:3px 8px; background:#ede9fe; color:#6d28d9; border: 1px solid #c4b5fd;">{root_stock}</span>{_extra_stock_tags}<span class="assembly-counts" style="font-weight: 600;">{count_str}</span><span class="wo-id-tag">Root: {root_id}</span></div><div class="status-badges">{badges_html}</div></button><div id="{div_id}" class="content-pane"><table class="wo-table"><thead><tr><th>Type</th><th>Workorder ID</th><th>Status</th><th>Stock ID</th><th>Created</th><th>TAT</th><th>Details</th><th style="width: 350px;">Queue</th></tr></thead><tbody>""")

            _emitted_parts_header = False
            _emitted_retry_header = False
            # Pre-check: does this section have any non-CANCELED rows other than the root?
            # Used to show a CANCELED root assembly that has no lab data of its own but
            # whose inputs DID have work done (e.g. Gibson canceled after parts were built).
            _section_inputs_have_work = any(
                r.get('wo_status') != 'CANCELED'
                for r in sorted_records
                if str(r.get('workorder_id', '')) != str(root_id)
            )
            tree_stock_ids = {str(r['STOCK_ID']) for r in sorted_records if pd.notna(r.get('STOCK_ID'))}
            # STOCK_IDs that have at least one non-CANCELED part row in this section.
            # A CANCELED part whose fragment is NOT covered here is the sole attempt for
            # a required input (keep it — it's real missing-input context); a CANCELED
            # part duplicating an already-covered fragment is just noise (skip it).
            _covered_part_stocks = {
                str(r.get('STOCK_ID')) for r in sorted_records
                if pd.notna(r.get('STOCK_ID')) and r.get('wo_status') != 'CANCELED'
                and (r.get('type') in _parts_types_dfs
                     or (r.get('type') in _asm_types_dfs and r.get('fulfills_request') == False))
            }
            for row in sorted_records:
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
                        _is_canceled_part = (row.get('type') in _parts_types_dfs
                                             or (row.get('type') in _asm_types_dfs
                                                 and row.get('fulfills_request') == False))
                        if (row.get('fulfills_request') == True
                                and row.get('type') in ('gibson_workorder', 'golden_gate_workorder')
                                and _section_inputs_have_work):
                            pass  # show it — parts had work done
                        elif (_is_canceled_part
                              and pd.notna(row.get('STOCK_ID'))
                              and str(row.get('STOCK_ID')) not in _covered_part_stocks):
                            pass  # show it — sole attempt for a required fragment (missing-input context)
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
                    html.append(f"""<tr><td colspan="8" class="u14"><span style="font-size:10px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em;">Parts / Inputs</span></td></tr>""")

                if _is_parts_row and _has_attempt and row.get('tree_depth', 0) == 0:
                    # "Retried Parts" divider — emitted once before the first retry attempt group
                    if not _emitted_retry_header:
                        _emitted_retry_header = True
                        html.append(f"""<tr><td colspan="8" style="padding:5px 10px 4px; background:#fef9ec; border-top:2px solid #fde68a; border-bottom:1px solid #fde68a;"><span style="font-size:10px; font-weight:700; color:#92400e; text-transform:uppercase; letter-spacing:0.05em;">Retried Parts</span></td></tr>""")
                    _atype = row.get('type', '')
                    _alabel = format_type_label(_atype)
                    _attempt_total = int(row.get('_attempt_total', 1))
                    _sid_raw = row.get('STOCK_ID')
                    _sid_lbl = '' if (_sid_raw is None or (isinstance(_sid_raw, float) and pd.isna(_sid_raw)) or str(_sid_raw).lower() in ('nan', 'none', '') or str(_sid_raw).startswith('#')) else str(_sid_raw).strip()
                    _vstatus = row.get('visual_status', '') or ''
                    _vstatus = '' if (isinstance(_vstatus, float) and pd.isna(_vstatus)) else str(_vstatus)
                    _vcolor = _status_colors.get(_vstatus, '#64748b')
                    _vicon = '✓ ' if _vstatus == 'SUCCEEDED' else '✗ ' if _vstatus == 'FAILED' else ''
                    html.append(f"""<tr><td colspan="8" style="padding:4px 10px; background:#f1f5f9; border-top:1px solid #cbd5e1; border-bottom:1px solid #e2e8f0;"><span style="font-size:10px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.04em;">{(_sid_lbl + ' — ') if _sid_lbl else ''}Attempt {int(_attempt_num)} of {_attempt_total}</span><span style="margin-left:8px; font-size:10px; font-weight:600; color:{_vcolor};">{_vicon}{_vstatus}</span></td></tr>""")

                elif not _is_parts_row and _has_attempt and row.get('tree_depth', 0) == 0:
                    # Assembly attempt banner (GG/Gibson) — uses best chain status
                    _atype = row.get('type', '')
                    _alabel = format_type_label(_atype)
                    _attempt_total = int(row.get('_attempt_total', 1))
                    _cs_raw = row.get('_attempt_chain_status')
                    _astatus = row.get('visual_status', '') if (_cs_raw is None or (isinstance(_cs_raw, float) and pd.isna(_cs_raw))) else str(_cs_raw)
                    _astatus_color = _status_colors.get(_astatus, '#64748b')
                    _status_icon = '✓ ' if _astatus == 'SUCCEEDED' else '✗ ' if _astatus == 'FAILED' else ''
                    html.append(f"""<tr><td colspan="8" style="padding:4px 10px; background:#f1f5f9; border-top:2px solid #cbd5e1; border-bottom:1px solid #e2e8f0;"><span style="font-size:10px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.04em;">{_alabel} — Attempt {int(_attempt_num)} of {_attempt_total}</span><span style="margin-left:8px; font-size:10px; font-weight:600; color:{_astatus_color};">{_status_icon}{_astatus}</span></td></tr>""")
                depth = row.get('tree_depth', 0)
                row_class = f"tree-row-{min(depth, 2)}"  # CSS only has 0, 1, 2
                _row_order_style = ""
                if row['type'] in ('syn_part_synthesis_workorder', 'oligo_synthesis_workorder', 'plasmid_synthesis_workorder'):
                    _rpn = row.get('protocol_name'); _rps = row.get('operation_state')
                    if isinstance(_rpn, np.ndarray): _rpn = _rpn.tolist()
                    if isinstance(_rps, np.ndarray): _rps = _rps.tolist()
                    _row_ap = {p for p, s in zip(_rpn, _rps) if s in ('RD', 'RU')} if isinstance(_rpn, list) and isinstance(_rps, list) else set()
                    if _row_ap & proto.ORDER_PROTOS:
                        try:
                            from datetime import timedelta as _otd
                            _rc = pd.Timestamp(row.get('wo_created_at'))
                            if _rc.tzinfo is None:
                                _rc = _rc.tz_localize('UTC')
                            if (datetime.now(pytz.UTC) - _rc.to_pydatetime()) > _otd(hours=4):
                                _row_order_style = ' style="background:#faf5ff !important; border-left:3px solid #7c3aed !important;"'
                        except Exception:
                            pass
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
                        for _pkey, plate_str in data.items():
                            if plate_str:
                                pids = [clean_plate_id(p) for p in str(plate_str).split(',') if clean_plate_id(p)]
                                if _pkey not in lims_plate_map: lims_plate_map[_pkey] = []
                                lims_plate_map[_pkey].extend(pids)
                    except: pass
                for col in ['colony_plates', 'all_locations']:
                    val = row.get(col, '')
                    if pd.notna(val):
                        _already_catalogued = {pid for pids in lims_plate_map.values() for pid in pids}
                        for entry in str(val).split(' | '):
                            match = re.search(r'Plate(\d+)\s*\(([^)]+)\)', entry)
                            if match:
                                pid, _pkey = match.group(1), match.group(2).strip()
                                if pid in _already_catalogued: continue
                                if _pkey not in lims_plate_map: lims_plate_map[_pkey] = []
                                if pid not in lims_plate_map[_pkey]: lims_plate_map[_pkey].append(pid)
                step_keywords = {proto.GOLDEN_GATE: ['Golden Gate'], proto.GIBSON: ['Gibson'], proto.STAR_TRANSF: ['Transformation', 'Agar'], proto.MINIPREP: ['Overnight', 'Miniprep', 'Glycerol'], proto.REARRAY: ['Rearray'], proto.DNA_QUANT: ['Quant', 'DNA'], proto.NGS: ['NGS', 'Sequence'], proto.PCR: ['PCR'], proto.LSP_RECEIVING: ['LSP Receiving'], 'Manual: Miniprep/Glycerol/Media created': ['Overnight', 'Miniprep', 'Glycerol'], proto.REPICK: ['Manual Repick']}

                pipeline_html = ['<div class="timeline-container">']
                if row['type'] == 'transformation_workorder':
                    src_id = row.get('source_asm_process_id')
                    if src_id and src_id in parent_details:
                        parent = parent_details[src_id]
                        src_name = parent['type'].replace(" Assembly", "")
                        det_pills = f'<span class="t-pill">Linked</span>'
                        if pd.notna(parent['job']): det_pills += f'<a href="https://op-tracker.asimov.io/job/{int(parent["job"])}/group/0/step/0/" target="_blank" class="t-pill">Job {int(parent["job"])}</a>'
                        if pd.notna(parent['plate']): det_pills += f'<a href="https://bios.asimov.io/inventory/plates/{clean_plate_id(parent["plate"])}" target="_blank" class="t-pill">Plate{clean_plate_id(parent["plate"])}</a>'
                        pipeline_html.append(f"""<div class="timeline-row"><div class="t-dot source"></div><div class="t-content"><div class="t-header"><span class="t-name" style="color:#1e3a5f">Source: {src_name}</span><span class="t-time">{parent["completion_str"]}</span></div><div class="t-details">{det_pills}</div></div></div>""")
                if queue_data:
                    _past_repick = False
                    _has_repick_plate_keys = any(lp.endswith(' (Repick)') for lp in lims_plate_map)
                    for item in queue_data:
                        is_ready = item['state'] == 'Ready'
                        time_str = item["ready_time"].strftime("%m/%d/%Y %H:%M") + " (Ready)" if is_ready and pd.notna(item["ready_time"]) else (item["start_time"].strftime("%m/%d/%Y %H:%M") if pd.notna(item["start_time"]) else "")
                        _row_is_post_repick = _past_repick
                        if item['queue'] == proto.REPICK:
                            _past_repick = True
                        tooltip_groups = {}
                        keywords = step_keywords.get(item['queue'], [])
                        for lims_proto, pids in lims_plate_map.items():
                            _is_repick_key = lims_proto.endswith(' (Repick)')
                            # When repick plate keys exist, pre-repick rows skip (Repick) keys
                            # and post-repick rows skip original keys — prevents double-counting.
                            if _has_repick_plate_keys and (_row_is_post_repick != _is_repick_key):
                                continue
                            _match_proto = lims_proto[:-len(' (Repick)')] if _is_repick_key else lims_proto
                            match = False
                            for kw in keywords:
                                if kw.lower() in _match_proto.lower():
                                    if item['queue'] == proto.MINIPREP and 'scinomix' in _match_proto.lower(): continue
                                    if item['queue'] == proto.LSP_RECEIVING and 'scinomix' in _match_proto.lower(): continue
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
                                tooltip_html += f'<div class="popover-group"><div class="popover-title">{proto_name}</div><div class="u2">'
                                sorted_plates = sorted(list(p_set), key=lambda x: int(x) if x.isdigit() else 0)
                                for i, pid in enumerate(sorted_plates):
                                    tooltip_html += f'<a href="https://bios.asimov.io/inventory/plates/{pid}" target="_blank" class="popover-link">Plate {pid}</a>'
                                    if (i + 1) % 3 == 0 and i < len(sorted_plates) - 1: tooltip_html += '<br>'
                                tooltip_html += '</div></div>'
                            details_pills += f"""<div class="plate-hover-container"><span class="plate-trigger">{total_plates} Plates</span><div class="plate-popover" data-pop="{_intern_popover(tooltip_html)}"></div></div>"""
                        _dot_cls = "repick" if item['queue'] == proto.REPICK else item['class']
                        pipeline_html.append(f"""<div class="timeline-row"><div class="t-dot {_dot_cls}"></div><div class="t-content"><div class="t-header"><span class="t-name">{item['queue']}</span><span class="t-time">{time_str}</span></div><div class="t-details">{details_pills}</div></div></div>""")
                else:
                    if lims_plate_map:
                        # Group all LIMS plates under a single manual entry with the same
                        # plate-hover popover format used for normal OpTracker timeline rows.
                        tooltip_html = ""
                        all_pids = set()
                        for proto_name, pids in lims_plate_map.items():
                            pids_sorted = sorted(pids, key=lambda x: int(x) if x.isdigit() else 0)
                            all_pids.update(pids_sorted)
                            tooltip_html += f'<div class="popover-group"><div class="popover-title">{proto_name}</div><div class="u2">'
                            for i, pid in enumerate(pids_sorted):
                                tooltip_html += f'<a href="https://bios.asimov.io/inventory/plates/{pid}" target="_blank" class="popover-link">Plate {pid}</a>'
                                if (i + 1) % 3 == 0 and i < len(pids_sorted) - 1: tooltip_html += '<br>'
                            tooltip_html += '</div></div>'
                        lims_pills = f'<div class="plate-hover-container"><span class="plate-trigger">{len(all_pids)} Plates</span><div class="plate-popover" data-pop="{_intern_popover(tooltip_html)}"></div></div>'
                        _fallback_labels = {
                            'pcr_workorder': 'PCR',
                            'oligo_synthesis_workorder': 'Oligo Synthesis',
                            'plasmid_synthesis_workorder': 'Plasmid Synthesis',
                            'syn_part_synthesis_workorder': 'Syn Part Synthesis',
                        }
                        _fallback_label = _fallback_labels.get(row['type'], 'Manual: Miniprep/Glycerol/Media created')
                        pipeline_html.append(f"""<div class="timeline-row"><div class="t-dot succeeded"></div><div class="t-content"><div class="t-header"><span class="t-name">{_fallback_label}</span><span class="t-time"></span></div><div class="t-details">{lims_pills}</div></div></div>""")
                    else:
                        # No OpTracker queue and no LIMS plates. If this workorder is
                        # WAITING/BLOCKED on parts, say what it's waiting on (and where
                        # each part is being made) instead of a bare "No queue data".
                        _wraw = row.get('Waiting')
                        _wl = ([x.strip() for x in str(_wraw).split(',') if x.strip()]
                               if row.get('wo_status') in ('WAITING', 'BLOCKED') and pd.notna(_wraw) else [])
                        _seen = set(); _wu = [w for w in _wl if not (w in _seen or _seen.add(w))]
                        if _wu:
                            _rows = []
                            for _w in _wu:
                                if _w in tree_stock_ids:
                                    _loc, _cls, _pri = 'being made in this workflow', 'here', 2
                                else:
                                    _src = _stock_to_req.get(_w)
                                    if _src and _src.get('exp_name'):
                                        _loc, _cls, _pri = _src['exp_name'], 'elsewhere', 0
                                    else:
                                        _loc, _cls, _pri = 'not yet started', 'none', 1
                                _rows.append((_pri, _w, _loc, _cls))
                            _rows.sort(key=lambda t: (t[0], t[1]))  # external blockers first
                            _lines = ''.join(
                                f'<div class="wait-ops-row"><span class="wait-part">{_w}</span>'
                                f'<span class="wait-loc {_cls}">{_loc}</span></div>'
                                for _pri, _w, _loc, _cls in _rows
                            )
                            _n = len(_wu)
                            pipeline_html.append(
                                f'<div class="wait-ops"><div class="wait-ops-hd">'
                                f'Waiting on {_n} part{"s" if _n != 1 else ""}</div>{_lines}</div>'
                            )
                        else:
                            pipeline_html.append('<span style="color: #9ca3af; font-size: 11px;">No queue data</span>')
                pipeline_html.append('</div>')

                # --- DETAILS INFO ---
                details_info = []
                waiting_items = set()
                if row['wo_status'] in ['WAITING', 'BLOCKED'] and pd.notna(row.get('Waiting')):
                    waiting_items = set([x.strip() for x in str(row['Waiting']).split(',') if x.strip()])
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
                _confirmed_raw = row.get('confirmed_input_ids', '')
                _has_confirmed = (
                    pd.notna(_confirmed_raw)
                    and str(_confirmed_raw).strip() not in ('', 'nan', 'None')
                )
                if _has_confirmed:
                    # Assembly has run — show physically used stocks, BB first.
                    _bb_name = str(row.get('backbone', '') or '').split(':')[0].strip()
                    _conf_ids = [x.strip() for x in str(_confirmed_raw).split('|') if x.strip()]
                    _parsed = []
                    for _entry in _conf_ids:
                        _parts = _entry.split('~')
                        _pid = _parts[0]
                        _well_id   = _parts[1] if len(_parts) > 1 else ''
                        _plate_lbl = _parts[2] if len(_parts) > 2 else ''
                        _pos       = int(_parts[3]) if len(_parts) > 3 and _parts[3].lstrip('-').isdigit() else None
                        _wc        = int(_parts[4]) if len(_parts) > 4 and _parts[4].isdigit() else 96
                        _is_bb = (_pid == _bb_name)
                        if _pos is not None:
                            _key = str(_pos + 1)
                            if _wc == 384:
                                _well_alpha = _WELL_MAP_384.get(_key, _key)
                            elif _wc == 8:
                                _well_alpha = _WELL_MAP_AGAR.get(_key, _key)
                            else:
                                _well_alpha = _WELL_MAP_96.get(_key, _key)
                        else:
                            _well_alpha = ''
                        _parsed.append((_is_bb, _pid, _well_id, _plate_lbl, _well_alpha))
                    _parsed.sort(key=lambda x: (0 if x[0] else 1))
                    _tip_rows = []
                    _tags_html = ''
                    for _is_bb, _stock, _wid, _plate_lbl, _well_alpha in _parsed:
                        _label = ('BB: ' if _is_bb else '') + _stock
                        _tags_html += f'<span class="part-tag">{_label}</span>'
                        _plate_num = ''.join(filter(str.isdigit, _plate_lbl))
                        _plate_cell = (
                            f'<a href="https://bios.asimov.io/inventory/plates/{_plate_num}" target="_blank" class="ci-tip-plate">{_plate_lbl} {_well_alpha}</a>'
                            if _plate_num else ''
                        )
                        _tip_rows.append(
                            f'<div class="ci-tip-row">'
                            f'<span class="ci-tip-stock">{"BB: " if _is_bb else ""}{_stock}</span>'
                            f'{_plate_cell}'
                            f'{"<span class=ci-tip-pid>well " + _wid + "</span>" if _wid else ""}'
                            f'</div>'
                        )
                    _tip_html = (
                        f'<span class="ci-tip">'
                        f'<div class="ci-tip-header">Input Wells</div>'
                        f'{"".join(_tip_rows)}'
                        f'</span>'
                    )
                    inputs_html = f'<div class="parts-container ci-wrap">{_tags_html}{_tip_html}</div>'
                else:
                    # Not yet assembled — show design inputs with waiting/missing styling.
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
                if 'part-tag' in inputs_html: details_info.append(inputs_html)

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
                        _batch_label = f'<a href="{_batch_href}" target="_blank" class="u12">{display_lsp_id}</a>'
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
                        source_row = record_by_wid.get(proc_id)
                        if source_row is not None:
                            resolved_construct = source_row.get('construct_name')
                            if pd.notna(resolved_construct) and str(resolved_construct) not in ('nan', ''):
                                construct_name = str(resolved_construct)
                            resolved_exp = source_row.get('experiment_name')
                            if pd.notna(resolved_exp) and str(resolved_exp) not in ('nan', ''):
                                exp_name = str(resolved_exp)
                            job_val = source_row.get('job_id')
                            if isinstance(job_val, list) and len(job_val) > 0 and pd.notna(job_val[0]):
                                proc_id = f"job__{int(job_val[0])}"

                    pai_val = row.get('STOCK_ID', 'N/A')

                    # Look up source workorder for full experiment name, req_id, request_status
                    src_req_id, src_req_status, src_exp_name = "N/A", "", exp_name
                    if proc_id and proc_id != "N/A" and not proc_id.startswith("job__"):
                        _r = _global_src_lookup.get(proc_id)
                        if _r is not None:
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
                        src_badge = f'<span style="background:#f1f5f9;color:#6b7280;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;border:1px solid #cbd5e1;margin-left:6px;">{src_req_status}</span>'
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
                                    <span class="u1">Input:</span>
                                    <span style="font-size:11px;padding:4px 0;border-bottom:1px solid #f1f5f9;"><a href="https://bios.asimov.io/inventory/wells/{_fid}" target="_blank" class="u12">well{_fid}</a></span>"""

                    lsp_parts.append(f"""
                        <div class="plate-hover-container" style="display: inline-block; margin-bottom: 6px;">
                            <span class="plate-trigger" style="background: #e5e7eb; color: #4b5563; cursor: pointer; font-size: 9px; font-weight: 600; padding: 2px 6px; border-radius: 3px; border: 1px solid #cbd5e1;">
                                Source Info
                            </span>
                            <div class="plate-popover" style="width: 460px; white-space: normal; padding: 15px; border-top: 4px solid #6b7280; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                                <div class="u6">
                                    Source Material Context
                                </div>
                                <div style="display: grid; grid-template-columns: 115px 1fr; gap: 0; font-size: 12px; line-height: 1.5; color: #1f2937;">
                                    <span class="u1">Experiment:</span>
                                    <span style="padding:4px 0;border-bottom:1px solid #f1f5f9;">{src_exp_name}</span>
                                    <span class="u1">Process ID:</span>
                                    <div style="padding:4px 0;border-bottom:1px solid #f1f5f9;">
                                        <a href="{proc_href}" target="_blank" class="u8">{proc_label}</a>
                                        <div style="font-size: 9px; color: #9ca3af; margin-top: 4px; padding-left: 2px;">{construct_name} &nbsp;·&nbsp; <span style="font-family: monospace; color: #6b7280;">{pai_val}</span></div>
                                    </div>
                                    <span class="u1">Request ID:</span>
                                    <span style="font-family: monospace; font-size: 11px; color: #4b5563; padding:4px 0;border-bottom:1px solid #f1f5f9;">{src_req_id}{src_badge}</span>
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
                        _qc_btn_style = "background:#e5e7eb;color:#4b5563;border:1px solid #cbd5e1;"

                    def _qc_dot(val):
                        v = str(val).lower()
                        if v == "pass" or v == "yes":
                            return '<span class="u11">●</span>'
                        elif v == "fail" or v == "no":
                            return '<span style="color:#dc2626;font-size:11px;margin-right:4px;">●</span>'
                        else:
                            return '<span style="color:#9ca3af;font-size:11px;margin-right:4px;">○</span>'

                    if _qc_rows:
                        _qc_grid = "".join(
                            f'<span class="u1">{lbl}:</span>'
                            f'<span class="u4">{"" if lbl == "Comment" else _qc_dot(val)}{val}</span>'
                            for lbl, val in _qc_rows
                        )
                        lsp_parts.append(f"""
                        <div class="plate-hover-container" style="display: inline-block; margin-bottom: 6px; margin-left: 4px;">
                            <span class="plate-trigger" style="{_qc_btn_style} cursor: pointer; font-size: 9px; font-weight: 600; padding: 2px 6px; border-radius: 3px;">
                                QC Details
                            </span>
                            <div class="plate-popover" style="width: 340px; white-space: normal; padding: 15px; border-top: 4px solid #6b7280; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                                <div class="u6">
                                    QC Details
                                </div>
                                <div style="display: grid; grid-template-columns: 160px 1fr; gap: 0; line-height: 1.6;">
                                    {_qc_grid}
                                </div>
                            </div>
                        </div>""")

                    import re as _re
                    def _fv(v):
                        try: return f"{float(v):.1f}"
                        except: return str(v)
                    def _ok(v): return pd.notna(v) and str(v) not in ('nan', 'None', '')
                    strain    = row.get('comp_cell') or row.get('cloning_strain')
                    dl_fmt    = row.get('delivery_format')
                    q_conc    = row.get('qubit_concentration_ngul')
                    q_yld     = row.get('qubit_yield')
                    nd_conc   = row.get('nanodrop_concentration_ngul')
                    nd_yld    = row.get('nanodrop_yield')
                    r_260_280 = row.get('ratio_260_280')
                    r_260_230 = row.get('ratio_260_230')
                    loc       = row.get('location')
                    qc_tube   = row.get('qc_tube_location')
                    # Parse format threshold: e.g. MIDIPREP_LSP_60_UG_800_NG_UL → conc=800, yld=60
                    _fmt_str  = str(dl_fmt) if _ok(dl_fmt) else ""
                    _customer = str(row.get('customer') or "").upper()
                    _use_nd   = "TECH_OUT" in _customer
                    _conc_thr, _yld_thr = None, None
                    _m = _re.search(r'(\d+)_UG[_\s].*?(\d+)_NG_UL', _fmt_str)
                    if _m:
                        try: _yld_thr  = float(_m.group(1))
                        except: pass
                        try: _conc_thr = float(_m.group(2))
                        except: pass
                    def _pass_badge(val, thr):
                        if thr is None or not _ok(val): return ""
                        try:
                            ok = float(val) >= thr
                            return (' <span style="font-size:9px;font-weight:700;color:%s;">%s</span>'
                                    % ("#16a34a" if ok else "#dc2626", "✓" if ok else "✗"))
                        except: return ""
                    rows2 = []
                    if _ok(dl_fmt): rows2.append(("Format", str(dl_fmt)))
                    if _ok(strain): rows2.append(("Strain", str(strain)))
                    # Qubit row — badge if qubit is the QC measurement
                    if _ok(q_conc) or _ok(q_yld):
                        _qv = []
                        if _ok(q_conc):
                            _badge = "" if _use_nd else _pass_badge(q_conc, _conc_thr)
                            _qv.append(f"{_fv(q_conc)} ng/µL{_badge}")
                        if _ok(q_yld):
                            _badge = "" if _use_nd else _pass_badge(q_yld, _yld_thr)
                            _qv.append(f"{_fv(q_yld)} µg{_badge}")
                        rows2.append(("Qubit", " · ".join(_qv)))
                    # Nanodrop row — badge if nanodrop is the QC measurement
                    if _ok(nd_conc) or _ok(nd_yld):
                        _nv = []
                        if _ok(nd_conc):
                            _badge = _pass_badge(nd_conc, _conc_thr) if _use_nd else ""
                            _nv.append(f"{_fv(nd_conc)} ng/µL{_badge}")
                        if _ok(nd_yld):
                            _badge = _pass_badge(nd_yld, _yld_thr) if _use_nd else ""
                            _nv.append(f"{_fv(nd_yld)} µg{_badge}")
                        rows2.append(("Nanodrop", " · ".join(_nv)))
                    # 260/280 and 260/230
                    if _ok(r_260_280) or _ok(r_260_230):
                        _rv = []
                        if _ok(r_260_280): _rv.append(f"260/280: {_fv(r_260_280)}")
                        if _ok(r_260_230): _rv.append(f"260/230: {_fv(r_260_230)}")
                        rows2.append(("Ratios", " · ".join(_rv)))
                    # Nanodrop/Qubit fold difference
                    if _ok(nd_conc) and _ok(q_conc):
                        try:
                            _fold = float(nd_conc) / float(q_conc)
                            rows2.append(("ND/QB", f"{_fold:.2f}×"))
                        except: pass
                    _etoh = row.get('etoh_precipitation')
                    _prep = row.get('prep_method')
                    if _ok(_prep):
                        rows2.append(("Prep", str(_prep)))
                    if _etoh is True or _etoh == True:
                        rows2.append(("EtOH Precip", "Yes"))
                    elif _etoh is False or _etoh == False:
                        rows2.append(("EtOH Precip", "No"))
                    if _ok(loc): rows2.append(("Aliquot", str(loc)))
                    if _ok(qc_tube): rows2.append(("QC Tube", str(qc_tube)))
                    if rows2:
                        grid_cells = "".join(
                            f'<span class="u3">{lbl}</span><span class="u7">{val}</span>'
                            for lbl, val in rows2
                        )
                        lsp_parts.append(
                            f'<div style="display:grid;grid-template-columns:58px 1fr;gap:2px 6px;margin-bottom:4px;">{grid_cells}</div>'
                        )
                    details_info.append("".join(lsp_parts))

                elif row['type'] in ['oligo_synthesis_workorder', 'pcr_workorder', 'plasmid_synthesis_workorder', 'syn_part_synthesis_workorder']:
                    _vendor = row.get('vendor')
                    _vorder = row.get('vendor_order_id')
                    _vorder_clean = str(_vorder).strip() if _vorder is not None and not (isinstance(_vorder, float) and pd.isna(_vorder)) and str(_vorder).strip() not in ('', 'nan', 'None') else ''
                    if pd.notna(_vendor) and str(_vendor).strip() not in ('', 'nan', 'None'):
                        _vendor_str = str(_vendor).strip()
                        _order_tag = f' <span class="u13">{_vorder_clean}</span>' if _vorder_clean else ''
                        details_info.append(f"<div style='font-size:10px;color:#64748b;margin-top:2px;'>Vendor: {_vendor_str}{_order_tag}</div>")
                    if row['type'] == 'pcr_workorder':
                        _wc = row.get('well_comments')
                        _wc_clean = str(_wc).strip().strip(';').strip() if _wc is not None and not (isinstance(_wc, float) and pd.isna(_wc)) else ''
                        if _wc_clean and _wc_clean not in ('nan', 'None', '{;}'):
                            details_info.append(f"<div style='font-size:10px;color:#b45309;background:#fffbeb;border:1px solid #fcd34d;border-radius:3px;padding:2px 5px;margin-top:26px;'>&#9888; {_wc_clean}</div>")

                elif row['type'] in ['golden_gate_workorder', 'gibson_workorder', 'transformation_workorder', 'transformation_offline_operation', 'streakout_operation']:
                    strain = row.get('cloning_strain')
                    if pd.notna(strain): details_info.append(f"<div style='font-size:10px;color:#64748b;margin-top:2px;'>Strain: {strain}</div>")
                    _ab = row.get('antibiotic')
                    _ab_str = str(_ab).strip() if (_ab is not None and not (isinstance(_ab, float) and pd.isna(_ab)) and str(_ab).strip() not in ('', 'nan', 'None')) else ''
                    if _ab_str:
                        _is_mismatch = row.get('antibiotic_mismatch') is True or row.get('antibiotic_mismatch') == True
                        if _is_mismatch:
                            _lims_ab = str(row.get('lims_antibiotic') or '')
                            details_info.append(
                                f"<div style='font-size:10px;margin-top:2px;'>"
                                f"Antibiotic: <span style='color:#b91c1c;font-weight:600;"
                                f"background:#fee2e2;border-radius:3px;padding:0 4px;'>"
                                f"{_ab_str} &#10007;</span>"
                                f"<span style='font-size:9px;color:#9ca3af;margin-left:4px;'>"
                                f"(LIMS: {_lims_ab})</span></div>"
                            )
                        elif row.get('lims_double_marker') is True or row.get('lims_double_marker') == True:
                            # Show every marker LIMS lists (raw set), not the neo-adjusted single value.
                            _lims_ab = str(row.get('lims_all_markers') or row.get('lims_antibiotic') or '')
                            details_info.append(
                                f"<div style='font-size:10px;color:#64748b;margin-top:2px;'>"
                                f"Antibiotic: {_ab_str} "
                                f"<span style='font-size:9px;color:#b45309;background:#fef3c7;"
                                f"border-radius:3px;padding:0 4px;' title='LIMS records more than one "
                                f"selection marker for this plasmid'>&#9888; LIMS: 2 markers ({_lims_ab})</span></div>"
                            )
                        else:
                            details_info.append(f"<div style='font-size:10px;color:#64748b;margin-top:2px;'>Antibiotic: {_ab_str}</div>")
                    _imaged   = row.get('imaged_colonies')
                    _pickable = row.get('pickable_colonies')
                    _picked   = row.get('picked_colonies')
                    _n_repick = row.get('repick_total_colonies', 0)
                    try: _n_repick = int(_n_repick) if pd.notna(_n_repick) else 0
                    except Exception: _n_repick = 0
                    if any(pd.notna(v) for v in [_imaged, _pickable, _picked]):
                        # Colony counts (plain, no link wrapping)
                        _grid = (
                            f"<div style='display:grid;grid-template-columns:58px 1fr;gap:0 4px;margin-top:2px;'>"
                            f"<span style='font-size:9px;font-weight:700;text-transform:uppercase;color:#6b7280;'>Imaged</span><span style='font-size:10px;color:#1e293b;'>{int(_imaged)}</span>"
                            f"<span style='font-size:9px;font-weight:700;text-transform:uppercase;color:#6b7280;'>Pickable</span><span style='font-size:10px;color:#1e293b;'>{int(_pickable) if pd.notna(_pickable) else 0}</span>"
                            f"<span style='font-size:9px;font-weight:700;text-transform:uppercase;color:#6b7280;'>Picked</span><span style='font-size:10px;color:#1e293b;'>{int(_picked) if pd.notna(_picked) else 0}</span>"
                        )
                        if _n_repick > 0:
                            _grid += (
                                f"<span style='font-size:9px;font-weight:700;text-transform:uppercase;color:#7c3aed;'>Repick</span>"
                                f"<span style='font-size:10px;color:#7c3aed;'>{_n_repick}</span>"
                            )
                        _grid += "</div>"
                        details_info.append(_grid)
                        # Agar plate link — plate_id + alphanumeric well from colonypickingcounts
                        _cpid   = row.get('colony_plate_id')
                        _cwpos  = row.get('colony_well_position')
                        _cwcnt  = row.get('colony_plate_well_count')
                        if _cpid and pd.notna(_cpid) and str(_cpid) not in ('nan', 'None', ''):
                            _plate_url  = f'https://bios.asimov.io/inventory/plates/{int(_cpid)}'
                            _well_alpha = ''
                            try:
                                if pd.notna(_cwpos) and pd.notna(_cwcnt):
                                    _key = str(int(_cwpos) + 1)  # LIMS is 0-indexed
                                    _cnt = int(_cwcnt)
                                    if _cnt == 384:
                                        _well_alpha = _WELL_MAP_384.get(_key, '')
                                    elif _cnt == 96:
                                        _well_alpha = _WELL_MAP_96.get(_key, '')
                                    else:
                                        _well_alpha = _WELL_MAP_AGAR.get(_key, '')
                            except Exception:
                                pass
                            _well_label = f' · {_well_alpha}' if _well_alpha else ''
                            details_info.append(
                                f"<div style='margin-top:2px;margin-bottom:2px;'>"
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
                            details_info.append(f'<br><span class="colony-badge" style="background:#fce7f3;color:#be185d;">0 colonies</span>')
                    if row['visual_status'] in ['SUCCEEDED', 'FAILED'] or row['wo_status'] == 'FAILED' or _n_repick > 0:
                        tot = row.get('total_colonies'); seq = row.get('seq_confirmed')
                        if pd.notna(tot) and tot > 0:
                            tot = int(tot); seq = int(seq) if pd.notna(seq) else 0
                            color, bg = ("#0e7490", "#cffafe") if seq > 0 else ("#be185d", "#fce7f3")
                            seq_conf_list = row.get('seq_confirmed_colonies', '')
                            selected_col = row.get('selected_colony', 'None'); selected_col_num = None
                            if selected_col != 'None' and ':' in str(selected_col):
                                clean = re.sub(r'\[.*?\]', '', str(selected_col)).strip()
                                parts = clean.split(':', 1)
                                selected_col_num = parts[1] if len(parts) == 2 else None
                            protocol_wells = {}
                            if pd.notna(seq_conf_list) and seq_conf_list:
                                for entry in str(seq_conf_list).split(','):
                                    entry = entry.strip()
                                    match = re.match(r'(\d+):(\d+)\[([^\]]+)\]', entry)
                                    if match:
                                        well_id, col_num, protocol_name = match.groups()
                                        if protocol_name.strip() not in protocol_wells: protocol_wells[protocol_name.strip()] = {}
                                        protocol_wells[protocol_name.strip()][col_num] = well_id
                            protocol_order = [proto.LSP_RECEIVING, 'Miniprep', 'Bank Overnights', proto.REARRAY, proto.GLYCEROL_STOCKING, 'Glycerol'] if row['type'] == 'lsp_workorder' else ['Miniprep', 'Bank Overnights', proto.REARRAY, proto.GLYCEROL_STOCKING]
                            popover_content = ""
                            for protocol_name in protocol_order:
                                matching_wells = {}
                                for _pkey, wells in protocol_wells.items():
                                    if protocol_name.lower() in _pkey.lower(): matching_wells.update(wells)
                                if matching_wells:
                                    links = ""
                                    for col_num in sorted(matching_wells.keys(), key=lambda x: int(x)):
                                        w_id = matching_wells[col_num]; url = f"https://bios.asimov.io/inventory/wells/{w_id}"; label = f"well{w_id}_col{col_num}"
                                        if col_num == selected_col_num: links += f'<a href="{url}" target="_blank" class="popover-link" style="color:#0891b2;">★ {label}</a>'
                                        else: links += f'<a href="{url}" target="_blank" class="popover-link">{label}</a>'
                                    popover_content += f'<div class="popover-group"><div class="popover-title">{protocol_name}</div>{links}</div>'
                            if popover_content: details_info.append(f'<br><div class="plate-hover-container"><span class="colony-badge" style="background: {bg}; color: {color}; cursor:pointer;">{seq}/{tot} colonies seq confirmed</span><div class="plate-popover" data-pop="{_intern_popover(popover_content)}"></div></div>')
                            else: details_info.append(f'<br><span class="colony-badge" style="background: {bg}; color: {color};">{seq}/{tot} colonies seq confirmed</span>')
                            if _n_repick > 0:
                                details_info.append(f'<br><span class="colony-badge" style="background:#ede9fe;color:#7c3aed;">Repick: 0/{_n_repick} colonies seq confirmed</span>')
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
                _bios_override = row.get('is_status_override') is True or row.get('is_status_override') == True
                _bios_raw = str(row.get('wo_status') or '').upper()
                _status_cell = f'<span class="badge {badge_class}">{effective_status}</span>'
                if _bios_override and _bios_raw and _bios_raw not in ('NAN', 'NONE', ''):
                    _status_cell += f'<div class="bios-override-label">BIOS: {_bios_raw}</div>'
                if row.get('antibiotic_mismatch') is True or row.get('antibiotic_mismatch') == True:
                    _status_cell += '<div style="font-size:9px;font-weight:700;color:#b91c1c;margin-top:2px;">🚨 ANTIBIOTIC</div>'
                _fulfills_attr = "1" if row.get('fulfills_request') else "0"
                _partner_attr = "1" if str(row.get('for_partner', '')).lower() == 'true' else "0"
                # Defer the heavy Queue/timeline cell: park its HTML in an inert
                # <template> (built on pane-open by _buildQueues) so it stays out of
                # the render tree until needed. Queue content is not full-text
                # searchable while collapsed (stock/workorder IDs live in other cells).
                _queue_cell = f'<td class="queue-cell"><template class="qtpl">{"".join(pipeline_html)}</template></td>'
                html.append(f"""<tr class="{row_class}"{_row_order_style} data-wo-type="{row['type']}" data-wo-stock="{(_sid or '').lower()}" data-wo-fulfills="{_fulfills_attr}" data-wo-id="{row['workorder_id']}" data-wo-partner="{_partner_attr}"><td><span class="type-label">{type_display}</span></td><td><code class="wo-id-tag" title="{row['workorder_id']}">{row['workorder_id']}</code></td><td>{_status_cell}</td><td><span class="{_sid_class}">{_sid}</span></td><td><div class="date-tag">{pd.to_datetime(row['wo_created_at']).strftime('%Y-%m-%d') if pd.notna(row['wo_created_at']) else ''}</div></td><td class="tat-cell">{tat_display}</td><td class="details-cell">{"".join(details_info)}</td>{_queue_cell}</tr>""")
            html.append("</tbody></table></div></div>")
        html.append("</div></div>")
        return "".join(html)

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
                <div style="font-size:9px;color:#9ca3af;margin-bottom:8px;font-family:monospace;">{f_total} fulfilled — production TAT distribution</div>
                <div style="display:flex;border-radius:4px;overflow:hidden;gap:1px;">{segs}</div>
                <div style="font-size:8px;color:#9ca3af;margin-top:6px;">weeks from request creation to LSP ready-to-ship</div>
            </div>'''

        max_c = max((sum(wc.values()) for wc in stage_counts.values()), default=1)
        rows  = ''
        for stage_key, _ in _BUCKET_STAGES:
            wc  = stage_counts.get(stage_key, {})
            cnt = sum(wc.values())
            label_style = "color:#374151;font-weight:600;" if cnt > 0 else "color:#9ca3af;font-weight:400;"
            if cnt == 0:
                bar_html = '<div style="flex:1;height:20px;background:#f1f5f9;border-radius:3px;border:1px dashed #e5e7eb;"></div>'
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
            count_html = f'<span style="font-size:10px;color:#374151;font-weight:700;font-family:monospace;margin-left:4px;">{cnt}</span>' if cnt > 0 else ''
            rows += f'''<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                <div style="width:88px;font-size:9px;{label_style}text-align:right;white-space:nowrap;font-family:monospace;">{stage_key}</div>
                <div style="flex:1;display:flex;align-items:center;">{bar_html}{count_html}</div>
            </div>'''
        total_str = f'{total} active request{"s" if total != 1 else ""}' if total > 0 else 'No active requests'
        _mid_blue = _BLUE_RAMP[len(_BLUE_RAMP) // 2]
        legend = f'''<div style="display:flex;gap:12px;margin-top:8px;padding-top:6px;border-top:1px solid #e5e7eb;">
            <span style="font-size:8px;color:#6b7280;display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:{_BLUE_RAMP[0]};border:1px solid rgba(255,255,255,0.25);border-radius:2px;display:inline-block;"></span>Newer</span>
            <span style="font-size:8px;color:#6b7280;display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:{_mid_blue};border-radius:2px;display:inline-block;"></span>On Track</span>
            <span style="font-size:8px;color:#6b7280;display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:{_WARN_COLOR};border-radius:2px;display:inline-block;"></span>Warning</span>
            <span style="font-size:8px;color:#6b7280;display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;background:{_OVER_COLOR};border-radius:2px;display:inline-block;"></span>Overdue</span>
        </div>'''
        return f'''<div style="padding:10px 14px 8px 14px;">
            <div style="font-size:9px;color:#9ca3af;margin-bottom:8px;font-family:monospace;">{total_str}</div>
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

    # Lookup maps for confirmed assembly inputs (DNA Stocks to Assemble).
    # process_id from lims__src.well is either an LSP batch ID or workorder UUID.
    _lsp_df = df[df['lsp_batch_id'].notna()][['lsp_batch_id', 'STOCK_ID']].drop_duplicates(subset='lsp_batch_id')
    _lsp_batch_to_stock: dict = {
        k: str(v or '') for k, v in zip(
            _lsp_df['lsp_batch_id'].astype(str).str.strip().str.upper(),
            _lsp_df['STOCK_ID']
        ) if k not in ('NAN', 'NONE', '')
    }
    _wo_df = df[df['STOCK_ID'].notna() & df['workorder_id'].notna()][['workorder_id', 'STOCK_ID']].drop_duplicates(subset='workorder_id')
    _woid_to_stock: dict = {
        str(wo).strip(): str(st).strip()
        for wo, st in zip(_wo_df['workorder_id'], _wo_df['STOCK_ID'])
        if str(st).strip() not in ('nan', 'None', '')
    }

    # Load experiment due dates (written by fetch_due_dates() before render)
    _due_date_map: dict[str, str] = {}
    try:
        from dnasc.extractors.sheets import load_due_dates
        _due_date_map = load_due_dates()
    except Exception:
        pass

    # Partner experiments with no Asana due date in the override sheet — collected
    # during render so they can be flagged on the card and logged at the end.
    _missing_partner_exps: list[str] = []

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

    # Stream mode (out_fh set): flush completed HTML to disk and reset the
    # buffer so the full ~160 MB document is never resident at once. In the
    # default mode (out_fh is None) every _flush() is a no-op and `html`
    # accumulates exactly as before.
    def _flush():
        nonlocal html
        if out_fh is not None and html:
            out_fh.write(html)
            html = ""
    _flush()  # write the document head before the per-project loop

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
        count_blocked = 0; count_stalled = 0; count_in_lsp = 0; count_asm_review = 0; count_seq_winner = 0; count_order_pending = 0
        count_in_assembly = 0; count_active_waiting = 0; count_ship_ready = 0
        new_req_list = []; active_req_list = []; fulfilled_req_list = []; canceled_req_list = []
        stalled_reqs = set(); asm_review_reqs = set(); seq_winner_reqs = set(); order_pending_reqs = set(); antibiotic_mismatch_reqs = set(); dual_antibiotic_reqs = set(); production_tats = []; total_tats = []
        count_antibiotic_mismatch = 0
        count_dual_antibiotic = 0
        _root_only = project_df[project_df['workorder_id'] == project_df['root_work_order_id']]
        _ptr_source = _root_only if not _root_only.empty else project_df
        has_ptr = _ptr_source['for_partner'].astype(str).str.lower().str.contains('true').any()
        _card_accent = "#7461b8" if has_ptr else "#3a5c7a"  # matches the muted gradient's start tone
        # Partner vs R&D run badge — appended right after the design title.
        if has_ptr:
            _run_badge = ('<span style="margin-left:12px;background:#f5f3ff;color:#7c3aed;'
                          'border:1px solid #ddd6fe;font-weight:600;font-size:10px;padding:2px 8px;'
                          'border-radius:5px;white-space:nowrap;">Partner</span>')
        else:
            _run_badge = ('<span style="margin-left:12px;background:#eff6ff;color:#1e40af;'
                          'border:1px solid #bfdbfe;font-weight:600;font-size:10px;padding:2px 8px;'
                          'border-radius:5px;white-space:nowrap;">R&amp;D</span>')
        # Customer tags (experiment header) — token colors + leading dot, mixed-case.
        _ecg = tok.GEOM['customer']
        _ec_geom = (f"display:inline-block;font-size:{_ecg['size']};font-weight:{_ecg['weight']};"
                    f"padding:{_ecg['pad']};border-radius:{_ecg['radius']};white-space:nowrap;")
        _exp_customers = []
        if 'customer' in project_df.columns:
            # Only look at request-fulfilling root workorders, not shared parts
            # pulled in from other experiments (which inherit a foreign experiment_name
            # via assembly_plan_id but carry a different experiment's customer value).
            _req_rows = project_df[project_df['workorder_id'] == project_df['root_work_order_id']]
            _cust_source = _req_rows if not _req_rows.empty else project_df
            for _cv in _cust_source['customer'].dropna().unique():
                _cv = str(_cv)
                if _cv not in ('nan', 'None', '') and _cv not in [c for c, *_ in _exp_customers]:
                    _cl, _cbg, _cfg = tok.CUSTOMER.get(_cv, (_cv.replace('_', ' '),) + tok.CUSTOMER_FALLBACK[1:])
                    _exp_customers.append((_cv, _cl, _cbg, _cfg))
        exp_customer_tags = " ".join(
            f'<span style="{_ec_geom}background:{bg};color:{fg};border:1px solid {fg}55;">{tok.CUSTOMER_DOT}{lbl}</span>'
            for _, lbl, bg, fg in _exp_customers
        )
        # ── Experiment creation date + due/seq override + timeline axis ───────
        # Computed BEFORE the request loop so the age-dots can be positioned on the
        # same axis, anchored to the dnasc-created (BIOS) date rather than week 0.
        exp_created_str = "N/A"
        exp_created_dt = None
        # VRT002 (A748-1-AG_VRT002_Vector_Design, created 2026-06-02) is the first
        # project whose timeline dot tracks Avg Available (LSP delivered). Every
        # project created before it keeps tracking Avg QC Confirmed. Cutoff =
        # VRT002's experiment creation timestamp (UTC).
        _use_delivered = False
        if 'experiment_created_at' in project_df.columns:
            exp_created_raw = project_df['experiment_created_at'].iloc[0]
            exp_created_dt = to_est(exp_created_raw)
            if exp_created_dt:
                exp_created_str = exp_created_dt.strftime('%Y-%m-%d · %-I:%M %p EST')
            _ec_utc = pd.to_datetime(exp_created_raw, errors='coerce', utc=True)
            _use_delivered = bool(pd.notna(_ec_utc) and _ec_utc >= pd.Timestamp('2026-06-02 17:55:14', tz='UTC'))

        _NO_TIMELINE_MARKERS = PipelineConfig.PINNED_INFRA_EXPERIMENTS
        _is_infra_exp = experiment_name in _NO_TIMELINE_MARKERS
        _due_raw         = None if _is_infra_exp else _due_date_map.get(experiment_name)
        _due_badge_html  = ""
        _due_marker_html = ""
        _sort_due_date   = "9999-99-99"   # default: no due date → sorts to end
        if _due_raw is None:
            _due_entry_data = None
        elif isinstance(_due_raw, str):
            _due_entry_data = {"due_date": _due_raw, "sequence_transferred": ""}
        elif isinstance(_due_raw, dict):
            _due_entry_data = _due_raw
        else:
            _due_entry_data = _due_raw[0] if _due_raw else None
        _due_entry = bool(_due_entry_data)

        # Sequence-transferred date = the real experiment start (sequences delivered),
        # which precedes the dnasc BIOS-created date. NOT used in any TAT math.
        _seq_exp_dt = None
        if _due_entry_data:
            _seq_raw = str(_due_entry_data.get("sequence_transferred", "") or "").strip()
            if _seq_raw and _seq_raw not in ("nan", "None"):
                try:
                    _seq_exp_dt = datetime.strptime(_seq_raw, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                except Exception:
                    _seq_exp_dt = None

        # Timeline axis. Week numbers ALWAYS count from dnasc-created (0w = entry) so the
        # orange/red TAT thresholds (4/5w partner, 5/6w R&D) sit on their week marks.
        # For partner experiments with a seq-transfer date, the left edge = seq and the
        # axis is PROPORTIONAL: a longer upstream lead → a longer hatched band (0w/dnasc
        # sits further right). R&D / no-seq: linear 8-week axis from dnasc-created.
        # `<=` (not `<`): a same-day seq (lead = 0, e.g. sequences delivered the day dnasc
        # opened the work) still gets the rich axis — the "now" dot and the 0w·dnasc start
        # marker — just with no hatched band (there's no lead to draw).
        _tl_origin_dt = exp_created_dt
        _tl_days      = 56.0
        _tl_weeks     = 8
        _show_dnasc_start = False
        _dnasc_pct    = 0.0
        _lead_days    = 0
        if has_ptr and exp_created_dt and _seq_exp_dt and _seq_exp_dt.date() <= exp_created_dt.date():
            _show_dnasc_start = True
            _tl_origin_dt = _seq_exp_dt
            _lead_days = (exp_created_dt.date() - _seq_exp_dt.date()).days
            _tl_days = float(((exp_created_dt + pd.Timedelta(weeks=8)).date() - _seq_exp_dt.date()).days)
            _dnasc_pct = min(100, max(0, _lead_days / _tl_days * 100))

        def _axpos(dt):
            """Position (0-100%) of a date on the timeline axis — linear from the left
            edge (dnasc-created normally; seq-transfer for the seq-extended partner axis)."""
            d = dt.date() if hasattr(dt, 'date') else dt
            if not _tl_origin_dt:
                return 0
            return min(100, max(0, (d - _tl_origin_dt.date()).days / _tl_days * 100))

        # Upstream-lead note (partner w/ seq date): experiment start + days before dnasc
        # opened the work — informational, NOT part of dnasc TAT.
        _seq_note_html = ""
        if _show_dnasc_start:
            _tot_txt = ""
            try:
                _dd = str((_due_entry_data or {}).get("due_date", "")).strip()
                if _dd:
                    _tot_days = (datetime.strptime(_dd, "%Y-%m-%d").date() - _seq_exp_dt.date()).days
                    _tot_txt = f" · {_tot_days}d total to due (incl. lead)"
            except Exception:
                _tot_txt = ""
            # lead = 0 (seq delivered the same day dnasc opened): no upstream window, so
            # show "= dnasc start" instead of a meaningless "+0d upstream".
            _lead_txt = (f'+{_lead_days}d upstream (not in TAT)' if _lead_days > 0
                         else '= dnasc start (same day)')
            _lead_title = (f'{_lead_days}d before dnasc opened the work; upstream, not counted in dnasc TAT.'
                           if _lead_days > 0 else 'same day dnasc opened the work — no upstream lead.')
            _seq_note_html = (
                f'<span class="kpill" style="background:#fff7ed;border:1px solid #fed7aa;" '
                f'title="Sequences transferred (experiment start) {_seq_exp_dt:%a %b %-d} — '
                f'{_lead_title}{_tot_txt}">'
                f'<span class="kk" style="color:#9a3412;">Seq transferred {_seq_exp_dt:%-m/%-d}</span> '
                f'<b style="color:#c2410c;">{_lead_txt}</b></span>'
            )

        dots_html = ""; stage_counts = {}; stage_items = {}; fulfilled_week_counts = {}
        _exp_end_times = []; _exp_any_active = False   # for the TOTAL RUNNING headline
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
            status = str(r_df['request_status'].iloc[0] if 'request_status' in r_df.columns else 'NEW').upper()
            is_partner = 'true' in str(r_df['for_partner'].iloc[0] if 'for_partner' in r_df.columns else False).lower()
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
                            if name == proto.LSP_REVIEWING and state == 'SC': ready_to_ship_time = to_est(start)
                            if name == proto.LSP_RELEASING and state == 'SC': final_release_time = to_est(start)

            is_finished = status in ['FULFILLED', 'SUCCEEDED']
            if is_finished:
                production_end = ready_to_ship_time or now
                total_end = final_release_time or production_end
                # Timeline dot: VRT002-and-newer track Available/delivered (LSP
                # Releasing = total_end); older projects track QC-confirmed (LSP
                # Reviewing = production_end). Position + age-color key off the
                # chosen end. TAT kpills below always report both regardless.
                _dot_end = total_end if _use_delivered else production_end
                age_weeks = (_dot_end - r_created).days / 7
                production_tats.append((production_end - r_created).days)
                total_tats.append((total_end - r_created).days)
                dot_color = _age_color(age_weeks, yellow_limit, red_limit, ramp=_BLUE_RAMP if has_ptr else _PURPLE_RAMP)
                _dot_date = _dot_end
                _exp_end_times.append(total_end)
            else:
                age_weeks = (now - r_created).days / 7
                dot_color = _age_color(age_weeks, yellow_limit, red_limit, ramp=_BLUE_RAMP if has_ptr else _PURPLE_RAMP)
                _dot_date = now
                if status not in ('CANCELED',):
                    _exp_any_active = True
            # Seq-extended axis: place dots by calendar date so they start at 0w (dnasc
            # created). Standard axis: age/8 (origin already = dnasc-created).
            if _show_dnasc_start:
                pos = _axpos(_dot_date)
            else:
                pos = max(0, min(100, (age_weeks / 8) * 100))

            # is_stalled / is_asm_review / is_blocked / has_real_workorders all
            # pre-computed by the pipeline enrichment step — read from parquet.
            is_stalled         = bool(r_df['is_stalled'].iloc[0])      if 'is_stalled'       in r_df.columns else False
            is_asm_review      = bool(r_df['is_asm_review'].iloc[0])   if 'is_asm_review'    in r_df.columns else False
            is_blocked         = bool(r_df['is_blocked'].iloc[0])      if 'is_blocked'       in r_df.columns else False
            has_seq_winner     = bool(r_df['has_seq_winner'].iloc[0])     if 'has_seq_winner'    in r_df.columns else False
            has_order_pending  = bool(r_df['has_order_pending'].iloc[0]) if 'has_order_pending' in r_df.columns else False
            has_antibiotic_mismatch = False
            if 'antibiotic_mismatch' in r_df.columns:
                _ab_active = r_df[
                    r_df['type'].isin(['golden_gate_workorder', 'gibson_workorder', 'transformation_workorder'])
                    & r_df['antibiotic_mismatch'].eq(True)
                    & ~r_df['visual_status'].isin(['SUCCEEDED', 'FAILED', 'CANCELED'])
                ]
                has_antibiotic_mismatch = not _ab_active.empty
            has_dual_antibiotic = False
            if 'lims_double_marker' in r_df.columns:
                _dual_active = r_df[
                    r_df['type'].isin(['golden_gate_workorder', 'gibson_workorder', 'transformation_workorder'])
                    & r_df['lims_double_marker'].eq(True)
                    & ~r_df['visual_status'].isin(['SUCCEEDED', 'FAILED', 'CANCELED'])
                ]
                has_dual_antibiotic = not _dual_active.empty
            _draft_mask        = r_df['data_source'].eq('BIOS_DRAFT') if 'data_source' in r_df.columns else pd.Series(False, index=r_df.index)
            has_real_workorders = not r_df[
                r_df['workorder_id'].notna()
                & ~r_df['workorder_id'].astype(str).str.startswith('REQ-')
                & ~_draft_mask
            ].empty

            if not is_finished and status != 'CANCELED':
                _week_bucket = min(int(age_weeks), 8)
                _tip_cn  = str(r_df['construct_name'].iloc[0] if 'construct_name' in r_df.columns and not r_df['construct_name'].dropna().empty else '') or ''
                _tip_sids = r_df[r_df['STOCK_ID'].notna()]['STOCK_ID'].dropna().unique().tolist()
                _tip_sid  = str(_tip_sids[0]) if _tip_sids else ''
                _tip_label = f"{_tip_sid}: {_tip_cn}".strip(': ') if (_tip_sid or _tip_cn) else str(rid)[:8]
                _stage = str(r_df['stage'].iloc[0]) if 'stage' in r_df.columns else ('In Design' if not has_real_workorders else 'Stalled')
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
                if has_seq_winner: count_seq_winner += 1; seq_winner_reqs.add(rid)
                if has_order_pending: count_order_pending += 1; order_pending_reqs.add(rid)
                if has_dual_antibiotic: count_dual_antibiotic += 1; dual_antibiotic_reqs.add(rid)
                if has_antibiotic_mismatch: count_antibiotic_mismatch += 1; antibiotic_mismatch_reqs.add(rid)
                elif is_blocked: count_blocked += 1
                else:
                    is_ship_ready = False
                    _active_wo = r_df[r_df['wo_status'] != 'CANCELED']
                    lsp_active = _active_wo[(_active_wo['type'] == 'lsp_workorder') & (_active_wo['visual_status'].isin(['RUNNING', 'READY', 'IN_PROGRESS']))]
                    for _, lsp_row in lsp_active.iterrows():
                        pnames = lsp_row.get('protocol_name', []); pstates = lsp_row.get('operation_state', [])
                        if isinstance(pnames, list) and isinstance(pstates, list):
                            for name, state in zip(pnames, pstates):
                                if name == proto.LSP_RELEASING and state == 'RD': is_ship_ready = True; break
                        if is_ship_ready: break
                    if is_ship_ready: count_ship_ready += 1
                    elif not lsp_active.empty: count_in_lsp += 1
                    elif not _active_wo[(_active_wo['type'].isin(['golden_gate_workorder', 'gibson_workorder', 'transformation_workorder', 'transformation_offline_operation', 'streakout_operation', 'pcr_workorder'])) & (_active_wo['visual_status'].isin(['RUNNING', 'READY', 'IN_PROGRESS']))].empty: count_in_assembly += 1
                    else: count_active_waiting += 1
            else: count_new += 1; new_req_list.append((rid, r_df))

            # Hover tooltip: the milestone this dot marks. Finished dots sit at
            # Available (LSP delivered); active dots sit at "today" (current age).
            _dc = str(r_df['construct_name'].iloc[0]) if 'construct_name' in r_df.columns and not r_df['construct_name'].dropna().empty else ''
            _ds_l = r_df[r_df['STOCK_ID'].notna()]['STOCK_ID'].dropna().unique().tolist() if 'STOCK_ID' in r_df.columns else []
            _dot_lbl = (f"{(str(_ds_l[0]) if _ds_l else '')}: {_dc}".strip(': ')) or str(rid)[:8]
            if is_finished:
                _milestone = "Available (LSP delivered)" if _use_delivered else "QC confirmed"
                _dd = max(0, (_dot_date - r_created).days); _dw, _drem = divmod(_dd, 7)
                _dot_title = f"{_dot_lbl} — {_milestone} · {_dot_date.strftime('%b %-d')} · {_dw}w {_drem}d"
            elif status == 'CANCELED':
                _dot_title = f"{_dot_lbl} — Canceled"
            else:
                _ad = max(0, (now - r_created).days); _aw, _arem = divmod(_ad, 7)
                _dot_title = f"{_dot_lbl} — In progress · {_aw}w {_arem}d"
            _dot_title = _dot_title.replace('"', '&quot;')

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
            dots_html += f'''<div title="{_dot_title}" style="position:absolute; left:{pos}%; top:{v_jitter}px; width:{dot_size}; height:{dot_size}; background:{dot_color}; {shape_css} {border_css} z-index:{z_idx}; opacity:{dot_opacity};"></div>'''


        avg_tat_html = ""
        if production_tats or total_tats:
            tat_parts = []
            if production_tats:
                avg_f = sum(production_tats) / len(production_tats); weeks, days = int(avg_f//7), int(avg_f%7)
                tat_parts.append(f"<span class='kpill'><span class='kk' title='Request created → LSP Reviewing (QC confirmed)'>Avg to QC Confirmed:</span><b style='color:#2563eb;'>{weeks}w {days}d</b></span>")
            if total_tats:
                avg_t = sum(total_tats) / len(total_tats); weeks, days = int(avg_t//7), int(avg_t%7)
                tat_parts.append(f"<span class='kpill'><span class='kk' title='Request created → LSP Releasing (available) — the dnasc TAT'>Avg to Available:</span><b style='color:#2563eb;'>{weeks}w {days}d</b></span>")
            avg_tat_html = f'''<div style="display:flex; gap:10px; font-weight:700;">{" ".join(tat_parts)}</div>'''

        # (exp_created_dt, due/seq override, and the timeline axis were computed
        # before the request loop above so the age-dots could use them.)

        # Partner project with no Asana due date in the sheet → flag it. Scoped to
        # ACTIVE experiments (≥1 in-progress request) so fulfilled/old partner work
        # isn't flagged. Infra experiments (refills, etc.) are exempt.
        # Any non-terminal request counts as active — a brand-new partner project
        # arrives as NEW/PLANNED (not yet IN_PROGRESS), and those are precisely the
        # ones that still need an Asana due date added. Terminal = FULFILLED/CANCELED/NONE.
        _exp_has_active = (
            project_df['request_status'].astype(str).str.upper()
            .isin({'IN_PROGRESS', 'NEW', 'PLANNED', 'RUNNING', 'READY', 'WAITING', 'BLOCKED'})
        ).any() if 'request_status' in project_df.columns else False
        _missing_asana = has_ptr and not _due_entry and not _is_infra_exp and _exp_has_active
        if _missing_asana:
            _missing_partner_exps.append(experiment_name)
        _missing_due_badge = (
            '<span class="badge" title="Partner project not found in the Asana due-date '
            'sheet — add it so milestones can anchor on the committed date." '
            'style="background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;margin-left:4px;'
            'font-weight:800;white-space:nowrap;">⚠ MISSING ASANA DATE</span>'
            if _missing_asana else ''
        )

        if _due_entry_data:
            try:
                _now_utc      = datetime.now(pytz.UTC)
                def _pos(dt):
                    return _axpos(dt)

                _due_date_str = _due_entry_data.get("due_date", "")
                _seq_str      = _due_entry_data.get("sequence_transferred", "")
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
                        _seq_dt    = datetime.strptime(_seq_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC) if _seq_str else None
                        # NGS day = last Mon/Thu strictly before the Asana due date
                        # (weekday Mon=0, Thu=3); ASM bracket starts 13 days before it.
                        _due_ngs_adj = _ngs_dt
                        for _bi in range(7):
                            _bc = _due_dt - pd.Timedelta(days=_bi+1)
                            if _bc.weekday() in (0, 3): _due_ngs_adj = _bc; break
                        _due_asm_dt  = _due_ngs_adj - pd.Timedelta(days=13)
                        _due_asm_pos = max(0, _pos(_due_asm_dt))
                        _ngs_pos   = _pos(_ngs_dt)
                        _due_pos   = _pos(_due_dt)
                        if 0 < _due_pos <= 100:
                            _dr            = (_due_dt.date() - _now_utc.date()).days
                            _urgency_color = "#f87171" if _dr < 0 else "#fcd34d" if _dr <= 7 else "#6ee7b7"
                            # Range bar now runs ASM → due (single Asana date; no second date).
                            _range_width   = max(0.5, _due_pos - _due_asm_pos)
                            _due_label     = _due_dt.strftime("%a %b %-d")
                            _seq_label     = _seq_dt.strftime("%a %-m/%-d") if _seq_dt else ""
                            _pop_id        = f"duepop_{safe_exp_id}"
                            _pill_text     = f"ASANA {_due_dt.strftime('%a %-m/%-d')}"
                            _due_marker_html = (
                                # Range bar: ASM → due/gantt
                                f'<div style="position:absolute;left:{_due_asm_pos:.2f}%;top:0;'
                                f'width:{_range_width:.2f}%;height:100%;'
                                f'background:rgba(124,58,237,0.07);border:1px solid #cbd5e1;'
                                f'border-radius:3px;z-index:1;pointer-events:none;"></div>'
                                # Vertical line
                                f'<div style="position:absolute;left:{_due_pos:.2f}%;top:0px;'
                                f'width:4px;height:24px;background:#7c3aed;border-radius:1px;'
                                f'box-shadow:0 0 6px rgba(124,58,237,0.45);z-index:4;transform:translateX(-50%);pointer-events:none;"></div>'
                                # Hover wrapper + pill
                                f'<div style="position:absolute;left:{_due_pos:.2f}%;top:26px;'
                                f'width:90px;height:16px;transform:translateX(-50%);z-index:25;cursor:pointer;"'
                                f' onmouseenter="document.getElementById(\'{_pop_id}\').style.display=\'block\'"'
                                f' onmouseleave="document.getElementById(\'{_pop_id}\').style.display=\'none\'">'
                                f'<div style="position:absolute;left:50%;top:0;transform:translateX(-50%);'
                                f'background:#f1f5f9;color:#6d28d9;font-size:10px;font-weight:700;'
                                f'padding:2px 6px;border-radius:4px;white-space:nowrap;letter-spacing:0.02em;'
                                f'line-height:1;box-shadow:0 1px 2px rgba(0,0,0,0.1);border:1px solid #7c3aed;">{_pill_text}</div>'
                                # Popover
                                f'<div id="{_pop_id}" style="display:none;position:absolute;left:38px;top:8px;'
                                f'background:#1e1b4b;color:white;font-size:10px;padding:8px 11px;border-radius:5px;'
                                f'white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,0.7);z-index:100;border:1px solid #cbd5e1;">'
                                f'<div style="font-weight:800;color:white;margin-bottom:5px;font-size:11px;">Asana Due Date</div>'
                                f'<div style="display:grid;grid-template-columns:90px 1fr;gap:2px 8px;">'
                                + (f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Seq Transferred</span>'
                                   f'<span style="font-weight:600;">{_seq_label}</span>' if _seq_dt else '') +
                                f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">Due (Asana)</span>'
                                f'<span style="font-weight:700;color:{_urgency_color};">{_due_label}</span>'
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
                def _pos2(dt):
                    return _axpos(dt)
                _def_asm_dt  = _last_ngs_b(_red_thresh_dt - _td2(days=1)) - _td2(days=13)
                _def_ngs_pos = _pos2(_def_ngs_dt)
                _def_red_pos = _pos2(_red_thresh_dt)
                _def_due_pos = _pos2(_def_due_dt)
                _def_asm_pos = max(0, _pos2(_def_asm_dt))
                _def_width   = max(0.5, _def_red_pos - _def_asm_pos)
                _def_pop_id  = f"defduepop_{safe_exp_id}"
                _def_ngs_str = _def_ngs_dt.strftime("%a %b %-d")
                _def_due_str = _def_due_dt.strftime("%a %-m/%-d")
                _def_red_str = _red_thresh_dt.strftime("%a %b %-d")
                _default_bracket_html = (
                    # White semi-transparent bracket: ASM → red threshold
                    f'<div style="position:absolute;left:{_def_asm_pos:.2f}%;top:0;'
                    f'width:{_def_width:.2f}%;height:100%;'
                    f'background:rgba(124,58,237,0.07);border:1px solid #cbd5e1;'
                    f'border-radius:3px;z-index:1;pointer-events:none;"></div>'
                    + (
                    # Purple vertical line at day-after-NGS
                    f'<div style="position:absolute;left:{_def_due_pos:.2f}%;top:0px;'
                    f'width:4px;height:24px;background:#7c3aed;border-radius:1px;'
                    f'box-shadow:0 0 6px rgba(124,58,237,0.45);z-index:4;transform:translateX(-50%);pointer-events:none;"></div>'
                    # Hover wrapper: pill + popover
                    f'<div style="position:absolute;left:{_def_due_pos:.2f}%;top:26px;'
                    f'width:80px;height:16px;transform:translateX(-50%);z-index:25;cursor:pointer;"'
                    f' onmouseenter="document.getElementById(\'{_def_pop_id}\').style.display=\'block\'"'
                    f' onmouseleave="document.getElementById(\'{_def_pop_id}\').style.display=\'none\'">'
                    # Light-purple DUE pill
                    f'<div style="position:absolute;left:50%;top:0;transform:translateX(-50%);'
                    f'background:#f1f5f9;color:#6d28d9;font-size:10px;font-weight:700;'
                    f'padding:2px 6px;border-radius:4px;white-space:nowrap;letter-spacing:0.02em;'
                    f'line-height:1;box-shadow:0 1px 2px rgba(0,0,0,0.1);border:1px solid #7c3aed;">'
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
        def _threshold_bar(week, color, glow):
            """Colored vertical bar at `week` weeks from dnasc-created, on the axis."""
            if not exp_created_dt or _is_refill:
                return ""
            _left = f"{_axpos(exp_created_dt + _td(weeks=week)):.2f}%"
            return (f'<div style="position:absolute;left:{_left};width:2px;height:24px;background:{color};'
                    f'top:0px;border-radius:1px;box-shadow:0 0 6px {glow};z-index:2;"></div>')
        # Threshold bars removed — the colored week dates (amber 4w, red 5w) carry the
        # TAT-warning cue instead; the week gridlines at 4w/5w stay normal like the rest.
        _orange_html = ""
        _red_html    = ""

        # Dynamic week header: two-line labels absolutely positioned at exact grid-line %
        # Row 1: week number (bold)  Row 2: date (smaller)
        # Pills use height:20px so their hover area stays in row-1 zone, clear of row-2 dates
        _bs = 'position:absolute;font-family:monospace;white-space:nowrap;text-align:center;line-height:1.4;'
        # Threshold dates (from dnasc-created) so the matching week label can be colored.
        _orange_d = (exp_created_dt + _td(weeks=orange_week)).date() if exp_created_dt else None
        _red_d    = (exp_created_dt + _td(weeks=red_week)).date()    if exp_created_dt else None
        def _hpos(dt):
            return _axpos(dt)
        _wn = 'font-size:11px;font-weight:600;color:#ffffff;letter-spacing:0.2px;text-shadow:0 1px 2px rgba(0,0,0,0.3);'
        _wd_n = 'font-size:11px;font-weight:500;color:rgba(255,255,255,0.95);text-shadow:0 1px 2px rgba(0,0,0,0.3);'

        if _show_dnasc_start:
            # Seq-extended axis: SEQ (experiment start) at the far left, then dnasc
            # weeks 0w..8w positioned by date. The SEQ label only shows when the band is
            # wide enough to clear the 0w label (else it's in the band hover); on tight
            # leads the 0w label left-aligns so it neither overlaps SEQ nor spills left.
            _weeks_header_html = ""
            if _dnasc_pct >= 10:
                _weeks_header_html = (
                    f'<span style="{_bs}left:0;transform:translateX(0);text-align:left;">'
                    f'<span style="font-size:11px;font-weight:700;color:rgba(255,255,255,0.85);">SEQ</span>'
                    f'<br><span style="font-size:11px;font-weight:500;color:rgba(255,255,255,0.85);">{_seq_exp_dt:%-m/%-d}</span></span>'
                )
            for _wh in range(0, 9):
                _wd = exp_created_dt + _td(weeks=_wh)
                _wpct = _hpos(_wd)
                _wdate = _wd.strftime("%a %-m/%-d")
                if _wh == 8:
                    _tx = 'translateX(-100%)'
                elif _wh == 0 and _dnasc_pct < 12:
                    _tx = 'translateX(0)'
                else:
                    _tx = 'translateX(-50%)'
                if _wh == 0:
                    _num = '<span style="font-size:11px;font-weight:800;color:#ffffff;text-shadow:0 1px 3px rgba(0,0,0,0.6);">0w · dnasc</span>'
                    _dtt = f'<span style="font-size:11px;font-weight:700;color:#ffffff;text-shadow:0 1px 3px rgba(0,0,0,0.6);">{_wdate}</span>'
                elif _wd.date() == _orange_d:
                    _num = f'<span style="{_wn}">{_wh}w</span>'; _dtt = f'<span style="font-size:11px;font-weight:800;color:#fbbf24;">{_wdate}</span>'
                elif _wd.date() == _red_d:
                    _num = f'<span style="{_wn}">{_wh}w</span>'; _dtt = f'<span style="font-size:11px;font-weight:800;color:#fb7185;">{_wdate}</span>'
                else:
                    _num = f'<span style="{_wn}">{_wh}w{"+" if _wh==8 else ""}</span>'; _dtt = f'<span style="{_wd_n}">{_wdate}</span>'
                _weeks_header_html += f'<span style="{_bs}left:{_wpct:.2f}%;transform:{_tx};">{_num}<br>{_dtt}</span>'
        else:
            _start_date_str = _tl_origin_dt.strftime("%a %-m/%-d") if _tl_origin_dt else ""
            _weeks_header_html = (
                f'<span style="{_bs}left:0;transform:translateX(0);text-align:left;">'
                f'<span style="{_wn}">START</span>'
                + (f'<br><span style="{_wd_n}">{_start_date_str}</span>' if _start_date_str else '')
                + f'</span>'
            )
            for _wh in range(1, _tl_weeks):
                _wh_pct = _wh / _tl_weeks * 100
                if _tl_origin_dt and not _is_refill:
                    _wh_dt_date = (_tl_origin_dt + _td(weeks=_wh)).date()
                    _wh_date = _wh_dt_date.strftime("%a %-m/%-d")
                    if _wh_dt_date == _orange_d:
                        _wh_dt  = f'<span style="font-size:11px;font-weight:800;color:#fbbf24;">{_wh_date}</span>'
                    elif _wh_dt_date == _red_d:
                        _wh_dt  = f'<span style="font-size:11px;font-weight:800;color:#fb7185;">{_wh_date}</span>'
                    else:
                        _wh_dt  = f'<span style="{_wd_n}">{_wh_date}</span>'
                    _wh_txt = f'<span style="{_wn}">{_wh}w</span><br>{_wh_dt}'
                else:
                    _wh_txt = f'<span style="{_wn}">{_wh}w</span>'
                _weeks_header_html += f'<span style="{_bs}left:{_wh_pct:.2f}%;transform:translateX(-50%);">{_wh_txt}</span>'
            if _tl_origin_dt and not _is_refill:
                _wh_last_date = (_tl_origin_dt + _td(weeks=_tl_weeks)).strftime("%a %-m/%-d")
                _wh_last_txt = (f'<span style="{_wn}">{_tl_weeks}w+</span>'
                                f'<br><span style="{_wd_n}">{_wh_last_date}</span>')
            else:
                _wh_last_txt = f'<span style="{_wn}">{_tl_weeks}w+</span>'
            _weeks_header_html += f'<span style="{_bs}right:0;transform:translateX(0);text-align:right;">{_wh_last_txt}</span>'
        if _is_refill:
            _default_bracket_html = ""
            _due_marker_html = ""

        # ── ASM / LSP scale-up markers (predetermined from TAT / due-date schedule) ──
        _asm_markers_html = ""
        if exp_created_dt and not _is_refill:
            try:
                # Determine reference NGS date from due date override or TAT threshold
                _tat_weeks2 = 5 if has_ptr else 6
                _thresh2 = exp_created_dt + _td(weeks=_tat_weeks2)
                _due_ref_str = (_due_entry_data or {}).get("due_date", "") if _due_entry_data else ""
                if _due_ref_str:
                    _due_ref_dt = datetime.strptime(_due_ref_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                    _ref_ngs_dt = _last_ngs_before(_due_ref_dt - _td(days=1))
                else:
                    _ref_ngs_dt = _last_ngs_before(_thresh2 - _td(days=1))

                _asm_dt   = _ref_ngs_dt - _td(days=13)
                _dnasc_dt = _ref_ngs_dt - _td(days=6)
                _ngs_wd   = _ref_ngs_dt.weekday()
                _lsp_start_dt = (_ref_ngs_dt - _td(days=5) if _ngs_wd == 0
                                 else _ref_ngs_dt - _td(days=3) if _ngs_wd == 3
                                 else None)
                _received_dt = (_ref_ngs_dt - _td(days=3) if _ngs_wd == 0
                                else _ref_ngs_dt - _td(days=1))
                _rel_dt = _ref_ngs_dt + _td(days=1)

                def _posm(dt):
                    return _axpos(dt)

                _chain_pop_id = f"chainpop_{safe_exp_id}"
                _pop_rows = [("Assembly", _asm_dt.strftime("%a %b %-d")),
                             ("ASM NGS", _dnasc_dt.strftime("%a %b %-d"))]
                if _lsp_start_dt:
                    _pop_rows.append(("LSP scale-up", _lsp_start_dt.strftime("%a %b %-d")))
                _pop_rows += [("LSP received", _received_dt.strftime("%a %b %-d")),
                              ("LSP NGS", _ref_ngs_dt.strftime("%a %b %-d")),
                              ("Release", _rel_dt.strftime("%a %b %-d"))]
                _pop_grid = "".join(
                    f'<span style="color:rgba(255,255,255,0.6);font-size:9px;font-weight:700;text-transform:uppercase;">{_k}</span>'
                    f'<span style="font-weight:600;">{_v}</span>'
                    for _k, _v in _pop_rows
                )

                _chain_pop_html = (
                    f'<div id="{_chain_pop_id}" style="display:none;position:absolute;left:38px;top:8px;'
                    f'background:#1e1b4b;color:white;font-size:10px;padding:8px 11px;border-radius:5px;'
                    f'white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,0.7);z-index:100;border:1px solid #34d399;">'
                    f'<div style="font-weight:800;color:white;margin-bottom:5px;font-size:11px;">Last Feasible Cycle</div>'
                    f'<div style="display:grid;grid-template-columns:90px 1fr;gap:2px 8px;">{_pop_grid}</div>'
                    f'</div>'
                )

                # ASM marker (green) + chain popover
                _asm_pos = _posm(_asm_dt)
                _asm_str = _asm_dt.strftime("%a %-m/%-d")
                if 0 <= _asm_pos <= 100:
                    _asm_markers_html += (
                        f'<div style="position:absolute;left:{_asm_pos:.2f}%;top:0px;'
                        f'width:3px;height:24px;background:#f97316;border-radius:1px;'
                        f'box-shadow:0 0 6px rgba(249,115,22,0.55);z-index:5;transform:translateX(-50%);pointer-events:none;"></div>'
                        f'<div style="position:absolute;left:{_asm_pos:.2f}%;top:26px;'
                        f'width:90px;height:16px;transform:translateX(-50%);z-index:26;cursor:pointer;"'
                        f' onmouseenter="document.getElementById(\'{_chain_pop_id}\').style.display=\'block\'"'
                        f' onmouseleave="document.getElementById(\'{_chain_pop_id}\').style.display=\'none\'">'
                        f'<div style="position:absolute;left:50%;top:0;transform:translateX(-50%);'
                        f'background:#f1f5f9;color:#9a3412;font-size:10px;font-weight:700;'
                        f'padding:2px 6px;border-radius:4px;white-space:nowrap;letter-spacing:0.02em;'
                        f'line-height:1;box-shadow:0 1px 2px rgba(0,0,0,0.1);border:1px solid #f97316;">ASM {_asm_str}</div>'
                        f'{_chain_pop_html}'
                        f'</div>'
                    )

                # LSP scale-up marker (sky blue) + same popover
                if _lsp_start_dt:
                    _lsp_pos = _posm(_lsp_start_dt)
                    _lsp_str = _lsp_start_dt.strftime("%a %-m/%-d")
                    _lsp_pop_id = f"lsppop_{safe_exp_id}"
                    _lsp_pop_html = (
                        f'<div id="{_lsp_pop_id}" style="display:none;position:absolute;left:38px;top:8px;'
                        f'background:#1e1b4b;color:white;font-size:10px;padding:8px 11px;border-radius:5px;'
                        f'white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,0.7);z-index:100;border:1px solid #38bdf8;">'
                        f'<div style="font-weight:800;color:white;margin-bottom:5px;font-size:11px;">Last Feasible Cycle</div>'
                        f'<div style="display:grid;grid-template-columns:90px 1fr;gap:2px 8px;">{_pop_grid}</div>'
                        f'</div>'
                    )
                    if 0 <= _lsp_pos <= 100:
                        _asm_markers_html += (
                            f'<div style="position:absolute;left:{_lsp_pos:.2f}%;top:0px;'
                            f'width:3px;height:24px;background:#2563eb;border-radius:1px;'
                            f'box-shadow:0 0 6px rgba(37,99,235,0.5);z-index:5;transform:translateX(-50%);pointer-events:none;"></div>'
                            f'<div style="position:absolute;left:{_lsp_pos:.2f}%;top:26px;'
                            f'width:90px;height:16px;transform:translateX(-50%);z-index:26;cursor:pointer;"'
                            f' onmouseenter="document.getElementById(\'{_lsp_pop_id}\').style.display=\'block\'"'
                            f' onmouseleave="document.getElementById(\'{_lsp_pop_id}\').style.display=\'none\'">'
                            f'<div style="position:absolute;left:50%;top:0;transform:translateX(-50%);'
                            f'background:#f1f5f9;color:#1e40af;font-size:10px;font-weight:700;'
                            f'padding:2px 6px;border-radius:4px;white-space:nowrap;letter-spacing:0.02em;'
                            f'line-height:1;box-shadow:0 1px 2px rgba(0,0,0,0.1);border:1px solid #2563eb;">LSP {_lsp_str}</div>'
                            f'{_lsp_pop_html}'
                            f'</div>'
                        )

                # Last Prod Day marker (teal) — the results/LSP-release day (Tue/Fri),
                # i.e. last NGS + 1. Only shown when there's an Asana override AND it
                # differs from the prod day. Without an override the default DUE marker
                # already sits on the release day, so PROD would duplicate it.
                _show_prod = bool(_due_ref_str) and (_rel_dt.date() != _due_ref_dt.date())
                if _show_prod:
                    _prod_pos = _posm(_rel_dt)
                    _prod_str = _rel_dt.strftime("%a %-m/%-d")
                    _prod_pop_id = f"prodpop_{safe_exp_id}"
                    _prod_pop_html = (
                        f'<div id="{_prod_pop_id}" style="display:none;position:absolute;left:38px;top:8px;'
                        f'background:#1e1b4b;color:white;font-size:10px;padding:8px 11px;border-radius:5px;'
                        f'white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,0.7);z-index:100;border:1px solid #14b8a6;">'
                        f'<div style="font-weight:800;color:white;margin-bottom:5px;font-size:11px;">Last Prod Day</div>'
                        f'<div style="display:grid;grid-template-columns:90px 1fr;gap:2px 8px;">{_pop_grid}</div>'
                        f'</div>'
                    )
                    if 0 <= _prod_pos <= 100:
                        # Second row (top:44px) so it never collides with the ASANA box,
                        # which sits ~1-2 days away on the top row. Line extends down to it.
                        _asm_markers_html += (
                            f'<div style="position:absolute;left:{_prod_pos:.2f}%;top:0px;'
                            f'width:3px;height:44px;background:#0d9488;border-radius:1px;'
                            f'box-shadow:0 0 6px rgba(13,148,136,0.5);z-index:3;transform:translateX(-50%);pointer-events:none;"></div>'
                            f'<div style="position:absolute;left:{_prod_pos:.2f}%;top:44px;'
                            f'width:90px;height:16px;transform:translateX(-50%);z-index:26;cursor:pointer;"'
                            f' onmouseenter="document.getElementById(\'{_prod_pop_id}\').style.display=\'block\'"'
                            f' onmouseleave="document.getElementById(\'{_prod_pop_id}\').style.display=\'none\'">'
                            f'<div style="position:absolute;left:50%;top:0;transform:translateX(-50%);'
                            f'background:#f1f5f9;color:#0f766e;font-size:10px;font-weight:700;'
                            f'padding:2px 6px;border-radius:4px;white-space:nowrap;letter-spacing:0.02em;'
                            f'line-height:1;box-shadow:0 1px 2px rgba(0,0,0,0.1);border:1px solid #0d9488;">PROD {_prod_str}</div>'
                            f'{_prod_pop_html}'
                            f'</div>'
                        )
            except Exception:
                pass

        # Color-coded lane: Partner = deep-purple system, R&D = tech-blue system.
        # Flooded GRADIENT timeline box (Partner=purple, R&D=blue). Week axis + legend
        # ride inside the box (white text); milestone chips sit just below the track.
        _grad = ("linear-gradient(90deg, #7461b8 0%, #a05f8a 100%)" if has_ptr
                 else "linear-gradient(90deg, #3a5c7a 0%, #3d8aa2 100%)")
        # Elapsed-time progress fill (origin -> now), as a fraction of the track.
        # When the experiment is done (nothing active), the clock has stopped — freeze
        # the reference at the last fulfillment so the fill ends there and the live
        # "now" dot is dropped (it'd otherwise keep drifting right forever post-delivery).
        _exp_done = bool(_exp_end_times) and not _exp_any_active
        _now_ref  = max(_exp_end_times) if _exp_done else now
        try:
            _now_pct = _axpos(_now_ref)
        except Exception:
            _now_pct = 0.0

        # Seq-extended axis: a purple line marks 0w (dnasc created = entry), and a green
        # dot marks "now". SEQ / 0w·dnasc labels live in the week header above.
        _seq_marker_html = ""
        if _show_dnasc_start:
            _now_lbl = now.strftime("%-m/%-d")
            # White 0w line (dnasc entry) always shows. The green "now" dot + label only
            # show while the experiment is live — once done the clock is frozen, so a
            # drifting "now" would misrepresent a finished project.
            _seq_marker_html = (
                f'<div title="0w — dnasc created (entry); dnasc TAT starts here" '
                f'style="position:absolute;left:{_dnasc_pct:.2f}%;top:0;width:3px;height:100%;'
                f'background:#ffffff;z-index:6;transform:translateX(-50%);box-shadow:0 0 6px rgba(0,0,0,0.55);"></div>'
            )
            if not _exp_done:
                _seq_marker_html += (
                    f'<div title="now {_now_lbl}" style="position:absolute;left:{_now_pct:.2f}%;top:50%;transform:translate(-50%,-50%);'
                    f'width:12px;height:12px;border-radius:50%;background:#10b981;border:2px solid #fff;z-index:8;box-shadow:0 1px 3px rgba(0,0,0,0.4);"></div>'
                    f'<div style="position:absolute;left:{_now_pct:.2f}%;top:44px;transform:translateX(-50%);'
                    f'background:#ecfdf5;color:#047857;font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;'
                    f'white-space:nowrap;line-height:1;border:1px solid #10b981;z-index:7;pointer-events:none;">now {_now_lbl}</div>'
                )
        # Greyed hatched band over the pre-dnasc lead (seq → dnasc start) = upstream.
        _upstream_band = ""
        if _show_dnasc_start and _dnasc_pct > 0:
            _up_pop_id = f"uppop_{safe_exp_id}"
            _upstream_band = (
                # Hatched band (visual only)
                f'<div style="position:absolute;left:0;top:0;width:{_dnasc_pct:.2f}%;height:100%;'
                f'background:repeating-linear-gradient(45deg,rgba(71,85,105,0.30),rgba(71,85,105,0.30) 5px,rgba(71,85,105,0.12) 5px,rgba(71,85,105,0.12) 10px);'
                f'border-right:1px dashed rgba(255,255,255,0.7);border-radius:11px 0 0 11px;z-index:2;pointer-events:none;"></div>'
                # Hover zone on top (above the dots layer) + styled popover
                f'<div style="position:absolute;left:0;top:0;width:{_dnasc_pct:.2f}%;height:100%;z-index:27;cursor:help;"'
                f' onmouseenter="document.getElementById(\'{_up_pop_id}\').style.display=\'block\'"'
                f' onmouseleave="document.getElementById(\'{_up_pop_id}\').style.display=\'none\'">'
                f'<div id="{_up_pop_id}" style="display:none;position:absolute;left:2px;top:26px;'
                f'background:#1e1b4b;color:white;font-size:10px;padding:8px 11px;border-radius:5px;'
                f'white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,0.7);z-index:100;border:1px solid #94a3b8;">'
                f'<div style="font-weight:800;margin-bottom:4px;">Experiment Start (upstream)</div>'
                f'<div>Seq transferred {_seq_exp_dt:%a %b %-d}<br>'
                f'<span style="color:rgba(255,255,255,0.6);">+{_lead_days}d before dnasc · not counted in TAT</span></div>'
                f'</div></div>'
            )
        # Elapsed (now) fill starts at 0w (dnasc entry), not the seq edge.
        _nowfill_left  = _dnasc_pct
        _nowfill_width = max(0.0, _now_pct - _dnasc_pct)
        # Gridlines: dnasc weeks 0w..8w by date when seq-extended, else even eighths.
        if _show_dnasc_start:
            _gridlines_html = "".join(
                f'<div style="position:absolute; left:{_hpos(exp_created_dt + _td(weeks=w)):.2f}%; top:0; width:1px; height:100%; background:rgba(255,255,255,0.25); z-index:1;"></div>'
                for w in range(0, 9))
        else:
            _gridlines_html = "".join(
                f'<div style="position:absolute; left:{(w/_tl_weeks)*100:.2f}%; top:0; width:1px; height:100%; background:rgba(255,255,255,0.25); z-index:1;"></div>'
                for w in range(1, _tl_weeks))
        # Legend (seq-extended axis only). Only list keys actually drawn: drop the
        # upstream-band item when there's no lead, and the "now" dot when the clock is
        # frozen (experiment done).
        _tl_legend = ""
        if _show_dnasc_start:
            _legend_items = []
            if _lead_days > 0:
                _legend_items.append('<span style="display:flex;align-items:center;gap:5px;"><span style="width:18px;height:9px;background:repeating-linear-gradient(45deg,rgba(255,255,255,0.55),rgba(255,255,255,0.55) 3px,rgba(255,255,255,0.15) 3px,rgba(255,255,255,0.15) 6px);border:1px solid rgba(255,255,255,0.5);"></span>upstream seq→dnasc (not in TAT)</span>')
            _legend_items.append('<span style="display:flex;align-items:center;gap:5px;"><span style="width:3px;height:11px;background:#ffffff;"></span>0w = dnasc created (entry)</span>')
            if not _exp_done:
                _legend_items.append('<span style="display:flex;align-items:center;gap:5px;"><span style="width:10px;height:10px;border-radius:50%;background:#10b981;border:1px solid #fff;"></span>now</span>')
            _tl_legend = (
                '<div style="display:flex;gap:16px;align-items:center;margin-top:8px;font-size:9px;color:rgba(255,255,255,0.92);flex-wrap:wrap;">'
                + "".join(_legend_items)
                + '</div>'
            )
        timeline_bar = f"""<div style="margin:8px 0 4px 0; padding:12px 14px; border-radius:8px; background:{_grad}; box-shadow:0 1px 3px rgba(15,23,42,0.18);"><div style="position:relative; width:100%; height:42px; margin-bottom:6px;">{_weeks_header_html}</div><div style="position:relative; width:100%; height:22px; margin-bottom:46px; background:rgba(255,255,255,0.14); border-radius:11px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.25); border:1px solid #cbd5e1;"><div style="position:absolute;left:{_nowfill_left:.1f}%;top:0;height:100%;width:{_nowfill_width:.1f}%;background:rgba(255,255,255,0.33);z-index:1;"></div>{_upstream_band}{_gridlines_html}{_orange_html}{_red_html}<div style="position:absolute; width:100%; height:100%; top:50%; left:0; z-index:10;">{dots_html}</div>{_default_bracket_html}{_due_marker_html}{_asm_markers_html}{_seq_marker_html}</div></div>"""

        if experiment_active_map is not None:
            db_active = experiment_active_map.get(experiment_name, True)
        else:
            _ea_vals = project_df['experiment_active'].dropna() if 'experiment_active' in project_df.columns else pd.Series([], dtype=object)
            db_active = bool((_ea_vals.astype(str).str.lower().isin(['true', '1'])).any()) if not _ea_vals.empty else True
        exp_header_gradient = "#ffffff"  # Option B: header stays white; color lives in the track bar

        _exp_emails_raw = [str(e).strip() for e in project_df['submitter_email'].dropna().unique() if str(e).strip() not in ('', 'nan', 'none', 'None')] if 'submitter_email' in project_df.columns else []
        _exp_email_str = ' / '.join(_exp_emails_raw[:2]) if 1 <= len(_exp_emails_raw) <= 2 else ''

        # ── TOTAL RUNNING / TOTAL headline ────────────────────────────────────
        # Elapsed since the project's earliest real start: seq transfer if present,
        # else dnasc-created (fallback also covers a missing/invalid seq so the box
        # never breaks or goes negative). Live "RUNNING" while any request is active;
        # frozen "TOTAL (seq → fulfilled)" once everything is done. Includes the
        # upstream lead — deliberately labeled so it's not mistaken for TAT.
        _run_html = ""
        _run_start = _seq_exp_dt if _seq_exp_dt else exp_created_dt
        if _run_start and exp_created_dt and _run_start > exp_created_dt:
            _run_start = exp_created_dt
        if _run_start and not _is_infra_exp:
            if _exp_any_active or not _exp_end_times:
                _run_end, _running = now, True
            else:
                _run_end, _running = max(_exp_end_times), False
            _run_days = max(0, (_run_end.date() - _run_start.date()).days)
            _rw, _rd = _run_days // 7, _run_days % 7
            _run_dur = f"{_rw}w {_rd}d" if _rw else f"{_rd}d"
            _from_seq = bool(_seq_exp_dt and _run_start == _seq_exp_dt)
            if _running:
                _run_lbl = "RUNNING (since sequence transfer)" if _from_seq else "RUNNING"
                _run_tip = (f"Elapsed since {'sequences transferred' if _from_seq else 'dnasc created'} "
                            f"{_run_start:%b %-d} — live clock; includes upstream lead, unlike the dnasc TAT.")
                _run_bg, _run_fg, _run_bd = "#eff6ff", "#1e40af", "#bfdbfe"
            else:
                _run_lbl = "TOTAL"
                _run_tip = (f"Final elapsed {_run_start:%b %-d} → {_run_end:%b %-d} "
                            f"({'seq' if _from_seq else 'dnasc'} → fulfilled); includes upstream lead, unlike TAT.")
                _run_bg, _run_fg, _run_bd = "#f0fdf4", "#166534", "#bbf7d0"
            _run_html = (
                f'<span class="kpill" title="{_run_tip}" style="margin-left:auto;background:{_run_bg};border:1px solid {_run_bd};">'
                f'<span class="kk" style="color:{_run_fg};">{_run_lbl}</span> '
                f'<b style="color:{_run_fg};">{_run_dur}</b></span>'
            )

        # Tiny centered caption under the timeline stating what the completion
        # dots represent: Avg Available (VRT002-and-newer) or Avg QC Confirmed
        # (older projects), matching _use_delivered. Hover a dot for the exact
        # milestone + date + TAT.
        _cap_metric = ('Avg. Available <span style="color:#94a3b8;">(LSP delivered)</span>'
                       if _use_delivered else 'Avg. QC Confirmed')
        _timeline_note = (
            '<div style="text-align:right; margin:5px 2px 0; font-size:9px;'
            ' font-weight:600; color:#64748b; letter-spacing:.02em;">'
            '<span style="display:inline-block; width:8px; height:8px;'
            ' background:#94a3b8; border:1.5px solid #1e3a5f; transform:rotate(45deg);'
            ' vertical-align:middle; margin-right:6px;"></span>'
            '= ' + _cap_metric + '</div>'
        )

        html += f"""
            <div class="project-wrapper" data-active="{"true" if db_active else "false"}" data-due-date="{_sort_due_date}" style="border-left:4px solid {_card_accent};">
                <div class="header-banner" style="background: {exp_header_gradient}; min-height: auto; padding: 12px 18px;" onclick="toggleSection('{safe_exp_id}')">
                    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:8px;">
                        <div class="header-title" style="margin-bottom:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:55%;">{experiment_name}</div>
                        {_run_badge}
                        {_missing_due_badge}
                        {exp_customer_tags}
                        <span class="kpill"><span class="kk">Requests:</span><b>{len(req_groups)}</b></span>
                        <span class="kpill"><span class="kk">Fulfilled:</span><b style="color:#2563eb;">{count_fulfilled}</b></span>
                        {avg_tat_html}
                        {_run_html}
                        <button id="bucket_btn_{safe_exp_id}" onclick="event.stopPropagation();toggleBucketView('{safe_exp_id}')" style="margin-left:8px;background:#fff;border:1px solid #e5e7eb;color:#374151;font-size:10px;padding:3px 9px;border-radius:6px;cursor:pointer;font-weight:600;white-space:nowrap;">Stage View</button>
                    </div>
                    <div style="font-size:10px; color:#9ca3af; margin-bottom:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">Created: {exp_created_str}{(f' &nbsp;<span style="color:#2563eb; font-weight:600;">' + _exp_email_str + '</span>') if _exp_email_str else ''}</div>
                    <div id="timeline_{safe_exp_id}">{timeline_bar}</div>
                    {_timeline_note}
                    <div id="bucket_{safe_exp_id}" style="display:none;background:#f8f9fa;border-radius:8px;border:1px solid #e5e7eb;margin-top:8px;">{_render_bucket_chart(stage_counts, fulfilled_week_counts, orange_week, red_week, stage_items, ramp=_BLUE_RAMP if has_ptr else _PURPLE_RAMP)}</div>
                    <div class="header-stats" style="margin-top: 0; display: flex; gap: 6px; flex-wrap: wrap;">
                        {f'<span class="stat-item" title="Requests submitted but no work has started yet" style="background:rgba(217,119,6,0.6); border:1px solid rgba(255,255,255,0.4);"><span class="stat-label" style="font-size:11px;">{count_new}</span> <span style="font-size:10px;">New</span></span>' if count_new > 0 else ''}
                        {f'<span class="stat-item" title="LSP prep complete — ready to ship" style="background:rgba(34,197,94,0.4); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">{count_ship_ready}</span> <span style="font-size:10px;">Ship Ready</span></span>' if count_ship_ready > 0 else ''}
                        {f'<span class="stat-item" title="Requests actively in LSP processing" style="background:rgba(8,145,178,0.4); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">{count_in_lsp}</span> <span style="font-size:10px;">In LSP</span></span>' if count_in_lsp > 0 else ''}
                        {f'<span class="stat-item" title="Requests actively in assembly (GG/Gibson)" style="background:rgba(124,58,237,0.4); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">{count_in_assembly}</span> <span style="font-size:10px;">In Assembly</span></span>' if count_in_assembly > 0 else ''}
                        {f'<span class="stat-item" title="Requests waiting on parts or upstream dependencies" style="background:rgba(249,115,22,0.4); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">{count_active_waiting}</span> <span style="font-size:10px;">Waiting</span></span>' if count_active_waiting > 0 else ''}
                        {f'<span class="stat-item" title="No pipeline progress detected — may need intervention" style="background:rgba(190,24,93,0.5); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">⚠️ {count_stalled}</span> <span style="font-size:10px;">Stalled</span></span>' if count_stalled > 0 else ''}
                        {f'<span class="stat-item" title="Assembly needs review before proceeding" style="background:rgba(217,119,6,0.5); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">🔬 {count_asm_review}</span> <span style="font-size:10px;">ASM Review</span></span>' if count_asm_review > 0 else ''}
                        {f'<span class="stat-item" title="A sequencing winner has been identified — ready for LSP" style="background:rgba(5,150,105,0.5); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">🏆 {count_seq_winner}</span> <span style="font-size:10px;">Seq Winner</span></span>' if count_seq_winner > 0 else ''}
                        {f'<span class="stat-item" title="Parts order submitted to synthesis vendor — waiting on delivery" style="background:rgba(124,58,237,0.5); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">⏳ {count_order_pending}</span> <span style="font-size:10px;">Order Pending</span></span>' if count_order_pending > 0 else ''}
                        {f'<span class="stat-item" title="A Gibson or Golden Gate workorder has an antibiotic that does not match LIMS" style="background:rgba(220,38,38,0.6); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">🚨 {count_antibiotic_mismatch}</span> <span style="font-size:10px;">Antibiotic Mismatch</span></span>' if count_antibiotic_mismatch > 0 else ''}
                        {f'<span class="stat-item" title="LIMS lists two bacterial antibiotics on a plasmid (often NeoR mis-flagged as Kan) — correct one is selected, but the BIOS/LIMS record should be fixed" style="background:rgba(245,158,11,0.5); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">⚠️ {count_dual_antibiotic}</span> <span style="font-size:10px;">Dual Antibiotic (LIMS)</span></span>' if count_dual_antibiotic > 0 else ''}
                        {f'<span class="stat-item" title="Assembly is blocked — upstream dependency unresolved" style="background:rgba(190,24,93,0.5); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">{count_blocked}</span> <span style="font-size:10px;">Blocked</span></span>' if count_blocked > 0 else ''}
                        {f'<span class="stat-item" title="Requests canceled" style="background:rgba(100,116,139,0.4); border:1px solid #cbd5e1;"><span class="stat-label" style="font-size:11px;">{count_canceled}</span> <span style="font-size:10px;">Canceled</span></span>' if count_canceled > 0 else ''}
                    </div>
                </div>"""

        if active_req_list:
            html += f"""
                <details>
                    <summary class="group-header in-progress">
                        <span class="group-arrow">▶</span> Planned / In-Progress ({len(active_req_list)}{f' - ⚠️ {count_stalled} Stalled' if count_stalled > 0 else ''}{f' - 🔬 {count_asm_review} ASM Review' if count_asm_review > 0 else ''}{f' - 🏆 {count_seq_winner} Seq Winner' if count_seq_winner > 0 else ''}{f' - ⏳ {count_order_pending} Order Pending' if count_order_pending > 0 else ''}{f' - 🚨 {count_antibiotic_mismatch} Antibiotic Mismatch' if count_antibiotic_mismatch > 0 else ''}{f' - ⚠️ {count_dual_antibiotic} Dual Antibiotic' if count_dual_antibiotic > 0 else ''})
                    </summary>"""
            _active_parts = []
            for rid, r_df in active_req_list:
                is_stalled_req = rid in stalled_reqs
                is_asm_review_req = rid in asm_review_reqs
                is_seq_winner_req    = rid in seq_winner_reqs
                is_order_pending_req = bool(r_df['has_order_pending'].iloc[0]) if 'has_order_pending' in r_df.columns else False
                is_antibiotic_mismatch_req = rid in antibiotic_mismatch_reqs
                _active_parts.append(render_single_request_html(rid, r_df, is_stalled_req, is_asm_review_req, is_seq_winner_req, is_order_pending_req, is_antibiotic_mismatch_req))
            html += "".join(_active_parts)
            html += "</details>"
        if new_req_list:
            html += f'<details><summary class="group-header new"><span class="group-arrow">▶</span> New ({len(new_req_list)})</summary>'
            _new_parts = []
            for rid, r_df in new_req_list: _new_parts.append(render_single_request_html(rid, r_df, False))
            html += "".join(_new_parts)
            html += "</details>"
        if fulfilled_req_list:
            html += f'<details><summary class="group-header fulfilled"><span class="group-arrow">▶</span> Fulfilled ({len(fulfilled_req_list)})</summary>'
            _fulfilled_parts = []
            for rid, r_df in fulfilled_req_list: _fulfilled_parts.append(render_single_request_html(rid, r_df, False))
            html += "".join(_fulfilled_parts)
            html += "</details>"
        if canceled_req_list:
            html += f'<details><summary class="group-header canceled"><span class="group-arrow">▶</span> Canceled ({len(canceled_req_list)})</summary>'
            _canceled_parts = []
            for rid, r_df in canceled_req_list:
                _canceled_parts.append(render_single_request_html(rid, r_df, False))
            html += "".join(_canceled_parts)
            html += "</details>"

        html += "</div>"
        _flush()  # stream this project's HTML to disk; reset the buffer

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

            <!-- REQUESTS IN FLIGHT TAB -->
            <div id="tab-inflight" class="tab-content" style="padding:0;overflow-y:auto;height:calc(100vh - 130px);">
                __INFLIGHT_FRAGMENT__
            </div>

            <!-- PARTS INVENTORY TAB -->
            <div id="tab-parts" class="tab-content" style="padding:0;overflow-y:auto;height:calc(100vh - 130px);">
                __PARTS_FRAGMENT__
            </div>

        </div>
    </div>
    """
    html = html.replace("__LSP_CAPACITY_TAB_CONTENT__", lsp_capacity_html)
    html = html.replace("__INFLIGHT_FRAGMENT__", _inflight_fragment)
    html = html.replace("__PARTS_FRAGMENT__", _parts_fragment)
    # Deduped plate-popover pool, emitted once after the body. Build sites emitted
    # empty <div class="plate-popover" data-pop="N">; JS fills each from PLATE_POP
    # on first hover/click. ~91% of popovers are dupes, so this replaces ~21 MB of
    # inlined content with a ~2 MB pool.
    _pool_list = [None] * len(_popover_pool)
    for _content, _idx in _popover_pool.items():
        _pool_list[_idx] = _content
    # Escape </ so popover HTML can't break out of the <script>. Kept on its own
    # line (not inside the f-string expression) — Python < 3.12 forbids backslashes
    # inside f-string replacement fields, and the server runs 3.9.
    _pool_js = json.dumps(_pool_list).replace("</", "<\\/")
    html += f"\n<script>window.PLATE_POP={_pool_js};</script>\n"
    _flush()  # stream the document tail (LSP + inflight tabs) to disk

    _uniq_missing = sorted(set(_missing_partner_exps))
    if _uniq_missing:
        log.warning("Partner experiments missing an Asana due date (%d): %s",
                    len(_uniq_missing), ", ".join(_uniq_missing))
    # Persist the missing list so full_refresh.py can append these names to the sheet.
    try:
        import json as _json
        from pathlib import Path as _Path
        _mp = _Path("dashboard_state/missing_asana_dates.json")
        _mp.parent.mkdir(parents=True, exist_ok=True)
        _mp.write_text(_json.dumps(_uniq_missing, indent=2))
    except Exception:
        pass
    return html


# ── Public API ────────────────────────────────────────────────────────────────

def render_dashboard(df: pd.DataFrame, experiment_active_map: dict | None = None, out_path=None):
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
    generated_ts = int(time.time())

    # Full HTML document template. The __INNER_BODY__ sentinel is replaced with
    # the rendered per-project body below (default mode), or split on so the
    # body can be streamed straight to disk between the head and tail.
    _doc_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>DNA SC Dashboard</title>
    <script>
    (function() {{
        var _VER = {generated_ts};
        var _base = window.location.pathname.replace(/\/[^/]+$/, '/');
        function _reload() {{ window.location.href = window.location.pathname + '?v=' + Date.now(); }}
        function _checkAndReload() {{
            fetch(_base + 'dnasc_version.txt?_nc=' + Date.now(), {{cache: 'no-store'}})
                .then(function(r) {{ return r.text(); }})
                .then(function(t) {{ if (parseInt(t, 10) > _VER + 120) {{ _reload(); }} }})
                .catch(function() {{ _reload(); }});
        }}
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'F5' || (e.ctrlKey && e.key === 'r') || (e.metaKey && e.key === 'r')) {{
                e.preventDefault();
                _checkAndReload();
            }}
        }});
    }})();
    </script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
    #_loading_overlay {{
        position: fixed; inset: 0; z-index: 99999;
        background: #f8fafc;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        gap: 14px;
        transition: opacity 0.25s ease;
    }}
    #_loading_overlay.fade-out {{ opacity: 0; pointer-events: none; }}
    #_loading_overlay .spinner {{
        width: 36px; height: 36px;
        border: 3px solid #cbd5e1;
        border-top-color: #6366f1;
        border-radius: 50%;
        animation: _spin 0.7s linear infinite;
    }}
    @keyframes _spin {{ to {{ transform: rotate(360deg); }} }}
    #_loading_overlay .label {{
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 13px; color: #6b7280; letter-spacing: 0.02em;
    }}
    </style>
</head>
<body>
<div id="_loading_overlay"><div class="spinner"></div><div class="label">Loading dashboard…</div></div>
__INNER_BODY__
</body>
</html>"""

    _render_kwargs = dict(
        logo_b64               = _ASSETS["logo"],
        tracking_icon_b64      = _ASSETS["tracking"],
        metrics_icon_b64       = _ASSETS["metrics"],
        cost_icon_b64          = _ASSETS["cost"],
        generated_at           = generated_at,
        experiment_active_map  = experiment_active_map,
    )

    if out_path is not None:
        # Stream the document to disk: head, then each project's body (flushed and
        # freed as it renders), then tail. The full ~160 MB string is never held in
        # memory at once — this is what keeps render under the server's RAM limit.
        #
        # Write to a sibling temp file, then atomically os.replace() it onto the
        # served path. The previous approach opened the served file itself in "w"
        # mode, which truncates it instantly and leaves it half-written for the
        # many seconds the ~160 MB stream takes. Any browser fetch during that
        # window (10-min auto-refresh, manual reload, or a concurrent re-run) got a
        # truncated document — and since the metrics/costs/capacity/inflight tabs
        # are emitted last (after the giant Tracking body), those tabs were the
        # ones that came up blank. os.replace() is atomic on POSIX, so a reader
        # always sees either the complete old file or the complete new one.
        _head, _tail = _doc_template.split("__INNER_BODY__")
        out_path = Path(out_path)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as _fh:
            _fh.write(_head)
            render_all_projects_dashboard(df, out_fh=_fh, **_render_kwargs)
            _fh.write(_tail)
        os.replace(tmp_path, out_path)
        log.info("Dashboard rendered successfully")
        return None

    inner = render_all_projects_dashboard(df, **_render_kwargs)
    html = _doc_template.replace("__INNER_BODY__", inner)
    log.info("Dashboard rendered successfully")
    return html

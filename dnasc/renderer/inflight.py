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
        })

    # Sort: pinned last; flagged experiments first; within experiment flagged rows
    # first; then by assembly date. exp_min and exp come BEFORE per-row flags so
    # all rows of an experiment stay contiguous (ifRender relies on this).
    _exp_min = {}
    for r in records:
        if r['assembly']:
            _exp_min[r['exp']] = min(_exp_min.get(r['exp'], '9999'), r['assembly'])
    _exp_has_flags = {r['exp'] for r in records if r['flags']}
    records.sort(key=lambda r: (
        1 if r['pinned'] else 0,
        0 if r['exp'] in _exp_has_flags else 1,
        _exp_min.get(r['exp'], '9999'),
        r['exp'],
        0 if r['flags'] else 1,
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
  var S_ST = {{
    'IN_PROGRESS':'background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;',
    'FULFILLED':  'background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;',
    'CANCELED':   'background:#f5f5f7;color:#6b7280;border:1px solid #d1d5db;',
    'PLANNED':    'background:#f5f3ff;color:#6d28d9;border:1px solid #ddd6fe;',
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
  var F_BG = {{'PAST_DUE':'#fff5f5','AT_RISK':'#fffef0','STALLED':'#fafafa'}};
  var BDG = 'display:inline-block;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:700;white-space:nowrap;margin:1px 1px;';
  var TD  = 'padding:5px 6px;border-bottom:1px solid #f0f0f2;vertical-align:top;font-size:10px;';

  function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}}
  function bdg(t,st){{return '<span style="'+BDG+st+'">'+esc(t)+'</span>';}}
  var PAI_STY   ='display:inline-block;background:#ede9fe;color:#6d28d9;border:1px solid #c4b5fd;padding:1px 4px;border-radius:2px;font-family:monospace;font-weight:700;font-size:9px;white-space:nowrap;margin:1px 1px;';
  var PAI_STY_RD='display:inline-block;background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;padding:1px 4px;border-radius:2px;font-family:monospace;font-weight:700;font-size:9px;white-space:nowrap;margin:1px 1px;';
  function paiBadges(s,cust){{if(!s)return'';var st=cust==='R_D'?PAI_STY_RD:PAI_STY;return s.split(',').map(function(p){{p=p.trim();return p?'<span style="'+st+'">'+esc(p)+'</span>':'';}}).join('');}}
  var CUST_MAP={{'R_D':['R&D','#f0fdf4','#166534'],'INTERNAL_CLD':['CLD','#dbeafe','#1d4ed8'],'TECH_OUT':['Tech Out','#ffedd5','#c2410c'],'EXTERNAL_TECH_OUT':['Ext TechOut','#fce7f3','#be185d']}};
  function custBadge(s,fp){{var m=CUST_MAP[s]||['—','#f3f4f6','#6b7280'];return'<span style="padding:2px 6px;border-radius:3px;font-size:10px;background:'+m[1]+';color:'+m[2]+';">'+m[0]+'</span>';}}
  var _DPILL='display:inline-block;padding:0px 5px;border-radius:3px;font-size:9px;font-weight:600;white-space:nowrap;margin-top:2px;';
  function fmtDate(s){{if(!s)return'';var diff=Math.round((new Date(s)-new Date(_TODAY))/(864e5));var bg,clr,lbl;if(diff<0){{bg='#fee2e2';clr='#991b1b';lbl=Math.abs(diff)+'d ago';}}else if(diff===0){{bg='#fef3c7';clr='#92400e';lbl='today';}}else if(diff<=7){{bg='#fef9c3';clr='#713f12';lbl='in '+diff+'d';}}else{{bg='#f3f4f6';clr='#6b7280';lbl='in '+diff+'d';}}return'<span style="color:#374151;">'+esc(s)+'</span><br><span style="background:'+bg+';color:'+clr+';'+_DPILL+'">'+lbl+'</span>';}}
  function fmtSubmitter(s){{if(!s||s.indexOf('@')===-1)return esc(s);var parts=s.split('@');var local=parts[0];var domain=parts[1];var org=domain.split('.')[0];org=org.charAt(0).toUpperCase()+org.slice(1);var name=local.split('.').map(function(p){{return p.charAt(0).toUpperCase()+p.slice(1);}}).join(' ');var ext=!domain.toLowerCase().startsWith('asimov.');var orgSty=ext?'display:inline-block;font-size:9px;font-weight:600;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;border-radius:3px;padding:1px 5px;margin-top:1px;':'display:block;color:#9ca3af;font-size:9px;';return'<span style="display:block;">'+esc(name)+'</span><span style="'+orgSty+'">'+esc(org)+'</span>';}}

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
    var st  = bdg(r.status.replace(/_/g,' '), S_ST[r.status]||'background:#f5f5f7;color:#6b7280;border:1px solid #d1d5db;');
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
  window.ifRender = function() {{
    window.ifBuildHead();
    var tbody = document.getElementById('inflight-tbody');
    if (!tbody) return;
    var expOrder = [], buckets = {{}};
    _IFD.forEach(function(r) {{
      if (!_pass(r)) return;
      if (!buckets.hasOwnProperty(r.exp)) {{ expOrder.push(r.exp); buckets[r.exp] = {{rows:[], fp:r.fp, pinned:r.pinned}}; }}
      buckets[r.exp].rows.push(r);
    }});
    var html = '';
    expOrder.forEach(function(exp) {{
      var g = buckets[exp];
      var grpSt = g.pinned
        ? 'background:linear-gradient(90deg,#f3f4f6,#f9fafb);border-top:2px solid #9ca3af;color:#4b5563;'
        : g.fp
          ? 'background:linear-gradient(90deg,#ede9fe,#faf5ff);border-top:2px solid #7c3aed;color:#4c1d95;'
          : 'background:linear-gradient(90deg,#dbeafe,#eff6ff);border-top:2px solid #2563eb;color:#1e3a8a;';
      html += '<tr class="if-grp"><td colspan="14" style="padding:4px 8px;font-size:10px;font-weight:700;'+grpSt+'">'
            + esc(exp) + (g.fp ? ' ★' : '') + '</td></tr>';
      g.rows.forEach(function(r) {{ html += _rowHtml(r); }});
    }});
    if (!html) html = '<tr><td colspan="14" style="padding:20px;color:#6b7280;font-size:11px;text-align:center;">No matching requests.</td></tr>';
    tbody.innerHTML = html;
  }};

  // ── Sort — sorts _IFD then re-renders (group headers rebuild correctly) ───
  var _sortKey = null, _sortDir = 1;
  function _ifSort(k) {{
    _sortDir = (_sortKey === k) ? _sortDir * -1 : 1;
    _sortKey = k;
    _IFD.sort(function(a,b) {{
      if (a.pinned !== b.pinned) return a.pinned ? 1 : -1;
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

  // ── Build column headers (called once on tab show) ────────────────────────
  var _headBuilt = false;
  window.ifBuildHead = function() {{
    if (_headBuilt) return;
    _headBuilt = true;
    var thead = document.getElementById('inflight-thead');
    if (!thead) return;
    var TH = 'padding:5px 6px;text-align:left;border-bottom:2px solid #d1d5db;font-size:9px;color:#374151;'
           + 'font-weight:700;text-transform:uppercase;background:#f9fafb;position:sticky;top:0;z-index:2;'
           + 'white-space:nowrap;cursor:pointer;user-select:none;';
    var COLS = [
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

}})();
</script>
"""

"""Colony Picking tab — paste a colony-count export, get the pick decision now.

The dashboard is a static HTML rendered on a cron, so a colony count taken this
morning is invisible until the next refresh. This tab closes that gap without a
server: paste the export straight out of Google Sheets (Sheets copies as TSV), and
the page parses and scores it entirely in the browser.

Scoring uses the SAME thresholds as the In-Flight tab, injected from PipelineConfig
rather than duplicated here, so the two views can never disagree about what LOW
means.

Editable Manual Pick column: edits are highlighted and can be copied back out as a
column to paste into the sheet — bold on yellow, which Sheets honours.

Saved state survives the dashboard's own auto-reload (a new render landing mid-session
would otherwise wipe a half-finished set of picks) but NOT a deliberate refresh, which
comes up empty. See isManualRefresh().
"""
from __future__ import annotations

from dnasc.config import PipelineConfig

# Picks wanted, keyed to the YIELD band — how many colonies actually grew. A thin plate
# needs more taken to have a fair chance of a confirmed clone, so the target rises as the
# count falls. With two strains the target is a TOTAL and any split satisfies it (4+4, 8+0).
#
# Keyed to Yield, NOT Risk. They are near-inverses drawn from the same number, so having
# Risk set the target meant a plate showing "HIGH yield / ok" still asked for the mid-tier
# count, which reads as a contradiction on the row.
#
# On a thin plate the target is capped at what physically grew, so the tab never asks for 8
# colonies off a plate that produced 3. The wanted figure is kept alongside it (g.want) because
# hitting a capped target is not success: a plate that gave everything it had and still came up
# short reads "missed target", and one that grew nothing reads "cannot pick".
#
# NOTE on the rich tier: measured over 2,600 attempts with sequencing outcomes, plates with
# 12+ pickable confirmed a clone 48.7% of the time when 2 colonies were taken (n=76) against
# 71.8% at 4 (n=1,288) — fewer draws at a ~0.335 per-colony confirm rate. 2 is the lab's
# call for plates with colonies to spare; a failure there is cheap to repick, which the
# success rate on its own does not capture.
PICK_TARGET_THIN = 8      # LOW yield  — capped at the colonies available
PICK_TARGET_MED  = 4      # MED yield
PICK_TARGET_RICH = 2      # HIGH yield — plenty of colonies, take a couple

# The pipeline re-runs an attempt on its own only when BOTH hold: the workorder is a Gibson or
# Golden Gate (below), and it came back with fewer than this many picked colonies.
AUTO_RETRY_UNDER  = 4

# Coverage is decided by ASSEMBLY TYPE alone. Gibson and Golden Gate are re-run automatically;
# Transformation never is. The strain does not enter into it — a Gibson in NEB10b or EPI400 is
# covered exactly as a NEBstable one is. So a HIGH RISK attempt falls through the gap, and needs
# putting back in by hand, when it is either a Transformation or a Gibson/Golden Gate that
# already reached the pick threshold. Matched as a substring of the export's Assembly Type,
# case-insensitively, so "Golden Gate" and "GoldenGate" both land.
AUTO_RETRY_TYPES = ("gibson", "golden")

# When an attempt is short of its pick target and both strains have colonies left, take from
# NEBstable first — it is the preferred strain for downstream work, so the extra picks are
# worth more there. EPI400 only makes up whatever NEBstable cannot cover.
STRAIN_PRIORITY = ["NEBstable", "EPI400"]


def render_colony_pick_tab() -> str:
    """Return the Colony Picking tab fragment (scoped style + markup + script)."""
    js_nums = {
        "__BAND_LOW__": PipelineConfig.PICK_BAND_LOW_MAX,
        "__BAND_MED__": PipelineConfig.PICK_BAND_MED_MAX,
        "__RISK_HIGH__": PipelineConfig.COLONY_RISK_HIGH_MAX,
        "__RISK_MED__": PipelineConfig.COLONY_RISK_MED_MAX,
        "__T_THIN__": PICK_TARGET_THIN,
        "__T_MED__": PICK_TARGET_MED,
        "__T_RICH__": PICK_TARGET_RICH,
        "__AUTO_UNDER__": AUTO_RETRY_UNDER,
    }
    frag = _FRAGMENT.replace("__AUTO_TYPES__",
                             ",".join('"%s"' % t.lower() for t in AUTO_RETRY_TYPES))
    frag = frag.replace("__PRIORITY__", ",".join('"%s"' % x for x in STRAIN_PRIORITY))
    for token, val in js_nums.items():
        frag = frag.replace(token, str(val))
    return frag


_FRAGMENT = r"""
<style>
 #tab-cpick{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#e9ecf2;color:#1d1d1f;padding:0 0 40px}
 #tab-cpick .cp-hd{background:#fff;border-bottom:1px solid #e5e7eb;padding:16px 20px}
 #tab-cpick .cp-hd h1{font-size:18px;margin:0 0 4px}
 #tab-cpick .cp-hd p{font-size:12px;color:#6b7280;margin:0}
 #tab-cpick .cp-box{margin:16px;background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px}
 #tab-cpick textarea{width:100%;height:110px;font-family:ui-monospace,Menlo,monospace;font-size:10px;border:1px solid #d1d5db;border-radius:6px;padding:8px;resize:vertical;box-sizing:border-box}
 #tab-cpick .cp-btn{font-size:11px;font-weight:600;padding:6px 13px;border-radius:6px;cursor:pointer;border:1px solid #93c5fd;background:#dbeafe;color:#1d4ed8}
 #tab-cpick .cp-btn.sec{border-color:#d1d5db;background:#f9fafb;color:#374151}
 #tab-cpick .cp-btn:disabled{opacity:.45;cursor:default}
 #tab-cpick .cp-kpis{display:flex;gap:10px;margin:0 16px 4px;flex-wrap:wrap}
 #tab-cpick .cp-k{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:8px 14px;min-width:96px}
 #tab-cpick .cp-kn{font-size:20px;font-weight:700;line-height:1}
 #tab-cpick .cp-kl{font-size:10px;color:#6b7280;margin-top:2px}
 #tab-cpick table{width:100%;border-collapse:collapse;font-size:11.5px;background:#fff}
 #tab-cpick th{text-align:left;padding:7px 8px;font-size:9px;letter-spacing:.04em;color:#0f172a;border-bottom:1px solid #cbd5e1;background:#f8fafc;position:sticky;top:0;z-index:2}
 #tab-cpick td{padding:5px 8px;border-bottom:1px solid #f1f5f9;vertical-align:middle}
 #tab-cpick tr.cp-grp td{border-top:2px solid #e5e7eb}
 #tab-cpick tr.cp-edited td{background:#fffbeb}
 #tab-cpick tr.cp-edited td:first-child{box-shadow:inset 3px 0 0 #f59e0b}
 #tab-cpick .cp-pill{display:inline-block;font-size:8px;font-weight:700;padding:1px 6px;border-radius:8px;white-space:nowrap}
 #tab-cpick .cp-inp{width:46px;font-size:11px;padding:2px 4px;border:1px solid #d1d5db;border-radius:4px;text-align:center;font-family:inherit}
 #tab-cpick .cp-inp:focus{outline:2px solid #93c5fd;border-color:#93c5fd}
 #tab-cpick .cp-mono{font-family:ui-monospace,Menlo,monospace}
 #tab-cpick .cp-note{font-size:11px;color:#6b7280;margin:8px 16px}
 #tab-cpick .cp-err{color:#b91c1c;font-size:11px;margin-top:6px}
 #tab-cpick .cp-two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 @media (max-width:1100px){#tab-cpick .cp-two{grid-template-columns:1fr}}
 #tab-cpick .cp-lab{font-size:11px;font-weight:700;color:#374151;margin-bottom:4px}
 #tab-cpick .cp-cnt{font-size:10px;font-weight:600;color:#15803d;margin-left:4px}
 #tab-cpick .cp-two textarea{height:96px}
 #tab-cpick .cp-two .cp-btn{margin-top:6px}
 #tab-cpick .cp-exp{max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#6b7280;font-size:10px}
 #tab-cpick .cp-src{font-size:8px;font-weight:700;color:#6b7280;background:#f1f5f9;border-radius:3px;padding:0 4px;margin-left:4px}
</style>

<div class="cp-hd">
  <h1>Colony Picking &mdash; paste today's counts</h1>
  <p>Paste the colony export straight from Google Sheets (including the header row). Scored in your
     browser with the same LOW/MED/HIGH thresholds the Requests In&nbsp;Flight tab uses &mdash; nothing is
     uploaded anywhere, and it works before the counts reach the dashboard's data.</p>
</div>

<div class="cp-box">
  <div class="cp-two">
    <div>
      <div class="cp-lab">Sheet 1 <span id="cp-n-a" class="cp-cnt"></span></div>
      <textarea id="cp-paste-a" autocomplete="off" placeholder="paste the first strain sheet here, header row included"></textarea>
      <button class="cp-btn sec" id="cp-copy-a" onclick="cpCopyPicks('a')" disabled>Copy Manual Pick &rarr; Sheet 1</button>
    </div>
    <div>
      <div class="cp-lab">Sheet 2 <span style="color:#9ca3af;font-weight:400">(optional)</span> <span id="cp-n-b" class="cp-cnt"></span></div>
      <textarea id="cp-paste-b" autocomplete="off" placeholder="paste the second strain sheet here (e.g. EPI400)"></textarea>
      <button class="cp-btn sec" id="cp-copy-b" onclick="cpCopyPicks('b')" disabled>Copy Manual Pick &rarr; Sheet 2</button>
    </div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap">
    <button class="cp-btn" onclick="cpScore()">Score</button>
    <button class="cp-btn sec" id="cp-sheets" onclick="cpCopySheets()" title="Paste into a blank Google Sheet, select the range and copy again — pasting THAT into Slack gives a real table. Slack only builds tables from a Sheets clipboard, not from this page.">Copy high risk &rarr; Sheets</button>
    <button class="cp-btn sec" onclick="cpLoadFile('a')">Load file &rarr; 1</button>
    <button class="cp-btn sec" onclick="cpLoadFile('b')">Load file &rarr; 2</button>
    <input type="file" id="cp-file" accept=".tsv,.csv,.txt" style="display:none">
    <button class="cp-btn sec" onclick="cpClear()">Clear both</button>
    <span id="cp-thr" style="font-size:10px;color:#9ca3af"></span>
  </div>
  <div id="cp-err" class="cp-err"></div>
</div>

<div id="cp-kpis" class="cp-kpis"></div>
<div id="cp-out" style="margin:8px 16px 0;background:#fff;border:1px solid #e5e7eb;border-radius:8px;overflow:auto;max-height:calc(100vh - 430px)"></div>

<script>
(function(){
  var BAND_LOW=__BAND_LOW__, BAND_MED=__BAND_MED__;
  var RISK_HIGH=__RISK_HIGH__, RISK_MED=__RISK_MED__;
  var T_THIN=__T_THIN__, T_MED=__T_MED__, T_RICH=__T_RICH__;
  var AUTO_UNDER=__AUTO_UNDER__;
  var AUTO_TYPES=[__AUTO_TYPES__];
  var PRIORITY=[__PRIORITY__];   // which strain to take the extra picks from first
  var LSKEY='cpick_state_v1';
  var ROWS=[];          // parsed input rows
  var EDITS={};         // rowKey -> manual pick override

  // Spell the risk tier and its pick target out together. Listing them as two separate
  // scales ("risk: HIGH 3 … picks: HIGH 8") reads as if HIGH meant the same thing in both,
  // when a HIGH YIELD plate wants the FEWEST picks and a HIGH RISK one wants the most.
  document.getElementById('cp-thr').textContent =
    'yield/strain: LOW ≤'+BAND_LOW+' · MED ≤'+BAND_MED+' · HIGH >'+BAND_MED+
    '   •   picks by yield: LOW → '+T_THIN+' (or all that are left) · MED → '+T_MED+
    ' · HIGH → '+T_RICH+
    '   •   risk (Round 2 only): HIGH RISK ≤'+RISK_HIGH+' · WATCH ≤'+RISK_MED;

  function num(v){ v=(v==null?'':(''+v)).trim(); if(v==='') return 0; var n=parseFloat(v); return isNaN(n)?0:Math.round(n); }
  function band(n){ return n<=BAND_LOW?'LOW':(n<=BAND_MED?'MED':'HIGH'); }
  function risk(n){ return n<=RISK_HIGH?'HIGH':(n<=RISK_MED?'MED':''); }
  function wantFor(bandName){
    if(bandName==='HIGH') return T_RICH;
    if(bandName==='MED')  return T_MED;
    return T_THIN;
  }
  function normStrain(s){
    var u=(''+s).toUpperCase().replace(/[_\-\s]/g,'');
    if(u.indexOf('EPI')>=0) return 'EPI400';
    if(u.indexOf('NEBSTABLE')>=0||u.indexOf('NEBSTBL')>=0) return 'NEBstable';
    return (''+s).trim()||'?';
  }
  function esc(s){ return (''+(s==null?'':s)).replace(/[&<>"]/g,function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; }); }

  // ── parse: TSV (what Sheets copies) or CSV, header row required ──────────────
  var MISSING=[];
  function parse(text, src){
    var lines=text.replace(/\r/g,'').split('\n').filter(function(l){ return l.trim()!==''; });
    if(!lines.length) throw new Error('nothing pasted');
    var delim = (lines[0].split('\t').length >= lines[0].split(',').length) ? '\t' : ',';
    // Two strain sheets pasted one after another puts a SECOND header row mid-paste.
    // Re-derive the column map whenever a header appears, which also survives the sheets
    // having different column orders. Sheets' trailing tab-only rows drop out above.
    function isHdr(cells){
      var low=cells.map(function(c){ return c.trim().toLowerCase(); });
      return low.indexOf('plasmid')>=0 && (low.indexOf('process id')>=0 || low.indexOf('agar plate id')>=0);
    }
    var ix=null, out=[], sawHdr=false;
    for(var i=0;i<lines.length;i++){
      var c=lines[i].split(delim);
      if(isHdr(c)){
        var low=c.map(function(x){ return x.trim().toLowerCase(); });
        function col(){
          for(var a=0;a<arguments.length;a++){ var j=low.indexOf(arguments[a]); if(j>=0) return j; }
          return -1;
        }
        ix={ pid:col('process id'), plasmid:col('plasmid'), plate:col('agar plate id'),
             well:col('agar well position'), exp:col('experiment'), job:col('assembly job'),
             atype:col('assembly type'), strain:col('cloning strain'),
             imaged:col('qpix imaged'), qpk:col('qpix pickable'), qpd:col('qpix picked'),
             mpk:col('manual pickable'), mpd:col('manual picked') };
        if(ix.qpk<0 && ix.mpk<0) throw new Error('no pickable column found ("QPix Pickable" / "Manual Pickable")');
        // A missing Manual column reads as zero everywhere downstream, which silently under-counts
        // both pickable and picked and makes the tab disagree with LIMS. Say so instead.
        if(ix.mpk<0) MISSING.push('Manual Pickable');
        if(ix.mpd<0) MISSING.push('Manual Picked');
        sawHdr=true;
        continue;
      }
      if(!ix) continue;                       // data before any header — nothing to key on
      function g(j){ return j>=0 && j<c.length ? c[j] : ''; }
      var pl=(g(ix.plasmid)||'').trim();
      if(!pl) continue;
      out.push({ src:src, si:out.length, pid:(g(ix.pid)||'').trim(), plasmid:pl, plate:(g(ix.plate)||'').trim(),
                 well:(g(ix.well)||'').trim(), exp:(g(ix.exp)||'').trim(), job:(g(ix.job)||'').trim(),
                 atype:(g(ix.atype)||'').trim(), strain:normStrain(g(ix.strain)),
                 imaged:num(g(ix.imaged)),
                 // BAND on the sum (the pipeline does the same): a plate with 0 QPix pickable
                 // but 20 manual is healthy, not dead.
                 pickable:num(g(ix.qpk))+num(g(ix.mpk)),
                 // Keep the two picked figures APART. QPix's picks are fixed history; the manual
                 // ones are what this tab edits and copies back into the sheet's Manual Picked
                 // column, so a single combined total cannot serve both.
                 qpd:num(g(ix.qpd)), mpd:num(g(ix.mpd)) });
    }
    if(!sawHdr) throw new Error('no header row found — paste the header too');
    if(!out.length) throw new Error('header found but no data rows');
    return out;
  }

  function rowKey(r){ return r.pid || (r.plasmid+'|'+r.plate+'|'+r.well+'|'+r.strain); }
  function rowByKey(k){
    for(var i=0;i<ROWS.length;i++){ if(rowKey(ROWS[i])===k) return ROWS[i]; }
    return null;
  }
  // Ceiling on the Manual Picked box: everything QPix did not take is fair game, and there is
  // nothing beyond that to take. Without this the box accepts 8 on a plate that grew 3.
  function manualCap(r){ return Math.max(0, r.pickable-r.qpd); }
  // The box holds MANUAL picks only — what a human took, which is exactly what the Copy button
  // pastes into the sheet's Manual Picked column. Seeding it with QPix + manual instead credited
  // a human with QPix's fixed 2 and double-counted them on the next paste of that sheet.
  function manualOf(r){
    var k=rowKey(r);
    return (k in EDITS) ? Math.min(EDITS[k], manualCap(r)) : r.mpd;
  }
  function pickedOf(r){ return r.qpd+manualOf(r); }          // total in tubes
  function availOf(r){ return Math.max(0, r.pickable-pickedOf(r)); }
  // What the Manual picked box SHOWS: hand picks already recorded, plus the picks this tab is
  // telling you to take. Filling that in is the tab's job — the Take column on its own left every
  // box at 0, so the column copied back to the sheet said nothing was picked.
  //
  // Deliberately NOT fed back into scoring. Left and Action describe what has actually happened;
  // if a suggestion counted as done, headroom would drop, the recommendation would recompute to
  // zero, and the box would empty itself on the next render.
  function suggestedOf(r){ return Math.min(r.mpd+(r.take||0), manualCap(r)); }
  function plannedOf(r){
    var k=rowKey(r);
    return (k in EDITS) ? Math.min(EDITS[k], manualCap(r)) : suggestedOf(r);
  }
  function leftTip(r){
    var a=availOf(r), m=manualOf(r);
    if(a<=0) return 'every pickable colony on this plate has been taken';
    return a+' of the '+r.pickable+' pickable colonies are still on the agar — QPix took '+r.qpd
         + (m>0 ? ' and '+m+' were taken by hand' : '')+'. A human can take these.';
  }
  // Highlight only where you overrode the tab, not where you accepted its number.
  function isEdited(r){ var k=rowKey(r); return (k in EDITS) && plannedOf(r)!==suggestedOf(r); }

  // ── group into attempts (a plasmid, across its strains) ─────────────────────
  function isAutoType(r){
    var t=(r.atype||'').toLowerCase();
    return AUTO_TYPES.some(function(x){ return t.indexOf(x)>=0; });
  }
  var HAS_ATYPE=false;
  function group(){
    HAS_ATYPE=ROWS.some(function(r){ return !!(r.atype||'').trim(); });
    var by={}, order=[];
    ROWS.forEach(function(r){
      var k=r.plasmid+'||'+r.job;
      if(!by[k]){ by[k]={plasmid:r.plasmid, job:r.job, exp:r.exp, rows:[]}; order.push(k); }
      by[k].rows.push(r);
    });
    return order.map(function(k){
      var g=by[k];
      g.pickable=g.rows.reduce(function(a,r){ return a+r.pickable; },0);
      g.picked  =g.rows.reduce(function(a,r){ return a+pickedOf(r); },0);
      // DISTINCT strains, not row count — _nstrain() in the In-Flight tab counts strains, and an
      // attempt can carry several plates of the same one. Counting rows halved the per-strain
      // figure for a two-plate NEBstable attempt and dropped it a whole risk band below what
      // In-Flight showed for the identical data.
      var _st={}; g.rows.forEach(function(r){ _st[r.strain]=1; });
      g.strains=Object.keys(_st);
      g.nStrain =g.strains.length || 1;
      // ROUND, not floor — this has to match _perStrain() in the In-Flight tab, which rounds.
      // Flooring made the two views disagree on the same attempt: 7 pickable over 2 strains read
      // as 3 here (HIGH RISK) and 4 there (WATCH), so a plasmid could be flagged as critical on
      // one tab and merely watched on the other. The bands were calibrated on the rounded figure.
      g.perStrain=Math.round(g.pickable/g.nStrain);
      g.band=band(g.perStrain);
      g.risk=risk(g.perStrain);
      g.want=wantFor(g.band);                    // what the policy asks for
      g.target=Math.min(g.want, g.pickable);      // ...and what this plate can actually give
      g.headroom=g.rows.reduce(function(a,r){ return a+availOf(r); },0);   // colonies still on the agar
      g.shortOf=Math.max(0,g.target-g.picked);
      g.pickMore=Math.min(g.shortOf,Math.max(0,g.headroom));
      // Auto-retry keys off the ASSEMBLY TYPE and the picked count — never the strain. Gibson
      // and Golden Gate are re-run when under the threshold; a Transformation is not re-run at
      // all. So a high-risk attempt needs a human when it is a Transformation, or when it is a
      // Gibson/Golden Gate that already reached the threshold.
      var covered=g.rows.filter(isAutoType);
      g.typeOk=covered.length>0;                       // Gibson / Golden Gate, any strain
      g.autoPicked=covered.reduce(function(a,r){ return a+pickedOf(r); },0);
      g.autoRetry=g.typeOk && g.autoPicked<AUTO_UNDER;
      g.typeKnown=HAS_ATYPE;                           // no Assembly Type column -> cannot judge
      g.needsManual=g.typeKnown && (g.risk==='HIGH') && !g.autoRetry;
      // Spread the shortfall over the strains, favouring NEBstable: it only says "pick 2 more"
      // if it also says WHERE, and taking them from the preferred strain is worth more.
      var need=g.pickMore;
      var order=g.rows.slice().sort(function(a,b){
        var ia=PRIORITY.indexOf(a.strain), ib=PRIORITY.indexOf(b.strain);
        if(ia<0) ia=99; if(ib<0) ib=99;
        if(ia!==ib) return ia-ib;
        return availOf(b)-availOf(a);            // then whichever plate has more left
      });
      g.rows.forEach(function(r){ r.take=0; });
      order.forEach(function(r){
        if(need<=0) return;
        var room=availOf(r);
        var t=Math.min(room,need);
        r.take=t; need-=t;
      });
      return g;
    });
  }

  function pill(txt,bg,fg){ return '<span class="cp-pill" style="background:'+bg+';color:'+fg+'">'+txt+'</span>'; }
  function bandPill(b){
    if(b==='LOW')  return pill('LOW','#FDE2E2','#B42318');
    if(b==='MED')  return pill('MED','#FEF3C7','#92400E');
    return pill('HIGH','#DCFCE7','#15803D');
  }
  function riskPill(rk){
    if(rk==='HIGH') return pill('HIGH RISK','#FEE2E2','#991B1B');
    if(rk==='MED')  return pill('WATCH','#FEF3C7','#92400E');
    return '<span style="color:#9ca3af;font-size:9px">ok</span>';
  }

  var GROUPS=[];
  function render(){
    var groups=group();
    GROUPS=groups;
    // worst first: unmet pick target, then thinnest per strain
    groups.sort(function(a,b){
      var d=(b.shortOf>0)-(a.shortOf>0); if(d) return d;
      if(a.perStrain!==b.perStrain) return a.perStrain-b.perStrain;
      return a.plasmid<b.plasmid?-1:1;
    });
    var nMore=0,nMissed=0,nEdit=0;
    var h='<table><thead><tr>'
        + '<th>Plasmid</th><th>Experiment</th><th>Plate &middot; well</th>'
        + '<th title="Process ID of this transformation, as it appears in the export">Process&nbsp;ID</th>'
        + '<th>Strain</th>'
        + '<th style="text-align:right">Imaged</th><th style="text-align:right">Pickable</th>'
        + '<th style="text-align:right" title="What QPix picked. Read-only — nothing is ever written back here.">Picked</th>'
        + '<th style="text-align:right" title="Pickable minus Picked: colonies still on the agar that a human can take. QPix only takes its fixed 2, so the rest are still there.">Left</th>'
        + '<th title="Hand picks. Filled in for you: what the sheet already records, plus the Take this tab recommends. Edit any box to overrule it, then Copy back to the sheet.">Manual picked</th>'
        + '<th title="How many MORE to take by hand, on top of what Picked already shows">Take</th>'
        + '<th style="text-align:right">/strain</th>'
        + '<th title="Colonies that GREW, per strain, and what sets the pick target. HIGH is good: '
        +      'LOW &le;'+BAND_LOW+' &rarr; pick '+T_THIN+' (or all that are left) &middot; MED &le;'
        +      BAND_MED+' &rarr; pick '+T_MED+' &middot; HIGH &gt;'+BAND_MED+' &rarr; pick '+T_RICH
        +      '">Yield</th>'
        + '<th title="Chance this attempt fails to yield a clone. HIGH RISK is bad: &le;'+RISK_HIGH
        +      ' &middot; WATCH &le;'+RISK_MED+'. Drives Round 2 only — the pick target comes from Yield.'
        +      '">Risk</th>'
        + '<th style="text-align:right" title="Picks wanted for this attempt, set by Yield. On a thin plate it caps at the number of colonies that actually grew.">Target</th>'
        + '<th>Action</th>'
        + '<th title="Shown for high-risk attempts only. The pipeline re-runs one automatically '
        +      'only if it is a Gibson or Golden Gate AND fewer than '+AUTO_UNDER+' were picked. '
        +      'A Transformation, or a Gibson/Golden Gate that already reached '+AUTO_UNDER
        +      ', has to be put back in by hand.">Retry</th>'
        + '</tr></thead><tbody>';
    groups.forEach(function(g){
      if(g.shortOf>0 && g.headroom>0) nMore++;
      if(g.headroom<=0 && g.picked<g.want) nMissed++;
      g.rows.forEach(function(r,i){
        var first=(i===0), ed=isEdited(r);
        if(ed) nEdit++;
        h+='<tr class="'+(first?'cp-grp ':'')+(ed?'cp-edited':'')+'">'
         + '<td class="cp-mono" style="font-weight:700">'+(first?esc(g.plasmid):'')+'</td>'
         + '<td class="cp-exp" title="'+esc(g.exp||'')+'">'+(first?esc(g.exp||''):'')+'</td>'
         + '<td class="cp-mono" style="color:#6b7280">'+esc(r.plate)+' &middot; '+esc(r.well)
         +   '<span class="cp-src">'+(r.src==='b'?'S2':'S1')+'</span></td>'
         + '<td class="cp-mono" style="color:#9ca3af;font-size:10px" title="'+esc(r.pid||'')+'">'
         +   esc(r.pid ? r.pid.slice(0,8)+'\u2026' : '')+'</td>'
         + '<td>'+esc(r.strain)+'</td>'
         + '<td style="text-align:right;color:#6b7280">'+r.imaged+'</td>'
         + '<td style="text-align:right;font-weight:700">'+r.pickable+'</td>'
         // QPix picks only, mirroring the sheet's own QPix Picked column. Hand picks live in
         // Manual picked and are never folded in here.
         + '<td style="text-align:right;color:#6b7280" title="QPix picks only — hand picks are in Manual picked">'
         +   r.qpd+'</td>'
         + '<td style="text-align:right;'+(availOf(r)>0?'font-weight:700':'color:#d1d5db')+'" '
         +      'title="'+esc(leftTip(r))+'">'+availOf(r)+'</td>'
         + '<td><input class="cp-inp" type="number" min="0" max="'+manualCap(r)+'" '
         +      'data-k="'+esc(rowKey(r))+'" value="'+plannedOf(r)+'" '
         +      'title="Hand picks: '+r.mpd+' already recorded in the sheet'
         +      (r.take>0? ' + '+r.take+' this tab is telling you to take':'')
         +      '. Filled in for you — change it if you take a different number. Max '+manualCap(r)
         +      ', which is all this plate grew." oninput="cpEdit(this)"></td>'
         + '<td>'+(r.take>0
              ? '<span class="cp-pill" style="background:#FEF3C7;color:#92400E">+'+r.take+'</span>'
                + (PRIORITY.indexOf(r.strain)===0 && g.rows.length>1
                     ? '<span style="color:#9ca3af;font-size:9px"> NEB first</span>' : '')
              : '<span style="color:#d1d5db">&mdash;</span>')+'</td>';
        if(first){
          var rs=g.rows.length;
          h+='<td rowspan="'+rs+'" style="text-align:right;font-weight:700">'+g.perStrain+'</td>'
           + '<td rowspan="'+rs+'">'+bandPill(g.band)+'</td>'
           + '<td rowspan="'+rs+'">'+riskPill(g.risk)+'</td>'
           + '<td rowspan="'+rs+'" style="text-align:right">'+g.target+'</td>'
           + '<td rowspan="'+rs+'">'+actionCell(g)+'</td>'
           + '<td rowspan="'+rs+'">'+retryCell(g)+'</td>';
        }
        h+='</tr>';
      });
    });
    h+='</tbody></table>';
    document.getElementById('cp-out').innerHTML=h;

    document.getElementById('cp-kpis').innerHTML =
        k(groups.length,'attempts','#111827')
      + k(nMore,'pick more','#b45309')
      + k(nMissed,'missed target','#b91c1c')
      + k(nEdit,'edited','#f59e0b');

    save();
  }
  function k(n,l,c){ return '<div class="cp-k"><div class="cp-kn" style="color:'+c+'">'+n+'</div><div class="cp-kl">'+l+'</div></div>'; }

  function whyManual(g){
    if(!g.typeOk)
      return 'Assembly Type is not Gibson or Golden Gate, and only those are re-run automatically.';
    return 'Gibson/Golden Gate, but '+g.autoPicked+' already picked — the auto-retry only fires under '
         + AUTO_UNDER+'.';
  }
  function retryCell(g){
    if(g.needsManual)
      return '<span class="cp-pill" style="background:#FEE2E2;color:#991B1B" title="'+esc(whyManual(g))
           + '">PUT BACK IN</span>';
    // Only high-risk attempts raise the question at all; the rest are not waiting on a re-run.
    if(g.risk!=='HIGH') return '<span style="color:#d1d5db">&mdash;</span>';
    if(!g.typeKnown)
      return '<span style="color:#d1d5db" title="No Assembly Type column in the paste, so whether '
           + 'this auto-retries cannot be judged">?</span>';
    return '<span style="color:#9ca3af" title="Gibson/Golden Gate with '+g.autoPicked+' picked, under '
         + AUTO_UNDER+' — the pipeline re-runs this one on its own">auto</span>';
  }
  function actionCell(g){
    var red='color:#b91c1c;font-weight:600', grey='color:#9ca3af';
    // Nothing grew. Capping the target at what the plate produced made this compare 0 picks
    // against a target of 0 and call it a success — the one row that most needs an alarm.
    if(g.pickable<=0)
      return '<span style="'+red+'">nothing grew &mdash; cannot pick</span>';
    // Everything the plate had is in a tube and it still did not reach the wanted count. That is
    // a miss, not a finish: "at target" here only meant "at the capped target".
    if(g.headroom<=0 && g.picked<g.want)
      return '<span style="'+red+'">missed target</span>'
           + '<span style="'+grey+'"> &mdash; only '+g.pickable+' grew, wanted '+g.want+'</span>';
    if(g.shortOf<=0){
      if(g.picked<g.want)
        return '<span style="'+red+'">missed target</span>'
             + '<span style="'+grey+'"> &mdash; only '+g.pickable+' grew, wanted '+g.want+'</span>';
      return '<span style="color:#15803d">at target</span>';
    }
    var out='<span style="color:#b45309;font-weight:600">pick '+g.pickMore+' more by hand</span>';
    if(g.pickMore<g.shortOf)
      out+='<span style="'+grey+'"> &mdash; all '+g.headroom+' that are left (want '+g.shortOf+' more)</span>';
    return out;
  }

  // ── edit / persist ─────────────────────────────────────────────────────────
  window.cpEdit=function(el){
    var k=el.getAttribute('data-k'), v=parseInt(el.value,10);
    if(isNaN(v)||v<0) v=0;
    var r=rowByKey(k);
    if(r) v=Math.min(v, manualCap(r));      // cannot take colonies that are not on the plate
    EDITS[k]=v;
    render();
    // keep focus where the user was typing
    var again=document.querySelector('#tab-cpick .cp-inp[data-k="'+k.replace(/"/g,'\\"')+'"]');
    if(again){ again.focus(); again.select(); }
  };
  function save(){
    try{ localStorage.setItem(LSKEY, JSON.stringify({
      a:document.getElementById('cp-paste-a').value,
      b:document.getElementById('cp-paste-b').value, edits:EDITS })); }catch(e){}
  }
  // A deliberate refresh should come up empty; the dashboard's own auto-reload should not.
  // Those are different navigations: the version poller in the document head goes to
  // ?v=<timestamp>, which reports as 'navigate', while Cmd-R reports as 'reload'. Telling them
  // apart is what lets a refresh clear the tab without handing back the data-loss the saved
  // state exists to prevent — a new render landing mid-session would otherwise wipe a
  // half-finished set of picks.
  function isManualRefresh(){
    try{
      var nav=performance.getEntriesByType && performance.getEntriesByType('navigation')[0];
      if(nav) return nav.type==='reload';
      return !!(performance.navigation && performance.navigation.type===1);
    }catch(e){ return false; }
  }
  function restore(){
    try{
      if(isManualRefresh()){
        localStorage.removeItem(LSKEY);
        // Dropping the saved state is not enough. The browser restores typed-in form values on a
        // reload by itself, and it does that AFTER this inline script has run, so the textareas
        // refill from under us and the tab comes back looking exactly as it did. Clear once now
        // and again after that restoration pass.
        cpClear();
        setTimeout(cpClear, 0);
        window.addEventListener('pageshow', function once(){
          cpClear(); window.removeEventListener('pageshow', once);
        });
        return;
      }
      var s=JSON.parse(localStorage.getItem(LSKEY)||'null');
      if(s && (s.a||s.b)){
        document.getElementById('cp-paste-a').value=s.a||'';
        document.getElementById('cp-paste-b').value=s.b||'';
        EDITS=s.edits||{}; cpScore(true);
      }
    }catch(e){}
  }

  window.cpScore=function(quiet){
    var err=document.getElementById('cp-err'); err.textContent='';
    var ta=document.getElementById('cp-paste-a').value.trim();
    var tb=document.getElementById('cp-paste-b').value.trim();
    var A=[],B=[],msg=[]; MISSING=[];
    // Parse the sheets SEPARATELY so each keeps its own row order — the Copy button has to
    // emit a column that lines up with the sheet it came from. Scoring merges them after.
    if(ta){ try{ A=parse(ta,'a'); }catch(e){ msg.push('Sheet 1: '+e.message); } }
    if(tb){ try{ B=parse(tb,'b'); }catch(e){ msg.push('Sheet 2: '+e.message); } }
    ROWS=A.concat(B);
    document.getElementById('cp-n-a').textContent = A.length? A.length+' rows' : '';
    document.getElementById('cp-n-b').textContent = B.length? B.length+' rows' : '';
    document.getElementById('cp-copy-a').disabled = !A.length;
    document.getElementById('cp-copy-b').disabled = !B.length;
    if(!ROWS.length){
      document.getElementById('cp-out').innerHTML='';
      document.getElementById('cp-kpis').innerHTML='';
      if(!quiet && msg.length) err.textContent=msg.join('   ·   ');
      else if(!quiet && !ta && !tb) err.textContent='nothing pasted';
      return;
    }
    if(MISSING.length)
      msg.push('column not found: '+MISSING.filter(function(v,i,a){ return a.indexOf(v)===i; }).join(', ')
               +' — those picks are NOT being counted, so totals will read low against the dashboard');
    if(msg.length) err.textContent=msg.join('   ·   ');
    render();
  };
  // Plain-text digest of the high-risk attempts, sized to paste straight into the dnasc channel.
  // Slack renders pasted text verbatim, so this stays deliberately free of markup and tabs — a
  // tab-separated block collapses into an unreadable single line there.
  function highRiskTsv(){
    // High risk OR falling through the auto-retry — the second group is the one nobody is
    // watching, so it belongs in the same message even when its yield was not flagged.
    var hi=GROUPS.filter(function(g){ return g.risk==='HIGH' || g.needsManual; });
    if(!hi.length) return null;
    // Grouped by experiment so one message can be actioned project by project, and worst first
    // within each: thinnest plate at the top, and where two are equally thin the one furthest
    // short of its target leads.
    hi=hi.slice().sort(function(a,b){
      var ea=(a.exp||'').toLowerCase(), eb=(b.exp||'').toLowerCase();
      if(ea!==eb) return ea<eb ? -1 : 1;
      if(a.perStrain!==b.perStrain) return a.perStrain-b.perStrain;
      var ga=a.want-a.picked, gb=b.want-b.picked;
      if(ga!==gb) return gb-ga;
      return a.plasmid<b.plasmid ? -1 : 1;
    });


    // The sheet gets filled in FROM this output, so the export reports the planned picks as
    // picked — the pre-filled Manual picked boxes are what will be recorded. On screen those
    // suggestions are deliberately excluded (counting them would zero the recommendation out
    // and blank the boxes), but by the time this message is sent they are the outcome.
    function plannedTotal(g){
      return g.rows.reduce(function(a,r){ return a+r.qpd+plannedOf(r); }, 0);
    }
    // ...and the retry verdict has to be read off those same numbers. Doing the recommended
    // picks is what pushes a Gibson/Golden Gate over the threshold and switches the auto-retry
    // OFF, so a row reporting the planned picks while reporting the CURRENT retry state would
    // say "5 picked" and "auto" side by side when 5 picked is precisely why it is no longer auto.
    function plannedPutBack(g){
      if(!g.typeKnown || g.risk!=='HIGH') return '-';
      var autoPicked=g.rows.filter(isAutoType).reduce(function(a,r){
        return a+r.qpd+plannedOf(r); }, 0);
      var willAuto = g.typeOk && autoPicked<AUTO_UNDER;
      return willAuto ? 'auto' : 'BY HAND';
    }
    var COLS=[
      {h:'Plasmid',    get:function(g){ return g.plasmid; }},
      {h:'Experiment', get:function(g){ return trunc(g.exp||'', 40); }},
      {h:'Total pickable', get:function(g){ return ''+g.pickable; }},
      {h:'Total picked',   get:function(g){ return ''+plannedTotal(g); }},
      {h:'Put back in',    get:function(g){ return plannedPutBack(g); }}
    ];
    // Tab-separated, header in row 1, no summary line — this is pasted into a blank sheet, and
    // a heading above the header row would push every column out of line. From there: select,
    // copy, paste into Slack, which builds the table. Slack will not build one from anything
    // this page puts on the clipboard directly; that was tried exhaustively, including a payload
    // verified byte-identical to a real Sheets copy.
    var tsv=[COLS.map(function(c){ return c.h; }).join('\t')];
    hi.forEach(function(g){
      tsv.push(COLS.map(function(c){ return c.get(g); }).join('\t'));
    });
    return tsv.join('\n');





  }
  // '...' not '…' — the legacy clipboard text flavour cannot encode U+2026 and truncates the
  // whole payload at the first one, which cut 808 bytes of picks down to 170.
  function trunc(v,n){ return v.length>n ? v.slice(0,n-3)+'...' : v; }
  function shortStrain(x){
    if(x==='NEBstable') return 'NEB';
    if(x==='EPI400')    return 'EPI';
    return x;
  }
  window.cpCopySheets=function(){
    var b=document.getElementById('cp-sheets'), lab=b.getAttribute('data-lab');
    if(!lab){ lab=b.textContent; b.setAttribute('data-lab', lab); }
    var tsv=highRiskTsv();
    if(!tsv){ b.textContent='No high-risk attempts'; setTimeout(function(){ b.textContent=lab; },2200); return; }
    cpCopy(tsv, null, function(ok){
      b.textContent = ok ? 'Copied — paste into a blank sheet' : 'Copy failed';
      setTimeout(function(){ b.textContent=lab; },2600);
    });
  };
  window.cpClear=function(){
    ROWS=[]; EDITS={};
    document.getElementById('cp-paste-a').value='';
    document.getElementById('cp-paste-b').value='';
    document.getElementById('cp-n-a').textContent='';
    document.getElementById('cp-n-b').textContent='';
    document.getElementById('cp-copy-a').disabled=true;
    document.getElementById('cp-copy-b').disabled=true;
    document.getElementById('cp-out').innerHTML='';
    document.getElementById('cp-kpis').innerHTML='';
    document.getElementById('cp-err').textContent='';
    try{ localStorage.removeItem(LSKEY); }catch(e){}
  };
  window.cpCopyPicks=function(src){
    // One value per row OF THAT SHEET, in the order it was pasted, so it pastes back aligned.
    // A merged column across both sheets would be off by the other sheet's row count.
    var rows=ROWS.filter(function(r){ return r.src===src; }).sort(function(a,b){ return a.si-b.si; });
    // MANUAL picks only — this column pastes into the sheet's Manual Picked, which sits next to
    // QPix Picked. Emitting the combined total re-recorded QPix's fixed 2 as hand picks, and the
    // next paste of that sheet then read more picked than the plate ever grew.
    // Blank, not 0, where there is nothing to take. The lab pastes this straight into the sheet
    // and a column of zeros reads as "assessed and none available"; empty leaves those cells as
    // they were and puts a number only where a pick is actually wanted. Parsing treats blank as
    // 0 anyway, so a sheet filled this way round-trips unchanged.
    var vals=rows.map(function(r){ return plannedOf(r); });
    var out=vals.join('\n');
    var nPick=vals.filter(function(v){ return v>0; }).length;
    // Also offer the column as an HTML table. Sheets honours inline style on paste, so the rows
    // that actually want a pick land BOLD on YELLOW and the rest paste as a plain 0. Every row
    // carries its number: as blanks, a run of trailing empty lines was trimmed on paste and the
    // last rows of an all-at-target sheet silently covered nothing.
    var html='<table>'+vals.map(function(v){
      return v>0
        ? '<tr><td style="background-color:#FFEB3B;font-weight:bold">'+v+'</td></tr>'
        : '<tr><td>'+v+'</td></tr>';
    }).join('')+'</table>';
    // Say how many actual picks went across, not just the row count. When every attempt is at
    // target the column is legitimately all blanks, and a bare "Copied 12 values" then looks
    // exactly like a broken button.
    cpCopy(out, html, function(ok){
      var b=document.getElementById('cp-copy-'+src), t=b.getAttribute('data-lab');
      if(!t){ t=b.textContent; b.setAttribute('data-lab', t); }
      b.textContent = !ok ? 'Copy failed — see the note below'
                    : (nPick ? 'Copied '+nPick+' pick'+(nPick>1?'s':'')+' over '+rows.length+' rows'
                             : 'Nothing to pick — all at target');
      setTimeout(function(){ b.textContent=t; },2200);
      document.getElementById('cp-err').textContent = ok ? ''
        : 'Could not reach the clipboard. This needs the https dashboard URL — a page opened '
          + 'straight off disk (file://) blocks copying.';
    });
  };
  // navigator.clipboard exists only in a secure context, so a dashboard opened as a local file
  // has no clipboard at all and the old call rejected into a promise nobody was catching — the
  // button simply did nothing, with no error anywhere. Fall back to a hidden textarea, and
  // report failure instead of swallowing it.
  function cpCopy(text, html, done){
    // Copy a contenteditable holding the HTML rather than a textarea holding the text: selecting
    // rendered markup puts BOTH flavours on the clipboard, so Sheets takes the bold table and a
    // plain text field still receives the newline-separated column.
    function fallback(){
      try{
        if(!html){                            // plain-text target: a textarea copies verbatim
          var ta=document.createElement('textarea');
          ta.value=text; ta.setAttribute('readonly','');
          ta.style.position='fixed'; ta.style.top='-1000px'; ta.style.opacity='0';
          document.body.appendChild(ta);
          ta.select(); ta.setSelectionRange(0, text.length);
          var tok=document.execCommand('copy');
          document.body.removeChild(ta);
          return done(tok);
        }
        var d=document.createElement('div');
        d.innerHTML=html;
        d.setAttribute('contenteditable','true');
        d.style.position='fixed'; d.style.top='-1000px'; d.style.opacity='0';
        document.body.appendChild(d);
        var rng=document.createRange(); rng.selectNodeContents(d);
        var sel=window.getSelection(); sel.removeAllRanges(); sel.addRange(rng);
        var ok=document.execCommand('copy');
        sel.removeAllRanges(); document.body.removeChild(d);
        done(ok);
      }catch(e){ done(false); }
    }
    if(!html && navigator.clipboard && navigator.clipboard.writeText){
      navigator.clipboard.writeText(text).then(function(){ done(true); }, fallback);
      return;
    }
    if(navigator.clipboard && navigator.clipboard.write && typeof ClipboardItem!=='undefined'){
      try{
        navigator.clipboard.write([new ClipboardItem({
          'text/html' : new Blob([html], {type:'text/html'}),
          'text/plain': new Blob([text], {type:'text/plain'})
        })]).then(function(){ done(true); }, fallback);
        return;
      }catch(e){ /* older ClipboardItem shapes throw — fall through */ }
    }
    fallback();
  }
  window.cpLoadFile=function(src){
    var f=document.getElementById('cp-file');
    f.value='';
    f.onchange=function(){
      var file=f.files && f.files[0]; if(!file) return;
      var rd=new FileReader();
      rd.onload=function(){ document.getElementById('cp-paste-'+src).value=rd.result; cpScore(); };
      rd.readAsText(file);
    };
    f.click();
  };

  restore();
})();
</script>
"""

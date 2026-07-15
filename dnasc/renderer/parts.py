"""Parts inventory dashboard tab — self-contained, SCOPED fragment (#tab-parts).

Reads parts_result.pkl (built by the standalone gen_parts_pkl.py pull) and renders the
Parts view as a fragment for the main dashboard, modeled on inflight.render_inflight_tab.
All CSS is namespaced under #tab-parts so nothing leaks into the other tabs; the JS toggle
is partsToggle() to avoid colliding with the host. Never raises to the host: on any failure
(missing/unreadable pkl, BigQuery error) it returns a small "data unavailable" notice.
"""
from __future__ import annotations  # 3.9 server compat (lazy annotations)
import os
import sys
import html
import pickle
import datetime
from zoneinfo import ZoneInfo

import pandas as pd

_ET = ZoneInfo("America/New_York")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))   # repo/scripts root
_PKL = os.path.join(_ROOT, "dashboard_state", "parts_result.pkl")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_FALLBACK = ('<div style="padding:24px;color:#6b7280;font:14px -apple-system,sans-serif">'
             'Parts inventory data unavailable — the parts pull (gen_parts_pkl.py) has not run '
             'yet or failed. Check the parts cron / logs/parts_pull.log.</div>')


def render_parts_tab() -> str:
    """Return the Parts tab fragment (<style scoped> + content + <script>). Never raises."""
    try:
        return _render()
    except Exception:
        import traceback
        traceback.print_exc()
        return _FALLBACK


def _render() -> str:
    import parts_inventory as P
    from dnasc.utils import parse_parts, parse_backbone, extract_pcr_info

    r = pickle.load(open(_PKL, "rb"))
    apd, dpd, now = r["all_plate_data"], r["dpart_data"], r["generated_at"]
    render_ts = datetime.datetime.now(_ET)   # when THIS html was built (Eastern, so freshness is visible)
    # data-pull time → Eastern. generated_at may be a plain datetime (tz-aware UTC) OR a pandas
    # Timestamp; .astimezone() works on both. tz-naive values are assumed UTC first.
    try:
        _n = now if getattr(now, "tzinfo", None) else pd.Timestamp(now).tz_localize("UTC")
        now_et = _n.astimezone(_ET)
    except Exception:
        now_et = now
    ctrl = set(P.CONTROL_PARTS)
    # LSP-dedicated Echo plates (rearrays in the 4B-LSP rack) are allocated to specific LSP
    # orders — NOT general stock. Exclude them from inventory counts AND the on-hand display.
    _lsp = r.get("lsp_plates")
    LSP_PLATE_IDS = set(_lsp["PLATE_ID"].astype(str)) if _lsp is not None and "PLATE_ID" in getattr(_lsp, "columns", []) else set()
    dmeta = dpd.drop_duplicates("DPART_NAME").set_index("DPART_NAME")
    apd = apd.copy(); apd["CREATED_AT"] = pd.to_datetime(apd["CREATED_AT"], errors="coerce", utc=True)

    # Use the parts list straight from the data pull (result["parts"]) instead of re-querying
    # workorders and re-deriving here. The rebuild used to run its own live workorder query, which
    # drifts from the pull as the lab generates/completes workorders between the two — so parts the
    # pull found (e.g. BLOCKED gibson plasmids, in-demand dParts) would silently vanish on rebuild.
    out = r["parts"].copy()
    # Oligos are still TBD — exclude o- parts from the actionable views for now.
    out = out[~out["Part"].astype(str).str.startswith("o")].reset_index(drop=True)

    # All workorder inputs come from the PULL (parts_result.pkl) — the render does ZERO BigQuery.
    import json as _json
    wod = r["wod_df"]                                  # active GG/Gibson/PCR workorders
    _blk = r["blk_df"]                                 # blocked workorder queue
    _succ_names = set(r.get("blk_succ_names") or set())  # products with a SUCCEEDED maker

    def _inp_names(pj, bj):
        ns=[]
        for j,single in ((pj,False),(bj,True)):
            if j is None or (isinstance(j,float)): continue
            try: d=_json.loads(j)
            except Exception: continue
            for it in ([d] if single else (d or [])):
                if isinstance(it,dict) and it.get("name"): ns.append(it["name"])
        return ns

    # (blocked_html is RENDERED later — after builds_for/cons exist — so it can show "what it's blocking")

    TY = {"golden_gate_workorder":"GG","gibson_workorder":"Gibson","pcr_workorder":"PCR"}
    def _n(s): return [i.split(":")[0] for i in s.split(", ") if i.split(":")[0]]
    cons = {}
    for _, w in wod.iterrows():
        wt = w["WT"]; names = []
        if wt in ("golden_gate_workorder","gibson_workorder"):
            names = _n(parse_parts(w.get("parts_json")))
            bb = parse_backbone(w.get("backbone_json"))
            if bb: names.append(bb.split(":")[0])
        elif wt == "pcr_workorder":
            names = _n(extract_pcr_info(w))
        for n in names:
            cons.setdefault(n, []).append((str(w["PRODUCT"]), TY[wt], str(w["ST"]), str(w.get("EXP") or "—")))

    tmpl_kids = {}
    for d, row in dmeta.iterrows():
        t = row.get("DPART_TEMPLATE")
        if pd.notna(t): tmpl_kids.setdefault(str(t), []).append(d)
    flagged = set(out["Part"].astype(str))

    # --- "already started?" refill signal: wells tagged REFILL_* in PROCESS_ID ---
    _refill = apd[apd["PROCESS_ID"].astype(str).str.contains("REFILL", case=False, na=False)].copy()
    _refill["age"]=(now-_refill["CREATED_AT"]).dt.days
    def refill_status(part):
        w=_refill[_refill["STOCK_ID"]==str(part)].sort_values("CREATED_AT", ascending=False)
        if w.empty: return ("none", None, None, None)
        x=w.iloc[0]; age=int(x["age"]) if pd.notna(x["age"]) else None
        proto=str(x["PLATE_PROTOCOL"])
        return ("ngs" if "NGS" in proto else "pre_ngs", age, str(x["PROCESS_ID"]), proto)

    # --- "already ordered?" vendor-order signal (Reorder) — from the pull ---
    _ord = r["ord_df"].copy()
    _ord["CREATED"]=pd.to_datetime(_ord["CREATED"],errors="coerce",utc=True)
    def order_status(part):
        o=_ord[_ord["NAME"]==str(part)].sort_values("CREATED")
        if o.empty: return None
        active=o[o["STATUS"].isin(["RUNNING","WAITING","READY","BLOCKED"])]
        pick=active.iloc[-1] if not active.empty else o.iloc[-1]
        return {"active":not active.empty,"vendor":pick["VENDOR"],"status":pick["STATUS"],
                "date":pick["CREATED"],"order_id":pick["ORDER_ID"]}

    def builds_for(part):
        return sorted(set(cons.get(part, [])))

    # ---- Blocked workorder queue render (needs builds_for/cons for "what it's blocking") ----
    _TYB={"gibson_workorder":"Gibson","golden_gate_workorder":"GG","pcr_workorder":"PCR",
          "plasmid_synthesis_workorder":"PlasmidSynth","syn_part_synthesis_workorder":"SynPartSynth"}
    _SC={"RUNNING":"#1d4ed8","WAITING":"#b45309","READY":"#15803d","BLOCKED":"#b91c1c"}
    _BHDR='<tr><th></th><th>WO</th><th>Type</th><th>Product</th><th>Blocked inputs</th><th>Created</th><th>Note / action</th></tr>'
    _BCOLS='<colgroup><col style="width:26px"><col style="width:9%"><col style="width:7%"><col style="width:14%"><col style="width:16%"><col style="width:9%"><col></colgroup>'
    _bexp={}; _bi=300000   # experiment -> list of row-html
    for _,b in _blk.iterrows():
        prod=str(b["product"] or "?")
        inputs=_inp_names(b["parts"], b["backbone"])
        blocked_in=[i for i in inputs if i not in _succ_names]
        succ_wos=[w for w in (list(b["succeeded_wos"]) if b["succeeded_wos"] is not None else []) if w]
        if succ_wos:
            note=f'<span style="color:#15803d;font-weight:600">✓ Another design already SUCCEEDED (wo {str(succ_wos[0])[:8]}) → safe to cancel this WO</span>'
        else:
            warns="; ".join(b["warnings"]) if (b["warnings"] is not None and len(b["warnings"])) else ""
            note=f'<span style="color:#b45309">{html.escape(warns) or "blocked — investigate"}</span>'
        feeds=builds_for(prod)
        feeds_chips="".join(f'<span class="chip" style="border-color:{_SC.get(s,"#9ca3af")};color:{_SC.get(s,"#6b7280")}">{html.escape(p)} <em>{t}·{s.lower()}</em></span>' for p,t,s,_e in feeds[:40])
        feeds_html=(f'<div class="d-sub">Blocking {len(feeds)} downstream build{"s" if len(feeds)!=1 else ""}</div><div class="d-chips">{feeds_chips}{" +"+str(len(feeds)-40) if len(feeds)>40 else ""}</div>') if feeds else '<div style="font-size:11px;color:#9ca3af">nothing downstream depends on it right now</div>'
        det=(f'<table class="detail">'
             f'<tr><th class="d-lab">Working on</th><td class="d-cell"><b>{html.escape(prod)}</b> ({_TYB.get(b["type"],b["type"])}) · WO {str(b["wid"])[:8]}</td></tr>'
             f'<tr><th class="d-lab">Blocked inputs</th><td class="d-cell" style="color:#b91c1c;font-family:monospace">{html.escape(", ".join(blocked_in) or "—")} <span style="color:#9ca3af">(no successful maker)</span></td></tr>'
             f'<tr><th class="d-lab">All inputs</th><td class="d-cell" style="font-family:monospace;color:#6b7280">{html.escape(", ".join(inputs))}</td></tr>'
             f'<tr><th class="d-lab">What it&#39;s blocking</th><td class="d-cell">{feeds_html}</td></tr>'
             f'<tr><th class="d-lab">Action</th><td class="d-cell">{note}</td></tr></table>')
        row=(f'<tr class="prow" onclick="partsToggle({_bi})" style="cursor:pointer">'
             f'<td style="width:18px;color:#9ca3af" id="c{_bi}">▸</td>'
             f'<td style="font-family:monospace">{str(b["wid"])[:8]}</td>'
             f'<td>{_TYB.get(b["type"],b["type"])}</td>'
             f'<td style="font-family:monospace;font-weight:700">{html.escape(prod)}</td>'
             f'<td style="color:#b91c1c;font-family:monospace">{html.escape(", ".join(blocked_in) or "—")}</td>'
             f'<td style="white-space:nowrap">{b["created"]}</td><td style="font-size:11px">{note}</td></tr>'
             f'<tr id="d{_bi}" style="display:none"><td></td><td colspan="6">{det}</td></tr>')
        _bexp.setdefault(str(b["experiment"] or "— no experiment —"),[]).append(row)
        _bi+=1
    blocked_html=""
    if len(_blk):
        _bgroups=""
        for e in sorted(_bexp, key=lambda e:-len(_bexp[e])):
            rows=_bexp[e]
            _bgroups+=(f'<details class="expgrp"><summary>'
                       f'<span class="egname">{html.escape(e)}</span>'
                       f'<span class="egcount">{len(rows)}</span></summary>'
                       f'<table class="ptbl">{_BCOLS}<tbody>{"".join(rows)}</tbody></table></details>')
        blocked_html=_bgroups

    def typ(pt): return "Plasmid" if pt.startswith("pAI") else ("Oligo" if pt.startswith("o") else "dPart")
    ST_COLOR = {"RUNNING":"#1d4ed8","WAITING":"#92400e","READY":"#15803d","BLOCKED":"#b91c1c"}
    def st_pill(s):
        cc = ST_COLOR.get(s, "#6b7280")
        return f'<span style="font-size:9px;font-weight:700;color:{cc}">{s}</span>'
    def act_badge(a, age=None):
        if str(a).startswith("Mark"): lbl,bg,fg = "Mark available","#eff6ff","#1d4ed8"
        elif a=="Refill": lbl,bg,fg = "Refill","#fef3c7","#92400e"
        elif a=="Transform": lbl,bg,fg = "Transform","#fff7ed","#c2410c"
        elif a=="True":   lbl,bg,fg = "Reorder","#fff1f5","#be185d"
        else: lbl,bg,fg = str(a),"#f3f4f6","#6b7280"
        return f'<span style="background:{bg};color:{fg};font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;white-space:nowrap">{lbl}</span>'

    def coord384(wn):
        if pd.isna(wn): return ""
        wn=int(wn); return f"{chr(65+wn%16)}{wn//16+1}"

    def coord96(wn):
        if pd.isna(wn): return ""
        wn=int(wn); return f"{chr(65+wn%8)}{wn//8+1}"

    def glycerol_streak(part):
        _loc=apd["PLATE_LOCATION_BOX"].fillna("").astype(str)
        _lw =apd["LABWARE"].fillna("").astype(str)
        _nw =pd.to_numeric(apd["PLATE_NUMBER_OF_WELLS"], errors="coerce")
        g=apd[(apd["STOCK_ID"]==part) & (apd["WELL_TYPE"]=="Glycerol") & (apd["AVAILABLE"]=="True")
              & (apd["SEQ_CONFIRMED"]=="True")
              & ~_loc.str.upper().str.contains("DISCARD")
              & ~_loc.str.upper().str.contains("TEMP")                    # never temp glycerol
              & (_lw.str.contains("Micronic") | (_nw==96))]               # only micronic OR non-temp 96-well
        def _ab(x):
            a=[n for n,col in (("Kan","ANTI_KAN"),("Spec","ANTI_SPEC"),("Carb","ANTI_CARB")) if str(x.get(col))=="True"]
            return "/".join(a) or "?"
        rows=[]
        for _,x in g.iterrows():
            nw=x["PLATE_NUMBER_OF_WELLS"]
            co=coord384(x["WELL_NUMBER"]) if nw==384 else coord96(x["WELL_NUMBER"])
            rows.append((x["PLATE_ID"], co, str(x["PLATE_LOCATION_BOX"]), _ab(x), str(x.get("COMP_CELL") or ""), x["WELL_ID"]))
        return rows

    def dna_stock(part, seq_only=True):
        s = apd[(apd["STOCK_ID"]==part) & (apd["WELL_TYPE"]=="Stock")
                & ~apd["PLATE_LOCATION_BOX"].fillna("").astype(str).str.upper().str.contains("DISCARD")]
        if seq_only:
            s = s[s["SEQ_CONFIRMED"]=="True"]
        outd={"384":[],"96":[]}
        for _,x in s.iterrows():
            n=x["PLATE_NUMBER_OF_WELLS"]
            fmt="384" if n==384 else ("96" if n==96 else None)
            if not fmt: continue
            co=coord384(x["WELL_NUMBER"]) if fmt=="384" else coord96(x["WELL_NUMBER"])
            age=(now-x["CREATED_AT"]).days if pd.notna(x["CREATED_AT"]) else None
            outd[fmt].append((x["PLATE_ID"],co,x["WELL_ID"],str(x["PLATE_LOCATION_BOX"]),x["VOLUME_UL"],x["CONCENTRATION_NGUL"],age,x["AVAILABLE"]=="True"))
        for k in outd:
            outd[k]=sorted(outd[k], key=lambda rr:-(pd.to_numeric(rr[5],errors="coerce") or 0))
        return outd

    def avail_wells(part):
        av = apd[(apd["STOCK_ID"]==part) & (apd["AVAILABLE"]=="True")
                 & (apd["WELL_TYPE"]=="Stock") & (apd["LABWARE"]=="384 Echo Source Plate")
                 & ~apd["PLATE_LOCATION_BOX"].fillna("").astype(str).str.upper().str.contains("DISCARD")]
        rows=[]
        for _,x in av.iterrows():
            age=(now-x["CREATED_AT"]).days if pd.notna(x["CREATED_AT"]) else None
            rows.append((x["PLATE_ID"], str(x["PLATE_LOCATION_BOX"]), coord384(x["WELL_NUMBER"]),
                         x["WELL_ID"], x["VOLUME_UL"], x["CONCENTRATION_NGUL"], age))
        rows.sort(key=lambda rr:(rr[6] is None, rr[6] if rr[6] is not None else 0))
        return rows

    def make_avail_wells(part):
        win = 730 if part.startswith("o") else 200
        s = apd[(apd["STOCK_ID"]==part) & (apd["WELL_TYPE"]=="Stock") & (apd["LABWARE"]=="384 Echo Source Plate")
                & (apd["SEQ_CONFIRMED"]=="True") & (apd["AVAILABLE"]!="True")]
        rows=[]
        for _,x in s.iterrows():
            loc=str(x["PLATE_LOCATION_BOX"]) if pd.notna(x["PLATE_LOCATION_BOX"]) else ""
            if "DISCARD" in loc.upper(): continue
            if not (loc.startswith("4B-") or loc in ("","None","nan")): continue
            v=pd.to_numeric(x["VOLUME_UL"],errors="coerce"); cc=pd.to_numeric(x["CONCENTRATION_NGUL"],errors="coerce")
            age=(now-x["CREATED_AT"]).days if pd.notna(x["CREATED_AT"]) else None
            if not (pd.notna(v) and v>25 and pd.notna(cc) and cc>5): continue
            if age is not None and age>win: continue
            rows.append((x["PLATE_ID"], coord384(x["WELL_NUMBER"]), x["WELL_ID"], loc or "(no loc)", v, cc, age))
        rows.sort(key=lambda rr:-(pd.to_numeric(rr[5],errors="coerce") or 0))
        return rows

    def newest_age(part):
        ages=[a for *_,a in avail_wells(part) if a is not None]
        return min(ages) if ages else None

    _fmt=lambda v: "?" if pd.isna(v) else (f"{v:g}" if isinstance(v,(int,float)) else str(v))
    ST_CHIP={"RUNNING":"#1d4ed8","WAITING":"#b45309","READY":"#15803d","BLOCKED":"#b91c1c"}

    def _block(label, body):
        return f'<tr><th class="d-lab">{label}</th><td class="d-cell">{body}</td></tr>'

    # --- assign each part to a section ---
    ctrl_related = set(ctrl)
    for d in (set(dmeta.index) & ctrl):
        for col in ("OLIGO_1", "OLIGO_2", "DPART_TEMPLATE"):
            v = dmeta.loc[d].get(col)
            if pd.notna(v): ctrl_related.add(str(v))
    def section_of(part):
        if part in ctrl_related: return "Controls"
        if part.startswith("pAI"): return "Plasmids"
        if part.startswith("syn"): return "SynParts"
        if part.startswith("d"):  return "dParts"
        if part.startswith("o"):  return "Oligos"
        return "Other"

    # ---- build detail panel HTML for one part ----
    def detail_html(x):
        part=str(x["Part"]); act=x["Action Suggested"]
        have=int(x["Reactions Available"]); need=int(x["Reactions Required"])
        win = 730 if part.startswith("o") else 200
        nage = newest_age(part)
        is_ctrl = part in ctrl_related
        target = 96 if is_ctrl else need + max(10, need)   # buffer = min 10, else 2× (buffer == need over 10)

        if act=="Refill":
            gp=str(x.get("Glycerol Plate","") or ""); gw=str(x.get("Glycerol Well","") or "")
            gl=str(x.get("Glycerol Location","") or ""); cs=str(x.get("Cell Strain","") or "")
            src=" · ".join(b for b in [f"plate {gp}" if gp not in("","nan") else "", f"well {gw}" if gw not in("","nan") else "", f"({gl})" if gl not in("","nan","None") else "", cs if cs not in("","nan") else ""] if b)
            situation, guidance, tone = "Below target — top up", f"Streak from glycerol {src}" if src else "No glycerol source recorded", "#92400e"
        elif act=="Transform":
            situation, guidance, tone = "No glycerol stock — transform fresh", "Transform the plasmid DNA below → overnight → miniprep → re-stock", "#c2410c"
        elif act=="True":
            situation, guidance, tone = "No DNA on hand", "Reorder / synthesize", "#be185d"
        else:
            situation, guidance, tone = "", "", "#6b7280"

        blocks=[]
        pct = max(3, min(100, int(round(100*have/target)))) if target else 0
        note = f"{need} needed + {max(10, need)} buffer" if not is_ctrl else "control buffer"
        blocks.append(_block("Status",
            f'<div class="d-stat"><span class="d-have">{have}</span><span class="d-of"> on hand · target {target}</span>'
            f'<span class="d-note">({note})</span></div>'
            f'<div class="d-barwrap"><div class="d-bar" style="width:{pct}%;background:{tone}"></div></div>'
            f'<div class="d-sit" style="color:{tone}">{html.escape(situation)}</div>'))
        wl=avail_wells(part)
        if wl:
            hdr='<tr><td><b>Plate</b></td><td><b>Well</b></td><td><b>Well ID</b></td><td><b>Location</b></td><td><b>Vol</b></td><td><b>Conc</b></td><td><b>Age</b></td></tr>'
            rws="".join(f'<tr><td>plate {p}</td><td>{co or "?"}</td><td style="font-family:monospace">{wid}</td><td>{html.escape(loc)}</td><td>{_fmt(v)}µL</td><td>{_fmt(cc)} ng/µL</td><td>{a if a is not None else "?"}d</td></tr>' for p,loc,co,wid,v,cc,a in wl)
            blocks.append(_block("On hand · 4B freezer",
                f'<table class="d-tbl"><tbody>{hdr}{rws}</tbody></table>'))
        else:
            blocks.append(_block("On hand · 4B freezer", '<span style="font-size:11px;color:#9ca3af">none in 4B freezer</span>'))
        gs=glycerol_streak(part)
        if gs:
            hdr='<tr><td><b>pAI</b></td><td><b>Antibiotic</b></td><td><b>Strain</b></td><td><b>Plate</b></td><td><b>Coord</b></td><td><b>Location</b></td><td><b>Well ID</b></td></tr>'
            rws="".join(f'<tr><td style="font-family:monospace;font-weight:700">{part}</td><td>{html.escape(ab)}</td><td>{html.escape(strain or "?")}</td><td>plate {p}</td><td style="font-family:monospace">{co or "?"}</td><td>{html.escape(loc)}</td><td style="font-family:monospace">{wid}</td></tr>' for p,co,loc,ab,strain,wid in gs)
            blocks.append(_block(f"Streak from · glycerol ({len(gs)})",
                f'<table class="d-tbl"><tbody>{hdr}{rws}</tbody></table>'))
        # (Make-available / flip-ON block removed — we don't recommend flipping wells available
        #  without partner-association + 5µL/rule confirmation.)
        if act=="Transform":
            ds=dna_stock(part)
            def _dna_tbl(fmt, wells):
                if not wells: return f'<div style="font-size:11px;color:#9ca3af">no seq-confirmed {fmt}-well DNA</div>'
                hdr='<tr><td><b>Plate</b></td><td><b>Well</b></td><td><b>Well ID</b></td><td><b>Location</b></td><td><b>Vol</b></td><td><b>Conc</b></td><td><b>Age</b></td><td><b>Avail</b></td></tr>'
                rws="".join(f'<tr><td>plate {p}</td><td>{co or "?"}</td><td style="font-family:monospace">{wid}</td><td>{html.escape(loc)}</td><td>{_fmt(v)}µL</td><td>{_fmt(cc)} ng/µL</td><td>{a if a is not None else "?"}d</td><td>{"✓" if av else "—"}</td></tr>' for p,co,wid,loc,v,cc,a,av in wells)
                return f'<div style="font-size:10px;font-weight:700;color:#6b7280;margin:2px 0">{fmt}-well plates</div><table class="d-tbl"><tbody>{hdr}{rws}</tbody></table>'
            n_total=len(ds["384"])+len(ds["96"])
            cap='<div style="font-size:10px;color:#6b7280;margin-bottom:3px">seq-confirmed DNA only, highest conc first</div>' if n_total else '<div style="font-size:11px;color:#be185d">No seq-confirmed DNA on hand — would need reorder/synthesis, not transform</div>'
            blocks.append(_block("DNA to transform", cap + (_dna_tbl("384",ds["384"]) + '<div style="height:6px"></div>' + _dna_tbl("96",ds["96"]) if n_total else "")))
        if guidance:
            blocks.append(_block("Do this", f'<div class="d-do">{html.escape(guidance)}</div>'))
        if act in ("Refill","Transform"):
            st,age,proc,proto = refill_status(part)
            if st=="pre_ngs":
                prog=f'<span style="color:#15803d;font-weight:700">⟳ Batch in progress</span> — plate map up, at <b>{html.escape(proto)}</b> (waiting before NGS), {age}d ago · {html.escape(proc)}'
            elif st=="ngs":
                prog=f'<span style="color:#be185d;font-weight:700">⚠ Needs batching</span> <span style="color:#9ca3af">(last batch reached NGS {age}d ago · {html.escape(proc)})</span>'
            else:
                prog='<span style="color:#be185d;font-weight:700">⚠ Needs batching</span> <span style="color:#9ca3af">— no refill on record</span>'
            blocks.append(_block("In progress?", f'<div style="font-size:11px">{prog}</div>'))
        elif act=="True":
            o=order_status(part)
            if o is None:
                prog='<span style="color:#9ca3af">No vendor order on record</span>'
            elif o["active"]:
                prog=f'<span style="color:#15803d;font-weight:700">⟳ On order</span> — {html.escape(str(o["vendor"]))} {o["status"]} (order {html.escape(str(o["order_id"]))})'
            else:
                prog=f'<span style="color:#9ca3af">Last ordered</span> {o["date"]:%Y-%m-%d} · {html.escape(str(o["vendor"]))} ({o["status"]}) — needs new order'
            blocks.append(_block("In progress?", f'<div style="font-size:11px">{prog}</div>'))
        bb=builds_for(part)
        if bb:
            exps={}
            for p,t,s,e in bb: exps.setdefault(e or "—",[]).append((p,t,s))
            body=f'<div class="d-sub">{len(bb)} build{"s" if len(bb)!=1 else ""} across {len(exps)} experiment{"s" if len(exps)!=1 else ""}</div>'
            for e,builds in sorted(exps.items(), key=lambda kv:-len(kv[1])):
                chips="".join(f'<span class="chip" style="border-color:{ST_CHIP.get(s,"#9ca3af")};color:{ST_CHIP.get(s,"#6b7280")}">{html.escape(p)} <em>{t}·{s.lower()}</em></span>' for p,t,s in builds[:24])
                more = f' <span class="d-more">+{len(builds)-24}</span>' if len(builds)>24 else ""
                body+=f'<div class="d-exp"><div class="d-expname">{html.escape(e)} <span class="d-cnt">{len(builds)}</span></div><div class="d-chips">{chips}{more}</div></div>'
            blocks.append(_block("Needed for", body))
        else:
            via=[d for d in tmpl_kids.get(part,[]) if d in flagged]
            co=[d for d in dmeta.index if str(dmeta.loc[d].get("OLIGO_1"))==part or str(dmeta.loc[d].get("OLIGO_2"))==part]
            co=[d for d in co if d in ctrl]
            if via: msg=f"PCR template for {', '.join(via)}"
            elif co: msg=f"Primer for control dPart {', '.join(co)} — kept permanently stocked"
            elif part in ctrl: msg="Control — kept permanently stocked"
            else: msg="Nothing live needs it right now"
            blocks.append(_block("Needed for", f'<div class="d-do">{html.escape(msg)}</div>'))
        return '<table class="detail">'+"".join(blocks)+'</table>'

    order = {"Refill":0,"Transform":1,"True":2}
    out["_o"]=out["Action Suggested"].map(lambda a: order.get(a,3 if not str(a).startswith("Mark") else 1))
    out["_sec"]=out["Part"].astype(str).map(section_of)

    def batch_cell(part, act):
        if act in ("Refill","Transform"):
            st,age,proc,proto = refill_status(part)
            if st=="pre_ngs": return '<span style="color:#15803d;font-weight:700">⟳ batch in progress</span>'
            return '<span style="color:#be185d;font-weight:700">⚠ needs batch</span>'
        if act=="True":
            o=order_status(part)
            if o and o["active"]: return f'<span style="color:#15803d">on order · {html.escape(str(o["vendor"]))}</span>'
            return '<span style="color:#be185d;font-weight:700">⚠ needs order</span>'
        return '<span style="color:#9ca3af">—</span>'

    def repeat_badge(demand):
        d=int(demand) if pd.notna(demand) else 0
        if d>=20:   lvl,bg,fg="HIGH","#fee2e2","#b91c1c"
        elif d>=10: lvl,bg,fg="MED","#fef3c7","#92400e"
        else:       lvl,bg,fg="LOW","#f0fdf4","#15803d"
        return f'<span title="feeds {d} downstream builds" style="background:{bg};color:{fg};font-size:8px;font-weight:700;padding:1px 5px;border-radius:8px;margin-left:5px">{lvl}</span>'

    _rowstate = {"i": 0}
    def row_html(i, x):
        part=str(x["Part"]); act=x["Action Suggested"]
        age=None
        bb=builds_for(part)
        if bb: summary=f'{bb[0][0]} ({bb[0][2]})'+(f' +{len(bb)-1} more' if len(bb)>1 else '')
        elif part in ctrl: summary="control — kept permanently stocked"
        elif part in ctrl_related: summary="primer/template for a control dPart"
        elif [d for d in tmpl_kids.get(part,[]) if d in flagged]: summary="template for "+", ".join(d for d in tmpl_kids.get(part,[]) if d in flagged)
        else: summary="—"
        return (f'<tr class="prow" onclick="partsToggle({i})" style="cursor:pointer">'
                f'<td style="width:18px;color:#9ca3af" id="c{i}">▸</td>'
                f'<td style="font-family:monospace;font-weight:700">{part}{repeat_badge(x["Reactions Required"])}</td>'
                f'<td style="text-align:center">{int(x["Reactions Available"])} / <strong style="color:#b45309;font-size:13px">{int(x["Reactions Required"])}</strong></td>'
                f'<td>{act_badge(act,age)}</td>'
                f'<td style="font-size:11px">{batch_cell(part,act)}</td>'
                f'<td style="font-size:11px;color:#374151">{html.escape(summary)}</td></tr>'
                f'<tr id="d{i}" style="display:none"><td></td><td colspan="5">{detail_html(x)}</td></tr>')

    # New builds = parts whose own assembly workorder is in flight (BLOCKED/READY/RUNNING/WAITING)
    out["_isbuild"]=out["Action Suggested"].astype(str).str.contains("workorder", case=False)
    builds_all = r.get("builds")
    if builds_all is None or builds_all.empty:
        builds_all = out[out["_isbuild"]].copy()
    builds_all = builds_all[~builds_all["Part"].astype(str).str.startswith("o")].copy()

    def has_avail_glycerol(part):
        g=apd[(apd["STOCK_ID"]==part)&(apd["WELL_TYPE"]=="Glycerol")&(apd["AVAILABLE"]=="True")&(apd["SEQ_CONFIRMED"]=="True")]
        return len(g)>0
    _refillable={p for p in builds_all["Part"].astype(str).unique() if has_avail_glycerol(p)}
    if _refillable:
        mv=builds_all[builds_all["Part"].astype(str).isin(_refillable)].copy()
        mv["Action Suggested"]="Refill"
        builds_all=builds_all[~builds_all["Part"].astype(str).isin(_refillable)]
        out=out[~out["Part"].astype(str).isin(_refillable)]
        out=pd.concat([out,mv],ignore_index=True)
        out["_o"]=out["Action Suggested"].map(lambda a: order.get(a,3 if not str(a).startswith("Mark") else 1))
        out["_isbuild"]=out["Action Suggested"].astype(str).str.contains("workorder", case=False)
        out["_sec"]=out["Part"].astype(str).map(section_of)

    def _blocked_only(part):
        bb=builds_for(part)
        return bool(bb) and all(s=="BLOCKED" for _p,_t,s,_e in bb)
    _phantom={p for p in out["Part"].astype(str).unique() if _blocked_only(p)}
    if _phantom:
        out=out[~out["Part"].astype(str).isin(_phantom)].reset_index(drop=True)

    # ---- Demand = DIRECT in-flight builds only (reconciles the number with "Needed for") ----
    def direct_need(part):
        return len(builds_for(str(part)))
    out["Reactions Required"]=out.apply(
        lambda x: int(x["Reactions Required"]) if str(x["Part"]) in ctrl_related else direct_need(x["Part"]),
        axis=1)
    builds_all["Reactions Required"]=builds_all.apply(
        lambda x: int(x["Reactions Required"]) if str(x["Part"]) in ctrl_related else direct_need(x["Part"]),
        axis=1)

    def _target(x):
        need=int(x["Reactions Required"])
        return 96 if str(x["Part"]) in ctrl_related else need + max(10, need)
    out=out[out.apply(
        lambda x: bool(x["_isbuild"]) or (str(x["Part"]) in ctrl_related)
                  or int(x["Reactions Available"]) < _target(x), axis=1)].reset_index(drop=True)

    i = 0
    def part_exps(part):
        return sorted({e for _,_,_,e in builds_for(part) if e})
    _PCOLS='<colgroup><col style="width:26px"><col style="width:19%"><col style="width:11%"><col style="width:12%"><col style="width:23%"><col></colgroup>'
    _HDR='<tr><th></th><th>Part</th><th>Have / Need</th><th>Action</th><th>Batch / order</th><th>Needed for</th></tr>'
    def _exp_group(title, rowobjs, accent="#7c3aed", desc="", open_=True):
        nonlocal i
        body=""
        for x in rowobjs:
            body+=row_html(i,x); i+=1
        md=max((int(x["Reactions Required"]) for x in rowobjs), default=0)
        dsc=f'<span class="secdesc">{desc}</span>' if desc else ''
        return (f'<details class="expgrp"{" open" if open_ else ""}><summary>'
                f'<span class="egname">{html.escape(title)}</span>'
                f'<span class="egcount">{len(rowobjs)}</span>'
                f'{repeat_badge(md)}{dsc}</summary>'
                f'<table class="ptbl">{_PCOLS}<tbody>{body}</tbody></table></details>')
    def grouped_by_experiment(rowobjs, multi_accent="#7c3aed", noexp_title="Controls / no live demand", open_=True):
        by={}; multi=[]; noexp=[]
        for x in rowobjs:
            exps=part_exps(str(x["Part"]))
            if not exps: noexp.append(x)
            elif len(exps)==1: by.setdefault(exps[0],[]).append(x)
            else: multi.append(x)
        hh=""
        for e in sorted(by, key=lambda e:-sum(int(x["Reactions Required"]) for x in by[e])):
            hh+=_exp_group(e, sorted(by[e], key=lambda x:-int(x["Reactions Required"])), open_=open_)
        if multi:
            hh+=_exp_group("Multi-project parts", sorted(multi,key=lambda x:-int(x["Reactions Required"])),
                           accent=multi_accent, desc="feed more than one experiment — see “Needed for”", open_=open_)
        if noexp:
            hh+=_exp_group(noexp_title, noexp, open_=open_)
        return hh

    _nb_rows=[x for _,x in builds_all.iterrows()] if not builds_all.empty else []
    _pa_rows=[x for _,x in out[~out["_isbuild"]].iterrows()]
    newbuilds_html = grouped_by_experiment(_nb_rows, open_=False) if _nb_rows else ""
    parts_html     = grouped_by_experiment(_pa_rows, open_=True)

    # ============================================================================
    # Well/plate action sections: Make Unavailable, Trash
    # ============================================================================
    lsp_plates = r.get("lsp_plates")
    lsp_ids = set()
    if lsp_plates is not None and not lsp_plates.empty and "PLATE_ID" in lsp_plates.columns:
        lsp_ids = set(pd.to_numeric(lsp_plates["PLATE_ID"], errors="coerce").dropna().astype(int))

    clean_wells = P.build_clean_inventory_queue(apd, now, exclude_oligos=True)
    mp_wells    = P.build_miniprep_unavail_queue(apd, now)
    exhausted   = P.build_exhausted_plates_queue(apd)
    dispose     = [d for d in P.build_dispose_queue(lsp_plates, now)
                   if d["old"] and str(d["location"]) not in ("(no location)","None","","nan")]
    dispose.sort(key=lambda d: -(d["age_days"] or 0))

    def _copybox(bid, text):
        return (f'<div style="display:flex;gap:6px;margin:6px 14px 12px"><textarea id="{bid}" readonly '
                f'style="flex:1;height:54px;font-family:monospace;font-size:10px;border:1px solid #d1d5db;'
                f'border-radius:4px;padding:5px 7px;background:#fafafa;resize:vertical">{text}</textarea>'
                f'<button onclick="var t=document.getElementById(\'{bid}\');t.select();navigator.clipboard.writeText(t.value)" '
                f'style="font-size:10px;font-weight:600;padding:5px 10px;border:1px solid #c4b5fd;border-radius:4px;'
                f'background:#ede9fe;color:#6d28d9;cursor:pointer;white-space:nowrap;align-self:flex-start">Copy</button></div>')

    def wells_section(sid, title, desc, accent, tokens, show_plates=True):
        body = (_copybox(sid, ",".join(tokens))) if tokens \
            else '<div style="padding:10px 14px;font-size:11px;color:#86868b">None.</div>'
        return (f'<div class="sec"><div class="sechd" style="border-left:4px solid {accent}">{title} '
                f'<span class="seccount" style="background:{accent}">{len(tokens)}</span>'
                f'<span class="secdesc">{desc}</span></div>{body}</div>')

    # "Make Available" removed — can't confirm these wells aren't partner-associated or rule-compliant.
    extra = wells_section("mk_un","Make Unavailable · 384 Echo source",
              "available Echo source wells that are ≤25µL (near-empty), past expiration (200d), OR &lt;5 ng/µL (too dilute) → flip OFF in LIMS","#be185d", clean_wells, show_plates=False)
    extra += wells_section("mk_un_mp","Make Unavailable · 96-well miniprep stock",
              "available miniprep-stock wells (96-well) past expiration (200d) → flip OFF in LIMS","#9d174d", mp_wells, show_plates=False)

    # Trash — LSP Echo plates >2mo
    if dispose:
        drows="".join(f'<tr><td style="font-family:monospace">{d["plate_id"]}</td>'
                      f'<td>{html.escape(str(d["location"]))}</td>'
                      f'<td style="color:#6b7280">{html.escape(str(d["protocol"]))}</td>'
                      f'<td style="color:#86868b">{d["created"]}</td>'
                      f'<td style="white-space:nowrap;color:#be123c;font-weight:700">{(d["age_days"]or 0)//30}mo</td></tr>'
                      for d in dispose)
        dispbody=(f'<table class="platetbl"><thead><tr>'
                  f'<th style="text-align:left;padding:3px 10px">Plate</th><th style="text-align:left;padding:3px 10px">Location</th>'
                  f'<th style="text-align:left;padding:3px 10px">Protocol</th><th style="text-align:left;padding:3px 10px">Created</th>'
                  f'<th style="text-align:left;padding:3px 10px">Age</th></tr></thead><tbody>{drows}</tbody></table>')
    else:
        dispbody='<div style="padding:10px 14px;font-size:11px;color:#86868b">No LSP Echo plates older than 2 months.</div>'
    extra += (f'<div class="sec"><div class="sechd" style="border-left:4px solid #6b7280">Trash — LSP Echo plates &gt;2mo '
              f'<span class="seccount" style="background:#6b7280">{len(dispose)}</span>'
              f'<span class="secdesc">LSP dilution plates past 2 months · physically toss (wells stay unavailable)</span></div>'
              f'{dispbody}</div>')

    # Trash — plates marked unusable (0 µL)
    if exhausted:
        erows="".join(f'<tr><td style="font-family:monospace">{e["plate_id"]}</td>'
                      f'<td>{html.escape(str(e["location"]))}</td>'
                      f'<td style="text-align:right;color:#6b7280">{e["wells"]}</td>'
                      f'<td style="color:#86868b">{e["created"]}</td></tr>'
                      for e in exhausted)
        exhbody=(f'<table class="platetbl"><thead><tr>'
                 f'<th style="text-align:left;padding:3px 10px">Plate</th><th style="text-align:left;padding:3px 10px">Location</th>'
                 f'<th style="text-align:right;padding:3px 10px">Wells</th><th style="text-align:left;padding:3px 10px">Created</th>'
                 f'</tr></thead><tbody>{erows}</tbody></table>')
    else:
        exhbody='<div style="padding:10px 14px;font-size:11px;color:#86868b">No plates marked unusable.</div>'
    extra += (f'<div class="sec"><div class="sechd" style="border-left:4px solid #6b7280">Trash — plates marked unusable (0 µL) '
              f'<span class="seccount" style="background:#6b7280">{len(exhausted)}</span>'
              f'<span class="secdesc">every well drained to 0 µL (marked unusable) → confirm the plate is physically discarded and update its location to DISCARDED · some still show a stale/active location</span></div>'
              f'{exhbody}</div>')

    # Trash by part type: STOCK plates in the 4B freezer, past expiration
    def _ptype(sid):
        sid=str(sid)
        if sid.startswith("pAI"): return "Plasmid"
        if sid.startswith("syn"): return "SynPart"
        if sid.startswith("d"):  return "dPart"
        if sid.startswith("o"):  return "Oligo"
        return "Other"
    EXPIRE_DAYS={"Plasmid":200,"dPart":200,"SynPart":200,"Oligo":730}

    stk = apd[(apd["WELL_TYPE"]=="Stock") & (apd["LABWARE"]=="384 Echo Source Plate")].copy()
    stk = stk[stk["PLATE_LOCATION_BOX"].astype(str).str.startswith("4B-")]
    stk["ptype"]=stk["STOCK_ID"].map(_ptype)
    stk = stk[stk["ptype"]!="Other"]
    stk["CREATED_AT"]=pd.to_datetime(stk["CREATED_AT"],errors="coerce",utc=True)
    exp_by_type={"Plasmid":[],"dPart":[],"SynPart":[],"Oligo":[]}
    for pid,g in stk.groupby("PLATE_ID"):
        dom=g["ptype"].mode().iloc[0] if not g["ptype"].mode().empty else "Other"
        if dom not in exp_by_type: continue
        created=g["CREATED_AT"].min()
        if pd.isna(created): continue
        age=(now-created).days
        if age > EXPIRE_DAYS[dom]:
            loc=g["PLATE_LOCATION_BOX"].mode().iloc[0]
            exp_by_type[dom].append((int(pid), str(loc), created.strftime("%Y-%m-%d"), age, g["WELL_ID"].nunique()))

    def trash_section(sid, label, accent, win, plates):
        if not plates:
            return (f'<div class="sec"><div class="sechd" style="border-left:4px solid {accent}">{label} '
                    f'<span class="seccount" style="background:{accent}">0</span>'
                    f'<span class="secdesc">no plates past {win}</span></div>'
                    f'<div style="padding:10px 14px;font-size:11px;color:#86868b">None.</div></div>')
        plates=sorted(plates, key=lambda x:-x[3])
        rows="".join(f'<tr><td style="font-family:monospace">plate {p}</td><td>{html.escape(loc)}</td>'
                     f'<td style="color:#86868b">{cr}</td><td style="white-space:nowrap;color:#be123c;font-weight:700">{a}d</td></tr>'
                     for p,loc,cr,a,n in plates)
        tbl=(f'<table class="platetbl"><thead><tr>'
             f'<th style="text-align:left;padding:3px 10px">Plate</th><th style="text-align:left;padding:3px 10px">Location</th>'
             f'<th style="text-align:left;padding:3px 10px">Created</th><th style="text-align:left;padding:3px 10px">Age</th>'
             f'</tr></thead><tbody>{rows}</tbody></table>')
        return (f'<div class="sec"><div class="sechd" style="border-left:4px solid {accent}">{label} '
                f'<span class="seccount" style="background:{accent}">{len(plates)}</span>'
                f'<span class="secdesc">stock plates older than {win} · physically toss</span></div>{tbl}</div>')

    extra += trash_section("trash_pl","Trash — Plasmid stock plates","#7c3aed","200 days", exp_by_type["Plasmid"])
    extra += trash_section("trash_dp","Trash — dPart stock plates","#0891b2","200 days", exp_by_type["dPart"])
    extra += trash_section("trash_sp","Trash — 384 rearray with synparts","#15803d","200 days", exp_by_type["SynPart"])

    _n_refill = int((out["Action Suggested"]=="Refill").sum())
    _n_xform  = int((out["Action Suggested"]=="Transform").sum())
    _n_nosrc  = int((out["Action Suggested"]=="True").sum())
    _n_trash  = (len(dispose) + len(exhausted)
                 + sum(len(exp_by_type[t]) for t in ("Plasmid","dPart","SynPart")))

    def section_card(title, tri, count, body, desc="", colhdr=""):
        if not body: return ""
        bg,txt,bd = tri
        dsc=f'<div class="scdesc">{html.escape(desc)}</div>' if desc else ''
        return (f'<div class="secblock"><div class="schd" style="background:{bg};border-left:4px solid {txt}">'
                f'<span class="schd-t">{html.escape(title)}</span>'
                f'<span class="sccount" style="background:{txt};color:#fff">{count}</span>{dsc}</div>'
                f'{colhdr}'
                f'<div class="scbody">{body}</div></div>')
    _nb_count=int(len(builds_all)) if builds_all is not None else 0
    _pa_count=int(len(out[~out["_isbuild"]]))
    _PHDR_ROW=f'<table class="ptbl hdrow">{_PCOLS}<thead>{_HDR}</thead></table>'
    _BHDR_ROW=f'<table class="ptbl hdrow">{_BCOLS}<thead>{_BHDR}</thead></table>'

    # ---- SCOPED fragment: every CSS rule namespaced under #tab-parts so nothing leaks ----
    frag=f"""<style>
 #tab-parts{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f1f5f9;color:#1d1d1f}}
 #tab-parts .hd{{background:#fff;border-bottom:1px solid #e5e7eb;padding:16px 20px}} #tab-parts .hd h1{{font-size:18px;margin:0 0 4px}} #tab-parts .hd p{{font-size:12px;color:#6b7280;margin:0}}
 #tab-parts .sec{{margin:16px;background:#fff;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden}}
 #tab-parts .sechd{{padding:10px 14px;font-size:14px;font-weight:700;background:#faf8ff;border-bottom:1px solid #ece8f5}}
 #tab-parts .seccount{{background:#6d28d9;color:#fff;font-size:11px;font-weight:700;border-radius:10px;padding:1px 8px;margin-left:6px}}
 #tab-parts .secdesc{{font-size:11px;font-weight:400;color:#9ca3af;margin-left:10px}}
 #tab-parts table{{width:100%;border-collapse:collapse;font-size:12px}}
 #tab-parts thead th{{text-align:left;background:#f3f4f6;color:#0f172a;font-size:9px;text-transform:uppercase;letter-spacing:.04em;padding:8px 10px;border-bottom:1px solid #cbd5e1;white-space:nowrap}}
 #tab-parts .prow td{{padding:8px 10px;border-bottom:1px solid #f0f0f3}} #tab-parts .prow:hover{{background:#f5f3ff}}
 #tab-parts .note{{margin:0 16px 16px;font-size:11px;color:#6b7280;line-height:1.6}}
 #tab-parts .secgroup-title{{margin:22px 16px 2px;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#9ca3af}}
 #tab-parts .secblock{{margin:20px 16px;background:#fdfdfd;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(15,23,42,.06)}}
 #tab-parts .schd{{padding:12px 16px;border-bottom:1px solid #eef2f6;display:flex;align-items:center;flex-wrap:wrap;gap:8px}}
 #tab-parts .schd-t{{font-size:15px;font-weight:700;color:#0f172a;letter-spacing:-.01em}}
 #tab-parts .sccount{{font-size:10px;font-weight:700;border-radius:4px;padding:1px 7px}}
 #tab-parts .scdesc{{flex-basis:100%;font-size:11px;font-weight:400;color:#94a3b8;margin-top:1px}}
 #tab-parts .expgrp{{border-bottom:1px solid #f1f5f9}} #tab-parts .expgrp:last-child{{border-bottom:none}}
 #tab-parts .expgrp>summary{{padding:8px 16px 8px 14px;font-size:12px;font-weight:600;cursor:pointer;list-style:none;display:flex;align-items:center;gap:7px;color:#475569;background:#fff}}
 #tab-parts .expgrp>summary::-webkit-details-marker{{display:none}}
 #tab-parts .expgrp>summary::before{{content:'▸';color:#94a3b8;font-size:9px}}
 #tab-parts .expgrp[open]>summary::before{{content:'▾'}}
 #tab-parts .expgrp>summary:hover{{background:#f8fafc}}
 #tab-parts .expgrp[open]>summary{{background:#f8fafc;border-bottom:1px solid #f1f5f9}}
 #tab-parts .egname{{font-weight:600;color:#334155}}
 #tab-parts .egcount{{background:#f1f5f9;color:#475569;border:1px solid #e2e8f0;font-size:10px;font-weight:700;border-radius:4px;padding:1px 6px}}
 #tab-parts .ptbl{{table-layout:fixed;width:100%}}
 #tab-parts .ptbl td{{overflow-wrap:anywhere}}
 #tab-parts .ptbl.hdrow{{border-bottom:1px solid #e5e7eb}}
 #tab-parts .ptbl.hdrow thead th{{background:#f8fafc}}
 #tab-parts .platetbl{{width:auto;border-collapse:collapse;font-size:11px;margin:2px 16px 10px}}
 #tab-parts .platetbl thead th{{text-align:left;background:#f8fafc;color:#475569;font-size:9px;text-transform:uppercase;letter-spacing:.04em;padding:6px 14px;border-bottom:1px solid #e5e7eb;white-space:nowrap}}
 #tab-parts .platetbl td{{padding:5px 14px;border-bottom:1px solid #f1f5f9;color:#374151;white-space:nowrap}}
 #tab-parts .platetbl tbody tr:nth-child(even) td{{background:#fafbfc}}
 #tab-parts .ov{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 16px 4px}}
 #tab-parts .ovc{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:10px 16px;min-width:96px}}
 #tab-parts .ovn{{font-size:24px;font-weight:800;line-height:1}} #tab-parts .ovl{{font-size:11px;color:#6b7280;margin-top:3px}}
 #tab-parts table.detail{{width:100%;border-collapse:collapse;background:#fbfaff;border-top:1px solid #d9d4ea}}
 #tab-parts table.detail th.d-lab{{width:118px;text-align:left;vertical-align:top;padding:10px 14px;font-size:9px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#9ca3af;border-bottom:1px solid #e7e3f2;border-right:1px solid #e7e3f2;background:#f6f4fc}}
 #tab-parts table.detail td.d-cell{{padding:10px 14px;border-bottom:1px solid #e7e3f2;vertical-align:top}}
 #tab-parts .d-stat{{display:flex;align-items:baseline;gap:6px;flex-wrap:wrap}}
 #tab-parts .d-have{{font-size:22px;font-weight:800}} #tab-parts .d-of{{font-size:12px;color:#374151}} #tab-parts .d-note{{font-size:11px;color:#9ca3af}}
 #tab-parts .d-barwrap{{height:6px;background:#eceaf3;border-radius:4px;margin:6px 0 4px;max-width:320px;overflow:hidden}} #tab-parts .d-bar{{height:100%;border-radius:4px}}
 #tab-parts .d-sit{{font-size:12px;font-weight:600}}
 #tab-parts .d-tbl{{font-size:11px;border-collapse:collapse}} #tab-parts .d-tbl td{{border:1px solid #d8d4e6;padding:3px 9px;color:#374151;white-space:nowrap}}
 #tab-parts .d-tbl tr:first-child td{{background:#f3f1fa}}
 #tab-parts .d-do{{font-size:12px;color:#1f2937}}
 #tab-parts .d-sub{{font-size:11px;color:#6b7280;margin-bottom:6px}}
 #tab-parts .d-exp{{margin-bottom:8px}} #tab-parts .d-expname{{font-size:11px;font-weight:700;color:#0f172a}} #tab-parts .d-cnt{{background:#eef;color:#3730a3;border-radius:9px;font-size:9px;padding:0 6px;margin-left:4px}}
 #tab-parts .d-chips{{margin-top:3px;line-height:2}}
 #tab-parts .chip{{font-size:10px;font-family:monospace;border:1px solid;border-radius:4px;padding:1px 5px;margin:0 4px 3px 0;display:inline-block}}
 #tab-parts .chip em{{font-style:normal;color:#9ca3af;font-size:9px}}
 #tab-parts .d-more{{font-size:10px;color:#9ca3af}}
</style>
<div class="hd" style="position:relative"><h1>Parts Inventory</h1>
<p>click any part for detail</p>
<div style="position:absolute;top:14px;right:20px;font-size:11px;font-weight:700;color:#1d4ed8;background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:4px 11px"><span style="color:#93a5c9;letter-spacing:.04em">PARTS DATA PULLED</span> {now_et:%Y-%m-%d %-I:%M %p} ET</div></div>
<div class="ov">
  <div class="ovc"><div class="ovn" style="color:#92400e">{_n_refill}</div><div class="ovl">Refill</div></div>
  <div class="ovc"><div class="ovn" style="color:#c2410c">{_n_xform}</div><div class="ovl">Transform</div></div>
  <div class="ovc"><div class="ovn" style="color:#be185d">{_n_nosrc}</div><div class="ovl">Reorder</div></div>
  <div class="ovc"><div class="ovn" style="color:#be185d">{len(clean_wells)+len(mp_wells)}</div><div class="ovl">Wells → unavailable</div></div>
  <div class="ovc"><div class="ovn" style="color:#6b7280">{_n_trash}</div><div class="ovl">Plates to trash</div></div>
</div>
{section_card("Parts needing attention", ("#fffbeb","#b45309","#fde68a"), _pa_count, parts_html, "restock / refill / reorder — grouped by experiment · Need = direct in-flight builds", colhdr=_PHDR_ROW)}
{section_card("Blocked workorders", ("#fef2f2","#b91c1c","#fca5a5"), len(_blk), blocked_html, "stuck assembly WOs — grouped by experiment · click a row for what it's blocking + cancel note", colhdr=_BHDR_ROW)}
{section_card("New builds — feed into requests", ("#f5f3ff","#6d28d9","#ddd6fe"), _nb_count, newbuilds_html, "net-new parts being assembled (workorder in flight) that feed downstream requests", colhdr=_PHDR_ROW)}
<p class="note"><b>Refill</b> = has glycerol → streak &nbsp;·&nbsp; <b>Transform</b> = no glycerol but DNA on hand → transform fresh &nbsp;·&nbsp; <b>Reorder</b> = no DNA → order/synthesize</p>
<div class="secgroup-title">Well &amp; plate actions</div>
{extra}
<script>
function partsToggle(i){{var d=document.getElementById('d'+i),c=document.getElementById('c'+i);
 var open=d.style.display==='none'; d.style.display=open?'table-row':'none'; c.textContent=open?'▾':'▸';}}
</script>"""
    return frag

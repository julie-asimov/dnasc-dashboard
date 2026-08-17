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
import re
import html
import pickle
import math
import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from dnasc.config import PipelineConfig

_ET = ZoneInfo("America/New_York")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))   # repo/scripts root
_PKL = os.path.join(_ROOT, "dashboard_state", "parts_result.pkl")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_FALLBACK = ('<div style="padding:24px;color:#6b7280;font:14px -apple-system,sans-serif">'
             'Parts inventory data unavailable — the parts pull (gen_parts_pkl.py) has not run '
             'yet or failed. Check the parts cron / logs/parts_pull.log.</div>')

# --- "already started?" refill signal: wells tagged REFILL_* in PROCESS_ID ---
# A refill is finished when it has rearrayed into the 384 Echo source plate — that step is what
# CREATES the Echo stock wells, so "this process has Echo wells" is the completion test. It is
# exact, not a heuristic: across all 317 refill processes in the pull, every one that reached
# NGS or rearray has Echo wells (17 + 293) and every one that has not (7, all at miniprep) has
# none. Age is reported but never decides the state — a batch that went through NGS is done
# whether that was yesterday or two years ago, which is why an 80-day-old refill used to sit
# here claiming to be in progress.
_REFILL_TYPICAL_DAYS = 9        # p95 of first→last well span (median 3d); "looks stalled" past this
# Furthest stage reached, in real refill order (streak → overnights → miniprep → quant →
# NGS confirm → rearray into the 384 Echo source). Rank by the LAST stage reached, never by
# the newest well's protocol — a process touches several plates on the same day, so
# "newest well" picks an arbitrary one and mislabels the stage.
_STAGE_RANK = {"Overnight Culture":1, "Bank Overnights":1, "Miniprep":2, "DNA Quant":3,
               "Sequence Plasmid":4, "NGS Sequence Confirmation":4, "Rearray 96 to 384":5}
_STAGE_LABEL = {1:"overnight culture", 2:"miniprep", 3:"DNA quant",
                4:"NGS (sequencing)", 5:"rearray into 384 Echo"}


def _is_echo384(d):
    # A REAL Echo source plate is 384-well. Some plates carry the '384 Echo
    # Source Plate' labware while being physically 96-well (LIMS data error) —
    # those are not Echo sources and must never enter the Echo-source flows
    # (make-available, on-hand, trash). Gate on the actual well_count.
    return ((d["LABWARE"] == "384 Echo Source Plate")
            & (pd.to_numeric(d["PLATE_NUMBER_OF_WELLS"], errors="coerce") == 384))


def _make_refill_status(apd, ngs_df, now):
    """Build the refill_status(part) reader over a loaded pull.

    Module-level (not a closure inside _render) so the in-flight tab can ask the same question
    about a part that is blocking a request without re-deriving any of it.
    """
    # NGS job state per well, from the pull. This is the real "is it still running?" signal: a
    # plate sitting on an NGS protocol only means the samples were submitted. RUNNING = still
    # sequencing, SUCCEEDED/FAILED/CANCELED = the job closed and the answer is in. Older pkls
    # have no ngs_df.
    _ngs_by_well = {}
    if ngs_df is not None and len(ngs_df):
        _n = ngs_df.copy()
        _n["WELL_ID"] = pd.to_numeric(_n["WELL_ID"], errors="coerce")
        for _w, _g in _n.dropna(subset=["WELL_ID"]).groupby("WELL_ID"):
            _ngs_by_well[int(_w)] = (set(_g["STATUS"].astype(str)), _g["UPDATED"].max())
    _refill = apd[apd["PROCESS_ID"].astype(str).str.contains("REFILL", case=False, na=False)].copy()
    _refill["age"]=(now-_refill["CREATED_AT"]).dt.days

    def refill_status(part):
        """State of the most recent refill batch for `part`.

        Returns a dict: state = 'inflight' | 'done' | 'none'.
          inflight — the batch has not finished: either it has not reached NGS yet, or its NGS job
                     is still RUNNING. `stage` is the furthest stage reached; `stalled` if it has
                     gone quiet for longer than a refill normally takes.
          done     — its NGS job closed (or it rearrayed into the Echo plate). `ngs` carries the
                     closing status, `landed` how much of the stock is still usable — together
                     those decide whether a new batch is needed.
        """
        w=_refill[_refill["STOCK_ID"]==str(part)]
        if w.empty: return {"state":"none"}
        proc=str(w.sort_values("CREATED_AT").iloc[-1]["PROCESS_ID"])   # newest process for this part
        g=w[w["PROCESS_ID"].astype(str)==proc]
        age=int((now-g["CREATED_AT"].max()).days) if pd.notna(g["CREATED_AT"].max()) else None
        rank=max((_STAGE_RANK.get(str(p),0) for p in g["PLATE_PROTOCOL"]), default=0)
        stage=_STAGE_LABEL.get(rank, "started")
        # SEQ_CONFIRMED is the sequencing verdict. The NGS *workorder status* is not: a job routinely
        # closes FAILED while its wells carry SEQ_CONFIRMED=True, because the job can close short for
        # reasons that have nothing to do with the read (low yield, for one). So never report a
        # closed-FAILED workorder as "sequencing failed" when the wells are confirmed.
        seq_ok=bool((g["SEQ_CONFIRMED"].astype(str)=="True").any())
        # NGS job state over this process's wells decides "in progress". NGS only runs on the
        # picked samples, so most wells carry no job — what matters is whether the process has any
        # job and whether they have closed.
        _st=set(); _upd=None
        for _wid in g["WELL_ID"]:
            try: _hit=_ngs_by_well.get(int(_wid))
            except (TypeError, ValueError): _hit=None
            if _hit:
                _st |= _hit[0]
                if _upd is None or (pd.notna(_hit[1]) and _hit[1] > _upd): _upd = _hit[1]
        ngs=None
        if _st:
            closed=not ("RUNNING" in _st)
            ngs={"statuses":_st, "closed":closed,
                 "outcome":("SUCCEEDED" if "SUCCEEDED" in _st else
                            ("FAILED" if "FAILED" in _st else
                             ("CANCELED" if "CANCELED" in _st else "RUNNING"))),
                 "closed_days":(int((now-_upd).days) if _upd is not None and pd.notna(_upd) else None)}
        # Rearray into the 384 Echo source comes after NGS, so wells there also mean it finished —
        # it covers the legacy processes whose NGS job predates the ngsworkorder records.
        e=apd[(apd["PROCESS_ID"].astype(str)==proc) & (apd["STOCK_ID"]==str(part))
              & (apd["WELL_TYPE"]=="Stock") & _is_echo384(apd)]
        still_sequencing = bool(ngs and not ngs["closed"])
        if still_sequencing or (ngs is None and rank < 4 and not len(e)):
            return {"state":"inflight","age":age,"proc":proc,"stage":stage,"rank":rank,"ngs":ngs,
                    "sequencing":still_sequencing,"seq_ok":seq_ok,
                    "stalled":bool(not still_sequencing and age is not None and age > _REFILL_TYPICAL_DAYS)}
        # Finished. How much of what it left is still usable? Only ever report the CURRENT state of
        # those wells — never why they got that way. LIMS overwrites VOLUME_UL in place, so a well
        # that was consumed normally is indistinguishable from a prep that never worked. Either way
        # what matters operationally is the same: nothing left → a new batch is needed.
        landed=None
        if len(e):      # through NGS but never rearrayed → no wells to report (0 such cases today)
            vol=pd.to_numeric(e["VOLUME_UL"],errors="coerce"); cc=pd.to_numeric(e["CONCENTRATION_NGUL"],errors="coerce")
            disc=e["PLATE_LOCATION_BOX"].fillna("").astype(str).str.upper().str.contains("DISCARD")
            landed={"wells":int(len(e)),"usable":int(((vol>25)&(cc>5)&~disc).sum()),
                    "gone":int((disc|(vol<=0)).sum())}
        return {"state":"done","age":age,"proc":proc,"stage":stage,"rank":rank,
                "landed":landed,"ngs":ngs,"seq_ok":seq_ok}

    return refill_status


def _consumers(wod):
    """{part: [(product, type_label, workorder_status, experiment), ...]} over the active GG /
    Gibson / PCR workorders in the pull. Module-level so both the Parts tab and the in-flight
    tab read consumption the same way."""
    from dnasc.utils import parse_parts, parse_backbone, extract_pcr_info
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
    return cons


def blocking_refill_progress() -> dict:
    """{product: [{part, stage, age, proc, sequencing, stalled}, ...]} — for every product whose
    build is BLOCKED on a flagged part, where that part's refill batch stands right now.

    The in-flight tab shows this in the Operation column: a request sitting BLOCKED/STALLED with
    an empty operation is not idle if the part it waits on is mid-refill, and that batch is the
    only thing that will move it. Only parts with a batch actually in flight are returned — a
    missing part with nothing running has nothing to report here (the Parts tab says that).

    Reads parts_result.pkl on its own (like the Parts and NGS tabs do) and frees it on the way
    out, so it adds no lasting memory to the render. Never raises: {} on any failure.
    """
    try:
        r = pickle.load(open(_PKL, "rb"))
        # Only the REFILL wells are needed here, and every well refill_status looks at carries a
        # REFILL process id (including the Echo wells it checks for, which it finds by process id).
        # Narrowing first keeps this off the 1.2M-row frame — the whole call runs in ~1s.
        apd = r["all_plate_data"]
        apd = apd[apd["PROCESS_ID"].astype(str).str.contains("REFILL", case=False, na=False)].copy()
        apd["CREATED_AT"] = pd.to_datetime(apd["CREATED_AT"], errors="coerce", utc=True)
        refill_status = _make_refill_status(apd, r.get("ngs_df"), r["generated_at"])
        cons = _consumers(r["wod_df"])
        # Only parts the pull already flagged as needing action — those are the ones a blocked
        # build is actually waiting on. Oligos are excluded from the Parts tab, so skip them here
        # too rather than reporting a refill nobody can see.
        flagged = [p for p in r["parts"]["Part"].astype(str) if not p.startswith("o")]
        out: dict = {}
        for part in flagged:
            blocked = sorted({prod for prod, _t, st, _e in cons.get(part, []) if st == "BLOCKED"})
            if not blocked: continue
            rs = refill_status(part)
            if rs.get("state") != "inflight": continue
            note = {"part": part, "stage": rs["stage"], "age": rs["age"], "proc": rs["proc"],
                    "sequencing": bool(rs.get("sequencing")), "stalled": bool(rs.get("stalled"))}
            for prod in blocked:
                out.setdefault(prod, []).append(note)
        return out
    except Exception:
        import traceback
        traceback.print_exc()
        return {}


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
    wod = r["wod_df"]                                  # active GG/Gibson/PCR workorders
    _blk = r["blk_df"]                                 # blocked workorder queue

    # (blocked WOs are indexed by product LATER — after builds_for/cons exist — so each missing
    #  part can list the workorders it is holding up. There is no standalone blocked-WO section.)

    cons = _consumers(wod)

    tmpl_kids = {}
    for d, row in dmeta.iterrows():
        t = row.get("DPART_TEMPLATE")
        if pd.notna(t): tmpl_kids.setdefault(str(t), []).append(d)
    flagged = set(out["Part"].astype(str))

    refill_status = _make_refill_status(apd, r.get("ngs_df"), now)

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

    # --- dParts are made by PCR, never ordered: "is a PCR queued, and how did the last one go?" ---
    _OPEN_ST = ("RUNNING","WAITING","READY","BLOCKED")
    _pcr = r.get("pcr_df")
    if _pcr is not None and len(_pcr):
        _pcr = _pcr.copy()
        _pcr["CREATED"] = pd.to_datetime(_pcr["CREATED"], errors="coerce", utc=True)
    def pcr_status(part):
        """Newest PCR workorder that PRODUCES `part` (not ones consuming it)."""
        if _pcr is None or not len(_pcr): return None
        p=_pcr[_pcr["NAME"]==str(part)].sort_values("CREATED")
        if p.empty: return None
        openq=p[p["STATUS"].isin(_OPEN_ST)]
        pick=openq.iloc[-1] if not openq.empty else p.iloc[-1]
        age=(now-pick["CREATED"]).days if pd.notna(pick["CREATED"]) else None
        return {"open":not openq.empty,"status":str(pick["STATUS"]),"days":age,
                "date":pick["CREATED"],"n":int(len(p))}

    _pi_cache={}
    def pcr_inputs(part):
        if str(part) in _pi_cache: return _pi_cache[str(part)]
        _pi_cache[str(part)] = _pcr_inputs(part)
        return _pi_cache[str(part)]

    def _pcr_inputs(part):
        """The template + two oligos a PCR for this dPart needs, with what's on hand.

        Uses the same reaction math as the pull for each type: plasmid/dPart wells from the 384
        Echo source (200d freshness), oligo tubes from molarity (730d, (vol-5)*nM/100k).
        """
        if str(part) not in dmeta.index: return []
        row=dmeta.loc[str(part)]
        want=[("template", row.get("DPART_TEMPLATE")),
              ("oligo 1", row.get("OLIGO_1")), ("oligo 2", row.get("OLIGO_2"))]
        outl=[]
        for role,name in want:
            # None/NaN means LIMS has no oligo_N_id on the dPart record at all. That is "we don't
            # know what it needs", NOT "we have zero of it" — showing it as ✗ 0 rxns reads as a
            # stock problem when it is a missing-data problem.
            if pd.isna(name) or str(name) in ("","nan","None"):
                outl.append((role,None,None,"")); continue
            n=str(name)
            if n.startswith("o"):
                w=apd[(apd["STOCK_ID"]==n) & (apd["AVAILABLE"]=="True")
                      & (apd["CREATED_AT"] > (now - pd.Timedelta(days=730)))]
                if len(w):
                    vol=pd.to_numeric(w["VOLUME_UL"],errors="coerce")
                    mol=pd.to_numeric(w["MOLARITY_NM"],errors="coerce")
                    rx=int(((vol-5).clip(lower=0)*mol/100_000).fillna(0).clip(lower=0).sum())
                else: rx=0
            else:
                w=apd[(apd["STOCK_ID"]==n) & (apd["WELL_TYPE"]=="Stock") & _is_echo384(apd)
                      & (apd["AVAILABLE"]=="True")
                      & (apd["CREATED_AT"] > (now - pd.Timedelta(days=_FRESH_DAYS)))]
                rx=int(_rxns(w).sum()) if len(w) else 0
            loc=""
            if len(w):
                _l=w["PLATE_LOCATION_BOX"].dropna().astype(str)
                loc=_l.mode().iloc[0] if not _l.empty else ""
            outl.append((role,n,rx,loc))
        return outl

    # ---- Blocked workorders, indexed by the product they make -------------------------------
    # There is no standalone "Blocked workorders" section any more. A blocked WO is never its own
    # piece of work — it is stuck because an input part does not exist — so it is listed inside
    # the missing part that blocks it. Two missing parts can block the same WO (a Gibson short two
    # inputs), so every total below counts DISTINCT wids.
    _TYB={"gibson_workorder":"Gibson","golden_gate_workorder":"GG","pcr_workorder":"PCR",
          "plasmid_synthesis_workorder":"PlasmidSynth","syn_part_synthesis_workorder":"SynPartSynth"}
    # Open WO statuses per final product, so a blocked WO can be checked for a twin that RUNS.
    _open_by_prod={}
    for _,w in wod.iterrows():
        _open_by_prod.setdefault(str(w["PRODUCT"]),[]).append(str(w["ST"]))

    def _unblocked_twin(prod):
        """How many OTHER open workorders make this same final product and are NOT blocked.

        This is the useful question, and LIMS's own "Duplicate Product" warning does not answer it:
        that warning only compares product NAMES, so it fires whether the twin can run or is stuck
        in the same ditch. Counting non-BLOCKED statuses here also excludes the blocked WO itself.
        """
        return sum(1 for s in _open_by_prod.get(prod,[]) if s!="BLOCKED")

    _blk_by_prod={}
    for _,b in _blk.iterrows():
        _w = b["warnings"]
        _w = list(dict.fromkeys(str(x) for x in _w)) if (_w is not None and len(_w)) else []  # dedupe, keep order
        _sw=[w for w in (list(b["succeeded_wos"]) if b["succeeded_wos"] is not None else []) if w]
        _prod=str(b["product"] or "?")
        # Three DIFFERENT claims, never conflated:
        #   succeeded_wos      → a WO for this product actually SUCCEEDED. The product exists.
        #   unblocked twin     → another open WO for the same final product is running/waiting, so
        #                        this blocked one is redundant work — the product still gets made.
        #   "Duplicate Product" → LIMS saw two WOs with the same product name. Says nothing about
        #                        whether either can run; only used when there is no unblocked twin.
        _blk_by_prod.setdefault(_prod,[]).append({
            "wid":str(b["wid"]), "type":_TYB.get(b["type"],str(b["type"])),
            "product":_prod, "created":str(b["created"]),
            "exp":str(b["experiment"] or "— no experiment —"),
            "dup":any("Duplicate Product" in x for x in _w), "succ":_sw,
            "twin":_unblocked_twin(_prod),
            "warns":[x for x in _w if "Duplicate Product" not in x]})
    _blk_all={w["wid"]:w for _lst in _blk_by_prod.values() for w in _lst}

    def blocked_wos_for(part):
        """The BLOCKED workorders this missing part is holding up (its direct consumers)."""
        seen={}
        for prod,_t,st,_e in builds_for(part):
            if st!="BLOCKED": continue
            for w in _blk_by_prod.get(prod,[]): seen[w["wid"]]=w
        return sorted(seen.values(), key=lambda w:(w["product"],w["wid"]))

    def _wo_note(w):
        """Why this blocked WO is here, and whether it needs its own action."""
        if w["succ"]:
            return ('<span style="color:#15803d;font-weight:600">✓ already produced by wo '
                    f'{html.escape(str(w["succ"][0])[:8])} → safe to cancel</span>')
        if w["twin"]:
            return ('<span style="color:#b45309;font-weight:600">other unblocked WO for the same '
                    'final product</span><span style="color:#6b7280"> — that one can run, so this '
                    'one is redundant → cancel it</span>')
        if w["dup"]:
            return ('<span style="color:#b45309;font-weight:600">second WO for the same final '
                    'product</span><span style="color:#6b7280"> — but it is blocked too, so '
                    'neither can run yet</span>')
        return '<span style="color:#6b7280">waiting on this part — no action of its own</span>'

    def _wo_tbl(wos):
        """Static table of blocked WOs (no nested toggles — it lives inside an open panel)."""
        hdr=('<tr><td><b>WO</b></td><td><b>Type</b></td><td><b>Making</b></td>'
             '<td><b>Created</b></td><td><b>Status / note</b></td></tr>')
        rws=""
        for w in wos:
            note=_wo_note(w)
            if w["warns"]:
                # trim absurd float precision (Tm 51.7194… → 51.7)
                _wt=re.sub(r"(\d+\.\d)\d+", r"\1", "; ".join(w["warns"]))
                note+=(f'<div style="color:#b45309;font-size:10px;margin-top:2px">'
                       f'{html.escape(_wt)}</div>')
            rws+=(f'<tr><td style="font-family:monospace">{html.escape(w["wid"][:8])}</td>'
                  f'<td>{html.escape(w["type"])}</td>'
                  f'<td style="font-family:monospace;font-weight:700">{html.escape(w["product"])}</td>'
                  f'<td style="white-space:nowrap">{html.escape(w["created"])}</td>'
                  f'<td style="font-size:11px">{note}</td></tr>')
        return f'<table class="d-tbl"><tbody>{hdr}{rws}</tbody></table>'

    def typ(pt): return "Plasmid" if pt.startswith("pAI") else ("Oligo" if pt.startswith("o") else "dPart")
    ST_COLOR = {"RUNNING":"#1d4ed8","WAITING":"#92400e","READY":"#15803d","BLOCKED":"#b91c1c"}
    def st_pill(s):
        cc = ST_COLOR.get(s, "#6b7280")
        return f'<span style="font-size:9px;font-weight:700;color:{cc}">{s}</span>'
    def _no_lims_wells(part):
        """Not one well on record for this part — LIMS has never held any of it."""
        return not len(apd[apd["STOCK_ID"].astype(str)==str(part)])

    def act_badge(a, age=None, not_in_lims=False, muted=False):
        # "Reorder" is wrong for a part LIMS has never held: there is nothing to RE-order, and the
        # row's real state is that it isn't there at all. Say that instead of naming an action.
        if not_in_lims: lbl,bg,fg = "Not in LIMS","#f3f4f6","#b91c1c"
        elif a=="Make by PCR": lbl,bg,fg = "Add PCR WO","#ecfeff","#0e7490"
        elif str(a).startswith("Mark"): lbl,bg,fg = "Mark available","#eff6ff","#1d4ed8"
        elif a=="Refill": lbl,bg,fg = "Refill","#fef3c7","#92400e"
        elif a=="Transform": lbl,bg,fg = "Transform","#fff7ed","#c2410c"
        elif a=="True":   lbl,bg,fg = "Reorder","#fff1f5","#be185d"
        else: lbl,bg,fg = str(a),"#f3f4f6","#6b7280"
        # Covered for its queued need: the row is only here because of worst-case exposure, so
        # the action is what you WOULD do, not what you must do today. Grey it out rather than
        # letting "Add PCR WO" shout next to 11 rxns on hand for a need of 2.
        if muted and not not_in_lims: bg,fg = "#f3f4f6","#9ca3af"
        return f'<span style="background:{bg};color:{fg};font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;white-space:nowrap">{lbl}</span>'

    def coord384(wn):
        if pd.isna(wn): return ""
        wn=int(wn); return f"{chr(65+wn%16)}{wn//16+1}"

    def coord96(wn):
        if pd.isna(wn): return ""
        wn=int(wn); return f"{chr(65+wn%8)}{wn//8+1}"

    def coord_wc(wn, nwells):
        # The '384 Echo Source Plate' labware is overloaded in LIMS: at least one
        # plate carries it while being physically 96-well. Pick the layout from the
        # plate's real well_count, never from the labware name.
        return coord96(wn) if pd.to_numeric(nwells, errors="coerce")==96 else coord384(wn)

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
        # Exclude error plates (a '384 Echo Source Plate' that is not physically
        # 384-well) — they should be found & discarded, never offered as transform
        # DNA. Genuine 96-well plates (other labware) still qualify.
        _errp = ((apd["LABWARE"]=="384 Echo Source Plate")
                 & (pd.to_numeric(apd["PLATE_NUMBER_OF_WELLS"],errors="coerce")!=384))
        s = apd[(apd["STOCK_ID"]==part) & (apd["WELL_TYPE"]=="Stock") & ~_errp
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
                 & (apd["WELL_TYPE"]=="Stock") & _is_echo384(apd)
                 & ~apd["PLATE_LOCATION_BOX"].fillna("").astype(str).str.upper().str.contains("DISCARD")]
        rows=[]
        for _,x in av.iterrows():
            age=(now-x["CREATED_AT"]).days if pd.notna(x["CREATED_AT"]) else None
            rows.append((x["PLATE_ID"], str(x["PLATE_LOCATION_BOX"]), coord_wc(x["WELL_NUMBER"], x["PLATE_NUMBER_OF_WELLS"]),
                         x["WELL_ID"], x["VOLUME_UL"], x["CONCENTRATION_NGUL"], age))
        rows.sort(key=lambda rr:(rr[6] is None, rr[6] if rr[6] is not None else 0))
        return rows

    _ma_cache = {}
    def make_avail_wells(part):
        # memoized: called per row, again by flip_gain, and again by the Make Available section —
        # each call is a full scan of the 1.1M-row well frame.
        if part in _ma_cache: return _ma_cache[part]
        _ma_cache[part] = _make_avail_wells(part)
        return _ma_cache[part]

    def _make_avail_wells(part):
        win = 730 if part.startswith("o") else 200
        s = apd[(apd["STOCK_ID"]==part) & (apd["WELL_TYPE"]=="Stock") & _is_echo384(apd)
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
            rows.append((x["PLATE_ID"], coord_wc(x["WELL_NUMBER"], x["PLATE_NUMBER_OF_WELLS"]), x["WELL_ID"], loc or "(no loc)", v, cc, age))
        rows.sort(key=lambda rr:-(pd.to_numeric(rr[5],errors="coerce") or 0))
        return rows

    def newest_age(part):
        ages=[a for *_,a in avail_wells(part) if a is not None]
        return min(ages) if ages else None

    # ---- reactions math: what flipping the make-available wells ON is actually worth ----
    # Same formula and same well set the pull uses for "Reactions Available" (verified to
    # reproduce it exactly for every part in the pull), so the two numbers are comparable:
    # rxns = (volume - dead volume) * concentration / (weight * sequence length * 6e9).
    _WEIGHT, _DEAD_VOL = 1e-12, 20
    _FRESH_DAYS = 200

    def _rxns(sub):
        sl = pd.to_numeric(sub["DPART_SEQUENCE_LENGTH"], errors="coerce").where(
             sub["STOCK_ID"].astype(str).str.startswith("d"),
             pd.to_numeric(sub["SEQUENCE_LENGTH"], errors="coerce"))
        v = pd.to_numeric(sub["VOLUME_UL"], errors="coerce")
        c = pd.to_numeric(sub["CONCENTRATION_NGUL"], errors="coerce")
        return (((v - _DEAD_VOL) * c) / (_WEIGHT * sl * 6e9)).clip(lower=0).fillna(0)

    _flip_cache = {}
    def flip_gain(part, have):
        """(n_wells, rxns gained, total after flipping) for the make-available wells.

        Anchored on the pull's `have` and adding the delta, so the panel can never disagree
        with the "on hand" number it prints right above it.
        """
        part = str(part)
        key = (part, int(have))
        if key in _flip_cache: return _flip_cache[key]
        ma = make_avail_wells(part)
        if not ma:
            _flip_cache[key] = (0, 0, int(have)); return _flip_cache[key]
        ids = {int(rr[2]) for rr in ma}
        base = apd[(apd["STOCK_ID"]==part) & (apd["WELL_TYPE"]=="Stock") & _is_echo384(apd)
                   & (apd["AVAILABLE"]=="True")
                   & (apd["CREATED_AT"] > (now - pd.Timedelta(days=_FRESH_DAYS)))]
        b = float(_rxns(base).sum())
        f = float(_rxns(apd[apd["WELL_ID"].isin(ids)]).sum())
        gain = int(b + f) - int(b)
        _flip_cache[key] = (len(ma), gain, int(have) + gain)
        return _flip_cache[key]

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
        part=str(x["Part"]); act=disp_act(x)
        have=int(x["Reactions Available"]); need=int(x["Reactions Required"])
        win = 730 if part.startswith("o") else 200
        nage = newest_age(part)
        is_ctrl = part in ctrl_related
        target = _target(x)      # buffer = min 10, else 2× (buffer == need over 10); 96 for controls
        n_flip, flip_rxns, after_flip = flip_gain(part, have)

        if act=="Mark available" and not n_flip:
            # Pull flagged wells but none pass the 4B-fridge / volume / freshness rules used here.
            situation, guidance, tone = "Below target", "", "#92400e"
        elif act=="Mark available":
            situation = f"Below target — but {n_flip} seq-confirmed well{'s' if n_flip!=1 else ''} already in the fridge"
            guidance  = (f"Flip {n_flip} well{'s' if n_flip!=1 else ''} ON in LIMS (well IDs below) "
                         f"→ {after_flip}/{target} rxns. No streak or batch needed.")
            tone      = "#1d4ed8"
        elif act=="Refill":
            gp=str(x.get("Glycerol Plate","") or ""); gw=str(x.get("Glycerol Well","") or "")
            gl=str(x.get("Glycerol Location","") or ""); cs=str(x.get("Cell Strain","") or "")
            src=" · ".join(b for b in [f"plate {gp}" if gp not in("","nan") else "", f"well {gw}" if gw not in("","nan") else "", f"({gl})" if gl not in("","nan","None") else "", cs if cs not in("","nan") else ""] if b)
            situation, guidance, tone = "Below target — top up", f"Streak from glycerol {src}" if src else "No glycerol source recorded", "#92400e"
        elif act=="Transform":
            situation, guidance, tone = "No glycerol stock — transform fresh", "Transform the plasmid DNA below → overnight → miniprep → re-stock", "#c2410c"
        elif act=="Make by PCR":
            _pi=[i for i in pcr_inputs(part) if i[1]]
            _short=[i[1] for i in _pi if not i[2]]
            _unknown=[i[0] for i in pcr_inputs(part) if not i[1]]
            situation = "Not enough dPart stock — PCR it"
            guidance  = ("Queue a PCR workorder for this dPart"
                         + (f" · missing inputs: {', '.join(_short)} — sort those first" if _short
                            else " · all known inputs on hand")
                         + (f" · {', '.join(_unknown)} not recorded in LIMS" if _unknown else ""))
            tone      = "#0e7490"
        elif act=="True" and x.get("_blockedpart") and _no_lims_wells(part):
            situation, guidance, tone = "Not in LIMS — no wells on record", "", "#b91c1c"
        elif act=="True":
            situation, guidance, tone = "No DNA on hand", "Reorder / synthesize", "#be185d"
        else:
            situation, guidance, tone = "", "", "#6b7280"

        blocks=[]
        pct = max(3, min(100, int(round(100*have/target)))) if target else 0
        _bf=_buffer(need)
        _rate=(f" @ {int(round(100*PipelineConfig.REFILL_BUFFER_FRAC))}%"
               if _bf > PipelineConfig.REFILL_BUFFER_MIN else " floor")
        note = f"{need} needed + {_bf} buffer{_rate}" if not is_ctrl else "control buffer"
        # What the shortfall really is: buffer-only shortfalls are not the same urgency as a
        # part live builds are waiting on.
        if not is_ctrl and need == 0:
            note += " · nothing live needs it — shortfall is buffer only"
        # The post-flip number, on the same line as "on hand", so the free stock can't be missed.
        flip_line = ""
        if n_flip:
            fpct = max(3, min(100, int(round(100*after_flip/target)))) if target else 0
            flip_line = (f'<div style="font-size:11px;color:#1d4ed8;margin-top:2px">&uarr; <b>{after_flip}</b> '
                         f'after flipping {n_flip} well{"s" if n_flip!=1 else ""} ON '
                         f'(+{flip_rxns} rxns) — {"clears" if after_flip>=target else "still short of"} '
                         f'the target of {target}</div>')
        # bar = what's available now (tone) + what a flip would add (light blue), 2px gap between
        seg = f'<span class="d-seg" style="width:{pct}%;background:{tone}"></span>'
        if n_flip and fpct > pct:
            seg += (f'<span class="d-seg" style="width:{fpct-pct}%;background:#93c5fd;margin-left:2px" '
                    f'title="reachable by flipping {n_flip} well(s) ON"></span>')
        _ex=exposure(part); _tmax=_target_max(x)
        _rng=(f'{target}&ndash;{_tmax}' if _tmax>target else f'{target}')
        # Capacity-to-fail instead of a guessed retry rate: how many of the already-drawn
        # builds could come back before this part is short.
        _spare=have-int(x["Reactions Required"])
        if _ex["drawn"] and _spare>0:
            _pct=int(round(100*min(_spare,_ex["drawn"])/_ex["drawn"]))
            _hd=(f'<div style="font-size:11px;color:#15803d;margin-top:3px">Headroom &mdash; '
                 f'<b>{_spare}</b> rxns above the queued need: absorbs <b>{min(_spare,_ex["drawn"])}</b> '
                 f'of the {_ex["drawn"]} running build{"s" if _ex["drawn"]!=1 else ""} coming back '
                 f'({_pct}%)</div>')
        elif _ex["drawn"]:
            _hd=(f'<div style="font-size:11px;color:#be185d;margin-top:3px">No headroom &mdash; '
                 f'nothing spare above the queued need, and {_ex["drawn"]} running '
                 f'build{"s" if _ex["drawn"]!=1 else ""} could still come back</div>')
        else:
            _hd=''
        _brk=(f'<div style="font-size:10px;color:#6b7280;margin-top:2px">'
              f'{_ex["queued"]} queued now &middot; {_ex["drawn"]} already drew material &middot; '
              f'target {target} if none retry, {_tmax} if all do</div>') if _ex["drawn"] else ''
        blocks.append(_block("Status",
            f'<div class="d-stat"><span class="d-have">{have}</span><span class="d-of"> on hand · target {_rng}</span>'
            f'<span class="d-note">({note})</span></div>'
            f'<div class="d-barwrap">{seg}</div>{flip_line}{_brk}{_hd}'
            f'<div class="d-sit" style="color:{tone}">{html.escape(situation)}</div>'))
        wl=avail_wells(part)
        if wl:
            hdr='<tr><td><b>Plate</b></td><td><b>Well</b></td><td><b>Well ID</b></td><td><b>Location</b></td><td><b>Vol</b></td><td><b>Conc</b></td><td><b>Age</b></td></tr>'
            rws="".join(f'<tr><td>plate {p}</td><td>{co or "?"}</td><td style="font-family:monospace">{wid}</td><td>{html.escape(loc)}</td><td>{_fmt(v)}µL</td><td>{_fmt(cc)} ng/µL</td><td>{a if a is not None else "?"}d</td></tr>' for p,loc,co,wid,v,cc,a in wl)
            blocks.append(_block("On hand · 4B fridge (4°C)",
                f'<table class="d-tbl"><tbody>{hdr}{rws}</tbody></table>'))
        else:
            blocks.append(_block("On hand · 4B fridge (4°C)", '<span style="font-size:11px;color:#9ca3af">none in 4B fridge</span>'))
        gs=glycerol_streak(part)
        if gs:
            hdr='<tr><td><b>pAI</b></td><td><b>Antibiotic</b></td><td><b>Strain</b></td><td><b>Plate</b></td><td><b>Coord</b></td><td><b>Location</b></td><td><b>Well ID</b></td></tr>'
            rws="".join(f'<tr><td style="font-family:monospace;font-weight:700">{part}</td><td>{html.escape(ab)}</td><td>{html.escape(strain or "?")}</td><td>plate {p}</td><td style="font-family:monospace">{co or "?"}</td><td>{html.escape(loc)}</td><td style="font-family:monospace">{wid}</td></tr>' for p,co,loc,ab,strain,wid in gs)
            blocks.append(_block(f"Streak from · glycerol ({len(gs)})",
                f'<table class="d-tbl"><tbody>{hdr}{rws}</tbody></table>'))
        # Make-available (flip ON) — seq-confirmed wells eligible to be re-enabled in LIMS.
        ma=make_avail_wells(part)
        if ma:
            hdr='<tr><td><b>Plate</b></td><td><b>Well</b></td><td><b>Well ID</b></td><td><b>Location</b></td><td><b>Vol</b></td><td><b>Conc</b></td><td><b>Age</b></td><td><b>Seq</b></td></tr>'
            rws="".join(f'<tr><td>plate {p}</td><td>{co or "?"}</td><td style="font-family:monospace">{wid}</td><td>{html.escape(loc)}</td><td>{_fmt(v)}µL</td><td>{_fmt(cc)} ng/µL</td><td>{a if a is not None else "?"}d</td><td style="color:#15803d">✓</td></tr>' for p,co,wid,loc,v,cc,a in ma)
            ids=",".join(f"well{int(r[2])}" for r in ma)
            cb=(f'<div style="display:flex;gap:6px;margin-top:6px"><textarea readonly onclick="this.select()" '
                f'style="flex:1;height:38px;font-family:monospace;font-size:10px;border:1px solid #bfdbfe;border-radius:4px;'
                f'padding:4px 6px;background:#f8fafc;resize:vertical">{ids}</textarea>'
                f'<button onclick="var t=this.previousElementSibling;t.select();navigator.clipboard.writeText(t.value)" '
                f'style="font-size:10px;font-weight:600;padding:4px 10px;border:1px solid #93c5fd;border-radius:4px;'
                f'background:#dbeafe;color:#1d4ed8;cursor:pointer;white-space:nowrap">Copy</button></div>')
            blocks.append(_block(f"Make available &rarr; flip ON ({len(ma)})",
                f'<div style="font-size:10px;color:#6b7280;margin-bottom:3px">seq-confirmed · &gt;25µL · &gt;5 ng/µL · fresh · not yet available</div>'
                f'<table class="d-tbl"><tbody>{hdr}{rws}</tbody></table>{cb}'))
        if act=="Make by PCR":
            pi=pcr_inputs(part)
            if pi:
                hdr='<tr><td><b>Role</b></td><td><b>Part</b></td><td><b>On hand</b></td><td><b>Location</b></td></tr>'
                rws=""
                for role,name,rx,loc in pi:
                    if not name:   # LIMS has no oligo/template id on the dPart record
                        rws+=(f'<tr><td style="color:#6b7280">{role}</td>'
                              f'<td colspan="3" style="color:#b45309">not recorded in LIMS '
                              f'<span style="color:#9ca3af">— can&#39;t tell what this PCR needs</span></td></tr>')
                        continue
                    mark = ('<span style="color:#15803d">✓</span>' if rx>0
                            else '<span style="color:#be185d">✗</span>')
                    rws+=(f'<tr><td style="color:#6b7280">{role}</td>'
                          f'<td style="font-family:monospace;font-weight:700">{html.escape(str(name))}</td>'
                          f'<td>{mark} {rx} rxns</td><td>{html.escape(loc) or "—"}</td></tr>')
                blocks.append(_block("PCR inputs",
                    '<div style="font-size:10px;color:#6b7280;margin-bottom:3px">template + both oligos must be '
                    'on hand to run the PCR</div>'
                    f'<table class="d-tbl"><tbody>{hdr}{rws}</tbody></table>'))
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
        if act in ("Refill","Transform","Mark available"):
            rs = refill_status(part)
            if rs["state"]=="inflight":
                head=(f'<span style="color:#be185d;font-weight:700">⚠ Batch stalled</span>'
                      if rs.get("stalled") else
                      f'<span style="color:#15803d;font-weight:700">⟳ Batch in progress</span>')
                prog=(f'{head} — furthest stage <b>{html.escape(rs["stage"])}</b>, last activity '
                      f'{rs["age"]}d ago · {html.escape(rs["proc"])}')
                if rs.get("sequencing"):
                    prog+=('<div style="color:#6b7280;margin-top:2px">its <b>NGS job is still open</b> — '
                           'the result lands when that job closes</div>')
                else:
                    prog+=('<div style="color:#6b7280;margin-top:2px">no NGS job on it yet'
                           + (f' · a refill normally finishes in {_REFILL_TYPICAL_DAYS}d — chase it'
                              if rs.get("stalled") else "") + '</div>')
            elif rs["state"]=="done":
                ld=rs.get("landed"); ng=rs.get("ngs")
                # It rearrayed into the Echo plate, which is the finish line — so it is over no
                # matter how long ago that was. Report only what is left of it today; the snapshot
                # cannot tell a batch that was used up from one that never worked, so it says neither.
                if ng:
                    _oc=ng["outcome"]
                    _when=(f', closed {ng["closed_days"]}d ago' if ng["closed_days"] is not None else "")
                    if _oc=="SUCCEEDED":
                        prog=(f'<span style="color:#9ca3af">Last refill</span> — NGS job '
                              f'<b style="color:#15803d">SUCCEEDED</b>{_when} · {html.escape(rs["proc"])}')
                    elif rs.get("seq_ok"):
                        # Wells are seq-confirmed: the read was fine and the batch simply came up
                        # short. Do not print the workorder status at all here. Showing "FAILED"
                        # and then annotating it away just makes the reader cancel out a word that
                        # should not have been in the sentence — and once the sequence is confirmed,
                        # the status changes nothing about what to do. It stays in the tooltip so
                        # the row is still reconcilable against LIMS.
                        _fin=(f' · finished {ng["closed_days"]}d ago'
                              if ng["closed_days"] is not None else "")
                        prog=(f'<span title="LIMS NGS workorder status: {html.escape(_oc)}'
                              f'{html.escape(_when)} — the wells are seq-confirmed, so that status '
                              f'is not a sequencing verdict" style="color:#9ca3af">Last refill</span> — '
                              f'<b style="color:#15803d">sequence confirmed</b>'
                              f'{_fin} · {html.escape(rs["proc"])}')
                    else:
                        prog=(f'<span style="color:#9ca3af">Last refill</span> — NGS job '
                              f'<b style="color:#be185d">{_oc}</b>{_when} · nothing seq-confirmed came '
                              f'out of it · {html.escape(rs["proc"])}')
                else:
                    prog=(f'<span style="color:#9ca3af">Last refill</span> — finished at '
                          f'<b>{html.escape(rs["stage"])}</b>, {rs["age"]}d ago · {html.escape(rs["proc"])}')
                if ld:
                    prog+=(f'<div style="color:#6b7280;margin-top:2px">put {ld["wells"]} well'
                           f'{"s" if ld["wells"]!=1 else ""} in the 384 Echo source · '
                           f'<b>{ld["usable"]}</b> still usable today'
                           + (f' ({ld["gone"]} empty or discarded)' if ld["gone"] else "") + '</div>')
                else:
                    prog+=('<div style="color:#6b7280;margin-top:2px">through NGS but nothing in the '
                           '384 Echo source plate</div>')
                if not (n_flip and after_flip>=target):
                    # Say what's short, so the line reads as a yield shortfall and never as a
                    # consequence of the NGS status printed above it.
                    prog+=(f'<div style="color:#be185d;margin-top:2px">→ needs a new batch — '
                           f'{have} of {target} rxns</div>')
            else:
                prog='<span style="color:#be185d;font-weight:700">⚠ Needs batching</span> <span style="color:#9ca3af">— no refill on record</span>'
            if n_flip and after_flip>=target:
                prog=(f'<span style="color:#1d4ed8;font-weight:700">No batch needed</span> '
                      f'<span style="color:#6b7280">— flipping {n_flip} well{"s" if n_flip!=1 else ""} ON '
                      f'reaches {after_flip}/{target}</span><div style="margin-top:3px">{prog}</div>')
            blocks.append(_block("In progress?", f'<div style="font-size:11px">{prog}</div>'))
        elif act=="Make by PCR":
            p=pcr_status(part)
            if p is None:
                prog='<span style="color:#9ca3af">No PCR workorder on record for this dPart</span>'
            elif p["open"]:
                _c,_ic = (("#b91c1c","⚠") if p["status"]=="BLOCKED" else ("#15803d","⟳"))
                prog=(f'<span style="color:{_c};font-weight:700">{_ic} PCR {html.escape(p["status"])}</span> — '
                      f'queued {p["days"]}d ago ({p["date"]:%Y-%m-%d})')
                if p["status"]=="BLOCKED":
                    _sh=[i[1] for i in pcr_inputs(part) if i[1] and not i[2]]
                    prog+=('<div style="color:#b91c1c;margin-top:2px">missing '
                           f'{html.escape(", ".join(_sh))} — sort that first</div>' if _sh else
                           '<div style="color:#6b7280;margin-top:2px">inputs look present — check the WO warnings</div>')
            else:
                _c="#be185d" if p["status"] in ("FAILED","CANCELED") else "#9ca3af"
                prog=(f'<span style="color:{_c};font-weight:700">Last PCR {html.escape(p["status"])}</span> '
                      f'{p["days"]}d ago ({p["date"]:%Y-%m-%d}) · {p["n"]} PCR workorder'
                      f'{"s" if p["n"]!=1 else ""} on record'
                      '<div style="color:#be185d;margin-top:2px">→ no PCR queued now — needs one</div>')
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
        if x.get("_blockedpart"):
            ws=blocked_wos_for(part)
            if ws:
                blocks.append(_block(f"Blocking {len(ws)} workorder{'s' if len(ws)!=1 else ''}",
                    '<div style="font-size:10px;color:#6b7280;margin-bottom:3px">stuck until this part '
                    'exists — they unblock themselves once it does, so they need no action of their own '
                    'unless flagged cancelable</div>' + _wo_tbl(ws)))
        bb=builds_for(part)
        if bb:
            exps={}
            for p,t,s,e in bb: exps.setdefault(e or "—",[]).append((p,t,s))
            _todo=sum(1 for _p,_t,st,_e in bb if st in _NOT_DRAWN)
            _drawn=len(bb)-_todo
            body=(f'<div class="d-sub">{len(bb)} build{"s" if len(bb)!=1 else ""} across '
                  f'{len(exps)} experiment{"s" if len(exps)!=1 else ""} &middot; '
                  f'<b>{_todo}</b> still to run'
                  + (f' &middot; {_drawn} already drew material (not counted as need)' if _drawn else '')
                  + '</div>')
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

    def batch_state(part, need, is_ctrl):
        """The batch half of the 'Batch / order' cell — never claims a dead batch is running.

        A shortfall that is pure buffer (nothing live needs the part) is called that, so it
        doesn't read as urgent as a part that real builds are waiting on.
        """
        rs = refill_status(part)
        urgent = bool(need > 0 or is_ctrl)
        col = "#be185d" if urgent else "#6b7280"
        pre = "" if urgent else '<span style="color:#9ca3af">buffer only · </span>'
        if rs["state"] == "inflight":
            if rs.get("sequencing"):
                return (f'<span style="color:#15803d;font-weight:700">⟳ batch in progress</span>'
                        f'<span style="color:#9ca3af"> · NGS job open, {rs["age"]}d</span>')
            if rs.get("stalled"):
                return (f'{pre}<span style="color:{col};font-weight:700">⚠ batch stalled</span>'
                        f'<span style="color:#9ca3af"> · stuck at {html.escape(rs["stage"])}, quiet {rs["age"]}d</span>')
            return (f'<span style="color:#15803d;font-weight:700">⟳ batch in progress</span>'
                    f'<span style="color:#9ca3af"> · {html.escape(rs["stage"])}, {rs["age"]}d</span>')
        if rs["state"] == "done":
            ld = rs.get("landed"); ng = rs.get("ngs")
            # NGS FAILED/CANCELED is a real failure — the job itself says so, unlike guessing from
            # well volume. Worth naming, because the fix differs: re-sequence or re-streak.
            if ng and ng["outcome"] in ("FAILED","CANCELED"):
                # Seq-confirmed wells → the read was fine, the batch just came up short. Naming NGS
                # here blamed the sequencing for what is a yield problem.
                _why=(f' · last batch came up short, {rs["age"]}d ago' if rs.get("seq_ok")
                      else f' · NGS {ng["outcome"].lower()}, nothing confirmed, {rs["age"]}d ago')
                return (f'{pre}<span style="color:{col};font-weight:700">⚠ needs batch</span>'
                        f'<span style="color:#9ca3af">{_why}</span>')
            tail = (f' · last batch {rs["age"]}d ago, nothing left from it'
                    if ld and ld["usable"] == 0 else f' · last batch {rs["age"]}d ago')
            return (f'{pre}<span style="color:{col};font-weight:700">⚠ needs batch</span>'
                    f'<span style="color:#9ca3af">{tail}</span>')
        return f'{pre}<span style="color:{col};font-weight:700">⚠ needs batch</span>'

    def _covered(x):
        """Every build actually WAITING on this part can run.

        This deliberately ignores the buffer. Urgency is about whether work is blocked, not
        whether the spare is topped up — testing against the full target (need + buffer) made
        d8260 read "needs PCR" with 4 rxns on hand and only 2 builds waiting, while d8278 with
        7 on hand read "covered". Same situation, opposite wording.
        """
        if x.get("_blockedpart") or bool(x.get("_isbuild")): return False
        return int(x["Reactions Available"]) >= int(x["Reactions Required"])

    def batch_cell(x):
        part=str(x["Part"]); act=x["Action Suggested"]
        if _covered(x):
            # Never print "needs PCR"/"needs batch" on a part whose waiting builds can all run.
            # One vocabulary for the calm states, so two rows in the same position cannot read
            # as though one were urgent: "buffer only" when nothing is waiting at all, "queue
            # covered" when something is waiting and the stock covers it. Any remaining gap is
            # buffer and is stated as such, never as an alarm.
            _ex=exposure(part); _nd=int(x["Reactions Required"]); _hv=int(x["Reactions Available"])
            _short=max(0, _target(x)-_hv)
            if part in ctrl_related:
                # Controls carry the pull's own reaction figure, not a count of waiting builds
                # (they are stocked regardless of demand), so "N waiting builds can run" would be
                # invented — pAI-13500 has zero open workorders consuming it.
                head=('<span style="color:#6b7280;font-weight:600">control stock</span>'
                      f'<span style="color:#9ca3af"> · held at {_target(x)} rxns regardless of '
                      f'live demand</span>')
            elif _nd>0:
                head=('<span style="color:#15803d;font-weight:600">queue covered</span>'
                      f'<span style="color:#9ca3af"> · all {_nd} waiting build'
                      f'{"s" if _nd!=1 else ""} can run</span>')
            else:
                head=('<span style="color:#6b7280;font-weight:600">buffer only</span>'
                      '<span style="color:#9ca3af"> · nothing is waiting on it</span>')
            tail=''
            if _short:
                tail=(f'<div style="color:#9ca3af;font-size:10px">{_short} below the buffer '
                      f'target of {_target(x)} — spare stock, not blocked work</div>')
            elif _ex["drawn"]:
                tail=(f'<div style="color:#9ca3af;font-size:10px">exposed only if the '
                      f'{_ex["drawn"]} running build{"s" if _ex["drawn"]!=1 else ""} '
                      f'{"retry" if _ex["drawn"]!=1 else "retries"}</div>')
            return head+tail
        if x.get("_blockedpart"):
            # Lead with the cost (how many WOs are stuck), because that is what makes one missing
            # part more urgent than another — the action itself is already in the Action column.
            ws=blocked_wos_for(part)
            nd=sum(1 for w in ws if w["twin"] and not w["succ"])
            ns=sum(1 for w in ws if w["succ"])
            cell=(f'<span style="color:#b91c1c;font-weight:700">⚠ blocking {len(ws)} '
                  f'WO{"s" if len(ws)!=1 else ""}</span>'
                  f'<span style="color:#9ca3af"> · nothing queued to make it</span>')
            if ns:
                cell+=(f'<div style="color:#15803d;font-size:10px">{ns} already produced '
                       f'elsewhere → cancelable</div>')
            if nd:
                cell+=(f'<div style="color:#b45309;font-size:10px">{nd} of them have another '
                       f'unblocked WO for the same final product</div>')
            return cell
        # A dPart with an OPEN pcr workorder: the pull writes "pcr_workorder is <STATUS>" as the
        # action, which left this column empty. Say how long it has been queued, and for BLOCKED
        # ones name the missing inputs — that is usually why it is stuck.
        if "pcr_workorder is" in str(act) and part.startswith("d"):
            p=pcr_status(part)
            if p and p["open"]:
                # BLOCKED is not progress — it is stuck, so it reads red like every other stall.
                _blk_pcr = p["status"]=="BLOCKED"
                _c, _ic = ("#b91c1c","⚠") if _blk_pcr else ("#15803d","⟳")
                cell=(f'<span style="color:{_c};font-weight:700">{_ic} PCR {html.escape(p["status"].lower())}</span>'
                      f'<span style="color:#9ca3af"> · queued {p["days"]}d ago</span>')
                if _blk_pcr:
                    short=[i[1] for i in pcr_inputs(part) if i[1] and not i[2]]
                    if short:
                        cell+=(f'<div style="color:#b91c1c;font-size:10px">missing '
                               f'{html.escape(", ".join(short))}</div>')
                return cell
        if act=="True" and part.startswith("d"):
            # PCR is the only maker for a dPart, so the question is PCR state, not order state.
            p=pcr_status(part)
            if p and p["open"]:
                return (f'<span style="color:#15803d;font-weight:700">⟳ PCR {html.escape(p["status"].lower())}</span>'
                        f'<span style="color:#9ca3af"> · queued {p["days"]}d ago</span>')
            if p:
                return (f'<span style="color:#be185d;font-weight:700">⚠ needs PCR</span>'
                        f'<span style="color:#9ca3af"> · last PCR {html.escape(p["status"].lower())} '
                        f'{p["days"]}d ago</span>')
            return ('<span style="color:#be185d;font-weight:700">⚠ needs PCR</span>'
                    '<span style="color:#9ca3af"> · none on record</span>')
        if act=="True":
            o=order_status(part)
            if o and o["active"]: return f'<span style="color:#15803d">on order · {html.escape(str(o["vendor"]))}</span>'
            return '<span style="color:#be185d;font-weight:700">⚠ needs order</span>'
        # "Mark ..." rows come from the pull already knowing wells can be flipped ON — they get
        # the same treatment as Refill/Transform, so the cell still says what to do. Before, they
        # fell straight through to "—" and the flip wells only appeared inside the open panel.
        if act not in ("Refill","Transform") and not str(act).startswith("Mark"):
            return '<span style="color:#9ca3af">—</span>'
        have=int(x["Reactions Available"]); need=int(x["Reactions Required"])
        is_ctrl=str(part) in ctrl_related
        tgt=_target(x)
        n,gain,after = flip_gain(part, have)
        # Wells already sitting in the fridge, seq-confirmed, that only need the LIMS available
        # flag flipped ON — free stock. Say so BEFORE asking for a batch: this is why a part
        # could read "needs batch" while its own panel listed wells ready to flip.
        if n:
            flip=(f'<span style="color:#1d4ed8;font-weight:700">&uarr; flip {n} well{"s" if n!=1 else ""} ON</span>'
                  f'<span style="color:#9ca3af"> &rarr; {after}/{tgt} rxns</span>')
            if after >= tgt:
                return flip + '<div style="color:#15803d;font-size:10px">covers target — no batch needed</div>'
            return flip + f'<div style="font-size:10px;margin-top:1px">{batch_state(part,need,is_ctrl)}</div>'
        return batch_state(part, need, is_ctrl)

    def repeat_badge(demand):
        d=int(demand) if pd.notna(demand) else 0
        if d>=20:   lvl,bg,fg="HIGH","#fee2e2","#b91c1c"
        elif d>=10: lvl,bg,fg="MED","#fef3c7","#92400e"
        else:       lvl,bg,fg="LOW","#f0fdf4","#15803d"
        return f'<span title="feeds {d} downstream builds" style="background:{bg};color:{fg};font-size:8px;font-weight:700;padding:1px 5px;border-radius:8px;margin-left:5px">{lvl}</span>'

    def stall_rank(x):
        """0 = waiting builds CANNOT run (no stock), 1 = only some can, 2 = queue is covered.

        The LOW/MED/HIGH badge bands the SIZE of demand, so a part with 0 on hand and 2 builds
        waiting looked identical to one with 5 on hand and nothing waiting. This is the thing
        that actually decides urgency: can the builds that are waiting on it proceed today.
        """
        if x.get("_blockedpart") or bool(x.get("_isbuild")): return 2
        have=int(x["Reactions Available"]); need=int(x["Reactions Required"])
        if need<=0:        return 2
        if have<=0:        return 0
        if have<need:      return 1
        return 2

    def stall_badge(x):
        r=stall_rank(x)
        if r==2: return ''
        need=int(x["Reactions Required"]); have=int(x["Reactions Available"])
        if r==0:
            return (f'<span title="No stock and {need} build(s) waiting — they cannot run until '
                    f'this is made" style="background:#b91c1c;color:#fff;font-size:8px;'
                    f'font-weight:700;padding:1px 6px;border-radius:8px;margin-left:5px">'
                    f'STALLS {need}</span>')
        return (f'<span title="{have} on hand but {need} build(s) waiting — only {have} can run" '
                f'style="background:#fef3c7;color:#92400e;font-size:8px;font-weight:700;'
                f'padding:1px 6px;border-radius:8px;margin-left:5px">SHORT {need-have}</span>')

    def disp_act(x):
        """Action to SHOW. If flipping the make-available wells ON already clears the target,
        the job is 'Mark available', not 'Refill' — the pull can't reach this verdict because it
        picks the action from raw demand while the target (need + buffer) is computed here."""
        act=x["Action Suggested"]
        if str(act).startswith("Mark"): return "Mark available"    # the pull already reached it
        # "True" means no fresh source to make more from. For a dPart that is never an order —
        # dParts are PCR'd in-house (there is no dPart synthesis workorder type), so "Reorder /
        # needs order" was unactionable: order_status() only covers oligo/plasmid/synpart
        # synthesis, so a dPart could never show "on order" and always fell through to it.
        if act=="True" and str(x["Part"]).startswith("d"): return "Make by PCR"
        if act in ("Refill","Transform"):
            n,_g,after = flip_gain(str(x["Part"]), int(x["Reactions Available"]))
            if n and after >= _target(x): return "Mark available"
        return act

    _rowstate = {"i": 0}
    def row_html(i, x):
        part=str(x["Part"]); act=disp_act(x)
        # Third number in the Have column: builds already running. Without it the column read
        # "4 / 0" on a part whose own "Needed for" cell listed two RUNNING builds — technically
        # right (nothing queued) but flatly contradicting the row beside it.
        _ex_row=exposure(part)
        age=None
        bb=builds_for(part)
        if bb: summary=f'{bb[0][0]} ({bb[0][2]})'+(f' +{len(bb)-1} more' if len(bb)>1 else '')
        elif part in ctrl: summary="control — kept permanently stocked"
        elif part in ctrl_related: summary="primer/template for a control dPart"
        elif [d for d in tmpl_kids.get(part,[]) if d in flagged]: summary="template for "+", ".join(d for d in tmpl_kids.get(part,[]) if d in flagged)
        else: summary="—"
        # Build the stock cell as its own value. It was previously an `A if cond else B`
        # inline in the return's f-string chain — adjacent f-strings concatenate, so the
        # conditional split the WHOLE row rather than just this cell, and control rows
        # rendered only their first three <td>s with no closing tags.
        _hv=int(x["Reactions Available"]); _rq=int(x["Reactions Required"])
        _lbl='color:#9ca3af;font-size:9px'
        if part in ctrl_related:
            # Controls hold the pull's own reaction figure, not a count of waiting builds —
            # they are stocked to a fixed target regardless of live demand.
            stock_cell=(f'<td style="text-align:center;white-space:nowrap" '
                        f'title="Control part — kept permanently stocked at {_target(x)} rxns '
                        f'regardless of live demand. {_hv} rxns on hand.">'
                        f'<span style="font-weight:700">{_hv}</span>'
                        f'<span style="{_lbl}"> on hand</span>'
                        f'<span style="color:#d1d5db"> · </span>'
                        f'<span style="color:#6b7280;font-size:9px">control · target '
                        f'{_target(x)}</span></td>')
        else:
            # Bare "4 / 0" read as though the 0 were the stock, so each number is labelled
            # in place — a column header is too far away to disambiguate at a glance.
            stock_cell=(f'<td style="text-align:center;white-space:nowrap" '
                        f'title="{_hv} rxns on hand · {_rq} rxns for builds still WAITING to '
                        f'draw · {_ex_row["drawn"]} rxns already drawn by builds now RUNNING">'
                        f'<span style="font-weight:700">{_hv}</span>'
                        f'<span style="{_lbl}"> on hand</span>'
                        f'<span style="color:#d1d5db"> · </span>'
                        f'<strong style="color:#b45309;font-size:13px">{_rq}</strong>'
                        f'<span style="{_lbl}"> waiting</span>'
                        f'<span style="color:#d1d5db"> · </span>'
                        f'<span style="color:#6b7280;font-weight:700">{_ex_row["drawn"]}</span>'
                        f'<span style="{_lbl}"> running</span></td>')
        return (f'<tr class="prow" onclick="partsToggle({i})" style="cursor:pointer">'
                f'<td style="width:18px;color:#9ca3af" id="c{i}">▸</td>'
                f'<td style="font-family:monospace;font-weight:700">{part}'
                f'{stall_badge(x)}'
                f'{repeat_badge(x["Reactions Required"]) if stall_rank(x)==2 else ""}</td>'
                f'{stock_cell}'
                f'<td>{act_badge(act,age,not_in_lims=bool(x.get("_blockedpart") and act=="True" and _no_lims_wells(part)),muted=_covered(x))}</td>'
                f'<td style="font-size:11px">{batch_cell(x)}</td>'
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

    def _has_open_maker(part):
        """Something is already queued to make this part — a PCR for a dPart, a vendor/synthesis
        order for anything else. Such a part is ordinary restock work ("on order" / "PCR ready"),
        so it must NOT be called blocked; the section claims nothing is queued to make it."""
        if str(part).startswith("d"):
            p=pcr_status(part);  return bool(p and p["open"])
        o=order_status(part);    return bool(o and o["active"])

    _avail_by_part=dict(zip(out["Part"].astype(str),
                            pd.to_numeric(out["Reactions Available"], errors="coerce").fillna(0)))

    def _blocked_only(part):
        bb=builds_for(part)
        if not bb or _has_open_maker(part):
            return False
        if all(s=="BLOCKED" for _p,_t,s,_e in bb):
            return True
        # "EVERY consumer is blocked" was too strict. A part with nothing on hand and nothing
        # queued to make it is missing even when other consumers read RUNNING — those either
        # already drew their material or sit ahead of the draw, which is no evidence the part
        # exists. d8260 fell through exactly here: 0 rxns, no PCR workorder ever, consumed by
        # 1 BLOCKED + 2 RUNNING Gibsons. It stayed in the restock list, so the BLOCKED WO it
        # holds up matched no blocked part and landed in "Blocked workorders — cause unknown",
        # which then claimed no missing part explained a WO whose missing part was listed above.
        return any(s=="BLOCKED" for _p,_t,s,_e in bb) and _avail_by_part.get(str(part), 0) <= 0
    # A part with nothing queued to make it is BLOCKED: it can't be refilled (there is no stock to
    # top up and no workorder in flight), and everything downstream of it stops. So it does not
    # belong in the restock list — but it must not disappear either, which is what dropping it as
    # "phantom demand" used to do. It gets its own section instead, and that section now carries
    # the blocked WOs each part is holding up (see blocked_wos_for) so there is no separate
    # blocked-workorder list saying the same thing from the other end.
    _phantom={p for p in out["Part"].astype(str).unique() if _blocked_only(p)}
    _ph_rows=[]
    if _phantom:
        _ph_rows=[]
        for _,_x in out[out["Part"].astype(str).isin(_phantom)].iterrows():
            _x=_x.copy(); _x["_blockedpart"]=True; _ph_rows.append(_x)
        out=out[~out["Part"].astype(str).isin(_phantom)].reset_index(drop=True)

    # ---- Blocked totals, for the header strip and to prove nothing is being dropped ----------
    _bp_wos={}
    for _x in _ph_rows:
        for _w in blocked_wos_for(str(_x["Part"])): _bp_wos[_w["wid"]]=_w
    _n_bp      = len(_ph_rows)                                        # missing parts
    _n_bp_wo   = len(_bp_wos)                                         # distinct WOs they block
    _n_bp_twin = sum(1 for w in _bp_wos.values() if w["twin"] and not w["succ"])
    _n_bp_succ = sum(1 for w in _bp_wos.values() if w["succ"])
    # A blocked WO that no missing part accounts for would silently vanish now that the standalone
    # blocked-WO list is gone. Today that set is empty (all 15 resolve to one of the 8 parts), but
    # it is not guaranteed — so orphans still get shown, in their own small section.
    _orphan_wos=[w for wid,w in sorted(_blk_all.items()) if wid not in _bp_wos]

    # ---- Demand = DIRECT in-flight builds only (reconciles the number with "Needed for") ----
    # A RUNNING build has already drawn its material off the Echo plate — that draw is why
    # the on-hand number is low in the first place. Counting it as demand asked for the same
    # material twice: pAI-19910 sat at 42 rxns with 28 RUNNING + 10 queued builds and was told
    # to reach a target of 76. Demand is what has NOT been served yet.
    _NOT_DRAWN = {"WAITING","READY","BLOCKED","NEW"}
    def direct_need(part):
        return sum(1 for _p,_t,st,_e in builds_for(str(part)) if st in _NOT_DRAWN)

    def exposure(part):
        """Queued vs already-drawn demand, with NO assumed failure rate.

        Predicting how many RUNNING builds come back would mean inventing a retry
        percentage, and a made-up number either hoards Echo plates or strands a batch.
        So report the physical truth and let the operator judge it:
          queued  — builds that still have to draw material (the immediate need)
          drawn   — builds that already drew (why on-hand is low; NOT re-billed as need)
          lo/hi   — target if nothing retries / target if EVERY running build retries
          spare   — rxns above the immediate need
          tol     — how many of the drawn builds that spare can absorb coming back
        """
        bb=builds_for(str(part))
        queued=sum(1 for _p,_t,st,_e in bb if st in _NOT_DRAWN)
        drawn=len(bb)-queued
        return {"queued":queued,"drawn":drawn,"total":len(bb)}
    out["Reactions Required"]=out.apply(
        lambda x: int(x["Reactions Required"]) if str(x["Part"]) in ctrl_related else direct_need(x["Part"]),
        axis=1)
    builds_all["Reactions Required"]=builds_all.apply(
        lambda x: int(x["Reactions Required"]) if str(x["Part"]) in ctrl_related else direct_need(x["Part"]),
        axis=1)
    # The blocked rows were lifted out of `out` ABOVE this recompute, so they kept the pull's raw
    # demand figure while every other section switched to direct in-flight builds — pAI-22332 read
    # "0 / 12" next to its own "blocking 4 WOs" and "4 builds across 1 experiment". Same basis for
    # every section, so Need, the target, and the WO count can't contradict each other.
    for _x in _ph_rows:
        if str(_x["Part"]) not in ctrl_related:
            _x["Reactions Required"]=direct_need(_x["Part"])

    def _buffer(n):
        """Spare stock to hold on top of `n`, per the configured stocking policy."""
        return max(PipelineConfig.REFILL_BUFFER_MIN,
                   math.ceil(PipelineConfig.REFILL_BUFFER_FRAC * n))

    def _target(x):
        need=int(x["Reactions Required"])
        return 96 if str(x["Part"]) in ctrl_related else need + _buffer(need)

    def _target_max(x):
        """Target if every RUNNING build came back for another attempt — the worst case."""
        if str(x["Part"]) in ctrl_related: return 96
        tot=exposure(x["Part"])["total"]
        return tot + _buffer(tot)

    # Visibility uses the WORST case, urgency uses the immediate need. Filtering on the
    # immediate need alone silently dropped parts that are fine today but have no cover if
    # their running builds retry (pAI-19910: 42 on hand, 10 queued, 28 already drawn) — the
    # row vanished instead of saying so. Nothing disappears now; the row states both numbers.
    out=out[out.apply(
        lambda x: bool(x["_isbuild"]) or (str(x["Part"]) in ctrl_related)
                  or int(x["Reactions Available"]) < _target_max(x), axis=1)].reset_index(drop=True)

    i = 0
    def part_exps(part):
        # cons stores a missing experiment as the literal "—" (see the `EXP or "—"` above), which
        # is truthy — so a bare `if e` treated "unknown experiment" as an experiment NAMED "—" and
        # gave the part its own one-row group titled with a dash. That group also shadowed the
        # no-demand bucket: pAI-13500 (template for control dPart d4674, whose PCR workorders
        # carry no EXP) was split off from d4674 instead of sitting with it. Drop the sentinel so
        # those rows fall through to the `not exps` branch.
        return sorted({e for _,_,_,e in builds_for(part) if e and e != "—"})
    _PCOLS='<colgroup><col style="width:26px"><col style="width:19%"><col style="width:14%"><col style="width:12%"><col style="width:21%"><col></colgroup>'
    _HDR=('<tr><th></th><th>Part</th>'
          '<th title="rxns on hand / rxns for builds still waiting / rxns already drawn by '
          'running builds">On hand / waiting / running</th>'
          '<th>Action</th><th>Batch / order</th><th>Needed for</th></tr>')
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
    def grouped_by_experiment(rowobjs, multi_accent="#7c3aed", noexp_title="Controls & no live demand", open_=True):
        by={}; multi=[]; noexp=[]
        for x in rowobjs:
            exps=part_exps(str(x["Part"]))
            if not exps: noexp.append(x)
            elif len(exps)==1: by.setdefault(exps[0],[]).append(x)
            else: multi.append(x)
        hh=""
        # Group order follows the worst stall inside it: a part with builds that cannot run
        # must not sit below a bigger-but-healthy group where nobody scrolls to it.
        for e in sorted(by, key=lambda e:(min(stall_rank(x) for x in by[e]),
                                          -sum(int(x["Reactions Required"]) for x in by[e]))):
            hh+=_exp_group(e, sorted(by[e], key=lambda x:(stall_rank(x),-int(x["Reactions Required"]))), open_=open_)
        if multi:
            hh+=_exp_group("Multi-project parts", sorted(multi,key=lambda x:(stall_rank(x),-int(x["Reactions Required"]))),
                           accent=multi_accent, desc="feed more than one experiment — see “Needed for”", open_=open_)
        if noexp:
            hh+=_exp_group(noexp_title, noexp, open_=open_)
        return hh

    _nb_rows=[x for _,x in builds_all.iterrows()] if not builds_all.empty else []
    _pa_rows=[x for _,x in out[~out["_isbuild"]].iterrows()]
    newbuilds_html = grouped_by_experiment(_nb_rows, open_=False) if _nb_rows else ""
    parts_html     = grouped_by_experiment(_pa_rows, open_=True)
    blockedparts_html = grouped_by_experiment(_ph_rows, open_=True) if _ph_rows else ""

    # ============================================================================
    # Well/plate action sections: Make Unavailable, Trash
    # ============================================================================
    lsp_plates = r.get("lsp_plates")
    lsp_ids = set()
    if lsp_plates is not None and not lsp_plates.empty and "PLATE_ID" in lsp_plates.columns:
        lsp_ids = set(pd.to_numeric(lsp_plates["PLATE_ID"], errors="coerce").dropna().astype(int))

    clean_wells = P.build_clean_inventory_queue(apd, now, exclude_oligos=True)
    mp_wells    = P.build_miniprep_unavail_queue(apd, now)
    disc_wells  = P.build_discarded_available_queue(apd)
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

    # Error plates — labware says "384 Echo Source Plate" but the plate is not
    # physically 384-well (LIMS data error). These are NOT valid Echo sources and
    # are excluded from every Echo-source list above; surface them so they can be
    # found, verified, and discarded / relabeled in LIMS.
    _echo_lbl = apd[apd["LABWARE"] == "384 Echo Source Plate"].copy()
    _echo_lbl["_wc"] = pd.to_numeric(_echo_lbl["PLATE_NUMBER_OF_WELLS"], errors="coerce")
    _err = _echo_lbl[_echo_lbl["_wc"] != 384]
    # Already dealt with → not an action item. A plate whose location says DISCARD, or whose every
    # well is switched OFF, can no longer leak into an Echo-source list and there is nothing left
    # to relabel for. LIMS may still carry the wrong labware string, but nobody has to touch it —
    # this section is for plates you still have to go find, so drop the finished ones.
    _err = _err[~_err["PLATE_LOCATION_BOX"].fillna("").astype(str).str.upper().str.contains("DISCARD")]
    _err = _err[_err["PLATE_ID"].isin(set(_err[_err["AVAILABLE"].astype(str) == "True"]["PLATE_ID"]))]
    err_plates = []
    for pid, g in _err.groupby("PLATE_ID"):
        _loc = g["PLATE_LOCATION_BOX"].dropna().astype(str)
        _loc = _loc.mode().iloc[0] if not _loc.empty else "(no loc)"
        _wc = g["_wc"].dropna()
        _wc = int(_wc.iloc[0]) if not _wc.empty else "?"
        err_plates.append((int(pid), _loc, _wc, g["WELL_ID"].nunique()))
    err_plates.sort(key=lambda x: x[0])
    err_section = ""
    if err_plates:
        erows = "".join(f'<tr><td style="font-family:monospace">plate {p}</td>'
                        f'<td>{html.escape(str(loc))}</td>'
                        f'<td style="text-align:right;color:#be123c;font-weight:700">{wc}-well</td>'
                        f'<td style="text-align:right;color:#6b7280">{n}</td></tr>'
                        for p, loc, wc, n in err_plates)
        ebody = (f'<table class="platetbl"><thead><tr>'
                 f'<th style="text-align:left;padding:3px 10px">Plate</th>'
                 f'<th style="text-align:left;padding:3px 10px">Location</th>'
                 f'<th style="text-align:right;padding:3px 10px">Actual size</th>'
                 f'<th style="text-align:right;padding:3px 10px">Wells</th>'
                 f'</tr></thead><tbody>{erows}</tbody></table>')
        err_section = (f'<div class="sec"><div class="sechd" style="border-left:4px solid #dc2626">'
                       f'&#9888; Error plates — mislabeled Echo source '
                       f'<span class="seccount" style="background:#dc2626">{len(err_plates)}</span>'
                       f'<span class="secdesc">labware says &ldquo;384 Echo Source Plate&rdquo; but the plate is NOT 384-well '
                       f'(LIMS data error) &middot; find, verify, and discard or relabel &middot; excluded from all Echo-source lists</span>'
                       f'</div>{ebody}</div>')

    # Make Available — the flip-ON wells for ONLY the parts listed in this tab. A global
    # make-available list was pulled once before because there was no way to tell whether an
    # arbitrary well was partner-associated; scoping it to the parts we are actively restocking
    # keeps that bound: every well here is one already shown in that part's own row.
    # _ph_rows (the blocked parts) are in here too. They are correctly blocked — right now their
    # stuck WOs have no usable stock to draw on — but a flip-ON well is exactly what un-blocks
    # them, so leaving those wells out of the copy box hid the cheapest possible fix.
    mk_avail = []
    for _x in _pa_rows + _nb_rows + _ph_rows:
        for _p, _co, _wid, _loc, _v, _cc, _a in make_avail_wells(str(_x["Part"])):
            tok = f"well{int(_wid)}"
            if tok not in mk_avail: mk_avail.append(tok)
    # Copy-box sections first — the well strings are the thing you act on every day. The error /
    # trash plate lists are find-and-toss housekeeping, so they sit at the bottom (err_section is
    # appended after the trash sections below).
    extra = wells_section("mk_av","Make Available · 384 Echo source",
              "seq-confirmed Echo source wells for the parts listed above that are &gt;25µL, &gt;5 ng/µL, fresh, and "
              "not yet available → flip ON in LIMS · <b>these are the refills</b>: flipping them is usually enough "
              "to reach target without batching · includes blocked parts, where flipping is what "
              "un-blocks their stuck workorders","#1d4ed8", mk_avail, show_plates=False)
    extra += wells_section("mk_un","Make Unavailable · 384 Echo source",
              "available Echo source wells that are ≤25µL (near-empty), past expiration (200d), OR &lt;5 ng/µL (too dilute) → flip OFF in LIMS "
              "· <b>dParts are exempt from the &lt;5 ng/µL rule only</b> (a PCR product is expected to come off dilute) — "
              "they still flip OFF at ≤25µL or 200d","#be185d", clean_wells, show_plates=False)
    extra += wells_section("mk_un_mp","Make Unavailable · 96-well miniprep stock",
              "available miniprep-stock wells (96-well) past expiration (200d) → flip OFF in LIMS","#9d174d", mp_wells, show_plates=False)
    extra += wells_section("mk_un_disc","Make Unavailable · wells on DISCARDED plates",
              "wells still switched ON where the plate location reads DISCARDED → flip OFF in LIMS · the plate is gone, so the "
              "well cannot be usable · <b>no exemptions</b>: any labware, any part type, regardless of volume, ng/µL or age · "
              "the other lists skip DISCARDED plates (nothing left to find), which is what let these keep reading as available stock",
              "#7f1d1d", disc_wells, show_plates=False)

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

    # Trash by part type: STOCK plates in the 4B fridge (4°C), past expiration
    def _ptype(sid):
        sid=str(sid)
        if sid.startswith("pAI"): return "Plasmid"
        if sid.startswith("syn"): return "SynPart"
        if sid.startswith("d"):  return "dPart"
        if sid.startswith("o"):  return "Oligo"
        return "Other"
    EXPIRE_DAYS={"Plasmid":200,"dPart":200,"SynPart":200,"Oligo":730}

    stk = apd[(apd["WELL_TYPE"]=="Stock") & _is_echo384(apd)].copy()
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
    extra += err_section      # bottom of the well/plate housekeeping block, with the trash lists

    # Overview counts use the DISPLAYED action, so the cards can't disagree with the table
    # (a part whose flip-ON wells already cover the target counts as Mark available, not Refill).
    _disp = [disp_act(x) for _,x in out.iterrows()]
    _n_flip   = sum(1 for a in _disp if a=="Mark available")
    _n_refill = sum(1 for a in _disp if a=="Refill")
    _n_xform  = sum(1 for a in _disp if a=="Transform")
    _n_pcr    = sum(1 for a in _disp if a=="Make by PCR")
    _n_nosrc  = sum(1 for a in _disp if a=="True")
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

    # ---- "Add it up" strip for the blocked section: parts, WOs they stall, what's cancelable ----
    _bp_strip=""
    if _ph_rows:
        _bits=[f'<b style="color:#b91c1c">{_n_bp}</b> missing part{"s" if _n_bp!=1 else ""}',
               f'blocking <b style="color:#b91c1c">{_n_bp_wo}</b> '
               f'workorder{"s" if _n_bp_wo!=1 else ""}']
        if _n_bp_succ:
            _bits.append(f'<b style="color:#15803d">{_n_bp_succ}</b> of those already produced '
                         f'elsewhere → cancelable')
        if _n_bp_twin:
            _bits.append(f'<b style="color:#b45309">{_n_bp_twin}</b> of those have another '
                         f'unblocked WO for the same final product → cancel the blocked one')
        _bp_strip=('<div style="margin:0 0 8px;padding:7px 10px;background:#fef2f2;'
                   'border:1px solid #fecaca;border-radius:6px;font-size:11.5px;color:#374151">'
                   + ' &nbsp;·&nbsp; '.join(_bits) + '</div>')

    # Only rendered when a blocked WO can't be traced back to a missing part above (normally none).
    _orphan_html=""
    if _orphan_wos:
        _orphan_html=('<div style="padding:2px"><div style="font-size:11px;color:#6b7280;'
                      'margin:0 0 6px">no missing part above accounts for these — the WO itself '
                      'needs investigating</div>' + _wo_tbl(_orphan_wos) + '</div>')

    # ---- SCOPED fragment: every CSS rule namespaced under #tab-parts so nothing leaks ----
    frag=f"""<style>
 #tab-parts{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#e9ecf2;color:#1d1d1f}}
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
 #tab-parts .d-barwrap{{height:6px;background:#eceaf3;border-radius:4px;margin:6px 0 4px;max-width:320px;overflow:hidden;display:flex}} #tab-parts .d-bar{{height:100%;border-radius:4px}}
 #tab-parts .d-seg{{height:6px;border-radius:4px;display:inline-block}}
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
  <div class="ovc"><div class="ovn" style="color:#1d4ed8">{_n_flip}</div><div class="ovl">Flip wells ON</div></div>
  <div class="ovc"><div class="ovn" style="color:#92400e">{_n_refill}</div><div class="ovl">Refill</div></div>
  <div class="ovc"><div class="ovn" style="color:#c2410c">{_n_xform}</div><div class="ovl">Transform</div></div>
  <div class="ovc"><div class="ovn" style="color:#0e7490">{_n_pcr}</div><div class="ovl">Add PCR WO</div></div>
  <div class="ovc"><div class="ovn" style="color:#be185d">{_n_nosrc}</div><div class="ovl">Reorder</div></div>
  <div class="ovc"><div class="ovn" style="color:#b91c1c">{_n_bp}</div><div class="ovl">Blocked parts &rarr; {_n_bp_wo} WOs</div></div>
  <div class="ovc"><div class="ovn" style="color:#be185d">{len(set(clean_wells)|set(mp_wells)|set(disc_wells))}</div><div class="ovl">Wells → unavailable</div></div>
  <div class="ovc"><div class="ovn" style="color:#6b7280">{_n_trash}</div><div class="ovl">Plates to trash</div></div>
</div>
{section_card("Parts needing attention", ("#fffbeb","#b45309","#fde68a"), _pa_count, parts_html, "restock / refill / reorder — grouped by experiment · Need = in-flight builds that have not drawn their material yet (RUNNING ones already did)", colhdr=_PHDR_ROW)}
{section_card("Blocked — nothing queued to make it", ("#fef2f2","#b91c1c","#fca5a5"), _n_bp, _bp_strip+blockedparts_html, f"order or PCR these to unblock {_n_bp_wo} stuck workorder{'s' if _n_bp_wo!=1 else ''} · click a part to see exactly which WOs it stalls", colhdr=_PHDR_ROW)}
{section_card("Blocked workorders — cause unknown", ("#fef2f2","#b91c1c","#fca5a5"), len(_orphan_wos), _orphan_html, "stuck WOs that no missing part above explains — investigate the workorder itself")}
{section_card("New builds — feed into requests", ("#f5f3ff","#6d28d9","#ddd6fe"), _nb_count, newbuilds_html, "net-new parts being assembled (workorder in flight) that feed downstream requests", colhdr=_PHDR_ROW)}
<div class="secgroup-title">Well &amp; plate actions</div>
{extra}
<script>
function partsToggle(i){{var d=document.getElementById('d'+i),c=document.getElementById('c'+i);
 var open=d.style.display==='none'; d.style.display=open?'table-row':'none'; c.textContent=open?'▾':'▸';}}
</script>"""
    return frag

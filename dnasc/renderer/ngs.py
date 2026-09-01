"""NGS queue tab — self-contained, SCOPED fragment (#tab-ngs).

An NGS run fits 384 samples. When the queue overflows, the question is which queued samples do
not need to spend a slot. The rule this tab implements:

    an AVAILABLE glycerol stock for the plasmid
    AND an LSP prep already IN FLIGHT for it

Both halves matter. A glycerol on its own proves nothing here — the glycerol and the miniprep
being sequenced are created by the same pick ("Create Minipreps and Glycerol Stocks"), so almost
every queued sample has one. It is the running LSP prep that says the material is already moving
downstream, and the available glycerol that says we can still get back to it without re-picking.

Two structural facts about this data, both verified against LIMS:
  • the queue's own rows are nearly empty — plasmid_stock_id / dpart_stock_id / syn_part_stock_id
    are NULL on every RearrayQuantLsp sample, so identity comes from well → well_content
  • ngs_workorder carries NO request_id at all (0 of 141), so "an LSP for that request" can only
    be matched through the PLASMID; the LSP workorder carries both, and its request is displayed

Reads parts_result.pkl (gen_parts_pkl.py). CSS namespaced under #tab-ngs. Never raises.
"""
from __future__ import annotations  # 3.9 server compat (lazy annotations)
import os
import sys
import html

from dnasc import wells as _wells
import pickle
import datetime
from zoneinfo import ZoneInfo

import pandas as pd

_ET = ZoneInfo("America/New_York")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PKL = os.path.join(_ROOT, "dashboard_state", "parts_result.pkl")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_GLY_LABWARE = ["Thermo V Bottom Plate", "Eppendorf V Microplate"]
_BATCH_CAP = 384

_FALLBACK = ('<div style="padding:24px;color:#6b7280;font:14px -apple-system,sans-serif">'
             'NGS queue data unavailable — the parts pull (gen_parts_pkl.py) has not run yet '
             'or failed. Check the parts cron / logs/parts_pull.log.</div>')

_STALE = ('<div style="padding:24px;color:#6b7280;font:14px -apple-system,sans-serif">'
          'NGS queue not in this pull — regenerate parts_result.pkl with gen_parts_pkl.py '
          '(the queue was added in v1.11.38).</div>')


def render_ngs_tab() -> str:
    """Return the NGS tab fragment. Never raises."""
    try:
        return _render()
    except Exception:
        import traceback
        traceback.print_exc()
        return _FALLBACK


def _well_coord(pos, rows=8) -> str:
    """RAW 0-based LIMS position -> coordinate. See dnasc/wells.py for the rules;
    do not reimplement the arithmetic here."""
    return _wells.coord_rows(pos, rows)


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _render() -> str:
    r = pickle.load(open(_PKL, "rb"))
    q = r.get("ngs_queue")
    if q is None:
        return _STALE
    apd = r["all_plate_data"]
    lsp_act = r.get("lsp_active")
    lspb = r.get("lsp_batches")
    now = r["generated_at"]
    try:
        _n = now if getattr(now, "tzinfo", None) else pd.Timestamp(now).tz_localize("UTC")
        now_et = _n.astimezone(_ET)
    except Exception:
        now_et = now

    # ---- Available glycerol per plasmid (prefer a non-NEB strain, as the parts pull does).
    gly = apd[apd["LABWARE"].isin(_GLY_LABWARE) & (apd["AVAILABLE"].astype(str) == "True")].copy()
    if len(gly):
        gly["_neb"] = gly["COMP_CELL"].astype(str).str.startswith("NEB")
        gly = (gly.sort_values(["STOCK_ID", "_neb"]).drop_duplicates("STOCK_ID", keep="last")
                  .set_index("STOCK_ID"))
    gly_ids = set(gly.index.astype(str)) if len(gly) else set()

    # ---- LSP preps in flight, keyed by plasmid (the only link — see module docstring).
    lsp_by_part = {}
    if lsp_act is not None and len(lsp_act):
        for part, sub in lsp_act.groupby(lsp_act["PART"].astype(str)):
            lsp_by_part[part] = {
                "n": int(len(sub)),
                "statuses": sorted({str(s) for s in sub["STATUS"]}),
                "reqs": [str(x)[:8] for x in sub["REQUEST_ID"].dropna().astype(str)][:3],
                "batches": [str(x) for x in sub["BATCH_ID"].dropna().astype(str)][:3],
            }

    # ---- Past LSP verdicts, shown as context only — NOT part of the flag rule.
    past = {}
    if lspb is not None and len(lspb):
        for part, sub in lspb.groupby(lspb["PART"].astype(str)):
            sts = [str(s) for s in sub["NGS_STATUS"].dropna().astype(str)
                   if s not in ("", "None", "nan")]
            past[part] = {"n": int(len(sub)),
                          "passed": sum(1 for s in sts if s.strip().lower() == "pass")}

    rows = []
    for part, sub in q.groupby(q["PART"].astype(str), dropna=False):
        g = gly.loc[part] if part in gly_ids else None
        l = lsp_by_part.get(part)
        p = past.get(part)
        colonies = sorted({str(int(c)) for c in sub["COLONY"].dropna()})
        plates = sorted({str(int(x)) for x in sub["PLATE_ID"].dropna()})
        rows.append({
            "part": part,
            "n": int(len(sub)),
            "colonies": ", ".join(colonies),
            "plates": ", ".join(plates[:4]) + ("…" if len(plates) > 4 else ""),
            "status": ", ".join(sorted({str(s) for s in sub["STATUS"]})),
            "gly": g is not None,
            "gly_plate": "" if g is None else ("" if pd.isna(g.get("PLATE_ID")) else str(int(g["PLATE_ID"]))),
            "gly_well": "" if g is None else _well_coord(g.get("WELL_NUMBER")),
            "gly_loc": "" if g is None else str(g.get("PLATE_LOCATION_BOX") or ""),
            "lsp": l,
            "past_n": 0 if not p else p["n"],
            "past_pass": 0 if not p else p["passed"],
        })

    def _flagged(x):
        return bool(x["gly"] and x["lsp"])
    rows.sort(key=lambda x: (not _flagged(x), -x["n"], x["part"]))

    n_samples = int(sum(x["n"] for x in rows))
    n_parts = len(rows)
    n_flag_parts = sum(1 for x in rows if _flagged(x))
    n_flag_samples = sum(x["n"] for x in rows if _flagged(x))
    over = max(0, n_samples - _BATCH_CAP)

    body = []
    for x in rows:
        glyc = '<span class="no">none available</span>'
        if x["gly"]:
            bits = []
            if x["gly_plate"]: bits.append(f'plate <b>{_esc(x["gly_plate"])}</b>')
            if x["gly_well"]:  bits.append(f'well <b>{_esc(x["gly_well"])}</b>')
            if x["gly_loc"]:   bits.append(f'<span class="dim">{_esc(x["gly_loc"])}</span>')
            glyc = " · ".join(bits) or "available"
        if x["lsp"]:
            l = x["lsp"]
            lspc = (f'<b>{l["n"]}</b> in flight · {_esc(", ".join(l["statuses"]))}'
                    + (f'<div class="dim">req {_esc(", ".join(l["reqs"]))}</div>' if l["reqs"] else "")
                    + (f'<div class="dim">batch {_esc(", ".join(l["batches"]))}</div>' if l["batches"] else ""))
        else:
            lspc = '<span class="no">no prep running</span>'
        hist = "—" if not x["past_n"] else (
            f'{x["past_n"]} batch{"es" if x["past_n"] != 1 else ""}'
            + (f' · <span class="pass">{x["past_pass"]} passed</span>' if x["past_pass"] else ""))
        if _flagged(x):
            verdict = ('<span class="v vgo">glycerol banked + prep running</span>'
                       '<span class="vsub">the material is already moving downstream and we can '
                       'get back to it → drop from the run</span>')
        elif x["gly"]:
            verdict = ('<span class="v vneed">sequence it</span>'
                       '<span class="vsub">glycerol is available, but no LSP prep is running</span>')
        elif x["lsp"]:
            verdict = ('<span class="v vneed">sequence it</span>'
                       '<span class="vsub">a prep is running, but no glycerol to fall back on</span>')
        else:
            verdict = '<span class="v vneed">sequence it</span>'
        body.append(
            f'<tr class="{"hit" if _flagged(x) else ""}">'
            f'<td class="mono">{_esc(x["part"])}</td>'
            f'<td><b>{x["n"]}</b><div class="dim">colony {_esc(x["colonies"])}</div></td>'
            f'<td class="dim">{_esc(x["plates"])}</td>'
            f'<td>{glyc}</td><td>{lspc}</td><td>{hist}</td><td>{verdict}</td></tr>')

    tbl = ('<table class="ngstbl"><thead><tr>'
           '<th>Plasmid</th><th>Queued</th><th>Miniprep plate</th><th>Glycerol stock</th>'
           '<th>LSP prep in flight</th><th>Past LSP</th><th>Verdict</th>'
           '</tr></thead><tbody>' + "".join(body) + '</tbody></table>') if rows else \
          '<div class="empty">Nothing is queued for sequencing right now.</div>'

    cap_note = (f'<span class="over">{n_samples} queued · {over} over the {_BATCH_CAP} cap</span>'
                if over else
                f'<span class="under">{n_samples} queued · fits in one {_BATCH_CAP}-sample run</span>')
    ts = now_et.strftime("%Y-%m-%d %-I:%M %p ET") if hasattr(now_et, "strftime") else str(now_et)
    return f"""
<style>
#tab-ngs {{ font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#1d1d1f; }}
#tab-ngs .wrap {{ padding:18px 22px 40px; }}
#tab-ngs h2 {{ font-size:19px; margin:0 0 3px; font-weight:700; }}
#tab-ngs .sub {{ font-size:12px; color:#6b7280; margin:0 0 14px; max-width:930px; line-height:1.55; }}
#tab-ngs .stamp {{ float:right; font-size:11px; font-weight:700; color:#1d4ed8; background:#eff6ff;
                   border:1px solid #bfdbfe; border-radius:6px; padding:4px 11px; }}
#tab-ngs .ov {{ display:flex; gap:10px; margin:0 0 8px; flex-wrap:wrap; }}
#tab-ngs .ovc {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:10px 16px; min-width:100px; }}
#tab-ngs .ovn {{ font-size:22px; font-weight:800; line-height:1.1; }}
#tab-ngs .ovl {{ font-size:11px; color:#6b7280; margin-top:2px; }}
#tab-ngs .cap {{ font-size:12px; margin:0 0 14px; }}
#tab-ngs .over {{ color:#b91c1c; font-weight:700; }}
#tab-ngs .under {{ color:#15803d; font-weight:700; }}
#tab-ngs .ngstbl {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid #e5e7eb;
                    border-radius:8px; overflow:hidden; }}
#tab-ngs .ngstbl th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
                       color:#6b7280; background:#f9fafb; padding:8px 10px; border-bottom:1px solid #e5e7eb; }}
#tab-ngs .ngstbl td {{ padding:8px 10px; border-bottom:1px solid #f3f4f6; vertical-align:top; font-size:12px; }}
#tab-ngs .ngstbl tr.hit {{ background:#f0fdf4; }}
#tab-ngs .ngstbl tr.hit td:first-child {{ box-shadow:inset 3px 0 0 #15803d; }}
#tab-ngs .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-weight:600; white-space:nowrap; }}
#tab-ngs .dim {{ color:#9ca3af; font-size:11px; }}
#tab-ngs .no {{ color:#c4c4c6; }}
#tab-ngs .pass {{ color:#15803d; font-weight:600; }}
#tab-ngs .v {{ font-weight:700; display:block; }}
#tab-ngs .vsub {{ color:#6b7280; font-size:11px; display:block; margin-top:1px; }}
#tab-ngs .vgo {{ color:#15803d; }}
#tab-ngs .vneed {{ color:#6b7280; font-weight:600; }}
#tab-ngs .empty {{ padding:20px; color:#6b7280; background:#fff; border:1px solid #e5e7eb; border-radius:8px; }}
</style>
<div class="wrap">
  <div class="stamp">DATA PULLED {_esc(ts)}</div>
  <h2>NGS queue — what can come out of the run</h2>
  <p class="sub">Every sample with an open NGS workorder, grouped by plasmid. A run fits {_BATCH_CAP}
  samples, so when the queue overflows these are the candidates to drop: a plasmid with an
  <b>available glycerol stock</b> AND an <b>LSP prep already in flight</b>. Both halves are required —
  a glycerol alone means nothing here, because the glycerol and the miniprep being sequenced come
  from the same pick, so nearly every queued sample has one. The running prep is what says the
  material is already moving; the glycerol is what says we can get back to it without re-picking.
  Past LSP results are shown as context, not as part of the rule.</p>
  <div class="ov">
    <div class="ovc"><div class="ovn">{n_samples}</div><div class="ovl">Samples queued</div></div>
    <div class="ovc"><div class="ovn">{n_parts}</div><div class="ovl">Distinct plasmids</div></div>
    <div class="ovc"><div class="ovn" style="color:#15803d">{n_flag_samples}</div><div class="ovl">Can drop</div></div>
    <div class="ovc"><div class="ovn" style="color:#15803d">{n_flag_parts}</div><div class="ovl">Plasmids to drop</div></div>
  </div>
  <div class="cap">{cap_note}</div>
  {tbl}
</div>
"""

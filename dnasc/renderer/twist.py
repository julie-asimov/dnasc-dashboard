"""Twist tab — vendor synthesis orders, SCOPED fragment (#tab-twist).

Answers one question: where is every part we have ordered from Twist but not yet received?
Orders are split into the three states that call for different action, in that order:

    IN PROGRESS   still being synthesised — the only number that matters is ETA vs due date
    IN TRANSIT    shipped, not arrived — carrier scan and estimated delivery
    DELIVERED     arrived — and, crucially, whether the parts were RECEIVED INTO LIMS

That last column is the point of the Delivered table. A shipment can be sitting on the bench
while the pipeline still shows its parts READY (= ordered, not received), and nothing else in
the dashboard surfaces that gap.

Two things learned from the API and encoded here:
  • the order-level `estimated_completion_date` is never reconciled after shipping, so once a
    shipment exists the SHIPMENT dates are the truth and the order ETA is only context
  • the "plate-maps" endpoint serves a plain CSV, not a ZIP — that CSV is cached whole in the
    pkl and handed to the browser by the ↓ Plate map button, and its `Name` column is the LIMS
    STOCK_ID, which is what lets a delivered part show its physical plate/well/QC/yield

Reads dashboard_state/twist_result.pkl (gen_twist_pkl.py, its OWN cron — Twist staleness is
independent of the pipeline). CSS/JS namespaced under #tab-twist. Never raises.
"""
from __future__ import annotations  # 3.9 server compat (lazy annotations)

import datetime as dt
import html
import json
import os
import pickle
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PKL = os.path.join(_ROOT, "dashboard_state", "twist_result.pkl")

# The pull is cheap to re-run but slow (~75 s/page); anything older than this is called out
# rather than shown as if it were live.
_STALE_HOURS = 18

_FALLBACK = ('<div style="padding:24px;color:#6b7280;font:14px -apple-system,sans-serif">'
             'Twist order data unavailable — the Twist pull (gen_twist_pkl.py) has not run yet, '
             'failed, or its API tokens expired. Check the Twist cron / logs.</div>')


def render_twist_tab() -> str:
    """Return the Twist tab fragment. Never raises."""
    try:
        return _render()
    except Exception:
        import traceback
        traceback.print_exc()
        return _FALLBACK


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _date(s):
    """ISO-ish string → date, or None. Twist mixes plain dates and full timestamps."""
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _d(s) -> str:
    return str(s)[:10] if s else ""


def _badge(text, tone="grey") -> str:
    return f'<span class="bdg {tone}">{_esc(text)}</span>'


# ── row model ─────────────────────────────────────────────────────────────────
def _classify(order: dict) -> str:
    """in_progress | in_transit | delivered.

    Bucketed on SHIPMENTS, not on order status: `status` is 'past'/'open' bookkeeping that
    does not track the box. An order with several shipments lands in the bucket of its
    least-advanced open one, so a partial shipment stays visible as in-transit work.
    """
    ships = order.get("shipments") or []
    if not ships:
        return "in_progress"
    if any(s.get("status") == "shipped" for s in ships):
        return "in_transit"
    if any(s.get("status") == "received" for s in ships):
        return "delivered"
    return "in_progress"


def _row(order: dict, parts: list, wells: dict, changes: dict) -> dict:
    q = order.get("order_name", "")
    ships = order.get("shipments") or []
    waiting = [p for p in parts if p["vis_status"] == "READY"]
    received = [p for p in parts if p["vis_status"] == "SUCCEEDED"]
    eta = _date(order.get("estimated_completion_date"))
    due = _date(order.get("due_date"))
    delivered = sorted(d for d in (_date(s.get("received_at")) for s in ships
                                   if s.get("status") == "received") if d)
    return {
        "q": q,
        "bucket": _classify(order),
        "project": order.get("project_name") or "",
        "status": order.get("status") or "",
        "ordered": _date(order.get("received_date")),
        "eta": eta,
        "due": due,
        "delivered": delivered[-1] if delivered else None,
        "items": order.get("total_items") or 0,
        "in_prod": order.get("in_production_items") or 0,
        "completed": order.get("completed_items") or 0,
        "closed": order.get("closed_items") or 0,
        "failed": order.get("failed_items") or 0,
        "cancelled": order.get("cancelled_items") or 0,
        "high_priority": bool(order.get("high_priority")),
        "parts": sorted(parts, key=lambda p: p["stock_id"]),
        "n_waiting": len(waiting),
        "n_received": len(received),
        "ships": ships,
        "wells": wells,
        "changes": {p["stock_id"]: changes[p["stock_id"]]
                    for p in parts if p["stock_id"] in changes},
    }


# ── cell builders ─────────────────────────────────────────────────────────────
def _order_cell(r: dict) -> str:
    bits = [f'<span class="q">{_esc(r["q"])}</span>']
    if r["high_priority"]:
        bits.append(_badge("rush", "amber"))
    proj = f'<div class="dim">{_esc(r["project"])}</div>' if r["project"] else ""
    ord_d = f'<div class="dim">ordered {r["ordered"]}</div>' if r["ordered"] else ""
    return " ".join(bits) + proj + ord_d


def _parts_cell(r: dict, rid: str) -> str:
    """Count + a toggle into the per-part detail row."""
    n = len(r["parts"])
    if not n:
        return '<span class="no">no pipeline parts</span>'
    lab = f'<b>{r["n_waiting"]}</b> waiting' if r["n_waiting"] else f'{n} parts'
    sub = []
    if r["n_received"]:
        sub.append(f'{r["n_received"]} received')
    if r["items"] and r["items"] != n:
        sub.append(f'Twist lists {r["items"]}')
    subs = f'<div class="dim">{_esc(" · ".join(sub))}</div>' if sub else ""
    return (f'<span class="tog" onclick="twistToggle(\'{rid}\')">{lab}'
            f' <span class="caret">▸</span></span>{subs}')


def _eta_cell(r: dict) -> str:
    """ETA with slip against the promised due date, for orders still at Twist."""
    if not r["eta"]:
        return '<span class="no">—</span>'
    today = dt.date.today()
    late = (today - r["eta"]).days
    if late > 0:
        head = f'<b class="bad">{r["eta"]}</b> <span class="bad">overdue {late}d</span>'
    else:
        head = f'<b>{r["eta"]}</b> <span class="dim">in {-late}d</span>' if late < 0 else f'<b>{r["eta"]}</b> <span class="dim">today</span>'
    due = ""
    if r["due"]:
        slip = (r["eta"] - r["due"]).days
        tone = "bad" if slip > 0 else "dim"
        note = f"{slip}d past due" if slip > 0 else "within due date"
        due = f'<div class="{tone}">due {r["due"]} · {note}</div>'
    return head + due


def _progress_cell(r: dict) -> str:
    """Twist's own item counters — the only view into what is happening on their floor."""
    tot = r["items"] or 0
    done = r["completed"] + r["closed"]
    pct = int(round(100 * done / tot)) if tot else 0
    bits = []
    if r["in_prod"]:
        bits.append(f'{r["in_prod"]} in production')
    if r["completed"]:
        bits.append(f'{r["completed"]} complete')
    if r["closed"]:
        bits.append(f'{r["closed"]} closed')
    if r["failed"]:
        bits.append(f'<span class="bad">{r["failed"]} failed</span>')
    if r["cancelled"]:
        bits.append(f'<span class="bad">{r["cancelled"]} cancelled</span>')
    bar = (f'<div class="bar"><span style="width:{pct}%"></span></div>')
    return f'<div class="dim">{" · ".join(bits) or "—"}</div>{bar}'


def _ship_cell(r: dict, want: tuple) -> str:
    out = []
    for s in r["ships"]:
        if s.get("status") not in want:
            continue
        carrier = (s.get("carrier") or "").upper()
        trk = s.get("tracking_number") or ""
        url = s.get("tracking_url") or ""
        link = (f'<a href="{_esc(url)}" target="_blank" rel="noopener" class="trk">{_esc(trk)}</a>'
                if url and trk else _esc(trk))
        det = (s.get("status_detail") or {})
        prov = det.get("provider_detail") or det.get("provider_status") or ""
        loc = s.get("last_location") or ""
        scan = _d(s.get("last_updated_at"))
        out.append(
            f'<div class="ship"><span class="carrier">{_esc(carrier)}</span> {link}'
            + (f'<div class="dim">{_esc(str(prov).replace("_", " "))}</div>' if prov else "")
            + (f'<div class="dim">last scan {_esc(loc)}{" · " + scan if scan else ""}</div>' if loc else "")
            + '</div>')
    return "".join(out) or '<span class="no">—</span>'


def _transit_when(r: dict) -> str:
    for s in r["ships"]:
        if s.get("status") != "shipped":
            continue
        est = _date(s.get("estimated_delivery_date"))
        shipped = _d(s.get("shipped_date"))
        if not est:
            return f'<div class="dim">shipped {shipped}</div>'
        days = (est - dt.date.today()).days
        when = "arriving today" if days == 0 else (f"in {days}d" if days > 0 else f'<span class="bad">{-days}d overdue</span>')
        return f'<b>{est}</b> <span class="dim">{when}</span><div class="dim">shipped {shipped}</div>'
    return '<span class="no">—</span>'


def _delivered_when(r: dict) -> str:
    if not r["delivered"]:
        return '<span class="no">—</span>'
    ago = (dt.date.today() - r["delivered"]).days
    tag = ""
    if r["eta"]:
        d = (r["delivered"] - r["eta"]).days
        tag = (f' <span class="bad">{d}d late</span>' if d > 0 else
               f' <span class="good">{-d}d early</span>' if d < 0 else ' <span class="good">on time</span>')
    return (f'<b>{r["delivered"]}</b>{tag}'
            f'<div class="dim">{"today" if ago == 0 else f"{ago}d ago"}</div>')


def _lims_cell(r: dict) -> str:
    """The action column of the Delivered table: box is here, is it in LIMS?"""
    if r["n_waiting"]:
        return (f'<span class="v vneed">{r["n_waiting"]} not received</span>'
                '<span class="vsub">delivered, but still READY in LIMS — receive the plate</span>')
    if r["n_received"]:
        return ('<span class="v vgo">all received</span>'
                f'<span class="vsub">{r["n_received"]} parts stocked</span>')
    return '<span class="no">—</span>'


def _platemap_cell(r: dict, maps: dict) -> str:
    out = []
    for s in r["ships"]:
        key = f'{r["q"]}|{s.get("id")}'
        m = maps.get(key)
        if not m:
            continue
        cont = s.get("containers") or []
        plate = next((c.get("barcode") for c in cont if c.get("barcode")), "")
        out.append(
            f'<button class="dl" onclick="twistCSV(\'{_esc(key)}\',event)">&#8595; Plate map</button>'
            + (f'<div class="dim mono">{_esc(plate)}</div>' if plate else ""))
    return "".join(out) or '<span class="no">—</span>'


def _detail_row(r: dict, rid: str, ncols: int) -> str:
    """Hidden per-part row: stock id, attempt, LIMS status, and — once a plate map exists —
    the physical well, QC and yield straight from Twist's CSV."""
    if not r["parts"]:
        return ""
    head = ('<tr><th>Part</th><th>Attempt</th><th>LIMS</th>'
            '<th>Plate · well</th><th>QC</th><th>Yield (ng)</th><th>Length</th></tr>')
    body = []
    for p in r["parts"]:
        w = r["wells"].get(p["stock_id"]) or {}
        vs = p["vis_status"]
        st = ('<span class="bad">waiting</span>' if vs == "READY" else
              '<span class="good">received</span>' if vs == "SUCCEEDED" else
              f'<span class="dim">{_esc(vs or "—")}</span>')
        att = (_badge(f'↻ {p["attempt"]}', "amber") if p["attempt"] > 1 else str(p["attempt"]))
        chg = r["changes"].get(p["stock_id"])
        qc = " · ".join(x for x in (w.get("asm_qc"), w.get("yield_qc")) if x)
        qc_tone = "good" if qc and "fail" not in qc.lower() else ("bad" if qc else "")
        loc = f'{w.get("plate", "")} <b>{w.get("well", "")}</b>' if w.get("well") else '<span class="no">—</span>'
        body.append(
            f'<tr><td class="mono">{_esc(p["stock_id"])}'
            + (f'<div class="dim">{_esc(p["construct"])}</div>' if p["construct"] else "")
            + (f'<div class="amberv">{_esc(chg)}</div>' if chg else "")
            + f'</td><td>{att}</td><td>{st}</td>'
            f'<td class="mono">{loc}</td>'
            f'<td class="{qc_tone}">{_esc(qc) or "—"}</td>'
            f'<td>{_esc(w.get("yield") or "—")}</td>'
            f'<td class="dim">{_esc(w.get("bp") or "—")}</td></tr>')
    return (f'<tr class="detrow" id="{rid}" style="display:none"><td colspan="{ncols}">'
            f'<table class="det">{head}{"".join(body)}</table></td></tr>')


# ── tables ────────────────────────────────────────────────────────────────────
def _table(rows: list, cols: list, cells, empty: str, maps: dict, tag: str) -> str:
    if not rows:
        return f'<div class="empty">{_esc(empty)}</div>'
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = []
    for i, r in enumerate(rows):
        rid = f"tw-{tag}-{i}"
        tds = "".join(f"<td>{c}</td>" for c in cells(r, rid, maps))
        body.append(f'<tr class="main" onclick="twistToggle(\'{rid}\')">{tds}</tr>')
        body.append(_detail_row(r, rid, len(cols)))
    return (f'<table class="twtbl"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _render() -> str:
    with open(_PKL, "rb") as fh:
        data = pickle.load(fh)

    orders = data.get("orders") or []
    by_order = data.get("parts_by_order") or {}
    maps = data.get("platemaps") or {}
    wells_all = data.get("wells_by_order") or {}
    changes = data.get("eta_changes") or {}
    notes = data.get("notes") or []

    rows = [_row(o, by_order.get(o.get("order_name"), []),
                 wells_all.get(o.get("order_name"), {}), changes)
            for o in orders]

    prog = sorted([r for r in rows if r["bucket"] == "in_progress"],
                  key=lambda r: (r["eta"] or dt.date.max, r["q"]))
    trans = sorted([r for r in rows if r["bucket"] == "in_transit"],
                   key=lambda r: (r["ordered"] or dt.date.min), reverse=True)
    deliv = sorted([r for r in rows if r["bucket"] == "delivered"],
                   key=lambda r: (r["delivered"] or dt.date.min), reverse=True)

    t_prog = _table(
        prog, ["Order", "Parts waiting", "Twist progress", "ETA", "Status"],
        lambda r, rid, m: [_order_cell(r), _parts_cell(r, rid), _progress_cell(r),
                           _eta_cell(r), _badge(r["status"] or "—", "blue")],
        "Nothing in synthesis right now.", maps, "p")

    t_trans = _table(
        trans, ["Order", "Parts waiting", "Carrier", "Arriving", "Plate map"],
        lambda r, rid, m: [_order_cell(r), _parts_cell(r, rid), _ship_cell(r, ("shipped",)),
                           _transit_when(r), _platemap_cell(r, m)],
        "Nothing in transit.", maps, "t")

    t_deliv = _table(
        deliv, ["Order", "Parts", "Delivered", "Received into LIMS", "Carrier", "Plate map"],
        lambda r, rid, m: [_order_cell(r), _parts_cell(r, rid), _delivered_when(r),
                           _lims_cell(r), _ship_cell(r, ("received",)), _platemap_cell(r, m)],
        "No deliveries in the window.", maps, "d")

    n_wait_parts = sum(r["n_waiting"] for r in rows)
    n_unreceived = sum(r["n_waiting"] for r in deliv)
    overdue = sum(1 for r in prog if r["eta"] and r["eta"] < dt.date.today())

    # Data stamp + staleness. The Twist pull has its own cron, so it can be stale while the
    # rest of the dashboard is fresh — say so rather than let it read as live.
    gen = data.get("generated_at")
    try:
        gen_et = gen.astimezone(_ET)
        age_h = (dt.datetime.now(tz=dt.timezone.utc) - gen).total_seconds() / 3600
        stamp = gen_et.strftime("%Y-%m-%d %-I:%M %p ET")
    except Exception:
        age_h, stamp = 0.0, str(gen)
    stale = (f'<span class="stale">stale — {int(age_h)}h old</span>' if age_h > _STALE_HOURS else "")
    note_html = ("".join(f'<div class="note">{_esc(n)}</div>' for n in notes)
                 + (f'<div class="note">Delivered orders drop off after '
                    f'{data.get("window_days", 45)} days.</div>'))

    csv_map = {k: v.get("csv", "") for k, v in maps.items()}
    name_map = {k: v.get("filename", "platemap.csv") for k, v in maps.items()}
    payload = json.dumps({"csv": csv_map, "name": name_map}).replace("</", "<\\/")

    return _CSS + f"""
<div class="wrap">
  <div class="stamp">DATA PULLED {_esc(stamp)} {stale}</div>
  <h2>Twist orders — what we are waiting on</h2>
  <p class="sub">Every Twist order with a pipeline part on it, split by where the DNA physically is.
  ETAs come from Twist; once a shipment exists the shipment dates are authoritative, because Twist
  never reconciles the order-level ETA after shipping. Click any row for the parts on that order —
  after delivery each one shows its plate, well, QC and yield from Twist's own plate map.</p>
  <div class="ov">
    <div class="ovc"><div class="ovn">{n_wait_parts}</div><div class="ovl">Parts awaiting delivery</div></div>
    <div class="ovc"><div class="ovn">{len(prog)}</div><div class="ovl">Orders in progress</div></div>
    <div class="ovc"><div class="ovn">{len(trans)}</div><div class="ovl">In transit</div></div>
    <div class="ovc"><div class="ovn" style="color:{'#b91c1c' if n_unreceived else '#15803d'}">{n_unreceived}</div>
      <div class="ovl">Delivered, not in LIMS</div></div>
    <div class="ovc"><div class="ovn" style="color:{'#b91c1c' if overdue else '#15803d'}">{overdue}</div>
      <div class="ovl">Orders past ETA</div></div>
  </div>

  <h3 class="sec">In progress <span class="cnt">{len(prog)}</span></h3>
  {t_prog}

  <h3 class="sec">In transit <span class="cnt">{len(trans)}</span></h3>
  {t_trans}

  <h3 class="sec">Delivered <span class="cnt">{len(deliv)}</span></h3>
  {t_deliv}

  {note_html}
</div>
<script>
window.TWIST_PM = {payload};
function twistToggle(id) {{
  var el = document.getElementById(id);
  if (!el) return;
  el.style.display = (el.style.display === 'none') ? 'table-row' : 'none';
}}
function twistCSV(key, ev) {{
  if (ev) ev.stopPropagation();   // the row itself toggles the part detail
  var pm = window.TWIST_PM || {{}};
  var text = (pm.csv || {{}})[key];
  if (!text) {{ alert('No plate map cached for this shipment.'); return; }}
  var name = (pm.name || {{}})[key] || 'platemap.csv';
  var blob = new Blob([text], {{type: 'text/csv;charset=utf-8;'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(function() {{ URL.revokeObjectURL(a.href); a.remove(); }}, 0);
}}
</script>
"""


_CSS = """
<style>
#tab-twist { font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#1d1d1f; }
#tab-twist .wrap { padding:18px 22px 40px; }
#tab-twist h2 { font-size:19px; margin:0 0 3px; font-weight:700; }
#tab-twist .sub { font-size:12px; color:#6b7280; margin:0 0 14px; max-width:930px; line-height:1.55; }
#tab-twist .stamp { float:right; font-size:11px; font-weight:700; color:#1d4ed8; background:#eff6ff;
                    border:1px solid #bfdbfe; border-radius:6px; padding:4px 11px; }
#tab-twist .stale { color:#b45309; }
#tab-twist .ov { display:flex; gap:10px; margin:0 0 18px; flex-wrap:wrap; }
#tab-twist .ovc { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:10px 16px; min-width:110px; }
#tab-twist .ovn { font-size:22px; font-weight:800; line-height:1.1; }
#tab-twist .ovl { font-size:11px; color:#6b7280; margin-top:2px; }
#tab-twist .sec { font-size:13px; font-weight:700; margin:22px 0 7px; text-transform:uppercase;
                  letter-spacing:.05em; color:#374151; }
#tab-twist .cnt { font-size:11px; font-weight:700; color:#6b7280; background:#f3f4f6;
                  border-radius:9px; padding:1px 8px; margin-left:5px; }
#tab-twist .twtbl { width:100%; border-collapse:collapse; background:#fff; border:1px solid #e5e7eb;
                    border-radius:8px; overflow:hidden; }
#tab-twist .twtbl th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
                       color:#6b7280; background:#f9fafb; padding:8px 10px; border-bottom:1px solid #e5e7eb; }
#tab-twist .twtbl td { padding:9px 10px; border-bottom:1px solid #f3f4f6; vertical-align:top; font-size:12px; }
#tab-twist .twtbl tr.main { cursor:pointer; }
#tab-twist .twtbl tr.main:hover { background:#f9fafb; }
#tab-twist .q { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-weight:800; color:#7c3aed; }
#tab-twist .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
#tab-twist .dim { color:#9ca3af; font-size:11px; }
#tab-twist .no { color:#c4c4c6; }
#tab-twist .bad { color:#b91c1c; font-weight:600; }
#tab-twist .good { color:#15803d; font-weight:600; }
#tab-twist .amberv { color:#b45309; font-size:11px; font-weight:600; }
#tab-twist .tog { font-weight:600; cursor:pointer; border-bottom:1px dotted #c4c4c6; }
#tab-twist .caret { color:#9ca3af; font-size:10px; }
#tab-twist .bdg { display:inline-block; font-size:9px; font-weight:700; padding:2px 7px; border-radius:10px;
                  background:#f8fafc; color:#475569; border:1px solid #e2e8f0; white-space:nowrap; }
#tab-twist .bdg.blue { background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe; }
#tab-twist .bdg.amber { background:#fffbeb; color:#b45309; border-color:#fde68a; }
#tab-twist .bar { height:4px; background:#f3f4f6; border-radius:3px; margin-top:5px; max-width:150px; overflow:hidden; }
#tab-twist .bar span { display:block; height:100%; background:#1d4ed8; }
#tab-twist .carrier { font-size:10px; font-weight:700; color:#475569; text-transform:uppercase; margin-right:5px; }
#tab-twist .trk { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; color:#1d4ed8;
                  font-weight:600; text-decoration:none; }
#tab-twist .ship + .ship { margin-top:6px; padding-top:6px; border-top:1px solid #f3f4f6; }
#tab-twist .dl { font-size:11px; font-weight:600; color:#1d4ed8; background:#eff6ff; border:1px solid #bfdbfe;
                 border-radius:5px; padding:3px 9px; cursor:pointer; white-space:nowrap; }
#tab-twist .dl:hover { background:#dbeafe; }
#tab-twist .v { font-weight:700; display:block; }
#tab-twist .vsub { color:#6b7280; font-size:11px; display:block; margin-top:1px; }
#tab-twist .vgo { color:#15803d; }
#tab-twist .vneed { color:#b91c1c; }
#tab-twist .detrow td { background:#fbfbfd; padding:0 10px 10px; }
#tab-twist .det { width:100%; border-collapse:collapse; margin-top:2px; }
#tab-twist .det th { font-size:10px; text-transform:uppercase; letter-spacing:.04em; color:#9ca3af;
                     background:transparent; padding:6px 8px; border-bottom:1px solid #e5e7eb; text-align:left; }
#tab-twist .det td { padding:5px 8px; border-bottom:1px solid #f3f4f6; font-size:11px; }
#tab-twist .empty { padding:16px; color:#6b7280; background:#fff; border:1px solid #e5e7eb; border-radius:8px; }
#tab-twist .note { font-size:11px; color:#9ca3af; margin-top:10px; }
</style>
"""

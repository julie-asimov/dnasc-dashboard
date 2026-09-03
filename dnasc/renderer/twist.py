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
from urllib.parse import quote
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


# Twist hands back an EasyPost tracking link, which is their relabelled view of the carrier's
# data — a scan or two behind, and with none of the delivery controls. Send the tracking number
# to the carrier's own page instead, and keep EasyPost only as the fallback for a carrier we
# have not mapped (better a working link than none).
_CARRIER_TRACK = {
    "ups": "https://www.ups.com/track?loc=en_US&tracknum={}",
    "fedex": "https://www.fedex.com/fedextrack/?trknbr={}",
    "usps": "https://tools.usps.com/go/TrackConfirmAction?tLabels={}",
    "dhl": "https://www.dhl.com/en/express/tracking.html?AWB={}",
    "dhl_express": "https://www.dhl.com/en/express/tracking.html?AWB={}",
}


def _track_url(carrier: str, tracking: str, fallback: str) -> str:
    tpl = _CARRIER_TRACK.get(str(carrier or "").strip().lower())
    if tpl and tracking:
        return tpl.format(quote(str(tracking).strip(), safe=""))
    return fallback or ""


# ── row model ─────────────────────────────────────────────────────────────────
def _classify(order: dict, items: list = ()) -> str:
    """in_progress | in_transit | delivered — the LEAST-ADVANCED open work wins.

    Bucketed on physical state, not on `order.status`, which is 'past'/'open' bookkeeping
    that does not track the box.

    Unmade items count as open work, which shipments alone cannot tell you. Q-698815 has two
    delivered boxes and 176 of its 320 parts still in production, and reading shipments only
    it headlined the Delivered table — an order with weeks of synthesis left presented as
    finished. An order is only `delivered` once nothing is still being made.

    A partial order is therefore filed under its outstanding work, and `_parts_cell` badges it
    `partial` with the split so the boxes that already landed are not hidden.
    """
    ships = order.get("shipments") or []
    still_making = any(str(i.get("status")) == "in_production" for i in (items or ()))
    if any(s.get("status") == "shipped" for s in ships):
        return "in_transit"          # a box in flight outranks unmade parts: it needs receiving
    if still_making:
        return "in_progress"
    if any(s.get("status") == "received" for s in ships):
        return "delivered"
    return "in_progress"


_BAD_ITEM = ("failed", "cancelled")


def _row(order: dict, parts: list, wells: dict, changes: dict, items: list,
         glyc: dict) -> dict:
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
        "bucket": _classify(order, items),
        # Partial = some material has physically arrived while other parts are still being
        # made. The bucket alone cannot say that, so the row badges it and shows the split.
        "n_in_prod": sum(1 for i in items if str(i.get("status")) == "in_production"),
        "n_done": sum(1 for i in items
                      if str(i.get("status")) in ("closed", "completed")),
        "partial": bool(any(str(i.get("status")) == "in_production" for i in items)
                        and any(s.get("status") in ("shipped", "received") for s in ships)),
        # synpart | other. An order earns `synpart` only when a syn-part workorder in the
        # pipeline points at it. Everything else reads as `other`: orders with no pipeline
        # part at all, and plasmid-synthesis orders (which are pipeline work, but not
        # synparts — their rows still show their waiting count, they just group separately).
        "kind": "synpart" if any(p["kind"] == "synpart" for p in parts) else "other",
        "project": order.get("project_name") or "",
        "status": order.get("status") or "",
        "ordered": _date(order.get("received_date")),
        "eta": eta,
        "due": due,
        "delivered": delivered[-1] if delivered else None,
        "deliveries": delivered,   # ascending; a multi-box order has several
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
        # Glycerol stocks, kept out of `wells` at pull time because Twist ships them as their
        # own shipment naming the same parts — merged, they overwrote the DNA locations.
        "glyc": glyc,
        # Twist's own per-item view, from …/orders/<sfdc>/items/. Sorted so failures and
        # cancellations come first — the whole point of naming items is to see those.
        "items_list": sorted(items, key=lambda i: (str(i.get("status")) not in _BAD_ITEM,
                                                   str(i.get("name") or ""))),
        "n_bad": sum(1 for i in items if str(i.get("status")) in _BAD_ITEM),
        "n_failed": sum(1 for i in items if str(i.get("status")) == "failed"),
        "n_cancelled": sum(1 for i in items if str(i.get("status")) == "cancelled"),
        "n_delayed": sum(1 for i in items if i.get("delayed_status") == "DELAYED"),
        # Twist's delayed flag means "missed its promised date" and it STAYS SET after the
        # part ships, so a flat count conflates two very different situations: Q-698815 has
        # 176 of its 190 delayed parts still in production, while Q-698807 has 70 of its 102
        # already delivered, merely late. Only the still-open ones are actionable.
        "n_delayed_open": sum(1 for i in items if i.get("delayed_status") == "DELAYED"
                              and str(i.get("status")) == "in_production"),
        "max_redo": max((int(i.get("redo_count") or 0) for i in items), default=0),
        "redo_by_name": {str(i.get("name")): int(i.get("redo_count") or 0)
                         for i in items if (i.get("redo_count") or 0) > 0},
        "changes": {p["stock_id"]: changes[p["stock_id"]]
                    for p in parts if p["stock_id"] in changes},
    }


# ── cell builders ─────────────────────────────────────────────────────────────
def _order_cell(r: dict) -> str:
    bits = [f'<span class="q">{_esc(r["q"])}</span>',
            _badge("synpart", "blue") if r["kind"] == "synpart" else _badge("other")]
    if r["partial"]:
        bits.append(_badge("partial", "amber"))
    if r["high_priority"]:
        bits.append(_badge("rush", "amber"))
    proj = f'<div class="dim">{_esc(r["project"])}</div>' if r["project"] else ""

    # Turnaround once the order is in, age while it is not — the question this tab is actually
    # asked. On a multi-box order TAT is a RANGE, not a number: Q-693738 was ordered 07-22 and
    # delivered 07-29 and 07-31, so a bare "TAT 9d" hid the first box arriving on day 7.
    ord_d = ""
    if r["ordered"]:
        ord_d = f'<div class="dim">ordered {r["ordered"]}</div>'
        got = r["deliveries"]
        if r["bucket"] == "delivered" and got:
            lo = (got[0] - r["ordered"]).days
            hi = (got[-1] - r["ordered"]).days
            span = f"{lo}–{hi}d" if hi != lo else f"{hi}d"
            note = (f'<span class="tatsub">{len(got)} boxes</span>' if len(got) > 1 else "")
            ord_d += f'<div class="tat">TAT {span}{note}</div>'
        else:
            ord_d += (f'<div class="tat open">{(dt.date.today() - r["ordered"]).days}d'
                      '<span class="tatsub">open</span></div>')
    return " ".join(bits) + proj + ord_d


def _parts_cell(r: dict, rid: str) -> str:
    """Count + a toggle into the per-part detail row."""
    n = len(r["parts"])
    # Failures and delays are named by Twist per item, so surface the counts here — on the
    # order-level payload these were bare numbers with no way to learn WHICH part failed.
    # Name the two states separately. "1 failed / cancelled" on an order with one failure and
    # no cancellations read as two problems, and there was no way to tell which it was.
    flags = ""
    # On a partial order the split IS the story: how much is here versus still being made.
    if r["partial"]:
        flags += (f'<div class="dim">{r["n_done"]} arrived · '
                  f'<b>{r["n_in_prod"]}</b> still in production</div>')
    if r["n_failed"] or r["n_cancelled"]:
        parts_ = ([f'{r["n_failed"]} failed'] if r["n_failed"] else []) + \
                 ([f'{r["n_cancelled"]} cancelled'] if r["n_cancelled"] else [])
        flags += f'<div class="bad">{" · ".join(parts_)}</div>'
    if r["n_delayed_open"]:
        late_done = r["n_delayed"] - r["n_delayed_open"]
        flags += (f'<div class="amberv">{r["n_delayed_open"]} delayed, still in production'
                  + (f' <span class="dim">(+{late_done} arrived late)</span>'
                     if late_done > 0 else "") + '</div>')
    elif r["n_delayed"]:
        flags += f'<div class="dim">{r["n_delayed"]} arrived late</div>'

    if not n:
        # No pipeline workorder points here, but Twist still names every part on the order
        # via …/items/ — which works before shipping, unlike the plate map. "no pipeline
        # parts" read as an empty order when the order had contents worth seeing.
        vend = r["items_list"] or [{"name": k} for k in sorted(r["wells"])]
        if vend:
            return (f'<span class="tog" onclick="twistToggle(\'{rid}\')">{len(vend)} parts'
                    f' <span class="caret">▸</span></span>'
                    f'<div class="dim">named by Twist, not in LIMS</div>{flags}')
        return (f'<span class="dim">{r["items"]} items on order</span>{flags}' if r["items"]
                else '<span class="no">no pipeline parts</span>')

    lab = f'<b>{r["n_waiting"]}</b> waiting' if r["n_waiting"] else f'{n} parts'
    sub = []
    if r["n_received"]:
        sub.append(f'{r["n_received"]} received')
    if r["items"] and r["items"] != n:
        sub.append(f'Twist lists {r["items"]}')
    subs = f'<div class="dim">{_esc(" · ".join(sub))}</div>' if sub else ""
    return (f'<span class="tog" onclick="twistToggle(\'{rid}\')">{lab}'
            f' <span class="caret">▸</span></span>{subs}{flags}')


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


def _ship_cell(r: dict, want: tuple, maps: dict) -> str:
    """One block per shipment — carrier, tracking, last scan, and that shipment's plate map.

    The button lives INSIDE the block on purpose. It used to sit in its own column, which
    meant two independent stacks: on an order with two shipments the taller carrier blocks
    and the shorter button blocks drifted apart, so the second button lined up with the
    first shipment. Rendering the button as part of the shipment it belongs to makes that
    class of bug impossible.
    """
    out = []
    for s in r["ships"]:
        if s.get("status") not in want:
            continue
        carrier = (s.get("carrier") or "").upper()
        trk = s.get("tracking_number") or ""
        url = _track_url(s.get("carrier"), trk, s.get("tracking_url") or "")
        link = (f'<a href="{_esc(url)}" target="_blank" rel="noopener" class="trk">{_esc(trk)}</a>'
                if url and trk else _esc(trk))
        det = (s.get("status_detail") or {})
        prov = det.get("provider_detail") or det.get("provider_status") or ""
        loc = s.get("last_location") or ""
        scan = _d(s.get("last_updated_at"))

        # Plate barcode and its download sit on one line, so the barcode reads as the
        # button's label rather than a caption floating underneath it. The barcode still
        # shows when no map is cached — knowing the plate is useful even without the CSV.
        key = f'{r["q"]}|{s.get("id")}'
        m = maps.get(key)
        plate = next((c.get("barcode") for c in (s.get("containers") or []) if c.get("barcode")), "")
        pm = ""
        if m or plate:
            btn = (f'<button class="dl" onclick="twistCSV(\'{_esc(key)}\',event)">'
                   f'&#8595; Plate map</button>' if m else "")
            lbl = f'<span class="mono dim">{_esc(plate)}</span>' if plate else ""
            # Glycerol ships as its own shipment. Saying so on the block is the difference
            # between "why are there two plate maps" and "one is the DNA, one is the stab".
            tag = _badge("glycerol", "amber") if (m or {}).get("glycerol") else ""
            pm = f'<div class="pm">{btn}{lbl}{tag}</div>'

        # Each box's own delivery date lives in its own block, for the same reason the plate
        # map button does: on a multi-shipment order the Delivered column and this column are
        # separate stacks, so a reader cannot tell which date belongs to which tracking number.
        got = _date(s.get("received_at")) if s.get("status") == "received" else None

        out.append(
            f'<div class="ship"><span class="carrier">{_esc(carrier)}</span> {link}'
            + (f'<div class="dim">{_esc(str(prov).replace("_", " "))}</div>' if prov else "")
            + (f'<div class="dim">last scan {_esc(loc)}{" · " + scan if scan else ""}</div>' if loc else "")
            + (f'<div class="dim">delivered {got}</div>' if got else "")
            + pm + '</div>')
    return "".join(out) or '<span class="no">—</span>'


# The carrier's own words, relayed through EasyPost in status_detail. Any of these means the
# box is not on its original plan.
_DELAY_WORDS = ("delay", "exception", "failure", "return_to_sender", "damaged", "lost")


def _is_delayed(s: dict) -> bool:
    det = s.get("status_detail") or {}
    blob = " ".join(str(x or "").lower() for x in (det.get("provider_detail"),
                                                   det.get("provider_status"),
                                                   s.get("carrier_detail")))
    return any(w in blob for w in _DELAY_WORDS)


def _transit_when(r: dict) -> str:
    """Estimated arrival — unless the carrier says the shipment is delayed.

    `estimated_delivery_date` is Twist's number and Twist never revises it, so on a delayed
    shipment it kept rendering as fact: a box sitting in Louisville read "2026-09-02 arriving
    today" while UPS was already reporting a delay. The carrier's scan is the only live signal
    in this payload, so once it says delayed we lead with it and stop printing an arrival date
    the shipment is not going to hit. Twist's date stays visible as context, marked passed —
    the carrier does not publish a revised ETA through this API, so we do not invent one.
    """
    for s in r["ships"]:
        if s.get("status") != "shipped":
            continue
        est = _date(s.get("estimated_delivery_date"))
        shipped = _d(s.get("shipped_date"))
        ship_line = f'<div class="dim">shipped {shipped}</div>' if shipped else ""

        if _is_delayed(s):
            det = s.get("status_detail") or {}
            why = str(det.get("provider_detail") or det.get("provider_status") or "delayed")
            loc, scan = s.get("last_location") or "", _d(s.get("last_updated_at"))
            lines = [f'<b class="bad">DELAYED</b>'
                     f' <span class="carrier">{_esc((s.get("carrier") or "").upper())}</span>',
                     f'<div class="amberv">{_esc(why.replace("_", " "))}</div>']
            if loc or scan:
                lines.append(f'<div class="dim">last scan {_esc(loc)}'
                             f'{" · " + scan if scan else ""}</div>')
            lines.append('<div class="dim">no revised ETA from the carrier</div>')
            if est:
                passed = " · passed" if est < dt.date.today() else ""
                lines.append(f'<div class="dim">Twist ETA {est}{passed}</div>')
            return "".join(lines) + ship_line

        if not est:
            return ship_line or '<span class="no">—</span>'
        days = (est - dt.date.today()).days
        when = "arriving today" if days == 0 else (f"in {days}d" if days > 0 else f'<span class="bad">{-days}d overdue</span>')
        head = f'<b>{est}</b> <span class="dim">{when}</span>'

        # Twist's ETA has arrived but the feed has not said the box is out for delivery or
        # delivered — so "arriving today" is Twist's plan, not a carrier commitment. UPS can
        # be flagging a delay and a new date that this payload never carries (seen on
        # 1Z67R3110114943672: UPS said delayed, Twist still said in_transit, ETA unchanged).
        # Say the date is unconfirmed and send the reader to the carrier rather than assert
        # an arrival we cannot see.
        det = s.get("status_detail") or {}
        prov = str(det.get("provider_status") or "").lower()
        if days <= 0 and prov not in ("out_for_delivery", "delivered", "available_for_pickup"):
            head += ('<div class="amberv">unconfirmed — Twist has not reported it out for '
                     'delivery; check the carrier</div>')
        return head + ship_line
    return '<span class="no">—</span>'


def _delivered_when(r: dict) -> str:
    """EVERY received shipment's own delivery date, newest first — not just the last one.

    An order can arrive in several boxes on different days: Q-705566 landed in three, on
    08-27, 09-01 and 09-02, and showing only the newest read as though the whole order
    turned up that day while three tracking numbers sat in the next column.

    Each date is scored against THAT shipment's own estimated_delivery_date, not the
    order-level ETA. Twist never reconciles the order ETA after shipping, so scoring a
    box that arrived exactly when its carrier said against a stale order ETA invented an
    "early"/"late" that nobody promised.
    """
    got = sorted(((d, _date(s.get("estimated_delivery_date")))
                  for s in r["ships"] if s.get("status") == "received"
                  for d in [_date(s.get("received_at"))] if d),
                 key=lambda t: t[0], reverse=True)
    if not got:
        return '<span class="no">—</span>'

    lines = []
    for d, est in got:
        tag = ""
        if est:
            k = (d - est).days
            tag = (f' <span class="bad">{k}d late</span>' if k > 0 else
                   f' <span class="good">{-k}d early</span>' if k < 0 else
                   ' <span class="good">on time</span>')
        lines.append(f'<div><b>{d}</b>{tag}</div>')

    # No "Nd ago" line. How long ago a box landed is not a question this tab answers — the
    # turnaround is, and that lives in the Order column.
    foot = f'<div class="dim">{len(got)} shipments</div>' if len(got) > 1 else ""
    return "".join(lines) + foot


def _lims_cell(r: dict) -> str:
    """The action column of the Delivered table: box is here, is it in LIMS?"""
    if r["n_waiting"]:
        return (f'<span class="v vneed">{r["n_waiting"]} not received</span>'
                '<span class="vsub">delivered, but still READY in LIMS — receive the plate</span>')
    if r["n_received"]:
        return ('<span class="v vgo">all received</span>'
                f'<span class="vsub">{r["n_received"]} parts stocked</span>')
    return '<span class="no">—</span>'


def _loc_cell(w: dict) -> str:
    """Where the part physically is. Plate products give plate + well; tube products (clonal
    genes) give a tube id and no well, so printing only `well` showed "—" for a whole product
    line that Twist had in fact located."""
    plate, well = w.get("plate") or "", w.get("well") or ""
    if plate and well:
        return f'{_esc(plate)} <b>{_esc(well)}</b>'
    if plate:
        return _esc(plate)
    return '<span class="no">—</span>'


def _item_status(it: dict) -> str:
    st = str(it.get("status") or "")
    tone = ("bad" if st in _BAD_ITEM else
            "good" if st in ("completed", "closed") else "dim")
    out = f'<span class="{tone}">{_esc(st.replace("_", " ")) or "—"}</span>'
    if it.get("delayed_status") == "DELAYED":
        out += '<div class="amberv">delayed</div>'
    return out


def _retries_cell(p: dict, it: dict) -> str:
    """Retry count. TWO counters exist and only one of them ever moves for a vendor part.

    The pipeline's `resubmit_count` is 0 for all 8336 synthesis workorders in the baseline
    (it only ever moves on golden-gate/PCR/Gibson), so the old Attempt column could print
    nothing but "1". Twist's own `redo_count` is the counter that tracks VENDOR retries, and
    it disagrees: Q-693738 syn4500/syn4484 are on Twist redo 3 while the pipeline says
    attempt 1, and Q-698807 SRK-108-043 is on 37. Prefer Twist's, fall back to the pipeline's.
    """
    redo = int((it or {}).get("redo_count") or 0)
    if redo:
        return _badge(f"↻ {redo}", "amber") + '<div class="dim">Twist</div>'
    att = int((p or {}).get("attempt") or 0)
    if att > 1:
        return _badge(f"↻ {att}", "amber") + '<div class="dim">pipeline</div>'
    return '<span class="dim">0</span>'


def _detail_row(r: dict, rid: str, ncols: int) -> str:
    """Hidden per-part row — ONE table merging the pipeline's view with Twist's.

    These used to be two different tables picked by whether the order had a pipeline part: a
    pipeline one (Part / Attempt / LIMS) and a Twist one (Status / Progress / Retries). That
    made the visible facts depend on which GROUP an order landed in — when Q-715384 was
    correctly reclassified from `other` to `synpart`, its Twist status, its "3/5 Fragments
    assembled" progress and its retry counts all silently disappeared. Both sources exist for
    every order now, so every row shows both and says plainly when one side has nothing.
    """
    by_name = {str(i.get("name") or ""): i for i in r["items_list"]}
    if r["parts"]:
        rows = [(p["stock_id"], p, by_name.get(p["stock_id"]) or {}) for p in r["parts"]]
    elif r["items_list"]:
        rows = [(str(i.get("name") or ""), None, i) for i in r["items_list"]]
    else:
        rows = [(nm, None, {}) for nm in sorted(r["wells"])]
    if not rows:
        return ""

    # The Glycerol column only appears when the order actually has glycerol stocks — most
    # orders are DNA only, and an always-empty column just adds noise to read past.
    has_glyc = bool(r["glyc"])
    head = ('<tr><th>Part</th><th>LIMS</th><th>Twist status</th><th>Progress</th>'
            '<th>Retries</th><th>Plate / tube · well</th>'
            + ('<th>Glycerol</th>' if has_glyc else "")
            + '<th>QC</th><th>Yield (ng)</th><th>Length</th></tr>')
    body = []
    for nm, p, it in rows:
        w = r["wells"].get(nm) or {}

        if p:
            vs = p["vis_status"]
            lims = ('<span class="bad">waiting</span>' if vs == "READY" else
                    '<span class="good">received</span>' if vs == "SUCCEEDED" else
                    f'<span class="dim">{_esc(vs or "—")}</span>')
        else:
            lims = '<span class="dim">not in LIMS</span>'

        tw = _item_status(it) if it else '<span class="no">—</span>'
        # Twist reports a closed item's progress as the placeholder "0/0" / "Unknown". Printing
        # that verbatim looked like a data problem where there simply is no progress left.
        prog = str(it.get("progress") or "") if it else ""
        ev = (str(it.get("progress_event") or it.get("last_event") or "").replace("_", " ")
              if it else "")
        if prog in ("0/0", "/"):
            prog = ""
        if ev.strip().lower() == "unknown":
            ev = ""
        qc = " · ".join(x for x in (w.get("asm_qc"), w.get("yield_qc")) if x)
        qc_tone = "good" if qc and "fail" not in qc.lower() else ("bad" if qc else "")
        chg = r["changes"].get(nm)

        body.append(
            f'<tr><td class="mono">{_esc(nm)}'
            + (f'<div class="dim">{_esc(p["construct"])}</div>'
               if p and p.get("construct") else "")
            + (f'<div class="amberv">{_esc(chg)}</div>' if chg else "")
            + f'</td><td>{lims}</td><td>{tw}</td>'
            + f'<td>{_esc(prog) or "—"}'
            + (f'<div class="dim">{_esc(ev)}</div>' if ev else "")
            + '</td>'
            + f'<td>{_retries_cell(p, it)}</td>'
            + f'<td class="mono">{_loc_cell(w)}</td>'
            + (f'<td class="mono">{_loc_cell(r["glyc"].get(nm) or {})}</td>'
               if has_glyc else "")
            + f'<td class="{qc_tone}">{_esc(qc) or "—"}</td>'
            + f'<td>{_esc(w.get("yield") or "—")}</td>'
            + f'<td class="dim">{_esc(w.get("bp") or (it or {}).get("size") or "—")}</td></tr>')
    return (f'<tr class="detrow" id="{rid}" style="display:none"><td colspan="{ncols}">'
            f'<table class="det">{head}{"".join(body)}</table></td></tr>')


# ── tables ────────────────────────────────────────────────────────────────────
def _table(rows: list, cols: list, cells, empty: str, maps: dict, tag: str) -> str:
    """`cols` is [(label, width), …]. Widths are explicit because without them the browser
    hands all the slack to the last column, leaving a wide half-empty band on the right."""
    if not rows:
        return f'<div class="empty">{_esc(empty)}</div>'
    head = "".join(f'<th style="width:{w}">{_esc(c)}</th>' for c, w in cols)
    body = []
    for i, r in enumerate(rows):
        rid = f"tw-{tag}-{i}"
        det = _detail_row(r, rid, len(cols))
        tds = "".join(f"<td>{c}</td>" for c in cells(r, rid, maps))
        # Only a row that HAS a detail row gets the handler and the pointer cursor. An order
        # with no pipeline part and no plate map yet has nothing to expand, and a row that
        # invites a click and then does nothing reads as a broken dropdown.
        attrs = (f' class="main" onclick="twistToggle(\'{rid}\')"' if det else ' class="flat"')
        body.append(f'<tr{attrs}>{tds}</tr>')
        if det:
            body.append(det)
    return (f'<table class="twtbl"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


_PROG_COLS = [("Order", "24%"), ("Parts waiting", "14%"), ("Twist progress", "22%"),
              ("ETA", "26%"), ("Status", "14%")]
# In progress, when a PARTIAL order is present. Filing partials under their outstanding work
# is right, but this table had no shipment column at all — so Q-698815's two delivered boxes,
# their tracking numbers and its glycerol plate map disappeared from the tab completely. The
# extra column is added only when some row actually has material already in hand.
_PROG_COLS_PARTIAL = [("Order", "20%"), ("Parts waiting", "13%"), ("Twist progress", "17%"),
                      ("ETA", "21%"), ("Status", "9%"), ("Already arrived", "20%")]
_TRANS_COLS = [("Order", "24%"), ("Parts waiting", "14%"), ("Shipment", "40%"),
               ("Arriving", "22%")]
_DELIV_COLS = [("Order", "20%"), ("Parts", "12%"), ("Delivered", "18%"),
               ("Received into LIMS", "20%"), ("Shipment & plate map", "30%")]


def _state_tables(rows: list, maps: dict, tag: str, empties: tuple) -> tuple:
    """The three state tables for one group of orders, as (row lists, table html).

    Synparts and other orders render the same three tables rather than sharing one, so a
    group can be read on its own. `tag` keeps the detail-row ids unique between the two
    groups — colliding ids would make one group's expander open the other group's row.
    """
    prog = sorted([r for r in rows if r["bucket"] == "in_progress"],
                  key=lambda r: (r["eta"] or dt.date.max, r["q"]))
    trans = sorted([r for r in rows if r["bucket"] == "in_transit"],
                   key=lambda r: (r["ordered"] or dt.date.min), reverse=True)
    deliv = sorted([r for r in rows if r["bucket"] == "delivered"],
                   key=lambda r: (r["delivered"] or dt.date.min), reverse=True)

    any_partial = any(r["partial"] for r in prog)
    t_prog = _table(
        prog, _PROG_COLS_PARTIAL if any_partial else _PROG_COLS,
        lambda r, rid, m: [_order_cell(r), _parts_cell(r, rid), _progress_cell(r),
                           _eta_cell(r), _badge(r["status"] or "—", "blue")]
        + ([_ship_cell(r, ("shipped", "received"), m)] if any_partial else []),
        empties[0], maps, f"{tag}p")
    t_trans = _table(
        trans, _TRANS_COLS,
        lambda r, rid, m: [_order_cell(r), _parts_cell(r, rid),
                           _ship_cell(r, ("shipped",), m), _transit_when(r)],
        empties[1], maps, f"{tag}t")
    t_deliv = _table(
        deliv, _DELIV_COLS,
        lambda r, rid, m: [_order_cell(r), _parts_cell(r, rid), _delivered_when(r),
                           _lims_cell(r), _ship_cell(r, ("received",), m)],
        empties[2], maps, f"{tag}d")
    return (prog, trans, deliv), (t_prog, t_trans, t_deliv)


def _render() -> str:
    with open(_PKL, "rb") as fh:
        data = pickle.load(fh)

    orders = data.get("orders") or []
    by_order = data.get("parts_by_order") or {}
    maps = data.get("platemaps") or {}
    wells_all = data.get("wells_by_order") or {}
    changes = data.get("eta_changes") or {}
    notes = data.get("notes") or []

    items_all = data.get("items_by_order") or {}
    glyc_all = data.get("glycerol_by_order") or {}

    rows = [_row(o, by_order.get(o.get("order_name"), []),
                 wells_all.get(o.get("order_name"), {}), changes,
                 items_all.get(o.get("order_name"), []),
                 glyc_all.get(o.get("order_name"), {}))
            for o in orders]

    syn = [r for r in rows if r["kind"] == "synpart"]
    oth = [r for r in rows if r["kind"] != "synpart"]

    (prog, trans, deliv), (t_prog, t_trans, t_deliv) = _state_tables(
        syn, maps, "s", ("Nothing in synthesis right now.", "Nothing in transit.",
                         "No deliveries in the window."))
    (oprog, otrans, odeliv), (o_prog, o_trans, o_deliv) = _state_tables(
        oth, maps, "o", ("No other orders in synthesis.", "No other orders in transit.",
                         "No other deliveries in the window."))

    # The overview counts the synpart group only — those are the parts the pipeline is
    # waiting on, and that is what these tiles have always meant. Other orders get their
    # own tile so the section is not a surprise at the bottom of the page.
    n_wait_parts = sum(r["n_waiting"] for r in syn)
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
  <p class="sub">Every Twist order on this account, split by where the DNA physically is.
  <b>Synthesis parts</b> are the orders a pipeline syn-part workorder points at; <b>other orders</b>
  are everything else on the account. ETAs come from Twist; once a shipment exists the shipment
  dates are authoritative, because Twist never reconciles the order-level ETA after shipping — and
  when the carrier reports a delay, the last scan replaces the ETA rather than sitting under a date
  the box will not hit. Click any row for the parts on that order — after delivery each one shows
  its plate, well, QC and yield from Twist's own plate map.</p>
  <div class="ov">
    <div class="ovc"><div class="ovn">{n_wait_parts}</div><div class="ovl">Parts awaiting delivery</div></div>
    <div class="ovc"><div class="ovn">{len(prog)}</div><div class="ovl">Orders in progress</div></div>
    <div class="ovc"><div class="ovn">{len(trans)}</div><div class="ovl">In transit</div></div>
    <div class="ovc"><div class="ovn" style="color:{'#b91c1c' if n_unreceived else '#15803d'}">{n_unreceived}</div>
      <div class="ovl">Delivered, not in LIMS</div></div>
    <div class="ovc"><div class="ovn" style="color:{'#b91c1c' if overdue else '#15803d'}">{overdue}</div>
      <div class="ovl">Orders past ETA</div></div>
    <div class="ovc"><div class="ovn">{len(oth)}</div><div class="ovl">Other account orders</div></div>
  </div>

  <h2 class="grp">Synthesis parts <span class="cnt">{len(syn)}</span></h2>

  <h3 class="sec">In progress <span class="cnt">{len(prog)}</span></h3>
  {t_prog}

  <h3 class="sec">In transit <span class="cnt">{len(trans)}</span></h3>
  {t_trans}

  <h3 class="sec">Delivered <span class="cnt">{len(deliv)}</span></h3>
  {t_deliv}

  <h2 class="grp">Other orders on this account <span class="cnt">{len(oth)}</span></h2>
  <p class="sub">No pipeline syn-part points at these — oligos, genes, and other teams' orders.
  Same tracking, kept separate so the synthesis tables above stay the first thing you read.</p>

  <h3 class="sec">In progress <span class="cnt">{len(oprog)}</span></h3>
  {o_prog}

  <h3 class="sec">In transit <span class="cnt">{len(otrans)}</span></h3>
  {o_trans}

  <h3 class="sec">Delivered <span class="cnt">{len(odeliv)}</span></h3>
  {o_deliv}

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
/* Group heading. The two groups each carry three state tables, so the group needs to outrank
   the state headings visually or the page reads as six peer sections. */
#tab-twist .grp { font-size:17px; font-weight:800; color:#111827; margin:34px 0 2px;
                  padding-bottom:7px; border-bottom:2px solid #e5e7eb; }
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
/* Turnaround. Deliberately the loudest thing in the Order cell — it is the number the tab
   gets opened for, and as dim 11px text beside the order date nobody could find it. */
#tab-twist .tat { display:inline-block; margin-top:4px; font-size:13px; font-weight:800;
                  color:#1d4ed8; background:#eff6ff; border:1px solid #bfdbfe;
                  border-radius:6px; padding:2px 8px; letter-spacing:.01em; }
#tab-twist .tat.open { color:#b45309; background:#fffbeb; border-color:#fde68a; }
#tab-twist .tatsub { font-size:10px; font-weight:600; opacity:.75; margin-left:5px; }
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
#tab-twist .ship + .ship { margin-top:8px; padding-top:8px; border-top:1px solid #f3f4f6; }
#tab-twist .pm { display:flex; align-items:center; gap:7px; margin-top:5px; flex-wrap:wrap; }
#tab-twist .pm .mono { font-size:11px; }
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

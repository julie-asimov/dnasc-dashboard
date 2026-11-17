#!/usr/bin/env python3
"""Refresh the Twist order-tracking cache used by the dashboard's Twist tab.

Standalone pull on its own cron, exactly like gen_parts_pkl.py — the dashboard render
NEVER calls the Twist API. Writes dashboard_state/twist_result.pkl ATOMICALLY (temp file
+ os.replace) so the renderer always sees either the previous good pkl or the complete
new one.

ORDER TRACKING API (migrated 2026-11-06, v1.11.144)
----------------------------------------------------
The old `GET {twist-api.twistdna.com}/v1/users/<email>/orders/` LIST endpoint is BROKEN
vendor-side and has been since ~2026-09-10. Twist's gateway answers it with HTTP 500
`ReadTimeout` / `502 Bad Gateway` from `mes-prod.twistbioscience.com`: measured 1 success
in 13 attempts (~8%) on 2026-11-06, identical from the Mac and from the server, so it is
not egress and not the tokens. The damage was invisible because this script correctly
keeps the previous cache rather than caching an empty pull — twist_pull.log shows 925 runs
and 122 successes, 118 of them in 2026-09-01..09, and the tab silently served an Oct 30
snapshot for a week while *looking* live.

Tiffany Dai at Twist (2026-09-19) confirmed the investigation is open and pointed at the
replacement: https://app.swaggerhub.com/apis-docs/TwistDNA/synbio-order-intake/1.0.0

So the order list, order detail and plate maps now come from the Order Tracking API, which
is a different host and sub-second instead of ~75 s a page:

    /orders/summary          0.9 s   every order on the account (633 = 7 pages of 100)
    /orders/<Q-number>       ~1 s    dates, item statuses + ETAs, shipments, counters
    /orders/<Q>/collateral   ~6 s    plate maps as JSON — plateId, wellLocation, yield, QC

READ-ONLY. The same API can place real, billable orders (`POST /v1/orders/<draft-id>`).
This script issues GETs and nothing else; never add a write call to it.

What still comes from the OLD host, because it works and the new API does not carry it:
  • `/orders/<sfdc_id>/items/` (~2-3 s) — per-item `redo_count` and `delayed_status`,
    Twist's own synthesis-delay flag. The new `TrackingOrderItem` has neither.
  • the order LIST, on ONE bounded best-effort attempt per run, purely to harvest
    `project_name` (see `_scrape_project_names`) — the only field with no replacement.

Deliberate losses, all of them recorded in `notes` so the tab says so rather than
implying Twist sent nothing:
  • a shipment's DELIVERY timestamp. The new payload has no `received_at`, so a shipment
    marked `received` reports its SHIP date and is flagged `received_at_approx`.
  • `status_detail` / `carrier_detail` / `last_updated_at` / per-shipment
    `estimated_delivery_date`. `_is_delayed()` in the renderer reads `status_detail` and
    will simply never fire now; `lastLocation` survives and is the live scan signal.

`--days` (default 45) bounds DELIVERED orders only, measured from the DELIVERY date — the
latest shipment's date — not from when the order was placed. OPEN orders are never aged
out; the longer one has been open the more it needs looking at, so `_open_days` is stamped
on each for the tab to flag.

Summaries are cheap enough to take whole (3.7 s for all 633), so paging is no longer the
cost. DETAIL calls are, at ~1 s each, so the walk stops once the window is covered AND
every Q-number the pipeline is still waiting on has been seen — the same early-stop rule
as before, now measured in orders rather than pages. `--deadline` remains a hard
wall-clock stop so an hourly cron can never still be running when the next one fires.

Env:
    AUTHORIZATION_JWT   (required)
    X_END_USER_TOKEN    (required only for the two OLD-host calls above)
    TWIST_EMAIL         (default dna@asimov.io)

Usage:
    /opt/anaconda3/bin/python3 gen_twist_pkl.py
    /opt/anaconda3/bin/python3 gen_twist_pkl.py --days 90
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
import os
import pickle
import sys
import tempfile
import time
import traceback

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from twist_orders import (  # noqa: E402  (path juggling above is deliberate)
    BASE_URL,
    ETA_LOG,
    PLATEMAP_DIR,
    _append_eta_log,
    _headers,
    _load_eta_log,
)

OUT = os.path.join(HERE, "dashboard_state", "twist_result.pkl")
PARQUET = os.path.join(HERE, "dashboard_state", "baseline.parquet")
# Project names live nowhere in the new API, so the few we manage to scrape off the old
# one are kept forever. An order's project never changes, so one lucky hit is permanent.
PROJECTS = os.path.join(HERE, "dashboard_state", "twist_project_names.json")

# Workorder types whose vendor_order_id is a Twist Q-number.
_SYNTH_TYPES = ["syn_part_synthesis_workorder", "plasmid_synthesis_workorder"]

# New Order Tracking API. Host and prefix differ from BASE_URL (the old host), and auth is
# a bearer JWT with no end-user token.
NEW_BASE = "https://twist-api.prod.twistbio.co/ecommerce-synbio-intake/v1"

_SUMMARY_SIZE = 100     # max the endpoint accepts; the whole account is ~7 pages
_TIMEOUT = 120
_DEADLINE_SECONDS = 900

# Measured 2026-11-06. Used to decide whether there is budget left to start a call rather
# than discovering mid-flight that there was not.
_DETAIL_COST = 5
_COLLATERAL_COST = 20   # ~6 s typical, but a 160-item order 504s after ~30 s
# ── pipeline side ─────────────────────────────────────────────────────────────
def _pipeline_parts(parquet: str) -> tuple[dict, set]:
    """{Q-number: [part dicts]}, {Q-numbers with a part still waiting on delivery}.

    READY on a synthesis workorder means "ordered, not yet received into LIMS" — that is
    the set the tab exists to watch. Parts already SUCCEEDED are kept too so a delivered
    order can show what came in, but they never keep an order on screen by themselves.
    """
    df = pd.read_parquet(parquet, columns=[
        "STOCK_ID", "type", "vendor_order_id", "visual_status", "wo_status",
        "resubmit_count", "construct_name", "wo_updated_at",
    ])
    df["vendor_order_id"] = df["vendor_order_id"].astype(str).str.strip()
    synth = df[df["type"].isin(_SYNTH_TYPES)].copy()

    # Deterministic dedupe. 1222 of 2084 synthesis STOCK_IDs have more than one baseline row,
    # and on 72 of them a CANCELED attempt coexists with a SUCCEEDED one — so the old
    # drop_duplicates(keep=first) picked by row order and reported CANCELED for parts that
    # had in fact succeeded (syn1683, syn1684, pAI-21454, pAI-21720 all read CANCELED).
    # Rank by how far the workorder actually got, then by recency.
    synth["_rank"] = synth["visual_status"].map(
        {"SUCCEEDED": 0, "RUNNING": 1, "READY": 2, "CANCELED": 3}).fillna(4)
    synth = (synth.sort_values(["_rank", "wo_updated_at"], ascending=[True, False])
                  .drop_duplicates("STOCK_ID"))

    def _part(r) -> dict:
        return {
            "stock_id": str(r.STOCK_ID),
            "kind": "synpart" if "syn_part" in str(r.type) else "plasmid",
            "attempt": int(r.resubmit_count or 0) + 1,
            "vis_status": str(r.visual_status or ""),
            "wo_status": str(r.wo_status or ""),
            "construct": str(r.construct_name or ""),
        }

    # Keyed by STOCK_ID over EVERY synthesis workorder, not just the ones carrying a
    # Q-number. 731 of 8044 syn-part workorders have a blank vendor_order_id, so an order
    # whose parts have not been stamped yet is invisible to the by_order map — Q-715384's 77
    # parts all exist in the pipeline but none of them names its order. Twist's item list
    # names them, so this map lets the order be recovered by part name instead.
    by_stock = {str(r.STOCK_ID): _part(r) for r in synth.itertuples(index=False)}

    by_order: dict[str, list[dict]] = {}
    for r in synth[synth["vendor_order_id"].str.startswith("Q-", na=False)].itertuples(index=False):
        by_order.setdefault(r.vendor_order_id, []).append(_part(r))

    waiting = {q for q, parts in by_order.items()
               if any(p["vis_status"] == "READY" for p in parts)}
    return by_order, waiting, by_stock


# ── Twist side: the new Order Tracking API ────────────────────────────────────
_RETRIES = 3
_RETRY_WAIT = 5         # the new endpoints answer in ~1 s, so a tight retry is sensible
_RETRY_ON = (500, 502, 503, 504)


def _bearer(jwt: str) -> dict:
    """Headers for the new API. It takes a BEARER jwt and no end-user token — unlike
    `_headers()`, which is the old host's `JWT <tok>` + `X-End-User-Token` pair. Passing
    the old header shape here authenticates nothing."""
    return {"accept": "*/*", "Authorization": f"Bearer {jwt}"}


def _get_json(url, jwt, left, params=None, cost=10, label="", retries=_RETRIES):
    """GET → parsed JSON, or None.

    Retries 5xx because the vendor's gateway still blips (Q-698807's collateral 504s
    reproducibly — ~160 items with full sequences is apparently too much for it to
    assemble in time). Each request is capped by what is left of the pull's budget, so
    one slow call cannot overrun the deadline on its own, and a call is not started at
    all unless `cost` seconds remain.
    """
    for attempt in range(1, retries + 1):
        if left() < cost:
            return None
        tmo = max(5, min(_TIMEOUT, int(left())))
        try:
            r = requests.get(url, headers=_bearer(jwt), params=params, timeout=tmo)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in _RETRY_ON:
                if label:
                    body = " ".join((r.text or "").split())[:160]
                    print(f"    {label}: HTTP {r.status_code} — {body!r}", flush=True)
                return None
            why = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            why = type(e).__name__
        if attempt == retries:
            if label:
                print(f"    {label}: {why} — giving up after {attempt} attempt(s)",
                      flush=True)
            return None
        time.sleep(_RETRY_WAIT)
    return None


def _fetch_summaries(email, jwt, left) -> list[dict]:
    """Every order on the account, newest first.

    The whole history is 633 orders = 7 pages of 100, measured at 3.7 s total, so there is
    no reason to window this call — the expensive half is the per-order detail below.
    Ordering is by Q-number descending, which is chronological: the list opens with
    Q-749793 and ends at Q-52134.

    `fromDate` is required-ish in practice; 2015 is simply "everything".
    """
    out, page = [], 1
    while True:
        j = _get_json(f"{NEW_BASE}/users/{email}/orders/summary", jwt, left,
                      params={"fromDate": "2015-01-01T00:00:00Z",
                              "toDate": "2099-12-31T23:59:59Z",
                              "page": page, "size": _SUMMARY_SIZE},
                      cost=5, label=f"summary page {page}")
        if not j:
            break
        out.extend(j.get("orders") or [])
        if page >= int(j.get("total_pages") or 1):
            break
        page += 1
    return out


def _fetch_detail(email, jwt, q: str, left) -> dict | None:
    """One order: dates, per-item status and ETA, shipments, counters. ~1 s."""
    return _get_json(f"{NEW_BASE}/users/{email}/orders/{q}", jwt, left,
                     cost=_DETAIL_COST, label=f"detail {q}")


# Twist's summary vocabulary → the old list endpoint's 'open'/'past' bookkeeping, which is
# what `_is_open()` and the renderer have always read. 'In production' is the only live
# state; 'Cancelled' is finished work that never shipped, so it files as 'past'.
_OPEN_STATUSES = ("in production",)


def _ship_status(raw: str) -> str:
    """New shipment status → the old vocabulary the renderer buckets on ('shipped' =
    in flight, 'received' = landed). Anything unrecognised passes through untouched
    rather than being guessed into one of the two — a wrong bucket moves an order to the
    wrong table, a blank one merely fails to move it."""
    s = str(raw or "").strip().lower()
    if s in ("received", "delivered"):
        return "received"
    if s in ("shipped", "in_transit", "in transit"):
        return "shipped"
    return s


def _legacy_order(summary: dict, detail: dict, project: str) -> dict:
    """New-API order → the dict shape the renderer has always consumed.

    Deliberately a translation layer rather than a renderer rewrite: dnasc/renderer/
    twist.py reads ~20 snake_case fields off these dicts and is the part of this feature
    with the most behaviour in it. Keeping the contract means the migration is one file.

    Two fields change meaning and are commented where they are set: the order-level ETA
    (now derived from item ETAs, because the new API has no order-level one) and a
    shipment's `received_at` (which the new API does not report at all).
    """
    info = detail.get("info") or {}
    items = detail.get("orderItems") or []
    raw_status = str(summary.get("orderStatus") or "").strip().lower()
    is_open = raw_status in _OPEN_STATUSES

    # Order ETA. The old API carried `estimated_completion_date` per order; the new one
    # carries `estimatedShippingDate` per ITEM. The honest order-level answer is the LAST
    # outstanding item's date — that is when the order is actually done — so prefer the
    # max over items still being made, and fall back to the max over all of them once
    # nothing is outstanding.
    def _eta(pool):
        return max((str(i.get("estimatedShippingDate") or "")[:10] for i in pool
                    if i.get("estimatedShippingDate")), default="")
    eta = _eta([i for i in items if str(i.get("status")) == "in_production"]) or _eta(items)

    ships = []
    for s in (detail.get("shipments") or []):
        st = _ship_status(s.get("status"))
        shipped = str(s.get("shippedDate") or "")
        ships.append({
            "id": s.get("id"),
            "number": s.get("number"),
            "status": st,
            "shipped_date": shipped,
            # NO DELIVERY TIMESTAMP EXISTS in this API. A shipment marked received did
            # land, so reporting its ship date keeps the Arrived column alive, but it is
            # off by the transit time (typically a day) and must never be read as the
            # delivery date. The flag is here so a renderer can label it; `notes` says so
            # on the tab either way.
            "received_at": shipped if st == "received" else "",
            "received_at_approx": st == "received",
            "carrier": s.get("carrier") or "",
            "tracking_number": s.get("trackingNumber") or "",
            "tracking_url": s.get("trackingUrl") or "",
            "last_location": s.get("lastLocation") or "",
            # Absent from the new payload. Left empty rather than faked: `_is_delayed()`
            # reads status_detail and will correctly never fire instead of inventing a
            # verdict from a field that no longer exists.
            "status_detail": {},
            "carrier_detail": "",
            "last_updated_at": "",
            "estimated_delivery_date": "",
            "containers": s.get("containers") or [],
        })

    return {
        "order_name": detail.get("quoteNumber") or summary.get("orderId") or "",
        "sfdc_id": summary.get("salesId") or detail.get("salesId") or "",
        "status": "open" if is_open else "past",
        "order_status_raw": summary.get("orderStatus") or "",
        "project_name": project or "",
        "received_date": str(detail.get("receivedDate") or ""),
        "created_date": str(detail.get("receivedDate") or ""),
        "estimated_completion_date": eta,
        "due_date": "",
        "high_priority": False,
        "type": detail.get("type") or "",
        "container_code": detail.get("containerType") or "",
        "total_items": info.get("totalItems") or 0,
        "in_production_items": info.get("inProductionItems") or 0,
        "failed_items": info.get("failedItems") or 0,
        # `completed` and `closed` are SUMMED by the renderer (done = completed + closed),
        # so the delivered count goes in exactly one of them.
        "completed_items": info.get("deliveredItems") or 0,
        "closed_items": 0,
        "cancelled_items": (info.get("totalItems") or 0) if raw_status == "cancelled" else 0,
        "total_amount": summary.get("totalAmount"),
        "currency_iso_code": summary.get("currency") or "",
        "shipments": ships,
    }


# ── project names: the one field with no replacement ──────────────────────────
def _load_projects() -> dict:
    try:
        with open(PROJECTS) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_projects(names: dict) -> None:
    try:
        d = os.path.dirname(PROJECTS)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".twist_projects.", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(names, fh, indent=1, sort_keys=True)
        os.replace(tmp, PROJECTS)
    except Exception:
        traceback.print_exc()


def _seed_projects_from_pkl(names: dict) -> int:
    """First run only: lift project names out of the last good pkl.

    That cache was written while the old list endpoint still worked, so it holds a real
    `project_name` for every order it covered. Without this the names would have to be
    re-won one ~8%-successful scrape at a time, and the tab would read as a wall of bare
    Q-numbers until that happened.
    """
    if names:
        return 0
    try:
        with open(OUT, "rb") as fh:
            prev = pickle.load(fh)
    except Exception:
        return 0
    got = 0
    for o in prev.get("orders") or []:
        q, p = o.get("order_name"), (o.get("project_name") or "").strip()
        if q and p:
            names[q] = p
            got += 1
    return got


def _scrape_project_names(email, jwt, eut, names: dict, left) -> str:
    """ONE bounded best-effort call to the OLD list endpoint, purely for `project_name`.

    `project_name` ("A824_upsilon_syn_parts", "LND004_clonal") exists nowhere in the new
    API — not on the summary, not on the order detail, not on the quote — and it is what
    makes the tab readable at a glance. The old LIST endpoint is the only source, and it
    answers about 8% of the time.

    So: try once, cheaply, and treat success as a bonus. Names are written through to
    PROJECTS and kept forever, because an order's project never changes — every lucky hit
    permanently improves the tab. Never let this block the pull; the run is already
    complete without it.
    """
    if left() < 130:
        return "project-name scrape skipped — not enough budget left"
    try:
        r = requests.get(f"{BASE_URL}/v1/users/{email}/orders/",
                         headers=_headers(jwt, eut), timeout=min(130, int(left())),
                         params={"page_size": 25, "sort_by": "received_date",
                                 "reverse": "true", "page": 1})
    except requests.RequestException as e:
        return f"project-name scrape failed ({type(e).__name__}) — using {len(names)} cached"
    if r.status_code != 200:
        return (f"project-name scrape failed (HTTP {r.status_code}, the known-broken old "
                f"endpoint) — using {len(names)} cached")
    fresh = 0
    for o in (r.json().get("results") or []):
        q, p = o.get("order_name"), (o.get("project_name") or "").strip()
        if q and p and names.get(q) != p:
            names[q] = p
            fresh += 1
    return f"project-name scrape OK — {fresh} new, {len(names)} known"

def _delivered_on(order: dict) -> str:
    """ISO date the order actually LANDED, or "" if nothing has arrived yet.

    `received_at` is when the box was logged as received and is the honest answer;
    `shipped_date` is the fallback for a shipment still in transit. Same precedence the
    plate-map loop already uses for `ship_at`, so the two never disagree.

    An order can ship in pieces (Q-705566 has three shipments). The LATEST one is what
    matters for a drop-off window — the order is not finished with us until the last box
    is in, so keying off the first would retire it while parts were still arriving.
    """
    dates = [str(s.get("received_at") or s.get("shipped_date") or "")[:10]
             for s in (order.get("shipments") or [])
             if s.get("status") in ("shipped", "received")]
    return max((d for d in dates if d), default="")


def _is_open(order: dict) -> bool:
    """Twist's own verdict: status is 'open' until the order is done, then 'past'.

    Deliberately not derived from item counts — a partially shipped order (Q-698807: two
    shipments, still open) is exactly the case we must keep showing, and counting
    completed items would retire it early.
    """
    return str(order.get("status") or "").lower() == "open"


# ── plate maps: now JSON, not a CSV download ──────────────────────────────────
# Sequence fields, dropped from everything we keep. On a clonal-gene order these are
# ~95% of the payload (Q-698807's map was 4.4 MB of 4.6 MB as CSV), the tab renders none
# of them, and the cached copy is embedded verbatim in the dashboard HTML — so every
# viewer downloads them on every page load.
_DROP_KEYS = ("insertSequence", "constructSequence", "vectorSequence")

# JSON key → the dict key the renderer reads. Product lines still disagree about names,
# exactly as the six CSV layouts used to: gene fragments carry `targetYieldQc` and no NGS
# verdict, clonal genes carry `ngsQc` + `yieldQc` + `constructLength`. First key wins.
_WELL_FIELDS = {
    "plate":    ("plateId",),
    "well":     ("wellLocation",),
    "bp":       ("insertLength", "constructLength"),
    "asm_qc":   ("ngsQc",),
    "yield_qc": ("yieldQc", "targetYieldQc"),
    "yield":    ("actualYieldNg",),
    "product":  ("productType",),
}

# Column order for the synthesised CSV behind the ↓ Plate map button. Header names match
# what Twist's own CSV used, so a file saved today still opens like the ones already on
# disk in twist_platemaps/.
_CSV_COLS = (("Name", ("name",)), ("Plate ID", ("plateId",)),
             ("Well Location", ("wellLocation",)), ("Insert Length", ("insertLength",)),
             ("Construct Length", ("constructLength",)), ("NGS QC", ("ngsQc",)),
             ("Yield QC", ("yieldQc",)), ("Target Yield QC", ("targetYieldQc",)),
             ("Actual Yield (ng)", ("actualYieldNg",)), ("Product Type", ("productType",)),
             ("Custom Label", ("customLabel",)))


def _first(row: dict, keys) -> str:
    for k in keys:
        v = row.get(k)
        if v not in (None, "", "None"):
            return str(v)
    return ""


def _rows_to_csv(rows: list) -> str:
    """Plate-map items → a CSV for the download button, sequences excluded.

    The tab's ↓ Plate map button hands the browser a Blob built from this text. Twist no
    longer serves a file to pass through, so one is built from the JSON — which also ends
    the old ZIP-vs-CSV magic-byte sniffing and the six-header-layout merge, since every
    product line now arrives as the same parsed objects.
    """
    cols = [c for c, keys in _CSV_COLS if any(_first(r, keys) for r in rows)]
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(cols)
    lookup = dict(_CSV_COLS)
    for r in rows:
        wr.writerow([_first(r, lookup[c]) for c in cols])
    return buf.getvalue()


def _fetch_collateral(email, jwt, q: str, left) -> list:
    """`/orders/<Q>/collateral` → [{shipmentId, items:[...]}], or [].

    Replaces the per-shipment plate-map download. One call now covers every shipment on
    the order, and the items come back parsed — `plateId`, `wellLocation`, `actualYieldNg`
    and the QC verdicts — instead of as a CSV (or a ZIP of CSVs) to sniff and merge.

    404 means "no shipments yet", which is normal for an order still in production and not
    worth a line of output. Big orders can 504: Q-698807 (~160 items) does so reproducibly,
    so `_get_json` retries and then gives up, and the caller notes the gap.
    """
    # TWO attempts. Measured 2026-11-06: this call is usually ~4 s, but the same order can
    # take 58 s once and 4 s the next time, and a big one (Q-698807, ~160 items) 504s
    # outright. One attempt lost Q-745440's two maps to a single slow response; three spends
    # ~90 s re-failing on the orders that are genuinely too big. Two recovers the flaky ones
    # without letting a hopeless one eat the budget.
    j = _get_json(f"{NEW_BASE}/users/{email}/orders/{q}/collateral", jwt, left,
                  cost=_COLLATERAL_COST, label="", retries=2)
    return (j or {}).get("plateMaps") or []

_ITEM_FIELDS = ("name", "status", "last_event", "redo_count", "delayed_status",
                "estimated_shipping_date", "size", "type", "vector_name")
# An items call measured 1.8–3.0 s; without this much budget left, don't start one.
_ITEM_COST = 10


def _fetch_order_items(email, jwt, eut, sfdc: str, left=None) -> list[dict]:
    """`/orders/<sfdc>/items/` → that order's parts, trimmed to _ITEM_FIELDS.

    The only endpoint that NAMES what is on an order, and the only one that works before the
    order ships: the order list carries bare counters (`total_items`, `failed_items`) and the
    plate map does not exist until a shipment does. It also carries two things found nowhere
    else in this API — per-item `status`, so a failed or cancelled part can be named instead
    of merely counted, and `delayed_status`, Twist's OWN delay flag (376 of 2970 items read
    DELAYED on 2026-09-03, including a whole synpart order). ~2-3 s per order.

    Note the URL takes the order's `sfdc_id`, not its Q-number — `…/orders/Q-715423/items/`
    returns "Order not found." And `…/orders/<sfdc>/` with no suffix returns 403, so this
    endpoint is also the only way to read one order's detail.
    """
    tmo = 60 if left is None else max(5, min(60, int(left())))
    try:
        r = requests.get(f"{BASE_URL}/v1/users/{email}/orders/{sfdc}/items/",
                         headers=_headers(jwt, eut), timeout=tmo)
        if r.status_code != 200:
            return []
        raw = r.json().get("order_items") or []
    except Exception:
        return []

    out = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        row = {k: it.get(k) for k in _ITEM_FIELDS}
        prog = it.get("progress") or {}
        row["progress"] = prog.get("event_index") or ""
        row["progress_event"] = prog.get("latest_event") or ""
        out.append(row)
    return out


def _eta_changes(orders: list[dict], by_order: dict, waiting: set) -> dict:
    """{stock_id: 'eta 2026-08-31 → 2026-09-02'} for parts whose ETA moved since the last
    pull, then append this snapshot to twist_eta_log.csv so the history keeps building.
    The log is the only record of Twist quietly walking an ETA forward."""
    order_by_q = {o.get("order_name"): o for o in orders}
    prev = _load_eta_log()
    rows, changes = [], {}
    for q in sorted(waiting):
        o = order_by_q.get(q)
        if not o:
            continue
        eta = str(o.get("estimated_completion_date") or "")[:10]
        for p in by_order.get(q, []):
            if p["vis_status"] != "READY":
                continue
            row = {"STOCK_ID": p["stock_id"], "order": q, "eta": eta,
                   "attempt": p["attempt"], "order_status": o.get("status", "")}
            rows.append(row)
            was = prev.get(p["stock_id"])
            if was and was.get("eta") and eta and was["eta"] != eta:
                changes[p["stock_id"]] = f"ETA {was['eta']} → {eta}"
    if rows:
        ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        try:
            _append_eta_log(rows, ts)
        except Exception:
            traceback.print_exc()
    return changes


def _token_id(label: str, tok: str) -> str:
    """Fingerprint a token for the log — length, a hash prefix, and its expiry. NEVER the
    token itself.

    Worth a line of output because of how this failed on 2026-09-03: `~/.bashrc` exported
    stale 150-char tokens that shadowed the good 263-char ones in `~/.twist_env`, so the
    hourly cron (which sources that file) succeeded every hour while every interactive run
    got an instant Django 500 from Twist. Identical code, identical host, opposite results,
    and nothing in the output said which credentials were in play. This line does.
    """
    out = f"{label} len={len(tok)} sha={hashlib.sha256(tok.encode()).hexdigest()[:8]}"
    try:
        payload = tok.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        if exp:
            when = dt.datetime.fromtimestamp(exp, dt.timezone.utc)
            left = (when - dt.datetime.now(dt.timezone.utc)).days
            out += f" exp={when:%Y-%m-%d}" + (" EXPIRED" if left < 0 else f" ({left}d)")
    except Exception:
        out += " exp=? (payload did not decode — token may be truncated)"
    return out


def have_tokens() -> bool:
    """Whether the Twist API credentials are in the environment. full_refresh checks this
    before calling in, so a server without the tokens skips cleanly instead of erroring.

    Only the JWT is required now. The Order Tracking API — the orders, their detail and
    their plate maps — needs nothing else; X_END_USER_TOKEN buys the two OLD-host extras
    (per-item delay flags and the project-name scrape), and a pull without it is degraded
    rather than impossible. Demanding both would skip the pull entirely over an optional
    token.
    """
    return bool(os.environ.get("AUTHORIZATION_JWT"))


def build(days: int, max_pages: int, deadline: int = _DEADLINE_SECONDS) -> dict:
    email = os.environ.get("TWIST_EMAIL", "dna@asimov.io")
    jwt = os.environ.get("AUTHORIZATION_JWT", "")
    eut = os.environ.get("X_END_USER_TOKEN", "")
    # RuntimeError, not SystemExit: this is called from full_refresh inside an
    # `except Exception`, and SystemExit would sail past it and kill the refresh.
    if not jwt:
        raise RuntimeError("AUTHORIZATION_JWT must be set.")

    print(f"tokens: {_token_id('JWT', jwt)}"
          + (f" | {_token_id('EUT', eut)}" if eut else " | EUT absent"), flush=True)

    t0 = time.time()

    def left():
        return deadline - (time.time() - t0)

    def el():
        return f"{time.time() - t0:.0f}s"

    notes: list[str] = []
    by_order, waiting, by_stock = _pipeline_parts(PARQUET)
    print(f"pipeline: {len(by_order)} orders seen, {len(waiting)} still awaiting delivery"
          f" ({', '.join(sorted(waiting)) or 'none'}) [{el()}]", flush=True)

    # ── 1. every order on the account, newest first (~4 s) ────────────────────
    sums = _fetch_summaries(email, jwt, left)
    # Never cache an empty pull over a good one: with no orders the tab would go blank and
    # read as "nothing on order" rather than "the pull did not finish".
    if not sums:
        raise RuntimeError(f"no order summaries fetched in {el()} — keeping the previous "
                           f"cache")
    print(f"  {len(sums)} order summaries [{el()}]", flush=True)

    # ── 2. detail for as far back as the window reaches ───────────────────────
    # Summaries are ordered newest-first, so the walk stops once it is past the window —
    # but it keeps SCANNING (free, no network) for any Q-number the pipeline is still
    # waiting on, because an order can sit open well past `days` and those are precisely
    # the ones worth showing.
    today = dt.date.today()
    cutoff = (today - dt.timedelta(days=days)).isoformat()
    max_orders = max(25, max_pages * 25)
    want, pairs, seen, past_cutoff = set(waiting), [], set(), False

    for s in sums:
        q = str(s.get("orderId") or "")
        if not q or q in seen:
            continue
        if past_cutoff and q not in want:
            continue
        if len(seen) >= max_orders:
            notes.append(f"stopped after {max_orders} orders"
                         + (f"; not fetched: {', '.join(sorted(want))}" if want else ""))
            break
        if left() < _DETAIL_COST:
            notes.append(f"out of time after {len(seen)} orders"
                         + (f"; not fetched: {', '.join(sorted(want))}" if want else ""))
            break
        d = _fetch_detail(email, jwt, q, left)
        seen.add(q)
        if not d:
            continue
        pairs.append((s, d))
        want.discard(q)
        rec = str(d.get("receivedDate") or "")[:10]
        if rec and rec < cutoff:
            past_cutoff = True
        if past_cutoff and not want:
            break
    print(f"  {len(pairs)} orders detailed [{el()}]", flush=True)

    # ── 3. project names: cached, topped up on a best-effort scrape ───────────
    names = _load_projects()
    seeded = _seed_projects_from_pkl(names)
    if seeded:
        print(f"  seeded {seeded} project names from the previous pkl", flush=True)

    orders = [_legacy_order(s, d, names.get(str(s.get("orderId") or ""), ""))
              for s, d in pairs]

    # ── 4. window ─────────────────────────────────────────────────────────────
    keep = []
    for o in orders:
        q = o.get("order_name")
        # Orders with no pipeline part are kept too. They are still real work on this Twist
        # account — oligos, genes, another team's plasmids — and the tab shows them in their
        # own section rather than pretending the account only holds our synthesis parts.
        if q in waiting:
            keep.append(o)
        elif _is_open(o):
            # An OPEN order never ages out. The window exists to retire finished work, and
            # an order still open is the opposite of finished — the older it is, the more
            # it needs looking at.
            keep.append(o)
        else:
            # Delivered/closed: the window runs from when the DNA LANDED, not when it was
            # ordered. An order that closed without ever shipping has no delivery date, so
            # it falls back to `received_date` rather than living forever.
            landed = _delivered_on(o)
            if (landed or str(o.get("received_date") or "")[:10]) >= cutoff:
                keep.append(o)

    for o in keep:
        rec = str(o.get("received_date") or "")[:10]
        o["_delivered_on"] = _delivered_on(o)
        try:
            o["_open_days"] = (today - dt.date.fromisoformat(rec)).days if _is_open(o) else None
        except ValueError:
            o["_open_days"] = None

    n_other = sum(1 for o in keep if o.get("order_name") not in by_order)
    n_open = sum(1 for o in keep if _is_open(o))
    aged = sorted((o for o in keep if (o.get("_open_days") or 0) > days),
                  key=lambda o: -o["_open_days"])
    print(f"  {len(keep)} tracked ({n_open} open, kept regardless of age; "
          f"{len(keep) - n_open} delivered within {days}d of landing) "
          f"({len(keep) - n_other} with pipeline parts, {n_other} other) [{el()}]", flush=True)
    if aged:
        print(f"  {len(aged)} open >{days}d: "
              + ", ".join(f"{o['order_name']} ({o['_open_days']}d)" for o in aged[:10]),
              flush=True)

    # Pipeline orders lead in both loops below. With non-pipeline orders in `keep`, spending
    # the budget in list order could burn it on another team's parts and skip work for a
    # part the pipeline is actually waiting on.
    ordered = sorted(keep, key=lambda o: o.get("order_name") not in by_order)

    # ── 5. item lists — still the OLD host, the only source of the delay flags ─
    items_by_order, it_skipped = {}, 0
    if not eut:
        notes.append("X_END_USER_TOKEN absent — per-item redo_count and delayed_status "
                     "not fetched")
    else:
        for o in ordered:
            if left() < _ITEM_COST:
                it_skipped += 1
                continue
            items_by_order[o.get("order_name")] = _fetch_order_items(
                email, jwt, eut, o.get("sfdc_id"), left)
    if it_skipped:
        notes.append(f"out of time — item list not fetched for {it_skipped} order(s)")
    print(f"  {sum(len(v) for v in items_by_order.values())} order items "
          f"across {len(items_by_order)} orders [{el()}]", flush=True)

    # Recover orders whose parts exist in the pipeline but whose vendor_order_id is still
    # blank, by matching Twist's item names to known synthesis STOCK_IDs. Without this,
    # Q-715384's 77 syn parts (all READY in the pipeline, none stamped with its Q-number)
    # made the order look like someone else's work and it rendered in the "other" group.
    recovered = []
    for o in ordered:
        q = o.get("order_name")
        if q in by_order:
            continue
        matched = [by_stock[n] for n in
                   (str(i.get("name") or "") for i in items_by_order.get(q) or [])
                   if n in by_stock]
        if matched:
            by_order[q] = matched
            recovered.append(f"{q} ({len(matched)})")
    if recovered:
        print(f"  recovered by part name (blank vendor_order_id): "
              f"{', '.join(recovered)} [{el()}]", flush=True)

    # ── 6. plate maps, one collateral call per shipped order ──────────────────
    # `wells` holds the DNA, `glyc` the glycerol stocks — kept apart on purpose. Twist ships
    # glycerol as a SEPARATE shipment whose map names the very same parts as the DNA map
    # (Q-698807: 77 DNA rows on pSHPs0828B417001IQ, 77 glycerol rows on pSHPs0829B005801HX).
    # Merged into one dict per order they collide name-for-name and one map silently
    # overwrites the other.
    platemaps, wells, glyc = {}, {}, {}
    skipped, approx = 0, False
    for o in ordered:
        q = o.get("order_name")
        ships = {s.get("id"): s for s in (o.get("shipments") or [])
                 if s.get("status") in ("shipped", "received")}
        approx = approx or any(s.get("received_at_approx") for s in ships.values())
        if not ships:
            continue
        if left() < _COLLATERAL_COST:
            skipped += 1
            continue
        print(f"    plate maps {q} ...", flush=True)
        maps = _fetch_collateral(email, jwt, q, left)
        if not maps:
            skipped += 1
            continue
        for pm in maps:
            sid = pm.get("shipmentId")
            rows = [r for r in (pm.get("items") or []) if isinstance(r, dict)]
            if not rows:
                continue
            s = ships.get(sid) or {}
            # The COMPLETE payload goes to disk — disk is cheap and the full artifact
            # (sequences included) stays available independent of the pkl.
            plate = _first(rows[0], ("plateId",))
            try:
                PLATEMAP_DIR.mkdir(exist_ok=True)
                (PLATEMAP_DIR / f"{q}_platemap_{plate or sid}.json").write_text(
                    json.dumps(rows, indent=1))
            except Exception:
                pass
            slim = [{k: v for k, v in r.items() if k not in _DROP_KEYS} for r in rows]
            parsed = {}
            for r in slim:
                nm = str(r.get("name") or "").strip()
                if not nm:
                    continue
                info = {f: _first(r, keys) for f, keys in _WELL_FIELDS.items()}
                # Tag each row with the state of the SHIPMENT that carried it. Merged per
                # order, the maps otherwise lose which box a part travelled in, so the tab
                # could not tell a part that has landed from one still in the air.
                info["ship_status"] = s.get("status") or ""
                info["ship_at"] = str(s.get("received_at") or s.get("shipped_date") or "")[:10]
                parsed[nm] = info
            if not parsed:
                continue
            # A glycerol shipment's map is glycerol end to end, so the whole shipment can be
            # labelled as such and its download button say what it is.
            is_glyc = all("glycerol" in (i.get("product") or "").lower()
                          for i in parsed.values())
            platemaps[f"{q}|{sid}"] = {
                "filename": f"platemap_{plate or q}.csv",
                "csv": _rows_to_csv(slim),
                "glycerol": is_glyc,
            }
            for nm, info in parsed.items():
                tgt = glyc if "glycerol" in (info.get("product") or "").lower() else wells
                slot = tgt.setdefault(q, {})
                # A part can appear in two maps (a reship, or DNA then glycerol). Never let
                # a shipped row overwrite one already recorded as received.
                if slot.get(nm, {}).get("ship_status") == "received" != info["ship_status"]:
                    continue
                slot[nm] = info
    if skipped:
        # Say it out loud: a missing download button must not look like "Twist sent no map".
        notes.append(f"plate map not fetched for {skipped} shipped order(s) this run")
    print(f"  {len(platemaps)} plate maps [{el()}]", flush=True)

    # ── 7. project-name scrape last, on whatever budget is left ───────────────
    # Only reach for the broken endpoint when it could actually tell us something. Once
    # every kept order has a name, the scrape is 122 s spent to re-learn what is already
    # cached — and it fails ~92% of the time anyway.
    missing = [o for o in keep if not names.get(o.get("order_name"))]
    if not eut:
        scrape = "project-name scrape skipped — no X_END_USER_TOKEN"
    elif not missing:
        scrape = f"project names complete ({len(names)} cached) — scrape not needed"
    else:
        scrape = _scrape_project_names(email, jwt, eut, names, left)
    _save_projects(names)
    print(f"  {scrape} [{el()}]", flush=True)
    for o in keep:
        o["project_name"] = names.get(o.get("order_name"), "") or o.get("project_name", "")
    unnamed = sum(1 for o in keep if not o.get("project_name"))
    if unnamed:
        notes.append(f"{unnamed} order(s) have no project name — Twist's new API does not "
                     f"carry one, and the old endpoint it came from is broken")
    if approx:
        notes.append("Delivery dates shown are SHIP dates: Twist's new order API reports "
                     "when a box left, never when it landed")

    return {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc),
        "orders": keep,
        "parts_by_order": by_order,
        "items_by_order": items_by_order,
        "waiting_orders": sorted(waiting),
        "platemaps": platemaps,
        "wells_by_order": wells,
        "glycerol_by_order": glyc,
        "eta_changes": _eta_changes(keep, by_order, waiting),
        "window_days": days,
        "notes": notes,
        "eta_log": str(ETA_LOG),
    }

def refresh(days: int = 45, max_pages: int = 4,
            deadline: int = _DEADLINE_SECONDS) -> dict:
    """Pull and cache. Called directly by full_refresh.py — keep it argv-free, since
    argparse there would choke on the refresh's own flags."""
    t0 = time.time()
    result = build(days, max_pages, deadline)

    d = os.path.dirname(OUT)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".twist_result.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(result, fh)
        os.replace(tmp, OUT)  # atomic swap into place
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"cached fresh twist_result.pkl @ "
          f"{result['generated_at']:%Y-%m-%d %H:%M} UTC "
          f"({len(result['orders'])} orders, {len(result['platemaps'])} plate maps) "
          f"in {time.time() - t0:.0f}s")
    for n in result.get("notes") or []:
        print(f"  note: {n}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=45,
                    help="how long a DELIVERED order stays on the tab, measured from its "
                         "delivery date (default 45). Open orders are never aged out.")
    ap.add_argument("--max-pages", type=int, default=4,
                    help="hard cap on orders DETAILED, in units of 25 (default 4 = 100). "
                         "Named for the pages the old list endpoint fetched; the summary "
                         "list is now taken whole and the per-order detail is the cost.")
    ap.add_argument("--deadline", type=int, default=_DEADLINE_SECONDS,
                    help=f"wall-clock budget in seconds (default {_DEADLINE_SECONDS})")
    args = ap.parse_args()
    refresh(args.days, args.max_pages, args.deadline)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

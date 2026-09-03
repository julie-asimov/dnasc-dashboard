#!/usr/bin/env python3
"""Refresh the Twist order-tracking cache used by the dashboard's Twist tab.

Standalone pull on its own cron, exactly like gen_parts_pkl.py — the dashboard render
NEVER calls the Twist API. Two reasons that matters:

  • the orders endpoint is slow (~75 s per page regardless of page_size), so a live call
    inside render would add minutes to every rebuild
  • it needs AUTHORIZATION_JWT / X_END_USER_TOKEN, which expire; a render must not fail
    (or hang) because a token went stale overnight

Writes dashboard_state/twist_result.pkl ATOMICALLY (temp file + os.replace) so the
renderer always sees either the previous good pkl or the complete new one.

The order list is paged newest-first and STOPS EARLY: it pages only until every Q-number
the pipeline is still waiting on has been seen AND the window reaches back `--days`
(default 45). The full history is 300 orders — paging all of it would take ~20 minutes
for data nobody looks at. A typical run is ~3 min (2 pages + plate maps); `--max-pages`
bounds it at ~6, and `--deadline` is a wall-clock stop on top of that, so an hourly cron
can never still be running when the next one fires. Anything skipped for either reason is
recorded in `notes` and shown on the tab — a short pull must not read as "nothing new".

Env:
    AUTHORIZATION_JWT, X_END_USER_TOKEN   (required)
    TWIST_EMAIL                           (default dna@asimov.io)

Usage:
    /opt/anaconda3/bin/python3 gen_twist_pkl.py
    /opt/anaconda3/bin/python3 gen_twist_pkl.py --days 90 --max-pages 6
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import os
import pickle
import sys
import tempfile
import time
import traceback
import zipfile

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

# Workorder types whose vendor_order_id is a Twist Q-number.
_SYNTH_TYPES = ["syn_part_synthesis_workorder", "plasmid_synthesis_workorder"]

# One page costs ~75 s no matter its size, so take big pages and few of them.
_PAGE_SIZE = 25
_TIMEOUT = 300

# Hard wall-clock budget for the whole pull. The vendor API is the slow part and its latency
# is not ours to control, so the job stops fetching rather than running long — a cron firing
# hourly must never still be running when the next one starts, and it must be finished well
# before the dashboard refresh it feeds. Whatever was collected still gets cached, with a note
# on the tab saying what was skipped; a silent short pull would read as "nothing new".
# Raised 600→900 when the per-order items fetch was added (~150 s for 31 orders on top of
# ~190 s of order pages and ~100 s of plate maps). The cron leaves ~20 min before the parts
# pull at :50, so 15 min still lands with room to spare.
_DEADLINE_SECONDS = 900
# A page costs ~75 s; without this much left, starting one just burns the remaining budget.
_PAGE_COST = 90


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


# ── Twist side ────────────────────────────────────────────────────────────────
def _fetch_orders_window(email, jwt, eut, want: set, days: int, max_pages: int, left):
    """Page orders newest-first, stopping as soon as the window covers what we need.

    `left()` returns the seconds remaining in the pull's budget. Returns (orders, notes);
    `notes` records anything we deliberately did not fetch — page cap or deadline — so the
    tab can say so instead of silently looking complete.
    """
    url = f"{BASE_URL}/v1/users/{email}/orders/"
    cutoff = dt.date.today() - dt.timedelta(days=days)
    orders, notes, seen = [], [], set()
    total = None

    for page in range(1, max_pages + 1):
        if left() < _PAGE_COST:
            missing = sorted(want - seen)
            notes.append(f"out of time after {len(orders)} orders"
                         + (f"; not fetched: {', '.join(missing)}" if missing else ""))
            break
        resp = requests.get(url, headers=_headers(jwt, eut),
                            timeout=min(_TIMEOUT, max(30, int(left()))), params={
            "page_size": _PAGE_SIZE, "sort_by": "received_date",
            "reverse": "true", "page": page,
        })
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            batch, has_next = data, False
        else:
            batch = data.get("results", [])
            has_next = bool(data.get("next"))
            total = data.get("count", total)
        if not batch:
            break
        orders.extend(batch)
        seen |= {o.get("order_name") for o in batch}

        oldest = min((str(o.get("received_date") or "")[:10] for o in batch if o.get("received_date")),
                     default="")
        past_cutoff = bool(oldest) and oldest < cutoff.isoformat()
        if (past_cutoff and want <= seen) or not has_next:
            break
        if page == max_pages:
            missing = sorted(want - seen)
            notes.append(f"stopped after {max_pages} pages ({len(orders)} of {total} orders)"
                         + (f"; not found: {', '.join(missing)}" if missing else ""))
    return orders, notes


def _platemap_csv(email, jwt, eut, order_sfdc: str, shipment_id: str, order_name: str,
                  left=None):
    """Fetch a shipment's plate map. Despite the endpoint's name it serves a plain CSV
    (verified against twist_platemaps/*.zip — every saved file is CSV text, not a ZIP).

    Returns (filename, csv_text) or (None, None). Also mirrored to twist_platemaps/ so
    the files exist on disk independent of the pkl.
    """
    api = (f"{BASE_URL}/v1/users/{email}/orders/{order_sfdc}"
           f"/shipments/{shipment_id}/plate-maps/")
    # Cap each request by what is left of the pull's budget, so one slow map cannot
    # overrun the deadline on its own.
    tmo = 120 if left is None else max(5, min(120, int(left())))
    try:
        r = requests.get(api, headers=_headers(jwt, eut), timeout=tmo)
        if r.status_code != 200:
            return None, None
        file_url = r.json().get("platemaps_file_url", "")
        if not file_url:
            return None, None
        f = requests.get(file_url, timeout=tmo)
        if f.status_code != 200:
            return None, None
        raw = f.content
    except Exception:
        return None, None

    # MOST orders serve a plain CSV — but some genuinely serve a ZIP with the CSV inside
    # (2026-09-03: 4 of 26 maps began with the PK magic bytes). Decoding those as text gave
    # mojibake, so the parse found no parts AND the download button handed the browser a
    # corrupt file. Sniff the magic bytes rather than trusting the endpoint's content type.
    if raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                inner = next((n for n in z.namelist() if n.lower().endswith(".csv")), None)
                if not inner:
                    return None, None
                text = z.read(inner).decode("utf-8", "replace")
        except Exception:
            return None, None
    else:
        text = raw.decode("utf-8", "replace")

    plate = ""
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
        plate = _col(rows[0], "Plate ID", "Tube ID") if rows else ""
    except Exception:
        pass
    name = f"platemap_{plate or order_name}.csv"

    # The COMPLETE file goes to disk — disk is cheap and the full artifact stays available.
    try:
        PLATEMAP_DIR.mkdir(exist_ok=True)
        (PLATEMAP_DIR / f"{order_name}_{name}").write_text(text)
    except Exception:
        pass
    # Only the slim version is cached, because the pkl's copy is embedded verbatim in the
    # dashboard HTML for the ↓ Plate map button.
    return name, _slim_csv(text)


# Sequence columns, dropped from the CACHED copy of a plate map (the full file still goes to
# twist_platemaps/ on disk). On a clonal-gene map these are ~95% of the bytes — 4.4 MB of
# Q-698807's 4.6 MB — the tab renders none of them, and the pkl's copy is embedded verbatim in
# the dashboard HTML, so every viewer downloads them on every page load.
_DROP_COLS = ("Insert Sequence", "Construct Sequence", "Vector Sequence",
              "Construct Sequence (Insert + Adapters)")


def _slim_csv(text: str) -> str:
    """Same CSV without the sequence columns. Returns the input unchanged if it has none,
    or if anything about the parse looks off — a plate map is better whole than mangled."""
    try:
        rd = csv.DictReader(io.StringIO(text))
        orig = list(rd.fieldnames or [])
        cols = [c for c in orig if c not in _DROP_COLS]
        if not cols or cols == orig:
            return text
        buf = io.StringIO()
        wr = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for row in rd:
            wr.writerow({c: row.get(c) or "" for c in cols})
        return buf.getvalue()
    except Exception:
        return text


def _col(row: dict, *names: str) -> str:
    """First non-empty value among `names`. Twist serves at least SIX different plate-map
    layouts depending on product type, so a single column name is not enough."""
    for n in names:
        v = (row.get(n) or "").strip()
        if v:
            return v
    return ""


def _parse_platemap(text: str) -> dict:
    """CSV → {part name: {plate, well, yield, asm_qc, yield_qc, bp}}.

    `Name` is the LIMS STOCK_ID (syn4714, …), which is what makes this worth parsing:
    it gives the physical well for every part the moment a shipment lands. The bulky
    Insert Sequence column is dropped — it is ~90% of the file and the tab never shows it.

    Column names vary by product line, so each field reads every alias we have actually
    seen. Plate products carry `Plate ID` + `Well Location`; tube products (clonal genes)
    carry `Tube ID` and no well at all; and QC/yield are renamed again per product
    (`NGS QC`, `Actual Yield (ng)`). Reading one layout meant clonal-gene orders parsed
    with a blank location and no yield, which rendered as "—" as if Twist had sent nothing.
    """
    out = {}
    try:
        for row in csv.DictReader(io.StringIO(text)):
            nm = (row.get("Name") or "").strip()
            if not nm:
                continue
            out[nm] = {
                "plate": _col(row, "Plate ID", "Tube ID"),
                "well": _col(row, "Well Location"),
                "bp": _col(row, "Insert Length", "Construct Length"),
                "asm_qc": _col(row, "Assembly QC", "NGS QC"),
                "yield_qc": _col(row, "Yield QC", "Target Yield QC"),
                "yield": _col(row, "Yield (ng)", "Actual Yield (ng)"),
            }
    except Exception:
        pass
    return out


# Fields kept from an order item. The full payload repeats the order's shipping address and
# carries construct sequences we never render; keeping it whole would bloat the pkl for nothing.
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


def have_tokens() -> bool:
    """Whether the Twist API credentials are in the environment. full_refresh checks this
    before calling in, so a server without the tokens skips cleanly instead of erroring."""
    return bool(os.environ.get("AUTHORIZATION_JWT") and os.environ.get("X_END_USER_TOKEN"))


def build(days: int, max_pages: int, deadline: int = _DEADLINE_SECONDS) -> dict:
    email = os.environ.get("TWIST_EMAIL", "dna@asimov.io")
    jwt = os.environ.get("AUTHORIZATION_JWT", "")
    eut = os.environ.get("X_END_USER_TOKEN", "")
    # RuntimeError, not SystemExit: this is called from full_refresh inside an
    # `except Exception`, and SystemExit would sail past it and kill the refresh.
    if not jwt or not eut:
        raise RuntimeError("AUTHORIZATION_JWT and X_END_USER_TOKEN must be set.")

    t0 = time.time()

    def left():
        return deadline - (time.time() - t0)

    def el():
        return f"{time.time() - t0:.0f}s"

    by_order, waiting, by_stock = _pipeline_parts(PARQUET)
    print(f"pipeline: {len(by_order)} orders seen, {len(waiting)} still awaiting delivery"
          f" ({', '.join(sorted(waiting)) or 'none'}) [{el()}]", flush=True)

    print(f"fetching orders (~75 s per page, {deadline}s budget) ...", flush=True)
    orders, notes = _fetch_orders_window(email, jwt, eut, waiting, days, max_pages, left)
    print(f"  {len(orders)} orders in window [{el()}]", flush=True)
    # Never cache an empty pull over a good one: with no orders the tab would go blank and
    # read as "nothing on order" rather than "the pull did not finish".
    if not orders:
        raise RuntimeError(f"no orders fetched in {el()} — keeping the previous cache"
                           + (f" ({'; '.join(notes)})" if notes else ""))

    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    keep = []
    for o in orders:
        q = o.get("order_name")
        # Orders with no pipeline part are kept too. They are still real work on this Twist
        # account — oligos, genes, another team's plasmids — and the tab shows them in their
        # own section rather than pretending the account only holds our synthesis parts.
        # Only the window bounds them; `waiting` can never name an order we have no parts for.
        if q in waiting or str(o.get("received_date") or "")[:10] >= cutoff:
            keep.append(o)
    n_other = sum(1 for o in keep if o.get("order_name") not in by_order)
    print(f"  {len(keep)} tracked within {days}d "
          f"({len(keep) - n_other} with pipeline parts, {n_other} other) [{el()}]", flush=True)

    # Pipeline orders lead in both loops below. With non-pipeline orders now in `keep`,
    # spending the budget in list order could burn it on another team's parts and skip work
    # for a part the pipeline is actually waiting on.
    ordered = sorted(keep, key=lambda o: o.get("order_name") not in by_order)

    # Item lists before plate maps: an items call is ~2-3 s and is the ONLY source of part
    # names for an order that has not shipped, while a plate map is ~4 s and does not exist
    # until a shipment does. Cheaper and more broadly useful, so it gets the budget first.
    items_by_order, it_skipped = {}, 0
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

    platemaps, wells = {}, {}
    skipped = 0
    for o in ordered:
        q, sfdc = o.get("order_name"), o.get("sfdc_id")
        for s in (o.get("shipments") or []):
            if s.get("status") not in ("shipped", "received") or not (sfdc and s.get("id")):
                continue
            if left() <= 0:
                skipped += 1
                continue
            key = f"{q}|{s['id']}"
            print(f"    plate map {q} {s['id'][:8]} ...", flush=True)
            fname, text = _platemap_csv(email, jwt, eut, sfdc, s["id"], q, left)
            if not text:
                continue
            platemaps[key] = {"filename": fname, "csv": text}
            wells.setdefault(q, {}).update(_parse_platemap(text))
    if skipped:
        # Say it out loud: a missing download button must not look like "Twist sent no map".
        notes.append(f"out of time — {skipped} plate map(s) not fetched this run")
    print(f"  {len(platemaps)} plate maps [{el()}]", flush=True)

    return {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc),
        "orders": keep,
        "parts_by_order": by_order,
        "items_by_order": items_by_order,
        "waiting_orders": sorted(waiting),
        "platemaps": platemaps,
        "wells_by_order": wells,
        "eta_changes": _eta_changes(orders, by_order, waiting),
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
                    help="how far back delivered orders stay on the tab (default 45)")
    ap.add_argument("--max-pages", type=int, default=4,
                    help="hard cap on order pages fetched (default 4 = 100 orders)")
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

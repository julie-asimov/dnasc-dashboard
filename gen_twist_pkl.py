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
for data nobody looks at.

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

# Workorder types whose vendor_order_id is a Twist Q-number.
_SYNTH_TYPES = ["syn_part_synthesis_workorder", "plasmid_synthesis_workorder"]

# One page costs ~75 s no matter its size, so take big pages and few of them.
_PAGE_SIZE = 25
_TIMEOUT = 300


# ── pipeline side ─────────────────────────────────────────────────────────────
def _pipeline_parts(parquet: str) -> tuple[dict, set]:
    """{Q-number: [part dicts]}, {Q-numbers with a part still waiting on delivery}.

    READY on a synthesis workorder means "ordered, not yet received into LIMS" — that is
    the set the tab exists to watch. Parts already SUCCEEDED are kept too so a delivered
    order can show what came in, but they never keep an order on screen by themselves.
    """
    df = pd.read_parquet(parquet, columns=[
        "STOCK_ID", "type", "vendor_order_id", "visual_status", "wo_status",
        "resubmit_count", "construct_name",
    ])
    df["vendor_order_id"] = df["vendor_order_id"].astype(str).str.strip()
    synth = df[df["type"].isin(_SYNTH_TYPES) &
               df["vendor_order_id"].str.startswith("Q-", na=False)]
    synth = synth.drop_duplicates("STOCK_ID")

    by_order: dict[str, list[dict]] = {}
    for r in synth.itertuples(index=False):
        by_order.setdefault(r.vendor_order_id, []).append({
            "stock_id": str(r.STOCK_ID),
            "kind": "synpart" if "syn_part" in str(r.type) else "plasmid",
            "attempt": int(r.resubmit_count or 0) + 1,
            "vis_status": str(r.visual_status or ""),
            "wo_status": str(r.wo_status or ""),
            "construct": str(r.construct_name or ""),
        })
    waiting = {q for q, parts in by_order.items()
               if any(p["vis_status"] == "READY" for p in parts)}
    return by_order, waiting


# ── Twist side ────────────────────────────────────────────────────────────────
def _fetch_orders_window(email, jwt, eut, want: set, days: int, max_pages: int):
    """Page orders newest-first, stopping as soon as the window covers what we need.

    Returns (orders, notes). `notes` records anything we deliberately did not fetch, so
    the tab can say so instead of silently looking complete.
    """
    url = f"{BASE_URL}/v1/users/{email}/orders/"
    cutoff = dt.date.today() - dt.timedelta(days=days)
    orders, notes, seen = [], [], set()
    total = None

    for page in range(1, max_pages + 1):
        resp = requests.get(url, headers=_headers(jwt, eut), timeout=_TIMEOUT, params={
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


def _platemap_csv(email, jwt, eut, order_sfdc: str, shipment_id: str, order_name: str):
    """Fetch a shipment's plate map. Despite the endpoint's name it serves a plain CSV
    (verified against twist_platemaps/*.zip — every saved file is CSV text, not a ZIP).

    Returns (filename, csv_text) or (None, None). Also mirrored to twist_platemaps/ so
    the files exist on disk independent of the pkl.
    """
    api = (f"{BASE_URL}/v1/users/{email}/orders/{order_sfdc}"
           f"/shipments/{shipment_id}/plate-maps/")
    try:
        r = requests.get(api, headers=_headers(jwt, eut), timeout=120)
        if r.status_code != 200:
            return None, None
        file_url = r.json().get("platemaps_file_url", "")
        if not file_url:
            return None, None
        f = requests.get(file_url, timeout=120)
        if f.status_code != 200:
            return None, None
        text = f.content.decode("utf-8", "replace")
    except Exception:
        return None, None

    plate = ""
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
        plate = str(rows[0].get("Plate ID", "") or "") if rows else ""
    except Exception:
        pass
    name = f"platemap_{plate or order_name}.csv"
    try:
        PLATEMAP_DIR.mkdir(exist_ok=True)
        (PLATEMAP_DIR / f"{order_name}_{name}").write_text(text)
    except Exception:
        pass
    return name, text


def _parse_platemap(text: str) -> dict:
    """CSV → {part name: {plate, well, yield, asm_qc, yield_qc, bp}}.

    `Name` is the LIMS STOCK_ID (syn4714, …), which is what makes this worth parsing:
    it gives the physical well for every part the moment a shipment lands. The bulky
    Insert Sequence column is dropped — it is ~90% of the file and the tab never shows it.
    """
    out = {}
    try:
        for row in csv.DictReader(io.StringIO(text)):
            nm = (row.get("Name") or "").strip()
            if not nm:
                continue
            out[nm] = {
                "plate": (row.get("Plate ID") or "").strip(),
                "well": (row.get("Well Location") or "").strip(),
                "bp": (row.get("Insert Length") or "").strip(),
                "asm_qc": (row.get("Assembly QC") or "").strip(),
                "yield_qc": (row.get("Yield QC") or "").strip(),
                "yield": (row.get("Yield (ng)") or "").strip(),
            }
    except Exception:
        pass
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


def build(days: int, max_pages: int) -> dict:
    email = os.environ.get("TWIST_EMAIL", "dna@asimov.io")
    jwt = os.environ.get("AUTHORIZATION_JWT", "")
    eut = os.environ.get("X_END_USER_TOKEN", "")
    # RuntimeError, not SystemExit: this is called from full_refresh inside an
    # `except Exception`, and SystemExit would sail past it and kill the refresh.
    if not jwt or not eut:
        raise RuntimeError("AUTHORIZATION_JWT and X_END_USER_TOKEN must be set.")

    by_order, waiting = _pipeline_parts(PARQUET)
    print(f"pipeline: {len(by_order)} orders seen, {len(waiting)} still awaiting delivery"
          f" ({', '.join(sorted(waiting)) or 'none'})", flush=True)

    print("fetching orders (~75 s per page) ...", flush=True)
    orders, notes = _fetch_orders_window(email, jwt, eut, waiting, days, max_pages)
    print(f"  {len(orders)} orders in window", flush=True)

    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    keep = []
    for o in orders:
        q = o.get("order_name")
        if q not in by_order:
            continue  # a Twist order with no pipeline part is not ours to track
        if q in waiting or str(o.get("received_date") or "")[:10] >= cutoff:
            keep.append(o)
    print(f"  {len(keep)} tracked (pipeline parts + within {days}d)", flush=True)

    platemaps, wells = {}, {}
    for o in keep:
        q, sfdc = o.get("order_name"), o.get("sfdc_id")
        for s in (o.get("shipments") or []):
            if s.get("status") not in ("shipped", "received") or not (sfdc and s.get("id")):
                continue
            key = f"{q}|{s['id']}"
            print(f"    plate map {q} {s['id'][:8]} ...", flush=True)
            fname, text = _platemap_csv(email, jwt, eut, sfdc, s["id"], q)
            if not text:
                continue
            platemaps[key] = {"filename": fname, "csv": text}
            wells.setdefault(q, {}).update(_parse_platemap(text))

    return {
        "generated_at": dt.datetime.now(tz=dt.timezone.utc),
        "orders": keep,
        "parts_by_order": by_order,
        "waiting_orders": sorted(waiting),
        "platemaps": platemaps,
        "wells_by_order": wells,
        "eta_changes": _eta_changes(orders, by_order, waiting),
        "window_days": days,
        "notes": notes,
        "eta_log": str(ETA_LOG),
    }


def refresh(days: int = 45, max_pages: int = 4) -> dict:
    """Pull and cache. Called directly by full_refresh.py — keep it argv-free, since
    argparse there would choke on the refresh's own flags."""
    result = build(days, max_pages)

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
          f"({len(result['orders'])} orders, {len(result['platemaps'])} plate maps)")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=45,
                    help="how far back delivered orders stay on the tab (default 45)")
    ap.add_argument("--max-pages", type=int, default=4,
                    help="hard cap on order pages fetched (default 4 = 100 orders)")
    args = ap.parse_args()
    refresh(args.days, args.max_pages)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

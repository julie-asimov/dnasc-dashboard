#!/usr/bin/env python3
"""
Twist Bioscience Order Status

Default: shows open orders with per-part status and ETA.
--synparts: joins pipeline parquet against Twist ETAs for active syn/plasmid workorders.
--plates: shows plate maps for all delivered orders.
--explore: dumps raw JSON for one order's items (use sfdc_id or Q-number).

Usage:
    python twist_orders.py                        # open orders + per-part ETAs
    python twist_orders.py --synparts             # active pipeline parts + Twist ETAs
    python twist_orders.py --plates               # all delivered plate maps
    python twist_orders.py --explore Q-663075     # dump raw items JSON for one order
    python twist_orders.py --output items.csv     # save items table to CSV
"""

import argparse
import csv
import json
import os
import pathlib
import sys

import pandas as pd
import requests

BASE_URL = "https://twist-api.twistdna.com"


def _headers(jwt: str, end_user_token: str) -> dict:
    return {
        "Authorization": f"JWT {jwt}",
        "X-End-User-Token": end_user_token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def get_orders(email, jwt, eut, order_status=None, page_size=50) -> list[dict]:
    url = f"{BASE_URL}/v1/users/{email}/orders/"
    params = {"page_size": page_size, "sort_by": "received_date", "reverse": "true"}
    if order_status:
        params["order_status"] = order_status
    all_orders, page = [], 1
    while True:
        params["page"] = page
        resp = requests.get(url, headers=_headers(jwt, eut), params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            all_orders.extend(data)
            break
        all_orders.extend(data.get("results", data))
        if not data.get("next"):
            break
        page += 1
    return all_orders


def get_order_items(email, jwt, eut, sfdc_id: str) -> list[dict]:
    url = f"{BASE_URL}/v1/users/{email}/orders/{sfdc_id}/items/"
    resp = requests.get(url, headers=_headers(jwt, eut))
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    # Some responses wrap items; try common keys
    for key in ("items", "results", "sequences"):
        if key in data:
            return data[key]
    # Single object (shouldn't happen but handle gracefully)
    return [data] if data else []


def get_plate_maps(email, jwt, eut, page_size=100) -> list[dict]:
    url = f"{BASE_URL}/v1/users/{email}/platemaps/"
    params = {"page_size": page_size}
    all_maps, page = [], 1
    while True:
        params["page"] = page
        resp = requests.get(url, headers=_headers(jwt, eut), params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            all_maps.extend(data)
            break
        all_maps.extend(data.get("results", data))
        if not data.get("next"):
            break
        page += 1
    return all_maps


PLATEMAP_DIR = pathlib.Path(__file__).parent / "twist_platemaps"


def download_platemap_zip(email, jwt, eut, order_sfdc_id: str, shipment_id: str,
                          order_name: str) -> str:
    """
    Fetches the plate map ZIP for a shipment, saves it locally, and returns the
    local file path (relative to the script dir). Returns '' if unavailable.
    Pre-signed S3 URLs expire in minutes — we save locally so links never expire.
    """
    api_url = f"{BASE_URL}/v1/users/{email}/orders/{order_sfdc_id}/shipments/{shipment_id}/plate-maps/"
    try:
        resp = requests.get(api_url, headers=_headers(jwt, eut))
        if resp.status_code != 200:
            return ""
        zip_url = resp.json().get("platemaps_file_url", "")
        if not zip_url:
            return ""
        zip_resp = requests.get(zip_url)
        if zip_resp.status_code != 200:
            return ""
        PLATEMAP_DIR.mkdir(exist_ok=True)
        safe_sid = shipment_id.replace("/", "_")[:16]
        filename = f"platemap_{order_name}_{safe_sid}.zip"
        local_path = PLATEMAP_DIR / filename
        if local_path.exists():
            return str(local_path)  # already downloaded, skip
        local_path.write_bytes(zip_resp.content)
        return str(local_path)
    except Exception:
        return ""


def render_twist_html(email: str, jwt: str, eut: str, parquet_path: str = None) -> str:
    """Fetch all Twist data and return a complete HTML string for the tracking tab."""
    import datetime

    now_ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Load pipeline parquet for part→order mapping
    _pq = parquet_path or str(pathlib.Path(__file__).parent / "dashboard_state" / "baseline.parquet")
    try:
        df_pip = pd.read_parquet(_pq)
        df_pip["vendor_order_id"] = df_pip["vendor_order_id"].astype(str).str.strip()
        parts_by_order: dict[str, list[dict]] = {}
        # Only orders with at least one READY part (still awaiting receipt)
        active_vendor_ids = set(
            df_pip.loc[
                df_pip["type"].isin(["syn_part_synthesis_workorder", "plasmid_synthesis_workorder"]) &
                (df_pip["visual_status"] == "READY") &
                df_pip["vendor_order_id"].str.startswith("Q-", na=False),
                "vendor_order_id"
            ].str.strip()
        )
        synth_rows = df_pip[
            df_pip["type"].isin(["syn_part_synthesis_workorder", "plasmid_synthesis_workorder"]) &
            df_pip["vendor_order_id"].str.strip().isin(active_vendor_ids)
        ][["STOCK_ID", "vendor_order_id", "visual_status", "resubmit_count", "wo_status"]].drop_duplicates("STOCK_ID")
        for _, r in synth_rows.iterrows():
            q = r["vendor_order_id"]
            parts_by_order.setdefault(q, []).append({
                "stock_id":  r["STOCK_ID"],
                "attempt":   int(r.get("resubmit_count") or 0) + 1,
                "vis_status": r.get("visual_status", ""),
                "wo_status":  r.get("wo_status", ""),
            })
    except Exception:
        parts_by_order = {}

    # Fetch active orders
    print("  Fetching orders ...", flush=True)
    all_orders = get_orders(email, jwt, eut)
    active_q = set(parts_by_order.keys())
    # Show orders we have pipeline parts for, sorted newest first
    orders_to_show = [o for o in all_orders if o.get("order_name") in active_q]

    def _badge(text, bg, fg, border):
        return f'<span style="display:inline-block;font-size:9px;font-weight:600;padding:2px 7px;border-radius:10px;background:{bg};color:{fg};border:1px solid {border};white-space:nowrap">{text}</span>'

    def _status_badge(status):
        m = {"received": ("#f0fdf4","#16a34a","#bbf7d0"),
             "past":     ("#f0fdf4","#16a34a","#bbf7d0"),
             "open":     ("#eff6ff","#1d4ed8","#bfdbfe"),
             "shipped":  ("#eff6ff","#1d4ed8","#bfdbfe"),
             "cancelled":("#fff1f5","#be185d","#fecdd3")}
        bg, fg, bd = m.get(status, ("#f8fafc","#475569","#e2e8f0"))
        return _badge(status, bg, fg, bd)

    cards = []
    for order in orders_to_show:
        q           = order.get("order_name", "")
        sfdc_id     = order.get("sfdc_id", "")
        project     = order.get("project_name", "")
        status      = order.get("status", "")
        ordered     = _fmt_date(order.get("received_date", ""))
        eta         = _fmt_date(order.get("estimated_completion_date", ""))
        shipments   = order.get("shipments") or []
        parts       = parts_by_order.get(q, [])

        # Detect adaptor-off or clonal from shipment container data
        attempt_types = set()
        for s in shipments:
            for c in (s.get("containers") or []):
                attempt_types.update((c.get("statistics") or {}).get("attempt_types", {}).keys())
        is_adaptor_off = "shippable_adaptor_off" in attempt_types
        is_clonal      = any("microprep" in t for t in attempt_types)
        n_shipped      = sum(1 for s in shipments if s.get("status") in ("shipped", "received"))

        # Order header badges
        badges = _status_badge(status)
        if is_adaptor_off:
            badges += " " + _badge("adaptor-off", "#f8fafc", "#475569", "#e2e8f0")
        if is_clonal and len(shipments) > 1:
            badges += " " + _badge(f"clonal — ships per item", "#f8fafc", "#475569", "#e2e8f0")

        # ── Arrival / ETA line ────────────────────────────────────────────────
        # The order-level `estimated_completion_date` is a stale estimate Twist
        # never reconciles after shipping (e.g. ETA 6/15 on an order received 6/10).
        # So show the REAL shipment state; the order ETA is only meaningful while
        # nothing has shipped yet. Handles partial shipments at the whole-order level.
        import datetime as _dt
        def _pdate(s):
            try:
                return _dt.date.fromisoformat(str(s)[:10])
            except Exception:
                return None
        today  = _dt.date.today()
        eta_d  = _pdate(eta)
        recv   = sorted(d for d in (_pdate(s.get("received_at")) for s in shipments if s.get("status") == "received") if d)
        in_transit = [s for s in shipments if s.get("status") == "shipped"]
        order_done = status in ("past", "received") and not in_transit

        def _delta_tag(arrived):
            if not eta_d or not arrived:
                return ""
            d = (arrived - eta_d).days
            if d > 0:
                return f' <span style="color:#be185d;font-weight:700">⚠ {d}d late</span>'
            if d < 0:
                return f' <span style="color:#16a34a;font-weight:600">{-d}d early</span>'
            return ' <span style="color:#16a34a;font-weight:600">on time</span>'

        dates_html = f'<span>Ordered <strong>{ordered}</strong></span>'
        if order_done and recv:
            dates_html += f' <span>Received <strong>{recv[-1]}</strong>{_delta_tag(recv[-1])}</span>'
        elif recv or in_transit:
            # partially shipped: some containers out the door, order still open
            bits = []
            if recv:
                bits.append(f'{len(recv)} received (last {recv[-1]})')
            if in_transit:
                est = _pdate(in_transit[0].get("estimated_delivery_date"))
                bits.append(f'{len(in_transit)} in transit' + (f' (arriving {est})' if est else ''))
            rest = ''
            if status not in ("past", "received") and eta:
                od = (today - eta_d).days if eta_d else None
                overdue = f' <span style="color:#be185d;font-weight:700">overdue {od}d</span>' if (od and od > 0) else ''
                rest = f' · rest ETA <strong>{eta}</strong>{overdue}'
            dates_html += f' <span style="color:#1d4ed8;font-weight:700">Partially shipped</span> <span>{" · ".join(bits)}{rest}</span>'
        elif eta:
            od = (today - eta_d).days if eta_d else None
            overdue = od is not None and od > 0
            style = "color:#be185d;font-weight:700" if overdue else ""
            tag = f' <span style="color:#be185d;font-weight:700">overdue {od}d</span>' if overdue else ''
            dates_html += f' <span>ETA <strong style="{style}">{eta}</strong></span>{tag}'

        # Parts table rows
        part_rows = []
        for p in sorted(parts, key=lambda x: x["stock_id"]):
            vs = p["vis_status"]
            if vs == "READY":
                st = '<span style="font-size:9px;color:#d97706;font-weight:600">waiting</span>'
            elif vs == "SUCCEEDED" or p["wo_status"] == "SUCCEEDED":
                st = '<span style="font-size:9px;color:#16a34a;font-weight:600">✓ received</span>'
            elif vs == "RUNNING":
                st = '<span style="font-size:9px;color:#1d4ed8;font-weight:600">→ in transit</span>'
            else:
                st = f'<span style="font-size:9px;color:#86868b">{vs or "—"}</span>'
            attempt_html = (f'<span style="font-size:8px;font-weight:700;padding:1px 5px;border-radius:6px;background:#fffbeb;color:#d97706;border:1px solid #fde68a">↻ {p["attempt"]}</span>'
                            if p["attempt"] > 1 else str(p["attempt"]))
            part_rows.append(
                f'<tr><td style="font-family:monospace;font-weight:700;color:#7c3aed;padding:5px 12px">{p["stock_id"]}</td>'
                f'<td style="padding:5px 12px">{attempt_html}</td>'
                f'<td style="padding:5px 12px;font-size:10px;color:#86868b">{eta}</td>'
                f'<td style="padding:5px 12px">{st}</td></tr>'
            )

        # Shipment rows
        ship_rows = []
        for s in shipments:
            s_status   = s.get("status", "")
            dot_color  = "#16a34a" if s_status == "received" else "#1d4ed8" if s_status == "shipped" else "#d97706"
            carrier    = (s.get("carrier") or "").upper()
            tracking   = s.get("tracking_number", "")
            track_url  = s.get("tracking_url") or "#"
            shipped    = _fmt_date(s.get("shipped_date", ""))
            location   = s.get("last_location") or ""
            delivered  = _fmt_date(s.get("received_at") or s.get("estimated_delivery_date") or "")
            containers = s.get("containers") or []
            plate_str  = " · ".join(
                f'{c.get("barcode","?")} ({sum((c.get("statistics") or {}).get("attempt_types", {}).values())} wells)'
                for c in containers
            ) if containers else "tube"

            # Download plate map ZIP locally so the link never expires
            local_zip = ""
            if s_status == "received" and sfdc_id and s.get("id"):
                print(f"    Downloading plate map ZIP for {q} ...", flush=True)
                local_zip = download_platemap_zip(email, jwt, eut, sfdc_id, s["id"], q)

            dl_link = (f' <a href="{local_zip}" download style="font-size:9px;font-weight:600;color:#1d4ed8;'
                       f'background:#eff6ff;border:1px solid #bfdbfe;border-radius:5px;padding:2px 8px;text-decoration:none">↓ Plate maps ZIP</a>'
                       if local_zip else "")

            delivered_str = f"Delivered <strong>{delivered}</strong> · {location}" if delivered else ('<span style="color:#1d4ed8;font-weight:600">in transit</span>' if s_status == "shipped" else "")

            ship_rows.append(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:7px 14px;border-bottom:1px solid #f0f0f2;flex-wrap:wrap">
              <span style="width:8px;height:8px;border-radius:50%;background:{dot_color};flex-shrink:0;display:inline-block"></span>
              <span style="font-size:10px;font-weight:700;color:#475569;text-transform:uppercase">{carrier}</span>
              <a href="{track_url}" target="_blank" style="font-family:monospace;font-size:10px;color:#1d4ed8;font-weight:600;text-decoration:none">{tracking}</a>
              <span style="font-family:monospace;font-size:10px;font-weight:600;color:#86868b;background:#f8fafc;border:1px solid #e2e8f0;border-radius:4px;padding:1px 6px">{plate_str}</span>
              <div style="font-size:10px;color:#86868b;display:flex;gap:12px;margin-left:auto;align-items:center;flex-wrap:wrap">
                <span>Shipped <strong style="color:#1d1d1f">{shipped}</strong></span>
                <span>{delivered_str}</span>
                {dl_link}
              </div>
            </div>""")

        cards.append(f"""
<div style="background:#fff;border:1px solid #e5e5e7;border-radius:10px;margin-bottom:10px;overflow:hidden">
  <div style="display:flex;align-items:center;gap:8px;padding:9px 14px;background:#fafafa;border-bottom:1px solid #e5e5e7;flex-wrap:wrap">
    <span style="font-size:13px;font-weight:800;color:#7c3aed">{q}</span>
    <span style="font-size:11px;color:#86868b">{project}</span>
    {badges}
    <div style="font-size:10px;color:#86868b;display:flex;gap:12px;margin-left:auto;flex-wrap:wrap">{dates_html}</div>
  </div>
  <table style="width:100%;border-collapse:collapse;border-bottom:1px solid #e5e5e7">
    <thead><tr style="background:#fafafa">
      <th style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:#86868b;padding:5px 12px;text-align:left;border-bottom:1px solid #e5e5e7">Part</th>
      <th style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:#86868b;padding:5px 12px;text-align:left;border-bottom:1px solid #e5e5e7">Attempt</th>
      <th style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:#86868b;padding:5px 12px;text-align:left;border-bottom:1px solid #e5e5e7">ETA</th>
      <th style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:#86868b;padding:5px 12px;text-align:left;border-bottom:1px solid #e5e5e7">Status</th>
    </tr></thead>
    <tbody>{''.join(part_rows)}</tbody>
  </table>
  {''.join(ship_rows)}
</div>""")

    return f"""<div style="padding:12px 16px">
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:12px;flex-wrap:wrap">
    <span style="font-size:13px;font-weight:700;color:#1d1d1f">Twist Tracking</span>
    <span style="font-size:9px;color:#86868b">{now_ts}</span>
  </div>
  {''.join(cards) if cards else '<div style="color:#86868b;font-size:12px;padding:8px">No active orders found.</div>'}
</div>"""


TWIST_HTML_OUT = pathlib.Path(__file__).parent / "twist_tracking.html"
ETA_LOG = pathlib.Path(__file__).parent / "twist_eta_log.csv"
_LOG_FIELDS = ["timestamp", "stock_id", "order", "eta", "attempt", "order_status"]


def _load_eta_log() -> dict[str, dict]:
    """Returns {stock_id: last-row-dict} for the most recent entry per part."""
    if not ETA_LOG.exists():
        return {}
    last: dict[str, dict] = {}
    with open(ETA_LOG, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "stock_id" in row:
                last[row["stock_id"]] = row
    return last


def _append_eta_log(rows: list[dict], now_ts: str) -> None:
    write_header = not ETA_LOG.exists()
    with open(ETA_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({
                "timestamp":    now_ts,
                "stock_id":     r["STOCK_ID"],
                "order":        r["order"],
                "eta":          r["eta"],
                "attempt":      r.get("attempt", ""),
                "order_status": r.get("order_status", ""),
            })


def _fmt_date(s) -> str:
    if not s:
        return ""
    return str(s)[:10]


def _item_rows(order: dict, items: list[dict]) -> list[dict]:
    rows = []
    for it in items:
        name = (it.get("name") or it.get("sequence_name") or
                it.get("insert_name") or it.get("construct_name") or "")
        eta = _fmt_date(
            it.get("estimated_completion_date") or
            it.get("estimated_delivery_date") or
            it.get("due_date") or
            order.get("estimated_completion_date")
        )
        rows.append({
            "order":   order.get("order_name"),
            "part":    name,
            "status":  it.get("status", ""),
            "eta":     eta,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Twist Bioscience Order Status")
    parser.add_argument("--synparts", action="store_true",
                        help="Show active pipeline syn/plasmid workorders with Twist ETAs")
    parser.add_argument("--plates", action="store_true",
                        help="Show plate maps for all delivered orders")
    parser.add_argument("--plates-csv", metavar="FILE",
                        help="Download real plate map data from Twist API and save to CSV")
    parser.add_argument("--html", metavar="FILE",
                        help="Generate tracking HTML with real Twist data and plate map download links")
    parser.add_argument("--explore", metavar="ORDER",
                        help="Dump raw items JSON for an order (Q-number or sfdc_id)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save items table to CSV")
    parser.add_argument("--parquet", default=None,
                        help="Path to baseline.parquet (default: auto-detected)")
    args = parser.parse_args()

    email = os.environ.get("TWIST_EMAIL", "dna@asimov.io")
    jwt   = os.environ.get("AUTHORIZATION_JWT", "")
    eut   = os.environ.get("X_END_USER_TOKEN", "")
    if not jwt or not eut:
        print("Error: AUTHORIZATION_JWT and X_END_USER_TOKEN must be set.", file=sys.stderr)
        sys.exit(1)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 240)

    # --html: generate full tracking HTML with real Twist data
    if args.html:
        outfile = args.html
        print(f"Generating Twist tracking HTML → {outfile}", flush=True)
        html = render_twist_html(email, jwt, eut, parquet_path=args.parquet)
        with open(outfile, "w") as f:
            f.write(f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Twist Tracking</title></head><body style='font-family:-apple-system,sans-serif;background:#f5f5f7;margin:0;padding:16px'>{html}</body></html>")
        print(f"Done. Open: {outfile}")
        return

    # --synparts: parts sitting in Receive queues joined against Twist order ETAs
    if args.synparts:
        parquet_path = args.parquet or str(
            pathlib.Path(__file__).parent / "dashboard_state" / "baseline.parquet"
        )
        df_pip = pd.read_parquet(parquet_path)

        df_pip["vendor_order_id"] = df_pip["vendor_order_id"].astype(str).str.strip()
        active = df_pip[
            df_pip["type"].isin(["syn_part_synthesis_workorder", "plasmid_synthesis_workorder"]) &
            (df_pip["visual_status"] == "READY") &
            df_pip["vendor_order_id"].notna() &
            df_pip["vendor_order_id"].str.startswith("Q-")
        ][["STOCK_ID", "type", "vendor_order_id", "wo_status", "construct_name", "resubmit_count"]].drop_duplicates("STOCK_ID")

        q_numbers = set(active["vendor_order_id"].unique())
        print(f"{len(active)} parts in receive queue across {len(q_numbers)} orders: {sorted(q_numbers)}\n")

        # Single orders-list call — no per-item detail needed
        print("Fetching Twist order ETAs ...", flush=True)
        all_orders = get_orders(email, jwt, eut)
        order_map = {o["order_name"]: o for o in all_orders if o.get("order_name") in q_numbers}

        rows = []
        for _, row in active.iterrows():
            stock = row["STOCK_ID"]
            q = str(row["vendor_order_id"])
            order = order_map.get(q, {})
            attempt = int(row.get("resubmit_count") or 0) + 1
            rows.append({
                "STOCK_ID":     stock,
                "type":         "synpart" if "syn_part" in row["type"] else "plasmid",
                "attempt":      attempt,
                "order":        q,
                "project":      str(order.get("project_name", "") or "")[:30],
                "ordered":      _fmt_date(order.get("received_date", "")),
                "eta":          _fmt_date(order.get("estimated_completion_date", "")),
                "order_status": order.get("status", ""),
            })

        import datetime
        now_ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        prev = _load_eta_log()

        # Flag ETA or attempt changes vs last snapshot
        for r in rows:
            prev_row = prev.get(r["STOCK_ID"])
            changes = []
            if prev_row:
                if prev_row.get("eta") and r["eta"] and prev_row["eta"] != r["eta"]:
                    changes.append(f"eta {prev_row['eta']} → {r['eta']}")
                if prev_row.get("attempt") and str(prev_row["attempt"]) != str(r.get("attempt", "")):
                    changes.append(f"attempt {prev_row['attempt']} → {r['attempt']}")
            r["eta_changed"] = "; ".join(changes)

        _append_eta_log(rows, now_ts)

        result = pd.DataFrame(rows).sort_values(["order", "STOCK_ID"])

        # Print change summary first if anything shifted
        changed = result[result["eta_changed"] != ""]
        if not changed.empty:
            print("=== ETA changes since last run ===")
            print(changed[["STOCK_ID", "order", "eta_changed"]].to_string(index=False))
            print()
        elif len(prev) > 0:
            print(f"No ETA changes since last run.\n")

        # How often does Twist update? Show log stats if we have history
        if ETA_LOG.exists():
            log_df = pd.read_csv(ETA_LOG, on_bad_lines="skip")
            if len(log_df) > 0:
                n_runs = log_df["timestamp"].nunique()
                n_changes = (
                    log_df.sort_values("timestamp")
                    .groupby("stock_id")["eta"]
                    .apply(lambda s: (s != s.shift()).sum() - 1)
                    .sum()
                )
                print(f"ETA log: {n_runs} snapshots recorded, {int(n_changes)} ETA/attempt changes observed total.")
                if n_changes > 0:
                    first_ts = log_df["timestamp"].min()
                    print(f"  Tracking since: {first_ts}")
                print()

        print(result[["STOCK_ID", "type", "attempt", "order", "project", "ordered", "eta", "order_status"]].to_string(index=False))
        if args.output:
            result.to_csv(args.output, index=False)
            print(f"\nSaved to {args.output}")
        return

    # --explore: find the order, print raw items JSON
    if args.explore:
        target = args.explore
        print(f"Fetching orders to find {target} ...", flush=True)
        orders = get_orders(email, jwt, eut)
        match = [o for o in orders if o.get("order_name") == target or o.get("sfdc_id") == target]
        if not match:
            print(f"Order {target} not found.")
            sys.exit(1)
        order = match[0]
        sfdc_id = order["sfdc_id"]
        print(f"sfdc_id: {sfdc_id}")
        print(f"order keys: {list(order.keys())}\n")
        items = get_order_items(email, jwt, eut, sfdc_id)
        print(f"{len(items)} items found.")
        if items:
            print(f"First item keys: {list(items[0].keys())}\n")
            print(json.dumps(items[:3], indent=2, default=str))
        return

    # --plates: all delivered plate maps
    if args.plates:
        print("Fetching plate maps ...", flush=True)
        maps = get_plate_maps(email, jwt, eut)
        if not maps:
            print("No plate maps found.")
            return
        df = pd.DataFrame(maps)
        for col in ("delivery_date",):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        print(f"\n--- Plate Maps ({len(df)} containers) ---")
        print(df.to_string(index=False))
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved to {args.output}")
        return

    # --plates-csv: get real Twist plate map ZIP download URLs per shipment
    if args.plates_csv:
        outfile = args.plates_csv
        print("Fetching orders and shipment plate map URLs ...", flush=True)
        all_orders = get_orders(email, jwt, eut)
        rows = []
        for order in all_orders:
            sfdc_id = order.get("sfdc_id")
            if not sfdc_id:
                continue
            for shipment in (order.get("shipments") or []):
                shipment_id = shipment.get("id")
                if not shipment_id:
                    continue
                # Only fetch for shipped/received shipments
                if shipment.get("status") not in ("shipped", "received"):
                    continue
                url = (f"{BASE_URL}/v1/users/{email}/orders/{sfdc_id}"
                       f"/shipments/{shipment_id}/plate-maps/")
                resp = requests.get(url, headers=_headers(jwt, eut))
                if resp.status_code != 200:
                    continue
                zip_url = resp.json().get("platemaps_file_url", "")
                rows.append({
                    "order_name":      order.get("order_name"),
                    "project_name":    order.get("project_name"),
                    "shipment_id":     shipment_id,
                    "tracking_number": shipment.get("tracking_number"),
                    "shipped_date":    _fmt_date(shipment.get("shipped_date")),
                    "shipment_status": shipment.get("status"),
                    "platemaps_zip_url": zip_url,
                })
                print(f"  {order.get('order_name')} shipment {shipment_id[:8]}... → {zip_url[:60]}...")

        if not rows:
            print("No plate map URLs found.")
            return
        df = pd.DataFrame(rows)
        df.to_csv(outfile, index=False)
        print(f"\n{len(df)} shipment plate map URLs written to {outfile}")
        return

    # Default: open orders with per-part ETAs
    print("Fetching open orders ...", flush=True)
    orders = get_orders(email, jwt, eut, order_status="open")
    print(f"  {len(orders)} open orders\n")

    all_rows = []
    for order in orders:
        sfdc_id = order.get("sfdc_id")
        if not sfdc_id:
            continue
        print(f"  {order['order_name']} ({order.get('project_name', '')}) ...", flush=True)
        try:
            items = get_order_items(email, jwt, eut, sfdc_id)
            all_rows.extend(_item_rows(order, items))
        except requests.HTTPError as e:
            print(f"    Warning: {e}")
            # Fall back to order-level row
            all_rows.append({
                "order":  order.get("order_name"),
                "part":   f"({order.get('total_items')} items — fetch failed)",
                "status": order.get("status", ""),
                "eta":    _fmt_date(order.get("estimated_completion_date")),
            })

    if not all_rows:
        print("No items found.")
        return

    df = pd.DataFrame(all_rows)
    print(f"\n--- Open order items ({len(df)} parts) ---")
    print(df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

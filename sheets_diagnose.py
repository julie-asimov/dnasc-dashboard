#!/usr/bin/env python3
"""
sheets_diagnose.py — read-only report on the due-dates Google Sheet.

Run this wherever the pipeline actually runs (the server) to answer the
questions a local machine cannot:

  1. Which sheet ID is the pipeline pointed at, and is that the live document?
  2. Can this host read the sheet at all, and with which quota project?
  3. Does the sheet contain duplicate appended rows (the repeated-experiment
     blocks), and how many rows are junk?
  4. Is the renderer getting sheet dates, or silently falling back to the
     stale CSV?
  5. Which experiments have a date in the sheet but are missing from
     due_dates.json — i.e. would be wrongly flagged "missing an Asana due date"?

WRITES NOTHING. It never calls append_experiment_names() and never touches
due_dates.json. Safe to run on the server at any time, including mid-cron.

Usage:
    python3 sheets_diagnose.py                 # config sheet + the known alternate
    python3 sheets_diagnose.py <sheet_id> ...  # also probe extra sheet ids

Use the same interpreter that runs full_refresh.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# The id that appeared in a pasted sheet URL, which differs from config.
# Probed alongside the configured id to settle which document is authoritative.
ALTERNATE_SHEET_ID = "1MXlnr7-T_7VFYFGcVDo-dg5tHvTPXxidOo2VPqPPkvQ"

READ_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def _rule(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def _creds():
    from google.auth import default
    from google.auth.transport.requests import Request as GoogleRequest

    creds, proj = default(scopes=READ_SCOPES)
    creds.refresh(GoogleRequest())
    return creds, proj


def _describe_creds(creds, adc_project) -> None:
    kind = type(creds).__name__
    print(f"credential class : {kind}")
    print(f"ADC project      : {adc_project or '(none)'}")
    for attr in ("service_account_email", "quota_project_id"):
        if getattr(creds, attr, None):
            print(f"{attr:17s}: {getattr(creds, attr)}")
    scopes = getattr(creds, "scopes", None)
    print(f"scopes           : {', '.join(scopes) if scopes else '(not reported)'}")
    if kind == "Credentials" and not getattr(creds, "service_account_email", None):
        print("note             : user ADC, not a service account — Sheets needs a "
              "quota project with serviceusage.serviceUsageConsumer")


def _try_read(creds, sheet_id: str, quota_project: str | None):
    """Return (status_code, values|None, error_message)."""
    import requests

    url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/Sheet1"
    headers = {"Authorization": f"Bearer {creds.token}"}
    if quota_project:
        headers["x-goog-user-project"] = quota_project
    try:
        r = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", "")
        except Exception:
            msg = r.text[:200]
        return r.status_code, None, msg
    return 200, r.json().get("values", []), None


def _analyze(values: list[list[str]]) -> dict:
    """Duplicate / junk analysis on raw sheet values."""
    if not values:
        return {"rows": 0}
    header, rows = values[0], values[1:]
    name_i = 0
    # due date column: prefer the documented header, else last column
    due_i = None
    for i, h in enumerate(header):
        if str(h).strip().lower() in ("due_date_in_asana", "date_in_asana", "due_date"):
            due_i = i
            break
    if due_i is None:
        due_i = len(header) - 1

    def cell(row, i):
        return str(row[i]).strip() if len(row) > i else ""

    names = [cell(r, name_i) for r in rows]
    nonblank = [n for n in names if n]
    dated = [r for r in rows if cell(r, due_i)]

    # Blank column A is NOT one thing. A row with content elsewhere is a
    # misplaced append; a row with nothing at all is just grid padding and
    # matters to nobody. Reporting them as a single number and calling them all
    # "appended rows in the wrong column" is how this script raised a false
    # alarm after a successful cleanup.
    misplaced = sum(1 for r in rows
                    if not (r and str(r[0]).strip()) and any(str(c).strip() for c in r))
    empty = (len(names) - len(nonblank)) - misplaced

    counts: dict[str, int] = {}
    for n in nonblank:
        counts[n] = counts.get(n, 0) + 1
    dups = {n: c for n, c in counts.items() if c > 1}

    junk_url = [n for n in nonblank if "docs.google.com" in n or n.startswith("http")]
    ws = [n for n in names if n != n.strip() and n.strip()]

    return {
        "header": header,
        "rows": len(rows),
        "blank_names": len(names) - len(nonblank),
        "misplaced": misplaced,
        "empty": empty,
        "unique_names": len(counts),
        "dated_rows": len(dated),
        "dup_names": len(dups),
        "dup_extra_rows": sum(c - 1 for c in dups.values()),
        "worst": sorted(dups.items(), key=lambda kv: -kv[1])[:8],
        "junk_url_rows": junk_url,
        "whitespace_names": len(ws),
        "dated_map": {cell(r, name_i): cell(r, due_i) for r in dated if cell(r, name_i)},
    }


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dnasc.config import PipelineConfig as C

    configured = C.DUE_DATES_SHEET_ID
    quota = C.DUE_DATES_QUOTA_PROJECT or None

    _rule("CONFIG")
    print(f"DUE_DATES_SHEET_ID     : {configured}")
    print(f"DUE_DATES_QUOTA_PROJECT: {quota or '(empty)'}")
    print(f"DUE_DATES_CSV_FALLBACK : {C.DUE_DATES_CSV_FALLBACK}")

    _rule("CREDENTIALS ON THIS HOST")
    try:
        creds, adc_project = _creds()
        _describe_creds(creds, adc_project)
    except Exception as e:
        print(f"FAILED to obtain credentials: {type(e).__name__}: {e}")
        return 1

    candidates = [("configured", configured)]
    if ALTERNATE_SHEET_ID != configured:
        candidates.append(("alternate", ALTERNATE_SHEET_ID))
    for extra in sys.argv[1:]:
        candidates.append(("argv", extra))

    # Try each sheet with no quota project, then with the configured one, then
    # with the ADC project — first success wins and is reported.
    quota_attempts = [None]
    for q in (quota, adc_project):
        if q and q not in quota_attempts:
            quota_attempts.append(q)

    readable: dict[str, dict] = {}
    for label, sid in candidates:
        _rule(f"SHEET {label}: {sid}")
        got = False
        for q in quota_attempts:
            status, values, err = _try_read(creds, sid, q)
            tag = f"quota={q or '(none)'}"
            if status == 200:
                print(f"  {tag:42s} -> 200 OK, {len(values)} raw rows")
                info = _analyze(values)
                readable[sid] = info
                print(f"      header          : {info['header']}")
                print(f"      data rows        : {info['rows']}")
                print(f"      unique names     : {info['unique_names']}")
                print(f"      rows with a date : {info['dated_rows']}")
                print(f"      blank-name rows  : {info['blank_names']}")
                print(f"      duplicate names  : {info['dup_names']}  "
                      f"(extra rows: {info['dup_extra_rows']})")
                if info["worst"]:
                    print("      worst offenders:")
                    for n, c in info["worst"]:
                        print(f"        x{c:<4d} {n[:60]}")
                if info["junk_url_rows"]:
                    print(f"      URL/junk rows    : {len(info['junk_url_rows'])}")
                    for n in info["junk_url_rows"][:3]:
                        print(f"        {n[:70]}")
                if info["whitespace_names"]:
                    print(f"      names with stray whitespace: {info['whitespace_names']}")
                got = True
                break
            print(f"  {tag:42s} -> {status}: {str(err)[:110]}")
        if not got:
            print("  NOT READABLE from this host")

    _rule("CSV FALLBACK")
    p = Path(C.DUE_DATES_CSV_FALLBACK)
    if p.exists():
        age = (time.time() - p.stat().st_mtime) / 86400
        n = max(0, sum(1 for _ in p.open()) - 1)
        print(f"exists: {p}  ({n} rows, {age:.0f} days old)")
        print("If the sheet read above succeeded, this file is NOT being used.")
    else:
        print(f"absent: {p}")

    _rule("due_dates.json (what the renderer consumed last run)")
    dd_entries = 0
    j = Path("dashboard_state/due_dates.json")
    if j.exists():
        age = (time.time() - j.stat().st_mtime) / 86400
        try:
            data = json.loads(j.read_text())
        except Exception as e:
            data = {}
            print(f"unreadable: {e}")
        dd_entries = len(data)
        print(f"entries: {dd_entries}   written {age:.1f} days ago")
        sheet_info = readable.get(configured) or next(iter(readable.values()), None)
        if sheet_info:
            missing = {k: v for k, v in sheet_info["dated_map"].items() if k not in data}
            print(f"\ndated in sheet but ABSENT from due_dates.json: {len(missing)}")
            print("(each of these renders as 'missing an Asana due date')")
            for k, v in sorted(missing.items())[:40]:
                print(f"  {v}  {k[:70]}")
            if len(missing) > 40:
                print(f"  ... and {len(missing) - 40} more")
            if not missing:
                print("  none — the renderer is seeing every dated experiment")
    else:
        print(f"absent: {j}")

    _rule("VERDICT")
    # Readability alone is NOT success: the read can return 200 and the parse still
    # die on a ragged row, leaving due_dates.json empty. Reporting "getting live
    # dates" off the HTTP status was actively misleading.
    if configured in readable:
        info = readable[configured]
        if not dd_entries:
            print("Configured sheet reads OK, but due_dates.json is EMPTY — the read")
            print("succeeds and something downstream drops the data. Prime suspect:")
            print(f"rows wider than the {len(info['header'])}-column header crashing the parse.")
        elif dd_entries < info["dated_rows"]:
            print(f"Configured sheet reads OK, but due_dates.json has {dd_entries} of "
                  f"{info['dated_rows']} dated rows — partial load.")
        else:
            print("Configured sheet IS readable and fully loaded — dates are live.")
        named = info["rows"] - info["blank_names"]
        print(f"\nrows with a name in column A : {named}   <- the real content")
        print(f"misplaced (content, no col A): {info['misplaced']}")
        print(f"empty rows (nothing at all)  : {info['empty']}   <- harmless padding")
        if info["misplaced"]:
            print("\nThe misplaced rows are appends that landed outside column A. The parser")
            print("skips them (blank experiment_name). Clear them with sheets_cleanup.py.")
        else:
            print("\nNo misplaced rows. Empty rows are grid padding and affect nothing —")
            print("they are NOT failed appends.")
    elif readable:
        others = ", ".join(readable)
        print(f"Configured sheet NOT readable, but these are: {others}")
        print("=> DUE_DATES_SHEET_ID in config.py is probably pointing at the wrong doc.")
    else:
        print("No sheet readable from this host — the renderer is on the stale CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

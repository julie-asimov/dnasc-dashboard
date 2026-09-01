#!/usr/bin/env python3
"""
sheets_cleanup.py — tidy the due-dates sheet.

Two kinds of removable row:

  junk   — content somewhere but a BLANK column A. Appends that landed outside
           column A (see v1.11.82). They carry only an experiment name, never a
           date, so the parser already skips them; deleting them loses nothing.
           Always targeted.

  empty  — nothing at all. NOT harmless, despite what this script said at first.
           appendCells appends after the sheet's LAST row, so a block of trailing
           empties pushes every newly synced name hundreds of rows below the real
           data, where nobody scrolling the sheet will find it. Removed only with
           --trim-empty, since deleting rows is the more invasive of the two.

NEVER touches the header, or any row with something in column A (every real
experiment, including ones with a name but no date yet).

DRY RUN BY DEFAULT. It prints what it would delete, and the row numbers your real
rows occupy, then exits. Nothing is written without --apply.

    python3 sheets_cleanup.py                          # show me
    python3 sheets_cleanup.py --apply                  # delete junk
    python3 sheets_cleanup.py --apply --trim-empty     # delete junk + padding

Run it where the pipeline runs — the service account there can write.
"""
from __future__ import annotations

import sys
from pathlib import Path

WRITE_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _ranges(rows: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted 1-indexed row numbers into [start, end] inclusive runs."""
    out: list[tuple[int, int]] = []
    for r in rows:
        if out and r == out[-1][1] + 1:
            out[-1] = (out[-1][0], r)
        else:
            out.append((r, r))
    return out


def main() -> int:
    apply = "--apply" in sys.argv
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import requests
    from google.auth import default
    from google.auth.transport.requests import Request as GoogleRequest
    from dnasc.config import PipelineConfig as C

    creds, _ = default(scopes=WRITE_SCOPES)
    creds.refresh(GoogleRequest())
    base = f"https://sheets.googleapis.com/v4/spreadsheets/{C.DUE_DATES_SHEET_ID}"
    headers = {"Authorization": f"Bearer {creds.token}"}
    if C.DUE_DATES_QUOTA_PROJECT:
        headers["x-goog-user-project"] = C.DUE_DATES_QUOTA_PROJECT

    r = requests.get(f"{base}/values/Sheet1", headers=headers, timeout=30)
    if r.status_code != 200:
        print(f"read failed {r.status_code}: {r.text[:300]}")
        return 1
    values = r.json().get("values", [])
    if len(values) < 2:
        print("sheet has no data rows — nothing to do")
        return 0

    keep_named, junk, blank = [], [], []
    for i, row in enumerate(values[1:], start=2):        # row 1 = header
        col_a = str(row[0]).strip() if row else ""
        has_any = any(str(c).strip() for c in row)
        if col_a:
            keep_named.append(i)
        elif has_any:
            junk.append(i)                              # content, but not in column A
        else:
            blank.append(i)

    trim_empty = "--trim-empty" in sys.argv

    print(f"header            : row 1  {values[0]}")
    print(f"real rows (col A) : {len(keep_named)}  -> KEEP")
    if keep_named:
        nr = _ranges(keep_named)
        print(f"                    at rows {', '.join(f'{a}-{b}' if a != b else str(a) for a, b in nr[:6])}"
              + (" ..." if len(nr) > 6 else ""))
    print(f"junk (blank col A): {len(junk)}  -> DELETE")
    # Empty rows are NOT harmless. appendCells appends after the sheet's LAST
    # row, so a block of trailing empties pushes every new name hundreds of rows
    # below the real data, where nobody scrolling the sheet will ever see it.
    print(f"fully empty rows  : {len(blank)}  -> "
          + ("DELETE (--trim-empty)" if trim_empty else "KEEP  (pass --trim-empty to remove)"))
    if blank and not trim_empty:
        print("                    ^ these push new appends to the bottom of the sheet")

    targets = sorted(junk + (blank if trim_empty else []))
    if not targets:
        print("\nnothing to delete.")
        return 0

    runs = _ranges(targets)
    print(f"\n{len(runs)} contiguous block(s):")
    for start, end in runs[:20]:
        n = end - start + 1
        sample = values[start - 1]
        text = next((str(c).strip() for c in sample if str(c).strip()), "")
        print(f"  rows {start}-{end}  ({n:4d})  e.g. {text[:60]}")
    if len(runs) > 20:
        print(f"  ... and {len(runs) - 20} more block(s)")

    if not apply:
        print(f"\nDRY RUN — nothing written. Re-run with --apply to delete "
              f"{len(targets)} row(s).")
        return 0

    # Delete bottom-up so earlier row numbers stay valid as rows disappear.
    meta = requests.get(f"{base}?fields=sheets(properties(sheetId,title))",
                        headers=headers, timeout=30)
    gid = None
    for s in meta.json().get("sheets", []):
        if s.get("properties", {}).get("title") == "Sheet1":
            gid = s["properties"].get("sheetId")
            break
    if gid is None:
        print("could not resolve the Sheet1 sheetId — aborting")
        return 1

    requests_body = [
        {"deleteDimension": {"range": {
            "sheetId": gid, "dimension": "ROWS",
            "startIndex": start - 1,      # API is 0-indexed, end-exclusive
            "endIndex": end,
        }}}
        for start, end in reversed(runs)
    ]
    resp = requests.post(f"{base}:batchUpdate", headers=headers,
                         json={"requests": requests_body}, timeout=120)
    if resp.status_code != 200:
        print(f"delete failed {resp.status_code}: {resp.text[:400]}")
        return 1
    print(f"\ndeleted {len(targets)} row(s) in {len(runs)} block(s).")
    print("Verify with: sheets_diagnose.py  (expect blank-name rows: 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
sheets_verify_append.py — live round-trip test: does an append land in column A?

Unit tests prove the code builds the right request. Only this proves what Google
actually does to the real sheet. It runs the production code path
(append_experiment_names), then reads the sheet back and reports WHICH COLUMN the
row landed in, then removes its own canary row.

    python3 sheets_verify_append.py

Writes exactly one row (a uniquely-named canary) and deletes it again. Touches
nothing else. Run it where the pipeline runs.

Exit 0 = the canary landed in column A (fixed).
Exit 1 = it landed elsewhere, or something else went wrong (still broken).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

WRITE_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _client():
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
    return requests, base, headers


def _read(requests, base, headers):
    r = requests.get(f"{base}/values/Sheet1", headers=headers, timeout=30)
    if r.status_code != 200:
        print(f"  read failed {r.status_code}: {r.text[:200]}")
        return None
    return r.json().get("values", [])


def _survey(values):
    """(real rows with col A, junk rows with blank col A but content)."""
    real = junk = 0
    for row in values[1:]:
        if row and str(row[0]).strip():
            real += 1
        elif any(str(c).strip() for c in row):
            junk += 1
    return real, junk


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dnasc.extractors.sheets import append_experiment_names
    from dnasc.config import PipelineConfig as C

    requests, base, headers = _client()

    print(f"sheet   : {C.DUE_DATES_SHEET_ID}")
    print(f"version : {C.PIPELINE_VERSION}   <- must be 1.11.82 or newer\n")

    print("BEFORE")
    before = _read(requests, base, headers)
    if before is None:
        return 1
    real, junk = _survey(before)
    print(f"  data rows        : {len(before) - 1}")
    print(f"  real (col A set) : {real}")
    print(f"  junk (col A blank): {junk}")
    if junk:
        print(f"  ^ {junk} junk row(s) still present. This test does not remove them;")
        print("    use sheets_cleanup.py, and make sure cron is stopped first.")

    canary = f"__CANARY_DO_NOT_USE_{int(time.time())}"
    print(f"\nAPPEND\n  writing canary: {canary}")
    res = append_experiment_names([canary])
    print(f"  ok={res['ok']}  appended={len(res['appended'])}  error={res['error']}")
    if not res["ok"] or not res["appended"]:
        print("\nFAIL — the append did not run. Nothing to verify.")
        return 1

    print("\nAFTER")
    after = _read(requests, base, headers)
    if after is None:
        return 1

    hit_row = hit_col = None
    for idx, row in enumerate(after[1:], start=2):
        for col, cell in enumerate(row):
            if str(cell).strip() == canary:
                hit_row, hit_col = idx, col
                break
        if hit_row:
            break

    if hit_row is None:
        print("  canary NOT FOUND in the sheet — append reported success but wrote nothing?")
        return 1

    letter = chr(ord("A") + hit_col)
    print(f"  canary found at row {hit_row}, column {letter} (index {hit_col})")

    # Clean up the canary regardless of pass/fail.
    meta = requests.get(f"{base}?fields=sheets(properties(sheetId,title))",
                        headers=headers, timeout=30)
    gid = None
    for s in meta.json().get("sheets", []):
        if s.get("properties", {}).get("title") == "Sheet1":
            gid = s["properties"].get("sheetId")
            break
    if gid is not None:
        d = requests.post(f"{base}:batchUpdate", headers=headers, timeout=30,
                          json={"requests": [{"deleteDimension": {"range": {
                              "sheetId": gid, "dimension": "ROWS",
                              "startIndex": hit_row - 1, "endIndex": hit_row,
                          }}}]})
        print(f"  canary row removed: {d.status_code == 200}")
    else:
        print("  WARNING: could not resolve sheetId — delete the canary row by hand")

    print()
    if hit_col == 0:
        print("PASS — appends land in column A. Delete the junk rows and they stay gone.")
        return 0
    print(f"FAIL — appends still land in column {letter}, not A.")
    print("The v1.11.82 appendCells fix is not in effect here. Check that the server")
    print("pulled it (git log -1) and that PIPELINE_VERSION above says 1.11.82+.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

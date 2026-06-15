"""
dnasc/extractors/sheets.py
───────────────────────────
Reads experiment due dates from Google Sheets (or local CSV fallback).
Saves result to dashboard_state/due_dates.json for the renderer to consume.

To enable Google Sheets access, run once:
    gcloud auth application-default login \
        --scopes=https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/spreadsheets.readonly

Then have Ben enable the Sheets API and grant serviceusage.serviceUsageConsumer
on a quota project (e.g. foundry-prd).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger

log = get_logger(__name__)

DUE_DATES_FILE = Path("dashboard_state/due_dates.json")


def fetch_due_dates() -> dict[str, str]:
    """
    Returns {experiment_name: due_date_str (YYYY-MM-DD)}.
    Tries Google Sheets first; falls back to local CSV if Sheets is unavailable.
    Result is also written to dashboard_state/due_dates.json.
    """
    t0 = time.time()
    df = _try_google_sheets()
    if df is None:
        df = _try_csv_fallback()
    if df is None:
        log.warning("No due-date source available — skipping due dates")
        _save({})
        return {}

    # One entry per experiment name — last row wins if duplicates exist.
    # Schema: a single authoritative date (`due_date_in_asana`) drives the due
    # marker + the normal NGS/assembly back-calc. `date_sequence_transferred` is
    # informational only (the day sequences were delivered; precedes the BIOS
    # created date) and never re-anchors the timer.
    def _clean(v) -> str:
        s = str(v).strip()
        return "" if s in ("nan", "None", "") else s

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = _clean(row.get("experiment_name", ""))
        # Preferred column is `due_date_in_asana`; accept legacy headers as fallback.
        due = (_clean(row.get("due_date_in_asana", ""))
               or _clean(row.get("date_in_asana", ""))
               or _clean(row.get("due_date", ""))
               or _clean(row.get("date_in_cld_gnatt", "")))
        seq = (_clean(row.get("date_sequence_transferred", ""))
               or _clean(row.get("sequence_transferred", "")))
        if not name:
            continue
        if not due:
            continue
        result[name] = {
            # Internal key stays `due_date` so the In-Flight tab anchor needs no change.
            "due_date":             due,
            "sequence_transferred": seq,
        }

    _save(result)
    log.info("Due dates ready: %d experiments in %.1fs", len(result), time.time() - t0)
    return result


def load_due_dates() -> dict[str, str]:
    """Load previously saved due_dates.json without re-fetching."""
    if DUE_DATES_FILE.exists():
        try:
            return json.loads(DUE_DATES_FILE.read_text())
        except Exception:
            pass
    return {}


def append_experiment_names(names: list[str]) -> dict:
    """
    Append experiment names (with blank date columns) to the Google Sheet so missing
    partner projects show up as rows to be filled in. Requires the service account to
    have *Editor* access on the sheet — read-only access (the default) will 403.

    Only names not already present are appended. Returns a status dict:
        {"appended": [...], "skipped_existing": [...], "ok": bool, "error": str|None}
    """
    result = {"appended": [], "skipped_existing": [], "ok": False, "error": None}
    names = [str(n).strip() for n in (names or []) if str(n).strip()]
    if not names:
        result["ok"] = True
        return result

    try:
        import requests
        from google.auth import default
        from google.auth.transport.requests import Request as GoogleRequest

        # Full read/write scope — append is a write.
        creds, _ = default(scopes=["https://www.googleapis.com/auth/spreadsheets"])
        creds.refresh(GoogleRequest())

        sheet_id = PipelineConfig.DUE_DATES_SHEET_ID
        quota_proj = PipelineConfig.DUE_DATES_QUOTA_PROJECT
        base = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}"
        headers = {"Authorization": f"Bearer {creds.token}"}
        if quota_proj:
            headers["x-goog-user-project"] = quota_proj

        # Existing names (column A, skip header) so we never duplicate a row.
        r = requests.get(f"{base}/values/Sheet1!A2:A", headers=headers, timeout=20)
        existing = set()
        if r.status_code == 200:
            for row in r.json().get("values", []):
                if row and str(row[0]).strip():
                    existing.add(str(row[0]).strip())
        else:
            log.warning("append: could not read existing names (%d): %s", r.status_code, r.text[:200])

        to_add = [n for n in dict.fromkeys(names) if n not in existing]
        result["skipped_existing"] = [n for n in names if n in existing]
        if not to_add:
            result["ok"] = True
            return result

        # 3 columns: experiment_name, date_sequence_transferred (blank), due_date_in_asana (blank).
        body = {"values": [[n, "", ""] for n in to_add]}
        ar = requests.post(
            f"{base}/values/Sheet1!A:C:append",
            headers=headers,
            params={"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
            json=body,
            timeout=20,
        )
        if ar.status_code == 200:
            result["appended"] = to_add
            result["ok"] = True
            log.info("Appended %d missing experiment(s) to the sheet", len(to_add))
        else:
            result["error"] = f"{ar.status_code}: {ar.text[:300]}"
            log.warning("append failed: %s", result["error"])
    except Exception as e:
        result["error"] = str(e)
        log.warning("append exception: %s", e)
    return result


# ── private ──────────────────────────────────────────────────────────────────

def _try_google_sheets() -> pd.DataFrame | None:
    try:
        import requests
        from google.auth import default
        from google.auth.transport.requests import Request as GoogleRequest

        creds, _ = default(scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
        creds.refresh(GoogleRequest())

        sheet_id = PipelineConfig.DUE_DATES_SHEET_ID
        quota_proj = PipelineConfig.DUE_DATES_QUOTA_PROJECT
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/Sheet1"
        headers = {"Authorization": f"Bearer {creds.token}"}
        if quota_proj:
            headers["x-goog-user-project"] = quota_proj

        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            log.warning("Sheets API %d: %s", r.status_code, r.text[:200])
            return None

        data = r.json()
        values = data.get("values", [])
        if len(values) < 2:
            log.warning("Sheet has no data rows")
            return None

        df = pd.DataFrame(values[1:], columns=values[0])
        log.info("Due dates loaded from Google Sheet: %d rows", len(df))
        return df

    except Exception as e:
        log.info("Google Sheets unavailable (%s) — trying CSV fallback", e)
        return None


def _try_csv_fallback() -> pd.DataFrame | None:
    path = Path(PipelineConfig.DUE_DATES_CSV_FALLBACK)
    if not path.exists():
        log.warning("CSV fallback not found: %s", path)
        return None
    df = pd.read_csv(path)
    log.info("Due dates loaded from CSV: %d rows", len(df))
    return df


def _save(data: dict) -> None:
    DUE_DATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    DUE_DATES_FILE.write_text(json.dumps(data, indent=2))

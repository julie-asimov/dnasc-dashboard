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
    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        name  = str(row.get("experiment_name", "")).strip()
        due   = str(row.get("due_date", "")).strip()
        gantt = str(row.get("date_in_cld_gnatt", "")).strip()
        if not name or name in ("nan", "None", ""):
            continue
        if not due or due in ("nan", "None", ""):
            continue
        result[name] = {
            "due_date":          due,
            "date_in_cld_gnatt": gantt if gantt not in ("nan", "None", "") else "",
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

"""
debug_sheets_auth.py
─────────────────────
Diagnoses Google Sheets API access step by step.
Run on the script server: /opt/anaconda3/bin/python3 debug_sheets_auth.py
"""
import sys, json
print("── Step 1: Import google-auth ──────────────────────────────────────────")
try:
    from google.auth import default
    from google.auth.transport.requests import Request as GoogleRequest
    print("  OK")
except ImportError as e:
    print(f"  FAIL — google-auth not installed: {e}")
    sys.exit(1)

print("\n── Step 2: Load application default credentials ────────────────────────")
try:
    creds, project = default(scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    print(f"  OK — project: {project}")
    print(f"  credential type: {type(creds).__name__}")
except Exception as e:
    print(f"  FAIL — {e}")
    print("\n  Fix: run this on the server:")
    print("    gcloud auth application-default login \\")
    print("      --scopes=https://www.googleapis.com/auth/cloud-platform,\\")
    print("              https://www.googleapis.com/auth/spreadsheets.readonly")
    sys.exit(1)

print("\n── Step 3: Refresh token ────────────────────────────────────────────────")
try:
    creds.refresh(GoogleRequest())
    print(f"  OK — token obtained (first 20 chars): {creds.token[:20]}...")
except Exception as e:
    print(f"  FAIL — {e}")
    sys.exit(1)

print("\n── Step 4: Call Sheets API ──────────────────────────────────────────────")
import requests as _req
from dnasc.config import PipelineConfig

sheet_id    = PipelineConfig.DUE_DATES_SHEET_ID
quota_proj  = PipelineConfig.DUE_DATES_QUOTA_PROJECT
url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/Sheet1"
headers = {"Authorization": f"Bearer {creds.token}"}
if quota_proj:
    headers["x-goog-user-project"] = quota_proj
    print(f"  quota project: {quota_proj}")
else:
    print("  WARNING: DUE_DATES_QUOTA_PROJECT is empty — may need to set this")

r = _req.get(url, headers=headers, timeout=15)
print(f"  HTTP {r.status_code}")

if r.status_code == 200:
    data   = r.json()
    values = data.get("values", [])
    print(f"  OK — {len(values)} rows (including header)")
    if len(values) >= 1:
        print(f"  Header row: {values[0]}")
    if len(values) >= 2:
        print(f"  First data row: {values[1]}")
elif r.status_code == 403:
    body = r.json()
    print(f"  FAIL 403 — {body.get('error', {}).get('message', r.text[:200])}")
    print("\n  Possible causes:")
    print("  1. Google Sheets API not enabled on the GCP project.")
    print("     Fix: https://console.developers.google.com/apis/api/sheets.googleapis.com/overview")
    print("  2. Credentials don't include spreadsheets scope.")
    print("     Fix: re-run gcloud auth application-default login with the sheets scope (see Step 2)")
    print("  3. Need a quota project. Ask Ben to enable Sheets API and set DUE_DATES_QUOTA_PROJECT in config.py.")
elif r.status_code == 404:
    print(f"  FAIL 404 — sheet not found or not shared. Check sheet ID and that it's shared with the service account.")
else:
    print(f"  FAIL — {r.text[:300]}")

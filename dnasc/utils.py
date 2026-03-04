"""
dnasc/utils.py
───────────────
Shared helper functions used across extractors, transformers, and the renderer.
Nothing here imports from other dnasc submodules — it is the bottom of the
dependency graph.
"""

from __future__ import annotations
import re
import json
from datetime import datetime
from typing import Any

import pandas as pd
import pytz


# ── Timezone conversion ───────────────────────────────────────────────────────

_EST = pytz.timezone("US/Eastern")


def to_est(dt_val: Any) -> datetime | None:
    """Convert any datetime-like value to US/Eastern. Returns None on failure."""
    if pd.isna(dt_val) or dt_val == "":
        return None
    try:
        dt = pd.to_datetime(dt_val)
        if dt.tz is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert(_EST)
    except Exception:
        return None


# ── Plate ID helpers ──────────────────────────────────────────────────────────

def clean_plate_id(val: Any) -> str | None:
    """Extract the numeric part of a plate identifier string."""
    if pd.isna(val):
        return None
    m = re.search(r"(\d+)", str(val))
    return m.group(1) if m else None


# ── JSON helpers ──────────────────────────────────────────────────────────────

def safe_json_name(s: Any) -> str | None:
    """Parse a JSON string and return the 'name' field, or None."""
    try:
        return json.loads(s).get("name")
    except Exception:
        return None


def parse_backbone(x: Any) -> str:
    """Return 'name:available' string from a backbone JSON blob."""
    try:
        d = json.loads(x)
        return f"{d.get('name', '')}:{d.get('available', '')}" if d else ""
    except Exception:
        return ""


def parse_parts(x: Any) -> str:
    """Return comma-joined 'name:available' strings from a parts JSON array."""
    try:
        d = json.loads(x)
        return ", ".join(f"{i['name']}:{i.get('available', '')}" for i in d) if d else ""
    except Exception:
        return ""


def extract_pcr_info(row: pd.Series) -> str:
    """Build a comma-joined PCR parts string from forward/reverse/template columns."""
    parts = []
    try:
        if pd.notna(row.get("pcr_forward_primer")):
            fwd = json.loads(row["pcr_forward_primer"])
            if fwd.get("name"):
                parts.append(f"{fwd['name']}:{fwd.get('available', '')}")
        if pd.notna(row.get("pcr_reverse_primer")):
            rev = json.loads(row["pcr_reverse_primer"])
            if rev.get("name"):
                parts.append(f"{rev['name']}:{rev.get('available', '')}")
        if pd.notna(row.get("pcr_templates")):
            for t in json.loads(row["pcr_templates"]):
                if t.get("name"):
                    parts.append(f"{t['name']}:{t.get('available', '')}")
    except Exception:
        pass
    return ", ".join(parts)


# ── pAI sort key ──────────────────────────────────────────────────────────────

_PAI_RE = re.compile(r"pAI-(\d+)", re.IGNORECASE)


def get_pai_sort_key(req_df: pd.DataFrame) -> int:
    """Return the lowest pAI number found in a request DataFrame, for sorting."""
    stock_ids = req_df["STOCK_ID"].dropna().astype(str).values
    min_pai = 99_999_999
    for s in stock_ids:
        m = _PAI_RE.search(s)
        if m:
            min_pai = min(min_pai, int(m.group(1)))
    return min_pai


# ── Type label formatting ─────────────────────────────────────────────────────

def format_type_label(type_str: str) -> str:
    """Convert snake_case workorder type to a human-readable label."""
    label = type_str.replace("_workorder", "").replace("_", " ").title()
    return (
        label
        .replace("Lsp", "LSP")
        .replace("Pcr", "PCR")
        .replace("Ngs", "NGS")
    )


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def ensure_list(x: Any) -> list:
    """Coerce lists, arrays, NaN, and None to a plain Python list."""
    if isinstance(x, (list,)):
        return x
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    return []


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate column names, keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()]

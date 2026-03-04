"""
dnasc/config.py
────────────────
Central configuration for the DNA SC pipeline.
All tuneable constants live here — nothing else imports os/datetime for config.
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta
import pytz

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class PipelineConfig:
    """Centralized configuration for the pipeline."""

    # ── BigQuery ──────────────────────────────────────────────────────────────
    PROJECT_ID: str = "data-platform-core-prd"

    # ── Data filtering ────────────────────────────────────────────────────────
    DATE_FILTER: str = "2025-01-01"          # Historical cutoff (full mode)

    # ── Incremental mode ──────────────────────────────────────────────────────
    INCREMENTAL_MODE: bool = False
    INCREMENTAL_HOURS: int = 24              # How far back to look in incremental mode

    # ── Caching ───────────────────────────────────────────────────────────────
    CACHE_FILE: str = "dashboard_data.parquet"
    ENABLE_CACHE: bool = True

    # ── Validation ────────────────────────────────────────────────────────────
    ENABLE_VALIDATION: bool = True
    YIELD_TOLERANCE: float = 0.1             # µg tolerance for yield validation

    # ── LSP ───────────────────────────────────────────────────────────────────
    LSP_BLACKLIST: list[str] = ["LSP-7602"]
    LSP_CUTOFF_DATE: str = "2025-11-01"      # Secondary-pass identity recovery cutoff

    # ── Pipeline version (bump on every code push) ────────────────────────────
    PIPELINE_VERSION: str = "1.0.0"

    @classmethod
    def get_date_filter(cls) -> str:
        """
        Return the date filter string for BigQuery WHERE clauses.
        In incremental mode, returns a timestamp offset from now.
        In full mode, returns the historical DATE_FILTER constant.
        """
        if cls.INCREMENTAL_MODE and os.path.exists(cls.CACHE_FILE):
            cutoff = datetime.now(pytz.UTC) - timedelta(hours=cls.INCREMENTAL_HOURS)
            filter_date = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            return filter_date
        return cls.DATE_FILTER

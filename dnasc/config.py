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
    # Tracking history window — a ROLLING window ending today, so the refresh stops
    # growing as the lab accumulates history. The old fixed "2025-01-01" cutoff was
    # 19 months wide and widening every day.
    #
    # Verified safe at 365 days against the current baseline: it drops 38% of rows,
    # and of the 282 live requests, ZERO have a root_work_order_id or attempt_anchor_id
    # falling outside the window (the one live req_id spanning it is ACTIVE_WIP, the
    # synthetic no-request bucket, which the In-Flight tab already excludes).
    #
    # This is the TRACKING window only. The Parts tab is unaffected — gen_parts_pkl.py
    # carries its own freshness windows (200d stock, 730d oligos/orders).
    HISTORY_DAYS: int = 365
    DATE_FILTER_PIN: str = ""                # 'YYYY-MM-DD' to pin a fixed cutoff instead
    DATE_FILTER: str = (DATE_FILTER_PIN or
                        (datetime.now(pytz.UTC) - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d"))

    # ── Incremental mode ──────────────────────────────────────────────────────
    INCREMENTAL_MODE: bool = False
    INCREMENTAL_HOURS: int = 24              # How far back to look in incremental mode

    # ── Caching ───────────────────────────────────────────────────────────────
    CACHE_FILE: str = "dashboard_data.parquet"
    ENABLE_CACHE: bool = True

    # ── Validation ────────────────────────────────────────────────────────────
    ENABLE_VALIDATION: bool = True
    YIELD_TOLERANCE: float = 0.1             # µg tolerance for yield validation

    # ── Due-date Google Sheet ─────────────────────────────────────────────────
    DUE_DATES_SHEET_ID: str     = "1qnZdQcviM83FW2ELC-KUAYYLaZoC8GDfjNjjFyB4z0k"
    DUE_DATES_QUOTA_PROJECT: str = ""
    DUE_DATES_CSV_FALLBACK: str = "due_dates/due_dates_override.csv"   # local Sheets fallback

    # ── LSP ───────────────────────────────────────────────────────────────────
    LSP_BLACKLIST: list[str] = ["LSP-7602"]
    LSP_CUTOFF_DATE: str = "2025-11-01"      # Secondary-pass identity recovery cutoff

    # ── Colony Tracking view (Requests In Flight tab) ─────────────────────────
    MIN_PICKABLE_COLONIES: int = 3   # request-level pickable sum below this → red flag

    # ── Colony pickable bands (Requests In Flight) ────────────────────────────
    # Calibrated on 2,575 colony-bearing workorders / 1,342 attempts (Jan 2025-Jul 2026),
    # measured PER STRAIN: median 11, p75 26, p25 3, p20 2.
    #
    # The unit matters. The old hardcoded 7/22 matched the per-workorder distribution but
    # was applied to attempt TOTALS as well, where 7 is only the 22nd percentile. Since an
    # attempt sums its strain transformations, a 1-strain attempt was judged against the
    # same yardstick as a 2-strain one: 1-strain median 12 vs 2-strain 26, and 1-strain
    # attempts were ~1.9x more likely to be called LOW (Fisher p=0.0002). Per-strain yield
    # is near-identical between them (12 vs 13), so totals are roughly additive and dividing
    # by the strain count compares like with like.
    PICK_BAND_LOW_MAX: int = 11    # below the per-strain median
    PICK_BAND_MED_MAX: int = 26    # median -> p75
    # Risk is a stricter, separate question — "about to run out of viable picks" — so it does
    # NOT reuse the descriptive band. p25 per strain is 3, which is what MIN_PICKABLE_COLONIES
    # already uses for the per-row low-pick flag; MED watches up to the median.
    COLONY_RISK_HIGH_MAX: int = 3
    COLONY_RISK_MED_MAX:  int = 11

    # ── Parts restock buffer ──────────────────────────────────────────────────
    # Spare stock to hold on top of the immediate need, as a fraction of that need,
    # with REFILL_BUFFER_MIN as a floor for small needs.
    #
    # This is a STOCKING POLICY, not a prediction — it says how much spare to keep,
    # and the tab labels it as such rather than presenting it as a forecast. It used
    # to be 1.0 (hold 100% of need), which assumed every build might need doing twice
    # and put parts with ~2x coverage on the restock list; that same full-retry case is
    # already shown explicitly as the upper end of the target range, so counting it
    # here too double-counted the pessimism.
    #
    # The floor was 10, which was a huge ask for a dPart consumed 2 at a time: parts with
    # 3.5x-5.5x coverage (d8269 at 11 on hand for a need of 2) were told to run a PCR.
    # At 5 those clear while genuinely thin stock (d8260, 4 on hand for a need of 2) still
    # flags. The floor governs until the need reaches 7 (0.8 * 7 > 5).
    REFILL_BUFFER_FRAC: float = 0.80
    REFILL_BUFFER_MIN:  int   = 5

    # ── Pinned infrastructure experiments ────────────────────────────────────
    # Ongoing reference/infra projects (not normal customer experiments). Single
    # source of truth — used by the In-Flight tab (_PINNED_EXPS: sort to bottom,
    # hide their terminal requests) AND the Tracking tab (_NO_TIMELINE_MARKERS:
    # no timeline markers, exempt from the "missing Asana date" flag + due-date
    # sheet append). Keep this list authoritative; both tabs read it.
    PINNED_INFRA_EXPERIMENTS: frozenset = frozenset({
        "LSP Refill Requests",
        "A469-Build DNASC CHO Destination Vectors",
        "A385-DNASC_RD",
    })

    # ── Pipeline version (bump on every code push) ────────────────────────────
    PIPELINE_VERSION: str = "1.11.26"

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

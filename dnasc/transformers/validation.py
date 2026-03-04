"""
dnasc/transformers/validation.py
──────────────────────────────────
Data quality validation — checks qubit/nanodrop yields against
calculated values and reports mismatches.
"""

from __future__ import annotations

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger

log = get_logger(__name__)


class ValidationTransformer:
    """Validate yield columns and backfill missing STOCK_IDs."""

    @staticmethod
    def validate_yields(df: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Compare stored qubit/nanodrop yields against concentration × volume.

        Returns
        ───────
        (qubit_mismatch_df, nanodrop_mismatch_df)
        Both are None when validation is disabled.
        """
        if not PipelineConfig.ENABLE_VALIDATION:
            return None, None

        tol = PipelineConfig.YIELD_TOLERANCE
        log.info("Validating yields (tolerance: %.3f µg)...", tol)

        df["qubit_yield_calc"]    = (df["qubit_concentration_ngul"]    * df["total_volume_ul"]) / 1000
        df["nanodrop_yield_calc"] = (df["nanodrop_concentration_ngul"] * df["total_volume_ul"]) / 1000

        qubit_mismatch = df[
            df["qubit_yield"].notna() &
            df["qubit_yield_calc"].notna() &
            (abs(df["qubit_yield"] - df["qubit_yield_calc"]) > tol)
        ][["lsp_batch_id", "workorder_id", "qubit_yield", "qubit_yield_calc",
           "qubit_concentration_ngul", "total_volume_ul"]].copy()

        nanodrop_mismatch = df[
            df["nanodrop_yield"].notna() &
            df["nanodrop_yield_calc"].notna() &
            (abs(df["nanodrop_yield"] - df["nanodrop_yield_calc"]) > tol)
        ][["lsp_batch_id", "workorder_id", "nanodrop_yield", "nanodrop_yield_calc",
           "nanodrop_concentration_ngul", "total_volume_ul"]].copy()

        if qubit_mismatch.empty:
            log.info("Qubit yields validated ✓")
        else:
            log.warning("%d rows have qubit_yield mismatch > %.3f µg", len(qubit_mismatch), tol)

        if nanodrop_mismatch.empty:
            log.info("Nanodrop yields validated ✓")
        else:
            log.warning("%d rows have nanodrop_yield mismatch > %.3f µg", len(nanodrop_mismatch), tol)

        df.drop(["qubit_yield_calc", "nanodrop_yield_calc"], axis=1, inplace=True, errors="ignore")

        # Backfill STOCK_ID from plasmid_id for synthetic LSPs
        if "plasmid_id" in df.columns:
            missing = df["STOCK_ID"].isna() & df["plasmid_id"].notna()
            df.loc[missing, "STOCK_ID"] = df.loc[missing, "plasmid_id"]
            missing_root = df["root_STOCK_ID"].isna() & df["plasmid_id"].notna()
            df.loc[missing_root, "root_STOCK_ID"] = df.loc[missing_root, "plasmid_id"]

        if "qubit_concentration_ngul" in df.columns:
            df["qubit_concentration"] = df["qubit_concentration_ngul"]

        return qubit_mismatch, nanodrop_mismatch

"""
dnasc/transformers/processing.py
──────────────────────────────────
Core workorder processing: JSON parsing, status enrichment,
yield calculation, STOCK_ID resolution, and source material linking.
"""

from __future__ import annotations
import json

import pandas as pd

from dnasc.logger import get_logger
from dnasc.utils import (
    safe_json_name,
    parse_backbone,
    parse_parts,
    extract_pcr_info,
)

log = get_logger(__name__)


class ProcessingTransformer:
    """Parse and enrich raw workorder data."""

    @staticmethod
    def process_workorder_data(df: pd.DataFrame) -> pd.DataFrame:
        log.info("Processing workorder data (%d rows)...", len(df))
        df = df.copy()

        if "source_lsp_process_id" in df.columns:
            df["source_lsp_process_id"] = (
                df["source_lsp_process_id"].astype(str).replace("nan", None)
            )

        # ── Cleanup: remove experiments with only canceled/empty work ─────────
        df = ProcessingTransformer._filter_canceled_experiments(df)

        # ── Dedup on workorder_id (safety net) ───────────────────────────────
        # BQ now returns 1 row per workorder via assembly_plan_id join.
        df = df.drop_duplicates(subset=["workorder_id"])

        # ── JSON parsing ──────────────────────────────────────────────────────
        df["backbone_json"] = df["backbone_json"].fillna("{}")

        if "STOCK_ID" not in df.columns:
            df["STOCK_ID"] = df["product_json"].apply(safe_json_name)
        df.drop("product_json", axis=1, inplace=True, errors="ignore")

        df["backbone"] = df["backbone_json"].apply(parse_backbone)
        df.drop("backbone_json", axis=1, inplace=True, errors="ignore")

        df["parts"] = df["parts_json"].apply(parse_parts)
        df.drop("parts_json", axis=1, inplace=True, errors="ignore")

        df["pcr_info"] = df.apply(extract_pcr_info, axis=1)
        df.drop(
            ["pcr_forward_primer", "pcr_reverse_primer", "pcr_templates"],
            axis=1, inplace=True, errors="ignore",
        )

        # ── Waiting parts ─────────────────────────────────────────────────────
        df["Waiting"] = df.apply(
            lambda row: ", ".join([
                item.split(":")[0]
                for col in ["backbone", "parts", "pcr_info"]
                for item in str(row.get(col, "")).split(", ")
                if "False" in item
            ]),
            axis=1,
        )

        # ── STOCK_ID from synthesis columns ───────────────────────────────────
        for col in ["synpartsynthesis_syn_part", "oligosynthesis_oligo", "plasmidsynthesis_plasmid"]:
            if col in df.columns:
                df["STOCK_ID"] = df["STOCK_ID"].fillna(df[col].apply(safe_json_name))
        df.drop(
            ["synpartsynthesis_syn_part", "oligosynthesis_oligo",
             "plasmidsynthesis_plasmid", "plasmidsynthesis_insert_sequence"],
            axis=1, inplace=True, errors="ignore",
        )

        # ── Root STOCK_ID ─────────────────────────────────────────────────────
        df["root_work_order_id"] = df["root_work_order_id"].fillna("")
        stock_map = (
            df[["workorder_id", "STOCK_ID"]]
            .dropna(subset=["STOCK_ID"])
            .drop_duplicates(subset=["workorder_id"])
        )
        df = df.merge(
            stock_map,
            left_on="root_work_order_id",
            right_on="workorder_id",
            how="left",
            suffixes=("", "_root"),
        )
        df.rename(columns={"STOCK_ID_root": "root_STOCK_ID"}, inplace=True)
        df.drop("workorder_id_root", axis=1, inplace=True, errors="ignore")

        # ── Concentration column aliases ───────────────────────────────────────
        if "qubit_concentration_ngul" in df.columns:
            df["qubit_concentration"] = df["qubit_concentration_ngul"]
        if "nanodrop_concentration_ngul" in df.columns:
            df["nanodrop_concentration"] = df["nanodrop_concentration_ngul"]

        # ── Yield calculation (fill missing + fix stored zeros) ───────────────
        df = ProcessingTransformer._calculate_yields(df)

        # ── Date coercion ─────────────────────────────────────────────────────
        for col in ["wo_created_at", "wo_updated_at", "request_created_at", "deleted_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # ── Source material links ─────────────────────────────────────────────
        df = ProcessingTransformer._generate_source_links(df)

        log.info("Processing complete: %d workorders", len(df))
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _filter_canceled_experiments(df: pd.DataFrame) -> pd.DataFrame:
        log.debug("Filtering experiments with only canceled/empty workorders...")

        def _has_work(row):
            if row.get("data_source") in ["SYNTHETIC_LSP", "LSP"]:
                return True
            status = str(row.get("wo_status", "")).upper()
            if status in {"SUCCEEDED", "FAILED", "RUNNING", "READY", "IN_PROGRESS"}:
                return True
            pnames = row.get("protocol_name")
            if isinstance(pnames, list) and len(pnames) > 0 and pd.notna(pnames[0]):
                return True
            return False

        df["has_work_done"] = df.apply(_has_work, axis=1)

        roots_with_live_children = set(
            df.loc[
                (df["wo_status"] != "CANCELED") & df["has_work_done"],
                "root_work_order_id",
            ]
            .dropna()
            .unique()
        )

        def _displayable(row):
            if row.get("root_work_order_id") in roots_with_live_children:
                return True
            if row.get("request_status") == "NEW":
                return True
            if row.get("wo_status") != "CANCELED":
                return True
            return row.get("has_work_done", False)

        df["is_displayable"] = df.apply(_displayable, axis=1)

        is_lsp = df["data_source"].isin(["SYNTHETIC_LSP", "LSP"])
        root_to_exp = (
            df[df["experiment_name"].notna()]
            .groupby("root_work_order_id")["experiment_name"]
            .first()
            .to_dict()
        )
        df["experiment_name_for_grouping"] = (
            df["experiment_name"]
            .fillna(df["root_work_order_id"].map(root_to_exp))
            .fillna(df["workorder_id"].where(is_lsp))
            .fillna(df["workorder_id"])
        )

        keep_experiments = (
            df.groupby("experiment_name_for_grouping")["is_displayable"]
            .any()
            .pipe(lambda s: s[s].index)
        )
        before = len(df)
        df = df[df["experiment_name_for_grouping"].isin(keep_experiments)]
        removed = before - len(df)
        if removed:
            log.info("Removed %d rows from all-canceled/empty experiments", removed)

        df.drop(
            ["has_work_done", "is_displayable", "experiment_name_for_grouping"],
            axis=1, inplace=True, errors="ignore",
        )
        return df

    @staticmethod
    def _calculate_yields(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing qubit yields and fix stored zeros."""
        conc_col = "qubit_concentration_ngul"
        vol_col  = "total_volume_ul"
        yld_col  = "qubit_yield"

        if not all(c in df.columns for c in [conc_col, vol_col, yld_col]):
            return df

        missing = df[yld_col].isna() & df[conc_col].notna() & df[vol_col].notna()
        df.loc[missing, yld_col] = (df.loc[missing, conc_col] * df.loc[missing, vol_col]) / 1000

        zero = (
            (df[yld_col] == 0.0) &
            df[conc_col].notna() & df[vol_col].notna() &
            (df[vol_col] > 0) & (df[conc_col] > 0)
        )
        df.loc[zero, yld_col] = (df.loc[zero, conc_col] * df.loc[zero, vol_col]) / 1000
        return df

    @staticmethod
    def _generate_source_links(df: pd.DataFrame) -> pd.DataFrame:
        """Build human-readable source material link strings for LSP rows."""
        log.debug("Generating source material links...")
        id_to_name  = df.set_index("workorder_id")["construct_name"].to_dict()
        id_to_stock = df.set_index("workorder_id")["STOCK_ID"].to_dict()

        def _link(row):
            if row["type"] != "lsp_workorder":
                return None
            src = row.get("source_lsp_process_id") or row.get("source_workorder_id")
            if pd.isna(src):
                return None
            if src in id_to_name:
                name = id_to_name[src] or id_to_stock.get(src) or src
                return f"{name} ({src})"
            hits = df[df["workorder_id"] == src]
            if not hits.empty:
                r = hits.iloc[0]
                return r.get("construct_name") or r.get("STOCK_ID") or src
            return f"Source: {src}"

        df["source_material_link"] = df.apply(_link, axis=1)
        return df

"""
dnasc/extractors/lims.py
─────────────────────────
Extracts colony, plate, and well data from LIMS (BigQuery).
Batches large workorder ID lists to avoid BQ query size limits.
"""

from __future__ import annotations
import time

import pandas as pd
from google.cloud import bigquery

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger

log = get_logger(__name__)

_BATCH_SIZE = 5_000


class LIMSExtractor:
    """Extract colony and well data from LIMS."""

    @staticmethod
    def get_colony_data(workorder_ids: list) -> pd.DataFrame:
        if not workorder_ids:
            log.warning("get_colony_data called with empty workorder_ids list")
            return pd.DataFrame()

        t0 = time.time()
        proj = PipelineConfig.PROJECT_ID
        clean_ids = list(set(str(w) for w in workorder_ids))
        client = bigquery.Client(project=proj)
        log.info("Querying LIMS colony data for %d workorders...", len(clean_ids))

        # ── Batched raw pull ──────────────────────────────────────────────────
        raw_dfs: list[pd.DataFrame] = []
        for i in range(0, len(clean_ids), _BATCH_SIZE):
            batch = clean_ids[i : i + _BATCH_SIZE]
            ids_str = "', '".join(batch)
            query = f"""
            SELECT
                COALESCE(d.process_id, g.process_id, a.process_id) AS workorder_id,
                COALESCE(d.colony_number, g.colony_number) AS colony_number,
                a.available,
                COALESCE(d.seq_confirmed, g.seq_confirmed) AS seq_confirmed,
                a.id AS well_id,
                b.id AS plate_id,
                b.protocol AS plate_protocol
            FROM `{proj}.lims__src.well` a
            LEFT JOIN `{proj}.lims__src.plate` b ON a.plate_id = b.id
            LEFT JOIN `{proj}.lims__src.well_content` c ON c.well_id = a.id
            LEFT JOIN `{proj}.lims__src.plasmid_stock` d ON d.id = c.plasmid_stock_id
            LEFT JOIN `{proj}.lims__src.strain` g ON g.id = c.strain_id
            WHERE a.type != 'Empty'
              AND COALESCE(d.process_id, g.process_id, a.process_id) IN ('{ids_str}')
            """
            raw_dfs.append(client.query(query).to_dataframe())

        raw_df = pd.concat(raw_dfs, ignore_index=True)
        if raw_df.empty:
            log.info("No colony data found")
            return pd.DataFrame()

        # ── Pre-compute string columns once ───────────────────────────────────
        raw_df["well_id_str"]      = raw_df["well_id"].astype(str)
        raw_df["colony_num_str"]   = raw_df["colony_number"].fillna(-1).astype(int).astype(str)
        raw_df["well_col_combined"]= raw_df["well_id_str"] + ":" + raw_df["colony_num_str"]

        unique_colonies = raw_df[raw_df["colony_number"].notna()].copy()

        # ── Colony counts ─────────────────────────────────────────────────────
        colony_summary = unique_colonies.groupby("workorder_id").agg(
            total_colonies    =("colony_number", "nunique"),
            available_colonies=("available", lambda x: x[x == True].count()),
            all_colonies      =("well_col_combined", lambda x: ", ".join(x)),
        ).reset_index()

        seq_conf = (
            unique_colonies[unique_colonies["seq_confirmed"] == True]
            .groupby(["workorder_id", "colony_number"])
            .size()
            .reset_index(name="cnt")
        )
        seq_count_map = seq_conf.groupby("workorder_id").size()
        colony_summary["seq_confirmed"] = (
            colony_summary["workorder_id"].map(seq_count_map).fillna(0).astype(int)
        )

        # ── Available list ────────────────────────────────────────────────────
        avail_df = unique_colonies[unique_colonies["available"] == True]
        avail_str = avail_df["well_col_combined"] + "[" + avail_df["plate_protocol"] + "]"
        avail_map = avail_str.groupby(avail_df["workorder_id"]).apply(", ".join)
        colony_summary["available_colonies_list"] = (
            colony_summary["workorder_id"].map(avail_map).fillna("")
        )

        # ── Seq confirmed list ────────────────────────────────────────────────
        seq_df = unique_colonies[unique_colonies["seq_confirmed"] == True]
        if not seq_df.empty:
            seq_str = seq_df["well_col_combined"] + "[" + seq_df["plate_protocol"] + "]"
            seq_map = seq_str.groupby(seq_df["workorder_id"]).apply(", ".join)
            colony_summary["seq_confirmed_colonies"] = (
                colony_summary["workorder_id"].map(seq_map).fillna("")
            )
        else:
            colony_summary["seq_confirmed_colonies"] = ""

        # ── Selected colony ───────────────────────────────────────────────────
        selected_map = (
            avail_df.sort_values("colony_number")
            .groupby("workorder_id")["well_col_combined"]
            .first()
        )
        colony_summary["selected_colony"] = (
            colony_summary["workorder_id"].map(selected_map).fillna("None")
        )

        # ── Plate strings ─────────────────────────────────────────────────────
        raw_df["plate_id_str"] = "Plate" + raw_df["plate_id"].astype(str)
        raw_df["col_label"]    = "col" + raw_df["colony_num_str"]

        plate_info = (
            raw_df.groupby(["workorder_id", "plate_id_str", "plate_protocol"])["col_label"]
            .apply(lambda x: ", ".join(sorted(x.unique())) if x.notna().any() else "")
            .reset_index()
        )
        plate_info["loc_string"] = (
            plate_info["plate_id_str"]
            + " (" + plate_info["plate_protocol"] + "): "
            + plate_info["col_label"]
        )
        loc_map = plate_info.groupby("workorder_id")["loc_string"].apply(lambda x: " | ".join(x))
        colony_summary["all_locations"] = colony_summary["workorder_id"].map(loc_map).fillna("")

        # ── Protocol plates JSON ──────────────────────────────────────────────
        proto_plates = (
            raw_df.groupby(["workorder_id", "plate_protocol"])["plate_id"]
            .apply(lambda x: ",".join(x.unique().astype(str)))
            .reset_index()
        )
        proto_plates["pair"] = (
            '"' + proto_plates["plate_protocol"] + '":"' + proto_plates["plate_id"] + '"'
        )
        json_map = proto_plates.groupby("workorder_id")["pair"].apply(
            lambda x: "{" + ",".join(x) + "}"
        )
        colony_summary["all_protocol_plates"] = (
            colony_summary["workorder_id"].map(json_map).fillna("{}")
        )

        log.info(
            "Colony data retrieved: %d workorders in %.2fs",
            len(colony_summary), time.time() - t0,
        )
        return colony_summary

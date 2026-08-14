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

        # ── Single raw pull via array parameter ───────────────────────────────
        # The COALESCE(...) IN (...) filter depends on JOINED columns, so it's
        # non-sargable — BQ scans + 4-way-joins the full well table before it can
        # filter. Batching the IN-list into 5k chunks therefore re-ran that full
        # scan+join PER batch (7x). One query with IN UNNEST(@ids) does it once:
        # ~10x faster (172s -> 18s), ~10x less slot time, identical rows (the
        # union of the old batches == one query over the full id list).
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
          AND COALESCE(d.process_id, g.process_id, a.process_id) IN UNNEST(@ids)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", clean_ids)]
        )
        raw_df = client.query(query, job_config=job_config).to_dataframe()
        if raw_df.empty:
            log.info("No colony data found")
            return pd.DataFrame()

        # ── Pre-compute string columns once ───────────────────────────────────
        raw_df["well_id_str"]      = raw_df["well_id"].astype(str)
        raw_df["colony_num_str"]   = raw_df["colony_number"].fillna(-1).astype(int).astype(str)
        raw_df["well_col_combined"]= raw_df["well_id_str"] + ":" + raw_df["colony_num_str"]

        # Dedup to one row per (workorder_id, colony_number) — a colony can produce
        # multiple well_content JOIN rows (e.g. plasmid_stock + strain records).
        # Sort seq_confirmed then available descending so True rows win ties.
        unique_colonies = (
            raw_df[raw_df["colony_number"].notna()]
            .sort_values(["seq_confirmed", "available"], ascending=False)
            .drop_duplicates(subset=["workorder_id", "colony_number"], keep="first")
            .copy()
        )

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
            raw_df[raw_df["colony_number"].notna()]
            .groupby(["workorder_id", "plate_id_str", "plate_protocol"])["col_label"]
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

        # ── Plate-only rows (e.g. PCR workorders with no colonies) ────────────
        # json_map is built from all wells in raw_df, not just colony wells.
        # Workorders that appear in raw_df but have no colonies are absent from
        # colony_summary, so all_protocol_plates would be NULL after the merge.
        # Add stub rows for these workorders so plate data flows through.
        plates_only_ids = set(json_map.index) - set(colony_summary["workorder_id"])
        if plates_only_ids:
            plates_only = pd.DataFrame({
                "workorder_id":           list(plates_only_ids),
                "all_protocol_plates":    [json_map[w] for w in plates_only_ids],
            })
            colony_summary = pd.concat([colony_summary, plates_only], ignore_index=True)
            log.info("Added %d plate-only (no-colony) workorder rows", len(plates_only_ids))

        log.info(
            "Colony data retrieved: %d workorders in %.2fs",
            len(colony_summary), time.time() - t0,
        )
        return colony_summary

    @staticmethod
    def get_plasmid_antibiotics(plasmid_ids: list) -> pd.DataFrame:
        """
        Query anti_kan/spec/carb from lims__src.plasmid by numeric ID.
        Returns DataFrame with (plasmid_id, anti_kan, anti_spec, anti_carb).
        Used to detect antibiotic mismatches against BIOS workorder antibiotic field.
        """
        if not plasmid_ids:
            return pd.DataFrame()
        proj = PipelineConfig.PROJECT_ID
        client = bigquery.Client(project=proj)
        clean_ids = list({int(x) for x in plasmid_ids if x is not None})
        if not clean_ids:
            return pd.DataFrame()
        ids_str = ", ".join(str(x) for x in clean_ids)
        query = f"""
        SELECT id AS plasmid_id, anti_kan, anti_spec, anti_carb
        FROM `{proj}.lims__src.plasmid`
        WHERE id IN ({ids_str})
        """
        return client.query(query).to_dataframe()

    @staticmethod
    def get_colony_picking_counts(workorder_ids: list) -> pd.DataFrame:
        """
        Pull QPix colony picking counts from bios__src.colonypickingcounts,
        joined to lims__src.well via well_id → process_id (= workorder UUID).
        Returns one row per workorder with summed imaged/pickable/picked counts.
        """
        if not workorder_ids:
            return pd.DataFrame()

        t0 = time.time()
        proj = PipelineConfig.PROJECT_ID
        clean_ids = list(set(str(w) for w in workorder_ids))
        client = bigquery.Client(project=proj)
        log.info("Querying colony picking counts for %d workorders...", len(clean_ids))

        # Single query via array param (was 5k-batched). Each workorder's wells
        # are all matched within one query, and GROUP BY workorder_id aggregates
        # them fully, so this is identical to the union of the old batches — and
        # scans colonypickingcounts once instead of per batch.
        query = f"""
        WITH per_well AS (
          -- Aggregate colonypickingcounts per (workorder, well) first,
          -- so a workorder with multiple wells gets deterministic plate/position.
          SELECT
            w.process_id AS workorder_id,
            w.plate_id, w.position, p.well_count,
            SUM(cpc.imaged) AS well_imaged,
            -- SUM, deliberately: this mirrors what LIMS holds and what OpTracker and the retry
            -- service compute, so the dashboard cannot silently disagree with them. The sum can
            -- be inflated — a blank Manual Pickable inherits the Manual Picked value, and that
            -- colony is already inside the QPix count, so 3 pickable with 1 leftover picked
            -- reads 4. Correcting it here would hide a discrepancy the lab needs to SEE, so the
            -- count stays as LIMS reports it and is flagged instead (see pickable_suspect).
            SUM(COALESCE(cpc.pickable_automated, 0) + COALESCE(cpc.pickable_manual, 0)) AS well_pickable,
            -- Flags the rows where that sum counts a colony twice: the summed figure exceeds
            -- what any single assessment, or the picked total, can account for. Nothing renders
            -- it at the moment — the '?' marker was pulled until the upstream inheritance is
            -- fixed and we can see what that does to the data. Kept because it costs nothing to
            -- compute and puts the marker one render away instead of one full refresh away.
            MAX(CASE WHEN COALESCE(cpc.pickable_automated,0) + COALESCE(cpc.pickable_manual,0)
                        > GREATEST(COALESCE(cpc.pickable_automated,0),
                                   COALESCE(cpc.pickable_manual,0),
                                   COALESCE(cpc.picked_automated,0) + COALESCE(cpc.picked_manual,0))
                     THEN 1 ELSE 0 END) AS well_pickable_suspect,
            SUM(COALESCE(cpc.picked_automated,   0) + COALESCE(cpc.picked_manual,   0)) AS well_picked
          FROM `{proj}.bios__src.colonypickingcounts` cpc
          JOIN `{proj}.lims__src.well`  w ON w.id = cpc.well_id
          JOIN `{proj}.lims__src.plate` p ON p.id = w.plate_id
          WHERE w.process_id IN UNNEST(@ids)
            AND cpc.deleted_at IS NULL
          GROUP BY w.process_id, w.plate_id, w.position, p.well_count
        )
        SELECT
          workorder_id,
          -- Pick the well with the most imaged colonies as the display well.
          ARRAY_AGG(plate_id    ORDER BY well_imaged DESC, plate_id ASC LIMIT 1)[OFFSET(0)] AS colony_plate_id,
          ARRAY_AGG(position    ORDER BY well_imaged DESC, plate_id ASC LIMIT 1)[OFFSET(0)] AS colony_well_position,
          ARRAY_AGG(well_count  ORDER BY well_imaged DESC, plate_id ASC LIMIT 1)[OFFSET(0)] AS colony_plate_well_count,
          SUM(well_imaged)   AS imaged_colonies,
          SUM(well_pickable) AS pickable_colonies,
          SUM(well_picked)   AS picked_colonies,
          MAX(well_pickable_suspect) AS pickable_suspect
        FROM per_well
        GROUP BY workorder_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", clean_ids)]
        )
        result = client.query(query, job_config=job_config).to_dataframe()
        if result.empty:
            log.info("No colony picking count data found")
            return pd.DataFrame()
        log.info("Colony picking counts: %d workorders in %.2fs", len(result), time.time() - t0)
        return result

    @staticmethod
    def get_repick_plates(workorder_cutoffs: dict) -> pd.DataFrame:
        """
        For GG/Gibson/Transformation workorders that have FAILED NGS, detect
        manually-created LIMS miniprep plates made AFTER the NGS failure timestamp.
        These represent a repick of colonies from the original agar plate.

        workorder_cutoffs: {workorder_id: ngs_fa_timestamp_str}
        Returns DataFrame with (workorder_id, plate_id, plate_created_at).
        """
        if not workorder_cutoffs:
            return pd.DataFrame()

        proj = PipelineConfig.PROJECT_ID
        client = bigquery.Client(project=proj)

        ids_upper = {k.upper(): v for k, v in workorder_cutoffs.items()}
        min_cutoff = min(ids_upper.values())
        ids_str = "', '".join(ids_upper.keys())

        query = f"""
        SELECT DISTINCT
            UPPER(COALESCE(d.process_id, g.process_id, a.process_id)) AS workorder_id,
            b.id AS plate_id,
            b.protocol AS plate_protocol,
            b.created_at AS plate_created_at,
            COALESCE(d.colony_number, g.colony_number) AS colony_number
        FROM `{proj}.lims__src.well` a
        JOIN `{proj}.lims__src.plate` b ON a.plate_id = b.id
        LEFT JOIN `{proj}.lims__src.well_content` c ON c.well_id = a.id
        LEFT JOIN `{proj}.lims__src.plasmid_stock` d ON d.id = c.plasmid_stock_id
        LEFT JOIN `{proj}.lims__src.strain` g ON g.id = c.strain_id
        WHERE UPPER(COALESCE(d.process_id, g.process_id, a.process_id)) IN ('{ids_str}')
          AND b.protocol IN ('Miniprep', 'Bank Overnights', 'Overnight Culture')
          AND b.created_at > '{min_cutoff}'
        """
        df = client.query(query).to_dataframe()
        if df.empty:
            return pd.DataFrame()

        # Filter per-workorder by its specific NGS FA cutoff
        rows = []
        for wo_id, cutoff in ids_upper.items():
            sub = df[df['workorder_id'] == wo_id]
            sub = sub[sub['plate_created_at'] > pd.Timestamp(cutoff, tz='UTC')]
            if not sub.empty:
                rows.append(sub)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    @staticmethod
    def get_well_comments(workorder_ids: list) -> pd.DataFrame:
        """
        Pull non-null well comments from lims__src.well for the given workorder IDs.
        Returns one row per workorder with all unique comments joined by ' | '.
        Used to surface PCR stock plate "at risk" comments in the dashboard.
        """
        if not workorder_ids:
            return pd.DataFrame()

        t0 = time.time()
        proj = PipelineConfig.PROJECT_ID
        clean_ids = list(set(str(w) for w in workorder_ids))
        client = bigquery.Client(project=proj)

        # Single query via array param (was 5k-batched); GROUP BY workorder_id is
        # complete within one query, so identical to the union of old batches.
        query = f"""
        SELECT
            w.process_id AS workorder_id,
            STRING_AGG(DISTINCT w.comments, ' | ' ORDER BY w.comments) AS well_comments
        FROM `{proj}.lims__src.well` w
        WHERE w.process_id IN UNNEST(@ids)
          AND w.comments IS NOT NULL
          AND TRIM(w.comments) != ''
        GROUP BY w.process_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", clean_ids)]
        )
        result = client.query(query, job_config=job_config).to_dataframe()
        if result.empty:
            log.info("No well comments found")
            return pd.DataFrame()
        log.info("Well comments: %d workorders in %.2fs", len(result), time.time() - t0)
        return result

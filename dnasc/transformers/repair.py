"""
dnasc/transformers/repair.py
─────────────────────────────
Two responsibilities:

1. RepairTransformer.create_synthetic_streakouts()
   Creates synthetic workorder rows for offline streakout / transformation
   processes that exist in OpTracker but have no BIOS workorder.

2. RepairTransformer.repair_data()
   Fixes broken root links and backfills metadata gaps using recursive
   well-to-workorder mapping from LIMS.

3. populate_synthetic_optracker_batch()
   Module-level function that attaches OpTracker operation lists to
   synthetic workorder rows by tracing through plate/well relationships.
"""

from __future__ import annotations
import re
from collections import Counter

import pandas as pd
from google.cloud import bigquery

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger

log = get_logger(__name__)

# ── Shared regex ──────────────────────────────────────────────────────────────
_UUID_RE  = re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", re.I)
_WELL_RE  = re.compile(r"well[_\s]*(\d+)", re.I)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: recursive well → workorder mapping
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_well_mapping(client: bigquery.Client, proj: str) -> dict[str, str]:
    query = f"""
    WITH RECURSIVE well_trace AS (
        SELECT w.id AS well_id,
               COALESCE(ps.process_id, s.process_id, w.process_id) AS source_workorder_id,
               0 AS depth
        FROM `{proj}.lims__src.well` w
        LEFT JOIN `{proj}.lims__src.well_content` wc ON wc.well_id = w.id
        LEFT JOIN `{proj}.lims__src.plasmid_stock` ps ON ps.id = wc.plasmid_stock_id
        LEFT JOIN `{proj}.lims__src.strain` s ON s.id = wc.strain_id
        WHERE COALESCE(ps.process_id, s.process_id, w.process_id) IS NOT NULL
        UNION ALL
        SELECT wt.well_id,
               COALESCE(ps2.process_id, s2.process_id, w2.process_id) AS source_workorder_id,
               wt.depth + 1
        FROM well_trace wt
        JOIN `{proj}.lims__src.well` w2
             ON CAST(REGEXP_EXTRACT(wt.source_workorder_id, r'well[_\\s]*(\\d+)') AS INT64) = w2.id
        LEFT JOIN `{proj}.lims__src.well_content` wc2 ON wc2.well_id = w2.id
        LEFT JOIN `{proj}.lims__src.plasmid_stock` ps2 ON ps2.id = wc2.plasmid_stock_id
        LEFT JOIN `{proj}.lims__src.strain` s2 ON s2.id = wc2.strain_id
        WHERE wt.source_workorder_id LIKE '%well%' AND wt.depth < 5
          AND COALESCE(ps2.process_id, s2.process_id, w2.process_id) IS NOT NULL
    )
    SELECT well_id, source_workorder_id, depth FROM well_trace
    WHERE source_workorder_id NOT LIKE '%well%' OR depth = 4
    QUALIFY ROW_NUMBER() OVER (PARTITION BY well_id ORDER BY depth DESC) = 1
    """
    df = client.query(query).to_dataframe()
    return dict(zip(df["well_id"].astype(str), df["source_workorder_id"].astype(str)))


# ─────────────────────────────────────────────────────────────────────────────
class RepairTransformer:
# ─────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def create_synthetic_streakouts(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic workorder rows for OpTracker processes that have no
        corresponding BIOS workorder (e.g. offline streakouts, cell-name UUIDs).
        Must run BEFORE merging so LSP rows can link to them.
        """
        log.info("Creating synthetic streakout workorders...")
        proj   = PipelineConfig.PROJECT_ID
        client = bigquery.Client(project=proj)
        df     = df.copy()

        # ── Build recursive well → workorder map ──────────────────────────────
        try:
            well_id_to_workorder = _fetch_well_mapping(client, proj)
            log.info("Mapped %d wells to workorders", len(well_id_to_workorder))
        except Exception as exc:
            log.error("Could not build recursive well mapping: %s", exc)
            return df

        # ── Collect source IDs that are missing from the dataset ─────────────
        all_source_ids: set[str] = set()
        for col in ["source_lsp_process_id", "source_workorder_id", "source_asm_process_id"]:
            if col not in df.columns:
                continue
            for val in df[col].dropna().astype(str):
                if re.search(r"(pos-ctrl|neg-ctrl|control)", val, re.I):
                    continue
                if val.upper().startswith("LSP-") or val.upper().startswith("ORPHAN_LSP"):
                    continue
                if re.search(r"STREAK|well\d+", val, re.I):
                    all_source_ids.add(val)
                elif _UUID_RE.search(val) and not _UUID_RE.fullmatch(val):
                    all_source_ids.add(val)

        existing_ids = set(df["workorder_id"].astype(str))
        missing      = all_source_ids - existing_ids
        log.info("Found %d missing OpTracker process IDs", len(missing))

        # ── Direct well → process_id lookup for simple well references ────────
        well_direct_links: dict[int, str] = {}
        potential_wells = [
            _WELL_RE.search(x).group(1) for x in missing if _WELL_RE.search(x)
        ]
        if potential_wells:
            try:
                meta_q = f"""
                    SELECT id, process_id FROM `{proj}.lims__src.well`
                    WHERE id IN ({','.join(potential_wells)})
                """
                meta_df = pd.read_gbq(meta_q, project_id=proj, dialect="standard")
                well_direct_links = meta_df.set_index("id")["process_id"].to_dict()
            except Exception as exc:
                log.warning("Metadata fetch for wells failed: %s", exc)

        # ── Build synthetic rows ──────────────────────────────────────────────
        synthetic_rows: list[dict] = []
        skipped = 0

        for process_id in missing:
            if re.search(r"(pos-ctrl|neg-ctrl|control|SCALEUP)", process_id, re.I):
                skipped += 1
                continue

            well_match = _WELL_RE.search(process_id)
            source_row = None

            if not well_match:
                uuid_match = _UUID_RE.search(process_id)
                if not uuid_match:
                    continue
                source_wo   = uuid_match.group(0)
                source_rows = df[df["workorder_id"] == source_wo]
                if source_rows.empty:
                    continue
                source_row = source_rows.iloc[0]
            else:
                well_id = well_match.group(1)
                if well_id not in well_id_to_workorder:
                    direct = well_direct_links.get(int(well_id))
                    if direct and str(direct) in existing_ids:
                        source_wo = str(direct)
                    else:
                        continue
                else:
                    source_wo = well_id_to_workorder[well_id]

                if "well" in source_wo.lower():
                    continue
                source_rows = df[df["workorder_id"] == source_wo]
                if source_rows.empty:
                    continue
                source_row = source_rows.iloc[0]

            # Skip control sources
            if pd.notna(source_row.get("STOCK_ID")):
                if re.search(r"control|ctrl", str(source_row["STOCK_ID"]), re.I):
                    skipped += 1
                    continue

            # Determine synthetic type
            has_uuid = bool(_UUID_RE.search(process_id))
            if has_uuid or "TFM" in process_id.upper() or "SUB-" in process_id.upper():
                proc_type = "transformation_offline_operation"
            elif "STREAK" in process_id.upper():
                proc_type = "streakout_operation"
            else:
                proc_type = "optracker_operation"

            synthetic_rows.append({
                "workorder_id":       process_id,
                "type":               proc_type,
                "wo_status":          "SUCCEEDED",
                "visual_status":      "SUCCEEDED",
                "STOCK_ID":           source_row.get("STOCK_ID"),
                "root_work_order_id": source_row.get("root_work_order_id"),
                "root_STOCK_ID":      source_row.get("root_STOCK_ID"),
                "req_id":             source_row.get("req_id"),
                "experiment_name":    source_row.get("experiment_name"),
                "construct_name":     source_row.get("construct_name"),
                "for_partner":        source_row.get("for_partner"),
                "request_status":     source_row.get("request_status"),
                "priority":           source_row.get("priority"),
                "wo_created_at":      pd.Timestamp.now(tz="UTC"),
                "wo_updated_at":      pd.Timestamp.now(tz="UTC"),
                "source_asm_process_id": source_wo if not well_match else source_wo,
                "is_visible":         True,
                "is_software_fail":   False,
                "status_rank":        3,
                "group_rank":         3,
                "req_rank":           source_row.get("req_rank", 3),
                "data_source":        "SYNTHETIC",
                "fulfills_request":   False,
                "deleted_at":         None,
                "is_fulfillment":     False,
                "cloning_strain":     source_row.get("cloning_strain"),
            })
            log.debug("Created synthetic row: %s → %s", process_id, source_wo)

        if skipped:
            log.info("Skipped %d control workorders", skipped)

        if synthetic_rows:
            log.info("Created %d synthetic rows", len(synthetic_rows))
            df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def repair_data(df: pd.DataFrame) -> pd.DataFrame:
        """Fix broken root links and backfill metadata."""
        log.info("Repairing root links and backfilling metadata...")
        proj   = PipelineConfig.PROJECT_ID
        client = bigquery.Client(project=proj)
        df     = df.copy()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        id_to_root = (
            df.dropna(subset=["root_work_order_id"])
            .set_index("workorder_id")["root_work_order_id"]
            .astype(str)
            .to_dict()
        )
        valid_ids = set(df["workorder_id"].astype(str).unique())

        # ── Well → root from existing location columns ─────────────────────────
        well_to_root: dict[str, str] = {}
        loc_cols = [c for c in ["all_locations", "operation_well_locations"] if c in df.columns]
        if loc_cols:
            search = df[loc_cols].fillna("").astype(str).agg(" ".join, axis=1)
            has_root = df["root_work_order_id"].notna()
            found = search[has_root].str.extractall(r"\b(\d{5,8})\b")[0]
            if not found.empty:
                well_to_root = dict(zip(
                    found.values,
                    df.loc[found.index.get_level_values(0), "root_work_order_id"],
                ))

        # ── Recursive well → workorder from LIMS ──────────────────────────────
        try:
            well_id_to_workorder = _fetch_well_mapping(client, proj)
        except Exception as exc:
            log.warning("Well mapping fetch failed: %s", exc)
            well_id_to_workorder = {}

        # ── Fix transformation workorder roots ────────────────────────────────
        # Root is resolved from the LIMS input-well's process_id (physical
        # lineage), not from the BIOS AssemblyDesign (logical grouping).
        # After setting the root, clear req_id/experiment_name so that
        # _finalize_metadata fills them from the physical-lineage root rather
        # than the BIOS AssemblyDesign root (which may be a different request).
        tfm_mask = df["type"] == "transformation_workorder"
        df.loc[tfm_mask, "root_work_order_id"] = (
            df.loc[tfm_mask, "source_asm_process_id"]
            .map(id_to_root)
            .fillna(df.loc[tfm_mask, "source_asm_process_id"])
        )
        # Cross-request batch transformations: the transformation's own request_id
        # differs from the GG root's request_id (i.e. a batch GG that assembled
        # constructs for multiple requests simultaneously).  These should self-root
        # so they appear in their own request section, not under the batch GG.
        # Same-request transformations (the actual target for that GG) clear
        # req_id/experiment_name as before so _finalize_metadata inherits from root.
        root_req_map = df.set_index("workorder_id")["req_id"].to_dict()
        has_source = tfm_mask & df["source_asm_process_id"].notna()
        # Prefer the resolved root's req_id; if the root isn't in df (e.g. an
        # older pre-cutoff GG that plan_roots picked but wasn't fetched), fall
        # back to the direct source workorder's req_id.  This prevents a NaN
        # root_req from falsely triggering cross-request self-rooting.
        source_req = df.loc[has_source, "source_asm_process_id"].map(root_req_map)
        root_req   = df.loc[has_source, "root_work_order_id"].map(root_req_map).fillna(source_req)
        own_req    = df.loc[has_source, "req_id"]
        cross_req  = own_req.notna() & root_req.notna() & (own_req != root_req)
        # Revert cross-request transformations to self-root (keep their own req_id)
        df.loc[has_source & cross_req, "root_work_order_id"] = (
            df.loc[has_source & cross_req, "workorder_id"]
        )
        # Clear only same-request transformations so _finalize_metadata picks up root metadata
        df.loc[has_source & ~cross_req, "req_id"]          = None
        df.loc[has_source & ~cross_req, "experiment_name"] = None
        id_to_root.update(
            df[tfm_mask].dropna(subset=["root_work_order_id"])
            .set_index("workorder_id")["root_work_order_id"].astype(str).to_dict()
        )

        # ── Fix synthetic/streakout roots ─────────────────────────────────────
        syn_mask = df["type"].isin(["streakout_operation", "transformation_offline_operation"])
        df.loc[syn_mask, "root_work_order_id"] = (
            df.loc[syn_mask, "source_asm_process_id"]
            .map(id_to_root)
            .fillna(df.loc[syn_mask, "source_asm_process_id"])
        )
        id_to_root.update(
            df[syn_mask].dropna(subset=["root_work_order_id"])
            .set_index("workorder_id")["root_work_order_id"].astype(str).to_dict()
        )

        # ── Fix LSP roots that chain through synthetics ────────────────────────
        lsp_mask       = df["type"] == "lsp_workorder"
        intermediate   = set(df.loc[syn_mask | tfm_mask, "workorder_id"].astype(str))
        for src_col in ["source_lsp_process_id", "source_workorder_id", "middle_root"]:
            if src_col not in df.columns:
                continue
            needs_fix = lsp_mask & (
                (df["root_work_order_id"].astype(str) == df["workorder_id"].astype(str)) |
                df["root_work_order_id"].astype(str).isin(intermediate)
            )
            if not needs_fix.any():
                continue
            resolved = df.loc[needs_fix, src_col].astype(str).map(id_to_root)
            valid    = resolved.notna() & (resolved != df.loc[needs_fix, "workorder_id"].astype(str))
            df.loc[valid[valid].index, "root_work_order_id"] = resolved[valid]
        id_to_root.update(
            df[lsp_mask].dropna(subset=["root_work_order_id"])
            .set_index("workorder_id")["root_work_order_id"].astype(str).to_dict()
        )

        # ── Vectorised root resolution for remaining gaps ──────────────────────
        df["_raw_src"] = (
            df["source_lsp_process_id"].fillna("").astype(str) +
            df.get("source_workorder_id", pd.Series("", index=df.index)).fillna("").astype(str) +
            df["source_asm_process_id"].fillna("").astype(str)
        )
        uuids = df["_raw_src"].str.extract(r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", flags=re.I)[0]
        wells = df["_raw_src"].str.extract(r"well[_\s]*(\d+)", flags=re.I)[0]

        res_uuid = uuids.map(id_to_root).fillna(uuids.where(uuids.isin(valid_ids)))
        lims_res = wells.map(well_id_to_workorder).map(id_to_root)
        res_well = wells.map(well_to_root).fillna(lims_res)

        missing_mask = (
            df["root_work_order_id"].isna() |
            df["root_work_order_id"].astype(str).isin(["nan", "None", ""])
        )
        df.loc[missing_mask, "root_work_order_id"] = res_uuid.fillna(res_well)

        # ── Backfill req_id + experiment_name: root row first, sibling fallback ─
        # Root-first priority prevents LSP siblings from contaminating cleared
        # Transformation rows, while still filling legitimate nulls for LSP chains.
        _rroot = df[df["workorder_id"] == df["root_work_order_id"]]
        if "req_id" in df.columns:
            _root_req_map = _rroot.set_index("root_work_order_id")["req_id"].to_dict()
            df["req_id"] = df["req_id"].fillna(df["root_work_order_id"].map(_root_req_map))
            df["req_id"] = df["req_id"].fillna(
                df.groupby("root_work_order_id")["req_id"].transform("first")
            )
        if "experiment_name" in df.columns:
            _root_exp_map = _rroot.set_index("root_work_order_id")["experiment_name"].to_dict()
            df["experiment_name"] = df["experiment_name"].fillna(
                df["root_work_order_id"].map(_root_exp_map)
            )
            # remaining nulls (root itself was null) get sibling fill below

        # ── Backfill metadata ──────────────────────────────────────────────────
        for col in ["experiment_name", "root_STOCK_ID", "for_partner"]:
            if col in df.columns:
                df[col] = df[col].fillna(
                    df.groupby("root_work_order_id")[col].transform("first")
                )

        df.drop(columns=["_raw_src"], inplace=True, errors="ignore")
        log.info("Data repair complete")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Module-level function (called from pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def populate_synthetic_optracker_batch(
    final_df: pd.DataFrame,
    project_id: str = PipelineConfig.PROJECT_ID,
) -> pd.DataFrame:
    """
    Attach OpTracker operation lists to synthetic workorder rows by tracing
    through plate/well relationships in LIMS and OpTracker.

    Approach:
    1. LIMS plates query (with one source_well_id hop) → plate_id → synthetic_id
    2. OpTracker query filtered by known plate_ids via "Plate ID" parameter
    3. Join to assign synthetic_id to each operation

    The source_well_id hop in step 1 captures downstream plates (e.g. Rearray
    destination plates) whose wells don't carry process_id directly but whose
    source wells do.
    """
    log.info("Populating synthetic OpTracker operations...")
    # Also include real LSP rows (data_source="LSP") with no OpTracker data yet.
    # Real LSP rows have no OpTracker process_id match so they never get queue data
    # from the main Step 9 merge; they need the same plate-based tracing.
    _has_queue = final_df["protocol_name"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    syn_mask = (
        final_df["data_source"].isin(["SYNTHETIC", "SYNTHETIC_LSP"]) |
        (
            (final_df["data_source"] == "LSP") &
            final_df["workorder_id"].astype(str).str.upper().str.startswith("LSP-") &
            ~_has_queue
        )
    )
    syn_ids     = final_df[syn_mask]["workorder_id"].dropna().unique().tolist()

    if not syn_ids:
        log.info("No synthetic workorders to process")
        return final_df

    client    = bigquery.Client(project=project_id)
    lsp_ids   = [s for s in syn_ids if s.upper().startswith("LSP-")]
    uuid_ids  = [s for s in syn_ids if not s.upper().startswith("LSP-")]

    # ── QUERY 1: plates ───────────────────────────────────────────────────────
    plates_dfs: list[pd.DataFrame] = []

    if uuid_ids:
        ids_str = ",".join(f"'{s}'" for s in uuid_ids)
        plates_dfs.append(client.query(f"""
            SELECT DISTINCT
                COALESCE(d.process_id, g.process_id, a.process_id) AS synthetic_id,
                b.id AS plate_id, b.protocol AS plate_protocol,
                b.created_at AS plate_created_at
            FROM `{project_id}.lims__src.well` a
            JOIN `{project_id}.lims__src.plate` b ON a.plate_id = b.id
            LEFT JOIN `{project_id}.lims__src.well_content` c ON c.well_id = a.id
            LEFT JOIN `{project_id}.lims__src.plasmid_stock` d ON d.id = c.plasmid_stock_id
            LEFT JOIN `{project_id}.lims__src.strain` g ON g.id = c.strain_id
            WHERE COALESCE(d.process_id, g.process_id, a.process_id) IN ({ids_str})
        """).to_dataframe())

    if lsp_ids:
        upper_map = {s.upper(): s for s in lsp_ids}
        ids_str   = ",".join(f"'{s}'" for s in upper_map)
        lsp_pl    = client.query(f"""
            SELECT DISTINCT
                UPPER(COALESCE(d.process_id, g.process_id, a.process_id)) AS synthetic_id,
                b.id AS plate_id, b.protocol AS plate_protocol,
                b.created_at AS plate_created_at
            FROM `{project_id}.lims__src.well` a
            JOIN `{project_id}.lims__src.plate` b ON a.plate_id = b.id
            LEFT JOIN `{project_id}.lims__src.well_content` c ON c.well_id = a.id
            LEFT JOIN `{project_id}.lims__src.plasmid_stock` d ON d.id = c.plasmid_stock_id
            LEFT JOIN `{project_id}.lims__src.strain` g ON g.id = c.strain_id
            WHERE UPPER(COALESCE(d.process_id, g.process_id, a.process_id)) IN ({ids_str})
        """).to_dataframe()
        lsp_pl["synthetic_id"] = lsp_pl["synthetic_id"].map(upper_map)
        plates_dfs.append(lsp_pl.dropna(subset=["synthetic_id"]))

    if not plates_dfs:
        log.info("No plates found for synthetic workorders")
        return final_df

    plates_df = pd.concat(plates_dfs, ignore_index=True)
    if plates_df.empty:
        return final_df

    log.info(
        "populate_synthetic_optracker_batch: %d plates found for %d synthetic workorders",
        plates_df["plate_id"].nunique(), plates_df["synthetic_id"].nunique(),
    )

    manual_times = (
        plates_df[plates_df["plate_protocol"].isin(["Overnight Culture", "Bank Overnights", "Miniprep"])]
        .groupby("synthetic_id")["plate_created_at"]
        .min()
        .to_dict()
    )

    # ── QUERY 2: operations — three-pass plate+well+job approach ──────────────
    # Pass 1a (plate-level): ops where "Plate ID" IN our known plates.
    #   → finds NGS, Miniprep, and any op whose OpTracker plate IS in LIMS.
    # Pass 1b (well-level): ops where sw/qw/dw references a well on our plates.
    #   → finds Quant, whose "Plate ID" is the Rearray input (not in LIMS set)
    #     but whose "Well to Quant" is a well on the miniprep plate (which is).
    # Pass 2 (job-level): expand to all ops in the same jobs.
    #   → finds Rearray, which runs in the same job as Miniprep/Quant but whose
    #     "Plate ID" is the output plate (not reachable via LIMS alone).
    # Keep ALL plate → synthetic_id pairs. A plate shared by N workorders
    # produces N rows here; all N workorders get credit for the ops on that plate.
    # (The old dict-based approach silently dropped all but the last workorder
    # for any shared plate, which caused most of the 1549 syn_ids to get nothing.)
    plate_syn_pairs = plates_df[["plate_id", "synthetic_id"]].drop_duplicates()
    plate_ids_str   = ",".join(str(p) for p in plates_df["plate_id"].unique())

    # Pass 1a: plate-level
    ops_1a = client.query(f"""
        SELECT DISTINCT
            o.id AS op_id, o.job_id, o.state, o.date_created, o.date_ready,
            p.name AS protocol_name,
            SAFE_CAST(REPLACE(op_param.value, '"', '') AS INT64) AS ref_id
        FROM `{project_id}.op_tracker__src.op_tracker_api_operation` o
        JOIN `{project_id}.op_tracker__src.op_tracker_api_protocol` p ON o.protocol_id = p.id
        JOIN `{project_id}.op_tracker__src.op_tracker_api_parameter` op_param ON o.id = op_param.operation_id
        JOIN `{project_id}.op_tracker__src.op_tracker_api_parametertype` pt ON op_param.parameter_type_id = pt.id
        WHERE pt.name = 'Plate ID'
          AND o.state IN ('SC', 'FA', 'RD', 'RU')
          AND SAFE_CAST(REPLACE(op_param.value, '"', '') AS INT64) IN ({plate_ids_str})
    """).to_dataframe()
    ops_1a = ops_1a.merge(
        plate_syn_pairs.rename(columns={"plate_id": "ref_id"}),
        on="ref_id", how="inner",
    )

    # Pass 1b: well-level — join entirely in BigQuery to avoid huge IN clauses.
    # Finds ops (e.g. Quant) whose "Plate ID" is not in our LIMS set but whose
    # per-well parameter (Source Well / Well to Quant / Destination Well) IS on
    # one of our known plates.
    plate_ids_array = ",".join(str(p) for p in plates_df["plate_id"].unique())
    ops_1b = client.query(f"""
        WITH our_wells AS (
            SELECT w.id AS well_id, w.plate_id
            FROM `{project_id}.lims__src.well` w
            WHERE w.plate_id IN ({plate_ids_array})
        )
        SELECT DISTINCT
            o.id AS op_id, o.job_id, o.state, o.date_created, o.date_ready,
            p.name AS protocol_name,
            ow.plate_id AS ref_id
        FROM `{project_id}.op_tracker__src.op_tracker_api_operation` o
        JOIN `{project_id}.op_tracker__src.op_tracker_api_protocol` p ON o.protocol_id = p.id
        JOIN `{project_id}.op_tracker__src.op_tracker_api_parameter` op_param ON o.id = op_param.operation_id
        JOIN `{project_id}.op_tracker__src.op_tracker_api_parametertype` pt ON op_param.parameter_type_id = pt.id
        JOIN our_wells ow
            ON SAFE_CAST(JSON_EXTRACT_SCALAR(op_param.value, '$.id') AS INT64) = ow.well_id
        WHERE pt.name IN ('Source Well', 'Well to Quant', 'Destination Well')
          AND o.state IN ('SC', 'FA', 'RD', 'RU')
    """).to_dataframe()
    ops_1b = ops_1b.merge(
        plate_syn_pairs.rename(columns={"plate_id": "ref_id"}),
        on="ref_id", how="inner",
    )

    # Combine pass 1a + 1b; dedup on (op_id, synthetic_id) so each pair appears once
    ops_pass1 = (
        pd.concat([ops_1a, ops_1b], ignore_index=True)
        .dropna(subset=["synthetic_id"])
        .drop_duplicates(subset=["op_id", "synthetic_id"])
    )

    # Pass 2: job-level expansion — catches Rearray in the same job as Miniprep/Quant
    job_ids = ops_pass1["job_id"].dropna().unique()
    if len(job_ids):
        # Keep all job → synthetic_id pairs so every workorder gets job-level ops
        job_syn_pairs = ops_pass1[["job_id", "synthetic_id"]].drop_duplicates()
        job_ids_str   = ",".join(str(j) for j in job_ids)
        ops_pass2     = client.query(f"""
            SELECT DISTINCT
                o.id AS op_id, o.job_id, o.state, o.date_created, o.date_ready,
                p.name AS protocol_name,
                SAFE_CAST(REPLACE(op_param.value, '"', '') AS INT64) AS ref_id
            FROM `{project_id}.op_tracker__src.op_tracker_api_operation` o
            JOIN `{project_id}.op_tracker__src.op_tracker_api_protocol` p ON o.protocol_id = p.id
            JOIN `{project_id}.op_tracker__src.op_tracker_api_parameter` op_param ON o.id = op_param.operation_id
            JOIN `{project_id}.op_tracker__src.op_tracker_api_parametertype` pt ON op_param.parameter_type_id = pt.id
            WHERE o.job_id IN ({job_ids_str})
              AND o.state IN ('SC', 'FA', 'RD', 'RU')
              AND pt.name = 'Plate ID'
        """).to_dataframe()
        ops_pass2 = ops_pass2.merge(job_syn_pairs, on="job_id", how="inner")
        # Dedup on (op_id, synthetic_id); Pass 1 attribution wins on conflicts
        ops_df = (
            pd.concat([ops_pass1, ops_pass2], ignore_index=True)
            .drop_duplicates(subset=["op_id", "synthetic_id"])
        )
    else:
        ops_df = ops_pass1.copy()

    ops_df = ops_df.dropna(subset=["synthetic_id"]).copy()

    if ops_df.empty:
        log.info("No OpTracker operations found for synthetic plates")
        return final_df

    # ── Run number lookup (NGS Sequence Confirmation) ─────────────────────────
    op_ids_str = ",".join(str(i) for i in ops_df["op_id"].unique())
    try:
        run_num_df = client.query(f"""
            SELECT op_param.operation_id AS op_id,
                   REPLACE(op_param.value, '"', '') AS ngs_run_number
            FROM `{project_id}.op_tracker__src.op_tracker_api_parameter` op_param
            JOIN `{project_id}.op_tracker__src.op_tracker_api_parametertype` pt
                ON op_param.parameter_type_id = pt.id
            WHERE pt.name = 'Run Number'
              AND op_param.operation_id IN ({op_ids_str})
        """).to_dataframe()
        ops_df = ops_df.merge(run_num_df, on="op_id", how="left")
    except Exception as exc:
        log.warning("populate_synthetic_optracker_batch: run number lookup failed: %s", exc)
        ops_df["ngs_run_number"] = None

    # ── Protocol filtering ────────────────────────────────────────────────────
    lsp_only = {
        "LSP Order", "LSP Receiving", "LSP Reviewing", "LSP Releasing",
        "LSP Aliquoting", "LSP QC", "LSP Processing", "LSP Shipping",
    }
    tfm_only = {"STAR Transformation", "Create Minipreps and Glycerol Stocks"}
    is_tfm     = ops_df["synthetic_id"].str.upper().str.contains("STBL3|EPI400|TRANSFORMATION|TFM|STREAK")
    is_lsp_orp = ops_df["synthetic_id"].str.upper().str.startswith("LSP-")
    ops_df = ops_df[
        ~(is_tfm     & ops_df["protocol_name"].isin(lsp_only)) &
        ~(is_lsp_orp & ops_df["protocol_name"].isin(tfm_only))
    ]
    
    # ── Aggregation ───────────────────────────────────────────────────────────
    # Sort first so drop_duplicates below keeps the earliest op per group.
    ops_df = ops_df.sort_values(["synthetic_id", "date_created"])
    # Collapse same-job duplicate-protocol ops (e.g. two NGS op_ids in Job 7650
    # for two different plates).  Both survive (op_id, synthetic_id) dedup above
    # but represent the same logical sequencing event.  Keeping both causes NGS
    # to appear non-consecutive in the renderer (one may have date_ready=None)
    # so the grouper renders both.  Ops without a job_id are left as-is.
    _has_job = ops_df["job_id"].notna()
    ops_df = pd.concat([
        ops_df[_has_job].drop_duplicates(
            subset=["synthetic_id", "protocol_name", "job_id"]
        ),
        ops_df[~_has_job],
    ], ignore_index=True).sort_values(["synthetic_id", "date_created"])
    agg = ops_df.groupby("synthetic_id").agg(
        protocol_name  =("protocol_name",   list),
        state          =("state",           list),
        date_created   =("date_created",    list),
        date_ready     =("date_ready",      list),
        job_id         =("job_id",          list),
        ngs_run_number =("ngs_run_number",  list),
    ).reset_index()

    # ── Normalize datetime lists to pd.Timestamp(tz=UTC) ─────────────────────
    # BigQuery returns numpy.datetime64 values which PyArrow cannot serialize
    # when mixed with pd.Timestamp or None values from the baseline parquet.
    def _normalize_dt_list(lst: list) -> list:
        result = []
        for v in lst:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                result.append(None)
            elif isinstance(v, pd.Timestamp):
                result.append(v if v.tzinfo else v.tz_localize("UTC"))
            else:
                try:
                    result.append(pd.Timestamp(v, tz="UTC"))
                except Exception:
                    result.append(None)
        return result

    agg["date_created"] = agg["date_created"].map(_normalize_dt_list)
    agg["date_ready"]   = agg["date_ready"].map(_normalize_dt_list)

    # ── Apply updates ─────────────────────────────────────────────────────────
    update_cols = ["protocol_name", "operation_state", "operation_start",
                   "operation_ready", "job_id", "ngs_run_number", "wo_created_at", "wo_updated_at"]
    for col in update_cols:
        if col not in final_df.columns:
            final_df[col] = None

    updates: dict[str, dict] = {k: {} for k in update_cols}

    for _, row in agg.iterrows():
        sid    = row["synthetic_id"]
        m_time = manual_times.get(sid)
        is_lsp = sid.upper().startswith("LSP-")

        if is_lsp and m_time:
            updates["protocol_name"][sid]    = ["Manual: LSP Receiving"]   + row["protocol_name"]
            updates["operation_state"][sid]  = ["SC"]                      + row["state"]
            updates["operation_start"][sid]  = [m_time]                    + row["date_created"]
            updates["operation_ready"][sid]  = [m_time]                    + row["date_ready"]
            updates["job_id"][sid]           = [None]                      + row["job_id"]
            updates["ngs_run_number"][sid]   = [None]                      + row["ngs_run_number"]
        elif is_lsp:
            updates["protocol_name"][sid]    = row["protocol_name"]
            updates["operation_state"][sid]  = row["state"]
            updates["operation_start"][sid]  = row["date_created"]
            updates["operation_ready"][sid]  = row["date_ready"]
            updates["job_id"][sid]           = row["job_id"]
            updates["ngs_run_number"][sid]   = row["ngs_run_number"]
        else:
            mt = m_time or pd.Timestamp.now(tz="UTC")
            updates["protocol_name"][sid]    = ["Manual: Miniprep/Glycerol/Media created"] + row["protocol_name"]
            updates["operation_state"][sid]  = ["SC"]                                       + row["state"]
            updates["operation_start"][sid]  = [mt]                                         + row["date_created"]
            updates["operation_ready"][sid]  = [mt]                                         + row["date_ready"]
            updates["job_id"][sid]           = [None]                                       + row["job_id"]
            updates["ngs_run_number"][sid]   = [None]                                       + row["ngs_run_number"]
            # Backfill wo_created_at with earliest LIMS plate time so TAT is
            # based on when the physical work happened, not pipeline run time.
            updates["wo_created_at"][sid]    = mt
            updates["wo_updated_at"][sid]    = mt

    for col, update_dict in updates.items():
        if update_dict:
            mask = final_df["workorder_id"].isin(update_dict)
            if mask.any():
                final_df.loc[mask, col] = final_df.loc[mask, "workorder_id"].map(
                    update_dict
                )

    log.info("Synthetic OpTracker population complete: %d workorders updated", len(agg))
    return final_df

# ─────────────────────────────────────────────────────────────────────────────
# Module-level function (called from pipeline.py between Steps 9 and 10)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_optracker_streakouts(
    final_df: pd.DataFrame,
    optracker_raw: pd.DataFrame,
    project_id: str = PipelineConfig.PROJECT_ID,
) -> pd.DataFrame:
    """
    Create synthetic workorder rows for OpTracker STREAK/TFM process_ids that
    exist in optracker_raw but have no matching workorder in final_df.

    These are missed by create_synthetic_streakouts() because no BIOS/LSP
    workorder references them as a source — they are only discoverable by
    scanning the OpTracker process_id list directly (e.g. a streakout whose
    input well's process_id in LIMS points straight to the BIOS workorder,
    bypassing the streakout process_id entirely).
    """
    # Match STREAK anywhere in the ID (catches PARTNER_STREAK, etc.)
    # Keep anchored starts for STBL3/EPI400/TFM to avoid false positives.
    STREAK_RE = re.compile(r"STREAK|^(STBL3|EPI400|TFM)", re.I)

    existing_ids = set(final_df["workorder_id"].astype(str).str.lower())
    op_pids = optracker_raw["process_id"].dropna().astype(str).unique()
    missing_streaks = [
        pid for pid in op_pids
        if STREAK_RE.search(pid) and pid.lower() not in existing_ids
    ]

    if not missing_streaks:
        log.info("resolve_optracker_streakouts: no unmatched streakout process_ids")
        return final_df

    log.info("resolve_optracker_streakouts: %d unmatched process_ids to resolve", len(missing_streaks))

    # Only process ones that have a well reference in the name
    well_pids = {
        pid: _WELL_RE.search(pid).group(1)
        for pid in missing_streaks
        if _WELL_RE.search(pid)
    }

    if not well_pids:
        log.info("resolve_optracker_streakouts: none have well references, skipping")
        return final_df

    client = bigquery.Client(project=project_id)
    well_ids_str = ",".join(set(well_pids.values()))

    try:
        well_df = client.query(f"""
            SELECT w.id AS well_id,
                   COALESCE(ps.process_id, s.process_id, w.process_id) AS source_workorder_id
            FROM `{project_id}.lims__src.well` w
            LEFT JOIN `{project_id}.lims__src.well_content` wc ON wc.well_id = w.id
            LEFT JOIN `{project_id}.lims__src.plasmid_stock` ps ON ps.id = wc.plasmid_stock_id
            LEFT JOIN `{project_id}.lims__src.strain` s ON s.id = wc.strain_id
            WHERE w.id IN ({well_ids_str})
              AND COALESCE(ps.process_id, s.process_id, w.process_id) IS NOT NULL
        """).to_dataframe()
    except Exception as exc:
        log.warning("resolve_optracker_streakouts: LIMS well lookup failed: %s", exc)
        return final_df

    if well_df.empty:
        log.info("resolve_optracker_streakouts: no well-to-workorder mappings found")
        return final_df

    well_to_source = dict(
        zip(well_df["well_id"].astype(str), well_df["source_workorder_id"].astype(str))
    )

    synthetic_rows: list[dict] = []
    for pid, well_id in well_pids.items():
        source_wo = well_to_source.get(well_id)
        if not source_wo or source_wo in ("nan", "None", ""):
            continue
        if "well" in source_wo.lower():
            continue
        source_rows = final_df[final_df["workorder_id"] == source_wo]
        if source_rows.empty:
            continue
        source_row = source_rows.iloc[0]

        proc_type = "streakout_operation" if "STREAK" in pid.upper() else "transformation_offline_operation"
        synthetic_rows.append({
            "workorder_id":          pid,
            "type":                  proc_type,
            "wo_status":             "SUCCEEDED",
            "visual_status":         "SUCCEEDED",
            "STOCK_ID":              source_row.get("STOCK_ID"),
            "root_work_order_id":    source_row.get("root_work_order_id"),
            "root_STOCK_ID":         source_row.get("root_STOCK_ID"),
            "req_id":                source_row.get("req_id"),
            "experiment_name":       source_row.get("experiment_name"),
            "construct_name":        source_row.get("construct_name"),
            "for_partner":           source_row.get("for_partner"),
            "request_status":        source_row.get("request_status"),
            "priority":              source_row.get("priority"),
            "wo_created_at":         pd.Timestamp.now(tz="UTC"),
            "wo_updated_at":         pd.Timestamp.now(tz="UTC"),
            "source_asm_process_id": source_wo,
            "is_visible":            True,
            "is_software_fail":      False,
            "status_rank":           3,
            "group_rank":            3,
            "req_rank":              source_row.get("req_rank", 3),
            "data_source":           "SYNTHETIC",
            "fulfills_request":      False,
            "deleted_at":            None,
            "is_fulfillment":        False,
            "cloning_strain":        source_row.get("cloning_strain"),
        })
        log.debug("resolve_optracker_streakouts: %s → %s", pid, source_wo)

    if synthetic_rows:
        log.info("resolve_optracker_streakouts: created %d synthetic rows", len(synthetic_rows))
        if final_df.columns.duplicated().any():
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        final_df = pd.concat([final_df, pd.DataFrame(synthetic_rows)], ignore_index=True)
    else:
        log.info("resolve_optracker_streakouts: source workorders not found in df for any candidate")

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# Module-level function (called from pipeline.py as Step 9c)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_lims_streakouts(
    final_df: pd.DataFrame,
    project_id: str = PipelineConfig.PROJECT_ID,
) -> pd.DataFrame:
    """
    LIMS-based streakout recovery (Step 9c).

    Scans lims__src.well for ALL wells whose process_id matches the STREAK/TFM
    pattern, extracts the source well ID embedded in the name, traces it back
    to a BIOS workorder already in final_df, and creates synthetic rows for
    any streakouts not yet present.

    This catches the common case where:
    - The streak was performed and wells were created in LIMS
    - BUT no OpTracker operation has been logged under the streak process_id
    - AND no downstream BIOS/LSP workorder references the streak as a source

    Complements resolve_optracker_streakouts (which requires OpTracker ops)
    and create_synthetic_streakouts (which requires df source-column references).
    """
    # Match STREAK anywhere in the ID (catches PARTNER_STREAK, etc.)
    STREAK_RE = re.compile(r"STREAK.*well\d+|^(STBL3|EPI400|TFM).*well\d+", re.I)
    existing_ids = set(final_df["workorder_id"].astype(str))
    client = bigquery.Client(project=project_id)

    # Query LIMS for all STREAK-pattern process_ids and trace the source well
    # back to a BIOS workorder via well_content → plasmid_stock / strain.
    try:
        streak_df = client.query(f"""
            SELECT DISTINCT
                w.process_id AS streak_process_id,
                COALESCE(ps.process_id, s.process_id, src_well.process_id)
                    AS source_workorder_id
            FROM `{project_id}.lims__src.well` w
            JOIN `{project_id}.lims__src.well` src_well
                ON src_well.id =
                   SAFE_CAST(REGEXP_EXTRACT(w.process_id, r'well(\\d+)') AS INT64)
            LEFT JOIN `{project_id}.lims__src.well_content` wc
                ON wc.well_id = src_well.id
            LEFT JOIN `{project_id}.lims__src.plasmid_stock` ps
                ON ps.id = wc.plasmid_stock_id
            LEFT JOIN `{project_id}.lims__src.strain` s
                ON s.id = wc.strain_id
            WHERE REGEXP_CONTAINS(w.process_id, r'(?i)(STREAK.*well\\d+|^(STBL3|EPI400|TFM).*well\\d+)')
              AND COALESCE(ps.process_id, s.process_id, src_well.process_id) IS NOT NULL
        """).to_dataframe()
    except Exception as exc:
        log.warning("resolve_lims_streakouts: LIMS query failed: %s", exc)
        return final_df

    if streak_df.empty:
        log.info("resolve_lims_streakouts: no LIMS streakouts found")
        return final_df

    # Keep only streakouts not already in df whose source IS in df
    new_streaks = streak_df[
        ~streak_df["streak_process_id"].isin(existing_ids)
        & streak_df["source_workorder_id"].isin(existing_ids)
        & ~streak_df["source_workorder_id"].str.lower().str.contains("well", na=False)
    ].drop_duplicates(subset=["streak_process_id"])

    if new_streaks.empty:
        log.info("resolve_lims_streakouts: no new LIMS streakouts to add")
        return final_df

    log.info("resolve_lims_streakouts: %d new streakout(s) found in LIMS", len(new_streaks))

    synthetic_rows: list[dict] = []
    for _, row in new_streaks.iterrows():
        pid = row["streak_process_id"]
        source_wo = row["source_workorder_id"]
        source_rows = final_df[final_df["workorder_id"] == source_wo]
        if source_rows.empty:
            continue
        src = source_rows.iloc[0]

        proc_type = (
            "streakout_operation"
            if "STREAK" in pid.upper()
            else "transformation_offline_operation"
        )
        synthetic_rows.append({
            "workorder_id":          pid,
            "type":                  proc_type,
            "wo_status":             "SUCCEEDED",
            "visual_status":         "SUCCEEDED",
            "STOCK_ID":              src.get("STOCK_ID"),
            "root_work_order_id":    src.get("root_work_order_id"),
            "root_STOCK_ID":         src.get("root_STOCK_ID"),
            "req_id":                src.get("req_id"),
            "experiment_name":       src.get("experiment_name"),
            "construct_name":        src.get("construct_name"),
            "for_partner":           src.get("for_partner"),
            "request_status":        src.get("request_status"),
            "priority":              src.get("priority"),
            "wo_created_at":         pd.Timestamp.now(tz="UTC"),
            "wo_updated_at":         pd.Timestamp.now(tz="UTC"),
            "source_asm_process_id": source_wo,
            "is_visible":            True,
            "is_software_fail":      False,
            "status_rank":           3,
            "group_rank":            3,
            "req_rank":              src.get("req_rank", 3),
            "data_source":           "SYNTHETIC",
            "fulfills_request":      False,
            "deleted_at":            None,
            "is_fulfillment":        False,
            "cloning_strain":        src.get("cloning_strain"),
        })
        log.debug("resolve_lims_streakouts: %s → %s", pid, source_wo)

    if synthetic_rows:
        log.info("resolve_lims_streakouts: created %d synthetic rows", len(synthetic_rows))
        if final_df.columns.duplicated().any():
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        final_df = pd.concat([final_df, pd.DataFrame(synthetic_rows)], ignore_index=True)
    else:
        log.info("resolve_lims_streakouts: source workorders not in df for any candidate")

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# Module-level function (called from pipeline.py after populate_synthetic_optracker_batch)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_downstream_plates(
    final_df: pd.DataFrame,
    project_id: str = PipelineConfig.PROJECT_ID,
) -> pd.DataFrame:
    """
    For workorders that completed Miniprep but are missing Rearray/Quant/NGS,
    trace forward via miniprep plate IDs to find the downstream OpTracker jobs.

    Flow:
      1. Find workorders where protocol_name has Miniprep but no Rearray/Quant/NGS
      2. Parse miniprep plate_id from all_protocol_plates JSON
      3. Get all wells on those plates from LIMS
      4. Reverse-lookup OpTracker jobs that touched those wells
      5. Append found operations to existing protocol_name/operation_state/job_id lists
    """
    import json

    # Rearray alone doesn't satisfy downstream — NGS/Quant must be present
    SEQ_PROTOCOLS = {'DNA Quantification', 'NGS Sequence Confirmation'}
    MINIPREP_KEYS = {'Miniprep', 'miniprep', 'Minipreps', 'minipreps'}

    # ── STEP 1: Find workorders missing downstream ops ────────────────────────
    def _needs_downstream(row):
        import json as _json
        x = row.get('protocol_name')
        if hasattr(x, 'tolist'):
            x = x.tolist()
        if not isinstance(x, list) or len(x) == 0:
            return False
        # Gibson/GG: has Miniprep step but no Quant/NGS yet
        if 'Create Minipreps and Glycerol Stocks' in x:
            return not any(p in x for p in SEQ_PROTOCOLS)
        # Transformation: has STAR Transformation + Rearray but no Quant/NGS,
        # AND all_protocol_plates declares an NGS plate (meaning NGS was planned)
        if 'STAR Transformation' in x and 'Rearray 96 to 384' in x:
            if any(p in x for p in SEQ_PROTOCOLS):
                return False
            plates_json = row.get('all_protocol_plates')
            if plates_json:
                try:
                    plates = _json.loads(plates_json) if isinstance(plates_json, str) else {}
                    return 'NGS Sequence Confirmation' in plates
                except Exception:
                    pass
        return False

    missing_mask = final_df.apply(_needs_downstream, axis=1)
    missing_df = final_df[missing_mask].copy()

    if missing_df.empty:
        log.info("resolve_downstream_plates: no workorders missing downstream ops")
        return final_df

    log.info("resolve_downstream_plates: %d workorders missing Rearray/Quant/NGS", len(missing_df))

    # ── STEP 2: Parse miniprep plate IDs from all_protocol_plates ────────────
    def _get_miniprep_plate(val):
        if pd.isna(val) or val in ('{}', '', None):
            return None
        try:
            d = json.loads(val) if isinstance(val, str) else val
            for key in MINIPREP_KEYS:
                if key in d:
                    return str(d[key])
        except Exception:
            pass
        return None

    missing_df['_miniprep_plate_id'] = missing_df['all_protocol_plates'].apply(_get_miniprep_plate)
    missing_df = missing_df[missing_df['_miniprep_plate_id'].notna()]

    if missing_df.empty:
        log.info("resolve_downstream_plates: no miniprep plate IDs found in all_protocol_plates")
        return final_df

    plate_ids = missing_df['_miniprep_plate_id'].unique().tolist()
    # Keep as DataFrame — multiple workorders can share the same miniprep plate
    plate_wid_df = missing_df[['_miniprep_plate_id', 'workorder_id']].drop_duplicates()

    log.info("resolve_downstream_plates: tracing %d miniprep plates", len(plate_ids))

    client = bigquery.Client(project=project_id)
    plate_ids_str = ','.join(plate_ids)

    # ── STEP 3: Get all wells on those miniprep plates ────────────────────────
    well_to_plate = client.query(f"""
        SELECT id AS well_id, plate_id
        FROM `{project_id}.lims__src.well`
        WHERE plate_id IN ({plate_ids_str})
    """).to_dataframe()

    if well_to_plate.empty:
        log.info("resolve_downstream_plates: no wells found for miniprep plates")
        return final_df

    # Build well_id → workorder_id pairs via merge (many-to-many safe)
    well_to_plate['plate_id_str'] = well_to_plate['plate_id'].astype(str)
    well_wid_df = well_to_plate.merge(
        plate_wid_df.rename(columns={'_miniprep_plate_id': 'plate_id_str'}),
        on='plate_id_str', how='inner'
    )[['well_id', 'workorder_id']].drop_duplicates()

    # ── STEP 4: Reverse-lookup OpTracker jobs touching those wells ────────────
    raw_ops = client.query(f"""
    WITH all_ops AS (
        SELECT
            o.id, o.job_id, o.plan_id, o.state, o.date_created, o.date_ready,
            p.name AS protocol_name,
            MAX(CASE WHEN pt.name = 'Source Well'       THEN op_param.value END) AS sw,
            MAX(CASE WHEN pt.name = 'Well to Quant'     THEN op_param.value END) AS qw,
            MAX(CASE WHEN pt.name = 'Destination Well'  THEN op_param.value END) AS dw,
            MAX(CASE WHEN pt.name = 'Plate ID'
                THEN CAST(REPLACE(op_param.value, '"', '') AS INT64) END) AS nps
        FROM `{project_id}.op_tracker__src.op_tracker_api_operation` o
        JOIN `{project_id}.op_tracker__src.op_tracker_api_protocol` p
            ON o.protocol_id = p.id
        JOIN `{project_id}.op_tracker__src.op_tracker_api_parameter` op_param
            ON o.id = op_param.operation_id
        JOIN `{project_id}.op_tracker__src.op_tracker_api_parametertype` pt
            ON op_param.parameter_type_id = pt.id
        WHERE o.state IN ('SC', 'FA', 'RD', 'RU', 'CA')
          AND p.name IN ('Rearray 96 to 384', 'DNA Quantification', 'NGS Sequence Confirmation')
        GROUP BY 1,2,3,4,5,6,7
    )
    SELECT a.*,
        CAST(JSON_EXTRACT_SCALAR(a.sw, '$.id') AS INT64) AS sw_id,
        CAST(JSON_EXTRACT_SCALAR(a.qw, '$.id') AS INT64) AS qw_id,
        CAST(JSON_EXTRACT_SCALAR(a.dw, '$.id') AS INT64) AS dw_id
    FROM all_ops a
    """).to_dataframe()

    if raw_ops.empty:
        log.info("resolve_downstream_plates: no downstream ops found in OpTracker")
        return final_df

    # ── STEP 5: Collect resolved (op, workorder_id) pairs — pass 1 ────────────
    # Each op may resolve to multiple workorders if they share a miniprep plate.
    resolved_parts = []
    for well_col in ['sw_id', 'qw_id', 'dw_id']:
        matched = (
            raw_ops[['id', well_col]].dropna(subset=[well_col])
            .merge(well_wid_df.rename(columns={'well_id': well_col}), on=well_col, how='inner')
            [['id', 'workorder_id']]
        )
        resolved_parts.append(matched)
    # Also resolve via nps (plate_id parameter → miniprep plate)
    nps_resolved = (
        raw_ops[['id', 'nps']].dropna(subset=['nps'])
        .assign(plate_id_str=lambda d: d['nps'].astype(str))
        .merge(plate_wid_df.rename(columns={'_miniprep_plate_id': 'plate_id_str'}),
               on='plate_id_str', how='inner')
        [['id', 'workorder_id']]
    )
    resolved_parts.append(nps_resolved)

    all_resolved_pass1 = (
        pd.concat(resolved_parts, ignore_index=True)
        .drop_duplicates(subset=['id', 'workorder_id'])
    ) if any(len(p) > 0 for p in resolved_parts) else pd.DataFrame(columns=['id', 'workorder_id'])

    # ── STEP 5b: Second-hop via all_protocol_plates (merge-based) ─────────────
    # Build plate→workorder pairs from all_protocol_plates for rearray/quant plates.
    # Dict-based approach silently drops all-but-last workorder per shared plate;
    # use a DataFrame instead so all (plate, workorder) pairs are preserved.
    REARRAY_KEYS = {'Rearray 96 to 384'}
    QUANT_KEYS   = {'DNA Quant', 'DNA Quantification', 'Quant'}

    downstream_plate_rows: list[dict] = []
    for _, mrow in missing_df.iterrows():
        plates_json = mrow.get('all_protocol_plates')
        if not plates_json:
            continue
        try:
            plates = json.loads(plates_json) if isinstance(plates_json, str) else {}
            for key, val in plates.items():
                if key in REARRAY_KEYS or key in QUANT_KEYS:
                    pid = str(val).split(',')[0].strip()
                    if pid and pid not in ('nan', 'None', ''):
                        downstream_plate_rows.append({'plate_id_str': pid, 'workorder_id': mrow['workorder_id']})
        except Exception:
            pass

    all_resolved_pass2 = pd.DataFrame(columns=['id', 'workorder_id'])
    if downstream_plate_rows:
        downstream_plate_wid_df = pd.DataFrame(downstream_plate_rows).drop_duplicates()
        ds_plate_ids_str = ','.join(downstream_plate_wid_df['plate_id_str'].unique().tolist())
        ds_wells_df = client.query(f"""
            SELECT id AS well_id, plate_id
            FROM `{project_id}.lims__src.well`
            WHERE plate_id IN ({ds_plate_ids_str})
        """).to_dataframe()

        if not ds_wells_df.empty:
            ds_wells_df['plate_id_str'] = ds_wells_df['plate_id'].astype(str)
            downstream_well_wid_df = ds_wells_df.merge(
                downstream_plate_wid_df, on='plate_id_str', how='inner'
            )[['well_id', 'workorder_id']].drop_duplicates()

            # Only look at Quant/NGS ops for second-hop
            quant_ngs_ops = raw_ops[raw_ops['protocol_name'].isin(
                ['DNA Quantification', 'NGS Sequence Confirmation']
            )]
            ds_parts = []
            for well_col in ['sw_id', 'qw_id', 'dw_id']:
                matched = (
                    quant_ngs_ops[['id', well_col]].dropna(subset=[well_col])
                    .merge(downstream_well_wid_df.rename(columns={'well_id': well_col}),
                           on=well_col, how='inner')
                    [['id', 'workorder_id']]
                )
                ds_parts.append(matched)
            nps_ds = (
                quant_ngs_ops[['id', 'nps']].dropna(subset=['nps'])
                .assign(plate_id_str=lambda d: d['nps'].astype(str))
                .merge(downstream_plate_wid_df, on='plate_id_str', how='inner')
                [['id', 'workorder_id']]
            )
            ds_parts.append(nps_ds)

            if any(len(p) > 0 for p in ds_parts):
                all_resolved_pass2 = (
                    pd.concat(ds_parts, ignore_index=True)
                    .drop_duplicates(subset=['id', 'workorder_id'])
                )
                log.info(
                    "resolve_downstream_plates: second-hop resolved %d (op, workorder) pairs "
                    "(%d downstream plates)",
                    len(all_resolved_pass2), len(downstream_plate_wid_df['plate_id_str'].unique()),
                )

    # Combine both passes and expand raw_ops (one row per op-workorder pair)
    all_resolved = (
        pd.concat([all_resolved_pass1, all_resolved_pass2], ignore_index=True)
        .drop_duplicates(subset=['id', 'workorder_id'])
    )

    if all_resolved.empty:
        log.info("resolve_downstream_plates: no ops resolved via miniprep wells")
        return final_df

    raw_ops = raw_ops.merge(all_resolved, on='id', how='left')
    raw_ops = raw_ops.rename(columns={'workorder_id': 'resolved_workorder'})

    ops_df = raw_ops.dropna(subset=['resolved_workorder']).copy()

    if ops_df.empty:
        log.info("resolve_downstream_plates: ops found but none mapped to workorders")
        return final_df

    # ── STEP 6: Normalize datetime lists ─────────────────────────────────────
    def _normalize_dt_list(lst):
        result = []
        for v in lst:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                result.append(None)
            elif isinstance(v, pd.Timestamp):
                result.append(v if v.tzinfo else v.tz_localize('UTC'))
            else:
                try:
                    result.append(pd.Timestamp(v, tz='UTC'))
                except Exception:
                    result.append(None)
        return result

    # Dedup: keep one op per (workorder, protocol, state) — well-level lookups
    # produce one row per plate well; collapse to one representative per group.
    ops_df = ops_df.sort_values(['resolved_workorder', 'date_created'])
    ops_df = ops_df.drop_duplicates(subset=['resolved_workorder', 'protocol_name', 'state'])

    agg = ops_df.groupby('resolved_workorder').agg(
        protocol_name=('protocol_name', list),
        state        =('state',         list),
        date_created =('date_created',  list),
        date_ready   =('date_ready',    list),
        job_id       =('job_id',        list),
    ).reset_index()

    agg['date_created'] = agg['date_created'].map(_normalize_dt_list)
    agg['date_ready']   = agg['date_ready'].map(_normalize_dt_list)

    # ── STEP 6b: Enrich ops — plate from job (dw/qw → LIMS plate) and
    #             job from plate (null job_id → plan → op_tracker_api_job) ──────

    # Collect all well IDs we need plate lookups for
    _well_ids_needed: set = set()
    for _col in ['dw_id', 'qw_id']:
        for _v in ops_df[_col].tolist():
            try:
                if pd.notna(_v):
                    _well_ids_needed.add(int(_v))
            except (TypeError, ValueError):
                pass

    _op_well_plate: dict = {}  # well_id → plate_id
    if _well_ids_needed:
        _wids_str = ','.join(str(w) for w in _well_ids_needed)
        _op_wp_df = client.query(f"""
            SELECT id AS well_id, plate_id
            FROM `{project_id}.lims__src.well`
            WHERE id IN ({_wids_str})
        """).to_dataframe()
        _op_well_plate = dict(zip(
            _op_wp_df['well_id'].astype(int),
            _op_wp_df['plate_id'].astype(int)
        ))

    def _get_op_plate(row):
        """Return LIMS plate_id for an op row (dw_id > qw_id > nps)."""
        for _col in ['dw_id', 'qw_id']:
            _v = row.get(_col)
            try:
                if pd.notna(_v):
                    _p = _op_well_plate.get(int(_v))
                    if _p is not None:
                        return _p
            except (TypeError, ValueError):
                pass
        _nps = row.get('nps')
        try:
            if pd.notna(_nps):
                return int(_nps)
        except (TypeError, ValueError):
            pass
        return None

    ops_df['_plate_id'] = ops_df.apply(_get_op_plate, axis=1)

    # Recover null job_ids via plan_id → op_tracker_api_job
    def _is_null(v):
        try:
            return not pd.notna(v)
        except (TypeError, ValueError):
            return True

    _null_plan_ids = []
    for _, _r in ops_df[ops_df['job_id'].apply(_is_null)].iterrows():
        _p = _r.get('plan_id')
        try:
            if pd.notna(_p):
                _null_plan_ids.append(int(_p))
        except (TypeError, ValueError):
            pass
    _plan_to_job: dict = {}
    if _null_plan_ids:
        _plan_ids_str = ','.join(str(int(p)) for p in _null_plan_ids)
        # op_tracker_api_job has no plan_id column — recover via other ops
        # in the same plan that DO have a job_id set.
        _plan_job_df = client.query(f"""
            SELECT plan_id, job_id
            FROM `{project_id}.op_tracker__src.op_tracker_api_operation`
            WHERE plan_id IN ({_plan_ids_str})
              AND job_id IS NOT NULL
            ORDER BY job_id DESC
        """).to_dataframe()
        # Take most recent job per plan
        _plan_to_job = (
            _plan_job_df.drop_duplicates(subset=['plan_id'], keep='first')
            .set_index('plan_id')['job_id']
            .to_dict()
        )
        log.info(
            "resolve_downstream_plates: recovered job_id for %d plans with null job",
            len(_plan_to_job),
        )

    def _enrich_job(row):
        try:
            if pd.notna(row.get('job_id')):
                return row['job_id']
        except (TypeError, ValueError):
            pass
        _plan = row.get('plan_id')
        try:
            if pd.notna(_plan):
                return _plan_to_job.get(int(_plan))
        except (TypeError, ValueError):
            pass
        return None

    ops_df['job_id'] = ops_df.apply(_enrich_job, axis=1)

    # Build (resolved_workorder, protocol_name) → plate_id for all_protocol_plates writeback
    _proto_plate_map: dict = {}  # (workorder_id, protocol_name) → plate_id
    for _, _row in ops_df[ops_df['_plate_id'].notna()].iterrows():
        _key = (_row['resolved_workorder'], _row['protocol_name'])
        if _key not in _proto_plate_map:
            _proto_plate_map[_key] = int(_row['_plate_id'])

    # Re-aggregate job_id after enrichment
    agg = ops_df.groupby('resolved_workorder').agg(
        protocol_name=('protocol_name', list),
        state        =('state',         list),
        date_created =('date_created',  list),
        date_ready   =('date_ready',    list),
        job_id       =('job_id',        list),
    ).reset_index()
    agg['date_created'] = agg['date_created'].map(_normalize_dt_list)
    agg['date_ready']   = agg['date_ready'].map(_normalize_dt_list)

    # ── STEP 7: Append to existing protocol lists ─────────────────────────────
    update_map = agg.set_index('resolved_workorder').to_dict('index')

    def _append_ops(row):
        wid = row['workorder_id']
        if wid not in update_map:
            return row
        new = update_map[wid]
        for src_col, dst_col in [
            ('protocol_name', 'protocol_name'),
            ('state',         'operation_state'),
            ('date_created',  'operation_start'),
            ('date_ready',    'operation_ready'),
            ('job_id',        'job_id'),
        ]:
            existing = row.get(dst_col)
            existing = existing if isinstance(existing, list) else []
            row[dst_col] = existing + new[src_col]

        # Write discovered plate IDs back to all_protocol_plates
        # so the renderer can display them even when LIMS colony data didn't capture them.
        # Map: OpTracker protocol_name → all_protocol_plates key
        _proto_to_lims_key = {
            'Rearray 96 to 384':          'Rearray 96 to 384',
            'DNA Quantification':          'DNA Quant',
            'NGS Sequence Confirmation':   'NGS Sequence Confirmation',
        }
        try:
            _plates_str = row.get('all_protocol_plates') or '{}'
            _plates = json.loads(_plates_str) if isinstance(_plates_str, str) else {}
            _updated = False
            for _proto in new['protocol_name']:
                _lims_key = _proto_to_lims_key.get(_proto)
                if not _lims_key:
                    continue
                if _lims_key in _plates:
                    continue  # already present, don't overwrite
                _plate_id = _proto_plate_map.get((wid, _proto))
                if _plate_id:
                    _plates[_lims_key] = str(_plate_id)
                    _updated = True
            if _updated:
                row['all_protocol_plates'] = json.dumps(_plates)
        except Exception:
            pass

        return row

    final_df = final_df.apply(_append_ops, axis=1)

    log.info(
        "resolve_downstream_plates: appended downstream ops to %d workorders",
        len(update_map)
    )
    return final_df
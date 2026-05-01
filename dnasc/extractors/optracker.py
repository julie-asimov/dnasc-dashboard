"""
dnasc/extractors/optracker.py
──────────────────────────────
Extracts OpTracker operation data from BigQuery, including well locations,
TAT metrics, and protocol success/failure state.
"""

from __future__ import annotations
import os
import time

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger

log = get_logger(__name__)

_EXCLUDED_JOBS_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "excluded_optracker_jobs.csv")


def _load_excluded_jobs() -> tuple[list[int], list[int]]:
    """Return (all_excluded_job_ids, jobs_where_process_ids_should_also_be_excluded)."""
    path = os.path.normpath(_EXCLUDED_JOBS_CSV)
    if not os.path.exists(path):
        return [], []
    try:
        df = pd.read_csv(path, comment="#")
        all_ids = [int(x) for x in df["job_id"].dropna().tolist()]
        pid_jobs = []
        if "exclude_process_ids" in df.columns:
            mask = df["exclude_process_ids"].astype(str).str.lower().isin(["true", "1", "yes"])
            pid_jobs = [int(x) for x in df.loc[mask, "job_id"].dropna().tolist()]
        return all_ids, pid_jobs
    except Exception as e:
        log.warning("Could not load excluded_optracker_jobs.csv: %s", e)
        return [], []



class OpTrackerExtractor:
    """Extract OpTracker operations with timing and state."""

    @staticmethod
    def get_optracker_operations() -> tuple[pd.DataFrame, set[str]]:
        t0 = time.time()
        proj = PipelineConfig.PROJECT_ID
        date_filter = PipelineConfig.get_date_filter()
        log.info("Querying OpTracker operations (since %s)...", date_filter)

        query = f"""
        WITH kicked_back_jobs AS (
          -- Pattern 1: Failed Precheck where user chose "Retry all Operations" (user_input=0).
          -- user_input=1 means "Continue with manual protocol" — PCR still ran, not a kickback.
          SELECT id AS job_id
          FROM `{proj}.op_tracker__src.op_tracker_api_job`
          WHERE REGEXP_CONTAINS(step_groups, r'"tag":\\s*"manual-run"[^}}]*"user_input":\\s*0')
          UNION DISTINCT
          -- Pattern 2: Confirmation step where user actually chose Retry.
          -- user_input=0 means "Continue" was selected (job ran fine, not a kickback).
          -- user_input>=1 means user chose "Retry all operations" — true kickback.
          -- Matching just the label text is too broad: it appears as an unselected option.
          SELECT id AS job_id
          FROM `{proj}.op_tracker__src.op_tracker_api_job`
          WHERE EXISTS (
            SELECT 1 FROM UNNEST(JSON_EXTRACT_ARRAY(step_groups)) AS sg
            WHERE JSON_EXTRACT_SCALAR(sg, '$.name') = 'Confirmation'
          )
          AND REGEXP_CONTAINS(step_groups, r'"tag":\\s*"confirmation"[^}}]*"user_input":\\s*[1-9]')
          UNION DISTINCT
          -- Pattern 4: PCR jobs that never completed (didn't reach Cleanup PCRs)
          -- Exclude jobs that still have active (RU/RD) operations — those are in progress, not kicked back
          SELECT j.id AS job_id
          FROM `{proj}.op_tracker__src.op_tracker_api_job` j
          JOIN `{proj}.op_tracker__src.op_tracker_api_protocol` p ON j.protocol_id = p.id
          WHERE p.name = 'PCR'
            AND NOT EXISTS (
              SELECT 1 FROM UNNEST(JSON_EXTRACT_ARRAY(j.step_groups)) AS sg
              WHERE JSON_EXTRACT_SCALAR(sg, '$.name') = 'Cleanup PCRs'
            )
            AND NOT EXISTS (
              SELECT 1 FROM `{proj}.op_tracker__src.op_tracker_api_operation` o
              WHERE o.job_id = j.id AND o.state IN ('RU', 'RD')
            )
          UNION DISTINCT
          -- Pattern 3: Synthesis Order where all completion-toggles are False (nothing ordered)
          SELECT j.id AS job_id
          FROM `{proj}.op_tracker__src.op_tracker_api_job` j
          JOIN `{proj}.op_tracker__src.op_tracker_api_protocol` p ON j.protocol_id = p.id
          WHERE p.name = 'Synthesis Order'
            AND REGEXP_CONTAINS(j.step_groups, r'"tag":\\s*"completion-toggle"')
            AND NOT REGEXP_CONTAINS(j.step_groups, r'"tag":\\s*"completion-toggle"[^}}]*"user_input":\\s*true')
        ),
        all_operations AS (
          SELECT
            o.id AS operation_id, o.job_id, o.plan_id, o.state,
            o.date_created, o.date_ready,
            p.name AS protocol_name,
            MAX(CASE WHEN pt.name = 'Product'     THEN op_param.value END) AS product,
            MAX(CASE WHEN pt.name = 'Process'     THEN REPLACE(op_param.value, '"', '') END) AS process_id,
            MAX(CASE WHEN pt.name = 'DNA to Assemble'        THEN op_param.value END) AS input_dna_plasmids,
            MAX(CASE WHEN pt.name = 'DNA Stocks to Assemble' THEN op_param.value END) AS input_stock_wells,
            MAX(CASE WHEN pt.name = 'Assembly Well'         THEN op_param.value END) AS output_assembly_well_json,
            MAX(CASE WHEN pt.name = 'Source Well'           THEN op_param.value END) AS source_well_json,
            MAX(CASE WHEN pt.name = 'Destination Well'      THEN op_param.value END) AS destination_well_json,
            MAX(CASE WHEN pt.name = 'Agar Well'             THEN op_param.value END) AS agar_well_json,
            MAX(CASE WHEN pt.name = 'Miniprep Plates'       THEN op_param.value END) AS miniprep_plates_json,
            MAX(CASE WHEN pt.name = 'Glycerol Plate'        THEN op_param.value END) AS glycerol_plate_json,
            MAX(CASE WHEN pt.name = 'Experiment'            THEN REPLACE(op_param.value, '"', '') END) AS experiment,
            MAX(CASE WHEN pt.name = 'Result Status'         THEN REPLACE(op_param.value, '"', '') END) AS result_status,
            MAX(CASE WHEN pt.name = 'LSP Batch'
                THEN COALESCE(JSON_EXTRACT_SCALAR(op_param.value, '$.name'), REPLACE(op_param.value, '"', ''))
                END) AS lsp_batch_id_from_optracker,
            MAX(CASE WHEN pt.name = 'Run Number'      THEN REPLACE(op_param.value, '"', '') END) AS ngs_run_number,
            MAX(CASE WHEN pt.name = 'Template'        THEN op_param.value END) AS pcr_template,
            MAX(CASE WHEN pt.name = 'Template Stock'  THEN op_param.value END) AS pcr_template_stock,
            MAX(CASE WHEN pt.name = 'Forward Primer'  THEN op_param.value END) AS pcr_fwd_primer,
            MAX(CASE WHEN pt.name = 'Fwd Primer Stock' THEN op_param.value END) AS pcr_fwd_primer_stock,
            MAX(CASE WHEN pt.name = 'Reverse Primer'  THEN op_param.value END) AS pcr_rev_primer,
            MAX(CASE WHEN pt.name = 'Rev Primer Stock' THEN op_param.value END) AS pcr_rev_primer_stock
          FROM `{proj}.op_tracker__src.op_tracker_api_operation` o
          JOIN `{proj}.op_tracker__src.op_tracker_api_protocol` p ON o.protocol_id = p.id
          JOIN `{proj}.op_tracker__src.op_tracker_api_parameter` op_param ON o.id = op_param.operation_id
          JOIN `{proj}.op_tracker__src.op_tracker_api_parametertype` pt ON op_param.parameter_type_id = pt.id
          WHERE o.date_created >= '{date_filter}'
            AND (o.job_id IS NULL OR o.job_id NOT IN (SELECT job_id FROM kicked_back_jobs))
          GROUP BY o.id, o.job_id, o.plan_id, o.state, o.date_created, o.date_ready, p.name
        ),
        successful_protocols AS (
          SELECT DISTINCT process_id, protocol_name FROM all_operations WHERE state = 'SC'
        ),
        miniprep_plates AS (
          SELECT a.operation_id,
            STRING_AGG(CONCAT('Plate', CAST(pl_mini.id AS STRING), '-miniprep'), '|') AS miniprep_location
          FROM all_operations a
          CROSS JOIN UNNEST(JSON_EXTRACT_ARRAY(a.miniprep_plates_json)) AS plate_json
          LEFT JOIN `{proj}.lims__src.plate` pl_mini
            ON pl_mini.id = CAST(JSON_EXTRACT_SCALAR(plate_json, '$.id') AS INT64)
          WHERE a.miniprep_plates_json IS NOT NULL
          GROUP BY a.operation_id
        ),
        glycerol_plates AS (
          SELECT a.operation_id,
            STRING_AGG(CONCAT('Plate', CAST(pl_glyc.id AS STRING), '-glycerol'), '|') AS glycerol_location
          FROM all_operations a
          CROSS JOIN UNNEST(JSON_EXTRACT_ARRAY(a.glycerol_plate_json)) AS plate_json
          LEFT JOIN `{proj}.lims__src.plate` pl_glyc
            ON pl_glyc.id = CAST(JSON_EXTRACT_SCALAR(plate_json, '$.id') AS INT64)
          WHERE a.glycerol_plate_json IS NOT NULL
          GROUP BY a.operation_id
        ),
        confirmed_inputs AS (
          -- Zip input_dna_plasmids (stock names) with input_stock_wells (well objects) by index.
          -- Format per entry: "stock_name~well_id~plate_label~position~well_count"
          SELECT
            a.operation_id,
            STRING_AGG(
              CONCAT(
                JSON_EXTRACT_SCALAR(dp_json, '$.name'), '~',
                CAST(w.id AS STRING), '~',
                COALESCE(pl.barcode, CONCAT('Plate', CAST(pl.id AS STRING))), '~',
                CAST(w.position AS STRING), '~',
                CAST(pl.well_count AS STRING)
              ), '|' ORDER BY di
            ) AS confirmed_input_ids
          FROM all_operations a
          CROSS JOIN UNNEST(JSON_EXTRACT_ARRAY(a.input_dna_plasmids))  AS dp_json WITH OFFSET di
          CROSS JOIN UNNEST(JSON_EXTRACT_ARRAY(a.input_stock_wells))   AS sw_json WITH OFFSET si
          JOIN `{proj}.lims__src.well` w
            ON w.id = SAFE_CAST(JSON_VALUE(sw_json, '$.id') AS INT64)
          JOIN `{proj}.lims__src.plate` pl ON pl.id = w.plate_id
          WHERE a.input_dna_plasmids IS NOT NULL
            AND a.input_stock_wells IS NOT NULL
            AND di = si
          GROUP BY a.operation_id
        ),
        pcr_confirmed_inputs AS (
          -- Template (LIMS well via Template Stock) + primers (LIMS well via well_content.oligo_stock_id).
          SELECT operation_id, STRING_AGG(entry, '|' ORDER BY sort_key) AS confirmed_input_ids
          FROM (
            SELECT 0 AS sort_key, a.operation_id,
              CONCAT(
                JSON_EXTRACT_SCALAR(JSON_EXTRACT_ARRAY(a.pcr_template)[SAFE_OFFSET(0)], '$.name'), '~',
                CAST(w.id AS STRING), '~',
                COALESCE(pl.barcode, CONCAT('Plate', CAST(pl.id AS STRING))), '~',
                CAST(w.position AS STRING), '~',
                CAST(pl.well_count AS STRING)
              ) AS entry
            FROM all_operations a
            CROSS JOIN UNNEST(JSON_EXTRACT_ARRAY(a.pcr_template_stock)) AS ts_json
            JOIN `{proj}.lims__src.well` w ON w.id = SAFE_CAST(JSON_VALUE(ts_json, '$.id') AS INT64)
            JOIN `{proj}.lims__src.plate` pl ON pl.id = w.plate_id
            WHERE a.pcr_template IS NOT NULL AND a.pcr_template_stock IS NOT NULL
            UNION ALL
            SELECT 1, a.operation_id,
              CONCAT(
                JSON_EXTRACT_SCALAR(a.pcr_fwd_primer, '$.name'), '~',
                CAST(w.id AS STRING), '~',
                COALESCE(pl.barcode, CONCAT('Plate', CAST(pl.id AS STRING))), '~',
                CAST(w.position AS STRING), '~',
                CAST(pl.well_count AS STRING)
              )
            FROM all_operations a
            JOIN `{proj}.lims__src.well` w
              ON w.id = SAFE_CAST(JSON_VALUE(a.pcr_fwd_primer_stock, '$.id') AS INT64)
            JOIN `{proj}.lims__src.plate` pl ON pl.id = w.plate_id
            WHERE a.pcr_fwd_primer IS NOT NULL AND a.pcr_fwd_primer_stock IS NOT NULL
            UNION ALL
            SELECT 2, a.operation_id,
              CONCAT(
                JSON_EXTRACT_SCALAR(a.pcr_rev_primer, '$.name'), '~',
                CAST(w.id AS STRING), '~',
                COALESCE(pl.barcode, CONCAT('Plate', CAST(pl.id AS STRING))), '~',
                CAST(w.position AS STRING), '~',
                CAST(pl.well_count AS STRING)
              )
            FROM all_operations a
            JOIN `{proj}.lims__src.well` w
              ON w.id = SAFE_CAST(JSON_VALUE(a.pcr_rev_primer_stock, '$.id') AS INT64)
            JOIN `{proj}.lims__src.plate` pl ON pl.id = w.plate_id
            WHERE a.pcr_rev_primer IS NOT NULL AND a.pcr_rev_primer_stock IS NOT NULL
          )
          GROUP BY operation_id
        ),
        well_info AS (
          SELECT a.operation_id, a.protocol_name,
            CASE
              WHEN a.destination_well_json IS NOT NULL THEN
                CONCAT(COALESCE(pl_dest.barcode, CONCAT('Plate', CAST(pl_dest.id AS STRING))), '-',
                  CHR(65 + CAST(FLOOR(w_dest.position / 24) AS INT64)),
                  CAST(MOD(w_dest.position, 24) + 1 AS STRING))
              WHEN a.agar_well_json IS NOT NULL THEN
                CONCAT(COALESCE(pl_agar.barcode, CONCAT('Plate', CAST(pl_agar.id AS STRING))), '-',
                  CHR(65 + CAST(FLOOR(w_agar.position / 12) AS INT64)),
                  CAST(MOD(w_agar.position, 12) + 1 AS STRING))
              WHEN mp.miniprep_location IS NOT NULL OR gp.glycerol_location IS NOT NULL THEN
                CONCAT(
                  COALESCE(mp.miniprep_location, ''),
                  CASE WHEN mp.miniprep_location IS NOT NULL AND gp.glycerol_location IS NOT NULL THEN '|' ELSE '' END,
                  COALESCE(gp.glycerol_location, ''))
              WHEN a.output_assembly_well_json IS NOT NULL THEN
                CONCAT(COALESCE(pl.barcode, CONCAT('Plate', CAST(pl.id AS STRING))), '-',
                  CHR(65 + CAST(FLOOR(w.position / 12) AS INT64)),
                  CAST(MOD(w.position, 12) + 1 AS STRING))
              ELSE NULL
            END AS well_location
          FROM all_operations a
          LEFT JOIN `{proj}.lims__src.well` w
            ON w.id = CAST(JSON_EXTRACT_SCALAR(a.output_assembly_well_json, '$.id') AS INT64)
          LEFT JOIN `{proj}.lims__src.plate` pl ON w.plate_id = pl.id
          LEFT JOIN `{proj}.lims__src.well` w_dest
            ON w_dest.id = CAST(JSON_EXTRACT_SCALAR(a.destination_well_json, '$.id') AS INT64)
          LEFT JOIN `{proj}.lims__src.plate` pl_dest ON w_dest.plate_id = pl_dest.id
          LEFT JOIN `{proj}.lims__src.well` w_agar
            ON w_agar.id = CAST(JSON_EXTRACT_SCALAR(a.agar_well_json, '$.id') AS INT64)
          LEFT JOIN `{proj}.lims__src.plate` pl_agar ON w_agar.plate_id = pl_agar.id
          LEFT JOIN miniprep_plates mp ON mp.operation_id = a.operation_id
          LEFT JOIN glycerol_plates gp ON gp.operation_id = a.operation_id
          WHERE a.output_assembly_well_json IS NOT NULL
             OR a.destination_well_json IS NOT NULL
             OR a.agar_well_json IS NOT NULL
             OR mp.miniprep_location IS NOT NULL
             OR gp.glycerol_location IS NOT NULL
        ),
        operations_with_tat AS (
          SELECT a.*, wi.well_location,
            COALESCE(ci.confirmed_input_ids, pci.confirmed_input_ids) AS confirmed_input_ids,
            sp.process_id IS NOT NULL AS protocol_has_success,
            LAG(a.date_created) OVER (PARTITION BY a.process_id ORDER BY a.date_created) AS prev_operation_end,
            TIMESTAMP_DIFF(a.date_ready, a.date_created, HOUR) AS ready_wait_hours,
            ROW_NUMBER() OVER (PARTITION BY a.process_id, a.protocol_name ORDER BY a.date_created DESC) AS operation_rank
          FROM all_operations a
          LEFT JOIN successful_protocols sp
            ON a.process_id = sp.process_id AND a.protocol_name = sp.protocol_name
          LEFT JOIN well_info wi ON a.operation_id = wi.operation_id
          LEFT JOIN confirmed_inputs ci ON ci.operation_id = a.operation_id
          LEFT JOIN pcr_confirmed_inputs pci ON pci.operation_id = a.operation_id
        )
        SELECT
          process_id, plan_id AS op_batch_id, operation_id, job_id,
          protocol_name, state AS operation_state,
          date_created AS operation_start, date_ready AS operation_ready,
          ready_wait_hours,
          LAG(date_created) OVER (PARTITION BY process_id ORDER BY date_ready) AS prev_operation_completed,
          TIMESTAMP_DIFF(
            date_ready,
            LAG(date_created) OVER (PARTITION BY process_id ORDER BY date_ready),
            HOUR
          ) AS tat_from_prev_hours,
          product, input_dna_plasmids, input_stock_wells,
          output_assembly_well_json, well_location,
          experiment, result_status,
          operation_rank, protocol_has_success,
          lsp_batch_id_from_optracker, ngs_run_number,
          confirmed_input_ids
        FROM operations_with_tat
        WHERE state = 'SC'
           OR (state = 'FA' AND protocol_has_success = FALSE)
           OR state IN ('RD', 'RU')
        ORDER BY process_id, date_created
        """

        df = pd.read_gbq(query, project_id=proj, dialect="standard")

        excluded_job_ids, pid_job_ids = _load_excluded_jobs()

        if pid_job_ids:
            excluded_pids = set(df.loc[df["job_id"].isin(pid_job_ids), "process_id"].dropna())
        else:
            excluded_pids = set()

        if excluded_job_ids:
            before = len(df)
            df = df[~df["job_id"].isin(excluded_job_ids)]
            if before - len(df):
                log.info("Excluded %d rows from job(s): %s", before - len(df), excluded_job_ids)

        if excluded_pids:
            before = len(df)
            df = df[~df["process_id"].isin(excluded_pids)]
            if before - len(df):
                log.info("Excluded %d unbatched rows for job(s) %s", before - len(df), pid_job_ids)

        log.info("OpTracker operations retrieved: %d rows in %.2fs", len(df), time.time() - t0)
        return df, excluded_pids

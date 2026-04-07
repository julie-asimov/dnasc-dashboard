"""
dnasc/extractors/lsp.py
────────────────────────
Extracts LSP batch, aliquot, and workorder data from BigQuery.
"""

from __future__ import annotations
import time

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger

log = get_logger(__name__)


class LSPExtractor:
    """Extract LSP workorders, aliquots, and batch lineage."""

    # ── Workorders ────────────────────────────────────────────────────────────

    @staticmethod
    def get_lsp_workorders() -> pd.DataFrame:
        t0 = time.time()
        proj = PipelineConfig.PROJECT_ID
        log.info("Querying LSP workorders...")

        query = f"""
        WITH lsp_base AS (
            SELECT
                wo.fulfills_request, wo.type, wo.status AS wo_status,
                wo.op_tracker_plan_id AS op_batch_id,
                wo.id AS workorder_id, wo.deleted_at,
                wo.created_at AS wo_created_at, wo.updated_at AS wo_updated_at,
                JSON_VALUE(lsp.plasmid, '$.name') AS STOCK_ID,
                lsp.plasmid AS product_json,
                -- well.process_id is often null for cell bank wells; fall through
                -- to plasmid_stock/strain content to find the source workorder
                COALESCE(ps.process_id, st.process_id, well.process_id) AS source_lsp_process_id,
                'LSP' AS data_source,
                COALESCE(source_wo_ps.id, source_wo_st.id, source_wo.id) AS source_workorder_id,
                lsp.input_lims_record AS lsp_input_well,
                CASE
                    WHEN lsp.lsp_batch_key IS NOT NULL AND lsp.lsp_batch_key != 'null'
                    THEN CONCAT('LSP-', JSON_VALUE(lsp.lsp_batch_key, '$.id'))
                    ELSE NULL
                END AS bios_batch_id,
                wo.request_id AS lsp_own_request_id
            FROM `{proj}.bios__src.workorder` wo
            LEFT JOIN `{proj}.bios__src.lspworkorder` lsp ON wo.id = lsp.id
            LEFT JOIN `{proj}.lims__src.well` AS well
                ON well.id = SAFE_CAST(JSON_VALUE(lsp.input_lims_record, '$.id') AS INT64)
            LEFT JOIN `{proj}.lims__src.well_content` AS wc ON wc.well_id = well.id
            LEFT JOIN `{proj}.lims__src.plasmid_stock` AS ps ON ps.id = wc.plasmid_stock_id
            LEFT JOIN `{proj}.lims__src.strain` AS st ON st.id = wc.strain_id
            LEFT JOIN `{proj}.bios__src.workorder` source_wo
                ON well.process_id = source_wo.id
            LEFT JOIN `{proj}.bios__src.workorder` source_wo_ps
                ON ps.process_id = source_wo_ps.id
            LEFT JOIN `{proj}.bios__src.workorder` source_wo_st
                ON st.process_id = source_wo_st.id
            WHERE wo.type = 'lsp_workorder'
              AND wo.status NOT IN ('DRAFT')
              AND wo.created_at >= '2025-01-01'
        )
        SELECT
            lb.*,
            req.id AS req_id, req.priority, req.submitter_email,
            req.type AS request_type, req.status AS request_status,
            req.created_at AS request_created_at,
            pr.construct_name, pr.delivery_container, pr.delivery_format,
            pr.for_partner, pr.customer,
            exp.name AS experiment_name,
            exp.created_at AS experiment_created_at,
            exp.updated_at AS experiment_updated_at,
            exp.active AS experiment_active,
            ad.root_work_order_id
        FROM lsp_base lb
        LEFT JOIN `{proj}.bios__src.request` req ON lb.lsp_own_request_id = req.id
        LEFT JOIN `{proj}.bios__src.plasmidrequest` pr ON lb.lsp_own_request_id = pr.id
        LEFT JOIN `{proj}.bios__src.experiment` exp ON pr.experiment_id = exp.id
        LEFT JOIN `{proj}.bios__src.workorder` parent_wo ON lb.source_workorder_id = parent_wo.id
        LEFT JOIN `{proj}.bios__src.assemblydesignworkorderassociation` adwoa
            ON adwoa.workorder_id = parent_wo.id
        LEFT JOIN `{proj}.bios__src.assemblydesign` ad ON adwoa.assemblydesign_id = ad.id
        """

        df = pd.read_gbq(query, project_id=proj, dialect="standard")
        log.info("LSP workorders retrieved: %d rows in %.2fs", len(df), time.time() - t0)
        return df

    # ── Aliquots ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_lsp_aliquots() -> pd.DataFrame:
        t0 = time.time()
        proj = PipelineConfig.PROJECT_ID
        date_filter = PipelineConfig.get_date_filter()
        log.info("Querying LSP aliquots (since %s)...", date_filter)

        query = f"""
        SELECT
            COALESCE(aliq.process_id, w.process_id) AS lsp_process_id,
            w.id AS input_well_id,
            CONCAT('LSP-', CAST(batch.id AS STRING)) AS lsp_batch_id,
            COALESCE(
                CONCAT('strain', CAST(source.strain_id AS STRING)),
                CONCAT('ps', CAST(source.plasmid_stock_id AS STRING)),
                CONCAT('LSP', CAST(source.parent_lsp_batch_id AS STRING))
            ) AS source_material,
            CONCAT('pAI-', CAST(batch.plasmid_id AS STRING)) AS plasmid_id,
            batch.nanodrop_concentration_ngul, batch.qubit_concentration_ngul,
            batch.concentration_ngul AS deprecated_concentration_ngul,
            batch.nanodrop_yield, batch.qubit_yield,
            batch.ratio_260_280, batch.ratio_260_230,
            batch.available, batch.created_at AS batch_created_at,
            batch.etoh_precipitation,
            batch.comments AS batch_comments, batch.prep_method, batch.buffer,
            batch._order AS vendor_order_id, batch.deposited_by, batch.qc_status,
            batch.ngs_status, batch.concentration_status, batch.yield_status,
            batch.digest, batch.digest_note,
            MAX(p.cell_strain) AS comp_cell,
            STRING_AGG(CAST(aliq.id AS STRING), ', ')         AS aliquot_ids,
            STRING_AGG(CAST(aliq.volume_ul AS STRING), ', ')  AS volume_ul,
            SUM(aliq.volume_ul)                               AS total_volume_ul,
            STRING_AGG(CAST(aliq.available AS STRING), ', ')  AS aliq_available,
            STRING_AGG(CAST(aliq.created_at AS STRING), ', ') AS aliq_created_at,
            STRING_AGG(CAST(aliq.date_received AS STRING), ', ') AS date_received,
            STRING_AGG(aliq.location, ', ')                   AS location,
            STRING_AGG(aliq.comments, ', ')                   AS aliquot_comments
        FROM `{proj}.lims__src.lsp_batch` AS batch
        LEFT JOIN `{proj}.lims__src.lsp_aliquot` AS aliq ON aliq.lsp_batch_id = batch.id
        LEFT JOIN `{proj}.lims__src.lsp_batch_source` AS source ON source.lsp_batch_id = batch.id
        LEFT JOIN `{proj}.lims__src.strain` AS p ON p.id = source.strain_id
        LEFT JOIN `{proj}.lims__src.well_content` AS wc ON wc.strain_id = p.id
        LEFT JOIN `{proj}.lims__src.well` AS w ON w.id = wc.well_id
        WHERE batch.created_at >= '{date_filter}'
        GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26
        """

        df = pd.read_gbq(query, project_id=proj, dialect="standard")
        log.info("LSP aliquots retrieved: %d rows in %.2fs", len(df), time.time() - t0)
        return df

    # ── Lineage tracing ───────────────────────────────────────────────────────

    @staticmethod
    def trace_lsp_to_workorder() -> dict[str, str]:
        """Return a mapping of lsp_batch_id → source workorder_id."""
        t0 = time.time()
        proj = PipelineConfig.PROJECT_ID
        date_filter = PipelineConfig.get_date_filter()
        log.info("Tracing LSP batches to source workorders...")

        query = f"""
        WITH lsp_sources AS (
            SELECT
                CONCAT('LSP-', CAST(batch.id AS STRING)) AS lsp_batch_id,
                COALESCE(
                    CONCAT('strain', CAST(source.strain_id AS STRING)),
                    CONCAT('ps', CAST(source.plasmid_stock_id AS STRING)),
                    CONCAT('LSP-', CAST(source.parent_lsp_batch_id AS STRING))
                ) AS source_material
            FROM `{proj}.lims__src.lsp_batch` batch
            LEFT JOIN `{proj}.lims__src.lsp_batch_source` source
                ON source.lsp_batch_id = batch.id
            WHERE batch.created_at >= '{date_filter}'
        ),
        material_to_workorder AS (
            SELECT CONCAT('strain', CAST(s.id AS STRING)) AS material_id, s.process_id AS workorder_id
            FROM `{proj}.lims__src.strain` s WHERE s.process_id IS NOT NULL
            UNION ALL
            SELECT CONCAT('ps', CAST(ps.id AS STRING)) AS material_id, ps.process_id AS workorder_id
            FROM `{proj}.lims__src.plasmid_stock` ps WHERE ps.process_id IS NOT NULL
        )
        SELECT lsp.lsp_batch_id, mtw.workorder_id AS source_workorder_id
        FROM lsp_sources lsp
        LEFT JOIN material_to_workorder mtw ON lsp.source_material = mtw.material_id
        WHERE mtw.workorder_id IS NOT NULL
        """

        df = pd.read_gbq(query, project_id=proj, dialect="standard")
        mapping = dict(zip(df["lsp_batch_id"], df["source_workorder_id"]))
        log.info("Mapped %d LSP batches in %.2fs", len(mapping), time.time() - t0)
        return mapping

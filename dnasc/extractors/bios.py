"""
dnasc/extractors/bios.py
─────────────────────────
Extracts BIOS workorder data from BigQuery, including NEW requests
that have no workorders yet (or only DRAFTs).
"""

from __future__ import annotations
import time

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger

log = get_logger(__name__)


class BIOSExtractor:
    """Extract BIOS workorder and request data."""

    @staticmethod
    def get_bios_workorders() -> pd.DataFrame:
        """
        Pull all non-LSP workorders plus NEW requests with no active workorders.
        Returns a combined DataFrame with data_source in {'BIOS', 'BIOS_REQUEST'}.
        """
        t0 = time.time()
        date_filter = PipelineConfig.get_date_filter()
        proj = PipelineConfig.PROJECT_ID
        log.info("Querying BIOS workorders (since %s)...", date_filter)

        query = f"""
        -- 1. Standard Workorders
        SELECT
            wo.fulfills_request, wo.type, wo.status AS wo_status,
            wo.op_tracker_plan_id AS op_batch_id,
            wo.id AS workorder_id,
            tw_well.process_id AS source_asm_process_id,
            wo.deleted_at,
            wo.created_at AS wo_created_at, wo.updated_at AS wo_updated_at,
            ad.root_work_order_id, ad.id AS ad_id, ad.created_at AS ad_created_at,
            req.id AS req_id, req.priority, req.submitter_email,
            req.type AS request_type, req.status AS request_status,
            req.created_at AS request_created_at,
            pr.construct_name, pr.delivery_container, pr.delivery_format,
            pr.for_partner,
            exp.name AS experiment_name,
            exp.created_at AS experiment_created_at,
            exp.updated_at AS experiment_updated_at,
            exp.active AS experiment_active,
            JSON_VALUE(COALESCE(
                ggw.product, gw.product, uw.product, pcrw.product,
                tw.product, psw.plasmid, ssw.syn_part, osw.oligo
            ), '$.name') AS STOCK_ID,
            osw.oligo AS oligosynthesis_oligo,
            pcrw.forward_primer AS pcr_forward_primer,
            pcrw.reverse_primer AS pcr_reverse_primer,
            pcrw.templates AS pcr_templates,
            psw.plasmid AS plasmidsynthesis_plasmid,
            psw.insert_sequence AS plasmidsynthesis_insert_sequence,
            psw.vector AS plasmidsynthesis_vector,
            ssw.syn_part AS synpartsynthesis_syn_part,
            COALESCE(ssw.vendor, psw.vendor, osw.vendor) AS vendor,
            uw.description AS untracked_description,
            COALESCE(ggw.product, gw.product, uw.product, pcrw.product, tw.product) AS product_json,
            COALESCE(ggw.backbone, gw.backbone) AS backbone_json,
            COALESCE(ggw.parts, gw.parts) AS parts_json,
            COALESCE(tw.cloning_strain, gw.cloning_strain, ggw.cloning_strain) AS cloning_strain,
            COALESCE(tw.antibiotic, gw.antibiotic, ggw.antibiotic) AS antibiotic,
            COALESCE(tw.expected_color, ggw.expected_color, gw.expected_color) AS expected_color,
            COALESCE(tw.background_color, ggw.background_color, gw.background_color) AS background_color,
            'BIOS' AS data_source
        FROM `{proj}.bios__src.workorder` wo
        LEFT JOIN `{proj}.bios__src.assemblydesignworkorderassociation` adwoa ON adwoa.workorder_id = wo.id
        LEFT JOIN `{proj}.bios__src.assemblydesign` ad ON adwoa.assemblydesign_id = ad.id
        LEFT JOIN `{proj}.bios__src.gibsonworkorder` gw ON wo.id = gw.id
        LEFT JOIN `{proj}.bios__src.goldengateworkorder` ggw ON wo.id = ggw.id
        LEFT JOIN `{proj}.bios__src.oligosynthesisworkorder` osw ON wo.id = osw.id
        LEFT JOIN `{proj}.bios__src.pcrworkorder` pcrw ON wo.id = pcrw.id
        LEFT JOIN `{proj}.bios__src.plasmidsynthesisworkorder` psw ON wo.id = psw.id
        LEFT JOIN `{proj}.bios__src.synpartsynthesisworkorder` ssw ON wo.id = ssw.id
        LEFT JOIN `{proj}.bios__src.untrackedworkorder` uw ON wo.id = uw.id
        LEFT JOIN `{proj}.bios__src.transformationworkorder` tw ON wo.id = tw.id
        LEFT JOIN `{proj}.lims__src.well` AS tw_well
            ON tw_well.id = CAST(JSON_VALUE(tw.input_well_key, '$.id') AS INT64)
        LEFT JOIN `{proj}.bios__src.workorder` root_wo
            ON root_wo.id = COALESCE(ad.root_work_order_id, wo.id)
        LEFT JOIN `{proj}.bios__src.request` req ON root_wo.request_id = req.id
        LEFT JOIN `{proj}.bios__src.plasmidrequest` pr ON root_wo.request_id = pr.id
        LEFT JOIN `{proj}.bios__src.experiment` exp ON pr.experiment_id = exp.id
        WHERE wo.status NOT IN ('DRAFT')
          AND wo.type != 'lsp_workorder'
          AND wo.created_at >= '{date_filter}'

        UNION ALL

        -- 2. NEW requests (no active workorders yet)
        SELECT
            FALSE AS fulfills_request, 'request_placeholder' AS type,
            'NEW' AS wo_status, NULL AS op_batch_id,
            CONCAT('REQ-', req.id) AS workorder_id,
            NULL AS source_asm_process_id, NULL AS deleted_at,
            NULL AS wo_created_at, NULL AS wo_updated_at, NULL AS root_work_order_id,
            NULL AS ad_id, NULL AS ad_created_at,
            req.id AS req_id, req.priority, req.submitter_email,
            req.type AS request_type, req.status AS request_status,
            req.created_at AS request_created_at,
            pr.construct_name, pr.delivery_container, pr.delivery_format,
            pr.for_partner,
            exp.name AS experiment_name,
            exp.created_at AS experiment_created_at,
            exp.updated_at AS experiment_updated_at,
            exp.active AS experiment_active,
            NULL AS STOCK_ID,
            NULL AS oligosynthesis_oligo, NULL AS pcr_forward_primer,
            NULL AS pcr_reverse_primer, NULL AS pcr_templates,
            NULL AS plasmidsynthesis_plasmid, NULL AS plasmidsynthesis_insert_sequence,
            NULL AS plasmidsynthesis_vector, NULL AS synpartsynthesis_syn_part,
            NULL AS vendor, NULL AS untracked_description,
            NULL AS product_json, NULL AS backbone_json, NULL AS parts_json,
            NULL AS cloning_strain, NULL AS antibiotic,
            NULL AS expected_color, NULL AS background_color,
            'BIOS_REQUEST' AS data_source
        FROM `{proj}.bios__src.request` req
        LEFT JOIN `{proj}.bios__src.plasmidrequest` pr ON req.id = pr.id
        LEFT JOIN `{proj}.bios__src.experiment` exp ON pr.experiment_id = exp.id
        WHERE req.created_at >= '{date_filter}'
          AND NOT EXISTS (
              SELECT 1 FROM `{proj}.bios__src.workorder` w
              WHERE w.request_id = req.id AND w.status != 'DRAFT'
          )
        """

        df = pd.read_gbq(query, project_id=proj, dialect="standard")
        log.info("BIOS workorders retrieved: %d rows in %.2fs", len(df), time.time() - t0)
        return df

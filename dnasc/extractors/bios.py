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
        -- ad_roots: for each workorder, find the authoritative fulfills_request=TRUE
        -- GG/Gibson root via assemblydesignworkorderassociation (explicit BIOS linkage).
        -- Used for parts/transformations to locate their parent GG/Gibson.
        -- fulfills_request=TRUE GG/Gibson workorders use plan_attempt_roots (CASE in SELECT).
        WITH ad_roots AS (
            -- For each workorder, find the canonical root GG/Gibson in its assembly
            -- design. Priority order:
            --   1. assemblydesign.root_work_order_id match — BIOS's own declared root
            --      for the AD (most authoritative; prevents cross-request fan-in where
            --      a newer GG added to the same AD steals parts from the original owner)
            --   2. RUNNING/READY > WAITING/BLOCKED > FAILED/CANCELED (status activity)
            --   3. Oldest created_at (original context wins among same-status ties)
            SELECT
                adwoa.workorder_id,
                ARRAY_AGG(
                    root_wo.id
                    ORDER BY IF(root_wo.id = ad.root_work_order_id, 0, 1),
                             IF(root_wo.status IN ('RUNNING','READY'), 0,
                                IF(root_wo.status IN ('FAILED','CANCELED'), 2, 1)),
                             root_wo.created_at ASC
                    LIMIT 1
                )[OFFSET(0)] AS root_wo_id,
                ARRAY_AGG(
                    root_wo.request_id
                    ORDER BY IF(root_wo.id = ad.root_work_order_id, 0, 1),
                             IF(root_wo.status IN ('RUNNING','READY'), 0,
                                IF(root_wo.status IN ('FAILED','CANCELED'), 2, 1)),
                             root_wo.created_at ASC
                    LIMIT 1
                )[OFFSET(0)] AS root_request_id
            FROM `{proj}.bios__src.assemblydesignworkorderassociation` adwoa
            JOIN `{proj}.bios__src.assemblydesign` ad
                ON ad.id = adwoa.assemblydesign_id
            JOIN `{proj}.bios__src.assemblydesignworkorderassociation` adwoa2
                ON adwoa2.assemblydesign_id = adwoa.assemblydesign_id
            JOIN `{proj}.bios__src.workorder` root_wo
                ON root_wo.id = adwoa2.workorder_id
               AND root_wo.fulfills_request = TRUE
               AND root_wo.type IN ('gibson_workorder', 'golden_gate_workorder')
               AND root_wo.status NOT IN ('DRAFT')
            GROUP BY adwoa.workorder_id
        ),

        consuming_roots AS (
            -- For each workorder, find the fulfills_request=TRUE GG/Gibson workorder IDs
            -- that share an assembly design with it (via ADWOA). These are the assemblies
            -- that consume this workorder as an input (parts, PCR, oligos, etc.).
            -- Stored as a comma-separated string for pandas compatibility.
            SELECT
                adwoa1.workorder_id AS part_workorder_id,
                STRING_AGG(DISTINCT gg_wo.id ORDER BY gg_wo.id) AS consuming_root_ids
            FROM `{proj}.bios__src.assemblydesignworkorderassociation` adwoa1
            JOIN `{proj}.bios__src.assemblydesignworkorderassociation` adwoa2
                ON adwoa1.assemblydesign_id = adwoa2.assemblydesign_id
               AND adwoa1.workorder_id != adwoa2.workorder_id
            JOIN `{proj}.bios__src.workorder` gg_wo
                ON adwoa2.workorder_id = gg_wo.id
               AND gg_wo.fulfills_request = TRUE
               AND gg_wo.type IN ('gibson_workorder', 'golden_gate_workorder')
               AND gg_wo.deleted_at IS NULL
               AND gg_wo.status NOT IN ('DRAFT')
            GROUP BY adwoa1.workorder_id
        )

        -- 1. Standard Workorders
        SELECT
            wo.fulfills_request, wo.type, wo.status AS wo_status,
            wo.op_tracker_plan_id AS op_batch_id,
            wo.id AS workorder_id,
            wo.resubmit_count,
            tw_well.process_id AS source_asm_process_id,
            wo.deleted_at,
            wo.assembly_plan_id,
            wo.created_at AS wo_created_at, wo.updated_at AS wo_updated_at,
            -- GG/Gibson roots always self-root; attempt grouping is expressed via
            -- attempt_anchor_id (computed by the attempt_anchors CTE above) rather than
            -- collapsing root_work_order_id. All other workorders use the ADWOA root.
            CASE
                WHEN wo.fulfills_request = TRUE
                     AND wo.type IN ('gibson_workorder', 'golden_gate_workorder')
                THEN wo.id
                ELSE COALESCE(ad_root.root_wo_id, wo.id)
            END AS root_work_order_id,
            req.id AS req_id, req.priority, req.submitter_email,
            req.type AS request_type, req.status AS request_status,
            req.created_at AS request_created_at,
            pr.construct_name, pr.delivery_container, pr.delivery_format,
            pr.for_partner, pr.customer,
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
            cr.consuming_root_ids,
            NULL AS attempt_anchor_id,
            NULL AS attempt_number,
            NULL AS attempt_total,
            'BIOS' AS data_source
        FROM `{proj}.bios__src.workorder` wo
        LEFT JOIN ad_roots ad_root ON ad_root.workorder_id = wo.id
        LEFT JOIN consuming_roots cr ON cr.part_workorder_id = wo.id
        LEFT JOIN `{proj}.bios__src.assemblyplan` ap ON ap.id = wo.assembly_plan_id
        LEFT JOIN `{proj}.bios__src.experiment` exp ON exp.id = ap.experiment_id
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
        LEFT JOIN `{proj}.bios__src.request` req
            ON req.id = COALESCE(wo.request_id, ad_root.root_request_id)
        LEFT JOIN `{proj}.bios__src.plasmidrequest` pr
            ON pr.id = COALESCE(wo.request_id, ad_root.root_request_id)
        WHERE wo.status NOT IN ('DRAFT')
          AND wo.type != 'lsp_workorder'
          AND wo.created_at >= '{date_filter}'

        UNION ALL

        -- 2. DRAFT root workorders whose assembly plan also has active (non-DRAFT)
        -- workorders. These provide context for parts ordered from a draft plan
        -- (e.g. parts ordered in advance before submitting the draft to production).
        -- DRAFT GGs are always self-rooted (root = their own id) so they render as
        -- their own separate section rather than mixing into the RUNNING GG's section.
        SELECT
            wo.fulfills_request, wo.type, wo.status AS wo_status,
            wo.op_tracker_plan_id AS op_batch_id,
            wo.id AS workorder_id,
            wo.resubmit_count,
            tw_well.process_id AS source_asm_process_id,
            wo.deleted_at,
            wo.assembly_plan_id,
            wo.created_at AS wo_created_at, wo.updated_at AS wo_updated_at,
            wo.id AS root_work_order_id,
            req.id AS req_id, req.priority, req.submitter_email,
            req.type AS request_type, req.status AS request_status,
            req.created_at AS request_created_at,
            pr.construct_name, pr.delivery_container, pr.delivery_format,
            pr.for_partner, pr.customer,
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
            NULL AS consuming_root_ids,
            NULL AS attempt_anchor_id, NULL AS attempt_number, NULL AS attempt_total,
            'BIOS_DRAFT' AS data_source
        FROM `{proj}.bios__src.workorder` wo
        LEFT JOIN `{proj}.bios__src.assemblyplan` ap ON ap.id = wo.assembly_plan_id
        LEFT JOIN `{proj}.bios__src.experiment` exp ON exp.id = ap.experiment_id
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
        LEFT JOIN `{proj}.bios__src.request` req ON req.id = wo.request_id
        LEFT JOIN `{proj}.bios__src.plasmidrequest` pr ON pr.id = wo.request_id
        WHERE wo.status = 'DRAFT'
          AND wo.type IN ('golden_gate_workorder', 'gibson_workorder')
          AND wo.created_at >= '{date_filter}'
          AND EXISTS (
            -- Only surface a DRAFT when its assembly plan has non-DRAFT parts workorders
            -- (syn, oligo, PCR, plasmid synthesis) already created — i.e. someone ordered
            -- inputs from this draft plan before submitting the GG to production.
            SELECT 1 FROM `{proj}.bios__src.workorder` wo3
            WHERE wo3.assembly_plan_id = wo.assembly_plan_id
              AND wo3.status NOT IN ('DRAFT')
              AND wo3.type NOT IN ('golden_gate_workorder', 'gibson_workorder',
                                   'transformation_workorder', 'lsp_workorder')
              AND wo3.deleted_at IS NULL
          )
          AND (wo.request_id IS NULL
            OR NOT EXISTS (
              -- Hide DRAFT when a non-DRAFT/CANCELED/FAILED GG/Gibson already exists
              -- for the same request — production work has started.
              SELECT 1 FROM `{proj}.bios__src.workorder` wo2
              WHERE wo2.request_id = wo.request_id
                AND wo2.status NOT IN ('DRAFT', 'CANCELED', 'FAILED')
                AND wo2.type IN ('golden_gate_workorder', 'gibson_workorder')
                AND wo2.id != wo.id
            ))

        UNION ALL

        -- 3. NEW requests (no active workorders yet)
        SELECT
            FALSE AS fulfills_request, 'request_placeholder' AS type,
            'NEW' AS wo_status, NULL AS op_batch_id,
            CONCAT('REQ-', req.id) AS workorder_id,
            NULL AS resubmit_count,
            NULL AS source_asm_process_id, NULL AS deleted_at,
            NULL AS assembly_plan_id,
            NULL AS wo_created_at, NULL AS wo_updated_at, NULL AS root_work_order_id,
            req.id AS req_id, req.priority, req.submitter_email,
            req.type AS request_type, req.status AS request_status,
            req.created_at AS request_created_at,
            pr.construct_name, pr.delivery_container, pr.delivery_format,
            pr.for_partner, pr.customer,
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
            NULL AS consuming_root_ids,
            NULL AS attempt_anchor_id, NULL AS attempt_number, NULL AS attempt_total,
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

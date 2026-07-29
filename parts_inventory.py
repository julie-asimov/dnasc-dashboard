#!/usr/bin/env python3
"""
Parts Inventory Tool

Queries active BIOS workorders and LIMS inventory and produces three copy-paste
lab action queues:
  1. Mark Available  — seq-confirmed, not-yet-available Echo wells > 25 µL
                       (whole inventory) to flip to available.
  2. Clean Inventory — available Echo wells <= 25 µL to mark unavailable.
  3. Refill          — per part: glycerol plate/well to streak from, plus a
                       ready-to-paste PCR-workorder CSV block for dParts.

Each part needed by a WAITING/READY/RUNNING GG or Gibson workorder is checked
against Echo source-plate stock. Control plasmids and dParts also get a fixed
96-reaction buffer (30-reaction refill trigger) independent of workorder demand.

Intended to become a tab in dna_sc_dashboard.html.

Usage:
    /opt/anaconda3/bin/python3 parts_inventory.py
    /opt/anaconda3/bin/python3 parts_inventory.py --output parts.csv --html queues.html
"""
from __future__ import annotations  # 3.9 server: lazy annotations so `X | None` unions parse

import argparse
import datetime as dt
import json
import math
import sys

import numpy as np
import pandas as pd
from google.cloud import bigquery

from dnasc.utils import parse_parts, parse_backbone, extract_pcr_info, safe_json_name

PROJECT = "data-platform-core-prd"

WELLS_96 = {
    "1": "A1", "2": "B1", "3": "C1", "4": "D1", "5": "E1", "6": "F1", "7": "G1", "8": "H1",
    "9": "A2", "10": "B2", "11": "C2", "12": "D2", "13": "E2", "14": "F2", "15": "G2", "16": "H2",
    "17": "A3", "18": "B3", "19": "C3", "20": "D3", "21": "E3", "22": "F3", "23": "G3", "24": "H3",
    "25": "A4", "26": "B4", "27": "C4", "28": "D4", "29": "E4", "30": "F4", "31": "G4", "32": "H4",
    "33": "A5", "34": "B5", "35": "C5", "36": "D5", "37": "E5", "38": "F5", "39": "G5", "40": "H5",
    "41": "A6", "42": "B6", "43": "C6", "44": "D6", "45": "E6", "46": "F6", "47": "G6", "48": "H6",
    "49": "A7", "50": "B7", "51": "C7", "52": "D7", "53": "E7", "54": "F7", "55": "G7", "56": "H7",
    "57": "A8", "58": "B8", "59": "C8", "60": "D8", "61": "E8", "62": "F8", "63": "G8", "64": "H8",
    "65": "A9", "66": "B9", "67": "C9", "68": "D9", "69": "E9", "70": "F9", "71": "G9", "72": "H9",
    "73": "A10", "74": "B10", "75": "C10", "76": "D10", "77": "E10", "78": "F10", "79": "G10", "80": "H10",
    "81": "A11", "82": "B11", "83": "C11", "84": "D11", "85": "E11", "86": "F11", "87": "G11", "88": "H11",
    "89": "A12", "90": "B12", "91": "C12", "92": "D12", "93": "E12", "94": "F12", "95": "G12", "96": "H12",
}


def _echo384(df):
    """Mask of REAL 384 Echo source wells: the '384 Echo Source Plate' labware AND
    a physically 384-well plate. Excludes mislabeled plates (e.g. a 96-well plate
    carrying that labware — a LIMS data error) from all Echo-source availability /
    mark-available logic, so they never drive a suggested action. Those plates are
    surfaced separately as "error plates" in the Parts tab renderer."""
    return ((df["LABWARE"] == "384 Echo Source Plate")
            & (pd.to_numeric(df["PLATE_NUMBER_OF_WELLS"], errors="coerce") == 384))


# ---------------------------------------------------------------------------
# Control parts: fixed 96-reaction buffer (4 plates/wk × 2 controls × 12 wks).
# Refill is triggered when Echo stock drops below CONTROL_REFILL_TRIGGER (30 rxns).
# Template/oligo demand is propagated when a control dPart needs restocking.
# ---------------------------------------------------------------------------
CONTROL_BUFFER_RXNS  = 96   # target inventory (3-month supply)
CONTROL_REFILL_TRIGGER = 30  # low-water mark — ~2 weeks of runway
OLIGO_FRESHNESS_DAYS = 730  # oligos are stable for years at −20°C; only treat as stale past ~2yr

CONTROL_PARTS = [
    # Gibson — Kan
    "d3550",   # shared Kan + Spec
    "d3551",   # shared Kan + Spec
    "d3391",
    "d3236",
    "d4674",   # backbone
    # Gibson — Spec
    "d3464",
    # Gibson — Carb
    "d4266",
    "d4268",
    "d4269",
    # Golden Gate — Carb (used for all GG types)
    "d4642",
    "pAI-456",
]


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

def _query_all_plate_data() -> str:
    return """
SELECT
  well.id                                                            AS WELL_ID,
  well.plate_id                                                      AS PLATE_ID,
  plate.type                                                         AS PLATE_TYPE,
  well.type                                                          AS WELL_TYPE,
  plate.labware                                                      AS LABWARE,
  plate.protocol                                                     AS PLATE_PROTOCOL,
  COALESCE(
    CONCAT('pAI-', plasmid_stock.plasmid_id),
    CONCAT('pAI-', strain_plasmid.plasmid_id),
    CONCAT('pAI-', well_content.plasmid_id),
    CONCAT('syn',  well_content.syn_part_id),
    CONCAT('d',    dpart_stock.dpart_id),
    CONCAT('o',    oligo_stock.oligo_id),
    -- SynPart 384-rearray stock: the Rearray 96→384 stock well itself carries no content id;
    -- its synpart identity comes from the DOWNSTREAM DNA-Quant well (which has syn_part_id)
    -- pointing back to it via well_source.parent_well_id. Resolve that lineage here.
    CONCAT('syn',  syn_child.syn_part_id)
  )                                                                  AS STOCK_ID,
  COALESCE(plasmid_stock.colony_number, strain.colony_number)        AS COLONY,
  well.available                                                     AS AVAILABLE,
  COALESCE(well.comments, strain.comments, plasmid_stock.comments)  AS COMMENTS,
  COALESCE(plasmid_stock.seq_confirmed, strain.seq_confirmed)        AS SEQ_CONFIRMED,
  well.volume_ul                                                     AS VOLUME_UL,
  well.concentration_ngul                                            AS CONCENTRATION_NGUL,
  LENGTH(plasmid.sequence)                                           AS SEQUENCE_LENGTH,
  dpart_template.name                                                AS DPART_TEMPLATE,
  LENGTH(dpart.sequence)                                             AS DPART_SEQUENCE_LENGTH,
  CONCAT('o', dpart.oligo_1_id)                                     AS DPART_OLIGO1,
  CONCAT('o', dpart.oligo_2_id)                                     AS DPART_OLIGO2,
  LENGTH(oligo.sequence)                                             AS OLIGO_SEQUENCE_LENGTH,
  oligo_stock.molarity * 1000                                        AS MOLARITY_NM,  -- µM→nM: source is µM (≈100), the rxns formula expects nM
  well.position                                                      AS WELL_NUMBER,
  plate.well_count                                                   AS PLATE_NUMBER_OF_WELLS,
  COALESCE(well.process_id, plasmid_stock.process_id,
           strain.process_id, strain_j.process_id)                  AS PROCESS_ID,
  COALESCE(plasmid_stock.updated_at, strain.updated_at)             AS UPDATED_AT,
  COALESCE(plasmid.anti_kan,
           Strain_Plasmid_Plasmid.anti_kan, plasmid_k.anti_kan)     AS ANTI_KAN,
  COALESCE(plasmid.anti_spec, Strain_Plasmid_Plasmid.anti_spec,
           strain.anti_spec)                                         AS ANTI_SPEC,
  COALESCE(plasmid.anti_carb, Strain_Plasmid_Plasmid.anti_carb,
           strain.anti_carb)                                         AS ANTI_CARB,
  plate.location                                                     AS PLATE_LOCATION_BOX,
  plate.comments                                                     AS PLATE_COMMENTS,
  plate.barcode                                                      AS PLATE_BARCODE,
  well.created_at                                                    AS CREATED_AT,
  strain_p.cell_strain                                               AS COMP_CELL
FROM lims__src.well well
LEFT JOIN lims__src.plate               plate             ON plate.id              = well.plate_id
LEFT JOIN lims__src.well_content        well_content      ON well_content.well_id  = well.id
LEFT JOIN lims__src.plasmid_stock       plasmid_stock     ON plasmid_stock.id      = well_content.plasmid_stock_id
LEFT JOIN lims__src.dpart_stock         dpart_stock       ON dpart_stock.id        = well_content.dpart_stock_id
LEFT JOIN lims__src.dpart               dpart             ON dpart.id              = dpart_stock.dpart_id
LEFT JOIN lims__src.plasmid             dpart_template    ON dpart_template.id     = dpart.plasmid_source_id
LEFT JOIN lims__src.well_source         well_source       ON well_source.well_id   = well.id
-- Reverse lineage: a child DNA-Quant well (carrying syn_part_id) points at THIS well as its
-- parent → this well is the synpart's 384-rearray stock. One synpart_id per parent well.
LEFT JOIN (
  SELECT ws_syn.parent_well_id AS parent_well_id, MAX(wc_syn.syn_part_id) AS syn_part_id
  FROM lims__src.well_source ws_syn
  JOIN lims__src.well_content wc_syn ON wc_syn.well_id = ws_syn.well_id
  WHERE wc_syn.syn_part_id IS NOT NULL
  GROUP BY ws_syn.parent_well_id
) syn_child ON syn_child.parent_well_id = well.id
LEFT JOIN lims__src.strain_plasmid      strain_plasmid    ON strain_plasmid.strain_id = well_content.strain_id
LEFT JOIN lims__src.strain              strain            ON strain.id             = well_content.strain_id
LEFT JOIN lims__src.plasmid             plasmid           ON plasmid.id            = plasmid_stock.plasmid_id
LEFT JOIN lims__src.strain_plasmid      strain_plasmid_i  ON strain_plasmid_i.plasmid_id = plasmid.id
LEFT JOIN lims__src.strain              strain_j          ON strain_j.id           = strain_plasmid_i.strain_id
LEFT JOIN lims__src.plasmid             plasmid_k         ON plasmid_k.id          = well_content.plasmid_id
LEFT JOIN lims__src.strain              strain_p          ON strain_p.id           = well_content.strain_id
LEFT JOIN lims__src.plasmid             Strain_Plasmid_Plasmid ON Strain_Plasmid_Plasmid.id = strain_plasmid.plasmid_id
LEFT JOIN lims__src.oligo_stock         oligo_stock       ON oligo_stock.id        = well_content.oligo_stock_id
LEFT JOIN lims__src.oligo               oligo             ON oligo.id              = oligo_stock.oligo_id
WHERE well.type != 'Empty'
-- Collapse the metadata-join fan-out (strain_plasmid → strain → plasmid chains multiply each
-- well ~4.8×) to ONE row per well IN SQL, instead of downloading ~5.5M rows and deduping in
-- pandas. Deterministic pick (prefer content-bearing rows) also kills the old run-to-run
-- nondeterminism where an arbitrary fanned row won. Cuts the transfer ~4-5×.
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY well.id
  ORDER BY strain_plasmid.plasmid_id NULLS LAST,
           strain_plasmid_i.strain_id NULLS LAST,
           well_content.id NULLS LAST
) = 1
ORDER BY well.id ASC
"""


def _query_workorder_data() -> str:
    return """
SELECT
  COALESCE(
    JSON_VALUE(GGwo.product,       '$.name'),
    JSON_VALUE(GIBwo.product,      '$.name'),
    JSON_VALUE(LSPwo.plasmid,      '$.name'),
    JSON_VALUE(PCRwo.product,      '$.name'),
    JSON_VALUE(OligoSynthwo.oligo, '$.name'),
    JSON_VALUE(PlasmidSynthwo.plasmid,  '$.name'),
    JSON_VALUE(SynPartSynthwo.syn_part, '$.name')
  )                                           AS STOCK_ID,
  COALESCE(GGwo.parts,  GIBwo.parts)         AS parts_json,
  COALESCE(GGwo.backbone, GIBwo.backbone)    AS backbone_json,
  PCRwo.templates                            AS pcr_templates,
  PCRwo.forward_primer                       AS pcr_forward_primer,
  PCRwo.reverse_primer                       AS pcr_reverse_primer,
  wo.type                                    AS WORKORDER_TYPE,
  wo.status                                  AS STATUS
FROM bios__src.workorder wo
LEFT JOIN bios__src.goldengateworkorder      GGwo           ON GGwo.id           = wo.id
LEFT JOIN bios__src.gibsonworkorder          GIBwo          ON GIBwo.id          = wo.id
LEFT JOIN bios__src.pcrworkorder             PCRwo          ON PCRwo.id          = wo.id
LEFT JOIN bios__src.lspworkorder             LSPwo          ON LSPwo.id          = wo.id
LEFT JOIN bios__src.oligosynthesisworkorder  OligoSynthwo   ON OligoSynthwo.id   = wo.id
LEFT JOIN bios__src.plasmidsynthesisworkorder PlasmidSynthwo ON PlasmidSynthwo.id = wo.id
LEFT JOIN bios__src.synpartsynthesisworkorder SynPartSynthwo ON SynPartSynthwo.id = wo.id
WHERE wo.status IN ('RUNNING', 'WAITING', 'READY', 'BLOCKED')
  AND wo.type IN (
    'pcr_workorder', 'lsp_workorder',
    'golden_gate_workorder', 'gibson_workorder',
    'oligo_synthesis_workorder',
    'plasmid_synthesis_workorder', 'syn_part_synthesis_workorder'
  )
"""


def _query_dparts() -> str:
    return """
SELECT
  CONCAT('d', dpart.id)             AS DPART_NAME,
  dpart_template.name               AS DPART_TEMPLATE,
  LENGTH(dpart.sequence)            AS DPART_SEQUENCE_LENGTH,
  CONCAT('o', dpart.oligo_1_id)     AS OLIGO_1,
  LENGTH(oligo1.sequence)           AS OLIGO_1_SEQUENCE_LENGTH,
  CONCAT('o', dpart.oligo_2_id)     AS OLIGO_2,
  LENGTH(oligo2.sequence)           AS OLIGO_2_SEQUENCE_LENGTH
FROM lims__src.dpart dpart
LEFT JOIN lims__src.plasmid dpart_template ON dpart_template.id = dpart.plasmid_source_id
LEFT JOIN lims__src.oligo   oligo1         ON oligo1.id         = dpart.oligo_1_id
LEFT JOIN lims__src.oligo   oligo2         ON oligo2.id         = dpart.oligo_2_id
ORDER BY dpart.id ASC
"""


def _query_oligo_stocks() -> str:
    """
    Standalone query for all oligo stocks from LIMS.
    Concentration is stored as molarity in nM (nanomolar) in lims__src.oligo_stock.molarity.
    Oligos are tube-stored (not Echo plates); reaction capacity differs from plasmid/dpart inventory.
    """
    return """
SELECT
  CONCAT('o', oligo.id)       AS STOCK_ID,
  oligo_stock.id               AS OLIGO_STOCK_ID,
  oligo.name                   AS OLIGO_NAME,
  oligo_stock.available        AS AVAILABLE,
  oligo_stock.volume_ul        AS VOLUME_UL,
  oligo_stock.molarity         AS MOLARITY_NM,
  oligo_stock.scale            AS SCALE,
  LENGTH(oligo.sequence)       AS SEQUENCE_LENGTH,
  oligo_stock.created_at       AS CREATED_AT,
  oligo_stock.updated_at       AS UPDATED_AT
FROM lims__src.oligo_stock oligo_stock
JOIN lims__src.oligo oligo ON oligo.id = oligo_stock.oligo_id
ORDER BY oligo_stock.created_at DESC
"""


def _query_lsp_echo_plates() -> str:
    """
    384 Echo Source plates linked to an LSP workorder — either a rearray from the LSP
    workflow or any plate whose well process_id maps to an lsp_workorder. process_id can
    be a bare workorder UUID or a wrapped form (e.g. 'Stbl3_<uuid>'), so the embedded UUID
    is extracted before joining. These plates are slated for disposal; their wells must
    stay unavailable.
    """
    return r"""
WITH lsp_wo AS (
  SELECT id FROM bios__src.workorder WHERE type = 'lsp_workorder'
),
echo_proc AS (
  -- Resolve each well's process_id via the well-content chain (plasmid_stock /
  -- strain) then the well itself, matching how the rest of the pipeline resolves it.
  SELECT plate.id AS PLATE_ID, plate.location AS LOCATION, plate.protocol AS PROTOCOL,
         plate.barcode AS BARCODE, plate.available AS AVAILABLE, plate.created_at AS CREATED_AT,
         COALESCE(ps.process_id, s.process_id, well.process_id) AS proc
  FROM lims__src.well well
  JOIN lims__src.plate plate ON plate.id = well.plate_id
  LEFT JOIN lims__src.well_content wc ON wc.well_id = well.id
  LEFT JOIN lims__src.plasmid_stock ps ON ps.id = wc.plasmid_stock_id
  LEFT JOIN lims__src.strain s ON s.id = wc.strain_id
  WHERE plate.labware = '384 Echo Source Plate'
    AND COALESCE(ps.process_id, s.process_id, well.process_id) IS NOT NULL
)
SELECT DISTINCT ep.PLATE_ID, ep.LOCATION, ep.PROTOCOL, ep.BARCODE, ep.AVAILABLE, ep.CREATED_AT
FROM echo_proc ep
WHERE (
        -- legacy LSP-#### batches (incl. wrapped forms: scaleup_LSP-…, SUB_NEB_LSP-…, PARTNER_CHECK_…LSP-…)
        UPPER(ep.proc) LIKE '%LSP%'
        -- LSP now runs through OpTracker: the process_id is (or wraps) the lsp_workorder UUID
        OR REGEXP_EXTRACT(ep.proc,
             r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}') IN (SELECT id FROM lsp_wo)
      )
  -- The '384 Echo Source Plate' labware is overloaded in LIMS — DNA Quant / NGS Sequence
  -- Confirmation / Sequence Plasmid plates inherit it too. Keep the real source plates
  -- (Rearray, Miniprep) by excluding those operation protocols.
  AND NOT ( UPPER(COALESCE(ep.PROTOCOL,'')) LIKE '%NGS%'
         OR UPPER(COALESCE(ep.PROTOCOL,'')) LIKE '%QUANT%'
         OR UPPER(COALESCE(ep.PROTOCOL,'')) LIKE '%SEQUENCE PLASMID%' )
ORDER BY ep.CREATED_AT DESC
"""


def _query_partner_closeout_products() -> str:
    """
    Products safe to retire when their INACTIVE PARTNER project closes: made by an inactive
    (active=FALSE) partner experiment (>=1 plasmidrequest.for_partner=TRUE), and NOT used —
    as a product OR as an input part/backbone — by any still-ACTIVE experiment. Returns
    (eid, ename, product). Map products → available Stock/Glycerol wells downstream.
    """
    prod = ("COALESCE(JSON_VALUE(gg.product,'$.name'),JSON_VALUE(gib.product,'$.name'),"
            "JSON_VALUE(pcr.product,'$.name'),JSON_VALUE(psy.plasmid,'$.name'),"
            "JSON_VALUE(ssy.syn_part,'$.name'),JSON_VALUE(lsp.plasmid,'$.name'))")
    pjoins = """
 LEFT JOIN bios__src.goldengateworkorder gg ON gg.id=w.id
 LEFT JOIN bios__src.gibsonworkorder gib ON gib.id=w.id
 LEFT JOIN bios__src.pcrworkorder pcr ON pcr.id=w.id
 LEFT JOIN bios__src.plasmidsynthesisworkorder psy ON psy.id=w.id
 LEFT JOIN bios__src.synpartsynthesisworkorder ssy ON ssy.id=w.id
 LEFT JOIN bios__src.lspworkorder lsp ON lsp.id=w.id"""
    return f"""
WITH exp_flag AS (
  SELECT e.id, e.active, ANY_VALUE(e.name) AS ename,
         LOGICAL_OR(COALESCE(pr.for_partner,FALSE)) AS is_partner
  FROM bios__src.experiment e
  LEFT JOIN bios__src.plasmidrequest pr ON pr.experiment_id=e.id
  WHERE e.deleted_at IS NULL GROUP BY e.id, e.active
),
wo AS (
  SELECT COALESCE(ap.experiment_id, pr2.experiment_id) AS eid, {prod} AS product,
    COALESCE(gg.parts, gib.parts) AS parts, COALESCE(gg.backbone, gib.backbone) AS backbone
  FROM bios__src.workorder w {pjoins}
  LEFT JOIN bios__src.assemblyplan ap ON ap.id=w.assembly_plan_id
  LEFT JOIN bios__src.plasmidrequest pr2 ON pr2.id=w.request_id
  WHERE w.deleted_at IS NULL
),
usage AS (
  SELECT eid, product AS name, 'product' AS role FROM wo WHERE product IS NOT NULL
  UNION ALL SELECT eid, JSON_VALUE(p,'$.name'), 'input' FROM wo, UNNEST(JSON_EXTRACT_ARRAY(parts)) p WHERE JSON_VALUE(p,'$.name') IS NOT NULL
  UNION ALL SELECT eid, JSON_VALUE(backbone,'$.name'), 'input' FROM wo WHERE JSON_VALUE(backbone,'$.name') IS NOT NULL
),
uf AS (SELECT u.name, u.role, ef.is_partner, ef.active, ef.id AS eid, ef.ename FROM usage u JOIN exp_flag ef ON ef.id=u.eid),
nonpartner_used AS (SELECT DISTINCT name FROM uf WHERE is_partner=FALSE AND name IS NOT NULL),
active_used     AS (SELECT DISTINCT name FROM uf WHERE active=TRUE AND name IS NOT NULL)
SELECT DISTINCT ef.id AS eid, ef.ename, uf.name AS product
FROM uf JOIN exp_flag ef ON ef.id=uf.eid
WHERE uf.role='product' AND ef.is_partner=TRUE AND ef.active=FALSE AND uf.name IS NOT NULL
  -- PARTNER-SPECIFIC only: never used (product or input) by any non-partner (R&D) project,
  -- so shared R&D/library parts (control dParts etc.) are never retired ...
  AND uf.name NOT IN (SELECT name FROM nonpartner_used)
  -- ... and not still needed by any active partner project.
  AND uf.name NOT IN (SELECT name FROM active_used)
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(client: bigquery.Client) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("  Loading plate inventory ...", flush=True)
    all_plate_data = client.query(_query_all_plate_data()).to_dataframe()

    # The metadata join chain (strain_plasmid → strain → plasmid …) fans out, so
    # each physical well appears multiple times (~4.8× on average). Every inventory
    # count/sum is per-well, so this duplication inflates them. Collapse to one row
    # per well — verified that duplicate rows are identical on all inventory columns.
    if "WELL_ID" in all_plate_data.columns:
        _before = len(all_plate_data)
        all_plate_data = all_plate_data.drop_duplicates(subset=["WELL_ID"]).reset_index(drop=True)
        print(f"  Deduped plate rows: {_before:,} → {len(all_plate_data):,} unique wells", flush=True)

    print("  Loading workorders ...", flush=True)
    workorder_data = client.query(_query_workorder_data()).to_dataframe()

    print("  Loading dpart metadata ...", flush=True)
    dpart_data = client.query(_query_dparts()).to_dataframe()

    # Normalize boolean-like columns so comparisons always work as strings
    for col in ("AVAILABLE", "SEQ_CONFIRMED", "ANTI_KAN", "ANTI_SPEC", "ANTI_CARB"):
        if col in all_plate_data.columns:
            all_plate_data[col] = all_plate_data[col].astype(str)
    if "MOLARITY_NM" in all_plate_data.columns:
        all_plate_data["MOLARITY_NM"] = pd.to_numeric(all_plate_data["MOLARITY_NM"], errors="coerce")

    # Downcast low-cardinality string columns to `category`. all_plate_data is ~1.1M
    # rows; as plain object columns it needs ~1.1 GB in RAM, and the dashboard render
    # loads the WHOLE frame (then .copy()s it in parts.py) just to emit a ~0.2 MB
    # fragment — that transient spike OOM-killed the render on the 8 GB server. Category
    # encoding cuts the frame ~63% (1,132 MB → 419 MB), verified byte-identical render
    # output. We add "" as a category on each so the render's `.fillna("")` calls stay
    # legal; NaNs are preserved (isna/notna semantics unchanged). Numeric / id / barcode
    # / free-text / timestamp columns are left as-is (high cardinality or numeric).
    _CAT_SKIP = {"WELL_ID", "PLATE_ID", "PLATE_BARCODE", "COMMENTS",
                 "PLATE_COMMENTS", "CREATED_AT", "UPDATED_AT"}
    _n_rows = len(all_plate_data)
    for col in all_plate_data.columns:
        if col in _CAT_SKIP or all_plate_data[col].dtype != object:
            continue
        _nu = all_plate_data[col].nunique(dropna=True)
        if _nu <= 2000 and _nu < 0.02 * _n_rows:
            _s = all_plate_data[col].astype("category")
            if "" not in _s.cat.categories:
                _s = _s.cat.add_categories([""])
            all_plate_data[col] = _s

    return all_plate_data, workorder_data, dpart_data


# ---------------------------------------------------------------------------
# Part extraction from workorders
# ---------------------------------------------------------------------------

def extract_required_parts(workorder_data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [New Parts, Reactions Required, Is_Control] —
    one row per unique part, with reaction count = number of assemblies needing it.

    Control parts are seeded with Reactions Required = CONTROL_BUFFER_RXNS regardless
    of whether they appear in any workorder.
    """
    rows: list[str] = []

    def _names_from_str(s: str) -> list[str]:
        """Split a 'name:available, name:available' string into just names."""
        return [item.split(":")[0] for item in s.split(", ") if item.split(":")[0]]

    # Builds that have reached LSP have already assembled (and won't be re-prepped),
    # so they no longer need their input parts — drop them from demand. Builds still
    # in PARTS/ASM are kept (an ASM build can fail and need a re-prep). Proxy for
    # "in LSP" = the product has an (active) lsp_workorder.
    lsp_products = set(
        workorder_data.loc[workorder_data["WORKORDER_TYPE"] == "lsp_workorder", "STOCK_ID"]
        .dropna().astype(str)
    ) if "WORKORDER_TYPE" in workorder_data.columns else set()

    for _, wo in workorder_data.iterrows():
        wtype = wo.get("WORKORDER_TYPE", "")
        if wtype in ("golden_gate_workorder", "gibson_workorder"):
            if str(wo.get("STOCK_ID")) in lsp_products:
                continue   # this build is already in LSP — part consumed, not re-needed
            rows.extend(_names_from_str(parse_parts(wo.get("parts_json"))))
            bb = parse_backbone(wo.get("backbone_json"))
            if bb:
                rows.append(bb.split(":")[0])
        elif wtype == "pcr_workorder":
            rows.extend(_names_from_str(extract_pcr_info(wo)))

    # pAI-, d-, o- prefix parts only; skip UUIDs (≥36 chars) and empty strings
    valid_rows = [
        r for r in rows
        if r and str(r).startswith(("pAI", "d", "o")) and len(str(r)) < 36
    ]

    # Double-count PCR parts from WAITING/READY workorders (mirrors notebook dpart_page logic)
    dparts_wos = workorder_data[
        workorder_data["STATUS"].isin(["WAITING", "READY"]) &
        (workorder_data["WORKORDER_TYPE"] == "pcr_workorder")
    ]
    for _, wo in dparts_wos.iterrows():
        for name in _names_from_str(extract_pcr_info(wo)):
            if name and str(name).startswith(("pAI", "d", "o")) and len(str(name)) < 36:
                valid_rows.append(name)

    if valid_rows:
        counts = pd.Series(valid_rows).value_counts().reset_index()
        counts.columns = ["New Parts", "Reactions Required"]
    else:
        counts = pd.DataFrame(columns=["New Parts", "Reactions Required"])

    counts["Is_Control"] = False

    # --- Seed control parts ---
    # Controls always appear in the report with CONTROL_BUFFER_RXNS as the target.
    # If a control also appeared in workorders, take the larger of the two demands.
    for ctrl in CONTROL_PARTS:
        existing = counts[counts["New Parts"] == ctrl]
        if existing.empty:
            counts = pd.concat(
                [counts, pd.DataFrame([{
                    "New Parts": ctrl,
                    "Reactions Required": CONTROL_BUFFER_RXNS,
                    "Is_Control": True,
                }])],
                ignore_index=True,
            )
        else:
            idx = existing.index[0]
            counts.at[idx, "Reactions Required"] = max(
                int(counts.at[idx, "Reactions Required"]), CONTROL_BUFFER_RXNS
            )
            counts.at[idx, "Is_Control"] = True

    if counts.empty:
        return pd.DataFrame(columns=["New Parts", "Reactions Required", "Is_Control"])

    # pAI- first, then d-, then o-
    pai = counts[counts["New Parts"].str.startswith("pAI")].copy()
    d   = counts[counts["New Parts"].str.startswith("d")].copy()
    o   = counts[counts["New Parts"].str.startswith("o")].copy()
    return pd.concat([pai, d, o], ignore_index=True)


# ---------------------------------------------------------------------------
# Inventory workflow
# ---------------------------------------------------------------------------

def run_optimized_lab_workflow(
    parts_list: pd.DataFrame,
    all_plate_data: pd.DataFrame,
    dpart_data: pd.DataFrame,
    now: dt.datetime,
) -> pd.DataFrame:
    WEIGHT = 1e-12  # pg per dalton
    DEAD_VOL = 20   # µL dead volume in Echo plate

    df = parts_list.drop_duplicates(subset=["Part"]).reset_index(drop=True)
    # Preserve Is_Control flag through the workflow
    if "Is_Control" not in df.columns:
        df["Is_Control"] = False
    # PCR runs needed to refill this dpart (used to size the Refill-queue PCR CSV)
    df["PCR Runs Needed"] = 0

    # Inventory = DNA source stock only. The Echo source plate also carries Temp/Glycerol/
    # working wells; restrict to WELL_TYPE 'Stock' so those never count toward "Have".
    if "LABWARE" in all_plate_data.columns:
        _echo_mask = _echo384(all_plate_data)
        if "WELL_TYPE" in all_plate_data.columns:
            _echo_mask = _echo_mask & (all_plate_data["WELL_TYPE"] == "Stock")
        echo_plates = all_plate_data[_echo_mask].copy()
    else:
        echo_plates = pd.DataFrame()

    cutoff = now - dt.timedelta(days=200)

    if len(echo_plates) == 0:
        is_fresh = pd.Series(dtype=bool)
    else:
        if "CREATED_AT" in echo_plates.columns:
            if echo_plates["CREATED_AT"].dtype == "object":
                echo_plates["CREATED_AT"] = pd.to_datetime(
                    echo_plates["CREATED_AT"], errors="coerce", utc=True
                )
            tz = echo_plates["CREATED_AT"].dt.tz
            if tz is not None and getattr(cutoff, "tzinfo", None) is None:
                cutoff = pd.Timestamp(cutoff, tz="UTC")
            elif tz is None and getattr(cutoff, "tzinfo", None) is not None:
                cutoff = cutoff.replace(tzinfo=None)
            is_fresh = echo_plates["CREATED_AT"] > cutoff
        else:
            # No created date → can't establish freshness → treat as not fresh.
            is_fresh = pd.Series(False, index=echo_plates.index)

    def calc_rxns(sub, dead):
        seq_len = np.where(
            sub["STOCK_ID"].str.startswith("d", na=False),
            pd.to_numeric(sub["DPART_SEQUENCE_LENGTH"], errors="coerce"),
            pd.to_numeric(sub["SEQUENCE_LENGTH"], errors="coerce"),
        )
        vol  = pd.to_numeric(sub["VOLUME_UL"], errors="coerce")
        conc = pd.to_numeric(sub["CONCENTRATION_NGUL"], errors="coerce")
        return (((vol - dead) * conc) / (WEIGHT * seq_len * 6e9)).clip(lower=0).fillna(0)

    if len(echo_plates) > 0:
        echo_plates["rxns_fresh"] = np.where(
            (echo_plates["AVAILABLE"] == "True") & is_fresh,
            calc_rxns(echo_plates, DEAD_VOL), 0
        )
        inventory_map = echo_plates.groupby("STOCK_ID")["rxns_fresh"].sum().astype(int).to_dict()
    else:
        inventory_map = {}

    # --- Oligo tube inventory (uses molarity in nM, 365-day freshness cutoff) ---
    oligo_cutoff = now - dt.timedelta(days=OLIGO_FRESHNESS_DAYS)
    oligo_wells = all_plate_data[
        all_plate_data["STOCK_ID"].str.startswith("o", na=False) &
        (all_plate_data["AVAILABLE"] == "True")
    ].copy() if "MOLARITY_NM" in all_plate_data.columns else pd.DataFrame()

    if len(oligo_wells) > 0 and "CREATED_AT" in oligo_wells.columns:
        if oligo_wells["CREATED_AT"].dtype == "object":
            oligo_wells["CREATED_AT"] = pd.to_datetime(
                oligo_wells["CREATED_AT"], errors="coerce", utc=True
            )
        tz = oligo_wells["CREATED_AT"].dt.tz
        _ocutoff = oligo_cutoff
        if tz is not None and getattr(_ocutoff, "tzinfo", None) is None:
            _ocutoff = pd.Timestamp(_ocutoff, tz="UTC")
        elif tz is None and getattr(_ocutoff, "tzinfo", None) is not None:
            _ocutoff = _ocutoff.replace(tzinfo=None)
        is_fresh_oligo = oligo_wells["CREATED_AT"] > _ocutoff
        # reactions ≈ (vol - 5µL dead) × molarity_nM / 100_000
        # tuned so a 50µL tube at 100µM (~100,000 nM) ≈ 45 reactions
        oligo_wells["rxns"] = np.where(
            is_fresh_oligo,
            (
                (pd.to_numeric(oligo_wells["VOLUME_UL"], errors="coerce") - 5).clip(lower=0) *
                pd.to_numeric(oligo_wells["MOLARITY_NM"], errors="coerce") / 100_000
            ).fillna(0).clip(lower=0),
            0,
        )
        oligo_inv = oligo_wells.groupby("STOCK_ID")["rxns"].sum().apply(math.ceil).astype(int).to_dict()
        for k, v in oligo_inv.items():
            inventory_map[k] = inventory_map.get(k, 0) + v

    # --- Template expansion for dparts ---
    # For demand-driven dparts: proportional to workorder demand.
    # For control dparts: proportional to shortfall below CONTROL_BUFFER_RXNS.
    new_templates: list[dict] = []
    for idx, row in df.copy().iterrows():
        part = str(row["Part"])
        if part.startswith("d"):
            match = dpart_data[dpart_data["DPART_NAME"] == part]
            if not match.empty:
                template = match["DPART_TEMPLATE"].values[0]
                if pd.notna(template) and str(template) not in ("None", "nan", ""):
                    df.at[idx, "dPart Template"] = template
                    is_ctrl = bool(row.get("Is_Control", False))
                    child_avail = inventory_map.get(part, 0)

                    if is_ctrl:
                        # For controls: demand on template = shortfall below buffer target
                        child_req = CONTROL_BUFFER_RXNS
                        needed = math.ceil((max(child_req - child_avail, 0) / 10) + 1) \
                            if child_avail < CONTROL_REFILL_TRIGGER else 0
                    else:
                        child_req = float(row.get("Reactions Required", 0))
                        needed = math.ceil((child_req / 10) + 1) if child_req > child_avail else 0

                    # Record this dpart's own PCR-run count for the Refill queue CSV
                    df.at[idx, "PCR Runs Needed"] = int(needed)

                    if needed > 0:
                        existing = df["Part"] == template
                        if existing.any():
                            t_idx = df.index[existing][0]
                            df.at[t_idx, "Reactions Required"] = float(df.at[t_idx, "Reactions Required"]) + needed
                        else:
                            new_templates.append({"Part": template, "Reactions Required": needed, "Is_Control": False})

                    # Also propagate to oligos (both demand and control dparts)
                    oligo1 = match["OLIGO_1"].values[0] if "OLIGO_1" in match.columns else None
                    oligo2 = match["OLIGO_2"].values[0] if "OLIGO_2" in match.columns else None
                    for oligo in [oligo1, oligo2]:
                        if oligo and pd.notna(oligo) and str(oligo) not in ("None", "nan", "o"):
                            oligo_avail = inventory_map.get(str(oligo), 0)
                            if is_ctrl:
                                oligo_needed = math.ceil((max(CONTROL_BUFFER_RXNS - oligo_avail, 0) / 10) + 1) \
                                    if oligo_avail < CONTROL_REFILL_TRIGGER else 0
                            else:
                                oligo_needed = math.ceil((float(row.get("Reactions Required", 0)) / 10) + 1) \
                                    if float(row.get("Reactions Required", 0)) > oligo_avail else 0
                            if oligo_needed > 0:
                                existing_o = df["Part"] == str(oligo)
                                if existing_o.any():
                                    o_idx = df.index[existing_o][0]
                                    df.at[o_idx, "Reactions Required"] = float(df.at[o_idx, "Reactions Required"]) + oligo_needed
                                else:
                                    new_templates.append({"Part": str(oligo), "Reactions Required": oligo_needed, "Is_Control": False})

    if new_templates:
        new_df = pd.DataFrame(new_templates).groupby("Part").agg(
            {"Reactions Required": "sum", "Is_Control": "first"}
        ).reset_index()
        df = pd.concat([df, new_df], ignore_index=True)

    df["Reactions Required"] = pd.to_numeric(df["Reactions Required"]).apply(math.ceil).astype(int)
    if "PCR Runs Needed" not in df.columns:
        df["PCR Runs Needed"] = 0
    df["PCR Runs Needed"] = pd.to_numeric(df["PCR Runs Needed"], errors="coerce").fillna(0).astype(int)

    # --- Sequence length enrichment (pAI parts → colonies-to-pick calc; oligos → display) ---
    if "SEQUENCE_LENGTH" in all_plate_data.columns:
        pai_seqlen = (
            all_plate_data[all_plate_data["STOCK_ID"].str.startswith("pAI", na=False)]
            .dropna(subset=["SEQUENCE_LENGTH"])
            .drop_duplicates("STOCK_ID")
            .set_index("STOCK_ID")["SEQUENCE_LENGTH"]
        )
        df["SEQUENCE_LENGTH"] = df["Part"].map(pai_seqlen)
    else:
        df["SEQUENCE_LENGTH"] = np.nan

    if "OLIGO_SEQUENCE_LENGTH" in all_plate_data.columns:
        oligo_seqlen = (
            all_plate_data[all_plate_data["STOCK_ID"].str.startswith("o", na=False)]
            .drop_duplicates("STOCK_ID")
            .set_index("STOCK_ID")["OLIGO_SEQUENCE_LENGTH"]
        )
        df["OLIGO_SEQUENCE_LENGTH"] = df["Part"].map(oligo_seqlen)
    else:
        df["OLIGO_SEQUENCE_LENGTH"] = np.nan

    if len(echo_plates) > 0:
        echo_plates["rxns_old"] = np.where(
            (echo_plates["AVAILABLE"] == "True") & ~is_fresh,
            calc_rxns(echo_plates, DEAD_VOL), 0
        )
        echo_plates["rxns_confirmed"] = np.where(
            (echo_plates["AVAILABLE"] != "True") & (echo_plates["SEQ_CONFIRMED"] == "True") & is_fresh,
            calc_rxns(echo_plates, 15), 0
        )
        old_plates_map = (
            echo_plates[echo_plates["rxns_old"] > 0]
            .groupby("STOCK_ID")["PLATE_ID"]
            .apply(lambda x: ",".join(str(p) for p in x.unique()) + ",")
            .to_dict()
        )
        rxns_old_series  = echo_plates.groupby("STOCK_ID")["rxns_old"].sum()
        rxns_conf_series = echo_plates.groupby("STOCK_ID")["rxns_confirmed"].sum()
    else:
        old_plates_map   = {}
        rxns_old_series  = pd.Series(dtype=float)
        rxns_conf_series = pd.Series(dtype=float)

    df["Reactions Available"]      = df["Part"].map(inventory_map).fillna(0).astype(int)
    df["Reactions Available Old"]  = df["Part"].map(rxns_old_series).fillna(0).astype(int)
    df["Reactions Seq Confirmed"]  = df["Part"].map(rxns_conf_series).fillna(0).astype(int)
    df["Old Plates"]               = df["Part"].map(old_plates_map).fillna("")

    # --- Micronic tube counts ---
    tubes = all_plate_data[
        (all_plate_data["LABWARE"] == "Micronic Tube Rack") &
        (all_plate_data["SEQ_CONFIRMED"] == "True") &
        (all_plate_data["WELL_TYPE"] == "Stock")
    ].copy()
    short_c = tubes[tubes["PLATE_LOCATION_BOX"] == "None"].groupby("STOCK_ID").size()
    long_c  = tubes[tubes["PLATE_LOCATION_BOX"] != "None"].groupby("STOCK_ID").size()
    df["Micronic Tubes"] = df["Part"].apply(lambda x: f"({short_c.get(x, 0)},{long_c.get(x, 0)})")

    # --- Glycerol info ---
    gly_labware = ["Thermo V Bottom Plate", "Eppendorf V Microplate"]
    gly = all_plate_data[
        all_plate_data["LABWARE"].isin(gly_labware) &
        (all_plate_data["AVAILABLE"] == "True")
    ].copy()
    gly["is_neb"] = gly["COMP_CELL"].str.startswith("NEB", na=False)
    gly = (
        gly.sort_values(["STOCK_ID", "is_neb"])
        .drop_duplicates("STOCK_ID", keep="last")
        .set_index("STOCK_ID")
    )

    def get_antibiotic(r):
        if r.get("ANTI_KAN") == "True":  return "Kan"
        if r.get("ANTI_SPEC") == "True": return "Spec"
        if r.get("ANTI_CARB") == "True": return "Carb"
        return ""

    df["Cell Strain"]      = df["Part"].map(gly["COMP_CELL"]).fillna("")
    df["Glycerol Plate"]   = df["Part"].map(gly["PLATE_ID"]).map(
        # PLATE_ID is a BigQuery nullable Int64 — can't .fillna("") directly
        lambda x: "" if pd.isna(x) else (str(int(x)) if not isinstance(x, str) else x)
    )
    df["Glycerol Location"]= df["Part"].map(gly["PLATE_LOCATION_BOX"]).fillna("")
    df["Antibiotic"]       = gly.apply(get_antibiotic, axis=1).reindex(df["Part"]).values
    df["Glycerol Well"]    = df["Part"].map(gly["WELL_NUMBER"]).map(WELLS_96).fillna("")
    def _well_id_str(x):
        if not pd.notna(x) or str(x) in ("", "nan"):
            return ""
        try:
            return f"well{int(float(x))}"
        except (ValueError, TypeError):
            return f"well{x}"

    df["Glycerol Well ID"] = df["Part"].map(gly["WELL_ID"]).apply(_well_id_str)

    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Action classification
# ---------------------------------------------------------------------------

def classify_actions(
    parts_list: pd.DataFrame,
    workorder_data: pd.DataFrame,
    all_plate_data: pd.DataFrame,
    now: dt.datetime,
) -> pd.DataFrame:
    df = parts_list.copy()
    df["Actions Required"] = ""
    df["Wells_To_mark_available"] = ""

    # Plasmid DNA we physically have anywhere (any Stock well, any age/labware).
    # When there's no glycerol to streak from, the refill route is TRANSFORMATION
    # of that DNA into fresh cells. With no DNA at all, it's a reorder/synthesis.
    if "WELL_TYPE" in all_plate_data.columns and "STOCK_ID" in all_plate_data.columns:
        _dna_ids = set(
            all_plate_data.loc[all_plate_data["WELL_TYPE"] == "Stock", "STOCK_ID"].dropna().astype(str)
        )
    else:
        _dna_ids = set()

    def _no_fresh_source(part) -> str:
        """Short, no glycerol & no fresh seq-confirmed wells. Only PLASMIDS can be
        transformed (from their DNA); oligos/dParts/no-DNA → reorder/synthesize."""
        p = str(part)
        if p.startswith("pAI") and p in _dna_ids:
            return "Transform"
        return "True"

    cutoff = now - dt.timedelta(days=200)
    if "CREATED_AT" in all_plate_data.columns and len(all_plate_data) > 0:
        if all_plate_data["CREATED_AT"].dtype != "object":
            tz = all_plate_data["CREATED_AT"].dt.tz
            if tz is not None and getattr(cutoff, "tzinfo", None) is None:
                cutoff = pd.Timestamp(cutoff, tz="UTC")
            elif tz is None and getattr(cutoff, "tzinfo", None) is not None:
                cutoff = cutoff.replace(tzinfo=None)

    for idx in df.index:
        req    = int(df.at[idx, "Reactions Required"])
        avail  = int(df.at[idx, "Reactions Available"])
        is_ctrl = bool(df.at[idx, "Is_Control"]) if "Is_Control" in df.columns else False

        # Sufficiency check differs for controls vs demand parts:
        #   Controls: sufficient when avail >= CONTROL_BUFFER_RXNS (96).
        #             Trigger refill when avail < CONTROL_REFILL_TRIGGER (30).
        #   Demand:   sufficient when avail >= req + max(10, req).
        if is_ctrl:
            if avail >= CONTROL_BUFFER_RXNS:
                continue  # well-stocked control — no action needed
            if avail >= CONTROL_REFILL_TRIGGER:
                continue  # below target but above trigger — no action yet
        else:
            if req + max(10, req) <= avail:
                continue  # sufficient demand stock — no action needed

        part = df.at[idx, "Part"]
        workorders = workorder_data[workorder_data["STOCK_ID"] == part]

        # For demand parts only: check if a workorder already covers this part
        if not is_ctrl and len(workorders) > 0:
            wo = workorders[workorders["STATUS"] != "BLOCKED"]
            if wo.empty:
                wo = workorders
            first = wo.iloc[0]
            df.at[idx, "Actions Required"] = f"{first['WORKORDER_TYPE']} is {first['STATUS']}"
            continue

        confirmed     = df.at[idx, "Reactions Seq Confirmed"]
        confirmed_val = float(confirmed) if str(confirmed) not in ("", "nan") else 0.0
        gly_well_id   = str(df.at[idx, "Glycerol Well ID"])

        # How many more reactions do we need? For controls use buffer target.
        effective_req = CONTROL_BUFFER_RXNS if is_ctrl else req
        shortfall = max(effective_req - avail, 0)

        if (
            confirmed_val > 0 and
            shortfall > 0 and
            shortfall <= confirmed_val
        ):
            # Seq-confirmed wells exist and could cover the shortfall — suggest marking available.
            # Volume threshold: > 30 µL (controls use same rule; >20 is our empty-well value).
            if "CREATED_AT" in all_plate_data.columns:
                freshness_mask = all_plate_data["CREATED_AT"] > cutoff
            else:
                freshness_mask = pd.Series(True, index=all_plate_data.index)

            wells_confirmed = all_plate_data[
                (all_plate_data["STOCK_ID"] == part) &
                (_echo384(all_plate_data)) &
                (all_plate_data["AVAILABLE"] != "True") &
                (pd.to_numeric(all_plate_data["CONCENTRATION_NGUL"], errors="coerce") > 5) &
                (pd.to_numeric(all_plate_data["VOLUME_UL"], errors="coerce") > 30) &   # >30 µL — exclude near-empty wells
                (all_plate_data["SEQ_CONFIRMED"] == "True") &
                freshness_mask
            ].copy()
            wells_confirmed = wells_confirmed.sort_values(
                by="CONCENTRATION_NGUL",
                key=lambda s: pd.to_numeric(s, errors="coerce"),
                ascending=False
            )
            if wells_confirmed.empty and gly_well_id:
                df.at[idx, "Actions Required"] = "Refill"
            elif wells_confirmed.empty:
                df.at[idx, "Actions Required"] = _no_fresh_source(part)
            else:
                well_list = list(reversed(("well" + wells_confirmed["WELL_ID"].astype(str)).tolist()))
                df.at[idx, "Actions Required"] = f"Mark seq confirmed wells available {well_list}"

        elif gly_well_id:
            df.at[idx, "Actions Required"] = "Refill"
        else:
            df.at[idx, "Actions Required"] = _no_fresh_source(part)

        # Informational: all seq-confirmed unavailable Echo wells with > 30 µL (no freshness filter)
        if not df.at[idx, "Actions Required"].startswith("Mark") and len(all_plate_data) > 0:
            _req_cols = {"STOCK_ID", "LABWARE", "AVAILABLE", "CONCENTRATION_NGUL",
                         "VOLUME_UL", "SEQ_CONFIRMED", "WELL_ID"}
            if _req_cols.issubset(all_plate_data.columns):
                all_conf = all_plate_data[
                    (all_plate_data["STOCK_ID"] == part) &
                    (_echo384(all_plate_data)) &
                    (all_plate_data["AVAILABLE"] != "True") &
                    (pd.to_numeric(all_plate_data["CONCENTRATION_NGUL"], errors="coerce") > 5) &
                    (pd.to_numeric(all_plate_data["VOLUME_UL"], errors="coerce") > 30) &   # >30 µL
                    (all_plate_data["SEQ_CONFIRMED"] == "True")
                ]
            else:
                all_conf = pd.DataFrame()
            if not all_conf.empty:
                df.at[idx, "Wells_To_mark_available"] = ", ".join(
                    "well" + str(w) for w in all_conf["WELL_ID"].tolist()
                )

    return df


# ---------------------------------------------------------------------------
# Output filtering
# ---------------------------------------------------------------------------

def build_output(parts_list: pd.DataFrame) -> pd.DataFrame:
    # astype(object) first so BigQuery nullable Int64/Float64 columns accept "" via fillna
    out = parts_list.astype(object).fillna("").copy()
    out = out[out["Actions Required"] != ""].rename(columns={"Actions Required": "Action Suggested"})

    # Demand parts: must have Reactions Required > 0.
    # Control parts: always include when they have an action (their Reactions Required = 96 by design).
    is_ctrl = out.get("Is_Control", pd.Series(False, index=out.index)).astype(bool)
    out = out[is_ctrl | (out["Reactions Required"] > 0)]

    # Exclude parts that are already being handled (lsp workorders, RUNNING/WAITING/READY)
    action = out["Action Suggested"]
    out = out[
        ~action.str[:3].eq("lsp") &
        ~action.str[-7:].eq("RUNNING") &
        ~action.str[-7:].eq("WAITING") &
        ~action.str[-5:].eq("READY")
    ].reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# HTML renderer  (dashboard tab fragment)
# ---------------------------------------------------------------------------

_ACTION_LABEL = {
    "Refill":    ("Refill", "#fef3c7", "#92400e", "#fde68a"),       # has glycerol → streak
    "Transform": ("Transform", "#fff7ed", "#c2410c", "#fed7aa"),    # no glycerol, have DNA → transform
    "True":      ("Reorder", "#fff1f5", "#be185d", "#fecdd3"),       # no DNA at all → reorder/synthesize
    "Mark":      ("Mark Available", "#eff6ff", "#1d4ed8", "#bfdbfe"),
}


def _action_badge(action: str) -> str:
    action_s = str(action)
    if action_s.startswith("Mark"):
        label, bg, fg, border = _ACTION_LABEL["Mark"]
    elif action_s == "Refill":
        label, bg, fg, border = _ACTION_LABEL["Refill"]
    elif action_s == "Transform":
        label, bg, fg, border = _ACTION_LABEL["Transform"]
    elif action_s == "True":
        label, bg, fg, border = _ACTION_LABEL["True"]
    else:
        label = action_s[:40]
        bg, fg, border = "#f5f5f7", "#6b7280", "#d1d5db"
    return (
        f'<span class="badge" style="background:{bg};color:{fg};'
        f'border:1px solid {border}">{label}</span>'
    )


def render_parts_inventory_html(df: pd.DataFrame, generated_at: dt.datetime | None = None) -> str:
    """
    Returns an HTML fragment (no <html>/<body> wrapper) suitable for embedding
    as a dashboard tab.  Matches the .wo-table / .badge CSS already in the dashboard.
    """
    if generated_at is None:
        generated_at = dt.datetime.now(tz=dt.timezone.utc)
    ts = generated_at.strftime("%Y-%m-%d %H:%M UTC")

    if df.empty:
        return (
            '<div style="padding:24px;color:#86868b;font-size:12px;">'
            "No parts require action.</div>"
        )

    # Summary counts
    action_col = "Action Suggested" if "Action Suggested" in df.columns else "Actions Required"
    action_s = df.get(action_col, pd.Series(dtype=str))
    is_ctrl_s = df.get("Is_Control", pd.Series(False, index=df.index)).astype(bool)

    n_total       = len(df)
    n_refill      = (action_s == "Refill").sum()
    n_mark        = action_s.str.startswith("Mark").sum()
    n_nosrc       = (action_s == "True").sum()
    n_ctrl        = is_ctrl_s.sum()

    rows_html = []
    for _, row in df.iterrows():
        part        = str(row.get("Part", ""))
        req         = int(row.get("Reactions Required", 0))
        avail       = int(row.get("Reactions Available", 0))
        old         = int(row.get("Reactions Available Old", 0))
        conf        = int(row.get("Reactions Seq Confirmed", 0))
        action      = str(row.get(action_col, ""))
        micro       = str(row.get("Micronic Tubes", ""))
        gly_plate   = str(row.get("Glycerol Plate", ""))
        gly_loc     = str(row.get("Glycerol Location", ""))
        gly_well    = str(row.get("Glycerol Well", ""))
        antibiotic  = str(row.get("Antibiotic", ""))
        cell_strain = str(row.get("Cell Strain", ""))
        template    = str(row.get("dPart Template", ""))
        old_plates  = str(row.get("Old Plates", ""))
        wells_mark  = str(row.get("Wells_To_mark_available", ""))
        oligo_len   = row.get("OLIGO_SEQUENCE_LENGTH", None)
        seq_len     = row.get("SEQUENCE_LENGTH", None)
        is_ctrl     = bool(row.get("Is_Control", False))

        # For controls, the "needed" bar is relative to the 96-reaction buffer target
        needed = CONTROL_BUFFER_RXNS if is_ctrl else req + max(10, req)

        def _blank(s): return not s or s in ("", "nan", "None")

        # Part cell — controls get a small "CTL" marker
        part_cell = f'<span class="stock-id-badge">{part}</span>'
        if is_ctrl:
            part_cell += '<span style="font-size:7px;background:#fef9c3;color:#854d0e;border:1px solid #fde68a;border-radius:3px;padding:0 3px;margin-left:3px">CTL</span>'
        if not _blank(template):
            part_cell += f'<br><span style="font-size:8px;color:#86868b;">tmpl: {template}</span>'
        if not _blank(oligo_len) and part.startswith("o"):
            part_cell += f'<br><span style="font-size:8px;color:#6b7280;">seq: {int(float(oligo_len))} bp</span>'

        # Reactions cell — avail / needed; bar fills against needed
        bar_pct = min(100, int(avail / max(needed, 1) * 100))
        bar_color = "#16a34a" if bar_pct >= 100 else ("#d97706" if bar_pct >= 50 else "#be185d")
        rxn_cell = (
            f'<span style="font-weight:700;color:#1d1d1f">{avail}</span>'
            f'<span style="color:#86868b"> / {needed}</span>'
            f'<div style="height:3px;border-radius:2px;background:#e5e5e7;margin-top:2px;width:60px">'
            f'<div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:2px"></div></div>'
        )
        if old > 0:
            rxn_cell += f'<br><span style="font-size:7px;color:#86868b">{old} old</span>'
        if conf > 0:
            rxn_cell += f'<br><span style="font-size:7px;color:#6d28d9">{conf} confirmed</span>'

        # Glycerol cell
        gly_parts = []
        if not _blank(cell_strain):
            gly_parts.append(f'<span style="font-size:8px;font-weight:600;color:#374151">{cell_strain}</span>')
        if not _blank(gly_loc):
            loc_txt = gly_loc
            if not _blank(gly_well):
                loc_txt += f' <span style="color:#86868b">{gly_well}</span>'
            gly_parts.append(f'<span style="font-size:8px">{loc_txt}</span>')
        if not _blank(gly_plate) and gly_plate != "0":
            gly_parts.append(f'<span style="font-size:7px;color:#86868b">plate {gly_plate}</span>')
        if not _blank(antibiotic):
            gly_parts.append(
                f'<span class="badge" style="font-size:7px;background:#f0f0f2;'
                f'color:#374151;border:1px solid #d1d5db">{antibiotic}</span>'
            )
        gly_cell = "<br>".join(gly_parts)

        # Old plates / tubes cell
        inv_parts = []
        if not _blank(old_plates):
            plates = [p.strip() for p in old_plates.split(",") if p.strip()][:4]
            inv_parts.append(", ".join(plates) + (" …" if len(plates) == 4 else ""))
        if not _blank(micro) and micro != "(0,0)":
            inv_parts.append(f'<span style="color:#6b7280">tubes: {micro}</span>')
        inv_cell = f'<span style="font-size:8px">{"<br>".join(inv_parts)}</span>'

        # Colonies to pick (Refill only, pAI parts with known sequence length)
        colonies_html = ""
        if action == "Refill" and not _blank(seq_len) and pd.notna(seq_len):
            shortfall = max(needed - avail, 0)
            if shortfall > 0:
                conc_avg = 10.0 if antibiotic == "Carb" else 20.0
                rxns_per_col = (30.0 * conc_avg) / (1e-12 * float(seq_len) * 6e9)
                if rxns_per_col > 0:
                    n_col = math.ceil(shortfall / rxns_per_col)
                    colonies_html = (
                        f'<br><span style="font-size:8px;color:#374151">'
                        f'~{n_col} {"colony" if n_col == 1 else "colonies"} to pick</span>'
                    )

        # Action cell
        action_cell = _action_badge(action) + colonies_html
        if not _blank(wells_mark):
            wells_list = [w.strip() for w in wells_mark.split(",") if w.strip()][:6]
            ellipsis = " …" if len(wells_list) == 6 else ""
            action_cell += (
                f'<br><span style="font-size:7px;color:#1d4ed8">'
                f'mark: {", ".join(wells_list)}{ellipsis}</span>'
            )

        ctrl_row_style = ' style="background:#fffdf0"' if is_ctrl else ''
        rows_html.append(
            f"<tr{ctrl_row_style}>"
            f"<td>{part_cell}</td>"
            f"<td>{rxn_cell}</td>"
            f"<td>{gly_cell}</td>"
            f"<td>{inv_cell}</td>"
            f"<td>{action_cell}</td>"
            f"</tr>"
        )

    rows_str = "\n".join(rows_html)

    return f"""
<div style="padding:12px 16px">
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;flex-wrap:wrap">
    <span style="font-size:13px;font-weight:700;color:#1d1d1f">Parts Inventory</span>
    <span style="font-size:9px;color:#86868b">{ts}</span>
    <span class="badge" style="background:#fff7ed;color:#c2410c;border:1px solid #fed7aa">{n_total} parts need action</span>
    {f'<span class="badge" style="background:#fef3c7;color:#92400e;border:1px solid #fde68a">{n_refill} refill</span>' if n_refill else ''}
    {f'<span class="badge" style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe">{n_mark} mark available</span>' if n_mark else ''}
    {f'<span class="badge" style="background:#fff1f5;color:#be185d;border:1px solid #fecdd3">{n_nosrc} no source</span>' if n_nosrc else ''}
    {f'<span class="badge" style="background:#fef9c3;color:#854d0e;border:1px solid #fde68a">{n_ctrl} controls</span>' if n_ctrl else ''}
  </div>
  <table class="wo-table" style="width:100%">
    <thead>
      <tr>
        <th>Part</th>
        <th>Rxns avail / req</th>
        <th>Glycerol / strain</th>
        <th>Old plates / tubes</th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody>
{rows_str}
    </tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# Action queues  (reframed output: three copy-paste lab queues)
# ---------------------------------------------------------------------------

MARK_AVAILABLE_VOL_MIN = 25   # µL — Echo wells with MORE than this can be marked available
MARK_AVAILABLE_CONC_MIN = 5   # ng/µL — below this is too dilute to mark available
CLEAN_INVENTORY_VOL_MAX = 25  # µL — Echo wells with this volume OR LESS get marked unavailable
LOW_CONC_MIN_NGUL = 5         # ng/µL — available Echo Stock wells below this are too dilute to use

# "Live well" liveness filter — keeps disposed/retired wells out of the queues.
FRESHNESS_DAYS = 200                                   # plate considered live within this window
DISPOSED_LOCATIONS = {"DISCARDED", "", "None", "nan"}  # not a real storage location


def _location_live(df: pd.DataFrame) -> pd.Series:
    """True where the plate is in a real storage box (not DISCARDED / blank)."""
    loc = (df["PLATE_LOCATION_BOX"] if "PLATE_LOCATION_BOX" in df.columns
           else pd.Series("", index=df.index)).astype(str)
    return ~loc.isin(DISPOSED_LOCATIONS)


def _is_fresh(df: pd.DataFrame, now: dt.datetime) -> pd.Series:
    """True where the plate is within the freshness window. (DVs/Deli Left are no
    longer special-cased — they age out under the same rule as every other plate.)"""
    cutoff = pd.Timestamp(now - dt.timedelta(days=FRESHNESS_DAYS))
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    created_raw = df["CREATED_AT"] if "CREATED_AT" in df.columns else pd.Series(pd.NaT, index=df.index)
    created = pd.to_datetime(created_raw, errors="coerce", utc=True)
    return created >= cutoff


def _well_tokens(wells: pd.DataFrame) -> list[str]:
    """Return ['well<id>', ...] for a filtered well DataFrame, ordered by WELL_ID."""
    if wells.empty or "WELL_ID" not in wells.columns:
        return []
    # The plate query LEFT JOINs several tables, so one physical well can appear in
    # multiple rows — dedupe WELL_ID so each well shows up once in the paste string.
    ids = pd.to_numeric(wells["WELL_ID"], errors="coerce").dropna().astype(int)
    return [f"well{i}" for i in sorted(ids.unique())]


def build_mark_available_queue(all_plate_data: pd.DataFrame, now: dt.datetime | None = None,
                               exclude_plate_ids: set | None = None) -> list[str]:
    """
    Whole-inventory queue (NOT scoped to active workorder parts): every seq-confirmed,
    not-yet-available 384 Echo source-plate well that is a *live* well — on a fresh plate,
    in a real storage box (not DISCARDED/blank), with volume >
    MARK_AVAILABLE_VOL_MIN µL and concentration > MARK_AVAILABLE_CONC_MIN ng/µL.
    These should be flipped to available=True. Returns a list of 'well<id>' tokens.

    `exclude_plate_ids`: plate IDs whose wells must NEVER be suggested for marking
    available (LSP-linked plates slated for disposal — they must stay unavailable).
    """
    if all_plate_data.empty:
        return []
    if now is None:
        now = dt.datetime.now(tz=dt.timezone.utc)
    vol  = pd.to_numeric(all_plate_data.get("VOLUME_UL"), errors="coerce")
    conc = pd.to_numeric(all_plate_data.get("CONCENTRATION_NGUL"), errors="coerce")
    mask = (
        (_echo384(all_plate_data)) &
        (all_plate_data["SEQ_CONFIRMED"] == "True") &
        (all_plate_data["AVAILABLE"] != "True") &
        (vol > MARK_AVAILABLE_VOL_MIN) &
        (conc > MARK_AVAILABLE_CONC_MIN) &
        _is_fresh(all_plate_data, now) &
        _location_live(all_plate_data)
    )
    if exclude_plate_ids and "PLATE_ID" in all_plate_data.columns:
        mask &= ~all_plate_data["PLATE_ID"].isin(exclude_plate_ids)
    return _well_tokens(all_plate_data[mask])


def build_clean_inventory_queue(all_plate_data: pd.DataFrame, now: dt.datetime | None = None,
                                exclude_oligos: bool = False) -> list[str]:
    """
    Whole-inventory queue: currently-available 384 Echo source-plate wells that should be
    flipped to unavailable for ANY of:
      • NEARLY EMPTY   — volume <= CLEAN_INVENTORY_VOL_MAX µL
      • PAST EXPIRATION — older than the freshness window (oligos OLIGO_FRESHNESS_DAYS, else FRESHNESS_DAYS)
      • LOW CONCENTRATION — a recorded conc below LOW_CONC_MIN_NGUL ng/µL (too dilute to use;
        oligos/unquantified wells have null ng/µL so they never match this clause)
    DISCARDED plates are skipped; blank-location wells ('in the bin, no home yet') ARE included.
    Already-unavailable wells are skipped (marking them is a no-op).
    """
    if all_plate_data.empty:
        return []
    if now is None:
        now = dt.datetime.now(tz=dt.timezone.utc)
    vol = pd.to_numeric(all_plate_data.get("VOLUME_UL"), errors="coerce")
    conc = pd.to_numeric(all_plate_data.get("CONCENTRATION_NGUL"), errors="coerce")
    created = pd.to_datetime(all_plate_data.get("CREATED_AT"), errors="coerce", utc=True)
    cnow = pd.Timestamp(now)
    if cnow.tzinfo is None:
        cnow = cnow.tz_localize("UTC")
    age = (cnow - created).dt.days
    sid = all_plate_data.get("STOCK_ID", pd.Series("", index=all_plate_data.index)).astype(str)
    exp_days = np.where(sid.str.startswith("o"), OLIGO_FRESHNESS_DAYS, FRESHNESS_DAYS)
    near_empty = vol <= CLEAN_INVENTORY_VOL_MAX
    expired = age > exp_days                      # NaT age → NaN > x → False (kept available)
    low_conc = conc.notna() & (conc < LOW_CONC_MIN_NGUL)
    loc = (all_plate_data["PLATE_LOCATION_BOX"] if "PLATE_LOCATION_BOX" in all_plate_data.columns
           else pd.Series("", index=all_plate_data.index)).fillna("").astype(str)
    mask = (
        (_echo384(all_plate_data)) &
        (all_plate_data["AVAILABLE"] == "True") &
        (near_empty | expired | low_conc) &
        ~loc.str.upper().str.contains("DISCARD")
    )
    if exclude_oligos:
        mask &= ~sid.str.startswith("o")
    return _well_tokens(all_plate_data[mask])


def build_miniprep_unavail_queue(all_plate_data: pd.DataFrame, now: dt.datetime | None = None) -> list[str]:
    """
    Whole-inventory queue: currently-available 96-well MINIPREP STOCK wells (the concentrated
    plasmid DNA pulled from overnight cultures, before dilution into 384 Echo source plates)
    that are PAST EXPIRATION (older than FRESHNESS_DAYS) and should be flipped to unavailable.
    Scoped to PLATE_PROTOCOL='Miniprep' + WELL_TYPE='Stock' + 96-well plates, real storage box
    only. Already-unavailable wells are skipped.
    """
    if all_plate_data.empty:
        return []
    if now is None:
        now = dt.datetime.now(tz=dt.timezone.utc)
    nwells = pd.to_numeric(all_plate_data.get("PLATE_NUMBER_OF_WELLS"), errors="coerce")
    created = pd.to_datetime(all_plate_data.get("CREATED_AT"), errors="coerce", utc=True)
    cnow = pd.Timestamp(now)
    if cnow.tzinfo is None:
        cnow = cnow.tz_localize("UTC")
    age = (cnow - created).dt.days
    mask = (
        (nwells == 96) &
        (all_plate_data["PLATE_PROTOCOL"] == "Miniprep") &
        (all_plate_data["WELL_TYPE"] == "Stock") &
        (all_plate_data["AVAILABLE"] == "True") &
        (age > FRESHNESS_DAYS) &                  # NaT age → NaN > x → False (kept available)
        _location_live(all_plate_data)
    )
    return _well_tokens(all_plate_data[mask])


def build_exhausted_plates_queue(all_plate_data: pd.DataFrame) -> list[dict]:
    """
    384 Echo Source plates where EVERY well is drained to 0 µL (fully used up) and the plate
    is not already DISCARDED — physical-disposal candidates. Returns per-plate dicts
    (plate_id, wells, created, location), newest first. Plates with no volume data at all
    (all-null) are NOT included — only plates with a recorded 0 across every well.
    """
    if all_plate_data.empty or "LABWARE" not in all_plate_data.columns:
        return []
    echo = all_plate_data[_echo384(all_plate_data)].copy()
    if echo.empty:
        return []
    echo["_vol"] = pd.to_numeric(echo.get("VOLUME_UL"), errors="coerce")
    loc = (echo["PLATE_LOCATION_BOX"] if "PLATE_LOCATION_BOX" in echo.columns
           else pd.Series("", index=echo.index)).fillna("").astype(str)
    echo = echo[~loc.str.upper().str.contains("DISCARD")]
    if echo.empty:
        return []
    created_all = pd.to_datetime(echo.get("CREATED_AT"), errors="coerce", utc=True)
    items: list[dict] = []
    for pid, sub in echo.groupby("PLATE_ID"):
        vols = sub["_vol"]
        if vols.notna().sum() == 0 or vols.max() > 0:
            continue  # no volume data (not "exhausted"), or still has usable wells
        created = created_all.loc[sub.index].min()
        locv = sub["PLATE_LOCATION_BOX"].astype(str)
        locmode = locv.mode().iloc[0] if len(locv) else ""
        items.append({
            "plate_id": int(pid) if pd.notna(pid) else pid,
            "wells": int(len(sub)),
            "created": created.date().isoformat() if pd.notna(created) else "",
            "location": locmode if locmode not in ("None", "", "nan") else "(no location)",
        })
    items.sort(key=lambda d: d["created"], reverse=True)
    return items


def _pcr_csv_block(dpart_name: str, oligo1: str, oligo2: str, template: str, n_runs: int) -> str:
    """
    Build a ready-to-paste PCR-workorder CSV block: one row per PCR run needed.
    Columns: dpart_name, oligo_1, oligo_2, sequence, templates. The sequence column
    is intentionally blank (filled downstream), matching the existing PCR format.
    """
    def _clean(v):
        s = str(v)
        return "" if s in ("", "nan", "None", "o") else s

    n_runs = max(int(n_runs), 1)
    header = "dpart_name,oligo_1,oligo_2,sequence,templates"
    row = f"{_clean(dpart_name)},{_clean(oligo1)},{_clean(oligo2)},,{_clean(template)}"
    return "\n".join([header] + [row] * n_runs)


def build_refill_queue(output_df: pd.DataFrame, dpart_data: pd.DataFrame) -> list[dict]:
    """
    For every part flagged 'Refill', produce a structured restock instruction:
      - glycerol V-bottom plate + well to streak out from
      - for dParts: a ready-to-paste PCR-workorder CSV block (one row per PCR run needed)
      - for pAI plasmids: streak + miniprep only (no PCR block)
    """
    if output_df.empty or "Action Suggested" not in output_df.columns:
        return []

    dpart_meta = (
        dpart_data.drop_duplicates("DPART_NAME").set_index("DPART_NAME")
        if not dpart_data.empty and "DPART_NAME" in dpart_data.columns
        else pd.DataFrame()
    )

    items: list[dict] = []
    refills = output_df[output_df["Action Suggested"] == "Refill"]
    for _, row in refills.iterrows():
        part    = str(row.get("Part", ""))
        is_ctrl = bool(row.get("Is_Control", False))
        avail   = int(row.get("Reactions Available", 0))
        req     = int(row.get("Reactions Required", 0))
        target  = CONTROL_BUFFER_RXNS if is_ctrl else req + max(10, req)
        pcr_runs = int(row.get("PCR Runs Needed", 0))

        item = {
            "part": part,
            "is_control": is_ctrl,
            "available": avail,
            "target": target,
            "antibiotic": str(row.get("Antibiotic", "")),
            "cell_strain": str(row.get("Cell Strain", "")),
            "glycerol_plate": str(row.get("Glycerol Plate", "")),
            "glycerol_well": str(row.get("Glycerol Well", "")),
            "glycerol_location": str(row.get("Glycerol Location", "")),
            "pcr_runs": pcr_runs,
            "csv_block": "",
            "colonies": 0,
        }

        if part.startswith("d") and not dpart_meta.empty and part in dpart_meta.index:
            meta = dpart_meta.loc[part]
            item["csv_block"] = _pcr_csv_block(
                part,
                meta.get("OLIGO_1", ""),
                meta.get("OLIGO_2", ""),
                meta.get("DPART_TEMPLATE", ""),
                pcr_runs,
            )
        elif part.startswith("pAI"):
            # Plasmid control/part: estimate colonies to pick (streak + miniprep, no PCR)
            # SEQUENCE_LENGTH may be "" (build_output fillna) or a BQ Int64 — coerce safely.
            seq_len = pd.to_numeric(row.get("SEQUENCE_LENGTH"), errors="coerce")
            shortfall = max(target - avail, 0)
            if shortfall > 0 and pd.notna(seq_len):
                conc_avg = 10.0 if item["antibiotic"] == "Carb" else 20.0
                rxns_per_col = (30.0 * conc_avg) / (1e-12 * float(seq_len) * 6e9)
                if rxns_per_col > 0:
                    item["colonies"] = math.ceil(shortfall / rxns_per_col)

        items.append(item)

    return items


def build_no_source_queue(output_df: pd.DataFrame) -> list[dict]:
    """
    Parts that are short on stock but have NO restock path — no glycerol stock to
    streak from and no seq-confirmed wells to mark available. Internally tagged
    "True" by classify_actions (its fallback). These need ordering / synthesis,
    which the streak/mark-available queues can't express.
    Returns [{part, is_control, required}] sorted: controls first, then by part.
    """
    if output_df.empty or "Action Suggested" not in output_df.columns:
        return []
    ns = output_df[output_df["Action Suggested"] == "True"]
    items = [
        {
            "part": str(r.get("Part", "")),
            "is_control": bool(r.get("Is_Control", False)),
            "required": int(r.get("Reactions Required", 0) or 0),
        }
        for _, r in ns.iterrows()
    ]
    items.sort(key=lambda d: (not d["is_control"], d["part"]))
    return items


DISPOSE_FLAG_DAYS = 60  # plates older than ~2 months are flagged for disposal

# Plates physically sitting in the LSP rack that are NOT actually LSP plates — co-mingled
# rearrays (refills + AAV inventory + partner-check QC) that the LSP query catches because a
# few partner-check wells reference an LSP strain. Treated as normal 200-day stock, NOT
# disposed on the 2-month LSP clock. Add a plate ID here to make it an exception.
LSP_DISPOSE_EXCEPTIONS = {15377}


def build_dispose_queue(lsp_plates: pd.DataFrame, now: dt.datetime | None = None) -> list[dict]:
    """
    LSP-linked 384 Echo plates to physically dispose of — returns plate id, location,
    protocol, created date and age per plate. Plates older than DISPOSE_FLAG_DAYS
    (~2 months) are flagged (`old=True`). Skips plates already in 'DISCARDED' and any
    plate in LSP_DISPOSE_EXCEPTIONS (mis-racked stock, not real LSP plates).
    Order is preserved from the query (newest first).
    """
    if lsp_plates is None or lsp_plates.empty:
        return []
    today = (now or dt.datetime.now(tz=dt.timezone.utc)).date()
    items: list[dict] = []
    for _, r in lsp_plates.iterrows():
        loc = str(r.get("LOCATION") or "").strip()
        if "DISCARD" in loc.upper():
            continue  # already disposed (bin location is 'DISCARD'/'DISCARDED')
        try:
            if int(r.get("PLATE_ID")) in LSP_DISPOSE_EXCEPTIONS:
                continue  # mis-racked plate — really stock, handled on the 200-day clock
        except (TypeError, ValueError):
            pass
        pid = r.get("PLATE_ID")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pass
        created = r.get("CREATED_AT")
        created_d = pd.Timestamp(created).date() if pd.notna(created) else None
        age_days = (today - created_d).days if created_d else None
        items.append({
            "plate_id": pid,
            "location": loc if loc not in ("", "None", "nan") else "(no location)",
            "protocol": str(r.get("PROTOCOL") or ""),
            "created": created_d.isoformat() if created_d else "",
            "age_days": age_days,
            "old": age_days is not None and age_days > DISPOSE_FLAG_DAYS,
        })
    return items


def _copy_box(box_id: str, text: str, height: str = "56px") -> str:
    """A read-only textarea + Copy button that copies the textarea's value."""
    return (
        f'<div style="display:flex;gap:6px;align-items:flex-start;margin-top:4px">'
        f'<textarea id="{box_id}" readonly style="flex:1;height:{height};font-family:monospace;'
        f'font-size:10px;border:1px solid #d1d5db;border-radius:4px;padding:5px 7px;'
        f'resize:vertical;color:#1f2937;background:#fafafa">{text}</textarea>'
        f'<button onclick="var t=document.getElementById(\'{box_id}\');t.select();'
        f'navigator.clipboard.writeText(t.value)" '
        f'style="font-size:10px;font-weight:600;padding:5px 10px;border:1px solid #c4b5fd;'
        f'border-radius:4px;background:#ede9fe;color:#6d28d9;cursor:pointer;white-space:nowrap">'
        f'Copy</button></div>'
    )


def render_action_queues_html(
    output_df: pd.DataFrame,
    all_plate_data: pd.DataFrame,
    dpart_data: pd.DataFrame,
    lsp_plates: pd.DataFrame | None = None,
    generated_at: dt.datetime | None = None,
) -> str:
    """
    Reframed dashboard fragment: four copy-paste action queues
    (Mark Available, Clean Inventory, Refill, Dispose) instead of a parts table.
    """
    if generated_at is None:
        generated_at = dt.datetime.now(tz=dt.timezone.utc)
    ts = generated_at.strftime("%Y-%m-%d %H:%M UTC")

    lsp_ids = set()
    if lsp_plates is not None and not lsp_plates.empty and "PLATE_ID" in lsp_plates.columns:
        lsp_ids = set(pd.to_numeric(lsp_plates["PLATE_ID"], errors="coerce").dropna().astype(int))

    mark_wells  = build_mark_available_queue(all_plate_data, generated_at, exclude_plate_ids=lsp_ids)
    clean_wells = build_clean_inventory_queue(all_plate_data)
    refills     = build_refill_queue(output_df, dpart_data)
    # Dispose queue = LSP Echo plates older than ~2 months — the ones to physically
    # toss. Newer plates are still in active use, so they're not shown here.
    dispose     = [d for d in build_dispose_queue(lsp_plates, generated_at) if d["old"]]
    dispose.sort(key=lambda d: -(d["age_days"] or 0))  # oldest first

    mark_str  = ",".join(mark_wells)
    clean_str = ",".join(clean_wells)

    def _section(title, subtitle, accent):
        return (
            f'<div style="font-size:12px;font-weight:700;color:#1d1d1f;margin-top:16px;'
            f'border-left:3px solid {accent};padding-left:8px">{title}'
            f'<span style="font-size:9px;font-weight:500;color:#86868b;margin-left:8px">{subtitle}</span></div>'
        )

    # Queue 1 — Mark Available
    q1 = _section("1 · Mark Available", f"{len(mark_wells)} wells · seq-confirmed, >{MARK_AVAILABLE_VOL_MIN} µL, not yet available", "#1d4ed8")
    q1 += _copy_box("q_mark_available", mark_str) if mark_wells else \
        '<div style="font-size:10px;color:#86868b;margin-top:4px">No wells to mark available.</div>'

    # Queue 2 — Clean Inventory
    q2 = _section("2 · Clean Inventory", f"{len(clean_wells)} wells · ≤{CLEAN_INVENTORY_VOL_MAX} µL → mark unavailable", "#be185d")
    q2 += _copy_box("q_clean_inventory", clean_str) if clean_wells else \
        '<div style="font-size:10px;color:#86868b;margin-top:4px">No near-empty wells to clean.</div>'

    # Queue 3 — Refill
    q3 = _section("3 · Refill", f"{len(refills)} parts to restock", "#92400e")
    if not refills:
        q3 += '<div style="font-size:10px;color:#86868b;margin-top:4px">No parts need restocking.</div>'
    else:
        cards = []
        for i, it in enumerate(refills):
            ctl = '<span style="font-size:7px;background:#fef9c3;color:#854d0e;border:1px solid #fde68a;border-radius:3px;padding:0 3px;margin-left:4px">CTL</span>' if it["is_control"] else ''
            ab = f'<span class="badge" style="font-size:7px;background:#f0f0f2;color:#374151;border:1px solid #d1d5db;margin-left:4px">{it["antibiotic"]}</span>' if it["antibiotic"] else ''

            # Streak-out line
            streak_bits = []
            if it["glycerol_plate"] and it["glycerol_plate"] not in ("", "0", "nan"):
                streak_bits.append(f'plate <b>{it["glycerol_plate"]}</b>')
            if it["glycerol_well"]:
                streak_bits.append(f'well <b>{it["glycerol_well"]}</b>')
            if it["glycerol_location"] and it["glycerol_location"] not in ("", "nan", "None"):
                streak_bits.append(f'({it["glycerol_location"]})')
            if it["cell_strain"] and it["cell_strain"] not in ("", "nan"):
                streak_bits.append(f'· {it["cell_strain"]}')
            streak = "Streak out: " + " ".join(streak_bits) if streak_bits else \
                '<span style="color:#be185d">No glycerol source found</span>'

            body = (
                f'<div style="font-size:11px;font-weight:700;color:#1d1d1f">{it["part"]}{ctl}{ab}'
                f'<span style="font-size:9px;font-weight:500;color:#86868b;margin-left:8px">'
                f'have {it["available"]} / target {it["target"]}</span></div>'
                f'<div style="font-size:10px;color:#374151;margin-top:2px">{streak}</div>'
            )
            if it["csv_block"]:
                body += (
                    f'<div style="font-size:9px;color:#6b7280;margin-top:4px">'
                    f'PCR workorder · {it["pcr_runs"]} run{"s" if it["pcr_runs"] != 1 else ""}</div>'
                )
                body += _copy_box(f"q_refill_csv_{i}", it["csv_block"], height="72px")
            elif it["colonies"]:
                body += (
                    f'<div style="font-size:10px;color:#374151;margin-top:3px">'
                    f'~{it["colonies"]} {"colony" if it["colonies"] == 1 else "colonies"} to pick, then miniprep</div>'
                )

            cards.append(
                f'<div style="border:1px solid #e5e5e7;border-radius:6px;padding:8px 10px;'
                f'margin-top:6px;background:{"#fffdf0" if it["is_control"] else "#fff"}">{body}</div>'
            )
        q3 += "".join(cards)

    # Queue 4 — Dispose (LSP-linked Echo plates to physically discard)
    q4 = _section("4 · Dispose (LSP Echo plates &gt;2mo)",
                  f"{len(dispose)} plates older than 2 months · physically toss · wells stay unavailable", "#6b7280")
    if not dispose:
        q4 += '<div style="font-size:10px;color:#86868b;margin-top:4px">No LSP Echo plates older than 2 months.</div>'
    else:
        def _age_lbl(d):
            n = d["age_days"]
            if n is None:
                return ""
            txt = f"{n//30}mo" if n >= 30 else f"{n}d"
            if d["old"]:
                return f'<span style="font-size:9px;font-weight:700;color:#be123c;background:#fff1f2;border:1px solid #fecdd3;border-radius:5px;padding:1px 5px">⚑ {txt}</span>'
            return f'<span style="font-size:9px;color:#86868b">{txt}</span>'
        disp_rows = "".join(
            f'<tr style="{"background:#fff7f7" if d["old"] else ""}">'
            f'<td style="padding:2px 8px;font-family:monospace">{d["plate_id"]}</td>'
            f'<td style="padding:2px 8px">{d["location"]}</td>'
            f'<td style="padding:2px 8px;color:#6b7280">{d["protocol"]}</td>'
            f'<td style="padding:2px 8px;color:#86868b;white-space:nowrap">{d["created"]}</td>'
            f'<td style="padding:2px 8px;white-space:nowrap">{_age_lbl(d)}</td></tr>'
            for d in dispose
        )
        disp_copy = "\n".join(f'{d["plate_id"]}\t{d["location"]}\t{d["created"]}' for d in dispose)
        q4 += (
            '<div style="max-height:260px;overflow:auto;margin-top:4px;border:1px solid #e5e5e7;border-radius:6px">'
            '<table class="wo-table" style="width:100%;font-size:10px"><thead><tr>'
            '<th style="text-align:left;padding:2px 8px">Plate ID</th>'
            '<th style="text-align:left;padding:2px 8px">Location</th>'
            '<th style="text-align:left;padding:2px 8px">Protocol</th>'
            '<th style="text-align:left;padding:2px 8px">Created</th>'
            '<th style="text-align:left;padding:2px 8px">Age</th></tr></thead>'
            f'<tbody>{disp_rows}</tbody></table></div>'
        )
        q4 += _copy_box("q_dispose", disp_copy, height="72px")

    # Queue 5 — No Source (short on stock, no streak source & no seq-confirmed wells)
    no_source = build_no_source_queue(output_df)
    q5 = _section("5 · No Source",
                  f"{len(no_source)} parts short with no glycerol stock or seq-confirmed wells · need ordering / synthesis",
                  "#be185d")
    if not no_source:
        q5 += '<div style="font-size:10px;color:#86868b;margin-top:4px">No parts in this state.</div>'
    else:
        _ctl_badge = (' <span style="font-size:7px;background:#fef9c3;color:#854d0e;'
                      'border:1px solid #fde68a;border-radius:3px;padding:0 3px">CTL</span>')
        ns_rows = "".join(
            '<tr>'
            f'<td style="padding:2px 8px;font-family:monospace">{d["part"]}'
            f'{_ctl_badge if d["is_control"] else ""}</td>'
            f'<td style="padding:2px 8px;color:#6b7280;white-space:nowrap">{d["required"]} rxn</td></tr>'
            for d in no_source
        )
        ns_copy = ",".join(d["part"] for d in no_source)
        q5 += (
            '<div style="max-height:240px;overflow:auto;margin-top:4px;border:1px solid #e5e5e7;border-radius:6px">'
            '<table class="wo-table" style="width:100%;font-size:10px"><thead><tr>'
            '<th style="text-align:left;padding:2px 8px">Part</th>'
            '<th style="text-align:left;padding:2px 8px">Required</th></tr></thead>'
            f'<tbody>{ns_rows}</tbody></table></div>'
        )
        q5 += _copy_box("q_no_source", ns_copy, height="56px")

    return f"""
<div style="padding:12px 16px">
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;flex-wrap:wrap">
    <span style="font-size:13px;font-weight:700;color:#1d1d1f">Parts Inventory — Action Queues</span>
    <span style="font-size:9px;color:#86868b">{ts}</span>
  </div>
  {q1}
  {q2}
  {q3}
  {q4}
  {q5}
</div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Partner close-out is paused (Steve's R&D-shared annotations pending) and not rendered in the
# tab, so its ~4-minute query is skipped in the pull. Flip to True to re-enable when it resumes.
PULL_PARTNER_CLOSEOUT = False


def _query_tab_inputs(client) -> dict:
    """Live workorder queries the Parts tab needs (active builds, blocked queue, orders).
    Run here in the PULL so the dashboard render is pure pkl->HTML with zero BigQuery."""
    Q_WOD = """
SELECT COALESCE(JSON_VALUE(GG.product,'$.name'),JSON_VALUE(GIB.product,'$.name'),JSON_VALUE(PCR.product,'$.name')) AS PRODUCT,
 COALESCE(GG.parts,GIB.parts) AS parts_json, COALESCE(GG.backbone,GIB.backbone) AS backbone_json,
 PCR.templates AS pcr_templates, PCR.forward_primer AS pcr_forward_primer, PCR.reverse_primer AS pcr_reverse_primer,
 wo.type AS WT, wo.status AS ST, exp.name AS EXP
FROM bios__src.workorder wo
LEFT JOIN bios__src.goldengateworkorder GG ON GG.id=wo.id
LEFT JOIN bios__src.gibsonworkorder GIB ON GIB.id=wo.id
LEFT JOIN bios__src.pcrworkorder PCR ON PCR.id=wo.id
LEFT JOIN bios__src.assemblyplan ap ON ap.id=wo.assembly_plan_id
LEFT JOIN bios__src.experiment exp ON exp.id=ap.experiment_id
WHERE wo.status IN ('RUNNING','WAITING','READY','BLOCKED')
 AND wo.type IN ('golden_gate_workorder','gibson_workorder','pcr_workorder')
"""
    _BLK_PROD = ("COALESCE(JSON_VALUE(GG.product,'$.name'),JSON_VALUE(GIB.product,'$.name'),"
                 "JSON_VALUE(PCR.product,'$.name'),JSON_VALUE(PSY.plasmid,'$.name'),JSON_VALUE(SSY.syn_part,'$.name'))")
    _BLK_JOINS = """
 LEFT JOIN bios__src.goldengateworkorder GG ON GG.id=wo.id
 LEFT JOIN bios__src.gibsonworkorder GIB ON GIB.id=wo.id
 LEFT JOIN bios__src.pcrworkorder PCR ON PCR.id=wo.id
 LEFT JOIN bios__src.plasmidsynthesisworkorder PSY ON PSY.id=wo.id
 LEFT JOIN bios__src.synpartsynthesisworkorder SSY ON SSY.id=wo.id"""
    Q_BLK = f"""
WITH prod AS (
  SELECT wo.id, wo.type, wo.status, DATE(wo.created_at) created, wo.warnings,
    {_BLK_PROD} AS product, COALESCE(GG.parts,GIB.parts) parts, COALESCE(GG.backbone,GIB.backbone) backbone,
    COALESCE(ap.experiment_id, pr.experiment_id) eid
  FROM bios__src.workorder wo {_BLK_JOINS}
  LEFT JOIN bios__src.assemblyplan ap ON ap.id=wo.assembly_plan_id
  LEFT JOIN bios__src.plasmidrequest pr ON pr.id=wo.request_id
  WHERE wo.deleted_at IS NULL
),
succ AS (SELECT DISTINCT product, id wid FROM prod WHERE status='SUCCEEDED' AND product IS NOT NULL)
SELECT b.id wid, b.type, b.product, CAST(b.created AS STRING) created, b.warnings, b.parts, b.backbone,
  e.name experiment, ARRAY_AGG(DISTINCT s.wid IGNORE NULLS) succeeded_wos
FROM prod b LEFT JOIN bios__src.experiment e ON e.id=b.eid LEFT JOIN succ s ON s.product=b.product
WHERE b.status='BLOCKED' GROUP BY 1,2,3,4,5,6,7,8
"""
    Q_SUCC = f"SELECT DISTINCT {_BLK_PROD} p FROM bios__src.workorder wo {_BLK_JOINS} WHERE wo.status='SUCCEEDED' AND wo.deleted_at IS NULL"
    Q_ORD = """
SELECT COALESCE(JSON_VALUE(osw.oligo,'$.name'),JSON_VALUE(psw.plasmid,'$.name'),JSON_VALUE(ssw.syn_part,'$.name')) AS NAME,
  wo.status AS STATUS, COALESCE(osw.vendor,psw.vendor,ssw.vendor) AS VENDOR,
  TRIM(COALESCE(osw.vendor_order_id,psw.vendor_order_id,ssw.vendor_order_id)) AS ORDER_ID, wo.created_at AS CREATED
FROM bios__src.workorder wo
LEFT JOIN bios__src.oligosynthesisworkorder osw ON osw.id=wo.id
LEFT JOIN bios__src.plasmidsynthesisworkorder psw ON psw.id=wo.id
LEFT JOIN bios__src.synpartsynthesisworkorder ssw ON ssw.id=wo.id
WHERE wo.type IN ('oligo_synthesis_workorder','plasmid_synthesis_workorder','syn_part_synthesis_workorder')
  -- order_status() prefers ACTIVE orders and else the most recent; stale old-completed orders
  -- can't win, so keep all active (any age) + anything from the last 2y and drop the rest.
  AND (wo.status IN ('RUNNING','WAITING','READY','BLOCKED')
       OR wo.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 730 DAY))
"""
    # NGS job state per well. A refill is "in progress" until its NGS job CLOSES — the plate
    # being on an NGS protocol only means it was submitted, not that results are in. Statuses:
    # RUNNING = still sequencing, SUCCEEDED/FAILED/CANCELED = closed. NGS runs on the picked
    # samples, so only a handful of a process's wells ever carry a job.
    Q_NGS = """
SELECT nw.well_id AS WELL_ID, wo.status AS STATUS, wo.updated_at AS UPDATED
FROM bios__src.ngsworkorder nw
JOIN bios__src.workorder wo ON wo.id = nw.id
WHERE wo.deleted_at IS NULL AND nw.well_id IS NOT NULL
"""
    # PCR workorder history per product. A dPart is made in-house by PCR — there is no dPart
    # synthesis workorder type — so "is a PCR queued, and how did the last one go" is the dPart
    # equivalent of the vendor-order question Q_ORD answers for plasmids/oligos/synparts. Q_WOD
    # only carries OPEN workorders, so closed outcomes (a PCR that FAILED yesterday) need this.
    Q_PCR = """
SELECT JSON_VALUE(PCR.product,'$.name') AS NAME, wo.status AS STATUS,
       wo.created_at AS CREATED, wo.updated_at AS UPDATED
FROM bios__src.workorder wo
JOIN bios__src.pcrworkorder PCR ON PCR.id = wo.id
WHERE wo.deleted_at IS NULL
  AND (wo.status IN ('RUNNING','WAITING','READY','BLOCKED')
       OR wo.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 730 DAY))
"""
    wod = client.query(Q_WOD).to_dataframe()
    blk = client.query(Q_BLK).to_dataframe()
    succ_names = set(client.query(Q_SUCC).to_dataframe()["p"].dropna())
    ordf = client.query(Q_ORD).to_dataframe()
    ngs = client.query(Q_NGS).to_dataframe()
    pcr = client.query(Q_PCR).to_dataframe()
    return {"wod": wod, "blk": blk, "blk_succ_names": succ_names, "ord": ordf,
            "ngs": ngs, "pcr": pcr}


def run_parts_inventory() -> dict:
    """
    Run the full pipeline and return everything the three action queues need:
        {parts, all_plate_data, dpart_data, generated_at}
    `parts` is the action-item DataFrame (back-compatible content).
    """
    client = bigquery.Client(project=PROJECT)
    now = dt.datetime.now(tz=dt.timezone.utc)

    # ── per-step profiling (like the dashboard) so slow steps are visible ──────
    import time as _time
    _t = [_time.perf_counter()]; _t0 = _t[0]
    def _lap(label):
        _n = _time.perf_counter(); print(f"  ⏱  {label}: {_n - _t[0]:.1f}s"); _t[0] = _n

    print("Loading data from BigQuery ...")
    all_plate_data, workorder_data, dpart_data = load_data(client)
    print(f"  {len(all_plate_data):,} well rows | {len(workorder_data):,} workorders | {len(dpart_data):,} dparts")
    _lap("load BigQuery (plate inventory + workorders + dparts)")

    print("Extracting required parts from workorders ...")
    raw_parts = extract_required_parts(workorder_data)
    ctrl_count = raw_parts["Is_Control"].sum() if "Is_Control" in raw_parts.columns else 0
    print(f"  {len(raw_parts)} unique parts required ({ctrl_count} controls seeded)")

    parts_list = raw_parts.rename(columns={"New Parts": "Part"})

    print("Computing inventory ...")
    parts_list = run_optimized_lab_workflow(parts_list, all_plate_data, dpart_data, now)
    _lap("compute inventory")

    print("Classifying actions ...")
    parts_list = classify_actions(parts_list, workorder_data, all_plate_data, now)
    _lap("classify actions")

    # In-flight builds: parts whose OWN assembly workorder is active (BLOCKED/READY/RUNNING/
    # WAITING) — net-new product being made that feeds downstream requests. build_output drops
    # READY/RUNNING/WAITING as "already handled", so capture these BEFORE that filter for the
    # "New builds — feed into requests" view (which wants everything in flight, not just blocked).
    _act = parts_list.get("Actions Required", pd.Series("", index=parts_list.index)).astype(str)
    builds = parts_list[_act.str.contains("workorder", case=False)].copy()
    builds = builds.rename(columns={"Actions Required": "Action Suggested"}).astype(object).fillna("")

    output = build_output(parts_list)
    ctrl_actions = output["Is_Control"].sum() if "Is_Control" in output.columns else 0
    print(f"  {len(output)} parts need attention ({ctrl_actions} controls) · {len(builds)} in-flight builds")

    print("Finding LSP-linked Echo plates (for disposal) ...")
    lsp_plates = client.query(_query_lsp_echo_plates()).to_dataframe()
    print(f"  {len(lsp_plates)} LSP-linked 384 Echo plates")
    _lap("query LSP Echo plates")

    # Partner close-out is PAUSED (not rendered — SHOW_CLOSEOUT=False in the tab), yet this query
    # was ~50% of the whole pull (~4 min). Skip it while paused; flip PULL_PARTNER_CLOSEOUT=True
    # to re-enable when the feature resumes.
    if PULL_PARTNER_CLOSEOUT:
        print("Finding partner-project close-out products ...")
        partner_closeout = client.query(_query_partner_closeout_products()).to_dataframe()
        print(f"  {partner_closeout['eid'].nunique() if len(partner_closeout) else 0} inactive partner projects "
              f"with {len(partner_closeout)} retirable products")
    else:
        partner_closeout = pd.DataFrame()
        print("Skipping partner close-out query (paused / not rendered) — saves ~4 min")
    _lap("query partner close-out")

    print("Fetching Parts-tab render inputs (active WOs, blocked queue, orders) ...")
    _tab = _query_tab_inputs(client)
    print(f"  {len(_tab['wod'])} active WOs · {len(_tab['blk'])} blocked · {len(_tab['ord'])} orders")
    _lap("query tab inputs (active WOs / blocked / orders)")
    print(f"  ⏱  TOTAL run_parts_inventory: {_time.perf_counter() - _t0:.1f}s")

    return {
        "parts": output,
        "builds": builds,
        "all_plate_data": all_plate_data,
        "dpart_data": dpart_data,
        "lsp_plates": lsp_plates,
        "partner_closeout": partner_closeout,
        "wod_df": _tab["wod"],
        "blk_df": _tab["blk"],
        "blk_succ_names": _tab["blk_succ_names"],
        "ord_df": _tab["ord"],
        "ngs_df": _tab["ngs"],
        "pcr_df": _tab["pcr"],
        "generated_at": now,
    }


def main():
    parser = argparse.ArgumentParser(description="Parts Inventory Tool")
    parser.add_argument(
        "--output", "-o",
        default=f"parts_inventory_{dt.date.today()}.csv",
        help="Parts CSV path (default: parts_inventory_YYYY-MM-DD.csv)",
    )
    parser.add_argument(
        "--html", default=None,
        help="Optional path to write the action-queues HTML fragment",
    )
    args = parser.parse_args()

    result = run_parts_inventory()
    parts = result["parts"]
    all_plate_data = result["all_plate_data"]
    dpart_data = result["dpart_data"]
    lsp_plates = result.get("lsp_plates")

    lsp_ids = set()
    if lsp_plates is not None and not lsp_plates.empty and "PLATE_ID" in lsp_plates.columns:
        lsp_ids = set(pd.to_numeric(lsp_plates["PLATE_ID"], errors="coerce").dropna().astype(int))

    # Build the action queues (LSP-linked plates excluded from Mark Available)
    mark_wells  = build_mark_available_queue(all_plate_data, result["generated_at"], exclude_plate_ids=lsp_ids)
    clean_wells = build_clean_inventory_queue(all_plate_data)
    refills     = build_refill_queue(parts, dpart_data)
    dispose     = [d for d in build_dispose_queue(lsp_plates, result["generated_at"]) if d["old"]]
    dispose.sort(key=lambda d: -(d["age_days"] or 0))  # oldest first

    # --- CLI: print copy-paste strings ---
    print("\n" + "=" * 70)
    print(f"1 · MARK AVAILABLE  ({len(mark_wells)} wells · seq-confirmed, >{MARK_AVAILABLE_VOL_MIN} µL, not available)")
    print("=" * 70)
    print(",".join(mark_wells) if mark_wells else "(none)")

    print("\n" + "=" * 70)
    print(f"2 · CLEAN INVENTORY ({len(clean_wells)} wells · ≤{CLEAN_INVENTORY_VOL_MAX} µL → mark unavailable)")
    print("=" * 70)
    print(",".join(clean_wells) if clean_wells else "(none)")

    print("\n" + "=" * 70)
    print(f"3 · REFILL  ({len(refills)} parts to restock)")
    print("=" * 70)
    for it in refills:
        ctl = " [CTL]" if it["is_control"] else ""
        gly = " ".join(p for p in [
            f'plate {it["glycerol_plate"]}' if it["glycerol_plate"] not in ("", "0", "nan") else "",
            f'well {it["glycerol_well"]}' if it["glycerol_well"] else "",
            f'({it["glycerol_location"]})' if it["glycerol_location"] not in ("", "nan", "None") else "",
        ] if p) or "no glycerol source"
        print(f'\n{it["part"]}{ctl}  have {it["available"]}/{it["target"]}  · streak: {gly}')
        if it["csv_block"]:
            print(f'  PCR workorder ({it["pcr_runs"]} run{"s" if it["pcr_runs"] != 1 else ""}):')
            for line in it["csv_block"].splitlines():
                print(f"    {line}")
        elif it["colonies"]:
            print(f'  ~{it["colonies"]} colonies to pick, then miniprep')

    print("\n" + "=" * 70)
    print(f"4 · DISPOSE — LSP Echo plates >2mo ({len(dispose)} plates · wells stay unavailable)")
    print("=" * 70)
    print(f'{"PLATE_ID":<10}{"AGE":<8}{"CREATED":<12}LOCATION')
    for d in dispose:
        age = f'{(d["age_days"] or 0)//30}mo'
        print(f'{str(d["plate_id"]):<10}{age:<8}{d["created"]:<12}{d["location"]}')
    if not dispose:
        print("(none)")

    # --- Parts CSV (still written for record-keeping) ---
    if not parts.empty:
        parts.to_csv(args.output, index=False)
        print(f"\nParts detail saved to {args.output}")

    # --- Optional HTML fragment ---
    if args.html:
        html = render_action_queues_html(
            parts, all_plate_data, dpart_data, lsp_plates=lsp_plates,
            generated_at=result["generated_at"],
        )
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"HTML fragment saved to {args.html}")


if __name__ == "__main__":
    main()

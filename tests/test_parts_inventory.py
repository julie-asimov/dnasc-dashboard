"""
Tests for parts_inventory.py — all pure-Python logic; no BigQuery calls.

Run: /opt/anaconda3/bin/python3 -m pytest tests/test_parts_inventory.py -v
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from parts_inventory import (
    _action_badge,
    _pcr_csv_block,
    build_clean_inventory_queue,
    build_dispose_queue,
    build_mark_available_queue,
    build_output,
    build_refill_queue,
    classify_actions,
    extract_required_parts,
    render_action_queues_html,
    render_parts_inventory_html,
    run_optimized_lab_workflow,
    CONTROL_BUFFER_RXNS,
    MARK_AVAILABLE_VOL_MIN,
    CLEAN_INVENTORY_VOL_MAX,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
FRESH_DATE = pd.Timestamp("2025-10-01", tz="UTC")   # within 200 days of NOW
OLD_DATE   = pd.Timestamp("2024-01-01", tz="UTC")   # older than 200 days


def _echo_well(stock_id, vol=100.0, conc=100.0, seq_len=3000, available="True",
               seq_confirmed="False", location="Bench", created_at=FRESH_DATE,
               dpart_seq_len=None, plate_id="P1", well_id="W1"):
    return {
        "WELL_ID": well_id,
        "PLATE_ID": plate_id,
        "STOCK_ID": stock_id,
        "LABWARE": "384 Echo Source Plate",
        "AVAILABLE": available,
        "SEQ_CONFIRMED": seq_confirmed,
        "VOLUME_UL": vol,
        "CONCENTRATION_NGUL": conc,
        "SEQUENCE_LENGTH": seq_len,
        "DPART_SEQUENCE_LENGTH": dpart_seq_len if dpart_seq_len is not None else np.nan,
        "PLATE_LOCATION_BOX": location,
        "CREATED_AT": created_at,
        "WELL_TYPE": "Stock",
        "PLATE_PROTOCOL": "",
        "WELL_NUMBER": "1",
        "PLATE_NUMBER_OF_WELLS": "96",
        "ANTI_KAN": "False",
        "ANTI_SPEC": "False",
        "ANTI_CARB": "False",
        "COMP_CELL": "",
        "COLONY": "col1",
        "COMMENTS": "",
    }


_PLATE_COLUMNS = [
    "WELL_ID", "PLATE_ID", "STOCK_ID", "LABWARE", "AVAILABLE", "SEQ_CONFIRMED",
    "VOLUME_UL", "CONCENTRATION_NGUL", "SEQUENCE_LENGTH", "DPART_SEQUENCE_LENGTH",
    "PLATE_LOCATION_BOX", "CREATED_AT", "WELL_TYPE", "PLATE_PROTOCOL",
    "WELL_NUMBER", "PLATE_NUMBER_OF_WELLS", "ANTI_KAN", "ANTI_SPEC", "ANTI_CARB",
    "COMP_CELL", "COLONY", "COMMENTS",
]


def _make_plate_data(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_PLATE_COLUMNS)
    return pd.DataFrame(rows)


def _workorder(stock_id, wo_type, status, parts_json=None, backbone_json=None,
               pcr_templates=None, pcr_forward_primer=None, pcr_reverse_primer=None):
    return {
        "STOCK_ID": stock_id,
        "WORKORDER_TYPE": wo_type,
        "STATUS": status,
        "parts_json": parts_json,
        "backbone_json": backbone_json,
        "pcr_templates": pcr_templates,
        "pcr_forward_primer": pcr_forward_primer,
        "pcr_reverse_primer": pcr_reverse_primer,
    }


# ---------------------------------------------------------------------------
# extract_required_parts
# ---------------------------------------------------------------------------

class TestExtractRequiredParts:

    def test_gg_parts_and_backbone_extracted(self):
        parts = json.dumps([{"name": "pAI-100"}, {"name": "pAI-200"}])
        backbone = json.dumps({"name": "pAI-300"})
        wos = pd.DataFrame([
            _workorder("pAI-999", "golden_gate_workorder", "WAITING",
                       parts_json=parts, backbone_json=backbone)
        ])
        df = extract_required_parts(wos)
        parts = set(df["New Parts"])
        assert "pAI-100" in parts
        assert "pAI-200" in parts
        assert "pAI-300" in parts

    def test_gibson_parts_and_backbone_extracted(self):
        parts = json.dumps([{"name": "pAI-101"}])
        backbone = json.dumps({"name": "pAI-301"})
        wos = pd.DataFrame([
            _workorder("pAI-999", "gibson_workorder", "READY",
                       parts_json=parts, backbone_json=backbone)
        ])
        df = extract_required_parts(wos)
        assert "pAI-101" in set(df["New Parts"])
        assert "pAI-301" in set(df["New Parts"])

    def test_pcr_template_added(self):
        wos = pd.DataFrame([
            _workorder("d5001", "pcr_workorder", "WAITING",
                       pcr_templates=json.dumps([{"name": "pAI-400", "available": True}]))
        ])
        df = extract_required_parts(wos)
        assert "pAI-400" in set(df["New Parts"])

    def test_pcr_template_only_from_waiting_ready(self):
        wos = pd.DataFrame([
            _workorder("d5001", "pcr_workorder", "SUCCEEDED",
                       pcr_templates=json.dumps([{"name": "pAI-400", "available": True}])),
        ])
        # SUCCEEDED PCR still appears in workorder_data in notebook via the cell-8 loop
        # but dpart_page only picks up WAITING/READY — both code paths tested here
        df = extract_required_parts(wos)
        # pAI-400 appears via the general PCR-workorder pass (cell 8 equivalent)
        assert "pAI-400" in set(df["New Parts"])

    def test_reactions_required_counts_demand(self):
        parts = json.dumps([{"name": "pAI-100"}, {"name": "pAI-100"}])
        backbone = json.dumps({"name": "pAI-100"})
        wos = pd.DataFrame([
            _workorder("pAI-999", "golden_gate_workorder", "WAITING",
                       parts_json=parts, backbone_json=backbone)
        ])
        df = extract_required_parts(wos)
        row = df[df["New Parts"] == "pAI-100"].iloc[0]
        assert row["Reactions Required"] == 3  # appears 3 times

    def test_uuid_and_non_pai_d_parts_excluded(self):
        parts = json.dumps([
            {"name": "pAI-100"},
            {"name": "550e8400-e29b-41d4-a716-446655440000"},  # UUID — excluded
            {"name": "synth_xyz"},                              # not pAI/d/o — excluded
        ])
        backbone = json.dumps({"name": "pAI-300"})
        wos = pd.DataFrame([
            _workorder("pAI-999", "golden_gate_workorder", "WAITING",
                       parts_json=parts, backbone_json=backbone)
        ])
        df = extract_required_parts(wos)
        assert "pAI-100" in set(df["New Parts"])
        assert "pAI-300" in set(df["New Parts"])
        for part in df["New Parts"]:
            assert not (len(part) >= 36), f"UUID-length part leaked: {part}"
            assert str(part).startswith(("pAI", "d", "o")), f"Non-pAI/d/o part leaked: {part}"

    def test_d_prefix_parts_included(self):
        parts = json.dumps([{"name": "d5650"}])
        backbone = json.dumps({"name": "pAI-300"})
        wos = pd.DataFrame([
            _workorder("pAI-999", "golden_gate_workorder", "WAITING",
                       parts_json=parts, backbone_json=backbone)
        ])
        df = extract_required_parts(wos)
        assert "d5650" in set(df["New Parts"])

    def test_empty_workorders_returns_only_seeded_controls(self):
        # With no workorders, the report still contains the always-stocked control
        # parts (seeded at CONTROL_BUFFER_RXNS), and nothing else.
        from parts_inventory import CONTROL_PARTS, CONTROL_BUFFER_RXNS
        wos = pd.DataFrame(columns=["STOCK_ID", "WORKORDER_TYPE", "STATUS", "parts_json",
                                    "backbone_json", "pcr_templates",
                                    "pcr_forward_primer", "pcr_reverse_primer"])
        df = extract_required_parts(wos)
        assert set(df["New Parts"]) == set(CONTROL_PARTS)
        assert df["Is_Control"].all()
        assert (df["Reactions Required"] == CONTROL_BUFFER_RXNS).all()

    def test_pAI_before_d_in_output(self):
        parts = json.dumps([{"name": "d5650"}, {"name": "pAI-100"}])
        backbone = json.dumps({"name": "pAI-200"})
        wos = pd.DataFrame([
            _workorder("pAI-999", "golden_gate_workorder", "WAITING",
                       parts_json=parts, backbone_json=backbone)
        ])
        df = extract_required_parts(wos)
        pai_idx = df.index[df["New Parts"].str.startswith("pAI")].tolist()
        d_idx   = df.index[df["New Parts"].str.startswith("d")].tolist()
        if pai_idx and d_idx:
            assert max(pai_idx) < min(d_idx)

    def test_malformed_json_does_not_crash(self):
        wos = pd.DataFrame([
            _workorder("pAI-999", "golden_gate_workorder", "WAITING",
                       parts_json="NOT_JSON", backbone_json="ALSO_NOT_JSON")
        ])
        df = extract_required_parts(wos)  # must not raise
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# run_optimized_lab_workflow
# ---------------------------------------------------------------------------

class TestRunOptimizedLabWorkflow:

    def _base_parts(self, part="pAI-100", req=5):
        return pd.DataFrame([{"Part": part, "Reactions Required": req}])

    def _dpart_data(self):
        return pd.DataFrame(columns=["DPART_NAME", "DPART_TEMPLATE", "DPART_SEQUENCE_LENGTH",
                                     "OLIGO_1", "OLIGO_1_SEQUENCE_LENGTH",
                                     "OLIGO_2", "OLIGO_2_SEQUENCE_LENGTH"])

    def test_fresh_echo_plate_counts_toward_available(self):
        plate_data = _make_plate_data([
            _echo_well("pAI-100", vol=100, conc=100, seq_len=3000, available="True",
                       created_at=FRESH_DATE)
        ])
        result = run_optimized_lab_workflow(
            self._base_parts(), plate_data, self._dpart_data(), NOW
        )
        # (100 - 20) * 100 / (1e-12 * 3000 * 6e9) = 8000/18000 ≈ 4444 reactions
        assert result.loc[0, "Reactions Available"] > 0

    def test_old_echo_plate_not_fresh_goes_to_old(self):
        plate_data = _make_plate_data([
            _echo_well("pAI-100", vol=100, conc=100, seq_len=3000, available="True",
                       created_at=OLD_DATE, location="Cold room")
        ])
        result = run_optimized_lab_workflow(
            self._base_parts(), plate_data, self._dpart_data(), NOW
        )
        assert result.loc[0, "Reactions Available"] == 0
        assert result.loc[0, "Reactions Available Old"] > 0

    def test_deli_left_always_fresh(self):
        plate_data = _make_plate_data([
            _echo_well("pAI-100", vol=100, conc=100, seq_len=3000, available="True",
                       created_at=OLD_DATE, location="Deli Left (DVs)")
        ])
        result = run_optimized_lab_workflow(
            self._base_parts(), plate_data, self._dpart_data(), NOW
        )
        assert result.loc[0, "Reactions Available"] > 0

    def test_unavailable_seq_confirmed_goes_to_confirmed(self):
        plate_data = _make_plate_data([
            _echo_well("pAI-100", vol=50, conc=100, seq_len=3000,
                       available="False", seq_confirmed="True",
                       created_at=FRESH_DATE)
        ])
        result = run_optimized_lab_workflow(
            self._base_parts(), plate_data, self._dpart_data(), NOW
        )
        assert result.loc[0, "Reactions Seq Confirmed"] > 0
        assert result.loc[0, "Reactions Available"] == 0

    def test_dpart_template_expansion_when_stock_short(self):
        dpart_data = pd.DataFrame([{
            "DPART_NAME": "d100",
            "DPART_TEMPLATE": "pAI-500",
            "DPART_SEQUENCE_LENGTH": 500,
            "OLIGO_1": "o1", "OLIGO_1_SEQUENCE_LENGTH": 30,
            "OLIGO_2": "o2", "OLIGO_2_SEQUENCE_LENGTH": 30,
        }])
        # d100 needs 10 reactions; no stock available → template should be added
        parts = pd.DataFrame([{"Part": "d100", "Reactions Required": 10}])
        plate_data = _make_plate_data([])  # empty inventory
        result = run_optimized_lab_workflow(parts, plate_data, dpart_data, NOW)
        assert "pAI-500" in result["Part"].values
        template_row = result[result["Part"] == "pAI-500"].iloc[0]
        assert template_row["Reactions Required"] > 0

    def test_dpart_template_not_expanded_when_stock_sufficient(self):
        dpart_data = pd.DataFrame([{
            "DPART_NAME": "d100",
            "DPART_TEMPLATE": "pAI-500",
            "DPART_SEQUENCE_LENGTH": 500,
            "OLIGO_1": "o1", "OLIGO_1_SEQUENCE_LENGTH": 30,
            "OLIGO_2": "o2", "OLIGO_2_SEQUENCE_LENGTH": 30,
        }])
        parts = pd.DataFrame([{"Part": "d100", "Reactions Required": 2}])
        # Sufficient dpart stock (> 2 reactions)
        plate_data = _make_plate_data([
            _echo_well("d100", vol=200, conc=200, seq_len=500,
                       dpart_seq_len=500, available="True", created_at=FRESH_DATE)
        ])
        result = run_optimized_lab_workflow(parts, plate_data, dpart_data, NOW)
        # pAI-500 should not appear (or have 0 required reactions)
        if "pAI-500" in result["Part"].values:
            assert result[result["Part"] == "pAI-500"].iloc[0]["Reactions Required"] == 0

    def test_dpart_uses_dpart_sequence_length_for_rxn_calc(self):
        plate_data = _make_plate_data([
            _echo_well("d100", vol=100, conc=100, seq_len=9999,  # SEQUENCE_LENGTH should be ignored
                       dpart_seq_len=500, available="True", created_at=FRESH_DATE)
        ])
        parts = pd.DataFrame([{"Part": "d100", "Reactions Required": 5}])
        result = run_optimized_lab_workflow(parts, plate_data, self._dpart_data(), NOW)
        rxns = result.loc[0, "Reactions Available"]
        # (100-20)*100 / (1e-12*500*6e9) ≈ 2666 rxns when using DPART_SEQUENCE_LENGTH=500
        # (100-20)*100 / (1e-12*9999*6e9) ≈  133 rxns if incorrectly using SEQUENCE_LENGTH=9999
        assert rxns > 500, f"Expected dpart seq_len=500, got {rxns} reactions (should be ~2666)"

    def test_old_plates_string_format(self):
        plate_data = _make_plate_data([
            _echo_well("pAI-100", vol=100, conc=100, seq_len=3000,
                       available="True", created_at=OLD_DATE,
                       location="Cold room", plate_id="P99")
        ])
        result = run_optimized_lab_workflow(
            self._base_parts(), plate_data, self._dpart_data(), NOW
        )
        assert "P99" in str(result.loc[0, "Old Plates"])

    def test_micronic_tubes_counted(self):
        tube_row = _echo_well("pAI-100", seq_len=3000)
        tube_row["LABWARE"] = "Micronic Tube Rack"
        tube_row["SEQ_CONFIRMED"] = "True"
        tube_row["WELL_TYPE"] = "Stock"
        tube_row["PLATE_LOCATION_BOX"] = "None"
        plate_data = _make_plate_data([tube_row])
        result = run_optimized_lab_workflow(
            self._base_parts(), plate_data, self._dpart_data(), NOW
        )
        assert result.loc[0, "Micronic Tubes"] == "(1,0)"

    def test_glycerol_well_info_populated(self):
        gly_row = _echo_well("pAI-100")
        gly_row["LABWARE"] = "Thermo V Bottom Plate"
        gly_row["AVAILABLE"] = "True"
        gly_row["PLATE_LOCATION_BOX"] = "Box A"
        gly_row["WELL_NUMBER"] = "1"
        gly_row["WELL_ID"] = "42"
        gly_row["COMP_CELL"] = "DH10B"
        plate_data = _make_plate_data([gly_row])
        result = run_optimized_lab_workflow(
            self._base_parts(), plate_data, self._dpart_data(), NOW
        )
        assert result.loc[0, "Glycerol Location"] == "Box A"
        assert result.loc[0, "Glycerol Well"] == "A1"
        assert result.loc[0, "Glycerol Well ID"] == "well42"

    def test_antibiotic_kan(self):
        gly_row = _echo_well("pAI-100")
        gly_row["LABWARE"] = "Eppendorf V Microplate"
        gly_row["AVAILABLE"] = "True"
        gly_row["ANTI_KAN"] = "True"
        gly_row["COMP_CELL"] = "DH10B"
        plate_data = _make_plate_data([gly_row])
        result = run_optimized_lab_workflow(
            self._base_parts(), plate_data, self._dpart_data(), NOW
        )
        assert result.loc[0, "Antibiotic"] == "Kan"

    def test_no_duplicates_in_output(self):
        parts = pd.DataFrame([
            {"Part": "pAI-100", "Reactions Required": 5},
            {"Part": "pAI-100", "Reactions Required": 5},  # duplicate
        ])
        result = run_optimized_lab_workflow(
            parts, _make_plate_data([]), self._dpart_data(), NOW
        )
        assert result["Part"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# classify_actions
# ---------------------------------------------------------------------------

class TestClassifyActions:

    def _empty_plate(self):
        return _make_plate_data([])

    def _parts_short(self, part="pAI-100", req=50, avail=0, confirmed=0, gly_well_id=""):
        return pd.DataFrame([{
            "Part": part,
            "Reactions Required": req,
            "Reactions Available": avail,
            "Reactions Seq Confirmed": confirmed,
            "Glycerol Well ID": gly_well_id,
            "Micronic Tubes": "(0,0)",
        }])

    def test_sufficient_stock_no_action(self):
        parts = pd.DataFrame([{
            "Part": "pAI-100",
            "Reactions Required": 5,
            "Reactions Available": 100,
            "Reactions Seq Confirmed": 0,
            "Glycerol Well ID": "",
            "Micronic Tubes": "(0,0)",
        }])
        wos = pd.DataFrame(columns=["STOCK_ID", "WORKORDER_TYPE", "STATUS"])
        result = classify_actions(parts, wos, self._empty_plate(), NOW)
        assert result.loc[0, "Actions Required"] == ""

    def test_existing_workorder_reported(self):
        parts = self._parts_short()
        wos = pd.DataFrame([
            _workorder("pAI-100", "pcr_workorder", "WAITING")
        ])
        result = classify_actions(parts, wos, self._empty_plate(), NOW)
        assert "pcr_workorder" in result.loc[0, "Actions Required"]
        assert "WAITING" in result.loc[0, "Actions Required"]

    def test_blocked_workorder_prefers_non_blocked(self):
        parts = self._parts_short()
        wos = pd.DataFrame([
            _workorder("pAI-100", "pcr_workorder", "BLOCKED"),
            _workorder("pAI-100", "lsp_workorder", "WAITING"),
        ])
        result = classify_actions(parts, wos, self._empty_plate(), NOW)
        assert "WAITING" in result.loc[0, "Actions Required"]

    def test_blocked_only_uses_blocked(self):
        parts = self._parts_short()
        wos = pd.DataFrame([
            _workorder("pAI-100", "pcr_workorder", "BLOCKED"),
        ])
        result = classify_actions(parts, wos, self._empty_plate(), NOW)
        assert "BLOCKED" in result.loc[0, "Actions Required"]

    def test_mark_available_when_seq_confirmed_wells_exist(self):
        parts = self._parts_short(req=5, avail=0, confirmed=100)
        conf_well = _echo_well("pAI-100", vol=50, conc=50, seq_len=3000,
                               available="False", seq_confirmed="True",
                               created_at=FRESH_DATE, well_id="99")
        plate_data = _make_plate_data([conf_well])
        wos = pd.DataFrame(columns=["STOCK_ID", "WORKORDER_TYPE", "STATUS"])
        result = classify_actions(parts, wos, plate_data, NOW)
        assert result.loc[0, "Actions Required"].startswith("Mark seq confirmed wells available")
        assert "well99" in result.loc[0, "Actions Required"]

    def test_refill_when_glycerol_exists_no_stock(self):
        parts = self._parts_short(gly_well_id="well42")
        wos = pd.DataFrame(columns=["STOCK_ID", "WORKORDER_TYPE", "STATUS"])
        result = classify_actions(parts, wos, self._empty_plate(), NOW)
        assert result.loc[0, "Actions Required"] == "Refill"

    def test_true_when_no_source(self):
        parts = self._parts_short(gly_well_id="")
        wos = pd.DataFrame(columns=["STOCK_ID", "WORKORDER_TYPE", "STATUS"])
        result = classify_actions(parts, wos, self._empty_plate(), NOW)
        assert result.loc[0, "Actions Required"] == "True"

    def test_within_10_rxn_buffer_no_action(self):
        parts = pd.DataFrame([{
            "Part": "pAI-100",
            "Reactions Required": 50,
            "Reactions Available": 50,   # req + 10 = 60 > 50, so should trigger
            "Reactions Seq Confirmed": 0,
            "Glycerol Well ID": "",
            "Micronic Tubes": "(0,0)",
        }])
        wos = pd.DataFrame(columns=["STOCK_ID", "WORKORDER_TYPE", "STATUS"])
        result = classify_actions(parts, wos, self._empty_plate(), NOW)
        # req + 10 (50+10=60) > avail (50) → action IS triggered
        assert result.loc[0, "Actions Required"] != ""

    def test_surplus_beyond_buffer_no_action(self):
        # buffer = max(10, req) = max(10, 40) = 40 → needed = 80
        parts = pd.DataFrame([{
            "Part": "pAI-100",
            "Reactions Required": 40,
            "Reactions Available": 80,   # 40 + max(10,40) = 80 <= 80
            "Reactions Seq Confirmed": 0,
            "Glycerol Well ID": "",
            "Micronic Tubes": "(0,0)",
        }])
        wos = pd.DataFrame(columns=["STOCK_ID", "WORKORDER_TYPE", "STATUS"])
        result = classify_actions(parts, wos, self._empty_plate(), NOW)
        assert result.loc[0, "Actions Required"] == ""


# ---------------------------------------------------------------------------
# build_output
# ---------------------------------------------------------------------------

class TestBuildOutput:

    def _parts(self, rows):
        return pd.DataFrame(rows)

    def test_removes_empty_action(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5, "Actions Required": ""},
            {"Part": "pAI-200", "Reactions Required": 5, "Actions Required": "Refill"},
        ])
        out = build_output(df)
        assert "pAI-100" not in out["Part"].values
        assert "pAI-200" in out["Part"].values

    def test_removes_zero_required(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 0, "Actions Required": "Refill"},
        ])
        out = build_output(df)
        assert len(out) == 0

    def test_excludes_lsp_actions(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5, "Actions Required": "lsp_workorder is RUNNING"},
        ])
        out = build_output(df)
        assert len(out) == 0

    def test_excludes_running_status(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5, "Actions Required": "pcr_workorder is RUNNING"},
        ])
        out = build_output(df)
        assert len(out) == 0

    def test_excludes_waiting_status(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5, "Actions Required": "pcr_workorder is WAITING"},
        ])
        out = build_output(df)
        assert len(out) == 0

    def test_excludes_ready_status(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5, "Actions Required": "pcr_workorder is READY"},
        ])
        out = build_output(df)
        assert len(out) == 0

    def test_keeps_refill(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5, "Actions Required": "Refill"},
        ])
        out = build_output(df)
        assert len(out) == 1
        assert out.iloc[0]["Action Suggested"] == "Refill"

    def test_keeps_mark_available(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5,
             "Actions Required": "Mark seq confirmed wells available ['well99']"},
        ])
        out = build_output(df)
        assert len(out) == 1

    def test_column_renamed_to_action_suggested(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5, "Actions Required": "Refill"},
        ])
        out = build_output(df)
        assert "Action Suggested" in out.columns
        assert "Actions Required" not in out.columns

    def test_blocked_workorder_kept_in_output(self):
        df = self._parts([
            {"Part": "pAI-100", "Reactions Required": 5, "Actions Required": "pcr_workorder is BLOCKED"},
        ])
        out = build_output(df)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# render_parts_inventory_html
# ---------------------------------------------------------------------------

class TestRenderPartsInventoryHtml:

    def _sample_df(self):
        return pd.DataFrame([{
            "Part": "pAI-100",
            "Reactions Required": 10,
            "Reactions Available": 2,
            "Reactions Available Old": 0,
            "Reactions Seq Confirmed": 0,
            "Micronic Tubes": "(1,0)",
            "Glycerol Plate": "P5",
            "Glycerol Location": "Box A",
            "Antibiotic": "Kan",
            "Glycerol Well": "A1",
            "Glycerol Well ID": "well42",
            "dPart Template": "",
            "Old Plates": "",
            "Action Suggested": "Refill",
        }])

    def test_returns_string(self):
        html = render_parts_inventory_html(self._sample_df())
        assert isinstance(html, str)

    def test_contains_part_name(self):
        html = render_parts_inventory_html(self._sample_df())
        assert "pAI-100" in html

    def test_empty_df_returns_no_action_message(self):
        html = render_parts_inventory_html(pd.DataFrame())
        assert "No parts require action" in html

    def test_refill_badge_in_html(self):
        html = render_parts_inventory_html(self._sample_df())
        assert "Refill" in html

    def test_summary_counts_in_html(self):
        html = render_parts_inventory_html(self._sample_df())
        assert "1 parts need action" in html

    def test_uses_wo_table_class(self):
        html = render_parts_inventory_html(self._sample_df())
        assert 'class="wo-table"' in html


# ---------------------------------------------------------------------------
# _action_badge
# ---------------------------------------------------------------------------

class TestActionBadge:

    def test_refill_badge(self):
        html = _action_badge("Refill")
        assert "Refill" in html
        assert "badge" in html

    def test_mark_available_badge(self):
        html = _action_badge("Mark seq confirmed wells available ['well1']")
        assert "Mark Available" in html

    def test_true_no_source_badge(self):
        html = _action_badge("True")
        assert "No Source" in html

    def test_unknown_action_uses_raw_text(self):
        html = _action_badge("pcr_workorder is BLOCKED")
        assert "pcr_workorder" in html


# ---------------------------------------------------------------------------
# Action queues (reframed output)
# ---------------------------------------------------------------------------

# Distinct helper (do NOT shadow the module-level _echo_well used by other tests).
# Defaults make a "live" well: fresh plate (relative to NOW), real location, ok conc.
def _q_well(well_id, vol, available, seq_confirmed=True, labware="384 Echo Source Plate",
            conc=100.0, location="4B-ECHO1", created_at=pd.Timestamp("2025-12-01", tz="UTC"),
            plate_id=1):
    return {
        "WELL_ID": well_id,
        "PLATE_ID": plate_id,
        "LABWARE": labware,
        "VOLUME_UL": vol,
        "AVAILABLE": "True" if available else "False",
        "SEQ_CONFIRMED": "True" if seq_confirmed else "False",
        "CONCENTRATION_NGUL": conc,
        "PLATE_LOCATION_BOX": location,
        "CREATED_AT": created_at,
    }


def _refill_row(**kw):
    base = {
        "Part": "d3550", "Is_Control": True, "Action Suggested": "Refill",
        "Reactions Available": 10, "Reactions Required": 96,
        "Antibiotic": "Kan", "Cell Strain": "DH5a", "Glycerol Plate": "555",
        "Glycerol Well": "A1", "Glycerol Location": "Box1", "PCR Runs Needed": 9,
        "SEQUENCE_LENGTH": None,
    }
    base.update(kw)
    return base


class TestMarkAvailableQueue:
    def test_includes_live_seq_confirmed_unavailable_high_volume(self):
        df = pd.DataFrame([
            _q_well(101, 40, available=False, seq_confirmed=True),   # ✓
            _q_well(102, 26, available=False, seq_confirmed=True),   # ✓ (>25)
        ])
        assert build_mark_available_queue(df, now=NOW) == ["well101", "well102"]

    def test_excludes_low_volume_available_unconfirmed_and_dilute(self):
        df = pd.DataFrame([
            _q_well(201, 25, available=False, seq_confirmed=True),   # ✗ not >25
            _q_well(202, 40, available=True,  seq_confirmed=True),   # ✗ already available
            _q_well(203, 40, available=False, seq_confirmed=False),  # ✗ not seq-confirmed
            _q_well(204, 40, available=False, seq_confirmed=True, labware="Micronic Tube Rack"),  # ✗ not echo
            _q_well(205, 40, available=False, seq_confirmed=True, conc=4),  # ✗ too dilute (<=5)
        ])
        assert build_mark_available_queue(df, now=NOW) == []

    def test_excludes_disposed_and_stale_plates(self):
        df = pd.DataFrame([
            _q_well(301, 40, available=False, seq_confirmed=True, location="DISCARDED"),  # ✗ disposed
            _q_well(302, 40, available=False, seq_confirmed=True, location=""),           # ✗ blank loc
            _q_well(303, 40, available=False, seq_confirmed=True,                         # ✗ old plate
                    created_at=pd.Timestamp("2024-01-01", tz="UTC")),
        ])
        assert build_mark_available_queue(df, now=NOW) == []

    def test_deli_left_is_live_even_when_old(self):
        df = pd.DataFrame([
            _q_well(401, 40, available=False, seq_confirmed=True, location="Deli Left (DVs)",
                    created_at=pd.Timestamp("2023-01-01", tz="UTC")),   # ✓ old but Deli Left
        ])
        assert build_mark_available_queue(df, now=NOW) == ["well401"]

    def test_null_volume_excluded(self):
        df = pd.DataFrame([_q_well(501, None, available=False, seq_confirmed=True)])
        assert build_mark_available_queue(df, now=NOW) == []

    def test_dedupes_wells_from_join_fanout(self):
        # The plate query LEFT JOINs fan out rows; a well must appear only once.
        df = pd.DataFrame([_q_well(601, 40, available=False, seq_confirmed=True)] * 3)
        assert build_mark_available_queue(df, now=NOW) == ["well601"]

    def test_excludes_lsp_linked_plates(self):
        # LSP-linked plates must never be suggested for marking available.
        df = pd.DataFrame([
            _q_well(701, 40, available=False, seq_confirmed=True, plate_id=10),  # ✓ normal
            _q_well(702, 40, available=False, seq_confirmed=True, plate_id=99),  # ✗ LSP plate
        ])
        assert build_mark_available_queue(df, now=NOW, exclude_plate_ids={99}) == ["well701"]


class TestCleanInventoryQueue:
    def test_includes_available_low_volume(self):
        df = pd.DataFrame([
            _q_well(401, 25, available=True),   # ✓ (<=25, available)
            _q_well(402, 10, available=True),   # ✓
        ])
        assert build_clean_inventory_queue(df) == ["well401", "well402"]

    def test_excludes_already_unavailable_and_high_volume(self):
        df = pd.DataFrame([
            _q_well(501, 10, available=False),   # ✗ already unavailable (no-op to re-mark)
            _q_well(502, 26, available=True),    # ✗ >25
        ])
        assert build_clean_inventory_queue(df) == []

    def test_excludes_disposed_location(self):
        df = pd.DataFrame([
            _q_well(601, 10, available=True, location="DISCARDED"),  # ✗ already gone
            _q_well(602, 10, available=True, location=""),           # ✗ blank loc
        ])
        assert build_clean_inventory_queue(df) == []

    def test_mutually_exclusive_with_mark_available(self):
        # No well should appear in both queues (disjoint by the 25 µL threshold).
        df = pd.DataFrame([
            _q_well(701, 40, available=False, seq_confirmed=True),  # mark-available
            _q_well(702, 15, available=True,  seq_confirmed=True),  # clean
            _q_well(703, 25, available=True,  seq_confirmed=True),  # clean (==25)
        ])
        mark = set(build_mark_available_queue(df, now=NOW))
        clean = set(build_clean_inventory_queue(df))
        assert mark.isdisjoint(clean)


class TestPcrCsvBlock:
    def test_header_and_blank_sequence(self):
        block = _pcr_csv_block("d3550", "o10", "o11", "pAI-900", 3)
        lines = block.splitlines()
        assert lines[0] == "dpart_name,oligo_1,oligo_2,sequence,templates"
        assert len(lines) == 1 + 3                       # header + 3 rows
        assert lines[1] == "d3550,o10,o11,,pAI-900"      # blank sequence column
        assert all(ln == lines[1] for ln in lines[1:])   # identical rows

    def test_minimum_one_row(self):
        block = _pcr_csv_block("d1", "o1", "o2", "pAI-1", 0)
        assert len(block.splitlines()) == 2              # header + 1 row floor


class TestRefillQueue:
    def test_dpart_gets_pcr_block(self):
        out = pd.DataFrame([_refill_row()])
        dparts = pd.DataFrame([{"DPART_NAME": "d3550", "OLIGO_1": "o10",
                                "OLIGO_2": "o11", "DPART_TEMPLATE": "pAI-900"}])
        items = build_refill_queue(out, dparts)
        assert len(items) == 1
        it = items[0]
        assert it["glycerol_plate"] == "555" and it["glycerol_well"] == "A1"
        assert it["csv_block"].count("d3550,o10,o11,,pAI-900") == 9   # one row per PCR run
        assert it["target"] == CONTROL_BUFFER_RXNS

    def test_pai_gets_colonies_not_pcr(self):
        out = pd.DataFrame([_refill_row(Part="pAI-456", SEQUENCE_LENGTH=5000, Antibiotic="Carb")])
        items = build_refill_queue(out, pd.DataFrame(columns=["DPART_NAME"]))
        it = items[0]
        assert it["csv_block"] == ""        # plasmid → no PCR block
        assert it["colonies"] > 0           # colonies-to-pick estimated

    def test_non_refill_actions_excluded(self):
        out = pd.DataFrame([{**_refill_row(), "Action Suggested": "True"}])
        assert build_refill_queue(out, pd.DataFrame(columns=["DPART_NAME"])) == []


class TestDisposeQueue:
    def _lsp(self, **kw):
        base = {"PLATE_ID": 100, "LOCATION": "4B-ECHO1", "PROTOCOL": "Rearray 96 to 384",
                "BARCODE": None, "AVAILABLE": False}
        base.update(kw)
        return base

    def test_lists_plate_id_and_location(self):
        df = pd.DataFrame([self._lsp(PLATE_ID=100, LOCATION="4B-ECHO1")])
        items = build_dispose_queue(df)
        # No CREATED_AT on the input → age fields default (created blank, age None, not old).
        assert items == [{"plate_id": 100, "location": "4B-ECHO1", "protocol": "Rearray 96 to 384",
                          "created": "", "age_days": None, "old": False}]

    def test_flags_old_plates_by_age(self):
        rows = pd.DataFrame([
            self._lsp(PLATE_ID=10, CREATED_AT=NOW - dt.timedelta(days=100)),  # > 60d → old
            self._lsp(PLATE_ID=11, CREATED_AT=NOW - dt.timedelta(days=10)),   # < 60d → fresh
        ])
        by_id = {d["plate_id"]: d for d in build_dispose_queue(rows, now=NOW)}
        assert by_id[10]["age_days"] == 100 and by_id[10]["old"] is True
        assert by_id[11]["age_days"] == 10 and by_id[11]["old"] is False

    def test_skips_already_discarded(self):
        df = pd.DataFrame([
            self._lsp(PLATE_ID=1, LOCATION="DISCARDED"),   # ✗ already gone
            self._lsp(PLATE_ID=2, LOCATION="4B-ECHO2"),    # ✓
        ])
        ids = [d["plate_id"] for d in build_dispose_queue(df)]
        assert ids == [2]

    def test_blank_location_labeled(self):
        df = pd.DataFrame([self._lsp(PLATE_ID=3, LOCATION=None)])
        assert build_dispose_queue(df)[0]["location"] == "(no location)"

    def test_empty_input(self):
        assert build_dispose_queue(pd.DataFrame(columns=["PLATE_ID", "LOCATION"])) == []
        assert build_dispose_queue(None) == []


class TestRenderActionQueues:
    def test_renders_three_sections(self):
        plate = pd.DataFrame([
            _q_well(701, 40, available=False, seq_confirmed=True),
            _q_well(702, 10, available=True),
        ])
        out = pd.DataFrame(columns=["Part", "Is_Control", "Action Suggested",
                                    "Reactions Available", "Reactions Required",
                                    "Antibiotic", "Cell Strain", "Glycerol Plate",
                                    "Glycerol Well", "Glycerol Location",
                                    "PCR Runs Needed", "SEQUENCE_LENGTH"])
        html = render_action_queues_html(out, plate, pd.DataFrame(columns=["DPART_NAME"]), generated_at=NOW)
        assert "Mark Available" in html
        assert "Clean Inventory" in html
        assert "Refill" in html
        assert "well701" in html    # mark-available well present in copy box
        assert "well702" in html    # clean well present

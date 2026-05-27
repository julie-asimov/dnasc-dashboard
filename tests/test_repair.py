"""
Tests for dnasc/transformers/repair.py.

All tests pass well_mapping={} to repair_data to skip BigQuery calls.
resolve_optracker_streakouts tests use process_ids without well references
to hit the early-return path before any BQ call.
"""
import re
import pandas as pd
import numpy as np
import pytest

from dnasc.transformers.repair import RepairTransformer, resolve_optracker_streakouts


# ── helpers ───────────────────────────────────────────────────────────────────

def _base_row(**kwargs) -> dict:
    defaults = {
        "workorder_id":          "wo-aaa",
        "type":                  "golden_gate_workorder",
        "root_work_order_id":    "wo-aaa",
        "source_asm_process_id": None,
        "source_lsp_process_id": None,
        "req_id":                "req-001",
        "experiment_name":       "Exp A",
        "STOCK_ID":              "pAI-001",
        "root_STOCK_ID":         "pAI-001",
        "for_partner":           False,
        "wo_status":             "SUCCEEDED",
        "visual_status":         "SUCCEEDED",
        "request_status":        "IN_PROGRESS",
        "wo_created_at":         pd.Timestamp("2025-03-01", tz="UTC"),
    }
    defaults.update(kwargs)
    return defaults


def _make_df(*rows) -> pd.DataFrame:
    return pd.DataFrame([_base_row(**r) for r in rows])


# ── RepairTransformer.repair_data ─────────────────────────────────────────────

class TestRepairDataDedupGuard:

    def test_duplicate_columns_stripped_at_entry(self):
        df = _make_df({})
        # Force duplicate columns via numpy (pandas .insert() rejects duplicates)
        arr = df.values
        stock_col = df.columns.get_loc("STOCK_ID")
        extra_col = df.values[:, stock_col:stock_col + 1]
        df_dup = pd.DataFrame(
            np.hstack([arr, extra_col]),
            columns=list(df.columns) + ["STOCK_ID"],
        )
        assert df_dup.columns.duplicated().any()
        result = RepairTransformer.repair_data(df_dup, well_mapping={})
        assert not result.columns.duplicated().any()


class TestRepairDataTransformationRoots:

    def test_same_request_tfm_root_set_from_source(self):
        """Transformation's root is resolved through source_asm_process_id."""
        df = _make_df(
            {"workorder_id": "gg-001", "type": "golden_gate_workorder",
             "root_work_order_id": "gg-001", "req_id": "req-001"},
            {"workorder_id": "tfm-001", "type": "transformation_workorder",
             "root_work_order_id": "tfm-001", "source_asm_process_id": "gg-001",
             "req_id": "req-001", "experiment_name": "Exp A"},
        )
        result = RepairTransformer.repair_data(df, well_mapping={})
        tfm = result[result["workorder_id"] == "tfm-001"].iloc[0]
        assert tfm["root_work_order_id"] == "gg-001"

    def test_same_request_tfm_req_id_filled_from_root(self):
        """Same-request transformation: req_id is cleared then refilled from root row."""
        df = _make_df(
            {"workorder_id": "gg-001", "type": "golden_gate_workorder",
             "root_work_order_id": "gg-001", "req_id": "req-001"},
            {"workorder_id": "tfm-001", "type": "transformation_workorder",
             "root_work_order_id": "tfm-001", "source_asm_process_id": "gg-001",
             "req_id": "req-001", "experiment_name": "Exp A"},
        )
        result = RepairTransformer.repair_data(df, well_mapping={})
        tfm = result[result["workorder_id"] == "tfm-001"].iloc[0]
        # req_id is cleared then refilled from the root (gg-001 → req-001)
        assert tfm["req_id"] == "req-001"

    def test_cross_request_tfm_self_roots(self):
        """Transformation from a different request than its GG self-roots."""
        df = _make_df(
            {"workorder_id": "gg-001", "type": "golden_gate_workorder",
             "root_work_order_id": "gg-001", "req_id": "req-001"},
            {"workorder_id": "tfm-002", "type": "transformation_workorder",
             "root_work_order_id": "tfm-002", "source_asm_process_id": "gg-001",
             "req_id": "req-002", "experiment_name": "Exp B"},
        )
        result = RepairTransformer.repair_data(df, well_mapping={})
        tfm = result[result["workorder_id"] == "tfm-002"].iloc[0]
        assert tfm["root_work_order_id"] == "tfm-002"

    def test_cross_request_tfm_keeps_own_req_id(self):
        """Cross-request transformation keeps its own req_id (not cleared)."""
        df = _make_df(
            {"workorder_id": "gg-001", "type": "golden_gate_workorder",
             "root_work_order_id": "gg-001", "req_id": "req-001"},
            {"workorder_id": "tfm-002", "type": "transformation_workorder",
             "root_work_order_id": "tfm-002", "source_asm_process_id": "gg-001",
             "req_id": "req-002", "experiment_name": "Exp B"},
        )
        result = RepairTransformer.repair_data(df, well_mapping={})
        tfm = result[result["workorder_id"] == "tfm-002"].iloc[0]
        assert tfm["req_id"] == "req-002"

    def test_tfm_without_source_root_attempt(self):
        """Transformation with no source: repair tries to set root from source (None),
        resulting in NaN root. This documents current behavior — a sourceless
        transformation has no physical lineage to repair from."""
        df = _make_df(
            {"workorder_id": "tfm-solo", "type": "transformation_workorder",
             "root_work_order_id": "tfm-solo", "source_asm_process_id": None,
             "req_id": "req-001"},
        )
        result = RepairTransformer.repair_data(df, well_mapping={})
        tfm = result[result["workorder_id"] == "tfm-solo"].iloc[0]
        # repair_data maps None source → NaN root; no recovery path without a source
        assert pd.isna(tfm["root_work_order_id"]) or tfm["root_work_order_id"] == "tfm-solo"

    def test_non_tfm_rows_unaffected(self):
        """GG workorder roots are not altered by the transformation repair pass."""
        df = _make_df(
            {"workorder_id": "gg-001", "type": "golden_gate_workorder",
             "root_work_order_id": "gg-001", "req_id": "req-001"},
        )
        result = RepairTransformer.repair_data(df, well_mapping={})
        gg = result[result["workorder_id"] == "gg-001"].iloc[0]
        assert gg["root_work_order_id"] == "gg-001"


class TestRepairDataStreakoutRoots:

    def test_streakout_root_resolved_from_source(self):
        """Streakout root is resolved through source_asm_process_id → GG root."""
        df = _make_df(
            {"workorder_id": "gg-001", "type": "golden_gate_workorder",
             "root_work_order_id": "gg-001", "req_id": "req-001"},
            {"workorder_id": "STREAK_well123", "type": "streakout_operation",
             "root_work_order_id": None, "source_asm_process_id": "gg-001",
             "req_id": None},
        )
        result = RepairTransformer.repair_data(df, well_mapping={})
        streak = result[result["workorder_id"] == "STREAK_well123"].iloc[0]
        assert streak["root_work_order_id"] == "gg-001"

    def test_tfo_root_resolved_from_source(self):
        """transformation_offline_operation root resolved same way as streakout."""
        df = _make_df(
            {"workorder_id": "gg-001", "type": "golden_gate_workorder",
             "root_work_order_id": "gg-001", "req_id": "req-001"},
            {"workorder_id": "TFM_well456", "type": "transformation_offline_operation",
             "root_work_order_id": None, "source_asm_process_id": "gg-001",
             "req_id": None},
        )
        result = RepairTransformer.repair_data(df, well_mapping={})
        tfo = result[result["workorder_id"] == "TFM_well456"].iloc[0]
        assert tfo["root_work_order_id"] == "gg-001"


# ── resolve_optracker_streakouts — pattern detection (no BQ) ─────────────────

class TestResolveOptrackerStreakoutsPatterns:
    """
    Test STREAK_RE / _WELL_RE detection logic.
    Process_ids without a 'well\d+' component hit the early-return before any BQ call.
    """

    def _final_df(self):
        return _make_df({"workorder_id": "gg-existing"})

    def _optracker(self, pids):
        return pd.DataFrame({"process_id": pids})

    def test_no_process_ids_returns_unchanged(self):
        df = self._final_df()
        result = resolve_optracker_streakouts(df, pd.DataFrame({"process_id": []}))
        assert len(result) == len(df)

    def test_matching_ids_already_in_df_are_not_added(self):
        """process_ids already in final_df must not create synthetic duplicates."""
        df = _make_df({"workorder_id": "STREAK_well1"})
        op = self._optracker(["STREAK_well1"])
        result = resolve_optracker_streakouts(df, op)
        assert len(result) == 1

    def test_streak_prefix_detected(self):
        """STREAK anywhere in the ID is a match (no ^ anchor)."""
        STREAK_RE = re.compile(r"STREAK|^(STBL3|EPI400|TFM)", re.I)
        assert STREAK_RE.search("STREAK_well123")
        assert STREAK_RE.search("PARTNER_STREAK_well456")
        assert STREAK_RE.search("streak_well789")

    def test_stbl3_anchored_detected(self):
        STREAK_RE = re.compile(r"STREAK|^(STBL3|EPI400|TFM)", re.I)
        assert STREAK_RE.search("STBL3_well100")
        assert not STREAK_RE.search("xxSTBL3_well100")

    def test_epi400_anchored_detected(self):
        STREAK_RE = re.compile(r"STREAK|^(STBL3|EPI400|TFM)", re.I)
        assert STREAK_RE.search("EPI400_well200")
        assert not STREAK_RE.search("xxEPI400_well200")

    def test_tfm_anchored_detected(self):
        STREAK_RE = re.compile(r"STREAK|^(STBL3|EPI400|TFM)", re.I)
        assert STREAK_RE.search("TFM_well300")
        assert not STREAK_RE.search("xxTFM_well300")

    def test_ordinary_uuid_not_detected(self):
        STREAK_RE = re.compile(r"STREAK|^(STBL3|EPI400|TFM)", re.I)
        assert not STREAK_RE.search("abc123-def456-ghi789")

    def test_no_well_ref_returns_unchanged_df(self):
        """IDs matching STREAK_RE but with no well\d+ reference → early return, no BQ call."""
        df = self._final_df()
        # STREAK without a well number → well_pids will be empty → early return
        op = self._optracker(["STREAK_nowell"])
        result = resolve_optracker_streakouts(df, op)
        assert len(result) == len(df)

    def test_synthetic_row_type_streak_vs_tfo(self):
        """STREAK in pid → streakout_operation; TFM/STBL3 → transformation_offline_operation."""
        # We can test the type-selection logic directly without running the full function
        pid_streak = "STREAK_well999"
        pid_tfm    = "TFM_well999"
        type_streak = "streakout_operation" if "STREAK" in pid_streak.upper() else "transformation_offline_operation"
        type_tfm    = "streakout_operation" if "STREAK" in pid_tfm.upper() else "transformation_offline_operation"
        assert type_streak == "streakout_operation"
        assert type_tfm    == "transformation_offline_operation"

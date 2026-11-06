"""
Tests for colony/repick visual_status derivation logic.

Covers _seq_status_from_ops (shared helper) and the key branches of
_apply_colony_status_overrides and _repick_status in pipeline.py.
"""
import pandas as pd
import pytest

from dnasc.pipeline import _seq_status_from_ops, _apply_colony_status_overrides
from dnasc import protocols as proto


# ── _seq_status_from_ops ──────────────────────────────────────────────────────

class TestSeqStatusFromOps:
    """The shared helper: given progress-SC ops and seq count, what's the status?"""

    def test_seq_confirmed_returns_succeeded(self):
        pn = [proto.REARRAY, proto.NGS]
        ps = ["SC", "SC"]
        assert _seq_status_from_ops(pn, ps, seq=2) == "SUCCEEDED"

    def test_ngs_sc_no_seq_returns_failed(self):
        pn = [proto.REARRAY, proto.NGS]
        ps = ["SC", "SC"]
        assert _seq_status_from_ops(pn, ps, seq=0) == "FAILED"

    def test_ngs_fa_no_seq_returns_failed(self):
        pn = [proto.REARRAY, proto.NGS]
        ps = ["SC", "FA"]
        assert _seq_status_from_ops(pn, ps, seq=0) == "FAILED"

    def test_fragment_analyzer_fa_no_seq_returns_failed(self):
        pn = [proto.REARRAY, proto.FRAGMENT_ANALYZER]
        ps = ["SC", "FA"]
        assert _seq_status_from_ops(pn, ps, seq=0) == "FAILED"

    def test_progress_sc_only_no_seq_returns_in_progress(self):
        """Rearray SC but no seq protocol result yet."""
        pn = [proto.REARRAY]
        ps = ["SC"]
        assert _seq_status_from_ops(pn, ps, seq=0) == "IN_PROGRESS"

    def test_dna_quant_sc_no_seq_returns_in_progress(self):
        pn = [proto.DNA_QUANT]
        ps = ["SC"]
        assert _seq_status_from_ops(pn, ps, seq=0) == "IN_PROGRESS"

    def test_seq_confirmed_wins_even_if_ngs_fa(self):
        """seq_confirmed > 0 wins regardless of protocol state."""
        pn = [proto.REARRAY, proto.NGS]
        ps = ["SC", "FA"]
        assert _seq_status_from_ops(pn, ps, seq=1) == "SUCCEEDED"

    def test_empty_ops_no_seq_returns_in_progress(self):
        assert _seq_status_from_ops([], [], seq=0) == "IN_PROGRESS"


# ── _apply_colony_status_overrides ────────────────────────────────────────────

def _make_df(**kwargs):
    """Build a single-row DataFrame for colony status override testing."""
    defaults = {
        "type":           "golden_gate_workorder",
        "wo_status":      "SUCCEEDED",
        "total_colonies": 1,
        "seq_confirmed":  0,
        "protocol_name":  None,
        "operation_state": None,
        "visual_status":  "SUCCEEDED",
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


class TestApplyColonyStatusOverrides:

    def test_no_colony_types_returns_unchanged(self):
        df = _make_df(type="lsp_workorder")
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"
        assert result["is_software_fail"].iloc[0] == False

    def test_failed_with_seq_confirmed_becomes_succeeded_software_fail(self):
        df = _make_df(wo_status="FAILED", seq_confirmed=1, total_colonies=1)
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"
        assert result["is_software_fail"].iloc[0] == True

    def test_running_with_seq_confirmed_becomes_succeeded(self):
        df = _make_df(wo_status="RUNNING", seq_confirmed=2, total_colonies=2)
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"
        assert result["is_software_fail"].iloc[0] == False

    def test_succeeded_zero_colonies_no_transf_done_running_op_returns_running(self):
        df = _make_df(
            wo_status="SUCCEEDED", total_colonies=0, seq_confirmed=0,
            protocol_name=[proto.STAR_TRANSF], operation_state=["RU"],
        )
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "RUNNING"

    def test_succeeded_zero_colonies_transf_done_no_colonies_returns_failed(self):
        df = _make_df(
            wo_status="SUCCEEDED", total_colonies=0, seq_confirmed=0,
            protocol_name=[proto.STAR_TRANSF, proto.MINIPREP],
            operation_state=["SC", "SC"],
        )
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "FAILED"

    def test_succeeded_with_colonies_progress_sc_seq_confirmed(self):
        df = _make_df(
            wo_status="SUCCEEDED", total_colonies=3, seq_confirmed=3,
            protocol_name=[proto.REARRAY, proto.NGS],
            operation_state=["SC", "SC"],
        )
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"

    def test_succeeded_with_colonies_progress_sc_ngs_fa_returns_failed(self):
        df = _make_df(
            wo_status="SUCCEEDED", total_colonies=3, seq_confirmed=0,
            protocol_name=[proto.REARRAY, proto.NGS],
            operation_state=["SC", "FA"],
        )
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "FAILED"

    def test_succeeded_with_colonies_progress_sc_another_op_running_returns_running(self):
        """Progress op SC but a downstream op is still RU → still running."""
        df = _make_df(
            wo_status="SUCCEEDED", total_colonies=3, seq_confirmed=0,
            protocol_name=[proto.REARRAY, proto.NGS],
            operation_state=["SC", "RU"],
        )
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "RUNNING"

    def test_succeeded_with_colonies_miniprep_sc_no_progress_returns_failed(self):
        """Miniprep SC but no rearray/quant/NGS SC yet → FAILED (no downstream progress)."""
        df = _make_df(
            wo_status="SUCCEEDED", total_colonies=3, seq_confirmed=0,
            protocol_name=[proto.MINIPREP],
            operation_state=["SC"],
        )
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "FAILED"

    def test_succeeded_with_colonies_miniprep_running_no_progress_returns_running(self):
        df = _make_df(
            wo_status="SUCCEEDED", total_colonies=3, seq_confirmed=0,
            protocol_name=[proto.MINIPREP],
            operation_state=["RU"],
        )
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "RUNNING"

    def test_succeeded_with_colonies_no_ops_returns_running(self):
        """Colonies picked but no ops yet — still in progress."""
        df = _make_df(
            wo_status="SUCCEEDED", total_colonies=3, seq_confirmed=0,
            protocol_name=None, operation_state=None,
        )
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "RUNNING"

    def test_is_status_override_set_when_status_changed(self):
        df = _make_df(wo_status="FAILED", seq_confirmed=1, total_colonies=1)
        result = _apply_colony_status_overrides(df)
        assert result["is_status_override"].iloc[0] == True

    def test_unknown_wo_status_returns_existing_visual_status(self):
        df = _make_df(wo_status="UNKNOWN", visual_status="IN_PROGRESS")
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "IN_PROGRESS"


class TestOptrackerManualRepickStatus:
    """
    Manual repicks logged in LIMS under a hand-typed process id (e.g.
    PICK_25Aug26_well2176911) surface as their own optracker_operation row.
    They carry real colony counts from get_colony_data (matched on
    well.process_id) but none of the transformation-shaped protocol sequence,
    so only the seq-confirmed rescue applies to them.
    """

    def test_seq_confirmed_pick_reads_succeeded(self):
        df = _make_df(type="optracker_operation", wo_status="SUCCEEDED",
                      total_colonies=6, seq_confirmed=3, visual_status="FAILED")
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"
        assert result["is_software_fail"].iloc[0] == False

    def test_bios_failed_but_seq_confirmed_is_software_fail(self):
        df = _make_df(type="optracker_operation", wo_status="FAILED",
                      total_colonies=8, seq_confirmed=4, visual_status="FAILED")
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"
        assert result["is_software_fail"].iloc[0] == True

    def test_colonies_but_none_confirmed_reads_running(self):
        """
        Colonies on the bench, nothing sequenced yet — in flight, the same read
        a transformation_offline_operation gets. The transformation-shaped
        SUCCEEDED logic still must not run (it would say IN_PROGRESS off a lone
        Miniprep SC), and the verdict must not depend on which recovery path
        created the row: off well2407817, sbclone_EPI read RUNNING while
        ubclone_10B read SUCCEEDED with both at 0 confirmed.
        """
        df = _make_df(type="optracker_operation", wo_status="SUCCEEDED",
                      workorder_id="ubclone_10B_04Nov26_well2407817",
                      total_colonies=6, seq_confirmed=0, visual_status="SUCCEEDED",
                      protocol_name=[proto.MINIPREP], operation_state=["SC"])
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "RUNNING"

    def test_refill_with_no_seq_stays_untouched(self):
        """
        A refill restocks an already-confirmed well and ends at the glycerol —
        no NGS is ever coming, so the in-flight read must not touch it. 169 of
        the 194 optracker_operation rows are refills; without this guard they
        all park at RUNNING forever.
        """
        df = _make_df(type="optracker_operation", wo_status="SUCCEEDED",
                      workorder_id="REFILL_STREAK_22Jul2026_well2105690",
                      total_colonies=4, seq_confirmed=0, visual_status="SUCCEEDED")
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"

    def test_kind_column_beats_the_id_text(self):
        """manual_op_kind is authoritative when the row carries it."""
        df = _make_df(type="optracker_operation", wo_status="SUCCEEDED",
                      workorder_id="some_free_text_well123", manual_op_kind="REFILL",
                      total_colonies=4, seq_confirmed=0, visual_status="SUCCEEDED")
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"

    def test_zero_colonies_leaves_status_untouched(self):
        df = _make_df(type="optracker_operation", wo_status="SUCCEEDED",
                      total_colonies=0, seq_confirmed=0, visual_status="SUCCEEDED")
        result = _apply_colony_status_overrides(df)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"

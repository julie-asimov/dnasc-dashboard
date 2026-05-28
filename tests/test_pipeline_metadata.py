"""
Tests for pipeline.py metadata functions: _assign_lsp_roots, _finalize_metadata,
and the visual_status bridge + filtering logic in _filter_and_enrich.

These three functions have caused the most documented bugs in the version history
and had zero test coverage before this file.
"""
import numpy as np
import pandas as pd
import pytest

from dnasc.pipeline import _assign_lsp_roots, _finalize_metadata, _filter_and_enrich


# ── helpers ───────────────────────────────────────────────────────────────────

_T0 = pd.Timestamp("2025-01-01", tz="UTC")
_T1 = pd.Timestamp("2025-06-01", tz="UTC")   # later
_T2 = pd.Timestamp("2025-12-01", tz="UTC")   # even later


def _lsp(**kw) -> dict:
    """Minimal LSP row with sensible defaults."""
    base = dict(
        workorder_id="wo-lsp-1",
        type="lsp_workorder",
        root_work_order_id="wo-lsp-1",
        data_source="LSP",
        req_id="req-001",
        experiment_name="Exp A",
        request_status="IN_PROGRESS",
        lsp_own_request_id=None,
        wo_created_at=_T1,
        request_created_at=_T0,
        STOCK_ID=None,
        plasmid_id="pAI-001",
        source_lsp_process_id=None,
        lsp_process_id=None,
        middle_root=None,
        source_workorder_id=None,
        input_well_id=None,
        lsp_input_well=None,
        cloning_strain=None,
        lsp_batch_id="LSP-10001",
        priority=None,
        construct_name=None,
        for_partner=False,
        batch_comments=None,
        qubit_concentration_ngul=None,
        deprecated_concentration_ngul=None,
        nanodrop_concentration_ngul=None,
    )
    base.update(kw)
    return base


def _gg(**kw) -> dict:
    """Minimal GG / non-LSP row."""
    base = dict(
        workorder_id="wo-gg-1",
        type="golden_gate_workorder",
        root_work_order_id="wo-gg-1",
        data_source="BIOS",
        req_id="req-001",
        experiment_name="Exp A",
        request_status="IN_PROGRESS",
        lsp_own_request_id=None,
        wo_created_at=_T1,
        request_created_at=_T0,
        STOCK_ID="pAI-001",
        plasmid_id=None,
        source_lsp_process_id=None,
        lsp_process_id=None,
        middle_root=None,
        source_workorder_id=None,
        input_well_id=None,
        lsp_input_well=None,
        cloning_strain=None,
        lsp_batch_id="",
        priority=None,
        construct_name=None,
        for_partner=False,
        batch_comments=None,
        qubit_concentration_ngul=None,
        deprecated_concentration_ngul=None,
        nanodrop_concentration_ngul=None,
    )
    base.update(kw)
    return base


def _df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _fe_row(**kw) -> dict:
    """Minimal row for _filter_and_enrich."""
    base = dict(
        workorder_id="wo-1",
        type="golden_gate_workorder",
        root_work_order_id="wo-1",
        data_source="BIOS",
        req_id="req-001",
        request_status="IN_PROGRESS",
        lsp_batch_id="",
        wo_status="SUCCEEDED",
        operation_state=[],
        protocol_name=[],
        source_lsp_process_id=None,
        total_volume_ul=None,
        STOCK_ID=None,
        root_STOCK_ID=None,
    )
    base.update(kw)
    return base


# ── _assign_lsp_roots ─────────────────────────────────────────────────────────

class TestAssignLspRoots:

    def test_non_lsp_row_root_unchanged(self):
        df = _df(_gg(root_work_order_id="wo-gg-1"))
        result = _assign_lsp_roots(df)
        assert result.loc[0, "root_work_order_id"] == "wo-gg-1"

    def test_lsp_resolves_root_via_source_lsp_process_id(self):
        # root workorder exists in df; LSP should inherit its root
        root_row = _gg(workorder_id="wo-gg-root", root_work_order_id="wo-gg-root")
        lsp_row  = _lsp(
            workorder_id="wo-lsp-1",
            root_work_order_id="wo-lsp-1",
            source_lsp_process_id="wo-gg-root",
        )
        result = _assign_lsp_roots(_df(root_row, lsp_row))
        assert result.loc[1, "root_work_order_id"] == "wo-gg-root"

    def test_lsp_prefix_source_skipped(self):
        lsp_row = _lsp(
            workorder_id="wo-lsp-1",
            root_work_order_id="wo-lsp-1",
            source_lsp_process_id="LSP-9999",
        )
        result = _assign_lsp_roots(_df(lsp_row))
        # Falls back to self
        assert result.loc[0, "root_work_order_id"] == "wo-lsp-1"

    def test_lsp_self_ref_source_skipped(self):
        lsp_row = _lsp(
            workorder_id="wo-lsp-1",
            root_work_order_id="wo-lsp-1",
            source_lsp_process_id="wo-lsp-1",
        )
        result = _assign_lsp_roots(_df(lsp_row))
        assert result.loc[0, "root_work_order_id"] == "wo-lsp-1"

    def test_lsp_uuid_source_not_in_df_used_directly(self):
        # source id not in df but len > 20 → used as root directly
        uuid = "abcdef1234567890abcde"  # 21 chars, strict > 20
        lsp_row = _lsp(source_lsp_process_id=uuid)
        result = _assign_lsp_roots(_df(lsp_row))
        assert result.loc[0, "root_work_order_id"] == uuid

    def test_backfill_source_lsp_from_lsp_process_id(self):
        # orphan LSP with no source_lsp_process_id but has lsp_process_id
        lsp_row = _lsp(
            source_lsp_process_id=None,
            lsp_process_id="wo-gg-root",
        )
        result = _assign_lsp_roots(_df(lsp_row))
        assert result.loc[0, "source_lsp_process_id"] == "wo-gg-root"

    def test_self_ref_source_cleared_after_backfill(self):
        # backfill assigns workorder_id to itself → should be cleared
        lsp_row = _lsp(
            workorder_id="wo-lsp-1",
            source_lsp_process_id=None,
            lsp_process_id="wo-lsp-1",
        )
        result = _assign_lsp_roots(_df(lsp_row))
        assert pd.isna(result.loc[0, "source_lsp_process_id"])


# ── _finalize_metadata ────────────────────────────────────────────────────────

class TestFinalizeMetadataTemporalDisqualifier:
    """LSP workorders created before their request should lose the req_id."""

    def test_lsp_before_request_loses_req_id(self):
        # Disqualifier clears req-old → ACTIVE_WIP sentinel fills it; experiment_name stays null
        lsp = _lsp(wo_created_at=_T0, request_created_at=_T1, req_id="req-old")
        result = _finalize_metadata(_df(lsp))
        assert result.loc[0, "req_id"] == "ACTIVE_WIP"
        assert pd.isna(result.loc[0, "experiment_name"])

    def test_lsp_after_request_keeps_req_id(self):
        lsp = _lsp(wo_created_at=_T1, request_created_at=_T0, req_id="req-001")
        result = _finalize_metadata(_df(lsp))
        assert result.loc[0, "req_id"] == "req-001"

    def test_gg_before_request_keeps_req_id(self):
        # Temporal disqualifier must NOT fire for non-LSP types (v1.0.4 regression)
        gg = _gg(wo_created_at=_T0, request_created_at=_T1, req_id="req-001")
        result = _finalize_metadata(_df(gg))
        assert result.loc[0, "req_id"] == "req-001"

    def test_transformation_before_request_keeps_req_id(self):
        row = _gg(
            workorder_id="wo-tfm-1",
            type="transformation_workorder",
            wo_created_at=_T0,
            request_created_at=_T1,
            req_id="req-001",
        )
        result = _finalize_metadata(_df(row))
        assert result.loc[0, "req_id"] == "req-001"

    def test_lsp_with_direct_link_exempt_from_disqualifier(self):
        # LSP Refill: created before request but lsp_own_request_id is set — keep req_id
        lsp = _lsp(
            wo_created_at=_T0,
            request_created_at=_T1,
            req_id="req-refill",
            lsp_own_request_id="req-refill",
        )
        result = _finalize_metadata(_df(lsp))
        assert result.loc[0, "req_id"] == "req-refill"


class TestFinalizeMetadataReqIdFill:
    """req_id propagates from root row only, not from siblings."""

    def test_req_id_fills_from_root(self):
        root = _gg(workorder_id="wo-root", root_work_order_id="wo-root", req_id="req-001")
        child = _lsp(
            workorder_id="wo-child",
            root_work_order_id="wo-root",
            req_id=None,
        )
        result = _finalize_metadata(_df(root, child))
        assert result.loc[1, "req_id"] == "req-001"

    def test_sibling_req_id_does_not_pollute_another_sibling(self):
        # Two LSPs under same root but different experiments: only root's req_id propagates
        root   = _gg(workorder_id="wo-root", root_work_order_id="wo-root", req_id="req-A")
        lsp_a  = _lsp(workorder_id="wo-lsp-a", root_work_order_id="wo-root", req_id="req-B")
        lsp_b  = _lsp(workorder_id="wo-lsp-b", root_work_order_id="wo-root", req_id=None)
        result = _finalize_metadata(_df(root, lsp_a, lsp_b))
        # lsp_b should fill from root (req-A), not from sibling lsp_a (req-B)
        assert result.loc[2, "req_id"] == "req-A"

    def test_experiment_name_fills_from_root(self):
        root  = _gg(workorder_id="wo-root", root_work_order_id="wo-root", experiment_name="Exp X")
        child = _lsp(workorder_id="wo-child", root_work_order_id="wo-root", experiment_name=None)
        result = _finalize_metadata(_df(root, child))
        assert result.loc[1, "experiment_name"] == "Exp X"


class TestFinalizeMetadataSentinels:
    """ORPHAN_LEGACY and ACTIVE_WIP sentinel assignment."""

    def test_synthetic_lsp_no_req_becomes_orphan_legacy(self):
        # experiment_name=None so root_map doesn't accidentally populate req_id
        row = _lsp(
            data_source="SYNTHETIC_LSP",
            req_id=None,
            experiment_name=None,
            root_work_order_id="wo-lsp-1",
            workorder_id="wo-lsp-1",
        )
        result = _finalize_metadata(_df(row))
        assert result.loc[0, "req_id"] == "ORPHAN_LEGACY"
        assert result.loc[0, "request_status"] == "SUCCEEDED"

    def test_lsp_no_req_becomes_active_wip(self):
        row = _lsp(data_source="LSP", req_id=None)
        result = _finalize_metadata(_df(row))
        assert result.loc[0, "req_id"] == "ACTIVE_WIP"
        assert result.loc[0, "request_status"] == "IN_PROGRESS"


class TestFinalizeMetadataStockId:
    """LSP STOCK_ID fills from plasmid_id before root-group fill."""

    def test_lsp_stock_fills_from_plasmid_id(self):
        row = _lsp(STOCK_ID=None, plasmid_id="pAI-999")
        result = _finalize_metadata(_df(row))
        assert result.loc[0, "STOCK_ID"] == "pAI-999"

    def test_lsp_plasmid_id_does_not_overwrite_existing_stock(self):
        row = _lsp(STOCK_ID="pAI-100", plasmid_id="pAI-999")
        result = _finalize_metadata(_df(row))
        assert result.loc[0, "STOCK_ID"] == "pAI-100"

    def test_stock_id_propagates_from_root_to_child(self):
        root  = _gg(workorder_id="wo-root", root_work_order_id="wo-root", STOCK_ID="pAI-200")
        child = _lsp(workorder_id="wo-child", root_work_order_id="wo-root", STOCK_ID=None, plasmid_id=None)
        result = _finalize_metadata(_df(root, child))
        assert result.loc[1, "STOCK_ID"] == "pAI-200"

    def test_sibling_lsp_plasmid_does_not_pollute_other_lsp(self):
        # Two LSPs same root: each should keep their own plasmid_id fill, not inherit the other's
        root  = _gg(workorder_id="wo-root", root_work_order_id="wo-root", STOCK_ID=None)
        lsp_a = _lsp(workorder_id="wo-lsp-a", root_work_order_id="wo-root", STOCK_ID=None, plasmid_id="pAI-111")
        lsp_b = _lsp(workorder_id="wo-lsp-b", root_work_order_id="wo-root", STOCK_ID=None, plasmid_id="pAI-222")
        result = _finalize_metadata(_df(root, lsp_a, lsp_b))
        assert result.loc[1, "STOCK_ID"] == "pAI-111"
        assert result.loc[2, "STOCK_ID"] == "pAI-222"


class TestFinalizeMetadataCloningStrain:

    def test_cloning_strain_fills_from_source_transformation(self):
        tfm = _gg(workorder_id="wo-tfm", type="transformation_workorder", cloning_strain="DH10B")
        lsp = _lsp(workorder_id="wo-lsp", source_lsp_process_id="wo-tfm", cloning_strain=None)
        result = _finalize_metadata(_df(tfm, lsp))
        assert result.loc[1, "cloning_strain"] == "DH10B"

    def test_cloning_strain_not_overwritten_if_already_set(self):
        tfm = _gg(workorder_id="wo-tfm", type="transformation_workorder", cloning_strain="DH10B")
        lsp = _lsp(workorder_id="wo-lsp", source_lsp_process_id="wo-tfm", cloning_strain="Stbl3")
        result = _finalize_metadata(_df(tfm, lsp))
        assert result.loc[1, "cloning_strain"] == "Stbl3"


# ── _filter_and_enrich — filtering ───────────────────────────────────────────

class TestFilterAndEnrichFiltering:

    def test_blacklisted_lsp_batch_removed(self):
        kept    = _fe_row(lsp_batch_id="LSP-10001")
        removed = _fe_row(workorder_id="wo-2", lsp_batch_id="LSP-7602")
        result  = _filter_and_enrich(_df(kept, removed))
        assert "wo-2" not in result["workorder_id"].values

    def test_req_id_with_active_status_kept(self):
        row = _fe_row(req_id="req-001", request_status="IN_PROGRESS")
        result = _filter_and_enrich(_df(row))
        assert len(result) == 1

    def test_no_req_id_no_ops_no_volume_dropped(self):
        row = _fe_row(req_id=None, request_status=None, protocol_name=[], total_volume_ul=0)
        result = _filter_and_enrich(_df(row))
        assert len(result) == 0

    def test_lsp10_prefix_batch_kept_without_req(self):
        row = _fe_row(req_id=None, request_status=None, lsp_batch_id="LSP-10050", protocol_name=[])
        result = _filter_and_enrich(_df(row))
        assert len(result) == 1

    def test_high_volume_kept_without_req(self):
        row = _fe_row(req_id=None, request_status=None, total_volume_ul=5.0, protocol_name=[])
        result = _filter_and_enrich(_df(row))
        assert len(result) == 1

    def test_ops_list_kept_without_req(self):
        row = _fe_row(req_id=None, request_status=None, protocol_name=["Miniprep"])
        result = _filter_and_enrich(_df(row))
        assert len(result) == 1

    def test_test_location_lsp_removed(self):
        row = _fe_row(
            type="lsp_workorder",
            location="TEST SHELF",
            req_id="req-001",
            request_status="IN_PROGRESS",
        )
        result = _filter_and_enrich(_df(row))
        assert len(result) == 0

    def test_canceled_lsp_no_ops_removed(self):
        row = _fe_row(
            type="lsp_workorder",
            wo_status="CANCELED",
            protocol_name=[],
            req_id="req-001",
            request_status="IN_PROGRESS",
        )
        result = _filter_and_enrich(_df(row))
        assert len(result) == 0

    def test_canceled_lsp_with_ops_kept(self):
        row = _fe_row(
            type="lsp_workorder",
            wo_status="CANCELED",
            protocol_name=["Miniprep"],
            req_id="req-001",
            request_status="IN_PROGRESS",
        )
        result = _filter_and_enrich(_df(row))
        assert len(result) == 1

    def test_duplicate_columns_stripped(self):
        row = _fe_row()
        df  = _df(row)
        # Inject duplicate column
        df2 = pd.concat([df, df[["req_id"]].rename(columns={"req_id": "req_id"})], axis=1)
        # Force actual duplicate via numpy trick
        arr = df.values
        extra = df.values[:, df.columns.get_loc("req_id"):df.columns.get_loc("req_id") + 1]
        df_dup = pd.DataFrame(
            np.hstack([arr, extra]),
            columns=list(df.columns) + ["req_id"],
        )
        result = _filter_and_enrich(df_dup)
        assert not result.columns.duplicated().any()


# ── _filter_and_enrich — visual_status bridge ────────────────────────────────

class TestFilterAndEnrichVisualStatus:
    """np.select priority: CANCELED > DRAFT > RU > RD > SC > FA > wo > req > default."""

    def test_canceled_wins_over_fa_ops(self):
        # v1.2.8 regression: CANCELED must win even when ops have FA states
        row = _fe_row(wo_status="CANCELED", operation_state=["FA"])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "CANCELED"

    def test_draft_wo_status(self):
        row = _fe_row(wo_status="DRAFT", operation_state=[])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "DRAFT"

    def test_running_state_gives_running(self):
        row = _fe_row(wo_status="SUCCEEDED", operation_state=["RU"])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "RUNNING"

    def test_ready_state_gives_ready(self):
        row = _fe_row(wo_status="SUCCEEDED", operation_state=["RD"])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "READY"

    def test_sc_terminal_gives_succeeded_for_non_lsp(self):
        row = _fe_row(type="golden_gate_workorder", wo_status="SUCCEEDED", operation_state=["SC"])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "SUCCEEDED"

    def test_fa_terminal_gives_failed_for_non_lsp(self):
        row = _fe_row(type="golden_gate_workorder", wo_status="SUCCEEDED", operation_state=["FA"])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "FAILED"

    def test_sc_then_fa_gives_failed(self):
        # FA is last terminal — should win over earlier SC
        row = _fe_row(type="golden_gate_workorder", wo_status="SUCCEEDED", operation_state=["SC", "FA"])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "FAILED"

    def test_fa_then_sc_gives_succeeded(self):
        # SC is last terminal — should win over earlier FA
        row = _fe_row(type="golden_gate_workorder", wo_status="SUCCEEDED", operation_state=["FA", "SC"])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "SUCCEEDED"

    def test_sc_on_lsp_does_not_give_succeeded(self):
        # is_not_lsp guard: SC/FA states should not override for LSP rows
        row = _fe_row(type="lsp_workorder", wo_status="IN_PROGRESS", operation_state=["SC"])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] != "SUCCEEDED"

    def test_valid_wo_status_used_when_no_ops(self):
        row = _fe_row(wo_status="RUNNING", operation_state=[])
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "RUNNING"

    def test_null_wo_backfilled_from_visual_status(self):
        row = _fe_row(wo_status=None, request_status="IN_PROGRESS", operation_state=[])
        result = _filter_and_enrich(_df(row))
        # wo_status should be filled from visual_status after bridge
        assert pd.notna(result.loc[result.index[0], "wo_status"])

    def test_ndarray_operation_state_handled(self):
        row = _fe_row(
            type="golden_gate_workorder",
            wo_status="SUCCEEDED",
            operation_state=np.array(["SC"]),
        )
        result = _filter_and_enrich(_df(row))
        assert result.loc[result.index[0], "visual_status"] == "SUCCEEDED"

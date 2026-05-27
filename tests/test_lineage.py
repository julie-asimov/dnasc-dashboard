"""
Regression tests for LineageTransformer.bridge_lsp_lineage.

Before fix: root_to_req was built with set_index().to_dict() which kept
the last req_id when the same root appeared under multiple requests.
A non-null req_id could be silently overwritten by a null one.
"""
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnasc.transformers.lineage import LineageTransformer


def _bios_row(workorder_id, root_id, req_id, wo_type="gibson_workorder"):
    return {
        "workorder_id":       workorder_id,
        "root_work_order_id": root_id,
        "req_id":             req_id,
        "type":               wo_type,
        "source_lsp_process_id": None,
    }


def _lsp_row(workorder_id, source_id, req_id):
    return {
        "workorder_id":          workorder_id,
        "root_work_order_id":    workorder_id,
        "req_id":                req_id,
        "type":                  "lsp_workorder",
        "source_lsp_process_id": source_id,
    }


class TestRootToReqPreservesNonNull:
    def test_non_null_req_id_wins_over_null(self):
        """
        When the same root appears in two rows — one with req_id, one without —
        the non-null req_id should be used for the anti-kidnapping check.
        """
        bios = pd.DataFrame([
            _bios_row("gg-001", "gg-001", "REQ-A"),   # root with req_id
            _bios_row("gg-001", "gg-001", None),       # same root, no req_id
        ])
        lsp = pd.DataFrame([
            _lsp_row("lsp-001", "gg-001", "REQ-A"),   # LSP from same request
        ])
        result = LineageTransformer.bridge_lsp_lineage(bios, lsp)
        lsp_row = result[result["workorder_id"] == "lsp-001"].iloc[0]
        # Should inherit root from parent (same request), NOT self-root
        assert lsp_row["root_work_order_id"] == "gg-001"

    def test_anti_kidnapping_still_fires_with_different_req(self):
        """LSP from REQ-B whose source is in REQ-A should self-root."""
        bios = pd.DataFrame([
            _bios_row("gg-001", "gg-001", "REQ-A"),
        ])
        lsp = pd.DataFrame([
            _lsp_row("lsp-002", "gg-001", "REQ-B"),
        ])
        result = LineageTransformer.bridge_lsp_lineage(bios, lsp)
        lsp_row = result[result["workorder_id"] == "lsp-002"].iloc[0]
        # Different request — should self-root
        assert lsp_row["root_work_order_id"] == "lsp-002"

    def test_no_parent_self_roots(self):
        """LSP with no matching source should always self-root."""
        bios = pd.DataFrame([_bios_row("gg-001", "gg-001", "REQ-A")])
        lsp  = pd.DataFrame([_lsp_row("lsp-003", "nonexistent-id", "REQ-A")])
        result = LineageTransformer.bridge_lsp_lineage(bios, lsp)
        lsp_row = result[result["workorder_id"] == "lsp-003"].iloc[0]
        assert lsp_row["root_work_order_id"] == "lsp-003"

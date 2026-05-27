"""
Regression tests for the isinstance(x, list) vs numpy.ndarray bug.

Before fix: canceled LSPs with ndarray protocol_name were dropped as if they
had no ops. _no_ops / _no_ops2 also misidentified ndarray rows as op-less.
"""
import numpy as np
import pandas as pd
import pytest


def has_ops(x):
    """The fixed predicate used in pipeline.py canceled_no_work and _no_ops."""
    return isinstance(x, (list, np.ndarray)) and len(x) > 0


def no_ops(x):
    return not has_ops(x)


class TestHasOpsPredicate:
    def test_list_with_ops(self):
        assert has_ops(["Miniprep", "NGS"]) is True

    def test_list_empty(self):
        assert has_ops([]) is False

    def test_ndarray_with_ops(self):
        # This was the bug: ndarray returned False from isinstance(x, list)
        assert has_ops(np.array(["Miniprep", "NGS"])) is True

    def test_ndarray_empty(self):
        assert has_ops(np.array([])) is False

    def test_none(self):
        assert has_ops(None) is False

    def test_nan(self):
        assert has_ops(float("nan")) is False

    def test_scalar_string(self):
        # A plain string is not a list/ndarray — correct to treat as no-ops
        assert has_ops("Miniprep") is False


class TestCanceledLSPFilter:
    """canceled_no_work mask should NOT drop LSPs that have ops in ndarray form."""

    def _make_df(self, protocol_name_val):
        return pd.DataFrame([{
            "type":          "lsp_workorder",
            "wo_status":     "CANCELED",
            "protocol_name": protocol_name_val,
            "workorder_id":  "LSP-0001",
        }])

    def test_ndarray_ops_not_dropped(self):
        df = self._make_df(np.array(["LSP Receiving", "LSP Reviewing"]))
        canceled_no_work = (
            (df["type"] == "lsp_workorder") &
            (df["wo_status"].astype(str).str.upper() == "CANCELED") &
            (~df["protocol_name"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0))
        )
        assert canceled_no_work.sum() == 0, "ndarray LSP with ops should NOT be in canceled_no_work"

    def test_list_ops_not_dropped(self):
        df = self._make_df(["LSP Receiving"])
        canceled_no_work = (
            (df["type"] == "lsp_workorder") &
            (df["wo_status"].astype(str).str.upper() == "CANCELED") &
            (~df["protocol_name"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0))
        )
        assert canceled_no_work.sum() == 0

    def test_no_ops_is_dropped(self):
        df = self._make_df([])
        canceled_no_work = (
            (df["type"] == "lsp_workorder") &
            (df["wo_status"].astype(str).str.upper() == "CANCELED") &
            (~df["protocol_name"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0))
        )
        assert canceled_no_work.sum() == 1, "canceled LSP with no ops SHOULD be in canceled_no_work"

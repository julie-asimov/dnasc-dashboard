"""
Regression tests for _apply_colony_status_overrides / _override.

Before fix: int("") raised ValueError when total_colonies or seq_confirmed
was an empty string, crashing apply() for the whole colony subset.
"""
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnasc.pipeline import _apply_colony_status_overrides


def _make_row(**kwargs):
    defaults = {
        "type":            "gibson_workorder",
        "wo_status":       "SUCCEEDED",
        "visual_status":   "SUCCEEDED",
        "total_colonies":  3,
        "seq_confirmed":   1,
        "protocol_name":   ["Create Minipreps and Glycerol Stocks"],
        "operation_state": ["SUCCEEDED"],
        "is_fulfillment":  True,
    }
    defaults.update(kwargs)
    return defaults


def _run(rows):
    df = pd.DataFrame(rows)
    return _apply_colony_status_overrides(df)


class TestIntEmptyStringCrash:
    def test_empty_string_total_colonies_no_crash(self):
        """Before fix: int("") raised ValueError."""
        rows = [_make_row(total_colonies="", seq_confirmed=0)]
        result = _run(rows)  # should not raise
        assert len(result) == 1

    def test_empty_string_seq_confirmed_no_crash(self):
        rows = [_make_row(total_colonies=3, seq_confirmed="")]
        result = _run(rows)
        assert len(result) == 1

    def test_none_total_colonies_treated_as_zero(self):
        rows = [_make_row(total_colonies=None, seq_confirmed=None)]
        result = _run(rows)
        assert len(result) == 1

    def test_nan_total_colonies_treated_as_zero(self):
        rows = [_make_row(total_colonies=float("nan"), seq_confirmed=float("nan"))]
        result = _run(rows)
        assert len(result) == 1


class TestOverrideLogic:
    def test_succeeded_with_seq_stays_succeeded(self):
        rows = [_make_row(wo_status="SUCCEEDED", total_colonies=3, seq_confirmed=2)]
        result = _run(rows)
        assert result["visual_status"].iloc[0] == "SUCCEEDED"

    def test_succeeded_zero_colonies_no_ops_becomes_in_progress(self):
        # SUCCEEDED with 0 colonies and no ops → conservatively IN_PROGRESS
        # (transformation happened but nothing recorded yet)
        rows = [_make_row(wo_status="SUCCEEDED", total_colonies=0,
                          seq_confirmed=0, protocol_name=[], operation_state=[])]
        result = _run(rows)
        assert result["visual_status"].iloc[0] == "IN_PROGRESS"

    def test_non_colony_type_untouched(self):
        rows = [_make_row(type="lsp_workorder", wo_status="SUCCEEDED",
                          total_colonies=0, seq_confirmed=0)]
        result = _run(rows)
        # lsp_workorder is not a colony type — visual_status should be unchanged
        assert result["visual_status"].iloc[0] == "SUCCEEDED"

"""
Tests for Step 9 op_agg merge column-management logic.

The key invariant: after the op_agg merge, final_df["protocol_name"] must
be the list-valued column from op_agg — not a stale null/scalar from a
pre-existing column in final_df.

Before the fix, using suffixes=("", "_raw_op") with a pre-existing protocol_name
column in the left df would keep the null/scalar as "protocol_name" and put the
aggregated list in "protocol_name_raw_op", which the dedup guard would then
silently discard.
"""
import pandas as pd
import pytest


def _simulate_step9_merge(left_has_protocol_name: bool):
    """
    Simulate the Step 9 op_agg merge. Returns (result_df, used_old_approach).
    The new approach explicitly drops op columns before merging.
    """
    left_df = pd.DataFrame({
        "workorder_id": ["abc-123", "def-456"],
        "join_key":     ["abc-123", "xyz-789"],
    })
    if left_has_protocol_name:
        left_df["protocol_name"] = [None, None]

    op_agg = pd.DataFrame({
        "process_id":      ["abc-123"],
        "protocol_name":   [["NGS Sequence Confirmation", "DNA Quantification"]],
        "operation_state": [["SC", "SC"]],
    })

    # New approach: explicit drop before merge
    _op_merge_cols = ["protocol_name", "operation_state"]
    left_clean = left_df.drop(
        columns=[c for c in _op_merge_cols if c in left_df.columns],
        errors="ignore",
    )
    result = left_clean.merge(op_agg, left_on="join_key", right_on="process_id", how="left")
    return result


class TestStep9MergeColumnManagement:

    def test_matched_row_gets_list_from_op_agg(self):
        result = _simulate_step9_merge(left_has_protocol_name=False)
        matched = result[result["join_key"] == "abc-123"].iloc[0]
        assert matched["protocol_name"] == ["NGS Sequence Confirmation", "DNA Quantification"]

    def test_unmatched_row_gets_null(self):
        result = _simulate_step9_merge(left_has_protocol_name=False)
        unmatched = result[result["join_key"] == "xyz-789"].iloc[0]
        assert pd.isna(unmatched["protocol_name"])

    def test_pre_existing_null_column_does_not_shadow_op_agg_list(self):
        """
        Core regression test: if left_df already had protocol_name (null),
        the explicit drop must ensure op_agg's list wins after merge.
        Without the drop, the old suffix approach would keep the null column.
        """
        result = _simulate_step9_merge(left_has_protocol_name=True)
        matched = result[result["join_key"] == "abc-123"].iloc[0]
        assert isinstance(matched["protocol_name"], list), (
            "protocol_name should be the aggregated list from op_agg, not null"
        )
        assert "NGS Sequence Confirmation" in matched["protocol_name"]

    def test_no_duplicate_columns_after_merge(self):
        result = _simulate_step9_merge(left_has_protocol_name=True)
        assert not result.columns.duplicated().any(), (
            "Merge must not produce duplicate column names"
        )

    def test_suffix_approach_would_have_failed(self):
        """
        Documents the old (broken) behavior: suffixes=("", "_raw_op") keeps
        the null scalar as "protocol_name" when left_df already has that column.
        This test verifies our understanding of the problem, not the fix.
        """
        left_df = pd.DataFrame({
            "workorder_id": ["abc-123"],
            "join_key":     ["abc-123"],
            "protocol_name": [None],
        })
        op_agg = pd.DataFrame({
            "process_id":    ["abc-123"],
            "protocol_name": [["NGS Sequence Confirmation"]],
        })
        # Old approach: suffixes — left's null wins as "protocol_name"
        result = left_df.merge(op_agg, left_on="join_key", right_on="process_id",
                               how="left", suffixes=("", "_raw_op"))
        # The left's None is in "protocol_name"; the list is in "protocol_name_raw_op"
        assert result["protocol_name"].iloc[0] is None, (
            "Documenting old bug: suffix approach keeps the null column"
        )
        assert result["protocol_name_raw_op"].iloc[0] == ["NGS Sequence Confirmation"], (
            "Documenting old bug: the list ends up in the suffixed column"
        )

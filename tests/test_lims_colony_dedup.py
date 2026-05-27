"""
Regression tests for the lims.py colony dedup fix.

Before fix: a colony well with both plasmid_stock and strain records produced
2 join rows. available_colonies used count() on raw rows, so a single available
colony was reported as 2.
"""
import pandas as pd
import numpy as np
import pytest


def _compute_colony_summary(raw_df):
    """
    Mirrors the fixed logic from LIMSExtractor.get_colony_data.
    Isolated here so we can test it without a BQ connection.
    """
    raw_df = raw_df.copy()
    raw_df["colony_num_str"]    = raw_df["colony_number"].fillna(-1).astype(int).astype(str)
    raw_df["well_id_str"]       = raw_df["well_id"].astype(str)
    raw_df["well_col_combined"] = raw_df["well_id_str"] + ":" + raw_df["colony_num_str"]
    raw_df["plate_id_str"]      = "Plate" + raw_df["plate_id"].astype(str)
    raw_df["col_label"]         = "col" + raw_df["colony_num_str"]

    # Fixed: dedup to one row per (workorder_id, colony_number)
    unique_colonies = (
        raw_df[raw_df["colony_number"].notna()]
        .sort_values(["seq_confirmed", "available"], ascending=False)
        .drop_duplicates(subset=["workorder_id", "colony_number"], keep="first")
        .copy()
    )

    colony_summary = unique_colonies.groupby("workorder_id").agg(
        total_colonies    =("colony_number", "nunique"),
        available_colonies=("available", lambda x: x[x == True].count()),
        seq_confirmed     =("seq_confirmed", lambda x: (x == True).sum()),
    ).reset_index()

    # Fixed: filter null-colony rows from plate labels
    plate_info = (
        raw_df[raw_df["colony_number"].notna()]
        .groupby(["workorder_id", "plate_id_str"])["col_label"]
        .apply(lambda x: ", ".join(sorted(x.unique())))
        .reset_index()
    )

    return colony_summary, plate_info


def _make_raw(workorder_id, colony_number, available, well_id=1, plate_id=100, n_join_rows=1, seq_confirmed=None):
    """Create n_join_rows rows for the same colony (simulates multi-join fan-out)."""
    return [
        {
            "workorder_id":   workorder_id,
            "colony_number":  colony_number,
            "available":      available,
            "seq_confirmed":  seq_confirmed,
            "well_id":        well_id,
            "plate_id":       plate_id,
            "plate_protocol": "Miniprep",
        }
        for _ in range(n_join_rows)
    ]


class TestColonyDedup:
    def test_single_colony_single_row(self):
        raw = pd.DataFrame(_make_raw("WO-1", 1, True, n_join_rows=1))
        summary, _ = _compute_colony_summary(raw)
        assert summary.loc[summary["workorder_id"] == "WO-1", "total_colonies"].iloc[0] == 1
        assert summary.loc[summary["workorder_id"] == "WO-1", "available_colonies"].iloc[0] == 1

    def test_single_colony_two_join_rows_counts_as_one(self):
        """The core bug: same colony appearing twice should not double-count."""
        raw = pd.DataFrame(_make_raw("WO-1", 1, True, n_join_rows=2))
        summary, _ = _compute_colony_summary(raw)
        assert summary.loc[summary["workorder_id"] == "WO-1", "total_colonies"].iloc[0] == 1
        assert summary.loc[summary["workorder_id"] == "WO-1", "available_colonies"].iloc[0] == 1

    def test_available_true_wins_when_rows_disagree(self):
        """If one row has available=True and another False for same colony, True wins."""
        rows = (
            _make_raw("WO-1", 1, True,  n_join_rows=1) +
            _make_raw("WO-1", 1, False, n_join_rows=1)
        )
        raw = pd.DataFrame(rows)
        summary, _ = _compute_colony_summary(raw)
        assert summary.loc[summary["workorder_id"] == "WO-1", "available_colonies"].iloc[0] == 1

    def test_two_distinct_colonies_counted_separately(self):
        rows = (
            _make_raw("WO-1", 1, True,  well_id=1, n_join_rows=2) +
            _make_raw("WO-1", 2, False, well_id=2, n_join_rows=2)
        )
        raw = pd.DataFrame(rows)
        summary, _ = _compute_colony_summary(raw)
        assert summary.loc[summary["workorder_id"] == "WO-1", "total_colonies"].iloc[0] == 2
        assert summary.loc[summary["workorder_id"] == "WO-1", "available_colonies"].iloc[0] == 1


    def test_seq_confirmed_true_wins_over_nan(self):
        """plasmid_stock row (seq_confirmed=True) must win over strain row (seq_confirmed=NaN)
        even when both have the same available value — the bug reported in code review."""
        rows = [
            _make_raw("WO-1", 5, True, seq_confirmed=True)[0],   # plasmid_stock row
            _make_raw("WO-1", 5, True, seq_confirmed=None)[0],   # strain row
        ]
        raw = pd.DataFrame(rows)
        summary, _ = _compute_colony_summary(raw)
        assert summary.loc[summary["workorder_id"] == "WO-1", "seq_confirmed"].iloc[0] == 1

    def test_seq_confirmed_not_lost_across_multiple_colonies(self):
        """seq_confirmed count is correct across a mix of sequenced and unsequenced colonies."""
        rows = (
            _make_raw("WO-1", 1, True, seq_confirmed=True)[0:1] +   # confirmed
            _make_raw("WO-1", 1, True, seq_confirmed=None)[0:1] +   # fan-out, should lose tie
            _make_raw("WO-1", 2, True, seq_confirmed=None)[0:1] +   # not confirmed
        [])
        raw = pd.DataFrame(rows)
        summary, _ = _compute_colony_summary(raw)
        assert summary.loc[summary["workorder_id"] == "WO-1", "total_colonies"].iloc[0] == 2
        assert summary.loc[summary["workorder_id"] == "WO-1", "seq_confirmed"].iloc[0] == 1


class TestPlateLabelNullFilter:
    def test_no_col_minus_one_in_plate_labels(self):
        """Null-colony rows (plate-only, no colony picked) must not produce 'col-1' labels."""
        rows = (
            _make_raw("WO-1", 1,    True,  well_id=1) +  # real colony
            [{"workorder_id": "WO-1", "colony_number": None,  # plate row, no colony
              "available": False, "well_id": 2, "plate_id": 100, "plate_protocol": "Miniprep"}]
        )
        raw = pd.DataFrame(rows)
        _, plate_info = _compute_colony_summary(raw)
        labels = plate_info[plate_info["workorder_id"] == "WO-1"]["col_label"].values
        assert all("col-1" not in str(l) for l in labels), f"Found col-1 in plate labels: {labels}"

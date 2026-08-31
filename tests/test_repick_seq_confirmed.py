"""
_detect_colony_repicks: repick colony counting.

repick_seq_confirmed is what lets a confirmed repick read as SUCCEEDED instead
of falling back to the original pick's count (0 by construction, since a repick
only fires on a FAILED parent). Live data currently has no confirmed repicks, so
these tests are the only proof the non-zero path works.
"""
import json
from unittest.mock import patch

import pandas as pd
import pytest

from dnasc.pipeline import _detect_colony_repicks


def _parent_df():
    """One FAILED Gibson with an NGS Fragment Analyzer op — a repick candidate."""
    return pd.DataFrame([{
        "workorder_id":       "WO-1",
        "type":               "gibson_workorder",
        "visual_status":      "FAILED",
        "total_colonies":     6,
        "seq_confirmed":      0,
        "protocol_name":      ["NGS Sequence Confirmation"],
        "operation_state":    ["FA"],
        "operation_start":    ["2026-08-01 10:00:00"],
        "operation_ready":    ["2026-08-01 10:00:00"],
        "job_id":             [1],
        "all_protocol_plates": "{}",
    }])


def _repick_plates(rows):
    return pd.DataFrame(rows)


def _run(df, plates):
    with patch("dnasc.extractors.lims.LIMSExtractor.get_repick_plates",
               return_value=plates):
        return _detect_colony_repicks(df)


class TestRepickSeqConfirmed:

    def test_counts_distinct_confirmed_colonies(self):
        """
        A colony is held in several wells (overnight, glycerol, miniprep) and only
        some carry the flag — count distinct colonies, not rows.
        """
        plates = _repick_plates([
            {"workorder_id": "WO-1", "plate_id": 100, "plate_protocol": "Bank Overnights",
             "plate_created_at": pd.Timestamp("2026-08-05", tz="UTC"), "colony_number": 1, "seq_confirmed": True},
            {"workorder_id": "WO-1", "plate_id": 101, "plate_protocol": "Miniprep",
             "plate_created_at": pd.Timestamp("2026-08-05", tz="UTC"), "colony_number": 1, "seq_confirmed": True},
            {"workorder_id": "WO-1", "plate_id": 100, "plate_protocol": "Bank Overnights",
             "plate_created_at": pd.Timestamp("2026-08-05", tz="UTC"), "colony_number": 2, "seq_confirmed": None},
            {"workorder_id": "WO-1", "plate_id": 100, "plate_protocol": "Bank Overnights",
             "plate_created_at": pd.Timestamp("2026-08-05", tz="UTC"), "colony_number": 3, "seq_confirmed": True},
        ])
        out = _run(_parent_df(), plates)
        assert out["repick_total_colonies"].iloc[0] == 3
        assert out["repick_seq_confirmed"].iloc[0] == 2   # colonies 1 and 3, not 4 rows

    def test_no_confirmed_colonies_is_zero(self):
        plates = _repick_plates([
            {"workorder_id": "WO-1", "plate_id": 100, "plate_protocol": "Miniprep",
             "plate_created_at": pd.Timestamp("2026-08-05", tz="UTC"), "colony_number": n, "seq_confirmed": None}
            for n in (1, 2, 3, 4)
        ])
        out = _run(_parent_df(), plates)
        assert out["repick_total_colonies"].iloc[0] == 4
        assert out["repick_seq_confirmed"].iloc[0] == 0

    def test_missing_seq_confirmed_column_does_not_crash(self):
        """Guards the older query shape, which returned no seq_confirmed column."""
        plates = _repick_plates([
            {"workorder_id": "WO-1", "plate_id": 100, "plate_protocol": "Miniprep",
             "plate_created_at": pd.Timestamp("2026-08-05", tz="UTC"), "colony_number": 1},
        ])
        out = _run(_parent_df(), plates)
        assert out["repick_seq_confirmed"].iloc[0] == 0

    def test_repick_plate_ids_deduped(self):
        """
        The query returns one row per (plate, colony), so a 6-colony plate used to
        be listed six times in the Manual Repick plate string.
        """
        plates = _repick_plates([
            {"workorder_id": "WO-1", "plate_id": 100, "plate_protocol": "Miniprep",
             "plate_created_at": pd.Timestamp("2026-08-05", tz="UTC"), "colony_number": n, "seq_confirmed": None}
            for n in (1, 2, 3)
        ])
        out = _run(_parent_df(), plates)
        listed = json.loads(out["all_protocol_plates"].iloc[0])["Manual Repick"]
        assert listed == "100"

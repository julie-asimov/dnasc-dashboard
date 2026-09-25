"""
_repick_status: the post-repick op slice is taken by TIME, not list position.

resolve_downstream_plates appends whatever it discovers to the END of the op
lists, and for a repick workorder it re-traces the ORIGINAL miniprep plate — so
the original pick's Rearray/Quant (SC) and NGS (FA) land after the repick marker
even though they ran days before it. Sliced positionally, a live repick resolved
straight back to FAILED off the original pick's NGS, which then made the request
read STALLED while its operation line read "REPICK MINIPREP: RUNNING"
(pAI-25730, Sept 2026).
"""
import pandas as pd

from dnasc import protocols as proto
from dnasc.pipeline import _repick_status


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def _row(protocol_name, operation_state, operation_start, **over):
    row = {
        "visual_status":         "RUNNING",
        "repick_total_colonies": 1,
        "repick_seq_confirmed":  0,
        "seq_confirmed":         0,
        "protocol_name":         protocol_name,
        "operation_state":       operation_state,
        "operation_start":       operation_start,
    }
    row.update(over)
    return row


class TestRepickSlice:

    def test_original_pick_ops_appended_after_repick_do_not_fail_it(self):
        """The pAI-25730 shape: repick is the newest thing that happened."""
        row = _row(
            [proto.GOLDEN_GATE, proto.MINIPREP, proto.REARRAY, proto.DNA_QUANT,
             proto.NGS, proto.REPICK,
             # re-appended by resolve_downstream_plates, out of time order:
             proto.REARRAY, proto.DNA_QUANT, proto.NGS],
            ["SC", "SC", "SC", "SC", "FA", "RU",
             "SC", "SC", "FA"],
            [_ts("2026-09-18 15:54"), _ts("2026-09-22 09:52"), _ts("2026-09-23 14:47"),
             _ts("2026-09-23 15:17"), _ts("2026-09-24 08:29"), _ts("2026-09-24 13:13"),
             _ts("2026-09-23 14:47"), _ts("2026-09-23 15:17"), _ts("2026-09-24 08:29")],
        )
        assert _repick_status(row) == "RUNNING"

    def test_repick_own_ngs_failure_still_fails(self):
        """A genuine post-repick NGS failure must still read FAILED."""
        row = _row(
            [proto.NGS, proto.REPICK, proto.REARRAY, proto.NGS],
            ["FA", "RU", "SC", "FA"],
            [_ts("2026-09-24 08:29"), _ts("2026-09-24 13:13"),
             _ts("2026-09-26 09:00"), _ts("2026-09-27 09:00")],
        )
        assert _repick_status(row) == "FAILED"

    def test_repick_own_ngs_confirmed_succeeds(self):
        row = _row(
            [proto.NGS, proto.REPICK, proto.REARRAY, proto.NGS],
            ["FA", "RU", "SC", "SC"],
            [_ts("2026-09-24 08:29"), _ts("2026-09-24 13:13"),
             _ts("2026-09-26 09:00"), _ts("2026-09-27 09:00")],
            repick_seq_confirmed=1,
        )
        assert _repick_status(row) == "SUCCEEDED"

    def test_post_repick_op_still_running_wins_over_earlier_verdict(self):
        row = _row(
            [proto.REPICK, proto.REARRAY, proto.NGS],
            ["RU", "SC", "RU"],
            [_ts("2026-09-24 13:13"), _ts("2026-09-26 09:00"), _ts("2026-09-27 09:00")],
        )
        assert _repick_status(row) == "RUNNING"

    def test_naive_pre_repick_ops_are_excluded(self):
        """
        Op lists mix tz-aware BQ timestamps with naive strings. A naive original-pick
        op must still compare as earlier than the repick, not blow up and not count.
        """
        row = _row(
            [proto.REARRAY, proto.NGS, proto.REPICK],
            ["SC", "FA", "RU"],
            ["2026-09-23 14:47:00", "2026-09-24 08:29:00", _ts("2026-09-24 13:13")],
        )
        assert _repick_status(row) == "RUNNING"

    def test_naive_post_repick_ops_are_included(self):
        row = _row(
            [proto.NGS, proto.REPICK, proto.REARRAY, proto.NGS],
            ["FA", "RU", "SC", "FA"],
            [_ts("2026-09-24 08:29"), _ts("2026-09-24 13:13"),
             "2026-09-26 09:00:00", "2026-09-27 09:00:00"],
        )
        assert _repick_status(row) == "FAILED"

    def test_missing_repick_timestamp_falls_back_to_position(self):
        row = _row(
            [proto.NGS, proto.REPICK, proto.REARRAY, proto.NGS],
            ["FA", "RU", "SC", "FA"],
            [_ts("2026-09-24 08:29"), None, _ts("2026-09-26 09:00"), _ts("2026-09-27 09:00")],
        )
        assert _repick_status(row) == "FAILED"

    def test_nothing_after_the_repick_is_running(self):
        row = _row(
            [proto.NGS, proto.REPICK],
            ["FA", "RU"],
            [_ts("2026-09-24 08:29"), _ts("2026-09-24 13:13")],
        )
        assert _repick_status(row) == "RUNNING"

    def test_non_running_rows_are_untouched(self):
        row = _row([proto.REPICK], ["RU"], [_ts("2026-09-24 13:13")], visual_status="SUCCEEDED")
        assert _repick_status(row) == "SUCCEEDED"

    def test_no_repick_colonies_is_untouched(self):
        row = _row([proto.REPICK], ["RU"], [_ts("2026-09-24 13:13")], repick_total_colonies=0)
        assert _repick_status(row) == "RUNNING"

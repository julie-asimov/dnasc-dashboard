"""
Regression tests for the due-date sheet reader.

The live sheet failed in a way no test covered: one row carried a 4th cell (a
pasted link) against a 3-column header, so
    pd.DataFrame(values[1:], columns=values[0])
raised "3 columns passed, passed data had 4 columns". The broad `except` reported
that as "Sheets unavailable", the CSV fallback was absent on the server, and
due_dates.json came out EMPTY — every experiment on the dashboard lost its due
date. The renderer then listed all of them as "missing an Asana due date", and
that manufactured list was appended to the sheet, which grew to 1412 rows of
which 1363 had a blank column A.

Covers the three defects that chain together:
  1. a ragged row must not abort the read
  2. dedup must see names in ANY column, not just A
  3. a failed dedup read must abort the append rather than treat the sheet as empty
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dnasc.extractors import sheets

HEADER = ["experiment_name", "date_sequence_transferred", "due_date_in_asana"]


def _mock_get(status=200, values=None):
    """Patch requests.get inside sheets.py to return one canned Sheets response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"values": values or []}
    resp.text = "mock error body"
    return resp


def _read_sheet(values):
    """Run _try_google_sheets against canned values, with auth stubbed out."""
    fake_creds = MagicMock()
    fake_creds.token = "tok"
    with patch("google.auth.default", return_value=(fake_creds, "proj")), \
         patch("google.auth.transport.requests.Request"), \
         patch("requests.get", return_value=_mock_get(200, values)):
        return sheets._try_google_sheets()


class TestRaggedRows:
    def test_extra_cell_does_not_abort_the_read(self):
        """The exact live failure: a pasted URL in a 4th cell.

        Before the fix this returned None, which sent the caller to a CSV that
        did not exist, which produced zero due dates.
        """
        values = [
            HEADER,
            ["A762 - BHR AAV Campaign", "2026-07-14", "2026-09-28"],
            ["", "", "", "https://docs.google.com/spreadsheets/d/abc/edit"],
            ["A764-LLY005_CLD campaign", "2026-07-30", "2026-08-31"],
        ]
        df = _read_sheet(values)
        assert df is not None, "a single stray cell must not fail the whole read"
        assert len(df) == 3
        assert list(df.columns)[:3] == HEADER

    def test_dates_survive_a_ragged_sheet(self):
        """Every named row with a date must still reach the caller."""
        values = [
            HEADER,
            ["A762", "2026-07-14", "2026-09-28"],
            ["", "", "", "junk"],
            ["A773_v2", "2026-08-04", "2026-09-08"],
        ]
        df = _read_sheet(values)
        got = {
            r["experiment_name"]: r["due_date_in_asana"]
            for _, r in df.iterrows() if str(r["experiment_name"]).strip()
        }
        assert got == {"A762": "2026-09-28", "A773_v2": "2026-09-08"}

    def test_short_rows_are_padded(self):
        """Sheets omits trailing empties, so rows are commonly shorter."""
        values = [HEADER, ["A762"], ["A764", "2026-07-30"]]
        df = _read_sheet(values)
        assert df is not None
        assert len(df) == 2
        assert df.iloc[0]["due_date_in_asana"] == ""

    def test_blank_name_rows_are_dropped_by_fetch(self):
        """Wrong-column junk rows must not become due-date entries.

        1363 of the live sheet's rows look like this. They have no name in
        column A, so they must be skipped rather than parsed as experiments.
        """
        values = [
            HEADER,
            ["A762", "2026-07-14", "2026-09-28"],
            ["", "", "A786-wave2_SRK-1323 > Destination vectors"],  # shifted junk
        ]
        with patch.object(sheets, "_try_google_sheets", return_value=pd.DataFrame(
                [row + [""] * (3 - len(row)) for row in values[1:]], columns=HEADER)), \
             patch.object(sheets, "_save"):
            result = sheets.fetch_due_dates()
        assert list(result) == ["A762"], "shifted junk must not become an entry"


class TestAppendSafety:
    def _run_append(self, read_status, read_values, post_status=200):
        fake_creds = MagicMock()
        fake_creds.token = "tok"
        post = MagicMock()
        post.status_code = post_status
        post.text = ""
        with patch("google.auth.default", return_value=(fake_creds, "proj")), \
             patch("google.auth.transport.requests.Request"), \
             patch("requests.get", return_value=_mock_get(read_status, read_values)), \
             patch("requests.post", return_value=post) as mock_post:
            res = sheets.append_experiment_names(["A786-wave2", "NewOne"])
        return res, mock_post

    def test_failed_dedup_read_aborts_the_append(self):
        """A failed read must never be treated as 'the sheet is empty'."""
        res, mock_post = self._run_append(403, None)
        assert res["ok"] is False
        assert "dedup read failed" in (res["error"] or "")
        assert not mock_post.called, "must not write when it cannot dedup"

    def test_dedup_sees_names_outside_column_a(self):
        """The live bug: names sitting in column C were invisible to dedup.

        A786-wave2 already exists, shifted two columns over. Only NewOne is new.
        """
        values = [
            HEADER,
            ["RealExperiment", "2026-01-01", "2026-02-01"],
            ["", "", "A786-wave2"],       # shifted, blank column A
        ]
        res, mock_post = self._run_append(200, values)
        assert res["ok"] is True
        assert res["appended"] == ["NewOne"], (
            "A786-wave2 exists in column C and must be recognised as a duplicate"
        )
        assert "A786-wave2" in res["skipped_existing"]

    def test_nothing_to_add_makes_no_write(self):
        values = [HEADER, ["", "", "A786-wave2"], ["", "NewOne", ""]]
        res, mock_post = self._run_append(200, values)
        assert res["ok"] is True
        assert res["appended"] == []
        assert not mock_post.called

    def test_empty_input_is_a_noop(self):
        res = sheets.append_experiment_names([])
        assert res["ok"] is True
        assert res["appended"] == []

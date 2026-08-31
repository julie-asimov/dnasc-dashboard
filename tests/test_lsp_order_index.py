"""
Tests for the LSP order handle — the "<job id>_<index>" the team now names an
LSP by ("9560: 8") instead of the batch ID.

Covers _attach_lsp_order_index (which of an LSP row's several identity columns
resolves the handle) and the two contracts the renderer/search rely on: the
handle must reach the row, and "9560: 8" typed into the search box must fold
onto the "9560_8" the badge renders as.
"""
import re

import pandas as pd
import pytest

from dnasc.pipeline import _attach_lsp_order_index


def _idx_df(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "lsp_batch_id", "lsp_order_index",
            "lsp_order_number", "lsp_order_process_id",
        ],
    )


# The real shape: one row per batch, index prefixed by the LSP Order job id.
IDX = _idx_df([
    ("LSP-11802", "9560_1", "9560", "0614f7f5-b1f9-491c-8eb0-ed796438d5c9"),
    ("LSP-11809", "9560_8", "9560", "74a433ad-0123-4771-aa73-63b04a38551a"),
    ("LSP-11724", "9559_5", "9559", "6ce1ac29-3794-4959-99d4-1eb55d254330"),
    # Outsourced prep: operator overrode Order Number with the vendor's.
    ("LSP-11118", "8806_1", "Batch_8806_Azenta_30-1343539039",
     "784915b9-f4bc-4ed7-b207-c7c5bfdc1faf"),
])


# ── _attach_lsp_order_index ───────────────────────────────────────────────────

class TestAttachLSPOrderIndex:
    """An LSP row can be keyed four ways depending on how it was recovered."""

    def test_matches_on_lims_batch_id(self):
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": "LSP-11809",
            "bios_batch_id": None, "workorder_id": "some-uuid",
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert out.loc[0, "lsp_order_index"] == "9560_8"
        assert out.loc[0, "lsp_order_number"] == "9560"

    def test_falls_back_to_bios_batch_id(self):
        """Real workorder that lost the LIMS aliquot join."""
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": None,
            "bios_batch_id": "LSP-11724", "workorder_id": "some-uuid",
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert out.loc[0, "lsp_order_index"] == "9559_5"

    def test_falls_back_to_synthetic_workorder_id(self):
        """Synthetic LSP rows carry the batch ID as their workorder_id."""
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": None,
            "bios_batch_id": None, "workorder_id": "LSP-11802",
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert out.loc[0, "lsp_order_index"] == "9560_1"

    def test_falls_back_to_process_id(self):
        """Last resort: the LSP Order op's Process param is the workorder UUID."""
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": None,
            "bios_batch_id": None,
            "workorder_id": "74a433ad-0123-4771-aa73-63b04a38551a",
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert out.loc[0, "lsp_order_index"] == "9560_8"

    def test_batch_id_match_is_case_insensitive(self):
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": " lsp-11809 ",
            "bios_batch_id": None, "workorder_id": None,
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert out.loc[0, "lsp_order_index"] == "9560_8"

    def test_vendor_override_keeps_job_prefix_on_the_index(self):
        """Order Number can be the vendor's, but the index stays job-prefixed."""
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": "LSP-11118",
            "bios_batch_id": None, "workorder_id": None,
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert out.loc[0, "lsp_order_index"] == "8806_1"
        assert out.loc[0, "lsp_order_number"] == "Batch_8806_Azenta_30-1343539039"

    def test_unmatched_row_gets_null_not_a_wrong_handle(self):
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": "LSP-99999",
            "bios_batch_id": None, "workorder_id": "unrelated-uuid",
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert pd.isna(out.loc[0, "lsp_order_index"])

    def test_non_lsp_rows_are_untouched(self):
        df = pd.DataFrame([{
            "type": "gibson_workorder", "lsp_batch_id": None,
            "bios_batch_id": None, "workorder_id": "gibson-uuid",
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert pd.isna(out.loc[0, "lsp_order_index"])

    def test_columns_exist_even_when_mapping_is_empty(self):
        """The renderer does row.get('lsp_order_index') unconditionally."""
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": "LSP-11809",
            "bios_batch_id": None, "workorder_id": None,
        }])
        out = _attach_lsp_order_index(df, _idx_df([]))
        assert "lsp_order_index" in out.columns
        assert "lsp_order_number" in out.columns
        assert pd.isna(out.loc[0, "lsp_order_index"])

    def test_missing_identity_columns_do_not_raise(self):
        """Older frames may not carry bios_batch_id at all."""
        df = pd.DataFrame([{"type": "lsp_workorder", "lsp_batch_id": "LSP-11724"}])
        out = _attach_lsp_order_index(df, IDX)
        assert out.loc[0, "lsp_order_index"] == "9559_5"

    def test_lims_batch_id_wins_over_a_conflicting_process_id(self):
        """First key that resolves wins; batch ID is the most trustworthy."""
        df = pd.DataFrame([{
            "type": "lsp_workorder", "lsp_batch_id": "LSP-11802",
            "bios_batch_id": None,
            "workorder_id": "74a433ad-0123-4771-aa73-63b04a38551a",  # LSP-11809's
        }])
        out = _attach_lsp_order_index(df, IDX)
        assert out.loc[0, "lsp_order_index"] == "9560_1"


# ── Search-term normalisation ─────────────────────────────────────────────────

# Mirrors _normSearch() in renderer/dashboard.py. Kept in step by
# TestSearchNormalisationSourceContract below.
_NORM_RE = re.compile(r"^(\d{3,6})\s*[:\-\s]\s*(\d{1,3})$")


def _norm_search(v: str) -> str:
    return _NORM_RE.sub(r"\1_\2", v.lower().strip())


class TestSearchNormalisation:
    """The team writes "9560: 8"; the badge renders "9560_8"."""

    @pytest.mark.parametrize("typed", ["9560: 8", "9560:8", "9560 8", "9560-8", " 9560 : 8 "])
    def test_spoken_forms_fold_onto_the_rendered_handle(self, typed):
        assert _norm_search(typed) == "9560_8"

    def test_already_normalised_is_left_alone(self):
        assert _norm_search("9560_8") == "9560_8"

    def test_bare_job_id_still_matches_every_row_in_the_job(self):
        """Searching "9560" must stay a prefix of "9560_8", not become "9560_"."""
        assert _norm_search("9560") == "9560"
        assert "9560_8".startswith(_norm_search("9560"))

    @pytest.mark.parametrize("typed", [
        "pAI-22509",                 # stock IDs must not be mangled
        "LSP-11809",
        "A773-1",                    # experiment names with digit-dash-digit
        "9560_8 extra",
    ])
    def test_non_handles_pass_through(self, typed):
        assert _norm_search(typed) == typed.lower().strip()


class TestSearchNormalisationSourceContract:
    """The JS in dashboard.py is the real implementation — keep this in step."""

    def test_js_regex_matches_the_python_mirror(self):
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "dnasc" / "renderer" / "dashboard.py"
        text = src.read_text()
        assert "function _normSearch(" in text, "_normSearch helper was renamed/removed"
        assert r"/^(\d{3,6})\s*[:\-\s]\s*(\d{1,3})$/" in text, (
            "the JS search-normalisation regex changed — update _NORM_RE here too"
        )
        # Both entry points must normalise, or the CSV/List export drifts
        # from what the table highlighted.
        assert text.count("_normSearch(document.getElementById('search_box').value)") == 2

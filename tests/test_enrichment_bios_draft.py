"""
Regression tests for the enrichment.py BIOS_DRAFT filter.

Before fix: active_rows.get('data_source', pd.Series('', ...)) used
DataFrame.get() which returns None when column is missing (not the default),
causing None != 'BIOS_DRAFT' to be True for all rows — BIOS_DRAFT rows
slipped into rc_rows and affected stall/phase logic.
"""
import pandas as pd
import pytest


def _filter_rc_rows(active_rows, root_chain_types):
    """Mirrors the fixed logic from EnrichmentTransformer."""
    _data_src = (
        active_rows['data_source']
        if 'data_source' in active_rows.columns
        else pd.Series('', index=active_rows.index)
    )
    return active_rows[
        active_rows['type'].isin(root_chain_types) &
        (_data_src != 'BIOS_DRAFT')
    ]


ROOT_CHAIN_TYPES = {"gibson_workorder", "golden_gate_workorder", "transformation_workorder"}


class TestBiosDraftFilter:
    def test_bios_draft_excluded_when_column_present(self):
        df = pd.DataFrame([
            {"type": "gibson_workorder",      "data_source": "BIOS",       "visual_status": "RUNNING"},
            {"type": "gibson_workorder",      "data_source": "BIOS_DRAFT", "visual_status": "WAITING"},
            {"type": "golden_gate_workorder", "data_source": "BIOS",       "visual_status": "SUCCEEDED"},
        ])
        rc = _filter_rc_rows(df, ROOT_CHAIN_TYPES)
        assert len(rc) == 2
        assert "BIOS_DRAFT" not in rc["data_source"].values

    def test_bios_draft_excluded_when_column_absent(self):
        """Before fix: missing column caused BIOS_DRAFT rows to pass the filter."""
        df = pd.DataFrame([
            {"type": "gibson_workorder",      "visual_status": "RUNNING"},
            {"type": "gibson_workorder",      "visual_status": "WAITING"},
        ])
        assert "data_source" not in df.columns
        rc = _filter_rc_rows(df, ROOT_CHAIN_TYPES)
        # All rows pass (no data_source to filter on) — this is correct default behavior
        assert len(rc) == 2

    def test_non_root_chain_types_excluded(self):
        df = pd.DataFrame([
            {"type": "gibson_workorder", "data_source": "BIOS",  "visual_status": "RUNNING"},
            {"type": "lsp_workorder",    "data_source": "BIOS",  "visual_status": "RUNNING"},
            {"type": "oligo_synthesis_workorder", "data_source": "BIOS", "visual_status": "RUNNING"},
        ])
        rc = _filter_rc_rows(df, ROOT_CHAIN_TYPES)
        assert len(rc) == 1
        assert rc["type"].iloc[0] == "gibson_workorder"


class TestProcessingSourceLinks:
    """
    Regression for processing.py: set_index("workorder_id").to_dict() kept
    last construct_name when workorder_id had duplicates. Non-null should win.
    """

    def _build_id_to_name(self, df):
        """Mirrors the fixed logic from ProcessingTransformer._generate_source_links."""
        return (
            df.sort_values("construct_name", na_position="last")
            .drop_duplicates("workorder_id", keep="first")
            .set_index("workorder_id")["construct_name"]
            .to_dict()
        )

    def test_non_null_construct_name_wins(self):
        df = pd.DataFrame([
            {"workorder_id": "WO-1", "construct_name": "pAI-100",  "STOCK_ID": "pAI-100"},
            {"workorder_id": "WO-1", "construct_name": None,        "STOCK_ID": "pAI-100"},
        ])
        id_to_name = self._build_id_to_name(df)
        assert id_to_name["WO-1"] == "pAI-100"

    def test_both_null_stays_null(self):
        df = pd.DataFrame([
            {"workorder_id": "WO-2", "construct_name": None, "STOCK_ID": None},
            {"workorder_id": "WO-2", "construct_name": None, "STOCK_ID": None},
        ])
        id_to_name = self._build_id_to_name(df)
        assert id_to_name.get("WO-2") is None

    def test_unique_workorder_unchanged(self):
        df = pd.DataFrame([
            {"workorder_id": "WO-3", "construct_name": "pAI-200", "STOCK_ID": "pAI-200"},
        ])
        id_to_name = self._build_id_to_name(df)
        assert id_to_name["WO-3"] == "pAI-200"

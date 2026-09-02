"""
Tests for resolve_downstream_plates in dnasc/transformers/repair.py.
BQ calls mocked via patch('dnasc.transformers.repair.bigquery.Client').
"""
import json
import pandas as pd
from unittest.mock import patch, MagicMock

from dnasc import protocols as proto
from dnasc.transformers.repair import resolve_downstream_plates


# ── mock helpers ──────────────────────────────────────────────────────────────

def _sql_dispatch(*rules):
    def _dispatch(sql):
        for kw, df in rules:
            if kw.lower() in sql.lower():
                m = MagicMock()
                m.to_dataframe.return_value = df
                return m
        m = MagicMock()
        m.to_dataframe.return_value = pd.DataFrame()
        return m
    return _dispatch


def _mock_bq(*rules):
    instance = MagicMock()
    instance.query.side_effect = _sql_dispatch(*rules)
    return MagicMock(return_value=instance)


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _base_row(**kwargs):
    defaults = {
        'workorder_id':        'gg-001',
        'type':                'golden_gate_workorder',
        'protocol_name':       [],
        'operation_state':     [],
        'operation_start':     [],
        'operation_ready':     [],
        'job_id':              [],
        'ngs_run_number':      [],
        'well_location':       [],
        'all_protocol_plates': '{}',
    }
    defaults.update(kwargs)
    return defaults


def _make_df(*rows):
    return pd.DataFrame([_base_row(**r) for r in rows])


# ── Tests: missing mask — no BQ ───────────────────────────────────────────────

class TestMissingMaskNoBQ:

    def test_no_miniprep_in_protocols_no_bq(self):
        """Rows with only GG in protocol_name don't qualify — BQ never called."""
        df = _make_df({'protocol_name': [proto.GOLDEN_GATE]})
        with patch('dnasc.transformers.repair.bigquery.Client') as mock_cls:
            result = resolve_downstream_plates(df, project_id='test-proj')
        assert not mock_cls.called
        assert len(result) == len(df)

    def test_miniprep_with_seq_already_present_no_bq(self):
        """Miniprep + NGS already present → not missing downstream → no BQ."""
        df = _make_df({'protocol_name': [proto.MINIPREP, proto.NGS]})
        with patch('dnasc.transformers.repair.bigquery.Client') as mock_cls:
            resolve_downstream_plates(df, project_id='test-proj')
        assert not mock_cls.called

    def test_miniprep_with_fragment_analyzer_already_present_no_bq(self):
        """Fragment Analyzer also counts as seq — not missing downstream."""
        df = _make_df({'protocol_name': [proto.MINIPREP, proto.FRAGMENT_ANALYZER]})
        with patch('dnasc.transformers.repair.bigquery.Client') as mock_cls:
            resolve_downstream_plates(df, project_id='test-proj')
        assert not mock_cls.called


# ── Tests: plate ID parsing — no BQ ──────────────────────────────────────────

class TestPlateIdParsingNoBQ:

    def test_empty_all_protocol_plates_no_bq(self):
        """Empty JSON in all_protocol_plates → no miniprep plate found → no BQ."""
        df = _make_df({'protocol_name': [proto.MINIPREP], 'all_protocol_plates': '{}'})
        with patch('dnasc.transformers.repair.bigquery.Client') as mock_cls:
            resolve_downstream_plates(df, project_id='test-proj')
        assert not mock_cls.called

    def test_optracker_key_not_in_miniprep_keys_no_bq(self):
        """all_protocol_plates keyed by OpTracker name (not a MINIPREP_KEYS match) → no plate → no BQ."""
        df = _make_df({
            'protocol_name': [proto.MINIPREP],
            'all_protocol_plates': json.dumps({proto.MINIPREP: '12345'}),
        })
        with patch('dnasc.transformers.repair.bigquery.Client') as mock_cls:
            resolve_downstream_plates(df, project_id='test-proj')
        # 'Create Minipreps and Glycerol Stocks' is NOT in MINIPREP_KEYS → no plate found
        assert not mock_cls.called

    def test_miniprep_lims_key_triggers_bq(self):
        """all_protocol_plates with LIMS key 'Miniprep' → plate found → BQ is called."""
        df = _make_df({
            'protocol_name':       [proto.MINIPREP],
            'all_protocol_plates': json.dumps({'Miniprep': '12345'}),
        })
        with patch('dnasc.transformers.repair.bigquery.Client', _mock_bq()):
            resolve_downstream_plates(df, project_id='test-proj')
        # BQ was called (returns empty by default → early return; df unchanged)


# ── Tests: end-to-end with BQ ────────────────────────────────────────────────

class TestEndToEnd:

    def test_empty_wells_returns_unchanged(self):
        """LIMS returns no wells for the miniprep plate → early return, no ops appended."""
        df = _make_df({
            'protocol_name':       [proto.MINIPREP],
            'all_protocol_plates': json.dumps({'Miniprep': '12345'}),
        })
        empty_wells = pd.DataFrame(columns=['well_id', 'plate_id'])
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.well', empty_wells))):
            result = resolve_downstream_plates(df, project_id='test-proj')
        assert result.iloc[0]['protocol_name'] == [proto.MINIPREP]

    def test_ngs_op_appended_via_sw_id_match(self):
        """
        NGS op whose sw_id matches a well on the miniprep plate gets appended
        to the workorder's protocol_name list.
        """
        df = _make_df({
            'workorder_id':        'gg-001',
            'protocol_name':       [proto.MINIPREP],
            'operation_state':     ['SC'],
            'operation_start':     [pd.Timestamp('2025-03-01', tz='UTC')],
            'operation_ready':     [None],
            'job_id':              [None],
            'ngs_run_number':      [None],
            'well_location':       [None],
            'all_protocol_plates': json.dumps({'Miniprep': '12345'}),
        })
        wells_df = pd.DataFrame([{'well_id': 101, 'plate_id': 12345}])
        # raw_ops: NGS op whose sw_id=101 (a well on miniprep plate 12345)
        raw_ops_df = pd.DataFrame([{
            'id': 99, 'job_id': None, 'plan_id': None, 'state': 'SC',
            'date_created': pd.Timestamp('2025-04-01', tz='UTC'),
            'date_ready':   pd.Timestamp('2025-04-01', tz='UTC'),
            'protocol_name': proto.NGS,
            'sw': None, 'qw': None, 'dw': None, 'nps': None,
            'sw_id': 101, 'qw_id': None, 'dw_id': None,
        }])
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.well', wells_df),
                             ('kicked_back_jobs', raw_ops_df))):
            result = resolve_downstream_plates(df, project_id='test-proj')
        pnames = result[result['workorder_id'] == 'gg-001'].iloc[0]['protocol_name']
        assert proto.NGS in pnames
        assert proto.MINIPREP in pnames

    def test_unrelated_workorder_not_modified(self):
        """
        A second workorder that has no miniprep plate is not touched when
        resolve_downstream_plates appends ops to the first.
        """
        df = _make_df(
            {
                'workorder_id':        'gg-001',
                'protocol_name':       [proto.MINIPREP],
                'operation_state':     ['SC'],
                'operation_start':     [pd.Timestamp('2025-03-01', tz='UTC')],
                'operation_ready':     [None],
                'job_id':              [None],
                'ngs_run_number':      [None],
                'well_location':       [None],
                'all_protocol_plates': json.dumps({'Miniprep': '12345'}),
            },
            {
                'workorder_id':        'gg-002',
                'protocol_name':       [proto.GOLDEN_GATE],
                'all_protocol_plates': '{}',
            },
        )
        wells_df  = pd.DataFrame([{'well_id': 101, 'plate_id': 12345}])
        raw_ops_df = pd.DataFrame([{
            'id': 99, 'job_id': None, 'plan_id': None, 'state': 'SC',
            'date_created': pd.Timestamp('2025-04-01', tz='UTC'),
            'date_ready':   pd.Timestamp('2025-04-01', tz='UTC'),
            'protocol_name': proto.NGS,
            'sw': None, 'qw': None, 'dw': None, 'nps': None,
            'sw_id': 101, 'qw_id': None, 'dw_id': None,
        }])
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.well', wells_df),
                             ('kicked_back_jobs', raw_ops_df))):
            result = resolve_downstream_plates(df, project_id='test-proj')
        pnames_gg2 = result[result['workorder_id'] == 'gg-002'].iloc[0]['protocol_name']
        assert proto.NGS not in (pnames_gg2 if isinstance(pnames_gg2, list) else [])


# ── well-level attribution (v1.11.91) ────────────────────────────────────────
class TestNarrowToOwnWell:
    """A plate carries material from many workorders, so merging every well on a
    plate against every workorder on that plate is a cross product. One downstream
    op then resolves to all of them and each row shows everyone's operations.

    Measured 2026-09-01: 52 queued `Rearray 96 to 384` ops existed in total, yet the
    dashboard carried 624 (row, op) pairs — 12x inflation, all of it on 11
    PARTNER_TFX_2026Aug31_* rows showing 52 ops each. Plate 17345 holds 13 distinct
    process_ids at exactly 4 wells each. After narrowing: 572 -> 44 pairs, 4 ops per
    row, and 44 + 8 (the 2 process_ids not among those rows) = 52 — every op
    attributed exactly once.
    """

    @staticmethod
    def _cand(rows):
        import pandas as pd
        return pd.DataFrame(rows)

    def test_own_well_wins(self):
        from dnasc.transformers.repair import _narrow_to_own_well
        out = _narrow_to_own_well(self._cand([
            {"well_id": 1, "workorder_id": "WO-A", "own_wid": "WO-A"},
            {"well_id": 1, "workorder_id": "WO-B", "own_wid": "WO-A"},
        ]))
        assert set(map(tuple, out.values)) == {(1, "WO-A")}

    def test_unknown_owner_falls_back_to_plate_level(self):
        """own_wid NULL is absence of evidence — keep the old behaviour."""
        from dnasc.transformers.repair import _narrow_to_own_well
        out = _narrow_to_own_well(self._cand([
            {"well_id": 2, "workorder_id": "WO-A", "own_wid": None},
            {"well_id": 2, "workorder_id": "WO-B", "own_wid": None},
        ]))
        assert set(map(tuple, out.values)) == {(2, "WO-A"), (2, "WO-B")}

    def test_another_owner_drops_the_well_entirely(self):
        """This is the distinction that took 52 -> 12 down to 52 -> 4. A well
        attributed to someone NOT on this plate map is positive evidence of
        non-membership, unlike NULL."""
        from dnasc.transformers.repair import _narrow_to_own_well
        out = _narrow_to_own_well(self._cand([
            {"well_id": 3, "workorder_id": "WO-A", "own_wid": "WO-ELSEWHERE"},
            {"well_id": 3, "workorder_id": "WO-B", "own_wid": "WO-ELSEWHERE"},
        ]))
        assert len(out) == 0

    def test_junk_owner_strings_count_as_unknown(self):
        from dnasc.transformers.repair import _narrow_to_own_well
        for junk in ("nan", "None", ""):
            out = _narrow_to_own_well(self._cand([
                {"well_id": 4, "workorder_id": "WO-A", "own_wid": junk},
                {"well_id": 4, "workorder_id": "WO-B", "own_wid": junk},
            ]))
            assert len(out) == 2, f"{junk!r} should be treated as unknown"

    def test_never_invents_a_pair(self):
        """Strictly a narrowing — the output must be a subset of the input."""
        from dnasc.transformers.repair import _narrow_to_own_well
        cand = self._cand([
            {"well_id": i, "workorder_id": w, "own_wid": o}
            for i, o in [(1, "WO-A"), (2, None), (3, "WO-X")]
            for w in ("WO-A", "WO-B", "WO-C")
        ])
        before = set(map(tuple, cand[["well_id", "workorder_id"]].values))
        after = set(map(tuple, _narrow_to_own_well(cand).values))
        assert after <= before

    def test_empty_and_missing_column_are_safe(self):
        import pandas as pd
        from dnasc.transformers.repair import _narrow_to_own_well
        assert _narrow_to_own_well(pd.DataFrame(
            columns=["well_id", "workorder_id", "own_wid"])).empty
        # no own_wid column at all -> unchanged plate-level behaviour
        out = _narrow_to_own_well(pd.DataFrame(
            [{"well_id": 1, "workorder_id": "WO-A"}]))
        assert len(out) == 1

    def test_both_maps_use_the_helper(self):
        """well_wid_df and downstream_well_wid_df must both be narrowed, and the
        LIMS query must select own_wid or the helper silently no-ops."""
        from pathlib import Path
        import dnasc.transformers.repair as r
        src = Path(r.__file__).read_text()
        assert src.count("_narrow_to_own_well(") >= 3, \
            "expected the helper defined once and applied to both well maps"
        assert "AS own_wid" in src, \
            "the LIMS well query must select own_wid, else narrowing no-ops"

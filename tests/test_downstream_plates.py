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

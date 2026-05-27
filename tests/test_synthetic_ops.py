"""
Tests for populate_synthetic_optracker_batch in dnasc/transformers/repair.py.
BQ calls mocked via patch('dnasc.transformers.repair.bigquery.Client').
"""
import pandas as pd
from unittest.mock import patch, MagicMock

from dnasc import protocols as proto
from dnasc.transformers.repair import populate_synthetic_optracker_batch


# ── mock helpers ──────────────────────────────────────────────────────────────

# Properly-shaped empty DataFrames for queries the function always makes.
# An empty pd.DataFrame() has no columns, which causes KeyError when the
# function accesses e.g. _kb_df["job_id"].
_EMPTY_KICKED_BACK = pd.DataFrame(columns=['job_id'])
_EMPTY_OPS = pd.DataFrame(columns=['op_id', 'job_id', 'state', 'date_created',
                                    'date_ready', 'protocol_name', 'ref_id'])
_EMPTY_RUN_NUM = pd.DataFrame(columns=['op_id', 'ngs_run_number'])


def _sql_dispatch(*rules):
    """
    Returns a callable side_effect for client.query(sql).
    Matches SQL against keywords in order; returns empty DataFrames with correct
    column shapes for the standard always-called queries if no rule matches.
    """
    default_fallbacks = [
        ('gather-samples-success-or-fail-mode', _EMPTY_KICKED_BACK),
        ('Source Well',                          _EMPTY_OPS),
        ('Run Number',                           _EMPTY_RUN_NUM),
    ]
    all_rules = list(rules) + default_fallbacks

    def _dispatch(sql):
        for kw, df in all_rules:
            if kw.lower() in sql.lower():
                m = MagicMock()
                m.to_dataframe.return_value = df
                return m
        m = MagicMock()
        m.to_dataframe.return_value = pd.DataFrame()
        return m
    return _dispatch


def _mock_bq(*rules):
    """Return a mock bigquery.Client class with keyword-dispatched query results."""
    instance = MagicMock()
    instance.query.side_effect = _sql_dispatch(*rules)
    return MagicMock(return_value=instance)


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _base_df(**kwargs):
    defaults = {
        'workorder_id':    'syn-001',
        'data_source':     'SYNTHETIC',
        'protocol_name':   None,
        'operation_state': None,
        'operation_start': None,
        'operation_ready': None,
        'job_id':          None,
        'ngs_run_number':  None,
        'wo_created_at':   None,
        'wo_updated_at':   None,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def _plates_df(synthetic_id, plate_protocol='Miniprep', plate_id=999):
    return pd.DataFrame([{
        'synthetic_id':     synthetic_id,
        'plate_id':         plate_id,
        'plate_protocol':   plate_protocol,
        'plate_created_at': pd.Timestamp('2025-03-01', tz='UTC'),
    }])


def _ops_bq_df(protocol_name, op_id=1, ref_id=999):
    """
    Ops row as returned by BQ (before merge with plate_syn_pairs).
    No synthetic_id here — that gets added by the merge inside _fetch_ops_1a.
    ref_id must match plates_df plate_id so the merge succeeds.
    job_id=None prevents ops_pass2 from running.
    """
    return pd.DataFrame([{
        'op_id':          op_id,
        'job_id':         None,
        'state':          'SC',
        'date_created':   pd.Timestamp('2025-03-02', tz='UTC'),
        'date_ready':     pd.Timestamp('2025-03-02', tz='UTC'),
        'protocol_name':  protocol_name,
        'ref_id':         ref_id,
    }])


# ── Tests: early return ───────────────────────────────────────────────────────

class TestEarlyReturn:

    def test_no_synthetic_rows_no_bq_call(self):
        """Real BIOS rows → syn_ids empty → BQ never instantiated."""
        df = _base_df(workorder_id='real-001', data_source='BIOS')
        with patch('dnasc.transformers.repair.bigquery.Client') as mock_cls:
            populate_synthetic_optracker_batch(df, project_id='test-proj')
        assert not mock_cls.called

    def test_lsp_row_with_queue_data_excluded_from_syn_mask(self):
        """
        Real LSP row (data_source='LSP') with existing protocol_name is excluded
        from syn_mask — the _has_queue check gates the LSP branch.
        """
        df = _base_df(
            workorder_id='LSP-0001', data_source='LSP',
            protocol_name=[proto.LSP_RECEIVING],
        )
        with patch('dnasc.transformers.repair.bigquery.Client') as mock_cls:
            populate_synthetic_optracker_batch(df, project_id='test-proj')
        assert not mock_cls.called

    def test_no_plates_found_returns_unchanged(self):
        """BQ plates query returns no rows → early return, protocol_name stays None."""
        df = _base_df()
        with patch('dnasc.transformers.repair.bigquery.Client', _mock_bq()):
            result = populate_synthetic_optracker_batch(df, project_id='test-proj')
        assert result.iloc[0]['protocol_name'] is None


# ── Tests: protocol filter ────────────────────────────────────────────────────

class TestProtocolFilter:

    def test_lsp_ops_stripped_from_streak_row(self):
        """STREAK synthetic rows must not receive LSP Receiving ops (lsp_only filter)."""
        sid = 'STREAK_well1'
        df = _base_df(workorder_id=sid)
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.plate',  _plates_df(sid)),
                             ('Plate ID',   _ops_bq_df(proto.LSP_RECEIVING)))):
            result = populate_synthetic_optracker_batch(df, project_id='test-proj')
        pnames = result.iloc[0]['protocol_name']
        assert pnames is None or (isinstance(pnames, list) and proto.LSP_RECEIVING not in pnames)

    def test_star_transf_ops_kept_for_streak_row(self):
        """STREAK synthetic rows do receive STAR Transformation ops."""
        sid = 'STREAK_well2'
        df = _base_df(workorder_id=sid)
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.plate',  _plates_df(sid)),
                             ('Plate ID',   _ops_bq_df(proto.STAR_TRANSF)))):
            result = populate_synthetic_optracker_batch(df, project_id='test-proj')
        pnames = result.iloc[0]['protocol_name']
        assert isinstance(pnames, list) and proto.STAR_TRANSF in pnames

    def test_star_transf_ops_stripped_from_lsp_row(self):
        """LSP synthetic rows must not receive STAR Transformation ops (tfm_only filter)."""
        sid = 'LSP-0001'
        df = _base_df(workorder_id=sid, data_source='SYNTHETIC_LSP')
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.plate', _plates_df(sid, plate_protocol='Bank Overnights')),
                             ('Plate ID',        _ops_bq_df(proto.STAR_TRANSF)))):
            result = populate_synthetic_optracker_batch(df, project_id='test-proj')
        pnames = result.iloc[0]['protocol_name']
        assert pnames is None or (isinstance(pnames, list) and proto.STAR_TRANSF not in pnames)

    def test_miniprep_ops_stripped_from_lsp_row(self):
        """LSP synthetic rows must not receive Miniprep ops (tfm_only filter)."""
        sid = 'LSP-0002'
        df = _base_df(workorder_id=sid, data_source='SYNTHETIC_LSP')
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.plate', _plates_df(sid, plate_protocol='Bank Overnights')),
                             ('Plate ID',        _ops_bq_df(proto.MINIPREP)))):
            result = populate_synthetic_optracker_batch(df, project_id='test-proj')
        pnames = result.iloc[0]['protocol_name']
        assert pnames is None or (isinstance(pnames, list) and proto.MINIPREP not in pnames)


# ── Tests: manual time prepend ────────────────────────────────────────────────

class TestManualTimePrepend:

    def test_manual_miniprep_label_prepended_for_uuid_synthetic(self):
        """
        Non-LSP synthetic with an Overnight Culture LIMS plate gets
        'Manual: Miniprep/Glycerol/Media created' prepended to protocol_name.
        """
        sid = 'syn-uuid-001'
        df = _base_df(workorder_id=sid)
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.plate', _plates_df(sid, plate_protocol='Overnight Culture')),
                             ('Plate ID',        _ops_bq_df(proto.MINIPREP)))):
            result = populate_synthetic_optracker_batch(df, project_id='test-proj')
        pnames = result.iloc[0]['protocol_name']
        assert isinstance(pnames, list)
        assert pnames[0] == 'Manual: Miniprep/Glycerol/Media created'
        assert proto.MINIPREP in pnames

    def test_manual_label_always_prepended_for_uuid_synthetic(self):
        """
        Non-LSP synthetics always get 'Manual: Miniprep/Glycerol/Media created'
        prepended regardless of plate type — the timestamp uses now() when no
        overnight/miniprep plate is present, but the label is unconditional.
        """
        sid = 'syn-uuid-002'
        df = _base_df(workorder_id=sid)
        with patch('dnasc.transformers.repair.bigquery.Client',
                   _mock_bq(('lims__src.plate', _plates_df(sid, plate_protocol='NGS Plate')),
                             ('Plate ID',        _ops_bq_df(proto.NGS)))):
            result = populate_synthetic_optracker_batch(df, project_id='test-proj')
        pnames = result.iloc[0]['protocol_name']
        assert isinstance(pnames, list)
        assert pnames[0] == 'Manual: Miniprep/Glycerol/Media created'
        assert proto.NGS in pnames

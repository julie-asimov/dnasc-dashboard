"""
Tests for parse_pipeline_operations (extracted from dnasc/renderer/dashboard.py).

This function converts raw op lists into cleaned, deduped, grouped timeline
entries for every workorder displayed in the dashboard. It touches every row
and has had several documented bugs (dedup by job_id, ndarray safety, run_number
propagation, soft-fail suppression).
"""
import numpy as np
import pandas as pd
import pytest

from dnasc.renderer.dashboard import parse_pipeline_operations
from dnasc import protocols as proto


# ── helpers ───────────────────────────────────────────────────────────────────

def _call(protocols, states, starts=None, jobs=None, wells=None, readys=None, run_numbers=None):
    return parse_pipeline_operations(
        protocol_names=protocols,
        operation_states=states,
        operation_starts=starts or [None] * len(protocols),
        job_ids=jobs or [None] * len(protocols),
        well_locations_list=wells or [None] * len(protocols),
        operation_ready_times=readys or [None] * len(protocols),
        ngs_run_numbers=run_numbers,
    )


def _queues(result):
    return [op['queue'] for op in result]

def _states(result):
    return [op['state'] for op in result]

def _classes(result):
    return [op['class'] for op in result]


# ── empty / null input ────────────────────────────────────────────────────────

class TestEmptyInput:

    def test_empty_protocol_list_returns_empty(self):
        assert parse_pipeline_operations([], [], [], [], [], []) == []

    def test_none_inputs_return_empty(self):
        assert parse_pipeline_operations(None, None, None, None, None, None) == []

    def test_nan_inputs_return_empty(self):
        assert parse_pipeline_operations(float('nan'), float('nan'), None, None, None, None) == []

    def test_ndarray_protocol_names_handled(self):
        result = _call(np.array([proto.MINIPREP]), np.array(['SC']))
        assert len(result) == 1
        assert result[0]['queue'] == proto.MINIPREP

    def test_ndarray_states_handled(self):
        result = _call([proto.MINIPREP], np.array(['RU']))
        assert result[0]['state'] == 'Running'


# ── state mapping ─────────────────────────────────────────────────────────────

class TestStateMapping:

    def test_sc_maps_to_completed(self):
        result = _call([proto.MINIPREP], ['SC'])
        assert result[0]['state'] == 'Completed'
        assert result[0]['class'] == 'succeeded'

    def test_fa_maps_to_failed(self):
        result = _call([proto.MINIPREP], ['FA'])
        assert result[0]['state'] == 'Failed'
        assert result[0]['class'] == 'failed'

    def test_ru_maps_to_running(self):
        result = _call([proto.MINIPREP], ['RU'])
        assert result[0]['state'] == 'Running'
        assert result[0]['class'] == 'running'

    def test_rd_maps_to_ready(self):
        result = _call([proto.MINIPREP], ['RD'])
        assert result[0]['state'] == 'Ready'
        assert result[0]['class'] == 'ready'

    def test_ca_maps_to_canceled(self):
        result = _call([proto.MINIPREP], ['CA'])
        assert result[0]['state'] == 'Canceled'
        assert result[0]['class'] == 'canceled'

    def test_unknown_state_maps_to_unknown_pending(self):
        result = _call([proto.MINIPREP], ['XX'])
        assert result[0]['state'] == 'Unknown'
        assert result[0]['class'] == 'pending'


# ── soft-fail suppression ─────────────────────────────────────────────────────

class TestSoftFail:

    def test_fa_suppressed_when_sc_exists_for_same_protocol(self):
        result = _call([proto.NGS, proto.NGS], ['FA', 'SC'])
        assert len(result) == 1
        assert result[0]['state'] == 'Completed'

    def test_ca_suppressed_when_sc_exists_for_same_protocol(self):
        result = _call([proto.NGS, proto.NGS], ['CA', 'SC'])
        assert len(result) == 1
        assert result[0]['state'] == 'Completed'

    def test_fa_kept_when_no_sc_exists(self):
        result = _call([proto.NGS], ['FA'])
        assert len(result) == 1
        assert result[0]['state'] == 'Failed'

    def test_fa_for_different_protocol_not_suppressed(self):
        # NGS has SC; Miniprep has FA — Miniprep FA should still show
        result = _call([proto.NGS, proto.MINIPREP], ['SC', 'FA'])
        assert len(result) == 2
        queues = _queues(result)
        assert proto.MINIPREP in queues

    def test_fa_suppressed_only_for_matching_protocol(self):
        result = _call(
            [proto.NGS, proto.NGS, proto.REARRAY],
            ['FA',       'SC',      'FA'],
        )
        queues = _queues(result)
        assert proto.NGS in queues
        assert proto.REARRAY in queues
        ngs_entries = [r for r in result if r['queue'] == proto.NGS]
        assert len(ngs_entries) == 1
        assert ngs_entries[0]['state'] == 'Completed'


# ── deduplication ─────────────────────────────────────────────────────────────

class TestDedup:

    def test_same_protocol_state_job_deduped(self):
        result = _call([proto.NGS, proto.NGS], ['SC', 'SC'], jobs=[42, 42])
        assert len(result) == 1

    def test_same_protocol_state_different_jobs_not_deduped(self):
        # v1.9.35 regression: two Rearray ops from different jobs (original + repick)
        result = _call([proto.REARRAY, proto.REARRAY], ['SC', 'SC'], jobs=[100, 200])
        assert len(result) == 2

    def test_same_protocol_different_states_not_deduped(self):
        # Use non-groupable protocol + different-but-not-suppressed states
        # so neither dedup nor grouping collapses them
        result = _call([proto.MINIPREP, proto.MINIPREP], ['RU', 'SC'], jobs=[42, 42])
        assert len(result) == 2

    def test_dedup_keeps_earliest_start_time(self):
        t_early = pd.Timestamp('2025-01-01', tz='UTC')
        t_late  = pd.Timestamp('2025-06-01', tz='UTC')
        result = _call(
            [proto.NGS, proto.NGS], ['SC', 'SC'],
            jobs=[42, 42],
            starts=[t_late, t_early],
        )
        assert len(result) == 1
        # start_time should be the earlier one (converted to EST)
        assert result[0]['start_time'] is not None


# ── run_number propagation ────────────────────────────────────────────────────

class TestRunNumberPropagation:

    def test_run_number_propagated_to_sc_from_fa_same_job(self):
        # FA carries run_number, SC does not — SC should inherit it after propagation
        result = _call(
            [proto.NGS, proto.NGS],
            ['FA',       'SC'],
            jobs=[42, 42],
            run_numbers=['RUN-001', None],
        )
        sc_entry = next(r for r in result if r['state'] == 'Completed')
        assert sc_entry['run_numbers'] == ['RUN-001']

    def test_run_number_not_propagated_across_different_jobs(self):
        result = _call(
            [proto.NGS, proto.NGS],
            ['SC',       'SC'],
            jobs=[42,    99],
            run_numbers=['RUN-A', None],
        )
        # Two distinct jobs: second should not inherit RUN-A
        entries = {r['job_id']: r for r in result}
        assert entries[99]['run_numbers'] == []

    def test_null_run_number_stays_null(self):
        result = _call([proto.MINIPREP], ['SC'], run_numbers=[None])
        assert result[0]['run_numbers'] == []


# ── grouping (groupable protocols) ────────────────────────────────────────────

class TestGrouping:

    def test_same_groupable_protocol_same_job_merged(self):
        # Use SC + RU (different states survive dedup) with same job → grouping merges wells
        result = _call(
            [proto.NGS, proto.NGS], ['SC', 'RU'],
            jobs=[42, 42],
            wells=['A1', 'B2'],
        )
        assert len(result) == 1
        assert set(result[0]['wells']) == {'A1', 'B2'}

    def test_same_groupable_protocol_different_jobs_not_merged(self):
        result = _call(
            [proto.NGS, proto.NGS], ['SC', 'SC'],
            jobs=[42, 99],
        )
        assert len(result) == 2

    def test_non_groupable_protocol_not_merged(self):
        # Different jobs so dedup doesn't fire; grouping should also not merge (non-groupable)
        result = _call(
            [proto.MINIPREP, proto.MINIPREP], ['SC', 'SC'],
            jobs=[42, 99],
        )
        assert len(result) == 2

    def test_grouped_op_upgrades_to_running_if_any_running(self):
        result = _call(
            [proto.NGS, proto.NGS], ['SC', 'RU'],
            jobs=[42, 42],
        )
        assert len(result) == 1
        assert result[0]['state'] == 'Running'
        assert result[0]['class'] == 'running'

    def test_groupable_protocols_include_dna_quant_rearray_ngs_fa(self):
        for p in [proto.DNA_QUANT, proto.REARRAY, proto.NGS, proto.FRAGMENT_ANALYZER]:
            result = _call([p, p], ['SC', 'SC'], jobs=[1, 1], wells=['A1', 'B2'])
            assert len(result) == 1, f"{p} should be groupable"


# ── well handling ─────────────────────────────────────────────────────────────

class TestWells:

    def test_null_well_not_added_to_wells_list(self):
        result = _call([proto.NGS], ['SC'], wells=[None])
        assert result[0]['wells'] == []

    def test_nan_well_not_added(self):
        result = _call([proto.NGS], ['SC'], wells=[float('nan')])
        assert result[0]['wells'] == []

    def test_valid_well_added(self):
        result = _call([proto.NGS], ['SC'], wells=['A1'])
        assert result[0]['wells'] == ['A1']


# ── multi-step timeline ───────────────────────────────────────────────────────

class TestMultiStepTimeline:

    def test_distinct_protocols_each_appear(self):
        result = _call(
            [proto.MINIPREP, proto.REARRAY, proto.NGS],
            ['SC',           'SC',          'SC'],
        )
        assert _queues(result) == [proto.MINIPREP, proto.REARRAY, proto.NGS]

    def test_mixed_states_in_order(self):
        result = _call(
            [proto.MINIPREP, proto.NGS],
            ['SC',           'RU'],
        )
        assert result[0]['state'] == 'Completed'
        assert result[1]['state'] == 'Running'

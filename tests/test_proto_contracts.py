"""
Contract tests for dnasc/protocols.py and the _PROTO_MAP in enrichment.py.

These tests lock in what each constant resolves to so a value change (e.g.
an OpTracker rename) can't silently break matching. They also verify that
_PROTO_MAP keys use the same strings, since a mismatch there causes the
req_operation label to silently not fire.

Validation methodology: all expected values were confirmed against
dashboard_state/baseline.parquet — only strings that appear in actual
pipeline data are asserted as present.
"""
import pytest
from dnasc import protocols as proto
from dnasc.transformers.enrichment import _PROTO_MAP, _PROTO_MAP_BY_PHASE, _TYPE_PHASE


class TestProtocolValues:
    """Constants resolve to the strings that actually appear in OpTracker."""

    def test_order_oligos_no_trailing_s(self):
        # Data shows "Order Oligo" — never "Order Oligos"
        assert proto.ORDER_OLIGOS == "Order Oligo"

    def test_ngs_value(self):
        assert proto.NGS == "NGS Sequence Confirmation"

    def test_fragment_analyzer_value(self):
        assert proto.FRAGMENT_ANALYZER == "Fragment Analyzer"

    def test_rearray_value(self):
        assert proto.REARRAY == "Rearray 96 to 384"

    def test_miniprep_value(self):
        assert proto.MINIPREP == "Create Minipreps and Glycerol Stocks"

    def test_star_transf_value(self):
        assert proto.STAR_TRANSF == "STAR Transformation"

    def test_no_sanger_in_seq_protos(self):
        assert "Sanger Sequencing" not in proto.SEQ_PROTOS

    def test_seq_protos_contains_only_real_protocols(self):
        assert proto.SEQ_PROTOS == frozenset({"NGS Sequence Confirmation", "Fragment Analyzer"})

    def test_order_protos_contains_correct_oligo_string(self):
        assert "Order Oligo" in proto.ORDER_PROTOS
        assert "Order Oligos" not in proto.ORDER_PROTOS


class TestProtoMapKeys:
    """_PROTO_MAP keys use the same strings as the protocol constants."""

    def test_order_oligo_rd_in_proto_map(self):
        assert (proto.ORDER_OLIGOS, 'RD') in _PROTO_MAP

    def test_order_oligo_ru_in_proto_map(self):
        assert (proto.ORDER_OLIGOS, 'RU') in _PROTO_MAP

    def test_old_order_oligos_with_s_not_in_proto_map(self):
        # Regression: "Order Oligos" (with 's') was the old wrong value
        assert ("Order Oligos", 'RD') not in _PROTO_MAP
        assert ("Order Oligos", 'RU') not in _PROTO_MAP

    def test_order_oligo_rd_label(self):
        _, label = _PROTO_MAP[(proto.ORDER_OLIGOS, 'RD')]
        assert label == 'ORDER OLIGOS: READY'

    def test_order_oligo_ru_label(self):
        _, label = _PROTO_MAP[(proto.ORDER_OLIGOS, 'RU')]
        assert label == 'ORDER OLIGOS: RUNNING'

    def test_ngs_in_proto_map(self):
        assert (proto.NGS, 'RD') in _PROTO_MAP
        assert (proto.NGS, 'RU') in _PROTO_MAP

    def test_sanger_not_in_proto_map(self):
        assert ("Sanger Sequencing", 'RD') not in _PROTO_MAP
        assert ("Sanger Sequencing", 'RU') not in _PROTO_MAP

    def test_all_proto_map_keys_use_known_constants(self):
        """Every protocol string in _PROTO_MAP should appear in the parquet data
        or be a known synthetic label (Manual: *). This catches future typos."""
        known = {
            proto.SYNTHESIS_ORDER, proto.ORDER_OLIGOS,
            proto.RECEIVE_SYNPART, proto.RECEIVE_PLASMID, proto.PCR,
            proto.FRAGMENT_ANALYZER, proto.GOLDEN_GATE, proto.GIBSON,
            proto.STAR_TRANSF, proto.MINIPREP, proto.REPICK,
            proto.REARRAY, proto.DNA_QUANT, proto.NGS,
            proto.LSP_ORDER, proto.LSP_RECEIVING, proto.GLYCEROL_STOCKING,
            proto.LSP_REVIEWING, proto.LSP_RELEASING, proto.DIGEST,
        }
        for (proto_name, _state), _ in _PROTO_MAP.items():
            assert proto_name in known, (
                f"_PROTO_MAP key {proto_name!r} is not a known protocol constant — "
                f"verify it exists in OpTracker before adding"
            )


def _resolve(proto_name, state, wo_type):
    """Mirror of the per-row lookup in _enrich_requests: phase override first,
    base map second."""
    ovr = _PROTO_MAP_BY_PHASE.get(_TYPE_PHASE.get(wo_type, ''))
    key = (proto_name, state)
    hit = ovr.get(key) if ovr else None
    return hit if hit is not None else _PROTO_MAP.get(key)


class TestPhaseScopedPriorities:
    """A protocol that runs in more than one phase scores on that phase's scale."""

    def test_digest_is_lsp_qc_tier(self):
        pri, label = _PROTO_MAP[(proto.DIGEST, 'RD')]
        assert label == 'DIGEST: READY'
        # after Glycerol Stocking, before LSP Reviewing
        assert _PROTO_MAP[(proto.GLYCEROL_STOCKING, 'RD')][0] < pri < _PROTO_MAP[(proto.LSP_REVIEWING, 'RD')][0]

    def test_fa_keeps_parts_priority_on_a_pcr_workorder(self):
        pri, _ = _resolve(proto.FRAGMENT_ANALYZER, 'RD', 'pcr_workorder')
        assert pri == _PROTO_MAP[(proto.FRAGMENT_ANALYZER, 'RD')][0]
        # loses to the assembly it feeds — it is PCR QC there, not a late step
        assert pri < _PROTO_MAP[(proto.GOLDEN_GATE, 'RD')][0]

    def test_fa_gets_lsp_priority_on_an_lsp_workorder(self):
        pri, label = _resolve(proto.FRAGMENT_ANALYZER, 'RD', 'lsp_workorder')
        assert label == 'FRAGMENT ANALYZER: READY'
        # digest-gel readout: after Digest, before LSP Reviewing
        assert _PROTO_MAP[(proto.DIGEST, 'RD')][0] < pri < _PROTO_MAP[(proto.LSP_REVIEWING, 'RD')][0]
        # and it now outranks the LSP QC steps that precede it
        assert pri > _PROTO_MAP[(proto.NGS, 'RD')][0]
        assert pri > _PROTO_MAP[(proto.DNA_QUANT, 'RD')][0]

    def test_both_fa_flavors_active_together_lsp_one_wins(self):
        """Same request, PCR-workorder FA and LSP-workorder FA both RD."""
        parts_pri, _ = _resolve(proto.FRAGMENT_ANALYZER, 'RD', 'pcr_workorder')
        lsp_pri, lsp_label = _resolve(proto.FRAGMENT_ANALYZER, 'RD', 'lsp_workorder')
        assert lsp_pri > parts_pri
        assert lsp_label == 'FRAGMENT ANALYZER: READY'

    def test_lsp_fa_beats_a_leftover_ready_pcr(self):
        # Regression: a stale RD parts op must not outrank live LSP work
        assert _resolve(proto.FRAGMENT_ANALYZER, 'RD', 'lsp_workorder')[0] > _PROTO_MAP[(proto.PCR, 'RD')][0]

    def test_non_overridden_protocol_is_phase_independent(self):
        for t in ('pcr_workorder', 'lsp_workorder', 'golden_gate_workorder'):
            assert _resolve(proto.NGS, 'RD', t) == _PROTO_MAP[(proto.NGS, 'RD')]

    def test_unknown_workorder_type_falls_back_to_base_map(self):
        assert _resolve(proto.FRAGMENT_ANALYZER, 'RD', 'some_new_workorder') == _PROTO_MAP[(proto.FRAGMENT_ANALYZER, 'RD')]

    def test_every_phase_override_key_exists_in_base_map(self):
        """An override is a re-tier of a real protocol, never a new one."""
        for phase, omap in _PROTO_MAP_BY_PHASE.items():
            for key in omap:
                assert key in _PROTO_MAP, f"{phase} override {key} has no base _PROTO_MAP entry"

    def test_lsp_workorder_maps_to_lsp_phase(self):
        assert _TYPE_PHASE['lsp_workorder'] == 'LSP'
        assert _TYPE_PHASE['pcr_workorder'] == 'PARTS'
        assert _TYPE_PHASE['golden_gate_workorder'] == 'ASM'

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
from dnasc.transformers.enrichment import _PROTO_MAP


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
            proto.LSP_REVIEWING, proto.LSP_RELEASING,
        }
        for (proto_name, _state), _ in _PROTO_MAP.items():
            assert proto_name in known, (
                f"_PROTO_MAP key {proto_name!r} is not a known protocol constant — "
                f"verify it exists in OpTracker before adding"
            )

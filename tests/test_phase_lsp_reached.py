"""
Regression test: a request that reached LSP must not fall back to ASM.

Before fix: an LSP workorder that is CANCELED (order pulled, re-order pending)
or SUCCEEDED drops out of the phase-active row set, and the already-executed
root-stock assembly below it re-labelled the request 'ASM'.  A773-1-PG_RHB001
showed 12 such requests as "ASM / STALLED" when every one of them was sitting
in LSP waiting on a fresh order.
"""
import pandas as pd

from dnasc.transformers.enrichment import EnrichmentTransformer


def _row(**kw):
    base = dict(
        req_id="R1", request_status="IN_PROGRESS", type="golden_gate_workorder",
        wo_status="SUCCEEDED", visual_status="SUCCEEDED", workorder_id="WO-1",
        STOCK_ID="pAI-1", fulfills_request=True, data_source="BIOS",
        wo_created_at=pd.Timestamp("2026-08-01", tz="UTC"),
        protocol_name=None, operation_state=None, experiment_name="E",
        request_created_at=pd.Timestamp("2026-07-01", tz="UTC"),
        backbone=None, Waiting=None, root_work_order_id=None,
    )
    base.update(kw)
    return base


def _phase(rows):
    out = EnrichmentTransformer.compute_request_enrichment(pd.DataFrame(rows))
    return out["req_phase"].iloc[0]


ASM = _row()
LSP = dict(type="lsp_workorder", workorder_id="LSP-1", STOCK_ID="pAI-1",
           fulfills_request=False, data_source="LSP")


class TestLspReached:
    def test_canceled_lsp_keeps_the_request_in_lsp(self):
        assert _phase([ASM, _row(**LSP, wo_status="CANCELED", visual_status="CANCELED",
                                 wo_created_at=pd.Timestamp("2026-08-20", tz="UTC"))]) == "LSP"

    def test_succeeded_lsp_keeps_the_request_in_lsp(self):
        assert _phase([ASM, _row(**LSP, wo_status="SUCCEEDED", visual_status="SUCCEEDED",
                                 wo_created_at=pd.Timestamp("2026-08-20", tz="UTC"))]) == "LSP"

    def test_live_lsp_still_reads_lsp(self):
        assert _phase([ASM, _row(**LSP, wo_status="RUNNING", visual_status="RUNNING",
                                 wo_created_at=pd.Timestamp("2026-08-20", tz="UTC"))]) == "LSP"

    def test_assembly_reopened_after_lsp_goes_back_to_asm(self):
        """A genuine trip back to assembly: new root-stock build created after the LSP."""
        rows = [
            ASM,
            _row(**LSP, wo_status="CANCELED", visual_status="CANCELED",
                 wo_created_at=pd.Timestamp("2026-08-20", tz="UTC")),
            _row(workorder_id="WO-2", wo_status="RUNNING", visual_status="RUNNING",
                 wo_created_at=pd.Timestamp("2026-08-25", tz="UTC")),
        ]
        assert _phase(rows) == "ASM"

    def test_failed_assembly_after_lsp_stays_in_lsp(self):
        """A dead post-LSP attempt is not a trip back to assembly.

        Reversed 2026-09-09 per Julie: this used to assert ASM, on the theory that
        any assembly created after the LSP meant the request had gone back to
        assembly. But a FAILED assembly is a dead branch, and the request then
        stalls reading 'ASM / Stalled' when what it actually needs is another LSP
        order. 18 requests in that day's baseline were mislabelled this way, every
        one with a FAILED assembly as its most recent. Reaching LSP is one-way.
        """
        rows = [
            ASM,
            _row(**LSP, wo_status="CANCELED", visual_status="CANCELED",
                 wo_created_at=pd.Timestamp("2026-08-20", tz="UTC")),
            _row(workorder_id="WO-2", wo_status="FAILED", visual_status="FAILED",
                 wo_created_at=pd.Timestamp("2026-08-25", tz="UTC")),
        ]
        assert _phase(rows) == "LSP"

    def test_no_lsp_workorder_is_untouched(self):
        assert _phase([ASM]) == "ASM"

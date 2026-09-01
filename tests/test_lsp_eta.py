"""
LSP release ETAs — guards the fix that came from step_ts rather than from here.

_lsp_batch_start() reads `operation_start` for a RUNNING LSP Order op and hands it
to _lsp_release_eta(). Before v1.11.85/89, operation_start was
operation.date_created, which under bios means "when the op was QUEUED". Measured
2026-09-01 across the 44 RU LSP Order ops then live: step_ts moves them +73.5h on
average (max 170h), and Glycerol Stocking Scinomix +259.5h (max 314h).

Measured PER JOB on 2026-09-01 — four running LSP Order jobs, same row before and
after, not a min-vs-min comparison (my first pass made that mistake and Julie
caught it):

    job 9607  start Aug 26 17:52 -> Aug 31 16:11   ETA Sep 01 -> Sep 04
    job 9608  start Aug 25 14:42 -> Aug 31 16:17   ETA Sep 01 -> Sep 04
    job 9620  start Sep 01 16:23 -> Sep 01 16:30   ETA Sep 08 -> Sep 08  (control)
    job 9622  start Aug 25 14:42 -> Sep 01 17:00   ETA Sep 01 -> Sep 08

Three of four batches carried a wrong release date, by 3 and by 7 days, and job
9620 — started today — is unchanged, which is what a correct fix looks like. At the
op level 14 of 44 ETAs move; 44 ops is only 4 batches, so op counts overstate it.

The ETA surfaces in exactly one place: the LSP Capacity tab, `LSP ETA` column
(lsp_capacity.py:1342, rendered :1361) plus its hover tooltip. The other two tables'
`LSP Start` columns come from _estimated_lsp_start, which reads operation
timestamps in 2 of 7 stage branches (In assembly / In transformation via
_running_op_start, and Waiting for synparts), so they shift too but narrowly.

No code in lsp_capacity.py was wrong; it faithfully rendered a timestamp that had
changed meaning underneath it.

So the guard has to be that _lsp_batch_start keeps reading operation_start (which
step_ts now feeds) and never re-derives a start of its own.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from dnasc.renderer import lsp_capacity as L


class TestReleaseEtaArithmetic:
    """Pins the received/release chain so a silent change to the model shows up."""

    # (job, old_start, new_start, old_eta, new_eta) — real values, same row each side
    MEASURED_JOBS = [
        (9607, dt.datetime(2026, 8, 26, 17, 52), dt.datetime(2026, 8, 31, 16, 11), (9, 1), (9, 4)),
        (9608, dt.datetime(2026, 8, 25, 14, 42), dt.datetime(2026, 8, 31, 16, 17), (9, 1), (9, 4)),
        (9620, dt.datetime(2026, 9, 1, 16, 23),  dt.datetime(2026, 9, 1, 16, 30),  (9, 8), (9, 8)),
        (9622, dt.datetime(2026, 8, 25, 14, 42), dt.datetime(2026, 9, 1, 17, 0),   (9, 1), (9, 8)),
    ]

    @pytest.mark.parametrize("job,old_s,new_s,old_e,new_e", MEASURED_JOBS,
                             ids=[f"job{j[0]}" for j in MEASURED_JOBS])
    def test_measured_per_job(self, job, old_s, new_s, old_e, new_e):
        got_old = L._lsp_release_eta(old_s)
        got_new = L._lsp_release_eta(new_s)
        assert (got_old.month, got_old.day) == old_e, f"job {job} old: got {got_old:%b %d}"
        assert (got_new.month, got_new.day) == new_e, f"job {job} new: got {got_new:%b %d}"

    def test_job_9620_is_the_control(self):
        """Started today, so queue-time and run-time agree — a correct fix must not
        move it. If this ever changes, the fix is touching rows it should not."""
        _, old_s, new_s, old_e, new_e = self.MEASURED_JOBS[2]
        assert old_e == new_e
        assert L._lsp_release_eta(old_s) == L._lsp_release_eta(new_s)

    def test_three_of_four_batches_were_wrong(self):
        moved = [j for j, _, _, oe, ne in self.MEASURED_JOBS if oe != ne]
        assert len(moved) == 3, f"expected 3 of 4 batches to shift, got {moved}"

    def test_a_later_start_never_yields_an_earlier_release(self):
        """Monotonic. A batch started later cannot release sooner."""
        base = dt.datetime(2026, 8, 3, 9, 0)
        etas = [L._lsp_release_eta(base + dt.timedelta(days=d)) for d in range(0, 28)]
        for earlier, later in zip(etas, etas[1:]):
            assert later >= earlier, f"{later} < {earlier}"

    def test_release_is_never_before_received(self):
        for d in range(0, 21):
            start = dt.datetime(2026, 8, 3, 9, 0) + dt.timedelta(days=d)
            assert L._lsp_release_eta(start) >= L._lsp_received(start)

    def test_release_is_never_before_the_start(self):
        for d in range(0, 21):
            start = dt.datetime(2026, 8, 3, 9, 0) + dt.timedelta(days=d)
            assert L._lsp_release_eta(start).date() >= start.date()


class TestStartComesFromOperationStart:
    """The actual regression surface. If these helpers stop reading
    operation_start, they stop benefiting from step_ts and the ETAs silently go
    back to promising three days early."""

    SRC = Path(L.__file__).read_text()

    def _body(self, name: str) -> str:
        i = self.SRC.index(f"def {name}(")
        j = self.SRC.find("\ndef ", i + 1)
        return self.SRC[i: j if j != -1 else len(self.SRC)]

    @pytest.mark.parametrize("fn", ["_op_start_for_protocol", "_running_op_start"])
    def test_reads_operation_start(self, fn):
        body = self._body(fn)
        assert "operation_start" in body, f"{fn} no longer reads operation_start"

    @pytest.mark.parametrize("fn", ["_op_start_for_protocol", "_running_op_start",
                                    "_lsp_batch_start"])
    def test_does_not_read_date_created_directly(self, fn):
        """date_created is the queue time. Reading it here would reintroduce the bug
        even with step_ts correct upstream."""
        body = self._body(fn)
        assert "date_created" not in body, (
            f"{fn} reads date_created directly — use operation_start, which "
            f"PipelineConfig.sql_step_ts() feeds"
        )

    def test_lsp_batch_start_still_requires_a_running_op(self):
        """It keys off state RU. If that changed to a terminal state, job.date_created
        would no longer mean 'when the operator started this batch'."""
        body = self._body("_lsp_batch_start")
        assert re.search(r"require_state\s*=\s*[\"']RU[\"']", body), \
            "_lsp_batch_start no longer requires state RU"

    def test_start_protocols_are_the_two_measured(self):
        assert set(L._LSP_START_PROTOCOLS) == {"LSP Order", "Glycerol Stocking Scinomix"}, (
            "the +73.5h / +259.5h shift measurements cover exactly these two; "
            "re-measure if the set changes"
        )

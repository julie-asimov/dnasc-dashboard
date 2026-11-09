"""
`workorder.status = BLOCKED` lags the bench.

When a missing part lands, bios readies the OpTracker operations a fraction of a
second AFTER writing the workorder rows, and never re-writes the status. On
2026-11-09 the GGN002/GGN004 campaigns had their syn parts received+quanted at
15:42-15:52 UTC and their Golden Gate operations readied at 15:59:46.0-46.2 — yet
8 of 9 workorders still read BLOCKED from a status write at 15:59:45.890. Only
34746dff got a later write and picked up READY.

The bridge in _filter_and_enrich prefers the op state, but these rows get their
operation lists completed by later pipeline steps, so at bridge time there was no
RD/RU to prefer. run_pipeline re-derives this class once the lists are final.

A genuinely blocked workorder has no runnable op at all — the 19
A824-upsilon-pilot retries waiting on the exhausted pAI-26906 are all WA.
"""
import numpy as np
import pandas as pd


def _runnable_op(s):
    """Mirrors the predicate in run_pipeline's stale-BLOCKED re-derive."""
    if hasattr(s, "tolist"):
        s = s.tolist()
    return isinstance(s, list) and any(v in ("RD", "RU") for v in s)


def _rederive(df):
    mask = (
        df["wo_status"].astype(str).str.upper().eq("BLOCKED")
        & df["visual_status"].astype(str).eq("BLOCKED")
        & df["operation_state"].map(_runnable_op)
    )
    out = df.copy()
    if mask.any():
        out.loc[mask, "visual_status"] = out.loc[mask, "operation_state"].map(
            lambda s: "RUNNING" if "RU" in (s.tolist() if hasattr(s, "tolist") else s) else "READY"
        )
    return out


def _row(ops, wo="BLOCKED", vs="BLOCKED"):
    return {"wo_status": wo, "visual_status": vs, "operation_state": ops}


class TestRunnableOp:
    def test_rd_and_ru_are_runnable(self):
        assert _runnable_op(["RD"]) is True
        assert _runnable_op(["WA", "RU"]) is True

    def test_waiting_only_is_not_runnable(self):
        assert _runnable_op(["WA", "WA", "WA"]) is False

    def test_empty_and_junk(self):
        assert _runnable_op([]) is False
        assert _runnable_op(None) is False
        assert _runnable_op("RD") is False      # a bare string is not an op list

    def test_ndarray_op_list(self):
        assert _runnable_op(np.array(["RD"])) is True


class TestStaleBlockedRederive:
    def test_rd_op_clears_a_stale_blocked(self):
        # The 8 GGN002/GGN004 builds Julie saw Ready in OpTracker.
        out = _rederive(pd.DataFrame([_row(["RD"])]))
        assert out["visual_status"].iloc[0] == "READY"

    def test_ru_op_clears_to_running(self):
        out = _rederive(pd.DataFrame([_row(["RU"])]))
        assert out["visual_status"].iloc[0] == "RUNNING"

    def test_waiting_only_stays_blocked(self):
        # The 19 A824-upsilon-pilot retries on the exhausted pAI-26906.
        out = _rederive(pd.DataFrame([_row(["WA", "WA", "WA"])]))
        assert out["visual_status"].iloc[0] == "BLOCKED"

    def test_no_ops_stays_blocked(self):
        out = _rederive(pd.DataFrame([_row([])]))
        assert out["visual_status"].iloc[0] == "BLOCKED"

    def test_only_touches_blocked_rows(self):
        df = pd.DataFrame([
            _row(["RD"], wo="READY",     vs="READY"),
            _row(["RD"], wo="SUCCEEDED", vs="SUCCEEDED"),
            _row(["RD"]),
        ])
        out = _rederive(df)
        assert list(out["visual_status"]) == ["READY", "SUCCEEDED", "READY"]

    def test_the_real_split(self):
        # 8 stale + 19 genuine, the exact 2026-11-09 shape.
        df = pd.DataFrame([_row(["RD"]) for _ in range(8)]
                          + [_row(["WA", "WA", "WA"]) for _ in range(19)])
        out = _rederive(df)
        assert (out["visual_status"] == "READY").sum() == 8
        assert (out["visual_status"] == "BLOCKED").sum() == 19

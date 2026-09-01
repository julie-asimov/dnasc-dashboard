"""
Manual LIMS operations: classification, and the type-safety guard.

resolve_lims_streakouts used to find rows with a keyword allowlist —
`STREAK.*well\\d+|^(STBL3|EPI400|TFM).*well\\d+` — over free text an operator
types by hand. Measured 2026-09-01, that missed 1016 of 1575 well-referencing
LIMS process_ids (31,101 wells): NEB_well2202668 was invisible purely because the
operator wrote NEB instead of STREAK, and bare `well12345` (489 distinct ids, the
largest single shape) was invisible because it has no prefix at all.

The filter is now structural — any process_id containing well<id> is a manual
operation on that well — and the keyword is used only to LABEL the row, never to
decide whether it exists.

The important test here is test_every_kind_maps_to_a_type_the_renderer_knows.
Introducing a new `type` value would silently drop these rows from the hardcoded
type lists in pipeline.py, dashboard.py and inflight.py, so they'd vanish from the
dashboard with nothing failing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dnasc.transformers.repair import (
    _MANUAL_KIND_PATTERNS,
    _MANUAL_KIND_TO_TYPE,
    _manual_op_kind,
)

# Every shape observed in lims__src.well.process_id on 2026-09-01, with the
# number of distinct ids carrying it, worst-first by volume.
REAL_SHAPES = [
    ("well17882",                            "OTHER"),           # 489 ids, largest shape
    ("Well2202668",                          "OTHER"),           # 38 ids, capital W
    ("NEBstable_well926888",                 "STRAIN"),          # 69 ids
    ("REFILL_well1531",                      "REFILL"),          # 50 ids
    ("refill_well2098847",                   "REFILL"),          # 34 ids
    ("SUB-NEBstable_well951",                "SUBCULTURE"),      # 23 ids
    ("SUB-NEBStable_well616",                "SUBCULTURE"),      # 15 ids
    ("EXT_well582",                          "EXT"),             # 27 ids
    ("SUB-Stbl3_well500",                    "SUBCULTURE"),      # 10 ids
    ("NEB_well2202668",                      "STRAIN"),          # 11 ids — Julie's
    ("SUB-EPI400_well400",                   "SUBCULTURE"),      # 8 ids
    ("PICK_25Aug26_well2157386",             "PICK"),            # 6 ids
    ("PARTNER_SUB-NEBstable_well288",        "SUBCULTURE"),      # 8 ids
    ("SUB-EPI400_TFM_well256",               "TFX"),             # 8 ids
    ("REFILL_INNOC_12Feb26_well99",          "REFILL"),
    ("PARTNER_GLYCEROL_CHECK_well77",        "GLYCEROL_CHECK"),
    ("PARTNER_STREAK_04May2026_well1884091", "STREAK"),
    ("PARTNER_TFX_2026Aug31_well2202668",    "TFX"),
    ("STREAK_31Aug26_well2224310",           "STREAK"),
    ("SUB_10beta_29Aug26_well123_Soyagar",   "SUBCULTURE"),
]

# The only type values anything downstream branches on. Sourced from the
# hardcoded lists in pipeline.py / dashboard.py / inflight.py.
RENDERER_KNOWN_TYPES = {
    "streakout_operation",
    "transformation_offline_operation",
    "optracker_operation",
}


class TestClassification:
    @pytest.mark.parametrize("pid,expected", REAL_SHAPES, ids=[p for p, _ in REAL_SHAPES])
    def test_real_lims_shapes(self, pid, expected):
        assert _manual_op_kind(pid) == expected

    def test_bare_well_id_is_not_dropped(self):
        """489 distinct ids look like this — the largest shape, previously invisible."""
        assert _manual_op_kind("well12345") == "OTHER"
        assert _MANUAL_KIND_TO_TYPE["OTHER"] in RENDERER_KNOWN_TYPES

    def test_nebs_reach_the_dashboard(self):
        """The three Julie reported. Classified, not discarded."""
        for pid in ("NEB_well2172126", "NEB_well2202611", "NEB_well2202668"):
            assert _manual_op_kind(pid) == "STRAIN"
            assert _MANUAL_KIND_TO_TYPE["STRAIN"] == "transformation_offline_operation"

    def test_refill_is_not_labelled_a_transformation(self):
        """The old code's else-branch called everything non-STREAK a transformation.
        A refill is not a transformation."""
        assert _MANUAL_KIND_TO_TYPE[_manual_op_kind("refill_well123")] != \
               "transformation_offline_operation"

    def test_pick_matches_the_existing_convention(self):
        """Manual repicks already ride as optracker_operation — pipeline.py:558."""
        assert _MANUAL_KIND_TO_TYPE[_manual_op_kind("PICK_25Aug26_well2176911")] == \
               "optracker_operation"

    def test_priority_order_is_intentional(self):
        """REFILL_INNOC is a refill; SUB-NEBstable is a subculture, not just 'a NEB
        thing'. Both ids contain more than one keyword, so order decides."""
        assert _manual_op_kind("REFILL_INNOC_12Feb26_well99") == "REFILL"
        assert _manual_op_kind("SUB-NEBstable_well951") == "SUBCULTURE"

    @pytest.mark.parametrize("junk", [None, "", "no-well-reference", 12345])
    def test_never_raises(self, junk):
        assert _manual_op_kind(junk) in _MANUAL_KIND_TO_TYPE


class TestTypeSafety:
    def test_every_kind_maps_to_a_type_the_renderer_knows(self):
        """THE guard. A new type value silently drops rows from the hardcoded type
        lists in pipeline.py/dashboard.py/inflight.py — nothing would fail, the
        rows would just stop appearing."""
        unknown = {k: v for k, v in _MANUAL_KIND_TO_TYPE.items()
                   if v not in RENDERER_KNOWN_TYPES}
        assert not unknown, (
            f"manual_op_kind maps to type value(s) the renderer does not handle: "
            f"{unknown}. Add them to the type lists in pipeline.py (556/698/986), "
            f"dashboard.py:3000 and inflight.py first, or reuse an existing type."
        )

    def test_every_pattern_has_a_type(self):
        for kind, _ in _MANUAL_KIND_PATTERNS:
            assert kind in _MANUAL_KIND_TO_TYPE, f"{kind} has no type mapping"
        assert "OTHER" in _MANUAL_KIND_TO_TYPE

    def test_renderer_labels_every_type_we_emit(self):
        """inflight.py had no label for optracker_operation at all before v1.11.88,
        so 17 existing rows rendered unlabelled."""
        from dnasc.renderer import inflight
        labels = getattr(inflight, "_TYPE_LABEL", None) or getattr(inflight, "_L2_LABEL", None)
        if labels is None:
            src = Path(inflight.__file__).read_text()
            for t in set(_MANUAL_KIND_TO_TYPE.values()):
                assert f"'{t}':" in src, f"inflight.py has no label for {t}"
        else:
            for t in set(_MANUAL_KIND_TO_TYPE.values()):
                assert t in labels, f"inflight.py has no label for {t}"


def test_discovery_is_structural_not_a_keyword_list():
    """The SQL predicate must not go back to matching STREAK/STBL3/EPI400/TFM.

    That allowlist is what hid NEB_well2202668 and 1015 other process_ids. If it
    reappears, this fails.
    """
    src = (Path(__file__).resolve().parents[1] /
           "dnasc" / "transformers" / "repair.py").read_text()
    fn = src[src.index("def resolve_lims_streakouts"):]
    fn = fn[:fn.index("\ndef ", 1)] if "\ndef " in fn[1:] else fn

    predicates = re.findall(r"REGEXP_CONTAINS\(w\.process_id,\s*r?'([^']+)'", fn)
    assert predicates, "no REGEXP_CONTAINS on w.process_id found — did the query move?"
    for pred in predicates:
        assert "STREAK" not in pred.upper(), (
            f"discovery predicate is keyword-gated again: {pred!r}"
        )
        assert re.search(r"well\[0-9\]\+|well\\\\?d\+", pred), (
            f"discovery predicate no longer matches well<id>: {pred!r}"
        )

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


# ── source well linking (v1.11.92) ───────────────────────────────────────────
class TestSourceWellLinking:
    """228 rows name their well in their own id yet showed no location, while
    ordinary rows showed theirs. All 222 distinct wells resolve in LIMS — the
    information was always there, nothing read it.

    Confirmed against LIMS: well 2202668 is plate17051, "59 - C8".
    """

    def test_extracts_the_well_id_from_every_real_shape(self):
        import re
        shapes = {
            "PARTNER_STREAK_26AUG26_well2170110": "2170110",
            "PARTNER_TFX_2026Aug31_well2202668":  "2202668",
            "REFILL_STREAK_07Jan2026_well1583422": "1583422",
            "refill_well2098847":                 "2098847",
            "STREAK_well1689712":                 "1689712",
            "well17882":                          "17882",
            "SUB_10beta_29Aug26_well123_Soyagar": "123",
        }
        for pid, expected in shapes.items():
            m = re.search(r"well(\d+)", pid, flags=re.I)
            assert m and m.group(1) == expected, pid

    def test_ignores_ids_with_no_well_reference(self):
        import re
        for pid in ("c0d78d82-2e08-404c-841c-128db81a17d4", "LSP-11810", "NEBstable_abc"):
            assert re.search(r"well(\d+)", pid, flags=re.I) is None, pid

    def test_position_shown_is_lims_numbering(self):
        """Julie: "I do not want to see position 58 -> C8, the position is 59."
        The column carries position+1, and the coordinate comes from wells.py."""
        from dnasc import wells
        assert wells.coord(58, 96) == "C8"
        assert wells.lims_number(58) == 59

    def test_is_a_scalar_column_not_the_ops_list(self):
        """`well_location` is a LIST parallel to the ops list. A source well is a
        property of the ROW, so writing a bare string into well_location would
        misalign parse_pipeline_operations."""
        from pathlib import Path
        import dnasc.pipeline as pl
        src = Path(pl.__file__).read_text()
        fn = src[src.index("def _attach_source_well_location"):]
        fn = fn[:fn.index("\ndef ", 1)]
        assert "source_well_location" in fn
        assert '"well_location"' not in fn and "'well_location'" not in fn, \
            "must not write into the parallel ops list"

    def test_renderer_displays_it_for_the_manual_types(self):
        from pathlib import Path
        from dnasc.renderer import dashboard as d
        src = Path(d.__file__).read_text()
        assert "source_well_location" in src, "renderer never reads the column"
        i = src.index("source_well_location")
        block = src[max(0, i - 2500):i]
        for t in ("streakout_operation", "transformation_offline_operation",
                  "optracker_operation"):
            assert t in block, f"the display block does not cover {t}"


# ── strain read from the operation, not inherited (v1.11.93) ─────────────────
class TestStrainInference:
    """Julie: "this is weird how is this epi? i thought we were putting it in neb"
    then "if the parent work order is epi and the glycerol says neb you know it's a
    transformation ... so it's not unknown but a transformation workorder."

    NEB_well2172126 displayed Strain: EPI400 because synthetic rows inherited
    `src.get("cloning_strain")` from the parent workorder. Its own NGS results and
    lims__src.strain.cell_strain both say NEBstable (8 wells each, for all three of
    NEB_well2172126 / 2202611 / 2202668).
    """

    def test_strain_variants_are_the_same_cells(self):
        """LIMS spells these several ways. Comparing raw would read as a strain
        change and mislabel an ordinary streakout a transformation."""
        from dnasc.transformers.repair import _norm_strain as n
        for a, b in [("NEBstable", "NEB_STABLE"), ("NEBStable", "NEBSTBL"),
                     ("NEB_10B", "NEB10beta"), ("EPI400", "EPI400")]:
            assert n(a) == n(b), f"{a} vs {b}"

    def test_a_real_strain_change_is_detected(self):
        from dnasc.transformers.repair import _norm_strain as n
        assert n("EPI400") != n("NEBstable")
        assert n("STBL3") != n("EPI400")

    def test_query_reads_cell_strain_per_process_not_per_well(self):
        """Joining well_content directly gave 1.93 rows per process_id, and since the
        caller keeps the first row a NULL-strain row could win and fall back to the
        parent — the bug itself. The aggregate keeps it at 1.00."""
        from pathlib import Path
        import dnasc.transformers.repair as r
        src = Path(r.__file__).read_text()
        fn = src[src.index("def resolve_lims_streakouts"):]
        fn = fn[:fn.index("\ndef ", 1)]
        assert "own_cell_strain" in fn, "the query must return the operation's own strain"
        assert "cell_strain" in fn
        assert "ORDER BY n DESC, cell_strain" in fn, \
            "the pick must be deterministic — 8 process_ids have wells that disagree"

    def test_strain_change_forces_the_transformation_type(self):
        from pathlib import Path
        import dnasc.transformers.repair as r
        src = Path(r.__file__).read_text()
        fn = src[src.index("def resolve_lims_streakouts"):]
        fn = fn[:fn.index("\ndef ", 1)]
        assert "strain_changed" in fn
        i = fn.index("if strain_changed:")
        assert '"transformation_offline_operation"' in fn[i:i + 220], \
            "a strain change must type the row as a transformation"

    def test_the_row_reports_its_own_strain(self):
        """cloning_strain must be what the material IS in, with the parent kept
        separately rather than overwriting it."""
        from pathlib import Path
        import dnasc.transformers.repair as r
        src = Path(r.__file__).read_text()
        fn = src[src.index("def resolve_lims_streakouts"):]
        fn = fn[:fn.index("\ndef ", 1)]
        assert '"cloning_strain":        row_strain' in fn
        assert '"parent_cloning_strain": parent_strain' in fn
        assert "row_strain = own_strain or parent_strain" in fn, \
            "fall back to the parent only when LIMS has no strain of its own"


class TestDisplayLabels:
    """v1.11.94 — what these rows are CALLED.

    Julie, on a manual op reading "Optracker Operation": if it is not in OpTracker
    then it is manual — not software aided. Backwards AND overclaiming: these are
    exactly the ops NOT in OpTracker. Not "unknown" either — every kind landing in
    the catch-all is identified (PICK/REFILL/INNOC/GLYCEROL_CHECK/EXT); the ones
    that are transformations or streakouts are typed as those. And on the
    offline-transformation badge: keep the
    transformation_offline_operation type (it already exists with its own
    badges) rather than promoting these rows to transformation_workorder,
    which would make a bench op with no workorder behind it indistinguishable
    from the ~2,700 real ones.

    So: a strain change -> transformation_offline_operation, badged with the real
    strain; everything else manual -> "Manual Operation".
    """

    @staticmethod
    def _src(mod):
        from pathlib import Path
        return Path(mod.__file__).read_text()

    def test_dashboard_labels_it_manual_not_optracker(self):
        """Not in OpTracker => manual. The label must not name the plumbing, and
        must not claim we do not know what happened."""
        from dnasc.renderer import dashboard as d
        src = self._src(d)
        assert "'optracker_operation': 'Manual Operation'" in src
        assert "_TYPE_LABEL_OVERRIDES" in src
        assert "Unknown Operation" not in src, \
            "these ops are identified by kind, not unknown"

    def test_the_override_is_consulted_before_the_generic_titlecase(self):
        """An override map that format_type_label never reads is decoration."""
        from dnasc.renderer import dashboard as d
        src = self._src(d)
        i = src.index("def format_type_label(")
        body = src[i:src.index("\n    #", i + 1)]
        assert "_TYPE_LABEL_OVERRIDES" in body, \
            "format_type_label must consult the override map"
        assert body.index("_TYPE_LABEL_OVERRIDES") < body.index(".title()"), \
            "the override must win over the generic title-casing"

    def test_inflight_agrees_with_the_dashboard(self):
        """Two tabs labelling the same type two different things is how 'Manual Op'
        and 'Optracker Operation' coexisted. Short form, same word."""
        from dnasc.renderer import inflight as f
        assert f._DTYPE_LABEL['optracker_operation'] == 'Manual Op'
        assert 'Unknown' not in self._src(f)

    def test_offline_transformation_keeps_its_own_type(self):
        """Julie's decision: do NOT retype these as transformation_workorder."""
        from dnasc.renderer import inflight as f
        from dnasc.transformers import repair as r
        assert 'transformation_offline_operation' in f._DTYPE_LABEL
        assert set(r._MANUAL_KIND_TO_TYPE.values()) <= {
            'streakout_operation', 'transformation_offline_operation',
            'optracker_operation',
        }, "a manual op must never be typed as a real BIOS workorder"
        assert 'transformation_workorder' not in set(r._MANUAL_KIND_TO_TYPE.values())

    def test_the_badge_reads_the_strain_not_the_parent_id_text(self):
        """The badge string-matched 'epi400'/'stbl3' in the PARENT's id, so a row
        whose parent is a UUID fell through to a generic badge — NEB_well2172126
        showed "Offline Trans" while its glycerol was NEBstable. cloning_strain now
        carries the real strain, so read it."""
        from dnasc.renderer import dashboard as d
        src = self._src(d)
        i = src.index("elif row['type'] == 'transformation_offline_operation':")
        block = src[i:i + 1400]
        assert "row.get('cloning_strain')" in block, \
            "the badge must read the strain, not infer it from the parent's id text"
        assert block.index("cloning_strain") < block.index("'STBL3' in suffix"), \
            "the strain must be preferred over the id-text fallback"
        assert "Offline</span>" in block

    def test_the_badge_never_prints_a_missing_strain(self):
        """cloning_strain is NaN/None for rows LIMS has no strain for; those must
        fall back, not render "nan Offline"."""
        from dnasc.renderer import dashboard as d
        src = self._src(d)
        i = src.index("elif row['type'] == 'transformation_offline_operation':")
        block = src[i:i + 1400]
        assert "pd.isna" in block
        assert "'nan'" in block and "'none'" in block, \
            "guard the string forms too — astype(str) on a NaN gives 'nan'"


    def test_the_catch_all_holds_no_transformation_or_streakout_kind(self):
        """What "Manual Operation" is allowed to cover. A kind that IS a
        transformation or a streakout must be typed as one, or the manual label
        starts absorbing things we can name — which is what sent 11 PARTNER_TFX
        and 1 SUB_NEBStable row to the catch-all before v1.11.88."""
        from dnasc.transformers import repair as r
        catch_all = {k for k, v in r._MANUAL_KIND_TO_TYPE.items()
                     if v == 'optracker_operation'}
        assert catch_all == {'PICK', 'REFILL', 'INNOC', 'GLYCEROL_CHECK',
                             'EXT', 'OTHER'}, catch_all
        for kind in ('TFX', 'SUBCULTURE', 'STRAIN'):
            assert r._MANUAL_KIND_TO_TYPE[kind] == 'transformation_offline_operation'
        assert r._MANUAL_KIND_TO_TYPE['STREAK'] == 'streakout_operation'

    def test_the_real_prefixes_seen_in_the_data_all_resolve(self):
        """The three prefixes actually present, measured 2026-09-02. None is
        unknown; only the pick belongs in the manual catch-all."""
        from dnasc.transformers import repair as r
        for pid, kind, typ in [
            ('PARTNER_TFX_2026Aug31_well2202668', 'TFX', 'transformation_offline_operation'),
            ('SUB_NEBStable_11May2026_well1745459', 'SUBCULTURE', 'transformation_offline_operation'),
            ('PICK_25Aug26_well2156655', 'PICK', 'optracker_operation'),
        ]:
            assert r._manual_op_kind(pid) == kind, pid
            assert r._MANUAL_KIND_TO_TYPE[r._manual_op_kind(pid)] == typ, pid

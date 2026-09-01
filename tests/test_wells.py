"""
Guards dnasc/wells.py as THE single definition of position → plate coordinate.

Six independent copies existed before it, and they had drifted: optracker.py was
row-major (so every coordinate the Tracking tab displayed was wrong) and treated
8-well agar plates as 8 rows where the renderers use 2. The interesting test here
is not the arithmetic — it is `test_no_other_module_implements_coordinates`, which
fails the build if anyone writes the formula a seventh time.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dnasc import wells


# (well_id, plate, well_count, raw 0-based position, LIMS number, coordinate)
# Every row confirmed by Julie against LIMS on 2026-09-01, chosen so column-major
# and row-major disagree — row-major would give C2, E11, A8, B1, A16 respectively.
LIMS_GROUND_TRUTH = [
    (1455307, "plate10991",  96, 25, 26, "B4"),
    (2202668, "plate17051",  96, 58, 59, "C8"),
    (2242513, "plate17358",  96,  7,  8, "H1"),
    (2242514, "plate17358",  96,  8,  9, "A2"),
    (2242617, "plate17359", 384, 15, 16, "P1"),
    (2238371, "plate17329",   8,  1,  2, "B1"),
]


class TestAgainstLIMS:
    @pytest.mark.parametrize("well_id,plate,wc,pos,lims_no,coord", LIMS_GROUND_TRUTH)
    def test_coord_matches_lims(self, well_id, plate, wc, pos, lims_no, coord):
        assert wells.coord(pos, wc) == coord, f"well {well_id} on {plate}"

    @pytest.mark.parametrize("well_id,plate,wc,pos,lims_no,coord", LIMS_GROUND_TRUTH)
    def test_lims_number_is_position_plus_one(self, well_id, plate, wc, pos, lims_no, coord):
        """Rule 3: the number shown to a person is position+1."""
        assert wells.lims_number(pos) == lims_no

    @pytest.mark.parametrize("well_id,plate,wc,pos,lims_no,coord", LIMS_GROUND_TRUTH)
    def test_label_reads_like_lims(self, well_id, plate, wc, pos, lims_no, coord):
        assert wells.label(pos, wc) == f"{lims_no} - {coord}"

    def test_row_major_would_fail_these(self):
        """Proves the ground truth actually discriminates, rather than passing either way."""
        for _, _, wc, pos, _, coord in LIMS_GROUND_TRUTH:
            if wc == 8:
                continue                      # 2 vs 8 rows agree at position 1
            cols = wc // wells.rows_for(wc)
            row_major = f"{chr(ord('A') + pos // cols)}{pos % cols + 1}"
            assert row_major != coord, f"pos {pos} on {wc}-well does not discriminate"


class TestColumnMajor:
    def test_96_fills_down_the_column(self):
        assert [wells.coord(i, 96) for i in range(9)] == \
               ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "A2"]

    def test_384_fills_down_the_column(self):
        assert wells.coord(0, 384) == "A1"
        assert wells.coord(15, 384) == "P1"
        assert wells.coord(16, 384) == "A2"
        assert wells.coord(383, 384) == "P24"

    def test_last_well_is_bottom_right(self):
        assert wells.coord(95, 96) == "H12"
        assert wells.coord(7, 8) == "B4"

    def test_unknown_well_count_assumes_96(self):
        assert wells.coord(8, None) == wells.coord(8, 96)
        assert wells.coord(8, 0) == wells.coord(8, 96)

    @pytest.mark.parametrize("bad", [None, "", "abc", -1, -100])
    def test_unusable_position_is_empty_not_an_exception(self, bad):
        assert wells.coord(bad, 96) == ""

    def test_coord_and_coord_rows_agree(self):
        for wc in (8, 96, 384):
            rows = wells.rows_for(wc)
            for pos in range(wc):
                assert wells.coord(pos, wc) == wells.coord_rows(pos, rows)


class TestCoordMap:
    """coord_map is keyed 1-BASED — its callers already hold a 1-based key."""

    @pytest.mark.parametrize("wc", [8, 96, 384])
    def test_map_is_one_based_and_complete(self, wc):
        m = wells.coord_map(wc)
        assert len(m) == wc
        assert m["1"] == "A1"
        for pos in range(wc):
            assert m[str(pos + 1)] == wells.coord(pos, wc)

    def test_matches_the_dicts_the_renderer_uses(self):
        """These literals were hand-written in dashboard.py; pin the values so the
        generated maps can never silently differ from what shipped."""
        m96, m384, m8 = wells.coord_map(96), wells.coord_map(384), wells.coord_map(8)
        assert (m96["1"], m96["8"], m96["9"], m96["96"]) == ("A1", "H1", "A2", "H12")
        assert (m384["1"], m384["16"], m384["17"], m384["384"]) == ("A1", "P1", "A2", "P24")
        assert (m8["1"], m8["2"], m8["8"]) == ("A1", "B1", "B4")

    def test_dashboard_uses_the_generated_maps(self):
        from dnasc.renderer import dashboard as d
        assert d._WELL_MAP_96 == wells.coord_map(96)
        assert d._WELL_MAP_384 == wells.coord_map(384)
        assert d._WELL_MAP_AGAR == wells.coord_map(8)


class TestSQLMatchesPython:
    def test_sql_is_built_not_hardcoded(self):
        sql = wells.sql_coord("w.position", "pl.well_count")
        assert "w.position" in sql and "pl.well_count" in sql
        assert "CHR(65 + MOD(" in sql
        # the old row-major forms must not reappear
        assert "/ 12" not in sql and "/ 24" not in sql

    def test_rows_expr_covers_every_supported_size(self):
        expr = wells.sql_rows_expr("pl.well_count")
        for wc, rows in wells.ROWS_BY_WELL_COUNT.items():
            assert f"WHEN {wc} THEN {rows}" in expr
        assert f"ELSE {wells.DEFAULT_ROWS}" in expr

    @pytest.mark.parametrize("wc", [8, 96, 384])
    def test_bigquery_agrees_with_python_on_every_position(self, wc):
        """The one test that can catch SQL/Python divergence for real. Skips when
        BigQuery is unreachable so the offline suite still runs."""
        try:
            from google.cloud import bigquery
            from dnasc.config import PipelineConfig as C
            client = bigquery.Client(project=C.PROJECT_ID)
        except Exception as exc:
            pytest.skip(f"BigQuery unreachable: {type(exc).__name__}: {exc}")

        expr = wells.sql_coord("pos", "wc")
        sql = f"""
        SELECT pos, {expr} AS coord
        FROM UNNEST(GENERATE_ARRAY(0, {wc - 1})) AS pos, UNNEST([{wc}]) AS wc
        ORDER BY pos
        """
        try:
            rows = list(client.query(sql).result())
        except Exception as exc:
            pytest.skip(f"BigQuery query failed: {type(exc).__name__}: {exc}")

        assert len(rows) == wc
        mismatches = [(r.pos, r.coord, wells.coord(r.pos, wc))
                      for r in rows if r.coord != wells.coord(r.pos, wc)]
        assert not mismatches, f"{wc}-well SQL/Python divergence: {mismatches[:5]}"


def test_no_other_module_implements_coordinates():
    """THE guard. Nothing but dnasc/wells.py may do the position arithmetic.

    This is what the whole module exists for: six copies had drifted and one was
    wrong for months without anything failing. A seventh copy fails here instead.
    """
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "dnasc" / "wells.py").resolve()

    # chr(65 + ...) / CHR(65 + ...) / chr(ord('A') + ...) — the letter arithmetic.
    letter_math = re.compile(r"(?:chr|CHR)\s*\(\s*(?:65|ord\s*\(\s*['\"]A['\"]\s*\))\s*\+")

    offenders = []
    for path in sorted((root / "dnasc").rglob("*.py")):
        if path.resolve() == canonical:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue                      # a comment describing it is fine
            if letter_math.search(line):
                offenders.append(f"  {path.relative_to(root)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "position→coordinate arithmetic outside dnasc/wells.py "
        f"({len(offenders)} site(s)) — call wells.coord()/coord_rows()/sql_coord() "
        "instead:\n" + "\n".join(offenders)
    )

"""
dnasc/wells.py
──────────────
THE single definition of LIMS well position → plate coordinate. Nothing else in
this repo may implement it; tests/test_wells.py fails the build if anything does.

Six independent copies existed before this module, and they had drifted:
extractors/optracker.py computed row-major (`/12`, `/24`) so every coordinate the
Tracking tab showed was misplaced — well 2170110 read C5 where it is E4 — while
renderer/{parts,ngs,inflight}.py and dashboard.py's _WELL_MAP_* dicts were all
column-major and correct. optracker.py also treated 8-well agar plates as 8 rows
where the other three use 2, so those disagreed from position 2 onward.

THE RULES
---------
1. Positions are COLUMN-MAJOR: they fill down a column before moving right. On a
   96-well plate A1=0, B1=1, ... H1=7, A2=8, ... H12=95.

2. BigQuery `lims__src.well.position` is 0-BASED. Every function here takes that
   raw value.

3. Every number shown to a person is position+1, matching LIMS: raw 58 is
   "position 59". Use lims_number(). Never quote the raw value at a person.

4. Row count comes from plate.well_count, NOT the labware name — a "384 Echo
   Source Plate" can be physically 96-well.

5. The +1 belongs at the display edge, once. coord_map() is keyed 1-based because
   its callers already hold a 1-based key; coord() takes the raw 0-based value.
   Applying +1 in both places shifts every well one row down.

VERIFIED against LIMS (2026-09-01), on wells chosen so column-major and row-major
disagree:

    well 1455307 / plate10991   96   pos 25 -> "26 - B4"   row-major: C2
    well 2202668 / plate17051   96   pos 58 -> "59 - C8"   row-major: E11
    well 2242513 / plate17358   96   pos  7 -> "8 - H1"    row-major: A8
    well 2242514 / plate17358   96   pos  8 -> "9 - A2"    row-major: B1
    well 2242617 / plate17359  384   pos 15 -> "16 - P1"   row-major: A16
    well 2238371 / plate17329    8   pos  1 -> "2 - B1"

Only position 1 of the 8-well layout is confirmed, and 2 rows vs 8 rows agree
there. ROWS_BY_WELL_COUNT[8] = 2 follows dashboard.py's _WELL_MAP_AGAR and
inflight.py, which have long used 2 — not an independent LIMS check.
"""
from __future__ import annotations

# well_count → number of rows. Anything unlisted falls back to DEFAULT_ROWS.
ROWS_BY_WELL_COUNT: dict[int, int] = {
    384: 16,   # 16 x 24
    96:   8,   # 8 x 12
    24:   4,   # 4 x 6
    8:    2,   # agar plates, 2 x 4 — per _WELL_MAP_AGAR / inflight.py
    6:    2,   # 2 x 3
}
DEFAULT_ROWS = 8


def rows_for(well_count) -> int:
    """Rows for a plate of this well_count. Unknown/missing sizes assume 96-well."""
    try:
        return ROWS_BY_WELL_COUNT[int(well_count)]
    except (TypeError, ValueError, KeyError):
        return DEFAULT_ROWS


def coord(position, well_count=96) -> str:
    """RAW 0-based position → coordinate, e.g. (58, 96) -> 'C8'.

    Returns "" for anything unparseable or negative, so callers can drop it
    straight into a template.
    """
    try:
        p = int(float(position))
    except (TypeError, ValueError):
        return ""
    if p < 0:
        return ""
    return coord_rows(p, rows_for(well_count))


def coord_rows(position, rows: int = DEFAULT_ROWS) -> str:
    """As coord(), but the caller already knows the row count instead of the
    well_count. Same column-major rule; still takes the RAW 0-based position."""
    try:
        p = int(float(position))
    except (TypeError, ValueError):
        return ""
    if p < 0 or rows <= 0:
        return ""
    return f"{chr(ord('A') + p % rows)}{p // rows + 1}"


def lims_number(position) -> int | None:
    """RAW 0-based position → the number LIMS displays (position+1). Rule 3."""
    try:
        return int(float(position)) + 1
    except (TypeError, ValueError):
        return None


def label(position, well_count=96) -> str:
    """How LIMS writes it: '59 - C8'. Empty string if position is unusable."""
    n, c = lims_number(position), coord(position, well_count)
    return f"{n} - {c}" if (n is not None and c) else ""


def coord_map(well_count=96) -> dict[str, str]:
    """{'1': 'A1', '2': 'B1', ...} keyed 1-BASED, for callers holding a 1-based
    key (dashboard.py's _WELL_MAP_* do). Generated here so the mapping cannot
    drift from coord()."""
    try:
        n = int(well_count)
    except (TypeError, ValueError):
        n = 96
    return {str(i + 1): coord(i, n) for i in range(n)}


def sql_rows_expr(well_count_expr: str) -> str:
    """BigQuery expression for the row count of a plate, from its well_count column."""
    cases = " ".join(
        f"WHEN {wc} THEN {rows}"
        for wc, rows in sorted(ROWS_BY_WELL_COUNT.items(), reverse=True)
    )
    return f"CASE {well_count_expr} {cases} ELSE {DEFAULT_ROWS} END"


def sql_coord(position_expr: str, well_count_expr: str) -> str:
    """BigQuery expression: RAW 0-based position column → coordinate string.

    Mirrors coord() exactly; tests/test_wells.py checks the two agree on every
    position of every supported plate size, so the SQL cannot drift from Python.
    """
    rows = sql_rows_expr(well_count_expr)
    return (
        f"CONCAT("
        f"CHR(65 + MOD({position_expr}, {rows})), "
        f"CAST(DIV({position_expr}, {rows}) + 1 AS STRING)"
        f")"
    )

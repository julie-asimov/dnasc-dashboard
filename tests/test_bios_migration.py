"""
Guards the Op Tracker → bios BigQuery migration (cutover 2026-08-28 21:27 UTC).

Two things have to stay true, and they fail in opposite directions:

  1. The legacy history survived. `bios__src` was backfilled from
     `op_tracker__src`, so every legacy id must still be present with identical
     values. A truncated or re-derived backfill would silently rewrite years of
     TAT history without erroring — the pipeline would just start reporting
     different numbers for 2020-2026.

  2. New data is arriving. `op_tracker__src` stopped receiving writes at the
     cutover. If the pipeline's queries still point at the legacy dataset, or
     if bios replication stalls, the dashboard goes stale *silently* — the
     queries succeed, they just stop seeing new operations.

`test_no_legacy_table_references_remain` is a source scan and needs no
BigQuery. The rest query live BigQuery and skip if it is unreachable, so the
offline suite still runs.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT   = "data-platform-core-prd"
LEGACY_DS = "op_tracker__src"
BIOS_DS   = "bios__src"

# Last write the legacy Django app made before the bios cutover.
LEGACY_FROZEN_AT = datetime(2026, 8, 28, 21, 27, 23, tzinfo=timezone.utc)

# Tolerate a quiet weekend before calling replication stalled. Lab volume drops
# to zero Fri evening → Mon morning, so anything under ~60h gives false alarms.
FRESHNESS_HOURS = 72

# Bound the row-level value comparison so the test stays cheap (parameter is 4M
# rows). The legacy parameter table has no date_created, so window via operation.
VALUE_CHECK_DAYS = 90

# Every table the pipeline actually reads: (bios name, legacy name).
TABLE_PAIRS = [
    ("operation",     "op_tracker_api_operation"),
    ("job",           "op_tracker_api_job"),
    ("protocol",      "op_tracker_api_protocol"),
    ("parameter",     "op_tracker_api_parameter"),
    ("parametertype", "op_tracker_api_parametertype"),
]

# `parameter` is checked at a different grain — see
# test_bios_covers_every_legacy_parameter_slot for why its ids legitimately churn.
ID_STABLE_PAIRS = [p for p in TABLE_PAIRS if p[0] != "parameter"]

# Columns the pipeline depends on, and the BQ type it depends on them having.
#
# The types are load-bearing, not decorative:
#   * parameter.value and job.step_groups MUST stay STRING. The extractor runs
#     REGEXP_CONTAINS() on step_groups, REPLACE(value, '"', '') on value, and
#     MAX(CASE WHEN ...) pivots over value. All three are illegal against a
#     native BQ JSON column, so a replication change to JSON would break Step 1
#     outright — and the migration guide flags JSON-vs-STRING as unconfirmed.
#   * the id columns MUST stay INTEGER. excluded_optracker_jobs.csv holds
#     integer job ids, the queries build integer IN-lists, and the baseline
#     parquet stores operation_id/job_id as integers.
REQUIRED_COLUMNS = {
    "operation": {
        "id": "INTEGER", "job_id": "INTEGER", "plan_id": "INTEGER",
        "protocol_id": "INTEGER", "state": "STRING",
        "date_created": "TIMESTAMP", "date_ready": "TIMESTAMP",
    },
    "job": {
        "id": "INTEGER", "protocol_id": "INTEGER", "step_groups": "STRING",
    },
    "protocol": {
        "id": "INTEGER", "name": "STRING",
    },
    "parameter": {
        "operation_id": "INTEGER", "parameter_type_id": "INTEGER", "value": "STRING",
    },
    "parametertype": {
        "id": "INTEGER", "name": "STRING",
    },
}


# ── live-BigQuery plumbing ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bq():
    """Live BigQuery client, or skip the whole module if BQ is unreachable."""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT)
        client.query("SELECT 1").result()
    except Exception as exc:  # no creds, no network, API disabled
        pytest.skip(f"BigQuery unreachable: {type(exc).__name__}: {exc}")
    return client


def _one(client, sql):
    """Run sql and return its single result row."""
    return list(client.query(sql))[0]


# ── 1. legacy history is preserved ───────────────────────────────────────────

@pytest.mark.parametrize("bios_tbl,legacy_tbl", ID_STABLE_PAIRS, ids=[p[0] for p in ID_STABLE_PAIRS])
def test_bios_contains_every_legacy_id(bq, bios_tbl, legacy_tbl):
    """bios must be a superset of legacy — nothing dropped by the backfill.

    This is also why we replace the legacy references rather than UNION the two
    datasets: a union would double-count all 353k historical operations.
    """
    row = _one(bq, f"""
        SELECT
          (SELECT COUNT(*) FROM (
             SELECT id FROM `{PROJECT}.{LEGACY_DS}.{legacy_tbl}`
             EXCEPT DISTINCT
             SELECT id FROM `{PROJECT}.{BIOS_DS}.{bios_tbl}`
          )) AS legacy_only,
          (SELECT COUNT(*) FROM `{PROJECT}.{LEGACY_DS}.{legacy_tbl}`) AS legacy_rows
    """)
    assert row.legacy_rows > 0, f"legacy {legacy_tbl} is empty — cannot verify the backfill"
    assert row.legacy_only == 0, (
        f"{row.legacy_only} ids exist in {LEGACY_DS}.{legacy_tbl} but not in "
        f"{BIOS_DS}.{bios_tbl} — the backfill lost history, do NOT drop the legacy table"
    )


def test_legacy_operation_values_unchanged(bq):
    """Backfilled operations must carry identical timestamps, not re-derived ones.

    date_created is the one to watch. Legacy Django set auto_now=True, making it
    a last-modified stamp; bios sets it insert-only. The backfill copied the old
    values, so history is stable — but if a future re-backfill regenerated them,
    every historical operation_start on the dashboard would shift by hours to
    days. Assert exact equality so that shows up here instead of in a chart.

    state and job_id are allowed to differ: 128 operations legitimately advanced
    (RD → RU) after the legacy table froze. Only forward drift is tolerated.
    """
    row = _one(bq, f"""
        SELECT
          COUNT(*) AS joined,
          COUNTIF(l.date_created != b.date_created) AS date_created_mismatch,
          COUNTIF(l.protocol_id IS DISTINCT FROM b.protocol_id) AS protocol_mismatch,
          COUNTIF(l.plan_id IS DISTINCT FROM b.plan_id) AS plan_mismatch
        FROM `{PROJECT}.{LEGACY_DS}.op_tracker_api_operation` l
        JOIN `{PROJECT}.{BIOS_DS}.operation` b USING(id)
    """)
    assert row.joined > 300_000, f"only {row.joined:,} operations joined — backfill looks partial"
    assert row.date_created_mismatch == 0, (
        f"{row.date_created_mismatch:,} operations have a different date_created in bios — "
        "historical TAT/timeline values would shift"
    )
    assert row.protocol_mismatch == 0, f"{row.protocol_mismatch:,} operations changed protocol_id"
    assert row.plan_mismatch == 0, f"{row.plan_mismatch:,} operations changed plan_id"


def test_bios_covers_every_legacy_parameter_slot(bq):
    """Every legacy (operation, parameter_type) slot must still exist in bios.

    `parameter` is the one table whose ids legitimately churn, so it cannot be
    checked at id grain like the others. When an operator re-stamps a parameter,
    the source row is deleted and reinserted with a *new* id — bios reflects live
    Postgres state, while the frozen legacy snapshot still holds the pre-restamp
    id. Asserting id-level containment therefore fails on normal activity (68
    `Op Input` params on `Digest` ops churned this way within 90 minutes of the
    cutover check).

    The grain that actually matters is (operation_id, parameter_type_id), because
    that is how every pipeline query joins parameters — `parameter.id` is never
    used. A slot disappearing means the dashboard loses a value; a new id under
    the same slot means nothing to it.
    """
    row = _one(bq, f"""
        SELECT COUNT(*) AS legacy_only_slots
        FROM (
          SELECT operation_id, parameter_type_id
          FROM `{PROJECT}.{LEGACY_DS}.op_tracker_api_parameter`
          EXCEPT DISTINCT
          SELECT operation_id, parameter_type_id
          FROM `{PROJECT}.{BIOS_DS}.parameter`
        )
    """)
    assert row.legacy_only_slots == 0, (
        f"{row.legacy_only_slots:,} (operation, parameter_type) slots exist in legacy "
        f"but not in bios — the dashboard would lose those parameter values"
    )


def test_frozen_parameter_values_unchanged(bq):
    """Parameter values on finished operations must survive byte-identically.

    Everything the dashboard identifies a construct by is a parameter value —
    Process, Product, Experiment, Result Status, the well JSON blobs. A changed
    or re-encoded value here silently reshapes the whole dashboard.

    Scoped to terminal-state operations (SC/FA/CA) created before the cutover:
    that is settled history, which must never move. Parameters on still-active
    operations are deliberately excluded — an operator re-stamping a value on a
    live op is real work, not a replication fault.
    """
    cutover = LEGACY_FROZEN_AT.strftime("%Y-%m-%d %H:%M:%S")
    row = _one(bq, f"""
        SELECT
          COUNT(*) AS joined,
          COUNTIF(l.value != b.value) AS value_mismatch
        FROM `{PROJECT}.{LEGACY_DS}.op_tracker_api_parameter` l
        JOIN `{PROJECT}.{BIOS_DS}.parameter` b USING(id)
        JOIN `{PROJECT}.{BIOS_DS}.operation` o ON o.id = b.operation_id
        WHERE o.state IN ('SC', 'FA', 'CA')
          AND o.date_created < TIMESTAMP '{cutover}'
          AND o.date_created >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {VALUE_CHECK_DAYS * 24} HOUR)
    """)
    assert row.joined > 0, f"no finished-op parameters in the last {VALUE_CHECK_DAYS} days to compare"
    assert row.value_mismatch == 0, (
        f"{row.value_mismatch:,} parameter values on finished operations differ "
        "between legacy and bios"
    )


def test_full_history_depth_retained(bq):
    """bios must reach as far back as legacy did (2020-03-16), not just recent data.

    A backfill windowed to "last N months" would pass the superset test on
    recent ids while quietly truncating the archive.
    """
    row = _one(bq, f"""
        SELECT
          (SELECT MIN(date_created) FROM `{PROJECT}.{LEGACY_DS}.op_tracker_api_operation`) AS legacy_earliest,
          (SELECT MIN(date_created) FROM `{PROJECT}.{BIOS_DS}.operation`) AS bios_earliest
    """)
    assert row.bios_earliest <= row.legacy_earliest, (
        f"bios history starts {row.bios_earliest}, legacy starts {row.legacy_earliest} — "
        "the bios backfill is truncated"
    )


# ── 2. new data is arriving ──────────────────────────────────────────────────

def test_new_operations_exist_after_cutover(bq):
    """bios must hold operations created after the legacy table froze.

    This is the direct "new stuff is coming in" assertion: rows that exist in
    bios and cannot exist in legacy.
    """
    cutover = LEGACY_FROZEN_AT.strftime("%Y-%m-%d %H:%M:%S")
    row = _one(bq, f"""
        SELECT COUNT(*) AS n, MAX(date_created) AS latest
        FROM `{PROJECT}.{BIOS_DS}.operation`
        WHERE date_created > TIMESTAMP '{cutover}'
    """)
    assert row.n > 0, (
        f"no operations in bios after the cutover ({cutover}) — bios replication "
        "never started or has stalled"
    )


def test_bios_is_fresh(bq):
    """bios replication must be current, not lagging."""
    row = _one(bq, f"""
        SELECT
          MAX(date_created) AS latest,
          TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(date_created), HOUR) AS age_hours
        FROM `{PROJECT}.{BIOS_DS}.operation`
    """)
    assert row.age_hours <= FRESHNESS_HOURS, (
        f"newest bios operation is {row.age_hours}h old ({row.latest}), over the "
        f"{FRESHNESS_HOURS}h threshold — bios → BigQuery replication has stalled"
    )


def test_bios_is_ahead_of_legacy(bq):
    """bios must be strictly newer than legacy.

    The canary for pointing at the wrong dataset. If this ever inverts, the
    legacy table resumed writes and the migration assumption is wrong.
    """
    row = _one(bq, f"""
        SELECT
          (SELECT MAX(date_created) FROM `{PROJECT}.{LEGACY_DS}.op_tracker_api_operation`) AS legacy_latest,
          (SELECT MAX(date_created) FROM `{PROJECT}.{BIOS_DS}.operation`) AS bios_latest
    """)
    assert row.bios_latest > row.legacy_latest, (
        f"bios latest ({row.bios_latest}) is not ahead of legacy ({row.legacy_latest}) — "
        "bios is not the live source"
    )


# ── 3. the schema contract the queries rely on ───────────────────────────────

@pytest.mark.parametrize("table", sorted(REQUIRED_COLUMNS), ids=sorted(REQUIRED_COLUMNS))
def test_bios_schema_contract(bq, table):
    """Columns the pipeline reads must exist with the types it assumes.

    See REQUIRED_COLUMNS for why STRING-vs-JSON on value/step_groups and
    INTEGER on the id columns are load-bearing rather than cosmetic.
    """
    schema = {f.name: f.field_type for f in bq.get_table(f"{PROJECT}.{BIOS_DS}.{table}").schema}
    for col, expected in REQUIRED_COLUMNS[table].items():
        assert col in schema, (
            f"{BIOS_DS}.{table} is missing column '{col}' that the pipeline reads "
            f"(present: {sorted(schema)})"
        )
        assert schema[col] == expected, (
            f"{BIOS_DS}.{table}.{col} is {schema[col]}, expected {expected}"
        )


# ── 4. the pipeline no longer reads the frozen dataset ───────────────────────

def test_step_ts_has_one_definition():
    """Every timeline query must call PipelineConfig.sql_step_ts(), not inline the
    cutover comparison.

    The op-vs-job split is subtle enough that I got it wrong on the first attempt —
    keying off o.date_created instead of j.date_created did nothing for the very
    rows it was meant to fix. Six hand-written copies of the well-coordinate
    formula is how optracker.py stayed row-major and wrong for months, so this one
    gets a single definition and a test.
    """
    pkg = Path(__file__).resolve().parents[1] / "dnasc"
    config = (pkg / "config.py").resolve()

    inline = re.compile(r"date_created\s*>=\s*TIMESTAMP")
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        if path.resolve() == config:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith(("#", "--")):
                continue
            if inline.search(line):
                offenders.append(f"  {path.relative_to(pkg.parent)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "cutover comparison inlined instead of calling PipelineConfig.sql_step_ts():\n"
        + "\n".join(offenders)
    )


def test_every_timeline_query_uses_step_ts():
    """The four repair.py timeline SELECTs and optracker.py must alias step_ts.

    repair.py feeds operation_start for synthetic and downstream-plate rows; if one
    of these silently reverts to o.date_created, those rows show queue time again —
    the bug that made an NGS step run on 08-31 print 08/11.
    """
    pkg = Path(__file__).resolve().parents[1] / "dnasc"
    repair = (pkg / "transformers" / "repair.py").read_text()
    optracker = (pkg / "extractors" / "optracker.py").read_text()

    assert repair.count("{step_ts} AS date_created") == 4, (
        "expected 4 timeline SELECTs in repair.py aliasing step_ts, found "
        f"{repair.count('{step_ts} AS date_created')}"
    )
    assert repair.count("PipelineConfig.sql_step_ts()") == 2, \
        "both repair.py functions must define step_ts from the shared helper"
    assert "{step_ts} AS step_ts" in optracker and \
           "PipelineConfig.sql_step_ts()" in optracker, \
           "optracker.py must use the shared helper too"

    # and the raw column must still gate the data window, not the step time
    assert "o.date_created >= '{date_filter}'" in repair, \
        "WHERE clauses should still filter raw o.date_created (data window)"


def test_no_legacy_table_references_remain():
    """No dnasc/ query may reference the frozen legacy dataset.

    Needs no BigQuery. This is the regression guard that matters most: legacy
    references do not error, they just return stale data, so nothing else in the
    suite would catch a reintroduced one.
    """
    pkg = Path(__file__).resolve().parents[1] / "dnasc"
    pattern = re.compile(r"op_tracker__src|op_tracker_api_")

    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                rel = path.relative_to(pkg.parent)
                offenders.append(f"  {rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        f"{len(offenders)} reference(s) to the frozen legacy Op Tracker dataset "
        f"(no new data since {LEGACY_FROZEN_AT:%Y-%m-%d %H:%M} UTC):\n"
        + "\n".join(offenders)
    )

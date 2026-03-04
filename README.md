# DNASC Dashboard

An automated dashboard for tracking DNA and Strain Construction (DNASC) pipeline progress across projects and experiments.

## What it does

Pulls data from BigQuery (OpTracker + LIMS) and generates an interactive HTML dashboard showing:

- **Per-experiment status** — which projects are active, waiting, or complete
- **Workorder timeline** — each step from LSP through assembly to strain construction
- **Colony / sequencing results** — pass/fail counts pulled directly from LIMS
- **NGS and downstream ops** — quant and sequencing confirmation linked to each construct
- **Phase detection** — automatically surfaces what is blocking progress (LSP, Assembly, or Parts)

The dashboard is regenerated hourly by a cron job

## Tech stack

- Python 3.10+
- BigQuery (`google-cloud-bigquery`, `pandas-gbq`)
- `pandas`, `pyarrow`, `pytz`

## Project structure

```
scripts/
├── dnasc/
│   ├── config.py              # All pipeline constants (version, date filters, etc.)
│   ├── pipeline.py            # Orchestration — 12-step ETL pipeline
│   ├── extractors/            # BigQuery + LIMS data extraction
│   ├── transformers/          # Repair, join, and enrichment logic
│   └── renderer/              # HTML dashboard generation
├── incremental_refresh.py     # Hourly delta update (upsert over baseline)
├── full_refresh.py            # Full rebuild from scratch
├── dashboard_state/
│   ├── baseline.parquet       # Cached pipeline output
│   └── pipeline_version.txt  # Version tag written after each full build
└── logs/
    └── incremental.log        # Cron job output
```

## Usage

**Full rebuild:**
```bash
python full_refresh.py
```

**Incremental update** (runs automatically via cron, or manually):
```bash
python incremental_refresh.py
```

## Configuration

All tunable constants are in `dnasc/config.py`:

| Setting | Default | Description |
|---|---|---|
| `DATE_FILTER` | `2025-01-01` | Historical cutoff for full mode |
| `INCREMENTAL_HOURS` | `24` | Lookback window for incremental mode |
| `PIPELINE_VERSION` | `1.0.0` | Bump on every code change to trigger full rebuild |


# StackSats

![StackSats Logo](https://raw.githubusercontent.com/hypertrial/stacksats/main/docs/assets/stacking-sats-logo.svg)

[![PyPI version](https://img.shields.io/pypi/v/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Python versions](https://img.shields.io/pypi/pyversions/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Package Check](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml/badge.svg)](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml)
[![License: MIT](https://img.shields.io/github/license/hypertrial/stacksats)](LICENSE)

StackSats is a **strategy-first backtesting and execution framework** for Bitcoin weight management. It separates **Strategy** intent from validated weight time-series outcomes, providing a strict boundary for causal validation.

Learn more at [www.stackingsats.org](https://www.stackingsats.org).

## Start Here

Start with the hosted docs: <https://hypertrial.github.io/stacksats/>.

User docs:

- [`docs/index.md`](docs/index.md) for the full map
- [`docs/start/quickstart.md`](docs/start/quickstart.md) for the packaged five-minute demo
- [`docs/tasks.md`](docs/tasks.md) for task-first workflows
- [`docs/commands.md`](docs/commands.md) for canonical CLI command reference
- [`docs/reference/public-api.md`](docs/reference/public-api.md) for the supported `1.x` library surface
- [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md) for a custom strategy walkthrough
- [`docs/migration.md`](docs/migration.md) for old-to-new breaking-change mappings

Maintainer docs:

- [`docs/release.md`](docs/release.md) for maintainers cutting releases

## Framework Principles

- The framework owns budget math, iteration, feasibility clipping, and lock semantics.
- Users own features, signals, hyperparameters, and daily intent.
- Strategy hooks support either day-level intent (`propose_weight(state)`) or batch intent (`build_target_profile(...)`).
- `BaseStrategy.spec()` is the canonical strategy contract for stable metadata, params, intent mode, and required columns.
- The same sealed allocation kernel runs in local, backtest, and production.

See [`docs/framework.md`](docs/framework.md) for the canonical contract.

## Installation

Choose your install mode:

| Use case | Install mode | Command |
|---|---|---|
| I want to use StackSats | package install | `pip install stacksats` |
| I am working from a checkout | editable install | `python -m pip install -c requirements/constraints-maintainer.txt -e ".[dev,all]"` |

```bash
pip install stacksats
```

Optional extras:

```bash
pip install "stacksats[viz]"
pip install "stacksats[network]"
pip install "stacksats[deploy]"
```

Use `viz` for animation and plotting, `network` for HTTP-backed BTC price helpers, and `deploy` for database/export integrations such as `stacksats-plot-weights`.
Helper console scripts such as `stacksats-plot-mvrv` and `stacksats-plot-weights` are convenience tools, not part of the frozen stable `1.x` CLI subset.

For local development from a checkout:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -c requirements/constraints-maintainer.txt -e ".[dev,all]"
pip install pre-commit
```

## Quick Start

Canonical first run:

```bash
stacksats demo backtest
```

Artifacts are written under:

```text
output/<strategy_id>/<version>/<run_id>/
```

Demo lifecycle commands:

```bash
stacksats demo validate
stacksats demo backtest
stacksats demo export
```

For full lifecycle commands (`validate`, `backtest`, `export`), see [`docs/commands.md`](docs/commands.md).
For task-first workflows, see [`docs/tasks.md`](docs/tasks.md).
For full BRK data setup, see [`docs/start/full-data-setup.md`](docs/start/full-data-setup.md).
For upgrades, see [`docs/migration.md`](docs/migration.md).
For stable imports and contract boundaries, see [`docs/reference/public-api.md`](docs/reference/public-api.md).
For a custom strategy template, see [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md).
`stacksats strategy validate` runs strict validation by default; use `--no-strict` only when you intentionally want the lighter path.

Create a high-definition strategy-vs-uniform animation from an existing backtest:

```bash
pip install "stacksats[viz]"
stacksats strategy animate \
  --backtest-json output/<strategy_id>/<version>/<run_id>/backtest_result.json \
  --output-dir output/<strategy_id>/<version>/<run_id> \
  --output-name strategy_vs_uniform_hd.gif
```

## Full BRK Data Setup

Use [docs/data-source.md](docs/data-source.md) as the canonical source for Drive linkage, manifest fields, checksum validation, and maintainer refresh workflow.

Fetch the canonical long-format source data and prepare the runtime parquet:

```bash
stacksats data fetch
stacksats data prepare
stacksats data doctor
```

Default locations:

- canonical source download: `~/.stacksats/data/brk/`
- prepared runtime parquet: `~/.stacksats/data/bitcoin_analytics.parquet`

Runtime resolution precedence:

- `STACKSATS_ANALYTICS_PARQUET`
- explicit `parquet_path`
- `~/.stacksats/data/bitcoin_analytics.parquet`
- `./bitcoin_analytics.parquet`

Current coverage and dataset scale are documented in the merged-metrics reference pages.

Detailed references:

- [docs/start/full-data-setup.md](docs/start/full-data-setup.md)
- [docs/reference/merged-metrics-data-guide.md](docs/reference/merged-metrics-data-guide.md)
- [docs/reference/merged-metrics-parquet-schema.md](docs/reference/merged-metrics-parquet-schema.md)
- [docs/reference/merged-metrics-taxonomy.md](docs/reference/merged-metrics-taxonomy.md)

Run idempotent daily execution (paper mode):

```bash
stacksats strategy run-daily \
  --strategy stacksats.strategies.examples:RunDailyPaperStrategy \
  --total-window-budget-usd 1000 \
  --mode paper
```

Daily state is persisted by default to `.stacksats/run_state.sqlite3`.

Export requires explicit date bounds:

```bash
stacksats strategy export \
  --strategy stacksats.strategies.examples:SimpleZScoreStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

Use date bounds that are covered by available BTC source data.

Exported `WeightTimeSeries` objects are read-only validated artifacts.
If you need to reload an export later, use `WeightTimeSeriesBatch.from_artifact_dir(...)` against the artifact directory.

## Public API

The stable `1.x` contract is intentionally narrow:

- top-level `stacksats` exports
- documented artifact payloads
- documented `stacksats demo`, `stacksats data`, and `stacksats strategy` CLI subset

Experimental/reference strategies live under `stacksats.strategies.experimental.*` and stay outside the stable `1.x` API boundary.
Helper scripts such as `stacksats-plot-mvrv` and `stacksats-plot-weights` are documented convenience entry points outside the frozen stable CLI contract.

`load_data()` uses strict source-only BRK validation (no synthetic gap-fill behavior) and supports an optional `end_date` bound.

For canonical `merged_metrics*.parquet` exploration, use `open_merged_metrics()` and `load_metric_catalog()`. See [docs/start/eda-quickstart.md](docs/start/eda-quickstart.md).
See [docs/reference/public-api.md](docs/reference/public-api.md) for the stable import surface and [docs/stability.md](docs/stability.md) for the support matrix, deprecation policy, and CLI/artifact boundary.

## Development

```bash
venv/bin/python -m pytest -q
venv/bin/python -m pytest -m "slow or integration or performance" -q
venv/bin/python -m pre_commit install -t pre-commit
venv/bin/python -m mkdocs build --strict
venv/bin/python -m ruff check .
venv/bin/python scripts/check_no_coinmetrics_refs.py
venv/bin/python scripts/generate_merged_metrics_taxonomy.py --check
bash scripts/check_docs_refs.sh
bash scripts/check_coverage.sh  # heavy; enforces 100% line + branch coverage on stacksats/
bash scripts/clean_local.sh
bash scripts/release_check.sh
```

`pre-commit` now includes a local CLI smoke lane covering `demo backtest`, `strategy export`, `strategy animate`, `data prepare`, `data doctor`, and the documented paper `run-daily` flow.

If the repo is moved or renamed locally, rerun `bash scripts/install_hooks.sh` to refresh git hook paths.

For command examples using the packaged strategy template, see `docs/commands.md`.

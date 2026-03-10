# StackSats

![StackSats Logo](https://raw.githubusercontent.com/hypertrial/stacksats/main/docs/assets/stacking-sats-logo.svg)

[![PyPI version](https://img.shields.io/pypi/v/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Python versions](https://img.shields.io/pypi/pyversions/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Package Check](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml/badge.svg)](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml)
[![License: MIT](https://img.shields.io/github/license/hypertrial/stacksats)](LICENSE)

StackSats, developed by [Hypertrial](https://www.hypertrial.ai), is a Python package for strategy-first Bitcoin dollar cost averaging (DCA) research and execution.

Learn more at [www.stackingsats.org](https://www.stackingsats.org).
Current release line: `0.7.1`.

## Start Here

Start with the hosted docs: <https://hypertrial.github.io/stacksats/>.

Local docs entry points:

- [`docs/index.md`](docs/index.md) for the full map
- [`docs/start/quickstart.md`](docs/start/quickstart.md) for five-minute setup
- [`docs/tasks.md`](docs/tasks.md) for task-first workflows
- [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md) for a custom strategy walkthrough
- [`docs/start/minimal-strategy-examples.md`](docs/start/minimal-strategy-examples.md) for copyable minimal strategy templates
- [`docs/commands.md`](docs/commands.md) for canonical CLI command reference
- [`docs/migration.md`](docs/migration.md) for old-to-new breaking-change mappings
- [`docs/faq.md`](docs/faq.md) for recurring docs and integration questions
- [`docs/framework.md`](docs/framework.md) for the framework contract

## Framework Principles

- The framework owns budget math, iteration, feasibility clipping, and lock semantics.
- Users own features, signals, hyperparameters, and daily intent.
- Strategy hooks support either day-level intent (`propose_weight(state)`) or batch intent (`build_target_profile(...)`).
- `BaseStrategy.spec()` is the canonical strategy contract for stable metadata, params, intent mode, and required columns.
- The same sealed allocation kernel runs in local, backtest, and production.

See [`docs/framework.md`](docs/framework.md) for the canonical contract.

## Installation

```bash
pip install stacksats
```

For local development:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pip install pre-commit
```

Optional deploy extras:

```bash
pip install "stacksats[deploy]"
```

## Quick Start

Run a packaged example strategy:

```bash
stacksats strategy backtest \
  --strategy stacksats.strategies.examples:SimpleZScoreStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

Artifacts are written under:

```text
output/<strategy_id>/<version>/<run_id>/
```

For full lifecycle commands (`validate`, `backtest`, `export`), see [`docs/commands.md`](docs/commands.md).
For task-first workflows, see [`docs/tasks.md`](docs/tasks.md).
For upgrades, see [`docs/migration.md`](docs/migration.md).
For a custom strategy template, see [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md).
`stacksats strategy validate` runs strict validation by default; use `--no-strict` only when you intentionally want the lighter path.

Create a high-definition strategy-vs-uniform animation from an existing backtest:

```bash
stacksats strategy animate \
  --backtest-json output/<strategy_id>/<version>/<run_id>/backtest_result.json \
  --output-dir output/<strategy_id>/<version>/<run_id> \
  --output-name strategy_vs_uniform_hd.gif
```

## Data Source (BRK DuckDB)

Use [docs/data-source.md](docs/data-source.md) as the canonical source for Drive linkage, manifest fields, checksum validation, and maintainer refresh workflow.

Quick command:

```bash
venv/bin/python scripts/fetch_brk_data.py --target-dir .
export STACKSATS_ANALYTICS_DUCKDB=$(pwd)/bitcoin_analytics.duckdb
```

DuckDB artifact size is `~10.13 GiB`; keep at least `~12 GiB` free disk.
If manifest file IDs are placeholders, fetch fails by design and you should place DuckDB locally, then export `STACKSATS_ANALYTICS_DUCKDB`.

DuckDB factor strategy workflow (shared-horizon research):

```bash
export STACKSATS_ANALYTICS_DUCKDB=./bitcoin_analytics.duckdb
venv/bin/python scripts/train_duckdb_factor_strategy.py --start-date 2018-01-01 --end-date 2025-05-31
venv/bin/python scripts/compare_duckdb_alpha.py --start-date 2018-01-01 --end-date 2025-05-31
```

If `STACKSATS_ANALYTICS_DUCKDB` is unset, runtime falls back to `./bitcoin_analytics.duckdb`.
StackSats runtime is BRK-only for strategy metrics sourcing.
For quick candidate-only loops, `scripts/compare_duckdb_alpha.py` supports `--skip-strict` plus baseline metric overrides.

Run idempotent daily execution (paper mode):

```bash
stacksats strategy run-daily \
  --strategy stacksats.strategies.examples:SimpleZScoreStrategy \
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

Exported `StrategyTimeSeries` objects are read-only validated artifacts.
If you need to reload an export later, use `StrategyTimeSeriesBatch.from_artifact_dir(...)` against the artifact directory.

## Public API

Top-level exports:

- `BaseStrategy`, `StrategyContext`, `DayState`, `TargetProfile`
- `BacktestConfig`, `ValidationConfig`, `ExportConfig`
- `RunDailyConfig`
- `StrategyMetadata`, `StrategySpec`, `StrategyContractWarning`
- `StrategyArtifactSet`
- `StrategyTimeSeries`, `StrategyTimeSeriesBatch`
- `BacktestResult`, `ValidationResult`, `DailyRunResult`
- `load_strategy()`, `load_data()`, `precompute_features()`
- `MVRVStrategy`

`load_data()` uses strict source-only BRK validation (no synthetic gap-fill behavior) and supports an optional `end_date` bound.

## Development

```bash
venv/bin/python -m pytest -q
venv/bin/python -m pytest -m "slow or integration or performance" -q
venv/bin/python -m pre_commit install -t pre-commit
venv/bin/python -m mkdocs build --strict
venv/bin/python -m ruff check .
venv/bin/python scripts/check_no_coinmetrics_refs.py
bash scripts/check_docs_refs.sh
bash scripts/check_coverage.sh  # heavy; mirrored by scheduled/manual coverage-report workflow
bash scripts/clean_local.sh
bash scripts/release_check.sh
venv/bin/python scripts/train_duckdb_factor_strategy.py --help
venv/bin/python scripts/compare_duckdb_alpha.py --help
```

If the repo is moved or renamed locally, rerun `bash scripts/install_hooks.sh` to refresh git hook paths.

For command examples using the packaged strategy template, see `docs/commands.md`.

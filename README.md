# StackSats

![StackSats Logo](https://raw.githubusercontent.com/hypertrial/stacksats/main/docs/assets/stacking-sats-logo.svg)

[![PyPI version](https://img.shields.io/pypi/v/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Python versions](https://img.shields.io/pypi/pyversions/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Package Check](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml/badge.svg)](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml)
[![License: MIT](https://img.shields.io/github/license/hypertrial/stacksats)](LICENSE)

StackSats, developed by [Hypertrial](https://www.hypertrial.ai), is a Python package for strategy-first Bitcoin dollar cost averaging (DCA) research and execution.

Learn more at [www.stackingsats.org](https://www.stackingsats.org).

## Start Here

Start with the hosted docs: <https://hypertrial.github.io/stacksats/>.

Local docs entry points:

- [`docs/index.md`](docs/index.md) for the full map
- [`docs/start/quickstart.md`](docs/start/quickstart.md) for five-minute setup
- [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md) for a custom strategy walkthrough
- [`docs/commands.md`](docs/commands.md) for canonical CLI command reference
- [`docs/framework.md`](docs/framework.md) for the framework contract

## Framework Principles

- The framework owns budget math, iteration, feasibility clipping, and lock semantics.
- Users own features, signals, hyperparameters, and daily intent.
- Strategy hooks support either day-level intent (`propose_weight(state)`) or batch intent (`build_target_profile(...)`).
- The same sealed allocation kernel runs in local, backtest, and production.

See [`docs/framework.md`](docs/framework.md) for the canonical contract.

## Installation

```bash
pip install stacksats
```

For local development:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Optional deploy extras:

```bash
pip install "stacksats[deploy]"
```

## Quick Start

Run the packaged example strategy:

```bash
python -m stacksats.strategies.model_example
```

Artifacts are written under:

```text
output/<strategy_id>/<version>/<run_id>/
```

For full lifecycle commands (`validate`, `backtest`, `export`), see [`docs/commands.md`](docs/commands.md).
For a custom strategy template, see [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md).

## Public API

Top-level exports:

- `BaseStrategy`, `StrategyContext`, `DayState`, `TargetProfile`
- `BacktestConfig`, `ValidationConfig`, `ExportConfig`
- `StrategyArtifactSet`
- `StrategyTimeSeries`, `StrategyTimeSeriesBatch`
- `BacktestResult`, `ValidationResult`
- `load_strategy()`, `load_data()`, `precompute_features()`
- `MVRVStrategy`

## Development

```bash
pytest tests/ -v
ruff check .
bash scripts/check_docs_refs.sh
```

For command examples using the packaged strategy template, see `docs/commands.md`.

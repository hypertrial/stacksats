# StackSats

*Python library for quantitative Bitcoin accumulation via DCA strategies.*

Not a general crypto trading bot or brokerage wrapper—see [What StackSats is](#what-stacksats-is).

![StackSats Logo](https://raw.githubusercontent.com/hypertrial/stacksats/main/docs/assets/stacking-sats-logo.svg)

[![PyPI version](https://img.shields.io/pypi/v/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Python versions](https://img.shields.io/pypi/pyversions/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Package Check](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml/badge.svg)](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml)
[![License: MIT](https://img.shields.io/github/license/hypertrial/stacksats)](LICENSE)

StackSats is a Python library for building, backtesting, and operationalizing quantitative dollar-cost averaging (DCA) strategies for Bitcoin accumulation. It helps you turn research signals and feature pipelines into validated BTC weight schedules and daily allocation decisions.

Use StackSats when you want to:

- build custom Bitcoin accumulation strategies in Python
- backtest DCA rules against BRK-derived canonical Bitcoin datasets
- validate causal constraints before shipping a strategy
- emit daily BTC allocation decisions for agents or external execution systems

StackSats is library-first: the CLI, demo flows, and hosted agent API all sit on top of the same Python package surface.

Learn more at [www.stackingsats.org](https://www.stackingsats.org).

**Hosted documentation:** <https://hypertrial.github.io/stacksats/> — start from [`docs/index.md`](docs/index.md).

## What StackSats is

StackSats is not a general crypto trading bot or brokerage wrapper. It is a research and decision engine for Bitcoin accumulation strategies:

- **Python library:** define strategies with `BaseStrategy`, run them with stable configs, and consume results from Python.
- **Quantitative DCA toolkit:** model how much BTC to accumulate over time instead of placing exchange-specific orders directly.
- **Backtesting framework:** compare strategies, validate constraints, and export artifact sets from repeatable runs.
- **Execution boundary:** StackSats computes decisions; brokerage execution stays outside the package unless you wire in an adapter intentionally.

## Documentation map

- [`docs/start/quickstart.md`](docs/start/quickstart.md) — first install, demo run, and Python entry points
- [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md) — write your first custom Bitcoin DCA strategy
- [`docs/start/minimal-strategy-examples.md`](docs/start/minimal-strategy-examples.md) — copyable strategy templates
- [`docs/reference/public-api.md`](docs/reference/public-api.md) — stable `1.x` Python library surface
- [`docs/start/system-overview.md`](docs/start/system-overview.md) — data flow and production paths
- [`docs/tasks.md`](docs/tasks.md) — task-first workflows
- [`docs/commands.md`](docs/commands.md) — CLI index
- [`docs/run/compare.md`](docs/run/compare.md) — compare multiple strategies on one window (`stacksats strategy compare`, `comparison_result.json`)
- [`docs/data-source.md`](docs/data-source.md) — Bitcoin Research Kit (BRK) dataset support, canonical source data, and manifests
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — symptom-based links
- [`docs/migration.md`](docs/migration.md) — breaking-change mappings
- [`docs/release.md`](docs/release.md) — maintainer releases

## BRK compatibility

StackSats is a Python library with explicit compatibility for the Bitcoin Research Kit (BRK) project and BRK-derived canonical data workflows. We document BRK as the upstream project and link to the official BRK surfaces: [bitcoinresearchkit/brk](https://github.com/bitcoinresearchkit/brk), [`brk` on crates.io](https://crates.io/crates/brk), and [`brk` on docs.rs](https://docs.rs/crate/brk/latest).

This is a project and data compatibility statement, not a promise that StackSats embeds BRK, re-exports Rust crates, or version-locks BRK crate APIs. StackSats remains a Python package with its own stable `1.x` support boundary.

## Framework principles

The framework owns budget math, iteration, feasibility clipping, and lock semantics; you own features, signals, hyperparameters, and daily intent (`propose_weight` or `build_target_profile`). The same sealed allocation kernel runs in local runs, backtests, and production. See [`docs/framework.md`](docs/framework.md).

## Primary production flow

1. StackSats computes a validated BTC accumulation decision.
2. An external agent or automation reads the decision payload.
3. Brokerage execution stays outside StackSats.

Use `stacksats strategy decide-daily` (or `strategy.decide_daily(...)`) for the agent-facing interface; [`docs/run/decide-daily.md`](docs/run/decide-daily.md) covers payloads and sensitivity. Use `stacksats serve agent-api` for a hosted `/v1` HTTP service ([`docs/run/agent-api.md`](docs/run/agent-api.md), including token policy). Use `stacksats strategy run-daily` when StackSats should submit through a configured adapter ([`docs/run/run-daily.md`](docs/run/run-daily.md)).

**Security:** follow [`SECURITY.md`](SECURITY.md) for reporting; treat decision and API tokens as secrets.

## Installation

| Use case | Command |
| --- | --- |
| Use StackSats from PyPI | `pip install stacksats` |
| Editable install from a checkout | `python -m pip install -c requirements/constraints-maintainer.txt -e ".[dev,all]"` |

Optional extras: `pip install "stacksats[viz]"` (animation/plots), `[network]` (HTTP BTC price helpers), `[service]` (agent API), `[deploy]` (Postgres/export helpers). The `stacksats-plot-weights` helper needs both `[viz]` and `[deploy]` plus a configured `DATABASE_URL` (it reads stored weight windows, then renders). Helper scripts are documented convenience tools, not part of the frozen stable `1.x` CLI subset.

**Development venv** (from repo root):

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -c requirements/constraints-maintainer.txt -e ".[dev,all]"
pip install pre-commit
venv/bin/python -m pre_commit install -t pre-commit
```

## Quick start

Install, import the stable `1.x` surface, then run the packaged demo:

```bash
pip install stacksats
```

```python
from stacksats import BaseStrategy, StrategyRunner, list_strategies
```

```bash
stacksats demo backtest
```

Artifacts: `output/<strategy_id>/<version>/<run_id>/`

If you want to build strategies in Python next, start with [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md) and [`docs/start/minimal-strategy-examples.md`](docs/start/minimal-strategy-examples.md). For the full CLI, use [`docs/commands.md`](docs/commands.md). For BRK data setup (`stacksats data fetch|prepare|doctor`), use [`docs/start/full-data-setup.md`](docs/start/full-data-setup.md). For the support boundary, use [`docs/stability.md`](docs/stability.md). `stacksats strategy validate` is strict by default; use `--no-strict` only when you intend the lighter path.

## Public API

The stable `1.x` contract covers top-level exports, documented artifacts, and the documented CLI/agent API subset. See [`docs/reference/public-api.md`](docs/reference/public-api.md) and [`docs/stability.md`](docs/stability.md). `load_data()` uses strict BRK validation; for long-format merged metrics exploration, see [`docs/start/eda-quickstart.md`](docs/start/eda-quickstart.md).

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full local quality matrix (tests, docs gates, coverage, release checks). Typical fast loop:

```bash
venv/bin/python -m pytest -q
venv/bin/python -m ruff check .
bash scripts/check_docs_refs.sh
venv/bin/python scripts/check_docs_ux.py
venv/bin/python -m mkdocs build --strict
```

If the repo path changes locally, rerun `bash scripts/install_hooks.sh` to refresh git hook paths.

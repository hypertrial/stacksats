# StackSats

![StackSats Logo](https://raw.githubusercontent.com/hypertrial/stacksats/main/docs/assets/stacking-sats-logo.svg)

[![PyPI version](https://img.shields.io/pypi/v/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Python versions](https://img.shields.io/pypi/pyversions/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Package Check](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml/badge.svg)](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml)
[![License: MIT](https://img.shields.io/github/license/hypertrial/stacksats)](LICENSE)

StackSats is a **strategy-first backtesting and decision framework** for Bitcoin weight management. It separates **Strategy** intent from validated weight time-series outcomes, providing a strict boundary for causal validation.

Learn more at [www.stackingsats.org](https://www.stackingsats.org).

**Hosted documentation:** <https://hypertrial.github.io/stacksats/> — start from [`docs/index.md`](docs/index.md).

## Documentation map

- [`docs/start/quickstart.md`](docs/start/quickstart.md) — packaged demo
- [`docs/start/system-overview.md`](docs/start/system-overview.md) — data flow and production paths
- [`docs/tasks.md`](docs/tasks.md) — task-first workflows
- [`docs/commands.md`](docs/commands.md) — CLI index
- [`docs/data-source.md`](docs/data-source.md) — Bitcoin Research Kit (BRK) dataset support, canonical source data, and manifests
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — symptom-based links
- [`docs/reference/public-api.md`](docs/reference/public-api.md) — stable `1.x` library surface
- [`docs/start/first-strategy-run.md`](docs/start/first-strategy-run.md) — custom strategy walkthrough
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

```bash
stacksats demo backtest
```

Artifacts: `output/<strategy_id>/<version>/<run_id>/`

Next: [`docs/commands.md`](docs/commands.md) for the full CLI, [`docs/start/full-data-setup.md`](docs/start/full-data-setup.md) for BRK data (`stacksats data fetch|prepare|doctor`), [`docs/start/eda-quickstart.md`](docs/start/eda-quickstart.md) for EDA, [`docs/stability.md`](docs/stability.md) for the support boundary. `stacksats strategy validate` is strict by default; use `--no-strict` only when you intend the lighter path.

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

---
title: CLI Commands
description: Command reference for validating, backtesting, and exporting StackSats strategies.
---

# Commands for `examples/model_example.py`

This file explains how to run checks, backtests, and exports using the standalone strategy file:

- `examples/model_example.py`
- strategy class: `ExampleMVRVStrategy`

Strategy implementations can use either:
- `propose_weight(state)` for per-day intent, or
- `build_target_profile(ctx, features_df, signals)` for batch intent.

## Strategy Lifecycle Flow

```mermaid
flowchart LR
    A["strategy validate"] --> B["strategy backtest"]
    B --> C["strategy export"]
    C --> D["output/<strategy_id>/<version>/<run_id>/"]
```

## Prerequisites

From the repo root:

```bash
pip install -e .
```

Optional dependencies:

```bash
# For local development tools
pip install -r requirements-dev.txt

# For export and database tooling
pip install -e ".[deploy]"
```

## Strategy Spec Format

CLI commands that load a strategy use:

```text
module_or_path:ClassName
```

For this example file:

```text
examples/model_example.py:ExampleMVRVStrategy
```

## 1) Quick Run (inside `examples/model_example.py`)

Run the file directly:

```bash
python examples/model_example.py
```

With custom options:

```bash
python examples/model_example.py \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --output-dir output \
  --strategy-label example-mvrv-strategy
```

What this does:
- Runs `strategy.validate(...)`
- Runs `strategy.backtest(...)`
- Writes plots + JSON output to `output/`

## 2) Validate Strategy via Strategy Lifecycle CLI

Check whether the model passes package validation gates:

```bash
stacksats strategy validate --strategy examples/model_example.py:ExampleMVRVStrategy
```

Common options:

```bash
stacksats strategy validate \
  --strategy examples/model_example.py:ExampleMVRVStrategy \
  --strategy-config strategy_config.json \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --strict \
  --min-win-rate 50.0
```

`--strict` enables additional robustness gates (determinism, mutation, leakage, OOS fold checks, and shuffled baseline checks).

## 3) Run Full Backtest via Strategy Lifecycle CLI

Basic:

```bash
stacksats strategy backtest --strategy examples/model_example.py:ExampleMVRVStrategy
```

With options:

```bash
stacksats strategy backtest \
  --strategy examples/model_example.py:ExampleMVRVStrategy \
  --strategy-config strategy_config.json \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --output-dir output \
  --strategy-label model-example
```

## 4) Export Strategy Artifacts

```bash
stacksats strategy export \
  --strategy examples/model_example.py:ExampleMVRVStrategy \
  --strategy-config strategy_config.json \
  --start-date 2025-12-01 \
  --end-date 2027-12-31 \
  --output-dir output
```

Artifacts are written under:

```text
output/<strategy_id>/<version>/<run_id>/
```

This includes:
- `weights.csv`
- `timeseries_schema.md`
- `artifacts.json` (`strategy_id`, `version`, `config_hash`, `run_id`, file map)
- canonical `weights.csv` columns: `start_date`, `end_date`, `day_index`, `date`, `price_usd`, `weight`

Notes:
- `stacksats strategy export` is strategy artifact export (filesystem output).

## 5) Useful Development Commands

Verify this document's example commands end-to-end:

```bash
# Requires ./venv (for example: python -m venv venv && source venv/bin/activate && pip install -e .)
python scripts/test_example_commands.py
```

Run tests:

```bash
pytest tests/ -v
```

Run lint:

```bash
ruff check .
```

## Troubleshooting

- **`Invalid strategy spec`**
  Ensure format is exactly `module_or_path:ClassName`.

- **`Class 'ExampleMVRVStrategy' not found`**
  Check class name spelling and file path.

- **`Strategy file not found`**
  Run from repo root or pass an absolute file path.

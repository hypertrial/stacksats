---
title: Quickstart
description: Five-minute setup and first execution path for StackSats.
---

# Quickstart

Use this page for a 5-minute first run.

!!! tip "Recommended Path"
    Start with editable install so local examples and docs stay in sync with your checkout.

## 1) Install

=== "Editable (recommended)"

    ```bash
    python -m venv venv
    source venv/bin/activate
    venv/bin/python -m pip install --upgrade pip
    venv/bin/python -m pip install -e ".[dev]"
    ```

=== "Package only"

    ```bash
    pip install stacksats
    ```

## 2) Make canonical data available

Canonical dataset schema and projection workflow:

- [BRK Data Source](../data-source.md)
- [Merged Metrics Parquet Schema](../reference/merged-metrics-parquet-schema.md)

If you already have a runtime-compatible BRK parquet:

```bash
export STACKSATS_ANALYTICS_PARQUET=$(pwd)/bitcoin_analytics.parquet
```

If you start from canonical `merged_metrics*.parquet`, use the projection step in [BRK Data Source](../data-source.md) to derive `bitcoin_analytics.parquet` first, then export `STACKSATS_ANALYTICS_PARQUET`.

## 3) Run an example strategy

```bash
stacksats strategy backtest \
  --strategy stacksats.strategies.examples:SimpleZScoreStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

Built-in strategy catalog (intent mode, required columns, and tuning parameters):
[Strategies](../reference/strategies.md).

This runs a packaged example through the canonical lifecycle and writes artifacts under:

```text
output/<strategy_id>/<version>/<run_id>/
```

## 4) Use the Strategy Lifecycle CLI

Run lifecycle commands from the canonical reference:

- [Validate Strategy](../run/validate.md)
- [Run Full Backtest](../run/backtest.md)
- [Export Strategy Artifacts](../run/export.md)
- [Animate Backtest Output](../run/animate.md)

## 5) Inspect outputs

Primary run artifacts are written under:

```text
output/<strategy_id>/<version>/<run_id>/
```

Typical files:

- `backtest_result.json`
- `metrics.json`
- plot `.svg` files

## 6) What to read next

- [Task Hub](../tasks.md)
- [Notebook Demo](notebook-demo.md)
- [First Strategy Run](first-strategy-run.md)
- [Minimal Strategy Examples](minimal-strategy-examples.md)
- [Strategies](../reference/strategies.md)
- [Framework Boundary](../framework.md)
- [CLI Commands](../commands.md)
- [Migration Guide](../migration.md)
- [FAQ](../faq.md)

## Success Criteria

A successful quickstart run should produce all of the following:

- CLI command exits without traceback.
- Backtest summary is printed.
- Backtest artifacts are written under one run directory.

## Troubleshooting

- If command import fails, confirm editable install from repo root.
- If dates or outputs look wrong, run explicit lifecycle commands from [CLI Commands](../commands.md).
- If upgrading and old helper names fail, use [Migration Guide](../migration.md).
- If you need minimal copy-paste templates, use [Minimal Strategy Examples](minimal-strategy-examples.md).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Quickstart)

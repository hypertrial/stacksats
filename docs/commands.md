---
title: Command Index
description: Canonical index for StackSats strategy lifecycle commands.
---

# Command Index

Use this page to find the canonical command reference for each lifecycle action.
For built-in strategy behavior and parameter defaults, see [Strategies](reference/strategies.md).

## Most Common Commands (copy/paste)

```bash
stacksats demo backtest
stacksats data fetch
stacksats data prepare
stacksats data doctor
stacksats strategy validate --strategy stacksats.strategies.examples:SimpleZScoreStrategy
stacksats strategy backtest --strategy stacksats.strategies.examples:SimpleZScoreStrategy --start-date 2024-01-01 --end-date 2024-12-31 --output-dir output
stacksats strategy export --strategy stacksats.strategies.examples:SimpleZScoreStrategy --start-date 2024-01-01 --end-date 2024-12-31 --output-dir output
stacksats strategy run-daily --strategy stacksats.strategies.examples:SimpleZScoreStrategy --total-window-budget-usd 1000 --mode paper
stacksats strategy animate --backtest-json output/<strategy_id>/<version>/<run_id>/backtest_result.json
```

## Command Pages

- [Validate Command](run/validate.md)
- [Backtest Command](run/backtest.md)
- [Export Command](run/export.md)
- [Demo Command](run/demo.md)
- [Data Command](run/data.md)
- [Run Daily Command](run/run-daily.md)
- [Animate Command](run/animate.md)

## Prerequisites

- The fastest first run is `stacksats demo backtest`.
- Canonical dataset is `merged_metrics*.parquet`; see [Merged Metrics Parquet Schema](reference/merged-metrics-parquet-schema.md).
- Runtime commands resolve a BRK-wide parquet via `STACKSATS_ANALYTICS_PARQUET`, managed default `~/.stacksats/data/bitcoin_analytics.parquet`, or legacy local fallback `./bitcoin_analytics.parquet`.
- Use editable install for local command consistency:

```bash
python -m venv venv
source venv/bin/activate
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -e ".[dev]"
```

## Compatibility Anchors

These sections intentionally preserve older in-page anchor links.

## 2) Validate Strategy via Strategy Lifecycle CLI

Canonical page: [Validate Command](run/validate.md)

## 3) Run Full Backtest via Strategy Lifecycle CLI

Canonical page: [Backtest Command](run/backtest.md)

## 4) Export Strategy Artifacts

Canonical page: [Export Command](run/export.md)

## 5) Run Idempotent Daily Execution

Canonical page: [Run Daily Command](run/run-daily.md)

## 6) Animate Backtest Output (HD GIF)

Canonical page: [Animate Command](run/animate.md)

## Troubleshooting

- If a command errors with import issues, verify editable install from repo root.
- If a command errors with data coverage, verify runtime parquet path/date bounds and confirm it was derived from canonical `merged_metrics`.
- For strict validation failures, use [Validation Checklist](validation_checklist.md).
- For task-first troubleshooting, use [Task Hub](tasks.md#i-want-to-troubleshoot-command-failures-quickly).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Command+Index)

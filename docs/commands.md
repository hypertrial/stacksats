---
title: Command Index
description: Canonical index for StackSats strategy lifecycle commands.
---

# Command Index

Use this page to find the canonical command reference for each lifecycle action.

## Most Common Commands (copy/paste)

```bash
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
- [Run Daily Command](run/run-daily.md)
- [Animate Command](run/animate.md)

## Prerequisites

- Use BRK-only runtime source (`STACKSATS_ANALYTICS_PARQUET` or `./bitcoin_analytics.parquet` fallback).
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
- If a command errors with data coverage/staleness, verify BRK parquet path and date bounds.
- For strict validation failures, use [Validation Checklist](validation_checklist.md).
- For task-first troubleshooting, use [Task Hub](tasks.md#i-want-to-troubleshoot-command-failures-quickly).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Command+Index)

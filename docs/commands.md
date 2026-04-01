---
title: Command Index
description: Canonical index for StackSats strategy lifecycle commands.
---

# Command Index

Use this page when you need the canonical CLI reference for the stable `stacksats` command groups.
For first-run onboarding, use [Quickstart](start/quickstart.md). For task-first routing, use [Task Hub](tasks.md).
For built-in strategy behavior and parameter defaults, see [Strategies](reference/strategies.md).

## Most Common Commands (copy/paste)

```bash
stacksats demo backtest
stacksats data fetch
stacksats data prepare
stacksats data doctor
stacksats strategy validate --strategy simple-zscore
stacksats strategy backtest --strategy simple-zscore --start-date 2024-01-01 --end-date 2024-12-31 --output-dir output
stacksats strategy export --strategy simple-zscore --start-date 2024-01-01 --end-date 2024-12-31 --output-dir output
stacksats strategy decide-daily --strategy run-daily-paper --total-window-budget-usd 1000
stacksats serve agent-api --registry-path .stacksats/agent_service_registry.json
stacksats strategy run-daily --strategy run-daily-paper --total-window-budget-usd 1000 --mode paper
stacksats strategy animate --backtest-json output/<strategy_id>/<version>/<run_id>/backtest_result.json
```

## Command Pages

- [Validate Command](run/validate.md)
- [Backtest Command](run/backtest.md)
- [Export Command](run/export.md)
- [Decide Daily Command](run/decide-daily.md)
- [Agent API Service](run/agent-api.md)
- [Demo Command](run/demo.md)
- [Data Command](run/data.md)
- [Run Daily Command](run/run-daily.md)
- [Animate Command](run/animate.md)

## Prerequisites

- If you have not run StackSats before, start with `stacksats demo backtest`.
- The stable CLI subset covers `stacksats demo`, `stacksats data`, `stacksats strategy`, and `stacksats serve`.
- Runtime commands resolve a BRK-wide parquet via `STACKSATS_ANALYTICS_PARQUET`, managed default `~/.stacksats/data/bitcoin_analytics.parquet`, or legacy local fallback `./bitcoin_analytics.parquet`.
- Visual commands (`stacksats strategy animate`, `stacksats-plot-mvrv`, `stacksats-plot-weights`) require `stacksats[viz]`.
- Hosted agent API commands (`stacksats serve agent-api`) require `stacksats[service]`.
- `stacksats-plot-weights` also needs `stacksats[deploy]` plus `DATABASE_URL` because it reads stored weight windows from Postgres before rendering.
- Helper scripts such as `stacksats-plot-mvrv` and `stacksats-plot-weights` are documented tools outside the frozen stable CLI contract.
- See [Stability Policy](stability.md) for the exact boundary and [Quickstart](start/quickstart.md) for install paths.

## Legacy Anchors

These sections are retained only to preserve older in-page links.

## 2) Validate Strategy via Strategy Lifecycle CLI

Canonical page: [Validate Command](run/validate.md)

## 3) Run Full Backtest via Strategy Lifecycle CLI

Canonical page: [Backtest Command](run/backtest.md)

## 4) Export Strategy Artifacts

Canonical page: [Export Command](run/export.md)

## 5) Run Idempotent Daily Execution

Canonical page: [Run Daily Command](run/run-daily.md)

## 6) Generate Agent-Facing Daily Decisions

Canonical page: [Decide Daily Command](run/decide-daily.md)

## 7) Host the Agent API Service

Canonical page: [Agent API Service](run/agent-api.md)

## 8) Animate Backtest Output (HD GIF)

Canonical page: [Animate Command](run/animate.md)

## Troubleshooting

- Start with the [Troubleshooting](troubleshooting.md) hub for symptom-based links.
- If a command errors with import issues, verify StackSats is installed; if you are using a checkout, confirm editable install from repo root.
- If a command errors with data coverage, verify runtime parquet path/date bounds and confirm it was derived from canonical `merged_metrics`.
- For strict validation failures, use [Validation Checklist](validation_checklist.md).
- For task-first troubleshooting, use [Task Hub](tasks.md#i-want-to-troubleshoot-command-failures-quickly).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Command+Index)

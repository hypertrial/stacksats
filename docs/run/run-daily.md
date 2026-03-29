---
title: Run Daily Command
description: Reference for `stacksats strategy run-daily`.
---

# Run Daily Command

Use this command when you intentionally want StackSats to submit an order through an execution adapter after generating the same validated daily decision payload that powers the agent-native flow.
If you only need a machine-consumable decision for an external AI agent, use [Decide Daily Command](decide-daily.md) instead.

## Prerequisites

- Strategy is validated and ready for execution.
- You know total window budget in USD.
- For live mode, provide an adapter spec.

## Command

```bash
stacksats strategy run-daily \
  --strategy stacksats.strategies.examples:RunDailyPaperStrategy \
  --total-window-budget-usd 1000 \
  --mode paper
```

Built-in strategy catalog and expected behavior: [Strategies](../reference/strategies.md).

## Expected output

- Structured JSON payload plus status line:
  - `Status: EXECUTED`
  - `Status: NO-OP (idempotent)`
  - `Status: FAILED`
- State persisted at `.stacksats/run_state.sqlite3` by default.

## Key options

- `--strategy-config <path>`: strategy params JSON.
- `--run-date YYYY-MM-DD`: execution date override (defaults to the current UTC date).
- `--state-db-path <path>`: idempotency/state DB (default `.stacksats/run_state.sqlite3`).
- `--output-dir <dir>`: artifact root for executed runs (default `output`).
- `--adapter module_or_path:ClassName`: required for `--mode live`.
- `--btc-price-col <name>`: BTC price column used for the run-date order calculation (default `price_usd`).
- `--force`: rerun when stored state detects changed parameters and would otherwise raise an idempotency conflict.

## Troubleshooting

- If live mode fails without adapter, pass `--adapter module_or_path:ClassName`.
- If a rerun fails after inputs changed, rerun with `--force`; unchanged reruns still return `NO-OP (idempotent)`.
- If price coverage or allocation-window checks fail, verify runtime BRK parquet path, run-date coverage, and the selected `--btc-price-col` (and confirm the parquet was derived from canonical `merged_metrics`).

## Next step

- Inspect the structured JSON result and the persisted state DB for the executed run. `reconcile-daily` remains an internal maintenance command and is not part of the stable CLI contract.
- Prefer [Decide Daily Command](decide-daily.md) when an external AI agent or brokerage workflow will execute the decision.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Run+Daily+Command)

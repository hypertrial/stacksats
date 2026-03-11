---
title: Run Daily Command
description: Reference for `stacksats strategy run-daily`.
---

# Run Daily Command

## Prerequisites

- Strategy is validated and ready for execution.
- You know total window budget in USD.
- For live mode, provide an adapter spec.

## Command

```bash
stacksats strategy run-daily \
  --strategy stacksats.strategies.examples:SimpleZScoreStrategy \
  --total-window-budget-usd 1000 \
  --mode paper
```

## Expected output

- Structured JSON payload plus status line:
  - `Status: EXECUTED`
  - `Status: NO-OP (idempotent)`
  - `Status: FAILED`
- State persisted at `.stacksats/run_state.sqlite3` by default.

## Key options

- `--run-date YYYY-MM-DD`: execution date override.
- `--state-db-path <path>`: idempotency/state DB.
- `--adapter module_or_path:ClassName`: required for `--mode live`.
- `--force`: bypass idempotency no-op guard.

## Troubleshooting

- If live mode fails without adapter, pass `--adapter`.
- If price coverage fails, verify BRK source freshness and run date.

## Next step

- Reconcile previous run state: `stacksats strategy reconcile-daily ...`.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Run+Daily+Command)

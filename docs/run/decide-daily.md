---
title: Decide Daily Command
description: Reference for `stacksats strategy decide-daily`.
---

# Decide Daily Command

Use this command when StackSats should emit a validated, execution-ready decision payload for an external AI agent or brokerage automation layer.

## Security and sensitivity

Decision payloads can include execution-relevant fields (for example sizing, hashes, and artifact paths). Treat stdout, logs, and stored JSON like operational data. Report security issues through the [GitHub security policy](https://github.com/hypertrial/stacksats/security/policy) (see the repository `SECURITY.md`).

## Prerequisites

- Strategy is validated and ready for decision generation.
- You know total window budget in USD.
- An external system will handle brokerage-specific execution.

## Command

```bash
stacksats strategy decide-daily \
  --strategy stacksats.strategies.examples:RunDailyPaperStrategy \
  --total-window-budget-usd 1000
```

## Python API

```python
from stacksats import DecideDailyConfig, RunDailyPaperStrategy

result = RunDailyPaperStrategy().decide_daily(
    DecideDailyConfig(total_window_budget_usd=1000.0)
)
print(result.to_json())
```

## Example payload

```json
{
  "schema_version": "1.0.0",
  "status": "decided",
  "strategy_id": "run-daily-paper",
  "strategy_version": "1.0.0",
  "run_date": "2024-12-31",
  "decision_key": "9ee2d0f0-8a15-46d8-9730-3f6f13d3150d",
  "weight_today": 0.0027397260273972603,
  "recommended_notional_usd": 2.73972602739726,
  "recommended_quantity_btc": 0.000045662100456621,
  "reference_price_usd": 60000.0,
  "btc_price_col": "price_usd",
  "validation_passed": true,
  "validation_receipt_id": 17,
  "data_hash": "…",
  "feature_snapshot_hash": "…",
  "artifact_path": "output/run-daily-paper/1.0.0/decisions/2024-12-31/<decision_key>/decision_result.json",
  "message": "Daily decision completed."
}
```

## Expected output

- Structured JSON decision payload plus status line:
  - `Status: DECIDED`
  - `Status: NO-OP (idempotent)`
  - `Status: FAILED`
- State persisted at `.stacksats/run_state.sqlite3` by default.

## Key options

- `--strategy-config <path>`: strategy params JSON.
- `--run-date YYYY-MM-DD`: decision date override (defaults to the current UTC date).
- `--state-db-path <path>`: idempotency/state DB (default `.stacksats/run_state.sqlite3`).
- `--output-dir <dir>`: artifact root for decision artifacts (default `output`).
- `--btc-price-col <name>`: BTC price column used for the run-date sizing calculation (default `price_usd`).
- `--force`: rerun when stored state detects changed parameters and would otherwise raise an idempotency conflict.

## Troubleshooting

- If a rerun fails after inputs changed, rerun with `--force`; unchanged reruns still return `NO-OP (idempotent)`.
- If price coverage or allocation-window checks fail, verify runtime BRK parquet path, run-date coverage, and the selected `--btc-price-col`.
- If strict validation fails, inspect `validation_receipt_id`, `data_hash`, and `feature_snapshot_hash` in the payload before handing it to your external agent.

## Next step

- Hand the payload to your external AI agent or broker-specific execution layer.
- Use [Run Daily Command](run-daily.md) only when you intentionally want StackSats to submit through an execution adapter.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Decide+Daily+Command)

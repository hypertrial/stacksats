---
title: Task Hub
description: Task-first workflows for common StackSats goals.
---

# I Want To...

Use this page to jump directly to the workflow you need.

Global source contract:
- Strategy runtime and validation are BRK-only in `0.7.0`.
- Set `STACKSATS_ANALYTICS_DUCKDB` or place `bitcoin_analytics.duckdb` at repo root.

## I want to validate a strategy

### Prerequisites

- Strategy file exists and can be loaded (`module_or_path:ClassName`).
- Local install is complete (`pip install -e ".[dev]"`).

### Command

```bash
stacksats strategy validate \
  --strategy my_strategy.py:MyStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### Expected output

- Terminal summary includes `Validation PASSED` or `Validation FAILED`.
- Win-rate and leakage/constraint checks are reported.
- Strict validation runs by default; use `--no-strict` only when you intentionally want the lighter path.

### Troubleshooting

- Check strategy spec format: `module_or_path:ClassName`.
- Check `NaN`/`inf` in custom features or signals.
- Use [Validation Checklist](validation_checklist.md) for strict-gate failures.

### Next step

- Run a full backtest: [CLI backtest command](commands.md#3-run-full-backtest-via-strategy-lifecycle-cli).

## I want to run a full backtest

### Prerequisites

- Strategy validates with acceptable quality.
- Start/end date range has available BTC data.

### Command

```bash
stacksats strategy backtest \
  --strategy my_strategy.py:MyStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

### Expected output

- Artifacts under `output/<strategy_id>/<version>/<run_id>/`.
- Files include `backtest_result.json`, `metrics.json`, and plot `.svg` outputs.

### Troubleshooting

- Ensure date bounds are valid and ordered.
- If outputs look unstable, re-run with strict validation and compare.

### Next step

- Interpret metrics consistently: [Interpret Backtest Metrics](recipes/interpret-backtest.md).

## I want to export strategy weights

### Prerequisites

- Strategy is loadable.
- Date bounds are explicit.

### Command

```bash
stacksats strategy export \
  --strategy my_strategy.py:MyStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

### Expected output

- Artifacts under `output/<strategy_id>/<version>/<run_id>/`.
- Files include `weights.csv`, `timeseries_schema.md`, and `artifacts.json`.

### Troubleshooting

- Export requires both `--start-date` and `--end-date`.
- Keep `--end-date` within available source data coverage.
- Validate `weights.csv` has canonical columns (`start_date`, `end_date`, `day_index`, `date`, `price_usd`, `weight`).

### Next step

- Review schema guarantees: [Strategy TimeSeries](reference/strategy-timeseries.md).

## I want to run daily execution safely

### Prerequisites

- Strategy loads successfully.
- You know your canonical total window budget in USD.
- Optional for live mode: an adapter class implementing `submit_order(...)`.

### Command

```bash
stacksats strategy run-daily \
  --strategy my_strategy.py:MyStrategy \
  --total-window-budget-usd 1000 \
  --mode paper
```

### Expected output

- Structured JSON summary with run metadata and order fields.
- Status line: `EXECUTED`, `NO-OP (idempotent)`, or `FAILED`.
- Ledger state written to `.stacksats/run_state.sqlite3` by default.

### Troubleshooting

- Live mode requires `--adapter module_or_path:ClassName`.
- Reruns with changed parameters require `--force`.
- Missing run-date BTC price coverage causes deterministic failure.

### Next step

- Inspect generated artifact JSON under `output/<strategy_id>/<version>/daily/<run_date>/`.

## I want to migrate from removed legacy internals

### Prerequisites

- You know old helper/constant usage in your integration code.

### Command

- Use the migration mapping directly: [Migration Guide](migration.md).

### Expected output

- No references to removed internals (`compute_weights_shared`, `_FEATURES_DF`, `BACKTEST_END`, old `generate_date_ranges` signature).

### Troubleshooting

- Search your repository for old symbols and replace using migration mappings.

### Next step

- Re-run fast tests and docs checks (`venv/bin/python -m pytest -q`, `venv/bin/python -m mkdocs build --strict`).

## I want minimal strategy code templates

### Prerequisites

- You know which intent style you want to start with (`propose_weight` or `build_target_profile`).

### Command

- Open and copy from: [Minimal Strategy Examples](start/minimal-strategy-examples.md).

### Expected output

- You can paste a working minimal strategy class and run validation immediately.

### Troubleshooting

- If unsure which hook to use, start with `build_target_profile` for vectorized logic.
- Use `propose_weight` when your policy is naturally day-by-day.

### Next step

- Validate your copied strategy using the [CLI validate command](commands.md#2-validate-strategy-via-strategy-lifecycle-cli).

## I want to train and promote the DuckDB alpha strategy

### Prerequisites

- `bitcoin_analytics.duckdb` is present locally, or `STACKSATS_ANALYTICS_DUCKDB` points to it.
- Local editable install is ready (`pip install -e ".[dev]"`).

### Commands

```bash
export STACKSATS_ANALYTICS_DUCKDB=./bitcoin_analytics.duckdb
venv/bin/python scripts/train_duckdb_factor_strategy.py \
  --start-date 2018-01-01 \
  --end-date 2025-05-31 \
  --output stacksats/strategies/duckdb_alpha_v1.json

venv/bin/python scripts/compare_duckdb_alpha.py \
  --start-date 2018-01-01 \
  --end-date 2025-05-31
```

### Expected output

- A refreshed frozen artifact JSON at `stacksats/strategies/duckdb_alpha_v1.json`.
- Baseline-vs-candidate comparison JSON with score, win-rate, and BTC-per-$1M uplift.
- Strict diagnostics included when `--skip-strict` is not set.

### Promotion checklist

- Candidate score delta and win-rate delta are positive on the shared horizon.
- Strict diagnostics are reviewed for permutation p-value, fold stability, and feature drift.
- Comparison output + artifact hash are captured in PR notes.

## I want to troubleshoot command failures quickly

### Prerequisites

- You have the exact command and full error output.

### Command

- Use [CLI Commands](commands.md#troubleshooting) and [Validation Checklist](validation_checklist.md).

### Expected output

- You can classify failures as strategy-loading, data-range, validation-gate, or export-configuration issues.

### Troubleshooting

- Re-run with explicit start/end dates.
- Confirm strategy spec format and module importability.
- Use strict mode for deeper diagnostics.

### Next step

- If still blocked, open a docs feedback issue with context or check [FAQ](faq.md).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Task+Hub)

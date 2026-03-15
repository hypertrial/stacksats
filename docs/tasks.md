---
title: Task Hub
description: Task-first workflows for common StackSats goals.
---

# I Want To...

Use this page to jump directly to the workflow you need.

Global source contract:
- Strategy runtime and validation use BRK parquet only (no other data backends).
- Set `STACKSATS_ANALYTICS_PARQUET` or place `bitcoin_analytics.parquet` at repo root.
- For canonical Drive-distributed artifacts + checksum workflow, see [BRK Data Source](data-source.md).

## I want to validate a strategy

### Prerequisites

- Strategy file exists and can be loaded (`module_or_path:ClassName`).
- Local install is complete (`venv/bin/python -m pip install -e ".[dev]"`).

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

- Run a full backtest: [Backtest Command](run/backtest.md).

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
- Full flag reference: [Backtest Command](run/backtest.md).

## I want animated performance visuals

### Prerequisites

- You already have a backtest artifact (`backtest_result.json`).
- The run directory has write access for GIF/manifest output.

### Command

```bash
stacksats strategy animate \
  --backtest-json output/<strategy_id>/<version>/<run_id>/backtest_result.json \
  --output-dir output/<strategy_id>/<version>/<run_id> \
  --output-name strategy_vs_uniform_hd.gif \
  --fps 20 \
  --width 1920 \
  --height 1080 \
  --max-frames 240 \
  --window-mode rolling
```

### Expected output

- `strategy_vs_uniform_hd.gif`
- `animation_manifest.json` with `frames`, `fps`, `window_mode`, and `source_backtest_json`.

### Troubleshooting

- Ensure `backtest_result.json` has `window_level_data` rows with percentile/SPD fields.
- If render time is high, lower `--fps` or `--max-frames`, or reduce output dimensions.
- Use `--window-mode non-overlapping` for a smaller, communication-safe timeline.

### Next step

- Share the GIF with fold/validation metrics for context (not as a standalone quality signal).
- Full flag reference: [Animate Command](run/animate.md).

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
- Full flag reference: [Export Command](run/export.md).

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
- Full flag reference: [Run Daily Command](run/run-daily.md).

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

- Validate your copied strategy using the [Validate Command](run/validate.md).

## I want to troubleshoot command failures quickly

### Prerequisites

- You have the exact command and full error output.

### Command

- Use [Command Index](commands.md), [Validation Checklist](validation_checklist.md), and the specific command page under `Run`.

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

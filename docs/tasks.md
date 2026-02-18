---
title: Task Hub
description: Task-first workflows for common StackSats goals.
---

# I Want To...

Use this page to jump directly to the workflow you need.

## I want to validate a strategy

### Prerequisites

- Strategy file exists and can be loaded (`module_or_path:ClassName`).
- Local install is complete (`pip install -e .`).

### Command

```bash
stacksats strategy validate \
  --strategy my_strategy.py:MyStrategy \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --strict
```

### Expected output

- Terminal summary includes `Validation PASSED` or `Validation FAILED`.
- Win-rate and leakage/constraint checks are reported.

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
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
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
  --start-date 2025-12-01 \
  --end-date 2027-12-31 \
  --output-dir output
```

### Expected output

- Artifacts under `output/<strategy_id>/<version>/<run_id>/`.
- Files include `weights.csv`, `timeseries_schema.md`, and `artifacts.json`.

### Troubleshooting

- Export requires both `--start-date` and `--end-date`.
- Validate `weights.csv` has canonical columns (`start_date`, `end_date`, `day_index`, `date`, `price_usd`, `weight`).

### Next step

- Review schema guarantees: [Strategy TimeSeries](reference/strategy-timeseries.md).

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

- Re-run tests and docs checks (`pytest -q`, `mkdocs build --strict`).

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

---
title: Task Hub
description: Task-first workflows for common StackSats goals.
---

# I Want To...

Use this page when you already know the outcome you want and need the shortest path to the right workflow.

For package install and first run, start with [Quickstart](start/quickstart.md).
For exact CLI syntax and command grouping, use [Command Index](commands.md).
For canonical source dataset details, use [BRK Data Source](data-source.md) and [Merged Metrics Parquet Schema](reference/merged-metrics-parquet-schema.md).

## I want to validate a strategy

### Prerequisites

- Strategy selector is valid (`strategy_id` for built-ins or `module_or_path:ClassName` for custom strategies).
- StackSats is installed and the command is importable.

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
- Strict validation runs by default in this CLI flow; use `--no-strict` only when you intentionally want the lighter path.

### Troubleshooting

- Check strategy selector format: built-in `strategy_id` or `module_or_path:ClassName`.
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

## I want to compare strategies against benchmarks

### Prerequisites

- You have two or more strategy selectors (built-in `strategy_id` or `module_or_path:ClassName`).
- For custom strategies, pass explicit `--start-date` and `--end-date` together (catalog defaults apply only when every selector resolves to a catalog entry).

### Command

```bash
stacksats strategy compare \
  --strategy simple-zscore \
  --strategy mvrv \
  --baseline uniform \
  --start-date 2018-01-01 \
  --end-date 2025-12-31 \
  --output-dir output
```

### Expected output

- Printed comparison table (win rate, score, exp-decay, vs uniform, judgment).
- `comparison_result.json` under `output/<baseline_strategy_id>/comparison/<run_id>/` with `schema_version`, shared `comparison_window`, and per-strategy rows (including deltas vs baseline).

### Library alternative

- Instantiate `BaseStrategy` objects and call `StrategyRunner().compare(strategies, ComparisonConfig(...))`, or for catalog strategies use `BaseStrategy.compare_to_benchmarks()` (reads `benchmark_strategy_ids` from the catalog entry).

### Troubleshooting

- **Missing dates for custom strategies:** supply both start and end dates.
- **Baseline not in set:** `--baseline` must match a strategy in the ordered set (baseline is prepended and de-duplicated).

### Next step

- Full flag reference: [Compare Command](run/compare.md).
- Deep dive on metrics: [Interpret Backtest Metrics](recipes/interpret-backtest.md).

## I want to create animated performance visuals

### Prerequisites

- You already have a backtest artifact (`backtest_result.json`).
- The run directory has write access for GIF/manifest output.
- Visual extras are installed: `pip install "stacksats[viz]"`.

### Command

```bash
stacksats strategy animate \
  --backtest-json output/<strategy_id>/<version>/<run_id>/backtest_result.json \
  --output-dir output/<strategy_id>/<version>/<run_id> \
  --output-name strategy_vs_uniform_hd.gif \
  --video-format mp4 \
  --fps 20 \
  --width 1920 \
  --height 1080 \
  --max-frames 240 \
  --window-mode non-overlapping
```

### Expected output

- `strategy_vs_uniform_hd.gif`
- `strategy_vs_uniform_hd.mp4` when `--video-format mp4` is set
- `animation_manifest.json` with `schema_version`, render settings, and source/provenance fields.

### Troubleshooting

- Ensure `backtest_result.json` has `window_level_data` rows with percentile/SPD fields.
- If render time is high, lower `--fps` or `--max-frames`, or reduce output dimensions.
- Use `--window-mode non-overlapping` for a smaller, communication-safe timeline.
- Video export requires a system `ffmpeg` binary.

### Next step

- Share the MP4 or GIF with fold/validation metrics for context (not as a standalone quality signal).
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

- Review schema guarantees: [WeightTimeSeries](reference/strategy-timeseries.md).
- Full flag reference: [Export Command](run/export.md).

## I want to generate an agent-consumable daily decision

### Prerequisites

- Strategy loads successfully.
- You know your canonical total window budget in USD.
- An external AI agent or downstream system will handle brokerage execution.

### Command

```bash
stacksats strategy decide-daily \
  --strategy run-daily-paper \
  --total-window-budget-usd 1000
```

### Expected output

- Structured JSON decision payload with recommended weight, notional, BTC quantity, and provenance fields.
- Status line: `DECIDED`, `NO-OP (idempotent)`, or `FAILED`.
- Decision state written to `.stacksats/run_state.sqlite3` by default.

### Troubleshooting

- If a rerun fails after inputs changed, rerun with `--force`; unchanged reruns still return `NO-OP (idempotent)`.
- If price coverage or allocation-window checks fail, verify runtime BRK parquet path, run-date coverage, and the selected `--btc-price-col`.
- If strict validation fails, inspect the decision payload fields `validation_receipt_id`, `data_hash`, and `feature_snapshot_hash`.

### Next step

- Hand the payload to your external AI agent or broker-specific automation layer.
- Full flag reference: [Decide Daily Command](run/decide-daily.md).

## I want to run daily execution safely

### Prerequisites

- Strategy loads successfully.
- You know your canonical total window budget in USD.
- Optional for live mode: an adapter class implementing `submit_order(...)`.
- Use this path only when you want StackSats to submit the order itself after generating the validated decision.

### Command

```bash
stacksats strategy run-daily \
  --strategy run-daily-paper \
  --total-window-budget-usd 1000 \
  --mode paper
```

Replace `run-daily-paper` with your own built-in `strategy_id` or custom strategy spec once you move past the canonical paper-flow example.

### Expected output

- Structured JSON summary with run metadata and order fields.
- Status line: `EXECUTED`, `NO-OP (idempotent)`, or `FAILED`.
- Ledger state written to `.stacksats/run_state.sqlite3` by default.

### Troubleshooting

- Live mode requires `--adapter module_or_path:ClassName`.
- Reruns with changed parameters require `--force`; unchanged reruns still return `NO-OP (idempotent)`.
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

- No references to removed internals (`compute_weights_shared`, `_FEATURES_DF`, old `generate_date_ranges` signature).

### Troubleshooting

- Search your repository for old symbols and replace using migration mappings.

### Next step

- Re-run your local checks after updating imports and symbols.

## I want to start from minimal strategy code templates

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

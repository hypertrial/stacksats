---
title: Backtest Command
description: Reference for `stacksats strategy backtest`.
---

# Backtest Command

## Prerequisites

- Strategy validates successfully on target date bounds.
- Canonical source dataset is `merged_metrics*.parquet` (see [Merged Metrics Parquet Schema](../reference/merged-metrics-parquet-schema.md)).
- Runtime BRK-wide parquet covers the requested window and can resolve from `STACKSATS_ANALYTICS_PARQUET`, managed default `~/.stacksats/data/bitcoin_analytics.parquet`, or `./bitcoin_analytics.parquet`.

## Command

```bash
stacksats strategy backtest \
  --strategy simple-zscore \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

Built-in strategy catalog and expected behavior: [Strategies](../reference/strategies.md).

## Expected output

- Run artifacts under `output/<strategy_id>/<version>/<run_id>/`.
- Includes `backtest_result.json`, `metrics.json`, and plot `.svg` files.

## Key options

- `--strategy-config <path>`: strategy params JSON.
- default scoring bounds if omitted: start `2018-01-01`, end `2025-12-31` (clamped to available data).
- `--strategy-label <text>`: label override for run context.
- `--output-dir <dir>`: artifact root (default `output`).

## Troubleshooting

- If date range fails, verify ordering and source coverage.
- If outputs drift across runs, run validation in strict mode and inspect diagnostics.

## Next step

- Compare multiple strategies on the same window (shared validate/backtest and `comparison_result.json`): [Compare Command](compare.md).
- Export fixed-window artifacts: [Export Command](export.md).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Backtest+Command)

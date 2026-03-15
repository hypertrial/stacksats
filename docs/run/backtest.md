---
title: Backtest Command
description: Reference for `stacksats strategy backtest`.
---

# Backtest Command

## Prerequisites

- Strategy validates successfully on target date bounds.
- BRK parquet (or data source) covers the requested window.

## Command

```bash
stacksats strategy backtest \
  --strategy stacksats.strategies.examples:SimpleZScoreStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

## Expected output

- Run artifacts under `output/<strategy_id>/<version>/<run_id>/`.
- Includes `backtest_result.json`, `metrics.json`, and plot `.svg` files.

## Key options

- `--strategy-config <path>`: strategy params JSON.
- `--strategy-label <text>`: label override for run context.
- `--output-dir <dir>`: artifact root (default `output`).

## Troubleshooting

- If date range fails, verify ordering and source coverage.
- If outputs drift across runs, run validation in strict mode and inspect diagnostics.

## Next step

- Export fixed-window artifacts: [Export Command](export.md).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Backtest+Command)

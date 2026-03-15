---
title: Validate Command
description: Reference for `stacksats strategy validate`.
---

# Validate Command

## Prerequisites

- Strategy spec format: `module_or_path:ClassName`
- BRK parquet available (`STACKSATS_ANALYTICS_PARQUET` or `./bitcoin_analytics.parquet`)

## Command

```bash
stacksats strategy validate \
  --strategy stacksats.strategies.examples:SimpleZScoreStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

## Expected output

- Summary line with pass/fail and gate statuses.
- Strict validation is enabled by default.

## Key options

- `--strategy-config <path>`: strategy params JSON.
- `--min-win-rate <float>`: validation threshold (default `50.0`).
- `--no-strict`: disable strict robustness gates intentionally.

## Troubleshooting

- If strategy import fails, verify module path and editable install.
- If strict gates fail, inspect diagnostics with [Validation Checklist](../validation_checklist.md).

## Next step

- Run a historical evaluation: [Backtest Command](backtest.md).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Validate+Command)

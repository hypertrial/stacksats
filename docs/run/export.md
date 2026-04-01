---
title: Export Command
description: Reference for `stacksats strategy export`.
---

# Export Command

## Prerequisites

- Explicit date bounds are required.
- Strategy loads successfully.
- Runtime BRK-wide parquet available (`STACKSATS_ANALYTICS_PARQUET`, managed default `~/.stacksats/data/bitcoin_analytics.parquet`, or `./bitcoin_analytics.parquet`).

## Command

```bash
stacksats strategy export \
  --strategy simple-zscore \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

Built-in strategy catalog and expected behavior: [Strategies](../reference/strategies.md).

## Expected output

- Artifacts under `output/<strategy_id>/<version>/<run_id>/`.
- Includes `weights.csv`, `timeseries_schema.md`, and `artifacts.json`.

## Key options

- `--strategy-config <path>`: strategy params JSON.
- `--output-dir <dir>`: artifact root (default `output`).

## Troubleshooting

- If export fails, verify `--start-date` and `--end-date` are both present and ordered.
- If runtime data is missing, run `stacksats data doctor` or follow [Full Data Setup](../start/full-data-setup.md).
- If schema checks fail, run `venv/bin/python scripts/sync_objects_schema_docs.py --check`.

## Next step

- Move to agent-facing daily decisions: [Decide Daily Command](decide-daily.md).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Export+Command)

---
title: Data Command
description: Reference for explicit BRK data setup and diagnostics commands.
---

# Data Command

Use the `data` command group for canonical Bitcoin Research Kit (BRK) setup and runtime diagnostics.

This command group supports StackSats' BRK-derived data workflow. Use [BRK Data Source](../data-source.md) for the canonical upstream links and support boundary.

## Fetch source data

```bash
stacksats data fetch
```

Downloads the canonical `merged_metrics*.parquet` asset into `~/.stacksats/data/brk/`
by default and writes the packaged schema markdown beside it.

## Prepare runtime parquet

```bash
stacksats data prepare
```

Builds `~/.stacksats/data/bitcoin_analytics.parquet` by default.

## Inspect runtime resolution

```bash
stacksats data doctor
```

Reports:

- paths checked
- resolved runtime parquet path
- coverage dates
- key columns
- suggested next steps

## Notes

- Runtime commands never auto-download data.
- Runtime readers load parquet lazily first and collect only when an eager frame is required.
- `data prepare` may also normalize an already-wide runtime parquet if you pass one
  explicitly via `--source`.
- Use [BRK Data Source](../data-source.md) for the full source contract.

---
title: Full Data Setup
description: Fetch canonical Bitcoin Research Kit (BRK) source data and prepare the runtime parquet used by StackSats.
---

# Full Data Setup

Use this path after the offline demo when you want the canonical Bitcoin Research Kit (BRK) dataset.

StackSats supports BRK as the upstream project for this data workflow. This page stays focused on the StackSats Python path around BRK-derived data artifacts; use [BRK Data Source](../data-source.md) for the canonical upstream links and support boundary.

## 1) Fetch the canonical source data

```bash
stacksats data fetch
```

Default download location:

```text
~/.stacksats/data/brk/
```

This downloads the canonical `merged_metrics*.parquet` source asset and writes the
packaged schema sidecar into the same directory.

## 2) Prepare the runtime parquet

```bash
stacksats data prepare
```

Default runtime output:

```text
~/.stacksats/data/bitcoin_analytics.parquet
```

## 3) Confirm runtime resolution

```bash
stacksats data doctor
```

Runtime resolution order:

1. `STACKSATS_ANALYTICS_PARQUET`
2. explicit `parquet_path`
3. `~/.stacksats/data/bitcoin_analytics.parquet`
4. `./bitcoin_analytics.parquet`

## 4) Run a standard strategy command

```bash
stacksats strategy backtest \
  --strategy simple-zscore \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

## Notes

- `data fetch` is explicit setup only; runtime commands do not auto-download data.
- `data prepare` accepts canonical long-format `merged_metrics*.parquet` and produces the
  runtime `bitcoin_analytics.parquet`.
- `data doctor` is the fastest way to debug missing-path and coverage problems.
- For notebook/script exploration of the canonical long-format parquet, use
  [EDA Quickstart](eda-quickstart.md) and `stacksats.eda`.

## Next Steps

- [EDA Quickstart](eda-quickstart.md)
- [Quickstart](quickstart.md)
- [BRK Data Source](../data-source.md)
- [Data Command](../run/data.md)

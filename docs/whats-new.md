---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.7.3 highlights

- Added `ColumnMapDataProvider` for flexible data ingestion without a parquet file. Users can supply any Pandas DataFrame by mapping library-canonical column names (e.g. `price_usd`, `mvrv`) to their DataFrame column names.
- Added `StrategyRunner.from_dataframe(df, column_map=...)` as the primary entry point for using StackSats without a BRK parquet file.
- Renamed `StrategyTimeSeries` → `TimeSeries` and `StrategyTimeSeriesBatch` → `TimeSeriesBatch` for a cleaner public API. `Strategy` and `TimeSeries` are now the two lead objects.

## Upgrade notes

- No new runtime feature APIs are introduced in this release.
- Source-contract posture is strict: BRK parquet (or user-supplied DataFrame) is the supported metrics source for strategy workflows.
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

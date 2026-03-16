---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.8.1 highlights

- Fixed default backtest/validation scoring horizon to `2018-01-01` -> `2025-12-31` for stable comparisons (while clamping to available data coverage).
- Enabled feature warmup history by default in loader-backed runtime paths so rolling features can use pre-start context.
- Added canonical merged-metrics parquet schema docs and aligned data-source guidance around long-format source + BRK-wide runtime projection.
- Hardened loader/export edge behavior and added targeted regression coverage for warmup and short-lookback runtime paths.

## Upgrade notes

- No new top-level public API families are introduced in this release.
- Source-contract posture remains strict: canonical source dataset is `merged_metrics*.parquet`, while runtime workflows consume a derived BRK-wide parquet (or user-supplied Polars DataFrame).
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

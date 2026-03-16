---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.8.0 highlights

- Completed the hard-cut Polars-only migration across runtime contracts, framework helpers, tests, and docs.
- Finalized the parquet-only BRK data path and removed the remaining DuckDB runtime surface.
- Added strategy hot-path profiling and hardened the built-in strategy audit flow with projected `merged_metrics*.parquet` loading and partial-failure reporting.
- Canonical dataset docs now center on long-format `merged_metrics*.parquet` schema, with runtime BRK-wide parquet documented as a derived projection.

## Upgrade notes

- No new top-level public API families are introduced in this release; the major change is the Polars-only and parquet-only contract cutover.
- Source-contract posture is strict: canonical source dataset is `merged_metrics*.parquet`, while runtime workflows consume a derived BRK-wide parquet (or user-supplied Polars DataFrame).
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

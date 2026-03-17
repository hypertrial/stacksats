---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.8.2 highlights

- Added an offline packaged first-run flow via `stacksats demo backtest`.
- Added explicit BRK setup commands: `stacksats data fetch`, `stacksats data prepare`, and `stacksats data doctor`.
- Added a release wheel-install smoke test to protect packaged CLI/runtime flows.
- Increased repository coverage gating to 100% with expanded branch-path regression coverage.
- Improved Polars backtest/runtime hot paths with lazy loading safeguards and batch-flow hardening.
- Restored loader/backtest edge-case handling for out-of-coverage windows and aligned docs/README command references.

## Upgrade notes

- The main CLI onboarding path is now `stacksats demo backtest`, not a manually prepared runtime parquet.
- Source-contract posture remains strict: canonical source dataset is `merged_metrics*.parquet`, while runtime workflows consume a derived BRK-wide parquet (or user-supplied Polars DataFrame).
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

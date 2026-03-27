---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 1.0.1 highlights

- Enforced true `100%` line and branch coverage for the `stacksats/` package in the release gate.
- Added targeted regression coverage for optional dependency paths, plotting/runtime fallbacks, EDA helpers, and strategy time-series edge cases.
- Synced maintainer docs and release guidance with the current branch-aware coverage contract.

## 1.0.0 highlights

- Froze the stable `1.x` contract around top-level `stacksats` exports, documented artifact payloads, and the documented `stacksats` CLI subtree.
- Added `schema_version = "1.0.0"` to stable JSON artifact payloads and locked those shapes in snapshot coverage.
- Split optional dependencies into `viz`, `network`, and `deploy` extras while keeping base installs focused on the stable core runtime.
- Moved advanced BRK overlay models under `stacksats.strategies.experimental.*` and kept them outside the stable `1.x` compatibility promise.
- Added a release-grade gate with isolated built-wheel smoke checks plus macOS supported-platform smoke coverage.
- Published the formal stability policy and aligned docs with best-effort causal linting rather than sandbox claims.

## Upgrade notes

- The main CLI onboarding path is now `stacksats demo backtest`, not a manually prepared runtime parquet.
- Source-contract posture remains strict: canonical source dataset is `merged_metrics*.parquet`, while runtime workflows consume a derived BRK-wide parquet (or user-supplied Polars DataFrame).
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- Use [Stability Policy](stability.md) and [Release Guide](release.md) as the source of truth for the supported `1.x` contract and release process.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.7.0 highlights

- Hard-break source contract is now fully BRK-only for runtime/docs/release guidance.
- Added required CI and release preflight guardrails to block deprecated CoinMetrics token reintroduction.
- Completed StrategyTimeSeries schema/lineage cleanup for BRK source-oriented naming and clearer export schema semantics.
- Migrated stale loader/history/prelude tests to BRK DuckDB-native behavior and removed obsolete legacy-loader tests.
- Updated release prep documentation so 0.7.0 release execution is fully aligned with current scripts and workflows.

## Upgrade notes

- No new runtime feature APIs are introduced in this release.
- Source-contract posture is strict: BRK DuckDB is the only supported metrics authority for strategy workflows.
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

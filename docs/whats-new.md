---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.7.1 highlights

- Stabilized BRK overlay and registry unit tests so they no longer require a pre-existing repository-root `bitcoin_analytics.duckdb` file.
- Added deterministic synthetic DuckDB fixtures in affected unit tests, removing environment-sensitive failures in clean CI runners.
- Preserved the hard-break BRK-only runtime/source contract introduced in `0.7.0`; no runtime API changes were introduced in this patch release.

## Upgrade notes

- No new runtime feature APIs are introduced in this release.
- Source-contract posture is strict: BRK DuckDB is the only supported metrics authority for strategy workflows.
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

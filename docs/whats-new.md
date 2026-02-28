---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the release landing pointer.

## 0.5.2 highlights

- Hardened `StrategyTimeSeries` into a read-only validated artifact object.
- Enforced exact daily-window coverage when export metadata defines `window_start` and `window_end`.
- Added explicit `extra_schema` support for strategy-owned export columns without weakening the core schema contract.
- Added `StrategyTimeSeriesBatch.from_artifact_dir(...)` and related CSV helpers for direct artifact reconstruction.
- Unified batch/window provenance and made export artifact file references portable within the artifact directory.

For full details and migration notes, see [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).
For consolidated upgrade mappings, see [Migration Guide](migration.md).

## Latest release

- `0.5.2` (2026-02-28): latest published release on PyPI.

## Recent PyPI releases

- `0.5.2` (2026-02-28)
- `0.5.1` (2026-02-27)
- `0.4.1` (2026-02-19)
- `0.4.0` (2026-02-18)
- `0.3.2` (2026-02-18)
- `0.3.1` (2026-02-17)
- `0.3.0.post1.dev2` (2026-02-17, pre-release)
- `0.3.0` (2026-02-16)
- `0.2.3` (2026-02-16)
- `0.2.2` (2026-02-16)
- `0.1.1` (2026-02-12)
- `0.1.0` (2026-02-12): initial public release of StackSats.

Tag-only (not published to PyPI): `0.2.0`, `0.2.1`.

## Full release history

For complete release notes and unreleased changes:

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)

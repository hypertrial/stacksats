---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the release landing pointer.

## 0.5.1 highlights

- Added idempotent daily execution via `stacksats strategy run-daily`.
- Added durable SQLite state for replay-safe daily runs and a pluggable execution adapter interface.
- Unified `stacksats.load_data(...)` to strict source-only CoinMetrics semantics with optional `end_date`.
- Removed compatibility APIs `stacksats.model_development.softmax(...)` and `BaseStrategy.export_weights(...)`.
- Hardened `BaseStrategy` with canonical `metadata()`, `params()`, `spec()`, and `intent_mode()` surfaces.
- Added warning-first handling for ambiguous dual-hook strategies and explicit `required_feature_columns()` validation.

For full details and migration notes, see [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).
For consolidated upgrade mappings, see [Migration Guide](migration.md).

## Latest release

- `0.5.1` (2026-02-27): latest published release on PyPI.

## Recent PyPI releases

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

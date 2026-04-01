---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 1.2.0 highlights

- Added a built-in strategy catalog with model cards, generated strategy reference pages, and stable public helpers for discovering cataloged strategies from Python.
- Added a Python-first research workflow for custom model development, including `scripts/research_strategy.py`, committed example strategy-config assets, and a copyable smoke-test template.
- Fixed the starter custom-strategy docs and locked them down with executable onboarding regression coverage so copy-paste model authoring flows stay runnable.

## 1.1.2 highlights

- Finished the package reorg by turning the major runtime roots into stable facade packages backed by smaller internal modules while preserving the documented CLI and public import surfaces.
- Reorganized `tests/unit/` to mirror the runtime domains and added architecture checks that guard facade-only exports plus the intended cross-package import boundaries.
- Fixed the packaged `service` install path so `stacksats.data` no longer pulls in the optional network dependency set at import time.

## 1.1.1 highlights

- No intended public API changes; `1.1.1` is the release-recovery follow-up to `1.1.0`.
- Restored the tag-triggered release gate to a true green state by closing the uncovered `decide-daily` branches and removing one unreachable exception path.
- Added targeted regression coverage for failed and noop daily decision CLI results, strict decision-validation failures, and the default decision config helper.

## 1.1.0 highlights

- Added a stable agent-facing daily decision interface with `DecideDailyConfig`, `DailyDecisionResult`, Python helpers, and the `stacksats strategy decide-daily` CLI command.
- Repositioned StackSats as a decision engine first, with external AI agents or brokerage layers handling execution and `run-daily` retained as the integrated convenience path.
- Added stable `decision_result.json` artifacts plus coverage for idempotent daily decisions, CLI contracts, and isolated example-command smoke verification.

## 1.0.2 highlights

- Fixed the documented stable paper `run-daily` path by adding a dedicated `RunDailyPaperStrategy` and moving daily preflight defaults onto the strategy contract.
- Added a local CLI smoke lane plus broader workflow/release contract coverage so docs examples, daily execution flows, and release scripts are tested together.
- Updated first-party GitHub Actions to Node 24-ready versions and clarified maintainer docs around local, scheduled/manual, and release-grade verification lanes.

## 1.0.1 highlights

- Enforced true `100%` line and branch coverage for the `stacksats/` package in the release gate.
- Added targeted regression coverage for optional dependency paths, plotting/runtime fallbacks, EDA helpers, and strategy time-series edge cases.
- Synced maintainer docs and release guidance with the current branch-aware coverage contract.

## 1.0.0 highlights

- Froze the stable `1.x` contract around top-level `stacksats` exports, documented artifact payloads, and the documented `stacksats` CLI subtree.
- Added `schema_version = "1.0.0"` to stable JSON artifact payloads and locked those shapes in snapshot coverage.
- Split optional dependencies into `viz`, `network`, and `deploy` extras while keeping base installs focused on the stable core runtime.
- Moved advanced BRK overlay models under `stacksats.strategies.experimental.*` and kept them outside the stable `1.x` compatibility promise via catalog-tier metadata.
- Added a release-grade gate with isolated built-wheel smoke checks plus macOS supported-platform smoke coverage.
- Published the formal stability policy and aligned docs with best-effort causal linting rather than sandbox claims.

## Upgrade notes

- If you relied on **direct imports** of internal modules (paths like `stacksats.runner_helpers` or `stacksats.prelude`), see [Python package layout (internal imports)](migration.md#python-package-layout-internal-imports) in the Migration Guide and the **Unreleased** section of [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).
- The main CLI onboarding path is now `stacksats demo backtest`, not a manually prepared runtime parquet.
- Source-contract posture remains strict: canonical source dataset is `merged_metrics*.parquet`, while runtime workflows consume a derived BRK-wide parquet (or user-supplied Polars DataFrame).
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- Use [Stability Policy](stability.md) and [Release Guide](release.md) as the source of truth for the supported `1.x` contract and release process.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

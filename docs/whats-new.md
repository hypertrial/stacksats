---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.6.1 highlights

- Added a reusable GitHub composite action for Python setup/dependency install to reduce CI workflow duplication.
- Added scheduled/manual docs command smoke checks so example lifecycle commands are continuously verified.
- Reset coverage policy to an actionable `97%` fail-under floor in `scripts/check_coverage.sh`, with ratchet-up guidance.
- Added a local cleanup utility script (`scripts/clean_local.sh`) for generated artifacts and caches.
- Improved runner/validation maintainability with targeted helper extraction and split, faster test modules.

## Upgrade notes

- No public runtime API changes are introduced in this release.
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

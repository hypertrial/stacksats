---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.6.0 highlights

- Corrected the release workflow for `setuptools-scm` so maintainers tag `v0.6.0` before building publishable artifacts.
- Strengthened `scripts/release_check.sh` so release preflight now runs docs checks plus the full non-performance test matrix instead of the fast local-default suite.
- Normalized docs around current behavior: strict validation is the default validate path, `run_daily` is strict-gated, and fast vs release-grade test commands are documented consistently.
- Expanded markdown quality coverage to all tracked `.md` files, including `.github` templates and repo policy documents.

## Upgrade notes

- No runtime API changes are introduced in this release-prep PR.
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

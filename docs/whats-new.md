---
title: What's New
description: Release pointers for user-visible StackSats changes.
---

# What's New

Use this page as the current-release landing pointer.

## 0.7.2 highlights

- Reorganized docs navigation around clear user intent paths (`Start`, `Run`, `Build`, `Reference`, `Maintainers`) while keeping compatibility anchors for high-traffic command links.
- Split lifecycle command documentation into canonical pages under `docs/run/` to reduce duplication and make flags/reference details easier to maintain.
- Refreshed docs visual styling with clearer light/dark theming, improved readability, and more consistent cards/code/admonition presentation.
- Refactored docs UX guardrails to check structural outcomes rather than brittle exact heading strings.

## Upgrade notes

- No new runtime feature APIs are introduced in this release.
- Source-contract posture is strict: BRK DuckDB is the only supported metrics authority for strategy workflows.
- If you maintain release workflows, rebuild publishable artifacts only after creating the annotated release tag.
- For behavior and compatibility notes, use [Migration Guide](migration.md) and [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md).

## Release details

- [`CHANGELOG.md`](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)
- [GitHub Releases](https://github.com/hypertrial/stacksats/releases)
- [PyPI project page](https://pypi.org/project/stacksats/)

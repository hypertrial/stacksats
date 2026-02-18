---
title: Docs UX Plan
description: UX goals, success metrics, and acceptance checks for StackSats documentation.
---

# Docs UX Plan

This plan defines UX standards for docs changes and release readiness.

## Primary user personas

- New strategy author: wants first successful validate/backtest/export run quickly.
- Returning integrator: wants stable API boundaries and migration clarity.
- Maintainer: wants docs changes to stay synchronized with runtime and CLI behavior.

## Top user tasks

1. Validate a strategy.
2. Run a backtest and read outputs.
3. Export weights with correct schema.
4. Migrate from breaking changes.
5. Troubleshoot failures quickly.

## UX quality targets

- Navigation depth: critical tasks reachable in two clicks from docs home.
- Task pages: include prerequisites, command/API, expected output, troubleshooting, next step.
- Migration changes: old/new mapping available in one page.
- Canonical commands: live in `docs/commands.md`; other pages link instead of duplicating.

## Metrics to track

- Broken links in docs CI: target `0`.
- Strict docs build failures: target `0`.
- Manual top-task walkthrough success rate: target `100%` on release PR.
- Docs feedback issue volume tagged as discoverability/confusion.

## Release acceptance checklist

- [ ] `bash scripts/check_docs_refs.sh` passes.
- [ ] `python scripts/check_docs_ux.py` passes.
- [ ] `mkdocs build --strict` passes.
- [ ] Top-task walkthroughs are verified by maintainer review.

## Governance

When docs UX targets are not met, release PR should include either:

- a fix, or
- a documented exception with follow-up issue.

---
title: Docs Ownership
description: Ownership model and update triggers for StackSats documentation.
---

# Docs Ownership

This page defines who updates what and when documentation updates are required.

## Canonical page responsibilities

- `docs/commands.md` is the canonical CLI command source.
- `docs/start/*.md` are onboarding guides and should link to canonical command/reference pages instead of duplicating option matrices.
- `README.md` stays concise and should link into docs for deep usage details.
- `docs/whats-new.md` is a release pointer page and must stay aligned with release metadata.

## Section owners

- `docs/framework.md`: framework contract maintainers.
- `docs/model*.md` and `docs/concepts/*`: model/runtime maintainers.
- `docs/reference/*`: API and object model maintainers.
- `docs/commands.md` and `docs/recipes/*`: CLI/runtime maintainers.
- `docs/release.md`: release maintainers.

## Required docs update triggers

Update docs in the same PR when any of these change:

- `stacksats/runner.py` or lifecycle APIs: update runtime/backtest/reference pages.
- `stacksats/strategy_types.py`: update strategy object docs and API reference.
- `stacksats/strategy_time_series.py`: run schema sync script and update TimeSeries docs.
- CLI flag or command behavior changes: update `docs/commands.md` and relevant recipes.

## Generated artifact policy

- `site/` is generated output from `mkdocs build` and must not be committed.
- Keep generated notebook exports only under `docs/assets/notebooks/` when source notebooks change.
- Do not add generated docs artifacts outside the docs asset folders.

## CI expectations

Docs quality gate runs in `docs-check`:

- markdown lint
- spelling checks
- link checks
- strict docs build

## Local checklist

```bash
bash scripts/check_docs_refs.sh
python scripts/sync_objects_schema_docs.py --check
mkdocs build --strict
```

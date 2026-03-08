---
title: Docs Ownership
description: Ownership model and update triggers for StackSats documentation.
---

# Docs Ownership

This page defines who updates what and when documentation updates are required.

## Canonical page responsibilities

- `docs/commands.md` is the canonical CLI command source.
- `docs/tasks.md` is the canonical task-first workflow hub.
- `docs/migration.md` is the canonical old->new compatibility mapping page.
- `docs/start/minimal-strategy-examples.md` is the canonical minimal template page for both strategy hook styles.
- `docs/faq.md` captures recurring docs questions sourced from docs feedback issues.
- `docs/start/*.md` are onboarding guides and should link to canonical command/reference pages instead of duplicating option matrices.
- `README.md` stays concise and should link into docs for deep usage details.
- `CHANGELOG.md` is the canonical release history source.
- `docs/whats-new.md` is the current-release summary page and must stay aligned with the latest changelog release.

## Section owners

- `docs/framework.md`: framework contract maintainers.
- `docs/model*.md` and `docs/concepts/*`: model/runtime maintainers.
- `docs/reference/*`: API and object model maintainers.
- `docs/commands.md` and `docs/recipes/*`: CLI/runtime maintainers.
- `docs/tasks.md` and `docs/migration.md`: CLI/runtime maintainers.
- `docs/start/minimal-strategy-examples.md` and `docs/faq.md`: CLI/runtime maintainers.
- `docs/release.md`: release maintainers.

## Required docs update triggers

Update docs in the same PR when any of these change:

- `stacksats/runner.py` or lifecycle APIs: update runtime/backtest/reference pages.
- `stacksats/strategy_types.py`: update strategy object docs and API reference.
- `stacksats/strategy_time_series.py`: run schema sync script and update TimeSeries docs.
- CLI flag or command behavior changes: update `docs/commands.md` and relevant recipes.
- Breaking or removed compatibility surfaces: update `docs/migration.md`, `docs/whats-new.md`, and `CHANGELOG.md`.
- Repeated docs feedback questions: fold updates into `docs/faq.md` and link affected task/start pages.
- Release tooling or release workflow changes: update `docs/release.md`, `CONTRIBUTING.md`, and release-facing sections in `README.md`.
- `pytest.ini` marker defaults or test-tier expectations: update `README.md`, `CONTRIBUTING.md`, and `docs/release.md`.
- BRK source-contract guardrails (`scripts/check_no_coinmetrics_refs.py`) or source nomenclature changes: update `README.md`, `docs/migration.md`, `docs/commands.md`, and `docs/release.md`.

## Generated artifact policy

- `site/` is generated output from `mkdocs build` and must not be committed.
- Keep generated notebook exports only under `docs/assets/` when source notebooks change.
- Do not add generated docs artifacts outside the docs asset folders.

## CI expectations

Docs quality gate runs in `docs-check`:

- markdown lint
- spelling checks
- link checks
- release docs sync check (`scripts/check_release_docs_sync.py`)
- UX structure checks (`scripts/check_docs_ux.py`)
- strict docs build

## Local checklist

```bash
bash scripts/check_docs_refs.sh
venv/bin/python scripts/check_docs_ux.py
venv/bin/python scripts/check_release_docs_sync.py
venv/bin/python scripts/sync_objects_schema_docs.py --check
venv/bin/python -m mkdocs build --strict
```

---
title: Docs Ownership
description: Ownership model and update triggers for StackSats documentation.
---

# Docs Ownership

This page defines who updates what and when documentation updates are required.

## Canonical page responsibilities

- `docs/tasks.md` is the canonical workflow-intent hub (`what to do`, `when to do it`, `what success looks like`).
- `docs/commands.md` is the canonical command index/routing page.
- `docs/run/*.md` pages are the canonical flag/reference truth for each lifecycle command.
- `docs/reference/strategies.md` is the canonical built-in strategy catalog (intent, required columns, tuning defaults, run guidance).
- `docs/reference/merged-metrics-data-guide.md` is the canonical user-facing answer to what data is available in the long-format BRK parquet.
- `docs/reference/merged-metrics-parquet-schema.md` is the canonical physical schema page for the long-format BRK parquet.
- `docs/reference/merged-metrics-taxonomy.md` is the canonical semantic taxonomy page for the merged-metrics namespace.
- `docs/migration.md` is the canonical old->new compatibility mapping page.
- `docs/start/minimal-strategy-examples.md` is the canonical minimal template page for both strategy hook styles.
- `docs/faq.md` captures recurring docs questions sourced from docs feedback issues.
- `docs/troubleshooting.md` is the link-only troubleshooting hub; detailed answers stay on target pages (`docs/tasks.md`, `docs/commands.md`, validation pages, and so on).
- `docs/start/system-overview.md` is the canonical high-level data-flow and production-path overview for onboarding.
- `docs/start/*.md` are onboarding guides and should link to `docs/run/*.md` instead of duplicating full option matrices.
- `README.md` stays concise and should link into docs for deep usage details.
- `CHANGELOG.md` is the canonical release history source.
- `docs/whats-new.md` is the current-release summary page and must stay aligned with the latest changelog release.

## Section owners

- `docs/framework.md`: framework contract maintainers.
- `docs/model*.md` and `docs/concepts/*`: model/runtime maintainers.
- `docs/reference/*`: API and object model maintainers.
- `docs/reference/strategies.md`: strategy/runtime maintainers.
- `docs/reference/merged-metrics-*.md`: data/runtime maintainers.
- `docs/commands.md`, `docs/run/*`, and `docs/recipes/*`: CLI/runtime maintainers.
- `docs/tasks.md` and `docs/migration.md`: CLI/runtime maintainers.
- `docs/start/minimal-strategy-examples.md`, `docs/faq.md`, and `docs/troubleshooting.md`: CLI/runtime maintainers.
- `docs/start/system-overview.md`: CLI/runtime maintainers (onboarding alignment with framework and data docs).
- `docs/release.md`: release maintainers.

## Required docs update triggers

Update docs in the same PR when any of these change:

- `stacksats/runner/` lifecycle packages (`__init__.py`, `core.py`, `daily.py`, `export.py`, `validation.py`): update runtime/backtest/reference pages.
- `stacksats/strategy_types/` or `stacksats/strategies/base.py`: update strategy object docs and API reference.
- `stacksats/strategy_time_series/` (`__init__.py`, `series.py`, `schema.py`, `metadata.py`): run schema sync script and update WeightTimeSeries docs.
- CLI flag or command behavior changes: update the relevant `docs/run/*.md` page, `docs/commands.md`, and relevant recipes.
- Breaking or removed compatibility surfaces: update `docs/migration.md`, `docs/whats-new.md`, and `CHANGELOG.md`.
- Repeated docs feedback questions: fold updates into `docs/faq.md` and link affected task/start pages.
- Release tooling or release workflow changes: update `docs/release.md`, `CONTRIBUTING.md`, and release-facing sections in `README.md`.
- `scripts/test_example_commands.py` changes: update `docs/release.md`, `README.md`, and `CONTRIBUTING.md`; update `docs/commands.md` too if user-visible example guidance changes.
- `scripts/release_wheel_smoke.py` changes: update `docs/release.md`, `README.md`, and `CONTRIBUTING.md`.
- `pytest.ini` marker defaults or test-tier expectations: update `README.md`, `CONTRIBUTING.md`, and `docs/release.md`.
- Docs/release workflow changes (`.github/workflows/docs-*.yml`, `.github/workflows/example-commands-smoke.yml`, `.github/workflows/coverage-report.yml`, `.github/workflows/release-gate.yml`): update `docs/release.md`, `README.md`, and `CONTRIBUTING.md`; update `docs/commands.md` too if user-visible command guidance changes.
- Test layout changes under `tests/unit/`: update workflow references, marker-contract tests, and any maintainer docs that cite specific test paths.
- BRK source-contract guardrails (`scripts/check_no_coinmetrics_refs.py`) or source nomenclature changes: update `README.md`, `docs/migration.md`, `docs/commands.md`, and `docs/release.md`.
- BRK data distribution changes (`stacksats/assets/brk_data_manifest.json`, `data/brk_data_manifest.json`, `scripts/fetch_brk_data.py`, `stacksats/data/data_setup.py`, Drive workflow): update `docs/data-source.md`, `README.md`, and relevant task/command pages.
- merged-metrics namespace changes (`merged_metrics*.parquet`, `scripts/generate_merged_metrics_taxonomy.py`): regenerate `data/brk_merged_metrics_taxonomy.json`, `data/brk_merged_metrics_catalog.json`, `docs/reference/merged-metrics-data-guide.md`, `docs/reference/merged-metrics-taxonomy.md`, and update `docs/reference/merged-metrics-parquet-schema.md` if the physical contract changes.
- Docs IA changes (`mkdocs.yml`, `docs/commands.md`, `docs/run/*`): update `scripts/check_docs_ux.py` rules in the same PR.
- `package-check` / `package-check-pr` path filters (`.github/workflows/package-check.yml`, `.github/workflows/package-check-pr.yml`): if you change which paths trigger the workflow, update `docs/release.md` when maintainer expectations change (for example skipping doc contract tests).

## Periodic review

When changing CLI flags or command behavior, confirm `docs/run/*.md` remains canonical and that `docs/start/*.md` and `docs/tasks.md` only link or use minimal snippets (avoid duplicating full option matrices).

## Generated artifact policy

- `site/` is generated output from `mkdocs build` and must not be committed.
- Keep generated notebook exports only under `docs/assets/` when source notebooks change.
- Generated merged-metrics reference artifacts are allowed at `data/brk_merged_metrics_*.json` and `docs/reference/merged-metrics-*.md`.

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
venv/bin/python scripts/generate_merged_metrics_taxonomy.py --check
venv/bin/python -m mkdocs build --strict
```

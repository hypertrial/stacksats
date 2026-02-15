---
title: Docs Ownership
description: Ownership model and update triggers for StackSats documentation.
---

# Docs Ownership

This page defines who updates what and when documentation updates are required.

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
- `examples/model_example_notebook.py` or `examples/model_example_notebook_browser.py`: regenerate notebook exports in `docs/assets/notebooks/` via `bash scripts/export_notebook_demo.sh`.

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

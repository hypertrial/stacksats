---
title: Add a Built-in Strategy
description: Maintainer workflow for adding, documenting, and promoting cataloged StackSats built-ins.
---

# Add a Built-in Strategy

Use this guide when you want a strategy to ship as part of the StackSats library.

For one-off or local experimentation, keep using a standalone `my_strategy.py` file and load it with `module_or_path:ClassName`. For maintained built-ins, add a cataloged strategy module and select it everywhere by `strategy_id`.

## When to use each path

- Custom strategy: local or project-specific work loaded with `module_or_path:ClassName`.
- Built-in strategy: maintained StackSats library model loaded by `strategy_id`.
- Agent service registry: built-ins should use `catalog_strategy_id`; custom strategies should keep using `strategy_spec`.

## Scaffold the built-in

Run the maintainer scaffold from the repo root:

```bash
python scripts/new_strategy.py \
  --tier experimental \
  --family overlays \
  --strategy-id alpha-beta \
  --class-name AlphaBetaStrategy \
  --intent profile
```

The scaffold creates:

- the strategy module under `stacksats/strategies/<tier>/<family>/`
- the matching unit test stub under `tests/unit/strategies/`
- a model card stub under `docs/reference/models/`
- missing package `__init__.py` markers for the new family chain
- a catalog stub in `stacksats/strategies/catalog.py`

Model cards under `docs/reference/models/` are intentionally built outside the main docs nav. They stay linked from the generated [Strategies](../reference/strategies.md) page, so adding a new built-in does not require a manual `mkdocs.yml` nav edit.

## Choose the inputs deliberately

- `tier`: support status. The catalog defines support tier; the directory layout is only an organizational convention.
- `family`: a lowercase snake_case grouping such as `signals`, `baselines`, or `overlays`.
- `strategy-id`: the canonical built-in selector. Use lowercase kebab-case because users will type this in the CLI.
- `class-name`: PascalCase Python class name.
- `intent`: choose `propose` for day-by-day strategies or `profile` for vectorized window strategies.

## Fill in the scaffolded stub

After scaffolding:

1. Implement the strategy logic and required feature declarations.
2. Replace placeholder description text in the class, model card, and catalog entry.
3. Set any durable configuration as public attrs or via `params()`.
4. Add or expand unit tests beyond the generated smoke coverage.
5. Review catalog metadata:
   - `tier`
   - `public_export`
   - `audit_enabled`
   - `family`
   - `tags`
   - `owner`
   - `benchmark_strategy_ids`
   - `promotion_stage`
   - docs/backtest/validation defaults

Keep class-level runtime metadata (`strategy_id`, `version`, `description`) authoritative for runtime identity. The catalog is for library-management metadata such as tier, exports, docs grouping, and audit inclusion.

For reusable feature and allocation helpers, use [Model Development Helpers](../concepts/model-development-helpers.md) instead of reimplementing common transforms inline.

## Regenerate docs and verify behavior

Regenerate the built-in strategy reference after changing catalog entries:

```bash
python scripts/generate_strategy_docs.py
```

Compare a candidate against baselines on a shared fixed window:

```bash
python scripts/compare_strategies.py \
  --strategy alpha-beta \
  --strategy mvrv \
  --baseline uniform \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --strict
```

Recommended checks before merging:

```bash
python -m pytest tests/unit/package tests/unit/scripts tests/unit/strategies -q
python -m pytest tests/unit/service tests/unit/cli -q
python -m pytest tests/unit/architecture/test_package_boundaries.py -q
python -m mkdocs build --strict
bash scripts/check_docs_refs.sh
python -m ruff check stacksats scripts tests
```

Run any additional targeted backtest or validation checks needed for the new model family.

## Promotion and selection rules

- Built-ins are loaded by `strategy_id`.
- Custom strategies are loaded by `module_or_path:ClassName`.
- Support tier is defined by the catalog entry, not by the implementation module path.
- Promoting a strategy from experimental to stable is a catalog metadata change first.
- Promotion stage should move deliberately from `research` to `candidate` to `promoted`.
- Moving the implementation module to a different directory is optional cleanup, not the source of truth for support status.

## Related references

- [Strategies](../reference/strategies.md)
- [Model Development Helpers](../concepts/model-development-helpers.md)
- [Create a Strategy](../recipes/create-strategy.md)
- [First Strategy Run](../start/first-strategy-run.md)
- [Agent API Service](../run/agent-api.md)

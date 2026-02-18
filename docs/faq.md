---
title: FAQ
description: Frequently asked questions based on recurring docs feedback themes.
---

# FAQ

This FAQ is sourced from recurring themes in docs feedback submissions and docs-related support questions.

- Browse docs issues: [Documentation issues](https://github.com/hypertrial/stacksats/issues?q=is%3Aissue+label%3Adocumentation)
- Submit new feedback: [Docs feedback template](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md)

## Strategy authoring

### Which hook should I implement: `propose_weight` or `build_target_profile`?

Use `propose_weight(state)` for day-by-day logic. Use `build_target_profile(...)` for vectorized, window-level logic.

See: [Minimal Strategy Examples](start/minimal-strategy-examples.md).

### Do I need to implement both hooks?

No. Implement one intent path. The framework will enforce the same allocation invariants either way.

### Why can’t I override `compute_weights` directly?

`compute_weights` is framework-owned and sealed to preserve invariant safety, lock semantics, and consistent runtime behavior.

See: [Framework Boundary](framework.md).

## CLI and outputs

### Where are artifacts written?

Backtest and export artifacts are written under:

`output/<strategy_id>/<version>/<run_id>/`

See: [CLI Commands](commands.md).

### Why does export require `--start-date` and `--end-date`?

Export requires explicit date bounds to avoid accidental implicit ranges and to make outputs reproducible.

### What files should I expect after a run?

Common files:

- backtest: `backtest_result.json`, `metrics.json`, plot `.svg` files
- export: `weights.csv`, `timeseries_schema.md`, `artifacts.json`

See: [Backtest Runtime](model_backtest.md) and [Strategy TimeSeries](reference/strategy-timeseries.md).

## Migration and compatibility

### My code used `compute_weights_shared` or `BACKTEST_END`. What do I use now?

Use explicit replacements from [Migration Guide](migration.md):

- `compute_weights_shared(...)` -> `compute_weights_with_features(..., features_df=...)`
- `BACKTEST_END` -> `get_backtest_end()`

### Which modules are stable public API?

Use top-level `stacksats` exports and documented API modules. Lower-level modules (`stacksats.backtest`, `stacksats.prelude`, `stacksats.export_weights`) are implementation details and may change.

See: [API Reference](reference/api/index.md).

## Docs feedback workflow

### How is this FAQ updated?

Maintainers periodically review docs feedback issues and fold recurring questions into this page.

### How should I report a docs gap?

Open an issue using the docs feedback template and include:

- page URL
- what you were trying to do
- what was unclear
- suggested improvement

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+FAQ)

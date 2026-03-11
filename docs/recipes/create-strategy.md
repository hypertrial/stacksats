---
title: Recipe - Create a Strategy
description: Task-focused recipe for creating and running a custom StackSats strategy.
---

# Recipe: Create a Strategy

## Goal

Create a custom strategy file and run it end-to-end.

## Steps

1. Create `my_strategy.py` with a `BaseStrategy` subclass.
2. Declare `required_feature_sets()` so the framework materializes the feature providers your strategy needs.
3. Implement `transform_features`, `build_signals`, and one intent hook.
   - Use copyable templates: [Minimal Strategy Examples](../start/minimal-strategy-examples.md).
4. Declare any hard-required transformed columns with `required_feature_columns()`.
5. Keep durable config in public attrs or `params()`, and keep runtime caches private.
6. Validate and backtest with CLI from the canonical command guide:
   - [Validate](../run/validate.md)
   - [Backtest](../run/backtest.md)

## Contract reminders

- `ctx.features_df` is already an observed-only, as-of-materialized frame. Do not assume rows after `current_date` exist.
- Strategy classes must not read files, call databases, or make network requests directly. Use provider-backed feature sets instead.
- Strict validation is the default CLI path and the same strict checks gate `run_daily` paper/live execution.

## Expected output

- Validation summary line with pass/fail status.
- Backtest artifacts in the standard run output directory (see [Export Command](../run/export.md)).

## Common failure patterns

- Non-finite signal values (`NaN`/`inf`).
- Incorrect index alignment across feature and signal series.
- Strategy bypass attempts of framework-owned kernel behavior.
- Missing or unregistered `required_feature_sets()`.
- AST lint blockers such as negative `shift`, centered rolling windows, or direct file/network I/O.
- Ambiguous dual-hook strategies: set `intent_preference` if both `propose_weight` and `build_target_profile` exist.
- Missing required transformed feature columns.

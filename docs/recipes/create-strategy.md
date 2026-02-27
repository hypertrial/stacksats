---
title: Recipe - Create a Strategy
description: Task-focused recipe for creating and running a custom StackSats strategy.
---

# Recipe: Create a Strategy

## Goal

Create a custom strategy file and run it end-to-end.

## Steps

1. Create `my_strategy.py` with a `BaseStrategy` subclass.
2. Implement `transform_features`, `build_signals`, and one intent hook.
   - Use copyable templates: [Minimal Strategy Examples](../start/minimal-strategy-examples.md).
3. Declare any hard-required transformed columns with `required_feature_columns()`.
4. Keep durable config in public attrs or `params()`, and keep runtime caches private.
5. Validate and backtest with CLI from the canonical command guide:
   - [Validate](../commands.md#2-validate-strategy-via-strategy-lifecycle-cli)
   - [Backtest](../commands.md#3-run-full-backtest-via-strategy-lifecycle-cli)

## Expected output

- Validation summary line with pass/fail status.
- Backtest artifacts in the standard run output directory (see [CLI Commands](../commands.md#4-export-strategy-artifacts)).

## Common failure patterns

- Non-finite signal values (`NaN`/`inf`).
- Incorrect index alignment across feature and signal series.
- Strategy bypass attempts of framework-owned kernel behavior.
- Ambiguous dual-hook strategies: set `intent_preference` if both `propose_weight` and `build_target_profile` exist.
- Missing required transformed feature columns.

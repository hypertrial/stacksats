---
title: Simple Z-Score Model Card
description: Model card for the simple-zscore built-in strategy.
---

# SimpleZScore Strategy

## Summary

- `strategy_id`: `simple-zscore`
- intent mode: `profile`
- support tier: `stable`
- promotion stage: `promoted`
- owner: `StackSats Maintainers`

## Why this model exists

`simple-zscore` is the smallest meaningful profile-mode reference model in the library. It provides a readable example of how to tilt preference toward lower valuation regimes without adding complex feature engineering.

## Feature dependencies

- required feature sets: `core_model_features_v1`
- required transformed columns: none

## Benchmarks

- benchmark strategy IDs: `uniform`, `mvrv`

## Expected comparison behavior

Expect finite, distinct profile behavior with modest deltas versus `uniform`. This model is useful when testing new comparison or profiling tooling because its logic is easy to reason about.

## Known failure modes and caveats

This is still a toy reference model. If it outperforms more sophisticated models by a wide margin, the evaluation setup deserves scrutiny before the result is trusted.

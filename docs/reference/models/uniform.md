---
title: Uniform Model Card
description: Model card for the uniform built-in strategy.
---

# Uniform Strategy

## Summary

- `strategy_id`: `uniform`
- intent mode: `propose`
- support tier: `stable`
- promotion stage: `promoted`
- owner: `StackSats Maintainers`

## Why this model exists

`uniform` is the baseline reference for validation, comparisons, audits, and regression checks. It exists to show what framework-default equal allocation looks like when no signal is applied.

## Feature dependencies

- required feature sets: `core_model_features_v1`
- required transformed columns: none

## Benchmarks

- benchmark strategy IDs: none

## Expected comparison behavior

This strategy should behave like the comparison baseline. `multiple_vs_uniform` should stay near `1.0` and comparison deltas should remain close to zero.

## Known failure modes and caveats

If `uniform` drifts materially from baseline metrics, the framework or evaluation setup is likely wrong rather than the model.

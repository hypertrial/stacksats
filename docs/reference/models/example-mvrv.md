---
title: Example MVRV Model Card
description: Model card for the example-mvrv built-in strategy.
---

# ExampleMVRV Strategy

## Summary

- `strategy_id`: `example-mvrv`
- intent mode: `profile`
- support tier: `experimental`
- promotion stage: `research`
- owner: `StackSats Maintainers`

## Why this model exists

`example-mvrv` is an experimental overlay model that shows how to extend the stable MVRV baseline with additional BRK-aware signals. It is kept to exercise the research-to-library path, not as a stable benchmark promise.

## Feature dependencies

- required feature sets: `core_model_features_v1`
- required transformed columns: overlay-specific transformed columns from the experimental implementation

## Benchmarks

- benchmark strategy IDs: `uniform`, `mvrv`

## Expected comparison behavior

Treat this model as a research candidate. Comparisons should focus on whether it behaves distinctly and plausibly versus `mvrv`, not on single-run headline wins.

## Known failure modes and caveats

Experimental overlays can overfit sparse regimes or depend on columns not present in every runtime parquet. Promotion should wait for repeated stable comparisons and clear operational value.

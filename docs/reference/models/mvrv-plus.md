---
title: MVRV Plus Model Card
description: Model card for the mvrv-plus built-in strategy.
---

# MVRVPlus Strategy

## Summary

- `strategy_id`: `mvrv-plus`
- intent mode: `profile`
- support tier: `experimental`
- promotion stage: `candidate`
- owner: `StackSats Maintainers`

## Why this model exists

`mvrv-plus` is an experimental candidate that layers BRK-aware regime logic on top of the stable MVRV baseline. It exists to test whether overlays can improve comparison results without breaking the maintainability of the core model family.

## Feature dependencies

- required feature sets: `core_model_features_v1`
- required transformed columns: overlay-specific transformed columns from the candidate implementation

## Benchmarks

- benchmark strategy IDs: `uniform`, `mvrv`

## Expected comparison behavior

Candidate-stage runs should show distinct, interpretable behavior against `mvrv` and `uniform` on the same window. Comparisons should be repeated, not judged from a single favorable run.

## Known failure modes and caveats

This model is not yet stable. If it only improves on narrow date ranges or becomes too sensitive to overlay availability, it should remain experimental.

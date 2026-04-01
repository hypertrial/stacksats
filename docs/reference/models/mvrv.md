---
title: MVRV Model Card
description: Model card for the mvrv built-in strategy.
---

# MVRV Strategy

## Summary

- `strategy_id`: `mvrv`
- intent mode: `profile`
- support tier: `stable`
- promotion stage: `promoted`
- owner: `StackSats Maintainers`

## Why this model exists

`mvrv` is the core package reference model for valuation-aware allocation. It is the main stable benchmark for comparing new valuation and overlay ideas against a maintained baseline.

## Feature dependencies

- required feature sets: `core_model_features_v1`
- required transformed columns: strategy-defined transformed MVRV features

## Benchmarks

- benchmark strategy IDs: `uniform`

## Expected comparison behavior

Expect clearly non-uniform profile behavior and meaningful comparison deltas on representative windows. Candidate overlay models should usually be compared against `mvrv` before any promotion discussion.

## Known failure modes and caveats

Because this model is a maintained baseline, regressions matter more than one-off wins. Treat unexplained behavior changes as release blockers until validated.

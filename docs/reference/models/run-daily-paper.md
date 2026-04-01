---
title: Run Daily Paper Model Card
description: Model card for the run-daily-paper built-in strategy.
---

# RunDailyPaper Strategy

## Summary

- `strategy_id`: `run-daily-paper`
- intent mode: `propose`
- support tier: `stable`
- promotion stage: `promoted`
- owner: `StackSats Maintainers`

## Why this model exists

`run-daily-paper` is the canonical daily-execution harness for agent and service flows. It is intentionally simple so maintainers can verify end-to-end daily decision plumbing without conflating infrastructure checks with model behavior.

## Feature dependencies

- required feature sets: `core_model_features_v1`
- required transformed columns: none

## Benchmarks

- benchmark strategy IDs: `uniform`

## Expected comparison behavior

This strategy should stay close to `uniform` in score and percentile behavior. The point is operational correctness, not benchmark outperformance.

## Known failure modes and caveats

Do not interpret this model as alpha. It intentionally uses relaxed daily-validation defaults and should not be treated as a production benchmark.

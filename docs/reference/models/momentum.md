---
title: Momentum Model Card
description: Model card for the momentum built-in strategy.
---

# Momentum Strategy

## Summary

- `strategy_id`: `momentum`
- intent mode: `profile`
- support tier: `stable`
- promotion stage: `promoted`
- owner: `StackSats Maintainers`

## Why this model exists

`momentum` is a simple counterexample to valuation-only logic. It gives maintainers a compact reference model for trend-sensitive profile construction and comparison tooling.

## Feature dependencies

- required feature sets: `core_model_features_v1`
- required transformed columns: `price_usd`

## Benchmarks

- benchmark strategy IDs: `uniform`, `mvrv`

## Expected comparison behavior

Expect stable, finite outputs with behavior that differs from `simple-zscore` on trend-led windows. It is useful for checking whether comparison tooling captures meaningful profile differences across simple model families.

## Known failure modes and caveats

This implementation is deliberately simple and uses a short lookback. It should be treated as a reference signal, not as a claim of durable momentum edge.

---
title: Model Development Helpers
description: Practical guide to the public stacksats.model_development helper surface.
---

# Model Development Helpers

This page covers the supported helper functions exposed by `stacksats.model_development`.

Use these helpers when you want to move faster on custom research models without rebuilding common feature or allocation plumbing by hand.

## When to use helpers

- Use a standalone `my_strategy.py` or the copyable templates under `stacksats/strategies/templates/` for fast local research.
- Use helpers when you want framework-compatible feature transforms, preference construction, or proposal-to-weight conversion.
- Use the built-in scaffold only when the model is worth adding to the maintained library.

## Feature precomputation

`precompute_features(df)` derives the core MVRV and MA features used by the built-in model family. Use it when you want a research strategy to start from the same base feature surface as the cataloged models.

## Signal normalization helpers

The public helper surface includes:

- `zscore`
- `rolling_percentile`
- `compute_percentile_signal`
- `compute_signal_confidence`
- `compute_mvrv_volatility`
- `classify_mvrv_zone`
- `compute_acceleration_modifier`
- `compute_adaptive_trend_modifier`
- `compute_asymmetric_extreme_boost`

These are useful when you want reusable, causal signal transforms instead of ad hoc inline math.

## Target-profile and proposal conversion

Use these functions when you already know the intent surface you want to construct:

- `compute_weights_from_target_profile(...)` converts a per-day target profile into final framework weights.
- `compute_weights_from_proposals(...)` converts per-day proposals into final weights with budget and lock semantics preserved.
- `compute_window_weights(...)` runs the full window computation path for precomputed features.

## Allocation helpers

`allocate_sequential_stable(...)` and `allocate_from_proposals(...)` are available when you need lower-level control over how a research signal turns into valid framework allocations.

## Recommended workflow

1. Start from `stacksats/strategies/templates/minimal_propose.py` or `stacksats/strategies/templates/minimal_profile.py`.
2. Use `precompute_features(...)` or the signal helpers where they simplify your model.
3. Validate with a custom selector such as `my_strategy.py:MyStrategy`.
4. Compare against built-ins with `python scripts/compare_strategies.py`.
5. Only move to the built-in scaffold once the model is worth maintaining in the library.

## Related references

- [First Strategy Run](../start/first-strategy-run.md)
- [Minimal Strategy Examples](../start/minimal-strategy-examples.md)
- [Create a Strategy](../recipes/create-strategy.md)
- [Add a Built-in Strategy](../maintainers/add-built-in-strategy.md)
- [Strategies](../reference/strategies.md)

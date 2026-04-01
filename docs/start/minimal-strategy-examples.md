---
title: Minimal Strategy Examples
description: Copyable minimal strategy templates for both propose_weight and build_target_profile styles.
---

# Minimal Strategy Examples

This page contains copyable minimal strategy templates for both supported intent styles in one place.
For built-in strategy behavior and tuning defaults, use [Strategies](../reference/strategies.md). For reusable helper functions, use [Model Development Helpers](../concepts/model-development-helpers.md).

The repo also ships copyable research templates under `stacksats/strategies/templates/minimal_propose.py` and `stacksats/strategies/templates/minimal_profile.py`.

## Choose a style

- Use `propose_weight(state)` when your logic is naturally day-by-day.
- Use `build_target_profile(ctx, features_df, signals)` when your logic is naturally vectorized over the window.

## Example A: `propose_weight(state)` style

Create `my_strategy.py`:

```python
import polars as pl

from stacksats import BaseStrategy, DayState, StrategyContext


class MinimalProposeWeightStrategy(BaseStrategy):
    strategy_id = "minimal-propose-weight"
    version = "1.0.0"
    description = "Minimal strategy using propose_weight(state)."

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ()

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def propose_weight(self, state: DayState) -> float:
        # Simple bias: buy slightly more when MVRV is lower.
        mvrv = (
            float(state.features["mvrv_zscore"][0])
            if "mvrv_zscore" in state.features.columns
            else 0.0
        )
        scale = 1.0 + (-0.05 * mvrv)
        proposal = state.uniform_weight * scale
        return max(0.0, min(state.remaining_budget, proposal))
```

Run:

```bash
stacksats strategy validate \
  --strategy my_strategy.py:MinimalProposeWeightStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

## Example B: `build_target_profile(...)` style

Create `my_strategy.py`:

```python
import polars as pl

from stacksats import BaseStrategy, StrategyContext, TargetProfile


class MinimalTargetProfileStrategy(BaseStrategy):
    strategy_id = "minimal-target-profile"
    version = "1.0.0"
    description = "Minimal strategy using build_target_profile(...)."

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("mvrv_zscore", "price_vs_ma")

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def build_signals(
        self, ctx: StrategyContext, features_df: pl.DataFrame
    ) -> dict[str, pl.Series]:
        del ctx
        value_signal = (-features_df["mvrv_zscore"]).clip(-4, 4)
        trend_signal = (-features_df["price_vs_ma"]).clip(-1, 1)
        return {
            "value": value_signal,
            "trend": trend_signal,
        }

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx
        preference = (0.7 * signals["value"]) + (0.3 * signals["trend"])
        return TargetProfile(
            values=pl.DataFrame({"date": features_df["date"], "value": preference}),
            mode="preference",
        )
```

Run:

```bash
stacksats strategy validate \
  --strategy my_strategy.py:MinimalTargetProfileStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

Strict validation runs by default in the CLI examples above. In Python, opt in with `ValidationConfig(strict=True, ...)`.

## Success Criteria

A successful run for either style should show:

- strategy loads with no import/spec errors
- `spec()` reports stable metadata, intent mode, and params
- validation summary prints pass/fail with gate details
- the leakage gate is reported as `No Forward Leakage`, where `True` means the strategy passed the check
- no forward-leakage or weight-constraint failures for a sane configuration

## Contract Notes

- Keep long-lived config in public attrs or override `params()`.
- Keep runtime caches private (for example `_cache`), so they are excluded from the stable strategy contract.
- Use `required_feature_sets()` for framework-owned provider inputs; do not load files or network data in strategy methods.
- If you implement both intent hooks, set `intent_preference` explicitly.

## Next Steps

- Run a full backtest: [Backtest Command](../run/backtest.md)
- Export weights: [Export Command](../run/export.md)
- Validate quickly: [Validate Command](../run/validate.md)
- Reuse supported feature/allocation helpers: [Model Development Helpers](../concepts/model-development-helpers.md)
- Compare built-ins and expected model behavior: [Strategies](../reference/strategies.md)
- Compare styles and internals: [Framework Boundary](../framework.md)

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Minimal+Strategy+Examples)

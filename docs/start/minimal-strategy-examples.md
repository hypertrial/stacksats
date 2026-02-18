---
title: Minimal Strategy Examples
description: Copyable minimal strategy templates for both propose_weight and build_target_profile styles.
---

# Minimal Strategy Examples

This page contains copyable minimal strategy templates for both supported intent styles in one place.

## Choose a style

- Use `propose_weight(state)` when your logic is naturally day-by-day.
- Use `build_target_profile(ctx, features_df, signals)` when your logic is naturally vectorized over the window.

## Example A: `propose_weight(state)` style

Create `my_strategy.py`:

```python
import pandas as pd

from stacksats import BaseStrategy, DayState, StrategyContext


class MinimalProposeWeightStrategy(BaseStrategy):
    strategy_id = "minimal-propose-weight"
    version = "1.0.0"
    description = "Minimal strategy using propose_weight(state)."

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        return ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()

    def propose_weight(self, state: DayState) -> float:
        # Simple bias: buy slightly more when MVRV is lower.
        mvrv = float(state.features.get("mvrv_zscore", 0.0))
        scale = 1.0 + (-0.05 * mvrv)
        proposal = state.uniform_weight * scale
        return max(0.0, min(state.remaining_budget, proposal))
```

Run:

```bash
stacksats strategy validate \
  --strategy my_strategy.py:MinimalProposeWeightStrategy \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --strict
```

## Example B: `build_target_profile(...)` style

Create `my_strategy.py`:

```python
import pandas as pd

from stacksats import BaseStrategy, StrategyContext, TargetProfile


class MinimalTargetProfileStrategy(BaseStrategy):
    strategy_id = "minimal-target-profile"
    version = "1.0.0"
    description = "Minimal strategy using build_target_profile(...)."

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        return ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()

    def build_signals(
        self, ctx: StrategyContext, features_df: pd.DataFrame
    ) -> dict[str, pd.Series]:
        del ctx
        value_signal = -features_df["mvrv_zscore"].clip(-4, 4)
        trend_signal = -features_df["price_vs_ma"].clip(-1, 1)
        return {
            "value": value_signal,
            "trend": trend_signal,
        }

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del ctx, features_df
        preference = (0.7 * signals["value"]) + (0.3 * signals["trend"])
        return TargetProfile(values=preference, mode="preference")
```

Run:

```bash
stacksats strategy validate \
  --strategy my_strategy.py:MinimalTargetProfileStrategy \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --strict
```

## Success Criteria

A successful run for either style should show:

- strategy loads with no import/spec errors
- validation summary prints pass/fail with gate details
- no forward-leakage or weight-constraint failures for a sane configuration

## Next Steps

- Run a full backtest: [CLI backtest command](../commands.md#3-run-full-backtest-via-strategy-lifecycle-cli)
- Export weights: [CLI export command](../commands.md#4-export-strategy-artifacts)
- Compare styles and internals: [Framework Boundary](../framework.md)

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Minimal+Strategy+Examples)

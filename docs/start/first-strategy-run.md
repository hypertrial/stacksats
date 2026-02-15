# First Strategy Run

This walkthrough gets a custom strategy running with minimal code.

## 1) Create a strategy file

Create `my_strategy.py`:

```python
import pandas as pd

from stacksats import BaseStrategy, StrategyContext, TargetProfile


class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"
    description = "First custom strategy."

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        return ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()

    def build_signals(
        self, ctx: StrategyContext, features_df: pd.DataFrame
    ) -> dict[str, pd.Series]:
        del ctx
        value_signal = -features_df["mvrv_zscore"].clip(-4, 4)
        trend_signal = -features_df["price_vs_ma"].clip(-1, 1)
        return {"value": value_signal, "trend": trend_signal}

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

## 2) Validate your strategy

```bash
stacksats strategy validate --strategy my_strategy.py:MyStrategy --strict
```

## 3) Run backtest and export

```bash
stacksats strategy backtest --strategy my_strategy.py:MyStrategy --output-dir output
stacksats strategy export --strategy my_strategy.py:MyStrategy --output-dir output
```

## 4) Keep strategy responsibilities clean

!!! info "Contract summary"
    You own features, signals, and intent. The framework owns iteration, clipping, and lock semantics.

Read [Framework Boundary](../framework.md) before increasing strategy complexity.

## 5) Troubleshooting

- If validation fails on constraints, check [Validation Checklist](../validation_checklist.md).
- If outputs look unexpected, compare runs with [CLI Commands](../commands.md).

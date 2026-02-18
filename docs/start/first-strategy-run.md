---
title: First Strategy Run
description: Build, validate, and backtest your first custom StackSats strategy.
---

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

Expected output:

- A validation summary line including pass/fail and gate results.

## 3) Run backtest and export

Use the canonical command reference for full option sets:

- [Backtest command](../commands.md#3-run-full-backtest-via-strategy-lifecycle-cli)
- [Export command](../commands.md#4-export-strategy-artifacts)

Expected output location:

```text
output/<strategy_id>/<version>/<run_id>/
```

## 4) Keep strategy responsibilities clean

!!! info "Contract summary"
    You own features, signals, and intent. The framework owns iteration, clipping, and lock semantics.

Read [Framework Boundary](../framework.md) before increasing strategy complexity.

## 5) Troubleshooting

- If validation fails on constraints, check [Validation Checklist](../validation_checklist.md).
- If outputs look unexpected, compare runs with [CLI Commands](../commands.md).
- If upgrading older code, use [Migration Guide](../migration.md).
- If you want copyable templates for both hook styles, use [Minimal Strategy Examples](minimal-strategy-examples.md).

## Success Criteria

A successful first run should include:

- Strategy loads via `my_strategy.py:MyStrategy`.
- Validation completes with clear pass/fail output.
- Backtest/export artifacts appear in run output directory.

## Next Steps

- Use [Task Hub](../tasks.md) for focused workflows.
- Use [Minimal Strategy Examples](minimal-strategy-examples.md) to compare `propose_weight` vs `build_target_profile`.
- Use [Interpret Backtest Metrics](../recipes/interpret-backtest.md) for comparison discipline.
- Use [FAQ](../faq.md) for recurring setup and integration questions.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+First+Strategy+Run)

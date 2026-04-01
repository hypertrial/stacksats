---
title: First Strategy Run
description: Build, validate, and backtest your first custom StackSats strategy.
---

# First Strategy Run

This walkthrough gets a custom strategy running with minimal code.

This page is for custom strategies loaded with `my_strategy.py:MyStrategy`. If you want to add a maintained built-in to the StackSats library, use [Add a Built-in Strategy](../maintainers/add-built-in-strategy.md) and the cataloged `strategy_id` workflow instead.

For copyable research starters inside the repo, see `stacksats/strategies/templates/minimal_propose.py` and `stacksats/strategies/templates/minimal_profile.py`, plus [Model Development Helpers](../concepts/model-development-helpers.md).

## 1) Create a strategy file

Create `my_strategy.py`:

```python
import polars as pl

from stacksats import BaseStrategy, StrategyContext, TargetProfile


class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"
    description = "First custom strategy."
    value_weight = 0.7
    trend_weight = 0.3

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        return ctx.features_df.clone()

    def build_signals(
        self, ctx: StrategyContext, features_df: pl.DataFrame
    ) -> dict[str, pl.Series]:
        del ctx
        value_signal = (-features_df["mvrv_zscore"]).clip(-4, 4)
        trend_signal = (-features_df["price_vs_ma"]).clip(-1, 1)
        return {"value": value_signal, "trend": trend_signal}

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del ctx
        preference = (
            self.value_weight * signals["value"]
        ) + (
            self.trend_weight * signals["trend"]
        )
        return TargetProfile(
            values=pl.DataFrame({"date": features_df["date"], "value": preference}),
            mode="preference",
        )
```

This example is intentionally parameterized with public attrs so you can drive it through
`--strategy-config`. The repo ships a matching starter config at
`examples/strategy_configs/first_strategy_run.example.json`.

## 2) Run the fast local research loop

Config-driven research run:

```bash
python scripts/research_strategy.py \
  --strategy my_strategy.py:MyStrategy \
  --strategy-config examples/strategy_configs/first_strategy_run.example.json \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --compare-strategy simple-zscore \
  --compare-strategy mvrv
```

Expected output:

- strict validation is enabled by default for this helper
- validation and backtest summaries print to the terminal
- `output/research_strategy.json` captures the run plus optional comparison rows

## 3) Try the same strategy on a custom dataframe

If your local research data does not use canonical StackSats column names, pass a column-map JSON:

```json
{
  "price_usd": "Close",
  "mvrv": "MVRV_Ratio"
}
```

Run:

```bash
python scripts/research_strategy.py \
  --strategy my_strategy.py:MyStrategy \
  --strategy-config examples/strategy_configs/first_strategy_run.example.json \
  --dataframe-parquet my_data.parquet \
  --column-map-config my_column_map.json \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --compare-strategy uniform
```

## 4) Make canonical data available

Use [BRK Data Source](../data-source.md) and [Merged Metrics Parquet Schema](../reference/merged-metrics-parquet-schema.md) before validation/backtest.

```bash
stacksats data fetch
stacksats data prepare
```

This prepares the managed runtime parquet at `~/.stacksats/data/bitcoin_analytics.parquet`.
If you already have a runtime-compatible parquet elsewhere, you can still export `STACKSATS_ANALYTICS_PARQUET` explicitly.

## 5) Validate your strategy on canonical runtime data

```bash
stacksats strategy validate --strategy my_strategy.py:MyStrategy
```

Expected output:

- A validation summary line including pass/fail and gate results.
- The leakage gate now prints as `No Forward Leakage: True/False` to make pass/fail semantics explicit.
- Strict validation is enabled by default in the CLI. In Python, opt in with `ValidationConfig(strict=True, ...)`.

## 6) Run backtest and export

Use the canonical command reference for full option sets:

- [Backtest command](../run/backtest.md)
- [Export command](../run/export.md)
- [Validate command](../run/validate.md)

You can also run lifecycle helpers from Python directly:

```python
from stacksats import BacktestConfig, ValidationConfig

strategy = MyStrategy()

run = strategy.run(
    validation_config=ValidationConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        strict=True,
    ),
    backtest_config=BacktestConfig(start_date="2024-01-01", end_date="2024-12-31"),
    include_export=False,
    save_backtest_artifacts=True,
    output_dir="output",
)
print(run.validation.summary())
print(run.backtest.summary())
print(run.output_dir)
```

Expected output location:

```text
output/<strategy_id>/<version>/<run_id>/
```

## 7) Troubleshooting

- If validation fails on constraints, check [Validation Checklist](../validation_checklist.md).
- If validation fails on lint or feature sourcing, confirm `required_feature_sets()` is provider-backed and remove direct file/network access from the strategy class.
- If outputs look unexpected, compare runs with [CLI Commands](../commands.md).
- If you want a copyable smoke test for your custom strategy, start from `examples/tests/custom_strategy_smoke.example.py`.
- If upgrading older code, use [Migration Guide](../migration.md).
- If you want copyable templates for both hook styles, use [Minimal Strategy Examples](minimal-strategy-examples.md).
- If you want reusable signal/allocation helpers, use [Model Development Helpers](../concepts/model-development-helpers.md).

## 8) Keep strategy responsibilities clean

!!! info "Contract summary"
    You own transforms, signals, and intent over observed data only. The framework owns feature sourcing, as-of materialization, iteration, clipping, and lock semantics.

Read [Framework Boundary](../framework.md) before increasing strategy complexity.

## Success Criteria

A successful first run should include:

- Strategy loads via `my_strategy.py:MyStrategy`.
- Validation completes with clear pass/fail output.
- Backtest/export artifacts appear in run output directory.

## Next Steps

- Use [Task Hub](../tasks.md) for focused workflows.
- Use [Minimal Strategy Examples](minimal-strategy-examples.md) to compare `propose_weight` vs `build_target_profile`.
- Use [Model Development Helpers](../concepts/model-development-helpers.md) to reuse framework-compatible transforms and allocation helpers.
- Use [Strategies](../reference/strategies.md) to compare built-in models and expected behavior.
- Use [Interpret Backtest Metrics](../recipes/interpret-backtest.md) for comparison discipline.
- Use [FAQ](../faq.md) for recurring setup and integration questions.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+First+Strategy+Run)

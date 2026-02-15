# StackSats

![Stacking Sats Logo](Stacking%20Sats%20Logo.svg)

[![PyPI version](https://img.shields.io/pypi/v/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Python versions](https://img.shields.io/pypi/pyversions/stacksats.svg)](https://pypi.org/project/stacksats/)
[![Package Check](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml/badge.svg)](https://github.com/hypertrial/stacksats/actions/workflows/package-check.yml)
[![License: MIT](https://img.shields.io/github/license/hypertrial/stacksats)](LICENSE)

StackSats, developed by [Hypertrial](https://www.hypertrial.ai), is a Python package for strategy-first Bitcoin DCA ("stacking sats") research and execution.

Learn more at [www.stackingsats.org](https://www.stackingsats.org).

## Start Here

Start with the docs at [`docs/framework.md`](docs/framework.md). It is the canonical framework contract and the best first read before using StackSats.

For the full docs index, see [`docs/index.md`](docs/index.md).
Hosted docs site: <https://hypertrial.github.io/stacksats/>.

## Framework Principles

- The framework owns budget math, iteration, feasibility clipping, and lock semantics.
- Users own features, signals, hyperparameters, and daily intent.
- Strategy hooks support either day-level intent (`propose_weight(state)`) or batch intent (`build_target_profile(...)`).
- The same sealed allocation kernel runs in local, backtest, and production.

See [`docs/framework.md`](docs/framework.md) for the canonical contract.

## Installation

```bash
pip install stacksats
```

For local development:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Optional deploy extras:

```bash
pip install "stacksats[deploy]"
```

## Quick Start

Create `my_strategy.py`:

```python
import pandas as pd

from stacksats import BaseStrategy, StrategyContext, TargetProfile


class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"
    description = "Example user strategy."

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

    # Optional alternative hook:
    # def propose_weight(self, state) -> float:
    #     return state.uniform_weight


if __name__ == "__main__":
    strategy = MyStrategy()
    validation = strategy.validate()
    print(validation.summary())
    result = strategy.backtest()
    print(result.summary())
    result.plot(output_dir="output")
    result.to_json("output/backtest_result.json")
```

Run it:

```bash
python my_strategy.py
```

## Strategy Lifecycle CLI

```bash
stacksats strategy validate --strategy my_strategy.py:MyStrategy
stacksats strategy backtest --strategy my_strategy.py:MyStrategy --output-dir output
stacksats strategy export --strategy my_strategy.py:MyStrategy --output-dir output
```

Artifacts are written to:

```text
output/<strategy_id>/<version>/<run_id>/
```

## Public API

Top-level exports:

- `BaseStrategy`, `StrategyContext`, `DayState`, `TargetProfile`
- `BacktestConfig`, `ValidationConfig`, `ExportConfig`
- `StrategyArtifactSet`
- `StrategyTimeSeries`, `StrategyTimeSeriesBatch`
- `BacktestResult`, `ValidationResult`
- `load_strategy()`, `load_data()`, `precompute_features()`
- `MVRVStrategy`

## Development

```bash
pytest tests/ -v
ruff check .
bash scripts/check_docs_refs.sh
```

For command examples using the packaged strategy template, see `docs/commands.md`.

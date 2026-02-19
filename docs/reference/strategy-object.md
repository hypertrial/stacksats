---
title: Strategy Object
description: Contract for BaseStrategy identity and user hook responsibilities.
---

# Strategy Object

A strategy subclasses `BaseStrategy` (`stacksats/strategy_types.py`) and defines:

- identity: `strategy_id`, `version`, `description`
- hooks: `transform_features`, `build_signals`
- intent path: `propose_weight(...)` or `build_target_profile(...)`
- lifecycle helpers: `validate(...)`, `backtest(...)`, `export(...)`, `run(...)`

## Ownership model

!!! info "Framework vs user"
    User code owns features/signals/intent. Framework code owns iteration, clipping, and invariants.

Framework-owned behavior remains sealed:

- compute kernel (`compute_weights` in `BaseStrategy`)
- clipping and remaining-budget enforcement
- lock semantics and final invariants

## Minimal strategy shape

```python
class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        ...

    def build_signals(self, ctx: StrategyContext, features_df: pd.DataFrame) -> dict[str, pd.Series]:
        ...

    def build_target_profile(self, ctx: StrategyContext, features_df: pd.DataFrame, signals: dict[str, pd.Series]) -> TargetProfile:
        ...
```

## Lifecycle methods

`BaseStrategy` includes convenience methods for common runtime flows:

- `validate(config=None, **kwargs)`: returns `ValidationResult`
- `backtest(config=None, **kwargs)`: returns `BacktestResult`
- `export(config=None, **kwargs)`: returns `StrategyTimeSeriesBatch`
- `export_weights(config=None, **kwargs)`: backward-compatible alias for `export(...)`
- `backtest_and_save(config=None, output_dir=..., ...)`: runs backtest and writes standard artifacts under `output/<strategy_id>/<version>/<run_id>/`
- `run(...)`: runs validate + backtest, with optional export and optional artifact writing
- `hook_status()`: returns which intent hook path is implemented
- `validate_contract()`: checks framework contract compliance for the strategy instance

## Related docs

- [Framework Boundary](../framework.md)
- [Minimal Strategy Examples](../start/minimal-strategy-examples.md)
- [Strategy TimeSeries](strategy-timeseries.md)
- [FAQ](../faq.md)

---
title: Strategy Object
description: Contract for BaseStrategy identity and user hook responsibilities.
---

# Strategy Object

A strategy subclasses `BaseStrategy` (`stacksats/strategy_types.py`) and defines:

- identity: `strategy_id`, `version`, `description`
- canonical contract surface: `metadata()`, `params()`, `spec()`
- hooks: `transform_features`, `build_signals`
- intent path: `propose_weight(...)` or `build_target_profile(...)`
- lifecycle helpers: `validate(...)`, `backtest(...)`, `export(...)`, `run(...)`

`BaseStrategy.spec()` is the canonical public contract for a strategy. It includes:

- `metadata`: normalized identity (`strategy_id`, `version`, `description`)
- `intent_mode`: the active execution path (`"propose"` or `"profile"`)
- `params`: stable, deterministic strategy configuration used for provenance/idempotency
- `required_feature_columns`: hard-required transformed columns

## Ownership model

!!! info "Framework vs user"
    User code owns features/signals/intent. Framework code owns iteration, clipping, and invariants.

Framework-owned behavior remains sealed:

- compute kernel (`compute_weights` in `BaseStrategy`)
- clipping and remaining-budget enforcement
- lock semantics and final invariants

## Stable Params vs Runtime State

Class attributes remain supported for strategy author ergonomics, but runtime caches are not part of the canonical contract.

- `params()` includes stable public config values from class attrs and instance attrs
- private attrs such as `_cache`, `_coinmetrics_features`, or `_coinmetrics_overlays` are excluded
- unsupported public values must be normalized by overriding `params()`

Use `spec()` when you need a durable view of strategy identity and configuration.

## Minimal strategy shape

```python
class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"
    description = "Example strategy."

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        ...

    def build_signals(self, ctx: StrategyContext, features_df: pd.DataFrame) -> dict[str, pd.Series]:
        ...

    def build_target_profile(self, ctx: StrategyContext, features_df: pd.DataFrame, signals: dict[str, pd.Series]) -> TargetProfile:
        ...
```

Optional contract helpers:

```python
    def required_feature_columns(self) -> tuple[str, ...]:
        return ("price_vs_ma", "mvrv_zscore", "mvrv_gradient")
```

## Intent Selection

- `hook_status()` reports which hooks are structurally implemented
- `intent_mode()` reports which hook path will actually execute

If a strategy implements both intent hooks:

- set `intent_preference = "propose"` or `intent_preference = "profile"` to make the active path explicit
- if unset, StackSats currently warns and falls back to `propose_weight(state)`
- that fallback will become an error in a future breaking release

## Lifecycle methods

`BaseStrategy` includes convenience methods for common runtime flows:

- `validate(config=None, **kwargs)`: returns `ValidationResult`
- `backtest(config=None, **kwargs)`: returns `BacktestResult`
- `export(config=None, **kwargs)`: returns `StrategyTimeSeriesBatch`
- `backtest_and_save(config=None, output_dir=..., ...)`: runs backtest and writes standard artifacts under `output/<strategy_id>/<version>/<run_id>/`
- `run(...)`: runs validate + backtest, with optional export and optional artifact writing
- `metadata()`: returns `StrategyMetadata`
- `params()`: returns stable strategy config
- `spec()`: returns `StrategySpec`
- `intent_mode()`: returns the active intent execution mode
- `required_feature_columns()`: returns hard-required transformed columns
- `hook_status()`: returns which intent hook path is implemented
- `validate_contract()`: checks framework contract compliance for the strategy instance

## Related docs

- [Framework Boundary](../framework.md)
- [Minimal Strategy Examples](../start/minimal-strategy-examples.md)
- [Strategy TimeSeries](strategy-timeseries.md)
- [FAQ](../faq.md)

---
title: Strategy Object
description: Contract for BaseStrategy identity and user hook responsibilities.
---

# Strategy Object

A strategy subclasses `BaseStrategy` (`stacksats/strategy_types/__init__.py`) and defines:

- identity: `strategy_id`, `version`, `description`
- canonical contract surface: `metadata()`, `params()`, `spec()`
- hooks: `transform_features`, `build_signals`
- intent path: `propose_weight(...)` or `build_target_profile(...)`
- optional lazy profile path: `transform_features_lazy(...)`, `build_signal_exprs(...)`, `build_target_profile_lazy(...)`
- lifecycle helpers: `validate(...)`, `backtest(...)`, `export(...)`, `run(...)`

Built-in strategy behavior, required columns, and tuning defaults are maintained in
[Strategies](strategies.md). This page focuses on the base API contract.

`BaseStrategy.spec()` is the canonical public contract for a strategy. It includes:

- `metadata`: normalized identity (`strategy_id`, `version`, `description`)
- `intent_mode`: the active execution path (`"propose"` or `"profile"`)
- `params`: stable, deterministic strategy configuration used for provenance/idempotency
- `required_feature_sets`: framework-owned feature providers needed to materialize `ctx.features_df`
- `required_feature_columns`: hard-required transformed columns

## Ownership model

!!! info "Framework vs user"
    User code owns transforms/signals/intent over observed inputs. Framework code owns feature sourcing, as-of materialization, iteration, clipping, and invariants.

Framework-owned behavior remains sealed:

- compute kernel (`compute_weights` in `BaseStrategy`)
- clipping and remaining-budget enforcement
- lock semantics and final invariants

## Stable Params vs Runtime State

Class attributes remain supported for strategy author ergonomics, but runtime caches are not part of the canonical contract.

- `params()` includes stable public config values from class attrs and instance attrs
- private attrs such as `_cache`, `_brk_features_cache`, or `_brk_overlay_cache` are excluded
- unsupported public values must be normalized by overriding `params()`

Use `spec()` when you need a durable view of strategy identity and configuration.

## Minimal strategy shape

```python
import polars as pl

class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"
    description = "Example strategy."

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        ...

    def build_signals(self, ctx: StrategyContext, features_df: pl.DataFrame) -> dict[str, pl.Series]:
        ...

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        ...

    def build_target_profile_lazy(
        self,
        ctx: StrategyLazyContext,
        features_lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        ...
```

Optional contract helpers:

```python
    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("price_vs_ma", "mvrv_zscore", "mvrv_gradient")
```

`ctx.features_df` is always observed-only. Strategy hooks do not receive rows after `current_date`.
StackSats also runs best-effort static causal lint checks for patterns such as direct file/network I/O, negative shifts, and centered rolling windows. These checks are heuristic and source-dependent; they are not a security sandbox or a complete leakage proof.

`build_target_profile_lazy(...)` is opt-in and only used for profile-mode strategies. `propose_weight(...)` remains eager because it depends on per-day state.

## Intent Selection

- `hook_status()` reports which hooks are structurally implemented
- `intent_mode()` reports which hook path will actually execute

If a strategy implements both intent hooks:

- set `intent_preference = "propose"` or `intent_preference = "profile"` to make the active path explicit
- if unset, StackSats raises a contract error during validation/load/spec access

## Lifecycle methods

`BaseStrategy` includes convenience methods for common runtime flows:

- `validate(config=None, **kwargs)`: returns `ValidationResult`
- `backtest(config=None, **kwargs)`: returns `BacktestResult`
- `export(config=None, **kwargs)`: returns `WeightTimeSeriesBatch`
- `backtest_and_save(config=None, output_dir=..., ...)`: runs backtest and writes standard artifacts under `output/<strategy_id>/<version>/<run_id>/`
- `run(...)`: runs validate + backtest, with optional export and optional artifact writing
- `metadata()`: returns `StrategyMetadata`
- `params()`: returns stable strategy config
- `spec()`: returns `StrategySpec`
- `intent_mode()`: returns the active intent execution mode
- `required_feature_sets()`: returns framework-owned provider IDs
- `required_feature_columns()`: returns hard-required transformed columns
- `hook_status()`: returns which intent hook path is implemented
- `validate_contract()`: checks framework contract compliance for the strategy instance

## Related docs

- [Strategies](strategies.md)
- [Framework Boundary](../framework.md)
- [Minimal Strategy Examples](../start/minimal-strategy-examples.md)
- [WeightTimeSeries](strategy-timeseries.md)
- [FeatureTimeSeries](feature-timeseries.md)
- [FAQ](../faq.md)

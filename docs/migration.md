---
title: Migration Guide
description: Breaking-change migration mappings for StackSats runtime and export APIs.
---

# Migration Guide

Use this guide when upgrading code that depended on removed compatibility helpers or older signatures.

## 0.7.x hard-break note

`0.7.0+` is BRK-only for strategy metrics/runtime data sourcing.
Legacy CoinMetrics source paths are removed from active runtime and docs workflows.

## Scope

This page covers migration for:

- backtest helper removals
- runtime end-date default changes
- export/date-range signature changes
- compatibility API removals
- strict source-only loader behavior
- strategy contract hardening for metadata, params, and intent selection
- v1 public-API narrowing and experimental strategy namespace moves

## Old -> New Mapping

| Old usage | New usage |
| --- | --- |
| `compute_weights_shared(window_feat)` | `compute_weights_with_features(window_feat, features_df=...)` |
| `stacksats.backtest._FEATURES_DF` global mutation | pass `features_df` explicitly at call sites |
| hardcoded backtest end constants | `get_backtest_end()` |
| `generate_date_ranges(start, end, min_length_days)` | `generate_date_ranges(start, end)` |
| `RANGE_START` / `RANGE_END` / `MIN_RANGE_LENGTH_DAYS` module defaults | set explicit app-level defaults in your own code/config |
| `stacksats.model_development.softmax(...)` | `stacksats.model_development.helpers.softmax(...)` |
| `BaseStrategy.export_weights(config=None, **kwargs)` | `BaseStrategy.export(config=None, **kwargs)` |
| `stacksats.load_data(cache_dir=..., max_age_hours=...)` | `stacksats.load_data(parquet_path=..., max_staleness_days=..., end_date=...)` |
| `coinmetrics_overlay_v1` provider ID | `brk_overlay_v1` provider ID |
| `PriceUSD_coinmetrics` / `CapMVRVCur` runtime columns | canonical runtime columns `price_usd` / `mvrv` |
| `stacksats.btc_api.coinmetrics_btc_csv` | removed; canonical source dataset is `merged_metrics*.parquet` and runtime consumes a derived BRK-wide parquet |
| `strategy.spec()` as the informal contract | `strategy.spec()` as the canonical public contract |
| `StrategyTimeSeries` | Removed. Use `WeightTimeSeries` |
| `StrategyTimeSeriesBatch` | Removed. Use `WeightTimeSeriesBatch` |
| `TimeSeries` | Removed. Use `WeightTimeSeries` |
| `TimeSeriesBatch` | Removed. Use `WeightTimeSeriesBatch` |
| `stacksats.strategies.model_example:ExampleMVRVStrategy` | `stacksats.strategies.experimental.model_example:ExampleMVRVStrategy` |
| `stacksats.strategies.model_mvrv_plus:MVRVPlusStrategy` | `stacksats.strategies.experimental.model_mvrv_plus:MVRVPlusStrategy` |
| importing advanced BRK overlay models from top-level `stacksats` | use the experimental namespace directly |

## Python package layout (internal imports)

If you imported **implementation modules** directly (not the stable top-level [`stacksats`](reference/public-api.md) surface), update paths after the domain package reorganization:

| Old | New |
| --- | --- |
| `stacksats.model_development_helpers` | `stacksats.model_development.helpers` |
| `stacksats.model_development_allocation` | `stacksats.model_development.allocation` |
| `stacksats.model_development_features` | `stacksats.model_development.features` |
| `stacksats.model_development_weights` | `stacksats.model_development.weights` |
| `stacksats.feature_registry` | `stacksats.features.registry` |
| `stacksats.feature_providers` | `stacksats.features.providers` |
| `stacksats.feature_materialization` | `stacksats.features.materialization` |
| `stacksats.feature_time_series` | `stacksats.features.time_series` |
| `stacksats.column_map_provider` | `stacksats.features.column_map_provider` |
| `stacksats.data_btc` | `stacksats.data.data_btc` |
| `stacksats.data_setup` | `stacksats.data.data_setup` |
| `stacksats.btc_price_fetcher` | `stacksats.data.btc_price_fetcher` |
| `stacksats.prelude` | `stacksats.data.prelude` |
| `stacksats.plot_mvrv` / `plot_weights` / `animation_render` / `animation_data` / `matplotlib_setup` | `stacksats.viz.<module>` |
| `stacksats.export_weights_core` | `stacksats.export_weights.core` |
| `stacksats.export_weights_db` | `stacksats.export_weights.db` |
| `stacksats.export_weights_runtime` | `stacksats.export_weights.runtime` |
| `stacksats.export_weights_sql` | `stacksats.export_weights.sql` |
| `stacksats.strategy_time_series_batch` | `stacksats.strategy_time_series.batch` |
| `stacksats.strategy_time_series_schema` | `stacksats.strategy_time_series.schema` |
| `stacksats.strategy_time_series_metadata` | `stacksats.strategy_time_series.metadata` |
| `stacksats.strategy_time_series_analysis` | `stacksats.strategy_time_series.analysis` |
| `stacksats.strategy_time_series_diagnostics` | `stacksats.strategy_time_series.diagnostics` |
| `stacksats.runner_helpers` | `stacksats.runner.helpers` |
| `stacksats.runner_validation` | `stacksats.runner.validation` |
| `stacksats.execution_state` | `stacksats.execution.state` |
| `stacksats.execution_adapters` | `stacksats.execution.adapters` |

These remain valid **import paths** as packages: `stacksats.runner`, `stacksats.strategy_time_series`, `stacksats.export_weights`, `stacksats.model_development`. Plotting console scripts target `stacksats.viz.plot_mvrv` and `stacksats.viz.plot_weights` (see `pyproject.toml` `[project.scripts]`).

## Polars migration (core objects)

`WeightTimeSeries`, `WeightTimeSeriesBatch`, `FeatureTimeSeries`, and the data layer use Polars internally.

| If you use | Update to |
| --- | --- |
| `series.to_dataframe()` expecting index-based DataFrame semantics | Returns `pl.DataFrame`; use `pl` APIs directly (e.g. `.filter()`, `.row()`, `.is_empty()`) |
| `batch.to_dataframe()` expecting index-based DataFrame semantics | Same as above |
| `WeightTimeSeries.from_dataframe(pd_df)` | Removed. Pass a Polars DataFrame |
| legacy `FeatureTimeSeries` constructor | Removed. Use `from_dataframe(pl_df)` |
| `StrategyContext.from_features_df(df)` | Polars-only; pass a DataFrame with canonical `date` column |
| Strategy hooks (`transform_features`, `build_signals`, `build_target_profile`) | Polars-only; framework passes `ctx.features_df` |

## Code Replacements

### 1) Backtest weight helper

```python
# old
weights = compute_weights_shared(window_feat)

# new
weights = compute_weights_with_features(
    window_feat,
    features_df=features_df,
)
```

### 2) Backtest end date

```python
from datetime import datetime

backtest_end = datetime.strptime(get_backtest_end(), "%Y-%m-%d")
assert max_received <= backtest_end
```

### 3) Date range generation

```python
# old
ranges = generate_date_ranges("2025-12-01", "2027-12-31", 120)

# new
ranges = generate_date_ranges("2025-12-01", "2027-12-31")
```

### 4) Softmax helper location

```python
# old
from stacksats.model_development import softmax

# new
from stacksats.model_development.helpers import softmax
```

### 5) Strategy export helper

```python
# old
batch = strategy.export_weights(config=my_export_config)

# new
batch = strategy.export(config=my_export_config)
```

### 6) Strict data loader contract

```python
# old (assumed synthetic fill/fallback behavior)
df = load_data(parquet_path="./bitcoin_analytics.parquet")

# new (strict source-only and optional explicit end bound)
df = load_data(
    parquet_path="~/.stacksats/data/bitcoin_analytics.parquet",
    max_staleness_days=3,
    end_date="2025-12-31",
)
```

`load_data(...)` now mirrors `BTCDataProvider.load(...)` behavior:
- no synthetic "today" row creation
- no historical date gap filling
- no MVRV fallback substitution
- runtime parquet scans are lazy-first; use `BTCDataProvider.load_lazy(...)` or `ColumnMapDataProvider.load_lazy(...)` if you need a `pl.LazyFrame` before the final eager collection boundary
- canonical source dataset is `merged_metrics*.parquet` (long-format), and runtime consumes a derived BRK-wide parquet via `STACKSATS_ANALYTICS_PARQUET`, managed default `~/.stacksats/data/bitcoin_analytics.parquet`, or legacy local fallback `./bitcoin_analytics.parquet`
- profile strategies can now opt into a lazy execution path with `StrategyLazyContext` and `build_target_profile_lazy(...)`; eager hooks remain supported and unchanged

### 7) Strategy contract hardening

```python
# old (informal)
class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"

# new (canonical contract still supports class attrs, but spec() is the durable surface)
class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"
    intent_preference = "profile"

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("price_vs_ma", "mvrv_zscore")
```

Notes:
- `metadata()` and `params()` now back durable provenance/idempotency behavior
- runtime caches should stay private (for example `_cache`)
- if both intent hooks exist, `intent_preference` must be set explicitly

## Upgrade Checklist

1. Replace removed helpers/constants/APIs using the mapping above.
2. Remove hidden globals and pass runtime dependencies explicitly.
3. Replace `strategy.export_weights(...)` call sites with `strategy.export(...)`.
4. Ensure export calls provide explicit `--start-date` and `--end-date`.
5. Validate `load_data(...)` consumers against strict source-only behavior.
6. If your strategy implements both intent hooks, set `intent_preference` explicitly.
7. Move runtime caches to private attrs and override `params()` if you expose non-serializable public config.
8. Update imports for experimental built-ins if you used the old pre-v1 paths.
9. Re-run fast tests:

```bash
venv/bin/python -m pytest -q
```

10. Rebuild docs:

```bash
venv/bin/python -m mkdocs build --strict
```

## Related docs

- [What's New](whats-new.md)
- [CLI Commands](commands.md)
- [Backtest Runtime](model_backtest.md)
- [Changelog](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Migration+Guide)

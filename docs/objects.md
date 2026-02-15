# Fundamental Objects

StackSats now has two fundamental runtime objects:

- `strategy`: user intent object (`BaseStrategy`)
- `StrategyTimeSeries` / `StrategyTimeSeriesBatch`: validated final output objects

## 1) `strategy`

A strategy subclasses `BaseStrategy` (`stacksats/strategy_types.py`) and defines:

- identity: `strategy_id`, `version`, `description`
- hooks: `transform_features`, `build_signals`
- intent path: `propose_weight(...)` or `build_target_profile(...)`

Framework-owned behavior remains sealed:

- compute kernel (`compute_weights` in `BaseStrategy`)
- clipping and remaining-budget enforcement
- lock semantics and final invariants

## 2) `StrategyTimeSeries`

`StrategyTimeSeries` (`stacksats/strategy_time_series.py`) is a single-window output object.

### Required metadata

- `strategy_id`
- `strategy_version`
- `run_id`
- `config_hash`
- `schema_version`
- `generated_at`
- `window_start`
- `window_end`

### Required data columns

- `date`
- `weight`
- `price_usd`

### Optional data columns

- `day_index`
- `locked`

### Core methods

- `schema()`
- `schema_markdown()`
- `validate_schema_coverage()`
- `validate()`
- `to_dataframe()`

### Validation guarantees

- required columns exist
- `date` is valid, unique, sorted ascending
- `weight` is finite, non-negative, sums to `1.0` (tolerance)
- `price_usd` is finite when present (nullable for future rows)
- all columns are covered by handwritten schema specs
- `window_start` / `window_end` match series boundaries

## 3) `StrategyTimeSeriesBatch`

`StrategyTimeSeriesBatch` is a multi-window container returned by export APIs.

### Batch guarantees

- contains one or more `StrategyTimeSeries` windows
- each window has a unique `(window_start, window_end)` key
- per-window provenance matches batch-level provenance

### Core methods

- `from_flat_dataframe(...)`
- `to_dataframe()`
- `iter_windows()`
- `for_window(start_date, end_date)`
- `schema_markdown()`

## Export contract

`StrategyRunner.export(...)` returns `StrategyTimeSeriesBatch`.

Export artifacts remain under:

`<output_dir>/<strategy_id>/<version>/<run_id>/`

and include:

- `weights.csv`
- `timeseries_schema.md`
- `artifacts.json`

Canonical `weights.csv` columns:

- `start_date`
- `end_date`
- `day_index`
- `date`
- `price_usd`
- `weight`

## References

- `stacksats/strategy_types.py`
- `stacksats/strategy_time_series.py`
- `stacksats/runner.py`
- `docs/framework.md`

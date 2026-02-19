---
title: Strategy TimeSeries
description: Metadata, guarantees, and export semantics for StrategyTimeSeries and StrategyTimeSeriesBatch.
---

# Strategy TimeSeries

`StrategyTimeSeries` (`stacksats/strategy_time_series.py`) is the single-window validated output object.

## Required metadata

- `strategy_id`
- `strategy_version`
- `run_id`
- `config_hash`
- `schema_version`
- `generated_at`
- `window_start`
- `window_end`

## Core methods

- `schema()`
- `schema_markdown()`
- `validate_schema_coverage()`
- `validate()`
- `to_dataframe()`

## EDA methods

- `profile()`: returns dataset-level and per-column summary metadata (counts, null rates, dtype, and numeric stats where applicable).
- `weight_diagnostics(top_k=5)`: returns weight concentration/distribution diagnostics including `hhi`, `effective_n`, summary quantiles, and top weighted rows.
- `returns_diagnostics()`: returns return/risk diagnostics derived from `price_usd` (observation counts, cumulative return, mean/std returns, annualized volatility, drawdown, best/worst day).
- `outlier_report(columns=None, method="mad", threshold=None)`: returns a tidy dataframe with `date`, `column`, `value`, `score`, `method`, and `threshold` for detected numeric outliers.

## Validation guarantees

- required columns exist
- `date` is valid, unique, and ascending
- `weight` is finite, non-negative, and sums to `1.0` (tolerance)
- `price_usd` is finite when present
- schema and lineage coverage stays synchronized

## Batch object

`StrategyTimeSeriesBatch` is a multi-window container returned by export APIs.

### Batch guarantees

- contains one or more windows
- unique `(window_start, window_end)` per window
- per-window provenance aligns with batch-level provenance

## Export contract

`StrategyRunner.export(...)` returns `StrategyTimeSeriesBatch`.

Artifacts are written under:

```text
<output_dir>/<strategy_id>/<version>/<run_id>/
```

Includes:

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

## Artifact preview

Example `weights.csv` header and first row:

```csv
start_date,end_date,day_index,date,price_usd,weight
2025-12-01,2026-11-30,0,2025-12-01,96250.12,0.0027397260
```

Example `artifacts.json` (shape):

```json
{
  "strategy_id": "example-mvrv",
  "version": "1.0.0",
  "run_id": "...",
  "files": {
    "weights_csv": "weights.csv",
    "timeseries_schema": "timeseries_schema.md"
  }
}
```

## Schema details

See [Strategy TimeSeries Schema](strategy-timeseries-schema.md) for generated column and lineage tables.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Strategy+TimeSeries)

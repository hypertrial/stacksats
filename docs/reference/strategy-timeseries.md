---
title: Strategy TimeSeries
description: Metadata, guarantees, and export semantics for StrategyTimeSeries and StrategyTimeSeriesBatch.
---

# Strategy TimeSeries

`StrategyTimeSeries` (`stacksats/strategy_time_series.py`) is the single-window validated export object used by `strategy export`.
`StrategyTimeSeriesBatch` is the multi-window container returned by export APIs and artifact loaders.

## Required metadata

Every window carries normalized `StrategySeriesMetadata`:

- `strategy_id`
- `strategy_version`
- `run_id`
- `config_hash`
- `schema_version`
- `generated_at`
- `window_start`
- `window_end`

Metadata invariants:

- required string fields must be non-empty
- `generated_at` is normalized to UTC
- `window_start` / `window_end` are normalized to daily timestamps
- `window_start <= window_end` when both are present

## Read-only object contract

`StrategyTimeSeries` is immutable after construction.

- the internal payload is stored privately
- `data` returns a defensive copy
- `to_dataframe()` returns a deep copy
- analysis and diagnostics methods operate on the validated internal payload, not on copied frames

If you need to transform data, do it outside the object and construct a new `StrategyTimeSeries`.

## Core methods

Single-window object:

- `schema()`
- `schema_markdown()`
- `validate_schema_coverage()`
- `validate()`
- `to_dataframe()`
- `to_csv(path)`
- `from_dataframe(...)`
- `from_csv(...)`
- `columns`
- `row_count`
- `date_index()`
- `window_key()`

Batch object:

- `to_dataframe()`
- `to_csv(path)`
- `from_flat_dataframe(...)`
- `from_csv(...)`
- `from_artifact_dir(...)`
- `schema_markdown()`
- `iter_windows()`
- `for_window(start_date, end_date)`
- `window_keys()`
- `date_span()`

## Validation guarantees

Core guarantees:

- required columns exist
- all columns must be covered by either the core schema or explicit `extra_schema`
- `date` is valid, unique, and ascending
- if both `window_start` and `window_end` exist, `date` must exactly equal the full daily range between them
- `weight` is finite, non-negative, and sums to `1.0` (tolerance)
- `price_usd` is finite when present
- schema and CoinMetrics lineage coverage stay synchronized

Important detail:

- exact daily coverage is enforced only when metadata bounds are present
- if a window intentionally has sparse dates, omit `window_start` and `window_end`

## Extensible schema

The framework schema stays strict by default.
Strategy-specific columns must be declared explicitly with `extra_schema`.

Example:

```python
from stacksats import ColumnSpec, StrategyTimeSeries

extra_schema = (
    ColumnSpec(
        name="custom_signal",
        dtype="float64",
        required=False,
        description="Strategy-owned export score.",
        constraints=("finite when present",),
        source="strategy",
    ),
)

series = StrategyTimeSeries(metadata=metadata, data=df, extra_schema=extra_schema)
```

Rules:

- undeclared extra columns fail validation
- extra-schema names cannot collide with core schema names
- duplicate extra-schema names are rejected

## Batch guarantees

`StrategyTimeSeriesBatch` guarantees:

- one or more windows
- unique `(window_start, window_end)` per window
- one coherent `generated_at` shared by the batch and all windows
- batch-level provenance matches every window
- all windows share the same `extra_schema`

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

## Artifact reconstruction

Rebuild a batch object from export artifacts:

```python
from stacksats import StrategyTimeSeriesBatch

batch = StrategyTimeSeriesBatch.from_artifact_dir("output/example-mvrv/1.0.0/<run_id>")
```

Or from a flattened CSV directly:

```python
batch = StrategyTimeSeriesBatch.from_csv(
    "weights.csv",
    strategy_id="example-mvrv",
    strategy_version="1.0.0",
    run_id="run-123",
    config_hash="abc123",
)
```

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
  "config_hash": "abc123",
  "run_id": "...",
  "files": {
    "weights_csv": "weights.csv",
    "timeseries_schema_md": "timeseries_schema.md"
  }
}
```

## Schema details

See [Strategy TimeSeries Schema](strategy-timeseries-schema.md) for generated core schema and CoinMetrics lineage tables.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Strategy+TimeSeries)

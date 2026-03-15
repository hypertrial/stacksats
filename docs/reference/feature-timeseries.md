---
title: FeatureTimeSeries
description: Validated feature input to a strategy (schema and time-series validation).
---

# FeatureTimeSeries

`FeatureTimeSeries` (`stacksats/feature_time_series.py`) is the **input** object passed into strategy computation. It wraps a Polars DataFrame with a datetime index (stored as a `date` column) and feature columns. It provides schema validation and time-series invariants so that strategies receive validated feature data.

Do not confuse with **WeightTimeSeries**, which is the **output** of a strategy (weights and prices).

## Contract

- **Backing store**: Polars DataFrame with a `date` column (sorted, unique, no nulls).
- **Feature columns**: All other columns are features; required columns can be enforced on construction.
- **Immutability**: The object is immutable after construction; use `.to_dataframe()` for a Polars copy if you need to transform.

## Construction

Build from a Polars DataFrame (for example from the feature registry or precomputed features):

```python
import polars as pl
from stacksats import FeatureTimeSeries

# Polars DataFrame with 'date' column (Datetime type)
pl_df = pl.DataFrame({
    "date": pl.Series(["2024-01-01", "2024-01-02"]).str.to_datetime(),
    "price_usd": [100, 101],
    "mvrv": [1.0, 1.1],
})
fts = FeatureTimeSeries.from_dataframe(
    pl_df,
    required_columns=("price_usd", "mvrv"),
    as_of_date="2024-01-02",  # optional: no data after this date
)

```

Parameters:

- `required_columns`: Column names that must be present.
- `as_of_date`: If set, validates that no row has date after this date (no forward-looking data).
- `require_finite`: Optional tuple of column names that must be finite numeric.

## Conversion

- **To DataFrame**: `fts.to_dataframe()` returns a Polars DataFrame with the canonical `date` column preserved. Strategy hooks are Polars-only and should use `ctx.features_df`.

## StrategyContext

`StrategyContext.features` is a `FeatureTimeSeries`. The preferred way to build a context from a feature DataFrame is **`StrategyContext.from_features_df(...)`**, which accepts a Polars DataFrame, wraps it in a `FeatureTimeSeries`, and returns a full context. The runner uses this (or the helper `strategy_context_from_features_df`, which delegates to it) so that strategies never see future data when the framework enforces it.

## Schema and validation

- **Date column**: Must exist, be sorted ascending, unique, and have no nulls.
- **Required columns**: Validated when provided to `from_dataframe(..., required_columns=...)`.
- **No forward-looking**: When `as_of_date` is set, `FeatureTimeSeries.from_dataframe` raises if any row has date &gt; `as_of_date`.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+FeatureTimeSeries)

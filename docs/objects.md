---
title: Runtime Objects Overview
description: Overview of strategy, FeatureTimeSeries, and WeightTimeSeries runtime object model.
---

# Runtime Objects Overview

StackSats has three fundamental runtime objects:

- **FeatureTimeSeries**: validated input to a strategy (Polars-backed feature time series with schema and time-series validation).
- **BaseStrategy**: user intent and logic (hooks, identity, allocation intent).
- **WeightTimeSeries** / **WeightTimeSeriesBatch**: validated output of a strategy (weights, prices, metadata; framework invariants per [Framework](framework.md)).

!!! tip "Input → Strategy → Output"
    **FeatureTimeSeries** is the input (features, schema, no forward-looking). **BaseStrategy** consumes it and produces allocation intent. **WeightTimeSeries** is the output object that enforces budget, daily bounds, and validation guards.

## Fundamental objects

| Object | Role | Validation |
|--------|------|------------|
| **FeatureTimeSeries** | Input to strategy | Schema (required columns), sorted unique dates, optional no-future-data and finite-numeric checks. |
| **BaseStrategy** | Strategy logic | User-defined hooks; framework runs the allocation kernel. |
| **WeightTimeSeries** / **WeightTimeSeriesBatch** | Output of strategy | Framework invariants: weight sum = 1, min/max daily weight, no forward-looking at export, NaN/inf/range guards. See [Framework](framework.md). |

## Read in this order

1. [Strategy Object](reference/strategy-object.md)
2. [FeatureTimeSeries](reference/feature-timeseries.md)
3. [WeightTimeSeries](reference/strategy-timeseries.md)
4. [WeightTimeSeries Schema](reference/strategy-timeseries-schema.md)

## Why this split exists

- **FeatureTimeSeries** defines validated feature input (schema and time-series checks).
- **BaseStrategy** defines user-owned hooks and metadata.
- **WeightTimeSeries** defines framework-validated outputs and export contracts.
- Schema docs are generated from code and kept synchronized in CI.

## Comprehensive list of library objects

All public types and functions below are exported from the top-level `stacksats` package (see `stacksats/__init__.py`).

### Primary objects

| Object | Description |
|--------|-------------|
| `FeatureTimeSeries` | Validated input to strategy: Polars DataFrame with datetime index; feature columns; schema and time-series validation. |
| `BaseStrategy` | Abstract base class for defining strategy logic (hooks, identity, intent). |
| `WeightTimeSeries` | Single-window validated strategy output (weights, prices, metadata). |
| `WeightTimeSeriesBatch` | Multi-window container of `WeightTimeSeries` (e.g. from export or backtest). |

### Strategy input and context

| Object | Description |
|--------|-------------|
| `StrategyContext` | Input passed into strategy computation: `features` (`FeatureTimeSeries`), date range, `current_date`, optional `locked_weights`, column names. Build from a pandas DataFrame with **`StrategyContext.from_features_df(...)`**. |

### Results

| Object | Description |
|--------|-------------|
| `BacktestResult` | Result of a backtest run. |
| `ValidationResult` | Result of strategy validation. |
| `DailyRunResult` | Result of a single daily run. |
| `DailyOrderRequest` | Request for a daily order. |
| `DailyOrderReceipt` | Receipt for a daily order. |
| `StrategyRunResult` | Result of a strategy run. |

### Config

| Object | Description |
|--------|-------------|
| `BacktestConfig` | Backtest options (dates, labels, etc.). |
| `ExportConfig` | Export options for strategy outputs. |
| `RunDailyConfig` | Config for daily run. |
| `ValidationConfig` | Config for validation. |

### Strategy metadata and contract

| Object | Description |
|--------|-------------|
| `StrategyMetadata` | Strategy identity (id, version, description). |
| `StrategySpec` | Full public contract (metadata, intent_mode, params, required features). |
| `StrategySeriesMetadata` | Metadata on a `WeightTimeSeries` (strategy_id, version, run_id, window, etc.). |
| `StrategyArtifactSet` | Set of artifacts produced by a strategy. |
| `StrategyContractWarning` | Warning type for contract violations. |

### Allocation and intent

| Object | Description |
|--------|-------------|
| `TargetProfile` | Target allocation profile returned by `build_target_profile`. |
| `DayState` | Per-day state used in allocation. |

### WeightTimeSeries schema

| Object | Description |
|--------|-------------|
| `ColumnSpec` | Spec for a column in the WeightTimeSeries schema. |

### Data and loading

| Object | Description |
|--------|-------------|
| `ColumnMapDataProvider` | Data provider that wraps a DataFrame and column map. |
| `ColumnMapError` | Error raised when column mapping fails. |

### Example strategies

| Object | Description |
|--------|-------------|
| `ExampleMVRVStrategy` | Example MVRV strategy. |
| `MVRVStrategy` | MVRV strategy. |
| `MVRVPlusStrategy` | MVRV+ strategy. |

### Runners and helpers

| Object | Description |
|--------|-------------|
| `StrategyRunner` | Orchestrates backtest, export, run, and validation. |
| `load_strategy` | Load a strategy (e.g. from JSON). |
| `load_data` | Load data from BRK parquet (prelude helper). Use `ColumnMapDataProvider` or `StrategyRunner.from_dataframe` for custom DataFrames. |
| `precompute_features` | Precompute features for a strategy. |

### Deprecated aliases

| Alias | Use instead | Notes |
|-------|-------------|-------|
| `TimeSeries` | `WeightTimeSeries` | Deprecated; removed in 0.9.0. |
| `TimeSeriesBatch` | `WeightTimeSeriesBatch` | Deprecated; removed in 0.9.0. |
| `StrategyTimeSeries` | `WeightTimeSeries` | Deprecated; removed in 0.9.0. |
| `StrategyTimeSeriesBatch` | `WeightTimeSeriesBatch` | Deprecated; removed in 0.9.0. |

## API pointers

- [API Index](reference/api/index.md)
- [strategy_types module](reference/api/strategy-types.md)
- [runner module](reference/api/runner.md)

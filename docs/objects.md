---
title: Runtime Objects Overview
description: Overview of strategy and TimeSeries runtime object model.
---

# Runtime Objects Overview

StackSats has two core runtime object families:

- `strategy`: user intent object (`BaseStrategy`)
- `TimeSeries` / `TimeSeriesBatch`: validated output objects

!!! tip "The Two-Object Model"
    StackSats enforces a strict separation between **identity** (the strategy you define) and **outcome** (the time-series results). Strategies are for research and logic; TimeSeries are for validation, backtesting, and production execution.

## Read in this order

1. [Strategy Object](reference/strategy-object.md)
2. [Strategy TimeSeries](reference/strategy-timeseries.md)
3. [Strategy TimeSeries Schema](reference/strategy-timeseries-schema.md)

## Why this split exists

- Strategy objects define user-owned hooks and metadata.
- TimeSeries objects define framework-validated outputs and export contracts.
- Schema docs are generated from code and kept synchronized in CI.

## Comprehensive list of library objects

All public types and functions below are exported from the top-level `stacksats` package (see `stacksats/__init__.py`).

### Primary objects

| Object | Description |
|--------|-------------|
| `BaseStrategy` | Abstract base class for defining strategy logic (hooks, identity, intent). |
| `TimeSeries` | Single-window validated strategy output (weights, prices, metadata). |
| `TimeSeriesBatch` | Multi-window container of `TimeSeries` (e.g. from export or backtest). |

### Strategy input and context

| Object | Description |
|--------|-------------|
| `StrategyContext` | Input passed into strategy computation: `features_df`, date range, `current_date`, optional `locked_weights`, column names. |

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
| `StrategySeriesMetadata` | Metadata on a `TimeSeries` (strategy_id, version, run_id, window, etc.). |
| `StrategyArtifactSet` | Set of artifacts produced by a strategy. |
| `StrategyContractWarning` | Warning type for contract violations. |

### Allocation and intent

| Object | Description |
|--------|-------------|
| `TargetProfile` | Target allocation profile returned by `build_target_profile`. |
| `DayState` | Per-day state used in allocation. |

### TimeSeries schema

| Object | Description |
|--------|-------------|
| `ColumnSpec` | Spec for a column in the TimeSeries schema. |

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
| `StrategyTimeSeries` | `TimeSeries` | Deprecated; removed in 0.9.0. |
| `StrategyTimeSeriesBatch` | `TimeSeriesBatch` | Deprecated; removed in 0.9.0. |

## API pointers

- [API Index](reference/api/index.md)
- [strategy_types module](reference/api/strategy-types.md)
- [runner module](reference/api/runner.md)

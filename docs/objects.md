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

## Stable public API

For the supported `1.x` import surface, CLI subset, and artifact contracts, use [Public API](reference/public-api.md).
This page stays focused on the runtime object model rather than acting as the full stable API index.

## API pointers

- [Public API](reference/public-api.md)
- [API Index](reference/api/index.md)
- [strategy_types module](reference/api/strategy-types.md)
- [runner module](reference/api/runner.md)

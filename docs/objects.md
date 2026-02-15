---
title: Runtime Objects Overview
description: Overview of strategy and StrategyTimeSeries runtime object model.
---

# Runtime Objects Overview

StackSats has two core runtime object families:

- `strategy`: user intent object (`BaseStrategy`)
- `StrategyTimeSeries` / `StrategyTimeSeriesBatch`: validated output objects

## Read in this order

1. [Strategy Object](reference/strategy-object.md)
2. [Strategy TimeSeries](reference/strategy-timeseries.md)
3. [Strategy TimeSeries Schema](reference/strategy-timeseries-schema.md)

## Why this split exists

- Strategy objects define user-owned hooks and metadata.
- TimeSeries objects define framework-validated outputs and export contracts.
- Schema docs are generated from code and kept synchronized in CI.

## API pointers

- [API Index](reference/api/index.md)
- [strategy_types module](reference/api/strategy-types.md)
- [runner module](reference/api/runner.md)

---
title: Welcome
description: Entry point for StackSats docs, quick navigation, and recommended learning path.
---

# StackSats Documentation

<div class="hero-block">
  <p class="hero-kicker">Strategy-first Bitcoin DCA toolkit</p>
  <h2>Build, validate, and emit agent-consumable Bitcoin dollar cost averaging (DCA) decisions with a sealed allocation framework.</h2>
  <p>
    StackSats, developed by Hypertrial, is a Python package for strategy-first Bitcoin dollar cost averaging (DCA) research and execution.
    Learn more at <a href="https://www.stackingsats.org">www.stackingsats.org</a>.
  </p>
</div>

## Start in 2 Clicks

Use this page when you want the fastest path into the hosted docs and need to choose the right next workflow.

Canonical first run:

1. `pip install stacksats`
2. `stacksats demo backtest`
3. Inspect `output/<strategy_id>/<version>/<run_id>/`

Next steps:

- [Quickstart](start/quickstart.md) for the offline packaged demo
- [Full Data Setup](start/full-data-setup.md) for the canonical BRK dataset
- [Task Hub](tasks.md) for task-first workflows
- [First Strategy Run](start/first-strategy-run.md) for custom strategy authoring
- [Public API](reference/public-api.md) for the stable `1.x` library surface

## Agent-Native Flow

The primary production flow is:

1. StackSats computes a validated daily BTC accumulation decision.
2. An external AI agent reads the structured decision payload.
3. Brokerage-specific execution happens outside StackSats.

Use [Task Hub](tasks.md) for the shortest path into `decide-daily` or the Python API.

## Canonical Dataset

StackSats is anchored on the canonical BRK `merged_metrics*.parquet` dataset.
Current coverage and scale are documented in the dataset-specific reference pages:

- [Merged Metrics Data Guide](reference/merged-metrics-data-guide.md)
- [BRK Data Source](data-source.md)
- [EDA Quickstart](start/eda-quickstart.md)
- [Merged Metrics Parquet Schema](reference/merged-metrics-parquet-schema.md)
- [Merged Metrics Taxonomy](reference/merged-metrics-taxonomy.md)

## Core Objects

StackSats is built around three fundamental runtime objects:

- **[FeatureTimeSeries](reference/feature-timeseries.md)**: Validated input to a strategy (feature time series with schema and time-series validation).
- **[Strategy](reference/strategy-object.md)**: User-defined logic for feature engineering, signals, and allocation intent.
- **[WeightTimeSeries](reference/strategy-timeseries.md)**: Framework-validated output data containing normalized weights and prices.

---

## Choose Your Path

<div class="grid cards" markdown>

-   :material-account-school: __I'm New to StackSats__

    ---

    Start with the offline packaged demo and expected outputs.

    [Quickstart](start/quickstart.md)

-   :material-format-list-checks: __I Know the Outcome I Want__

    ---

    Jump directly to the workflow you need.

    [Task Hub](tasks.md)

-   :material-code-braces: __I'm Building a Strategy__

    ---

    Build custom hooks and run validate/backtest/export.

    [First Strategy Run](start/first-strategy-run.md)

-   :material-database: __I Need Full BRK Data__

    ---

    Fetch canonical source data and prepare the managed runtime parquet.

    [Full Data Setup](start/full-data-setup.md)

</div>

## Core Concepts

- [Framework Boundary](framework.md) explains framework-owned invariants versus user-owned strategy logic.
- [Strategies](reference/strategies.md) documents stable built-ins and experimental reference strategies.
- [Command Index](commands.md) is the canonical CLI reference for the stable `stacksats` command groups.
- [Migration Guide](migration.md) maps old names and pre-v1 paths to the current `1.x` contract.

## Maintainer Links

- [What's New](whats-new.md)
- [Release Guide](release.md)
- [Docs Ownership](docs_ownership.md)

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Docs+Home)

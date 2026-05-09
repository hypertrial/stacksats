---
title: Welcome
description: Python library for quantitative Bitcoin DCA accumulation—docs entry, navigation, and learning path.
---

# StackSats Documentation

<div class="hero-block">
  <p class="hero-kicker">Python library for dynamic Bitcoin DCA models</p>
  <h2>Build dynamic Bitcoin DCA models that try to outperform uniform DCA under fixed-budget, fixed-horizon, no-forward-looking-data constraints.</h2>
  <p>
    StackSats, developed by Hypertrial, turns research signals and feature pipelines into validated BTC weight schedules, backtests them against uniform DCA, and emits operational daily decisions.
    It is not a brokerage or generic trading bot.
    Learn more at <a href="https://www.stackingsats.org">www.stackingsats.org</a>.
  </p>
</div>

## The Stacking Sats Problem

Given a fixed Bitcoin accumulation budget and an allocation horizon longer than six months, can a dynamic dollar-cost averaging (DCA) model acquire more BTC than uniform DCA without using future data?

Uniform DCA means equal daily allocation across the same budget and horizon. StackSats measures the dynamic model's edge as sats per dollar (SPD): more BTC acquired for the same dollars, not short-term USD ROI or CAGR. The framework keeps that comparison honest by enforcing fixed-budget allocation, a fixed horizon that defaults to 365 days, no forward-looking data, locked historical allocations, and a clean boundary where brokerage execution stays outside the library.

See [Framework Boundary](framework.md) for the allocation contract and [Backtest Runtime](model_backtest.md) for how strategy-vs-uniform outcomes are scored.

## Start in 2 Clicks

Use this page when you want the fastest path into the hosted docs and need to choose the right next workflow.

Canonical first run:

1. `pip install stacksats`
2. `stacksats demo backtest`
3. Inspect `output/<strategy_id>/<version>/<run_id>/`

Next steps:

- [Quickstart](start/quickstart.md) for the offline packaged demo
- [System Overview](start/system-overview.md) for data flow and production paths
- [Full Data Setup](start/full-data-setup.md) for the canonical BRK dataset
- [Task Hub](tasks.md) for task-first workflows (including [compare strategies on one window](tasks.md#i-want-to-compare-strategies-against-benchmarks))
- [Troubleshooting](troubleshooting.md) for symptom-based links
- [First Strategy Run](start/first-strategy-run.md) for custom strategy authoring
- [Public API](reference/public-api.md) for the stable `1.x` library surface

## What's new

- [What's New](whats-new.md) for release highlights
- [Changelog on GitHub](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md) for full version history

## Agent-Native Flow

The primary production flow is:

1. StackSats computes a validated daily BTC accumulation decision.
2. An external AI agent reads the structured decision payload.
3. Brokerage-specific execution happens outside StackSats.

Use [Task Hub](tasks.md) for the shortest path into `decide-daily` or the Python API.

## Canonical Dataset

StackSats is anchored on the canonical Bitcoin Research Kit (BRK) `merged_metrics*.parquet` dataset. StackSats supports BRK at the project and data-workflow level and documents BRK as the upstream project. See [bitcoinresearchkit/brk](https://github.com/bitcoinresearchkit/brk), [`brk` on crates.io](https://crates.io/crates/brk), and [`brk` on docs.rs](https://docs.rs/crate/brk/latest).

This does not mean StackSats embeds the Rust BRK crates or promises crate-level API compatibility. StackSats remains a Python package that consumes documented BRK-derived data artifacts.

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

- [System Overview](start/system-overview.md) sketches BRK data through strategy hooks to weights and artifacts.
- [Framework Boundary](framework.md) explains framework-owned invariants versus user-owned strategy logic.
- [Strategies](reference/strategies.md) documents stable built-ins and experimental reference strategies.
- [Command Index](commands.md) is the canonical CLI reference for the stable `stacksats` command groups.
- [Migration Guide](migration.md) maps old names and pre-v1 paths to the current `1.x` contract.

## Maintainer Links

- [Release Guide](release.md)
- [Docs Ownership](docs_ownership.md)

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Docs+Home)

---
title: Welcome
description: Entry point for StackSats docs, quick navigation, and recommended learning path.
---

# StackSats Documentation

<div class="hero-block">
  <p class="hero-kicker">Strategy-first Bitcoin DCA toolkit</p>
  <h2>Build, validate, and backtest Bitcoin dollar cost averaging (DCA) strategies with a sealed allocation framework.</h2>
  <p>
    StackSats, developed by Hypertrial, is a Python package for strategy-first Bitcoin dollar cost averaging (DCA) research and execution.
    Learn more at <a href="https://www.stackingsats.org">www.stackingsats.org</a>.
  </p>
</div>

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

    Start with install, first run, and expected outputs.

    [Quickstart](start/quickstart.md)

-   :material-code-braces: __I'm Building a Strategy__

    ---

    Build custom hooks and run validate/backtest/export.

    [First Strategy Run](start/first-strategy-run.md)

-   :material-code-json: __I Need Minimal Code Templates__

    ---

    Copy minimal strategy examples for both supported hook styles.

    [Minimal Strategy Examples](start/minimal-strategy-examples.md)

-   :material-console: __I Need Commands Fast__

    ---

    Use the command index and dedicated command reference pages.

    [CLI Commands](commands.md)

-   :material-format-list-checks: __I Want Task-Based Guidance__

    ---

    Jump directly to workflow outcomes and next steps.

    [Task Hub](tasks.md)

-   :material-progress-wrench: __I'm Upgrading Versions__

    ---

    Apply breaking-change mappings quickly.

    [Migration Guide](migration.md)

-   :material-help-circle: __I Have Questions__

    ---

    Read frequently asked questions sourced from docs feedback.

    [FAQ](faq.md)

-   :material-shield-check: __I Need Contract Details__

    ---

    Understand framework-owned invariants and user-owned hooks.

    [Framework Boundary](framework.md)

</div>

## Start in 2 Clicks

1. New user path: [Quickstart](start/quickstart.md) -> [First Strategy Run](start/first-strategy-run.md)
2. Runner path: [Task Hub](tasks.md) -> [Command Index](commands.md)
3. Builder path: [Create a Strategy](recipes/create-strategy.md) -> [Framework Boundary](framework.md)
4. Maintainer path: [Release Guide](release.md) -> [Docs Ownership](docs_ownership.md)

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Docs+Home)

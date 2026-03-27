---
title: Stability Policy
description: Supported platforms, public API boundary, and deprecation rules for StackSats.
---

# Stability Policy

StackSats uses SemVer for its documented stable surface.

## Official support

- Python: `3.11` and `3.12`
- Operating systems: Linux and macOS
  - `release-gate.yml` enforces Linux full-suite validation and a macOS supported-platform smoke lane before release tags are publishable.
- Windows: best-effort only until dedicated CI coverage is added

## Stable public API

The stable `1.x` contract is intentionally narrow:

- top-level `stacksats` exports
- documented artifact payloads emitted by the stable CLI flows
- documented CLI commands:
  - `stacksats demo validate|backtest|export`
  - `stacksats data fetch|prepare|doctor`
  - `stacksats strategy validate|backtest|export|run-daily|animate`
- Optional helper console scripts such as `stacksats-plot-mvrv` and `stacksats-plot-weights` are documented convenience tools, but they are outside the frozen stable CLI subset.

Lower-level modules are allowed to change between releases unless they are re-exported from top-level `stacksats`.

## Internal and experimental surfaces

- Internal reference modules include generated API pages such as `stacksats.runner`, `stacksats.strategy_types`, `stacksats.eda`, `stacksats.backtest`, and `stacksats.export_weights`.
- Experimental/reference strategies live under `stacksats.strategies.experimental.*`.
- Experimental surfaces may change without SemVer stability guarantees.

## Built-in strategy tiers

Stable supported built-ins:

- `UniformStrategy`
- `SimpleZScoreStrategy`
- `MomentumStrategy`
- `MVRVStrategy`

Experimental reference strategies:

- `ExampleMVRVStrategy`
- `MVRVPlusStrategy`

## Deprecation policy

After `1.0.0`, any incompatible change to the stable surface requires:

1. Documentation in the changelog and migration guide.
2. An explicit deprecation note in the docs for the affected stable surface.
3. At least one minor release of overlap before removal, unless a security or correctness issue requires an immediate break.

## Optional dependency policy

- Base install: stable core library and non-visual CLI flows.
- `viz`: plotting and animation commands.
- `network`: HTTP-backed helper modules such as BTC price fetching helpers.
- `deploy`: database/export integrations.
- `stacksats-plot-mvrv` is an optional helper script that requires `viz`.
- `stacksats-plot-weights` is an optional helper script that spans both `deploy` (DB access) and `viz` (rendering).

Install only the extras needed for your workflow.

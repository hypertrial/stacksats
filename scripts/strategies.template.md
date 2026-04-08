---
title: Strategies
description: Canonical reference for built-in StackSats strategies, contracts, and execution commands.
---

# Strategies

This is the canonical strategy catalog for StackSats built-ins. The published page is generated from the in-repo strategy catalog.

Built-in support tier is defined by catalog metadata, not by the implementation module path. For the maintainer workflow that adds new cataloged strategies, use [Add a Built-in Strategy](../maintainers/add-built-in-strategy.md).

Use this page for:

- built-in behavior and intent mode
- model card links, owners, and promotion stage
- required feature sets/columns
- exposed tuning parameters and defaults
- canonical validate/backtest/audit commands
- reasonableness expectations when interpreting model results
- support tier for stable versus experimental reference strategies

Model cards for all cataloged built-ins:

{{model_card_links}}

## Purpose and Scope

Framework boundary (canonical):

- User strategy code owns feature transforms, signal formulas, and allocation intent.
- Framework runtime owns feature sourcing/materialization, window iteration, clipping, lock semantics, and final invariants.

See [Framework Boundary](../framework.md) for the full contract.

## Global Strategy Contract

All strategy hooks are Polars-only:

- `transform_features(...) -> pl.DataFrame`
- `build_signals(...) -> dict[str, pl.Series]`
- `propose_weight(state) -> float` or `build_target_profile(...) -> TargetProfile | pl.DataFrame`

Data shape contract:

- strategy-facing frames use canonical `date` columns (no index semantics)
- `TargetProfile.values` payload must be `date` + `value`
- runtime rejects ambiguous dual-hook strategies unless `intent_preference` is set explicitly

## Canonical Dataset and Scoring Window

Dataset flow:

- Canonical source dataset is long-format `merged_metrics*.parquet` (`day_utc`, `metric`, `value`)
- Runtime commands consume a derived BRK-wide parquet via `STACKSATS_ANALYTICS_PARQUET`, managed default `~/.stacksats/data/bitcoin_analytics.parquet`, or legacy local fallback `./bitcoin_analytics.parquet`
- Canonical schema: [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md)
- Projection workflow: [BRK Data Source](../data-source.md)

Scoring defaults used by runtime and strategy audits:

- default backtest start: `2018-01-01`
- default backtest end: `2025-12-31` (or earlier if data coverage ends sooner)
- feature warmup history is included by default; scored windows still begin at the requested `start_date`

## Stable Supported Built-ins

These strategies are part of the stable `1.x` contract because their catalog entries are marked `tier=stable`. Stable built-in strategy IDs: {{stable_strategy_ids}}.

{{stable_table}}

## Experimental Reference Strategies

These strategies are cataloged as `tier=experimental` and are not part of the stable `1.x` contract. Their current implementation modules may live under `stacksats.strategies.experimental.*`, but support status comes from the catalog entry. Experimental built-in strategy IDs: {{experimental_strategy_ids}}.

{{experimental_table}}

Built-in intent is explicit by implementation shape:

{{stable_intent_bullets}}

## How to Run (Parquet-First)

Fetch and prepare the managed runtime parquet:

```bash
stacksats data fetch
stacksats data prepare
```

Validate a built-in strategy on fixed comparison bounds:

```bash
stacksats strategy validate \
  --strategy mvrv \
  --start-date 2018-01-01 \
  --end-date 2025-12-31
```

Run a backtest on the same canonical window:

```bash
stacksats strategy backtest \
  --strategy mvrv \
  --start-date 2018-01-01 \
  --end-date 2025-12-31 \
  --output-dir output
```

Run the built-in strategy audit matrix (uses local workspace code and emits `output/strategy_audit.json`):

```bash
python scripts/run_all_strategies.py
```

Compare candidates against baselines on a shared window (uses `strategy_id` for built-ins and `module_or_path:ClassName` for customs). Prefer the stable CLI; full flags and `comparison_result.json` are documented on [Compare Command](../run/compare.md).

```bash
stacksats strategy compare \
  --strategy my_strategy.py:MyStrategy \
  --strategy simple-zscore \
  --strategy mvrv \
  --baseline uniform \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --strict
```

From a repository checkout you can also run `python scripts/compare_strategies.py` with the same flags; it delegates to `StrategyRunner.compare()`.

Profile hot paths for strategy runtime:

```bash
python scripts/profile_strategy_hotpaths.py
```

Generate an execution-ready daily decision payload for an external agent:

```bash
stacksats strategy decide-daily \
  --strategy run-daily-paper \
  --total-window-budget-usd 1000
```

## Metric Interpretation Guide

Primary metrics:

- `win_rate`: percent of windows where strategy percentile beats uniform percentile
- `exp_decay_percentile`: recency-weighted strategy percentile
- `uniform_exp_decay_percentile`: recency-weighted baseline percentile
- `exp_decay_multiple_vs_uniform`: ratio of dynamic vs uniform recency-weighted percentile
- `score`: blended headline metric (`0.5 * win_rate + 0.5 * exp_decay_percentile`)

Reasonableness expectations by strategy family:

- `uniform`: baseline multiple should stay near `1.0`; material drift indicates a bug.
- `run-daily-paper`: treat as a daily-execution harness, not a benchmark; expect near-uniform behavior and use it to verify paper-flow correctness.
- `simple-zscore`, `momentum`: acceptable if finite/stable and non-pathological, even when close to baseline.
- `mvrv`: stable baseline model; should produce distinct, non-baseline behavior on representative windows.
- experimental overlay models: treat outputs as reference or research signals, not as stable benchmark promises.

Comparability discipline:

- compare strategies on the same start/end window
- keep allocation span and strictness settings fixed
- avoid comparing fixed-end runs with extended-to-latest runs without calling out recency effects
- treat all example interpretations as illustrative unless explicitly dated; regenerate current metrics from local parquet before decisions

## Troubleshooting

### Missing columns or provider data errors

- Confirm strategy `required_feature_sets()` and `required_feature_columns()` are satisfied.
- Verify runtime parquet was derived from canonical merged metrics with required BRK columns.
- For overlay models, ensure required overlay metrics exist in the runtime parquet.

### No backtest windows generated

- Ensure `end_date >= start_date`.
- Ensure date span is long enough for 365-day window generation.
- Confirm runtime parquet covers the full requested range.

### Forward leakage failures

- Ensure strategy logic uses observed-only inputs.
- Remove any dependence on future rows, centered/forward-looking transforms, or direct external I/O inside strategy hooks.
- Remember that causal lint is best-effort static analysis, not a runtime sandbox.
- Re-run strict validation to inspect leakage diagnostics.

### Warmup and horizon drift

- Keep explicit bounds (`2018-01-01` to `2025-12-31`) for stable model comparisons.
- Understand that extending end date changes recency-weighted metrics materially.
- Warmup history can shift early-window features; this is expected and should be held constant across comparisons.

## Extending with New Strategies

When adding a custom strategy, keep this contract:

1. Subclass `BaseStrategy` and provide stable metadata (`strategy_id`, `version`, `description`).
2. Declare provider-backed `required_feature_sets()` and hard-required transformed columns.
3. Keep hooks Polars-native and return canonical shapes.
4. If implementing both intent hooks, set `intent_preference` explicitly.
5. Keep external data access out of strategy methods; use framework providers.
6. Validate and backtest on explicit date bounds before claiming model deltas.

Selector rules:

- built-in strategies use `strategy_id`
- custom strategies use `module_or_path:ClassName`
- service registry built-ins use `catalog_strategy_id`

Related references:

- [Strategy Object](strategy-object.md)
- [Minimal Strategy Examples](../start/minimal-strategy-examples.md)
- [Model Development Helpers](../concepts/model-development-helpers.md)
- [Backtest Runtime](../model_backtest.md)
- [Validation Checklist](../validation_checklist.md)
- [Add a Built-in Strategy](../maintainers/add-built-in-strategy.md)

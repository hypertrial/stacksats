---
title: Strategies
description: Canonical reference for built-in StackSats strategies, contracts, and execution commands.
---

# Strategies

This is the canonical strategy catalog for StackSats built-ins.

Use this page for:

- built-in behavior and intent mode
- required feature sets/columns
- exposed tuning parameters and defaults
- canonical validate/backtest/audit commands
- reasonableness expectations when interpreting model results

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

## Built-in Strategy Catalog

| Strategy | Import spec | Intent mode | Required feature sets | Required columns | Configurable params (defaults) | Expected behavior | Common failure modes | Use when |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `UniformStrategy` | `stacksats.strategies.examples:UniformStrategy` | `propose` | `core_model_features_v1` | none | none | Uniform baseline allocation across each window. | Invalid/missing runtime data window, or date bounds too short for 365-day windows. | Baseline sanity checks and normalization reference. |
| `SimpleZScoreStrategy` | `stacksats.strategies.examples:SimpleZScoreStrategy` | `profile` | `core_model_features_v1` | none | none | Preference tilts toward lower `mvrv_zscore`; intended as a simple toy comparator. | Missing `mvrv_zscore` can collapse behavior near uniform-like outputs. | Quick toy benchmark for profile-hook flow and contract checks. |
| `MomentumStrategy` | `stacksats.strategies.examples:MomentumStrategy` | `profile` | `core_model_features_v1` | `price_usd` | none | Contrarian tilt from 30-day momentum (`pct_change(30)`), clipped and converted to preference scores. | Missing `price_usd`, sparse data windows, or very short ranges produce weak/flat signals. | Toy momentum baseline and sensitivity checks. |
| `MVRVStrategy` | `stacksats.strategies.mvrv:MVRVStrategy` | `profile` | `core_model_features_v1` | `price_vs_ma`, `mvrv_zscore`, `mvrv_gradient`, `mvrv_percentile`, `mvrv_acceleration`, `mvrv_zone`, `mvrv_volatility`, `signal_confidence` | none | Core package MVRV/MA preference model via `compute_preference_scores(...)`. | Missing transformed feature columns from provider/materialization path. | Canonical production-style baseline model. |
| `ExampleMVRVStrategy` | `stacksats.strategies.model_example:ExampleMVRVStrategy` | `profile` | `core_model_features_v1`, `brk_overlay_v1` | core MVRV feature set + BRK overlays (`brk_netflow_*`, activity/liquidity/exchange/miner/ROI columns) | `base_temperature=0.58`, `overlay_scale=0.75`, plus overlay component weights (`netflow_weight`, `activity_weight`, `liquidity_weight`, etc.) | Score-focused MVRV model with multi-horizon BRK overlays and interaction terms. | Missing overlay columns, non-finite upstream data, or horizon drift causing recency-weighted score swings. | Rich overlay experimentation and advanced profile-shape tuning. |
| `MVRVPlusStrategy` | `stacksats.strategies.model_mvrv_plus:MVRVPlusStrategy` | `profile` | `core_model_features_v1`, `brk_overlay_v1` | `price_usd` + core MVRV feature set + overlay columns (`brk_netflow`, `brk_exchange_share`, `brk_activity_div`, `brk_roi_context`, etc.) | `deep_value_boost=1.18`, `overheat_dampen=0.62`, `overlay_scale=0.20`, `smooth_alpha=0.10`, `risk_budget_base/min/max=1.02/0.72/1.22`, overlay weights for miner/hash components | MVRV baseline with regime/risk/disagreement gating and adaptive smoothing. | Missing overlay inputs, insufficient history for derived vol/drawdown features, recency-heavy end-range sensitivity. | Primary advanced built-in model for BRK-aware runtime behavior. |

Built-in intent is explicit by implementation shape:

- `UniformStrategy` uses `propose_weight(state)`.
- all other built-ins use `build_target_profile(...)`.
- no built-in currently requires dual-hook `intent_preference`.

## How to Run (Parquet-First)

Fetch and prepare the managed runtime parquet:

```bash
stacksats data fetch
stacksats data prepare
```

Validate a built-in strategy on fixed comparison bounds:

```bash
stacksats strategy validate \
  --strategy stacksats.strategies.model_mvrv_plus:MVRVPlusStrategy \
  --start-date 2018-01-01 \
  --end-date 2025-12-31
```

Run a backtest on the same canonical window:

```bash
stacksats strategy backtest \
  --strategy stacksats.strategies.model_mvrv_plus:MVRVPlusStrategy \
  --start-date 2018-01-01 \
  --end-date 2025-12-31 \
  --output-dir output
```

Run the built-in strategy audit matrix (uses local workspace code and emits `output/strategy_audit.json`):

```bash
python scripts/run_all_strategies.py
```

Profile hot paths for strategy runtime:

```bash
python scripts/profile_strategy_hotpaths.py
```

## Metric Interpretation Guide

Primary metrics:

- `win_rate`: percent of windows where strategy percentile beats uniform percentile
- `exp_decay_percentile`: recency-weighted strategy percentile
- `uniform_exp_decay_percentile`: recency-weighted baseline percentile
- `exp_decay_multiple_vs_uniform`: ratio of dynamic vs uniform recency-weighted percentile
- `score`: blended headline metric (`0.5 * win_rate + 0.5 * exp_decay_percentile`)

Reasonableness expectations by strategy family:

- `UniformStrategy`: baseline multiple should stay near `1.0`; material drift indicates a bug.
- `SimpleZScoreStrategy`, `MomentumStrategy`: acceptable if finite/stable and non-pathological, even when close to baseline.
- `MVRVStrategy`, `ExampleMVRVStrategy`, `MVRVPlusStrategy`: should produce distinct, non-baseline behavior on representative windows; near-identical baseline metrics require investigation.

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
- Remove any dependence on future rows, file I/O, DB/network access, or centered/forward-looking transforms.
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

Related references:

- [Strategy Object](strategy-object.md)
- [Minimal Strategy Examples](../start/minimal-strategy-examples.md)
- [Backtest Runtime](../model_backtest.md)
- [Validation Checklist](../validation_checklist.md)

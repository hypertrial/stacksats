---
title: Migration Guide
description: Breaking-change migration mappings for StackSats runtime and export APIs.
---

# Migration Guide

Use this guide when upgrading code that depended on removed compatibility helpers or older signatures.

## Scope

This page covers migration for:

- backtest helper removals
- runtime constant removals
- export/date-range signature changes
- compatibility API removals
- strict source-only loader behavior
- strategy contract hardening for metadata, params, and intent selection

## Old -> New Mapping

| Old usage | New usage |
| --- | --- |
| `compute_weights_shared(window_feat)` | `compute_weights_with_features(window_feat, features_df=...)` |
| `stacksats.backtest._FEATURES_DF` global mutation | pass `features_df` explicitly at call sites |
| `BACKTEST_END` constant | `get_backtest_end()` |
| `generate_date_ranges(start, end, min_length_days)` | `generate_date_ranges(start, end)` |
| `RANGE_START` / `RANGE_END` / `MIN_RANGE_LENGTH_DAYS` module defaults | set explicit app-level defaults in your own code/config |
| `stacksats.model_development.softmax(...)` | `stacksats.model_development_helpers.softmax(...)` |
| `BaseStrategy.export_weights(config=None, **kwargs)` | `BaseStrategy.export(config=None, **kwargs)` |
| `stacksats.load_data(cache_dir=..., max_age_hours=...)` with gap-fill assumptions | `stacksats.load_data(cache_dir=..., max_age_hours=..., end_date=...)` with strict CoinMetrics source-only validation |
| raw `strategy_id` / `version` / public attrs as the informal contract | `strategy.spec()` as the canonical public contract |
| implicit dual-hook precedence | explicit `intent_preference = "propose"` or `"profile"` when both hooks exist |

## Code Replacements

### 1) Backtest weight helper

```python
# old
weights = compute_weights_shared(window_feat)

# new
weights = compute_weights_with_features(
    window_feat,
    features_df=features_df,
)
```

### 2) Backtest end date

```python
# old
assert max_received <= BACKTEST_END

# new
backtest_end = pd.to_datetime(get_backtest_end())
assert max_received <= backtest_end
```

### 3) Date range generation

```python
# old
ranges = generate_date_ranges("2025-12-01", "2027-12-31", 120)

# new
ranges = generate_date_ranges("2025-12-01", "2027-12-31")
```

### 4) Softmax helper location

```python
# old
from stacksats.model_development import softmax

# new
from stacksats.model_development_helpers import softmax
```

### 5) Strategy export helper

```python
# old
batch = strategy.export_weights(config=my_export_config)

# new
batch = strategy.export(config=my_export_config)
```

### 6) Strict data loader contract

```python
# old (assumed synthetic fill/fallback behavior)
df = load_data(cache_dir="~/.stacksats/cache")

# new (strict source-only and optional explicit end bound)
df = load_data(cache_dir="~/.stacksats/cache", end_date="2025-12-31")
```

`load_data(...)` now mirrors `BTCDataProvider.load(...)` behavior:
- no synthetic "today" row creation
- no historical date gap filling
- no MVRV fallback substitution

### 7) Strategy contract hardening

```python
# old (informal)
class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"

# new (canonical contract still supports class attrs, but spec() is the durable surface)
class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"
    intent_preference = "profile"

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("price_vs_ma", "mvrv_zscore")
```

Notes:
- `metadata()` and `params()` now back durable provenance/idempotency behavior
- runtime caches should stay private (for example `_cache`)
- if both intent hooks exist and `intent_preference` is unset, StackSats currently warns and falls back to `propose_weight(state)`

## Upgrade Checklist

1. Replace removed helpers/constants/APIs using the mapping above.
2. Remove hidden globals and pass runtime dependencies explicitly.
3. Replace `strategy.export_weights(...)` call sites with `strategy.export(...)`.
4. Ensure export calls provide explicit `--start-date` and `--end-date`.
5. Validate `load_data(...)` consumers against strict source-only behavior.
6. If your strategy implements both intent hooks, set `intent_preference` explicitly.
7. Move runtime caches to private attrs and override `params()` if you expose non-serializable public config.
8. Re-run tests:

```bash
pytest -q
```

7. Rebuild docs:

```bash
mkdocs build --strict
```

## Related docs

- [What's New](whats-new.md)
- [CLI Commands](commands.md)
- [Backtest Runtime](model_backtest.md)
- [Changelog](https://github.com/hypertrial/stacksats/tree/main/CHANGELOG.md)

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Migration+Guide)

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

## Old -> New Mapping

| Old usage | New usage |
| --- | --- |
| `compute_weights_shared(window_feat)` | `compute_weights_with_features(window_feat, features_df=...)` |
| `stacksats.backtest._FEATURES_DF` global mutation | pass `features_df` explicitly at call sites |
| `BACKTEST_END` constant | `get_backtest_end()` |
| `generate_date_ranges(start, end, min_length_days)` | `generate_date_ranges(start, end)` |
| `RANGE_START` / `RANGE_END` / `MIN_RANGE_LENGTH_DAYS` module defaults | set explicit app-level defaults in your own code/config |

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

## Upgrade Checklist

1. Replace removed helpers/constants using the mapping above.
2. Remove hidden globals and pass runtime dependencies explicitly.
3. Ensure export calls provide explicit `--start-date` and `--end-date`.
4. Re-run tests:

```bash
pytest -q
```

5. Rebuild docs:

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

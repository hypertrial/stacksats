---
title: Compare Command
description: Reference for `stacksats strategy compare` and `comparison_result.json`.
---

# Compare Command

## Prerequisites

- At least one `--strategy` selector (repeatable). The CLI prepends `--baseline` (default `uniform`) and de-duplicates, so the baseline must resolve to a `strategy_id` present in the loaded set.
- Canonical data covers the resolved window (see [Merged Metrics Parquet Schema](../reference/merged-metrics-parquet-schema.md)).
- If **any** selector is not a catalog strategy, you must pass **both** `--start-date` and `--end-date`. When every strategy is cataloged, omitted dates resolve from catalog backtest defaults (intersection logic matches the maintainer script).

## Command

```bash
stacksats strategy compare \
  --strategy simple-zscore \
  --strategy mvrv \
  --baseline uniform \
  --start-date 2018-01-01 \
  --end-date 2025-12-31 \
  --output-dir output
```

Built-in catalog and behavior: [Strategies](../reference/strategies.md).

## Expected output

- Human-readable table printed to stdout (selector/intent/tier/win rate/score/exp-decay/vs uniform/judgment).
- Line `Saved: <path>` for the JSON artifact.
- Artifact path: `output/<baseline_strategy_id>/comparison/<run_id>/comparison_result.json`.

## Key options

| Flag | Description |
|------|-------------|
| `--strategy` | Repeatable. Built-in `strategy_id` or `module_or_path:ClassName`. |
| `--baseline` | `strategy_id` used for score/exp-decay deltas (default `uniform`). |
| `--start-date`, `--end-date` | Shared validation/backtest window (required together for non-catalog mixes). |
| `--strict` / `--no-strict` | Override strict validation for the shared run (default: derived from catalog when applicable). |
| `--min-win-rate` | Override minimum win rate (default: derived from catalog when applicable). |
| `--output-dir` | Artifact root (default `output`). |
| `--strategy-config` | Optional JSON path passed to each `load_strategy` call. |

## Library usage

The stable API operates on `BaseStrategy` instances (selectors are a CLI concern):

```python
from stacksats import ComparisonConfig, MVRVStrategy, StrategyRunner, UniformStrategy

result = StrategyRunner().compare(
    [UniformStrategy(), MVRVStrategy()],
    ComparisonConfig(
        start_date="2018-01-01",
        end_date="2025-12-31",
        baseline="uniform",
        output_dir="output",
    ),
)
print(result.summary())
print(result.render_table())
```

For catalog strategies, `BaseStrategy.compare_to_benchmarks()` loads `benchmark_strategy_ids` from the catalog and calls the same runner path.

The CLI passes `selectors=` into `StrategyRunner.compare()` so each `ComparisonRow.selector` matches the argv string (e.g. `path/to/model.py:Class`) even when `metadata().strategy_id` differs.

## `comparison_result.json` contract

- Includes `schema_version` (stable artifact contract; see [Public API](../reference/public-api.md)).
- Top-level fields include `baseline_selector`, `comparison_window` (`start_date`, `end_date`, `strict`, `min_win_rate`), and `rows`.
- Each row includes strategy metadata, validation outcome, `judgment_label`, headline metrics, `multiple_vs_uniform`, deltas vs baseline (`score_delta_vs_baseline`, `exp_decay_delta_vs_baseline`), and `is_baseline`.

## Troubleshooting

- **Baseline errors:** `--baseline` must match a loaded strategy's `strategy_id` after ordering/de-duplication.
- **Custom strategy without dates:** pass both start and end dates explicitly.
- **Strict failures:** inspect validation messages from a single-strategy `stacksats strategy validate` run on the same window.

## Next step

- Interpret metrics: [Interpret Backtest Metrics](../recipes/interpret-backtest.md).
- Maintainer script: `scripts/compare_strategies.py` (delegates to `StrategyRunner.compare()`).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Compare+Command)

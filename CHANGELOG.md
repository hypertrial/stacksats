# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- Repository governance and package policy docs (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`).
- Additional project metadata links and maintainer contact metadata for PyPI consumers.
- New typed export objects: `StrategyTimeSeries`, `StrategyTimeSeriesBatch`, `StrategySeriesMetadata`, and `ColumnSpec`.
- Handwritten schema APIs for export payloads (`schema`, `schema_markdown`, `validate_schema_coverage`).
- New docs UX surfaces: `docs/tasks.md` (task hub), `docs/migration.md` (upgrade mapping), `docs/start/minimal-strategy-examples.md` (copyable templates for both strategy styles), `docs/faq.md` (structured recurring questions), and `docs/ux-plan.md` (UX acceptance criteria).
- New docs feedback issue template: `.github/ISSUE_TEMPLATE/docs_feedback.md`.

### Changed
- Release process documentation now requires changelog updates for every release.
- Breaking: `StrategyRunner.export(...)` and `BaseStrategy.export_weights(...)` now return `StrategyTimeSeriesBatch` instead of `pandas.DataFrame`.
- Export CSV contract now uses canonical columns (`start_date`, `end_date`, `day_index`, `date`, `price_usd`, `weight`) and writes `timeseries_schema.md`.
- CLI usability improvements: user-facing error messages replace common tracebacks, help now includes examples/defaults, and export requires explicit date ranges.
- Backtest artifact path messaging is standardized across CLI/docs/examples to `output/<strategy_id>/<version>/<run_id>/`.
- Docs UX quality gate now runs in CI via `scripts/check_docs_ux.py` and `.github/workflows/docs-check.yml`.

### Removed
- Legacy backtest compatibility helpers were removed:
  - `stacksats.backtest.compute_weights_shared(...)`
  - `stacksats.backtest._FEATURES_DF`
- Legacy fixed-end constant was removed:
  - `stacksats.prelude.BACKTEST_END` (use `get_backtest_end()`).
- Legacy export module defaults were removed:
  - `RANGE_START`, `RANGE_END`, `MIN_RANGE_LENGTH_DAYS`.

### Migration Notes
- Replace `compute_weights_shared(window_feat)` with `compute_weights_with_features(window_feat, features_df=...)`.
- Replace `BACKTEST_END` constant reads with runtime `get_backtest_end()`.
- Update `generate_date_ranges(...)` call sites:
  - old: `generate_date_ranges(start, end, min_length_days)`
  - new: `generate_date_ranges(start, end)`

## [0.1.0] - 2026-02-12

### Added
- Initial public release of the `stacksats` package.
- Rolling-window Bitcoin DCA backtesting and validation APIs.
- Strategy loading, feature precomputation, CLI tools, and optional deployment extras.

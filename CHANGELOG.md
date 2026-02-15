# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- Repository governance and package policy docs (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`).
- Additional project metadata links and maintainer contact metadata for PyPI consumers.
- New typed export objects: `StrategyTimeSeries`, `StrategyTimeSeriesBatch`, `StrategySeriesMetadata`, and `ColumnSpec`.
- Handwritten schema APIs for export payloads (`schema`, `schema_markdown`, `validate_schema_coverage`).

### Changed
- Release process documentation now requires changelog updates for every release.
- Breaking: `StrategyRunner.export(...)` and `BaseStrategy.export_weights(...)` now return `StrategyTimeSeriesBatch` instead of `pandas.DataFrame`.
- Export CSV contract now uses canonical columns (`start_date`, `end_date`, `day_index`, `date`, `price_usd`, `weight`) and writes `timeseries_schema.md`.

## [0.1.0] - 2026-02-12

### Added
- Initial public release of the `stacksats` package.
- Rolling-window Bitcoin DCA backtesting and validation APIs.
- Strategy loading, feature precomputation, CLI tools, and optional deployment extras.

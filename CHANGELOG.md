# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

## [0.5.2] - 2026-02-28

### Added
- Added `StrategyTimeSeries` / `StrategyTimeSeriesBatch` artifact reload helpers (`from_csv(...)`, `from_artifact_dir(...)`, `to_csv(...)`) plus convenience accessors for window/date inspection.
- Added explicit extensible schema support for strategy-owned export columns via `extra_schema`.
- Added dedicated artifact and schema hardening coverage for StrategyTimeSeries exports.

### Changed
- Hardened `StrategyTimeSeries` into a read-only validated object backed by a private normalized payload.
- Enforced exact daily-window coverage whenever `window_start` and `window_end` are present in `StrategySeriesMetadata`.
- Normalized and validated `StrategySeriesMetadata` at construction time, including UTC `generated_at` and daily-normalized window bounds.
- Unified batch/window provenance so one batch shares one coherent `generated_at`.
- Export artifacts now record local file names so `artifacts.json` can be reloaded portably from the artifact directory.

## [0.5.1] - 2026-02-27

### Added
- New idempotent daily execution lifecycle command: `stacksats strategy run-daily`.
- New daily execution APIs/types: `RunDailyConfig`, `DailyOrderRequest`, `DailyOrderReceipt`, `DailyRunResult`, and `BaseStrategy.run_daily(...)`.
- New SQLite-backed daily run ledger and weight snapshot state for replay-safe execution (`.stacksats/run_state.sqlite3` by default).
- New execution adapter interface plus deterministic built-in `PaperExecutionAdapter`.

### Changed
- Removed the stale exported notebook asset from docs and updated the notebook demo page to use maintained CLI workflows.
- `stacksats.load_data(...)` now delegates to `BTCDataProvider` strict source-only semantics and accepts optional `end_date`.
- Schema sync script and pre-commit hook now resolve repository-local imports robustly without manual `PYTHONPATH` workarounds.
- Runtime modules no longer execute dotenv or matplotlib backend/style setup at import time.

### Removed
- Compatibility API `stacksats.model_development.softmax(...)` (use `stacksats.model_development_helpers.softmax(...)`).
- Compatibility API `BaseStrategy.export_weights(...)` (use `BaseStrategy.export(...)`).

## [0.4.0] - 2026-02-18

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

## [0.3.2] - 2026-02-18

### Added
- Added packaged example strategies under `stacksats.strategies`, including the MVRV+ strategy/model path.
- Added backtest outputs for uniform exp-decay and dynamic-vs-uniform multiple metrics.

### Changed
- Updated docs, scripts, and CLI examples to reference packaged example strategies.
- Focused notebook/docs flows on the model example path and removed the browser-safe notebook route.

### Fixed
- Extended unit/BDD/snapshot coverage for new strategy paths and backtest output metrics.

## [0.3.1] - 2026-02-17

### Added
- Added comprehensive failure-path rollback tests for export weight DB operations.

### Changed
- Hardened export weight DB transaction handling, including rollback behavior on failures.
- Refreshed docs intro and command documentation for clearer onboarding.

## [0.3.0.post1.dev2] - 2026-02-17

### Changed
- Published pre-release validation build from post-`0.3.0` commit state prior to the stable `0.3.1` tag.
- Included early post-`0.3.0` hardening/docs updates for release validation on PyPI.

## [0.3.0] - 2026-02-16

### Added
- Added issue templates for bug and feature requests.

### Changed
- Improved CLI usability with clearer user-facing behavior and docs/test alignment.
- Aligned export defaults and behavior across code/docs/tests around explicit date-bound export workflows.
- Refactored large core modules into focused helper modules to reduce file size and improve maintainability.
- Consolidated docs structure and strengthened docs quality checks.

### Fixed
- Updated dependency versions/pins for packaging/runtime consistency.

## [0.2.3] - 2026-02-16

### Changed
- Switched releases to manual token-based PyPI publishing by default.
- Added and documented `scripts/publish_pypi_manual.sh` as the primary publish path.

## [0.2.2] - 2026-02-16

### Added
- Added MkDocs site foundation plus docs-check/docs-pages workflows.
- Added typed strategy time-series object model docs and schema sync tooling.
- Added pre-commit/pre-push quality hooks and expanded runtime/docs test coverage.

### Changed
- Split package-check CI into PR and push workflows.
- Refined docs architecture/theme/assets and improved docs deployment flow.

### Fixed
- Fixed Python 3.12 deployment compatibility and optional `psycopg2` handling in CI/runtime paths.
- Hardened release preflight behavior for SSL-constrained environments.

## [0.2.1] - 2026-02-15

### Notes
- Git tag release; not published to PyPI.

### Fixed
- Fixed PyPI workflow version-check dependency ordering in release automation.

## [0.2.0] - 2026-02-15

### Notes
- Git tag release; not published to PyPI.

### Changed
- Removed marimo-driven docs flow in favor of the docs/site direction adopted in subsequent releases.

## [0.1.1] - 2026-02-12

### Changed
- Updated README to point users to hosted docs and canonical docs entry points.

## [0.1.0] - 2026-02-12

### Added
- Initial public release of the `stacksats` package.
- Rolling-window Bitcoin DCA backtesting and validation APIs.
- Strategy loading, feature precomputation, CLI tools, and optional deployment extras.

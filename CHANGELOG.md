# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

## [1.1.0] - 2026-03-29

### Added
- Added a stable agent-facing daily decision interface: `DecideDailyConfig`, `DailyDecisionResult`, `StrategyRunner.decide_daily(...)`, `BaseStrategy.decide_daily(...)`, and the `stacksats strategy decide-daily` CLI command.
- Added a documented `decision_result.json` artifact contract for execution-ready daily decision payloads consumed by external AI agents or brokerage layers.
- Added direct unit and end-to-end coverage for the new decision flow, including idempotent noop behavior, force reruns, CLI mapping, artifact snapshots, and package exports.

### Changed
- Repositioned StackSats docs around an agent-native production flow where StackSats computes the validated decision and external systems execute brokerage orders.
- Refactored `run-daily` to reuse the shared daily decision computation path so integrated execution stays aligned with the agent-facing interface.
- Expanded the stable `1.x` contract docs and command references to include `decide-daily` across README, task, command, public API, strategy, FAQ, and stability pages.

### Fixed
- Isolated the docs/example smoke harness so decision and output artifacts no longer reuse repository-local `.stacksats` or `output/` state during verification.

## [1.0.2] - 2026-03-28

### Added
- Added `RunDailyPaperStrategy` as the stable canonical paper-execution built-in for the documented `run-daily` flow.
- Added local CLI smoke coverage for `demo backtest`, `strategy export`, `strategy animate`, `data prepare`, `data doctor`, `run-daily`, and `reconcile-daily`, plus direct contract coverage for release/wheel smoke orchestration.
- Added workflow contract tests that lock first-party CI/docs workflow wiring and current GitHub Action major versions.

### Changed
- Moved `run-daily` validation defaults to a strategy-owned hook while preserving strict defaults for existing strategies.
- Updated first-party GitHub Actions to Node 24-ready action majors and documented the verification lanes more explicitly for maintainers.
- Expanded public API, strategy catalog, task, command, and release docs to match the stable `1.x` daily execution and verification contracts.

### Fixed
- Fixed the documented stable paper `run-daily` happy path on packaged demo data without weakening validation for existing strategies.

## [1.0.1] - 2026-03-27

### Changed
- Enforced `100%` line and branch coverage for `stacksats/` in the release gate so the published engineering contract now matches CI.
- Added targeted branch-path regression coverage across optional dependency handling, EDA helpers, plotting fallbacks, runner/runtime edges, and strategy time-series helpers.
- Tightened maintainer docs so release, contribution, and coverage guidance stay in sync with the current `1.x` process.

## [1.0.0] - 2026-03-27

### Added
- Added a published stability policy covering supported platforms, stable public API boundaries, experimental surfaces, and deprecation rules.
- Added `schema_version` to stable JSON artifact payloads (`backtest_result.json`, `metrics.json`, `animation_manifest.json`, `artifacts.json`).
- Added release-grade `release-gate.yml` CI for `release/*` branches and `v*` tags.
- Added a dedicated isolated release wheel smoke script to validate base and `viz` installs from the built wheel artifact.
- Added pytest marker-contract coverage that proves `-m "not performance"` excludes the entire `tests/performance/` tree.

### Changed
- Narrowed the stable `1.x` contract to top-level `stacksats` exports, documented artifact payloads, and the documented CLI subset.
- Promoted `UniformStrategy`, `SimpleZScoreStrategy`, `MomentumStrategy`, and `MVRVStrategy` as stable built-ins.
- Moved advanced BRK overlay strategies to `stacksats.strategies.experimental.*` and removed their old pre-v1 import paths.
- Split optional dependencies into `viz`, `network`, `deploy`, and `all` extras and added maintainer constraints for reproducible release environments.
- Updated docs to describe causal lint as best-effort static analysis rather than a runtime sandbox.
- Promoted package metadata from alpha to production/stable.
- Expanded the release gate with a macOS Python 3.11 smoke lane so the documented Linux/macOS support policy is backed by CI.
- Marked every benchmark, memory, and scaling test under `tests/performance/` as `performance` so release lanes and local selectors now match their names.

## [0.8.2] - 2026-03-17

### Added
- Added packaged first-run onboarding commands: `stacksats demo backtest`, `stacksats data fetch`, `stacksats data prepare`, and `stacksats data doctor`.
- Added a wheel-install smoke test to the release test suite to verify packaged CLI/runtime flows.

### Changed
- Increased repository coverage gating to 100% and expanded targeted branch-path coverage across runner, export, and loader edges.
- Improved Polars runtime/backtest hot paths with lazy guards and batch-oriented execution flow hardening.
- Included `wheel` in development extras to keep local release tooling aligned with package build requirements.

### Fixed
- Restored loader/backtest edge-case behavior for out-of-coverage windows and corresponding regression tests.
- Synced README/docs command references and export guidance for the current onboarding/runtime paths.

## [0.8.1] - 2026-03-16

### Changed
- Backtest/validation default scoring horizon is now fixed to `2018-01-01` through `2025-12-31` (with end-date clamped to available data).
- Data loader defaults now retain pre-start history for feature warmup while preserving requested scoring bounds.
- Strategy audit runner now enforces the canonical default end horizon (`2025-12-31`) when source data extends beyond it.

### Fixed
- `export_weights.get_current_btc_price(...)` now disables warmup for its short lookback fetch, avoiding unnecessary full-history loads.
- Warmup-inclusive loader paths now consistently reject missing/non-finite `price_usd` values in returned frames.

### Docs
- Added canonical merged-metrics schema reference page and aligned docs to the canonical long-format parquet + derived runtime parquet flow.
- Updated runtime/backtest/validation docs to reflect fixed default horizon and warmup-default behavior.

## [0.8.0] - 2026-03-15

### Changed
- **Pandas removed; Polars-only:** (1) `pandas` removed from project dependencies. (2) All data types use `datetime.datetime` and `pl.DataFrame`; `StrategyContext`, `DayState`, `WeightTimeSeriesBatch`, prelude, and framework contract are Polars-only. (3) `BTCDataProvider`, `ColumnMapDataProvider`, `FeatureTimeSeries`, `WeightTimeSeries`, `WeightTimeSeriesBatch`, export_weights, animation_data, and btc_price_fetcher use Polars. (4) Strategy hooks (`transform_features`, `build_signals`, `build_target_profile`) use `pl.DataFrame` and `pl.Series`. (5) Example strategies (`UniformStrategy`, `SimpleZScoreStrategy`, `MomentumStrategy`) migrated to Polars.
- Default data path is now 100% parquet: `BTCDataProvider` and `load_data()` use `STACKSATS_ANALYTICS_PARQUET` / `./bitcoin_analytics.parquet`. Removed DuckDB dependency and all DuckDB-only code (feature providers, strategies, scripts, docs).
- `BRKOverlayFeatureProvider` now reads overlay metrics from `btc_df` columns only (parquet or user DataFrame); no separate database.
- Fetch script and manifest use `parquet` asset; see [BRK Data Source](docs/data-source.md).
- Validation and strategy docs now report the leakage gate as `No Forward Leakage` to make pass/fail semantics explicit.
- Built-in strategy audit tooling now lazily projects long-format `merged_metrics*.parquet` inputs and lifts the overlay metrics required by BRK-aware strategies into the temporary BRK-wide frame.

### Added
- Added `scripts/profile_strategy_hotpaths.py` to profile validation, backtest window iteration, and allocation kernel performance against local workspace code.
- Added targeted runtime optimization regression coverage for allocation equivalence, `compute_cycle_spd(...)` equivalence, overlay-feature materialization, and validation cache lifecycle behavior.

### Fixed
- Optimized `allocate_sequential_stable(...)` to reuse prefix-stable signals instead of recomputing cumulative normalization for every day.
- Reused cached materialized features and computed weights across validation and nested backtests to reduce repeated work without changing validation semantics.
- Reduced `compute_cycle_spd(...)` window overhead by reusing sorted date indices for positional slicing and skipping unnecessary join work when strategy weights already align to the window dates.
- Fixed BRK overlay feature lagging so early observed prefixes preserve overlay columns instead of shifting away the `date` column.
- Hardened `scripts/run_all_strategies.py` so strategy failures are recorded in the audit report instead of aborting the full batch.

### Removed
- Optional extra `brk` (duckdb). Parquet support is provided by default dependency `pyarrow`.
- `DuckDBAnalyticsFeatureProvider`, `DuckDBAlphaStrategy`, DuckDB scripts (render_duckdb_schema_doc, train_duckdb_factor_strategy, compare_duckdb_alpha), and related tests/docs.

## [0.7.3] - 2026-03-15

### Added
- Added `ColumnMapDataProvider` for flexible data ingestion without DuckDB. Users can supply any Pandas DataFrame by mapping library-canonical column names (e.g. `price_usd`, `mvrv`) to their DataFrame column names.
- Added `StrategyRunner.from_dataframe(df, column_map=...)` as the primary entry point for using StackSats without a BRK DuckDB installation.

### Changed
- Renamed `StrategyTimeSeries` → `TimeSeries` and `StrategyTimeSeriesBatch` → `TimeSeriesBatch` for a cleaner public API. `Strategy` and `TimeSeries` are now the primary library objects.
- Strategy export methods and runners now return `TimeSeriesBatch` (deprecated `StrategyTimeSeriesBatch` name remains supported).

## [0.7.2] - 2026-03-11

### Added
- Added dedicated command reference pages under `docs/run/` for validate, backtest, export, run-daily, and animate flows.

### Changed
- Reorganized docs information architecture and navigation around `Start`, `Run`, `Build`, `Reference`, and `Maintainers`.
- Refreshed docs visual styling with tokenized light/dark themes, improved typography rhythm, and consistent cards/code/admonition presentation.
- Refactored docs UX checks to enforce structural intent instead of brittle literal heading matches.
- Updated docs ownership and cross-page links so task pages remain workflow-intent focused while command pages are the canonical flag/reference source.

## [0.7.1] - 2026-03-08

### Fixed
- Stabilized BRK overlay and feature registry unit tests by provisioning deterministic synthetic DuckDB fixtures in-test instead of relying on a repository-root `bitcoin_analytics.duckdb`.
- Removed environment-dependent unit test failures where `brk_overlay_v1` materialization could fail on missing local DuckDB files in clean CI runners.

## [0.7.0] - 2026-03-08

### Added
- Added a required source-contract guard script (`scripts/check_no_coinmetrics_refs.py`) to prevent reintroduction of deprecated CoinMetrics tokens in active runtime, test, script, and workflow paths.
- Added CI quality workflow enforcement of the BRK-only source guard in both PR and main package-check pipelines.
- Added explicit legacy-loader removal coverage with a dedicated regression test asserting the removed source module cannot be imported.

### Changed
- Finalized hard-break BRK-only docs and release messaging across README, commands, tasks, migration, and release guides.
- Updated release preflight (`scripts/release_check.sh`) to include the BRK source-reference guard in the required release-gate command chain.
- Completed StrategyTimeSeries schema/lineage cleanup to use BRK source-oriented naming and remove ambiguous duplicate lineage/schema entries.
- Migrated stale data-loader/historical-fetch/prelude contract tests to BRK DuckDB-native signatures and behavior.

### Removed
- Removed remaining legacy CoinMetrics CSV loader code and CoinMetrics-focused unit tests from active code paths.

## [0.6.1] - 2026-03-06

### Added
- Added a reusable GitHub composite action (`.github/actions/setup-python-project`) so Python setup and editable dependency installation are defined once across workflows.
- Added a scheduled/manual command-smoke workflow (`.github/workflows/example-commands-smoke.yml`) to continuously verify docs lifecycle command examples.
- Added `scripts/clean_local.sh` to remove local generated artifacts and caches (`.coverage*`, `coverage.xml`, `dist/`, `build/`, `site/`, `output/`, cache directories).
- Added focused runner-validation unit test modules and shared test kit utilities to improve test maintenance and xdist balancing.

### Changed
- Adjusted coverage policy to an actionable global fail-under floor of `97%` in `scripts/check_coverage.sh` (with explicit ratchet-up guidance).
- Consolidated CI workflow setup logic across package, docs, and coverage jobs to reduce duplicated installation/setup steps.
- Stabilized `scripts/test_example_commands.py` by using deterministic smoke variants and synthetic cache-backed data so scheduled/manual smoke runs are reliable.
- Expanded targeted runner/validation helper extraction and branch-path test coverage without changing public runtime APIs.
- Updated docs to reflect current CI/test-tier behavior, strict validation defaults, and data-coverage expectations for export command examples.

## [0.6.0] - 2026-03-04

### Added
- Added `scripts/check_markdown_scope.sh` to define the tracked markdown scope used by docs tooling and CI.
- Added `scripts/check_release_docs_sync.py` to verify `docs/whats-new.md` stays aligned with the latest released section in `CHANGELOG.md`.

### Changed
- Corrected the documented `setuptools-scm` release order so publishable artifacts are built after creating the annotated release tag.
- Strengthened `scripts/release_check.sh` to run docs checks plus the full non-performance test suite instead of inheriting the fast local pytest defaults.
- Expanded docs CI markdown coverage to all tracked `.md` files, including `.github` templates and root policy docs.
- Normalized docs around strict validation defaults, fast-vs-release test tiers, and repo-venv maintainer commands.
- Simplified `docs/whats-new.md` into a current-release summary page instead of a manually maintained release-history mirror.

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

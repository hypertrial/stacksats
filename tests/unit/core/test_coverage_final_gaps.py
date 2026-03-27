from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import stacksats.animation_data as animation_data
import stacksats.prelude as prelude_module
from stacksats.btc_price_fetcher import _fetch_with_retry
from stacksats.loader import load_strategy
from stacksats.plot_mvrv_render import plot_mvrv_metrics
from stacksats.runner import StrategyRunner
from stacksats.runner_helpers import build_fold_ranges, weights_match
from stacksats.runner_validation import _ValidationState
from stacksats.strategy_lint import _is_negative_integer
from stacksats.strategy_time_series import StrategySeriesMetadata, WeightTimeSeries
from stacksats.strategy_time_series_batch import WeightTimeSeriesBatch
from stacksats.strategy_types import BacktestConfig, BaseStrategy, ExportConfig, ValidationConfig
from tests.unit.core.runner_validation_testkit import (
    ProfileOffsetLeakStrategy,
    UniformProposeStrategy,
    btc_df,
)


def _write_module(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def _weight_series(prices: list[float]) -> WeightTimeSeries:
    dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(prices))]
    return WeightTimeSeries(
        metadata=StrategySeriesMetadata(
            strategy_id="coverage",
            strategy_version="1.0.0",
            run_id="run-1",
            config_hash="cfg",
            schema_version="1.0.0",
            window_start=dates[0],
            window_end=dates[-1],
        ),
        data=pl.DataFrame(
            {
                "date": dates,
                "weight": [1.0 / len(prices)] * len(prices),
                "price_usd": prices,
                "mvrv": [1.0] * len(prices),
            }
        ),
    )


def test_animation_parse_date_fallback_and_empty_prepared_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DateTimeFallback(dt.datetime):
        @classmethod
        def fromisoformat(cls, value: str):
            raise ValueError(value)

    monkeypatch.setattr(animation_data, "datetime", _DateTimeFallback)
    assert animation_data._parse_window_date("2024-01-03") == dt.datetime(2024, 1, 3)

    monkeypatch.setattr(
        animation_data,
        "_normalize_spd_frame",
        lambda frame: pl.DataFrame(
            schema={
                "window": pl.Utf8,
                "window_start": pl.Datetime,
                "window_end": pl.Datetime,
                "dynamic_percentile": pl.Float64,
                "uniform_percentile": pl.Float64,
                "excess_percentile": pl.Float64,
                "dynamic_sats_per_dollar": pl.Float64,
                "uniform_sats_per_dollar": pl.Float64,
            }
        ),
    )
    with pytest.raises(ValueError, match="SPD table is empty"):
        animation_data.prepare_animation_frame_data(pl.DataFrame({"window": []}, schema={"window": pl.Utf8}))


def test_fetch_with_retry_rejects_empty_retry_policy() -> None:
    class _EmptyRetryPolicy:
        def __iter__(self):
            return iter(())

    with pytest.raises(RuntimeError, match="Retry policy exhausted unexpectedly"):
        _fetch_with_retry(lambda: 42000.0, "EmptySource", retry_policy=_EmptyRetryPolicy())


def test_loader_rejects_empty_metadata_fields_post_contract_validation(tmp_path: Path) -> None:
    empty_id = _write_module(
        tmp_path / "empty_id.py",
        """
from types import SimpleNamespace
from stacksats.strategy_types import BaseStrategy, StrategyMetadata

class EmptyIdStrategy(BaseStrategy):
    strategy_id = "declared-id"
    version = "1.0.0"
    _calls = 0

    def propose_weight(self, state):
        return state.uniform_weight

    def metadata(self):
        self._calls += 1
        if self._calls == 1:
            return SimpleNamespace(strategy_id="", version="1.0.0")
        return StrategyMetadata(strategy_id="declared-id", version="1.0.0")
""",
    )
    with pytest.raises(ValueError, match="must define non-empty strategy_id metadata"):
        load_strategy(f"{empty_id}:EmptyIdStrategy")

    empty_version = _write_module(
        tmp_path / "empty_version.py",
        """
from types import SimpleNamespace
from stacksats.strategy_types import BaseStrategy, StrategyMetadata

class EmptyVersionStrategy(BaseStrategy):
    strategy_id = "declared-id"
    version = "1.0.0"
    _calls = 0

    def propose_weight(self, state):
        return state.uniform_weight

    def metadata(self):
        self._calls += 1
        if self._calls == 1:
            return SimpleNamespace(strategy_id="declared-id", version="")
        return StrategyMetadata(strategy_id="declared-id", version="1.0.0")
""",
    )
    with pytest.raises(ValueError, match="must define non-empty version metadata"):
        load_strategy(f"{empty_version}:EmptyVersionStrategy")


def test_plot_mvrv_requires_date_column_before_plotting() -> None:
    with pytest.raises(ValueError, match="DataFrame must have a 'date' column"):
        plot_mvrv_metrics(
            pl.DataFrame({"mvrv": [1.0]}),
            init_plot_env_fn=lambda: None,
            logging_mod=type("Log", (), {"info": staticmethod(lambda *args, **kwargs: None)})(),
            plt_mod=None,
            mdates_mod=None,
            path_cls=Path,
        )


def _sample_cycle_btc_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(370)],
            "price_usd": np.linspace(100.0, 200.0, 370),
            "mvrv": np.linspace(1.0, 2.0, 370),
        }
    )


def _uniform_window_weights(frame: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": frame["date"],
            "weight": np.full(frame.height, 1.0 / frame.height),
        }
    )


def test_compute_cycle_spd_skips_when_generated_window_start_is_past_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    btc = _sample_cycle_btc_frame()

    monkeypatch.setattr(
        prelude_module.pl,
        "datetime_range",
        lambda *args, **kwargs: pl.Series("date", [dt.datetime(2025, 1, 10)]),
    )
    skipped = prelude_module.compute_cycle_spd(
        btc,
        _uniform_window_weights,
        start_date="2024-01-01",
        end_date="2025-01-05",
        validate_weights=False,
    )
    assert skipped.is_empty()


def test_prelude_make_window_label() -> None:
    """_make_window_label formats window label."""
    from stacksats.prelude import _make_window_label

    label = _make_window_label(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 12, 31),
    )
    assert label == "2024-01-01 → 2024-12-31"


def test_compute_cycle_spd_price_can_slice_false_with_duplicate_dates() -> None:
    """When price_plan.has_unique_dates is False, we hit the else branch (price_can_slice=False)."""
    base_dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    dates = base_dates + [base_dates[0], base_dates[1]]
    btc = pl.DataFrame(
        {
            "date": dates,
            "price_usd": np.linspace(100.0, 200.0, len(dates)),
            "mvrv": np.linspace(1.0, 2.0, len(dates)),
        }
    )
    result = prelude_module.compute_cycle_spd(
        btc,
        _uniform_window_weights,
        start_date="2024-01-01",
        end_date="2025-01-05",
        validate_weights=True,
    )
    assert result.height >= 1


def test_compute_cycle_spd_batched_validation_failure() -> None:
    """Batched strategy returning invalid weight sums raises ValueError."""
    btc = _sample_cycle_btc_frame()

    class BadBatchStrategy:
        def _compute_window_weights_batch(
            self, full_feat, *, feature_plan, windows, expected_days
        ):
            rows = []
            for i in range(min(2, windows.height)):
                w_start = windows["window_start"][i]
                w_end = windows["window_end"][i]
                dates = pl.datetime_range(w_start, w_end, interval="1d", eager=True)
                n = len(dates)
                weights = [2.0] + [0.0] * (n - 1)
                rows.append(
                    pl.DataFrame(
                        {
                            "window_start": [w_start] * n,
                            "window_end": [w_end] * n,
                            "date": dates,
                            "weight": weights,
                        }
                    )
                )
            return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()

    with pytest.raises(ValueError, match="Batched framework weights failed sum validation"):
        prelude_module.compute_cycle_spd(
            btc,
            BadBatchStrategy(),
            start_date="2024-01-01",
            end_date="2025-01-05",
            validate_weights=True,
        )


def test_compute_cycle_spd_falls_back_to_uniform_when_weights_are_misaligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    btc = _sample_cycle_btc_frame()

    monkeypatch.setattr(
        prelude_module,
        "_normalize_weight_frame",
        lambda price_slice, _weight_df: pl.DataFrame(
            {"date": [price_slice["date"][0]], "weight": [1.0]}
        ),
    )
    result = prelude_module.compute_cycle_spd(
        btc,
        _uniform_window_weights,
        start_date="2024-01-01",
        end_date="2025-01-05",
        validate_weights=True,
    )
    assert result.height >= 1
    assert np.allclose(
        result["dynamic_sats_per_dollar"].to_numpy(),
        result["uniform_sats_per_dollar"].to_numpy(),
    )


def test_export_supports_non_polars_dataframe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()

    csv_calls: dict[str, object] = {}

    class _ExportFrame:
        def to_csv(self, path, index=False):
            csv_calls["path"] = Path(path)
            csv_calls["index"] = index
            Path(path).write_text("start_date,end_date,date,weight,price_usd\n", encoding="utf-8")

    class _FakeBatch:
        def to_dataframe(self):
            return _ExportFrame()

        def schema_markdown(self) -> str:
            return "# schema"

    monkeypatch.setattr("stacksats.prelude.generate_date_ranges", lambda *args, **kwargs: [(dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2))])
    monkeypatch.setattr(
        "stacksats.prelude.group_ranges_by_start_date",
        lambda ranges: {ranges[0][0]: [ranges[0][1]]},
    )
    monkeypatch.setattr(
        "stacksats.export_weights.process_start_date_batch",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "start_date": ["2024-01-01", "2024-01-01"],
                "end_date": ["2024-01-02", "2024-01-02"],
                "date": ["2024-01-01", "2024-01-02"],
                "weight": [0.5, 0.5],
                "price_usd": [100.0, 101.0],
            }
        ),
    )
    monkeypatch.setattr(
        "stacksats.runner.WeightTimeSeriesBatch.from_flat_dataframe",
        lambda *args, **kwargs: _FakeBatch(),
    )
    monkeypatch.setattr(
        runner,
        "_materialize_strategy_features",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
                "price_usd": [100.0, 101.0],
                "mvrv": [1.0, 1.1],
            }
        ),
    )

    runner.export(
        UniformProposeStrategy(),
        ExportConfig(
            range_start="2024-01-01",
            range_end="2024-01-02",
            output_dir=str(tmp_path),
        ),
        btc_df=btc_df(days=600),
        current_date=dt.datetime(2024, 1, 2),
    )

    assert csv_calls["index"] is False
    assert Path(csv_calls["path"]).exists()


def test_runner_helper_edge_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert weights_match(
        pl.DataFrame({"date": [dt.datetime(2024, 1, 1)]}),
        pl.DataFrame({"date": [dt.datetime(2024, 1, 2)]}),
    ) is False

    monkeypatch.setattr(
        "stacksats.runner_helpers.np.linspace",
        lambda *args, **kwargs: np.array([0, 0, 366, 732, 1098], dtype=int),
    )
    folds = build_fold_ranges(dt.datetime(2020, 1, 1), dt.datetime(2025, 12, 31))
    assert len(folds) >= 1


def test_runner_validation_remaining_edge_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    assert runner._last_runtime_profile() == {}
    runner._last_runtime_hotpath_profile = "bad"
    assert runner._last_runtime_profile() == {}

    dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    btc = pl.DataFrame({"date": dates, "price_usd": np.linspace(100.0, 500.0, len(dates))})
    features = btc.with_columns(pl.lit(1.0).alias("signal"))
    state = _ValidationState(messages=[])

    monkeypatch.setattr(
        runner,
        "_compute_strategy_weights",
        lambda *args, **kwargs: (
            pl.DataFrame({"date": [kwargs["end_date"]], "weight": [1.0]}),
            True,
        ),
    )
    runner._run_weight_constraint_checks(
        strategy=UniformProposeStrategy(),
        btc_df=btc,
        features_df=features,
        start_ts=dates[0],
        end_ts=dates[-1],
        strict_mode=True,
        config=ValidationConfig(strict=True),
        state=state,
    )
    assert state.strict_checks_ok is False

    state = _ValidationState(messages=[])
    profile_runner = StrategyRunner()
    monkeypatch.setattr(
        profile_runner,
        "_compute_strategy_weights",
        lambda *args, **kwargs: (
            pl.DataFrame({"date": [kwargs["current_date"]], "weight": [1.0]}),
            False,
        ),
    )
    monkeypatch.setattr(
        profile_runner,
        "_build_profile_with_mutation_guard",
        lambda *args, **kwargs: (
            pl.DataFrame({"date": [dates[0]], "value": [1.0]}),
            True,
        ),
    )
    monkeypatch.setattr(
        profile_runner,
        "_strategy_ctx_from_source",
        lambda **kwargs: type("Ctx", (), {"features": type("Features", (), {"data": features})()})(),
    )
    monkeypatch.setattr(profile_runner, "_strict_determinism_check", lambda **kwargs: True)
    monkeypatch.setattr(profile_runner, "_check_prefix_invariance", lambda **kwargs: True)
    profile_runner._run_forward_leakage_checks(
        strategy=ProfileOffsetLeakStrategy(),
        btc_df=btc,
        full_features_df=features,
        backtest_idx=pl.Series("date", [dates[2]]),
        start_ts=dates[0],
        strict_mode=True,
        has_propose_hook=False,
        has_profile_hook=True,
        probe_step=1,
        state=state,
    )
    assert any("profile build" in message for message in state.messages)

    state = _ValidationState(messages=[])
    observed_labels: list[str] = []

    def _prefix_check(**kwargs):
        observed_labels.append(str(kwargs["check_label"]))
        return len(observed_labels) != 4

    perturbed_runner = StrategyRunner()
    monkeypatch.setattr(
        perturbed_runner,
        "_compute_strategy_weights",
        lambda *args, **kwargs: (
            pl.DataFrame({"date": [kwargs["current_date"]], "weight": [1.0]}),
            False,
        ),
    )
    monkeypatch.setattr(
        perturbed_runner,
        "_build_profile_with_mutation_guard",
        lambda *args, **kwargs: (
            pl.DataFrame({"date": [dates[0]], "value": [1.0]}),
            False,
        ),
    )
    monkeypatch.setattr(perturbed_runner, "_check_prefix_invariance", _prefix_check)
    perturbed_runner._run_forward_leakage_checks(
        strategy=ProfileOffsetLeakStrategy(),
        btc_df=btc,
        full_features_df=features,
        backtest_idx=pl.Series("date", [dates[2]]),
        start_ts=dates[0],
        strict_mode=False,
        has_propose_hook=False,
        has_profile_hook=True,
        probe_step=1,
        state=state,
    )
    assert observed_labels[-1] == "profile values diverge (perturbed-future)"


def test_strict_statistical_checks_cover_fold_skip_and_missing_feature_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    state = _ValidationState(messages=[], diagnostics={})
    start_ts = dt.datetime(2024, 1, 1)
    end_ts = dt.datetime(2024, 12, 31)
    spd_table = pl.DataFrame(
        {
            "dynamic_percentile": [60.0, 55.0],
            "uniform_percentile": [50.0, 45.0],
            "excess_percentile": [10.0, 10.0],
        }
    )

    monkeypatch.setattr(
        "stacksats.runner_validation.build_purged_walk_forward_folds",
        lambda *args, **kwargs: [(None, None, start_ts, start_ts + dt.timedelta(days=1))],
    )
    runner._strict_statistical_checks(
        strategy=UniformProposeStrategy(),
        btc_df=btc_df(days=600),
        features_df=pl.DataFrame({"date": [start_ts], "signal": [1.0]}),
        start_ts=start_ts,
        end_ts=end_ts,
        config=ValidationConfig(strict=True),
        state=state,
        backtest_result=type("Result", (), {"spd_table": spd_table})(),
    )
    assert state.diagnostics["feature_drift"] == {"max_psi": 0.0, "max_ks": 0.0}

    class _RequiredColumnStrategy(BaseStrategy):
        strategy_id = "required-column"
        version = "1.0.0"

        def propose_weight(self, state):
            return state.uniform_weight

        def required_feature_columns(self) -> tuple[str, ...]:
            return ("signal",)

    materialized = iter(
        [
            pl.DataFrame({"date": [start_ts], "signal": [1.0]}),
            pl.DataFrame({"date": [end_ts], "other": [2.0]}),
        ]
    )
    monkeypatch.setattr(
        "stacksats.runner_validation.build_purged_walk_forward_folds",
        lambda *args, **kwargs: [
            (
                None,
                None,
                start_ts + dt.timedelta(days=366),
                start_ts + dt.timedelta(days=367),
            )
        ],
    )
    monkeypatch.setattr(
        runner,
        "_materialize_strategy_features",
        lambda *args, **kwargs: next(materialized),
    )
    state = _ValidationState(messages=[], diagnostics={})
    runner._strict_statistical_checks(
        strategy=_RequiredColumnStrategy(),
        btc_df=btc_df(days=800),
        features_df=pl.DataFrame({"date": [start_ts], "signal": [1.0]}),
        start_ts=start_ts,
        end_ts=end_ts + dt.timedelta(days=400),
        config=ValidationConfig(strict=True),
        state=state,
        backtest_result=type("Result", (), {"spd_table": spd_table})(),
    )
    assert state.diagnostics["feature_drift"] == {"max_psi": 0.0, "max_ks": 0.0}


def test_strategy_lint_and_time_series_remaining_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _is_negative_integer(ast.Constant(value=0)) is False
    assert _is_negative_integer(ast.Name(id="x")) is False

    series = _weight_series([100.0, 101.0, 102.0])
    object.__setattr__(
        series,
        "_data",
        series.data.with_columns(pl.col("date").dt.strftime("%Y-%m-%d").alias("date")),
    )
    series.validate()

    ts = _weight_series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    original_isclose = np.isclose
    monkeypatch.setattr(
        "stacksats.strategy_time_series_analysis.np.isclose",
        lambda value, other, *args, **kwargs: True
        if float(value) == 1.0 and float(other) == 0.0
        else original_isclose(value, other, *args, **kwargs),
    )
    dec = ts.decompose(period=2, model="multiplicative", series="price")
    assert "residual" in dec.columns


def test_artifact_dir_prefers_existing_cwd_relative_weights_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    cwd_csv = tmp_path / "weights.csv"
    cwd_csv.write_text(
        "start_date,end_date,date,weight,price_usd\n"
        "2024-01-01,2024-01-01,2024-01-01,1.0,42000.0\n",
        encoding="utf-8",
    )
    (artifact_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "strategy_id": "coverage",
                "version": "1.0.0",
                "run_id": "run-1",
                "config_hash": "cfg",
                "files": {"weights_csv": "weights.csv"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    batch = WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)

    assert batch.windows[0].data["weight"].to_list() == [1.0]


def test_compute_cycle_spd_covers_empty_source_and_batched_success_paths() -> None:
    empty = pl.DataFrame(
        schema={
            "date": pl.Datetime("us"),
            "price_usd": pl.Float64,
            "price_usd_source_exists": pl.Boolean,
        }
    )
    empty_features = pl.DataFrame(schema={"date": pl.Datetime("us"), "price_usd": pl.Float64})
    assert prelude_module.compute_cycle_spd(
        empty,
        _uniform_window_weights,
        features_df=empty_features,
        start_date="2024-01-01",
        end_date="2024-01-05",
        validate_weights=True,
    ).is_empty()

    invalid_source = pl.DataFrame(
        {
            "date": pl.Series("date", [None], dtype=pl.Utf8),
            "price_usd": [100.0],
            "price_usd_source_exists": [True],
        }
    )
    assert prelude_module.compute_cycle_spd(
        invalid_source,
        _uniform_window_weights,
        features_df=pl.DataFrame({"date": pl.Series("date", [None], dtype=pl.Utf8), "price_usd": [100.0]}),
        start_date="2024-01-01",
        end_date="2024-01-05",
        validate_weights=True,
    ).is_empty()

    masked_source = _sample_cycle_btc_frame().with_columns(pl.lit(False).alias("price_usd_source_exists"))
    unvalidated = prelude_module.compute_cycle_spd(
        masked_source,
        _uniform_window_weights,
        start_date="2024-01-01",
        end_date="2025-01-05",
        validate_weights=False,
    )
    assert unvalidated.height >= 1

    class GoodBatchStrategy:
        def _compute_window_weights_batch(
            self, full_feat, *, feature_plan, windows, expected_days
        ):
            del full_feat, feature_plan, expected_days
            rows = []
            for i in range(min(2, windows.height)):
                w_start = windows["window_start"][i]
                w_end = windows["window_end"][i]
                dates = pl.datetime_range(w_start, w_end, interval="1d", eager=True)
                n = len(dates)
                rows.append(
                    pl.DataFrame(
                        {
                            "window_start": [w_start] * n,
                            "window_end": [w_end] * n,
                            "date": dates,
                            "weight": [1.0 / n] * n,
                        }
                    )
                )
            return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()

    validated = prelude_module.compute_cycle_spd(
        _sample_cycle_btc_frame(),
        GoodBatchStrategy(),
        start_date="2024-01-01",
        end_date="2025-01-05",
        validate_weights=True,
    )
    assert validated.height >= 1


def test_runner_and_validation_cover_remaining_cache_and_noop_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    runner._release_runtime_cache(clear=False)

    metadata = UniformProposeStrategy().metadata()
    noop = runner._build_noop_daily_result(
        metadata=metadata,
        config=type("Cfg", (), {"mode": "paper", "total_window_budget_usd": 100.0})(),
        run_date="2024-01-01",
        state_store=type("StateStore", (), {"db_path": Path("state.sqlite3")})(),
        existing_run=type(
            "ExistingRun",
            (),
            {
                "payload": {"status": "noop"},
                "order_summary": None,
                "validation_receipt_id": None,
                "data_hash": "",
                "feature_snapshot_hash": "",
                "artifact_path": None,
                "run_key": "run-1",
                "idempotency_hit": True,
                "force_flag": False,
                "message": "noop",
            },
        )(),
        daily_order_receipt_cls=lambda **kwargs: kwargs,
        daily_run_result_cls=lambda **kwargs: kwargs,
    )
    assert noop["order_receipt"] is None

    dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    btc = pl.DataFrame({"date": dates, "price_usd": np.linspace(100.0, 500.0, len(dates))})
    state = _ValidationState(messages=[])
    runner._run_weight_constraint_checks(
        strategy=UniformProposeStrategy(),
        btc_df=btc,
        features_df=btc,
        start_ts=dates[-1],
        end_ts=dates[-1],
        strict_mode=True,
        config=ValidationConfig(strict=True),
        state=state,
    )
    assert state.strict_checks_ok is True

    class _LazyProfileStrategy(BaseStrategy):
        strategy_id = "lazy-profile"
        intent_preference = "profile"

        def build_target_profile_lazy(self, ctx, features_lf):
            del ctx
            return features_lf.select("date", pl.lit(1.0).alias("value"))

        def propose_weight(self, state):
            return state.uniform_weight

    cache, _ = runner._ensure_runtime_cache(_LazyProfileStrategy(), btc)
    monkeypatch.setattr(runner, "_materialize_strategy_features_lazy", lambda *args, **kwargs: btc.lazy())
    weights, mutated = runner._compute_strategy_weights(
        _LazyProfileStrategy(),
        btc_df=btc,
        features_df=btc,
        start_date=dates[0],
        end_date=dates[-1],
        current_date=dates[-1],
        strict_mode=True,
        cache_namespace="base",
    )
    assert mutated is False
    assert cache.profile.compute_weights_calls == 1
    assert cache.weight_frames == {}

    runner_no_cache = StrategyRunner()
    monkeypatch.setattr(
        runner_no_cache,
        "_materialize_strategy_features_lazy",
        lambda *args, **kwargs: btc.lazy(),
    )
    runner_no_cache._compute_strategy_weights(
        _LazyProfileStrategy(),
        btc_df=btc,
        features_df=btc,
        start_date=dates[0],
        end_date=dates[-1],
        current_date=dates[-1],
        strict_mode=True,
        cache_namespace="base",
    )

    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "_compute_strategy_weights",
        lambda *args, **kwargs: (
            pl.DataFrame({"date": dates[:365], "weight": [1.0 / 365.0] * 365}),
            False,
        ),
    )
    state = _ValidationState(messages=[])
    runner._run_weight_constraint_checks(
        strategy=UniformProposeStrategy(),
        btc_df=btc,
        features_df=btc,
        start_ts=dates[0],
        end_ts=dates[-1],
        strict_mode=True,
        config=ValidationConfig(strict=True, max_boundary_hit_rate=1.0),
        state=state,
    )
    assert state.strict_checks_ok is True
    assert any("Strict boundary diagnostics" in message for message in state.messages)

    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "_materialize_strategy_features",
        lambda *args, **kwargs: btc.filter(pl.col("date") <= kwargs["current_date"]),
    )
    monkeypatch.setattr(
        runner,
        "_compute_with_mutation_guard",
        lambda strategy, ctx, strict_mode: (
            pl.DataFrame({"date": [ctx.current_date + dt.timedelta(days=1)], "weight": [1.0]}),
            False,
        ),
    )
    state = _ValidationState(messages=[])
    runner._run_locked_prefix_check(
        strategy=UniformProposeStrategy(),
        btc_df=btc,
        features_df=btc,
        start_ts=dates[0],
        max_window_start=dates[0],
        strict_mode=True,
        state=state,
    )
    assert state.strict_checks_ok is True

    runner = StrategyRunner()
    monkeypatch.setattr("stacksats.runner_validation.block_bootstrap_confidence_interval", lambda *args, **kwargs: type("B", (), {"lower": 1.0, "upper": 2.0})())
    monkeypatch.setattr("stacksats.runner_validation.paired_block_permutation_pvalue", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr("stacksats.runner_validation.build_purged_walk_forward_folds", lambda *args, **kwargs: [])
    state = _ValidationState(messages=[], diagnostics={})
    runner._strict_statistical_checks(
        strategy=UniformProposeStrategy(),
        btc_df=btc_df(days=800),
        features_df=btc_df(days=800),
        start_ts=dt.datetime(2024, 1, 1),
        end_ts=dt.datetime(2026, 12, 31),
        config=ValidationConfig(strict=True),
        state=state,
        backtest_result=type("Result", (), {"spd_table": pl.DataFrame({"dynamic_percentile": [1.0], "uniform_percentile": [1.0], "excess_percentile": [0.0]})})(),
    )
    assert state.strict_checks_ok is True

    backtest_runner = StrategyRunner()
    monkeypatch.setattr(backtest_runner, "_runtime_profile", lambda: None)
    monkeypatch.setattr(backtest_runner, "_release_runtime_cache", lambda clear: None)
    result = backtest_runner.backtest(
        UniformProposeStrategy(),
        BacktestConfig(start_date="2024-01-01", end_date="2025-01-05"),
        btc_df=_sample_cycle_btc_frame(),
    )
    assert result is not None


def test_prelude_internal_spd_helpers_cover_unvalidated_and_batched_fast_paths() -> None:
    btc = _sample_cycle_btc_frame()
    start_ts = dt.datetime(2024, 1, 1)
    end = dt.datetime(2025, 1, 5)
    (
        dataframe,
        price_plan,
        full_feat,
        feature_plan,
        inv_price_frame,
        windows,
    ) = prelude_module._batched_spd_windows(
        btc,
        prelude_module.precompute_features(btc),
        start_ts=start_ts,
        end=end,
    )

    single = prelude_module._compute_cycle_spd_framework(
        dataframe,
        price_plan,
        full_feat,
        feature_plan,
        inv_price_frame,
        windows.head(1),
        lambda frame, **kwargs: _uniform_window_weights(frame),
        validate_weights=False,
    )
    assert single.height == 1

    class GoodBatchStrategy:
        def _compute_window_weights_batch(
            self, full_feat, *, feature_plan, windows, expected_days
        ):
            del full_feat, feature_plan, expected_days
            rows = []
            for i in range(min(1, windows.height)):
                w_start = windows["window_start"][i]
                w_end = windows["window_end"][i]
                dates = pl.datetime_range(w_start, w_end, interval="1d", eager=True)
                n = len(dates)
                rows.append(
                    pl.DataFrame(
                        {
                            "window_start": [w_start] * n,
                            "window_end": [w_end] * n,
                            "date": dates,
                            "weight": [1.0 / n] * n,
                        }
                    )
                )
            return pl.concat(rows, how="vertical_relaxed")

    batched = prelude_module._compute_cycle_spd_batched(
        full_feat,
        feature_plan,
        inv_price_frame,
        windows.head(1),
        GoodBatchStrategy(),
        validate_weights=False,
    )
    assert batched.height == 1

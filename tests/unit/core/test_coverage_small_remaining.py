from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stacksats.animation_data import _parse_window_date, prepare_animation_frame_data
from stacksats.animation_render import _axis_limits, _validate_animation_frame_data
from stacksats.column_map_provider import ColumnMapDataProvider, ColumnMapError
from stacksats.export_weights_runtime import _date_to_str, update_today_weights
from stacksats.model_development_features import precompute_features
from stacksats.plot_mvrv_render import _parse_date
from stacksats.prelude import compute_cycle_spd
from stacksats.runner import StrategyRunner
from stacksats.runner_helpers import build_fold_ranges, weights_match
from stacksats.strategy_time_series import StrategySeriesMetadata, WeightTimeSeries
from stacksats.strategy_time_series_schema import ColumnSpec
from stacksats.strategy_types import BacktestConfig, BaseStrategy, RunDailyConfig
from stacksats.strategies.examples import SimpleZScoreStrategy
from stacksats.strategies.experimental.model_example import ExampleMVRVStrategy
from stacksats.strategies.experimental.model_mvrv_plus import MVRVPlusStrategy


def _meta() -> StrategySeriesMetadata:
    return StrategySeriesMetadata(
        strategy_id="s",
        strategy_version="1.0.0",
        run_id="run",
        config_hash="cfg",
        window_start=dt.datetime(2024, 1, 1),
        window_end=dt.datetime(2024, 1, 3),
    )


def _series_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)],
            "weight": [0.2, 0.3, 0.5],
            "price_usd": [100.0, 101.0, 102.0],
        }
    )


def test_animation_and_plot_helpers_cover_edge_cases() -> None:
    assert _parse_window_date("2024-01-03") == dt.datetime(2024, 1, 3)
    with pytest.raises(ValueError, match="Animation frame data is empty"):
        _validate_animation_frame_data(pl.DataFrame())
    assert _axis_limits(np.array([np.nan]), np.array([np.nan])) == (-1.0, 1.0)
    assert _axis_limits(np.array([2.0]), None) == (1.0, 3.0)
    assert _parse_date("2024-01-01T12:00:00Z") == dt.datetime(2024, 1, 1, 12, 0)
    with pytest.raises(ValueError, match="SPD table contains no valid numeric values"):
        prepare_animation_frame_data(
            pl.DataFrame(
                {
                    "window": [],
                    "dynamic_percentile": [],
                    "uniform_percentile": [],
                    "excess_percentile": [],
                    "dynamic_sats_per_dollar": [],
                    "uniform_sats_per_dollar": [],
                },
                schema={
                    "window": pl.Utf8,
                    "dynamic_percentile": pl.Float64,
                    "uniform_percentile": pl.Float64,
                    "excess_percentile": pl.Float64,
                    "dynamic_sats_per_dollar": pl.Float64,
                    "uniform_sats_per_dollar": pl.Float64,
                },
            )
        )


def test_small_helper_branches_for_provider_export_and_features(tmp_path: Path) -> None:
    utf8_df = pl.DataFrame({"date": ["2024-01-02", "2024-01-01"]})
    normalized = ColumnMapDataProvider._to_daily_date(utf8_df)
    assert list(normalized["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]
    with pytest.raises(ColumnMapError, match="must have a 'date' column"):
        ColumnMapDataProvider._to_daily_date(pl.DataFrame({"x": [1]}))

    assert _date_to_str(dt.datetime(2024, 1, 1, 12, 0)) == "2024-01-01"
    assert _date_to_str("2024-01-02T12:00:00") == "2024-01-02"

    df = pl.DataFrame(
        {
            "day_index": [0],
            "start_date": [dt.datetime(2024, 1, 1)],
            "end_date": [dt.datetime(2024, 1, 10)],
            "date": [dt.datetime(2024, 1, 1)],
            "price_usd": [100.0],
            "weight": [1.0],
        }
    )
    class _Cursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            del query, params

        def fetchone(self):
            return None

    class _Conn:
        def cursor(self):
            return _Cursor()

        def commit(self):
            return None

        def rollback(self):
            return None

    updated = update_today_weights(
        conn=_Conn(),
        df=df,
        today_str="2024-01-01",
        get_current_btc_price_fn=lambda previous_price: 50000.0,
        build_update_rows_fn=lambda frame, current_btc_price=None: [("ok", current_btc_price, frame.height)],
        build_values_sql_fn=lambda rows, include_price=False, sql_quote=None: "VALUES_SQL",
        sql_quote_fn=lambda value: str(value),
    )
    assert updated == 1

    with pytest.raises(KeyError, match="DataFrame must have 'date' column"):
        precompute_features(
            pl.DataFrame({"price_usd": [1.0]}),
            price_col="price_usd",
            mvrv_col="mvrv",
            ma_window=2,
            mvrv_rolling_window=2,
            mvrv_cycle_window=2,
            mvrv_gradient_window=2,
            mvrv_accel_window=2,
            mvrv_zone_deep_value=-2.0,
            mvrv_zone_value=1.0,
            mvrv_zone_caution=2.0,
            mvrv_zone_danger=3.0,
            mvrv_volatility_window=2,
        )


def test_runner_and_prelude_edge_branches(monkeypatch) -> None:
    class _Uniform(BaseStrategy):
        strategy_id = "u"

        def propose_weight(self, state):
            return state.uniform_weight

    runner = StrategyRunner()

    class _FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 5, 12, 0, tzinfo=tz)

    monkeypatch.setattr("stacksats.runner.dt.datetime", _FixedDateTime)
    _, _, run_ts, run_date = runner._parse_run_daily_config(_Uniform(), RunDailyConfig())
    assert run_ts.date().isoformat() == "2024-01-05"
    assert run_ts.tzinfo == dt.timezone.utc
    assert run_date == "2024-01-05"

    btc = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)],
            "price_usd": np.linspace(100.0, 109.0, 10),
        }
    )
    spd = compute_cycle_spd(
        btc,
        lambda feat: pl.DataFrame({"date": feat["date"].head(1), "weight": [1.0]}),
        start_date="2024-01-01",
        end_date="2024-01-05",
        validate_weights=False,
    )
    assert spd.is_empty()

    with pytest.raises(ValueError, match="Backtest end date must be on or after start date"):
        runner.backtest(
            _Uniform(),
            BacktestConfig(start_date="2024-01-05", end_date="2024-01-01"),
            btc_df=pl.DataFrame(
                {
                    "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
                    "price_usd": [100.0, 101.0],
                }
            ),
        )


def test_misc_runtime_branches_and_strategy_edge_defaults() -> None:
    assert weights_match(
        pl.DataFrame({"date": [dt.datetime(2024, 1, 1)]}),
        pl.DataFrame({"date": [dt.datetime(2024, 1, 1)], "weight": [1.0]}),
    ) is False

    series = WeightTimeSeries(
        metadata=_meta(),
        data=_series_frame().with_columns(
            pl.Series("decision_time", [None, None, None], dtype=pl.Datetime)
        ),
        extra_schema=(
            ColumnSpec(
                name="decision_time",
                dtype="datetime64[ns]",
                required=False,
                description="decision time",
                source="strategy",
            ),
        ),
    )
    profile = series.profile()
    assert profile["columns_profile"]["decision_time"]["datetime_min"] is None
    assert profile["columns_profile"]["decision_time"]["datetime_max"] is None

    assert build_fold_ranges(dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1)) == []

    assert SimpleZScoreStrategy().build_target_profile(
        ctx=None,  # type: ignore[arg-type]
        features_df=pl.DataFrame({"date": [dt.datetime(2024, 1, 1)]}),
        signals={},
    )["value"][0] == 0.0

    example = ExampleMVRVStrategy()
    assert example._to_dt("2024-01-01") == dt.datetime(2024, 1, 1)
    assert np.array_equal(example._get_col_array(pl.DataFrame({"date": []}), "missing"), np.zeros(0))
    assert example.transform_features(
        type("Ctx", (), {"features": type("Features", (), {"data": pl.DataFrame(schema={"date": pl.Datetime, "price_usd": pl.Float64})})(), "start_date": "2024-01-02", "end_date": "2024-01-01"})()
    ).is_empty()

    plus = MVRVPlusStrategy()
    assert plus._safe_get(pl.DataFrame({"x": [1.0]}), "missing").to_list() == [0.0]
    assert plus.transform_features(
        type("Ctx", (), {"features": type("Features", (), {"data": pl.DataFrame(schema={"date": pl.Datetime, "price_usd": pl.Float64})})(), "start_date": "2024-01-01", "end_date": "2024-01-02"})()
    ).is_empty()
    outside_window = pl.DataFrame({"date": [dt.datetime(2024, 1, 1)], "price_usd": [100.0]})
    assert plus.transform_features(
        type("Ctx", (), {"features": type("Features", (), {"data": outside_window})(), "start_date": "2024-01-02", "end_date": "2024-01-03"})()
    ).is_empty()

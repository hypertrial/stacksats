from __future__ import annotations

import datetime as dt
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from stacksats.strategies.model_example import ExampleMVRVStrategy, main
from stacksats.strategy_types import (
    BaseStrategy,
    validate_strategy_contract,
    strategy_context_from_features_df,
)


def _base_features(dates: list) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame({
        "date": dates,
        "price_usd": np.linspace(10000.0, 20000.0, n),
        "mvrv": np.linspace(1.0, 2.0, n),
        "price_vs_ma": np.linspace(-0.3, 0.3, n),
        "mvrv_zscore": np.linspace(-2.0, 2.0, n),
        "mvrv_gradient": np.linspace(-1.0, 1.0, n),
        "mvrv_zone": np.tile(np.array([-1.5, 0.0, 1.5]), n // 3 + 1)[:n],
        "mvrv_volatility": np.linspace(0.2, 0.9, n),
        "signal_confidence": np.linspace(0.1, 0.9, n),
        "mvrv_percentile": np.linspace(0.1, 0.9, n),
        "mvrv_acceleration": np.linspace(-0.5, 0.5, n),
        "brk_netflow_fast": np.linspace(-1.0, 1.0, n),
        "brk_netflow_slow": np.linspace(-0.8, 0.8, n),
        "brk_netflow_slope": np.linspace(-0.6, 0.6, n),
        "brk_activity_level": np.linspace(-0.4, 0.4, n),
        "brk_activity_div_fast": np.linspace(-0.4, 0.4, n),
        "brk_activity_div_slow": np.linspace(-0.3, 0.3, n),
        "brk_liquidity_level": np.linspace(-0.2, 0.2, n),
        "brk_liquidity_impulse": np.linspace(-0.2, 0.2, n),
        "brk_exchange_share_level": np.linspace(-0.3, 0.3, n),
        "brk_exchange_share_delta": np.linspace(-0.3, 0.3, n),
        "brk_miner_pressure": np.linspace(-0.2, 0.2, n),
        "brk_hash_momentum": np.linspace(-0.2, 0.2, n),
        "brk_roi30": np.linspace(-0.6, 0.6, n),
        "brk_roi1y": np.linspace(-0.5, 0.5, n),
    })


def test_clean_array_helper() -> None:
    arr = ExampleMVRVStrategy._clean_array(pl.Series([1.0, np.inf, np.nan, -2.0]))
    assert np.array_equal(arr, np.array([1.0, 0.0, 0.0, -2.0]))


def test_strategy_contract_and_required_feature_metadata() -> None:
    strategy = ExampleMVRVStrategy()
    has_propose, has_profile = validate_strategy_contract(strategy)
    assert has_propose is False
    assert has_profile is True
    assert strategy.required_feature_sets() == ("core_model_features_v1", "brk_overlay_v1")
    assert "brk_netflow_fast" in strategy.required_feature_columns()


def test_transform_features_uses_observed_window_and_handles_nan_inf() -> None:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 10),
        interval="1d", eager=True
    ).to_list()
    base = _base_features(dates)
    base = base.with_columns(
        pl.when(pl.col("date") == dates[2]).then(float("inf")).otherwise(pl.col("brk_netflow_fast")).alias("brk_netflow_fast"),
        pl.when(pl.col("date") == dates[3]).then(float("nan")).otherwise(pl.col("brk_netflow_slow")).alias("brk_netflow_slow"),
    )

    strategy = ExampleMVRVStrategy()
    ctx = strategy_context_from_features_df(
        base, dates[1], dates[8], dates[8],
        required_columns=tuple(strategy.required_feature_columns())
    )
    transformed = strategy.transform_features(ctx)
    assert transformed["date"].min() == dates[1]
    assert transformed["date"].max() == dates[8]
    assert np.isfinite(transformed.select(pl.exclude("date")).to_numpy().astype(float)).all()
    assert set(transformed.columns) == set(base.columns)


def test_transform_features_returns_empty_when_window_invalid() -> None:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 5),
        interval="1d", eager=True
    ).to_list()
    strategy = ExampleMVRVStrategy()
    ctx = strategy_context_from_features_df(
        _base_features(dates),
        dates[-1],
        dates[0],
        dates[0],
        required_columns=tuple(strategy.required_feature_columns()),
        as_of_date=None,
    )
    transformed = strategy.transform_features(ctx)
    assert transformed.is_empty()


def test_build_signals_and_target_profile_paths() -> None:
    strategy = ExampleMVRVStrategy()
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 30),
        interval="1d", eager=True
    ).to_list()
    features = _base_features(dates)

    assert strategy.build_signals(ctx=object(), features_df=features) == {}

    empty = strategy.build_target_profile(
        ctx=object(),
        features_df=pl.DataFrame(),
        signals={},
    )
    assert empty.values.is_empty()

    profile = strategy.build_target_profile(ctx=object(), features_df=features, signals={})
    assert profile.values.height == len(dates)
    assert np.isfinite(profile.values["value"].to_numpy().astype(float)).all()

    slim = features.drop(
        ["mvrv_percentile", "mvrv_acceleration", "mvrv_volatility", "signal_confidence"]
    )
    profile_slim = strategy.build_target_profile(ctx=object(), features_df=slim, signals={})
    assert profile_slim.values.height == len(dates)
    assert np.isfinite(profile_slim.values["value"].to_numpy().astype(float)).all()


def test_main_executes_and_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    class _FakeValidation:
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class _FakeBacktest:
        strategy_id = "example-mvrv"
        strategy_version = "4.2.0"
        run_id = "run-main"

        @staticmethod
        def summary() -> str:
            return "Score: 50.0%"

        @staticmethod
        def plot(output_dir: str):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        @staticmethod
        def to_json(path: Path):
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ExampleMVRVStrategy, "validate", lambda self, config: _FakeValidation())
    monkeypatch.setattr(ExampleMVRVStrategy, "backtest", lambda self, config: _FakeBacktest())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_example.py",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-10",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    out = tmp_path / "example-mvrv" / "4.2.0" / "run-main" / "backtest_result.json"
    assert out.exists()


def test_module_dunder_main_executes(monkeypatch, tmp_path: Path) -> None:
    class _FakeValidation:
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class _FakeBacktest:
        strategy_id = "example-mvrv"
        strategy_version = "4.2.0"
        run_id = "run-dunder"

        @staticmethod
        def summary() -> str:
            return "Score: 55.0%"

        @staticmethod
        def plot(output_dir: str):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        @staticmethod
        def to_json(path: Path):
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(BaseStrategy, "validate", lambda self, config: _FakeValidation())
    monkeypatch.setattr(BaseStrategy, "backtest", lambda self, config: _FakeBacktest())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_example.py",
            "--output-dir",
            str(tmp_path),
        ],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'.*' found in sys.modules after import of package '.*'",
            category=RuntimeWarning,
        )
        runpy.run_module("stacksats.strategies.model_example", run_name="__main__")

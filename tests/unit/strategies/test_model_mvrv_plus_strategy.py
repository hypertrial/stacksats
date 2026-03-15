from __future__ import annotations

import datetime as dt
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from stacksats.strategies import model_mvrv_plus
from stacksats.strategies.model_mvrv_plus import MVRVPlusStrategy, main
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
        "price_vs_ma": np.linspace(-0.5, 0.5, n),
        "mvrv_zscore": np.linspace(-2.0, 2.0, n),
        "mvrv_zone": np.tile(np.array([-1.5, 0.0, 1.5]), n // 3 + 1)[:n],
        "mvrv_volatility": np.linspace(0.2, 0.95, n),
        "signal_confidence": np.linspace(0.1, 0.9, n),
        "brk_netflow": np.linspace(-0.7, 0.7, n),
        "brk_exchange_share": np.linspace(-0.6, 0.6, n),
        "brk_exchange_share_delta": np.linspace(-0.5, 0.5, n),
        "brk_activity_div": np.linspace(-0.4, 0.4, n),
        "brk_roi_context": np.linspace(-0.6, 0.6, n),
        "brk_liquidity_impulse": np.linspace(-0.3, 0.3, n),
        "brk_miner_pressure": np.linspace(-0.4, 0.4, n),
        "brk_hash_momentum": np.linspace(-0.4, 0.4, n),
    })


def test_safe_array_and_adaptive_ewma_paths() -> None:
    arr = MVRVPlusStrategy._safe_array(pl.Series([1.0, None, float("inf")]), fill=-1.0)
    assert np.array_equal(arr, np.array([1.0, -1.0, -1.0]))

    assert MVRVPlusStrategy._adaptive_ewma(np.array([]), np.array([])).size == 0
    smooth = MVRVPlusStrategy._adaptive_ewma(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 0.2, 0.8]),
    )
    assert np.isfinite(smooth).all()
    assert smooth.shape == (3,)


def test_strategy_contract_and_required_feature_metadata() -> None:
    strategy = MVRVPlusStrategy()
    has_propose, has_profile = validate_strategy_contract(strategy)
    assert has_propose is False
    assert has_profile is True
    assert strategy.required_feature_sets() == ("core_model_features_v1", "brk_overlay_v1")
    assert "brk_netflow" in strategy.required_feature_columns()


def test_transform_features_and_build_target_profile(monkeypatch) -> None:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1) + dt.timedelta(days=39),
        interval="1d", eager=True
    ).to_list()
    features = _base_features(dates)
    strategy = MVRVPlusStrategy()

    empty_ctx = strategy_context_from_features_df(
        features,
        dates[-1] + dt.timedelta(days=1),
        dates[-1],
        dates[-1],
        required_columns=(),
        as_of_date=None,
    )
    transformed_empty = strategy.transform_features(empty_ctx)
    assert transformed_empty.is_empty()

    ctx = strategy_context_from_features_df(
        features,
        dates[0],
        dates[-1],
        dates[-1],
        required_columns=(),
    )
    transformed = strategy.transform_features(ctx)
    assert "plus_vol21" in transformed.columns
    assert "plus_drawdown90" in transformed.columns
    assert "brk_netflow" in transformed.columns
    assert np.isfinite(transformed.select(pl.exclude("date")).to_numpy().astype(float)).all()

    def _fake_preference(features_df, start_date, end_date):
        return pl.DataFrame({
            "date": features_df["date"],
            "preference": np.linspace(-0.5, 0.5, features_df.height),
        })

    monkeypatch.setattr(model_mvrv_plus, "compute_preference_scores", _fake_preference)
    profile = strategy.build_target_profile(ctx=ctx, features_df=transformed, signals={})
    assert profile.values.height == transformed.height
    assert np.isfinite(profile.values["value"].to_numpy().astype(float)).all()

    empty_profile = strategy.build_target_profile(
        ctx=ctx, features_df=pl.DataFrame(), signals={}
    )
    assert empty_profile.values.is_empty()


def test_transform_features_is_collision_safe_with_provider_columns() -> None:
    dates = pl.datetime_range(
        dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 30),
        interval="1d", eager=True
    ).to_list()
    strategy = MVRVPlusStrategy()
    features = _base_features(dates)
    features = features.with_columns(
        pl.when(pl.col("date") == dates[0]).then(1.23).otherwise(float("nan")).alias("plus_vol21"),
        pl.when(pl.col("date") == dates[1]).then(-0.45).otherwise(float("nan")).alias("plus_drawdown90"),
    )
    ctx = strategy_context_from_features_df(
        features,
        dates[0],
        dates[-1],
        dates[-1],
        required_columns=(),
    )
    transformed = strategy.transform_features(ctx)
    assert "brk_exchange_share" in transformed.columns
    assert "plus_vol21" in transformed.columns
    row0 = transformed.filter(pl.col("date") == dates[0]).row(0, named=True)
    row1 = transformed.filter(pl.col("date") == dates[1]).row(0, named=True)
    assert abs(float(row0["plus_vol21"]) - 1.23) < 1e-9
    assert abs(float(row1["plus_drawdown90"]) - (-0.45)) < 1e-9


def test_main_executes(monkeypatch, tmp_path: Path) -> None:
    class _FakeValidation:
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class _FakeBacktest:
        strategy_id = "mvrv-plus"
        strategy_version = "0.1.0"
        run_id = "run-main"

        @staticmethod
        def summary() -> str:
            return "Score: 55.0%"

        @staticmethod
        def plot(output_dir: str):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        @staticmethod
        def to_json(path: Path):
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(MVRVPlusStrategy, "validate", lambda self, config: _FakeValidation())
    monkeypatch.setattr(MVRVPlusStrategy, "backtest", lambda self, config: _FakeBacktest())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_mvrv_plus.py",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    out = tmp_path / "mvrv-plus" / "0.1.0" / "run-main" / "backtest_result.json"
    assert out.exists()


def test_module_dunder_main_executes(monkeypatch, tmp_path: Path) -> None:
    class _FakeValidation:
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class _FakeBacktest:
        strategy_id = "mvrv-plus"
        strategy_version = "0.1.0"
        run_id = "run-dunder"

        @staticmethod
        def summary() -> str:
            return "Score: 56.0%"

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
            "model_mvrv_plus.py",
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
        runpy.run_module("stacksats.strategies.model_mvrv_plus", run_name="__main__")

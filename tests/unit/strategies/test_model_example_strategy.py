from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from stacksats.strategies.model_example import ExampleMVRVStrategy, main
from stacksats.strategy_types import BaseStrategy, validate_strategy_contract


def _base_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(10000.0, 20000.0, n),
            "CapMVRVCur": np.linspace(1.0, 2.0, n),
            "price_vs_ma": np.linspace(-0.3, 0.3, n),
            "mvrv_zscore": np.linspace(-2.0, 2.0, n),
            "mvrv_gradient": np.linspace(-1.0, 1.0, n),
            "mvrv_zone": np.tile(np.array([-1.5, 0.0, 1.5]), n // 3 + 1)[:n],
            "mvrv_volatility": np.linspace(0.2, 0.9, n),
            "signal_confidence": np.linspace(0.1, 0.9, n),
            "mvrv_percentile": np.linspace(0.1, 0.9, n),
            "mvrv_acceleration": np.linspace(-0.5, 0.5, n),
            "cm_netflow_fast": np.linspace(-1.0, 1.0, n),
            "cm_netflow_slow": np.linspace(-0.8, 0.8, n),
            "cm_netflow_slope": np.linspace(-0.6, 0.6, n),
            "cm_activity_level": np.linspace(-0.4, 0.4, n),
            "cm_activity_div_fast": np.linspace(-0.4, 0.4, n),
            "cm_activity_div_slow": np.linspace(-0.3, 0.3, n),
            "cm_liquidity_level": np.linspace(-0.2, 0.2, n),
            "cm_liquidity_impulse": np.linspace(-0.2, 0.2, n),
            "cm_exchange_share_level": np.linspace(-0.3, 0.3, n),
            "cm_exchange_share_delta": np.linspace(-0.3, 0.3, n),
            "cm_miner_pressure": np.linspace(-0.2, 0.2, n),
            "cm_hash_momentum": np.linspace(-0.2, 0.2, n),
            "cm_roi30": np.linspace(-0.6, 0.6, n),
            "cm_roi1y": np.linspace(-0.5, 0.5, n),
        },
        index=index,
    )


def test_clean_array_helper() -> None:
    arr = ExampleMVRVStrategy._clean_array(pd.Series([1.0, np.inf, np.nan, -2.0]))
    assert np.array_equal(arr, np.array([1.0, 0.0, 0.0, -2.0]))


def test_strategy_contract_and_required_feature_metadata() -> None:
    strategy = ExampleMVRVStrategy()
    has_propose, has_profile = validate_strategy_contract(strategy)
    assert has_propose is False
    assert has_profile is True
    assert strategy.required_feature_sets() == ("core_model_features_v1", "coinmetrics_overlay_v1")
    assert "cm_netflow_fast" in strategy.required_feature_columns()


def test_transform_features_uses_observed_window_and_handles_nan_inf() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    base = _base_features(idx)
    base.loc[idx[2], "cm_netflow_fast"] = np.inf
    base.loc[idx[3], "cm_netflow_slow"] = np.nan

    strategy = ExampleMVRVStrategy()
    ctx = type(
        "Ctx",
        (),
        {"features_df": base, "start_date": idx[1], "end_date": idx[8]},
    )()
    transformed = strategy.transform_features(ctx)
    assert transformed.index.min() == idx[1]
    assert transformed.index.max() == idx[8]
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()
    assert list(transformed.columns) == list(base.columns)


def test_transform_features_returns_empty_when_window_invalid() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    strategy = ExampleMVRVStrategy()
    ctx = type(
        "Ctx",
        (),
        {"features_df": _base_features(idx), "start_date": idx[-1], "end_date": idx[0]},
    )()
    transformed = strategy.transform_features(ctx)
    assert transformed.empty


def test_build_signals_and_target_profile_paths() -> None:
    strategy = ExampleMVRVStrategy()
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    features = _base_features(idx)

    assert strategy.build_signals(ctx=object(), features_df=features) == {}

    empty = strategy.build_target_profile(ctx=object(), features_df=pd.DataFrame(), signals={})
    assert empty.values.empty

    profile = strategy.build_target_profile(ctx=object(), features_df=features, signals={})
    assert len(profile.values) == len(idx)
    assert np.isfinite(profile.values.to_numpy(dtype=float)).all()

    # missing optional columns path
    slim = features.drop(
        columns=["mvrv_percentile", "mvrv_acceleration", "mvrv_volatility", "signal_confidence"]
    )
    profile_slim = strategy.build_target_profile(ctx=object(), features_df=slim, signals={})
    assert len(profile_slim.values) == len(idx)
    assert np.isfinite(profile_slim.values.to_numpy(dtype=float)).all()


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

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from stacksats.strategies.model_example import ExampleMVRVStrategy, main
from stacksats.strategy_types import BaseStrategy


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
        },
        index=index,
    )


def _coinmetrics_csv(path: Path, *, include_optional: bool) -> None:
    base = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=30, freq="D"),
            "PriceUSD": np.linspace(40000.0, 45000.0, 30),
        }
    )
    if include_optional:
        base["CapMrktCurUSD"] = np.linspace(1e12, 1.2e12, 30)
        base["FlowInExUSD"] = np.linspace(1e9, 1.3e9, 30)
        base["FlowOutExUSD"] = np.linspace(0.8e9, 1.1e9, 30)
        base["AdrActCnt"] = np.linspace(500000, 700000, 30)
        base["TxCnt"] = np.linspace(200000, 240000, 30)
        base["FeeTotNtv"] = np.linspace(100, 120, 30)
        base["volume_reported_spot_usd_1d"] = np.linspace(1e10, 1.2e10, 30)
        base["SplyExNtv"] = np.linspace(2e6, 2.1e6, 30)
        base["SplyCur"] = np.linspace(19e6, 19.2e6, 30)
        base["IssTotUSD"] = np.linspace(1e8, 1.1e8, 30)
        base["HashRate"] = np.linspace(350e6, 360e6, 30)
        base["ROI30d"] = np.linspace(-0.1, 0.2, 30)
        base["ROI1yr"] = np.linspace(-0.4, 0.8, 30)
    base.to_csv(path, index=False)


def test_clean_array_and_rolling_zscore_helpers() -> None:
    arr = ExampleMVRVStrategy._clean_array(pd.Series([1.0, np.inf, np.nan, -2.0]))
    assert np.array_equal(arr, np.array([1.0, 0.0, 0.0, -2.0]))

    z = ExampleMVRVStrategy._rolling_zscore(pd.Series(np.linspace(1.0, 60.0, 60)), 30)
    assert len(z) == 60
    assert np.isfinite(z.to_numpy(dtype=float)).all()


def test_to_numeric_only_converts_known_columns() -> None:
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    ExampleMVRVStrategy._to_numeric(df, ["a", "missing"])
    assert list(df["a"]) == [1, 2]
    assert list(df["b"]) == ["x", "y"]


def test_load_coinmetrics_features_handles_missing_and_invalid_time(tmp_path: Path) -> None:
    strategy = ExampleMVRVStrategy()
    strategy.coinmetrics_cache_path = tmp_path / "missing.csv"
    assert strategy._load_coinmetrics_features().empty

    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"not_time": [1, 2]}).to_csv(invalid, index=False)
    strategy = ExampleMVRVStrategy()
    strategy.coinmetrics_cache_path = invalid
    assert strategy._load_coinmetrics_features().empty


def test_load_coinmetrics_features_full_and_minimal_paths(tmp_path: Path) -> None:
    full_csv = tmp_path / "full.csv"
    _coinmetrics_csv(full_csv, include_optional=True)
    strategy_full = ExampleMVRVStrategy()
    strategy_full.coinmetrics_cache_path = full_csv
    full = strategy_full._load_coinmetrics_features()
    assert not full.empty
    assert "cm_netflow_fast" in full.columns
    assert "cm_hash_momentum" in full.columns

    # cache-hit branch
    cached = strategy_full._load_coinmetrics_features()
    assert cached is full

    minimal_csv = tmp_path / "minimal.csv"
    _coinmetrics_csv(minimal_csv, include_optional=False)
    strategy_minimal = ExampleMVRVStrategy()
    strategy_minimal.coinmetrics_cache_path = minimal_csv
    minimal = strategy_minimal._load_coinmetrics_features()
    assert not minimal.empty
    assert np.isfinite(minimal.to_numpy(dtype=float)).all()


def test_load_coinmetrics_features_activity_without_price_and_build_signals(tmp_path: Path) -> None:
    csv_path = tmp_path / "activity_only.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=20, freq="D"),
            "AdrActCnt": np.linspace(500000, 600000, 20),
            "TxCnt": np.linspace(200000, 210000, 20),
            "FeeTotNtv": np.linspace(100, 110, 20),
        }
    ).to_csv(csv_path, index=False)
    strategy = ExampleMVRVStrategy()
    strategy.coinmetrics_cache_path = csv_path
    features = strategy._load_coinmetrics_features()
    assert not features.empty
    assert np.isfinite(features[["cm_activity_div_fast", "cm_activity_div_slow"]].to_numpy(dtype=float)).all()
    assert strategy.build_signals(ctx=object(), features_df=features) == {}


def test_transform_features_without_and_with_coinmetrics(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    base = _base_features(idx)

    strategy_no_cm = ExampleMVRVStrategy()
    strategy_no_cm.coinmetrics_cache_path = tmp_path / "does-not-exist.csv"
    ctx = type(
        "Ctx",
        (),
        {"features_df": base, "start_date": idx[0], "end_date": idx[-1]},
    )()
    transformed_no_cm = strategy_no_cm.transform_features(ctx)
    assert list(transformed_no_cm.columns) == list(base.columns)

    cm_csv = tmp_path / "cm.csv"
    _coinmetrics_csv(cm_csv, include_optional=True)
    strategy_cm = ExampleMVRVStrategy()
    strategy_cm.coinmetrics_cache_path = cm_csv
    transformed_cm = strategy_cm.transform_features(ctx)
    assert "cm_netflow_fast" in transformed_cm.columns
    assert transformed_cm.shape[0] == base.shape[0]


def test_build_target_profile_empty_and_populated() -> None:
    strategy = ExampleMVRVStrategy()
    empty = strategy.build_target_profile(ctx=object(), features_df=pd.DataFrame(), signals={})
    assert empty.values.empty

    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    features = _base_features(idx)
    profile = strategy.build_target_profile(ctx=object(), features_df=features, signals={})
    assert len(profile.values) == len(idx)
    assert np.isfinite(profile.values.to_numpy(dtype=float)).all()

    # missing optional columns branch
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

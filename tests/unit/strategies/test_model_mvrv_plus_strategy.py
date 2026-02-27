from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stacksats.strategies import model_mvrv_plus
from stacksats.strategies.model_mvrv_plus import MVRVPlusStrategy, _load_coinmetrics_csv, main


def _base_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "date": index,
            "PriceUSD_coinmetrics": np.linspace(10000.0, 20000.0, n),
            "CapMVRVCur": np.linspace(1.0, 2.0, n),
            "price_vs_ma": np.linspace(-0.5, 0.5, n),
            "mvrv_zscore": np.linspace(-2.0, 2.0, n),
            "mvrv_zone": np.tile(np.array([-1.5, 0.0, 1.5]), n // 3 + 1)[:n],
            "mvrv_volatility": np.linspace(0.2, 0.95, n),
            "signal_confidence": np.linspace(0.1, 0.9, n),
        },
        index=index,
    )


def _coinmetrics_csv(path: Path, *, include_optional: bool) -> None:
    base = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=60, freq="D"),
            "PriceUSD": np.linspace(40000.0, 45000.0, 60),
        }
    )
    if include_optional:
        base["CapMrktCurUSD"] = np.linspace(1e12, 1.2e12, 60)
        base["FlowInExUSD"] = np.linspace(1e9, 1.3e9, 60)
        base["FlowOutExUSD"] = np.linspace(0.8e9, 1.1e9, 60)
        base["SplyExNtv"] = np.linspace(2e6, 2.1e6, 60)
        base["SplyCur"] = np.linspace(19e6, 19.2e6, 60)
        base["AdrActCnt"] = np.linspace(500000, 700000, 60)
        base["TxCnt"] = np.linspace(200000, 240000, 60)
        base["TxTfrCnt"] = np.linspace(300000, 360000, 60)
        base["ROI30d"] = np.linspace(-0.2, 0.2, 60)
        base["ROI1yr"] = np.linspace(-0.6, 0.8, 60)
        base["volume_reported_spot_usd_1d"] = np.linspace(1e10, 1.3e10, 60)
        base["IssTotUSD"] = np.linspace(1e8, 1.2e8, 60)
        base["HashRate"] = np.linspace(350e6, 365e6, 60)
    base.to_csv(path, index=False)


def test_safe_array_and_adaptive_ewma_paths() -> None:
    arr = MVRVPlusStrategy._safe_array(pd.Series(["1", None, np.inf]), fill=-1.0)
    assert np.array_equal(arr, np.array([1.0, -1.0, -1.0]))

    assert MVRVPlusStrategy._adaptive_ewma(np.array([]), np.array([])).size == 0
    smooth = MVRVPlusStrategy._adaptive_ewma(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 0.2, 0.8]),
    )
    assert np.isfinite(smooth).all()
    assert smooth.shape == (3,)


def test_load_coinmetrics_overlays_missing_file_and_missing_time(tmp_path: Path) -> None:
    strategy = MVRVPlusStrategy()
    strategy.coinmetrics_csv = tmp_path / "missing.csv"
    assert strategy._load_coinmetrics_overlays().empty

    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"not_time": [1]}).to_csv(invalid, index=False)
    strategy_invalid = MVRVPlusStrategy()
    strategy_invalid.coinmetrics_csv = invalid
    assert strategy_invalid._load_coinmetrics_overlays().empty


def test_load_coinmetrics_overlays_full_and_minimal(tmp_path: Path) -> None:
    full = tmp_path / "full.csv"
    _coinmetrics_csv(full, include_optional=True)
    s_full = MVRVPlusStrategy()
    s_full.coinmetrics_csv = full
    overlays_full = s_full._load_coinmetrics_overlays()
    assert not overlays_full.empty
    assert "cm_miner_pressure" in overlays_full.columns
    assert "cm_hash_momentum" in overlays_full.columns
    assert s_full._load_coinmetrics_overlays() is overlays_full

    minimal = tmp_path / "minimal.csv"
    _coinmetrics_csv(minimal, include_optional=False)
    s_minimal = MVRVPlusStrategy()
    s_minimal.coinmetrics_csv = minimal
    overlays_minimal = s_minimal._load_coinmetrics_overlays()
    assert not overlays_minimal.empty
    assert np.isfinite(overlays_minimal.to_numpy(dtype=float)).all()


def test_load_coinmetrics_overlays_activity_without_price_uses_zero_momentum(tmp_path: Path) -> None:
    csv_path = tmp_path / "activity_only.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=20, freq="D"),
            "AdrActCnt": np.linspace(500000, 700000, 20),
            "TxCnt": np.linspace(200000, 240000, 20),
            "TxTfrCnt": np.linspace(300000, 360000, 20),
        }
    ).to_csv(csv_path, index=False)
    strategy = MVRVPlusStrategy()
    strategy.coinmetrics_csv = csv_path
    overlays = strategy._load_coinmetrics_overlays()
    assert not overlays.empty
    assert np.isfinite(overlays["cm_activity_div"].to_numpy(dtype=float)).all()


def test_transform_features_and_build_target_profile(monkeypatch, tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    features = _base_features(idx)
    strategy = MVRVPlusStrategy()

    # Empty window branch.
    empty_ctx = type(
        "Ctx",
        (),
        {"features_df": features, "start_date": idx[-1] + pd.Timedelta(days=1), "end_date": idx[-1]},
    )()
    transformed_empty = strategy.transform_features(empty_ctx)
    assert transformed_empty.empty

    cm_csv = tmp_path / "cm.csv"
    _coinmetrics_csv(cm_csv, include_optional=True)
    strategy.coinmetrics_csv = cm_csv
    ctx = type("Ctx", (), {"features_df": features, "start_date": idx[0], "end_date": idx[-1]})()
    transformed = strategy.transform_features(ctx)
    assert "plus_vol21" in transformed.columns
    assert "cm_netflow" in transformed.columns

    monkeypatch.setattr(
        model_mvrv_plus,
        "compute_preference_scores",
        lambda features_df, start_date, end_date: pd.Series(
            np.linspace(-0.5, 0.5, len(features_df.index)), index=features_df.index, dtype=float
        ),
    )
    profile = strategy.build_target_profile(ctx=ctx, features_df=transformed, signals={})
    assert len(profile.values) == len(transformed.index)
    assert np.isfinite(profile.values.to_numpy(dtype=float)).all()

    empty_profile = strategy.build_target_profile(
        ctx=ctx, features_df=pd.DataFrame(), signals={}
    )
    assert empty_profile.values.empty


def test_load_coinmetrics_csv_error_and_success_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        _load_coinmetrics_csv(str(missing))

    no_time = tmp_path / "no_time.csv"
    pd.DataFrame({"PriceUSD": [1.0]}).to_csv(no_time, index=False)
    with pytest.raises(ValueError, match="time"):
        _load_coinmetrics_csv(str(no_time))

    no_price = tmp_path / "no_price.csv"
    pd.DataFrame({"time": ["2024-01-01"]}).to_csv(no_price, index=False)
    with pytest.raises(ValueError, match="PriceUSD"):
        _load_coinmetrics_csv(str(no_price))

    valid = tmp_path / "valid.csv"
    pd.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "PriceUSD": ["40000", "40100", "40200"],
        }
    ).to_csv(valid, index=False)
    parsed = _load_coinmetrics_csv(str(valid))
    assert "PriceUSD_coinmetrics" in parsed.columns
    assert parsed.index.is_monotonic_increasing
    assert len(parsed) == 2


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

    csv_path = tmp_path / "coinmetrics.csv"
    _coinmetrics_csv(csv_path, include_optional=False)

    monkeypatch.setattr(
        model_mvrv_plus.StrategyRunner,
        "validate",
        lambda self, strategy, config, btc_df: _FakeValidation(),
    )
    monkeypatch.setattr(
        model_mvrv_plus.StrategyRunner,
        "backtest",
        lambda self, strategy, config, btc_df: _FakeBacktest(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_mvrv_plus.py",
            "--coinmetrics-csv",
            str(csv_path),
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    out = tmp_path / "mvrv-plus" / "0.1.0" / "run-main" / "backtest_result.json"
    assert out.exists()


def test_main_invalid_end_date_raises(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "coinmetrics.csv"
    _coinmetrics_csv(csv_path, include_optional=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_mvrv_plus.py",
            "--coinmetrics-csv",
            str(csv_path),
            "--end-date",
            "not-a-date",
        ],
    )
    with pytest.raises(ValueError, match="Invalid --end-date"):
        main()


def test_main_nat_end_date_raises(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "coinmetrics.csv"
    _coinmetrics_csv(csv_path, include_optional=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_mvrv_plus.py",
            "--coinmetrics-csv",
            str(csv_path),
            "--end-date",
            "NaT",
        ],
    )
    with pytest.raises(ValueError, match="Invalid --end-date"):
        main()


def test_main_valid_end_date_executes(monkeypatch, tmp_path: Path) -> None:
    class _FakeValidation:
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class _FakeBacktest:
        strategy_id = "mvrv-plus"
        strategy_version = "0.1.0"
        run_id = "run-valid-end-date"

        @staticmethod
        def summary() -> str:
            return "Score: 57.0%"

        @staticmethod
        def plot(output_dir: str):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        @staticmethod
        def to_json(path: Path):
            path.write_text("{}", encoding="utf-8")

    csv_path = tmp_path / "coinmetrics.csv"
    _coinmetrics_csv(csv_path, include_optional=False)
    monkeypatch.setattr(
        model_mvrv_plus.StrategyRunner,
        "validate",
        lambda self, strategy, config, btc_df: _FakeValidation(),
    )
    monkeypatch.setattr(
        model_mvrv_plus.StrategyRunner,
        "backtest",
        lambda self, strategy, config, btc_df: _FakeBacktest(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_mvrv_plus.py",
            "--coinmetrics-csv",
            str(csv_path),
            "--end-date",
            "2024-01-20",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()


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

    csv_path = tmp_path / "coinmetrics.csv"
    _coinmetrics_csv(csv_path, include_optional=False)
    monkeypatch.setattr(
        "stacksats.runner.StrategyRunner.validate",
        lambda self, strategy, config, btc_df: _FakeValidation(),
    )
    monkeypatch.setattr(
        "stacksats.runner.StrategyRunner.backtest",
        lambda self, strategy, config, btc_df: _FakeBacktest(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_mvrv_plus.py",
            "--coinmetrics-csv",
            str(csv_path),
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

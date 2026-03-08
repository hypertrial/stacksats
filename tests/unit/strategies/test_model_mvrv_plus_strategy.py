from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from stacksats.strategies import model_mvrv_plus
from stacksats.strategies.model_mvrv_plus import MVRVPlusStrategy, main
from stacksats.strategy_types import BaseStrategy, validate_strategy_contract


def _base_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "date": index,
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
        },
        index=index,
    )


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


def test_strategy_contract_and_required_feature_metadata() -> None:
    strategy = MVRVPlusStrategy()
    has_propose, has_profile = validate_strategy_contract(strategy)
    assert has_propose is False
    assert has_profile is True
    assert strategy.required_feature_sets() == ("core_model_features_v1", "brk_overlay_v1")
    assert "plus_vol21" in strategy.required_feature_columns()


def test_transform_features_and_build_target_profile(monkeypatch) -> None:
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

    # Normal branch with provider-supplied overlay columns already present.
    ctx = type("Ctx", (), {"features_df": features, "start_date": idx[0], "end_date": idx[-1]})()
    transformed = strategy.transform_features(ctx)
    assert "plus_vol21" in transformed.columns
    assert "plus_drawdown90" in transformed.columns
    assert "brk_netflow" in transformed.columns
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()

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


def test_transform_features_is_collision_safe_with_provider_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    strategy = MVRVPlusStrategy()
    features = _base_features(idx)
    features["plus_vol21"] = np.nan
    features.loc[idx[0], "plus_vol21"] = 1.23
    features["plus_drawdown90"] = np.nan
    features.loc[idx[1], "plus_drawdown90"] = -0.45
    ctx = type("Ctx", (), {"features_df": features, "start_date": idx[0], "end_date": idx[-1]})()
    transformed = strategy.transform_features(ctx)
    assert transformed.index.equals(idx)
    assert "brk_exchange_share" in transformed.columns
    assert "plus_vol21" in transformed.columns
    assert transformed.loc[idx[0], "plus_vol21"] == 1.23
    assert transformed.loc[idx[1], "plus_drawdown90"] == -0.45


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

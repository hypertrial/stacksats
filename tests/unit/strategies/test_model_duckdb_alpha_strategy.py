from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from stacksats.strategies.model_duckdb_alpha import DuckDBAlphaStrategy, main
from stacksats.strategies.model_mvrv_plus import MVRVPlusStrategy
from stacksats.strategy_types import BaseStrategy, StrategyContext, validate_strategy_contract


def _feature_frame(strategy: DuckDBAlphaStrategy, index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = strategy.required_feature_columns()
    frame = pd.DataFrame(index=index)
    for idx, column in enumerate(cols):
        frame[column] = np.linspace(-1.0 + (idx * 0.01), 1.0 + (idx * 0.01), len(index))
    frame["PriceUSD_coinmetrics"] = np.linspace(10000.0, 20000.0, len(index))
    frame["mvrv_zone"] = np.tile(np.array([-1.2, 0.0, 1.3]), len(index) // 3 + 1)[: len(index)]
    frame["mvrv_volatility"] = np.linspace(0.25, 0.95, len(index))
    frame["signal_confidence"] = np.linspace(0.1, 0.9, len(index))
    return frame


def test_duckdb_alpha_strategy_contract_and_metadata() -> None:
    strategy = DuckDBAlphaStrategy()
    has_propose, has_profile = validate_strategy_contract(strategy)
    assert has_propose is False
    assert has_profile is True
    assert "duckdb_analytics_factors_v1" in strategy.required_feature_sets()
    assert len(strategy.artifact_feature_columns) > 0


def test_duckdb_alpha_profile_overlay(monkeypatch) -> None:
    strategy = DuckDBAlphaStrategy()
    index = pd.date_range("2024-01-01", periods=45, freq="D")
    frame = _feature_frame(strategy, index)
    ctx = StrategyContext(
        features_df=frame,
        start_date=index[0],
        end_date=index[-1],
        current_date=index[-1],
    )
    transformed = strategy.transform_features(ctx)

    monkeypatch.setattr(
        "stacksats.strategies.model_mvrv_plus.compute_preference_scores",
        lambda features_df, start_date, end_date: pd.Series(
            np.linspace(-0.3, 0.3, len(features_df.index)),
            index=features_df.index,
            dtype=float,
        ),
    )
    profile = strategy.build_target_profile(ctx, transformed, {})
    base = MVRVPlusStrategy().build_target_profile(ctx, transformed, {})
    assert len(profile.values) == len(index)
    assert np.isfinite(profile.values.to_numpy(dtype=float)).all()
    assert not np.allclose(
        profile.values.to_numpy(dtype=float),
        base.values.to_numpy(dtype=float),
    )

    empty_profile = strategy.build_target_profile(ctx, pd.DataFrame(), {})
    assert empty_profile.values.empty


def test_duckdb_alpha_main_executes(monkeypatch, tmp_path: Path) -> None:
    class _FakeValidation:
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class _FakeBacktest:
        strategy_id = "duckdb-alpha"
        strategy_version = "0.1.0"
        run_id = "run-main"

        @staticmethod
        def summary() -> str:
            return "Score: 70.0%"

        @staticmethod
        def plot(output_dir: str):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        @staticmethod
        def to_json(path: Path):
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(DuckDBAlphaStrategy, "validate", lambda self, config: _FakeValidation())
    monkeypatch.setattr(DuckDBAlphaStrategy, "backtest", lambda self, config: _FakeBacktest())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_duckdb_alpha.py",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    out = tmp_path / "duckdb-alpha" / "0.1.0" / "run-main" / "backtest_result.json"
    assert out.exists()


def test_duckdb_alpha_module_dunder_main_executes(monkeypatch, tmp_path: Path) -> None:
    class _FakeValidation:
        messages = ["ok"]

        @staticmethod
        def summary() -> str:
            return "Validation PASSED"

    class _FakeBacktest:
        strategy_id = "duckdb-alpha"
        strategy_version = "0.1.0"
        run_id = "run-dunder"

        @staticmethod
        def summary() -> str:
            return "Score: 71.0%"

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
            "model_duckdb_alpha.py",
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
        runpy.run_module("stacksats.strategies.model_duckdb_alpha", run_name="__main__")

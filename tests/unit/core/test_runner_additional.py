from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.runner import StrategyRunner, WeightValidationError
from stacksats.strategy_types import BacktestConfig, BaseStrategy, ExportConfig, ValidationConfig


class _UniformProposeStrategy(BaseStrategy):
    strategy_id = "runner-uniform"
    version = "1.0.0"

    def propose_weight(self, state):
        return state.uniform_weight


class _MutatingProposeStrategy(BaseStrategy):
    strategy_id = "runner-mutating"
    version = "1.0.0"

    def transform_features(self, ctx):
        # Intentional contract violation for strict-mode guard coverage.
        ctx.features_df["__mutated__"] = 1.0
        return ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()

    def propose_weight(self, state):
        return state.uniform_weight


class _RandomProposeStrategy(BaseStrategy):
    strategy_id = "runner-random"
    version = "1.0.0"

    def propose_weight(self, state):
        rng = np.random.default_rng()
        return float(rng.uniform(0.0, state.uniform_weight * 2.0))


def _btc_df(days: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(10000.0, 50000.0, len(idx)),
            "CapMVRVCur": np.linspace(1.0, 2.0, len(idx)),
        },
        index=idx,
    )


def test_validate_weights_rejects_sum_mismatch() -> None:
    runner = StrategyRunner()

    with pytest.raises(WeightValidationError, match="expected 1.0"):
        runner._validate_weights(
            pd.Series([0.4, 0.4]),
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-01-02"),
        )


def test_validate_weights_rejects_negative_values() -> None:
    runner = StrategyRunner()

    with pytest.raises(WeightValidationError, match="contain negative values"):
        runner._validate_weights(
            pd.Series([1.1, -0.1]),
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-01-02"),
        )


def test_backtest_raises_when_no_windows_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    strategy = _UniformProposeStrategy()
    monkeypatch.setattr(
        "stacksats.runner.backtest_dynamic_dca",
        lambda *args, **kwargs: (pd.DataFrame(), 50.0),
    )

    with pytest.raises(ValueError, match="No backtest windows were generated"):
        runner.backtest(
            strategy,
            BacktestConfig(start_date="2024-01-01", end_date="2024-02-01"),
            btc_df=_btc_df(days=60),
        )


def test_validate_reports_win_rate_threshold_failure_message() -> None:
    runner = StrategyRunner()
    strategy = _UniformProposeStrategy()

    result = runner.validate(
        strategy,
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=1000.0,
        ),
        btc_df=_btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert bool(result.win_rate_ok) is False
    assert any("Win rate below threshold" in message for message in result.messages)


def test_export_raises_when_no_ranges_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    strategy = _UniformProposeStrategy()
    monkeypatch.setattr("stacksats.prelude.generate_date_ranges", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="No export ranges generated"):
        runner.export(
            strategy,
            ExportConfig(range_start="2025-01-01", range_end="2025-01-02"),
            btc_df=_btc_df(days=1200),
            current_date=pd.Timestamp("2025-01-02"),
        )


def test_validate_strict_rejects_strategy_that_mutates_context_features() -> None:
    runner = StrategyRunner()
    result = runner.validate(
        _MutatingProposeStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
            strict=True,
        ),
        btc_df=_btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df" in message for message in result.messages)


def test_validate_strict_rejects_non_deterministic_strategy() -> None:
    runner = StrategyRunner()
    result = runner.validate(
        _RandomProposeStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
            strict=True,
        ),
        btc_df=_btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("non-deterministic" in message for message in result.messages)

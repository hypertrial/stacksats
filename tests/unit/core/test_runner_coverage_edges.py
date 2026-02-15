from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BacktestConfig, BaseStrategy, StrategyContext, ValidationConfig


class _UniformStrategy(BaseStrategy):
    strategy_id = "runner-coverage-uniform"

    def propose_weight(self, state):
        return state.uniform_weight


def _btc_df(days: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": np.linspace(10000.0, 50000.0, len(idx)),
            "CapMVRVCur": np.linspace(1.0, 2.0, len(idx)),
        },
        index=idx,
    )


def _uniform_weights(ctx: StrategyContext) -> pd.Series:
    idx = pd.date_range(ctx.start_date, ctx.end_date, freq="D")
    if len(idx) == 0:
        return pd.Series(dtype=float)
    return pd.Series(np.full(len(idx), 1.0 / len(idx), dtype=float), index=idx)


def test_weights_match_returns_false_when_index_differs() -> None:
    lhs = pd.Series([0.5, 0.5], index=pd.date_range("2024-01-01", periods=2, freq="D"))
    rhs = pd.Series([0.5, 0.5], index=pd.date_range("2024-01-02", periods=2, freq="D"))
    assert StrategyRunner._weights_match(lhs, rhs) is False


def test_perturb_future_features_returns_copy_when_no_future_rows() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    features = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx)

    perturbed = StrategyRunner._perturb_future_features(features, probe=idx.max())

    assert perturbed.equals(features)
    assert perturbed is not features


def test_perturb_future_features_reverses_non_numeric_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    features = pd.DataFrame(
        {
            "PriceUSD_coinmetrics": [100.0, 110.0, 120.0, 130.0],
            "label": ["a", "b", "c", "d"],
        },
        index=idx,
    )

    perturbed = StrategyRunner._perturb_future_features(features, probe=idx[1])

    assert list(perturbed.loc[idx[2]:, "label"]) == ["d", "c"]


def test_build_fold_ranges_returns_empty_when_max_folds_below_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_min = builtins.min

    def _fake_min(a, b):
        if a == 4 and isinstance(b, int):
            return 1
        return original_min(a, b)

    monkeypatch.setattr(builtins, "min", _fake_min)

    folds = StrategyRunner._build_fold_ranges(
        start_ts=pd.Timestamp("2022-01-01"),
        end_ts=pd.Timestamp("2025-01-01"),
    )
    assert folds == []


def test_build_fold_ranges_skips_non_increasing_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stacksats.runner.np.linspace",
        lambda *args, **kwargs: np.array([0, 0, 10, 20, 30], dtype=int),
    )

    folds = StrategyRunner._build_fold_ranges(
        start_ts=pd.Timestamp("2022-01-01"),
        end_ts=pd.Timestamp("2025-01-01"),
    )

    assert folds == []


def test_strict_fold_checks_skips_when_less_than_two_fold_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()

    class _TwoLenOneIter:
        def __len__(self):
            return 2

        def __iter__(self):
            yield (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))

    monkeypatch.setattr(runner, "_build_fold_ranges", lambda *args, **kwargs: _TwoLenOneIter())
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=55.0))

    ok, messages = runner._strict_fold_checks(
        strategy=_UniformStrategy(),
        btc_df=_btc_df(days=1000),
        start_ts=pd.Timestamp("2024-01-01"),
        end_ts=pd.Timestamp("2025-12-31"),
        config=ValidationConfig(strict=True),
    )

    assert ok is True
    assert any("not enough valid fold results" in message for message in messages)


def test_strict_shuffled_check_skips_empty_validation_window() -> None:
    runner = StrategyRunner()
    ok, messages = runner._strict_shuffled_check(
        strategy=_UniformStrategy(),
        btc_df=_btc_df(days=30),
        start_ts=pd.Timestamp("2035-01-01"),
        end_ts=pd.Timestamp("2035-01-05"),
        config=ValidationConfig(strict=True, shuffled_trials=1),
    )

    assert ok is True
    assert any("empty validation window" in message for message in messages)


def test_strict_shuffled_check_skips_when_no_runs_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr("stacksats.runner.range", lambda *args: [], raising=False)

    ok, messages = runner._strict_shuffled_check(
        strategy=_UniformStrategy(),
        btc_df=_btc_df(days=30),
        start_ts=pd.Timestamp("2024-01-01"),
        end_ts=pd.Timestamp("2024-01-05"),
        config=ValidationConfig(strict=True, shuffled_trials=1),
    )

    assert ok is True
    assert any("no shuffled runs completed" in message for message in messages)


def test_backtest_strategy_fn_handles_empty_window(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()

    def _fake_backtest_dynamic_dca(
        btc_df,
        strategy_fn,
        *,
        features_df,
        strategy_label,
        start_date,
        end_date,
    ):
        del btc_df, features_df, strategy_label, start_date, end_date
        empty_weights = strategy_fn(pd.DataFrame())
        assert empty_weights.empty
        spd_table = pd.DataFrame(
            {
                "dynamic_percentile": [60.0],
                "uniform_percentile": [40.0],
            }
        )
        return spd_table, 50.0

    monkeypatch.setattr("stacksats.runner.backtest_dynamic_dca", _fake_backtest_dynamic_dca)

    result = runner.backtest(
        strategy,
        BacktestConfig(start_date="2024-01-01", end_date="2024-01-10"),
        btc_df=_btc_df(days=30),
    )

    assert float(result.win_rate) == 100.0


def test_validate_continues_when_window_start_exceeds_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", -pd.Timedelta(days=1))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-03", min_win_rate=0.0),
        btc_df=_btc_df(days=10),
    )

    assert bool(result.passed) is True


def test_validate_strict_repeat_pass_mutation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", pd.Timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    counter = {"n": 0}

    def _compute(ctx: StrategyContext) -> pd.Series:
        counter["n"] += 1
        if counter["n"] == 2:
            ctx.features_df["__mutated__"] = 1.0
        return _uniform_weights(ctx)

    monkeypatch.setattr(strategy, "compute_weights", _compute)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", strict=True, min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df" in message for message in result.messages)


def test_validate_strict_masked_pass_mutation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", pd.Timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    def _compute(ctx: StrategyContext) -> pd.Series:
        if ctx.features_df.isna().any().any():
            ctx.features_df["__masked_mutation__"] = 1.0
        return _uniform_weights(ctx)

    monkeypatch.setattr(strategy, "compute_weights", _compute)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", strict=True, min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df" in message for message in result.messages)


def test_validate_strict_perturbed_pass_mutation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", pd.Timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    def _compute(ctx: StrategyContext) -> pd.Series:
        numeric = ctx.features_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if np.isfinite(numeric).any() and np.nanmin(numeric) < 0.0:
            ctx.features_df["__perturbed_mutation__"] = 1.0
        return _uniform_weights(ctx)

    monkeypatch.setattr(strategy, "compute_weights", _compute)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", strict=True, min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df" in message for message in result.messages)


def test_validate_leakage_loop_skips_when_prefix_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", pd.Timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    def _future_only_weights(ctx: StrategyContext) -> pd.Series:
        future_day = pd.Timestamp(ctx.end_date) + pd.Timedelta(days=1)
        return pd.Series([1.0], index=[future_day], dtype=float)

    monkeypatch.setattr(strategy, "compute_weights", _future_only_weights)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.passed) is True


def test_validate_weight_loop_skips_empty_weight_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", pd.Timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))
    monkeypatch.setattr(
        strategy,
        "compute_weights",
        lambda ctx: pd.Series(dtype=float, index=pd.DatetimeIndex([])),
    )

    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.passed) is True


def test_validate_weight_loop_records_weight_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", pd.Timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    def _invalid_weights(ctx: StrategyContext) -> pd.Series:
        idx = pd.date_range(ctx.start_date, ctx.end_date, freq="D")
        return pd.Series(np.zeros(len(idx), dtype=float), index=idx)

    monkeypatch.setattr(strategy, "compute_weights", _invalid_weights)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.weight_constraints_ok) is False
    assert any("expected 1.0" in message for message in result.messages)


def test_validate_strict_lock_base_mutation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", pd.Timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    counter = {"n": 0}

    def _compute(ctx: StrategyContext) -> pd.Series:
        counter["n"] += 1
        if counter["n"] == 24:
            ctx.features_df["__lock_base_mutation__"] = 1.0
        return _uniform_weights(ctx)

    monkeypatch.setattr(strategy, "compute_weights", _compute)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", strict=True, min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df" in message for message in result.messages)


def test_validate_strict_lock_perturbed_mutation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr("stacksats.runner.WINDOW_OFFSET", pd.Timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    counter = {"n": 0}

    def _compute(ctx: StrategyContext) -> pd.Series:
        counter["n"] += 1
        if counter["n"] == 25:
            ctx.features_df["__lock_perturbed_mutation__"] = 1.0
        return _uniform_weights(ctx)

    monkeypatch.setattr(strategy, "compute_weights", _compute)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", strict=True, min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df" in message for message in result.messages)

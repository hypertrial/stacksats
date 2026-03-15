from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BacktestConfig, BaseStrategy, StrategyContext, ValidationConfig
from tests.test_helpers import btc_frame

pytestmark = pytest.mark.slow


class _UniformStrategy(BaseStrategy):
    strategy_id = "runner-coverage-uniform"

    def propose_weight(self, state):
        return state.uniform_weight


def _btc_df(days: int = 60) -> pl.DataFrame:
    return btc_frame(start="2024-01-01", days=days)


def _window_dates(ctx: StrategyContext) -> list[dt.datetime]:
    return pl.datetime_range(ctx.start_date, ctx.end_date, interval="1d", eager=True).to_list()


def _uniform_weights(ctx: StrategyContext) -> pl.DataFrame:
    dates = _window_dates(ctx)
    if len(dates) == 0:
        return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})
    return pl.DataFrame({"date": dates, "weight": np.full(len(dates), 1.0 / len(dates), dtype=float)})


def test_weights_match_returns_false_when_dates_differ() -> None:
    lhs = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "weight": [0.5, 0.5],
        }
    )
    rhs = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)],
            "weight": [0.5, 0.5],
        }
    )
    assert StrategyRunner._weights_match(lhs, rhs) is False


def test_perturb_future_helpers_return_copy_when_no_future_rows() -> None:
    probe = dt.datetime(2024, 1, 3)
    features = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), probe],
            "x": [1.0, 2.0, 3.0],
        }
    )
    source = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), probe],
            "price_usd": [100.0, 101.0, 102.0],
        }
    )

    perturbed_features = StrategyRunner._perturb_future_features(features, probe=probe)
    perturbed_source = StrategyRunner._perturb_future_source_data(source, probe=probe)

    assert perturbed_features.equals(features)
    assert perturbed_source.equals(source)
    assert perturbed_features is not features
    assert perturbed_source is not source


def test_perturb_future_features_reverses_non_numeric_columns() -> None:
    features = pl.DataFrame(
        {
            "date": [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
                dt.datetime(2024, 1, 4),
            ],
            "price_usd": [100.0, 110.0, 120.0, 130.0],
            "label": ["a", "b", "c", "d"],
        }
    )

    perturbed = StrategyRunner._perturb_future_features(features, probe=dt.datetime(2024, 1, 2))
    assert perturbed.filter(pl.col("date") > dt.datetime(2024, 1, 2))["label"].to_list() == ["d", "c"]


def test_build_fold_ranges_handles_short_and_valid_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds = StrategyRunner._build_fold_ranges(
        start_ts=dt.datetime(2024, 1, 1),
        end_ts=dt.datetime(2024, 12, 31),
    )
    assert folds == []

    monkeypatch.setattr(
        "stacksats.runner_helpers.np.linspace",
        lambda *args, **kwargs: np.array([0, 0, 10, 20, 30], dtype=int),
    )
    skipped = StrategyRunner._build_fold_ranges(
        start_ts=dt.datetime(2022, 1, 1),
        end_ts=dt.datetime(2025, 1, 1),
    )
    assert skipped == []

    valid = StrategyRunner._build_fold_ranges(
        start_ts=dt.datetime(2020, 1, 1),
        end_ts=dt.datetime(2025, 12, 31),
    )
    assert len(valid) >= 1
    assert all((end - start).days + 1 >= ALLOCATION_SPAN_DAYS for start, end in valid)


def test_strict_shuffled_check_skips_empty_window_and_empty_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    ok, messages = runner._strict_shuffled_check(
        strategy=_UniformStrategy(),
        btc_df=_btc_df(days=30),
        start_ts=dt.datetime(2035, 1, 1),
        end_ts=dt.datetime(2035, 1, 5),
        config=ValidationConfig(strict=True, shuffled_trials=1),
    )
    assert ok is True
    assert any("empty validation window" in message for message in messages)

    monkeypatch.setattr(StrategyRunner, "ITER_RANGE", lambda *args: [])
    ok, messages = runner._strict_shuffled_check(
        strategy=_UniformStrategy(),
        btc_df=_btc_df(days=30),
        start_ts=dt.datetime(2024, 1, 1),
        end_ts=dt.datetime(2024, 1, 5),
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
        empty_weights = strategy_fn(pl.DataFrame(schema={"date": pl.Datetime("us"), "price_usd": pl.Float64}))
        assert empty_weights.is_empty()
        spd_table = pl.DataFrame(
            {
                "dynamic_percentile": [60.0],
                "uniform_percentile": [40.0],
            }
        )
        return spd_table, 50.0, 45.0

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
    monkeypatch.setattr(StrategyRunner, "WINDOW_OFFSET", -dt.timedelta(days=1))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-03", min_win_rate=0.0),
        btc_df=_btc_df(days=10),
    )

    assert bool(result.passed) is True


@pytest.mark.parametrize("mutation_call", [2, 3, 4, 24, 25])
def test_validate_strict_detects_mutation_across_repeat_masked_perturbed_and_lock_paths(
    monkeypatch: pytest.MonkeyPatch,
    mutation_call: int,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr(StrategyRunner, "WINDOW_OFFSET", dt.timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))
    counter = {"n": 0}

    def _compute(ctx: StrategyContext) -> pl.DataFrame:
        counter["n"] += 1
        if counter["n"] == mutation_call:
            ctx.features_df.insert_column(
                ctx.features_df.width,
                pl.Series(f"__mutation_{mutation_call}__", [1.0] * ctx.features_df.height),
            )
        return _uniform_weights(ctx)

    monkeypatch.setattr(strategy, "compute_weights", _compute)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", strict=True, min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features" in message for message in result.messages)


def test_validate_leakage_and_weight_loops_handle_empty_and_invalid_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr(StrategyRunner, "WINDOW_OFFSET", dt.timedelta(days=2))
    monkeypatch.setattr(runner, "backtest", lambda *args, **kwargs: SimpleNamespace(win_rate=100.0))

    def _future_only_weights(ctx: StrategyContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "date": [ctx.end_date + dt.timedelta(days=1)],
                "weight": [1.0],
            }
        )

    monkeypatch.setattr(strategy, "compute_weights", _future_only_weights)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )
    assert bool(result.passed) is True

    monkeypatch.setattr(
        strategy,
        "compute_weights",
        lambda ctx: pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64}),
    )
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )
    assert bool(result.passed) is True

    def _invalid_weights(ctx: StrategyContext) -> pl.DataFrame:
        dates = _window_dates(ctx)
        return pl.DataFrame({"date": dates, "weight": np.zeros(len(dates), dtype=float)})

    monkeypatch.setattr(strategy, "compute_weights", _invalid_weights)
    result = runner.validate(
        strategy,
        ValidationConfig(start_date="2024-01-01", end_date="2024-01-05", min_win_rate=0.0),
        btc_df=_btc_df(days=20),
    )
    assert bool(result.weight_constraints_ok) is False
    assert any("expected 1.0" in message for message in result.messages)

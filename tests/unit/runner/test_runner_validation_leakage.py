from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import polars as pl
import pytest

from stacksats.runner import StrategyRunner
from stacksats.strategy_types import ValidationConfig
from tests.unit.runner.runner_validation_testkit import (
    DualHookProfilePreferredLeakStrategy,
    ProfileOffsetLeakStrategy,
    ProfileValuePeekStrategy,
    UniformProposeStrategy,
    btc_df,
    patch_skip_weight_and_lock_checks,
)

pytestmark = pytest.mark.slow


def test_validate_reports_masked_future_weight_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
    matches = iter([False])
    monkeypatch.setattr(
        runner,
        "_weights_match",
        lambda *args, **kwargs: bool(next(matches)),
    )

    result = runner.validate(
        UniformProposeStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
        ),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.forward_leakage_ok) is False
    assert any("masked-future weights diverge" in msg for msg in result.messages)


def test_validate_reports_perturbed_future_weight_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
    matches = iter([True, False])
    monkeypatch.setattr(
        runner,
        "_weights_match",
        lambda *args, **kwargs: bool(next(matches)),
    )

    result = runner.validate(
        UniformProposeStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
        ),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.forward_leakage_ok) is False
    assert any("perturbed-future weights diverge" in msg for msg in result.messages)


def test_validate_observed_only_profile_input_blocks_future_offset_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )

    result = runner.validate(
        ProfileOffsetLeakStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
        ),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.forward_leakage_ok) is True
    assert any("All validation checks passed." in msg for msg in result.messages)


def test_validate_uses_profile_checks_for_dual_hook_profile_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )

    result = runner.validate(
        DualHookProfilePreferredLeakStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
        ),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.forward_leakage_ok) is True
    assert any("All validation checks passed." in msg for msg in result.messages)


def test_validate_detects_profile_value_peeking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )

    result = runner.validate(
        ProfileValuePeekStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
        ),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.forward_leakage_ok) is False
    assert any("profile values diverge" in msg for msg in result.messages)


def test_align_profile_horizon_keeps_feature_columns_for_empty_input() -> None:
    horizon = pl.DataFrame({"date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)]})
    empty_features = pl.DataFrame(
        schema={"date": pl.Datetime("us"), "price_usd": pl.Float64, "mvrv": pl.Float64}
    )

    aligned = StrategyRunner._align_profile_horizon(horizon, empty_features)

    assert aligned.columns == ["date", "price_usd", "mvrv"]
    assert aligned.height == horizon.height
    assert aligned["price_usd"].null_count() == horizon.height

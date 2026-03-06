from __future__ import annotations

from types import SimpleNamespace

import pytest

from stacksats.runner import StrategyRunner
from stacksats.strategy_types import ValidationConfig
from tests.unit.core.runner_validation_testkit import (
    DualHookProfilePreferredLeakStrategy,
    ProfileOffsetLeakStrategy,
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

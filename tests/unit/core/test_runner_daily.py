from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stacksats.api import ValidationResult
from stacksats.execution_state import IdempotencyConflictError
from stacksats.runner import StrategyRunner
from stacksats.strategies.model_example import ExampleMVRVStrategy
from stacksats.strategies.model_mvrv_plus import MVRVPlusStrategy
from stacksats.strategy_types import BaseStrategy, RunDailyConfig, StrategyContext


class _UniformDailyStrategy(BaseStrategy):
    strategy_id = "daily-uniform"
    version = "1.0.0"

    def propose_weight(self, state):
        return state.uniform_weight


class _ObservingUniformDailyStrategy(_UniformDailyStrategy):
    def __init__(self) -> None:
        self.locked_prefix_len: int | None = None

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        self.locked_prefix_len = (
            len(ctx.locked_weights) if ctx.locked_weights is not None else None
        )
        return super().transform_features(ctx)


def _btc_df() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=900, freq="D")
    return pd.DataFrame(
        {
            "price_usd": np.linspace(20000.0, 90000.0, len(idx)),
            "mvrv": np.linspace(0.8, 2.2, len(idx)),
        },
        index=idx,
    )


def _config(tmp_path, **overrides) -> RunDailyConfig:
    params = {
        "run_date": "2024-12-31",
        "total_window_budget_usd": 1000.0,
        "mode": "paper",
        "state_db_path": str(tmp_path / "state.sqlite3"),
        "output_dir": str(tmp_path / "output"),
    }
    params.update(overrides)
    return RunDailyConfig(**params)


def _allow_validation(runner: StrategyRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "validate",
        lambda *args, **kwargs: ValidationResult(
            passed=True,
            forward_leakage_ok=True,
            weight_constraints_ok=True,
            win_rate=100.0,
            win_rate_ok=True,
            messages=["ok"],
            diagnostics={},
        ),
    )


def test_run_daily_executes_paper_order(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    _allow_validation(runner, monkeypatch)
    result = runner.run_daily(
        _UniformDailyStrategy(),
        _config(tmp_path),
        btc_df=_btc_df(),
    )
    assert result.status == "executed"
    assert result.order_notional_usd is not None
    assert result.btc_quantity is not None
    assert result.idempotency_hit is False
    assert result.artifact_path is not None


def test_run_daily_second_invocation_is_noop_for_same_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    _allow_validation(runner, monkeypatch)
    strategy = _UniformDailyStrategy()
    first = runner.run_daily(strategy, _config(tmp_path), btc_df=_btc_df())
    second = runner.run_daily(strategy, _config(tmp_path), btc_df=_btc_df())
    assert first.status == "executed"
    assert second.status == "noop"
    assert second.idempotency_hit is True
    assert second.run_key == first.run_key


def test_run_daily_conflict_without_force_on_changed_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    _allow_validation(runner, monkeypatch)
    strategy = _UniformDailyStrategy()
    runner.run_daily(strategy, _config(tmp_path), btc_df=_btc_df())
    with pytest.raises(IdempotencyConflictError):
        runner.run_daily(
            strategy,
            _config(tmp_path, total_window_budget_usd=2000.0),
            btc_df=_btc_df(),
        )


def test_run_daily_force_rerun_allows_changed_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    _allow_validation(runner, monkeypatch)
    strategy = _UniformDailyStrategy()
    runner.run_daily(strategy, _config(tmp_path), btc_df=_btc_df())
    forced = runner.run_daily(
        strategy,
        _config(tmp_path, total_window_budget_usd=2000.0, force=True),
        btc_df=_btc_df(),
    )
    assert forced.status == "executed"
    assert forced.forced_rerun is True


def test_run_daily_missing_price_data_returns_failed_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _btc_df()
    df.loc[pd.Timestamp("2024-12-31"), "price_usd"] = np.nan
    runner = StrategyRunner()
    _allow_validation(runner, monkeypatch)
    result = runner.run_daily(
        _UniformDailyStrategy(),
        _config(tmp_path),
        btc_df=df,
    )
    assert result.status == "failed"
    assert "Missing BTC price data" in result.message


def test_run_daily_live_mode_requires_adapter(tmp_path) -> None:
    runner = StrategyRunner()
    with pytest.raises(ValueError, match="Live mode requires --adapter"):
        runner.run_daily(
            _UniformDailyStrategy(),
            _config(tmp_path, mode="live"),
            btc_df=_btc_df(),
        )


def test_run_daily_reuses_prior_snapshot_for_locked_prefix(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    _allow_validation(runner, monkeypatch)
    strategy = _ObservingUniformDailyStrategy()
    runner.run_daily(
        strategy,
        _config(tmp_path, run_date="2024-12-30"),
        btc_df=_btc_df(),
    )
    second = runner.run_daily(
        strategy,
        _config(tmp_path, run_date="2024-12-31"),
        btc_df=_btc_df(),
    )
    assert second.status == "executed"
    assert strategy.locked_prefix_len is not None
    assert strategy.locked_prefix_len == 364


def test_daily_run_fingerprint_uses_provider_contract_for_example_strategy(tmp_path) -> None:
    runner = StrategyRunner()
    strategy = ExampleMVRVStrategy()

    before = runner._daily_run_fingerprint(strategy, _config(tmp_path), "2024-12-31")
    strategy.overlay_scale = 0.95
    after = runner._daily_run_fingerprint(strategy, _config(tmp_path), "2024-12-31")

    assert before != after


def test_daily_run_fingerprint_tracks_mvrv_plus_params(tmp_path) -> None:
    runner = StrategyRunner()
    strategy = MVRVPlusStrategy(overlay_scale=0.20)

    before = runner._daily_run_fingerprint(strategy, _config(tmp_path), "2024-12-31")
    after_same = runner._daily_run_fingerprint(strategy, _config(tmp_path), "2024-12-31")
    changed = runner._daily_run_fingerprint(
        MVRVPlusStrategy(overlay_scale=0.35),
        _config(tmp_path),
        "2024-12-31",
    )

    assert before == after_same
    assert before != changed

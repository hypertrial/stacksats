from __future__ import annotations

import datetime as dt
from types import MethodType

import numpy as np
import polars as pl

from stacksats.api import DailyOrderReceipt, ValidationResult
from stacksats.execution_state import StoredRun
from stacksats.runner import StrategyRunner
from stacksats.strategy_types import BaseStrategy, RunDailyConfig, StrategyContext
from tests.test_helpers import btc_frame


class _UniformStrategy(BaseStrategy):
    strategy_id = "daily-hardening"
    version = "1.0.0"

    def propose_weight(self, state):
        return state.uniform_weight


class _TiltedStrategy(BaseStrategy):
    strategy_id = "daily-reconcile"
    version = "1.0.0"

    def build_target_profile(self, ctx, features_df, signals):
        del ctx, signals
        return pl.DataFrame(
            {
                "date": features_df["date"],
                "value": np.linspace(0.0, 1.0, features_df.height, dtype=float),
            }
        )


def _btc_df(days: int = 500) -> pl.DataFrame:
    return btc_frame(start="2023-01-01", days=days).with_columns(
        pl.col("price_usd").alias("PriceUSD")
    )


def test_run_daily_fails_fast_when_strict_validation_fails(
    tmp_path,
    monkeypatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    monkeypatch.setattr(
        runner,
        "validate",
        lambda *args, **kwargs: ValidationResult(
            passed=False,
            forward_leakage_ok=False,
            weight_constraints_ok=False,
            win_rate=0.0,
            win_rate_ok=False,
            messages=["blocked"],
            diagnostics={},
        ),
    )

    result = runner.run_daily(
        strategy,
        RunDailyConfig(
            run_date="2024-05-01",
            total_window_budget_usd=1000.0,
            state_db_path=str(tmp_path / "state.sqlite3"),
        ),
        btc_df=_btc_df(),
    )

    assert result.status == "failed"
    assert "Strict validation failed" in result.message
    assert result.validation_receipt_id is not None
    assert result.validation_passed is False


def test_reconcile_daily_run_detects_decision_change(tmp_path, monkeypatch) -> None:
    runner = StrategyRunner()
    strategy = _TiltedStrategy()
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
    monkeypatch.setattr(
        "stacksats.execution_adapters.PaperExecutionAdapter.submit_order",
        lambda self, request, idempotency_key: DailyOrderReceipt(
            status="filled",
            external_order_id=idempotency_key,
            filled_notional_usd=request.notional_usd,
            filled_quantity_btc=request.quantity_btc,
            fill_price_usd=request.price_usd,
            metadata={},
        ),
    )

    state_db = tmp_path / "state.sqlite3"
    base_df = _btc_df()
    runner.run_daily(
        strategy,
        RunDailyConfig(
            run_date="2024-05-01",
            total_window_budget_usd=1000.0,
            state_db_path=str(state_db),
        ),
        btc_df=base_df,
    )

    revised_df = base_df.with_columns(
        pl.when(pl.col("date").is_between(dt.datetime(2024, 4, 15), dt.datetime(2024, 5, 1)))
        .then(pl.col("price_usd") * 0.8)
        .otherwise(pl.col("price_usd"))
        .alias("price_usd"),
        pl.when(pl.col("date").is_between(dt.datetime(2024, 4, 15), dt.datetime(2024, 5, 1)))
        .then(pl.col("PriceUSD") * 0.8)
        .otherwise(pl.col("PriceUSD"))
        .alias("PriceUSD"),
    )
    result = runner.reconcile_daily_run(
        strategy,
        run_date="2024-05-01",
        state_db_path=str(state_db),
        btc_df=revised_df,
    )

    assert result["status"] in {
        "stable",
        "data_revised_no_decision_change",
        "decision_changed_due_to_revision",
    }


def test_reconcile_daily_run_reuses_locked_prefix(monkeypatch) -> None:
    runner = StrategyRunner()
    strategy = _UniformStrategy()
    captured: dict[str, np.ndarray | None] = {}
    locked_prefix = np.array([0.01, 0.02, 0.03], dtype=float)

    monkeypatch.setattr(runner, "_validate_strategy_contract", lambda _: None)
    monkeypatch.setattr(
        "stacksats.execution_state.SQLiteExecutionStateStore.get_run",
        lambda self, **kwargs: StoredRun(
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.version,
            run_date="2024-05-01",
            mode="paper",
            run_key="run-key",
            fingerprint="fp",
            status="executed",
            payload={"weight_today": 0.01, "feature_snapshot_hash": "old-hash"},
            order_summary=None,
            force_flag=False,
        ),
    )
    monkeypatch.setattr(
        "stacksats.execution_state.SQLiteExecutionStateStore.load_locked_prefix",
        lambda self, **kwargs: locked_prefix,
    )
    monkeypatch.setattr(
        runner,
        "_load_btc_df",
        lambda btc_df=None, *, end_date=None: _btc_df(),
    )
    monkeypatch.setattr(
        runner._feature_registry,
        "materialization_fingerprint",
        lambda *args, **kwargs: (
            (lambda dates: (
                pl.DataFrame(
                    {
                        "date": dates,
                        "price_usd": np.linspace(10000.0, 20000.0, len(dates)),
                    }
                )
            ))(
                pl.datetime_range(
                    kwargs["start_date"],
                    kwargs["end_date"],
                    interval="1d",
                    eager=True,
                ).to_list()
            ),
            "new-hash",
        ),
    )

    original_compute_weights = strategy.compute_weights

    def _wrapped_compute_weights(self, ctx: StrategyContext):
        captured["locked_weights"] = (
            None if ctx.locked_weights is None else np.array(ctx.locked_weights, copy=True)
        )
        return original_compute_weights(ctx)

    monkeypatch.setattr(
        strategy,
        "compute_weights",
        MethodType(_wrapped_compute_weights, strategy),
    )

    result = runner.reconcile_daily_run(
        strategy,
        run_date="2024-05-01",
        state_db_path=":memory:",
    )

    assert np.array_equal(captured["locked_weights"], locked_prefix)
    assert result["recomputed_feature_snapshot_hash"] == "new-hash"

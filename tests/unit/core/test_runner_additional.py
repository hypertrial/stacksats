from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS, MAX_DAILY_WEIGHT, MIN_DAILY_WEIGHT
from stacksats.runner import StrategyRunner, WeightValidationError
from stacksats.strategy_types import BacktestConfig, ExportConfig, StrategyMetadata, ValidationConfig
from tests.unit.core.runner_validation_testkit import (
    UniformProposeStrategy,
    btc_df,
    patch_skip_weight_and_lock_checks,
)

pytestmark = pytest.mark.slow


def _weight_df(values: np.ndarray | list[float], start: dt.datetime) -> pl.DataFrame:
    dates = pl.datetime_range(start, start + dt.timedelta(days=len(values) - 1), interval="1d", eager=True)
    return pl.DataFrame({"date": dates, "weight": values})


def test_validate_weights_rejects_sum_mismatch() -> None:
    runner = StrategyRunner()

    with pytest.raises(WeightValidationError, match="expected 1.0"):
        runner._validate_weights(
            _weight_df([0.4, 0.4], dt.datetime(2024, 1, 1)),
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 2),
        )


def test_validate_weights_rejects_negative_values() -> None:
    runner = StrategyRunner()

    with pytest.raises(WeightValidationError, match="contain negative values"):
        runner._validate_weights(
            _weight_df([1.1, -0.1], dt.datetime(2024, 1, 1)),
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 1, 2),
        )


def test_validate_weights_rejects_below_min_for_full_contract_span() -> None:
    runner = StrategyRunner()
    base = 1.0 / ALLOCATION_SPAN_DAYS
    weights = np.full(ALLOCATION_SPAN_DAYS, base, dtype=float)
    weights[0] = MIN_DAILY_WEIGHT / 10.0
    deficit = base - weights[0]
    weights[1:] += deficit / (ALLOCATION_SPAN_DAYS - 1)

    with pytest.raises(WeightValidationError, match="below minimum"):
        runner._validate_weights(
            _weight_df(weights, dt.datetime(2024, 1, 1)),
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 12, 30),
        )


def test_validate_weights_rejects_above_max_for_full_contract_span() -> None:
    runner = StrategyRunner()
    base = 1.0 / ALLOCATION_SPAN_DAYS
    weights = np.full(ALLOCATION_SPAN_DAYS, base, dtype=float)
    weights[0] = MAX_DAILY_WEIGHT + 1e-3
    excess = weights[0] - base
    weights[1:] -= excess / (ALLOCATION_SPAN_DAYS - 1)

    with pytest.raises(WeightValidationError, match="above maximum"):
        runner._validate_weights(
            _weight_df(weights, dt.datetime(2024, 1, 1)),
            window_start=dt.datetime(2024, 1, 1),
            window_end=dt.datetime(2024, 12, 30),
        )


def test_backtest_raises_when_no_windows_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    strategy = UniformProposeStrategy()
    monkeypatch.setattr(
        "stacksats.runner.backtest_dynamic_dca",
        lambda *args, **kwargs: (
            pl.DataFrame(schema={"dynamic_percentile": pl.Float64, "uniform_percentile": pl.Float64}),
            50.0,
            40.0,
        ),
    )

    with pytest.raises(ValueError, match="No backtest windows were generated"):
        runner.backtest(
            strategy,
            BacktestConfig(start_date="2024-01-01", end_date="2024-02-01"),
            btc_df=btc_df(days=60),
        )


def test_backtest_win_rate_ignores_tiny_float_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = UniformProposeStrategy()
    spd_table = pl.DataFrame(
        {
            "dynamic_percentile": [40.0 + 1e-12, 50.0 - 1e-12, 60.0 + 5e-11],
            "uniform_percentile": [40.0, 50.0, 60.0],
        }
    )
    monkeypatch.setattr(
        "stacksats.runner.backtest_dynamic_dca",
        lambda *args, **kwargs: (spd_table, 50.0, 40.0),
    )

    result = runner.backtest(
        strategy,
        BacktestConfig(start_date="2022-01-01", end_date="2022-12-31"),
        btc_df=btc_df(days=500),
    )

    assert float(result.win_rate) == 0.0


def test_backtest_win_rate_counts_only_deltas_above_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = UniformProposeStrategy()
    spd_table = pl.DataFrame(
        {
            "dynamic_percentile": [40.0 + 1e-8, 50.0 - 1e-8, 60.0 + 2e-10],
            "uniform_percentile": [40.0, 50.0, 60.0],
        }
    )
    monkeypatch.setattr(
        "stacksats.runner.backtest_dynamic_dca",
        lambda *args, **kwargs: (spd_table, 50.0, 40.0),
    )

    result = runner.backtest(
        strategy,
        BacktestConfig(start_date="2022-01-01", end_date="2022-12-31"),
        btc_df=btc_df(days=500),
    )

    assert result.win_rate == pytest.approx(66.6666666667, rel=0.0, abs=1e-9)


def test_validate_reports_win_rate_threshold_failure_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = UniformProposeStrategy()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=12.5),
    )
    monkeypatch.setattr(runner, "_run_forward_leakage_checks", lambda **kwargs: None)
    patch_skip_weight_and_lock_checks(monkeypatch, runner)

    result = runner.validate(
        strategy,
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=1000.0,
        ),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert bool(result.win_rate_ok) is False
    assert any("Win rate below threshold" in message for message in result.messages)


def test_export_raises_when_no_ranges_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    strategy = UniformProposeStrategy()
    monkeypatch.setattr("stacksats.prelude.generate_date_ranges", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="No export ranges generated"):
        runner.export(
            strategy,
            ExportConfig(range_start="2025-01-01", range_end="2025-01-02"),
            btc_df=btc_df(days=1200),
            current_date=dt.datetime(2025, 1, 2),
        )


def test_strict_fold_checks_skip_on_short_range() -> None:
    runner = StrategyRunner()
    ok, messages = runner._strict_fold_checks(
        strategy=UniformProposeStrategy(),
        btc_df=btc_df(days=120),
        start_ts=dt.datetime(2024, 1, 1),
        end_ts=dt.datetime(2024, 3, 31),
        config=ValidationConfig(strict=True),
    )

    assert ok is True
    assert any("insufficient date range" in msg for msg in messages)


def test_strict_fold_checks_reports_min_fold_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    fold_rates = iter([70.0, 50.0, 74.0, 69.0])
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=float(next(fold_rates))),
    )

    ok, messages = runner._strict_fold_checks(
        strategy=UniformProposeStrategy(),
        btc_df=btc_df(days=2000),
        start_ts=dt.datetime(2021, 1, 1),
        end_ts=dt.datetime(2025, 12, 31),
        config=ValidationConfig(strict=True, min_fold_win_rate=60.0, max_fold_win_rate_std=1000.0),
    )

    assert ok is False
    assert any("minimum fold win rate" in msg for msg in messages)
    assert any("Strict fold diagnostics" in msg for msg in messages)


def test_strict_fold_checks_reports_std_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    fold_rates = iter([10.0, 80.0, 10.0, 80.0])
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=float(next(fold_rates))),
    )

    ok, messages = runner._strict_fold_checks(
        strategy=UniformProposeStrategy(),
        btc_df=btc_df(days=2000),
        start_ts=dt.datetime(2021, 1, 1),
        end_ts=dt.datetime(2025, 12, 31),
        config=ValidationConfig(strict=True, min_fold_win_rate=0.0, max_fold_win_rate_std=5.0),
    )

    assert ok is False
    assert any("fold win-rate std" in msg for msg in messages)
    assert any("Strict fold diagnostics" in msg for msg in messages)


def test_strict_shuffled_check_reports_threshold_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    shuffled_rates = iter([92.0, 90.0, 88.0])
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=float(next(shuffled_rates))),
    )

    ok, messages = runner._strict_shuffled_check(
        strategy=UniformProposeStrategy(),
        btc_df=btc_df(days=2000),
        start_ts=dt.datetime(2022, 1, 1),
        end_ts=dt.datetime(2023, 12, 31),
        config=ValidationConfig(strict=True, shuffled_trials=3, max_shuffled_win_rate=80.0),
    )

    assert ok is False
    assert any("mean shuffled win rate" in msg for msg in messages)
    assert any("Strict shuffled diagnostics" in msg for msg in messages)


def test_strict_shuffled_check_skips_without_price_column() -> None:
    runner = StrategyRunner()
    df = pl.DataFrame(
        {
            "date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "Other": [1.0, 2.0],
        }
    )

    ok, messages = runner._strict_shuffled_check(
        strategy=UniformProposeStrategy(),
        btc_df=df,
        start_ts=dt.datetime(2024, 1, 1),
        end_ts=dt.datetime(2024, 1, 2),
        config=ValidationConfig(strict=True, shuffled_trials=3),
    )

    assert ok is True
    assert any("missing price_usd column" in msg for msg in messages)


def test_strict_shuffled_check_skips_when_trials_non_positive() -> None:
    runner = StrategyRunner()

    ok, messages = runner._strict_shuffled_check(
        strategy=UniformProposeStrategy(),
        btc_df=btc_df(days=500),
        start_ts=dt.datetime(2023, 1, 1),
        end_ts=dt.datetime(2023, 12, 31),
        config=ValidationConfig(strict=True, shuffled_trials=0),
    )

    assert ok is True
    assert any("shuffled_trials <= 0" in msg for msg in messages)


def test_frame_signature_handles_nested_payloads() -> None:
    df = pl.DataFrame({"obj": ['{"a": 1}', '{"b": 2}']})

    sig = StrategyRunner._frame_signature(df)
    assert isinstance(sig[0], int)
    assert sig[3] == (2, 1)


def test_strict_validation_failure_message_includes_details() -> None:
    msg = StrategyRunner._strict_validation_failure_message(["a", "b"])
    assert msg.endswith("a; b")

    fallback = StrategyRunner._strict_validation_failure_message([])
    assert fallback == "Strict validation failed before daily execution."


def test_classify_reconciliation_status_paths() -> None:
    assert (
        StrategyRunner._classify_reconciliation_status(
            previous_weight_today=None,
            recomputed_weight_today=0.1,
            previous_feature_snapshot_hash="x",
            recomputed_feature_snapshot_hash="y",
        )
        == "stable"
    )
    assert (
        StrategyRunner._classify_reconciliation_status(
            previous_weight_today=0.2,
            recomputed_weight_today=0.2,
            previous_feature_snapshot_hash="x",
            recomputed_feature_snapshot_hash="x",
        )
        == "stable"
    )
    assert (
        StrategyRunner._classify_reconciliation_status(
            previous_weight_today=0.2,
            recomputed_weight_today=0.2,
            previous_feature_snapshot_hash="x",
            recomputed_feature_snapshot_hash="z",
        )
        == "data_revised_no_decision_change"
    )
    assert (
        StrategyRunner._classify_reconciliation_status(
            previous_weight_today=0.2,
            recomputed_weight_today=0.25,
            previous_feature_snapshot_hash="x",
            recomputed_feature_snapshot_hash="z",
        )
        == "decision_changed_due_to_revision"
    )


def test_merge_strict_check_result_updates_state() -> None:
    state = SimpleNamespace(strict_checks_ok=True, messages=["initial"])
    StrategyRunner._merge_strict_check_result(
        state,
        ok=False,
        messages=["m1", "m2"],
    )
    assert state.strict_checks_ok is False
    assert state.messages == ["initial", "m1", "m2"]


def test_build_failed_daily_result_maps_expected_payload_fields() -> None:
    class _Result:
        def __init__(self, **kwargs):
            self.payload = kwargs

    runner = StrategyRunner()
    result = runner._build_failed_daily_result(
        metadata=StrategyMetadata(strategy_id="s", version="1.0.0"),
        config=SimpleNamespace(mode="paper"),
        run_date="2026-01-01",
        run_key="rk",
        state_store=SimpleNamespace(db_path=Path("/tmp/state.sqlite3")),
        forced_rerun=True,
        bootstrap=False,
        validation_receipt_id=10,
        data_hash="dhash",
        feature_snapshot_hash="fhash",
        error_message="boom",
        daily_run_result_cls=_Result,
    )

    assert result.payload["status"] == "failed"
    assert result.payload["forced_rerun"] is True
    assert result.payload["validation_receipt_id"] == 10
    assert result.payload["validation_passed"] is False
    assert result.payload["message"] == "Daily execution failed: boom"


def test_validate_returns_failed_result_when_contract_validation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "_validate_strategy_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid strategy")),
    )

    result = runner.validate(
        UniformProposeStrategy(),
        ValidationConfig(strict=True),
        btc_df=btc_df(days=120),
    )

    assert bool(result.passed) is False
    assert result.messages == ["invalid strategy"]


def test_reconcile_daily_run_raises_when_no_stored_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Store:
        def __init__(self, _path: str):
            pass

        def get_run(self, **kwargs):
            del kwargs
            return None

    monkeypatch.setattr("stacksats.execution_state.SQLiteExecutionStateStore", _Store)

    runner = StrategyRunner()
    with pytest.raises(ValueError, match="No stored daily run exists"):
        runner.reconcile_daily_run(
            UniformProposeStrategy(),
            run_date="2025-01-01",
            mode="paper",
            state_db_path=":memory:",
            btc_df=btc_df(days=120),
        )

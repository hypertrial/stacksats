"""Tests for enhanced public API functionality."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stacksats.api import (
    BacktestResult,
    DailyDecisionResult,
    DailyRunResult,
    ExecutionReceiptEvent,
    ExecutionReceiptHistoryResult,
    ExecutionStatusResult,
    ValidationResult,
)
from stacksats.strategy_types import (
    BaseStrategy,
    DecideDailyConfig,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
    strategy_context_from_features_df,
)
from stacksats.strategies.examples import (
    MomentumStrategy,
    RunDailyPaperStrategy,
    SimpleZScoreStrategy,
    UniformStrategy,
)


def _sample_btc_df() -> pl.DataFrame:
    base = datetime(2022, 1, 1)
    dates = [base + timedelta(days=i) for i in range(520)]
    price = np.linspace(20000.0, 45000.0, len(dates))
    mvrv = np.linspace(0.8, 2.2, len(dates))
    return pl.DataFrame({
        "date": dates,
        "price_usd": price,
        "mvrv": mvrv,
    })


def _sample_spd_df() -> pl.DataFrame:
    windows = [
        "2022-01-01 → 2023-01-01",
        "2022-02-01 → 2023-02-01",
        "2022-03-01 → 2023-03-01",
    ]
    return pl.DataFrame({
        "window": windows,
        "min_sats_per_dollar": [1000.0, 1200.0, 900.0],
        "max_sats_per_dollar": [5000.0, 5200.0, 4800.0],
        "uniform_sats_per_dollar": [2500.0, 2600.0, 2300.0],
        "dynamic_sats_per_dollar": [2800.0, 2900.0, 2400.0],
        "uniform_percentile": [37.5, 35.0, 36.8],
        "dynamic_percentile": [45.0, 42.0, 39.5],
        "excess_percentile": [7.5, 7.0, 2.7],
    })


def test_backtest_result_summary_dataframe_and_json(tmp_path: Path):
    """BacktestResult helper methods should return stable structured outputs."""
    result = BacktestResult(
        spd_table=_sample_spd_df(),
        exp_decay_percentile=44.2,
        win_rate=66.6,
        score=55.4,
        uniform_exp_decay_percentile=35.0,
    )

    summary = result.summary()
    assert "Score: 55.40%" in summary
    assert "Win Rate: 66.60%" in summary
    assert "Uniform Exp-Decay: 35.00%" in summary
    assert "Exp-Decay Multiple: 1.263x" in summary

    as_df = result.to_dataframe()
    assert as_df.equals(result.spd_table)
    assert as_df is not result.spd_table

    output_path = tmp_path / "result.json"
    payload = result.to_json(output_path)
    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == payload
    assert payload["summary_metrics"]["score"] == 55.4
    assert payload["summary_metrics"]["uniform_exp_decay_percentile"] == 35.0
    assert payload["summary_metrics"]["exp_decay_multiple_vs_uniform"] == 44.2 / 35.0
    assert len(payload["window_level_data"]) == result.spd_table.height


def test_backtest_result_plot_delegates_to_backtest_module(mocker):
    """BacktestResult.plot should call export_metrics_json and return metrics path."""
    result = BacktestResult(
        spd_table=_sample_spd_df(),
        exp_decay_percentile=44.2,
        win_rate=66.6,
        score=55.4,
        uniform_exp_decay_percentile=35.0,
    )

    mock_export = mocker.patch("stacksats.backtest.export_metrics_json")

    paths = result.plot(output_dir="my-output")

    mock_export.assert_called_once()
    metrics_args = mock_export.call_args.args
    assert "uniform_exp_decay_percentile" in metrics_args[1]
    assert "exp_decay_multiple_vs_uniform" in metrics_args[1]
    assert paths["metrics_json"].endswith("metrics.json")


def test_backtest_result_animate_writes_manifest_and_returns_paths(
    tmp_path: Path, mocker
) -> None:
    result = BacktestResult(
        spd_table=_sample_spd_df(),
        exp_decay_percentile=44.2,
        win_rate=66.6,
        score=55.4,
        uniform_exp_decay_percentile=35.0,
        strategy_id="example",
        strategy_version="1.2.3",
        run_id="run-123",
    )
    source_json = tmp_path / "backtest_result.json"
    source_json.write_text("{}", encoding="utf-8")

    def _fake_render(frame_data, output_path, *, fps, width, height, **kwargs):
        del kwargs
        Path(output_path).write_bytes(b"GIF89a")
        return {
            "gif_path": str(output_path),
            "frames": len(frame_data),
            "fps": fps,
            "width": width,
            "height": height,
        }

    mocker.patch("stacksats.animation_render.render_strategy_vs_uniform_gif", _fake_render)

    paths = result.animate(
        output_dir=str(tmp_path),
        fps=7,
        width=800,
        height=450,
        max_frames=2,
        filename="demo.gif",
        window_mode="non-overlapping",
        source_backtest_json=source_json,
    )
    manifest = json.loads(Path(paths["manifest_json"]).read_text(encoding="utf-8"))

    assert Path(paths["gif"]).exists()
    assert Path(paths["manifest_json"]).exists()
    assert manifest["fps"] == 7
    assert manifest["width"] == 800
    assert manifest["height"] == 450
    assert manifest["window_mode"] == "non-overlapping"
    assert manifest["source_backtest_json"] == str(source_json.resolve())
    assert manifest["strategy_id"] == "example"
    assert manifest["strategy_version"] == "1.2.3"
    assert manifest["run_id"] == "run-123"


def test_backtest_result_animate_records_null_source_backtest_json(
    tmp_path: Path,
    mocker,
) -> None:
    result = BacktestResult(
        spd_table=_sample_spd_df(),
        exp_decay_percentile=44.2,
        win_rate=66.6,
        score=55.4,
        uniform_exp_decay_percentile=35.0,
    )

    def _fake_render(frame_data, output_path, *, fps, width, height, **kwargs):
        del frame_data, fps, width, height, kwargs
        Path(output_path).write_bytes(b"GIF89a")
        return {
            "gif_path": str(output_path),
            "frames": 1,
            "fps": 6,
            "width": 900,
            "height": 600,
        }

    mocker.patch("stacksats.animation_render.render_strategy_vs_uniform_gif", _fake_render)

    paths = result.animate(output_dir=str(tmp_path))
    manifest = json.loads(Path(paths["manifest_json"]).read_text(encoding="utf-8"))

    assert manifest["source_backtest_json"] is None


def test_backtest_result_exp_decay_multiple_none_when_uniform_zero() -> None:
    result = BacktestResult(
        spd_table=_sample_spd_df(),
        exp_decay_percentile=44.2,
        win_rate=66.6,
        score=55.4,
        uniform_exp_decay_percentile=0.0,
    )
    payload = result.to_json()
    assert result.exp_decay_multiple_vs_uniform is None
    assert payload["summary_metrics"]["exp_decay_multiple_vs_uniform"] is None


def test_validation_result_summary_format():
    """ValidationResult summary should include primary gate statuses."""
    result = ValidationResult(
        passed=True,
        forward_leakage_ok=True,
        weight_constraints_ok=True,
        win_rate=72.3,
        win_rate_ok=True,
        messages=["All checks passed."],
    )

    summary = result.summary()
    assert "Validation PASSED" in summary
    assert "No Forward Leakage: True" in summary
    assert "Weight Constraints: True" in summary
    assert "Win Rate: 72.30%" in summary


def test_validation_result_summary_uses_configured_threshold():
    """ValidationResult summary should show configured win-rate threshold."""
    result = ValidationResult(
        passed=False,
        forward_leakage_ok=True,
        weight_constraints_ok=True,
        win_rate=72.3,
        win_rate_ok=False,
        messages=["Win rate below threshold"],
        min_win_rate=75.0,
    )

    summary = result.summary()
    assert ">=75.00%" in summary
    assert "False" in summary


def test_daily_decision_result_summary_and_json(tmp_path: Path) -> None:
    result = DailyDecisionResult(
        status="decided",
        strategy_id="uniform",
        strategy_version="1.0.0",
        run_date="2024-12-31",
        decision_key="decision-123",
        idempotency_hit=False,
        forced_rerun=False,
        weight_today=0.05,
        recommended_notional_usd=50.0,
        recommended_quantity_btc=0.001,
        reference_price_usd=50000.0,
        btc_price_col="price_usd",
        state_db_path=str(tmp_path / "state.sqlite3"),
        artifact_path=str(tmp_path / "decision_result.json"),
        message="Daily decision completed.",
        validation_receipt_id=7,
        validation_passed=True,
        data_hash="data-hash",
        feature_snapshot_hash="feature-hash",
    )

    summary = result.summary()
    assert "Daily Decision DECIDED" in summary
    assert "decision-123" in summary

    output_path = tmp_path / "decision_result.json"
    payload = result.to_json(output_path)
    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == payload
    assert payload["recommended_notional_usd"] == 50.0
    assert payload["recommended_quantity_btc"] == 0.001
    assert payload["reference_price_usd"] == 50000.0


def test_base_strategy_decide_daily_delegates_to_runner(mocker) -> None:
    expected = DailyDecisionResult(
        status="decided",
        strategy_id="uniform",
        strategy_version="1.0.0",
        run_date="2024-12-31",
        decision_key="decision-123",
        idempotency_hit=False,
        forced_rerun=False,
        weight_today=0.05,
        recommended_notional_usd=50.0,
        recommended_quantity_btc=0.001,
        reference_price_usd=50000.0,
        btc_price_col="price_usd",
        state_db_path=".stacksats/run_state.sqlite3",
        artifact_path=None,
        message="ok",
    )
    mock_decide = mocker.patch("stacksats.runner.StrategyRunner.decide_daily", return_value=expected)

    result = UniformStrategy().decide_daily(DecideDailyConfig(run_date="2024-12-31"))

    assert result is expected
    mock_decide.assert_called_once()


def test_execution_receipt_event_to_json() -> None:
    event = ExecutionReceiptEvent(
        decision_key="decision-123",
        event_id="evt-1",
        event_type="submitted",
        event_time="2025-01-01T10:00:00Z",
        metadata={"broker": "paper"},
    )

    payload = event.to_json()

    assert payload["schema_version"]
    assert payload["decision_key"] == "decision-123"
    assert payload["metadata"]["broker"] == "paper"


def test_execution_status_result_summary_and_json() -> None:
    result = ExecutionStatusResult(
        decision_key="decision-123",
        strategy_id="uniform",
        run_date="2025-01-01",
        decision_status="decided",
        execution_status="filled",
        reconciliation_status="matched",
        recommended_notional_usd=10.0,
        recommended_quantity_btc=0.0001,
        filled_notional_usd=10.0,
        filled_quantity_btc=0.0001,
        average_fill_price_usd=100000.0,
        receipt_count=2,
        latest_event_type="filled",
        latest_event_time="2025-01-01T10:02:00Z",
        message="matched",
    )

    summary = result.summary()
    payload = result.to_json()

    assert "Execution FILLED" in summary
    assert payload["reconciliation_status"] == "matched"
    assert payload["receipt_count"] == 2


def test_execution_receipt_history_result_to_json() -> None:
    history = ExecutionReceiptHistoryResult(
        decision_key="decision-123",
        receipts=[
            ExecutionReceiptEvent(
                decision_key="decision-123",
                event_id="evt-1",
                event_type="submitted",
                event_time="2025-01-01T10:00:00Z",
            )
        ],
    )

    payload = history.to_json()

    assert payload["decision_key"] == "decision-123"
    assert payload["receipts"][0]["event_id"] == "evt-1"


@pytest.mark.slow
def test_validate_strategy_passes_with_uniform_strategy():
    """Uniform example strategy should satisfy validation when win-rate floor is relaxed."""
    btc_df = _sample_btc_df()
    result = UniformStrategy().validate(
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            min_win_rate=0.0,
        ),
        btc_df=btc_df,
    )

    assert result.forward_leakage_ok is True
    assert result.weight_constraints_ok is True
    assert bool(result.win_rate_ok) is True
    assert bool(result.passed) is True


def test_validate_strategy_fails_weight_constraints_for_bad_strategy():
    """Invalid strategy should fail at backtest assertion on weight sums."""
    btc_df = _sample_btc_df()

    class BadWeightsStrategy(BaseStrategy):
        strategy_id = "bad-weights"
        version = "1.0.0"

        def build_target_profile(
            self,
            ctx: StrategyContext,
            features_df: pl.DataFrame,
            signals: dict[str, pl.Series],
        ) -> TargetProfile:
            del ctx, signals
            return TargetProfile(
                values=pl.DataFrame({
                    "date": features_df["date"],
                    "value": [float("nan")] * features_df.height,
                }),
                mode="preference",
            )

    import pytest

    with pytest.raises(ValueError, match="target profile must contain finite numeric values"):
        BadWeightsStrategy().validate(
            ValidationConfig(
                start_date="2022-01-01",
                end_date="2023-05-01",
                min_win_rate=0.0,
            ),
            btc_df=btc_df,
        )


@pytest.mark.slow
def test_validate_strategy_observed_only_input_blocks_peeking_strategy():
    """Observed-only context should remove access to rows beyond the decision date."""
    btc_df = _sample_btc_df()

    class LeakyStrategy(BaseStrategy):
        strategy_id = "leaky"
        version = "1.0.0"

        def build_target_profile(
            self,
            ctx: StrategyContext,
            features_df: pl.DataFrame,
            signals: dict[str, pl.Series],
        ) -> pl.DataFrame:
            del signals
            from datetime import timedelta

            dates = features_df["date"]
            lookahead_date = ctx.end_date + timedelta(days=1)
            future_signal = 0.0
            date_list = dates.to_list()
            if date_list and lookahead_date in date_list:
                future_signal = 1.0
            values = [0.0] * features_df.height
            if values:
                values[-1] = future_signal
            return pl.DataFrame({"date": dates, "value": values})

    result = LeakyStrategy().validate(
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-05-01",
            min_win_rate=0.0,
        ),
        btc_df=btc_df,
    )
    assert result.forward_leakage_ok is True


def test_example_strategies_return_valid_weight_vectors():
    """Example strategies should produce non-negative, normalized weights."""
    btc_df = _sample_btc_df()
    from stacksats.model_development import precompute_features

    features_df = precompute_features(btc_df)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)

    for strategy in (
        UniformStrategy(),
        RunDailyPaperStrategy(),
        SimpleZScoreStrategy(),
        MomentumStrategy(),
    ):
        weights = strategy.compute_weights(
            strategy_context_from_features_df(
                features_df,
                start_date,
                end_date,
                end_date,
            )
        )
        assert not weights.is_empty()
        assert bool((weights["weight"] >= 0).all())
        assert np.isclose(float(weights["weight"].sum()), 1.0, atol=1e-8)


def test_example_profile_strategies_return_empty_profile_for_empty_window():
    empty_features = pl.DataFrame(schema={"date": pl.Datetime("us")})
    ctx = strategy_context_from_features_df(
        empty_features,
        datetime(2024, 1, 1),
        datetime(2024, 1, 1),
        datetime(2024, 1, 1),
    )

    simple_profile = SimpleZScoreStrategy().build_target_profile(ctx, empty_features, {})
    momentum_profile = MomentumStrategy().build_target_profile(ctx, empty_features, {})

    assert simple_profile.is_empty()
    assert momentum_profile.is_empty()


def test_daily_run_result_to_json_supports_optional_output_path(tmp_path) -> None:
    result = DailyRunResult(
        status="executed",
        strategy_id="s",
        strategy_version="1.0.0",
        run_date="2024-01-01",
        run_key="rk",
        mode="paper",
        idempotency_hit=False,
        forced_rerun=False,
        weight_today=0.1,
        order_notional_usd=100.0,
        btc_quantity=0.001,
        price_usd=50000.0,
        adapter_name="paper",
        state_db_path="state.sqlite3",
        artifact_path=None,
        message="ok",
    )

    payload = result.to_json()
    assert payload["strategy_id"] == "s"

    output_path = tmp_path / "nested" / "daily.json"
    persisted = result.to_json(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == persisted

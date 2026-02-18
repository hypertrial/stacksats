from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS, MAX_DAILY_WEIGHT, MIN_DAILY_WEIGHT
from stacksats.runner import StrategyRunner, WeightValidationError
from stacksats.strategy_types import (
    BacktestConfig,
    BaseStrategy,
    ExportConfig,
    StrategyContext,
    ValidationConfig,
)


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


class _ProfileOffsetLeakStrategy(BaseStrategy):
    strategy_id = "runner-leak-profile-offset"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del signals
        future = ctx.features_df.loc[
            ctx.features_df.index > ctx.end_date, "PriceUSD_coinmetrics"
        ].dropna()
        offset = float(future.mean()) if not future.empty else 0.0
        # Additive offsets leave softmax weights unchanged, but profile checks should still catch leakage.
        base = np.linspace(-1.0, 1.0, len(features_df), dtype=float)
        return pd.Series(base + offset, index=features_df.index, dtype=float)


class _ProfileMutationStrategy(BaseStrategy):
    strategy_id = "runner-profile-mutation"
    version = "1.0.0"

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> pd.Series:
        del signals
        ctx.features_df["__profile_mutation__"] = 1.0
        if features_df.empty:
            return pd.Series(dtype=float)
        return pd.Series(np.ones(len(features_df), dtype=float), index=features_df.index)


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


def test_validate_weights_rejects_below_min_for_full_contract_span() -> None:
    runner = StrategyRunner()
    base = 1.0 / ALLOCATION_SPAN_DAYS
    weights = np.full(ALLOCATION_SPAN_DAYS, base, dtype=float)
    weights[0] = MIN_DAILY_WEIGHT / 10.0
    deficit = base - weights[0]
    weights[1:] += deficit / (ALLOCATION_SPAN_DAYS - 1)

    with pytest.raises(WeightValidationError, match="below minimum"):
        runner._validate_weights(
            pd.Series(weights),
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-12-30"),
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
            pd.Series(weights),
            window_start=pd.Timestamp("2024-01-01"),
            window_end=pd.Timestamp("2024-12-30"),
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


def test_backtest_win_rate_ignores_tiny_float_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformProposeStrategy()
    spd_table = pd.DataFrame(
        {
            "dynamic_percentile": [40.0 + 1e-12, 50.0 - 1e-12, 60.0 + 5e-11],
            "uniform_percentile": [40.0, 50.0, 60.0],
        }
    )
    monkeypatch.setattr(
        "stacksats.runner.backtest_dynamic_dca",
        lambda *args, **kwargs: (spd_table, 50.0),
    )

    result = runner.backtest(
        strategy,
        BacktestConfig(start_date="2022-01-01", end_date="2022-12-31"),
        btc_df=_btc_df(days=500),
    )

    # Deltas are below win-rate tolerance, so no wins should be counted.
    assert float(result.win_rate) == 0.0


def test_backtest_win_rate_counts_only_deltas_above_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformProposeStrategy()
    spd_table = pd.DataFrame(
        {
            "dynamic_percentile": [40.0 + 1e-8, 50.0 - 1e-8, 60.0 + 2e-10],
            "uniform_percentile": [40.0, 50.0, 60.0],
        }
    )
    monkeypatch.setattr(
        "stacksats.runner.backtest_dynamic_dca",
        lambda *args, **kwargs: (spd_table, 50.0),
    )

    result = runner.backtest(
        strategy,
        BacktestConfig(start_date="2022-01-01", end_date="2022-12-31"),
        btc_df=_btc_df(days=500),
    )

    # Two of three windows exceed tolerance.
    assert result.win_rate == pytest.approx(66.6666666667, rel=0.0, abs=1e-9)


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


def test_strict_fold_checks_skip_on_short_range() -> None:
    runner = StrategyRunner()
    ok, messages = runner._strict_fold_checks(
        strategy=_UniformProposeStrategy(),
        btc_df=_btc_df(days=120),
        start_ts=pd.Timestamp("2024-01-01"),
        end_ts=pd.Timestamp("2024-03-31"),
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
        strategy=_UniformProposeStrategy(),
        btc_df=_btc_df(days=2000),
        start_ts=pd.Timestamp("2021-01-01"),
        end_ts=pd.Timestamp("2025-12-31"),
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
        strategy=_UniformProposeStrategy(),
        btc_df=_btc_df(days=2000),
        start_ts=pd.Timestamp("2021-01-01"),
        end_ts=pd.Timestamp("2025-12-31"),
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
        strategy=_UniformProposeStrategy(),
        btc_df=_btc_df(days=2000),
        start_ts=pd.Timestamp("2022-01-01"),
        end_ts=pd.Timestamp("2023-12-31"),
        config=ValidationConfig(strict=True, shuffled_trials=3, max_shuffled_win_rate=80.0),
    )

    assert ok is False
    assert any("mean shuffled win rate" in msg for msg in messages)
    assert any("Strict shuffled diagnostics" in msg for msg in messages)


def test_strict_shuffled_check_skips_without_price_column() -> None:
    runner = StrategyRunner()
    btc_df = pd.DataFrame({"Other": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))

    ok, messages = runner._strict_shuffled_check(
        strategy=_UniformProposeStrategy(),
        btc_df=btc_df,
        start_ts=pd.Timestamp("2024-01-01"),
        end_ts=pd.Timestamp("2024-01-02"),
        config=ValidationConfig(strict=True, shuffled_trials=3),
    )

    assert ok is True
    assert any("missing PriceUSD_coinmetrics column" in msg for msg in messages)


def test_strict_shuffled_check_skips_when_trials_non_positive() -> None:
    runner = StrategyRunner()

    ok, messages = runner._strict_shuffled_check(
        strategy=_UniformProposeStrategy(),
        btc_df=_btc_df(days=500),
        start_ts=pd.Timestamp("2023-01-01"),
        end_ts=pd.Timestamp("2023-12-31"),
        config=ValidationConfig(strict=True, shuffled_trials=0),
    )

    assert ok is True
    assert any("shuffled_trials <= 0" in msg for msg in messages)


def test_validate_reports_masked_future_weight_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )
    matches = iter([False])
    monkeypatch.setattr(
        runner,
        "_weights_match",
        lambda *args, **kwargs: bool(next(matches)),
    )

    result = runner.validate(
        _UniformProposeStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
        ),
        btc_df=_btc_df(days=1200),
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
    matches = iter([True, False])
    monkeypatch.setattr(
        runner,
        "_weights_match",
        lambda *args, **kwargs: bool(next(matches)),
    )

    result = runner.validate(
        _UniformProposeStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
        ),
        btc_df=_btc_df(days=1200),
    )

    assert bool(result.forward_leakage_ok) is False
    assert any("perturbed-future weights diverge" in msg for msg in result.messages)


def test_validate_reports_profile_masked_future_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )

    result = runner.validate(
        _ProfileOffsetLeakStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
        ),
        btc_df=_btc_df(days=1200),
    )

    assert bool(result.forward_leakage_ok) is False
    assert any("profile values diverge (masked-future)" in msg for msg in result.messages)


def test_validate_strict_passes_and_emits_fold_and_shuffled_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=55.0),
    )

    result = runner.validate(
        _UniformProposeStrategy(),
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2024-12-31",
            min_win_rate=0.0,
            strict=True,
            min_fold_win_rate=20.0,
            max_fold_win_rate_std=35.0,
            max_shuffled_win_rate=80.0,
            shuffled_trials=3,
        ),
        btc_df=_btc_df(days=2000),
    )

    assert bool(result.passed) is True
    assert any("Strict fold diagnostics" in msg for msg in result.messages)
    assert any("Strict shuffled diagnostics" in msg for msg in result.messages)


def test_validate_strict_fails_when_boundary_hit_rate_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformProposeStrategy()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )

    def _boundary_saturating_compute_weights(ctx: StrategyContext) -> pd.Series:
        idx = pd.date_range(ctx.start_date, ctx.end_date, freq="D")
        n = len(idx)
        if n != ALLOCATION_SPAN_DAYS:
            return pd.Series(np.full(n, 1.0 / n, dtype=float), index=idx)
        # 355 days at min, 9 days at max, 1 day as remainder => valid sum and >99% at bounds.
        values = np.full(ALLOCATION_SPAN_DAYS, MIN_DAILY_WEIGHT, dtype=float)
        values[:9] = MAX_DAILY_WEIGHT
        values[-1] = 1.0 - float(values[:-1].sum())
        return pd.Series(values, index=idx)

    monkeypatch.setattr(strategy, "compute_weights", _boundary_saturating_compute_weights)

    result = runner.validate(
        strategy,
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
            strict=True,
            max_boundary_hit_rate=0.85,
        ),
        btc_df=_btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("boundary hit rate" in msg for msg in result.messages)
    assert any("exceeds" in msg for msg in result.messages)


def test_validate_strict_fails_when_locked_prefix_is_not_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = _UniformProposeStrategy()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )

    def _ignoring_locked_prefix_compute_weights(ctx: StrategyContext) -> pd.Series:
        idx = pd.date_range(ctx.start_date, ctx.end_date, freq="D")
        n = len(idx)
        values = np.full(n, 1.0 / n, dtype=float)
        if ctx.locked_weights is not None and n >= 2:
            values[0] += 1e-3
            values[1] -= 1e-3
        return pd.Series(values, index=idx)

    monkeypatch.setattr(strategy, "compute_weights", _ignoring_locked_prefix_compute_weights)

    result = runner.validate(
        strategy,
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
            strict=True,
        ),
        btc_df=_btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("locked prefix was not preserved exactly" in msg for msg in result.messages)


def test_validate_strict_detects_profile_build_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    strategy = _ProfileMutationStrategy()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )

    def _safe_compute_weights(ctx: StrategyContext) -> pd.Series:
        idx = pd.date_range(ctx.start_date, ctx.end_date, freq="D")
        return pd.Series(np.full(len(idx), 1.0 / len(idx), dtype=float), index=idx)

    monkeypatch.setattr(strategy, "compute_weights", _safe_compute_weights)

    result = runner.validate(
        strategy,
        ValidationConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            min_win_rate=0.0,
            strict=True,
        ),
        btc_df=_btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df during profile build" in msg for msg in result.messages)


def test_frame_signature_falls_back_for_unhashable_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame({"obj": [{"a": 1}, {"b": 2}]})
    monkeypatch.setattr(
        "stacksats.runner.pd.util.hash_pandas_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("unhashable")),
    )

    sig = StrategyRunner._frame_signature(df)
    assert isinstance(sig[0], int)
    assert sig[3] == tuple(df.shape)

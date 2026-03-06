from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS, MAX_DAILY_WEIGHT, MIN_DAILY_WEIGHT
from stacksats.runner import StrategyRunner
from stacksats.runner_validation import _ValidationState
from stacksats.strategy_types import StrategyContext, ValidationConfig
from tests.unit.core.runner_validation_testkit import (
    MutatingProposeStrategy,
    ProfileMutationStrategy,
    RandomProposeStrategy,
    UniformProposeStrategy,
    btc_df,
    fast_strict_validation_config,
    patch_skip_weight_and_lock_checks,
)

pytestmark = pytest.mark.slow


def test_validate_strict_rejects_strategy_that_mutates_context_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
    result = runner.validate(
        MutatingProposeStrategy(),
        fast_strict_validation_config(),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df" in message for message in result.messages)


def test_validate_strict_rejects_non_deterministic_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
    result = runner.validate(
        RandomProposeStrategy(),
        fast_strict_validation_config(),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("non-deterministic" in message for message in result.messages)


def test_validate_strict_passes_and_emits_fold_and_shuffled_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    monkeypatch.setattr(runner, "_run_forward_leakage_checks", lambda **kwargs: None)
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=55.0),
    )

    result = runner.validate(
        UniformProposeStrategy(),
        ValidationConfig(
            start_date="2019-01-01",
            end_date="2025-12-31",
            min_win_rate=0.0,
            strict=True,
            bootstrap_trials=25,
            permutation_trials=25,
            block_size=14,
            min_fold_win_rate=20.0,
            max_fold_win_rate_std=35.0,
            max_shuffled_win_rate=80.0,
            shuffled_trials=3,
        ),
        btc_df=btc_df(days=3200),
    )

    assert bool(result.passed) is True
    assert any("Strict fold diagnostics" in msg for msg in result.messages)
    assert any("Strict shuffled diagnostics" in msg for msg in result.messages)


def test_validate_strict_fails_when_boundary_hit_rate_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = UniformProposeStrategy()
    monkeypatch.setattr(runner, "_run_forward_leakage_checks", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_run_locked_prefix_check", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_strict_fold_checks", lambda **kwargs: (True, []))
    monkeypatch.setattr(runner, "_strict_shuffled_check", lambda **kwargs: (True, []))
    monkeypatch.setattr(runner, "_strict_statistical_checks", lambda **kwargs: None)
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
        values = np.full(ALLOCATION_SPAN_DAYS, MIN_DAILY_WEIGHT, dtype=float)
        values[:9] = MAX_DAILY_WEIGHT
        values[-1] = 1.0 - float(values[:-1].sum())
        return pd.Series(values, index=idx)

    monkeypatch.setattr(strategy, "compute_weights", _boundary_saturating_compute_weights)

    result = runner.validate(
        strategy,
        fast_strict_validation_config(max_boundary_hit_rate=0.85),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("boundary hit rate" in msg for msg in result.messages)
    assert any("exceeds" in msg for msg in result.messages)


def test_validate_strict_fails_when_locked_prefix_is_not_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    strategy = UniformProposeStrategy()
    monkeypatch.setattr(
        runner,
        "backtest",
        lambda *args, **kwargs: SimpleNamespace(win_rate=100.0),
    )
    monkeypatch.setattr(runner, "_run_forward_leakage_checks", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_strict_fold_checks", lambda **kwargs: (True, []))
    monkeypatch.setattr(runner, "_strict_shuffled_check", lambda **kwargs: (True, []))
    monkeypatch.setattr(runner, "_strict_statistical_checks", lambda **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_run_weight_constraint_checks",
        lambda **kwargs: kwargs["start_ts"],
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
        fast_strict_validation_config(),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("locked prefix was not preserved exactly" in msg for msg in result.messages)


def test_validate_strict_detects_profile_build_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = StrategyRunner()
    strategy = ProfileMutationStrategy()
    patch_skip_weight_and_lock_checks(monkeypatch, runner)
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
        fast_strict_validation_config(),
        btc_df=btc_df(days=1200),
    )

    assert bool(result.passed) is False
    assert any("mutated ctx.features_df during profile build" in msg for msg in result.messages)


def test_stop_on_mutation_helper_marks_state_only_in_strict_mode() -> None:
    runner = StrategyRunner()
    state = _ValidationState(messages=[])
    assert (
        runner._stop_on_mutation(
            strict_mode=False,
            mutated=True,
            state=state,
        )
        is False
    )
    assert state.strict_checks_ok is True
    assert state.mutation_safe is True

    assert (
        runner._stop_on_mutation(
            strict_mode=True,
            mutated=True,
            state=state,
        )
        is True
    )
    assert state.strict_checks_ok is False
    assert state.mutation_safe is False
    assert any("mutated ctx.features_df in-place" in msg for msg in state.messages)


def test_strict_statistical_checks_skip_when_spd_table_missing() -> None:
    runner = StrategyRunner()
    strategy = UniformProposeStrategy()
    state = _ValidationState(messages=[], diagnostics={})
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    features = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)

    runner._strict_statistical_checks(
        strategy=strategy,
        btc_df=btc_df(days=60),
        features_df=features,
        start_ts=index.min(),
        end_ts=index.max(),
        config=fast_strict_validation_config(),
        state=state,
        backtest_result=SimpleNamespace(),
    )

    assert any("lacks window diagnostics" in message for message in state.messages)
    assert state.diagnostics == {}


def test_strict_statistical_checks_emits_failures_and_drift_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = StrategyRunner()
    state = _ValidationState(messages=[], diagnostics={})
    start_ts = pd.Timestamp("2024-01-01")
    end_ts = pd.Timestamp("2024-12-31")
    test_start = pd.Timestamp("2024-10-01")
    test_end = pd.Timestamp("2024-10-30")
    features_idx = pd.date_range(start_ts, end_ts, freq="D")
    features = pd.DataFrame({"feat": np.linspace(0.0, 1.0, len(features_idx))}, index=features_idx)
    backtest_result = SimpleNamespace(
        spd_table=pd.DataFrame(
            {
                "dynamic_percentile": [55.0, 40.0, 62.0],
                "uniform_percentile": [50.0, 50.0, 50.0],
            }
        )
    )

    class _RequiredFeatureStrategy(UniformProposeStrategy):
        def required_feature_columns(self) -> tuple[str, ...]:
            return ("feat",)

    strategy = _RequiredFeatureStrategy()

    monkeypatch.setattr(
        "stacksats.runner_validation.anchored_window_excess",
        lambda *args, **kwargs: pd.Series([0.5, -0.2, 0.1], dtype=float),
    )
    monkeypatch.setattr(
        "stacksats.runner_validation.block_bootstrap_confidence_interval",
        lambda *args, **kwargs: SimpleNamespace(lower=-0.05, upper=0.25),
    )
    monkeypatch.setattr(
        "stacksats.runner_validation.paired_block_permutation_pvalue",
        lambda *args, **kwargs: 0.9,
    )
    monkeypatch.setattr(
        "stacksats.runner_validation.build_purged_walk_forward_folds",
        lambda *args, **kwargs: [(start_ts, test_start - pd.Timedelta(days=1), test_start, test_end)],
    )

    def _materialize(*args, **kwargs) -> pd.DataFrame:
        start_date = kwargs["start_date"]
        end_date = kwargs["end_date"]
        idx = pd.date_range(start_date, end_date, freq="D")
        base = 0.0 if end_date < test_start else 100.0
        return pd.DataFrame({"feat": np.full(len(idx), base, dtype=float)}, index=idx)

    monkeypatch.setattr(runner, "_materialize_strategy_features", _materialize)
    monkeypatch.setattr(
        "stacksats.runner_validation.population_stability_index",
        lambda *args, **kwargs: 0.5,
    )
    monkeypatch.setattr(
        "stacksats.runner_validation.ks_statistic",
        lambda *args, **kwargs: 0.6,
    )

    runner._strict_statistical_checks(
        strategy=strategy,
        btc_df=btc_df(days=800),
        features_df=features,
        start_ts=start_ts,
        end_ts=end_ts,
        config=ValidationConfig(
            strict=True,
            min_bootstrap_ci_lower_excess=0.0,
            max_permutation_pvalue=0.1,
            max_feature_psi=0.25,
            max_feature_ks=0.25,
            bootstrap_trials=5,
            permutation_trials=5,
            block_size=7,
        ),
        state=state,
        backtest_result=backtest_result,
    )

    assert state.strict_checks_ok is False
    assert state.diagnostics["bootstrap_ci"] == {"lower": -0.05, "upper": 0.25}
    assert state.diagnostics["permutation_pvalue"] == 0.9
    assert state.diagnostics["feature_drift"] == {"max_psi": 0.5, "max_ks": 0.6}
    assert any("bootstrap lower CI" in message for message in state.messages)
    assert any("permutation p-value" in message for message in state.messages)
    assert any("feature PSI" in message for message in state.messages)
    assert any("feature KS" in message for message in state.messages)

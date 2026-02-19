"""Validation mixin and related types for StrategyRunner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .framework_contract import ALLOCATION_SPAN_DAYS, MAX_DAILY_WEIGHT, MIN_DAILY_WEIGHT
from .prelude import WINDOW_OFFSET as DEFAULT_WINDOW_OFFSET
from .runner_helpers import (
    build_fold_ranges,
    frame_signature,
    perturb_future_features,
    profile_values,
    weights_match,
)
from .strategy_types import (
    BacktestConfig,
    BaseStrategy,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)

WIN_RATE_TOLERANCE = 1e-10
STRICT_MUTATION_MESSAGE = "Strict check failed: strategy mutated ctx.features_df in-place."
STRICT_PROFILE_MUTATION_MESSAGE = (
    "Strict check failed: strategy mutated ctx.features_df during profile build."
)


@dataclass
class _ValidationState:
    messages: list[str]
    forward_leakage_ok: bool = True
    weight_constraints_ok: bool = True
    strict_checks_ok: bool = True
    mutation_safe: bool = True
    deterministic_ok: bool = True


class WeightValidationError(ValueError):
    """Raised when strategy weights violate required constraints."""


class StrategyRunnerValidationMixin:
    """Validation and strict-check helpers for strategy execution."""

    WINDOW_OFFSET = DEFAULT_WINDOW_OFFSET
    ITER_RANGE = range
    BUILD_FOLD_RANGES = staticmethod(build_fold_ranges)

    def _validate_weights(
        self,
        weights: pd.Series,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> None:
        if weights.empty:
            return
        weight_sum = float(weights.sum())
        if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=1e-8):
            raise WeightValidationError(
                f"Weights for range {window_start.date()} to {window_end.date()} "
                f"sum to {weight_sum:.10f}, expected 1.0"
            )
        if bool((weights < 0).any()):
            raise WeightValidationError(
                f"Weights for range {window_start.date()} to {window_end.date()} "
                "contain negative values"
            )
        if len(weights) == ALLOCATION_SPAN_DAYS:
            if bool((weights < (MIN_DAILY_WEIGHT - 1e-12)).any()):
                raise WeightValidationError(
                    f"Weights for range {window_start.date()} to {window_end.date()} "
                    f"contain values below minimum {MIN_DAILY_WEIGHT}"
                )
            if bool((weights > (MAX_DAILY_WEIGHT + 1e-12)).any()):
                raise WeightValidationError(
                    f"Weights for range {window_start.date()} to {window_end.date()} "
                    f"contain values above maximum {MAX_DAILY_WEIGHT}"
                )

    @staticmethod
    def _weights_match(lhs: pd.Series, rhs: pd.Series, *, atol: float = 1e-12) -> bool:
        return weights_match(lhs, rhs, atol=atol)

    @staticmethod
    def _profile_values(profile: TargetProfile | pd.Series) -> pd.Series:
        return profile_values(profile)

    @staticmethod
    def _frame_signature(df: pd.DataFrame) -> tuple:
        return frame_signature(df)

    @staticmethod
    def _perturb_future_features(features_df: pd.DataFrame, probe: pd.Timestamp) -> pd.DataFrame:
        return perturb_future_features(features_df, probe)

    @staticmethod
    def _compute_win_rate(
        spd_table: pd.DataFrame,
        *,
        atol: float = WIN_RATE_TOLERANCE,
    ) -> float:
        """Compute win rate with tolerance against floating-point noise."""
        if spd_table.empty:
            return 0.0
        dynamic = pd.to_numeric(spd_table["dynamic_percentile"], errors="coerce")
        uniform = pd.to_numeric(spd_table["uniform_percentile"], errors="coerce")
        delta = (dynamic - uniform).to_numpy(dtype=float)
        finite = np.isfinite(delta)
        if not finite.any():
            return 0.0
        wins = delta[finite] > float(atol)
        return float(wins.mean() * 100.0)

    @classmethod
    def _build_fold_ranges(
        cls,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        return cls.BUILD_FOLD_RANGES(start_ts, end_ts)

    def _strict_fold_checks(
        self,
        strategy: BaseStrategy,
        btc_df: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        config: ValidationConfig,
    ) -> tuple[bool, list[str]]:
        messages: list[str] = []
        folds = self._build_fold_ranges(start_ts, end_ts)
        if len(folds) < 2:
            messages.append("Strict fold checks skipped: insufficient date range for >=2 folds.")
            return True, messages

        fold_win_rates: list[float] = []
        for idx, (fold_start, fold_end) in enumerate(folds, start=1):
            fold_result = self.backtest(
                strategy,
                BacktestConfig(
                    start_date=fold_start.strftime("%Y-%m-%d"),
                    end_date=fold_end.strftime("%Y-%m-%d"),
                    strategy_label=f"strict-fold-{idx}",
                ),
                btc_df=btc_df,
            )
            fold_win_rates.append(float(fold_result.win_rate))

        if len(fold_win_rates) < 2:
            messages.append("Strict fold checks skipped: not enough valid fold results.")
            return True, messages

        min_fold = float(np.min(fold_win_rates))
        std_fold = float(np.std(fold_win_rates))
        ok = True
        if min_fold < float(config.min_fold_win_rate):
            ok = False
            messages.append(
                "Strict fold check failed: minimum fold win rate "
                f"{min_fold:.2f}% < {config.min_fold_win_rate:.2f}%."
            )
        if std_fold > float(config.max_fold_win_rate_std):
            ok = False
            messages.append(
                "Strict fold check failed: fold win-rate std "
                f"{std_fold:.2f} > {config.max_fold_win_rate_std:.2f}."
            )
        messages.append(
            "Strict fold diagnostics: "
            f"fold_win_rates={[round(x, 2) for x in fold_win_rates]}."
        )
        return ok, messages

    def _strict_shuffled_check(
        self,
        strategy: BaseStrategy,
        btc_df: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        config: ValidationConfig,
    ) -> tuple[bool, list[str]]:
        messages: list[str] = []
        if "PriceUSD_coinmetrics" not in btc_df.columns:
            messages.append("Strict shuffled check skipped: missing PriceUSD_coinmetrics column.")
            return True, messages
        if config.shuffled_trials <= 0:
            messages.append("Strict shuffled check skipped: shuffled_trials <= 0.")
            return True, messages

        shuffled_win_rates: list[float] = []
        for seed in self.ITER_RANGE(int(config.shuffled_trials)):
            shuffled_df = btc_df.copy(deep=True)
            window_values = np.array(
                shuffled_df.loc[start_ts:end_ts, "PriceUSD_coinmetrics"].to_numpy(dtype=float),
                dtype=float,
                copy=True,
            )
            if window_values.size == 0:
                messages.append("Strict shuffled check skipped: empty validation window.")
                return True, messages
            rng = np.random.default_rng(seed)
            rng.shuffle(window_values)
            shuffled_df.loc[start_ts:end_ts, "PriceUSD_coinmetrics"] = window_values
            shuffled_result = self.backtest(
                strategy,
                BacktestConfig(
                    start_date=start_ts.strftime("%Y-%m-%d"),
                    end_date=end_ts.strftime("%Y-%m-%d"),
                    strategy_label=f"strict-shuffled-{seed}",
                ),
                btc_df=shuffled_df,
            )
            shuffled_win_rates.append(float(shuffled_result.win_rate))

        if len(shuffled_win_rates) == 0:
            messages.append("Strict shuffled check skipped: no shuffled runs completed.")
            return True, messages

        mean_shuffled = float(np.mean(shuffled_win_rates))
        ok = mean_shuffled <= float(config.max_shuffled_win_rate)
        if not ok:
            messages.append(
                "Strict shuffled check failed: mean shuffled win rate "
                f"{mean_shuffled:.2f}% > {config.max_shuffled_win_rate:.2f}%."
            )
        messages.append(
            "Strict shuffled diagnostics: "
            f"shuffled_win_rates={[round(x, 2) for x in shuffled_win_rates]}."
        )
        return ok, messages

    @staticmethod
    def _mark_mutation_failure(
        state: _ValidationState,
        *,
        message: str = STRICT_MUTATION_MESSAGE,
    ) -> None:
        state.mutation_safe = False
        state.strict_checks_ok = False
        state.messages.append(message)

    def _compute_with_mutation_guard(
        self,
        strategy: BaseStrategy,
        ctx: StrategyContext,
        *,
        strict_mode: bool,
    ) -> tuple[pd.Series, bool]:
        if not strict_mode:
            return strategy.compute_weights(ctx), False
        before = self._frame_signature(ctx.features_df)
        weights = strategy.compute_weights(ctx)
        after = self._frame_signature(ctx.features_df)
        return weights, before != after

    def _build_profile_with_mutation_guard(
        self,
        strategy: BaseStrategy,
        ctx: StrategyContext,
        *,
        strict_mode: bool,
    ) -> tuple[pd.Series, bool]:
        before = self._frame_signature(ctx.features_df) if strict_mode else ()
        profile_features = strategy.transform_features(ctx)
        profile_signals = strategy.build_signals(ctx, profile_features)
        profile = strategy.build_target_profile(ctx, profile_features, profile_signals)
        after = self._frame_signature(ctx.features_df) if strict_mode else before
        return self._profile_values(profile), before != after

    def _run_forward_leakage_checks(
        self,
        *,
        strategy: BaseStrategy,
        features_df: pd.DataFrame,
        backtest_idx: pd.DatetimeIndex,
        start_ts: pd.Timestamp,
        strict_mode: bool,
        has_propose_hook: bool,
        has_profile_hook: bool,
        probe_step: int,
        state: _ValidationState,
    ) -> None:
        window_offset = self.WINDOW_OFFSET
        for probe in backtest_idx[::probe_step]:
            window_start = max(start_ts, probe - window_offset)
            if window_start > probe:
                continue

            full_ctx = StrategyContext(
                features_df=features_df.copy(deep=True),
                start_date=window_start,
                end_date=probe,
                current_date=probe,
            )
            full_weights, full_mutated = self._compute_with_mutation_guard(
                strategy,
                full_ctx,
                strict_mode=strict_mode,
            )
            if strict_mode and full_mutated:
                self._mark_mutation_failure(state)
                break

            if strict_mode:
                repeat_ctx = StrategyContext(
                    features_df=features_df.copy(deep=True),
                    start_date=window_start,
                    end_date=probe,
                    current_date=probe,
                )
                repeat_weights, repeat_mutated = self._compute_with_mutation_guard(
                    strategy,
                    repeat_ctx,
                    strict_mode=True,
                )
                if repeat_mutated:
                    self._mark_mutation_failure(state)
                    break
                if not self._weights_match(full_weights, repeat_weights):
                    state.deterministic_ok = False
                    state.strict_checks_ok = False
                    state.messages.append(
                        "Strict check failed: strategy is non-deterministic for identical inputs."
                    )
                    break

            masked_features = features_df.copy(deep=True)
            masked_features.loc[masked_features.index > probe, :] = np.nan
            masked_ctx = StrategyContext(
                features_df=masked_features,
                start_date=window_start,
                end_date=probe,
                current_date=probe,
            )
            masked_weights, masked_mutated = self._compute_with_mutation_guard(
                strategy,
                masked_ctx,
                strict_mode=strict_mode,
            )
            if strict_mode and masked_mutated:
                self._mark_mutation_failure(state)
                break

            perturbed_ctx = StrategyContext(
                features_df=self._perturb_future_features(features_df, probe),
                start_date=window_start,
                end_date=probe,
                current_date=probe,
            )
            perturbed_weights, perturbed_mutated = self._compute_with_mutation_guard(
                strategy,
                perturbed_ctx,
                strict_mode=strict_mode,
            )
            if strict_mode and perturbed_mutated:
                self._mark_mutation_failure(state)
                break

            prefix_idx = full_weights.index[full_weights.index <= probe]
            if len(prefix_idx) == 0:
                continue
            full_prefix = full_weights.loc[prefix_idx]
            masked_prefix = masked_weights.reindex(prefix_idx)
            perturbed_prefix = perturbed_weights.reindex(prefix_idx)

            if not self._weights_match(full_prefix, masked_prefix):
                state.forward_leakage_ok = False
                state.messages.append(
                    "Forward leakage detected near "
                    f"{probe.strftime('%Y-%m-%d')}: masked-future weights diverge."
                )
                break
            if not self._weights_match(full_prefix, perturbed_prefix):
                state.forward_leakage_ok = False
                state.messages.append(
                    "Forward leakage detected near "
                    f"{probe.strftime('%Y-%m-%d')}: perturbed-future weights diverge."
                )
                break

            if has_profile_hook and not has_propose_hook:
                profile_full_ctx = StrategyContext(
                    features_df=features_df.copy(deep=True),
                    start_date=window_start,
                    end_date=probe,
                    current_date=probe,
                )
                profile_masked_features = features_df.copy(deep=True)
                profile_masked_features.loc[profile_masked_features.index > probe, :] = np.nan
                profile_masked_ctx = StrategyContext(
                    features_df=profile_masked_features,
                    start_date=window_start,
                    end_date=probe,
                    current_date=probe,
                )
                profile_perturbed_ctx = StrategyContext(
                    features_df=self._perturb_future_features(features_df, probe),
                    start_date=window_start,
                    end_date=probe,
                    current_date=probe,
                )
                full_profile_series, full_profile_mutated = self._build_profile_with_mutation_guard(
                    strategy,
                    profile_full_ctx,
                    strict_mode=strict_mode,
                )
                (
                    masked_profile_series,
                    masked_profile_mutated,
                ) = self._build_profile_with_mutation_guard(
                    strategy,
                    profile_masked_ctx,
                    strict_mode=strict_mode,
                )
                (
                    perturbed_profile_series,
                    perturbed_profile_mutated,
                ) = self._build_profile_with_mutation_guard(
                    strategy,
                    profile_perturbed_ctx,
                    strict_mode=strict_mode,
                )
                if strict_mode and (
                    full_profile_mutated or masked_profile_mutated or perturbed_profile_mutated
                ):
                    self._mark_mutation_failure(
                        state,
                        message=STRICT_PROFILE_MUTATION_MESSAGE,
                    )
                    break

                full_profile_prefix = full_profile_series.reindex(prefix_idx)
                masked_profile_prefix = masked_profile_series.reindex(prefix_idx)
                perturbed_profile_prefix = perturbed_profile_series.reindex(prefix_idx)
                if not self._weights_match(full_profile_prefix, masked_profile_prefix):
                    state.forward_leakage_ok = False
                    state.messages.append(
                        "Forward leakage detected near "
                        f"{probe.strftime('%Y-%m-%d')}: profile values diverge (masked-future)."
                    )
                    break
                if not self._weights_match(full_profile_prefix, perturbed_profile_prefix):
                    state.forward_leakage_ok = False
                    state.messages.append(
                        "Forward leakage detected near "
                        f"{probe.strftime('%Y-%m-%d')}: profile values diverge (perturbed-future)."
                    )
                    break

    def _run_weight_constraint_checks(
        self,
        *,
        strategy: BaseStrategy,
        features_df: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        strict_mode: bool,
        config: ValidationConfig,
        state: _ValidationState,
    ) -> pd.Timestamp:
        window_offset = self.WINDOW_OFFSET
        max_window_start = end_ts - window_offset
        boundary_hits = 0
        boundary_total = 0
        if start_ts <= max_window_start:
            window_starts = pd.date_range(start=start_ts, end=max_window_start, freq="D")
            for window_start in window_starts:
                window_end = window_start + window_offset
                ctx = StrategyContext(
                    features_df=features_df.copy(deep=True) if strict_mode else features_df,
                    start_date=window_start,
                    end_date=window_end,
                    current_date=window_end,
                )
                weights, mutated = self._compute_with_mutation_guard(
                    strategy,
                    ctx,
                    strict_mode=strict_mode,
                )
                if strict_mode and mutated:
                    self._mark_mutation_failure(state)
                    break
                if weights.empty:
                    continue
                try:
                    self._validate_weights(weights, window_start, window_end)
                except WeightValidationError as exc:
                    state.weight_constraints_ok = False
                    state.messages.append(str(exc))
                    break
                if strict_mode and len(weights) == ALLOCATION_SPAN_DAYS:
                    arr = weights.to_numpy(dtype=float)
                    at_bounds = np.isclose(arr, MIN_DAILY_WEIGHT, atol=1e-12) | np.isclose(
                        arr,
                        MAX_DAILY_WEIGHT,
                        atol=1e-12,
                    )
                    boundary_hits += int(at_bounds.sum())
                    boundary_total += int(len(arr))

        if strict_mode and boundary_total > 0:
            boundary_hit_rate = boundary_hits / boundary_total
            state.messages.append(
                "Strict boundary diagnostics: "
                f"{boundary_hit_rate * 100:.2f}% of days hit MIN/MAX bounds."
            )
            if boundary_hit_rate > float(config.max_boundary_hit_rate):
                state.strict_checks_ok = False
                state.messages.append(
                    "Strict check failed: boundary hit rate "
                    f"{boundary_hit_rate * 100:.2f}% exceeds "
                    f"{config.max_boundary_hit_rate * 100:.2f}%."
                )
        return max_window_start

    def _run_locked_prefix_check(
        self,
        *,
        strategy: BaseStrategy,
        features_df: pd.DataFrame,
        start_ts: pd.Timestamp,
        max_window_start: pd.Timestamp,
        strict_mode: bool,
        state: _ValidationState,
    ) -> None:
        if not (strict_mode and state.strict_checks_ok and start_ts <= max_window_start):
            return

        window_offset = self.WINDOW_OFFSET
        lock_start = start_ts
        lock_end = lock_start + window_offset
        lock_mid_offset = max(ALLOCATION_SPAN_DAYS // 2 - 1, 0)
        lock_current = min(lock_start + pd.Timedelta(days=lock_mid_offset), lock_end)
        base_lock_ctx = StrategyContext(
            features_df=features_df.copy(deep=True),
            start_date=lock_start,
            end_date=lock_end,
            current_date=lock_current,
        )
        base_lock_weights, base_mutated = self._compute_with_mutation_guard(
            strategy,
            base_lock_ctx,
            strict_mode=True,
        )
        if base_mutated:
            self._mark_mutation_failure(state)
            return
        if base_lock_weights.empty:
            return

        n_past = int((base_lock_weights.index <= lock_current).sum())
        locked_prefix = base_lock_weights.iloc[:n_past].to_numpy(dtype=float)
        locked_ctx = StrategyContext(
            features_df=self._perturb_future_features(features_df, lock_current),
            start_date=lock_start,
            end_date=lock_end,
            current_date=lock_current,
            locked_weights=locked_prefix,
        )
        locked_run_weights, locked_mutated = self._compute_with_mutation_guard(
            strategy,
            locked_ctx,
            strict_mode=True,
        )
        if locked_mutated:
            self._mark_mutation_failure(state)
            return
        if n_past > 0:
            observed_prefix = locked_run_weights.iloc[:n_past].to_numpy(dtype=float)
            if not np.allclose(observed_prefix, locked_prefix, atol=1e-12, rtol=0.0):
                state.strict_checks_ok = False
                state.messages.append(
                    "Strict check failed: locked prefix was not preserved exactly."
                )

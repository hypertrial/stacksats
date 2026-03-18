"""Validation mixin and related types for StrategyRunner."""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .feature_registry import DEFAULT_FEATURE_REGISTRY
from .framework_contract import (
    ALLOCATION_SPAN_DAYS,
    MAX_DAILY_WEIGHT,
    MIN_DAILY_WEIGHT,
    _to_naive_utc,
)
from .prelude import WINDOW_OFFSET as DEFAULT_WINDOW_OFFSET
from .runner_helpers import (
    build_window_index,
    build_fold_ranges,
    frame_signature,
    perturb_future_source_data,
    perturb_future_features,
    profile_values,
    slice_window_or_filter,
    weights_match,
)
from .statistical_validation import (
    anchored_window_excess,
    block_bootstrap_confidence_interval,
    build_purged_walk_forward_folds,
    ks_statistic,
    paired_block_permutation_pvalue,
    population_stability_index,
)
from .strategy_types import (
    BacktestConfig,
    BaseStrategy,
    StrategyContext,
    StrategyLazyContext,
    TargetProfile,
    ValidationConfig,
    strategy_context_from_features_df,
)

WIN_RATE_TOLERANCE = 1e-10
STRICT_MUTATION_MESSAGE = (
    "Strict check failed: strategy mutated ctx.features (feature data) in-place."
)
STRICT_PROFILE_MUTATION_MESSAGE = (
    "Strict check failed: strategy mutated ctx.features (feature data) during profile build."
)


@dataclass
class _ValidationState:
    messages: list[str]
    forward_leakage_ok: bool = True
    weight_constraints_ok: bool = True
    strict_checks_ok: bool = True
    mutation_safe: bool = True
    deterministic_ok: bool = True
    diagnostics: dict[str, object] | None = None


class WeightValidationError(ValueError):
    """Raised when strategy weights violate required constraints."""


@dataclass
class _HotPathProfile:
    materialize_calls: int = 0
    materialize_cache_hits: int = 0
    compute_weights_calls: int = 0
    compute_weights_cache_hits: int = 0
    forward_leakage_seconds: float = 0.0
    weight_constraint_seconds: float = 0.0
    backtest_seconds: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "materialize_calls": self.materialize_calls,
            "materialize_cache_hits": self.materialize_cache_hits,
            "compute_weights_calls": self.compute_weights_calls,
            "compute_weights_cache_hits": self.compute_weights_cache_hits,
            "forward_leakage_seconds": self.forward_leakage_seconds,
            "weight_constraint_seconds": self.weight_constraint_seconds,
            "backtest_seconds": self.backtest_seconds,
        }


@dataclass
class _StrategyRuntimeCache:
    strategy_key: tuple[str, str]
    base_source_id: int
    feature_frames: dict[tuple[str, str, str, str], pl.DataFrame] = field(
        default_factory=dict
    )
    weight_frames: dict[tuple[str, str, str, str], pl.DataFrame] = field(
        default_factory=dict
    )
    profile: _HotPathProfile = field(default_factory=_HotPathProfile)


class StrategyRunnerValidationMixin:
    """Validation and strict-check helpers for strategy execution."""

    WINDOW_OFFSET = DEFAULT_WINDOW_OFFSET
    ITER_RANGE = range
    BUILD_FOLD_RANGES = staticmethod(build_fold_ranges)
    FEATURE_REGISTRY = DEFAULT_FEATURE_REGISTRY

    @staticmethod
    def _cache_dt(value: dt.datetime) -> str:
        return value.strftime("%Y-%m-%d")

    @staticmethod
    def _runtime_cache_key(
        namespace: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        current_date: dt.datetime,
    ) -> tuple[str, str, str, str]:
        return (
            namespace,
            StrategyRunnerValidationMixin._cache_dt(start_date),
            StrategyRunnerValidationMixin._cache_dt(end_date),
            StrategyRunnerValidationMixin._cache_dt(current_date),
        )

    def _active_runtime_cache(self) -> _StrategyRuntimeCache | None:
        cache = getattr(self, "_strategy_runtime_cache", None)
        return cache if isinstance(cache, _StrategyRuntimeCache) else None

    def _last_runtime_profile(self) -> dict[str, float | int]:
        profile = getattr(self, "_last_runtime_hotpath_profile", None)
        return profile if isinstance(profile, dict) else {}

    def _runtime_profile(self) -> _HotPathProfile | None:
        cache = self._active_runtime_cache()
        return cache.profile if cache is not None else None

    def _ensure_runtime_cache(
        self,
        strategy: BaseStrategy,
        btc_df: pl.DataFrame,
    ) -> tuple[_StrategyRuntimeCache, bool]:
        metadata = strategy.metadata()
        strategy_key = (metadata.strategy_id, metadata.version)
        cache = self._active_runtime_cache()
        if (
            cache is not None
            and cache.strategy_key == strategy_key
            and cache.base_source_id == id(btc_df)
        ):
            return cache, False
        cache = _StrategyRuntimeCache(
            strategy_key=strategy_key,
            base_source_id=id(btc_df),
        )
        self._strategy_runtime_cache = cache
        return cache, True

    def _release_runtime_cache(self, *, clear: bool) -> None:
        cache = self._active_runtime_cache()
        if cache is not None:
            self._last_runtime_hotpath_profile = cache.profile.to_dict()
        if clear and hasattr(self, "_strategy_runtime_cache"):
            delattr(self, "_strategy_runtime_cache")

    def _validate_weights(
        self,
        weights: pl.DataFrame,
        window_start: dt.datetime,
        window_end: dt.datetime,
    ) -> None:
        if weights.is_empty() or "weight" not in weights.columns:
            return
        w = weights["weight"]
        weight_sum = float(w.sum())
        if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=1e-8):
            raise WeightValidationError(
                f"Weights for range {window_start.date()} to {window_end.date()} "
                f"sum to {weight_sum:.10f}, expected 1.0"
            )
        if bool((w < 0).any()):
            raise WeightValidationError(
                f"Weights for range {window_start.date()} to {window_end.date()} "
                "contain negative values"
            )
        if weights.height == ALLOCATION_SPAN_DAYS:
            if bool((w < (MIN_DAILY_WEIGHT - 1e-12)).any()):
                raise WeightValidationError(
                    f"Weights for range {window_start.date()} to {window_end.date()} "
                    f"contain values below minimum {MIN_DAILY_WEIGHT}"
                )
            if bool((w > (MAX_DAILY_WEIGHT + 1e-12)).any()):
                raise WeightValidationError(
                    f"Weights for range {window_start.date()} to {window_end.date()} "
                    f"contain values above maximum {MAX_DAILY_WEIGHT}"
                )

    @staticmethod
    def _weights_match(lhs: pl.DataFrame, rhs: pl.DataFrame, *, atol: float = 1e-12) -> bool:
        return weights_match(lhs, rhs, atol=atol)

    @staticmethod
    def _profile_values(profile: TargetProfile | pl.DataFrame) -> pl.DataFrame:
        return profile_values(profile)

    @staticmethod
    def _frame_signature(df: pl.DataFrame) -> tuple:
        return frame_signature(df)

    @staticmethod
    def _perturb_future_features(features_df: pl.DataFrame, probe: dt.datetime) -> pl.DataFrame:
        return perturb_future_features(features_df, probe)

    @staticmethod
    def _perturb_future_source_data(btc_df: pl.DataFrame, probe: dt.datetime) -> pl.DataFrame:
        return perturb_future_source_data(btc_df, probe)

    def _materialize_strategy_features(
        self,
        strategy: BaseStrategy,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        current_date: dt.datetime,
        cache_namespace: str | None = None,
    ) -> pl.DataFrame:
        cache = self._active_runtime_cache()
        cache_key = None
        if cache is not None and cache_namespace is not None:
            cache_key = self._runtime_cache_key(
                f"features:{cache_namespace}",
                start_date,
                end_date,
                current_date,
            )
            cached = cache.feature_frames.get(cache_key)
            if cached is not None:
                cache.profile.materialize_cache_hits += 1
                return cached.clone()
        frame = self.FEATURE_REGISTRY.materialize_for_strategy(
            strategy,
            btc_df,
            start_date=start_date,
            end_date=end_date,
            current_date=current_date,
        )
        if cache is not None:
            cache.profile.materialize_calls += 1
            if cache_key is not None:
                cache.feature_frames[cache_key] = frame.clone()
        return frame

    def _materialize_strategy_features_lazy(
        self,
        strategy: BaseStrategy,
        btc_df: pl.DataFrame,
        *,
        start_date: dt.datetime,
        end_date: dt.datetime,
        current_date: dt.datetime,
    ) -> pl.LazyFrame:
        return self.FEATURE_REGISTRY.materialize_for_strategy_lazy(
            strategy,
            btc_df,
            start_date=start_date,
            end_date=end_date,
            current_date=current_date,
        )

    def _compute_strategy_weights(
        self,
        strategy: BaseStrategy,
        *,
        features_df: pl.DataFrame,
        btc_df: pl.DataFrame | None = None,
        start_date: dt.datetime,
        end_date: dt.datetime,
        current_date: dt.datetime,
        strict_mode: bool,
        cache_namespace: str | None = None,
    ) -> tuple[pl.DataFrame, bool]:
        cache = self._active_runtime_cache()
        cache_key = None
        if cache is not None and cache_namespace is not None and not strict_mode:
            cache_key = self._runtime_cache_key(
                f"weights:{cache_namespace}",
                start_date,
                end_date,
                current_date,
            )
            cached = cache.weight_frames.get(cache_key)
            if cached is not None:
                cache.profile.compute_weights_cache_hits += 1
                return cached.clone(), False

        if (
            btc_df is not None
            and strategy.intent_mode() == "profile"
            and strategy.__class__.build_target_profile_lazy
            is not BaseStrategy.build_target_profile_lazy
        ):
            observed_end = current_date if current_date <= end_date else end_date
            date_range = pl.datetime_range(
                start_date,
                observed_end,
                interval="1d",
                eager=True,
            )
            lazy_ctx = StrategyLazyContext(
                start_date=start_date,
                end_date=end_date,
                current_date=current_date,
                locked_weights=None,
            )
            features_lf = self._materialize_strategy_features_lazy(
                strategy,
                btc_df,
                start_date=start_date,
                end_date=end_date,
                current_date=current_date,
            )
            weights = strategy._compute_weights_lazy_profile_from_features_lf(
                lazy_ctx,
                features_lf,
                date_range=date_range,
            )
            if cache is not None:
                cache.profile.compute_weights_calls += 1
                if cache_key is not None:
                    cache.weight_frames[cache_key] = weights.clone()
            return weights, False

        ctx = strategy_context_from_features_df(
            features_df,
            start_date,
            end_date,
            current_date,
            required_columns=tuple(strategy.required_feature_columns()),
            as_of_date=current_date,
        )
        weights, mutated = self._compute_with_mutation_guard(
            strategy,
            ctx,
            strict_mode=strict_mode,
        )
        if cache is not None:
            cache.profile.compute_weights_calls += 1
            if cache_key is not None and not mutated:
                cache.weight_frames[cache_key] = weights.clone()
        return weights, mutated

    @staticmethod
    def _compute_win_rate(
        spd_table: pl.DataFrame,
        *,
        atol: float = WIN_RATE_TOLERANCE,
    ) -> float:
        """Compute win rate with tolerance against floating-point noise."""
        if spd_table.is_empty():
            return 0.0
        dynamic = spd_table["dynamic_percentile"].cast(pl.Float64, strict=False)
        uniform = spd_table["uniform_percentile"].cast(pl.Float64, strict=False)
        delta = (dynamic - uniform).to_numpy().astype(float)
        finite = np.isfinite(delta)
        if not finite.any():
            return 0.0
        wins = delta[finite] > float(atol)
        return float(wins.mean() * 100.0)

    @classmethod
    def _build_fold_ranges(
        cls,
        start_ts: dt.datetime,
        end_ts: dt.datetime,
    ) -> list[tuple[dt.datetime, dt.datetime]]:
        return cls.BUILD_FOLD_RANGES(start_ts, end_ts)

    def _strict_fold_checks(
        self,
        strategy: BaseStrategy,
        btc_df: pl.DataFrame,
        start_ts: dt.datetime,
        end_ts: dt.datetime,
        config: ValidationConfig,
    ) -> tuple[bool, list[str]]:
        messages: list[str] = []
        folds = build_purged_walk_forward_folds(
            start_ts,
            end_ts,
            n_folds=4,
            min_train_days=ALLOCATION_SPAN_DAYS,
            test_days=ALLOCATION_SPAN_DAYS,
            embargo_days=ALLOCATION_SPAN_DAYS,
        )
        if len(folds) < 2:
            messages.append(
                "Strict fold checks skipped: insufficient date range for >=2 folds "
                "(not enough valid fold results)."
            )
            return True, messages

        fold_win_rates: list[float] = []
        for idx, (_, _, fold_start, fold_end) in enumerate(folds, start=1):
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
        btc_df: pl.DataFrame,
        start_ts: dt.datetime,
        end_ts: dt.datetime,
        config: ValidationConfig,
    ) -> tuple[bool, list[str]]:
        messages: list[str] = []
        if "price_usd" not in btc_df.columns:
            messages.append("Strict shuffled check skipped: missing price_usd column.")
            return True, messages
        if config.shuffled_trials <= 0:
            messages.append("Strict shuffled check skipped: shuffled_trials <= 0.")
            return True, messages

        shuffled_win_rates: list[float] = []
        for seed in self.ITER_RANGE(int(config.shuffled_trials)):
            window_df = btc_df.filter(
                (pl.col("date") >= start_ts) & (pl.col("date") <= end_ts)
            )
            window_values = window_df["price_usd"].cast(pl.Float64, strict=False).to_numpy()
            window_values = np.asarray(window_values, dtype=float)
            if window_values.size == 0:
                messages.append("Strict shuffled check skipped: empty validation window.")
                return True, messages
            rng = np.random.default_rng(seed)
            block_size = max(1, min(int(config.block_size), window_values.size))
            blocks = [
                window_values[idx : idx + block_size].copy()
                for idx in range(0, window_values.size, block_size)
            ]
            rng.shuffle(blocks)
            shuffled_prices = np.concatenate(blocks)[: window_values.size]
            outside = btc_df.filter((pl.col("date") < start_ts) | (pl.col("date") > end_ts))
            window_df = btc_df.filter(
                (pl.col("date") >= start_ts) & (pl.col("date") <= end_ts)
            ).with_columns(pl.Series("price_usd", shuffled_prices))
            shuffled_df = pl.concat([outside, window_df]).sort("date")
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
    def _mutation_message_for_strategy(strategy: BaseStrategy) -> str:
        return (
            STRICT_PROFILE_MUTATION_MESSAGE
            if strategy.intent_mode() == "profile"
            else STRICT_MUTATION_MESSAGE
        )

    @staticmethod
    def _mark_mutation_failure(
        state: _ValidationState,
        *,
        message: str = STRICT_MUTATION_MESSAGE,
    ) -> None:
        state.mutation_safe = False
        state.strict_checks_ok = False
        state.messages.append(message)

    @staticmethod
    def _stop_on_mutation(
        *,
        strict_mode: bool,
        mutated: bool,
        state: _ValidationState,
        message: str = STRICT_MUTATION_MESSAGE,
    ) -> bool:
        if strict_mode and mutated:
            StrategyRunnerValidationMixin._mark_mutation_failure(state, message=message)
            return True
        return False

    def _compute_with_mutation_guard(
        self,
        strategy: BaseStrategy,
        ctx: StrategyContext,
        *,
        strict_mode: bool,
    ) -> tuple[pl.DataFrame, bool]:
        if not strict_mode:
            return strategy.compute_weights(ctx), False
        # Use ctx.features_df so in-place mutation is detected against the
        # canonical Polars feature frame.
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
    ) -> tuple[pl.DataFrame, bool]:
        # Use ctx.features_df so mutation during profile build is detected.
        before = self._frame_signature(ctx.features_df) if strict_mode else ()
        if (
            strategy.intent_mode() == "profile"
            and strategy.__class__.build_target_profile_lazy
            is not BaseStrategy.build_target_profile_lazy
        ):
            _, profile = strategy._resolve_lazy_target_profile(
                StrategyLazyContext.from_strategy_context(ctx),
                ctx.features_df.lazy(),
            )
        else:
            profile_features = strategy.transform_features(ctx)
            profile_signals = strategy.build_signals(ctx, profile_features)
            profile = strategy.build_target_profile(ctx, profile_features, profile_signals)
        after = self._frame_signature(ctx.features_df) if strict_mode else before
        return self._profile_values(profile), before != after

    @staticmethod
    def _prefix_until(weights: pl.DataFrame, probe: dt.datetime) -> pl.DataFrame:
        return weights.filter(pl.col("date") <= probe)

    @staticmethod
    def _aligned_prefix(weights: pl.DataFrame, prefix_dates: pl.DataFrame) -> pl.DataFrame:
        return prefix_dates.join(weights, on="date", how="left")

    def _strict_determinism_check(
        self,
        *,
        strategy: BaseStrategy,
        full_features_df: pl.DataFrame,
        feature_index: dict[dt.datetime, int],
        window_start: dt.datetime,
        probe: dt.datetime,
        full_weights: pl.DataFrame,
        state: _ValidationState,
    ) -> bool:
        window_feat = slice_window_or_filter(
            full_features_df,
            feature_index,
            window_start,
            probe,
            expected_days=None,
        )
        repeat_ctx = strategy_context_from_features_df(
            window_feat,
            window_start,
            probe,
            probe,
            required_columns=tuple(strategy.required_feature_columns()),
            as_of_date=probe,
        )
        repeat_weights, repeat_mutated = self._compute_with_mutation_guard(
            strategy,
            repeat_ctx,
            strict_mode=True,
        )
        if repeat_mutated:
            self._mark_mutation_failure(state)
            return False
        if not self._weights_match(full_weights, repeat_weights):
            state.deterministic_ok = False
            state.strict_checks_ok = False
            state.messages.append(
                "Strict check failed: strategy is non-deterministic for identical inputs."
            )
            return False
        return True

    def _strategy_ctx_from_source(
        self,
        *,
        strategy: BaseStrategy,
        source_df: pl.DataFrame,
        window_start: dt.datetime,
        probe: dt.datetime,
        cache_namespace: str | None = None,
    ) -> StrategyContext:
        return strategy_context_from_features_df(
            self._materialize_strategy_features(
                strategy,
                source_df,
                start_date=window_start,
                end_date=probe,
                current_date=probe,
                cache_namespace=cache_namespace,
            ),
            window_start,
            probe,
            probe,
            required_columns=tuple(strategy.required_feature_columns()),
            as_of_date=probe,
        )

    @staticmethod
    def _align_profile_horizon(
        horizon_dates: pl.DataFrame,
        features_df: pl.DataFrame,
    ) -> pl.DataFrame:
        return horizon_dates.join(features_df, on="date", how="left")

    def _check_prefix_invariance(
        self,
        *,
        state: _ValidationState,
        probe: dt.datetime,
        full_prefix: pl.DataFrame,
        candidate_prefix: pl.DataFrame,
        check_label: str,
    ) -> bool:
        if self._weights_match(full_prefix, candidate_prefix):
            return True
        state.forward_leakage_ok = False
        state.messages.append(
            "Forward leakage detected near "
            f"{probe.strftime('%Y-%m-%d')}: {check_label}."
        )
        return False

    def _run_forward_leakage_checks(
        self,
        *,
        strategy: BaseStrategy,
        btc_df: pl.DataFrame,
        full_features_df: pl.DataFrame,
        backtest_idx: pl.Series,
        start_ts: dt.datetime,
        strict_mode: bool,
        has_propose_hook: bool,
        has_profile_hook: bool,
        probe_step: int,
        state: _ValidationState,
    ) -> None:
        started_at = time.perf_counter()
        window_offset = self.WINDOW_OFFSET
        probes = backtest_idx.to_list()[::probe_step]
        try:
            full_features_df, feature_index = build_window_index(full_features_df)
            for probe in probes:
                probe_ts = _to_naive_utc(probe)
                window_start = max(start_ts, probe_ts - window_offset)
                if window_start > probe_ts:
                    continue

                window_feat = slice_window_or_filter(
                    full_features_df,
                    feature_index,
                    window_start,
                    probe_ts,
                    expected_days=None,
                )
                full_weights, full_mutated = self._compute_strategy_weights(
                    strategy,
                    features_df=window_feat,
                    btc_df=btc_df,
                    start_date=window_start,
                    end_date=probe_ts,
                    current_date=probe_ts,
                    strict_mode=strict_mode,
                    cache_namespace="base",
                )
                if self._stop_on_mutation(
                    strict_mode=strict_mode,
                    mutated=full_mutated,
                    state=state,
                    message=self._mutation_message_for_strategy(strategy),
                ):
                    break

                if strict_mode and not self._strict_determinism_check(
                    strategy=strategy,
                    full_features_df=full_features_df,
                    feature_index=feature_index,
                    window_start=window_start,
                    probe=probe_ts,
                    full_weights=full_weights,
                    state=state,
                ):
                    break

                masked_source = btc_df.filter(pl.col("date") <= probe_ts)
                masked_ctx = self._strategy_ctx_from_source(
                    strategy=strategy,
                    source_df=masked_source,
                    window_start=window_start,
                    probe=probe_ts,
                    cache_namespace=f"masked:{self._cache_dt(probe_ts)}",
                )
                masked_weights, masked_mutated = self._compute_strategy_weights(
                    strategy,
                    features_df=masked_ctx.features.data,
                    btc_df=masked_source,
                    start_date=window_start,
                    end_date=probe_ts,
                    current_date=probe_ts,
                    strict_mode=strict_mode,
                    cache_namespace=f"masked:{self._cache_dt(probe_ts)}",
                )
                if self._stop_on_mutation(
                    strict_mode=strict_mode,
                    mutated=masked_mutated,
                    state=state,
                    message=self._mutation_message_for_strategy(strategy),
                ):
                    break

                perturbed_source = self._perturb_future_source_data(btc_df, probe_ts)
                perturbed_ctx = self._strategy_ctx_from_source(
                    strategy=strategy,
                    source_df=perturbed_source,
                    window_start=window_start,
                    probe=probe_ts,
                    cache_namespace=f"perturbed:{self._cache_dt(probe_ts)}",
                )
                perturbed_weights, perturbed_mutated = self._compute_strategy_weights(
                    strategy,
                    features_df=perturbed_ctx.features.data,
                    btc_df=perturbed_source,
                    start_date=window_start,
                    end_date=probe_ts,
                    current_date=probe_ts,
                    strict_mode=strict_mode,
                    cache_namespace=f"perturbed:{self._cache_dt(probe_ts)}",
                )
                if self._stop_on_mutation(
                    strict_mode=strict_mode,
                    mutated=perturbed_mutated,
                    state=state,
                    message=self._mutation_message_for_strategy(strategy),
                ):
                    break

                prefix_dates = self._prefix_until(full_weights, probe_ts).select("date")
                if prefix_dates.is_empty():
                    continue
                full_prefix = prefix_dates.join(full_weights, on="date", how="left")
                masked_prefix = self._aligned_prefix(masked_weights, prefix_dates)
                perturbed_prefix = self._aligned_prefix(perturbed_weights, prefix_dates)

                if not self._check_prefix_invariance(
                    state=state,
                    probe=probe_ts,
                    full_prefix=full_prefix,
                    candidate_prefix=masked_prefix,
                    check_label="masked-future weights diverge",
                ):
                    break
                if not self._check_prefix_invariance(
                    state=state,
                    probe=probe_ts,
                    full_prefix=full_prefix,
                    candidate_prefix=perturbed_prefix,
                    check_label="perturbed-future weights diverge",
                ):
                    break

                if has_profile_hook and not has_propose_hook:
                    horizon_end = full_features_df["date"].max()
                    horizon_dates = full_features_df.select("date")
                    profile_full_ctx = strategy_context_from_features_df(
                        full_features_df,
                        start_ts,
                        probe_ts,
                        probe_ts,
                        required_columns=tuple(strategy.required_feature_columns()),
                        # Keep future rows visible in ctx.features so profile
                        # leakage checks can distinguish date-spine access from
                        # illegal access to future feature values.
                        as_of_date=horizon_end,
                    )
                    masked_profile_features = self._strategy_ctx_from_source(
                        strategy=strategy,
                        source_df=masked_source,
                        window_start=start_ts,
                        probe=probe_ts,
                        cache_namespace=f"profile-masked:{self._cache_dt(probe_ts)}",
                    ).features.data
                    profile_masked_ctx = strategy_context_from_features_df(
                        self._align_profile_horizon(horizon_dates, masked_profile_features),
                        start_ts,
                        probe_ts,
                        probe_ts,
                        required_columns=tuple(strategy.required_feature_columns()),
                        as_of_date=horizon_end,
                    )
                    perturbed_profile_features = self._strategy_ctx_from_source(
                        strategy=strategy,
                        source_df=perturbed_source,
                        window_start=start_ts,
                        probe=probe_ts,
                        cache_namespace=f"profile-perturbed:{self._cache_dt(probe_ts)}",
                    ).features.data
                    profile_perturbed_ctx = strategy_context_from_features_df(
                        self._align_profile_horizon(horizon_dates, perturbed_profile_features),
                        start_ts,
                        probe_ts,
                        probe_ts,
                        required_columns=tuple(strategy.required_feature_columns()),
                        as_of_date=horizon_end,
                    )
                    (
                        full_profile_series,
                        full_profile_mutated,
                    ) = self._build_profile_with_mutation_guard(
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
                    if self._stop_on_mutation(
                        strict_mode=strict_mode,
                        mutated=(
                            full_profile_mutated
                            or masked_profile_mutated
                            or perturbed_profile_mutated
                        ),
                        state=state,
                        message=STRICT_PROFILE_MUTATION_MESSAGE,
                    ):
                        break

                    full_profile_prefix = self._aligned_prefix(
                        full_profile_series,
                        prefix_dates,
                    )
                    masked_profile_prefix = self._aligned_prefix(
                        masked_profile_series,
                        prefix_dates,
                    )
                    perturbed_profile_prefix = self._aligned_prefix(
                        perturbed_profile_series,
                        prefix_dates,
                    )
                    if not self._check_prefix_invariance(
                        state=state,
                        probe=probe_ts,
                        full_prefix=full_profile_prefix,
                        candidate_prefix=masked_profile_prefix,
                        check_label="profile values diverge (masked-future)",
                    ):
                        break
                    if not self._check_prefix_invariance(
                        state=state,
                        probe=probe_ts,
                        full_prefix=full_profile_prefix,
                        candidate_prefix=perturbed_profile_prefix,
                        check_label="profile values diverge (perturbed-future)",
                    ):
                        break
        finally:
            profile = self._runtime_profile()
            if profile is not None:
                profile.forward_leakage_seconds += time.perf_counter() - started_at

    def _run_weight_constraint_checks(
        self,
        *,
        strategy: BaseStrategy,
        btc_df: pl.DataFrame,
        features_df: pl.DataFrame,
        start_ts: dt.datetime,
        end_ts: dt.datetime,
        strict_mode: bool,
        config: ValidationConfig,
        state: _ValidationState,
    ) -> dt.datetime:
        started_at = time.perf_counter()
        window_offset = self.WINDOW_OFFSET
        max_window_start = end_ts - window_offset
        boundary_hits = 0
        boundary_total = 0
        try:
            if start_ts <= max_window_start:
                features_df, feature_index = build_window_index(features_df)
                window_starts = pl.datetime_range(
                    start_ts, max_window_start, interval="1d", eager=True
                ).to_list()
                step = 1 if (not strict_mode or len(window_starts) <= 200) else max(
                    len(window_starts) // 200,
                    1,
                )
                for window_start in window_starts[::step]:
                    window_end = window_start + window_offset
                    window_features = slice_window_or_filter(
                        features_df,
                        feature_index,
                        window_start,
                        window_end,
                        expected_days=None,
                    )
                    weights, mutated = self._compute_strategy_weights(
                        strategy,
                        features_df=window_features,
                        btc_df=btc_df,
                        start_date=window_start,
                        end_date=window_end,
                        current_date=window_end,
                        strict_mode=strict_mode,
                        cache_namespace="base",
                    )
                    if self._stop_on_mutation(
                        strict_mode=strict_mode,
                        mutated=mutated,
                        state=state,
                        message=self._mutation_message_for_strategy(strategy),
                    ):
                        break
                    if weights.is_empty():
                        continue
                    try:
                        self._validate_weights(weights, window_start, window_end)
                    except WeightValidationError as exc:
                        state.weight_constraints_ok = False
                        state.messages.append(str(exc))
                        break
                    if strict_mode and weights.height == ALLOCATION_SPAN_DAYS:
                        arr = weights["weight"].to_numpy().astype(float)
                        at_bounds = np.isclose(
                            arr,
                            MIN_DAILY_WEIGHT,
                            atol=1e-12,
                        ) | np.isclose(
                            arr,
                            MAX_DAILY_WEIGHT,
                            atol=1e-12,
                        )
                        boundary_hits += int(at_bounds.sum())
                        boundary_total += int(len(arr))
        finally:
            profile = self._runtime_profile()
            if profile is not None:
                profile.weight_constraint_seconds += time.perf_counter() - started_at

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
        btc_df: pl.DataFrame | None = None,
        features_df: pl.DataFrame,
        start_ts: dt.datetime,
        max_window_start: dt.datetime,
        strict_mode: bool,
        state: _ValidationState,
    ) -> None:
        if not (strict_mode and state.strict_checks_ok and start_ts <= max_window_start):
            return

        window_offset = self.WINDOW_OFFSET
        lock_start = start_ts
        lock_end = lock_start + window_offset
        lock_mid_offset = max(ALLOCATION_SPAN_DAYS // 2 - 1, 0)
        lock_current = min(
            lock_start + dt.timedelta(days=lock_mid_offset), lock_end
        )
        base_lock_ctx = strategy_context_from_features_df(
            self._materialize_strategy_features(
                strategy,
                btc_df if btc_df is not None else features_df,
                start_date=lock_start,
                end_date=lock_end,
                current_date=lock_current,
            ),
            lock_start,
            lock_end,
            lock_current,
            required_columns=tuple(strategy.required_feature_columns()),
            as_of_date=lock_current,
        )
        base_lock_weights, base_mutated = self._compute_with_mutation_guard(
            strategy,
            base_lock_ctx,
            strict_mode=True,
        )
        if self._stop_on_mutation(
            strict_mode=True,
            mutated=base_mutated,
            state=state,
            message=self._mutation_message_for_strategy(strategy),
        ):
            return
        if base_lock_weights.is_empty():
            return

        prefix_df = base_lock_weights.filter(pl.col("date") <= lock_current).sort("date")
        n_past = prefix_df.height
        locked_prefix = prefix_df["weight"].to_numpy().astype(float)
        perturbed_source = (
            self._perturb_future_source_data(btc_df, lock_current)
            if btc_df is not None
            else features_df
        )
        locked_ctx = strategy_context_from_features_df(
            self._materialize_strategy_features(
                strategy,
                perturbed_source,
                start_date=lock_start,
                end_date=lock_end,
                current_date=lock_current,
            ),
            lock_start,
            lock_end,
            lock_current,
            required_columns=tuple(strategy.required_feature_columns()),
            as_of_date=lock_current,
            locked_weights=locked_prefix,
        )
        locked_run_weights, locked_mutated = self._compute_with_mutation_guard(
            strategy,
            locked_ctx,
            strict_mode=True,
        )
        if self._stop_on_mutation(
            strict_mode=True,
            mutated=locked_mutated,
            state=state,
            message=self._mutation_message_for_strategy(strategy),
        ):
            return
        if n_past > 0:
            observed_prefix = (
                locked_run_weights.filter(pl.col("date") <= lock_current)
                .sort("date")["weight"]
                .to_numpy()
                .astype(float)
            )
            if not np.allclose(observed_prefix, locked_prefix, atol=1e-12, rtol=0.0):
                state.strict_checks_ok = False
                state.messages.append(
                    "Strict check failed: locked prefix was not preserved exactly."
                )

    def _strict_statistical_checks(
        self,
        *,
        strategy: BaseStrategy,
        btc_df: pl.DataFrame,
        features_df: pl.DataFrame,
        start_ts: dt.datetime,
        end_ts: dt.datetime,
        config: ValidationConfig,
        state: _ValidationState,
        backtest_result,
    ) -> None:
        diagnostics = state.diagnostics if state.diagnostics is not None else {}
        if not hasattr(backtest_result, "spd_table"):
            state.messages.append(
                "Strict statistical checks skipped: backtest result lacks window diagnostics."
            )
            state.diagnostics = diagnostics
            return

        anchored_excess = anchored_window_excess(
            backtest_result.spd_table,
            step=ALLOCATION_SPAN_DAYS,
        )
        bootstrap = block_bootstrap_confidence_interval(
            anchored_excess,
            block_size=config.block_size,
            trials=config.bootstrap_trials,
            seed=17,
        )
        permutation_pvalue = paired_block_permutation_pvalue(
            backtest_result.spd_table["dynamic_percentile"]
            if "dynamic_percentile" in backtest_result.spd_table.columns
            else pl.Series("d", []),
            backtest_result.spd_table["uniform_percentile"]
            if "uniform_percentile" in backtest_result.spd_table.columns
            else pl.Series("u", []),
            block_size=config.block_size,
            trials=config.permutation_trials,
            seed=23,
        )
        diagnostics["bootstrap_ci"] = {
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
        }
        diagnostics["permutation_pvalue"] = permutation_pvalue
        state.messages.append(
            "Strict statistical diagnostics: "
            f"bootstrap_ci=({bootstrap.lower:.2f}, {bootstrap.upper:.2f}), "
            f"permutation_pvalue={permutation_pvalue:.4f}."
        )
        if bootstrap.lower < float(config.min_bootstrap_ci_lower_excess):
            state.strict_checks_ok = False
            state.messages.append(
                "Strict check failed: bootstrap lower CI "
                f"{bootstrap.lower:.2f} < {config.min_bootstrap_ci_lower_excess:.2f}."
            )
        if permutation_pvalue > float(config.max_permutation_pvalue):
            state.strict_checks_ok = False
            state.messages.append(
                "Strict check failed: permutation p-value "
                f"{permutation_pvalue:.4f} > {config.max_permutation_pvalue:.4f}."
            )

        folds = build_purged_walk_forward_folds(
            start_ts,
            end_ts,
            n_folds=4,
            min_train_days=ALLOCATION_SPAN_DAYS,
            test_days=ALLOCATION_SPAN_DAYS,
            embargo_days=ALLOCATION_SPAN_DAYS,
        )
        required_columns = [
            column
            for column in strategy.required_feature_columns()
            if column in features_df.columns
        ]
        max_psi = 0.0
        max_ks = 0.0
        for _, _, test_start, test_end in folds:
            prior_end = test_start - dt.timedelta(days=1)
            prior_start = max(
                start_ts, prior_end - dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)
            )
            if prior_end < prior_start:
                continue
            baseline = self._materialize_strategy_features(
                strategy,
                btc_df,
                start_date=prior_start,
                end_date=prior_end,
                current_date=prior_end,
            )
            candidate = self._materialize_strategy_features(
                strategy,
                btc_df,
                start_date=test_start,
                end_date=test_end,
                current_date=test_end,
            )
            for column in required_columns:
                if column not in baseline.columns or column not in candidate.columns:
                    continue
                psi = population_stability_index(baseline[column], candidate[column])
                ks = ks_statistic(baseline[column], candidate[column])
                max_psi = max(max_psi, psi)
                max_ks = max(max_ks, ks)
        diagnostics["feature_drift"] = {"max_psi": max_psi, "max_ks": max_ks}
        state.messages.append(
            "Strict feature-drift diagnostics: "
            f"max_psi={max_psi:.4f}, max_ks={max_ks:.4f}."
        )
        if max_psi > float(config.max_feature_psi):
            state.strict_checks_ok = False
            state.messages.append(
                "Strict check failed: feature PSI "
                f"{max_psi:.4f} > {config.max_feature_psi:.4f}."
            )
        if max_ks > float(config.max_feature_ks):
            state.strict_checks_ok = False
            state.messages.append(
                "Strict check failed: feature KS "
                f"{max_ks:.4f} > {config.max_feature_ks:.4f}."
            )
        state.diagnostics = diagnostics

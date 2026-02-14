"""Strategy execution orchestration services."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .data_btc import BTCDataProvider
from .framework_contract import (
    ALLOCATION_SPAN_DAYS,
    MAX_DAILY_WEIGHT,
    MIN_DAILY_WEIGHT,
)
from .model_development import precompute_features
from .prelude import BACKTEST_START, WINDOW_OFFSET, backtest_dynamic_dca
from .strategy_types import (
    BacktestConfig,
    BaseStrategy,
    ExportConfig,
    StrategyArtifactSet,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)


class WeightValidationError(ValueError):
    """Raised when strategy weights violate required constraints."""


class StrategyRunner:
    """Single orchestration path for strategy lifecycle operations."""

    def __init__(self, data_provider=None):
        self._data_provider = data_provider or BTCDataProvider()

    def _load_btc_df(self, btc_df: pd.DataFrame | None) -> pd.DataFrame:
        if btc_df is not None:
            return btc_df
        return self._data_provider.load(backtest_start=BACKTEST_START)

    def _validate_strategy_contract(self, strategy: BaseStrategy) -> None:
        if strategy.__class__.compute_weights is not BaseStrategy.compute_weights:
            raise TypeError(
                "Custom compute_weights overrides are not supported. "
                "Use transform_features/build_signals with propose_weight "
                "or build_target_profile hooks."
            )
        strategy_cls = strategy.__class__
        has_propose_hook = strategy_cls.propose_weight is not BaseStrategy.propose_weight
        has_profile_hook = (
            strategy_cls.build_target_profile is not BaseStrategy.build_target_profile
        )
        if not (has_propose_hook or has_profile_hook):
            raise TypeError(
                "Strategy must implement propose_weight(state) or "
                "build_target_profile(ctx, features_df, signals)."
            )

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

    def _provenance(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig | ValidationConfig | ExportConfig,
    ) -> dict[str, str]:
        config_hash = hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        return {
            "strategy_id": strategy.strategy_id,
            "version": strategy.version,
            "config_hash": config_hash,
            "run_id": str(uuid4()),
        }

    @staticmethod
    def _weights_match(lhs: pd.Series, rhs: pd.Series, *, atol: float = 1e-12) -> bool:
        if not lhs.index.equals(rhs.index):
            return False
        left = pd.to_numeric(lhs, errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(rhs, errors="coerce").to_numpy(dtype=float)
        return bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right))) and bool(
            np.allclose(left, right, rtol=0.0, atol=atol)
        )

    @staticmethod
    def _profile_values(profile: TargetProfile | pd.Series) -> pd.Series:
        values = profile.values if isinstance(profile, TargetProfile) else profile
        return pd.to_numeric(values, errors="coerce")

    @staticmethod
    def _frame_signature(df: pd.DataFrame) -> tuple:
        try:
            row_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
        except TypeError:
            # Fallback for non-hashable object cells in strategy-added columns.
            row_hash = hash(df.to_json(date_format="iso", orient="split", default_handler=str))
        return (
            row_hash,
            tuple(str(col) for col in df.columns),
            tuple(str(dtype) for dtype in df.dtypes),
            tuple(df.shape),
        )

    @staticmethod
    def _perturb_future_features(features_df: pd.DataFrame, probe: pd.Timestamp) -> pd.DataFrame:
        perturbed = features_df.copy(deep=True)
        future_mask = perturbed.index > probe
        if not bool(future_mask.any()):
            return perturbed

        future = perturbed.loc[future_mask].copy()
        numeric_cols = list(future.select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            numeric = future[numeric_cols].to_numpy(dtype=float)
            ramp = np.linspace(1.0, 2.0, numeric.shape[0], dtype=float).reshape(-1, 1)
            shifted = np.where(np.isfinite(numeric), (-3.0 * numeric) + ramp, 0.0)
            future.loc[:, numeric_cols] = shifted

        non_numeric_cols = [col for col in future.columns if col not in numeric_cols]
        if len(non_numeric_cols) > 0 and len(future.index) > 1:
            future.loc[:, non_numeric_cols] = future[non_numeric_cols].iloc[::-1].to_numpy()

        perturbed.loc[future_mask, :] = future
        return perturbed

    @staticmethod
    def _build_fold_ranges(
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        all_days = pd.date_range(start=start_ts, end=end_ts, freq="D")
        if len(all_days) < (ALLOCATION_SPAN_DAYS * 2):
            return []
        max_folds = min(4, len(all_days) // ALLOCATION_SPAN_DAYS)
        if max_folds < 2:
            return []
        boundaries = np.linspace(0, len(all_days), num=max_folds + 1, dtype=int)
        folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for i in range(max_folds):
            left = int(boundaries[i])
            right = int(boundaries[i + 1]) - 1
            if right <= left:
                continue
            fold_start = all_days[left]
            fold_end = all_days[right]
            if len(pd.date_range(fold_start, fold_end, freq="D")) >= ALLOCATION_SPAN_DAYS:
                folds.append((fold_start, fold_end))
        return folds

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
        for seed in range(int(config.shuffled_trials)):
            shuffled_df = btc_df.copy(deep=True)
            window_values = shuffled_df.loc[start_ts:end_ts, "PriceUSD_coinmetrics"].to_numpy(
                dtype=float
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

    def backtest(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        *,
        btc_df: pd.DataFrame | None = None,
    ):
        from .api import BacktestResult

        self._validate_strategy_contract(strategy)
        btc_df = self._load_btc_df(btc_df)
        features_df = precompute_features(btc_df)

        def _strategy_fn(df_window: pd.DataFrame) -> pd.Series:
            if df_window.empty:
                return pd.Series(dtype=float)

            window_start = df_window.index.min()
            window_end = df_window.index.max()
            ctx = StrategyContext(
                features_df=features_df,
                start_date=window_start,
                end_date=window_end,
                current_date=window_end,
            )
            weights = strategy.compute_weights(ctx)
            self._validate_weights(weights, window_start, window_end)
            strategy.validate_weights(weights, ctx)
            return weights

        strategy_label = config.strategy_label or strategy.strategy_id
        spd_table, exp_decay_percentile = backtest_dynamic_dca(
            btc_df,
            _strategy_fn,
            features_df=features_df,
            strategy_label=strategy_label,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        if spd_table.empty:
            raise ValueError(
                "No backtest windows were generated for the requested date range."
            )

        win_rate = (
            (spd_table["dynamic_percentile"] > spd_table["uniform_percentile"]).mean()
            * 100
        )
        score = (0.5 * win_rate) + (0.5 * exp_decay_percentile)
        provenance = self._provenance(strategy, config)

        return BacktestResult(
            spd_table=spd_table,
            exp_decay_percentile=exp_decay_percentile,
            win_rate=win_rate,
            score=score,
            strategy_id=provenance["strategy_id"],
            strategy_version=provenance["version"],
            config_hash=provenance["config_hash"],
            run_id=provenance["run_id"],
        )

    def validate(
        self,
        strategy: BaseStrategy,
        config: ValidationConfig,
        *,
        btc_df: pd.DataFrame | None = None,
    ):
        from .api import ValidationResult

        self._validate_strategy_contract(strategy)
        btc_df = self._load_btc_df(btc_df)

        start_date = config.start_date or BACKTEST_START
        end_date = config.end_date or btc_df.index.max().strftime("%Y-%m-%d")
        features_df = precompute_features(btc_df)
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        backtest_idx = btc_df.loc[start_ts:end_ts].index

        messages: list[str] = []
        forward_leakage_ok = True
        weight_constraints_ok = True
        strict_checks_ok = True
        strict_mode = bool(config.strict)
        mutation_safe = True
        deterministic_ok = True

        if len(backtest_idx) == 0:
            return ValidationResult(
                passed=False,
                forward_leakage_ok=False,
                weight_constraints_ok=False,
                win_rate=0.0,
                win_rate_ok=False,
                messages=["No data available in the requested date range."],
            )

        probe_step = 1 if strict_mode else max(len(backtest_idx) // 50, 1)
        strategy_cls = strategy.__class__
        has_propose_hook = strategy_cls.propose_weight is not BaseStrategy.propose_weight
        has_profile_hook = (
            strategy_cls.build_target_profile is not BaseStrategy.build_target_profile
        )

        def _compute_with_mutation_guard(ctx: StrategyContext) -> tuple[pd.Series, bool]:
            if not strict_mode:
                return strategy.compute_weights(ctx), False
            before = self._frame_signature(ctx.features_df)
            weights = strategy.compute_weights(ctx)
            after = self._frame_signature(ctx.features_df)
            return weights, before != after

        for probe in backtest_idx[::probe_step]:
            window_start = max(start_ts, probe - WINDOW_OFFSET)
            if window_start > probe:
                continue

            full_ctx = StrategyContext(
                features_df=features_df.copy(deep=True),
                start_date=window_start,
                end_date=probe,
                current_date=probe,
            )
            full_weights, full_mutated = _compute_with_mutation_guard(full_ctx)
            if strict_mode and full_mutated:
                mutation_safe = False
                strict_checks_ok = False
                messages.append(
                    "Strict check failed: strategy mutated ctx.features_df in-place."
                )
                break

            if strict_mode:
                repeat_ctx = StrategyContext(
                    features_df=features_df.copy(deep=True),
                    start_date=window_start,
                    end_date=probe,
                    current_date=probe,
                )
                repeat_weights, repeat_mutated = _compute_with_mutation_guard(repeat_ctx)
                if repeat_mutated:
                    mutation_safe = False
                    strict_checks_ok = False
                    messages.append(
                        "Strict check failed: strategy mutated ctx.features_df in-place."
                    )
                    break
                if not self._weights_match(full_weights, repeat_weights):
                    deterministic_ok = False
                    strict_checks_ok = False
                    messages.append(
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
            masked_weights, masked_mutated = _compute_with_mutation_guard(masked_ctx)
            if strict_mode and masked_mutated:
                mutation_safe = False
                strict_checks_ok = False
                messages.append(
                    "Strict check failed: strategy mutated ctx.features_df in-place."
                )
                break

            perturbed_ctx = StrategyContext(
                features_df=self._perturb_future_features(features_df, probe),
                start_date=window_start,
                end_date=probe,
                current_date=probe,
            )
            perturbed_weights, perturbed_mutated = _compute_with_mutation_guard(perturbed_ctx)
            if strict_mode and perturbed_mutated:
                mutation_safe = False
                strict_checks_ok = False
                messages.append(
                    "Strict check failed: strategy mutated ctx.features_df in-place."
                )
                break

            prefix_idx = full_weights.index[full_weights.index <= probe]
            if len(prefix_idx) == 0:
                continue
            full_prefix = full_weights.loc[prefix_idx]
            masked_prefix = masked_weights.reindex(prefix_idx)
            perturbed_prefix = perturbed_weights.reindex(prefix_idx)

            if not self._weights_match(full_prefix, masked_prefix):
                forward_leakage_ok = False
                messages.append(
                    "Forward leakage detected near "
                    f"{probe.strftime('%Y-%m-%d')}: masked-future weights diverge."
                )
                break
            if not self._weights_match(full_prefix, perturbed_prefix):
                forward_leakage_ok = False
                messages.append(
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

                def _build_profile(ctx: StrategyContext) -> tuple[pd.Series, bool]:
                    before = self._frame_signature(ctx.features_df)
                    profile_features = strategy.transform_features(ctx)
                    profile_signals = strategy.build_signals(ctx, profile_features)
                    profile = strategy.build_target_profile(
                        ctx, profile_features, profile_signals
                    )
                    after = self._frame_signature(ctx.features_df)
                    return self._profile_values(profile), before != after

                full_profile_series, full_profile_mutated = _build_profile(profile_full_ctx)
                masked_profile_series, masked_profile_mutated = _build_profile(profile_masked_ctx)
                perturbed_profile_series, perturbed_profile_mutated = _build_profile(
                    profile_perturbed_ctx
                )
                if strict_mode and (
                    full_profile_mutated or masked_profile_mutated or perturbed_profile_mutated
                ):
                    mutation_safe = False
                    strict_checks_ok = False
                    messages.append(
                        "Strict check failed: strategy mutated ctx.features_df during profile build."
                    )
                    break

                full_profile_prefix = full_profile_series.reindex(prefix_idx)
                masked_profile_prefix = masked_profile_series.reindex(prefix_idx)
                perturbed_profile_prefix = perturbed_profile_series.reindex(prefix_idx)
                if not self._weights_match(full_profile_prefix, masked_profile_prefix):
                    forward_leakage_ok = False
                    messages.append(
                        "Forward leakage detected near "
                        f"{probe.strftime('%Y-%m-%d')}: profile values diverge (masked-future)."
                    )
                    break
                if not self._weights_match(full_profile_prefix, perturbed_profile_prefix):
                    forward_leakage_ok = False
                    messages.append(
                        "Forward leakage detected near "
                        f"{probe.strftime('%Y-%m-%d')}: profile values diverge (perturbed-future)."
                    )
                    break

        max_window_start = end_ts - WINDOW_OFFSET
        boundary_hits = 0
        boundary_total = 0
        if start_ts <= max_window_start:
            window_starts = pd.date_range(start=start_ts, end=max_window_start, freq="D")
            for window_start in window_starts:
                window_end = window_start + WINDOW_OFFSET
                ctx = StrategyContext(
                    features_df=features_df.copy(deep=True) if strict_mode else features_df,
                    start_date=window_start,
                    end_date=window_end,
                    current_date=window_end,
                )
                weights, mutated = _compute_with_mutation_guard(ctx)
                if strict_mode and mutated:
                    mutation_safe = False
                    strict_checks_ok = False
                    messages.append(
                        "Strict check failed: strategy mutated ctx.features_df in-place."
                    )
                    break
                if weights.empty:
                    continue
                try:
                    self._validate_weights(weights, window_start, window_end)
                except WeightValidationError as exc:
                    weight_constraints_ok = False
                    messages.append(str(exc))
                    break
                if strict_mode and len(weights) == ALLOCATION_SPAN_DAYS:
                    arr = weights.to_numpy(dtype=float)
                    at_bounds = np.isclose(arr, MIN_DAILY_WEIGHT, atol=1e-12) | np.isclose(
                        arr, MAX_DAILY_WEIGHT, atol=1e-12
                    )
                    boundary_hits += int(at_bounds.sum())
                    boundary_total += int(len(arr))

        if strict_mode and boundary_total > 0:
            boundary_hit_rate = boundary_hits / boundary_total
            messages.append(
                "Strict boundary diagnostics: "
                f"{boundary_hit_rate * 100:.2f}% of days hit MIN/MAX bounds."
            )
            if boundary_hit_rate > float(config.max_boundary_hit_rate):
                strict_checks_ok = False
                messages.append(
                    "Strict check failed: boundary hit rate "
                    f"{boundary_hit_rate * 100:.2f}% exceeds "
                    f"{config.max_boundary_hit_rate * 100:.2f}%."
                )

        if strict_mode and strict_checks_ok and start_ts <= max_window_start:
            lock_start = start_ts
            lock_end = lock_start + WINDOW_OFFSET
            lock_mid_offset = max(ALLOCATION_SPAN_DAYS // 2 - 1, 0)
            lock_current = min(lock_start + pd.Timedelta(days=lock_mid_offset), lock_end)
            base_lock_ctx = StrategyContext(
                features_df=features_df.copy(deep=True),
                start_date=lock_start,
                end_date=lock_end,
                current_date=lock_current,
            )
            base_lock_weights, base_mutated = _compute_with_mutation_guard(base_lock_ctx)
            if base_mutated:
                mutation_safe = False
                strict_checks_ok = False
                messages.append(
                    "Strict check failed: strategy mutated ctx.features_df in-place."
                )
            elif not base_lock_weights.empty:
                n_past = int((base_lock_weights.index <= lock_current).sum())
                locked_prefix = base_lock_weights.iloc[:n_past].to_numpy(dtype=float)
                locked_ctx = StrategyContext(
                    features_df=self._perturb_future_features(features_df, lock_current),
                    start_date=lock_start,
                    end_date=lock_end,
                    current_date=lock_current,
                    locked_weights=locked_prefix,
                )
                locked_run_weights, locked_mutated = _compute_with_mutation_guard(locked_ctx)
                if locked_mutated:
                    mutation_safe = False
                    strict_checks_ok = False
                    messages.append(
                        "Strict check failed: strategy mutated ctx.features_df in-place."
                    )
                elif n_past > 0:
                    observed_prefix = locked_run_weights.iloc[:n_past].to_numpy(dtype=float)
                    if not np.allclose(observed_prefix, locked_prefix, atol=1e-12, rtol=0.0):
                        strict_checks_ok = False
                        messages.append(
                            "Strict check failed: locked prefix was not preserved exactly."
                        )

        backtest_result = self.backtest(
            strategy,
            BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                strategy_label="validation-run",
            ),
            btc_df=btc_df,
        )
        win_rate_ok = backtest_result.win_rate >= config.min_win_rate
        if not win_rate_ok:
            messages.append(
                f"Win rate below threshold: {backtest_result.win_rate:.2f}% < "
                f"{config.min_win_rate:.2f}%."
            )

        if strict_mode and strict_checks_ok:
            fold_ok, fold_messages = self._strict_fold_checks(
                strategy=strategy,
                btc_df=btc_df,
                start_ts=start_ts,
                end_ts=end_ts,
                config=config,
            )
            strict_checks_ok = strict_checks_ok and fold_ok
            messages.extend(fold_messages)

            shuffled_days = max(ALLOCATION_SPAN_DAYS * 2, 730)
            shuffled_start = max(start_ts, end_ts - pd.Timedelta(days=shuffled_days - 1))
            shuffled_ok, shuffled_messages = self._strict_shuffled_check(
                strategy=strategy,
                btc_df=btc_df,
                start_ts=shuffled_start,
                end_ts=end_ts,
                config=config,
            )
            strict_checks_ok = strict_checks_ok and shuffled_ok
            messages.extend(shuffled_messages)

        if strict_mode and not mutation_safe:
            strict_checks_ok = False
        if strict_mode and not deterministic_ok:
            strict_checks_ok = False

        if not messages:
            messages.append("All validation checks passed.")

        return ValidationResult(
            passed=forward_leakage_ok
            and weight_constraints_ok
            and win_rate_ok
            and (strict_checks_ok if strict_mode else True),
            forward_leakage_ok=forward_leakage_ok,
            weight_constraints_ok=weight_constraints_ok,
            win_rate=float(backtest_result.win_rate),
            win_rate_ok=win_rate_ok,
            messages=messages,
        )

    def export(
        self,
        strategy: BaseStrategy,
        config: ExportConfig,
        *,
        btc_df: pd.DataFrame | None = None,
        current_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        from .export_weights import process_start_date_batch
        from .prelude import generate_date_ranges, group_ranges_by_start_date

        self._validate_strategy_contract(strategy)
        btc_df = self._load_btc_df(btc_df)
        features_df = precompute_features(btc_df)
        run_date = current_date or pd.Timestamp.now().normalize()

        date_ranges = generate_date_ranges(config.range_start, config.range_end)
        grouped_ranges = group_ranges_by_start_date(date_ranges)
        all_results = []
        for start_date, end_dates in sorted(grouped_ranges.items()):
            all_results.append(
                process_start_date_batch(
                    start_date,
                    end_dates,
                    features_df,
                    btc_df,
                    run_date,
                    config.btc_price_col,
                    strategy=strategy,
                    enforce_span_contract=True,
                )
            )
        if not all_results:
            raise ValueError("No export ranges generated from provided export config.")
        result_df = pd.concat(all_results, ignore_index=True)

        provenance = self._provenance(strategy, config)
        output_root = (
            Path(config.output_dir)
            / strategy.strategy_id
            / strategy.version
            / provenance["run_id"]
        )
        output_root.mkdir(parents=True, exist_ok=True)
        result_path = output_root / "weights.csv"
        result_df.to_csv(result_path, index=False)
        metadata = StrategyArtifactSet(
            strategy_id=strategy.strategy_id,
            version=strategy.version,
            config_hash=provenance["config_hash"],
            run_id=provenance["run_id"],
            output_dir=str(output_root),
            files={"weights_csv": str(result_path)},
        )
        (output_root / "artifacts.json").write_text(
            json.dumps(asdict(metadata), indent=2),
            encoding="utf-8",
        )
        return result_df

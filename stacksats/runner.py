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

        if len(backtest_idx) == 0:
            return ValidationResult(
                passed=False,
                forward_leakage_ok=False,
                weight_constraints_ok=False,
                win_rate=0.0,
                win_rate_ok=False,
                messages=["No data available in the requested date range."],
            )

        probe_step = max(len(backtest_idx) // 50, 1)
        strategy_cls = strategy.__class__
        has_propose_hook = strategy_cls.propose_weight is not BaseStrategy.propose_weight
        has_profile_hook = (
            strategy_cls.build_target_profile is not BaseStrategy.build_target_profile
        )
        for probe in backtest_idx[::probe_step]:
            window_start = max(
                start_ts, probe - WINDOW_OFFSET
            )
            if window_start > probe:
                continue
            full_ctx = StrategyContext(
                features_df=features_df,
                start_date=window_start,
                end_date=probe,
                current_date=probe,
            )
            full_weights = strategy.compute_weights(full_ctx)

            masked_features = features_df.copy()
            masked_features.loc[masked_features.index > probe, :] = np.nan
            masked_ctx = StrategyContext(
                features_df=masked_features,
                start_date=window_start,
                end_date=probe,
                current_date=probe,
            )
            masked_weights = strategy.compute_weights(masked_ctx)

            if probe in full_weights.index and probe in masked_weights.index:
                if not np.isclose(
                    float(full_weights.loc[probe]),
                    float(masked_weights.loc[probe]),
                    rtol=1e-9,
                    atol=1e-12,
                ):
                    forward_leakage_ok = False
                    messages.append(
                        f"Forward leakage detected near {probe.strftime('%Y-%m-%d')}."
                    )
                    break

            if has_profile_hook and not has_propose_hook:
                full_features = strategy.transform_features(full_ctx)
                masked_features = strategy.transform_features(masked_ctx)
                full_signals = strategy.build_signals(full_ctx, full_features)
                masked_signals = strategy.build_signals(masked_ctx, masked_features)
                full_profile = strategy.build_target_profile(
                    full_ctx, full_features, full_signals
                )
                masked_profile = strategy.build_target_profile(
                    masked_ctx, masked_features, masked_signals
                )

                def _profile_series(
                    profile: TargetProfile | pd.Series,
                ) -> pd.Series:
                    values = profile.values if isinstance(profile, TargetProfile) else profile
                    return pd.to_numeric(values, errors="coerce")

                full_profile_series = _profile_series(full_profile)
                masked_profile_series = _profile_series(masked_profile)
                if probe in full_profile_series.index and probe in masked_profile_series.index:
                    full_probe = full_profile_series.loc[probe]
                    masked_probe = masked_profile_series.loc[probe]
                    if not (
                        np.isfinite(full_probe)
                        and np.isfinite(masked_probe)
                        and np.isclose(float(full_probe), float(masked_probe), atol=1e-12)
                    ):
                        forward_leakage_ok = False
                        messages.append(
                            f"Forward leakage detected near {probe.strftime('%Y-%m-%d')}."
                        )
                        break

        max_window_start = end_ts - WINDOW_OFFSET
        if start_ts <= max_window_start:
            window_starts = pd.date_range(start=start_ts, end=max_window_start, freq="D")
            for window_start in window_starts:
                window_end = window_start + WINDOW_OFFSET
                ctx = StrategyContext(
                    features_df=features_df,
                    start_date=window_start,
                    end_date=window_end,
                    current_date=window_end,
                )
                weights = strategy.compute_weights(ctx)
                if weights.empty:
                    continue
                try:
                    self._validate_weights(weights, window_start, window_end)
                except WeightValidationError as exc:
                    weight_constraints_ok = False
                    messages.append(str(exc))
                    break

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

        if not messages:
            messages.append("All validation checks passed.")

        return ValidationResult(
            passed=forward_leakage_ok and weight_constraints_ok and win_rate_ok,
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

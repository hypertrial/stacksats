"""Strategy execution orchestration services."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .data_btc import BTCDataProvider
from .framework_contract import ALLOCATION_SPAN_DAYS
from .model_development import precompute_features
from .prelude import BACKTEST_START, backtest_dynamic_dca
from .runner_validation import (
    STRICT_MUTATION_MESSAGE,
    STRICT_PROFILE_MUTATION_MESSAGE,
    WIN_RATE_TOLERANCE,
    StrategyRunnerValidationMixin,
    WeightValidationError,
    _ValidationState,
)
from .strategy_time_series import StrategyTimeSeriesBatch
from .strategy_types import (
    BacktestConfig,
    BaseStrategy,
    ExportConfig,
    StrategyArtifactSet,
    StrategyContext,
    ValidationConfig,
    validate_strategy_contract,
)


class StrategyRunner(StrategyRunnerValidationMixin):
    """Single orchestration path for strategy lifecycle operations."""

    def __init__(self, data_provider=None):
        self._data_provider = data_provider or BTCDataProvider()

    def _load_btc_df(
        self,
        btc_df: pd.DataFrame | None,
        *,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if btc_df is not None:
            return btc_df
        return self._data_provider.load(backtest_start=BACKTEST_START, end_date=end_date)

    def _validate_strategy_contract(self, strategy: BaseStrategy) -> None:
        validate_strategy_contract(strategy)

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
        btc_df = self._load_btc_df(btc_df, end_date=config.end_date)
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
        spd_table, exp_decay_percentile, uniform_exp_decay_percentile = backtest_dynamic_dca(
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

        win_rate = self._compute_win_rate(spd_table)
        score = (0.5 * win_rate) + (0.5 * exp_decay_percentile)
        provenance = self._provenance(strategy, config)

        return BacktestResult(
            spd_table=spd_table,
            exp_decay_percentile=exp_decay_percentile,
            uniform_exp_decay_percentile=uniform_exp_decay_percentile,
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
        btc_df = self._load_btc_df(btc_df, end_date=config.end_date)

        start_date = config.start_date or BACKTEST_START
        end_date = config.end_date or btc_df.index.max().strftime("%Y-%m-%d")
        features_df = precompute_features(btc_df)
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        backtest_idx = btc_df.loc[start_ts:end_ts].index

        strict_mode = bool(config.strict)
        state = _ValidationState(messages=[])

        if len(backtest_idx) == 0:
            return ValidationResult(
                passed=False,
                forward_leakage_ok=False,
                weight_constraints_ok=False,
                win_rate=0.0,
                win_rate_ok=False,
                messages=["No data available in the requested date range."],
                min_win_rate=float(config.min_win_rate),
            )

        probe_step = 1 if strict_mode else max(len(backtest_idx) // 50, 1)
        strategy_cls = strategy.__class__
        has_propose_hook = strategy_cls.propose_weight is not BaseStrategy.propose_weight
        has_profile_hook = (
            strategy_cls.build_target_profile is not BaseStrategy.build_target_profile
        )
        self._run_forward_leakage_checks(
            strategy=strategy,
            features_df=features_df,
            backtest_idx=backtest_idx,
            start_ts=start_ts,
            strict_mode=strict_mode,
            has_propose_hook=has_propose_hook,
            has_profile_hook=has_profile_hook,
            probe_step=probe_step,
            state=state,
        )
        max_window_start = self._run_weight_constraint_checks(
            strategy=strategy,
            features_df=features_df,
            start_ts=start_ts,
            end_ts=end_ts,
            strict_mode=strict_mode,
            config=config,
            state=state,
        )
        self._run_locked_prefix_check(
            strategy=strategy,
            features_df=features_df,
            start_ts=start_ts,
            max_window_start=max_window_start,
            strict_mode=strict_mode,
            state=state,
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
            state.messages.append(
                f"Win rate below threshold: {backtest_result.win_rate:.2f}% < "
                f"{config.min_win_rate:.2f}%."
            )

        if strict_mode and state.strict_checks_ok:
            fold_ok, fold_messages = self._strict_fold_checks(
                strategy=strategy,
                btc_df=btc_df,
                start_ts=start_ts,
                end_ts=end_ts,
                config=config,
            )
            state.strict_checks_ok = state.strict_checks_ok and fold_ok
            state.messages.extend(fold_messages)

            shuffled_days = max(ALLOCATION_SPAN_DAYS * 2, 730)
            shuffled_start = max(start_ts, end_ts - pd.Timedelta(days=shuffled_days - 1))
            shuffled_ok, shuffled_messages = self._strict_shuffled_check(
                strategy=strategy,
                btc_df=btc_df,
                start_ts=shuffled_start,
                end_ts=end_ts,
                config=config,
            )
            state.strict_checks_ok = state.strict_checks_ok and shuffled_ok
            state.messages.extend(shuffled_messages)

        if strict_mode and not state.mutation_safe:
            state.strict_checks_ok = False
        if strict_mode and not state.deterministic_ok:
            state.strict_checks_ok = False

        if not state.messages:
            state.messages.append("All validation checks passed.")

        return ValidationResult(
            passed=state.forward_leakage_ok
            and state.weight_constraints_ok
            and win_rate_ok
            and (state.strict_checks_ok if strict_mode else True),
            forward_leakage_ok=state.forward_leakage_ok,
            weight_constraints_ok=state.weight_constraints_ok,
            win_rate=float(backtest_result.win_rate),
            win_rate_ok=win_rate_ok,
            messages=state.messages,
            min_win_rate=float(config.min_win_rate),
        )

    def export(
        self,
        strategy: BaseStrategy,
        config: ExportConfig,
        *,
        btc_df: pd.DataFrame | None = None,
        current_date: pd.Timestamp | None = None,
    ) -> StrategyTimeSeriesBatch:
        from .export_weights import process_start_date_batch
        from .prelude import generate_date_ranges, group_ranges_by_start_date

        self._validate_strategy_contract(strategy)
        btc_df = self._load_btc_df(btc_df, end_date=config.range_end)
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
        series_batch = StrategyTimeSeriesBatch.from_flat_dataframe(
            result_df,
            strategy_id=provenance["strategy_id"],
            strategy_version=provenance["version"],
            run_id=provenance["run_id"],
            config_hash=provenance["config_hash"],
        )
        output_root = (
            Path(config.output_dir)
            / strategy.strategy_id
            / strategy.version
            / provenance["run_id"]
        )
        output_root.mkdir(parents=True, exist_ok=True)
        result_path = output_root / "weights.csv"
        export_df = series_batch.to_dataframe()
        export_df.to_csv(result_path, index=False)
        schema_path = output_root / "timeseries_schema.md"
        schema_path.write_text(series_batch.schema_markdown(), encoding="utf-8")
        metadata = StrategyArtifactSet(
            strategy_id=strategy.strategy_id,
            version=strategy.version,
            config_hash=provenance["config_hash"],
            run_id=provenance["run_id"],
            output_dir=str(output_root),
            files={
                "weights_csv": str(result_path),
                "timeseries_schema_md": str(schema_path),
            },
        )
        (output_root / "artifacts.json").write_text(
            json.dumps(asdict(metadata), indent=2),
            encoding="utf-8",
        )
        return series_batch


__all__ = [
    "STRICT_MUTATION_MESSAGE",
    "STRICT_PROFILE_MUTATION_MESSAGE",
    "WIN_RATE_TOLERANCE",
    "StrategyRunner",
    "WeightValidationError",
]

"""Strategy execution orchestration services."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .data_btc import BTCDataProvider
from .feature_materialization import hash_dataframe
from .feature_registry import DEFAULT_FEATURE_REGISTRY
from .framework_contract import ALLOCATION_SPAN_DAYS, ALLOCATION_WINDOW_OFFSET
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
    RunDailyConfig,
    StrategyArtifactSet,
    StrategyContext,
    ValidationConfig,
    validate_strategy_contract,
)
from .strategy_lint import lint_strategy_class, summarize_lint_findings


class StrategyRunner(StrategyRunnerValidationMixin):
    """Single orchestration path for strategy lifecycle operations."""

    def __init__(self, data_provider=None):
        self._data_provider = data_provider or BTCDataProvider()
        self._feature_registry = DEFAULT_FEATURE_REGISTRY

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

    def _strategy_lint_warnings(self, strategy: BaseStrategy) -> list[str]:
        _, warnings = summarize_lint_findings(lint_strategy_class(strategy.__class__))
        return warnings

    def _provenance(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig | ValidationConfig | ExportConfig,
    ) -> dict[str, str]:
        metadata = strategy.metadata()
        config_hash = hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        return {
            "strategy_id": metadata.strategy_id,
            "version": metadata.version,
            "config_hash": config_hash,
            "run_id": str(uuid4()),
        }

    def _daily_run_fingerprint(
        self,
        strategy: BaseStrategy,
        config: RunDailyConfig,
        run_date: str,
    ) -> str:
        spec = strategy.spec()
        payload = {
            "strategy_metadata": asdict(spec.metadata),
            "strategy_params": spec.params,
            "strategy_intent_mode": spec.intent_mode,
            "required_feature_columns": list(spec.required_feature_columns),
            "required_feature_sets": list(strategy.required_feature_sets()),
            "strategy_class": strategy.__class__.__name__,
            "strategy_module": strategy.__class__.__module__,
            "run_date": run_date,
            "mode": config.mode,
            "total_window_budget_usd": float(config.total_window_budget_usd),
            "btc_price_col": config.btc_price_col,
            "adapter_spec": config.adapter_spec,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _config_hash(config: ValidationConfig) -> str:
        return hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def backtest(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        *,
        btc_df: pd.DataFrame | None = None,
    ):
        from .api import BacktestResult

        self._validate_strategy_contract(strategy)
        metadata = strategy.metadata()
        btc_df = self._load_btc_df(btc_df, end_date=config.end_date)
        start_ts = pd.Timestamp(config.start_date or BACKTEST_START)
        end_ts = pd.Timestamp(config.end_date or btc_df.index.max()).normalize()
        full_features_df = self._materialize_strategy_features(
            strategy,
            btc_df,
            start_date=start_ts,
            end_date=end_ts,
            current_date=end_ts,
        )

        def _strategy_fn(df_window: pd.DataFrame) -> pd.Series:
            if df_window.empty:
                return pd.Series(dtype=float)

            window_start = df_window.index.min()
            window_end = df_window.index.max()
            window_features = self._materialize_strategy_features(
                strategy,
                btc_df,
                start_date=window_start,
                end_date=window_end,
                current_date=window_end,
            )
            ctx = StrategyContext(
                features_df=window_features,
                start_date=window_start,
                end_date=window_end,
                current_date=window_end,
            )
            weights = strategy.compute_weights(ctx)
            self._validate_weights(weights, window_start, window_end)
            strategy.validate_weights(weights, ctx)
            return weights

        strategy_label = config.strategy_label or metadata.strategy_id
        spd_table, exp_decay_percentile, uniform_exp_decay_percentile = backtest_dynamic_dca(
            btc_df,
            _strategy_fn,
            features_df=full_features_df,
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

        try:
            self._validate_strategy_contract(strategy)
        except (TypeError, ValueError) as exc:
            return ValidationResult(
                passed=False,
                forward_leakage_ok=False,
                weight_constraints_ok=False,
                win_rate=0.0,
                win_rate_ok=False,
                messages=[str(exc)],
                min_win_rate=float(config.min_win_rate),
                diagnostics={},
            )
        btc_df = self._load_btc_df(btc_df, end_date=config.end_date)

        start_date = config.start_date or BACKTEST_START
        end_date = config.end_date or btc_df.index.max().strftime("%Y-%m-%d")
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        backtest_idx = btc_df.loc[start_ts:end_ts].index

        strict_mode = bool(config.strict)
        state = _ValidationState(messages=[], diagnostics={})
        state.messages.extend(
            f"Strategy lint warning: {warning}" for warning in self._strategy_lint_warnings(strategy)
        )

        if len(backtest_idx) == 0:
            return ValidationResult(
                passed=False,
                forward_leakage_ok=False,
                weight_constraints_ok=False,
                win_rate=0.0,
                win_rate_ok=False,
                messages=["No data available in the requested date range."],
                min_win_rate=float(config.min_win_rate),
                diagnostics={},
            )

        features_df = self._materialize_strategy_features(
            strategy,
            btc_df,
            start_date=start_ts,
            end_date=end_ts,
            current_date=end_ts,
        )
        if strict_mode:
            probe_step = 1 if len(backtest_idx) <= (ALLOCATION_SPAN_DAYS * 2) else max(
                len(backtest_idx) // 100,
                1,
            )
        else:
            probe_step = max(len(backtest_idx) // 50, 1)
        active_intent_mode = strategy.intent_mode()
        has_propose_hook = active_intent_mode == "propose"
        has_profile_hook = active_intent_mode == "profile"
        self._run_forward_leakage_checks(
            strategy=strategy,
            btc_df=btc_df,
            full_features_df=features_df,
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
            btc_df=btc_df,
            features_df=features_df,
            start_ts=start_ts,
            end_ts=end_ts,
            strict_mode=strict_mode,
            config=config,
            state=state,
        )
        self._run_locked_prefix_check(
            strategy=strategy,
            btc_df=btc_df,
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
            self._strict_statistical_checks(
                strategy=strategy,
                btc_df=btc_df,
                features_df=features_df,
                start_ts=start_ts,
                end_ts=end_ts,
                config=config,
                state=state,
                backtest_result=backtest_result,
            )

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
            diagnostics=state.diagnostics or {},
        )

    def run_daily(
        self,
        strategy: BaseStrategy,
        config: RunDailyConfig,
        *,
        btc_df: pd.DataFrame | None = None,
    ):
        from .api import DailyOrderReceipt, DailyOrderRequest, DailyRunResult
        from .execution_adapters import PaperExecutionAdapter, load_execution_adapter
        from .execution_state import IdempotencyConflictError, SQLiteExecutionStateStore

        self._validate_strategy_contract(strategy)
        metadata = strategy.metadata()
        budget = float(config.total_window_budget_usd)
        if not pd.notna(budget) or budget <= 0.0:
            raise ValueError("total_window_budget_usd must be a finite value greater than 0.")
        if config.mode not in {"paper", "live"}:
            raise ValueError("mode must be either 'paper' or 'live'.")
        if config.mode == "live" and not config.adapter_spec:
            raise ValueError("Live mode requires --adapter.")

        run_ts = (
            pd.Timestamp(config.run_date).normalize()
            if config.run_date is not None
            else pd.Timestamp.now().normalize()
        )
        run_date = run_ts.strftime("%Y-%m-%d")
        run_key = str(uuid4())
        state_store = SQLiteExecutionStateStore(config.state_db_path)
        fingerprint = self._daily_run_fingerprint(strategy, config, run_date)
        claim = state_store.claim_run(
            strategy_id=metadata.strategy_id,
            strategy_version=metadata.version,
            run_date=run_date,
            mode=config.mode,
            run_key=run_key,
            fingerprint=fingerprint,
            force=config.force,
        )
        if claim.status == "noop":
            existing = claim.existing_run
            assert existing is not None
            payload = existing.payload
            order_receipt = None
            if existing.order_summary is not None:
                order_receipt = DailyOrderReceipt(
                    status=str(existing.order_summary.get("status", "unknown")),
                    external_order_id=existing.order_summary.get("external_order_id"),
                    filled_notional_usd=float(
                        existing.order_summary.get("filled_notional_usd", 0.0)
                    ),
                    filled_quantity_btc=float(
                        existing.order_summary.get("filled_quantity_btc", 0.0)
                    ),
                    fill_price_usd=float(existing.order_summary.get("fill_price_usd", 0.0)),
                    metadata=dict(existing.order_summary.get("metadata") or {}),
                )
            return DailyRunResult(
                status="noop",
                strategy_id=metadata.strategy_id,
                strategy_version=metadata.version,
                run_date=run_date,
                run_key=existing.run_key,
                mode=config.mode,
                idempotency_hit=True,
                forced_rerun=False,
                weight_today=(
                    float(payload["weight_today"])
                    if payload.get("weight_today") is not None
                    else None
                ),
                order_notional_usd=(
                    float(payload["order_notional_usd"])
                    if payload.get("order_notional_usd") is not None
                    else None
                ),
                btc_quantity=(
                    float(payload["btc_quantity"])
                    if payload.get("btc_quantity") is not None
                    else None
                ),
                price_usd=(
                    float(payload["price_usd"]) if payload.get("price_usd") is not None else None
                ),
                adapter_name=str(payload.get("adapter_name", "unknown")),
                state_db_path=str(state_store.db_path),
                artifact_path=(
                    str(payload["artifact_path"])
                    if payload.get("artifact_path") is not None
                    else None
                ),
                message="Existing completed run reused via idempotency ledger.",
                order_receipt=order_receipt,
                bootstrap=bool(payload.get("bootstrap", False)),
                validation_receipt_id=(
                    int(payload["validation_receipt_id"])
                    if payload.get("validation_receipt_id") is not None
                    else None
                ),
                validation_passed=payload.get("validation_passed"),
                data_hash=str(payload.get("data_hash", "")),
                feature_snapshot_hash=str(payload.get("feature_snapshot_hash", "")),
            )

        bootstrap = False
        validation_receipt_id: int | None = None
        data_hash = ""
        feature_snapshot_hash = ""
        try:
            btc_df_loaded = self._load_btc_df(btc_df, end_date=run_date)
            if config.btc_price_col not in btc_df_loaded.columns:
                raise ValueError(
                    f"Missing required BTC price column '{config.btc_price_col}'."
                )
            window_end = run_ts
            window_start = window_end - ALLOCATION_WINDOW_OFFSET
            required_index = pd.date_range(window_start, window_end, freq="D")
            required_window = btc_df_loaded.reindex(required_index)
            if required_window[config.btc_price_col].isna().any():
                raise ValueError(
                    "Missing BTC price data for required allocation window ending on run_date."
                )

            validation_config = ValidationConfig(
                start_date=window_start.strftime("%Y-%m-%d"),
                end_date=window_end.strftime("%Y-%m-%d"),
                strict=True,
            )
            validation_result = self.validate(
                strategy,
                validation_config,
                btc_df=btc_df_loaded,
            )
            provider_hash = self._feature_registry.provider_fingerprint(strategy)
            observed_features, feature_snapshot_hash = self._feature_registry.materialization_fingerprint(
                strategy,
                btc_df_loaded,
                start_date=window_start,
                end_date=window_end,
                current_date=window_end,
            )
            data_hash = hash_dataframe(btc_df_loaded.loc[:window_end].copy())
            validation_receipt = state_store.create_validation_receipt(
                strategy_id=metadata.strategy_id,
                strategy_version=metadata.version,
                run_date=run_date,
                fingerprint=fingerprint,
                data_hash=data_hash,
                provider_hash=provider_hash,
                feature_snapshot_hash=feature_snapshot_hash,
                config_hash=self._config_hash(validation_config),
                passed=bool(validation_result.passed),
                diagnostics={
                    "summary": validation_result.summary(),
                    "messages": validation_result.messages,
                    "diagnostics": validation_result.diagnostics,
                },
            )
            validation_receipt_id = validation_receipt.receipt_id
            if not validation_result.passed:
                raise ValueError(
                    "Strict validation failed before daily execution: "
                    + "; ".join(validation_result.messages)
                )

            locked_prefix = state_store.load_locked_prefix(
                strategy_id=metadata.strategy_id,
                strategy_version=metadata.version,
                mode=config.mode,
                run_date=run_date,
                window_start=window_start,
            )
            if locked_prefix is None:
                bootstrap = True
            ctx = StrategyContext(
                features_df=observed_features,
                start_date=window_start,
                end_date=window_end,
                current_date=window_end,
                locked_weights=locked_prefix,
                btc_price_col=config.btc_price_col,
            )
            weights = strategy.compute_weights(ctx)
            self._validate_weights(weights, window_start, window_end)
            strategy.validate_weights(weights, ctx)
            if run_ts not in weights.index:
                raise ValueError("Strategy weights are missing run_date allocation.")

            weight_today = float(weights.loc[run_ts])
            price_usd = float(required_window.loc[run_ts, config.btc_price_col])
            if price_usd <= 0.0:
                raise ValueError("BTC price must be greater than 0 for run_date.")
            order_notional_usd = budget * weight_today
            btc_quantity = order_notional_usd / price_usd

            if config.mode == "paper":
                adapter = PaperExecutionAdapter()
            else:
                adapter = load_execution_adapter(config.adapter_spec or "")
            adapter_name = adapter.__class__.__name__
            order_request = DailyOrderRequest(
                strategy_id=metadata.strategy_id,
                strategy_version=metadata.version,
                run_date=run_date,
                mode=config.mode,
                weight_today=weight_today,
                notional_usd=order_notional_usd,
                price_usd=price_usd,
                quantity_btc=btc_quantity,
                btc_price_col=config.btc_price_col,
            )
            receipt = adapter.submit_order(order_request, idempotency_key=run_key)
            if isinstance(receipt, dict):
                receipt = DailyOrderReceipt(**receipt)
            if not isinstance(receipt, DailyOrderReceipt):
                raise TypeError("Execution adapter must return DailyOrderReceipt.")

            output_root = (
                Path(config.output_dir)
                / metadata.strategy_id
                / metadata.version
                / "daily"
                / run_date
            )
            output_root.mkdir(parents=True, exist_ok=True)
            artifact_path = output_root / f"{run_key}.json"
            result = DailyRunResult(
                status="executed",
                strategy_id=metadata.strategy_id,
                strategy_version=metadata.version,
                run_date=run_date,
                run_key=run_key,
                mode=config.mode,
                idempotency_hit=False,
                forced_rerun=bool(config.force or claim.forced_overwrite),
                weight_today=weight_today,
                order_notional_usd=order_notional_usd,
                btc_quantity=btc_quantity,
                price_usd=price_usd,
                adapter_name=adapter_name,
                state_db_path=str(state_store.db_path),
                artifact_path=str(artifact_path),
                message=(
                    "Daily execution completed."
                    if not bootstrap
                    else "Daily execution completed (bootstrap run with no prior snapshot)."
                ),
                order_receipt=receipt,
                bootstrap=bootstrap,
                validation_receipt_id=validation_receipt_id,
                validation_passed=True,
                data_hash=data_hash,
                feature_snapshot_hash=feature_snapshot_hash,
            )
            result_payload = result.to_json(path=artifact_path)
            state_store.mark_run_success_with_snapshot(
                strategy_id=metadata.strategy_id,
                strategy_version=metadata.version,
                run_date=run_date,
                mode=config.mode,
                payload=result_payload,
                order_summary=asdict(receipt),
                force_flag=bool(config.force or claim.forced_overwrite),
                snapshot_date=run_date,
                weights=weights,
                validation_receipt_id=validation_receipt_id,
                data_hash=data_hash,
                feature_snapshot_hash=feature_snapshot_hash,
            )
            return result
        except IdempotencyConflictError:
            raise
        except Exception as exc:
            error_message = str(exc)
            failure_payload = {
                "status": "failed",
                "strategy_id": metadata.strategy_id,
                "strategy_version": metadata.version,
                "run_date": run_date,
                "run_key": run_key,
                "mode": config.mode,
                "error": error_message,
            }
            state_store.mark_run_failure(
                strategy_id=metadata.strategy_id,
                strategy_version=metadata.version,
                run_date=run_date,
                mode=config.mode,
                payload=failure_payload,
                force_flag=bool(config.force or claim.forced_overwrite),
                validation_receipt_id=validation_receipt_id,
                data_hash=data_hash,
                feature_snapshot_hash=feature_snapshot_hash,
            )
            return DailyRunResult(
                status="failed",
                strategy_id=metadata.strategy_id,
                strategy_version=metadata.version,
                run_date=run_date,
                run_key=run_key,
                mode=config.mode,
                idempotency_hit=False,
                forced_rerun=bool(config.force or claim.forced_overwrite),
                weight_today=None,
                order_notional_usd=None,
                btc_quantity=None,
                price_usd=None,
                adapter_name=(
                    "PaperExecutionAdapter" if config.mode == "paper" else "custom-adapter"
                ),
                state_db_path=str(state_store.db_path),
                artifact_path=None,
                message=f"Daily execution failed: {error_message}",
                order_receipt=None,
                bootstrap=bootstrap,
                validation_receipt_id=validation_receipt_id,
                validation_passed=False if validation_receipt_id is not None else None,
                data_hash=data_hash,
                feature_snapshot_hash=feature_snapshot_hash,
            )

    def reconcile_daily_run(
        self,
        strategy: BaseStrategy,
        *,
        run_date: str,
        mode: str = "paper",
        state_db_path: str = ".stacksats/run_state.sqlite3",
        btc_df: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        from .execution_state import SQLiteExecutionStateStore

        self._validate_strategy_contract(strategy)
        metadata = strategy.metadata()
        state_store = SQLiteExecutionStateStore(state_db_path)
        stored = state_store.get_run(
            strategy_id=metadata.strategy_id,
            strategy_version=metadata.version,
            run_date=run_date,
            mode=mode,
        )
        if stored is None:
            raise ValueError("No stored daily run exists for the requested key.")
        btc_df_loaded = self._load_btc_df(btc_df, end_date=run_date)
        run_ts = pd.Timestamp(run_date).normalize()
        window_end = run_ts
        window_start = window_end - ALLOCATION_WINDOW_OFFSET
        locked_prefix = state_store.load_locked_prefix(
            strategy_id=metadata.strategy_id,
            strategy_version=metadata.version,
            mode=mode,
            run_date=run_date,
            window_start=window_start,
        )
        features_df, feature_snapshot_hash = self._feature_registry.materialization_fingerprint(
            strategy,
            btc_df_loaded,
            start_date=window_start,
            end_date=window_end,
            current_date=window_end,
        )
        weights = strategy.compute_weights(
            StrategyContext(
                features_df=features_df,
                start_date=window_start,
                end_date=window_end,
                current_date=window_end,
                locked_weights=locked_prefix,
            )
        )
        weight_today = float(weights.loc[run_ts]) if run_ts in weights.index else None
        prior_weight = stored.payload.get("weight_today")
        if prior_weight is None or weight_today is None:
            status = "stable"
        elif abs(float(prior_weight) - float(weight_today)) <= 1e-12:
            if stored.payload.get("feature_snapshot_hash", "") == feature_snapshot_hash:
                status = "stable"
            else:
                status = "data_revised_no_decision_change"
        else:
            status = "decision_changed_due_to_revision"
        return {
            "status": status,
            "previous_weight_today": prior_weight,
            "recomputed_weight_today": weight_today,
            "previous_feature_snapshot_hash": stored.payload.get("feature_snapshot_hash", ""),
            "recomputed_feature_snapshot_hash": feature_snapshot_hash,
        }

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
        metadata = strategy.metadata()
        btc_df = self._load_btc_df(btc_df, end_date=config.range_end)
        run_date = current_date or pd.Timestamp.now().normalize()
        features_df = self._materialize_strategy_features(
            strategy,
            btc_df,
            start_date=pd.Timestamp(config.range_start),
            end_date=pd.Timestamp(config.range_end),
            current_date=run_date,
        )

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
            / metadata.strategy_id
            / metadata.version
            / provenance["run_id"]
        )
        output_root.mkdir(parents=True, exist_ok=True)
        result_path = output_root / "weights.csv"
        export_df = series_batch.to_dataframe()
        export_df.to_csv(result_path, index=False)
        schema_path = output_root / "timeseries_schema.md"
        schema_path.write_text(series_batch.schema_markdown(), encoding="utf-8")
        artifact_metadata = StrategyArtifactSet(
            strategy_id=provenance["strategy_id"],
            version=provenance["version"],
            config_hash=provenance["config_hash"],
            run_id=provenance["run_id"],
            output_dir=str(output_root),
            files={
                "weights_csv": result_path.name,
                "timeseries_schema_md": schema_path.name,
            },
        )
        (output_root / "artifacts.json").write_text(
            json.dumps(asdict(artifact_metadata), indent=2),
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

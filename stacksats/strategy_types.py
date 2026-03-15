"""Strategy-first domain types for StackSats."""

from __future__ import annotations

import datetime as dt
import inspect
import math
import types as pytypes
import warnings
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from .feature_registry import DEFAULT_FEATURE_REGISTRY
from .feature_time_series import FeatureTimeSeries
from .framework_contract import ALLOCATION_SPAN_DAYS, MAX_DAILY_WEIGHT, MIN_DAILY_WEIGHT
from .strategy_lint import lint_strategy_class, summarize_lint_findings

if TYPE_CHECKING:
    from .api import BacktestResult, DailyRunResult, ValidationResult
    from .strategy_time_series import WeightTimeSeriesBatch


def _to_datetime(value: dt.datetime | str) -> dt.datetime:
    """Normalize date-like to naive datetime."""
    if isinstance(value, dt.datetime):
        out = value
    elif isinstance(value, str):
        try:
            out = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            out = dt.datetime.strptime(value[:10], "%Y-%m-%d")
    else:
        if hasattr(value, "to_pydatetime"):
            out = value.to_pydatetime()
        else:
            out = dt.datetime.strptime(str(value)[:10], "%Y-%m-%d")
    if out.tzinfo is not None:
        out = out.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return out.replace(hour=0, minute=0, second=0, microsecond=0)


def strategy_context_from_features_df(
    features_df: pl.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    current_date: dt.datetime | str,
    *,
    required_columns: tuple[str, ...] = (),
    as_of_date: dt.datetime | str | None = None,
    locked_weights: np.ndarray | None = None,
    btc_price_col: str = "price_usd",
    mvrv_col: str = "mvrv",
) -> StrategyContext:
    """Build a StrategyContext from a Polars feature DataFrame."""
    return StrategyContext.from_features_df(
        features_df,
        start_date,
        end_date,
        current_date,
        required_columns=required_columns,
        as_of_date=as_of_date,
        locked_weights=locked_weights,
        btc_price_col=btc_price_col,
        mvrv_col=mvrv_col,
    )


@dataclass(frozen=True)
class StrategyContext:
    """Normalized context passed into strategy computation."""

    features: FeatureTimeSeries
    start_date: dt.datetime
    end_date: dt.datetime
    current_date: dt.datetime
    locked_weights: np.ndarray | None = None
    btc_price_col: str = "price_usd"
    mvrv_col: str = "mvrv"

    @classmethod
    def from_features_df(
        cls,
        features_df: pl.DataFrame,
        start_date: dt.datetime | str,
        end_date: dt.datetime | str,
        current_date: dt.datetime | str,
        *,
        required_columns: tuple[str, ...] = (),
        as_of_date: dt.datetime | str | None = None,
        locked_weights: np.ndarray | None = None,
        btc_price_col: str = "price_usd",
        mvrv_col: str = "mvrv",
    ) -> StrategyContext:
        """Build a StrategyContext from a Polars feature DataFrame.

        When as_of_date is provided, validates no forward-looking data
        (max date in features <= as_of_date).
        """
        as_of = _to_datetime(as_of_date) if as_of_date is not None else None
        features = FeatureTimeSeries.from_dataframe(
            features_df,
            required_columns=required_columns,
            as_of_date=as_of,
        )
        return cls(
            features=features,
            start_date=_to_datetime(start_date),
            end_date=_to_datetime(end_date),
            current_date=_to_datetime(current_date),
            locked_weights=locked_weights,
            btc_price_col=btc_price_col,
            mvrv_col=mvrv_col,
        )

    @property
    def features_df(self) -> pl.DataFrame:
        """Polars DataFrame of features. Alias for ctx.features.data."""
        return self.features.data


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str | None = None
    end_date: str | None = None
    strategy_label: str | None = None
    output_dir: str = "output"


@dataclass(frozen=True)
class ValidationConfig:
    start_date: str | None = None
    end_date: str | None = None
    min_win_rate: float = 50.0
    strict: bool = False
    min_fold_win_rate: float = 20.0
    max_fold_win_rate_std: float = 35.0
    max_shuffled_win_rate: float = 80.0
    shuffled_trials: int = 3
    max_boundary_hit_rate: float = 0.85
    block_size: int = 30
    bootstrap_trials: int = 500
    permutation_trials: int = 200
    max_permutation_pvalue: float = 0.10
    min_bootstrap_ci_lower_excess: float = 0.0
    max_feature_psi: float = 0.25
    max_feature_ks: float = 0.25


def _default_export_range_start() -> str:
    today = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (today - dt.timedelta(days=364)).strftime("%Y-%m-%d")


def _default_export_range_end() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .strftime("%Y-%m-%d")
    )


@dataclass(frozen=True)
class ExportConfig:
    range_start: str = field(default_factory=_default_export_range_start)
    range_end: str = field(default_factory=_default_export_range_end)
    output_dir: str = "output"
    btc_price_col: str = "price_usd"


def _default_state_db_path() -> str:
    return str(Path(".stacksats") / "run_state.sqlite3")


@dataclass(frozen=True)
class RunDailyConfig:
    run_date: str | None = None
    total_window_budget_usd: float = 1000.0
    mode: Literal["paper", "live"] = "paper"
    state_db_path: str = field(default_factory=_default_state_db_path)
    output_dir: str = "output"
    adapter_spec: str | None = None
    force: bool = False
    btc_price_col: str = "price_usd"


@dataclass(frozen=True)
class StrategyMetadata:
    """Canonical strategy identity and descriptive metadata."""

    strategy_id: str
    version: str
    description: str = ""


@dataclass(frozen=True)
class StrategySpec:
    """Stable public strategy contract for identity, intent, and params."""

    metadata: StrategyMetadata
    intent_mode: Literal["propose", "profile"]
    params: dict[str, object]
    required_feature_sets: tuple[str, ...] = ()
    required_feature_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyArtifactSet:
    """Artifact bundle with strategy provenance metadata."""

    strategy_id: str
    version: str
    config_hash: str
    run_id: str
    output_dir: str
    files: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetProfile:
    """User-provided target profile or daily preference score.

    values: pl.DataFrame with columns 'date' and 'value'.
    """

    values: pl.DataFrame
    mode: Literal["preference", "absolute"] = "preference"


@dataclass(frozen=True)
class StrategyRunResult:
    """Composite result for a full strategy lifecycle run."""

    validation: "ValidationResult"
    backtest: "BacktestResult"
    export_batch: "WeightTimeSeriesBatch | None" = None
    output_dir: str | None = None


@dataclass(frozen=True)
class DayState:
    """Per-day user hook input for proposing today's weight.

    features: pl.DataFrame with one row (that day's feature values).
    """

    current_date: dt.datetime
    features: pl.DataFrame
    remaining_budget: float
    day_index: int
    total_days: int
    uniform_weight: float


class StrategyContractWarning(UserWarning):
    """Warning emitted for ambiguous or transitional strategy contracts."""


_METADATA_FIELD_NAMES = frozenset({"strategy_id", "version", "description", "intent_preference"})


def _normalize_param_value(value: object, *, key_path: str) -> object:
    if isinstance(value, np.generic):
        return _normalize_param_value(value.item(), key_path=key_path)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(
                f"Unsupported strategy param value for '{key_path}' ({value!r}). "
                "Override params() to return only finite JSON-serializable config values."
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [
            _normalize_param_value(item, key_path=f"{key_path}[{idx}]")
            for idx, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        dict_keys = list(value)
        for sub_key in dict_keys:
            if not isinstance(sub_key, str):
                raise TypeError(
                    f"Unsupported strategy param key at {key_path}: "
                    f"dict keys must be strings, got {type(sub_key).__name__}."
                )
        for sub_key in sorted(dict_keys):
            normalized[sub_key] = _normalize_param_value(
                value[sub_key], key_path=f"{key_path}.{sub_key}"
            )
        return normalized
    raise TypeError(
        f"Unsupported strategy param value for '{key_path}' ({type(value).__name__}). "
        "Override params() to return only JSON-serializable config values."
    )


def _validate_metadata(metadata: StrategyMetadata) -> StrategyMetadata:
    if not isinstance(metadata.strategy_id, str) or not metadata.strategy_id:
        raise ValueError("Strategy metadata must define non-empty strategy_id.")
    if not isinstance(metadata.version, str) or not metadata.version:
        raise ValueError("Strategy metadata must define non-empty version.")
    if not isinstance(metadata.description, str):
        raise TypeError("Strategy metadata description must be a string.")
    return metadata


def _validate_required_feature_columns(
    strategy: "BaseStrategy",
    features_df: pl.DataFrame,
) -> None:
    required_columns = tuple(strategy.required_feature_columns())
    if not required_columns:
        return
    missing = [column for column in required_columns if column not in features_df.columns]
    if not missing:
        return
    metadata = strategy.metadata()
    available = list(features_df.columns)
    preview = ", ".join(available[:10])
    if len(available) > 10:
        preview = f"{preview}, ..."
    raise ValueError(
        "Missing required feature columns for strategy "
        f"{metadata.strategy_id}@{metadata.version}: {missing}. "
        f"Available columns ({len(available)}): {preview}"
    )


class BaseStrategy(ABC):
    """Base class for strategy-first runtime behavior."""

    strategy_id: str = "custom-strategy"
    version: str = "0.1.0"
    description: str = ""
    intent_preference: Literal["propose", "profile"] | None = None

    def metadata(self) -> StrategyMetadata:
        """Return canonical strategy metadata."""
        return _validate_metadata(
            StrategyMetadata(
                strategy_id=self.strategy_id,
                version=self.version,
                description=self.description,
            )
        )

    def params(self) -> dict[str, object]:
        """Return stable strategy configuration for provenance and idempotency."""
        params: dict[str, object] = {}
        for cls in reversed(self.__class__.mro()):
            if cls in {object, BaseStrategy}:
                continue
            for name, value in cls.__dict__.items():
                if name.startswith("_") or name in _METADATA_FIELD_NAMES:
                    continue
                if isinstance(value, (staticmethod, classmethod, property)):
                    continue
                if inspect.isroutine(value) or inspect.isdatadescriptor(value):
                    continue
                if isinstance(value, (type, pytypes.ModuleType)):
                    continue
                params[name] = _normalize_param_value(value, key_path=name)
        for name, value in self.__dict__.items():
            if name.startswith("_") or name in _METADATA_FIELD_NAMES:
                continue
            if callable(value) or isinstance(value, (type, pytypes.ModuleType)):
                continue
            params[name] = _normalize_param_value(value, key_path=name)
        return {key: params[key] for key in sorted(params)}

    def spec(self) -> StrategySpec:
        """Return canonical public strategy contract surface."""
        return StrategySpec(
            metadata=self.metadata(),
            intent_mode=self.intent_mode(),
            params=self.params(),
            required_feature_sets=tuple(self.required_feature_sets()),
            required_feature_columns=tuple(self.required_feature_columns()),
        )

    def intent_mode(self) -> Literal["propose", "profile"]:
        """Return the active intent execution mode for this strategy."""
        has_propose_hook, has_profile_hook = self.hook_status()
        if not (has_propose_hook or has_profile_hook):
            raise TypeError(
                "Strategy must implement propose_weight(state) or "
                "build_target_profile(ctx, features_df, signals)."
            )
        if has_propose_hook and has_profile_hook:
            if self.intent_preference in {"propose", "profile"}:
                return self.intent_preference
            warnings.warn(
                "Strategy implements both propose_weight(state) and "
                "build_target_profile(ctx, features_df, signals). "
                "Current fallback uses propose_weight(state). "
                "Set intent_preference = 'propose' or 'profile' to make this explicit "
                "before the next breaking release.",
                StrategyContractWarning,
                stacklevel=2,
            )
            return "propose"
        return "propose" if has_propose_hook else "profile"

    def required_feature_columns(self) -> tuple[str, ...]:
        """Return transformed feature columns required for this strategy."""
        return ()

    def required_feature_sets(self) -> tuple[str, ...]:
        """Return framework-owned feature provider IDs required by this strategy."""
        return ("core_model_features_v1",)

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        """Hook for user-defined feature transforms on the active window."""
        return ctx.features.data.clone()

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
    ) -> dict[str, pl.Series]:
        """Hook for user-defined signal formulas."""
        del ctx, features_df
        return {}

    def propose_weight(self, state: DayState) -> float:
        """Optional per-day weight proposal hook."""
        del state
        raise NotImplementedError(
            "Override propose_weight(state) or build_target_profile(...) in your strategy."
        )

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile | pl.DataFrame:
        """Return daily preference scores or absolute target profile.

        Default behavior vectorizes `propose_weight` over the active window.
        Return TargetProfile with values as pl.DataFrame (date, value) or pl.DataFrame directly.
        """
        del signals
        if features_df.is_empty():
            return TargetProfile(
                values=pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64}),
                mode="absolute",
            )
        return TargetProfile(values=self._collect_proposals(features_df), mode="absolute")

    def _validate_target_df(
        self,
        df: pl.DataFrame,
        *,
        name: str,
        date_col: pl.Series,
    ) -> pl.DataFrame:
        """Validate target/profile DataFrame has date and value, finite numeric."""
        if "date" not in df.columns or "value" not in df.columns:
            raise ValueError(f"{name} must have 'date' and 'value' columns.")
        if df.height != date_col.len():
            raise ValueError(f"{name} length must match observed window.")
        vals = df["value"].cast(pl.Float64, strict=False)
        if vals.is_null().any() or not vals.is_finite().all():
            raise ValueError(f"{name} must contain finite numeric values.")
        return df.select(["date", pl.col("value").cast(pl.Float64)])

    def _collect_proposals(self, features_df: pl.DataFrame) -> pl.DataFrame:
        """Collect per-day proposals with shared framework safeguards."""
        if features_df.is_empty():
            return pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64})
        date_col = features_df["date"]
        total_days = features_df.height
        uniform_weight = 1.0 / total_days
        remaining_budget = 1.0
        proposed: list[float] = []
        for day_index in range(total_days):
            row = features_df.row(day_index, named=True)
            current_date = _to_datetime(row["date"])
            row_df = features_df.slice(day_index, 1)
            state = DayState(
                current_date=current_date,
                features=row_df,
                remaining_budget=remaining_budget,
                day_index=day_index,
                total_days=total_days,
                uniform_weight=uniform_weight,
            )
            proposal = float(self.propose_weight(state))
            if not math.isfinite(proposal):
                raise ValueError("propose_weight must return a finite numeric value.")
            proposed.append(proposal)
            remaining_budget -= float(np.clip(proposal, 0.0, remaining_budget))
        return pl.DataFrame({"date": date_col, "value": proposed})

    def compute_weights(self, ctx: StrategyContext) -> pl.DataFrame:
        """Framework-owned orchestration from hooks -> final weights.

        Returns pl.DataFrame with columns 'date' and 'weight'.
        """
        from .framework_contract import compute_n_past

        observed_end = (
            ctx.current_date
            if ctx.current_date <= ctx.end_date
            else ctx.end_date
        )
        date_range = pl.datetime_range(
            ctx.start_date, observed_end, interval="1d", eager=True
        )
        features_df = self.transform_features(ctx)
        if not isinstance(features_df, pl.DataFrame):
            raise TypeError("transform_features must return a polars DataFrame.")
        if date_range.len() == 0:
            return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})

        full_dates = pl.DataFrame({"date": date_range})
        features_df = full_dates.join(features_df, on="date", how="left")
        features_df = features_df.sort("date").fill_null(0.0)
        _validate_required_feature_columns(self, features_df)

        signals = self.build_signals(ctx, features_df)
        if not isinstance(signals, dict):
            raise TypeError("build_signals must return a dict[str, pl.Series].")
        for key, series in signals.items():
            if not isinstance(series, pl.Series) or series.len() != date_range.len():
                raise ValueError(
                    f"signal '{key}' must be pl.Series with length matching window."
                )
            arr = series.cast(pl.Float64, strict=False)
            if arr.is_null().any() or not arr.is_finite().all():
                raise ValueError(f"signal '{key}' must contain finite numeric values.")

        n_past = compute_n_past(date_range.to_list(), ctx.current_date)
        intent_mode = self.intent_mode()

        if intent_mode == "propose":
            from .model_development import compute_weights_from_proposals

            proposals_df = self._collect_proposals(features_df)
            return compute_weights_from_proposals(
                proposals=proposals_df,
                start_date=ctx.start_date,
                end_date=ctx.end_date,
                n_past=n_past,
                locked_weights=ctx.locked_weights,
            )

        profile = self.build_target_profile(ctx, features_df, signals)
        if isinstance(profile, TargetProfile):
            mode = profile.mode
            target_df = profile.values
        else:
            mode = "preference"
            target_df = profile

        if not isinstance(target_df, pl.DataFrame):
            raise TypeError("target profile must be a Polars DataFrame or TargetProfile.")
        if target_df.is_empty():
            return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})
        self._validate_target_df(target_df, name="target profile", date_col=date_range)

        from .model_development import compute_weights_from_target_profile

        return compute_weights_from_target_profile(
            features_df=features_df,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
            current_date=ctx.current_date,
            target_profile=target_df,
            mode=mode,
            n_past=n_past,
            locked_weights=ctx.locked_weights,
        )

    def default_backtest_config(self) -> BacktestConfig:
        return BacktestConfig(strategy_label=self.metadata().strategy_id)

    def default_validation_config(self) -> ValidationConfig:
        return ValidationConfig()

    def default_export_config(self) -> ExportConfig:
        return ExportConfig()

    def hook_status(self) -> tuple[bool, bool]:
        """Return which intent hooks this strategy overrides."""
        return strategy_hook_status(self.__class__)

    def validate_contract(self) -> tuple[bool, bool]:
        """Validate framework contract compliance for this strategy instance."""
        return validate_strategy_contract(self)

    def validate_weights(self, weights: pl.DataFrame, ctx: StrategyContext) -> None:
        """Optional strategy-specific weight checks.

        weights: pl.DataFrame with 'weight' column.
        """
        del ctx
        if weights.is_empty():
            return
        w = weights["weight"]
        if (w < 0).any():
            raise ValueError("Weights contain negative values.")
        if weights.height == ALLOCATION_SPAN_DAYS:
            if (w < (MIN_DAILY_WEIGHT - 1e-12)).any():
                raise ValueError(f"Weights must be >= {MIN_DAILY_WEIGHT}.")
            if (w > (MAX_DAILY_WEIGHT + 1e-12)).any():
                raise ValueError(f"Weights must be <= {MAX_DAILY_WEIGHT}.")
        weight_sum = float(w.sum())
        if abs(weight_sum - 1.0) > 1e-5:
            raise ValueError(f"Weights must sum to 1.0 (got {weight_sum:.10f}).")

    def backtest(
        self,
        config: BacktestConfig | None = None,
        **kwargs,
    ) -> "BacktestResult":
        from .runner import StrategyRunner

        runner = StrategyRunner()
        return runner.backtest(self, config or self.default_backtest_config(), **kwargs)

    def validate(
        self,
        config: ValidationConfig | None = None,
        **kwargs,
    ) -> "ValidationResult":
        from .runner import StrategyRunner

        runner = StrategyRunner()
        return runner.validate(self, config or self.default_validation_config(), **kwargs)

    def export(
        self,
        config: ExportConfig | None = None,
        **kwargs,
    ) -> "WeightTimeSeriesBatch":
        from .runner import StrategyRunner

        runner = StrategyRunner()
        return runner.export(self, config or self.default_export_config(), **kwargs)

    def backtest_and_save(
        self,
        config: BacktestConfig | None = None,
        *,
        output_dir: str | None = None,
        write_plots: bool = True,
        write_json: bool = True,
        **kwargs,
    ) -> tuple["BacktestResult", str]:
        """Run backtest and persist standard artifacts to disk."""
        result = self.backtest(config=config, **kwargs)
        root = (
            Path(output_dir or (config.output_dir if config else "output"))
            / result.strategy_id
            / result.strategy_version
            / result.run_id
        )
        root.mkdir(parents=True, exist_ok=True)
        if write_plots:
            result.plot(output_dir=str(root))
        if write_json:
            result.to_json(root / "backtest_result.json")
        return result, str(root)

    def run(
        self,
        *,
        validation_config: ValidationConfig | None = None,
        backtest_config: BacktestConfig | None = None,
        export_config: ExportConfig | None = None,
        include_export: bool = False,
        save_backtest_artifacts: bool = False,
        output_dir: str | None = None,
        btc_df: pl.DataFrame | None = None,
        current_date: dt.datetime | None = None,
    ) -> StrategyRunResult:
        """Run validate + backtest, with optional export and artifact persistence."""
        validation = self.validate(config=validation_config, btc_df=btc_df)
        if save_backtest_artifacts:
            backtest, saved_output_dir = self.backtest_and_save(
                config=backtest_config,
                output_dir=output_dir,
                btc_df=btc_df,
            )
        else:
            backtest = self.backtest(config=backtest_config, btc_df=btc_df)
            saved_output_dir = None

        export_batch = None
        if include_export:
            export_batch = self.export(
                config=export_config,
                btc_df=btc_df,
                current_date=current_date,
            )

        return StrategyRunResult(
            validation=validation,
            backtest=backtest,
            export_batch=export_batch,
            output_dir=saved_output_dir,
        )

    def run_daily(
        self,
        config: RunDailyConfig | None = None,
        **kwargs,
    ) -> "DailyRunResult":
        from .runner import StrategyRunner

        runner = StrategyRunner()
        return runner.run_daily(self, config or RunDailyConfig(), **kwargs)


def strategy_hook_status(strategy_cls: type[BaseStrategy]) -> tuple[bool, bool]:
    """Return whether strategy overrides propose/profile intent hooks."""
    has_propose_hook = strategy_cls.propose_weight is not BaseStrategy.propose_weight
    has_profile_hook = (
        strategy_cls.build_target_profile is not BaseStrategy.build_target_profile
    )
    return has_propose_hook, has_profile_hook


def validate_strategy_contract(strategy: BaseStrategy) -> tuple[bool, bool]:
    """Enforce framework-owned compute kernel boundaries for strategies."""
    if strategy.__class__.compute_weights is not BaseStrategy.compute_weights:
        raise TypeError(
            "Custom compute_weights overrides are not allowed. "
            "Implement propose_weight(state) or "
            "build_target_profile(ctx, features_df, signals) instead."
        )
    has_propose_hook, has_profile_hook = strategy_hook_status(strategy.__class__)
    if not (has_propose_hook or has_profile_hook):
        raise TypeError(
            "Strategy must implement propose_weight(state) or "
            "build_target_profile(ctx, features_df, signals)."
        )
    _validate_metadata(strategy.metadata())
    preference = strategy.intent_preference
    if preference is not None and preference not in {"propose", "profile"}:
        raise ValueError("intent_preference must be 'propose', 'profile', or None.")
    feature_sets = tuple(strategy.required_feature_sets())
    if len(feature_sets) == 0:
        raise ValueError("Strategy must declare at least one required_feature_set.")
    for provider_id in feature_sets:
        DEFAULT_FEATURE_REGISTRY.get(provider_id)
    lint_errors, _ = summarize_lint_findings(lint_strategy_class(strategy.__class__))
    if lint_errors:
        raise TypeError(
            "Strategy failed causal lint checks: " + " | ".join(lint_errors)
        )
    return has_propose_hook, has_profile_hook

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
import pandas as pd
from .framework_contract import ALLOCATION_SPAN_DAYS, MAX_DAILY_WEIGHT, MIN_DAILY_WEIGHT

if TYPE_CHECKING:
    from .api import BacktestResult, DailyRunResult, ValidationResult
    from .strategy_time_series import StrategyTimeSeriesBatch


@dataclass(frozen=True)
class StrategyContext:
    """Normalized context passed into strategy computation."""

    features_df: pd.DataFrame
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    current_date: pd.Timestamp
    locked_weights: np.ndarray | None = None
    btc_price_col: str = "PriceUSD_coinmetrics"
    mvrv_col: str = "CapMVRVCur"


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


def _default_export_range_start() -> str:
    today = pd.Timestamp.now().normalize()
    return (today - pd.Timedelta(days=364)).strftime("%Y-%m-%d")


def _default_export_range_end() -> str:
    return pd.Timestamp.now().normalize().strftime("%Y-%m-%d")


@dataclass(frozen=True)
class ExportConfig:
    range_start: str = field(default_factory=_default_export_range_start)
    range_end: str = field(default_factory=_default_export_range_end)
    output_dir: str = "output"
    btc_price_col: str = "PriceUSD_coinmetrics"


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
    btc_price_col: str = "PriceUSD_coinmetrics"


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
    required_feature_columns: tuple[str, ...]


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
    """User-provided target profile or daily preference score."""

    values: pd.Series
    mode: Literal["preference", "absolute"] = "preference"


@dataclass(frozen=True)
class StrategyRunResult:
    """Composite result for a full strategy lifecycle run."""

    validation: "ValidationResult"
    backtest: "BacktestResult"
    export_batch: "StrategyTimeSeriesBatch | None" = None
    output_dir: str | None = None


@dataclass(frozen=True)
class DayState:
    """Per-day user hook input for proposing today's weight."""

    current_date: pd.Timestamp
    features: pd.Series
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date)):
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
    features_df: pd.DataFrame,
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

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        """Hook for user-defined feature transforms on the active window."""
        return ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
    ) -> dict[str, pd.Series]:
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
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile | pd.Series:
        """Return daily preference scores or absolute target profile.

        Default behavior vectorizes `propose_weight` over the active window.
        """
        del signals
        if features_df.empty:
            return TargetProfile(values=pd.Series(dtype=float), mode="absolute")
        return TargetProfile(
            values=self._collect_proposals(features_df),
            mode="absolute",
        )

    def _validate_series(
        self,
        values: pd.Series,
        *,
        name: str,
        expected_index: pd.Index,
    ) -> pd.Series:
        if not isinstance(values, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")
        if values.index.has_duplicates:
            raise ValueError(f"{name} index must not contain duplicates.")
        if not values.index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be sorted ascending.")
        if not values.index.equals(expected_index):
            raise ValueError(f"{name} index must exactly match strategy window index.")
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain finite numeric values.")
        return numeric.astype(float)

    def _collect_proposals(self, features_df: pd.DataFrame) -> pd.Series:
        """Collect per-day proposals with shared framework safeguards."""
        if features_df.empty:
            return pd.Series(dtype=float)
        total_days = len(features_df.index)
        uniform_weight = 1.0 / total_days
        remaining_budget = 1.0
        proposed: list[float] = []
        for day_index, current_date in enumerate(features_df.index):
            state = DayState(
                current_date=current_date,
                features=features_df.loc[current_date],
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
        return pd.Series(proposed, index=features_df.index, dtype=float)

    def compute_weights(self, ctx: StrategyContext) -> pd.Series:
        """Framework-owned orchestration from hooks -> final weights."""
        from .framework_contract import compute_n_past

        expected_index = pd.date_range(ctx.start_date, ctx.end_date, freq="D")
        features_df = self.transform_features(ctx)
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError("transform_features must return a pandas DataFrame.")
        if len(expected_index) == 0:
            return pd.Series(dtype=float)
        features_df = features_df.reindex(expected_index).ffill().fillna(0.0)
        _validate_required_feature_columns(self, features_df)

        signals = self.build_signals(ctx, features_df)
        if not isinstance(signals, dict):
            raise TypeError("build_signals must return a dict[str, pandas.Series].")
        validated_signals: dict[str, pd.Series] = {}
        for key, series in signals.items():
            validated_signals[key] = self._validate_series(
                series, name=f"signal '{key}'", expected_index=expected_index
            )

        n_past = compute_n_past(expected_index, ctx.current_date)
        intent_mode = self.intent_mode()

        if intent_mode == "propose":
            from .model_development import compute_weights_from_proposals

            return compute_weights_from_proposals(
                proposals=self._collect_proposals(features_df),
                start_date=ctx.start_date,
                end_date=ctx.end_date,
                n_past=n_past,
                locked_weights=ctx.locked_weights,
            )

        profile = self.build_target_profile(ctx, features_df, validated_signals)
        if isinstance(profile, TargetProfile):
            mode = profile.mode
            values = profile.values
        else:
            mode = "preference"
            values = profile

        target_series = self._validate_series(
            values, name="target profile", expected_index=expected_index
        )
        from .model_development import compute_weights_from_target_profile

        return compute_weights_from_target_profile(
            features_df=features_df,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
            current_date=ctx.current_date,
            target_profile=target_series,
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

    def validate_weights(self, weights: pd.Series, ctx: StrategyContext) -> None:
        """Optional strategy-specific weight checks."""
        del ctx
        if weights.empty:
            return
        if bool((weights < 0).any()):
            raise ValueError("Weights contain negative values.")
        if len(weights) == ALLOCATION_SPAN_DAYS:
            if bool((weights < (MIN_DAILY_WEIGHT - 1e-12)).any()):
                raise ValueError(f"Weights must be >= {MIN_DAILY_WEIGHT}.")
            if bool((weights > (MAX_DAILY_WEIGHT + 1e-12)).any()):
                raise ValueError(f"Weights must be <= {MAX_DAILY_WEIGHT}.")
        weight_sum = float(weights.sum())
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
    ) -> "StrategyTimeSeriesBatch":
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
        btc_df: pd.DataFrame | None = None,
        current_date: pd.Timestamp | None = None,
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
    return has_propose_hook, has_profile_hook

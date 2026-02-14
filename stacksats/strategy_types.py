"""Strategy-first domain types for StackSats."""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from .framework_contract import ALLOCATION_SPAN_DAYS, MAX_DAILY_WEIGHT, MIN_DAILY_WEIGHT

if TYPE_CHECKING:
    from .api import BacktestResult, ValidationResult


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


@dataclass(frozen=True)
class ExportConfig:
    range_start: str = "2025-12-01"
    range_end: str = "2027-12-31"
    output_dir: str = "output"
    btc_price_col: str = "PriceUSD_coinmetrics"


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
class DayState:
    """Per-day user hook input for proposing today's weight."""

    current_date: pd.Timestamp
    features: pd.Series
    remaining_budget: float
    day_index: int
    total_days: int
    uniform_weight: float


class BaseStrategy(ABC):
    """Base class for strategy-first runtime behavior."""

    strategy_id: str = "custom-strategy"
    version: str = "0.1.0"
    description: str = ""

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
        return TargetProfile(
            values=pd.Series(proposed, index=features_df.index, dtype=float),
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

        signals = self.build_signals(ctx, features_df)
        if not isinstance(signals, dict):
            raise TypeError("build_signals must return a dict[str, pandas.Series].")
        validated_signals: dict[str, pd.Series] = {}
        for key, series in signals.items():
            validated_signals[key] = self._validate_series(
                series, name=f"signal '{key}'", expected_index=expected_index
            )

        strategy_cls = self.__class__
        has_propose_hook = strategy_cls.propose_weight is not BaseStrategy.propose_weight
        has_profile_hook = (
            strategy_cls.build_target_profile is not BaseStrategy.build_target_profile
        )
        if not (has_propose_hook or has_profile_hook):
            raise TypeError(
                "Strategy must implement propose_weight(state) or "
                "build_target_profile(ctx, features_df, signals)."
            )

        n_past = compute_n_past(expected_index, ctx.current_date)

        if has_propose_hook:
            from .model_development import compute_weights_from_proposals

            total_days = len(expected_index)
            uniform_weight = 1.0 / total_days
            remaining_budget = 1.0
            proposals: list[float] = []
            for day_index, current_date in enumerate(expected_index):
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
                proposals.append(proposal)
                remaining_budget -= float(np.clip(proposal, 0.0, remaining_budget))
            return compute_weights_from_proposals(
                proposals=pd.Series(proposals, index=expected_index, dtype=float),
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
        return BacktestConfig(strategy_label=self.strategy_id)

    def default_validation_config(self) -> ValidationConfig:
        return ValidationConfig()

    def default_export_config(self) -> ExportConfig:
        return ExportConfig()

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

    def export_weights(
        self,
        config: ExportConfig | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        from .runner import StrategyRunner

        runner = StrategyRunner()
        return runner.export(self, config or self.default_export_config(), **kwargs)

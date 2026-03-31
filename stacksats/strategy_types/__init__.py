"""Strategy-first domain types for StackSats."""

from ._legacy import (
    BaseStrategy,
    DayState,
    StrategyContext,
    StrategyLazyContext,
    TargetProfile,
    _normalize_param_value,
    _to_datetime,
    _validate_metadata,
    _validate_required_feature_column_names,
    _validate_required_feature_columns,
    strategy_context_from_features_df,
)
from .config import (
    AgentServiceConfig,
    BacktestConfig,
    DecideDailyConfig,
    ExportConfig,
    RunDailyConfig,
    ValidationConfig,
)
from .contracts import strategy_hook_status, validate_strategy_contract
from .metadata import (
    StrategyArtifactSet,
    StrategyMetadata,
    StrategyRunResult,
    StrategySpec,
)

__all__ = [
    "AgentServiceConfig",
    "BacktestConfig",
    "BaseStrategy",
    "DayState",
    "DecideDailyConfig",
    "ExportConfig",
    "RunDailyConfig",
    "StrategyArtifactSet",
    "StrategyContext",
    "StrategyLazyContext",
    "StrategyMetadata",
    "StrategyRunResult",
    "StrategySpec",
    "TargetProfile",
    "ValidationConfig",
    "_normalize_param_value",
    "_to_datetime",
    "_validate_metadata",
    "_validate_required_feature_column_names",
    "_validate_required_feature_columns",
    "strategy_context_from_features_df",
    "strategy_hook_status",
    "validate_strategy_contract",
]

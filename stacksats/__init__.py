"""StackSats package public API."""

from .api import (
    BacktestResult,
    ComparisonResult,
    ComparisonRow,
    DailyDecisionResult,
    DailyOrderReceipt,
    DailyOrderRequest,
    DailyRunResult,
    ExecutionReceiptEvent,
    ExecutionReceiptHistoryResult,
    ExecutionStatusResult,
    ValidationResult,
)
from .features.column_map_provider import ColumnMapDataProvider, ColumnMapError
from .eda import MergedMetricsDataset, MetricCatalog, load_metric_catalog, open_merged_metrics
from .features.time_series import FeatureTimeSeries
from .loader import load_strategy
from .model_development import precompute_features
from .data.prelude import load_data
from .runner import StrategyRunner
from .service import create_agent_service_app
from .strategies.catalog import (
    StrategyCatalogEntry,
    get_strategy_catalog_entry,
    iter_catalog_strategy_classes,
    list_strategies,
)
from .strategy_time_series import (
    ColumnSpec,
    StrategySeriesMetadata,
    WeightTimeSeries,
    WeightTimeSeriesBatch,
)
from .strategy_types import (
    AgentServiceConfig,
    BacktestConfig,
    BaseStrategy,
    ComparisonConfig,
    DecideDailyConfig,
    DayState,
    ExportConfig,
    RunDailyConfig,
    StrategyArtifactSet,
    StrategyContext,
    StrategyLazyContext,
    StrategyMetadata,
    StrategyRunResult,
    StrategySpec,
    TargetProfile,
    ValidationConfig,
    strategy_context_from_features_df,
)

_STABLE_PUBLIC_CLASSES = iter_catalog_strategy_classes(tier="stable", public_only=True)
for _strategy_cls in _STABLE_PUBLIC_CLASSES:
    globals()[_strategy_cls.__name__] = _strategy_cls

__all__ = [
    # Primary objects
    "FeatureTimeSeries",
    "MergedMetricsDataset",
    "WeightTimeSeries",
    "WeightTimeSeriesBatch",
    "BaseStrategy",
    # Results
    "BacktestResult",
    "ComparisonResult",
    "ComparisonRow",
    "DailyDecisionResult",
    "ExecutionReceiptEvent",
    "ExecutionReceiptHistoryResult",
    "ExecutionStatusResult",
    "ValidationResult",
    "DailyRunResult",
    # Configs
    "AgentServiceConfig",
    "BacktestConfig",
    "ComparisonConfig",
    "DecideDailyConfig",
    "ExportConfig",
    "RunDailyConfig",
    "ValidationConfig",
    # Supporting types
    "ColumnMapDataProvider",
    "ColumnMapError",
    "ColumnSpec",
    "MetricCatalog",
    "DailyOrderReceipt",
    "DailyOrderRequest",
    "DayState",
    "StrategyRunner",
    "StrategySeriesMetadata",
    "StrategyArtifactSet",
    "StrategyCatalogEntry",
    "StrategyContext",
    "StrategyLazyContext",
    "StrategyMetadata",
    "StrategyRunResult",
    "StrategySpec",
    "TargetProfile",
    "create_agent_service_app",
    "get_strategy_catalog_entry",
    "load_strategy",
    "load_metric_catalog",
    "load_data",
    "list_strategies",
    "open_merged_metrics",
    "precompute_features",
    "strategy_context_from_features_df",
]
__all__.extend(_strategy_cls.__name__ for _strategy_cls in _STABLE_PUBLIC_CLASSES)

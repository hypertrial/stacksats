"""StackSats package public API."""

from .api import (
    BacktestResult,
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
from .strategy_time_series import (
    ColumnSpec,
    StrategySeriesMetadata,
    WeightTimeSeries,
    WeightTimeSeriesBatch,
)
from .strategies.examples import (
    MomentumStrategy,
    RunDailyPaperStrategy,
    SimpleZScoreStrategy,
    UniformStrategy,
)
from .strategies.mvrv import MVRVStrategy
from .strategy_types import (
    AgentServiceConfig,
    BacktestConfig,
    BaseStrategy,
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

__all__ = [
    # Primary objects
    "FeatureTimeSeries",
    "MergedMetricsDataset",
    "WeightTimeSeries",
    "WeightTimeSeriesBatch",
    "BaseStrategy",
    # Results
    "BacktestResult",
    "DailyDecisionResult",
    "ExecutionReceiptEvent",
    "ExecutionReceiptHistoryResult",
    "ExecutionStatusResult",
    "ValidationResult",
    "DailyRunResult",
    # Configs
    "AgentServiceConfig",
    "BacktestConfig",
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
    "MomentumStrategy",
    "MVRVStrategy",
    "RunDailyPaperStrategy",
    "SimpleZScoreStrategy",
    "StrategyRunner",
    "StrategySeriesMetadata",
    "StrategyArtifactSet",
    "StrategyContext",
    "StrategyLazyContext",
    "StrategyMetadata",
    "StrategyRunResult",
    "StrategySpec",
    "TargetProfile",
    "UniformStrategy",
    "create_agent_service_app",
    "load_strategy",
    "load_metric_catalog",
    "load_data",
    "open_merged_metrics",
    "precompute_features",
    "strategy_context_from_features_df",
]

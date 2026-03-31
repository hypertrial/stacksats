"""Result types for strategy lifecycle operations."""

from .backtest import BacktestResult, ValidationResult, WIN_RATE_TOLERANCE
from .daily import DailyDecisionResult, DailyOrderReceipt, DailyOrderRequest, DailyRunResult
from .execution import ExecutionReceiptEvent, ExecutionReceiptHistoryResult, ExecutionStatusResult

__all__ = [
    "BacktestResult",
    "ValidationResult",
    "WIN_RATE_TOLERANCE",
    "DailyDecisionResult",
    "DailyOrderReceipt",
    "DailyOrderRequest",
    "DailyRunResult",
    "ExecutionReceiptEvent",
    "ExecutionReceiptHistoryResult",
    "ExecutionStatusResult",
]

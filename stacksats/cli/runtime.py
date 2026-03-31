"""Compatibility façade for split CLI helper modules."""

from . import (
    _HelpFormatter,
    _add_backtest_command,
    _add_export_command,
    _add_strategy_spec_arguments,
    _add_validate_command,
    _build_parser,
    _run_lifecycle_command,
    _start_agent_service_from_args,
    main,
)

__all__ = [
    "_HelpFormatter",
    "_add_backtest_command",
    "_add_export_command",
    "_add_strategy_spec_arguments",
    "_add_validate_command",
    "_build_parser",
    "_run_lifecycle_command",
    "_start_agent_service_from_args",
    "main",
]

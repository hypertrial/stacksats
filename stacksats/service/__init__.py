"""HTTP agent service public entrypoints."""

from __future__ import annotations

from typing import Any

from ..strategy_types import AgentServiceConfig


def create_agent_service_app(config: AgentServiceConfig) -> Any:
    """Create the FastAPI app for the StackSats agent service."""
    from .app import create_agent_service_app as _create_agent_service_app

    return _create_agent_service_app(config)


def start_agent_service(config: AgentServiceConfig) -> None:
    """Start the StackSats agent service with uvicorn."""
    from .app import start_agent_service as _start_agent_service

    _start_agent_service(config)


__all__ = ["create_agent_service_app", "start_agent_service"]

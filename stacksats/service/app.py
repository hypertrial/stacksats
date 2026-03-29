"""FastAPI service exposing StackSats daily decisions and receipt ingestion."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone
import time
from typing import Any, Literal
import uuid

from .._contract import PUBLIC_ARTIFACT_SCHEMA_VERSION
from .._optional import import_optional, missing_dependency_error

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ModuleNotFoundError as exc:
    if exc.name == "fastapi" or exc.name.startswith("fastapi."):
        raise missing_dependency_error(
            dependency="fastapi",
            extra="service",
            feature="agent HTTP service",
        ) from exc
    raise

try:
    from pydantic import BaseModel, Field, model_validator
except ModuleNotFoundError as exc:
    if exc.name == "pydantic" or exc.name.startswith("pydantic."):
        raise missing_dependency_error(
            dependency="pydantic",
            extra="service",
            feature="agent HTTP service",
        ) from exc
    raise

from fastapi.responses import JSONResponse

from ..api import ExecutionReceiptEvent
from ..execution_state import ReceiptConflictError, SQLiteExecutionStateStore
from ..loader import load_strategy
from ..runner import StrategyRunner
from ..strategy_types import AgentServiceConfig, DecideDailyConfig
from .registry import RegisteredStrategy, load_strategy_registry

logger = logging.getLogger(__name__)


class _HealthResponse(BaseModel):
    status: str
    api_version: str


class _DiscoveryAuth(BaseModel):
    type: str
    scheme: str
    header: str


class _DiscoveryEndpoints(BaseModel):
    healthz: str
    openapi: str
    decision_create: str
    decision_get: str
    execution_receipt_create: str
    execution_get: str
    execution_receipts_get: str


class _DiscoveryDocument(BaseModel):
    schema_version: str
    service: str
    api_version: str
    auth: _DiscoveryAuth
    endpoints: _DiscoveryEndpoints
    receipt_event_types: list[str]


class _DecisionRequest(BaseModel):
    strategy_id: str
    total_window_budget_usd: float = Field(gt=0.0)
    run_date: str | None = None
    btc_price_col: str | None = None
    force: bool = False


class _DecisionPayload(BaseModel):
    schema_version: str
    status: str
    strategy_id: str
    strategy_version: str
    run_date: str
    decision_key: str
    idempotency_hit: bool
    forced_rerun: bool
    weight_today: float | None
    recommended_notional_usd: float | None
    recommended_quantity_btc: float | None
    reference_price_usd: float | None
    btc_price_col: str
    state_db_path: str
    artifact_path: str | None
    message: str
    validation_receipt_id: int | None = None
    validation_passed: bool | None = None
    data_hash: str
    feature_snapshot_hash: str
    bootstrap: bool


class _ReceiptRequest(BaseModel):
    decision_key: str
    event_id: str
    event_type: Literal[
        "submitted",
        "partially_filled",
        "filled",
        "canceled",
        "rejected",
        "failed",
    ]
    event_time: datetime
    broker_name: str | None = None
    broker_account_ref: str | None = None
    external_order_id: str | None = None
    filled_notional_usd: float | None = Field(default=None, ge=0.0)
    filled_quantity_btc: float | None = Field(default=None, ge=0.0)
    fill_price_usd: float | None = Field(default=None, ge=0.0)
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fill_fields(self) -> "_ReceiptRequest":
        has_notional = self.filled_notional_usd is not None
        has_quantity = self.filled_quantity_btc is not None
        if has_notional != has_quantity:
            raise ValueError(
                "filled_notional_usd and filled_quantity_btc must be provided together."
            )
        if self.fill_price_usd is not None and not has_quantity:
            raise ValueError(
                "fill_price_usd requires filled_notional_usd and filled_quantity_btc."
            )
        return self


class _ExecutionReceiptPayload(BaseModel):
    schema_version: str
    decision_key: str
    event_id: str
    event_type: str
    event_time: str
    broker_name: str | None = None
    broker_account_ref: str | None = None
    external_order_id: str | None = None
    filled_notional_usd: float | None = None
    filled_quantity_btc: float | None = None
    fill_price_usd: float | None = None
    message: str | None = None
    metadata: dict[str, Any]


class _ExecutionStatusPayload(BaseModel):
    schema_version: str
    decision_key: str
    strategy_id: str
    run_date: str
    decision_status: str
    execution_status: str
    reconciliation_status: str
    recommended_notional_usd: float | None = None
    recommended_quantity_btc: float | None = None
    filled_notional_usd: float
    filled_quantity_btc: float
    average_fill_price_usd: float | None = None
    receipt_count: int
    latest_event_type: str | None = None
    latest_event_time: str | None = None
    message: str


class _ExecutionReceiptHistoryPayload(BaseModel):
    schema_version: str
    decision_key: str
    receipts: list[_ExecutionReceiptPayload]


def create_agent_service_app(config: AgentServiceConfig) -> FastAPI:
    """Create the FastAPI agent service."""
    auth_token = os.getenv(config.auth_token_env)
    if not auth_token:
        raise ValueError(
            f"Environment variable '{config.auth_token_env}' must be set before starting "
            "the agent API service."
        )
    registry = load_strategy_registry(config.registry_path)
    security = HTTPBearer(auto_error=False)
    app = FastAPI(
        title="StackSats Agent API",
        version="1.0.0",
        description=(
            "HTTP service for StackSats daily decisions and external execution receipt "
            "reconciliation."
        ),
    )

    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = (request.headers.get("X-Request-ID") or "").strip() or uuid.uuid4().hex
        request.state.request_id = request_id
        client_host = request.client.host if request.client else "-"
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "Unhandled agent API exception request_id=%s method=%s path=%s duration_ms=%.2f client=%s",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
                client_host,
                exc_info=True,
            )
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error.",
                    "request_id": request_id,
                },
            )
        duration_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "Agent API request complete request_id=%s method=%s path=%s status=%s duration_ms=%.2f client=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            client_host,
        )
        return response

    def require_bearer_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> None:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not secrets.compare_digest(credentials.credentials, auth_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/healthz", response_model=_HealthResponse, tags=["system"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "api_version": "v1"}

    @app.get(
        "/.well-known/agent-integration.json",
        response_model=_DiscoveryDocument,
        tags=["system"],
    )
    def discovery_document() -> dict[str, object]:
        return {
            "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
            "service": "stacksats-agent-api",
            "api_version": "v1",
            "auth": {
                "type": "http",
                "scheme": "bearer",
                "header": "Authorization",
            },
            "endpoints": {
                "healthz": "/healthz",
                "openapi": "/openapi.json",
                "decision_create": "/v1/decisions/daily",
                "decision_get": "/v1/decisions/{decision_key}",
                "execution_receipt_create": "/v1/executions/receipts",
                "execution_get": "/v1/executions/{decision_key}",
                "execution_receipts_get": "/v1/executions/{decision_key}/receipts",
            },
            "receipt_event_types": [
                "submitted",
                "partially_filled",
                "filled",
                "canceled",
                "rejected",
                "failed",
            ],
        }

    @app.post(
        "/v1/decisions/daily",
        response_model=_DecisionPayload,
        tags=["decisions"],
        dependencies=[Depends(require_bearer_token)],
    )
    def create_daily_decision(body: _DecisionRequest) -> dict[str, object]:
        entry = _resolve_registered_strategy(registry, body.strategy_id)
        strategy = load_strategy(
            entry.strategy_spec,
            config_path=entry.strategy_config_path,
        )
        result = StrategyRunner().decide_daily(
            strategy,
            DecideDailyConfig(
                run_date=body.run_date,
                total_window_budget_usd=body.total_window_budget_usd,
                state_db_path=config.state_db_path,
                output_dir=config.output_dir,
                force=body.force,
                btc_price_col=(
                    body.btc_price_col
                    or entry.btc_price_col
                    or config.btc_price_col_default
                ),
            ),
        )
        return result.to_json()

    @app.get(
        "/v1/decisions/{decision_key}",
        response_model=_DecisionPayload,
        tags=["decisions"],
        dependencies=[Depends(require_bearer_token)],
    )
    def get_daily_decision(decision_key: str) -> dict[str, object]:
        store = SQLiteExecutionStateStore(config.state_db_path)
        stored = store.get_run_by_run_key(run_key=decision_key, mode="decision")
        if stored is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unknown decision_key.",
            )
        return dict(stored.payload)

    @app.post(
        "/v1/executions/receipts",
        response_model=_ExecutionStatusPayload,
        tags=["executions"],
        dependencies=[Depends(require_bearer_token)],
    )
    def create_execution_receipt(body: _ReceiptRequest) -> dict[str, object]:
        store = SQLiteExecutionStateStore(config.state_db_path)
        try:
            status_result, _ = store.ingest_execution_receipt(
                decision_key=body.decision_key,
                event=_receipt_request_to_event(body),
            )
        except ReceiptConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return status_result.to_json()

    @app.get(
        "/v1/executions/{decision_key}",
        response_model=_ExecutionStatusPayload,
        tags=["executions"],
        dependencies=[Depends(require_bearer_token)],
    )
    def get_execution_status(decision_key: str) -> dict[str, object]:
        store = SQLiteExecutionStateStore(config.state_db_path)
        status_result = store.get_execution_status(decision_key=decision_key)
        if status_result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unknown decision_key.",
            )
        return status_result.to_json()

    @app.get(
        "/v1/executions/{decision_key}/receipts",
        response_model=_ExecutionReceiptHistoryPayload,
        tags=["executions"],
        dependencies=[Depends(require_bearer_token)],
    )
    def get_execution_receipts(decision_key: str) -> dict[str, object]:
        store = SQLiteExecutionStateStore(config.state_db_path)
        history = store.get_execution_receipts(decision_key=decision_key)
        if history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unknown decision_key.",
            )
        return history.to_json()

    return app


def start_agent_service(config: AgentServiceConfig) -> None:
    """Start the agent API service with uvicorn."""
    import_optional("uvicorn", extra="service", feature="agent HTTP service")
    import uvicorn

    logger.info(
        "Starting StackSats agent API host=%s port=%s registry_path=%s state_db_path=%s output_dir=%s auth_token_env=%s",
        config.host,
        config.port,
        config.registry_path,
        config.state_db_path,
        config.output_dir,
        config.auth_token_env,
    )
    app = create_agent_service_app(config)
    uvicorn.run(app, host=config.host, port=int(config.port))


def _resolve_registered_strategy(
    registry: dict[str, RegisteredStrategy],
    strategy_id: str,
) -> RegisteredStrategy:
    entry = registry.get(strategy_id)
    if entry is None or not entry.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown strategy_id.",
        )
    return entry


def _receipt_request_to_event(body: _ReceiptRequest) -> ExecutionReceiptEvent:
    event_time = body.event_time
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    else:
        event_time = event_time.astimezone(timezone.utc)
    return ExecutionReceiptEvent(
        decision_key=body.decision_key,
        event_id=body.event_id,
        event_type=body.event_type,
        event_time=event_time.isoformat().replace("+00:00", "Z"),
        broker_name=body.broker_name,
        broker_account_ref=body.broker_account_ref,
        external_order_id=body.external_order_id,
        filled_notional_usd=body.filled_notional_usd,
        filled_quantity_btc=body.filled_quantity_btc,
        fill_price_usd=body.fill_price_usd,
        message=body.message,
        metadata=body.metadata,
    )


__all__ = ["create_agent_service_app", "start_agent_service"]

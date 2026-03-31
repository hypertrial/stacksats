from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from stacksats.api import ValidationResult
from stacksats.runner import StrategyRunner
import stacksats.service as service_entrypoints
from stacksats.service import app as service_app
from stacksats.service.registry import load_strategy_registry
from stacksats.strategy_types import AgentServiceConfig, BaseStrategy

try:
    from fastapi.testclient import TestClient
except RuntimeError as exc:  # pragma: no cover - depends on local optional deps
    if "httpx package" in str(exc):
        pytest.skip(str(exc), allow_module_level=True)
    raise


class _UniformDailyStrategy(BaseStrategy):
    strategy_id = "service-uniform"
    version = "1.0.0"

    def propose_weight(self, state):
        return state.uniform_weight


def _btc_df() -> pl.DataFrame:
    dates = pl.datetime_range(
        dt.datetime(2023, 1, 1),
        dt.datetime(2023, 1, 1) + dt.timedelta(days=899),
        interval="1d",
        eager=True,
    )
    n = dates.len()
    return pl.DataFrame(
        {
            "date": dates,
            "price_usd": np.linspace(20000.0, 90000.0, n),
            "mvrv": np.linspace(0.8, 2.2, n),
        }
    )


def _allow_validation(runner: StrategyRunner) -> None:
    runner.validate = lambda *args, **kwargs: ValidationResult(  # type: ignore[method-assign]
        passed=True,
        forward_leakage_ok=True,
        weight_constraints_ok=True,
        win_rate=100.0,
        win_rate_ok=True,
        messages=["ok"],
        diagnostics={},
    )


def _deny_validation(runner: StrategyRunner) -> None:
    runner.validate = lambda *args, **kwargs: ValidationResult(  # type: ignore[method-assign]
        passed=False,
        forward_leakage_ok=True,
        weight_constraints_ok=True,
        win_rate=10.0,
        win_rate_ok=False,
        messages=["strict daily decision validation failed"],
        diagnostics={},
    )


def _registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "uniform-service": {
                    "strategy_spec": "ignored.module:IgnoredStrategy",
                    "enabled": True,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _service_config(tmp_path: Path, registry_path: Path) -> AgentServiceConfig:
    return AgentServiceConfig(
        registry_path=str(registry_path),
        state_db_path=str(tmp_path / "state.sqlite3"),
        output_dir=str(tmp_path / "output"),
    )


def test_load_strategy_registry_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "strategy.json"
    config_path.write_text("{}", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "svc": {
                    "strategy_spec": "strategies/demo.py:DemoStrategy",
                    "strategy_config_path": "strategy.json",
                    "enabled": True,
                    "btc_price_col": "close_usd",
                }
            }
        ),
        encoding="utf-8",
    )

    registry = load_strategy_registry(registry_path)

    assert registry["svc"].strategy_spec.endswith("strategies/demo.py:DemoStrategy")
    assert registry["svc"].strategy_config_path == str(config_path.resolve())
    assert registry["svc"].btc_price_col == "close_usd"

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        relative_registry = load_strategy_registry(Path("registry.json"))
    finally:
        os.chdir(cwd)

    assert relative_registry["svc"].strategy_spec.endswith("strategies/demo.py:DemoStrategy")


def test_load_strategy_registry_validation_errors(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_strategy_registry(missing_path)

    invalid_payload_path = tmp_path / "invalid-payload.json"
    invalid_payload_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_strategy_registry(invalid_payload_path)

    invalid_key_path = tmp_path / "invalid-key.json"
    invalid_key_path.write_text(json.dumps({"": {"strategy_spec": "demo"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="keys must be non-empty"):
        load_strategy_registry(invalid_key_path)

    invalid_entry_path = tmp_path / "invalid-entry.json"
    invalid_entry_path.write_text(json.dumps({"svc": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_strategy_registry(invalid_entry_path)

    invalid_spec_path = tmp_path / "invalid-spec.json"
    invalid_spec_path.write_text(json.dumps({"svc": {"enabled": True}}), encoding="utf-8")
    with pytest.raises(ValueError, match="strategy_spec"):
        load_strategy_registry(invalid_spec_path)

    invalid_config_path = tmp_path / "invalid-config.json"
    invalid_config_path.write_text(
        json.dumps({"svc": {"strategy_spec": "demo", "strategy_config_path": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strategy_config_path"):
        load_strategy_registry(invalid_config_path)

    invalid_enabled_path = tmp_path / "invalid-enabled.json"
    invalid_enabled_path.write_text(
        json.dumps({"svc": {"strategy_spec": "demo", "enabled": "yes"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid 'enabled'"):
        load_strategy_registry(invalid_enabled_path)

    invalid_price_col_path = tmp_path / "invalid-price-col.json"
    invalid_price_col_path.write_text(
        json.dumps({"svc": {"strategy_spec": "demo", "btc_price_col": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid 'btc_price_col'"):
        load_strategy_registry(invalid_price_col_path)

    no_colon_path = tmp_path / "no-colon.json"
    no_colon_path.write_text(json.dumps({"svc": {"strategy_spec": "demo"}}), encoding="utf-8")
    registry = load_strategy_registry(no_colon_path)
    assert registry["svc"].strategy_spec == "demo"

    absolute_config_path = tmp_path / "absolute-config.json"
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    absolute_config_path.write_text(
        json.dumps(
            {
                "svc": {
                    "strategy_spec": "demo",
                    "strategy_config_path": str(config_file.resolve()),
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_strategy_registry(absolute_config_path)
    assert registry["svc"].strategy_config_path == str(config_file.resolve())


def test_create_agent_service_app_requires_env_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = _registry_path(tmp_path)
    monkeypatch.delenv("STACKSATS_AGENT_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="STACKSATS_AGENT_API_TOKEN"):
        service_app.create_agent_service_app(_service_config(tmp_path, registry_path))


def test_agent_service_entrypoint_wrappers_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    stub_module = ModuleType("stacksats.service.app")
    stub_module.create_agent_service_app = lambda config: observed.setdefault("create", config)
    stub_module.start_agent_service = lambda config: observed.setdefault("start", config)

    monkeypatch.setitem(sys.modules, "stacksats.service.app", stub_module)
    config = AgentServiceConfig()

    service_entrypoints.create_agent_service_app(config)
    service_entrypoints.start_agent_service(config)

    assert observed["create"] is config
    assert observed["start"] is config


def test_agent_service_receipt_request_validation_and_naive_event_time() -> None:
    with pytest.raises(ValueError, match="provided together"):
        service_app._ReceiptRequest(
            decision_key="decision-1",
            event_id="evt-1",
            event_type="filled",
            event_time=dt.datetime(2025, 1, 1, 10, 0, 0),
            filled_notional_usd=1.0,
        )

    with pytest.raises(ValueError, match="fill_price_usd requires"):
        service_app._ReceiptRequest(
            decision_key="decision-1",
            event_id="evt-1",
            event_type="filled",
            event_time=dt.datetime(2025, 1, 1, 10, 0, 0),
            fill_price_usd=100000.0,
        )

    request = service_app._ReceiptRequest(
        decision_key="decision-1",
        event_id="evt-2",
        event_type="submitted",
        event_time=dt.datetime(2025, 1, 1, 10, 0, 0),
    )
    event = service_app._receipt_request_to_event(request)
    assert event.event_time.endswith("Z")


def test_start_agent_service_uses_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    observed: dict[str, object] = {}
    config = AgentServiceConfig()
    caplog.set_level(logging.INFO, logger=service_app.logger.name)

    monkeypatch.setattr(service_app, "import_optional", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service_app,
        "create_agent_service_app",
        lambda cfg: observed.setdefault("app", cfg),
    )
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(
            run=lambda app, host, port: observed.update(
                {"uvicorn_app": app, "host": host, "port": port}
            )
        ),
    )

    service_app.start_agent_service(config)

    assert observed["app"] is config
    assert observed["uvicorn_app"] is config
    assert observed["host"] == config.host
    assert observed["port"] == config.port
    assert f"host={config.host}" in caplog.text
    assert f"auth_token_env={config.auth_token_env}" in caplog.text


def test_agent_service_requires_valid_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = _registry_path(tmp_path)
    monkeypatch.setenv("STACKSATS_AGENT_API_TOKEN", "top-secret")
    monkeypatch.setattr(service_app, "load_strategy", lambda *args, **kwargs: _UniformDailyStrategy())
    monkeypatch.setattr(
        service_app,
        "StrategyRunner",
        lambda: SimpleNamespace(decide_daily=lambda strategy, config: None),
    )
    client = TestClient(service_app.create_agent_service_app(_service_config(tmp_path, registry_path)))

    missing = client.post(
        "/v1/decisions/daily",
        json={"strategy_id": "uniform-service", "total_window_budget_usd": 1000.0},
    )
    invalid = client.post(
        "/v1/decisions/daily",
        headers={
            "Authorization": "Bearer wrong",
            "X-Request-ID": "req-invalid",
        },
        json={"strategy_id": "uniform-service", "total_window_budget_usd": 1000.0},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert "X-Request-ID" in missing.headers
    assert invalid.headers["X-Request-ID"] == "req-invalid"


def test_agent_service_missing_decision_retrievals_return_404(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = _registry_path(tmp_path)
    monkeypatch.setenv("STACKSATS_AGENT_API_TOKEN", "top-secret")
    monkeypatch.setattr(service_app, "load_strategy", lambda *args, **kwargs: _UniformDailyStrategy())
    monkeypatch.setattr(
        service_app,
        "StrategyRunner",
        lambda: SimpleNamespace(decide_daily=lambda strategy, config: None),
    )
    client = TestClient(service_app.create_agent_service_app(_service_config(tmp_path, registry_path)))
    headers = {"Authorization": "Bearer top-secret"}

    decision = client.get("/v1/decisions/missing", headers=headers)
    execution = client.get("/v1/executions/missing", headers=headers)
    receipts = client.get("/v1/executions/missing/receipts", headers=headers)

    assert decision.status_code == 404
    assert execution.status_code == 404
    assert receipts.status_code == 404


def test_agent_service_decision_endpoints_and_discovery_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry_path = _registry_path(tmp_path)
    caplog.set_level(logging.INFO, logger=service_app.logger.name)
    monkeypatch.setenv("STACKSATS_AGENT_API_TOKEN", "top-secret")
    monkeypatch.setattr(service_app, "load_strategy", lambda *args, **kwargs: _UniformDailyStrategy())
    runner = StrategyRunner()
    _allow_validation(runner)
    monkeypatch.setattr(
        service_app,
        "StrategyRunner",
        lambda: SimpleNamespace(
            decide_daily=lambda strategy, config: runner.decide_daily(
                strategy,
                config,
                btc_df=_btc_df(),
            )
        ),
    )
    client = TestClient(service_app.create_agent_service_app(_service_config(tmp_path, registry_path)))
    headers = {"Authorization": "Bearer top-secret"}

    health = client.get("/healthz")
    discovery = client.get("/.well-known/agent-integration.json")
    openapi = client.get("/openapi.json")
    first = client.post(
        "/v1/decisions/daily",
        headers={
            **headers,
            "X-Request-ID": "req-decide",
        },
        json={
            "strategy_id": "uniform-service",
            "total_window_budget_usd": 1000.0,
            "run_date": "2024-12-31",
        },
    )
    second = client.post(
        "/v1/decisions/daily",
        headers=headers,
        json={
            "strategy_id": "uniform-service",
            "total_window_budget_usd": 1000.0,
            "run_date": "2024-12-31",
        },
    )

    assert health.status_code == 200
    assert "X-Request-ID" in health.headers
    assert len(health.headers["X-Request-ID"]) >= 8
    assert discovery.status_code == 200
    assert discovery.json()["endpoints"]["decision_create"] == "/v1/decisions/daily"
    assert openapi.status_code == 200
    assert first.status_code == 200
    assert first.json()["status"] == "decided"
    assert second.status_code == 200
    assert second.json()["status"] == "noop"

    decision_key = first.json()["decision_key"]
    fetched_decision = client.get(f"/v1/decisions/{decision_key}", headers=headers)
    execution_status = client.get(f"/v1/executions/{decision_key}", headers=headers)

    assert fetched_decision.status_code == 200
    assert fetched_decision.json()["decision_key"] == decision_key
    assert execution_status.status_code == 200
    assert execution_status.json()["execution_status"] == "pending"
    assert execution_status.json()["reconciliation_status"] == "pending"
    assert first.headers["X-Request-ID"] == "req-decide"
    assert "request_id=req-decide" in caplog.text
    assert "path=/v1/decisions/daily" in caplog.text


def test_agent_service_decision_failure_and_unknown_strategy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = _registry_path(tmp_path)
    monkeypatch.setenv("STACKSATS_AGENT_API_TOKEN", "top-secret")
    monkeypatch.setattr(service_app, "load_strategy", lambda *args, **kwargs: _UniformDailyStrategy())
    runner = StrategyRunner()
    _deny_validation(runner)
    monkeypatch.setattr(
        service_app,
        "StrategyRunner",
        lambda: SimpleNamespace(
            decide_daily=lambda strategy, config: runner.decide_daily(
                strategy,
                config,
                btc_df=_btc_df(),
            )
        ),
    )
    client = TestClient(service_app.create_agent_service_app(_service_config(tmp_path, registry_path)))
    headers = {"Authorization": "Bearer top-secret"}

    failed = client.post(
        "/v1/decisions/daily",
        headers=headers,
        json={
            "strategy_id": "uniform-service",
            "total_window_budget_usd": 1000.0,
            "run_date": "2024-12-31",
        },
    )
    unknown = client.post(
        "/v1/decisions/daily",
        headers=headers,
        json={"strategy_id": "missing", "total_window_budget_usd": 1000.0},
    )

    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    assert "daily decision" in failed.json()["message"].lower()
    fetched_failed = client.get(
        f"/v1/decisions/{failed.json()['decision_key']}",
        headers=headers,
    )
    assert fetched_failed.status_code == 200
    assert fetched_failed.json()["status"] == "failed"
    assert unknown.status_code == 404


def test_agent_service_receipt_ingestion_and_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = _registry_path(tmp_path)
    monkeypatch.setenv("STACKSATS_AGENT_API_TOKEN", "top-secret")
    monkeypatch.setattr(service_app, "load_strategy", lambda *args, **kwargs: _UniformDailyStrategy())
    runner = StrategyRunner()
    _allow_validation(runner)
    monkeypatch.setattr(
        service_app,
        "StrategyRunner",
        lambda: SimpleNamespace(
            decide_daily=lambda strategy, config: runner.decide_daily(
                strategy,
                config,
                btc_df=_btc_df(),
            )
        ),
    )
    client = TestClient(service_app.create_agent_service_app(_service_config(tmp_path, registry_path)))
    headers = {"Authorization": "Bearer top-secret"}

    decision = client.post(
        "/v1/decisions/daily",
        headers=headers,
        json={
            "strategy_id": "uniform-service",
            "total_window_budget_usd": 1000.0,
            "run_date": "2024-12-31",
        },
    )
    decision_key = decision.json()["decision_key"]
    half_notional = float(decision.json()["recommended_notional_usd"]) / 2.0
    half_quantity = float(decision.json()["recommended_quantity_btc"]) / 2.0
    reference_price = float(decision.json()["reference_price_usd"])
    submitted = client.post(
        "/v1/executions/receipts",
        headers=headers,
        json={
            "decision_key": decision_key,
            "event_id": "evt-1",
            "event_type": "submitted",
            "event_time": "2025-01-01T10:00:00Z",
            "external_order_id": "ord-1",
        },
    )
    partial = client.post(
        "/v1/executions/receipts",
        headers=headers,
        json={
            "decision_key": decision_key,
            "event_id": "evt-2",
            "event_type": "partially_filled",
            "event_time": "2025-01-01T10:01:00Z",
            "external_order_id": "ord-1",
            "filled_notional_usd": half_notional,
            "filled_quantity_btc": half_quantity,
            "fill_price_usd": reference_price,
        },
    )
    matched = client.post(
        "/v1/executions/receipts",
        headers=headers,
        json={
            "decision_key": decision_key,
            "event_id": "evt-3",
            "event_type": "filled",
            "event_time": "2025-01-01T10:02:00Z",
            "external_order_id": "ord-1",
            "filled_notional_usd": half_notional,
            "filled_quantity_btc": half_quantity,
            "fill_price_usd": reference_price,
        },
    )
    duplicate = client.post(
        "/v1/executions/receipts",
        headers=headers,
        json={
            "decision_key": decision_key,
            "event_id": "evt-3",
            "event_type": "filled",
            "event_time": "2025-01-01T10:02:00Z",
            "external_order_id": "ord-1",
            "filled_notional_usd": half_notional,
            "filled_quantity_btc": half_quantity,
            "fill_price_usd": reference_price,
        },
    )
    conflict = client.post(
        "/v1/executions/receipts",
        headers=headers,
        json={
            "decision_key": decision_key,
            "event_id": "evt-3",
            "event_type": "failed",
            "event_time": "2025-01-01T10:03:00Z",
        },
    )
    missing = client.post(
        "/v1/executions/receipts",
        headers=headers,
        json={
            "decision_key": "missing-decision",
            "event_id": "evt-x",
            "event_type": "submitted",
            "event_time": "2025-01-01T10:00:00Z",
        },
    )
    execution = client.get(f"/v1/executions/{decision_key}", headers=headers)
    receipts = client.get(f"/v1/executions/{decision_key}/receipts", headers=headers)

    assert submitted.status_code == 200
    assert partial.status_code == 200
    assert partial.json()["execution_status"] == "partially_filled"
    assert matched.status_code == 200
    assert matched.json()["reconciliation_status"] == "matched"
    assert duplicate.status_code == 200
    assert duplicate.json()["receipt_count"] == 3
    assert conflict.status_code == 409
    assert missing.status_code == 404
    assert execution.status_code == 200
    assert execution.json()["execution_status"] == "filled"
    assert execution.json()["receipt_count"] == 3
    assert receipts.status_code == 200
    assert len(receipts.json()["receipts"]) == 3


def test_agent_service_preserves_request_ids_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = _registry_path(tmp_path)
    monkeypatch.setenv("STACKSATS_AGENT_API_TOKEN", "top-secret")
    monkeypatch.setattr(service_app, "load_strategy", lambda *args, **kwargs: _UniformDailyStrategy())
    monkeypatch.setattr(
        service_app,
        "StrategyRunner",
        lambda: SimpleNamespace(decide_daily=lambda strategy, config: None),
    )
    client = TestClient(service_app.create_agent_service_app(_service_config(tmp_path, registry_path)))

    response = client.get("/healthz", headers={"X-Request-ID": "req-health"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-health"


def test_agent_service_logs_and_returns_request_id_for_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry_path = _registry_path(tmp_path)
    caplog.set_level(logging.INFO, logger=service_app.logger.name)
    monkeypatch.setenv("STACKSATS_AGENT_API_TOKEN", "top-secret")
    monkeypatch.setattr(service_app, "load_strategy", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    client = TestClient(
        service_app.create_agent_service_app(_service_config(tmp_path, registry_path)),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/v1/decisions/daily",
        headers={
            "Authorization": "Bearer top-secret",
            "X-Request-ID": "req-boom",
        },
        json={
            "strategy_id": "uniform-service",
            "total_window_budget_usd": 1000.0,
        },
    )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-boom"
    assert response.json() == {
        "detail": "Internal server error.",
        "request_id": "req-boom",
    }
    assert "Unhandled agent API exception request_id=req-boom" in caplog.text
    assert "Agent API request complete request_id=req-boom" in caplog.text

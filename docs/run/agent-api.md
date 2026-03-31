---
title: Agent API Service
description: Reference for `stacksats serve agent-api`.
---

# Agent API Service

Use this command when StackSats should expose the agent-native daily decision flow over a hosted HTTP service.

## Security

- Inject bearer tokens via environment variables or a secret manager; never commit tokens or log them in plaintext.
- Rotate bearer tokens by updating the secret behind `STACKSATS_AGENT_API_TOKEN` (or the env name passed to `--auth-token-env`), then restart or roll the service so it reloads the new value.
- Report vulnerabilities through the [GitHub security policy](https://github.com/hypertrial/stacksats/security/policy) (see the repository `SECURITY.md`).

## Prerequisites

- Install the service extra: `pip install "stacksats[service]"`.
- Set a bearer token in an environment variable such as `STACKSATS_AGENT_API_TOKEN`.
- Create a strategy registry JSON file for the strategies the service is allowed to run.

## Command

```bash
export STACKSATS_AGENT_API_TOKEN=replace-me

stacksats serve agent-api \
  --registry-path .stacksats/agent_service_registry.json
```

## Strategy registry

```json
{
  "btc-dca-paper": {
    "strategy_spec": "stacksats.strategies.examples:RunDailyPaperStrategy",
    "enabled": true,
    "btc_price_col": "price_usd"
  }
}
```

## Example requests

```bash
curl -sS http://127.0.0.1:8000/.well-known/agent-integration.json

curl -sS http://127.0.0.1:8000/v1/decisions/daily \
  -H "Authorization: Bearer $STACKSATS_AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"strategy_id":"btc-dca-paper","total_window_budget_usd":1000}'

curl -sS http://127.0.0.1:8000/v1/executions/receipts \
  -H "Authorization: Bearer $STACKSATS_AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"decision_key":"<decision_key>","event_id":"evt-1","event_type":"submitted","event_time":"2025-01-01T10:00:00Z"}'
```

## Stable endpoints

- `GET /healthz`
- `GET /.well-known/agent-integration.json`
- `POST /v1/decisions/daily`
- `GET /v1/decisions/{decision_key}`
- `POST /v1/executions/receipts`
- `GET /v1/executions/{decision_key}`
- `GET /v1/executions/{decision_key}/receipts`

## Key options

- `--host <addr>`: bind host (default `127.0.0.1`).
- `--port <port>`: bind port (default `8000`).
- `--registry-path <path>`: strategy registry JSON (default `.stacksats/agent_service_registry.json`).
- `--state-db-path <path>`: decision and receipt state DB (default `.stacksats/run_state.sqlite3`).
- `--output-dir <dir>`: decision artifact root (default `output`).
- `--auth-token-env <name>`: env var containing the bearer token (default `STACKSATS_AGENT_API_TOKEN`).

## Reconciliation behavior

- Receipt ingestion is append-only and keyed by `decision_key` plus `event_id`.
- Duplicate identical receipts are idempotent; duplicate conflicting receipts return HTTP `409`.
- Reconciliation compares ingested cumulative fills to the original StackSats recommendation.
- StackSats does not poll broker state or store broker credentials in hosted mode.

## Operations

- Every response includes `X-Request-ID`. If the client sends a non-empty `X-Request-ID`, StackSats preserves it; otherwise the service generates one.
- The hosted service logs one `INFO` line per request with request ID, method, path, status, duration, and client host when available.
- Unhandled server exceptions are logged at `ERROR` with the request ID and traceback, and the client receives a stable `500` response body with `detail` plus `request_id`.
- Token handling and rotation: see **Security** above.

## Next step

- Use [Decide Daily Command](decide-daily.md) for the same flow without HTTP.
- Use [Run Daily Command](run-daily.md) only when you want StackSats to submit orders itself.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Agent+API+Service)

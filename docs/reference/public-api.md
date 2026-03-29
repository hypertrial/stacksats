---
title: Public API
description: Stable StackSats 1.x imports, CLI subset, and artifact contracts.
---

# Public API

Use this page when you want the supported `1.x` library surface rather than the broader internal/generated module reference.

## Stable `1.x` contract

The stable public contract is intentionally narrow:

- top-level `stacksats` exports
- documented artifact payloads emitted by stable CLI flows
- documented CLI commands:
  - `stacksats demo validate|backtest|export`
  - `stacksats data fetch|prepare|doctor`
  - `stacksats strategy validate|backtest|export|decide-daily|run-daily|animate`
  - `stacksats serve agent-api`
- documented hosted HTTP service:
  - `POST /v1/decisions/daily`
  - `GET /v1/decisions/{decision_key}`
  - `POST /v1/executions/receipts`
  - `GET /v1/executions/{decision_key}`
  - `GET /v1/executions/{decision_key}/receipts`

Helper scripts such as `stacksats-plot-mvrv` and `stacksats-plot-weights` are documented convenience tools outside the frozen stable CLI subset.

## Stable top-level imports

Common stable imports include:

- runtime objects: `FeatureTimeSeries`, `WeightTimeSeries`, `WeightTimeSeriesBatch`
- strategy contract types: `BaseStrategy`, `StrategyContext`, `DayState`, `TargetProfile`
- configs and results: `AgentServiceConfig`, `BacktestConfig`, `ValidationConfig`, `ExportConfig`, `DecideDailyConfig`, `RunDailyConfig`, `BacktestResult`, `ValidationResult`, `DailyDecisionResult`, `DailyRunResult`, `ExecutionReceiptEvent`, `ExecutionReceiptHistoryResult`, `ExecutionStatusResult`
- metadata and schema types: `StrategyMetadata`, `StrategySpec`, `StrategySeriesMetadata`, `StrategyArtifactSet`, `ColumnSpec`
- runners and loaders: `StrategyRunner`, `create_agent_service_app`, `load_strategy`, `load_data`, `open_merged_metrics`, `load_metric_catalog`, `precompute_features`
- stable built-ins: `UniformStrategy`, `RunDailyPaperStrategy`, `SimpleZScoreStrategy`, `MomentumStrategy`, `MVRVStrategy`

See [Stability Policy](../stability.md) for the canonical support and deprecation rules.

## Most common imports

Run a strategy through `StrategyRunner`:

```python
from stacksats import BacktestConfig, StrategyRunner, UniformStrategy

runner = StrategyRunner()
result = runner.backtest(
    UniformStrategy(),
    BacktestConfig(start_date="2024-01-01", end_date="2024-12-31"),
)
print(result.summary())
```

Load canonical data for the stable runtime path:

```python
from stacksats import load_data, open_merged_metrics

btc_df = load_data()
dataset = open_merged_metrics()
print(btc_df.columns)
print(dataset.summary())
```

Generate an agent-facing daily decision payload:

```python
from stacksats import DecideDailyConfig, RunDailyPaperStrategy

result = RunDailyPaperStrategy().decide_daily(
    DecideDailyConfig(total_window_budget_usd=1000.0)
)
print(result.to_json())
```

Embed the hosted agent API in your own process:

```python
from stacksats import AgentServiceConfig, create_agent_service_app

app = create_agent_service_app(
    AgentServiceConfig(registry_path=".stacksats/agent_service_registry.json")
)
```

Reload exported artifacts:

```python
from stacksats import WeightTimeSeriesBatch

batch = WeightTimeSeriesBatch.from_artifact_dir(
    "output/simple-zscore/<version>/<run_id>"
)
print(batch.window_count)
print(batch.to_dataframe().columns)
```

## Stable artifact payloads

The documented stable JSON payloads are:

- `backtest_result.json`
- `decision_result.json`
- `metrics.json`
- `animation_manifest.json`
- `artifacts.json`

These payloads carry `schema_version` and are part of the stable `1.x` artifact contract documented across the runtime and model-backtest docs.

## Stable vs internal vs experimental

| Surface | Status | Notes |
|---|---|---|
| top-level `stacksats` exports | stable | covered by SemVer and deprecation policy |
| documented CLI subset | stable | `demo`, `data`, `strategy`, and `serve agent-api` |
| documented hosted HTTP service | stable | versioned `/v1` agent decision + receipt endpoints |
| documented artifact payloads | stable | frozen `1.x` JSON contract |
| generated module pages under API Reference | internal | useful for reading internals; not stable by default |
| lower-level modules such as `stacksats.runner` or `stacksats.strategy_types` | internal | may change even when still documented |
| `stacksats.strategies.experimental.*` | experimental | outside the `1.x` compatibility promise |
| helper scripts such as `stacksats-plot-mvrv` and `stacksats-plot-weights` | documented but outside stable CLI | useful tools, but not part of the frozen CLI subset |

## Where to go next

- [Stability Policy](../stability.md)
- [Runtime Objects Overview](../objects.md)
- [Strategies](strategies.md)
- [Command Index](../commands.md)
- [API Reference](api/index.md) for internal/generated module details

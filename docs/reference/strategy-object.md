---
title: Strategy Object
description: Contract for BaseStrategy identity and user hook responsibilities.
---

# Strategy Object

A strategy subclasses `BaseStrategy` (`stacksats/strategy_types.py`) and defines:

- identity: `strategy_id`, `version`, `description`
- hooks: `transform_features`, `build_signals`
- intent path: `propose_weight(...)` or `build_target_profile(...)`

## Ownership model

!!! info "Framework vs user"
    User code owns features/signals/intent. Framework code owns iteration, clipping, and invariants.

Framework-owned behavior remains sealed:

- compute kernel (`compute_weights` in `BaseStrategy`)
- clipping and remaining-budget enforcement
- lock semantics and final invariants

## Minimal strategy shape

```python
class MyStrategy(BaseStrategy):
    strategy_id = "my-strategy"
    version = "1.0.0"

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        ...

    def build_signals(self, ctx: StrategyContext, features_df: pd.DataFrame) -> dict[str, pd.Series]:
        ...

    def build_target_profile(self, ctx: StrategyContext, features_df: pd.DataFrame, signals: dict[str, pd.Series]) -> TargetProfile:
        ...
```

## Related docs

- [Framework Boundary](../framework.md)
- [Minimal Strategy Examples](../start/minimal-strategy-examples.md)
- [Strategy TimeSeries](strategy-timeseries.md)
- [FAQ](../faq.md)

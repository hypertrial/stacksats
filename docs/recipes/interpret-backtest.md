---
title: Recipe - Interpret Backtest Metrics
description: Task recipe for interpreting StackSats backtest results and artifacts.
---

# Recipe: Interpret Backtest Metrics

## Goal

Read backtest outputs consistently when comparing strategy changes.

## Generate a run

Use the canonical backtest command as reference:

- [Backtest Command](../run/backtest.md)

Example:

```bash
stacksats strategy backtest \
  --strategy my_strategy.py:MyStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

## Key metrics to inspect

- `win_rate`: windows where dynamic beats uniform.
- `exp_decay_percentile`: recency-weighted percentile.
- `score`: blended ranking metric.

## Artifact checklist

- `backtest_result.json`
- `metrics.json`
- plot `.svg` files

## Comparison rules

- Keep date range and allocation span identical.
- Compare against same baseline assumptions.
- Use strict validation outputs to flag overfitting risks.

## Related docs

- [Backtest Runtime](../model_backtest.md)
- [CLI Commands](../commands.md)

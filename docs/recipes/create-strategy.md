---
title: Recipe - Create a Strategy
description: Task-focused recipe for creating and running a custom StackSats strategy.
---

# Recipe: Create a Strategy

## Goal

Create a custom strategy file and run it end-to-end.

## Steps

1. Create `my_strategy.py` with a `BaseStrategy` subclass.
2. Implement `transform_features`, `build_signals`, and one intent hook.
3. Validate and backtest with CLI.

```bash
stacksats strategy validate --strategy my_strategy.py:MyStrategy --strict
stacksats strategy backtest --strategy my_strategy.py:MyStrategy --output-dir output
```

## Expected output

- Validation summary line with pass/fail status.
- Backtest artifacts in `output/<strategy_id>/<version>/<run_id>/`.

## Common failure patterns

- Non-finite signal values (`NaN`/`inf`).
- Incorrect index alignment across feature and signal series.
- Strategy bypass attempts of framework-owned kernel behavior.

---
title: Notebook Demo
description: Lightweight interactive workflow using maintained StackSats CLI and strategy examples.
---

# Notebook Demo

Use this page for a lightweight interactive workflow without exported notebook assets.

It demonstrates:

- package installation and environment setup
- running the packaged strategy entry point
- validating, backtesting, and exporting with CLI commands
- inspecting generated artifacts under `output/<strategy_id>/<version>/<run_id>/`

## Suggested interactive flow

1. Follow [Quickstart](quickstart.md) to install and verify environment.
2. Run the packaged example strategy:

```bash
python -m stacksats.strategies.model_example
```

3. Run the lifecycle commands from [CLI Commands](../commands.md):
   - `stacksats strategy validate ...`
   - `stacksats strategy backtest ...`
   - `stacksats strategy export ...`

## Why no hosted notebook export?

Notebook exports were removed from the maintained docs flow to avoid stale generated assets and outdated paths.

## Next step

- Use [First Strategy Run](first-strategy-run.md) to implement your own strategy class.

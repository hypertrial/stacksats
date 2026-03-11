---
title: Notebook Demo
description: Lightweight interactive workflow using maintained StackSats CLI and strategy examples.
---

# Notebook Demo

Use this page for a lightweight interactive workflow without exported notebook assets.

It demonstrates:

- package installation and environment setup
- running a packaged example strategy via CLI
- validating, backtesting, and exporting with CLI commands
- inspecting generated artifacts under `output/<strategy_id>/<version>/<run_id>/`

## Suggested interactive flow

1. Follow [Quickstart](quickstart.md) to install and verify environment.
2. Run a packaged example strategy:

```bash
stacksats strategy backtest \
  --strategy stacksats.strategies.examples:SimpleZScoreStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

3. Run the lifecycle commands from [Command Index](../commands.md):
   - `stacksats strategy validate ...`
   - `stacksats strategy backtest ...`
   - `stacksats strategy export ...`

## Why no hosted notebook export?

Notebook exports were removed from the maintained docs flow to avoid stale generated assets and outdated paths.

## Next step

- Use [First Strategy Run](first-strategy-run.md) to implement your own strategy class.

---
title: Notebook Demo
description: Lightweight interactive workflow using maintained StackSats CLI and strategy examples.
---

# Notebook Demo

Use this page for a lightweight interactive workflow without exported notebook assets.

It demonstrates:

- package installation and environment setup
- running a packaged example strategy via CLI
- iterating on a local custom strategy with the repo research helper
- validating, backtesting, and exporting with CLI commands
- inspecting generated artifacts under `output/<strategy_id>/<version>/<run_id>/`

## Suggested interactive flow

1. Follow [Quickstart](quickstart.md) to install and verify environment.
2. Run a packaged example strategy:

```bash
stacksats strategy backtest \
  --strategy simple-zscore \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-dir output
```

   Built-in strategy catalog: [Strategies](../reference/strategies.md).

3. For local custom model work, use the research helper:

```bash
python scripts/research_strategy.py \
  --strategy my_strategy.py:MyStrategy \
  --strategy-config examples/strategy_configs/first_strategy_run.example.json \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

4. Run the lifecycle commands from [Command Index](../commands.md):
   - `stacksats strategy validate ...`
   - `stacksats strategy backtest ...`
   - `stacksats strategy export ...`

## Why no hosted notebook export?

Notebook exports were removed from the maintained docs flow to avoid stale generated assets and outdated paths.

## Next step

- Use [First Strategy Run](first-strategy-run.md) to implement your own strategy class.

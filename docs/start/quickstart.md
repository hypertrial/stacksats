---
title: Quickstart
description: Five-minute setup and first execution path for StackSats.
---

# Quickstart

Use this page for a 5-minute first run.

!!! tip "Recommended Path"
    Start with editable install so local examples and docs stay in sync with your checkout.

## 1) Install

=== "Editable (recommended)"

    ```bash
    pip install -e .
    pip install -r requirements-dev.txt
    ```

=== "Package only"

    ```bash
    pip install stacksats
    ```

## 2) Run the example strategy

```bash
python examples/model_example.py
```

This runs validation and backtest, then writes artifacts under:

```text
output/<strategy_id>/<version>/<run_id>/
```

## 3) Use the Strategy Lifecycle CLI

Run lifecycle commands from the canonical reference:

- [Validate Strategy](../commandsmd#2-validate-strategy-via-strategy-lifecycle-cli)
- [Run Full Backtest](../commandsmd#3-run-full-backtest-via-strategy-lifecycle-cli)
- [Export Strategy Artifacts](../commandsmd#4-export-strategy-artifacts)

## 4) Inspect outputs

Primary run artifacts are written under:

```text
output/<strategy_id>/<version>/<run_id>/
```

## 5) What to read next

- [Notebook Demo](notebook-demo.md)
- [First Strategy Run](first-strategy-run.md)
- [Framework Boundary](../framework.md)
- [CLI Commands](../commands.md)

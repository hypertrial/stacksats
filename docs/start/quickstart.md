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

This runs validation and backtest, then writes artifacts to `output/`.

## 3) Use the Strategy Lifecycle CLI

```bash
stacksats strategy validate --strategy examples/model_example.py:ExampleMVRVStrategy
stacksats strategy backtest --strategy examples/model_example.py:ExampleMVRVStrategy --output-dir output
stacksats strategy export --strategy examples/model_example.py:ExampleMVRVStrategy --output-dir output
```

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

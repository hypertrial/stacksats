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
python -m stacksats.strategies.model_example
```

This runs validation and backtest, then writes artifacts under:

```text
output/<strategy_id>/<version>/<run_id>/
```

## 3) Use the Strategy Lifecycle CLI

Run lifecycle commands from the canonical reference:

- [Validate Strategy](../commands.md#2-validate-strategy-via-strategy-lifecycle-cli)
- [Run Full Backtest](../commands.md#3-run-full-backtest-via-strategy-lifecycle-cli)
- [Export Strategy Artifacts](../commands.md#4-export-strategy-artifacts)

## 4) Inspect outputs

Primary run artifacts are written under:

```text
output/<strategy_id>/<version>/<run_id>/
```

Typical files:

- `backtest_result.json`
- `metrics.json`
- plot `.svg` files

## 5) What to read next

- [Task Hub](../tasks.md)
- [Notebook Demo](notebook-demo.md)
- [First Strategy Run](first-strategy-run.md)
- [Minimal Strategy Examples](minimal-strategy-examples.md)
- [Framework Boundary](../framework.md)
- [CLI Commands](../commands.md)
- [Migration Guide](../migration.md)
- [FAQ](../faq.md)

## Success Criteria

A successful quickstart run should produce all of the following:

- CLI command exits without traceback.
- Validation summary is printed.
- Backtest artifacts are written under one run directory.

## Troubleshooting

- If command import fails, confirm editable install from repo root.
- If dates or outputs look wrong, run explicit lifecycle commands from [CLI Commands](../commands.md).
- If upgrading and old helper names fail, use [Migration Guide](../migration.md).
- If you need minimal copy-paste templates, use [Minimal Strategy Examples](minimal-strategy-examples.md).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Quickstart)

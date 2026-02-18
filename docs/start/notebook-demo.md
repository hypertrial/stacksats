---
title: Notebook Demo
description: Interactive marimo notebook demo for StackSats model strategy logic, metrics, and charts.
---

# Notebook Demo

Use this marimo notebook demo to see a StackSats model strategy workflow end-to-end.

It demonstrates:

- package installation and environment setup
- loading market data via the framework
- strategy signal and dynamic DCA policy logic
- backtest summary metrics and visualizations using the package API

## Open notebook views

- [Open exported notebook](https://hypertrial.github.io/stacksats/assets/notebooks/model_example_notebook.html)
- [View notebook source (`examples/model_example_notebook.py`)](https://github.com/hypertrial/stacksats/tree/main/examples/model_example_notebook.py)

## Run locally

From repository root in your active virtualenv:

```bash
marimo edit examples/model_example_notebook.py
```

Then run all cells from top to bottom.

## Regenerate notebook export

When the notebook source changes:

```bash
bash scripts/export_notebook_demo.sh
```

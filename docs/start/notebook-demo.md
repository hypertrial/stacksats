---
title: Notebook Demo
description: Embedded browser-safe marimo notebook demo for strategy logic, metrics, and charts.
---

# Notebook Demo

Use this embedded browser-safe marimo notebook to quickly see a StackSats-style strategy workflow end-to-end.

It demonstrates:

- in-memory market data generation
- strategy signal and dynamic DCA policy logic
- backtest summary metrics and visualizations

<div class="notebook-embed">
  <iframe
    title="StackSats browser-safe marimo notebook demo"
    src="../../assets/notebooks/model_example_notebook_browser.html"
    loading="lazy"
    referrerpolicy="no-referrer"
  ></iframe>
</div>

## Open notebook views

- [Open browser-safe exported notebook](../../assets/notebooks/model_example_notebook_browser.html)
- [View browser-safe notebook source (`examples/model_example_notebook_browser.py`)](https://github.com/hypertrial/stacksats/blob/main/examples/model_example_notebook_browser.py)
- [Open full local exported notebook](../../assets/notebooks/model_example_notebook.html)
- [View full local notebook source (`examples/model_example_notebook.py`)](https://github.com/hypertrial/stacksats/blob/main/examples/model_example_notebook.py)

## Run locally

From repository root in your active virtualenv:

```bash
marimo edit examples/model_example_notebook_browser.py
```

For the full package + CLI workflow notebook:

```bash
marimo edit examples/model_example_notebook.py
```

Then run all cells from top to bottom.

## Regenerate embedded export

When notebook source changes:

```bash
bash scripts/export_notebook_demo.sh
```

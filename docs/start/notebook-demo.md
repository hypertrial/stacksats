---
title: Notebook Demo
description: Embedded marimo notebook preview demonstrating the StackSats model example workflow.
---

# Notebook Demo

Use this embedded marimo notebook to quickly see the StackSats workflow end-to-end.

It demonstrates:

- environment/dependency checks in-notebook
- running a strategy backtest via the StackSats CLI
- generated artifact location and summary output

<div class="notebook-embed">
  <iframe
    title="StackSats marimo notebook demo"
    src="../../assets/notebooks/model_example_notebook.html"
    loading="lazy"
    referrerpolicy="no-referrer"
  ></iframe>
</div>

## Open full notebook view

- [Open the exported notebook directly](https://hypertrial.github.io/stacksats/assets/notebooks/model_example_notebook.html)
- [View notebook source (`examples/model_example_notebook.py`)](https://github.com/hypertrial/stacksats/blob/main/examples/model_example_notebook.py)

## Run it locally

From repository root in your active virtualenv:

```bash
marimo edit examples/model_example_notebook.py
```

Then run all cells from top to bottom.

## Regenerate embedded export

When notebook source changes:

```bash
bash scripts/export_notebook_demo.sh
```

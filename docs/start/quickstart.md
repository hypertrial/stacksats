---
title: Quickstart
description: Install StackSats and run the canonical demo—Python library for quantitative Bitcoin DCA accumulation.
---

# Quickstart

StackSats is a Python library for quantitative Bitcoin dollar-cost averaging (DCA) accumulation. Use this page for the canonical first-run path as a package user.

The core problem is whether a dynamic DCA model can robustly acquire more BTC than uniform DCA for the same fixed budget and allocation horizon. For the full framing, start from [The Stacking Sats Problem](../index.md#the-stacking-sats-problem).

## 1) Choose your install mode

| Use case | Install mode | Command |
|---|---|---|
| I want to use StackSats | package install | `pip install stacksats` |
| I am working from a checkout | editable install | `python -m pip install -c requirements/constraints-maintainer.txt -e ".[dev,all]"` |

Recommended first run:

```bash
pip install stacksats
```

If you are working from this repository checkout instead:

```bash
python -m venv venv
source venv/bin/activate
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -c requirements/constraints-maintainer.txt -e ".[dev,all]"
```

For plotting or animation commands later, install visual extras:

```bash
pip install "stacksats[viz]"
```

## 2) Run the packaged demo

```bash
stacksats demo backtest
```

Optional demo lifecycle commands:

```bash
stacksats demo validate
stacksats demo export
```

This runs a packaged example through the canonical lifecycle and writes artifacts under:

```text
output/<strategy_id>/<version>/<run_id>/
```

## 3) Use the full strategy lifecycle CLI

After the demo succeeds, move to the full `strategy` and `data` command families.

Use these pages next:

- [Validate Strategy](../run/validate.md)
- [Run Full Backtest](../run/backtest.md)
- [Export Strategy Artifacts](../run/export.md)
- [Animate Backtest Output](../run/animate.md)
- [Data Command](../run/data.md)

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
- [Full Data Setup](full-data-setup.md)
- [Notebook Demo](notebook-demo.md)
- [First Strategy Run](first-strategy-run.md)
- [Minimal Strategy Examples](minimal-strategy-examples.md)
- [Strategies](../reference/strategies.md)
- [Framework Boundary](../framework.md)
- [CLI Commands](../commands.md)
- [Migration Guide](../migration.md)
- [FAQ](../faq.md)

## Success Criteria

A successful quickstart run should produce all of the following:

- CLI command exits without traceback.
- Backtest summary is printed.
- Backtest artifacts are written under one run directory.

## Troubleshooting

- If command import fails, confirm you installed either the package or the editable checkout from repo root.
- If you want the canonical BRK workflow next, use [Full Data Setup](full-data-setup.md).
- If dates or outputs look wrong, run explicit lifecycle commands from [CLI Commands](../commands.md).
- If upgrading and old helper names fail, use [Migration Guide](../migration.md).
- If you need minimal copy-paste templates, use [Minimal Strategy Examples](minimal-strategy-examples.md).

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Quickstart)

# Commands for `examples/model_example.py`

This file explains how to run checks, backtests, exports, and deployment using the standalone strategy file:

- `examples/model_example.py`
- strategy class: `ExampleMVRVStrategy`

## Prerequisites

From the repo root:

```bash
pip install -e .
```

Optional dependencies:

```bash
# For local development tools
pip install -r requirements-dev.txt

# For export + Modal deployment
pip install -e ".[deploy]"
```

## Strategy Spec Format

CLI commands that load a strategy use:

```text
module_or_path:ClassName
```

For this example file:

```text
examples/model_example.py:ExampleMVRVStrategy
```

## 1) Quick Run (inside `examples/model_example.py`)

Run the file directly:

```bash
python examples/model_example.py
```

With custom options:

```bash
python examples/model_example.py \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --output-dir output \
  --strategy-label example-mvrv-strategy
```

What this does:
- Runs `validate_strategy(...)`
- Runs `run_backtest(...)`
- Writes plots + JSON output to `output/`

## 2) Validate Strategy via Package CLI

Check whether the model passes package validation gates:

```bash
stacksats-validate --strategy examples/model_example.py:ExampleMVRVStrategy
```

Common options:

```bash
stacksats-validate \
  --strategy examples/model_example.py:ExampleMVRVStrategy \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --min-win-rate 50.0
```

## 3) Run Full Backtest via Package CLI

Basic:

```bash
stacksats-backtest --strategy examples/model_example.py:ExampleMVRVStrategy
```

With options:

```bash
stacksats-backtest \
  --strategy examples/model_example.py:ExampleMVRVStrategy \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --output-dir output \
  --strategy-label model-example
```

## 4) Export Weights to Database

Requires:
- `DATABASE_URL` set
- deploy extras installed

```bash
export DATABASE_URL="postgresql://..."
stacksats-export --strategy examples/model_example.py:ExampleMVRVStrategy
```

## 5) Deploy to Modal

Use `STACKSATS_STRATEGY` to tell Modal which strategy class to load:

```bash
export STACKSATS_STRATEGY="examples/model_example.py:ExampleMVRVStrategy"
modal deploy stacksats/modal_app.py
```

## 6) Useful Development Commands

Verify this document's example commands end-to-end:

```bash
# Requires ./venv (for example: python -m venv venv && source venv/bin/activate && pip install -e .)
python scripts/test_example_commands.py
```

Run tests:

```bash
pytest tests/ -v
```

Run lint:

```bash
ruff check .
```

## Troubleshooting

- **`Invalid strategy spec`**  
  Ensure format is exactly `module_or_path:ClassName`.

- **`Class 'ExampleMVRVStrategy' not found`**  
  Check class name spelling and file path.

- **`Strategy file not found`**  
  Run from repo root or pass an absolute file path.

- **Modal deploy cannot import custom code**  
  Keep strategy self-contained in one file, or package your custom modules and use module-path strategy specs.

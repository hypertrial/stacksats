#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p docs/assets/notebooks

# marimo writes logs under HOME; force a writable HOME for CI and sandboxes.
export HOME="${MARIMO_HOME:-/tmp}"

marimo export html \
  examples/model_example_notebook.py \
  -o docs/assets/notebooks/model_example_notebook.html

marimo export html \
  examples/model_example_notebook_browser.py \
  -o docs/assets/notebooks/model_example_notebook_browser.html

echo "Exported docs/assets/notebooks/model_example_notebook.html"
echo "Exported docs/assets/notebooks/model_example_notebook_browser.html"

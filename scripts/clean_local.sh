#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[clean_local] Removing generated local artifacts..."

rm -rf \
  .coverage \
  .coverage.* \
  coverage.xml \
  dist \
  build \
  site \
  output \
  .pytest_cache \
  .ruff_cache \
  .hypothesis \
  .benchmarks

find . -type d -name "__pycache__" -prune -exec rm -rf {} +

echo "[clean_local] Done."

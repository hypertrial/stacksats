#!/usr/bin/env bash

set -euo pipefail

log() {
  echo "[release_check] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Must run inside a git repository." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit/stash changes before release." >&2
  git status --short
  exit 1
fi

require_cmd python
require_cmd git

log "Ensuring packaging tools are available"
python -m pip install --upgrade build twine >/dev/null

log "Running lint"
python -m ruff check .

log "Running tests"
python -m pytest tests/ -v

log "Cleaning build artifacts"
rm -rf dist/ build/ .eggs/ ./*.egg-info

log "Building source and wheel distributions"
python -m build

log "Validating package metadata/rendering"
python -m twine check dist/*

log "Release preflight passed"

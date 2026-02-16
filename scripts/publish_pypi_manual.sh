#!/usr/bin/env bash

set -euo pipefail

log() {
  echo "[publish_pypi_manual] $*"
}

if [[ -f ".env" ]]; then
  log "Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${PYPI_API_KEY:-}" ]]; then
  echo "PYPI_API_KEY is required (set it in your environment or .env)." >&2
  exit 1
fi

export TWINE_USERNAME=__token__
export TWINE_PASSWORD="${PYPI_API_KEY}"

log "Ensuring packaging tools are available"
python -m pip install --upgrade build twine >/dev/null

log "Cleaning previous artifacts"
rm -rf dist/ build/ .eggs/
rm -rf ./*.egg-info 2>/dev/null || true

log "Building distributions"
python -m build

log "Validating package metadata"
python -m twine check dist/*

log "Uploading to PyPI via token"
python -m twine upload dist/*

log "Publish complete"

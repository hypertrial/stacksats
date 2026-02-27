#!/usr/bin/env bash
set -euo pipefail

pytest --cov=stacksats --cov-report=term-missing --cov-report=xml --cov-fail-under=100 -q "$@"

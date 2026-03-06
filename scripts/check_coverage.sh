#!/usr/bin/env bash
set -euo pipefail

# Coverage is release-grade by design: include slow + integration tests and exclude only
# performance benchmarks.
python -m pytest --cov=stacksats --cov-report=term-missing --cov-report=xml --cov-fail-under=100 -q -m "not performance" "$@"

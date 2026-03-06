#!/usr/bin/env bash
set -euo pipefail

# Coverage is release-grade by design: include slow + integration tests and exclude only
# performance benchmarks.
#
# The fail-under floor is intentionally ratcheted upward over time and should not
# be lowered in routine cleanup PRs.
COVERAGE_FAIL_UNDER="${COVERAGE_FAIL_UNDER:-97}"

python -m pytest \
  --cov=stacksats \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under="${COVERAGE_FAIL_UNDER}" \
  -q \
  -m "not performance" \
  "$@"

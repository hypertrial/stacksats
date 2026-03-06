from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_collect(marker_expr: str, nodeid: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            marker_expr,
            nodeid,
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )


def test_fast_tier_excludes_slow_marked_runner_test() -> None:
    proc = _run_collect(
        'not slow and not integration and not performance',
        "tests/unit/core/test_runner.py::test_runner_backtest_with_uniform_strategy",
    )
    # pytest exits with code 5 when nothing is selected in collect-only mode.
    assert proc.returncode in {0, 5}
    assert "test_runner_backtest_with_uniform_strategy" not in proc.stdout
    assert "deselected" in f"{proc.stdout}\n{proc.stderr}"


def test_heavy_tier_includes_slow_marked_runner_test() -> None:
    proc = _run_collect(
        "slow or integration or performance",
        "tests/unit/core/test_runner.py::test_runner_backtest_with_uniform_strategy",
    )
    assert proc.returncode == 0
    assert "test_runner_backtest_with_uniform_strategy" in proc.stdout

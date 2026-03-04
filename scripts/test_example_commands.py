#!/usr/bin/env python3
"""Run all docs/commands.md example commands with pass/fail reporting."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EXAMPLE_SPEC = "stacksats.strategies.examples:SimpleZScoreStrategy"


def run_step(label: str, cmd: list[str], *, cwd: Path, env: dict[str, str]) -> bool:
    print(f"\n=== {label} ===")
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode == 0:
        print(f"PASS: {label}")
        return True
    print(f"FAIL: {label}")
    return False


def skip_step(label: str, reason: str) -> None:
    print(f"\n=== {label} ===")
    print(f"SKIP: {reason}")


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    venv_dir = root_dir / "venv"
    venv_python = venv_dir / "bin" / "python"
    venv_bin = venv_dir / "bin"

    if not venv_python.exists():
        print(f"ERROR: venv python not found at {venv_python}")
        return 1

    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    cli_prefix = [str(venv_python), "-m", "stacksats.cli"]

    passed = 0
    failed = 0
    skipped = 0

    steps: list[tuple[str, list[str]]] = [
        (
                "Quick run (validate)",
                [
                *cli_prefix,
                "strategy",
                "validate",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-31",
            ],
        ),
        (
            "Quick run (backtest)",
            [
                *cli_prefix,
                "strategy",
                "backtest",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-31",
                "--output-dir",
                "output",
                "--strategy-label",
                "simple-zscore",
            ],
        ),
        (
            "Validate strategy (basic)",
            [
                *cli_prefix,
                "strategy",
                "validate",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-31",
                "--min-win-rate",
                "25.0",
            ],
        ),
        (
            "Validate strategy (with options)",
            [
                *cli_prefix,
                "strategy",
                "validate",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-31",
                "--min-win-rate",
                "25.0",
            ],
        ),
        (
            "Backtest (basic)",
            [
                *cli_prefix,
                "strategy",
                "backtest",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-31",
            ],
        ),
        (
            "Backtest (with options)",
            [
                *cli_prefix,
                "strategy",
                "backtest",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-31",
                "--output-dir",
                "output",
                "--strategy-label",
                "simple-zscore",
            ],
        ),
        (
            "Export strategy artifacts",
            [
                *cli_prefix,
                "strategy",
                "export",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2025-12-01",
                "--end-date",
                "2027-12-31",
                "--output-dir",
                "output",
            ],
        ),
    ]

    for label, cmd in steps:
        if run_step(label, cmd, cwd=root_dir, env=env):
            passed += 1
        else:
            failed += 1

    skip_step("Run tests", "pytest disabled for this script")
    skipped += 1

    skip_step("Run lint", "repo-wide lint is validated separately")
    skipped += 1

    print("\n=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

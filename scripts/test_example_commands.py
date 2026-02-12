#!/usr/bin/env python3
"""Run all docs/commands.md example commands with pass/fail reporting."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

EXAMPLE_SPEC = "examples/model_example.py:ExampleMVRVStrategy"


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

    passed = 0
    failed = 0
    skipped = 0

    steps: list[tuple[str, list[str]]] = [
        ("Quick run (default)", [str(venv_python), "examples/model_example.py"]),
        (
            "Quick run (with options)",
            [
                str(venv_python),
                "examples/model_example.py",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-01-01",
                "--output-dir",
                "output",
                "--strategy-label",
                "example-mvrv-strategy",
            ],
        ),
        (
            "Validate strategy (basic)",
            ["stacksats-validate", "--strategy", EXAMPLE_SPEC],
        ),
        (
            "Validate strategy (with options)",
            [
                "stacksats-validate",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-01-01",
                "--min-win-rate",
                "50.0",
            ],
        ),
        (
            "Backtest (basic)",
            ["stacksats-backtest", "--strategy", EXAMPLE_SPEC],
        ),
        (
            "Backtest (with options)",
            [
                "stacksats-backtest",
                "--strategy",
                EXAMPLE_SPEC,
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2025-01-01",
                "--output-dir",
                "output",
                "--strategy-label",
                "model-example",
            ],
        ),
    ]

    for label, cmd in steps:
        if run_step(label, cmd, cwd=root_dir, env=env):
            passed += 1
        else:
            failed += 1

    database_url = env.get("DATABASE_URL", "")
    if database_url and database_url != "postgresql://...":
        if run_step(
            "Export weights to database",
            ["stacksats-export", "--strategy", EXAMPLE_SPEC],
            cwd=root_dir,
            env=env,
        ):
            passed += 1
        else:
            failed += 1
    else:
        skip_step(
            "Export weights to database",
            "DATABASE_URL is not set to a real Postgres URL",
        )
        skipped += 1

    if shutil.which("modal", path=env["PATH"]):
        modal_env = env.copy()
        modal_env["STACKSATS_STRATEGY"] = EXAMPLE_SPEC
        if run_step(
            "Modal deploy",
            ["modal", "deploy", "stacksats/modal_app.py"],
            cwd=root_dir,
            env=modal_env,
        ):
            passed += 1
        else:
            failed += 1
    else:
        skip_step("Modal deploy", "modal CLI is not installed")
        skipped += 1

    skip_step("Run tests", "pytest disabled for this script")
    skipped += 1

    if run_step("Run lint", ["ruff", "check", "."], cwd=root_dir, env=env):
        passed += 1
    else:
        failed += 1

    print("\n=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

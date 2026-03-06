#!/usr/bin/env python3
"""Run deterministic smoke variants of docs/commands.md lifecycle commands."""

from __future__ import annotations

from datetime import date, timedelta
import os
import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLE_SPEC = "stacksats.strategies.examples:SimpleZScoreStrategy"


def _write_synthetic_coinmetrics_cache(
    cache_csv: Path,
    *,
    end_date: date,
    lookback_days: int = 2200,
) -> None:
    start = end_date - timedelta(days=lookback_days)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    with cache_csv.open("w", encoding="utf-8") as handle:
        handle.write("time,PriceUSD\n")
        for offset in range(lookback_days + 1):
            day = start + timedelta(days=offset)
            price = 10000.0 + (offset * 6.5)
            handle.write(f"{day.isoformat()},{price:.2f}\n")


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
    runtime_python = Path(sys.executable).resolve()

    env = os.environ.copy()
    if venv_python.exists():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        cli_prefix = [str(venv_python), "-m", "stacksats.cli"]
    else:
        print(f"INFO: venv not found at {venv_python}; using current interpreter: {runtime_python}")
        cli_prefix = [str(runtime_python), "-m", "stacksats.cli"]

    passed = 0
    failed = 0
    skipped = 0
    today = date.today()
    one_year_ago = today - timedelta(days=365)
    start_date = one_year_ago.isoformat()
    end_date = today.isoformat()

    with tempfile.TemporaryDirectory(prefix="stacksats-example-smoke-") as tmp_home:
        synthetic_home = Path(tmp_home)
        cache_csv = synthetic_home / ".stacksats" / "cache" / "coinmetrics_btc.csv"
        _write_synthetic_coinmetrics_cache(cache_csv, end_date=today)
        mpl_config_dir = synthetic_home / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(synthetic_home)
        env["MPLCONFIGDIR"] = str(mpl_config_dir)

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
                    start_date,
                    "--end-date",
                    end_date,
                    "--min-win-rate",
                    "0.0",
                    "--no-strict",
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
                    start_date,
                    "--end-date",
                    end_date,
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
                    start_date,
                    "--end-date",
                    end_date,
                    "--min-win-rate",
                    "0.0",
                    "--no-strict",
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
                    start_date,
                    "--end-date",
                    end_date,
                    "--min-win-rate",
                    "0.0",
                    "--no-strict",
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
                    start_date,
                    "--end-date",
                    end_date,
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
                    start_date,
                    "--end-date",
                    end_date,
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
                    start_date,
                    "--end-date",
                    end_date,
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

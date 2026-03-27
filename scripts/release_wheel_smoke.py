#!/usr/bin/env python
"""Release-grade isolated smoke validation for built StackSats wheels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _venv_bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _run_checked(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(
            "Command failed:\n"
            f"$ {joined}\n"
            f"cwd={cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _create_venv(root: Path, name: str) -> tuple[Path, Path, Path]:
    venv_dir = root / name
    _run_checked([sys.executable, "-m", "venv", str(venv_dir)], cwd=root, env=os.environ.copy())
    bin_dir = _venv_bin_dir(venv_dir)
    python_path = bin_dir / ("python.exe" if os.name == "nt" else "python")
    pip_path = bin_dir / ("pip.exe" if os.name == "nt" else "pip")
    _run_checked([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], cwd=root, env=os.environ.copy())
    return venv_dir, python_path, pip_path


def _base_runtime_env(home_dir: Path, *, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    if extra_env:
        env.update(extra_env)
    return env


def _resolve_packaged_demo_parquet(python_path: Path, *, cwd: Path, env: dict[str, str]) -> Path:
    result = _run_checked(
        [
            str(python_path),
            "-c",
            "\n".join(
                [
                    "from pathlib import Path",
                    "from stacksats.data_setup import packaged_demo_parquet_path",
                    "with packaged_demo_parquet_path() as path:",
                    "    print(Path(path).resolve())",
                ]
            ),
        ],
        cwd=cwd,
        env=env,
    )
    return Path(result.stdout.strip())


def _install_wheel(pip_path: Path, wheel_path: Path, *, cwd: Path, env: dict[str, str]) -> None:
    _run_checked([str(pip_path), "install", str(wheel_path)], cwd=cwd, env=env)


def _install_viz_deps(
    pip_path: Path,
    constraints_file: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    _run_checked(
        [
            str(pip_path),
            "install",
            "-c",
            str(constraints_file),
            "matplotlib",
            "Pillow",
            "seaborn",
        ],
        cwd=cwd,
        env=env,
    )


def _base_smoke(root: Path, wheel_path: Path) -> None:
    home_dir = root / "base-home"
    runtime_dir = root / "base-runtime"
    output_dir = root / "base-output"
    home_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    base_venv_dir, python_path, pip_path = _create_venv(root, "base-venv")
    base_env = _base_runtime_env(home_dir)

    _install_wheel(pip_path, wheel_path, cwd=root, env=base_env)

    import_probe = _run_checked(
        [
            str(python_path),
            "-c",
            "import json, stacksats; print(json.dumps({'module': stacksats.__file__}))",
        ],
        cwd=runtime_dir,
        env=base_env,
    )
    module_path = Path(json.loads(import_probe.stdout)["module"]).resolve()
    if base_venv_dir.resolve() not in module_path.parents:
        raise RuntimeError(f"stacksats imported from unexpected location: {module_path}")

    cli_name = "stacksats.exe" if os.name == "nt" else "stacksats"
    cli_path = _venv_bin_dir(base_venv_dir) / cli_name
    _run_checked(
        [str(cli_path), "demo", "backtest", "--output-dir", str(output_dir)],
        cwd=runtime_dir,
        env=base_env,
    )

    backtest_result = next(output_dir.glob("**/backtest_result.json"), None)
    if backtest_result is None:
        raise RuntimeError(f"Missing backtest_result.json under {output_dir}")

    payload = json.loads(backtest_result.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0.0":
        raise RuntimeError(
            f"Unexpected schema_version in {backtest_result}: {payload.get('schema_version')!r}"
        )


def _viz_smoke(root: Path, wheel_path: Path, constraints_file: Path) -> None:
    home_dir = root / "viz-home"
    runtime_dir = root / "viz-runtime"
    mpl_config_dir = home_dir / ".mplconfig"
    output_path = runtime_dir / "mvrv_metrics.svg"
    home_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    _, python_path, pip_path = _create_venv(root, "viz-venv")
    viz_env = _base_runtime_env(home_dir, extra_env={"MPLCONFIGDIR": str(mpl_config_dir)})

    _install_wheel(pip_path, wheel_path, cwd=root, env=viz_env)
    _install_viz_deps(pip_path, constraints_file, cwd=root, env=viz_env)
    demo_parquet = _resolve_packaged_demo_parquet(python_path, cwd=runtime_dir, env=viz_env)
    viz_env["STACKSATS_ANALYTICS_PARQUET"] = str(demo_parquet)

    plot_name = "stacksats-plot-mvrv.exe" if os.name == "nt" else "stacksats-plot-mvrv"
    plot_cli_path = _venv_bin_dir(root / "viz-venv") / plot_name
    _run_checked(
        [str(plot_cli_path), "--output", str(output_path)],
        cwd=runtime_dir,
        env=viz_env,
    )

    if not output_path.exists():
        raise RuntimeError(f"Missing plot output at {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated release smoke checks against a built StackSats wheel."
    )
    parser.add_argument(
        "--wheel",
        required=True,
        help="Path to the built wheel. Wildcards are allowed when quoted by the shell.",
    )
    parser.add_argument(
        "--constraints-file",
        required=True,
        help="Constraints file used for reproducible viz dependency installation.",
    )
    parser.add_argument(
        "--mode",
        choices=("base", "all"),
        default="all",
        help="Run only the base smoke or both base and viz smokes.",
    )
    return parser.parse_args()


def _resolve_single_path(pattern: str, *, kind: str) -> Path:
    literal_path = Path(pattern)
    if literal_path.exists():
        return literal_path.resolve()

    matches = sorted(Path().glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one {kind} for pattern {pattern!r}, found: {matches}. "
            "Use an exact file path or clean old build artifacts first."
        )
    return matches[0].resolve()


def main() -> int:
    args = _parse_args()
    wheel_path = _resolve_single_path(args.wheel, kind="wheel")
    constraints_file = _resolve_single_path(args.constraints_file, kind="constraints file")

    if shutil.which(sys.executable) is None:
        raise SystemExit(f"Current Python executable is not available: {sys.executable}")

    with tempfile.TemporaryDirectory(prefix="stacksats-release-smoke-") as tmp:
        root = Path(tmp)
        _base_smoke(root, wheel_path)
        if args.mode == "all":
            _viz_smoke(root, wheel_path, constraints_file)

    print(f"Release wheel smoke passed for {wheel_path.name} ({args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

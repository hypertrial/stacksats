from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


def _venv_bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _run_checked(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"$ {' '.join(cmd)}\n"
            f"cwd={cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def test_built_wheel_supports_demo_backtest_via_console_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    dist_dir = tmp_path / "dist"
    venv_dir = tmp_path / "wheel-venv"
    home_dir = tmp_path / "home"
    output_dir = tmp_path / "wheel-output"
    runtime_dir = tmp_path / "runtime"
    no_viz_runtime_dir = tmp_path / "runtime-no-viz"
    mpl_config_dir = home_dir / ".mplconfig"

    home_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    no_viz_runtime_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    build_env = os.environ.copy()
    _run_checked(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        cwd=repo_root,
        env=build_env,
    )

    wheel_paths = sorted(dist_dir.glob("stacksats-*.whl"))
    assert len(wheel_paths) == 1, f"Expected one built wheel, found: {wheel_paths}"
    wheel_path = wheel_paths[0]

    _run_checked(
        # This regression helper intentionally inherits site-packages so it can run
        # in offline/local environments. The release gate uses a separate isolated
        # workflow smoke script for clean-install validation.
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        cwd=tmp_path,
        env=build_env,
    )

    bin_dir = _venv_bin_dir(venv_dir)
    pip_path = bin_dir / ("pip.exe" if os.name == "nt" else "pip")
    cli_path = bin_dir / ("stacksats.exe" if os.name == "nt" else "stacksats")
    plot_cli_path = bin_dir / ("stacksats-plot-mvrv.exe" if os.name == "nt" else "stacksats-plot-mvrv")
    python_path = bin_dir / ("python.exe" if os.name == "nt" else "python")

    assert pip_path.exists(), f"Missing venv pip at {pip_path}"
    assert cli_path.exists() is False

    _run_checked(
        [str(pip_path), "install", "--no-deps", str(wheel_path)],
        cwd=tmp_path,
        env=build_env,
    )

    assert cli_path.exists(), f"Missing installed console script at {cli_path}"
    assert plot_cli_path.exists(), f"Missing installed plot console script at {plot_cli_path}"

    runtime_env = os.environ.copy()
    runtime_env["HOME"] = str(home_dir)
    runtime_env["MPLCONFIGDIR"] = str(mpl_config_dir)
    runtime_env.pop("STACKSATS_ANALYTICS_PARQUET", None)
    runtime_env.pop("PYTHONPATH", None)
    runtime_env["PYTHONNOUSERSITE"] = "1"

    asset_check = _run_checked(
        [
            str(python_path),
            "-c",
            "\n".join(
                [
                    "from pathlib import Path",
                    "import stacksats",
                    "from stacksats.data.data_setup import packaged_demo_parquet_path, packaged_text",
                    "from stacksats.eda import load_metric_catalog, open_merged_metrics",
                    "import polars as pl",
                    "print(Path(stacksats.__file__).resolve())",
                    "with packaged_demo_parquet_path() as path:",
                    "    print(path.resolve())",
                    "    print(path.exists())",
                    "    print(pl.read_parquet(path).height)",
                    "catalog = load_metric_catalog()",
                    "print(catalog.summary()['metric_count'])",
                    "payload = packaged_text('brk_merged_metrics_catalog.json')",
                    "print('market_cap' in payload)",
                    "canonical = Path('synthetic_merged_metrics.parquet')",
                    "pl.DataFrame([",
                    "    {'day_utc': '2024-01-01', 'metric': 'market_cap', 'value': 100.0},",
                    "    {'day_utc': '2024-01-01', 'metric': 'adjusted_sopr', 'value': 1.1},",
                    "    {'day_utc': '2024-01-02', 'metric': 'market_cap', 'value': 101.0},",
                    "]).with_columns(pl.col('day_utc').str.to_date()).write_parquet(canonical)",
                    "dataset = open_merged_metrics(canonical)",
                    "print(dataset.summary()['row_count'])",
                ]
            ),
        ],
        cwd=runtime_dir,
        env=runtime_env,
    )
    asset_lines = [line.strip() for line in asset_check.stdout.splitlines() if line.strip()]
    assert len(asset_lines) >= 7, f"Unexpected asset probe output: {asset_check.stdout}"
    module_path = Path(asset_lines[0])
    asset_path = Path(asset_lines[1])
    assert str(module_path).startswith(str(venv_dir.resolve()))
    assert str(asset_path).startswith(str(venv_dir.resolve()))
    assert asset_lines[2] == "True"
    assert int(asset_lines[3]) > 3000
    assert int(asset_lines[4]) > 40_000
    assert asset_lines[5] == "True"
    assert asset_lines[6] == "3"
    runtime_env["STACKSATS_ANALYTICS_PARQUET"] = str(asset_path)

    result = _run_checked(
        [str(cli_path), "demo", "backtest", "--output-dir", str(output_dir)],
        cwd=runtime_dir,
        env=runtime_env,
    )

    assert "Score:" in result.stdout
    assert "Saved:" in result.stdout
    backtest_results = sorted(output_dir.glob("**/backtest_result.json"))
    assert backtest_results, f"No backtest_result.json found under {output_dir}"

    plot_output = runtime_dir / "mvrv_metrics.svg"
    _run_checked(
        [str(plot_cli_path), "--output", str(plot_output)],
        cwd=runtime_dir,
        env=runtime_env,
    )
    assert plot_output.exists(), f"Missing plot output at {plot_output}"

    # Simulate a base install without viz extras by shadowing matplotlib/seaborn.
    (no_viz_runtime_dir / "matplotlib.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'matplotlib'\")\n",
        encoding="utf-8",
    )
    (no_viz_runtime_dir / "seaborn.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'seaborn'\")\n",
        encoding="utf-8",
    )
    no_viz_env = dict(runtime_env)
    no_viz_env["PYTHONPATH"] = str(no_viz_runtime_dir)
    no_viz_result = subprocess.run(
        [str(plot_cli_path), "--output", str(no_viz_runtime_dir / "unused.svg")],
        cwd=str(no_viz_runtime_dir),
        env=no_viz_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert no_viz_result.returncode != 0
    combined_output = f"{no_viz_result.stdout}\n{no_viz_result.stderr}"
    assert 'stacksats[viz]' in combined_output

#!/usr/bin/env python
"""Release-grade isolated smoke validation for built StackSats wheels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


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


def _install_wheel_with_extra(
    pip_path: Path,
    wheel_path: Path,
    *,
    extra: str,
    cwd: Path,
    env: dict[str, str],
) -> None:
    _run_checked(
        [str(pip_path), "install", f"stacksats[{extra}] @ {wheel_path.resolve().as_uri()}"],
        cwd=cwd,
        env=env,
    )


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


def _choose_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _http_json(
    *,
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, object] | None = None,
    timeout: int = 10,
) -> tuple[int, dict[str, str], dict[str, object]]:
    request_headers = dict(headers or {})
    body = None
    if payload is not None:
        request_headers.setdefault("Content-Type", "application/json")
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(response.status)
            response_headers = {
                str(key).lower(): str(value) for key, value in response.headers.items()
            }
            response_payload = json.loads(response.read().decode("utf-8"))
            return status_code, response_headers, response_payload
    except urllib.error.HTTPError as exc:
        response_headers = {str(key).lower(): str(value) for key, value in exc.headers.items()}
        response_payload = json.loads(exc.read().decode("utf-8"))
        return int(exc.code), response_headers, response_payload


def _wait_for_service_ready(*, url: str, process: subprocess.Popen[str], timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout = process.communicate(timeout=5)[0]
            raise RuntimeError(f"Hosted service exited before readiness check passed.\n{stdout}")
        try:
            status_code, _, payload = _http_json(url=url, timeout=2)
            if status_code == 200 and payload.get("status") == "ok":
                return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    process.terminate()
    stdout = process.communicate(timeout=5)[0]
    raise RuntimeError(
        "Timed out waiting for hosted service readiness.\n"
        f"Last error: {last_error}\n{stdout}"
    )


def _service_smoke(root: Path, wheel_path: Path) -> None:
    home_dir = root / "service-home"
    runtime_dir = root / "service-runtime"
    output_dir = runtime_dir / "output"
    state_db_path = runtime_dir / ".stacksats" / "run_state.sqlite3"
    registry_path = runtime_dir / "agent_service_registry.json"
    home_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_db_path.parent.mkdir(parents=True, exist_ok=True)

    service_token = "service-smoke-token"
    service_port = _choose_free_port()
    service_url = f"http://127.0.0.1:{service_port}"

    service_venv_dir, python_path, pip_path = _create_venv(root, "service-venv")
    service_env = _base_runtime_env(home_dir)
    _install_wheel_with_extra(
        pip_path,
        wheel_path,
        extra="service",
        cwd=root,
        env=service_env,
    )

    demo_parquet = _resolve_packaged_demo_parquet(python_path, cwd=runtime_dir, env=service_env)
    service_env["STACKSATS_ANALYTICS_PARQUET"] = str(demo_parquet)
    service_env["STACKSATS_AGENT_API_TOKEN"] = service_token
    registry_path.write_text(
        json.dumps(
            {
                "btc-dca-paper": {
                    "strategy_spec": "stacksats.strategies.examples:RunDailyPaperStrategy",
                    "enabled": True,
                    "btc_price_col": "price_usd",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cli_name = "stacksats.exe" if os.name == "nt" else "stacksats"
    cli_path = _venv_bin_dir(service_venv_dir) / cli_name
    process = subprocess.Popen(
        [
            str(cli_path),
            "serve",
            "agent-api",
            "--host",
            "127.0.0.1",
            "--port",
            str(service_port),
            "--registry-path",
            str(registry_path),
            "--state-db-path",
            str(state_db_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(runtime_dir),
        env=service_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_service_ready(url=f"{service_url}/healthz", process=process)
        health_status, health_headers, health_payload = _http_json(url=f"{service_url}/healthz")
        if health_status != 200 or health_payload.get("api_version") != "v1":
            raise RuntimeError(f"Unexpected healthz response: {health_status} {health_payload}")
        if "x-request-id" not in health_headers:
            raise RuntimeError("Hosted service healthz response is missing X-Request-ID header.")

        discovery_status, _, discovery_payload = _http_json(
            url=f"{service_url}/.well-known/agent-integration.json"
        )
        if discovery_status != 200 or discovery_payload.get("api_version") != "v1":
            raise RuntimeError(
                f"Unexpected discovery response: {discovery_status} {discovery_payload}"
            )

        decision_status, decision_headers, decision_payload = _http_json(
            url=f"{service_url}/v1/decisions/daily",
            method="POST",
            headers={
                "Authorization": f"Bearer {service_token}",
                "X-Request-ID": "service-smoke-request",
            },
            payload={
                "strategy_id": "btc-dca-paper",
                "total_window_budget_usd": 1000.0,
            },
        )
        if decision_status != 200:
            raise RuntimeError(
                f"Unexpected decision response status: {decision_status} {decision_payload}"
            )
        if decision_payload.get("status") != "decided":
            raise RuntimeError(f"Unexpected decision payload: {decision_payload}")
        if decision_payload.get("strategy_id") != "run-daily-paper":
            raise RuntimeError(f"Unexpected strategy_id in decision payload: {decision_payload}")
        if decision_headers.get("x-request-id") != "service-smoke-request":
            raise RuntimeError(
                "Hosted service decision response did not preserve the X-Request-ID header."
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
        else:
            process.communicate(timeout=5)


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
        choices=("base", "service", "all"),
        default="all",
        help="Run base-only, service-only, or base+viz+service smokes.",
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
        if args.mode in {"base", "all"}:
            _base_smoke(root, wheel_path)
        if args.mode == "service":
            _service_smoke(root, wheel_path)
        if args.mode == "all":
            _viz_smoke(root, wheel_path, constraints_file)
            _service_smoke(root, wheel_path)

    print(f"Release wheel smoke passed for {wheel_path.name} ({args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import marimo  # pyright: ignore[reportMissingImports]

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import shlex
    import subprocess
    import sys
    from importlib import util as importlib_util
    from pathlib import Path

    return Path, importlib_util, shlex, subprocess, sys


@app.cell
def _(mo):
    mo.md(
        """
        # StackSats model example notebook

        This notebook installs the package in the **current venv** and runs a
        backtest using the packaged example strategy.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("Repository root: current working directory")
    return


@app.cell
def _(Path, shlex, subprocess):
    def run_cmd(command: list[str], *, env: dict[str, str] | None = None) -> int:
        print("$ " + " ".join(shlex.quote(arg) for arg in command))
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            check=False,
            text=True,
        )
        print(f"[exit code: {completed.returncode}]")
        return completed.returncode

    return (run_cmd,)


@app.cell
def _(mo, sys):
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    mo.md(f"Python version: `{py_version}`  \nInside venv: `{'yes' if in_venv else 'no'}`")
    return in_venv


@app.cell
def _(importlib_util, run_cmd, sys):
    needs_stacksats = importlib_util.find_spec("stacksats") is None
    needs_marimo = importlib_util.find_spec("marimo") is None

    if needs_stacksats or needs_marimo:
        print("Installing missing dependencies into current venv...")
        if needs_stacksats:
            run_cmd([sys.executable, "-m", "pip", "install", "-e", "."])
        else:
            print("`stacksats` already installed; skipping.")
        if needs_marimo:
            run_cmd([sys.executable, "-m", "pip", "install", "marimo"])
        else:
            print("`marimo` already installed; skipping.")
    else:
        print("Dependencies already installed; skipping install step.")
    return


@app.cell
def _(run_cmd):
    print("1) Backtest")
    strategy_spec = "examples/model_example.py:ExampleMVRVStrategy"
    run_cmd(
        [
            "stacksats",
            "strategy",
            "backtest",
            "--strategy",
            strategy_spec,
            "--end-date",
            "2026-01-31",
            "--output-dir",
            "output",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Run the notebook

        From your active venv at repo root:

        ```bash
        marimo edit examples/model_example_notebook.py
        ```

        Then run all cells from top to bottom.
        """
    )
    return


if __name__ == "__main__":
    app.run()

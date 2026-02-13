import marimo  # pyright: ignore[reportMissingImports]

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from importlib import util as importlib_util
    import shlex
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    return Path, importlib_util, shlex, subprocess, sys, textwrap


@app.cell
def _(mo):
    mo.md(
        """
        # StackSats model example notebook

        This notebook installs the package in the **current venv**, defines
        the strategy in-notebook, and runs one backtest command.
        """
    )
    return


@app.cell
def _(Path):
    repo_root = Path.cwd()
    strategy_file = repo_root / "output" / "notebook_model_strategy.py"
    return repo_root, strategy_file


@app.cell
def _(mo, repo_root):
    mo.md(f"Repository root: `{repo_root}`")
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
    mo.md(f"Python: `{sys.executable}`  \nInside venv: `{'yes' if in_venv else 'no'}`")
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
def _(strategy_file, textwrap):
    strategy_source = textwrap.dedent(
        """
        \"\"\"Notebook-defined example strategy for StackSats.\"\"\"

        from __future__ import annotations

        from dataclasses import dataclass

        import pandas as pd

        from stacksats import BaseStrategy, StrategyContext, TargetProfile


        @dataclass(frozen=True)
        class ExampleConfig:
            value_weight: float = 0.7
            trend_weight: float = 0.3
            strength: float = 4.0
            trend_lookback_days: int = 30


        class ExampleMVRVStrategy(BaseStrategy):
            strategy_id = "example-mvrv-notebook"
            version = "1.0.0"
            description = "Notebook-defined strategy."

            def __init__(
                self,
                value_weight: float = 0.7,
                trend_weight: float = 0.3,
                strength: float = 4.0,
                trend_lookback_days: int = 30,
            ):
                self.cfg = ExampleConfig(
                    value_weight=value_weight,
                    trend_weight=trend_weight,
                    strength=strength,
                    trend_lookback_days=trend_lookback_days,
                )

            def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
                window = ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()
                if window.empty:
                    return window
                price = window[ctx.btc_price_col]
                trend = price.pct_change(self.cfg.trend_lookback_days).fillna(0.0)
                window["trend_signal_input"] = trend.clip(-1.0, 1.0)
                return window

            def build_signals(
                self,
                ctx: StrategyContext,
                features_df: pd.DataFrame,
            ) -> dict[str, pd.Series]:
                del ctx
                value_signal = -features_df.get(
                    "mvrv_zscore",
                    pd.Series(0.0, index=features_df.index),
                ).clip(-4, 4)
                trend_signal = -features_df.get(
                    "trend_signal_input",
                    pd.Series(0.0, index=features_df.index),
                )
                return {"value": value_signal, "trend": trend_signal}

            def build_target_profile(
                self,
                ctx: StrategyContext,
                features_df: pd.DataFrame,
                signals: dict[str, pd.Series],
            ) -> TargetProfile:
                del ctx, features_df
                preference = (
                    (self.cfg.value_weight * signals["value"])
                    + (self.cfg.trend_weight * signals["trend"])
                ) * self.cfg.strength
                return TargetProfile(values=preference, mode="preference")
        """
    ).lstrip()
    strategy_file.parent.mkdir(parents=True, exist_ok=True)
    strategy_file.write_text(strategy_source, encoding="utf-8")
    print(f"Wrote strategy file: {strategy_file}")
    return strategy_file


@app.cell
def _(run_cmd, strategy_file):
    print("1) Backtest")
    strategy_spec = f"{strategy_file}:ExampleMVRVStrategy"
    run_cmd(
        [
            "stacksats",
            "strategy",
            "backtest",
            "--strategy",
            strategy_spec,
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

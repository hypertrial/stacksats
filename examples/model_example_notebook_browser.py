import marimo  # pyright: ignore[reportMissingImports]

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return np, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
        # StackSats browser-safe notebook demo

        This demo intentionally avoids subprocess calls, CLI execution, network access,
        and file writes so it can run in browser-only Python runtimes (Pyodide/WASM).

        It showcases a strategy-first DCA workflow:

        - generate a deterministic market dataset in-memory
        - compute features and a dynamic DCA policy
        - compare dynamic vs uniform DCA outcomes
        - inspect metrics and charts
        """
    )
    return


@app.cell
def _(np, pd):
    rng = np.random.default_rng(42)
    n_days = 9 * 365
    dates = pd.date_range("2017-01-01", periods=n_days, freq="D")

    drift = 0.00055
    cycle = 0.0009 * np.sin(np.linspace(0, 9 * np.pi, n_days))
    shock = rng.normal(0.0, 0.012, n_days)
    daily_returns = drift + cycle + shock

    price = 1200.0 * np.exp(np.cumsum(daily_returns))
    btc_df = pd.DataFrame({"price": price}, index=dates)
    btc_df.index.name = "date"
    btc_df["return_1d"] = btc_df["price"].pct_change().fillna(0.0)

    return (btc_df,)


@app.cell
def _(btc_df, np, pd):
    features = pd.DataFrame(index=btc_df.index)
    features["price"] = btc_df["price"]
    features["ma_60"] = features["price"].rolling(60, min_periods=20).mean()
    features["trend_signal"] = (features["price"] / features["ma_60"]) - 1.0
    features["momentum_14"] = features["price"].pct_change(14)
    features["volatility_21"] = btc_df["return_1d"].rolling(21, min_periods=7).std()

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (features,)


@app.cell
def _(btc_df, features, np, pd):
    policy_score = (
        (-2.2 * features["trend_signal"])
        + (1.1 * features["momentum_14"])
        - (4.2 * features["volatility_21"])
    )
    multiplier = 1.0 + 0.6 * np.tanh(policy_score)
    multiplier = np.clip(multiplier, 0.40, 1.60)

    base_usd = 25.0
    simulation = pd.DataFrame(index=btc_df.index)
    simulation["price"] = btc_df["price"]
    simulation["uniform_usd"] = base_usd
    simulation["dynamic_usd"] = base_usd * multiplier
    simulation["dynamic_multiplier"] = multiplier

    simulation["uniform_btc"] = simulation["uniform_usd"] / simulation["price"]
    simulation["dynamic_btc"] = simulation["dynamic_usd"] / simulation["price"]
    simulation["uniform_btc_cum"] = simulation["uniform_btc"].cumsum()
    simulation["dynamic_btc_cum"] = simulation["dynamic_btc"].cumsum()
    simulation["uniform_spend_cum"] = simulation["uniform_usd"].cumsum()
    simulation["dynamic_spend_cum"] = simulation["dynamic_usd"].cumsum()
    simulation["uniform_value"] = simulation["uniform_btc_cum"] * simulation["price"]
    simulation["dynamic_value"] = simulation["dynamic_btc_cum"] * simulation["price"]

    # Framework-like invariants for this demo policy.
    assert bool((simulation["dynamic_usd"] > 0.0).all())
    assert bool(np.isfinite(simulation["dynamic_multiplier"]).all())

    return (simulation,)


@app.cell
def _(np, pd, simulation):
    def max_drawdown_pct(series: pd.Series) -> float:
        running_max = series.cummax()
        drawdown = (series / running_max) - 1.0
        return float(drawdown.min() * 100.0)

    final_row = simulation.iloc[-1]
    uniform_final_value = float(final_row["uniform_value"])
    dynamic_final_value = float(final_row["dynamic_value"])
    uniform_final_btc = float(final_row["uniform_btc_cum"])
    dynamic_final_btc = float(final_row["dynamic_btc_cum"])

    value_edge_pct = ((dynamic_final_value / uniform_final_value) - 1.0) * 100.0
    btc_edge_pct = ((dynamic_final_btc / uniform_final_btc) - 1.0) * 100.0

    monthly_values = simulation[["uniform_value", "dynamic_value"]].resample("ME").last()
    monthly_returns = monthly_values.pct_change().dropna()
    monthly_win_rate = float(
        (monthly_returns["dynamic_value"] > monthly_returns["uniform_value"]).mean() * 100.0
    )

    uniform_mdd = max_drawdown_pct(simulation["uniform_value"])
    dynamic_mdd = max_drawdown_pct(simulation["dynamic_value"])
    mdd_delta = dynamic_mdd - uniform_mdd

    demo_score = (0.55 * value_edge_pct) + (0.35 * (monthly_win_rate - 50.0)) - (
        0.10 * mdd_delta
    )

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Final value (uniform)",
                "Final value (dynamic)",
                "Value edge",
                "BTC edge",
                "Monthly win rate",
                "Max drawdown (uniform)",
                "Max drawdown (dynamic)",
                "Demo score",
            ],
            "Value": [
                f"${uniform_final_value:,.2f}",
                f"${dynamic_final_value:,.2f}",
                f"{value_edge_pct:.2f}%",
                f"{btc_edge_pct:.2f}%",
                f"{monthly_win_rate:.2f}%",
                f"{uniform_mdd:.2f}%",
                f"{dynamic_mdd:.2f}%",
                f"{demo_score:.2f}",
            ],
        }
    )

    return metrics, monthly_returns


@app.cell
def _(metrics, mo):
    mo.md("## Backtest summary")
    return


@app.cell
def _(metrics):
    metrics
    return


@app.cell
def _(plt, simulation):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(simulation.index, simulation["price"], label="BTC price", color="#F8931A", lw=1.2)
    axes[0].set_title("Synthetic BTC price path")
    axes[0].set_ylabel("Price (USD)")
    axes[0].grid(alpha=0.20)
    axes[0].legend(loc="upper left")

    axes[1].plot(
        simulation.index,
        simulation["dynamic_multiplier"],
        label="Dynamic multiplier",
        color="#EFD7BD",
        lw=1.1,
    )
    axes[1].axhline(1.0, color="#A5A5B4", ls="--", lw=1.0, label="Uniform baseline")
    axes[1].set_title("Strategy output (multiplier over base daily DCA)")
    axes[1].set_ylabel("Multiplier")
    axes[1].set_ylim(0.3, 1.7)
    axes[1].grid(alpha=0.20)
    axes[1].legend(loc="upper left")

    axes[2].plot(
        simulation.index,
        simulation["uniform_value"],
        label="Uniform DCA value",
        color="#A5A5B4",
        lw=1.0,
    )
    axes[2].plot(
        simulation.index,
        simulation["dynamic_value"],
        label="Dynamic DCA value",
        color="#F8931A",
        lw=1.4,
    )
    axes[2].set_title("Portfolio value comparison")
    axes[2].set_ylabel("Value (USD)")
    axes[2].grid(alpha=0.20)
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    fig
    return


@app.cell
def _(monthly_returns, mo):
    mo.md("## Monthly return sample")
    return


@app.cell
def _(monthly_returns):
    monthly_returns.tail(12)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Notes

        - This notebook is self-contained and browser-safe by design.
        - For full StackSats package + CLI workflows, use:
          `examples/model_example_notebook.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()

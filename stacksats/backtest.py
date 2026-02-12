import argparse
import json
import logging
import os
from datetime import datetime

from . import matplotlib_setup  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .api import run_backtest
from .loader import load_strategy
from .model_development import compute_window_weights, precompute_features
from .prelude import check_strategy_submission_ready, load_data, parse_window_dates
from .strategies.mvrv import MVRVStrategy

# Set seaborn style for all plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

# Global variable to store precomputed features (Modal-aligned)
_FEATURES_DF = None


def compute_weights_modal(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper using compute_window_weights for validation.

    Matches the interface expected by check_strategy_submission_ready().
    Uses precomputed features stored in _FEATURES_DF.

    For backtesting historical data, current_date is set to end_date since
    all dates in the window are in the "past" (we have price data for them).

    This implementation uses the shared compute_window_weights() from
    model_development.py to ensure backtest results match production behavior.
    """
    global _FEATURES_DF

    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")

    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()

    # For backtesting, current_date = end_date (all dates are in the past)
    # This means all weights come from the model (no uniform future weights)
    current_date = end_date

    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def create_performance_comparison_chart(
    df_spd: pd.DataFrame, output_dir: str = "output"
):
    """Create line chart comparing dynamic vs uniform percentile over time."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract dates from window labels and sort
    dates_series = df_spd.index.map(parse_window_dates)
    df_with_dates = df_spd.copy()
    df_with_dates["_date"] = dates_series
    df_sorted = df_with_dates.sort_values("_date")

    # Prepare data for seaborn
    plot_df = pd.DataFrame(
        {
            "Date": df_sorted["_date"],
            "Dynamic DCA": df_sorted["dynamic_percentile"],
            "Uniform DCA": df_sorted["uniform_percentile"],
        }
    )
    plot_df = plot_df.melt(
        id_vars=["Date"], var_name="Strategy", value_name="SPD Percentile"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=plot_df,
        x="Date",
        y="SPD Percentile",
        hue="Strategy",
        style="Strategy",
        markers=False,
        linewidth=2.5,
        ax=ax,
    )

    ax.set_xlabel("Window Start Date", fontsize=12)
    ax.set_ylabel("SPD Percentile (%)", fontsize=12)
    ax.set_title(
        "Performance Comparison: Dynamic vs Uniform DCA", fontsize=14, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "performance_comparison.svg")
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    logging.info(f"✓ Saved: {output_path}")


def create_excess_percentile_distribution(
    df_spd: pd.DataFrame, output_dir: str = "output"
):
    """Create histogram of excess percentile distribution."""
    os.makedirs(output_dir, exist_ok=True)

    excess_percentile = df_spd["dynamic_percentile"] - df_spd["uniform_percentile"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        excess_percentile,
        bins=30,
        kde=True,
        edgecolor="black",
        alpha=0.7,
        color="#10b981",
        ax=ax,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Break-even")
    ax.axvline(
        excess_percentile.mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {excess_percentile.mean():.2f}%",
    )

    ax.set_xlabel("Excess Percentile (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Distribution of Excess Percentile (Dynamic - Uniform)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "excess_percentile_distribution.svg")
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    logging.info(f"✓ Saved: {output_path}")


def create_win_loss_comparison(df_spd: pd.DataFrame, output_dir: str = "output"):
    """Create bar chart showing win/loss comparison."""
    os.makedirs(output_dir, exist_ok=True)

    wins = (df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]).sum()
    losses = len(df_spd) - wins
    win_rate = wins / len(df_spd) * 100

    # Prepare data for seaborn
    comparison_df = pd.DataFrame(
        {
            "Outcome": ["Wins", "Losses"],
            "Count": [wins, losses],
            "Percentage": [wins / len(df_spd) * 100, losses / len(df_spd) * 100],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = sns.barplot(
        data=comparison_df,
        x="Outcome",
        y="Count",
        hue="Outcome",
        palette={"Wins": "#10b981", "Losses": "#ef4444"},
        edgecolor="black",
        linewidth=1.5,
        legend=False,
        ax=ax,
    )

    # Add value labels on bars
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}\n({comparison_df.iloc[i]['Percentage']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Number of Windows", fontsize=12)
    ax.set_title(
        f"Win/Loss Comparison\nWin Rate: {win_rate:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "win_loss_comparison.svg")
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    logging.info(f"✓ Saved: {output_path}")


def create_cumulative_performance(df_spd: pd.DataFrame, output_dir: str = "output"):
    """Create area chart showing cumulative performance difference."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract dates and sort
    dates_series = df_spd.index.map(parse_window_dates)
    df_with_dates = df_spd.copy()
    df_with_dates["_date"] = dates_series
    df_sorted = df_with_dates.sort_values("_date")
    dates = df_sorted["_date"]

    excess_percentile = (
        df_sorted["dynamic_percentile"] - df_sorted["uniform_percentile"]
    )
    cumulative_excess = excess_percentile.cumsum()

    # Prepare data for seaborn
    plot_df = pd.DataFrame({"Date": dates, "Cumulative Excess": cumulative_excess})

    fig, ax = plt.subplots(figsize=(12, 6))
    # Use seaborn for the line plot
    sns.lineplot(
        data=plot_df,
        x="Date",
        y="Cumulative Excess",
        linewidth=2.5,
        color="#059669",
        ax=ax,
    )
    # Fill area using matplotlib
    ax.fill_between(
        dates,
        0,
        cumulative_excess,
        alpha=0.4,
        color="#10b981",
        label="Cumulative Excess",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("Window Start Date", fontsize=12)
    ax.set_ylabel("Cumulative Excess Percentile (%)", fontsize=12)
    ax.set_title(
        "Cumulative Performance Advantage Over Time", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "cumulative_performance.svg")
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    logging.info(f"✓ Saved: {output_path}")


def create_performance_metrics_summary(
    df_spd: pd.DataFrame, metrics: dict, output_dir: str = "output"
):
    """Create a summary table visualization of key metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # Use seaborn style for the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = [
        ["Metric", "Value"],
        ["Final Model Score", f"{metrics['score']:.2f}%"],
        ["Win Rate", f"{metrics['win_rate']:.2f}%"],
        ["Exponential Decay Percentile", f"{metrics['exp_decay_percentile']:.2f}%"],
        ["Mean Excess Percentile", f"{metrics['mean_excess']:.2f}%"],
        ["Median Excess Percentile", f"{metrics['median_excess']:.2f}%"],
        [
            "Mean Relative Improvement",
            f"{metrics['relative_improvement_pct_mean']:.2f}%",
        ],
        [
            "Median Relative Improvement",
            f"{metrics['relative_improvement_pct_median']:.2f}%",
        ],
        ["Mean Ratio (Dynamic/Uniform)", f"{metrics['mean_ratio']:.2f}"],
        ["Median Ratio (Dynamic/Uniform)", f"{metrics['median_ratio']:.2f}"],
        ["Total Windows", f"{metrics['total_windows']}"],
        ["Wins", f"{metrics['wins']}"],
        ["Losses", f"{metrics['losses']}"],
    ]

    table = ax.table(
        cellText=table_data, cellLoc="left", loc="center", colWidths=[0.6, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor("#2563eb")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f1f5f9")

    ax.set_title(
        "Backtest Performance Metrics Summary", fontsize=16, fontweight="bold", pad=20
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "metrics_summary.svg")
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    logging.info(f"✓ Saved: {output_path}")


def export_metrics_json(
    df_spd: pd.DataFrame, metrics: dict, output_dir: str = "output"
):
    """Export all metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare JSON data
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": metrics,
        "window_level_data": [],
    }

    # Add window-level data
    for window_label in df_spd.index:
        window_data = {
            "window": window_label,
            "start_date": parse_window_dates(window_label).isoformat(),
            "dynamic_percentile": float(df_spd.loc[window_label, "dynamic_percentile"]),
            "uniform_percentile": float(df_spd.loc[window_label, "uniform_percentile"]),
            "excess_percentile": float(df_spd.loc[window_label, "excess_percentile"]),
            "dynamic_sats_per_dollar": float(
                df_spd.loc[window_label, "dynamic_sats_per_dollar"]
            ),
            "uniform_sats_per_dollar": float(
                df_spd.loc[window_label, "uniform_sats_per_dollar"]
            ),
            "min_sats_per_dollar": float(
                df_spd.loc[window_label, "min_sats_per_dollar"]
            ),
            "max_sats_per_dollar": float(
                df_spd.loc[window_label, "max_sats_per_dollar"]
            ),
        }
        json_data["window_level_data"].append(window_data)

    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logging.info(f"✓ Saved: {output_path}")


def main() -> None:
    global _FEATURES_DF

    parser = argparse.ArgumentParser(description="Run StackSats rolling-window backtest.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Custom strategy spec in 'module_or_path:ClassName' format",
    )
    parser.add_argument(
        "--strategy-label",
        type=str,
        default=None,
        help="Optional strategy label for reports",
    )
    args = parser.parse_args()

    logging.info("Starting Bitcoin DCA Strategy Analysis")

    if args.strategy:
        strategy = load_strategy(args.strategy)
        strategy_label = args.strategy_label or args.strategy
        btc_df = None
    else:
        strategy = MVRVStrategy()
        strategy_label = args.strategy_label or "Dynamic DCA"
        btc_df = load_data()

        # Keep the existing submission-readiness check for the built-in strategy.
        logging.info("Precomputing features for submission checks...")
        _FEATURES_DF = precompute_features(btc_df)
        logging.info("Running strategy submission checks...")
        check_strategy_submission_ready(btc_df, compute_weights_modal)

    result = run_backtest(
        strategy,
        btc_df=btc_df,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_label=strategy_label,
    )

    print(result.summary())
    result.plot(output_dir=args.output_dir)
    result.to_json(os.path.join(args.output_dir, "backtest_result.json"))
    logging.info(f"All outputs saved to '{args.output_dir}/' directory")


if __name__ == "__main__":
    main()

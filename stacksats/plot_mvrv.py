"""Plot CoinMetrics MVRV metrics over time.

This script fetches Bitcoin data from CoinMetrics and creates line plots for:
- CapMVRVCur: Market Value to Realized Value ratio
- CapMVRVZ: MVRV Z-Score

Usage:
    python plot_mvrv.py                    # Creates plots with default settings
    python plot_mvrv.py --start 2020-01-01  # Filter by start date
    python plot_mvrv.py --output mvrv.png   # Custom output filename
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from . import matplotlib_setup  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .btc_api.coinmetrics_btc_csv import fetch_coinmetrics_btc_csv

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set seaborn style for all plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300


def plot_mvrv_metrics(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_path: str = "mvrv_metrics.svg",
) -> None:
    """Plot CapMVRVCur and CapMVRVZ metrics over time.

    Args:
        df: DataFrame with CoinMetrics BTC data (must have 'CapMVRVCur' column)
        start_date: Optional start date filter (YYYY-MM-DD format)
        end_date: Optional end date filter (YYYY-MM-DD format)
        output_path: Path to save the plot

    Raises:
        ValueError: If CapMVRVCur column is missing from the DataFrame
    """
    # Validate required column exists
    if "CapMVRVCur" not in df.columns:
        available_cols = [
            c for c in df.columns if "MVRV" in c.upper() or "CAP" in c.upper()
        ]
        raise ValueError(
            f"Missing required column: CapMVRVCur. "
            f"Available MVRV/Cap columns: {available_cols if available_cols else 'None'}"
        )

    # Calculate CapMVRVZ if it doesn't exist (MVRV Z-Score)
    if "CapMVRVZ" not in df.columns:
        logging.info(
            "CapMVRVZ not found in data. Calculating MVRV Z-Score from CapMVRVCur..."
        )
        # MVRV Z-Score = (MVRV - mean(MVRV)) / std(MVRV)
        # Use rolling window to calculate mean and std
        mvrv_mean = df["CapMVRVCur"].rolling(window=365, min_periods=30).mean()
        mvrv_std = df["CapMVRVCur"].rolling(window=365, min_periods=30).std()
        df["CapMVRVZ"] = (df["CapMVRVCur"] - mvrv_mean) / mvrv_std
        logging.info(
            "✓ Calculated CapMVRVZ from CapMVRVCur using 365-day rolling window"
        )

    # Filter by date range if specified
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if len(df) == 0:
        raise ValueError("No data available for the specified date range")

    # Remove any rows with missing MVRV data
    df_clean = df[["CapMVRVCur", "CapMVRVZ"]].dropna()

    if len(df_clean) == 0:
        raise ValueError("No valid MVRV data available after removing missing values")

    logging.info(
        f"Plotting MVRV metrics: {len(df_clean)} data points from "
        f"{df_clean.index.min().date()} to {df_clean.index.max().date()}"
    )

    # Calculate 30-day moving averages
    mvrv_ma30 = df_clean["CapMVRVCur"].rolling(window=30, min_periods=1).mean()
    zscore_ma30 = df_clean["CapMVRVZ"].rolling(window=30, min_periods=1).mean()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: CapMVRVCur (MVRV Ratio)
    # Raw data (lighter/thinner)
    ax1.plot(
        df_clean.index,
        df_clean["CapMVRVCur"],
        linewidth=1.5,
        color="#2563eb",
        alpha=0.5,
        label="MVRV Ratio (Daily)",
    )
    ax1.fill_between(
        df_clean.index,
        df_clean["CapMVRVCur"],
        alpha=0.2,
        color="#2563eb",
    )
    # 30-day moving average (thicker/more prominent)
    ax1.plot(
        df_clean.index,
        mvrv_ma30,
        linewidth=2.5,
        color="#1e40af",
        label="30-Day MA",
    )

    # Add reference line at 1.0 (fair value)
    ax1.axhline(
        y=1.0,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Fair Value (1.0)",
    )

    ax1.set_title(
        "Bitcoin MVRV Ratio (Market Value / Realized Value)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_ylabel("MVRV Ratio", fontsize=12, fontweight="medium")
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # Add statistics text box
    mvrv_mean = df_clean["CapMVRVCur"].mean()
    mvrv_median = df_clean["CapMVRVCur"].median()
    mvrv_min = df_clean["CapMVRVCur"].min()
    mvrv_max = df_clean["CapMVRVCur"].max()
    mvrv_current = df_clean["CapMVRVCur"].iloc[-1]
    mvrv_ma30_current = mvrv_ma30.iloc[-1]

    stats_text1 = (
        f"Current: {mvrv_current:.2f}\n"
        f"30-Day MA: {mvrv_ma30_current:.2f}\n"
        f"Mean: {mvrv_mean:.2f}\n"
        f"Median: {mvrv_median:.2f}\n"
        f"Range: {mvrv_min:.2f} - {mvrv_max:.2f}"
    )

    ax1.text(
        0.98,
        0.98,
        stats_text1,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            alpha=0.95,
            edgecolor="#e5e7eb",
            linewidth=1.5,
        ),
        family="monospace",
    )

    # Plot 2: CapMVRVZ (MVRV Z-Score)
    # Raw data (lighter/thinner)
    ax2.plot(
        df_clean.index,
        df_clean["CapMVRVZ"],
        linewidth=1.5,
        color="#16a34a",
        alpha=0.5,
        label="MVRV Z-Score (Daily)",
    )
    ax2.fill_between(
        df_clean.index,
        df_clean["CapMVRVZ"],
        alpha=0.2,
        color="#16a34a",
    )
    # 30-day moving average (thicker/more prominent)
    ax2.plot(
        df_clean.index,
        zscore_ma30,
        linewidth=2.5,
        color="#15803d",
        label="30-Day MA",
    )

    # Add reference lines for Z-Score interpretation
    ax2.axhline(
        y=0, color="#6b7280", linestyle="-", linewidth=1, alpha=0.5, label="Mean (0)"
    )
    ax2.axhline(
        y=2,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Overvalued (+2σ)",
    )
    ax2.axhline(
        y=-2,
        color="#3b82f6",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Undervalued (-2σ)",
    )

    ax2.set_title("Bitcoin MVRV Z-Score", fontsize=16, fontweight="bold", pad=15)
    ax2.set_xlabel("Date", fontsize=12, fontweight="medium")
    ax2.set_ylabel("MVRV Z-Score", fontsize=12, fontweight="medium")
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # Add statistics text box for Z-Score
    zscore_mean = df_clean["CapMVRVZ"].mean()
    zscore_median = df_clean["CapMVRVZ"].median()
    zscore_min = df_clean["CapMVRVZ"].min()
    zscore_max = df_clean["CapMVRVZ"].max()
    zscore_current = df_clean["CapMVRVZ"].iloc[-1]
    zscore_ma30_current = zscore_ma30.iloc[-1]

    stats_text2 = (
        f"Current: {zscore_current:.2f}\n"
        f"30-Day MA: {zscore_ma30_current:.2f}\n"
        f"Mean: {zscore_mean:.2f}\n"
        f"Median: {zscore_median:.2f}\n"
        f"Range: {zscore_min:.2f} - {zscore_max:.2f}"
    )

    ax2.text(
        0.98,
        0.98,
        stats_text2,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            alpha=0.95,
            edgecolor="#e5e7eb",
            linewidth=1.5,
        ),
        family="monospace",
    )

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    date_range_days = (df_clean.index.max() - df_clean.index.min()).days
    if date_range_days > 365:
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    elif date_range_days > 90:
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_minor_locator(mdates.WeekdayLocator())
    else:
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(
        output_path,
        format=Path(output_path).suffix[1:] or "svg",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    logging.info(f"✓ Plot saved to {output_path}")
    logging.info(
        f"  Date range: {df_clean.index.min().date()} to {df_clean.index.max().date()}"
    )
    logging.info(f"  Data points: {len(df_clean)}")
    logging.info(
        f"  Current MVRV Ratio: {mvrv_current:.2f} (30-Day MA: {mvrv_ma30_current:.2f})"
    )
    logging.info(
        f"  Current MVRV Z-Score: {zscore_current:.2f} (30-Day MA: {zscore_ma30_current:.2f})"
    )


def main() -> None:
    """Main function to fetch data and create MVRV plots."""
    parser = argparse.ArgumentParser(
        description="Plot CoinMetrics MVRV metrics (CapMVRVCur and CapMVRVZ) over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_mvrv.py                           # Create plots with all available data
  python plot_mvrv.py --start 2020-01-01        # Filter from start date
  python plot_mvrv.py --start 2020-01-01 --end 2024-12-31  # Filter date range
  python plot_mvrv.py --output mvrv_analysis.png # Custom output filename
        """,
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date filter (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date filter (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mvrv_metrics.svg",
        help="Output filename (default: mvrv_metrics.svg)",
    )

    args = parser.parse_args()

    try:
        # Fetch CoinMetrics data
        logging.info("Fetching CoinMetrics BTC data...")
        df = fetch_coinmetrics_btc_csv()

        # Create plots
        plot_mvrv_metrics(
            df,
            start_date=args.start,
            end_date=args.end,
            output_path=args.output,
        )

        print(f"\n✓ Successfully created MVRV plots: {args.output}")

    except ValueError as e:
        logging.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        print(
            "\nTip: The CoinMetrics CSV may not include MVRV columns in all versions."
        )
        print("     Check available columns by running:")
        print(
            '     python -c "from stacksats.btc_api.coinmetrics_btc_csv import fetch_coinmetrics_btc_csv; df = fetch_coinmetrics_btc_csv(); print(list(df.columns))"'
        )
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

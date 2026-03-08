"""Plot BRK MVRV metrics over time.

Usage:
    stacksats-plot-mvrv                     # Creates plots with default settings
    stacksats-plot-mvrv --start 2020-01-01 # Filter by start date
    stacksats-plot-mvrv --output mvrv.png  # Custom output filename
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .data_btc import BTCDataProvider
from .matplotlib_setup import configure_matplotlib_env
from .plot_mvrv_render import plot_mvrv_metrics as _plot_mvrv_metrics

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _init_plot_env() -> None:
    configure_matplotlib_env()
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300


def plot_mvrv_metrics(
    df,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_path: str = "mvrv_metrics.svg",
) -> None:
    """Plot mvrv and CapMVRVZ metrics over time."""
    return _plot_mvrv_metrics(
        df,
        start_date,
        end_date,
        output_path,
        init_plot_env_fn=_init_plot_env,
        logging_mod=logging,
        plt_mod=plt,
        mdates_mod=mdates,
        path_cls=Path,
    )


def main() -> None:
    """Main function to fetch data and create MVRV plots."""
    _init_plot_env()
    parser = argparse.ArgumentParser(
        description="Plot BRK MVRV metrics (mvrv and CapMVRVZ) over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  stacksats-plot-mvrv                           # Create plots with all available data
  stacksats-plot-mvrv --start 2020-01-01        # Filter from start date
  stacksats-plot-mvrv --start 2020-01-01 --end 2024-12-31  # Filter date range
  stacksats-plot-mvrv --output mvrv_analysis.png # Custom output filename
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
        logging.info("Loading BRK BTC data...")
        df = BTCDataProvider().load()

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
        print("\nTip: Verify your BRK DuckDB has metrics_distribution.mvrv.")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

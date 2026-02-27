"""Plot DCA weights for a specified start_date and end_date pair from NeonDB.

Usage:
    stacksats-plot-weights                           # Uses oldest range
    stacksats-plot-weights 2025-01-01 2025-12-31    # Uses specified range
    stacksats-plot-weights --help                    # Shows help
"""

from __future__ import annotations

import argparse
import os
import sys
from types import ModuleType
from typing import Tuple

from .matplotlib_setup import configure_matplotlib_env
from .plot_weights_data import (
    fetch_weights_for_date_range as _fetch_weights_for_date_range,
)
from .plot_weights_data import get_date_range_options as _get_date_range_options
from .plot_weights_data import get_db_connection as _get_db_connection
from .plot_weights_data import get_oldest_date_range as _get_oldest_date_range
from .plot_weights_data import validate_date_range as _validate_date_range
from .plot_weights_render import plot_dca_weights as _plot_dca_weights

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

try:
    import psycopg2  # noqa: E402
except ImportError:  # pragma: no cover - exercised only without deploy extras
    psycopg2 = ModuleType("psycopg2")

    def _missing_connect(*_args, **_kwargs):
        raise ImportError(
            "Missing optional dependency 'psycopg2-binary'. "
            "Install deploy extras with: pip install stacksats[deploy]"
        )

    psycopg2.connect = _missing_connect
    sys.modules.setdefault("psycopg2", psycopg2)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _init_plot_env() -> None:
    configure_matplotlib_env()
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300


def get_db_connection():
    """Get database connection using DATABASE_URL environment variable."""
    return _get_db_connection(
        load_dotenv_fn=_load_dotenv_if_available,
        getenv_fn=os.getenv,
        psycopg2_module=psycopg2,
    )


def get_date_range_options(conn):
    """Get all available date range options from the database."""
    return _get_date_range_options(conn)


def get_oldest_date_range(conn) -> Tuple[str, str]:
    """Find the oldest start_date and its corresponding end_date."""
    return _get_oldest_date_range(conn)


def validate_date_range(conn, start_date: str, end_date: str) -> bool:
    """Check if the specified date range exists in the database."""
    return _validate_date_range(conn, start_date, end_date)


def fetch_weights_for_date_range(conn, start_date: str, end_date: str):
    """Fetch all DCA weights for a specific start_date and end_date pair."""
    return _fetch_weights_for_date_range(conn, start_date, end_date)


def plot_dca_weights(
    df,
    start_date: str,
    end_date: str,
    output_path: str = "oldest_weights_plot.svg",
):
    """Create and save a plot of DCA weights over time."""
    return _plot_dca_weights(
        df,
        start_date,
        end_date,
        output_path,
        init_plot_env_fn=_init_plot_env,
        plt_mod=plt,
        mdates_mod=mdates,
    )


def main():
    """Main function to plot DCA weights for specified or oldest date range."""
    _init_plot_env()
    parser = argparse.ArgumentParser(
        description="Plot DCA weights for a specified start_date and end_date pair from NeonDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  stacksats-plot-weights                           # Uses oldest range
  stacksats-plot-weights 2025-01-01 2025-12-31    # Uses specified range
  stacksats-plot-weights --list                    # Lists all available ranges
        """,
    )
    parser.add_argument(
        "start_date",
        nargs="?",
        help="Start date in YYYY-MM-DD format (optional, uses oldest if not specified)",
    )
    parser.add_argument(
        "end_date",
        nargs="?",
        help="End date in YYYY-MM-DD format (optional, uses oldest if not specified)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available date ranges and exit"
    )
    parser.add_argument(
        "--output",
        default="oldest_weights_plot.svg",
        help="Output filename (default: oldest_weights_plot.svg)",
    )

    args = parser.parse_args()

    print("Connecting to database...")
    conn = None

    try:
        conn = get_db_connection()

        if args.list:
            print("\nAvailable date ranges:")
            options = get_date_range_options(conn)
            for _, row in options.iterrows():
                print(
                    f"  {row['start_date'].date()} to {row['end_date'].date()} ({row['count']} weights)"
                )
            print(f"\nTotal ranges: {len(options)}")
            return

        if args.start_date and args.end_date:
            start_date = args.start_date
            end_date = args.end_date
            print(f"Using specified date range: {start_date} to {end_date}")

            if not validate_date_range(conn, start_date, end_date):
                print(
                    f"Error: Date range {start_date} to {end_date} not found in database."
                )
                print("Use --list to see available ranges.")
                sys.exit(1)
        else:
            print("Finding oldest date range...")
            start_date, end_date = get_oldest_date_range(conn)
            print(f"Using oldest date range: {start_date} to {end_date}")

        print("Fetching DCA weights...")
        df = fetch_weights_for_date_range(conn, start_date, end_date)

        output_path = args.output
        plot_dca_weights(df, start_date, end_date, output_path)

        print("\n✓ Successfully created DCA weights plot")
        if args.start_date:
            print(f"  Range: {start_date} to {end_date}")
        else:
            print("  Range: oldest available")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()

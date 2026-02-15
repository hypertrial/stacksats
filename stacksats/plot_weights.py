"""Plot DCA weights for a specified start_date and end_date pair from NeonDB.

This script connects to the NeonDB database and creates a visualization of the DCA weights over time.
Can be run with specific start_date and end_date arguments, or automatically uses the oldest range.

Usage:
    python plot_oldest_weights.py                           # Uses oldest range
    python plot_oldest_weights.py 2025-01-01 2025-12-31    # Uses specified range
    python plot_oldest_weights.py --help                    # Shows help
"""

import argparse
import os
import sys
from types import ModuleType
from typing import Tuple

from .matplotlib_setup import configure_matplotlib_env

configure_matplotlib_env()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass

# Set seaborn style for all plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300


def get_db_connection():
    """Get database connection using DATABASE_URL environment variable."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return psycopg2.connect(database_url)


def get_date_range_options(conn) -> pd.DataFrame:
    """Get all available date range options from the database.

    Returns:
        DataFrame with start_date, end_date, and count columns
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT start_date, end_date, COUNT(*) as count
            FROM bitcoin_dca
            GROUP BY start_date, end_date
            ORDER BY start_date ASC
        """)
        rows = cur.fetchall()

    if not rows:
        raise ValueError("No data found in bitcoin_dca table")

    df = pd.DataFrame(rows, columns=["start_date", "end_date", "count"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def get_oldest_date_range(conn) -> Tuple[str, str]:
    """Find the oldest start_date and its corresponding end_date.

    Returns:
        Tuple of (start_date, end_date) as strings
    """
    options = get_date_range_options(conn)
    oldest = options.iloc[0]
    return oldest["start_date"].date().isoformat(), oldest[
        "end_date"
    ].date().isoformat()


def validate_date_range(conn, start_date: str, end_date: str) -> bool:
    """Check if the specified date range exists in the database.

    Args:
        conn: Database connection
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        True if the date range exists
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM bitcoin_dca
            WHERE start_date = %s AND end_date = %s
        """,
            (start_date, end_date),
        )
        count = cur.fetchone()[0]
        return count > 0


def fetch_weights_for_date_range(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch all DCA weights for a specific start_date and end_date pair.

    Args:
        conn: Database connection
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        DataFrame with columns: date, weight, price_usd, day_index
    """
    query = """
        SELECT DCA_date AS date, weight, btc_usd AS price_usd, id AS day_index
        FROM bitcoin_dca
        WHERE start_date = %s AND end_date = %s
        ORDER BY DCA_date ASC
    """

    with conn.cursor() as cur:
        cur.execute(query, (start_date, end_date))
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No data found for date range {start_date} to {end_date}")

    df = pd.DataFrame(rows, columns=["date", "weight", "price_usd", "day_index"])
    df["date"] = pd.to_datetime(df["date"])

    return df


def plot_dca_weights(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_path: str = "oldest_weights_plot.svg",
):
    """Create and save a plot of DCA weights over time.

    Differentiates between:
    - Past weights (with price_usd price data) - computed by the model
    - Future weights (no price_usd price data) - placeholder/projected weights

    Args:
        df: DataFrame with date, weight, price_usd, day_index columns
        start_date: Start date string for plot title
        end_date: End date string for plot title
        output_path: Path to save the plot
    """
    # Ensure weights sum to 1.0 (normalize if necessary due to database update issues)
    original_weight_sum = df["weight"].sum()
    if original_weight_sum > 0 and abs(original_weight_sum - 1.0) > 1e-10:
        df = df.copy()
        df["weight"] = df["weight"] / original_weight_sum

    fig, ax = plt.subplots(figsize=(14, 8))

    # Split data into past (has price) and future (no price)
    past_df = df[df["price_usd"].notna()].copy()
    future_df = df[df["price_usd"].isna()].copy()

    has_past = len(past_df) > 0
    has_future = len(future_df) > 0

    # Calculate statistics for past weights only (model-computed)
    if has_past:
        past_weights = past_df["weight"].values
        past_mean = past_weights.mean()
        past_min = past_weights.min()
        past_max = past_weights.max()
        past_min_date = past_df.loc[past_df["weight"].idxmin(), "date"]
        past_max_date = past_df.loc[past_df["weight"].idxmax(), "date"]

    # Overall stats for y-axis limits
    all_weights = df["weight"].values
    min_weight = all_weights.min()
    max_weight = all_weights.max()

    # Plot PAST weights (model-computed) - solid blue
    if has_past:
        ax.fill_between(
            past_df["date"],
            past_df["weight"],
            alpha=0.3,
            color="#2563eb",
            label=f"Past Weights (n={len(past_df)})",
        )
        ax.plot(
            past_df["date"],
            past_df["weight"],
            linewidth=2.5,
            color="#1e40af",
            marker="o",
            markersize=3,
            markevery=max(1, len(past_df) // 30),
            zorder=3,
        )

    # Plot FUTURE weights (projected) - dashed orange
    if has_future:
        ax.fill_between(
            future_df["date"],
            future_df["weight"],
            alpha=0.2,
            color="#f97316",
            label=f"Future Weights (n={len(future_df)})",
        )
        ax.plot(
            future_df["date"],
            future_df["weight"],
            linewidth=2,
            color="#ea580c",
            linestyle="--",
            marker="s",
            markersize=2,
            markevery=max(1, len(future_df) // 30),
            alpha=0.8,
            zorder=3,
        )

    # Add vertical line at boundary between past and future
    if has_past and has_future:
        boundary_date = past_df["date"].max()
        ax.axvline(
            x=boundary_date,
            color="#6b7280",
            linestyle=":",
            linewidth=2,
            alpha=0.8,
            label=f"Today: {boundary_date.strftime('%Y-%m-%d')}",
            zorder=2,
        )

    # Add horizontal line for mean of PAST weights only
    if has_past:
        ax.axhline(
            y=past_mean,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Past Mean: {past_mean:.6f}",
            zorder=2,
        )

        # Highlight min and max of PAST weights
        ax.scatter(
            [past_min_date],
            [past_min],
            color="#16a34a",
            s=150,
            marker="v",
            edgecolors="white",
            linewidths=2,
            zorder=4,
            label=f"Past Min: {past_min:.6f}",
        )
        ax.scatter(
            [past_max_date],
            [past_max],
            color="#dc2626",
            s=150,
            marker="^",
            edgecolors="white",
            linewidths=2,
            zorder=4,
            label=f"Past Max: {past_max:.6f}",
        )

    # Format the plot
    ax.set_title(
        f"DCA Investment Weights Distribution\n{start_date} to {end_date}",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )
    ax.set_xlabel("DCA Date", fontsize=13, fontweight="medium")
    ax.set_ylabel("Investment Weight (log scale)", fontsize=13, fontweight="medium")

    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Format x-axis dates with better spacing
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(df) // 365)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=10)

    # Format y-axis with log scale
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(bottom=min_weight * 0.8, top=max_weight * 1.5)

    # Calculate statistics - separate past and future
    if has_past:
        past_stats = past_df["weight"].describe()
        past_p25 = past_df["weight"].quantile(0.25)
        past_p75 = past_df["weight"].quantile(0.75)
        past_median = past_df["weight"].median()

        stats_text = (
            f"Past Weight Stats (n={len(past_df)}):\n"
            f"Mean:   {past_stats['mean']:.6f}\n"
            f"Median: {past_median:.6f}\n"
            f"Std:    {past_stats['std']:.6f}\n"
            f"Min:    {past_stats['min']:.6f}\n"
            f"Max:    {past_stats['max']:.6f}\n"
            f"P25:    {past_p25:.6f}\n"
            f"P75:    {past_p75:.6f}"
        )
    else:
        stats_text = "No past weights available"

    if has_future:
        future_stats = future_df["weight"].describe()
        stats_text += (
            f"\n\nFuture Weights (n={len(future_df)}):\n"
            f"Mean:   {future_stats['mean']:.6f}\n"
            f"Min:    {future_stats['min']:.6f}\n"
            f"Max:    {future_stats['max']:.6f}"
        )

    # Position stats box in upper right
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
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
        zorder=5,
    )

    # Add legend
    ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.95,
        edgecolor="#e5e7eb",
        fancybox=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✓ Plot saved to {output_path}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Total data points: {len(df)}")
    print(f"  Past weights (with price): {len(past_df)}")
    print(f"  Future weights (no price): {len(future_df)}")

    # Report normalization if it occurred
    if abs(original_weight_sum - 1.0) > 1e-10:
        print(f"  ⚠ Weights normalized from {original_weight_sum:.6f} to 1.0")
    else:
        print("  Weights sum to 1.0 ✓")

    # Print detailed statistics to console
    if has_past:
        past_stats = past_df["weight"].describe()
        past_p25 = past_df["weight"].quantile(0.25)
        past_p75 = past_df["weight"].quantile(0.75)
        past_median = past_df["weight"].median()

        print(f"\nPast Weight Statistics (n={len(past_df)}):")
        print(f"  Mean:   {past_stats['mean']:.6f}")
        print(f"  Median: {past_median:.6f}")
        print(f"  Std:    {past_stats['std']:.6f}")
        print(f"  Min:    {past_stats['min']:.6f}")
        print(f"  Max:    {past_stats['max']:.6f}")
        print(f"  P25:    {past_p25:.6f}")
        print(f"  P75:    {past_p75:.6f}")
        print(f"  Range:  {past_max - past_min:.6f}")

    if has_future:
        future_stats = future_df["weight"].describe()
        print(f"\nFuture Weight Statistics (n={len(future_df)}):")
        print(f"  Mean:   {future_stats['mean']:.6f}")
        print(f"  Min:    {future_stats['min']:.6f}")
        print(f"  Max:    {future_stats['max']:.6f}")
        print(f"  Range:  {future_stats['max'] - future_stats['min']:.6f}")

    if has_past:
        print("\nSummary:")
        print(f"  Past mean weight: {past_mean:.6f}")
        print(f"  Past weight range: {past_min:.6f} to {past_max:.6f}")


def main():
    """Main function to plot DCA weights for specified or oldest date range."""
    parser = argparse.ArgumentParser(
        description="Plot DCA weights for a specified start_date and end_date pair from NeonDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_oldest_weights.py                           # Uses oldest range
  python plot_oldest_weights.py 2025-01-01 2025-12-31    # Uses specified range
  python plot_oldest_weights.py --list                    # Lists all available ranges
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

        # List available ranges if requested
        if args.list:
            print("\nAvailable date ranges:")
            options = get_date_range_options(conn)
            for _, row in options.iterrows():
                print(
                    f"  {row['start_date'].date()} to {row['end_date'].date()} ({row['count']} weights)"
                )
            print(f"\nTotal ranges: {len(options)}")
            return

        # Determine which date range to use
        if args.start_date and args.end_date:
            start_date = args.start_date
            end_date = args.end_date
            print(f"Using specified date range: {start_date} to {end_date}")

            # Validate the range exists
            if not validate_date_range(conn, start_date, end_date):
                print(
                    f"Error: Date range {start_date} to {end_date} not found in database."
                )
                print("Use --list to see available ranges.")
                sys.exit(1)
        else:
            # Get the oldest date range
            print("Finding oldest date range...")
            start_date, end_date = get_oldest_date_range(conn)
            print(f"Using oldest date range: {start_date} to {end_date}")

        # Fetch weights for this date range
        print("Fetching DCA weights...")
        df = fetch_weights_for_date_range(conn, start_date, end_date)

        # Create and save plot
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

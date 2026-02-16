"""Export daily model weights and BTC prices for multiple date ranges.

Generates weights for every day from RANGE_START to RANGE_END.
Core business logic that can run locally or be imported by runtime functions.

Weight computation strategy:
- Past dates (up to current_date): Use ML model weights
- Future dates (after current_date): Uniform weights for remaining budget
- Total always sums to 1.0 without normalization
- Each day the export process runs, future uniform weights are recalculated
"""

import sys
from types import ModuleType

import numpy as np
import pandas as pd
try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:  # pragma: no cover - exercised only without deploy extras
    psycopg2 = ModuleType("psycopg2")
    sys.modules.setdefault("psycopg2", psycopg2)
    execute_values = None

from .btc_price_fetcher import fetch_btc_price_robust
from .export_weights_db import (
    create_table_if_not_exists as _create_table_if_not_exists,
    get_db_connection as _get_db_connection,
    sql_quote as _db_sql_quote,
    table_is_empty as _table_is_empty,
    today_data_exists as _today_data_exists,
)
from .framework_contract import validate_span_length
from .model_development import compute_window_weights
from .prelude import generate_date_ranges, group_ranges_by_start_date  # noqa: F401
from .strategy_types import BaseStrategy, StrategyContext, validate_strategy_contract


def _missing_psycopg2(*_args, **_kwargs):
    raise ImportError(
        "Missing optional dependency 'psycopg2-binary'. "
        "Install deploy extras with: pip install stacksats[deploy]"
    )


if not hasattr(psycopg2, "connect"):
    psycopg2.connect = _missing_psycopg2


# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass

# Configuration constants
RANGE_START = "2025-12-01"  # First day of December 2025
RANGE_END = "2027-12-31"  # Last day of 2027
MIN_RANGE_LENGTH_DAYS = 120  # Minimum range length
DATE_FREQ = "D"  # Daily frequency
BTC_PRICE_COL = "PriceUSD_coinmetrics"  # Use CoinMetrics PriceUSD data


def _require_deploy_dependency(name: str, imported_obj):
    """Raise a consistent error when deploy-only dependencies are missing."""
    if imported_obj is None:
        raise ImportError(
            f"Missing optional dependency '{name}'. "
            "Install deploy extras with: pip install stacksats[deploy]"
        )


def _sql_quote(value) -> str:
    """Quote values for dynamic VALUES SQL when psycopg2 adapt isn't available."""
    return _db_sql_quote(value)


def process_start_date_batch(
    start_date,
    end_dates,
    features_df,
    btc_df,
    current_date,
    btc_price_col,
    strategy=None,
    locked_weights_by_end_date: dict[str, np.ndarray] | None = None,
    enforce_span_contract: bool = True,
):
    """Process all date ranges sharing the same start_date.

    Uses the shared compute_window_weights() from model_development.py for
    weight computation to ensure parity between backtest and production.

    Args:
        start_date: Shared start date
        end_dates: List of end dates
        features_df: DataFrame with precomputed features
        btc_df: Original BTC DataFrame (for price data)
        current_date: Current date (determines past/future boundary)
        btc_price_col: Column name for BTC price

    Returns:
        DataFrame with canonical columns:
        day_index, start_date, end_date, date, price_usd, weight
    """
    if strategy is not None:
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("strategy must subclass BaseStrategy.")
        validate_strategy_contract(strategy)

    results = []

    for end_date in end_dates:
        if enforce_span_contract:
            validate_span_length(start_date, end_date)
        full_range = pd.date_range(start=start_date, end=end_date, freq="D")
        n_total = len(full_range)
        end_date_key = end_date.strftime("%Y-%m-%d")
        locked_weights = None
        if locked_weights_by_end_date is not None:
            locked_weights = locked_weights_by_end_date.get(end_date_key)

        # Compute weights using either provided strategy or default model function.
        if strategy is None:
            weights = compute_window_weights(
                features_df,
                start_date,
                end_date,
                current_date,
                locked_weights=locked_weights,
            )
        else:
            weights = strategy.compute_weights(
                StrategyContext(
                    features_df=features_df,
                    start_date=start_date,
                    end_date=end_date,
                    current_date=current_date,
                    locked_weights=locked_weights,
                )
            )
            if weights.empty:
                weights = compute_window_weights(
                    features_df,
                    start_date,
                    end_date,
                    current_date,
                    locked_weights=locked_weights,
                )
            else:
                weights = weights.reindex(full_range).fillna(0.0)

        # Get prices: past prices are known, future prices are NaN
        range_prices = btc_df[btc_price_col].reindex(full_range).values

        # Create DataFrame for this range
        range_df = pd.DataFrame(
            {
                "day_index": range(n_total),
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "date": full_range.strftime("%Y-%m-%d"),
                "price_usd": range_prices,
                "weight": weights.values,
            }
        )
        results.append(range_df)

    return pd.concat(results, ignore_index=True)


def load_locked_weights_for_window(
    conn,
    start_date: str,
    end_date: str,
    lock_end_date: str,
) -> np.ndarray | None:
    """Load immutable locked prefix from DB for a specific allocation window.

    The returned array is validated to be a contiguous prefix from start_date
    through min(lock_end_date, end_date), with no missing DB days.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    lock_end_ts = min(pd.to_datetime(lock_end_date), end_ts)
    if lock_end_ts < start_ts:
        return None

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DCA_date, weight
            FROM bitcoin_dca
            WHERE start_date = %s
              AND end_date = %s
              AND DCA_date >= %s
              AND DCA_date <= %s
            ORDER BY DCA_date ASC
            """,
            (start_date, end_date, start_date, lock_end_date),
        )
        rows = cur.fetchall()
    if not rows:
        return None

    expected_dates = pd.date_range(start=start_ts, end=lock_end_ts, freq="D")
    actual_dates = pd.to_datetime([row[0] for row in rows]).normalize()
    if len(actual_dates) != len(expected_dates) or not actual_dates.equals(expected_dates):
        missing = expected_dates.difference(actual_dates)
        extra = actual_dates.difference(expected_dates)
        raise ValueError(
            "Locked history is not a contiguous prefix for "
            f"{start_date}..{end_date}. Missing={list(missing.strftime('%Y-%m-%d'))}, "
            f"extra={list(extra.strftime('%Y-%m-%d'))}"
        )

    return np.array([float(row[1]) for row in rows], dtype=float)


def get_db_connection():
    """Get database connection using DATABASE_URL environment variable."""
    if psycopg2 is None:
        _missing_psycopg2()
    return _get_db_connection(psycopg2)


def create_table_if_not_exists(conn):
    """Create bitcoin_dca table if it doesn't already exist."""
    _create_table_if_not_exists(conn)


def table_is_empty(conn):
    """Check if bitcoin_dca table is empty."""
    return _table_is_empty(conn)


def today_data_exists(conn, today_str):
    """Check if data exists for today's date in bitcoin_dca table.

    Args:
        conn: Database connection
        today_str: Date string in YYYY-MM-DD format

    Returns:
        bool: True if data exists for today, False otherwise
    """
    return _today_data_exists(conn, today_str)


def get_current_btc_price(previous_price=None):
    """Fetch current BTC price using robust fetcher with retry logic and multiple sources.

    Args:
        previous_price: Optional previous price for validation/sanity checks

    Returns:
        float: Current BTC price in USD, or None if all sources fail
    """
    import logging

    logging.info("Fetching current BTC price with retry logic and multiple sources...")
    price_usd = fetch_btc_price_robust(previous_price=previous_price)

    if price_usd is None:
        logging.error("Failed to fetch BTC price from all available sources")
    else:
        logging.info(f"Successfully fetched current BTC price: ${price_usd:,.2f}")

    return price_usd


def insert_all_data(conn, df):
    """Insert all data into bitcoin_dca table using optimized bulk insertion.

    Uses COPY FROM for maximum performance, falling back to execute_values if COPY fails.
    """
    import logging
    import time
    from io import StringIO

    _require_deploy_dependency("psycopg2-binary", execute_values)

    total_rows = len(df)
    logging.info(f"Starting bulk insertion of {total_rows} rows into bitcoin_dca table")

    # Get count before insertion
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM bitcoin_dca")
        count_before = cur.fetchone()[0]

    start_time = time.time()

    # Try using COPY FROM to temp table, then INSERT with ON CONFLICT (fastest method)
    try:
        logging.info("Attempting fast bulk insert using COPY FROM with temp table...")

        # Prepare data as tab-separated string for COPY (much faster than row-by-row)
        # Use pandas to_csv for efficient conversion
        buffer = StringIO()
        # Select and format columns for COPY
        copy_df = df[
            ["day_index", "start_date", "end_date", "date", "price_usd", "weight"]
        ].copy()
        copy_df = copy_df.rename(
            columns={
                "day_index": "id",
                "date": "DCA_date",
                "price_usd": "btc_usd",
            }
        )
        # Convert to proper types and handle NaN
        copy_df["id"] = copy_df["id"].astype(int)
        copy_df["btc_usd"] = copy_df["btc_usd"].where(pd.notna(copy_df["btc_usd"]), "")
        copy_df["weight"] = copy_df["weight"].where(pd.notna(copy_df["weight"]), "")
        # Write as tab-separated (no header, no index)
        copy_df.to_csv(buffer, sep="\t", header=False, index=False, na_rep="")
        buffer.seek(0)

        with conn.cursor() as cur:
            # Create temporary table with same structure
            cur.execute("""
                CREATE TEMP TABLE temp_bitcoin_dca (
                    id INTEGER,
                    start_date DATE,
                    end_date DATE,
                    DCA_date DATE,
                    btc_usd FLOAT,
                    weight FLOAT
                ) ON COMMIT DROP
            """)

            # Use COPY FROM to temp table (much faster than INSERT)
            cur.copy_from(
                buffer,
                "temp_bitcoin_dca",
                columns=(
                    "id",
                    "start_date",
                    "end_date",
                    "DCA_date",
                    "btc_usd",
                    "weight",
                ),
                null="",  # Empty string represents NULL
            )

            # Insert from temp table with ON CONFLICT handling
            cur.execute("""
                INSERT INTO bitcoin_dca (id, start_date, end_date, DCA_date, btc_usd, weight)
                SELECT id, start_date, end_date, DCA_date, btc_usd, weight
                FROM temp_bitcoin_dca
                ON CONFLICT (id, start_date, end_date, DCA_date) DO NOTHING
            """)

            conn.commit()

        # Get final count after insertion
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bitcoin_dca")
            final_count = cur.fetchone()[0]

        actual_inserted = final_count - count_before
        elapsed = time.time() - start_time
        logging.info(
            f"✓ COPY FROM completed: {actual_inserted} rows inserted in {elapsed:.2f}s "
            f"({actual_inserted / elapsed:.0f} rows/sec)"
        )

        return actual_inserted

    except Exception as e:
        # Fallback to execute_values if COPY fails (e.g., permission issues, remote DB)
        logging.warning(
            f"COPY FROM failed ({e}), falling back to execute_values method..."
        )
        conn.rollback()

        # Prepare data for insertion
        logging.info("Preparing data for bulk insertion...")
        data = [
            (
                int(row["day_index"]),
                row["start_date"],
                row["end_date"],
                row["date"],
                float(row["price_usd"]) if pd.notna(row["price_usd"]) else None,
                float(row["weight"]) if pd.notna(row["weight"]) else None,
            )
            for _, row in df.iterrows()
        ]

        # Use larger batches and commit less frequently for better performance
        batch_size = 50000  # Increased from 10,000 for better throughput
        total_batches = (len(data) + batch_size - 1) // batch_size

        logging.info(
            f"Inserting {len(data)} rows in {total_batches} batches of {batch_size}..."
        )

        with conn.cursor() as cur:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                if batch_num % 10 == 0 or batch_num == total_batches:
                    logging.info(
                        f"Processing batch {batch_num}/{total_batches} ({len(batch)} rows)..."
                    )

                # Use execute_values with explicit page_size for better performance
                execute_values(
                    cur,
                    """
                    INSERT INTO bitcoin_dca (id, start_date, end_date, DCA_date, btc_usd, weight)
                    VALUES %s
                    ON CONFLICT (id, start_date, end_date, DCA_date) DO NOTHING
                    """,
                    batch,
                    page_size=len(batch),  # Process entire batch at once
                )

                # Commit every batch (or less frequently for even better performance)
                if batch_num % 5 == 0 or batch_num == total_batches:
                    conn.commit()
                    if batch_num % 10 == 0 or batch_num == total_batches:
                        logging.info(f"  ✓ Committed through batch {batch_num}")

        # Final commit
        conn.commit()

        # Get final count after insertion
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bitcoin_dca")
            final_count = cur.fetchone()[0]

        actual_inserted = final_count - count_before
        elapsed = time.time() - start_time
        logging.info(
            f"Bulk insertion completed: {actual_inserted} rows inserted in {elapsed:.2f}s "
            f"({actual_inserted / elapsed:.0f} rows/sec)"
        )

        return actual_inserted


def update_today_weights(conn, df, today_str):
    """Update weight and BTC price columns for rows where date equals today.

    Fetches current BTC price using robust fetcher with retry logic and multiple sources,
    and updates both weight and btc_usd columns for all rows matching today's date.
    Uses bulk SQL UPDATE for efficiency.
    """
    import logging
    import time

    required_cols = {"day_index", "start_date", "end_date", "date", "price_usd", "weight"}
    missing_cols = sorted(col for col in required_cols if col not in df.columns)
    if missing_cols:
        raise ValueError(
            "update_today_weights requires canonical columns: "
            + ", ".join(sorted(required_cols))
            + f". Missing: {', '.join(missing_cols)}"
        )

    # Try to get previous day's price from database for validation
    previous_price = None
    try:
        today_date = pd.to_datetime(today_str)
        previous_day_str = (today_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT btc_usd FROM bitcoin_dca
                WHERE DCA_date = %s AND btc_usd IS NOT NULL
                LIMIT 1
                """,
                (previous_day_str,),
            )
            result = cur.fetchone()
            if result and result[0] is not None:
                previous_price = float(result[0])
                logging.info(
                    f"Found previous day's price for validation: ${previous_price:,.2f}"
                )
    except Exception as e:
        logging.debug(f"Could not fetch previous day's price for validation: {e}")

    # Fetch current BTC price with retry logic and multiple sources
    current_btc_price = get_current_btc_price(previous_price=previous_price)
    if current_btc_price is None:
        logging.warning(
            "Failed to fetch BTC price from all API sources. Will use price from dataframe if available."
        )
        # Fallback to price from dataframe if available
        day_mask = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d") == today_str
        today_df_temp = df[day_mask]
        if not today_df_temp.empty:
            current_btc_price = today_df_temp["price_usd"].iloc[0]
            if pd.notna(current_btc_price):
                logging.info(
                    f"Using BTC price from dataframe: ${current_btc_price:,.2f}"
                )
            else:
                logging.error(
                    "No BTC price available from API sources or dataframe. Skipping BTC price update."
                )
                current_btc_price = None
        else:
            logging.error(
                "No BTC price available from API sources or dataframe. Skipping BTC price update."
            )
            current_btc_price = None

    # Filter for today's data
    logging.info(f"Filtering data for date = {today_str}")
    day_mask = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d") == today_str
    today_df = df[day_mask].copy()

    if today_df.empty:
        logging.warning(f"No data found for today ({today_str})")
        return 0

    # GUARD: If we don't have a valid price for today (neither from API nor from DataFrame),
    # skip the update to avoid writing invalid/zero weights.
    if current_btc_price is None:
        # Check if dataframe has valid prices
        if today_df["price_usd"].notna().any():
            logging.info("Using existing BTC prices from dataframe for update.")
        else:
            logging.warning(
                f"Skipping DB update for {today_str}: No valid BTC price available from API or DataFrame. "
                "Preventing overwrite with invalid weights."
            )
            return 0

    total_rows = len(today_df)
    logging.info(f"Found {total_rows} rows to update where date = {today_str}")

    # Log some sample data to verify
    sample_row = today_df.iloc[0]
    logging.info(
        f"Sample row - day_index: {sample_row['day_index']}, start_date: {sample_row['start_date']}, end_date: {sample_row['end_date']}, weight: {sample_row['weight']:.6f}"
    )
    if current_btc_price is not None:
        logging.info(f"Will update BTC price to: ${current_btc_price:,.2f}")

    start_time = time.time()

    # Prepare data for bulk update
    if current_btc_price is not None:
        # Prepare tuples: (id, start_date, end_date, DCA_date, weight, btc_usd)
        update_data = [
            (
                int(row["day_index"]),
                row["start_date"],
                row["end_date"],
                row["date"],
                float(row["weight"]) if pd.notna(row["weight"]) else None,
                current_btc_price,
            )
            for _, row in today_df.iterrows()
        ]
    else:
        # Fallback: only update weight if BTC price fetch failed
        update_data = [
            (
                int(row["day_index"]),
                row["start_date"],
                row["end_date"],
                row["date"],
                float(row["weight"]) if pd.notna(row["weight"]) else None,
            )
            for _, row in today_df.iterrows()
        ]

    # Use bulk UPDATE with VALUES clause for efficiency
    batch_size = 10000  # Larger batch size since we're using bulk operations
    total_updated = 0

    logging.info(
        f"Starting bulk weight and BTC price updates in batches of {batch_size}..."
    )

    with conn.cursor() as cur:
        for i in range(0, len(update_data), batch_size):
            batch = update_data[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(update_data) + batch_size - 1) // batch_size

            batch_start_time = time.time()

            logging.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} rows)..."
            )

            if current_btc_price is not None:
                values_list = []
                for row in batch:
                    (
                        id_val,
                        start_date_val,
                        end_date_val,
                        DCA_date_val,
                        weight_val,
                        btc_usd_val,
                    ) = row
                    # Properly escape and format values
                    weight_sql = _sql_quote(weight_val)
                    btc_usd_sql = _sql_quote(btc_usd_val)
                    values_list.append(
                        f"({id_val}, {_sql_quote(start_date_val)}::date, "
                        f"{_sql_quote(end_date_val)}::date, "
                        f"{_sql_quote(DCA_date_val)}::date, "
                        f"{weight_sql}::float, {btc_usd_sql}::float)"
                    )
                values_str = ", ".join(values_list)

                cur.execute(
                    f"""
                    UPDATE bitcoin_dca AS t
                    SET weight = v.weight, btc_usd = v.btc_usd
                    FROM (VALUES {values_str}) AS v(id, start_date, end_date, DCA_date, weight, btc_usd)
                    WHERE t.id = v.id
                    AND t.start_date = v.start_date
                    AND t.end_date = v.end_date
                    AND t.DCA_date = v.DCA_date
                    """
                )
            else:
                values_list = []
                for row in batch:
                    id_val, start_date_val, end_date_val, DCA_date_val, weight_val = row
                    # Properly escape and format values
                    weight_sql = _sql_quote(weight_val)
                    values_list.append(
                        f"({id_val}, {_sql_quote(start_date_val)}::date, "
                        f"{_sql_quote(end_date_val)}::date, "
                        f"{_sql_quote(DCA_date_val)}::date, "
                        f"{weight_sql}::float)"
                    )
                values_str = ", ".join(values_list)

                cur.execute(
                    f"""
                    UPDATE bitcoin_dca AS t
                    SET weight = v.weight
                    FROM (VALUES {values_str}) AS v(id, start_date, end_date, DCA_date, weight)
                    WHERE t.id = v.id
                    AND t.start_date = v.start_date
                    AND t.end_date = v.end_date
                    AND t.DCA_date = v.DCA_date
                    """
                )

            batch_updated = cur.rowcount
            total_updated += batch_updated

            batch_time = time.time() - batch_start_time
            logging.info(
                f"  ✓ Batch {batch_num}/{total_batches} completed: {batch_updated} rows updated in {batch_time:.2f}s"
            )

        # Final commit
        logging.info("Committing all weight and BTC price updates...")
        conn.commit()
        commit_time = time.time() - start_time

    logging.info(
        f"Update completed. Total rows updated: {total_updated} in {commit_time:.2f}s"
    )
    logging.info(f"Average update rate: {total_updated / commit_time:.1f} rows/second")
    return total_updated

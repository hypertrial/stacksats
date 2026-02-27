"""Data-access helpers for plot_weights CLI/runtime."""

from __future__ import annotations

import pandas as pd


def get_db_connection(*, load_dotenv_fn, getenv_fn, psycopg2_module):
    """Get database connection using DATABASE_URL environment variable."""
    load_dotenv_fn()
    database_url = getenv_fn("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return psycopg2_module.connect(database_url)


def get_date_range_options(conn) -> pd.DataFrame:
    """Get all available date range options from the database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT start_date, end_date, COUNT(*) as count
            FROM bitcoin_dca
            GROUP BY start_date, end_date
            ORDER BY start_date ASC
            """
        )
        rows = cur.fetchall()

    if not rows:
        raise ValueError("No data found in bitcoin_dca table")

    df = pd.DataFrame(rows, columns=["start_date", "end_date", "count"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def get_oldest_date_range(conn):
    """Find the oldest start_date and its corresponding end_date."""
    options = get_date_range_options(conn)
    oldest = options.iloc[0]
    return oldest["start_date"].date().isoformat(), oldest["end_date"].date().isoformat()


def validate_date_range(conn, start_date: str, end_date: str) -> bool:
    """Check if the specified date range exists in the database."""
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
    """Fetch all DCA weights for a specific start_date and end_date pair."""
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

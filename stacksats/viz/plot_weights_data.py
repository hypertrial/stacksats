"""Data-access helpers for plot_weights CLI/runtime."""

from __future__ import annotations

import polars as pl


def get_db_connection(*, load_dotenv_fn, getenv_fn, psycopg2_module):
    """Get database connection using DATABASE_URL environment variable."""
    load_dotenv_fn()
    database_url = getenv_fn("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return psycopg2_module.connect(database_url)


def get_date_range_options(conn) -> pl.DataFrame:
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

    cols = ["start_date", "end_date", "count"]
    df = pl.DataFrame(
        {col: [r[i] for r in rows] for i, col in enumerate(cols)}
    )
    for col in ["start_date", "end_date"]:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(col).str.to_datetime().dt.replace_time_zone(None)
            )
    return df


def get_oldest_date_range(conn):
    """Find the oldest start_date and its corresponding end_date."""
    options = get_date_range_options(conn)
    first = options.row(0, named=True)
    start_dt = first["start_date"]
    end_dt = first["end_date"]
    if hasattr(start_dt, "date"):
        start_str = start_dt.date().isoformat()
    else:
        start_str = str(start_dt)[:10]
    if hasattr(end_dt, "date"):
        end_str = end_dt.date().isoformat()
    else:
        end_str = str(end_dt)[:10]
    return start_str, end_str


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


def fetch_weights_for_date_range(conn, start_date: str, end_date: str) -> pl.DataFrame:
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

    cols = ["date", "weight", "price_usd", "day_index"]
    df = pl.DataFrame({col: [r[i] for r in rows] for i, col in enumerate(cols)})
    df = df.with_columns(
        pl.col("date").str.to_datetime().dt.replace_time_zone(None)
    )
    return df

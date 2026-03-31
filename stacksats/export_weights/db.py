"""Database helper functions for export_weights."""

from __future__ import annotations

import os

import numpy as np


def sql_quote(value) -> str:
    """Quote values for dynamic VALUES SQL when psycopg2 adapt isn't available."""
    if value is None:
        return "NULL"
    if isinstance(value, (float, np.floating)):
        return str(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value).replace("'", "''")
    return f"'{text}'"


def get_db_connection(psycopg2_module):
    """Get database connection using DATABASE_URL environment variable."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return psycopg2_module.connect(database_url)


def create_table_if_not_exists(conn):
    """Create bitcoin_dca table if it doesn't already exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bitcoin_dca (
                id INTEGER,
                start_date DATE,
                end_date DATE,
                DCA_date DATE,
                btc_usd FLOAT,
                weight FLOAT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, start_date, end_date, DCA_date)
            )
        """
        )
        cur.execute(
            """
            CREATE OR REPLACE FUNCTION update_bitcoin_dca_last_updated()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.last_updated = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """
        )
        cur.execute(
            """
            DROP TRIGGER IF EXISTS bitcoin_dca_last_updated_trigger ON bitcoin_dca;
            CREATE TRIGGER bitcoin_dca_last_updated_trigger
                BEFORE UPDATE ON bitcoin_dca
                FOR EACH ROW
                EXECUTE FUNCTION update_bitcoin_dca_last_updated();
        """
        )
        conn.commit()


def table_is_empty(conn):
    """Check if bitcoin_dca table is empty."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM bitcoin_dca")
        count = cur.fetchone()[0]
        return count == 0


def today_data_exists(conn, today_str):
    """Check if data exists for today's date in bitcoin_dca table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM bitcoin_dca
            WHERE DCA_date = %s AND btc_usd IS NOT NULL AND weight > 0
            """,
            (today_str,),
        )
        count = cur.fetchone()[0]
        return count > 0

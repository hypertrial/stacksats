"""Export daily model weights and BTC prices for multiple date ranges.

This public module preserves historical imports while delegating heavy internals
to focused helper modules.
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:  # pragma: no cover - exercised only without deploy extras
    psycopg2 = ModuleType("psycopg2")
    sys.modules.setdefault("psycopg2", psycopg2)
    execute_values = None

from .btc_price_fetcher import fetch_btc_price_robust
from .export_weights_core import (
    load_locked_weights_for_window as _load_locked_weights_for_window,
)
from .export_weights_core import (
    process_start_date_batch as _process_start_date_batch,
)
from .export_weights_db import create_table_if_not_exists as _create_table_if_not_exists
from .export_weights_db import get_db_connection as _get_db_connection
from .export_weights_db import sql_quote as _db_sql_quote
from .export_weights_db import table_is_empty as _table_is_empty
from .export_weights_db import today_data_exists as _today_data_exists
from .export_weights_runtime import get_current_btc_price as _get_current_btc_price
from .export_weights_runtime import insert_all_data as _insert_all_data
from .export_weights_runtime import update_today_weights as _update_today_weights
from .export_weights_sql import (
    build_insert_rows,
    build_update_rows,
    build_values_sql,
    prepare_copy_dataframe,
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


def _load_dotenv_if_available() -> None:
    """Best-effort dotenv loading for runtime DB entrypoints."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


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
    """Process all date ranges sharing the same start_date."""
    return _process_start_date_batch(
        start_date,
        end_dates,
        features_df,
        btc_df,
        current_date,
        btc_price_col,
        strategy=strategy,
        locked_weights_by_end_date=locked_weights_by_end_date,
        enforce_span_contract=enforce_span_contract,
        compute_window_weights_fn=compute_window_weights,
        validate_span_length_fn=validate_span_length,
        strategy_context_cls=StrategyContext,
        base_strategy_cls=BaseStrategy,
        validate_strategy_contract_fn=validate_strategy_contract,
    )


def load_locked_weights_for_window(
    conn,
    start_date: str,
    end_date: str,
    lock_end_date: str,
) -> np.ndarray | None:
    """Load immutable locked prefix from DB for a specific allocation window."""
    return _load_locked_weights_for_window(
        conn,
        start_date=start_date,
        end_date=end_date,
        lock_end_date=lock_end_date,
    )


def get_db_connection():
    """Get database connection using DATABASE_URL environment variable."""
    _load_dotenv_if_available()
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
    """Check if data exists for today's date in bitcoin_dca table."""
    return _today_data_exists(conn, today_str)


def get_current_btc_price(previous_price=None):
    """Fetch current BTC price using robust fetcher with retry logic and multiple sources."""
    return _get_current_btc_price(
        previous_price=previous_price,
        fetch_btc_price_fn=fetch_btc_price_robust,
    )


def insert_all_data(conn, df):
    """Insert all data into bitcoin_dca table using optimized bulk insertion."""
    return _insert_all_data(
        conn,
        df,
        execute_values=execute_values,
        prepare_copy_dataframe_fn=prepare_copy_dataframe,
        build_insert_rows_fn=build_insert_rows,
    )


def update_today_weights(conn, df, today_str):
    """Update weight and BTC price columns for rows where date equals today."""
    return _update_today_weights(
        conn,
        df,
        today_str,
        get_current_btc_price_fn=get_current_btc_price,
        build_update_rows_fn=build_update_rows,
        build_values_sql_fn=build_values_sql,
        sql_quote_fn=_sql_quote,
    )

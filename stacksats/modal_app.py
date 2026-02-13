"""Modal app for exporting model weights and BTC prices.

Deploys web endpoints and scheduled jobs on Modal.
"""

import os
from pathlib import Path

import numpy as np

try:
    import modal
except ImportError:  # pragma: no cover - used in local/test environments
    class _LocalModalFunction:
        """Local stand-in that mimics Modal function wrappers for tests."""

        def __init__(self, fn):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def get_raw_f(self):
            return self._fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def map(self, iterable):
            return (self._fn(item) for item in iterable)

    class _LocalApp:
        def __init__(self, _name):
            self._name = _name

        def function(self, *args, **kwargs):
            del args, kwargs

            def decorator(fn):
                return _LocalModalFunction(fn)

            return decorator

        def local_entrypoint(self, *args, **kwargs):
            del args, kwargs

            def decorator(fn):
                return fn

            return decorator

    class _LocalSecret:
        @classmethod
        def from_name(cls, _name):
            return cls()

    class _LocalImage:
        @classmethod
        def debian_slim(cls, **kwargs):
            del kwargs
            return cls()

        def pip_install(self, *packages):
            del packages
            return self

        def add_local_dir(self, *_args, **_kwargs):
            return self

        def add_local_file(self, *_args, **_kwargs):
            return self

    class _LocalCron:
        def __init__(self, _expr):
            self.expr = _expr

    class _ModalShim:
        App = _LocalApp
        Secret = _LocalSecret
        Image = _LocalImage
        Cron = _LocalCron

    modal = _ModalShim()

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass


# Optional strategy override loaded from environment.
# Supports either module path (`pkg.module:Class`) or file path (`my_strategy.py:Class`).
RAW_STRATEGY_SPEC = os.getenv("STACKSATS_STRATEGY")
MODAL_STRATEGY_SPEC = RAW_STRATEGY_SPEC

# Create Modal app and image
app = modal.App("export-weights")
secret = modal.Secret.from_name("bitcoin-weights-secret")
database_secret = modal.Secret.from_name("database_url")

# Create image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas",
        "numpy",
        "scipy",
        "requests",
        "pyarrow",  # for parquet support
        "fastapi",  # for web endpoint responses
        "psycopg2-binary",  # for PostgreSQL/NeonDB connectivity
        "python-dotenv",  # for loading .env files
        "tenacity",  # for retry logic
    )
    .add_local_dir("stacksats", "/root/stacksats")
)

if RAW_STRATEGY_SPEC and ":" in RAW_STRATEGY_SPEC:
    strategy_module_or_path, strategy_class_name = RAW_STRATEGY_SPEC.rsplit(":", 1)
    if strategy_module_or_path.endswith(".py"):
        local_strategy_path = Path(strategy_module_or_path).expanduser().resolve()
        if local_strategy_path.exists():
            container_strategy_path = f"/root/{local_strategy_path.name}"
            image = image.add_local_file(str(local_strategy_path), container_strategy_path)
            MODAL_STRATEGY_SPEC = f"{container_strategy_path}:{strategy_class_name}"


@app.function(image=image)
def process_start_date_batch_modal(args_tuple):
    """Process a batch of date ranges sharing the same start date on Modal worker.

    Args:
        args_tuple: Tuple of (start_date_str, end_date_strs_list, current_date_str, btc_price_col, features_df_pickle, btc_df_pickle, strategy_spec)
    """
    import pickle
    import sys

    sys.path.insert(0, "/root")

    import pandas as pd

    from .export_weights import process_start_date_batch
    from .loader import load_strategy

    if len(args_tuple) == 8:
        (
            start_date_str,
            end_date_strs,
            current_date_str,
            btc_price_col,
            features_df_pickle,
            btc_df_pickle,
            strategy_spec,
            locked_weights_by_end_date,
        ) = args_tuple
    elif len(args_tuple) == 7:
        (
            start_date_str,
            end_date_strs,
            current_date_str,
            btc_price_col,
            features_df_pickle,
            btc_df_pickle,
            trailing_arg,
        ) = args_tuple
        if isinstance(trailing_arg, (dict, type(None))):
            strategy_spec = None
            locked_weights_by_end_date = trailing_arg or {}
        else:
            strategy_spec = trailing_arg
            locked_weights_by_end_date = {}
    elif len(args_tuple) == 6:
        (
            start_date_str,
            end_date_strs,
            current_date_str,
            btc_price_col,
            features_df_pickle,
            btc_df_pickle,
        ) = args_tuple
        strategy_spec = None
        locked_weights_by_end_date = {}
    else:
        raise ValueError(
            f"process_start_date_batch_modal expected 6, 7, or 8 args, got {len(args_tuple)}"
        )

    # Reconstruct DataFrames from pickle
    features_df = pickle.loads(features_df_pickle)
    btc_df = pickle.loads(btc_df_pickle)

    start_date = pd.to_datetime(start_date_str)
    end_dates = [pd.to_datetime(d) for d in end_date_strs]
    current_date = pd.to_datetime(current_date_str)

    strategy = load_strategy(strategy_spec) if strategy_spec else None
    return process_start_date_batch(
        start_date,
        end_dates,
        features_df,
        btc_df,
        current_date,
        btc_price_col,
        strategy=strategy,
        locked_weights_by_end_date=locked_weights_by_end_date,
        enforce_span_contract=True,
    )


@app.function(
    image=image, timeout=1800, secrets=[database_secret]
)  # 30 minutes timeout for large exports
def run_export(
    range_start: str = None,
    range_end: str = None,
    min_range_length_days: int = None,
    btc_price_col: str = None,
):
    """Helper function to run the export process on Modal.

    Returns:
        tuple: (final_df, metadata_dict)
    """
    import pickle
    import sys

    sys.path.insert(0, "/root")

    import pandas as pd

    from .export_weights import (
        BTC_PRICE_COL,
        MIN_RANGE_LENGTH_DAYS,
        RANGE_END,
        RANGE_START,
        get_db_connection,
        generate_date_ranges,
        group_ranges_by_start_date,
        load_locked_weights_for_window,
    )
    from .model_development import precompute_features
    from .prelude import load_data

    # Use defaults if not provided
    range_start = range_start or RANGE_START
    range_end = range_end or RANGE_END
    min_range_length_days = min_range_length_days or MIN_RANGE_LENGTH_DAYS
    btc_price_col = btc_price_col or BTC_PRICE_COL

    strategy_spec = MODAL_STRATEGY_SPEC

    print("Loading data...")
    btc_df = load_data()

    # Determine BTC price column
    if btc_price_col not in btc_df.columns:
        alt_cols = [col for col in btc_df.columns if "BTC" in col and "USD" in col]
        btc_price_col = alt_cols[0] if alt_cols else "PriceUSD_coinmetrics"
        print(f"Warning: {btc_price_col} not found. Using {btc_price_col} instead.")

    current_date = pd.Timestamp.now().normalize()

    # Generate date ranges
    date_ranges = generate_date_ranges(range_start, range_end, min_range_length_days)
    print(f"Generated {len(date_ranges)} date ranges")

    # Precompute features locally (on the coordinator)
    print("Precomputing features...")
    features_df = precompute_features(btc_df)

    # Group ranges by start date
    grouped_ranges = group_ranges_by_start_date(date_ranges)
    sorted_start_dates = sorted(grouped_ranges.keys())
    print(
        f"Grouped into {len(sorted_start_dates)} unique start dates for batched processing"
    )

    # Serialize DataFrames for Modal workers
    # features_df is small enough to broadcast; btc_df is also needed for raw prices
    btc_df_pickle = pickle.dumps(btc_df)
    features_df_pickle = pickle.dumps(features_df)
    current_date_str = current_date.strftime("%Y-%m-%d")
    lock_end_date_str = (current_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Load immutable historical lock prefixes from production DB if available.
    locked_by_start_end: dict[str, dict[str, np.ndarray]] = {}
    try:
        conn = get_db_connection()
        try:
            for start_date in sorted_start_dates:
                start_key = start_date.strftime("%Y-%m-%d")
                per_end: dict[str, np.ndarray] = {}
                for end_date in grouped_ranges[start_date]:
                    end_key = end_date.strftime("%Y-%m-%d")
                    locked = load_locked_weights_for_window(
                        conn,
                        start_key,
                        end_key,
                        lock_end_date_str,
                    )
                    if locked is not None:
                        per_end[end_key] = locked
                locked_by_start_end[start_key] = per_end
        finally:
            conn.close()
    except Exception as exc:
        print(f"Warning: could not load locked DB prefixes, continuing unlocked ({exc})")
        locked_by_start_end = {}

    # Prepare arguments for batches
    batch_args = []
    for start_date in sorted_start_dates:
        end_dates = grouped_ranges[start_date]
        batch_args.append(
            (
                start_date.strftime("%Y-%m-%d"),
                [d.strftime("%Y-%m-%d") for d in end_dates],
                current_date_str,
                btc_price_col,
                features_df_pickle,
                btc_df_pickle,
                strategy_spec,
                locked_by_start_end.get(start_date.strftime("%Y-%m-%d"), {}),
            )
        )

    # Process batches in parallel using Modal
    print(f"Processing {len(batch_args)} batches (start dates) in parallel...")

    # We can likely process these in larger chunks or even all at once depending on limit
    # But let's stick to a reasonable batch size for the map call itself if needed,
    # though map handles this well usually.
    # The previous code batched the *inputs* to map to avoid memory spikes during result collection.

    submission_batch_size = 50
    results = []

    for i in range(0, len(batch_args), submission_batch_size):
        batch = batch_args[i : i + submission_batch_size]
        batch_num = (i // submission_batch_size) + 1
        total_batches = (
            len(batch_args) + submission_batch_size - 1
        ) // submission_batch_size

        print(
            f"Submitting batch {batch_num}/{total_batches} ({len(batch)} start dates)..."
        )

        batch_results = list(process_start_date_batch_modal.map(batch))

        # Each result from map is a DataFrame (concatenated result of that start date batch)
        results.extend(batch_results)
        print(f"  Completed batch {batch_num}/{total_batches}")

    print(f"All {len(date_ranges)} date ranges processed successfully")

    # Combine results
    final_df = pd.concat(results, ignore_index=True)[
        ["id", "start_date", "end_date", "DCA_date", "btc_usd", "weight"]
    ]

    metadata = {
        "rows": len(final_df),
        "date_ranges": len(date_ranges),
        "unique_start_dates": len(sorted_start_dates),
        "range_start": range_start,
        "range_end": range_end,
        "export_date": current_date_str,
    }

    return final_df, metadata


@app.function(
    image=image,
    schedule=modal.Cron(
        "0 14 * * *"
    ),  # Run daily at 4 AM Hawaiian time (UTC-10 = 14:00 UTC)
    secrets=[secret, database_secret],
    timeout=1800,  # 30 minutes timeout for large exports
    retries=3,  # Modal-level retries: retry up to 3 times on failure
)
def daily_export():
    """Scheduled function that runs daily to export weights and save to database."""
    import sys
    from datetime import datetime

    sys.path.insert(0, "/root")

    from .export_weights import (
        create_table_if_not_exists,
        get_db_connection,
        insert_all_data,
        table_is_empty,
        update_today_weights,
    )

    print(f"Starting daily export at {datetime.now()}...")

    # Run the export (use .remote() to call another Modal function)
    final_df, metadata = run_export.remote()

    # Connect to database and handle insert/update logic
    print("Connecting to database...")
    conn = get_db_connection()

    try:
        # Create table if it doesn't exist
        print("Ensuring bitcoin_dca table exists...")
        create_table_if_not_exists(conn)

        # Check if table is empty
        is_empty = table_is_empty(conn)

        today_str = metadata["export_date"]

        if is_empty:
            # Initial run: insert all data
            print("Table is empty. Inserting all data...")
            inserted_count = insert_all_data(conn, final_df)
            print(f"✓ Successfully inserted {inserted_count} rows into bitcoin_dca")
            result_count = inserted_count
        else:
            # Subsequent run: update only today's weights and BTC price
            print(
                f"Table has data. Updating weights and BTC price for date={today_str}..."
            )
            updated_count = update_today_weights(conn, final_df, today_str)
            print(f"✓ Successfully updated {updated_count} rows for date={today_str}")
            result_count = updated_count

        print(f"  Date ranges: {metadata['date_ranges']}")
        print(f"  Range: {metadata['range_start']} to {metadata['range_end']}")

        return {
            "status": "success",
            "rows_affected": result_count,
            **metadata,
        }
    finally:
        conn.close()


@app.function(
    image=image,
    schedule=modal.Cron(
        "0 20 * * *"
    ),  # Run daily at 10 AM Hawaiian time (UTC-10 = 20:00 UTC) - 6 hours after main run
    secrets=[secret, database_secret],
    timeout=1800,  # 30 minutes timeout for large exports
    retries=2,  # Fewer retries for the retry function
)
def daily_export_retry():
    """Retry function that runs daily to check if today's export succeeded.

    If today's data is missing or incomplete, retries the export and update.
    This provides a second chance if the morning run failed due to API issues.
    """
    import sys
    from datetime import datetime

    sys.path.insert(0, "/root")

    import pandas as pd

    from .export_weights import (
        create_table_if_not_exists,
        get_db_connection,
        table_is_empty,
        today_data_exists,
        update_today_weights,
    )

    today_str = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    print(
        f"Starting daily export retry check at {datetime.now()} for date {today_str}..."
    )

    # Connect to database to check if today's data exists
    conn = get_db_connection()

    try:
        # Create table if it doesn't exist
        create_table_if_not_exists(conn)

        # Check if table is empty (if so, skip retry - initial run needed)
        is_empty = table_is_empty(conn)
        if is_empty:
            print("Table is empty. Skipping retry - initial full export needed.")
            return {
                "status": "skipped",
                "reason": "table_empty",
                "date": today_str,
            }

        # Check if today's data already exists and is valid
        if today_data_exists(conn, today_str):
            print(
                f"✓ Today's data ({today_str}) already exists in database. No retry needed."
            )
            return {
                "status": "skipped",
                "reason": "data_already_exists",
                "date": today_str,
            }

        # Today's data is missing - retry the export
        print(f"⚠ Today's data ({today_str}) missing. Retrying export...")

        # Run the export (use .remote() to call another Modal function)
        final_df, metadata = run_export.remote()

        # Verify we got data for today
        today_df = final_df[final_df["DCA_date"] == today_str].copy()
        if today_df.empty:
            print(
                f"⚠ Export completed but no data for {today_str}. May need manual intervention."
            )
            return {
                "status": "partial_failure",
                "reason": "no_data_in_export",
                "date": today_str,
                **metadata,
            }

        # Attempt to update today's weights
        updated_count = update_today_weights(conn, final_df, today_str)

        if updated_count > 0:
            print(
                f"✓ Retry successful: Updated {updated_count} rows for date={today_str}"
            )
            return {
                "status": "success",
                "rows_affected": updated_count,
                "retry": True,
                **metadata,
            }
        else:
            print(
                f"⚠ Retry completed but no rows updated for {today_str}. Price fetch may have failed."
            )
            return {
                "status": "partial_failure",
                "reason": "no_rows_updated",
                "date": today_str,
                **metadata,
            }

    except Exception as e:
        print(f"❌ Retry failed with error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "date": today_str,
        }
    finally:
        conn.close()


@app.local_entrypoint()
def main():
    """Local entrypoint to test the export functionality."""
    print("Running export via Modal...")
    final_df, metadata = run_export.remote()

    print(f"\n✓ Successfully exported {metadata['rows']} rows")
    print(f"  Number of date ranges: {metadata['date_ranges']}")
    print(f"  Range: {metadata['range_start']} to {metadata['range_end']}")
    print(f"  Export date: {metadata['export_date']}")
    print("\nFirst few rows:")
    print(final_df.head(10))
    print("\nLast few rows:")
    print(final_df.tail(10))

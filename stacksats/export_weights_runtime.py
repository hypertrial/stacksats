"""Runtime DB/batch update helpers for export_weights."""

from __future__ import annotations

import logging
import time
from io import StringIO

import pandas as pd


def get_current_btc_price(previous_price=None, *, fetch_btc_price_fn):
    """Fetch current BTC price with robust fallback behavior."""
    logging.info("Fetching current BTC price with retry logic and multiple sources...")
    price_usd = fetch_btc_price_fn(previous_price=previous_price)

    if price_usd is None:
        logging.error("Failed to fetch BTC price from all available sources")
    else:
        logging.info(f"Successfully fetched current BTC price: ${price_usd:,.2f}")

    return price_usd


def insert_all_data(
    conn,
    df,
    *,
    execute_values,
    prepare_copy_dataframe_fn,
    build_insert_rows_fn,
):
    """Insert all rows with COPY-fastpath and execute_values fallback."""
    if execute_values is None:
        raise ImportError(
            "Missing optional dependency 'psycopg2-binary'. "
            "Install deploy extras with: pip install stacksats[deploy]"
        )

    total_rows = len(df)
    logging.info(f"Starting bulk insertion of {total_rows} rows into bitcoin_dca table")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM bitcoin_dca")
        count_before = cur.fetchone()[0]

    start_time = time.time()

    try:
        logging.info("Attempting fast bulk insert using COPY FROM with temp table...")

        buffer = StringIO()
        copy_df = prepare_copy_dataframe_fn(df)
        copy_df.to_csv(buffer, sep="\t", header=False, index=False, na_rep="")
        buffer.seek(0)

        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TEMP TABLE temp_bitcoin_dca (
                    id INTEGER,
                    start_date DATE,
                    end_date DATE,
                    DCA_date DATE,
                    btc_usd FLOAT,
                    weight FLOAT
                ) ON COMMIT DROP
                """
            )

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
                null="",
            )

            cur.execute(
                """
                INSERT INTO bitcoin_dca (id, start_date, end_date, DCA_date, btc_usd, weight)
                SELECT id, start_date, end_date, DCA_date, btc_usd, weight
                FROM temp_bitcoin_dca
                ON CONFLICT (id, start_date, end_date, DCA_date) DO NOTHING
                """
            )

            conn.commit()

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

    except Exception as exc:
        logging.warning(
            f"COPY FROM failed ({exc}), falling back to execute_values method..."
        )
        conn.rollback()

        logging.info("Preparing data for bulk insertion...")
        data = build_insert_rows_fn(df)

        batch_size = 50000
        total_batches = (len(data) + batch_size - 1) // batch_size

        logging.info(
            f"Inserting {len(data)} rows in {total_batches} batches of {batch_size}..."
        )

        failed_batch_num = None
        failed_batch_size = 0
        try:
            with conn.cursor() as cur:
                for i in range(0, len(data), batch_size):
                    batch = data[i : i + batch_size]
                    batch_num = (i // batch_size) + 1
                    failed_batch_num = batch_num
                    failed_batch_size = len(batch)

                    if batch_num % 10 == 0 or batch_num == total_batches:
                        logging.info(
                            f"Processing batch {batch_num}/{total_batches} ({len(batch)} rows)..."
                        )

                    execute_values(
                        cur,
                        """
                        INSERT INTO bitcoin_dca (id, start_date, end_date, DCA_date, btc_usd, weight)
                        VALUES %s
                        ON CONFLICT (id, start_date, end_date, DCA_date) DO NOTHING
                        """,
                        batch,
                        page_size=len(batch),
                    )

                    if batch_num % 5 == 0 or batch_num == total_batches:
                        conn.commit()
                        if batch_num % 10 == 0 or batch_num == total_batches:
                            logging.info(f"  ✓ Committed through batch {batch_num}")

            conn.commit()
        except Exception as exc:
            logging.error(
                "Fallback bulk insert failed at batch %s/%s (%s rows): %s",
                failed_batch_num,
                total_batches,
                failed_batch_size,
                exc,
            )
            try:
                conn.rollback()
            except Exception as rollback_error:
                logging.error(
                    "Rollback failed after fallback insert error: %s", rollback_error
                )
            raise

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


def update_today_weights(
    conn,
    df,
    today_str,
    *,
    get_current_btc_price_fn,
    build_update_rows_fn,
    build_values_sql_fn,
    sql_quote_fn,
):
    """Update today rows with model weights and latest BTC price when available."""
    required_cols = {
        "day_index",
        "start_date",
        "end_date",
        "date",
        "price_usd",
        "weight",
    }
    missing_cols = sorted(col for col in required_cols if col not in df.columns)
    if missing_cols:
        raise ValueError(
            "update_today_weights requires canonical columns: "
            + ", ".join(sorted(required_cols))
            + f". Missing: {', '.join(missing_cols)}"
        )

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
    except Exception as exc:
        logging.debug(f"Could not fetch previous day's price for validation: {exc}")

    current_btc_price = get_current_btc_price_fn(previous_price=previous_price)
    if current_btc_price is None:
        logging.warning(
            "Failed to fetch BTC price from all API sources. Will use price from dataframe if available."
        )
        day_mask = (
            pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            == today_str
        )
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

    logging.info(f"Filtering data for date = {today_str}")
    day_mask = (
        pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d") == today_str
    )
    today_df = df[day_mask].copy()

    if today_df.empty:
        logging.warning(f"No data found for today ({today_str})")
        return 0

    if current_btc_price is None:
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

    sample_row = today_df.iloc[0]
    logging.info(
        f"Sample row - day_index: {sample_row['day_index']}, start_date: {sample_row['start_date']}, end_date: {sample_row['end_date']}, weight: {sample_row['weight']:.6f}"
    )
    if current_btc_price is not None:
        logging.info(f"Will update BTC price to: ${current_btc_price:,.2f}")

    start_time = time.time()
    update_data = build_update_rows_fn(today_df, current_btc_price=current_btc_price)

    batch_size = 10000
    total_updated = 0

    logging.info(
        f"Starting bulk weight and BTC price updates in batches of {batch_size}..."
    )

    total_batches = (len(update_data) + batch_size - 1) // batch_size
    failed_batch_num = None
    failed_batch_size = 0
    update_mode = "weight_and_price" if current_btc_price is not None else "weight_only"
    try:
        with conn.cursor() as cur:
            for i in range(0, len(update_data), batch_size):
                batch = update_data[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                failed_batch_num = batch_num
                failed_batch_size = len(batch)

                batch_start_time = time.time()

                logging.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} rows)..."
                )

                if current_btc_price is not None:
                    values_str = build_values_sql_fn(
                        batch,
                        include_price=True,
                        sql_quote=sql_quote_fn,
                    )

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
                    values_str = build_values_sql_fn(
                        batch,
                        include_price=False,
                        sql_quote=sql_quote_fn,
                    )

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

            logging.info("Committing all weight and BTC price updates...")
            conn.commit()
            commit_time = time.time() - start_time
    except Exception as exc:
        logging.error(
            "update_today_weights failed in %s mode at batch %s/%s (%s rows): %s",
            update_mode,
            failed_batch_num,
            total_batches,
            failed_batch_size,
            exc,
        )
        try:
            conn.rollback()
        except Exception as rollback_error:
            logging.error(
                "Rollback failed after update_today_weights error: %s", rollback_error
            )
        raise

    logging.info(
        f"Update completed. Total rows updated: {total_updated} in {commit_time:.2f}s"
    )
    logging.info(f"Average update rate: {total_updated / commit_time:.1f} rows/second")
    return total_updated

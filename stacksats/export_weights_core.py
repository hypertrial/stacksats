"""Core range-processing helpers for export_weights."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from .strategy_types import strategy_context_from_features_df


def _to_dt(val) -> datetime:
    if isinstance(val, datetime):
        if val.tzinfo is not None:
            val = val.astimezone(timezone.utc).replace(tzinfo=None)
        return val.replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime.strptime(str(val)[:10], "%Y-%m-%d")


def _date_range(start: datetime, end: datetime) -> list[datetime]:
    """Generate list of daily datetimes from start to end inclusive."""
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _normalize_date_frame(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Normalize a date-like column to canonical daily datetimes."""
    if date_col not in df.columns or df.is_empty():
        return df
    dtype = df[date_col].dtype
    if dtype == pl.Utf8:
        expr = pl.col(date_col).str.to_datetime(strict=False)
    elif dtype == pl.Date:
        expr = pl.col(date_col).cast(pl.Datetime)
    else:
        expr = pl.col(date_col).cast(pl.Datetime, strict=False)
    return df.with_columns(expr.dt.replace_time_zone(None).dt.truncate("1d").alias(date_col))


def process_start_date_batch(
    start_date,
    end_dates,
    features_df: pl.DataFrame,
    btc_df: pl.DataFrame,
    current_date,
    btc_price_col: str,
    strategy=None,
    locked_weights_by_end_date: dict[str, np.ndarray] | None = None,
    enforce_span_contract: bool = True,
    *,
    compute_window_weights_fn,
    validate_span_length_fn,
    base_strategy_cls,
    validate_strategy_contract_fn,
):
    """Process all date ranges that share a start_date."""
    if strategy is not None:
        if not isinstance(strategy, base_strategy_cls):
            raise TypeError("strategy must subclass BaseStrategy.")
        validate_strategy_contract_fn(strategy)

    date_col = "date"
    features_df = _normalize_date_frame(features_df, date_col)
    btc_df = _normalize_date_frame(btc_df, date_col)
    results = []

    for end_date in end_dates:
        if enforce_span_contract:
            validate_span_length_fn(start_date, end_date)
        start_ts = _to_dt(start_date)
        end_ts = _to_dt(end_date)
        full_range = _date_range(start_ts, end_ts)
        n_total = len(full_range)
        end_date_key = end_ts.strftime("%Y-%m-%d")
        locked_weights = None
        if locked_weights_by_end_date is not None:
            locked_weights = locked_weights_by_end_date.get(end_date_key)

        if strategy is None:
            weights_df = compute_window_weights_fn(
                features_df,
                start_date,
                end_date,
                current_date,
                locked_weights=locked_weights,
            )
        else:
            observed_end = min(_to_dt(current_date), end_ts)
            slice_df = features_df.filter(
                (pl.col(date_col) >= start_ts) & (pl.col(date_col) <= observed_end)
            )
            ctx = strategy_context_from_features_df(
                slice_df,
                start_date,
                end_date,
                current_date,
                required_columns=tuple(strategy.required_feature_columns()),
                as_of_date=current_date,
                locked_weights=locked_weights,
            )
            weights_df = strategy.compute_weights(ctx)
            if weights_df.height == 0:
                weights_df = compute_window_weights_fn(
                    features_df,
                    start_date,
                    end_date,
                    current_date,
                    locked_weights=locked_weights,
                )
            else:
                full_dates_df = pl.DataFrame({date_col: full_range})
                weights_df = full_dates_df.join(
                    weights_df.select([date_col, "weight"]),
                    on=date_col,
                    how="left",
                ).with_columns(pl.col("weight").fill_null(0.0))

        full_dates_df = pl.DataFrame({date_col: full_range})
        btc_slice = btc_df.filter(
            (pl.col(date_col) >= start_ts) & (pl.col(date_col) <= end_ts)
        )
        range_prices_df = full_dates_df.join(
            btc_slice.select([date_col, btc_price_col]),
            on=date_col,
            how="left",
        )
        range_prices = range_prices_df[btc_price_col].to_numpy()
        weight_vals = weights_df["weight"].to_numpy() if "weight" in weights_df.columns else np.ones(n_total) / n_total

        range_df = pl.DataFrame({
            "day_index": list(range(n_total)),
            "start_date": start_ts.strftime("%Y-%m-%d"),
            "end_date": end_ts.strftime("%Y-%m-%d"),
            "date": [d.strftime("%Y-%m-%d") for d in full_range],
            "price_usd": range_prices,
            "weight": weight_vals,
        })
        results.append(range_df)

    return pl.concat(results, how="vertical_relaxed")


def load_locked_weights_for_window(
    conn,
    start_date: str,
    end_date: str,
    lock_end_date: str,
) -> np.ndarray | None:
    """Load immutable locked prefix from DB for one allocation window."""
    start_ts = _to_dt(start_date)
    end_ts = _to_dt(end_date)
    lock_end_ts = min(_to_dt(lock_end_date), end_ts)
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

    expected = _date_range(start_ts, lock_end_ts)
    actual_dates = [row[0] for row in rows]
    if len(actual_dates) != len(expected):
        actual_set = set(str(d)[:10] for d in actual_dates)
        expected_set = set(d.strftime("%Y-%m-%d") for d in expected)
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        raise ValueError(
            "Locked history is not a contiguous prefix for "
            f"{start_date}..{end_date}. Missing={list(missing)}, "
            f"extra={list(extra)}"
        )

    return np.array([float(row[1]) for row in rows], dtype=float)

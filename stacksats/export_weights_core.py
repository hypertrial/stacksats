"""Core range-processing helpers for export_weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    *,
    compute_window_weights_fn,
    validate_span_length_fn,
    strategy_context_cls,
    base_strategy_cls,
    validate_strategy_contract_fn,
):
    """Process all date ranges that share a start_date."""
    if strategy is not None:
        if not isinstance(strategy, base_strategy_cls):
            raise TypeError("strategy must subclass BaseStrategy.")
        validate_strategy_contract_fn(strategy)

    results = []

    for end_date in end_dates:
        if enforce_span_contract:
            validate_span_length_fn(start_date, end_date)
        full_range = pd.date_range(start=start_date, end=end_date, freq="D")
        n_total = len(full_range)
        end_date_key = end_date.strftime("%Y-%m-%d")
        locked_weights = None
        if locked_weights_by_end_date is not None:
            locked_weights = locked_weights_by_end_date.get(end_date_key)

        if strategy is None:
            weights = compute_window_weights_fn(
                features_df,
                start_date,
                end_date,
                current_date,
                locked_weights=locked_weights,
            )
        else:
            weights = strategy.compute_weights(
                strategy_context_cls(
                    features_df=features_df,
                    start_date=start_date,
                    end_date=end_date,
                    current_date=current_date,
                    locked_weights=locked_weights,
                )
            )
            if weights.empty:
                weights = compute_window_weights_fn(
                    features_df,
                    start_date,
                    end_date,
                    current_date,
                    locked_weights=locked_weights,
                )
            else:
                weights = weights.reindex(full_range).fillna(0.0)

        range_prices = btc_df[btc_price_col].reindex(full_range).values

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
    """Load immutable locked prefix from DB for one allocation window."""
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

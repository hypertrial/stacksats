import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .btc_price_fetcher import fetch_btc_price_historical, fetch_btc_price_robust
from .framework_contract import ALLOCATION_SPAN_DAYS, ALLOCATION_WINDOW_OFFSET
from .model_development import precompute_features

# Configuration
BACKTEST_START = "2018-01-01"
# Fixed allocation span used across modules.
WINDOW_DAYS = ALLOCATION_SPAN_DAYS
WINDOW_OFFSET = ALLOCATION_WINDOW_OFFSET

# Tolerance for weight sum validation (small leniency for floating-point precision)
WEIGHT_SUM_TOLERANCE = 1e-5


def get_backtest_end() -> str:
    """Return dynamic default end date as yesterday (UTC-localized date)."""
    return (pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# Backward-compatible constant for callers/tests that import this symbol directly.
BACKTEST_END = get_backtest_end()


def load_data(*, cache_dir: str | None = "~/.stacksats/cache", max_age_hours: int = 24):
    """Load BTC data from CoinMetrics CSV with optional local caching.

    This keeps legacy prelude behavior:
    - fresh/stale cache handling
    - gap filling for missing historical ``PriceUSD`` values
    - best-effort inclusion of today's row
    - MVRV fallback from yesterday when today's value is missing
    """
    url = "https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv"
    now = pd.Timestamp.now().normalize()
    backtest_start_ts = pd.to_datetime(BACKTEST_START)

    def _download_csv() -> bytes:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser() / "coinmetrics_btc.csv"

    csv_bytes: bytes
    if cache_path is not None and cache_path.exists():
        age_hours = (pd.Timestamp.now().timestamp() - cache_path.stat().st_mtime) / 3600.0
        if age_hours <= max_age_hours:
            csv_bytes = cache_path.read_bytes()
        else:
            csv_bytes = _download_csv()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(csv_bytes)
    else:
        csv_bytes = _download_csv()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(csv_bytes)

    df = pd.read_csv(BytesIO(csv_bytes))
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.index = df.index.normalize().tz_localize(None)
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()

    if "PriceUSD" not in df.columns:
        raise ValueError("PriceUSD column not found in CoinMetrics data")
    df["PriceUSD"] = pd.to_numeric(df["PriceUSD"], errors="coerce")
    # True only where CoinMetrics provided a concrete price in the raw CSV.
    df["PriceUSD_source_exists"] = df["PriceUSD"].notna()
    df["PriceUSD_coinmetrics"] = df["PriceUSD"]
    window = df.loc[df.index >= backtest_start_ts].copy()

    price_col = "PriceUSD_coinmetrics"
    # Fill historical holes in existing rows first.
    for hole_date in window.index[window[price_col].isna()]:
        prev_series = window.loc[window.index < hole_date, price_col].dropna()
        prev_price = float(prev_series.iloc[-1]) if len(prev_series) else None
        fetched = fetch_btc_price_historical(hole_date, previous_price=prev_price)
        if fetched is not None:
            window.at[hole_date, price_col] = float(fetched)
        elif prev_price is not None:
            window.at[hole_date, price_col] = prev_price

    # Ensure today exists in index; use robust fetch, then fallback to previous known price.
    if now not in window.index:
        window.loc[now] = pd.NA
        window.at[now, "PriceUSD_source_exists"] = False
    if pd.isna(window.at[now, price_col]):
        prev_series = window.loc[window.index < now, price_col].dropna()
        prev_price = float(prev_series.iloc[-1]) if len(prev_series) else None
        today_price = fetch_btc_price_robust(previous_price=prev_price)
        if today_price is None:
            today_price = prev_price
        if today_price is not None:
            window.at[now, price_col] = float(today_price)

    # Strict guard: unresolved gaps are a hard failure for prelude consumers.
    missing_dates = window.index[window[price_col].isna()]
    if len(missing_dates):
        raise AssertionError(
            "CoinMetrics dates still missing BTC-USD prices after gap-fill attempts: "
            f"{missing_dates[0].date()}"
        )

    # MVRV fallback: if today's value is missing, copy yesterday when available.
    if "CapMVRVCur" in window.columns and now in window.index:
        today_mvrv = window.at[now, "CapMVRVCur"]
        if pd.isna(today_mvrv):
            yesterday = now - pd.Timedelta(days=1)
            if yesterday in window.index and pd.notna(window.at[yesterday, "CapMVRVCur"]):
                window.at[now, "CapMVRVCur"] = window.at[yesterday, "CapMVRVCur"]
                logging.info(
                    "Used yesterday's MVRV value for today (%s).",
                    now.date(),
                )
            else:
                logging.warning(
                    "Could not find valid MVRV for yesterday; today's MVRV remains missing."
                )

    return window.sort_index()


def _make_window_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format rolling window label as 'YYYY-MM-DD → YYYY-MM-DD'."""
    return f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"


def parse_window_dates(window_label: str) -> pd.Timestamp:
    """Extract start date from window label like '2016-01-01 → 2017-01-01'.

    Args:
        window_label: Window label in format 'YYYY-MM-DD → YYYY-MM-DD'

    Returns:
        Start date as pandas Timestamp
    """
    return pd.to_datetime(window_label.split(" → ")[0])


def generate_date_ranges(
    range_start: str, range_end: str, min_length_days: int = 120
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate date ranges where each start_date has fixed-span end_date.

    Uses DATE_FREQ (daily) for start date generation.
    Each start_date is paired with exactly one end_date at fixed span length.
    Uses WINDOW_OFFSET from prelude.py for consistency across modules.

    Args:
        range_start: Start of the date range (YYYY-MM-DD format)
        range_end: End of the date range (YYYY-MM-DD format)
        min_length_days: Minimum range length in days (default 120)

    Returns:
        List of (start_date, end_date) tuples
    """
    del min_length_days
    start, end = pd.to_datetime(range_start), pd.to_datetime(range_end)
    max_start_date = end - WINDOW_OFFSET
    start_dates = pd.date_range(start=start, end=max_start_date, freq="D")

    def _window_end(start_date: pd.Timestamp) -> pd.Timestamp:
        return start_date + WINDOW_OFFSET

    date_ranges = []
    for start_date in start_dates:
        end_date = _window_end(start_date)
        # Only include if end_date is within range_end
        if end_date <= end:
            date_ranges.append((start_date, end_date))

    return date_ranges


def group_ranges_by_start_date(
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[pd.Timestamp, list[pd.Timestamp]]:
    """Group list of (start, end) tuples by start_date.

    Args:
        date_ranges: List of (start_date, end_date) tuples

    Returns:
        Dictionary mapping start_date -> list of end_dates
    """
    grouped: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for start, end in date_ranges:
        if start not in grouped:
            grouped[start] = []
        grouped[start].append(end)
    return grouped


def compute_cycle_spd(
    dataframe: pd.DataFrame,
    strategy_function,
    features_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    validate_weights: bool = True,
) -> pd.DataFrame:
    """Compute sats-per-dollar (SPD) statistics over rolling windows.

    Unified function that supports both simple usage and shared runtime logic with
    precomputed features. Uses 1-year windows for consistency across modules.

    Args:
        dataframe: DataFrame containing price data with 'PriceUSD_coinmetrics' column
        strategy_function: Function that takes features DataFrame and returns weights
        features_df: Optional precomputed features. If None, computes them internally.
        start_date: Optional start date (default: BACKTEST_START)
        end_date: Optional end date (default: BACKTEST_END)
        validate_weights: Whether to validate that weights sum to 1.0 (default: True)

    Returns:
        DataFrame with SPD statistics indexed by window label
    """
    start = start_date or BACKTEST_START
    end = pd.to_datetime(end_date or get_backtest_end())

    # Use provided features or compute them
    if features_df is None:
        full_feat = precompute_features(dataframe).loc[start:end]
    else:
        full_feat = features_df.loc[start:end]

    source_mask: pd.Series | None = None
    if "PriceUSD_source_exists" in dataframe.columns:
        # Backtests must only use rows that exist in CoinMetrics history.
        source_mask = (
            dataframe["PriceUSD_source_exists"]
            .reindex(dataframe.index)
            .fillna(False)
            .astype(bool)
        )
        available_index = source_mask[source_mask].index
        if len(available_index) > 0:
            source_end = available_index.max()
            end = min(end, source_end)
            full_feat = full_feat.loc[:end]

    def _window_end(start_dt: pd.Timestamp) -> pd.Timestamp:
        return start_dt + WINDOW_OFFSET

    # Generate start dates daily (matching export_weights.py DATE_FREQ)
    max_start_date = pd.to_datetime(end) - WINDOW_OFFSET
    start_dates = pd.date_range(
        start=pd.to_datetime(start),
        end=max_start_date,
        freq="D",  # Daily frequency for consistency
    )

    if len(start_dates) > 0:
        actual_end_date = _window_end(start_dates[-1]).date()
        logging.info(
            f"Backtesting date range: {start_dates[0].date()} to {actual_end_date} "
            f"({len(start_dates)} total windows)"
        )

    results = []
    validated_windows = 0
    for window_start in start_dates:
        window_end = _window_end(window_start)

        # Only include if end_date is within range
        if window_end > pd.to_datetime(end):
            continue

        price_slice = dataframe["PriceUSD_coinmetrics"].loc[window_start:window_end]
        if price_slice.empty:
            continue
        if source_mask is not None:
            source_slice = source_mask.loc[window_start:window_end]
            if source_slice.empty or not bool(source_slice.all()):
                continue

        # Compute weights using strategy_function
        window_feat = full_feat.loc[window_start:window_end]
        # Under the strict span contract, strategies only accept full-span windows.
        # Historical datasets that begin after BACKTEST_START can create partial windows
        # at the front edge; skip those instead of surfacing per-window contract errors.
        if len(window_feat) != WINDOW_DAYS:
            continue
        weight_slice = strategy_function(window_feat)
        if weight_slice.empty:
            # Some strategies may return empty weights for low-feature windows.
            # Fall back to uniform weights over available price dates.
            weight_slice = pd.Series(
                np.full(len(price_slice), 1.0 / len(price_slice)),
                index=price_slice.index,
            )
        else:
            # Align strategy output to price index and normalize defensively.
            weight_slice = weight_slice.reindex(price_slice.index).fillna(0.0)
            weight_total = float(weight_slice.sum())
            if not np.isfinite(weight_total) or weight_total <= 0:
                weight_slice = pd.Series(
                    np.full(len(price_slice), 1.0 / len(price_slice)),
                    index=price_slice.index,
                )
            else:
                weight_slice = weight_slice / weight_total

        # Validate weights sum to 1.0 if requested
        if validate_weights:
            weight_sum = weight_slice.sum()
            if not np.isclose(weight_sum, 1.0, atol=WEIGHT_SUM_TOLERANCE):
                raise ValueError(
                    f"Weights for range {window_start.date()} to {window_end.date()} "
                    f"sum to {weight_sum:.10f}, expected 1.0 "
                    f"(tolerance: {WEIGHT_SUM_TOLERANCE})"
                )
            validated_windows += 1

        inv_price = 1e8 / price_slice  # sats per dollar
        min_spd, max_spd = inv_price.min(), inv_price.max()
        span = max_spd - min_spd
        uniform_spd = inv_price.mean()
        dynamic_spd = (weight_slice * inv_price).sum()

        # Handle edge case where span is zero (all prices identical)
        if span > 0:
            uniform_pct = (uniform_spd - min_spd) / span * 100
            dynamic_pct = (dynamic_spd - min_spd) / span * 100
        else:
            # When all prices are identical, percentile is undefined
            uniform_pct = float("nan")
            dynamic_pct = float("nan")

        results.append(
            {
                "window": _make_window_label(window_start, window_end),
                "min_sats_per_dollar": min_spd,
                "max_sats_per_dollar": max_spd,
                "uniform_sats_per_dollar": uniform_spd,
                "dynamic_sats_per_dollar": dynamic_spd,
                "uniform_percentile": uniform_pct,
                "dynamic_percentile": dynamic_pct,
                "excess_percentile": dynamic_pct - uniform_pct,
            }
        )

    if validate_weights and validated_windows > 0:
        logging.info(
            f"✓ Validated weight sums for {validated_windows} windows (all sum to 1.0)"
        )

    return pd.DataFrame(results).set_index("window")


def backtest_dynamic_dca(
    dataframe: pd.DataFrame,
    strategy_function,
    features_df: pd.DataFrame | None = None,
    *,
    strategy_label: str = "strategy",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Run rolling-window SPD backtest and log aggregated performance metrics.

    Unified function that supports both simple usage and shared runtime logic with
    precomputed features.

    Args:
        dataframe: DataFrame containing price data with 'PriceUSD_coinmetrics' column
        strategy_function: Function that takes features DataFrame and returns weights
        features_df: Optional precomputed features. If None, computes them internally.
        strategy_label: Label for logging (default: "strategy")
        start_date: Optional start date (default: BACKTEST_START)
        end_date: Optional end date (default: dynamic yesterday)

    Returns:
        Tuple of:
        - SPD table DataFrame
        - exponential-decay average dynamic percentile
        - exponential-decay average uniform percentile
    """
    spd_table = compute_cycle_spd(
        dataframe,
        strategy_function,
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
    )
    dynamic_spd = spd_table["dynamic_sats_per_dollar"]
    dynamic_pct = spd_table["dynamic_percentile"]
    uniform_pct = spd_table["uniform_percentile"]

    # Exponential decay weighting (recent windows weighted more)
    N = len(dynamic_spd)
    exp_weights = 0.9 ** np.arange(N - 1, -1, -1)
    exp_weights /= exp_weights.sum()
    exp_avg_pct = (dynamic_pct.values * exp_weights).sum()
    uniform_exp_avg_pct = (uniform_pct.values * exp_weights).sum()

    logging.info(f"Aggregated Metrics for {strategy_label}:")
    logging.info(
        f"  SPD: min={dynamic_spd.min():.2f}, max={dynamic_spd.max():.2f}, "
        f"mean={dynamic_spd.mean():.2f}, median={dynamic_spd.median():.2f}"
    )
    logging.info(
        f"  Percentile: min={dynamic_pct.min():.2f}%, max={dynamic_pct.max():.2f}%, "
        f"mean={dynamic_pct.mean():.2f}%, median={dynamic_pct.median():.2f}%"
    )
    logging.info(f"  Exp-decay avg SPD percentile: {exp_avg_pct:.2f}%")
    logging.info(f"  Exp-decay avg uniform percentile: {uniform_exp_avg_pct:.2f}%")

    return spd_table, exp_avg_pct, uniform_exp_avg_pct

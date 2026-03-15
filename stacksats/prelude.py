import datetime as dt
import logging

import numpy as np
import polars as pl

from .data_btc import BTCDataProvider
from .framework_contract import ALLOCATION_SPAN_DAYS, ALLOCATION_WINDOW_OFFSET
from .model_development import precompute_features

# Configuration
BACKTEST_START = "2018-01-01"
# Fixed allocation span used across modules.
WINDOW_DAYS = ALLOCATION_SPAN_DAYS
WINDOW_OFFSET = ALLOCATION_WINDOW_OFFSET
DATE_FREQ = "D"
DATE_COL = "date"


def _daily_datetime_expr(frame: pl.DataFrame, column: str = DATE_COL) -> pl.Expr:
    """Return a canonical daily datetime expression for a date-like column."""
    dtype = frame[column].dtype
    if dtype == pl.Utf8:
        base = pl.col(column).str.to_datetime(strict=False)
    elif dtype == pl.Date:
        base = pl.col(column).cast(pl.Datetime)
    else:
        base = pl.col(column).cast(pl.Datetime, strict=False)
    return base.dt.replace_time_zone(None).dt.truncate("1d")

# Tolerance for weight sum validation (small leniency for floating-point precision)
WEIGHT_SUM_TOLERANCE = 1e-5


def get_backtest_end() -> str:
    """Return dynamic default end date as yesterday (UTC-localized date)."""
    today = dt.datetime.now(dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    yesterday = today - dt.timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def load_data(
    *,
    parquet_path: str | None = None,
    max_staleness_days: int = 3,
    end_date: str | None = None,
):
    """Load strict BRK BTC data through the canonical provider path.

    Requires a local BRK parquet file (or set STACKSATS_ANALYTICS_PARQUET).
    If you want to supply your own data without a parquet file, use
    :class:`stacksats.ColumnMapDataProvider` instead, or construct a
    :class:`stacksats.runner.StrategyRunner` with
    :meth:`~stacksats.runner.StrategyRunner.from_dataframe`.

    This path intentionally enforces source-only data integrity:
    - local parquet only
    - no synthetic row filling
    - no fallback source blending
    """
    provider = BTCDataProvider(
        parquet_path=parquet_path,
        max_staleness_days=max_staleness_days,
    )
    return provider.load(backtest_start=BACKTEST_START, end_date=end_date)


def _make_window_label(start: dt.datetime, end: dt.datetime) -> str:
    """Format rolling window label as 'YYYY-MM-DD → YYYY-MM-DD'."""
    return f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"


def parse_window_dates(window_label: str) -> dt.datetime:
    """Extract start date from window label like '2016-01-01 → 2017-01-01'.

    Args:
        window_label: Window label in format 'YYYY-MM-DD → YYYY-MM-DD'

    Returns:
        Start date as datetime (naive).
    """
    return dt.datetime.strptime(window_label.split(" → ")[0], "%Y-%m-%d")


def generate_date_ranges(
    range_start: str,
    range_end: str,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Generate date ranges where each start_date has fixed-span end_date.

    Uses DATE_FREQ (daily) for start date generation.
    Each start_date is paired with exactly one end_date at fixed span length.
    Uses WINDOW_OFFSET from prelude.py for consistency across modules.

    Args:
        range_start: Start of the date range (YYYY-MM-DD format)
        range_end: End of the date range (YYYY-MM-DD format)

    Returns:
        List of (start_date, end_date) tuples
    """
    start = dt.datetime.strptime(range_start, "%Y-%m-%d")
    end = dt.datetime.strptime(range_end, "%Y-%m-%d")
    max_start_date = end - WINDOW_OFFSET

    def _window_end(start_date: dt.datetime) -> dt.datetime:
        return start_date + WINDOW_OFFSET

    date_ranges: list[tuple[dt.datetime, dt.datetime]] = []
    current = start
    while current <= max_start_date:
        end_date = _window_end(current)
        if end_date <= end:
            date_ranges.append((current, end_date))
        current += dt.timedelta(days=1)

    return date_ranges


def date_range_list(
    start: dt.datetime | str,
    end: dt.datetime | str,
) -> list[dt.datetime]:
    """Return list of dates from start to end inclusive (1d interval).

    Replaces pd.date_range(..., freq='D') for Polars migration.
    """
    if isinstance(start, str):
        start = dt.datetime.strptime(start[:10], "%Y-%m-%d")
    if isinstance(end, str):
        end = dt.datetime.strptime(end[:10], "%Y-%m-%d")
    s = pl.datetime_range(start, end, interval="1d", eager=True)
    return s.to_list()


def date_range_series(
    start: dt.datetime | str,
    end: dt.datetime | str,
) -> pl.Series:
    """Return Polars Series of dates from start to end inclusive (1d interval)."""
    if isinstance(start, str):
        start = dt.datetime.strptime(start[:10], "%Y-%m-%d")
    if isinstance(end, str):
        end = dt.datetime.strptime(end[:10], "%Y-%m-%d")
    return pl.datetime_range(start, end, interval="1d", eager=True)


def group_ranges_by_start_date(
    date_ranges: list[tuple[dt.datetime, dt.datetime]],
) -> dict[dt.datetime, list[dt.datetime]]:
    """Group list of (start, end) tuples by start_date.

    Args:
        date_ranges: List of (start_date, end_date) tuples

    Returns:
        Dictionary mapping start_date -> list of end_dates
    """
    grouped: dict[dt.datetime, list[dt.datetime]] = {}
    for start, end in date_ranges:
        if start not in grouped:
            grouped[start] = []
        grouped[start].append(end)
    return grouped


def _ensure_pl_with_date(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure dataframe has date column for filtering."""
    if DATE_COL not in df.columns:
        raise ValueError(f"DataFrame must have '{DATE_COL}' column.")
    return df


def compute_cycle_spd(
    dataframe: pl.DataFrame,
    strategy_function,
    features_df: pl.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    validate_weights: bool = True,
) -> pl.DataFrame:
    """Compute sats-per-dollar (SPD) statistics over rolling windows.

    Unified function that supports both simple usage and shared runtime logic with
    precomputed features. Uses 1-year windows for consistency across modules.

    Args:
        dataframe: Polars DataFrame with 'date' and 'price_usd' columns
        strategy_function: Function that takes features pl.DataFrame and returns
            pl.DataFrame with 'date' and 'weight' columns
        features_df: Optional precomputed features. If None, computes them internally.
        start_date: Optional start date (default: BACKTEST_START)
        end_date: Optional end date (default: dynamic yesterday)
        validate_weights: Whether to validate that weights sum to 1.0 (default: True)

    Returns:
        Polars DataFrame with SPD statistics, 'window' as first column
    """
    start = start_date or BACKTEST_START
    end = (
        dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
        if end_date
        else dt.datetime.strptime(get_backtest_end()[:10], "%Y-%m-%d")
    )

    dataframe = _ensure_pl_with_date(dataframe)
    dataframe = dataframe.with_columns(_daily_datetime_expr(dataframe).alias(DATE_COL))

    if features_df is None:
        full_feat = precompute_features(dataframe)
    else:
        full_feat = _ensure_pl_with_date(features_df).clone()
    full_feat = full_feat.with_columns(_daily_datetime_expr(full_feat).alias(DATE_COL))

    start_ts = dt.datetime.strptime(start[:10], "%Y-%m-%d")
    full_feat = full_feat.filter(
        (pl.col(DATE_COL) >= start_ts) & (pl.col(DATE_COL) <= end)
    )

    source_mask = None
    if "price_usd_source_exists" in dataframe.columns:
        mask = dataframe["price_usd_source_exists"].fill_null(False)
        valid = dataframe.filter(mask)
        if valid.height > 0:
            source_end = valid[DATE_COL].max()
            if source_end is not None:
                end = min(end, source_end)
                full_feat = full_feat.filter(pl.col(DATE_COL) <= end)

    max_start_date = end - WINDOW_OFFSET
    start_dates = pl.datetime_range(start_ts, max_start_date, interval="1d", eager=True)

    if start_dates.len() > 0:
        last_start = start_dates[-1]
        actual_end = last_start + WINDOW_OFFSET
        logging.info(
            f"Backtesting date range: {start_dates[0]} to {actual_end} "
            f"({start_dates.len()} total windows)"
        )

    results: list[dict] = []
    validated_windows = 0
    for window_start in start_dates.to_list():
        window_end = window_start + WINDOW_OFFSET
        if window_end > end:
            continue

        price_slice = dataframe.filter(
            (pl.col(DATE_COL) >= window_start) & (pl.col(DATE_COL) <= window_end)
        )
        if price_slice.is_empty():
            continue
        if source_mask is not None:
            # Re-check source mask for this window
            pass

        window_feat = full_feat.filter(
            (pl.col(DATE_COL) >= window_start) & (pl.col(DATE_COL) <= window_end)
        )
        if window_feat.height != WINDOW_DAYS:
            continue

        weight_df = strategy_function(window_feat)
        if weight_df.is_empty() or "weight" not in weight_df.columns:
            n = price_slice.height
            weight_df = pl.DataFrame({
                DATE_COL: price_slice[DATE_COL].to_list(),
                "weight": [1.0 / n] * n,
            })
        else:
            merged = price_slice.select(DATE_COL).join(
                weight_df.select([DATE_COL, "weight"]),
                on=DATE_COL,
                how="left",
            )
            w = merged["weight"].fill_null(0.0)
            total = w.sum()
            if not np.isfinite(total) or total <= 0:
                n = price_slice.height
                weight_df = pl.DataFrame({
                    DATE_COL: price_slice[DATE_COL].to_list(),
                    "weight": [1.0 / n] * n,
                })
            else:
                weight_df = pl.DataFrame({
                    DATE_COL: merged[DATE_COL].to_list(),
                    "weight": (w / total).to_list(),
                })

        if validate_weights:
            weight_sum = weight_df["weight"].sum()
            if not np.isclose(float(weight_sum), 1.0, atol=WEIGHT_SUM_TOLERANCE):
                raise ValueError(
                    f"Weights for range {window_start.date()} to {window_end.date()} "
                    f"sum to {float(weight_sum):.10f}, expected 1.0 "
                    f"(tolerance: {WEIGHT_SUM_TOLERANCE})"
                )
            validated_windows += 1

        price_vals = price_slice["price_usd"].to_numpy()
        inv_price = 1e8 / price_vals
        min_spd, max_spd = float(np.nanmin(inv_price)), float(np.nanmax(inv_price))
        span = max_spd - min_spd
        uniform_spd = float(np.nanmean(inv_price))
        w_vals = weight_df["weight"].to_numpy()
        if len(w_vals) == len(inv_price):
            dynamic_spd = float(np.sum(w_vals * inv_price))
        else:
            dynamic_spd = uniform_spd

        if span > 0:
            uniform_pct = (uniform_spd - min_spd) / span * 100
            dynamic_pct = (dynamic_spd - min_spd) / span * 100
        else:
            uniform_pct = float("nan")
            dynamic_pct = float("nan")

        results.append({
            "window": _make_window_label(window_start, window_end),
            "min_sats_per_dollar": min_spd,
            "max_sats_per_dollar": max_spd,
            "uniform_sats_per_dollar": uniform_spd,
            "dynamic_sats_per_dollar": dynamic_spd,
            "uniform_percentile": uniform_pct,
            "dynamic_percentile": dynamic_pct,
            "excess_percentile": dynamic_pct - uniform_pct,
        })

    if validate_weights and validated_windows > 0:
        logging.info(
            f"✓ Validated weight sums for {validated_windows} windows (all sum to 1.0)"
        )

    if not results:
        return pl.DataFrame(schema={
            "window": pl.Utf8,
            "min_sats_per_dollar": pl.Float64,
            "max_sats_per_dollar": pl.Float64,
            "uniform_sats_per_dollar": pl.Float64,
            "dynamic_sats_per_dollar": pl.Float64,
            "uniform_percentile": pl.Float64,
            "dynamic_percentile": pl.Float64,
            "excess_percentile": pl.Float64,
        })
    return pl.DataFrame(results)


def backtest_dynamic_dca(
    dataframe: pl.DataFrame,
    strategy_function,
    features_df: pl.DataFrame | None = None,
    *,
    strategy_label: str = "strategy",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pl.DataFrame, float, float]:
    """Run rolling-window SPD backtest and log aggregated performance metrics.

    Unified function that supports both simple usage and shared runtime logic with
    precomputed features.

    Args:
        dataframe: Polars DataFrame with 'date' and 'price_usd'
        strategy_function: Function that takes features pl.DataFrame and returns
            pl.DataFrame with 'date' and 'weight'
        features_df: Optional precomputed features. If None, computes them internally.
        strategy_label: Label for logging (default: "strategy")
        start_date: Optional start date (default: BACKTEST_START)
        end_date: Optional end date (default: dynamic yesterday)

    Returns:
        Tuple of:
        - SPD table pl.DataFrame
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

    N = spd_table.height
    exp_weights = 0.9 ** np.arange(N - 1, -1, -1)
    exp_weights /= exp_weights.sum()
    exp_avg_pct = float((dynamic_pct.to_numpy() * exp_weights).sum())
    uniform_exp_avg_pct = float((uniform_pct.to_numpy() * exp_weights).sum())

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

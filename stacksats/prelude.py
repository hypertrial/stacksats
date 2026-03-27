import datetime as dt
import logging

import numpy as np
import polars as pl

from .data_btc import BTCDataProvider
from .framework_contract import ALLOCATION_SPAN_DAYS, ALLOCATION_WINDOW_OFFSET
from .model_development import precompute_features
from .runner_helpers import build_window_bounds, build_window_index, slice_window_or_filter

# Configuration
BACKTEST_START = "2018-01-01"
BACKTEST_END = "2025-12-31"
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
    """Return canonical default scoring end date for backtests."""
    return BACKTEST_END


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
        date_ranges.append((current, end_date))
        current += dt.timedelta(days=1)

    return date_ranges


def date_range_list(
    start: dt.datetime | str,
    end: dt.datetime | str,
) -> list[dt.datetime]:
    """Return list of dates from start to end inclusive (1d interval).

    Generate an inclusive daily date range using canonical Python datetimes.
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


def _uniform_weight_frame(price_slice: pl.DataFrame) -> pl.DataFrame:
    """Return uniform weights aligned to the provided price window."""
    n = price_slice.height
    return pl.DataFrame({
        DATE_COL: price_slice[DATE_COL],
        "weight": [1.0 / n] * n,
    })


def _normalize_weight_frame(
    price_slice: pl.DataFrame,
    weight_df: pl.DataFrame,
) -> pl.DataFrame:
    """Align strategy weights to a price window and normalize safely."""
    if weight_df.is_empty() or "weight" not in weight_df.columns:
        return _uniform_weight_frame(price_slice)

    if (
        DATE_COL in weight_df.columns
        and weight_df.height == price_slice.height
        and weight_df[DATE_COL].equals(price_slice[DATE_COL])
    ):
        aligned = weight_df.select([
            DATE_COL,
            pl.col("weight").cast(pl.Float64, strict=False).fill_null(0.0).alias("weight"),
        ])
    else:
        merged = price_slice.select(DATE_COL).join(
            weight_df.select([DATE_COL, "weight"]),
            on=DATE_COL,
            how="left",
        )
        aligned = merged.select(
            DATE_COL,
            pl.col("weight").cast(pl.Float64, strict=False).fill_null(0.0),
        )

    weights = aligned["weight"]
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _uniform_weight_frame(price_slice)
    return aligned.with_columns((pl.col("weight") / total).alias("weight"))


def _empty_spd_frame() -> pl.DataFrame:
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


def _candidate_windows(start_ts: dt.datetime, end: dt.datetime) -> pl.DataFrame:
    max_start_date = end - WINDOW_OFFSET
    if max_start_date < start_ts:
        return pl.DataFrame(schema={
            "window_start": pl.Datetime("us"),
            "window_end": pl.Datetime("us"),
        })
    start_dates = pl.datetime_range(start_ts, max_start_date, interval="1d", eager=True)
    return pl.DataFrame({"window_start": start_dates}).with_columns(
        (pl.col("window_start") + pl.duration(days=WINDOW_DAYS - 1)).alias("window_end")
    )


def _window_label_expr(start_col: str = "window_start", end_col: str = "window_end") -> pl.Expr:
    return pl.format(
        "{} → {}",
        pl.col(start_col).dt.strftime("%Y-%m-%d"),
        pl.col(end_col).dt.strftime("%Y-%m-%d"),
    ).alias("window")


def _window_slice_from_row(
    frame: pl.DataFrame,
    plan,
    row: dict[str, object],
    *,
    prefix: str,
    expected_days: int | None,
) -> pl.DataFrame:
    can_slice = bool(row.get(f"{prefix}_can_slice", False))
    start_idx = row.get(f"{prefix}_start_idx")
    end_idx = row.get(f"{prefix}_end_idx")
    if can_slice and start_idx is not None and end_idx is not None:
        return frame.slice(int(start_idx), int(end_idx) - int(start_idx) + 1)
    return slice_window_or_filter(
        frame,
        plan,
        row["window_start"],
        row["window_end"],
        expected_days=expected_days,
    )


def _window_percentiles(
    *,
    min_spd: float,
    max_spd: float,
    uniform_spd: float,
    dynamic_spd: float,
) -> tuple[float, float]:
    span = max_spd - min_spd
    if span > 0:
        uniform_pct = (uniform_spd - min_spd) / span * 100
        dynamic_pct = (dynamic_spd - min_spd) / span * 100
    else:
        uniform_pct = float("nan")
        dynamic_pct = float("nan")
    return float(uniform_pct), float(dynamic_pct)


def _batched_spd_windows(
    dataframe: pl.DataFrame,
    full_feat: pl.DataFrame,
    *,
    start_ts: dt.datetime,
    end: dt.datetime,
) -> tuple[pl.DataFrame, pl.DataFrame, object, pl.DataFrame, object, pl.DataFrame]:
    dataframe, price_plan = build_window_index(dataframe.filter(pl.col(DATE_COL) >= start_ts))
    full_feat, feature_plan = build_window_index(full_feat)
    inv_price_frame = dataframe.select(
        DATE_COL,
        (pl.lit(1e8) / pl.col("price_usd").cast(pl.Float64, strict=False)).alias("_inv_price"),
    )
    windows = _candidate_windows(start_ts, end)
    if windows.is_empty():
        return dataframe, price_plan, full_feat, feature_plan, inv_price_frame, windows

    windows = build_window_bounds(
        price_plan,
        windows,
        expected_days=WINDOW_DAYS,
        prefix="price",
    )
    windows = build_window_bounds(
        feature_plan,
        windows,
        expected_days=WINDOW_DAYS,
        prefix="feature",
    )
    if price_plan.has_unique_dates and not inv_price_frame.is_empty():
        inv_metrics = inv_price_frame.with_columns(
            pl.col("_inv_price")
            .rolling_min(window_size=WINDOW_DAYS, min_samples=WINDOW_DAYS)
            .alias("min_sats_per_dollar"),
            pl.col("_inv_price")
            .rolling_max(window_size=WINDOW_DAYS, min_samples=WINDOW_DAYS)
            .alias("max_sats_per_dollar"),
            pl.col("_inv_price")
            .rolling_mean(window_size=WINDOW_DAYS, min_samples=WINDOW_DAYS)
            .alias("uniform_sats_per_dollar"),
        ).select(
            pl.col(DATE_COL).alias("window_end"),
            "min_sats_per_dollar",
            "max_sats_per_dollar",
            "uniform_sats_per_dollar",
        )
        windows = windows.join(inv_metrics, on="window_end", how="left").with_columns(
            pl.when(pl.col("price_can_slice"))
            .then(pl.col("min_sats_per_dollar"))
            .otherwise(None)
            .alias("min_sats_per_dollar"),
            pl.when(pl.col("price_can_slice"))
            .then(pl.col("max_sats_per_dollar"))
            .otherwise(None)
            .alias("max_sats_per_dollar"),
            pl.when(pl.col("price_can_slice"))
            .then(pl.col("uniform_sats_per_dollar"))
            .otherwise(None)
            .alias("uniform_sats_per_dollar"),
        )
    else:
        windows = windows.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("min_sats_per_dollar"),
            pl.lit(None, dtype=pl.Float64).alias("max_sats_per_dollar"),
            pl.lit(None, dtype=pl.Float64).alias("uniform_sats_per_dollar"),
        )
    windows = windows.with_columns(_window_label_expr())
    return dataframe, price_plan, full_feat, feature_plan, inv_price_frame, windows


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
        end_date: Optional end date (default: BACKTEST_END)
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
    data_end = dataframe[DATE_COL].max()
    if data_end is not None:
        end = min(end, data_end)

    if features_df is None:
        full_feat = precompute_features(dataframe)
    else:
        full_feat = _ensure_pl_with_date(features_df).clone()
    full_feat = full_feat.with_columns(_daily_datetime_expr(full_feat).alias(DATE_COL))

    start_ts = dt.datetime.strptime(start[:10], "%Y-%m-%d")
    full_feat = full_feat.filter(
        (pl.col(DATE_COL) >= start_ts) & (pl.col(DATE_COL) <= end)
    )

    if "price_usd_source_exists" in dataframe.columns:
        mask = dataframe["price_usd_source_exists"].fill_null(False)
        valid = dataframe.filter(mask)
        if valid.height > 0:
            source_end = valid[DATE_COL].max()
            if source_end is not None:
                end = min(end, source_end)
                full_feat = full_feat.filter(pl.col(DATE_COL) <= end)
    (
        dataframe,
        price_plan,
        full_feat,
        feature_plan,
        inv_price_frame,
        windows,
    ) = _batched_spd_windows(
        dataframe,
        full_feat,
        start_ts=start_ts,
        end=end,
    )

    if windows.is_empty():
        return _empty_spd_frame()

    logging.info(
        "Backtesting date range: %s to %s (%s total windows)",
        windows["window_start"][0],
        windows["window_end"][-1],
        windows.height,
    )
    if hasattr(strategy_function, "_compute_window_weights_batch"):
        return _compute_cycle_spd_batched(
            full_feat,
            feature_plan,
            inv_price_frame,
            windows,
            strategy_function,
            validate_weights=validate_weights,
        )
    if getattr(strategy_function, "_stacksats_framework_fast_path", False):
        return _compute_cycle_spd_framework(
            dataframe,
            price_plan,
            full_feat,
            feature_plan,
            inv_price_frame,
            windows,
            strategy_function,
            validate_weights=validate_weights,
        )
    return _compute_cycle_spd_generic(
        dataframe,
        price_plan,
        full_feat,
        feature_plan,
        inv_price_frame,
        windows,
        strategy_function,
        validate_weights=validate_weights,
    )


def _validate_window_weights(
    weight_df: pl.DataFrame,
    *,
    window_start: dt.datetime,
    window_end: dt.datetime,
) -> None:
    weight_sum = weight_df["weight"].sum()
    if not np.isclose(float(weight_sum), 1.0, atol=WEIGHT_SUM_TOLERANCE):
        raise ValueError(
            f"Weights for range {window_start.date()} to {window_end.date()} "
            f"sum to {float(weight_sum):.10f}, expected 1.0 "
            f"(tolerance: {WEIGHT_SUM_TOLERANCE})"
        )


def _spd_metrics_for_row(
    row: dict[str, object],
    spd_frame: pl.DataFrame,
) -> tuple[float, float, float]:
    min_spd = row.get("min_sats_per_dollar")
    max_spd = row.get("max_sats_per_dollar")
    uniform_spd = row.get("uniform_sats_per_dollar")
    if min_spd is None or max_spd is None or uniform_spd is None:
        inv_price = spd_frame["_inv_price"]
        return (
            float(inv_price.min()),
            float(inv_price.max()),
            float(inv_price.mean()),
        )
    return float(min_spd), float(max_spd), float(uniform_spd)


def _dynamic_spd(spd_frame: pl.DataFrame, weight_df: pl.DataFrame, uniform_spd: float) -> float:
    if weight_df.height != spd_frame.height:
        return float(uniform_spd)  # pragma: no cover
    return float(
        spd_frame.with_columns(weight_df["weight"].alias("_weight"))
        .select((pl.col("_weight") * pl.col("_inv_price")).sum())
        .item()
    )


def _compute_cycle_spd_generic(
    dataframe: pl.DataFrame,
    price_plan,
    full_feat: pl.DataFrame,
    feature_plan,
    inv_price_frame: pl.DataFrame,
    windows: pl.DataFrame,
    strategy_function,
    *,
    validate_weights: bool,
) -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    validated_windows = 0
    for idx in range(windows.height):
        row = windows.row(idx, named=True)
        price_slice = _window_slice_from_row(
            dataframe,
            price_plan,
            row,
            prefix="price",
            expected_days=WINDOW_DAYS,
        )
        if price_slice.is_empty():
            continue

        window_feat = _window_slice_from_row(
            full_feat,
            feature_plan,
            row,
            prefix="feature",
            expected_days=WINDOW_DAYS,
        )
        if window_feat.height != WINDOW_DAYS:
            continue

        weight_df = _normalize_weight_frame(price_slice, strategy_function(window_feat))
        if validate_weights:
            _validate_window_weights(
                weight_df,
                window_start=row["window_start"],
                window_end=row["window_end"],
            )
            validated_windows += 1

        spd_frame = _window_slice_from_row(
            inv_price_frame,
            price_plan,
            row,
            prefix="price",
            expected_days=WINDOW_DAYS,
        )
        min_spd, max_spd, uniform_spd = _spd_metrics_for_row(row, spd_frame)
        dynamic_spd = _dynamic_spd(spd_frame, weight_df, uniform_spd)
        uniform_pct, dynamic_pct = _window_percentiles(
            min_spd=min_spd,
            max_spd=max_spd,
            uniform_spd=uniform_spd,
            dynamic_spd=dynamic_spd,
        )
        rows.append({
            "window": row["window"],
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
            "✓ Validated weight sums for %s windows (all sum to 1.0)",
            validated_windows,
        )
    return pl.DataFrame(rows) if rows else _empty_spd_frame()


def _compute_cycle_spd_framework(
    dataframe: pl.DataFrame,
    price_plan,
    full_feat: pl.DataFrame,
    feature_plan,
    inv_price_frame: pl.DataFrame,
    windows: pl.DataFrame,
    strategy_function,
    *,
    validate_weights: bool,
) -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    validated_windows = 0
    compute_window_weights = getattr(strategy_function, "_compute_window_weights", strategy_function)

    for idx in range(windows.height):
        row = windows.row(idx, named=True)
        price_slice = _window_slice_from_row(
            dataframe,
            price_plan,
            row,
            prefix="price",
            expected_days=WINDOW_DAYS,
        )
        if price_slice.is_empty():
            continue  # pragma: no cover
        window_feat = _window_slice_from_row(
            full_feat,
            feature_plan,
            row,
            prefix="feature",
            expected_days=WINDOW_DAYS,
        )
        if window_feat.height != WINDOW_DAYS:
            continue  # pragma: no cover
        weight_df = _normalize_weight_frame(
            price_slice,
            compute_window_weights(
                window_feat,
                window_start=row["window_start"],
                window_end=row["window_end"],
            ),
        )
        if validate_weights:
            _validate_window_weights(
                weight_df,
                window_start=row["window_start"],
                window_end=row["window_end"],
            )
            validated_windows += 1

        spd_frame = _window_slice_from_row(
            inv_price_frame,
            price_plan,
            row,
            prefix="price",
            expected_days=WINDOW_DAYS,
        )
        min_spd, max_spd, uniform_spd = _spd_metrics_for_row(row, spd_frame)
        dynamic_spd = _dynamic_spd(spd_frame, weight_df, uniform_spd)
        uniform_pct, dynamic_pct = _window_percentiles(
            min_spd=min_spd,
            max_spd=max_spd,
            uniform_spd=uniform_spd,
            dynamic_spd=dynamic_spd,
        )
        rows.append({
            "window": row["window"],
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
            "✓ Validated weight sums for %s windows (all sum to 1.0)",
            validated_windows,
        )
    if not rows:
        return _empty_spd_frame()  # pragma: no cover
    return pl.DataFrame(rows)


def _compute_cycle_spd_batched(
    full_feat: pl.DataFrame,
    feature_plan,
    inv_price_frame: pl.DataFrame,
    windows: pl.DataFrame,
    strategy_function,
    *,
    validate_weights: bool,
) -> pl.DataFrame:
    batch_fn = getattr(strategy_function, "_compute_window_weights_batch")
    weights_df = batch_fn(
        full_feat,
        feature_plan=feature_plan,
        windows=windows,
        expected_days=WINDOW_DAYS,
    )
    if weights_df.is_empty():
        return _empty_spd_frame()  # pragma: no cover

    if validate_weights:
        invalid = weights_df.group_by(["window_start", "window_end"]).agg(
            pl.col("weight").sum().alias("_weight_sum"),
        ).filter((pl.col("_weight_sum") - 1.0).abs() > WEIGHT_SUM_TOLERANCE)
        if not invalid.is_empty():
            row = invalid.row(0, named=True)
            raise ValueError(
                "Batched framework weights failed sum validation for range "
                f"{row['window_start'].date()} to {row['window_end'].date()} "
                f"(sum={float(row['_weight_sum']):.10f})."
            )

    dynamic = (
        weights_df.join(inv_price_frame, on=DATE_COL, how="left")
        .with_columns((pl.col("weight") * pl.col("_inv_price")).alias("_dynamic_spd"))
        .group_by(["window_start", "window_end"])
        .agg(pl.col("_dynamic_spd").sum().alias("dynamic_sats_per_dollar"))
    )
    result = windows.select(
        "window_start",
        "window_end",
        "window",
        "min_sats_per_dollar",
        "max_sats_per_dollar",
        "uniform_sats_per_dollar",
    ).join(
        dynamic,
        on=["window_start", "window_end"],
        how="inner",
    )
    result = result.with_columns(
        pl.when((pl.col("max_sats_per_dollar") - pl.col("min_sats_per_dollar")) > 0.0)
        .then(
            (
                (pl.col("uniform_sats_per_dollar") - pl.col("min_sats_per_dollar"))
                / (pl.col("max_sats_per_dollar") - pl.col("min_sats_per_dollar"))
            )
            * 100.0
        )
        .otherwise(float("nan"))
        .alias("uniform_percentile"),
        pl.when((pl.col("max_sats_per_dollar") - pl.col("min_sats_per_dollar")) > 0.0)
        .then(
            (
                (pl.col("dynamic_sats_per_dollar") - pl.col("min_sats_per_dollar"))
                / (pl.col("max_sats_per_dollar") - pl.col("min_sats_per_dollar"))
            )
            * 100.0
        )
        .otherwise(float("nan"))
        .alias("dynamic_percentile"),
    ).with_columns(
        (pl.col("dynamic_percentile") - pl.col("uniform_percentile")).alias("excess_percentile")
    )
    return result.select(
        "window",
        "min_sats_per_dollar",
        "max_sats_per_dollar",
        "uniform_sats_per_dollar",
        "dynamic_sats_per_dollar",
        "uniform_percentile",
        "dynamic_percentile",
        "excess_percentile",
    ).sort("window")


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
        end_date: Optional end date (default: BACKTEST_END)

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

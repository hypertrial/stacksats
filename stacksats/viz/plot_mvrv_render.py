"""Rendering helpers for plot_mvrv."""

from __future__ import annotations

import datetime as dt

import polars as pl


def _parse_date(s: str | None) -> dt.datetime | None:
    if s is None:
        return None
    if "T" in s or " " in s:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    return dt.datetime.strptime(s, "%Y-%m-%d")


def plot_mvrv_metrics(
    df: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str = "mvrv_metrics.svg",
    *,
    init_plot_env_fn,
    logging_mod,
    plt_mod,
    mdates_mod,
    path_cls,
) -> None:
    """Plot mvrv and CapMVRVZ metrics over time."""
    init_plot_env_fn()

    if "mvrv" not in df.columns:
        available_cols = [
            c for c in df.columns if "MVRV" in c.upper() or "CAP" in c.upper()
        ]
        raise ValueError(
            f"Missing required column: mvrv. "
            f"Available MVRV/Cap columns: {available_cols if available_cols else 'None'}"
        )

    if "date" not in df.columns:
        raise ValueError("DataFrame must have a 'date' column.")

    if "CapMVRVZ" not in df.columns:
        logging_mod.info(
            "CapMVRVZ not found in data. Calculating MVRV Z-Score from mvrv..."
        )
        mvrv_mean = df["mvrv"].rolling_mean(window_size=365, min_samples=30)
        mvrv_std = df["mvrv"].rolling_std(window_size=365, min_samples=30)
        df = df.with_columns(
            ((pl.col("mvrv") - mvrv_mean) / mvrv_std).alias("CapMVRVZ")
        )
        logging_mod.info(
            "✓ Calculated CapMVRVZ from mvrv using 365-day rolling window"
        )

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    if start_dt is not None:
        df = df.filter(pl.col("date") >= start_dt)
    if end_dt is not None:
        df = df.filter(pl.col("date") <= end_dt)

    if df.is_empty():
        raise ValueError("No data available for the specified date range")

    df_clean = (
        df.select(["date", "mvrv", "CapMVRVZ"])
        .drop_nulls()
        .filter(pl.col("mvrv").is_finite() & pl.col("CapMVRVZ").is_finite())
    )

    if df_clean.is_empty():
        raise ValueError("No valid MVRV data available after removing missing values")

    date_col = df_clean["date"]
    date_min = date_col.min()
    date_max = date_col.max()
    date_min_py = date_min if isinstance(date_min, dt.datetime) else dt.datetime.fromisoformat(str(date_min)[:10])
    date_max_py = date_max if isinstance(date_max, dt.datetime) else dt.datetime.fromisoformat(str(date_max)[:10])

    logging_mod.info(
        f"Plotting MVRV metrics: {len(df_clean)} data points from "
        f"{date_min_py.date()} to {date_max_py.date()}"
    )

    mvrv_ma30 = df_clean["mvrv"].rolling_mean(window_size=30, min_samples=1)
    zscore_ma30 = df_clean["CapMVRVZ"].rolling_mean(window_size=30, min_samples=1)

    fig, (ax1, ax2) = plt_mod.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(
        date_col.to_list(),
        df_clean["mvrv"].to_list(),
        linewidth=1.5,
        color="#2563eb",
        alpha=0.5,
        label="MVRV Ratio (Daily)",
    )
    ax1.fill_between(
        date_col.to_list(),
        df_clean["mvrv"].to_list(),
        alpha=0.2,
        color="#2563eb",
    )
    ax1.plot(
        date_col.to_list(),
        mvrv_ma30.to_list(),
        linewidth=2.5,
        color="#1e40af",
        label="30-Day MA",
    )

    ax1.axhline(
        y=1.0,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Fair Value (1.0)",
    )

    ax1.set_title(
        "Bitcoin MVRV Ratio (Market Value / Realized Value)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_ylabel("MVRV Ratio", fontsize=12, fontweight="medium")
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.95)

    mvrv_mean = float(df_clean["mvrv"].mean())
    mvrv_median = float(df_clean["mvrv"].median())
    mvrv_min = float(df_clean["mvrv"].min())
    mvrv_max = float(df_clean["mvrv"].max())
    mvrv_current = float(df_clean["mvrv"][-1])
    mvrv_ma30_current = float(mvrv_ma30[-1])

    stats_text1 = (
        f"Current: {mvrv_current:.2f}\n"
        f"30-Day MA: {mvrv_ma30_current:.2f}\n"
        f"Mean: {mvrv_mean:.2f}\n"
        f"Median: {mvrv_median:.2f}\n"
        f"Range: {mvrv_min:.2f} - {mvrv_max:.2f}"
    )

    ax1.text(
        0.98,
        0.98,
        stats_text1,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            alpha=0.95,
            edgecolor="#e5e7eb",
            linewidth=1.5,
        ),
        family="monospace",
    )

    ax2.plot(
        date_col.to_list(),
        df_clean["CapMVRVZ"].to_list(),
        linewidth=1.5,
        color="#16a34a",
        alpha=0.5,
        label="MVRV Z-Score (Daily)",
    )
    ax2.fill_between(
        date_col.to_list(),
        df_clean["CapMVRVZ"].to_list(),
        alpha=0.2,
        color="#16a34a",
    )
    ax2.plot(
        date_col.to_list(),
        zscore_ma30.to_list(),
        linewidth=2.5,
        color="#15803d",
        label="30-Day MA",
    )

    ax2.axhline(y=0, color="#6b7280", linestyle="-", linewidth=1, alpha=0.5, label="Mean (0)")
    ax2.axhline(
        y=2,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Overvalued (+2σ)",
    )
    ax2.axhline(
        y=-2,
        color="#3b82f6",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Undervalued (-2σ)",
    )

    ax2.set_title("Bitcoin MVRV Z-Score", fontsize=16, fontweight="bold", pad=15)
    ax2.set_xlabel("Date", fontsize=12, fontweight="medium")
    ax2.set_ylabel("MVRV Z-Score", fontsize=12, fontweight="medium")
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)

    zscore_mean = float(df_clean["CapMVRVZ"].mean())
    zscore_median = float(df_clean["CapMVRVZ"].median())
    zscore_min = float(df_clean["CapMVRVZ"].min())
    zscore_max = float(df_clean["CapMVRVZ"].max())
    zscore_current = float(df_clean["CapMVRVZ"][-1])
    zscore_ma30_current = float(zscore_ma30[-1])

    stats_text2 = (
        f"Current: {zscore_current:.2f}\n"
        f"30-Day MA: {zscore_ma30_current:.2f}\n"
        f"Mean: {zscore_mean:.2f}\n"
        f"Median: {zscore_median:.2f}\n"
        f"Range: {zscore_min:.2f} - {zscore_max:.2f}"
    )

    ax2.text(
        0.98,
        0.98,
        stats_text2,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            alpha=0.95,
            edgecolor="#e5e7eb",
            linewidth=1.5,
        ),
        family="monospace",
    )

    ax2.xaxis.set_major_formatter(mdates_mod.DateFormatter("%Y-%m-%d"))
    date_range_days = (date_max_py - date_min_py).days
    if date_range_days > 365:
        ax2.xaxis.set_major_locator(mdates_mod.YearLocator())
        ax2.xaxis.set_minor_locator(mdates_mod.MonthLocator((1, 7)))
    elif date_range_days > 90:
        ax2.xaxis.set_major_locator(mdates_mod.MonthLocator())
        ax2.xaxis.set_minor_locator(mdates_mod.WeekdayLocator())
    else:
        ax2.xaxis.set_major_locator(mdates_mod.WeekdayLocator())
    plt_mod.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=10)

    plt_mod.tight_layout()
    plt_mod.savefig(
        output_path,
        format=path_cls(output_path).suffix[1:] or "svg",
        bbox_inches="tight",
        dpi=300,
    )
    plt_mod.close()

    logging_mod.info(f"✓ Plot saved to {output_path}")
    logging_mod.info(
        f"  Date range: {date_min_py.date()} to {date_max_py.date()}"
    )
    logging_mod.info(f"  Data points: {len(df_clean)}")
    logging_mod.info(
        f"  Current MVRV Ratio: {mvrv_current:.2f} (30-Day MA: {mvrv_ma30_current:.2f})"
    )
    logging_mod.info(
        f"  Current MVRV Z-Score: {zscore_current:.2f} (30-Day MA: {zscore_ma30_current:.2f})"
    )

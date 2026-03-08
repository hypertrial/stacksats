"""Rendering helpers for plot_mvrv."""

from __future__ import annotations

import pandas as pd


def plot_mvrv_metrics(
    df: pd.DataFrame,
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

    if "CapMVRVZ" not in df.columns:
        logging_mod.info(
            "CapMVRVZ not found in data. Calculating MVRV Z-Score from mvrv..."
        )
        mvrv_mean = df["mvrv"].rolling(window=365, min_periods=30).mean()
        mvrv_std = df["mvrv"].rolling(window=365, min_periods=30).std()
        df["CapMVRVZ"] = (df["mvrv"] - mvrv_mean) / mvrv_std
        logging_mod.info(
            "✓ Calculated CapMVRVZ from mvrv using 365-day rolling window"
        )

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if len(df) == 0:
        raise ValueError("No data available for the specified date range")

    df_clean = df[["mvrv", "CapMVRVZ"]].dropna()

    if len(df_clean) == 0:
        raise ValueError("No valid MVRV data available after removing missing values")

    logging_mod.info(
        f"Plotting MVRV metrics: {len(df_clean)} data points from "
        f"{df_clean.index.min().date()} to {df_clean.index.max().date()}"
    )

    mvrv_ma30 = df_clean["mvrv"].rolling(window=30, min_periods=1).mean()
    zscore_ma30 = df_clean["CapMVRVZ"].rolling(window=30, min_periods=1).mean()

    fig, (ax1, ax2) = plt_mod.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(
        df_clean.index,
        df_clean["mvrv"],
        linewidth=1.5,
        color="#2563eb",
        alpha=0.5,
        label="MVRV Ratio (Daily)",
    )
    ax1.fill_between(
        df_clean.index,
        df_clean["mvrv"],
        alpha=0.2,
        color="#2563eb",
    )
    ax1.plot(
        df_clean.index,
        mvrv_ma30,
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

    mvrv_mean = df_clean["mvrv"].mean()
    mvrv_median = df_clean["mvrv"].median()
    mvrv_min = df_clean["mvrv"].min()
    mvrv_max = df_clean["mvrv"].max()
    mvrv_current = df_clean["mvrv"].iloc[-1]
    mvrv_ma30_current = mvrv_ma30.iloc[-1]

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
        df_clean.index,
        df_clean["CapMVRVZ"],
        linewidth=1.5,
        color="#16a34a",
        alpha=0.5,
        label="MVRV Z-Score (Daily)",
    )
    ax2.fill_between(
        df_clean.index,
        df_clean["CapMVRVZ"],
        alpha=0.2,
        color="#16a34a",
    )
    ax2.plot(
        df_clean.index,
        zscore_ma30,
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

    zscore_mean = df_clean["CapMVRVZ"].mean()
    zscore_median = df_clean["CapMVRVZ"].median()
    zscore_min = df_clean["CapMVRVZ"].min()
    zscore_max = df_clean["CapMVRVZ"].max()
    zscore_current = df_clean["CapMVRVZ"].iloc[-1]
    zscore_ma30_current = zscore_ma30.iloc[-1]

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
    date_range_days = (df_clean.index.max() - df_clean.index.min()).days
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
        f"  Date range: {df_clean.index.min().date()} to {df_clean.index.max().date()}"
    )
    logging_mod.info(f"  Data points: {len(df_clean)}")
    logging_mod.info(
        f"  Current MVRV Ratio: {mvrv_current:.2f} (30-Day MA: {mvrv_ma30_current:.2f})"
    )
    logging_mod.info(
        f"  Current MVRV Z-Score: {zscore_current:.2f} (30-Day MA: {zscore_ma30_current:.2f})"
    )

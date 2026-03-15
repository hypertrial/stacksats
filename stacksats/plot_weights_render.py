"""Rendering helpers for plot_weights."""

from __future__ import annotations

import polars as pl


def plot_dca_weights(
    df,
    start_date: str,
    end_date: str,
    output_path: str,
    *,
    init_plot_env_fn,
    plt_mod,
    mdates_mod,
):
    """Create and save a plot of DCA weights over time."""
    init_plot_env_fn()

    original_weight_sum = float(df["weight"].sum())
    if original_weight_sum > 0 and abs(original_weight_sum - 1.0) > 1e-10:
        df = df.with_columns((pl.col("weight") / original_weight_sum).alias("weight"))

    fig, ax = plt_mod.subplots(figsize=(14, 8))

    price_expr = pl.col("price_usd").cast(pl.Float64, strict=False)
    past_df = df.filter(price_expr.is_not_null() & price_expr.is_not_nan())
    future_df = df.filter(pl.col("price_usd").is_null())

    has_past = past_df.height > 0
    has_future = future_df.height > 0

    if has_past:
        past_weights = past_df["weight"].to_numpy()
        past_mean = float(past_weights.mean())
        past_min = float(past_weights.min())
        past_max = float(past_weights.max())
        past_min_row = past_df.filter(pl.col("weight") == past_df["weight"].min()).row(0, named=True)
        past_max_row = past_df.filter(pl.col("weight") == past_df["weight"].max()).row(0, named=True)
        past_min_date = past_min_row["date"]
        past_max_date = past_max_row["date"]

    all_weights = df["weight"].to_numpy()
    min_weight = all_weights.min()
    max_weight = all_weights.max()

    if has_past:
        ax.fill_between(
            past_df["date"].to_list(),
            past_df["weight"].to_list(),
            alpha=0.3,
            color="#2563eb",
            label=f"Past Weights (n={len(past_df)})",
        )
        ax.plot(
            past_df["date"].to_list(),
            past_df["weight"].to_list(),
            linewidth=2.5,
            color="#1e40af",
            marker="o",
            markersize=3,
            markevery=max(1, len(past_df) // 30),
            zorder=3,
        )

    if has_future:
        ax.fill_between(
            future_df["date"].to_list(),
            future_df["weight"].to_list(),
            alpha=0.2,
            color="#f97316",
            label=f"Future Weights (n={len(future_df)})",
        )
        ax.plot(
            future_df["date"].to_list(),
            future_df["weight"].to_list(),
            linewidth=2,
            color="#ea580c",
            linestyle="--",
            marker="s",
            markersize=2,
            markevery=max(1, len(future_df) // 30),
            alpha=0.8,
            zorder=3,
        )

    if has_past and has_future:
        boundary_date = past_df["date"].max()
        bd_str = boundary_date.strftime("%Y-%m-%d") if hasattr(boundary_date, "strftime") else str(boundary_date)[:10]
        ax.axvline(
            x=boundary_date,
            color="#6b7280",
            linestyle=":",
            linewidth=2,
            alpha=0.8,
            label=f"Today: {bd_str}",
            zorder=2,
        )

    if has_past:
        ax.axhline(
            y=past_mean,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Past Mean: {past_mean:.6f}",
            zorder=2,
        )

        ax.scatter(
            [past_min_date],
            [past_min],
            color="#16a34a",
            s=150,
            marker="v",
            edgecolors="white",
            linewidths=2,
            zorder=4,
            label=f"Past Min: {past_min:.6f}",
        )
        ax.scatter(
            [past_max_date],
            [past_max],
            color="#dc2626",
            s=150,
            marker="^",
            edgecolors="white",
            linewidths=2,
            zorder=4,
            label=f"Past Max: {past_max:.6f}",
        )

    ax.set_title(
        f"DCA Investment Weights Distribution\n{start_date} to {end_date}",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )
    ax.set_xlabel("DCA Date", fontsize=13, fontweight="medium")
    ax.set_ylabel("Investment Weight (log scale)", fontsize=13, fontweight="medium")

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    ax.xaxis.set_major_formatter(mdates_mod.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates_mod.MonthLocator(interval=max(1, df.height // 365)))
    plt_mod.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=10)

    ax.set_yscale("log")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(bottom=min_weight * 0.8, top=max_weight * 1.5)

    if has_past:
        past_stats = past_df["weight"].describe()
        past_p25 = float(past_df["weight"].quantile(0.25))
        past_p75 = float(past_df["weight"].quantile(0.75))
        past_median = float(past_df["weight"].median())
        mean_val = past_stats.filter(pl.col("statistic") == "mean")["value"][0]
        std_val = past_stats.filter(pl.col("statistic") == "std")["value"][0]
        min_val = past_stats.filter(pl.col("statistic") == "min")["value"][0]
        max_val = past_stats.filter(pl.col("statistic") == "max")["value"][0]

        stats_text = (
            f"Past Weight Stats (n={past_df.height}):\n"
            f"Mean:   {mean_val:.6f}\n"
            f"Median: {past_median:.6f}\n"
            f"Std:    {std_val:.6f}\n"
            f"Min:    {min_val:.6f}\n"
            f"Max:    {max_val:.6f}\n"
            f"P25:    {past_p25:.6f}\n"
            f"P75:    {past_p75:.6f}"
        )
    else:
        stats_text = "No past weights available"

    if has_future:
        f_mean = float(future_df["weight"].mean())
        f_min = float(future_df["weight"].min())
        f_max = float(future_df["weight"].max())
        stats_text += (
            f"\n\nFuture Weights (n={future_df.height}):\n"
            f"Mean:   {f_mean:.6f}\n"
            f"Min:    {f_min:.6f}\n"
            f"Max:    {f_max:.6f}"
        )

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
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
        zorder=5,
    )

    ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.95,
        edgecolor="#e5e7eb",
        fancybox=True,
    )

    plt_mod.tight_layout()
    plt_mod.savefig(output_path, format="svg", bbox_inches="tight", dpi=300)
    plt_mod.close()

    print(f"✓ Plot saved to {output_path}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Total data points: {df.height}")
    print(f"  Past weights (with price): {past_df.height}")
    print(f"  Future weights (no price): {future_df.height}")

    if abs(original_weight_sum - 1.0) > 1e-10:
        print(f"  ⚠ Weights normalized from {original_weight_sum:.6f} to 1.0")
    else:
        print("  Weights sum to 1.0 ✓")

    if has_past:
        past_p25 = float(past_df["weight"].quantile(0.25))
        past_p75 = float(past_df["weight"].quantile(0.75))
        past_median = float(past_df["weight"].median())
        past_std = float(past_df["weight"].std())

        print(f"\nPast Weight Statistics (n={past_df.height}):")
        print(f"  Mean:   {past_mean:.6f}")
        print(f"  Median: {past_median:.6f}")
        print(f"  Std:    {past_std:.6f}")
        print(f"  Min:    {past_min:.6f}")
        print(f"  Max:    {past_max:.6f}")
        print(f"  P25:    {past_p25:.6f}")
        print(f"  P75:    {past_p75:.6f}")
        print(f"  Range:  {past_max - past_min:.6f}")

    if has_future:
        f_mean = float(future_df["weight"].mean())
        f_min = float(future_df["weight"].min())
        f_max = float(future_df["weight"].max())
        print(f"\nFuture Weight Statistics (n={future_df.height}):")
        print(f"  Mean:   {f_mean:.6f}")
        print(f"  Min:    {f_min:.6f}")
        print(f"  Max:    {f_max:.6f}")
        print(f"  Range:  {f_max - f_min:.6f}")

    if has_past:
        print("\nSummary:")
        print(f"  Past mean weight: {past_mean:.6f}")
        print(f"  Past weight range: {past_min:.6f} to {past_max:.6f}")

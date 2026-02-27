"""Rendering helpers for plot_weights."""

from __future__ import annotations


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

    original_weight_sum = df["weight"].sum()
    if original_weight_sum > 0 and abs(original_weight_sum - 1.0) > 1e-10:
        df = df.copy()
        df["weight"] = df["weight"] / original_weight_sum

    fig, ax = plt_mod.subplots(figsize=(14, 8))

    past_df = df[df["price_usd"].notna()].copy()
    future_df = df[df["price_usd"].isna()].copy()

    has_past = len(past_df) > 0
    has_future = len(future_df) > 0

    if has_past:
        past_weights = past_df["weight"].values
        past_mean = past_weights.mean()
        past_min = past_weights.min()
        past_max = past_weights.max()
        past_min_date = past_df.loc[past_df["weight"].idxmin(), "date"]
        past_max_date = past_df.loc[past_df["weight"].idxmax(), "date"]

    all_weights = df["weight"].values
    min_weight = all_weights.min()
    max_weight = all_weights.max()

    if has_past:
        ax.fill_between(
            past_df["date"],
            past_df["weight"],
            alpha=0.3,
            color="#2563eb",
            label=f"Past Weights (n={len(past_df)})",
        )
        ax.plot(
            past_df["date"],
            past_df["weight"],
            linewidth=2.5,
            color="#1e40af",
            marker="o",
            markersize=3,
            markevery=max(1, len(past_df) // 30),
            zorder=3,
        )

    if has_future:
        ax.fill_between(
            future_df["date"],
            future_df["weight"],
            alpha=0.2,
            color="#f97316",
            label=f"Future Weights (n={len(future_df)})",
        )
        ax.plot(
            future_df["date"],
            future_df["weight"],
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
        ax.axvline(
            x=boundary_date,
            color="#6b7280",
            linestyle=":",
            linewidth=2,
            alpha=0.8,
            label=f"Today: {boundary_date.strftime('%Y-%m-%d')}",
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
    ax.xaxis.set_major_locator(mdates_mod.MonthLocator(interval=max(1, len(df) // 365)))
    plt_mod.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=10)

    ax.set_yscale("log")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(bottom=min_weight * 0.8, top=max_weight * 1.5)

    if has_past:
        past_stats = past_df["weight"].describe()
        past_p25 = past_df["weight"].quantile(0.25)
        past_p75 = past_df["weight"].quantile(0.75)
        past_median = past_df["weight"].median()

        stats_text = (
            f"Past Weight Stats (n={len(past_df)}):\n"
            f"Mean:   {past_stats['mean']:.6f}\n"
            f"Median: {past_median:.6f}\n"
            f"Std:    {past_stats['std']:.6f}\n"
            f"Min:    {past_stats['min']:.6f}\n"
            f"Max:    {past_stats['max']:.6f}\n"
            f"P25:    {past_p25:.6f}\n"
            f"P75:    {past_p75:.6f}"
        )
    else:
        stats_text = "No past weights available"

    if has_future:
        future_stats = future_df["weight"].describe()
        stats_text += (
            f"\n\nFuture Weights (n={len(future_df)}):\n"
            f"Mean:   {future_stats['mean']:.6f}\n"
            f"Min:    {future_stats['min']:.6f}\n"
            f"Max:    {future_stats['max']:.6f}"
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
    print(f"  Total data points: {len(df)}")
    print(f"  Past weights (with price): {len(past_df)}")
    print(f"  Future weights (no price): {len(future_df)}")

    if abs(original_weight_sum - 1.0) > 1e-10:
        print(f"  ⚠ Weights normalized from {original_weight_sum:.6f} to 1.0")
    else:
        print("  Weights sum to 1.0 ✓")

    if has_past:
        past_stats = past_df["weight"].describe()
        past_p25 = past_df["weight"].quantile(0.25)
        past_p75 = past_df["weight"].quantile(0.75)
        past_median = past_df["weight"].median()

        print(f"\nPast Weight Statistics (n={len(past_df)}):")
        print(f"  Mean:   {past_stats['mean']:.6f}")
        print(f"  Median: {past_median:.6f}")
        print(f"  Std:    {past_stats['std']:.6f}")
        print(f"  Min:    {past_stats['min']:.6f}")
        print(f"  Max:    {past_stats['max']:.6f}")
        print(f"  P25:    {past_p25:.6f}")
        print(f"  P75:    {past_p75:.6f}")
        print(f"  Range:  {past_max - past_min:.6f}")

    if has_future:
        future_stats = future_df["weight"].describe()
        print(f"\nFuture Weight Statistics (n={len(future_df)}):")
        print(f"  Mean:   {future_stats['mean']:.6f}")
        print(f"  Min:    {future_stats['min']:.6f}")
        print(f"  Max:    {future_stats['max']:.6f}")
        print(f"  Range:  {future_stats['max'] - future_stats['min']:.6f}")

    if has_past:
        print("\nSummary:")
        print(f"  Past mean weight: {past_mean:.6f}")
        print(f"  Past weight range: {past_min:.6f} to {past_max:.6f}")

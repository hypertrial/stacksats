"""Backtest utilities: weight computation and metrics export."""

import json
import logging
import os
from datetime import datetime

import polars as pl

from .._contract import PUBLIC_ARTIFACT_SCHEMA_VERSION
from ..model_development import compute_window_weights


def compute_weights_with_features(
    df_window: pl.DataFrame,
    *,
    features_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute window weights with explicit feature input."""
    if df_window.is_empty():
        return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})

    start_date = df_window["date"].min()
    end_date = df_window["date"].max()
    current_date = end_date
    return compute_window_weights(features_df, start_date, end_date, current_date)


def export_metrics_json(
    df_spd: pl.DataFrame, metrics: dict, output_dir: str = "output"
) -> str:
    """Export all metrics to JSON file. Returns path to written file."""
    os.makedirs(output_dir, exist_ok=True)

    json_data = {
        "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": metrics,
        "window_level_data": [],
    }

    artifact_df = df_spd.with_columns(
        pl.col("window")
        .str.split_exact(" → ", 1)
        .struct.field("field_0")
        .str.strip_chars()
        .str.to_datetime(strict=False)
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
        .alias("start_date")
    ).select(
        "window",
        "start_date",
        pl.col("dynamic_percentile").cast(pl.Float64, strict=False),
        pl.col("uniform_percentile").cast(pl.Float64, strict=False),
        pl.col("excess_percentile").cast(pl.Float64, strict=False),
        pl.col("dynamic_sats_per_dollar").cast(pl.Float64, strict=False),
        pl.col("uniform_sats_per_dollar").cast(pl.Float64, strict=False),
        pl.col("min_sats_per_dollar").cast(pl.Float64, strict=False),
        pl.col("max_sats_per_dollar").cast(pl.Float64, strict=False),
    )
    json_data["window_level_data"] = artifact_df.to_dicts()

    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logging.info("✓ Saved: %s", output_path)
    return output_path

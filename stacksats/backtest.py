"""Backtest utilities: weight computation and metrics export."""

import json
import logging
import os
from datetime import datetime

import polars as pl

from .model_development import compute_window_weights
from .prelude import parse_window_dates


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
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": metrics,
        "window_level_data": [],
    }

    for row in df_spd.iter_rows(named=True):
        window_label = row.get("window", "")
        json_data["window_level_data"].append({
            "window": window_label,
            "start_date": parse_window_dates(str(window_label)).isoformat(),
            "dynamic_percentile": float(row.get("dynamic_percentile", 0)),
            "uniform_percentile": float(row.get("uniform_percentile", 0)),
            "excess_percentile": float(row.get("excess_percentile", 0)),
            "dynamic_sats_per_dollar": float(row.get("dynamic_sats_per_dollar", 0)),
            "uniform_sats_per_dollar": float(row.get("uniform_sats_per_dollar", 0)),
            "min_sats_per_dollar": float(row.get("min_sats_per_dollar", 0)),
            "max_sats_per_dollar": float(row.get("max_sats_per_dollar", 0)),
        })

    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logging.info("✓ Saved: %s", output_path)
    return output_path

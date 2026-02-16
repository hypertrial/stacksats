"""Metadata models and time helpers for strategy time series."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@dataclass(frozen=True, slots=True)
class StrategySeriesMetadata:
    """Provenance and window metadata for a single strategy time series."""

    strategy_id: str
    strategy_version: str
    run_id: str
    config_hash: str
    schema_version: str = "1.0.0"
    generated_at: pd.Timestamp = field(default_factory=_utc_now)
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None

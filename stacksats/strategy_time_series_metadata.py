"""Metadata models and time helpers for strategy time series."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _normalize_generated_at(value: pd.Timestamp | str | None) -> pd.Timestamp:
    ts = _utc_now() if value is None else pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_window_date(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


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

    def __post_init__(self) -> None:
        for field_name in (
            "strategy_id",
            "strategy_version",
            "run_id",
            "config_hash",
            "schema_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"StrategySeriesMetadata.{field_name} must be a non-empty string.")

        generated_at = _normalize_generated_at(self.generated_at)
        window_start = _normalize_window_date(self.window_start)
        window_end = _normalize_window_date(self.window_end)
        if window_start is not None and window_end is not None and window_start > window_end:
            raise ValueError("StrategySeriesMetadata.window_start must be <= window_end.")

        object.__setattr__(self, "generated_at", generated_at)
        object.__setattr__(self, "window_start", window_start)
        object.__setattr__(self, "window_end", window_end)

"""Metadata models and time helpers for strategy time series."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _normalize_generated_at(value: dt.datetime | str | None) -> dt.datetime:
    ts = _utc_now() if value is None else _parse_datetime(value)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _parse_datetime(value: str | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return dt.datetime.strptime(value[:10], "%Y-%m-%d")
    raise TypeError(f"Expected datetime or str, got {type(value)}")


def _normalize_window_date(value: dt.datetime | str | None) -> dt.datetime | None:
    if value is None:
        return None
    ts = _parse_datetime(value) if isinstance(value, str) else value
    if ts.tzinfo is not None:
        ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass(frozen=True, slots=True)
class StrategySeriesMetadata:
    """Provenance and window metadata for a single strategy time series."""

    strategy_id: str
    strategy_version: str
    run_id: str
    config_hash: str
    schema_version: str = "1.0.0"
    generated_at: dt.datetime = field(default_factory=_utc_now)
    window_start: dt.datetime | None = None
    window_end: dt.datetime | None = None

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

"""Batch container for strategy time-series outputs."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import polars as pl

from .strategy_time_series_metadata import (
    StrategySeriesMetadata,
    _normalize_generated_at,
    _utc_now,
)
from .strategy_time_series_schema import ColumnSpec, validate_schema_specs

if TYPE_CHECKING:
    from .strategy_time_series import WeightTimeSeries


def _norm_window_dt(value: dt.datetime | str | object) -> dt.datetime:
    """Normalize window date to naive midnight UTC."""
    from .framework_contract import _to_naive_utc

    return _to_naive_utc(value)


@dataclass(frozen=True, slots=True)
class WeightTimeSeriesBatch:
    """Collection of single-window strategy weight time-series outputs."""

    strategy_id: str
    strategy_version: str
    run_id: str
    config_hash: str
    windows: tuple["WeightTimeSeries", ...]
    schema_version: str = "1.0.0"
    generated_at: dt.datetime = field(default_factory=_utc_now)
    extra_schema: tuple[ColumnSpec, ...] = ()
    _window_index: dict[tuple[dt.datetime, dt.datetime], "WeightTimeSeries"] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if len(self.windows) == 0:
            raise ValueError("WeightTimeSeriesBatch.windows must not be empty.")
        normalized_generated_at = _normalize_generated_at(self.generated_at)
        if self.windows:
            normalized_generated_at = self.windows[0].metadata.generated_at
        batch_extra_schema = self.extra_schema
        if not batch_extra_schema and self.windows:
            batch_extra_schema = self.windows[0].extra_schema
        normalized_extra_schema = validate_schema_specs(
            batch_extra_schema,
            forbid_core_name_collisions=True,
        )
        object.__setattr__(self, "generated_at", normalized_generated_at)
        object.__setattr__(self, "extra_schema", normalized_extra_schema)
        window_index = self._build_window_index(self.windows)
        object.__setattr__(self, "_window_index", window_index)
        self.validate()

    @staticmethod
    def _build_window_index(
        windows: tuple["WeightTimeSeries", ...],
    ) -> dict[tuple[dt.datetime, dt.datetime], "WeightTimeSeries"]:
        index: dict[tuple[dt.datetime, dt.datetime], "WeightTimeSeries"] = {}
        for window in windows:
            md = window.metadata
            if md.window_start is None or md.window_end is None:
                continue
            key = (md.window_start, md.window_end)
            index[key] = window
        return index

    @classmethod
    def from_flat_dataframe(
        cls,
        data: pl.DataFrame,
        *,
        strategy_id: str,
        strategy_version: str,
        run_id: str,
        config_hash: str,
        schema_version: str = "1.0.0",
        generated_at: dt.datetime | None = None,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> "WeightTimeSeriesBatch":
        """Build a batch object from a flattened export dataframe."""
        from .strategy_time_series import WeightTimeSeries

        if not isinstance(data, pl.DataFrame):
            raise TypeError("data must be a Polars DataFrame.")

        required = {"start_date", "end_date", "date", "weight", "price_usd"}
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(
                "Flat dataframe missing required columns: " + ", ".join(sorted(missing))
            )

        normalized_extra_schema = validate_schema_specs(
            extra_schema,
            forbid_core_name_collisions=True,
        )
        batch_generated_at = _normalize_generated_at(generated_at)

        def _to_dt(col: str):
            c = pl.col(col)
            if data[col].dtype == pl.Utf8:
                c = c.str.to_datetime().dt.replace_time_zone(None).dt.truncate("1d")
            else:
                c = c.cast(pl.Datetime).dt.replace_time_zone(None).dt.truncate("1d")
            return c

        normalized = data.with_columns(
            _to_dt("start_date").alias("start_date"),
            _to_dt("end_date").alias("end_date"),
            _to_dt("date").alias("date"),
        ).sort(["start_date", "end_date", "date"])
        if (
            normalized["start_date"].null_count() > 0
            or normalized["end_date"].null_count() > 0
            or normalized["date"].null_count() > 0
        ):
            raise ValueError("start_date, end_date, and date must be valid datetimes.")

        windows: list[WeightTimeSeries] = []
        for sub_df in normalized.partition_by(["start_date", "end_date"]):
            window_start = sub_df["start_date"][0]
            window_end = sub_df["end_date"][0]
            payload = sub_df.drop(["start_date", "end_date"])
            if "day_index" not in payload.columns:
                payload = payload.with_columns(pl.arange(0, payload.height).alias("day_index"))
            ws = _norm_window_dt(window_start)
            we = _norm_window_dt(window_end)
            metadata = StrategySeriesMetadata(
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                run_id=run_id,
                config_hash=config_hash,
                schema_version=schema_version,
                generated_at=batch_generated_at,
                window_start=ws,
                window_end=we,
            )
            windows.append(
                WeightTimeSeries(
                    metadata=metadata,
                    data=payload,
                    extra_schema=normalized_extra_schema,
                )
            )

        return cls(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            run_id=run_id,
            config_hash=config_hash,
            windows=tuple(windows),
            schema_version=schema_version,
            generated_at=batch_generated_at,
            extra_schema=normalized_extra_schema,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        strategy_id: str,
        strategy_version: str,
        run_id: str,
        config_hash: str,
        schema_version: str = "1.0.0",
        generated_at: dt.datetime | None = None,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> "WeightTimeSeriesBatch":
        """Load a batch object from a flattened CSV export."""
        return cls.from_flat_dataframe(
            pl.read_csv(Path(path)),
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            run_id=run_id,
            config_hash=config_hash,
            schema_version=schema_version,
            generated_at=generated_at,
            extra_schema=extra_schema,
        )

    @classmethod
    def from_artifact_dir(
        cls,
        path: str | Path,
        *,
        extra_schema: tuple[ColumnSpec, ...] = (),
    ) -> "WeightTimeSeriesBatch":
        """Load a batch object from a strategy export artifact directory."""
        artifact_dir = Path(path)
        artifact_json_path = artifact_dir / "artifacts.json"
        if not artifact_json_path.exists():
            raise FileNotFoundError(f"Artifact metadata file not found: {artifact_json_path}")
        try:
            artifact_payload = json.loads(artifact_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid artifacts.json: {artifact_json_path}") from exc

        required_fields = ("strategy_id", "version", "run_id", "config_hash", "files")
        missing_fields = [field for field in required_fields if field not in artifact_payload]
        if missing_fields:
            raise ValueError(
                "artifacts.json missing required provenance fields: "
                + ", ".join(missing_fields)
            )
        files = artifact_payload["files"]
        if not isinstance(files, dict):
            raise ValueError("artifacts.json field 'files' must be an object.")
        weights_entry = files.get("weights_csv")
        if not isinstance(weights_entry, str) or not weights_entry:
            raise ValueError("artifacts.json must include a non-empty 'weights_csv' file entry.")

        weights_path = Path(weights_entry)
        if not weights_path.is_absolute():
            artifact_relative = artifact_dir / weights_path
            if artifact_relative.exists():
                weights_path = artifact_relative
            elif weights_path.exists():
                weights_path = weights_path
            else:
                artifact_local = artifact_dir / weights_path.name
                if artifact_local.exists():
                    weights_path = artifact_local
                else:
                    weights_path = artifact_relative
        if not weights_path.exists():
            raise FileNotFoundError(f"weights.csv not found: {weights_path}")

        return cls.from_csv(
            weights_path,
            strategy_id=str(artifact_payload["strategy_id"]),
            strategy_version=str(artifact_payload["version"]),
            run_id=str(artifact_payload["run_id"]),
            config_hash=str(artifact_payload["config_hash"]),
            schema_version=str(artifact_payload.get("schema_version", "1.0.0")),
            generated_at=artifact_payload.get("generated_at"),
            extra_schema=extra_schema,
        )

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def row_count(self) -> int:
        return int(sum(window.row_count for window in self.windows))

    def validate(self) -> None:
        """Validate cross-window metadata and uniqueness invariants."""
        seen_keys: set[tuple[dt.datetime, dt.datetime]] = set()
        for window in self.windows:
            window.validate()
            md = window.metadata
            if md.strategy_id != self.strategy_id:
                raise ValueError("Window metadata strategy_id does not match batch strategy_id.")
            if md.strategy_version != self.strategy_version:
                raise ValueError(
                    "Window metadata strategy_version does not match batch strategy_version."
                )
            if md.run_id != self.run_id:
                raise ValueError("Window metadata run_id does not match batch run_id.")
            if md.config_hash != self.config_hash:
                raise ValueError("Window metadata config_hash does not match batch config_hash.")
            if md.schema_version != self.schema_version:
                raise ValueError(
                    "Window metadata schema_version does not match batch schema_version."
                )
            if md.window_start is None or md.window_end is None:
                raise ValueError("Each window must define metadata.window_start and metadata.window_end.")
            if md.generated_at != self.generated_at:
                raise ValueError("Window metadata generated_at does not match batch generated_at.")
            if window.extra_schema != self.extra_schema:
                raise ValueError("All windows must share the batch extra_schema definition.")
            key = (md.window_start, md.window_end)
            if key in seen_keys:
                raise ValueError(
                    "Duplicate window key detected in batch: "
                    f"{key[0].strftime('%Y-%m-%d')} -> {key[1].strftime('%Y-%m-%d')}"
                )
            seen_keys.add(key)

    def to_dataframe(self) -> pl.DataFrame:
        """Flatten the batch into one canonical dataframe."""
        frames: list[pl.DataFrame] = []
        for window in self.windows:
            md = window.metadata
            payload = window.to_dataframe()
            payload = payload.with_columns(
                pl.lit(md.window_start).alias("start_date"),
                pl.lit(md.window_end).alias("end_date"),
            )
            payload = payload.select(
                ["start_date", "end_date"] + [c for c in payload.columns if c not in ("start_date", "end_date")]
            )
            frames.append(payload)
        return pl.concat(frames, how="vertical_relaxed")

    def to_csv(self, path: str | Path, *, index: bool = False) -> None:
        """Write the canonical flattened batch dataframe to CSV."""
        self.to_dataframe().write_csv(Path(path), include_header=True)

    def schema_markdown(self) -> str:
        """Render the shared window schema as markdown."""
        return self.windows[0].schema_markdown()

    def iter_windows(self) -> Iterable["WeightTimeSeries"]:
        """Yield windows in batch order."""
        return iter(self.windows)

    def window_keys(self) -> tuple[tuple[dt.datetime, dt.datetime], ...]:
        """Return all batch window keys in batch order."""
        return tuple(
            (window.metadata.window_start, window.metadata.window_end)
            for window in self.windows
            if window.metadata.window_start is not None and window.metadata.window_end is not None
        )

    def date_span(self) -> tuple[dt.datetime, dt.datetime]:
        """Return the full date span covered by the batch."""
        starts = [window.metadata.window_start for window in self.windows]
        ends = [window.metadata.window_end for window in self.windows]
        return (min(starts), max(ends))

    def for_window(
        self,
        start_date: str | dt.datetime,
        end_date: str | dt.datetime,
    ) -> "WeightTimeSeries":
        """Return the window object for a specific date range."""
        start = _norm_window_dt(start_date)
        end = _norm_window_dt(end_date)
        try:
            return self._window_index[(start, end)]
        except KeyError as exc:
            raise KeyError(
                f"Window not found: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}"
            ) from exc


# Deprecated aliases — remove in 0.9.0
TimeSeriesBatch = WeightTimeSeriesBatch
StrategyTimeSeriesBatch = WeightTimeSeriesBatch

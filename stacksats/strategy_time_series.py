"""Typed time-series output objects for strategy export runs."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Iterable

import numpy as np
import pandas as pd


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Handwritten schema specification for a single column."""

    name: str
    dtype: str
    required: bool
    description: str
    unit: str | None = None
    constraints: tuple[str, ...] = ()
    source: str = "framework"
    formula: str | None = None


@dataclass(frozen=True, slots=True)
class CoinMetricsLineageSpec:
    """CoinMetrics source column lineage into StrategyTimeSeries columns."""

    source_column: str
    required: bool
    description: str
    strategy_column: str | None
    notes: str = ""


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


@dataclass(frozen=True, slots=True)
class StrategyTimeSeries:
    """Single-window normalized strategy output time series."""

    metadata: StrategySeriesMetadata
    data: pd.DataFrame

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("date", "weight", "price_usd")
    COINMETRICS_BTC_CSV_COLUMNS: ClassVar[tuple[str, ...]] = (
        "time",
        "AdrActCnt",
        "AdrBalCnt",
        "AssetCompletionTime",
        "AssetEODCompletionTime",
        "BlkCnt",
        "CapMVRVCur",
        "CapMrktCurUSD",
        "CapMrktEstUSD",
        "FeeTotNtv",
        "FlowInExNtv",
        "FlowInExUSD",
        "FlowOutExNtv",
        "FlowOutExUSD",
        "HashRate",
        "IssTotNtv",
        "IssTotUSD",
        "PriceBTC",
        "PriceUSD",
        "ROI1yr",
        "ROI30d",
        "ReferenceRate",
        "ReferenceRateETH",
        "ReferenceRateEUR",
        "ReferenceRateUSD",
        "SplyCur",
        "SplyExNtv",
        "SplyExUSD",
        "SplyExpFut10yr",
        "TxCnt",
        "TxTfrCnt",
        "volume_reported_spot_usd_1d",
    )
    COINMETRICS_LINEAGE: ClassVar[tuple[CoinMetricsLineageSpec, ...]] = (
        CoinMetricsLineageSpec(
            source_column="time",
            required=True,
            description="CoinMetrics daily timestamp column.",
            strategy_column="date",
            notes="Loaded as index, then represented by StrategyTimeSeries.date.",
        ),
        CoinMetricsLineageSpec(
            source_column="AdrActCnt",
            required=False,
            description="CoinMetrics active addresses count.",
            strategy_column="AdrActCnt",
        ),
        CoinMetricsLineageSpec(
            source_column="AdrBalCnt",
            required=False,
            description="CoinMetrics addresses with non-zero balance.",
            strategy_column="AdrBalCnt",
        ),
        CoinMetricsLineageSpec(
            source_column="AssetCompletionTime",
            required=False,
            description="CoinMetrics ingestion completion timestamp for asset-day data.",
            strategy_column="AssetCompletionTime",
        ),
        CoinMetricsLineageSpec(
            source_column="AssetEODCompletionTime",
            required=False,
            description="CoinMetrics end-of-day completion timestamp for asset metrics.",
            strategy_column="AssetEODCompletionTime",
        ),
        CoinMetricsLineageSpec(
            source_column="BlkCnt",
            required=False,
            description="CoinMetrics blocks mined during the day.",
            strategy_column="BlkCnt",
        ),
        CoinMetricsLineageSpec(
            source_column="PriceUSD",
            required=True,
            description="CoinMetrics BTC close price in USD.",
            strategy_column="price_usd",
            notes="Aliased to PriceUSD_coinmetrics before export normalization.",
        ),
        CoinMetricsLineageSpec(
            source_column="CapMVRVCur",
            required=False,
            description="CoinMetrics current market-value-to-realized-value ratio.",
            strategy_column="CapMVRVCur",
        ),
        CoinMetricsLineageSpec(
            source_column="CapMrktCurUSD",
            required=False,
            description="CoinMetrics current market capitalization in USD.",
            strategy_column="CapMrktCurUSD",
        ),
        CoinMetricsLineageSpec(
            source_column="CapMrktEstUSD",
            required=False,
            description="CoinMetrics estimated market capitalization in USD.",
            strategy_column="CapMrktEstUSD",
        ),
        CoinMetricsLineageSpec(
            source_column="FeeTotNtv",
            required=False,
            description="CoinMetrics total transaction fees in native BTC units.",
            strategy_column="FeeTotNtv",
        ),
        CoinMetricsLineageSpec(
            source_column="FlowInExNtv",
            required=False,
            description="CoinMetrics exchange inflow in native BTC units.",
            strategy_column="FlowInExNtv",
        ),
        CoinMetricsLineageSpec(
            source_column="FlowInExUSD",
            required=False,
            description="CoinMetrics exchange inflow valued in USD.",
            strategy_column="FlowInExUSD",
        ),
        CoinMetricsLineageSpec(
            source_column="FlowOutExNtv",
            required=False,
            description="CoinMetrics exchange outflow in native BTC units.",
            strategy_column="FlowOutExNtv",
        ),
        CoinMetricsLineageSpec(
            source_column="FlowOutExUSD",
            required=False,
            description="CoinMetrics exchange outflow valued in USD.",
            strategy_column="FlowOutExUSD",
        ),
        CoinMetricsLineageSpec(
            source_column="HashRate",
            required=False,
            description="CoinMetrics network hash rate estimate.",
            strategy_column="HashRate",
        ),
        CoinMetricsLineageSpec(
            source_column="IssTotNtv",
            required=False,
            description="CoinMetrics total daily issuance in native BTC units.",
            strategy_column="IssTotNtv",
        ),
        CoinMetricsLineageSpec(
            source_column="IssTotUSD",
            required=False,
            description="CoinMetrics total daily issuance valued in USD.",
            strategy_column="IssTotUSD",
        ),
        CoinMetricsLineageSpec(
            source_column="PriceBTC",
            required=False,
            description="CoinMetrics BTC reference price quoted in BTC.",
            strategy_column="PriceBTC",
        ),
        CoinMetricsLineageSpec(
            source_column="PriceUSD_coinmetrics",
            required=True,
            description="Runtime alias of CoinMetrics PriceUSD.",
            strategy_column="price_usd",
            notes="Canonical runtime price input consumed by model and export.",
        ),
        CoinMetricsLineageSpec(
            source_column="ROI1yr",
            required=False,
            description="CoinMetrics trailing 1-year return metric.",
            strategy_column="ROI1yr",
        ),
        CoinMetricsLineageSpec(
            source_column="ROI30d",
            required=False,
            description="CoinMetrics trailing 30-day return metric.",
            strategy_column="ROI30d",
        ),
        CoinMetricsLineageSpec(
            source_column="ReferenceRate",
            required=False,
            description="CoinMetrics reference rate for BTC.",
            strategy_column="ReferenceRate",
        ),
        CoinMetricsLineageSpec(
            source_column="ReferenceRateETH",
            required=False,
            description="CoinMetrics reference rate for BTC quoted in ETH.",
            strategy_column="ReferenceRateETH",
        ),
        CoinMetricsLineageSpec(
            source_column="ReferenceRateEUR",
            required=False,
            description="CoinMetrics reference rate for BTC quoted in EUR.",
            strategy_column="ReferenceRateEUR",
        ),
        CoinMetricsLineageSpec(
            source_column="ReferenceRateUSD",
            required=False,
            description="CoinMetrics reference rate for BTC quoted in USD.",
            strategy_column="ReferenceRateUSD",
        ),
        CoinMetricsLineageSpec(
            source_column="SplyCur",
            required=False,
            description="CoinMetrics current circulating BTC supply.",
            strategy_column="SplyCur",
        ),
        CoinMetricsLineageSpec(
            source_column="SplyExNtv",
            required=False,
            description="CoinMetrics supply held on exchanges in native BTC units.",
            strategy_column="SplyExNtv",
        ),
        CoinMetricsLineageSpec(
            source_column="SplyExUSD",
            required=False,
            description="CoinMetrics supply held on exchanges valued in USD.",
            strategy_column="SplyExUSD",
        ),
        CoinMetricsLineageSpec(
            source_column="SplyExpFut10yr",
            required=False,
            description="CoinMetrics projected BTC supply 10 years ahead.",
            strategy_column="SplyExpFut10yr",
        ),
        CoinMetricsLineageSpec(
            source_column="TxCnt",
            required=False,
            description="CoinMetrics on-chain transaction count.",
            strategy_column="TxCnt",
        ),
        CoinMetricsLineageSpec(
            source_column="TxTfrCnt",
            required=False,
            description="CoinMetrics transfer transaction count.",
            strategy_column="TxTfrCnt",
        ),
        CoinMetricsLineageSpec(
            source_column="volume_reported_spot_usd_1d",
            required=False,
            description="CoinMetrics reported spot exchange volume in USD for 1 day.",
            strategy_column="volume_reported_spot_usd_1d",
        ),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("StrategyTimeSeries.data must be a pandas DataFrame.")

        normalized = self.data.copy(deep=True)
        if "date" in normalized.columns:
            normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
            normalized = normalized.sort_values("date").reset_index(drop=True)
        if "weight" in normalized.columns:
            normalized["weight"] = pd.to_numeric(normalized["weight"], errors="coerce")
        if "price_usd" in normalized.columns:
            normalized["price_usd"] = pd.to_numeric(normalized["price_usd"], errors="coerce")
        if "day_index" in normalized.columns:
            normalized["day_index"] = pd.to_numeric(normalized["day_index"], errors="coerce")

        object.__setattr__(self, "data", normalized)
        self.validate()

    @staticmethod
    def _schema_specs() -> tuple[ColumnSpec, ...]:
        return (
            ColumnSpec(
                name="day_index",
                dtype="int64",
                required=False,
                description="Zero-based day index within the allocation window.",
                constraints=(">=0", "strictly increasing by 1"),
            ),
            ColumnSpec(
                name="date",
                dtype="datetime64[ns]",
                required=True,
                description="Calendar day for this allocation row.",
                constraints=("unique", "sorted ascending", "daily grain"),
            ),
            ColumnSpec(
                name="weight",
                dtype="float64",
                required=True,
                description=(
                    "Final feasible daily allocation after clipping, lock preservation, "
                    "and remaining-budget constraints."
                ),
                constraints=("finite", ">=0", "sum ~= 1.0"),
            ),
            ColumnSpec(
                name="price_usd",
                dtype="float64",
                required=True,
                description="BTC price in USD for the given date when available.",
                unit="USD",
                constraints=("finite when present", "nullable for future dates"),
            ),
            ColumnSpec(
                name="locked",
                dtype="bool",
                required=False,
                description="True when a row belongs to an immutable locked history prefix.",
                constraints=("boolean values only",),
            ),
            ColumnSpec(
                name="PriceUSD",
                dtype="float64",
                required=False,
                description="Raw CoinMetrics BTC price column when preserved in payloads.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
                formula="raw PriceUSD",
            ),
            ColumnSpec(
                name="PriceUSD_coinmetrics",
                dtype="float64",
                required=False,
                description="Runtime alias for CoinMetrics BTC price when retained.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
                formula="PriceUSD -> PriceUSD_coinmetrics",
            ),
            ColumnSpec(
                name="CapMVRVCur",
                dtype="float64",
                required=False,
                description="CoinMetrics MVRV ratio when retained in strategy payloads.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="time",
                dtype="datetime64[ns]",
                required=False,
                description="CoinMetrics daily timestamp column.",
                constraints=("valid datetime when present",),
                source="coinmetrics",
                formula="raw time",
            ),
            ColumnSpec(
                name="AdrActCnt",
                dtype="float64",
                required=False,
                description="CoinMetrics active addresses count.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="AdrBalCnt",
                dtype="float64",
                required=False,
                description="CoinMetrics addresses with non-zero balance.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="AssetCompletionTime",
                dtype="datetime64[ns]",
                required=False,
                description="CoinMetrics ingestion completion timestamp for asset-day data.",
                constraints=("valid datetime when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="AssetEODCompletionTime",
                dtype="datetime64[ns]",
                required=False,
                description="CoinMetrics end-of-day completion timestamp for asset metrics.",
                constraints=("valid datetime when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="BlkCnt",
                dtype="float64",
                required=False,
                description="CoinMetrics blocks mined during the day.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="CapMrktCurUSD",
                dtype="float64",
                required=False,
                description="CoinMetrics current market capitalization in USD.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="CapMrktEstUSD",
                dtype="float64",
                required=False,
                description="CoinMetrics estimated market capitalization in USD.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="FeeTotNtv",
                dtype="float64",
                required=False,
                description="CoinMetrics total transaction fees in native BTC units.",
                unit="BTC",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="FlowInExNtv",
                dtype="float64",
                required=False,
                description="CoinMetrics exchange inflow in native BTC units.",
                unit="BTC",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="FlowInExUSD",
                dtype="float64",
                required=False,
                description="CoinMetrics exchange inflow valued in USD.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="FlowOutExNtv",
                dtype="float64",
                required=False,
                description="CoinMetrics exchange outflow in native BTC units.",
                unit="BTC",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="FlowOutExUSD",
                dtype="float64",
                required=False,
                description="CoinMetrics exchange outflow valued in USD.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="HashRate",
                dtype="float64",
                required=False,
                description="CoinMetrics network hash rate estimate.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="IssTotNtv",
                dtype="float64",
                required=False,
                description="CoinMetrics total daily issuance in native BTC units.",
                unit="BTC",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="IssTotUSD",
                dtype="float64",
                required=False,
                description="CoinMetrics total daily issuance valued in USD.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="PriceBTC",
                dtype="float64",
                required=False,
                description="CoinMetrics BTC reference price quoted in BTC.",
                unit="BTC",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="ROI1yr",
                dtype="float64",
                required=False,
                description="CoinMetrics trailing 1-year return metric.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="ROI30d",
                dtype="float64",
                required=False,
                description="CoinMetrics trailing 30-day return metric.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="ReferenceRate",
                dtype="float64",
                required=False,
                description="CoinMetrics reference rate for BTC.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="ReferenceRateETH",
                dtype="float64",
                required=False,
                description="CoinMetrics reference rate for BTC quoted in ETH.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="ReferenceRateEUR",
                dtype="float64",
                required=False,
                description="CoinMetrics reference rate for BTC quoted in EUR.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="ReferenceRateUSD",
                dtype="float64",
                required=False,
                description="CoinMetrics reference rate for BTC quoted in USD.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="SplyCur",
                dtype="float64",
                required=False,
                description="CoinMetrics current circulating BTC supply.",
                unit="BTC",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="SplyExNtv",
                dtype="float64",
                required=False,
                description="CoinMetrics supply held on exchanges in native BTC units.",
                unit="BTC",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="SplyExUSD",
                dtype="float64",
                required=False,
                description="CoinMetrics supply held on exchanges valued in USD.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="SplyExpFut10yr",
                dtype="float64",
                required=False,
                description="CoinMetrics projected BTC supply 10 years ahead.",
                unit="BTC",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="TxCnt",
                dtype="float64",
                required=False,
                description="CoinMetrics on-chain transaction count.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="TxTfrCnt",
                dtype="float64",
                required=False,
                description="CoinMetrics transfer transaction count.",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
            ColumnSpec(
                name="volume_reported_spot_usd_1d",
                dtype="float64",
                required=False,
                description="CoinMetrics reported spot exchange volume in USD for 1 day.",
                unit="USD",
                constraints=("finite when present",),
                source="coinmetrics",
            ),
        )

    def schema(self) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return self.schema_dict()

    @classmethod
    def schema_dict(cls) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return {spec.name: spec for spec in cls._schema_specs()}

    @classmethod
    def validate_coinmetrics_lineage_coverage(cls) -> None:
        """Ensure lineage mappings target documented StrategyTimeSeries columns."""
        schema_names = {spec.name for spec in cls._schema_specs()}
        missing_targets = [
            lineage.source_column
            for lineage in cls.COINMETRICS_LINEAGE
            if lineage.strategy_column is not None and lineage.strategy_column not in schema_names
        ]
        if missing_targets:
            raise ValueError(
                "CoinMetrics lineage mappings reference undocumented StrategyTimeSeries "
                "columns for source columns: " + ", ".join(missing_targets)
            )

        lineage_sources = {lineage.source_column for lineage in cls.COINMETRICS_LINEAGE}
        missing_sources = [
            column for column in cls.COINMETRICS_BTC_CSV_COLUMNS if column not in lineage_sources
        ]
        if missing_sources:
            raise ValueError(
                "CoinMetrics lineage missing BTC CSV source columns: "
                + ", ".join(missing_sources)
            )

    @classmethod
    def coinmetrics_lineage_markdown(cls) -> str:
        """Render CoinMetrics source-to-schema lineage as a markdown table."""
        cls.validate_coinmetrics_lineage_coverage()
        header = (
            "| source_column | required | description | strategy_column | notes |\n"
            "| --- | --- | --- | --- | --- |"
        )
        rows = [
            "| {source} | {required} | {description} | {target} | {notes} |".format(
                source=spec.source_column,
                required=str(spec.required),
                description=spec.description.replace("|", "\\|"),
                target=(spec.strategy_column or ""),
                notes=spec.notes.replace("|", "\\|"),
            )
            for spec in cls.COINMETRICS_LINEAGE
        ]
        return "\n".join([header, *rows])

    @classmethod
    def schema_markdown_table(cls) -> str:
        """Render StrategyTimeSeries schema specs as a markdown table."""
        cls.validate_coinmetrics_lineage_coverage()
        specs = cls._schema_specs()
        return cls._render_schema_markdown(specs)

    @staticmethod
    def _render_schema_markdown(specs: Iterable[ColumnSpec]) -> str:
        """Render schema specs as a markdown table."""
        header = (
            "| name | dtype | required | description | unit | constraints | source | formula |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        rows = [
            "| {name} | {dtype} | {required} | {description} | {unit} | {constraints} | {source} | {formula} |".format(
                name=spec.name,
                dtype=spec.dtype,
                required=str(spec.required),
                description=spec.description.replace("|", "\\|"),
                unit=(spec.unit or ""),
                constraints=", ".join(spec.constraints),
                source=spec.source,
                formula=(spec.formula or ""),
            )
            for spec in specs
        ]
        return "\n".join([header, *rows])

    def schema_markdown(self) -> str:
        """Render schema specs as a markdown table."""
        return self._render_schema_markdown(self.schema().values())

    def validate_schema_coverage(self) -> None:
        """Ensure each column has an explicit handwritten schema entry."""
        covered = set(self.schema().keys())
        unknown = [col for col in self.data.columns if col not in covered]
        if unknown:
            raise ValueError(
                "Schema coverage missing for columns: " + ", ".join(str(col) for col in unknown)
            )

    def validate(self) -> None:
        """Validate data and metadata invariants."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing:
            raise ValueError(
                "StrategyTimeSeries missing required columns: "
                + ", ".join(str(col) for col in missing)
            )

        self.validate_schema_coverage()

        dates = pd.to_datetime(self.data["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Column 'date' must contain valid datetimes.")
        if dates.duplicated().any():
            raise ValueError("Column 'date' must not contain duplicates.")
        if not dates.is_monotonic_increasing:
            raise ValueError("Column 'date' must be sorted ascending.")

        if self.metadata.window_start is not None and len(dates) > 0:
            start = pd.Timestamp(self.metadata.window_start)
            if pd.Timestamp(dates.iloc[0]) != start:
                raise ValueError(
                    "Series start date does not match metadata.window_start: "
                    f"{dates.iloc[0]!s} != {start!s}"
                )
        if self.metadata.window_end is not None and len(dates) > 0:
            end = pd.Timestamp(self.metadata.window_end)
            if pd.Timestamp(dates.iloc[-1]) != end:
                raise ValueError(
                    "Series end date does not match metadata.window_end: "
                    f"{dates.iloc[-1]!s} != {end!s}"
                )

        weights = pd.to_numeric(self.data["weight"], errors="coerce")
        if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
            raise ValueError("Column 'weight' must contain finite numeric values.")
        if bool((weights < 0).any()):
            raise ValueError("Column 'weight' must not contain negative values.")
        if len(weights) > 0:
            weight_sum = float(weights.sum())
            if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=1e-8):
                raise ValueError(
                    "Column 'weight' must sum to 1.0 "
                    f"(got {weight_sum:.10f})."
                )

        raw_price = self.data["price_usd"]
        prices = pd.to_numeric(raw_price, errors="coerce")
        invalid_non_null = raw_price.notna() & prices.isna()
        if invalid_non_null.any():
            raise ValueError("Column 'price_usd' must be numeric when present.")
        finite_mask = prices.notna()
        if finite_mask.any() and not np.isfinite(prices.loc[finite_mask].to_numpy(dtype=float)).all():
            raise ValueError("Column 'price_usd' must be finite when present.")

        if "locked" in self.data.columns:
            locked = self.data["locked"]
            valid_locked = locked.isin([True, False])
            if not bool(valid_locked.all()):
                raise ValueError("Column 'locked' must contain only boolean values.")

        if "day_index" in self.data.columns:
            day_index = pd.to_numeric(self.data["day_index"], errors="coerce")
            if day_index.isna().any():
                raise ValueError("Column 'day_index' must contain integer values.")
            if bool((day_index < 0).any()):
                raise ValueError("Column 'day_index' must be >= 0.")
            if len(day_index) > 0:
                expected = np.arange(len(day_index), dtype=float)
                if not np.array_equal(day_index.to_numpy(dtype=float), expected):
                    raise ValueError("Column 'day_index' must be contiguous starting at 0.")

        for spec in self._schema_specs():
            if spec.source != "coinmetrics":
                continue
            if spec.name not in self.data.columns:
                continue
            if spec.dtype in {"int64", "float64"}:
                self._validate_optional_numeric_column(spec.name)
            elif spec.dtype.startswith("datetime64"):
                self._validate_optional_datetime_column(spec.name)

    def _validate_optional_numeric_column(self, column: str) -> None:
        """Validate optional CoinMetrics passthrough numeric columns."""
        raw = self.data[column]
        values = pd.to_numeric(raw, errors="coerce")
        invalid_non_null = raw.notna() & values.isna()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be numeric when present.")
        finite_mask = values.notna()
        if finite_mask.any() and not np.isfinite(values.loc[finite_mask].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{column}' must be finite when present.")

    def _validate_optional_datetime_column(self, column: str) -> None:
        """Validate optional CoinMetrics passthrough datetime columns."""
        raw = self.data[column]
        values = pd.to_datetime(raw, errors="coerce")
        invalid_non_null = raw.notna() & values.isna()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be datetime when present.")

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the normalized dataframe payload."""
        return self.data.copy(deep=True)


@dataclass(frozen=True, slots=True)
class StrategyTimeSeriesBatch:
    """Collection of single-window strategy time-series outputs."""

    strategy_id: str
    strategy_version: str
    run_id: str
    config_hash: str
    windows: tuple[StrategyTimeSeries, ...]
    schema_version: str = "1.0.0"
    generated_at: pd.Timestamp = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if len(self.windows) == 0:
            raise ValueError("StrategyTimeSeriesBatch.windows must not be empty.")
        self.validate()

    @classmethod
    def from_flat_dataframe(
        cls,
        data: pd.DataFrame,
        *,
        strategy_id: str,
        strategy_version: str,
        run_id: str,
        config_hash: str,
        schema_version: str = "1.0.0",
    ) -> "StrategyTimeSeriesBatch":
        """Build a batch object from a flattened export dataframe."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        required = {"start_date", "end_date", "date", "weight", "price_usd"}
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(
                "Flat dataframe missing required columns: " + ", ".join(sorted(missing))
            )

        normalized = data.copy(deep=True)
        normalized["start_date"] = pd.to_datetime(normalized["start_date"], errors="coerce")
        normalized["end_date"] = pd.to_datetime(normalized["end_date"], errors="coerce")
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        if (
            normalized["start_date"].isna().any()
            or normalized["end_date"].isna().any()
            or normalized["date"].isna().any()
        ):
            raise ValueError("start_date, end_date, and date must be valid datetimes.")

        normalized = normalized.sort_values(["start_date", "end_date", "date"]).reset_index(
            drop=True
        )

        windows: list[StrategyTimeSeries] = []
        grouped = normalized.groupby(["start_date", "end_date"], sort=True, dropna=False)
        for (window_start, window_end), window_frame in grouped:
            payload_columns = [
                col for col in window_frame.columns if col not in {"start_date", "end_date"}
            ]
            payload = window_frame[payload_columns].reset_index(drop=True)
            if "day_index" not in payload.columns:
                payload.insert(0, "day_index", np.arange(len(payload), dtype=int))
            metadata = StrategySeriesMetadata(
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                run_id=run_id,
                config_hash=config_hash,
                schema_version=schema_version,
                window_start=pd.Timestamp(window_start),
                window_end=pd.Timestamp(window_end),
            )
            windows.append(StrategyTimeSeries(metadata=metadata, data=payload))

        return cls(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            run_id=run_id,
            config_hash=config_hash,
            windows=tuple(windows),
            schema_version=schema_version,
        )

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def row_count(self) -> int:
        return int(sum(len(window.data) for window in self.windows))

    def validate(self) -> None:
        """Validate cross-window metadata and uniqueness invariants."""
        seen_keys: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
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
                raise ValueError("Window metadata schema_version does not match batch schema_version.")
            if md.window_start is None or md.window_end is None:
                raise ValueError("Each window must define metadata.window_start and metadata.window_end.")
            key = (pd.Timestamp(md.window_start), pd.Timestamp(md.window_end))
            if key in seen_keys:
                raise ValueError(
                    "Duplicate window key detected in batch: "
                    f"{key[0].strftime('%Y-%m-%d')} -> {key[1].strftime('%Y-%m-%d')}"
                )
            seen_keys.add(key)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the batch into one canonical dataframe."""
        frames: list[pd.DataFrame] = []
        for window in self.windows:
            md = window.metadata
            payload = window.to_dataframe()
            payload.insert(0, "end_date", pd.Timestamp(md.window_end))
            payload.insert(0, "start_date", pd.Timestamp(md.window_start))
            frames.append(payload)
        return pd.concat(frames, ignore_index=True)

    def schema_markdown(self) -> str:
        """Render the shared window schema as markdown."""
        return self.windows[0].schema_markdown()

    def iter_windows(self) -> Iterable[StrategyTimeSeries]:
        """Yield windows in batch order."""
        return iter(self.windows)

    def for_window(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> StrategyTimeSeries:
        """Return the window object for a specific date range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        for window in self.windows:
            md = window.metadata
            if pd.Timestamp(md.window_start) == start and pd.Timestamp(md.window_end) == end:
                return window
        raise KeyError(f"Window not found: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}")

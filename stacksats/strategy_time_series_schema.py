"""Schema and lineage definitions for StrategyTimeSeries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


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


COINMETRICS_BTC_CSV_COLUMNS: tuple[str, ...] = (
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

COINMETRICS_LINEAGE: tuple[CoinMetricsLineageSpec, ...] = (
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


def schema_specs() -> tuple[ColumnSpec, ...]:
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


def schema_dict(specs: Iterable[ColumnSpec] | None = None) -> dict[str, ColumnSpec]:
    target_specs = tuple(specs) if specs is not None else schema_specs()
    return {spec.name: spec for spec in target_specs}


def validate_coinmetrics_lineage_coverage(
    *,
    lineage: Iterable[CoinMetricsLineageSpec],
    schema_specs_iter: Iterable[ColumnSpec],
    source_columns: Iterable[str],
) -> None:
    schema_names = {spec.name for spec in schema_specs_iter}
    lineage_list = tuple(lineage)
    missing_targets = [
        item.source_column
        for item in lineage_list
        if item.strategy_column is not None and item.strategy_column not in schema_names
    ]
    if missing_targets:
        raise ValueError(
            "CoinMetrics lineage mappings reference undocumented StrategyTimeSeries "
            "columns for source columns: " + ", ".join(missing_targets)
        )

    lineage_sources = {item.source_column for item in lineage_list}
    missing_sources = [column for column in source_columns if column not in lineage_sources]
    if missing_sources:
        raise ValueError("CoinMetrics lineage missing BTC CSV source columns: " + ", ".join(missing_sources))


def coinmetrics_lineage_markdown(lineage: Iterable[CoinMetricsLineageSpec]) -> str:
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
        for spec in lineage
    ]
    return "\n".join([header, *rows])


def render_schema_markdown(specs: Iterable[ColumnSpec]) -> str:
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

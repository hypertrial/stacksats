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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ColumnSpec.name must be a non-empty string.")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError(f"ColumnSpec.dtype must be a non-empty string for {self.name!r}.")
        if not isinstance(self.required, bool):
            raise TypeError(f"ColumnSpec.required must be boolean for {self.name!r}.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                f"ColumnSpec.description must be a non-empty string for {self.name!r}."
            )
        if self.unit is not None and not isinstance(self.unit, str):
            raise TypeError(f"ColumnSpec.unit must be a string or None for {self.name!r}.")
        if not isinstance(self.constraints, tuple):
            raise TypeError(f"ColumnSpec.constraints must be a tuple for {self.name!r}.")
        if any(not isinstance(item, str) or not item.strip() for item in self.constraints):
            raise ValueError(
                f"ColumnSpec.constraints must contain only non-empty strings for {self.name!r}."
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(f"ColumnSpec.source must be a non-empty string for {self.name!r}.")
        if self.formula is not None and not isinstance(self.formula, str):
            raise TypeError(f"ColumnSpec.formula must be a string or None for {self.name!r}.")


@dataclass(frozen=True, slots=True)
class BRKLineageSpec:
    """BRK source column lineage into StrategyTimeSeries columns."""

    source_column: str
    required: bool
    description: str
    strategy_column: str | None
    notes: str = ""


BRK_SOURCE_COLUMNS: tuple[str, ...] = (
    "time",
    "AdrActCnt",
    "AdrBalCnt",
    "AssetCompletionTime",
    "AssetEODCompletionTime",
    "BlkCnt",
    "mvrv",
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

# Backward-compatible alias kept for callers/tests that still reference legacy name.
BRK_BTC_CSV_COLUMNS: tuple[str, ...] = BRK_SOURCE_COLUMNS

BRK_LINEAGE: tuple[BRKLineageSpec, ...] = (
    BRKLineageSpec(
        source_column="time",
        required=True,
        description="BRK daily timestamp column.",
        strategy_column="date",
        notes="Loaded as index, then represented by StrategyTimeSeries.date.",
    ),
    BRKLineageSpec(
        source_column="AdrActCnt",
        required=False,
        description="BRK active addresses count.",
        strategy_column="AdrActCnt",
    ),
    BRKLineageSpec(
        source_column="AdrBalCnt",
        required=False,
        description="BRK addresses with non-zero balance.",
        strategy_column="AdrBalCnt",
    ),
    BRKLineageSpec(
        source_column="AssetCompletionTime",
        required=False,
        description="BRK ingestion completion timestamp for asset-day data.",
        strategy_column="AssetCompletionTime",
    ),
    BRKLineageSpec(
        source_column="AssetEODCompletionTime",
        required=False,
        description="BRK end-of-day completion timestamp for asset metrics.",
        strategy_column="AssetEODCompletionTime",
    ),
    BRKLineageSpec(
        source_column="BlkCnt",
        required=False,
        description="BRK blocks mined during the day.",
        strategy_column="BlkCnt",
    ),
    BRKLineageSpec(
        source_column="PriceUSD",
        required=True,
        description="BRK BTC close price in USD.",
        strategy_column="price_usd",
        notes="Aliased to price_usd before export normalization.",
    ),
    BRKLineageSpec(
        source_column="mvrv",
        required=False,
        description="BRK current market-value-to-realized-value ratio.",
        strategy_column="mvrv",
    ),
    BRKLineageSpec(
        source_column="CapMrktCurUSD",
        required=False,
        description="BRK current market capitalization in USD.",
        strategy_column="CapMrktCurUSD",
    ),
    BRKLineageSpec(
        source_column="CapMrktEstUSD",
        required=False,
        description="BRK estimated market capitalization in USD.",
        strategy_column="CapMrktEstUSD",
    ),
    BRKLineageSpec(
        source_column="FeeTotNtv",
        required=False,
        description="BRK total transaction fees in native BTC units.",
        strategy_column="FeeTotNtv",
    ),
    BRKLineageSpec(
        source_column="FlowInExNtv",
        required=False,
        description="BRK exchange inflow in native BTC units.",
        strategy_column="FlowInExNtv",
    ),
    BRKLineageSpec(
        source_column="FlowInExUSD",
        required=False,
        description="BRK exchange inflow valued in USD.",
        strategy_column="FlowInExUSD",
    ),
    BRKLineageSpec(
        source_column="FlowOutExNtv",
        required=False,
        description="BRK exchange outflow in native BTC units.",
        strategy_column="FlowOutExNtv",
    ),
    BRKLineageSpec(
        source_column="FlowOutExUSD",
        required=False,
        description="BRK exchange outflow valued in USD.",
        strategy_column="FlowOutExUSD",
    ),
    BRKLineageSpec(
        source_column="HashRate",
        required=False,
        description="BRK network hash rate estimate.",
        strategy_column="HashRate",
    ),
    BRKLineageSpec(
        source_column="IssTotNtv",
        required=False,
        description="BRK total daily issuance in native BTC units.",
        strategy_column="IssTotNtv",
    ),
    BRKLineageSpec(
        source_column="IssTotUSD",
        required=False,
        description="BRK total daily issuance valued in USD.",
        strategy_column="IssTotUSD",
    ),
    BRKLineageSpec(
        source_column="PriceBTC",
        required=False,
        description="BRK BTC reference price quoted in BTC.",
        strategy_column="PriceBTC",
    ),
    BRKLineageSpec(
        source_column="price_usd",
        required=True,
        description="Runtime alias of BRK PriceUSD.",
        strategy_column="price_usd",
        notes="Canonical runtime price input consumed by model and export.",
    ),
    BRKLineageSpec(
        source_column="ROI1yr",
        required=False,
        description="BRK trailing 1-year return metric.",
        strategy_column="ROI1yr",
    ),
    BRKLineageSpec(
        source_column="ROI30d",
        required=False,
        description="BRK trailing 30-day return metric.",
        strategy_column="ROI30d",
    ),
    BRKLineageSpec(
        source_column="ReferenceRate",
        required=False,
        description="BRK reference rate for BTC.",
        strategy_column="ReferenceRate",
    ),
    BRKLineageSpec(
        source_column="ReferenceRateETH",
        required=False,
        description="BRK reference rate for BTC quoted in ETH.",
        strategy_column="ReferenceRateETH",
    ),
    BRKLineageSpec(
        source_column="ReferenceRateEUR",
        required=False,
        description="BRK reference rate for BTC quoted in EUR.",
        strategy_column="ReferenceRateEUR",
    ),
    BRKLineageSpec(
        source_column="ReferenceRateUSD",
        required=False,
        description="BRK reference rate for BTC quoted in USD.",
        strategy_column="ReferenceRateUSD",
    ),
    BRKLineageSpec(
        source_column="SplyCur",
        required=False,
        description="BRK current circulating BTC supply.",
        strategy_column="SplyCur",
    ),
    BRKLineageSpec(
        source_column="SplyExNtv",
        required=False,
        description="BRK supply held on exchanges in native BTC units.",
        strategy_column="SplyExNtv",
    ),
    BRKLineageSpec(
        source_column="SplyExUSD",
        required=False,
        description="BRK supply held on exchanges valued in USD.",
        strategy_column="SplyExUSD",
    ),
    BRKLineageSpec(
        source_column="SplyExpFut10yr",
        required=False,
        description="BRK projected BTC supply 10 years ahead.",
        strategy_column="SplyExpFut10yr",
    ),
    BRKLineageSpec(
        source_column="TxCnt",
        required=False,
        description="BRK on-chain transaction count.",
        strategy_column="TxCnt",
    ),
    BRKLineageSpec(
        source_column="TxTfrCnt",
        required=False,
        description="BRK transfer transaction count.",
        strategy_column="TxTfrCnt",
    ),
    BRKLineageSpec(
        source_column="volume_reported_spot_usd_1d",
        required=False,
        description="BRK reported spot exchange volume in USD for 1 day.",
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
            description="Raw BRK BTC price column when preserved in payloads.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
            formula="raw PriceUSD",
        ),
        ColumnSpec(
            name="mvrv",
            dtype="float64",
            required=False,
            description="BRK MVRV ratio when retained in strategy payloads.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="time",
            dtype="datetime64[ns]",
            required=False,
            description="BRK daily timestamp column.",
            constraints=("valid datetime when present",),
            source="brk",
            formula="raw time",
        ),
        ColumnSpec(
            name="AdrActCnt",
            dtype="float64",
            required=False,
            description="BRK active addresses count.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="AdrBalCnt",
            dtype="float64",
            required=False,
            description="BRK addresses with non-zero balance.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="AssetCompletionTime",
            dtype="datetime64[ns]",
            required=False,
            description="BRK ingestion completion timestamp for asset-day data.",
            constraints=("valid datetime when present",),
            source="brk",
        ),
        ColumnSpec(
            name="AssetEODCompletionTime",
            dtype="datetime64[ns]",
            required=False,
            description="BRK end-of-day completion timestamp for asset metrics.",
            constraints=("valid datetime when present",),
            source="brk",
        ),
        ColumnSpec(
            name="BlkCnt",
            dtype="float64",
            required=False,
            description="BRK blocks mined during the day.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="CapMrktCurUSD",
            dtype="float64",
            required=False,
            description="BRK current market capitalization in USD.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="CapMrktEstUSD",
            dtype="float64",
            required=False,
            description="BRK estimated market capitalization in USD.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="FeeTotNtv",
            dtype="float64",
            required=False,
            description="BRK total transaction fees in native BTC units.",
            unit="BTC",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="FlowInExNtv",
            dtype="float64",
            required=False,
            description="BRK exchange inflow in native BTC units.",
            unit="BTC",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="FlowInExUSD",
            dtype="float64",
            required=False,
            description="BRK exchange inflow valued in USD.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="FlowOutExNtv",
            dtype="float64",
            required=False,
            description="BRK exchange outflow in native BTC units.",
            unit="BTC",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="FlowOutExUSD",
            dtype="float64",
            required=False,
            description="BRK exchange outflow valued in USD.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="HashRate",
            dtype="float64",
            required=False,
            description="BRK network hash rate estimate.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="IssTotNtv",
            dtype="float64",
            required=False,
            description="BRK total daily issuance in native BTC units.",
            unit="BTC",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="IssTotUSD",
            dtype="float64",
            required=False,
            description="BRK total daily issuance valued in USD.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="PriceBTC",
            dtype="float64",
            required=False,
            description="BRK BTC reference price quoted in BTC.",
            unit="BTC",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="ROI1yr",
            dtype="float64",
            required=False,
            description="BRK trailing 1-year return metric.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="ROI30d",
            dtype="float64",
            required=False,
            description="BRK trailing 30-day return metric.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="ReferenceRate",
            dtype="float64",
            required=False,
            description="BRK reference rate for BTC.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="ReferenceRateETH",
            dtype="float64",
            required=False,
            description="BRK reference rate for BTC quoted in ETH.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="ReferenceRateEUR",
            dtype="float64",
            required=False,
            description="BRK reference rate for BTC quoted in EUR.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="ReferenceRateUSD",
            dtype="float64",
            required=False,
            description="BRK reference rate for BTC quoted in USD.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="SplyCur",
            dtype="float64",
            required=False,
            description="BRK current circulating BTC supply.",
            unit="BTC",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="SplyExNtv",
            dtype="float64",
            required=False,
            description="BRK supply held on exchanges in native BTC units.",
            unit="BTC",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="SplyExUSD",
            dtype="float64",
            required=False,
            description="BRK supply held on exchanges valued in USD.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="SplyExpFut10yr",
            dtype="float64",
            required=False,
            description="BRK projected BTC supply 10 years ahead.",
            unit="BTC",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="TxCnt",
            dtype="float64",
            required=False,
            description="BRK on-chain transaction count.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="TxTfrCnt",
            dtype="float64",
            required=False,
            description="BRK transfer transaction count.",
            constraints=("finite when present",),
            source="brk",
        ),
        ColumnSpec(
            name="volume_reported_spot_usd_1d",
            dtype="float64",
            required=False,
            description="BRK reported spot exchange volume in USD for 1 day.",
            unit="USD",
            constraints=("finite when present",),
            source="brk",
        ),
    )


def schema_dict(specs: Iterable[ColumnSpec] | None = None) -> dict[str, ColumnSpec]:
    target_specs = tuple(specs) if specs is not None else schema_specs()
    return {spec.name: spec for spec in target_specs}


def validate_schema_specs(
    extra_specs: Iterable[ColumnSpec],
    *,
    forbid_core_name_collisions: bool = True,
) -> tuple[ColumnSpec, ...]:
    specs = tuple(extra_specs)
    seen: set[str] = set()
    duplicates: list[str] = []
    for spec in specs:
        if spec.name in seen:
            duplicates.append(spec.name)
        seen.add(spec.name)
    if duplicates:
        names = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Schema specs contain duplicate column names: {names}")

    if forbid_core_name_collisions:
        core_names = {spec.name for spec in schema_specs()}
        collisions = sorted(name for name in seen if name in core_names)
        if collisions:
            raise ValueError(
                "Extra schema columns collide with core StrategyTimeSeries schema: "
                + ", ".join(collisions)
            )
    return specs


def merge_schema_specs(
    core_specs: Iterable[ColumnSpec],
    extra_specs: Iterable[ColumnSpec],
) -> tuple[ColumnSpec, ...]:
    core = tuple(core_specs)
    extra = validate_schema_specs(extra_specs, forbid_core_name_collisions=True)
    return core + extra


def validate_brk_lineage_coverage(
    *,
    lineage: Iterable[BRKLineageSpec],
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
            "BRK lineage mappings reference undocumented StrategyTimeSeries "
            "columns for source columns: " + ", ".join(missing_targets)
        )

    lineage_sources = {item.source_column for item in lineage_list}
    missing_sources = [column for column in source_columns if column not in lineage_sources]
    if missing_sources:
        raise ValueError("BRK lineage missing source columns: " + ", ".join(missing_sources))


def brk_lineage_markdown(lineage: Iterable[BRKLineageSpec]) -> str:
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

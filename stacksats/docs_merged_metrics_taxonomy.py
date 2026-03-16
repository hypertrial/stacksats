"""Helpers for generating merged-metrics taxonomy docs and JSON artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl

PHYSICAL_SCHEMA = (
    {
        "name": "day_utc",
        "dtype": "Date",
        "required": True,
        "nullable": False,
        "description": "UTC calendar day for the metric observation.",
    },
    {
        "name": "metric",
        "dtype": "String",
        "required": True,
        "nullable": False,
        "description": "Long-format metric key.",
    },
    {
        "name": "value",
        "dtype": "Float64",
        "required": True,
        "nullable": False,
        "description": "Metric value for the (day_utc, metric) pair.",
    },
)
EXPECTED_PHYSICAL_COLUMNS = frozenset(item["name"] for item in PHYSICAL_SCHEMA)

SEMANTIC_CLASS_SPECS: dict[str, dict[str, str]] = {
    "core_market_metrics": {
        "description": (
            "Standalone market, valuation, supply, and realized-value metric families."
        ),
        "matching_rule": (
            "Known standalone prefixes such as price, realized, investor, cost, "
            "invested, subsidy, supply, market, sopr, mvrv, and similar domains."
        ),
    },
    "windowed_return_and_path_metrics": {
        "description": (
            "Duration-led rolling return, DCA, lump-sum, and path-statistic metrics."
        ),
        "matching_rule": "Top-level prefix matches a duration token such as 1m, 1y, or 10y.",
    },
    "utxo_cohorts": {
        "description": "UTXO age-bucket cohort metrics.",
        "matching_rule": "Top-level prefix is utxos.",
    },
    "address_balance_cohorts": {
        "description": "Address balance-bucket cohort metrics.",
        "matching_rule": "Top-level prefix is addrs.",
    },
    "vintage_year_cohorts": {
        "description": "Vintage cohort metrics partitioned by originating year.",
        "matching_rule": "Top-level prefix is year with year_<yyyy>_* metric patterns.",
    },
    "halving_epoch_cohorts": {
        "description": "Halving-epoch cohort metrics partitioned by epoch id.",
        "matching_rule": "Top-level prefix is epoch with epoch_<n>_* metric patterns.",
    },
    "holder_cohorts": {
        "description": "Short-term and long-term holder cohort metrics.",
        "matching_rule": "Top-level prefix is sth or lth.",
    },
    "script_output_type_cohorts": {
        "description": "Script type and output-type cohort metrics.",
        "matching_rule": "Top-level prefix starts with p2 or is unknown, empty, or opreturn.",
    },
    "mining_pool_metrics": {
        "description": "Per-pool mining production, dominance, fee, coinbase, and subsidy metrics.",
        "matching_rule": (
            "Exact top-level family exposes the mining-pool signature "
            "(blocks_mined, dominance, coinbase, fee, subsidy)."
        ),
    },
    "address_activity_aggregates": {
        "description": "Address activity distribution aggregates.",
        "matching_rule": "Top-level prefix is address.",
    },
    "block_aggregates": {
        "description": "Block production, size, interval, weight, and fullness aggregates.",
        "matching_rule": "Top-level prefix is block.",
    },
    "benchmark_class_metrics": {
        "description": "Benchmark class path metrics such as DCA cohort ladders.",
        "matching_rule": "Top-level prefix is dca.",
    },
    "other_standalone_metrics": {
        "description": "Fallback class for documented families that do not match a more specific rule.",
        "matching_rule": "All remaining top-level families.",
    },
}

CORE_MARKET_PREFIXES = {
    "price",
    "market",
    "realized",
    "investor",
    "cost",
    "invested",
    "subsidy",
    "supply",
    "sopr",
    "mvrv",
    "cointime",
    "profit",
    "loss",
    "coinblocks",
    "coindays",
    "capitulation",
    "pain",
    "greed",
    "nupl",
    "sell",
    "spot",
    "total",
    "min",
    "max",
    "lower",
    "upper",
    "net",
    "unrealized",
    "adjusted",
    "value",
    "peak",
}

EXACT_SCRIPT_OUTPUT_PREFIXES = {"unknown", "empty", "opreturn"}
KNOWN_SUFFIXES = (
    ("_usd", "USD-denominated value"),
    ("_sats", "Satoshi-denominated value"),
    ("_btc", "BTC-denominated value"),
    ("_cents", "Cent-denominated fiat value"),
    ("_cumulative", "Cumulative running total"),
    ("_ema", "Exponential moving average"),
    ("_sma", "Simple moving average"),
    ("_ratio", "Ratio or relative-value transform"),
    ("_zscore", "Z-score normalization"),
)
PERCENTILE_SUFFIX_RE = re.compile(r"_pct\d{1,2}$")
DURATION_FAMILY_RE = re.compile(r"^\d+[hdwmy]$")
YEAR_TOKEN_RE = re.compile(r"^year_(\d{4})_")
EPOCH_TOKEN_RE = re.compile(r"^epoch_(\d+)_")
MINING_POOL_SIGNATURE_STEMS = {
    "blocks_mined",
    "dominance",
    "coinbase",
    "fee",
    "subsidy",
}


def taxonomy_json_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[1]
    return base / "data" / "brk_merged_metrics_taxonomy.json"


def taxonomy_docs_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[1]
    return base / "docs" / "reference" / "merged-metrics-taxonomy.md"


def resolve_default_parquet_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[1]
    matches = sorted(base.glob("merged_metrics*.parquet"))
    if not matches:
        raise FileNotFoundError(
            "No merged_metrics*.parquet file found under repository root."
        )
    return matches[-1]


def _family_token(metric_name: str) -> str:
    return metric_name.split("_", 1)[0]


def _is_mining_pool_family(family: str, metrics: list[str]) -> bool:
    if family.startswith("p2") or family in EXACT_SCRIPT_OUTPUT_PREFIXES:
        return False
    stems = {
        metric[len(family) + 1 :]
        for metric in metrics
        if metric.startswith(f"{family}_") and len(metric) > len(family) + 1
    }
    return MINING_POOL_SIGNATURE_STEMS.issubset(stems)


def classify_family(family: str, metrics: list[str]) -> str:
    if DURATION_FAMILY_RE.match(family):
        return "windowed_return_and_path_metrics"
    if family == "utxos":
        return "utxo_cohorts"
    if family == "addrs":
        return "address_balance_cohorts"
    if family == "year":
        return "vintage_year_cohorts"
    if family == "epoch":
        return "halving_epoch_cohorts"
    if family in {"sth", "lth"}:
        return "holder_cohorts"
    if family.startswith("p2") or family in EXACT_SCRIPT_OUTPUT_PREFIXES:
        return "script_output_type_cohorts"
    if family == "address":
        return "address_activity_aggregates"
    if family == "block":
        return "block_aggregates"
    if family == "dca":
        return "benchmark_class_metrics"
    if _is_mining_pool_family(family, metrics):
        return "mining_pool_metrics"
    if family in CORE_MARKET_PREFIXES:
        return "core_market_metrics"
    return "other_standalone_metrics"


def _family_pattern_summary(family: str, semantic_class: str, metrics: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if semantic_class == "windowed_return_and_path_metrics":
        summary["pattern"] = f"{family}_<measure>"
        summary["duration_token"] = family
        return summary
    if semantic_class == "vintage_year_cohorts":
        years = sorted({int(match.group(1)) for metric in metrics if (match := YEAR_TOKEN_RE.match(metric))})
        summary["pattern"] = "year_<yyyy>_<measure>"
        summary["cohort_values"] = years
        return summary
    if semantic_class == "halving_epoch_cohorts":
        epochs = sorted({int(match.group(1)) for metric in metrics if (match := EPOCH_TOKEN_RE.match(metric))})
        summary["pattern"] = "epoch_<n>_<measure>"
        summary["cohort_values"] = epochs
        return summary
    if semantic_class == "utxo_cohorts":
        summary["pattern"] = "utxos_<age_bucket>_<measure>"
        summary["cohort_examples"] = metrics[:8]
        return summary
    if semantic_class == "address_balance_cohorts":
        summary["pattern"] = "addrs_<balance_bucket>_<measure>"
        summary["cohort_examples"] = metrics[:8]
        return summary
    if semantic_class == "holder_cohorts":
        summary["pattern"] = f"{family}_<measure>"
        return summary
    if semantic_class == "script_output_type_cohorts":
        summary["pattern"] = f"{family}_<measure>"
        return summary
    if semantic_class == "mining_pool_metrics":
        summary["pattern"] = f"{family}_<measure>"
        summary["signature_measures"] = sorted(
            stem for stem in MINING_POOL_SIGNATURE_STEMS if f"{family}_{stem}" in metrics
        )
        return summary
    if semantic_class == "block_aggregates":
        summary["pattern"] = "block_<measure>"
        return summary
    if semantic_class == "address_activity_aggregates":
        summary["pattern"] = "address_<measure>"
        return summary
    if semantic_class == "benchmark_class_metrics":
        summary["pattern"] = "dca_class_<yyyy>_<measure>"
        return summary
    summary["pattern"] = f"{family}_<measure>"
    return summary


def build_taxonomy_from_metrics(
    *,
    parquet_name: str,
    row_count: int,
    distinct_days: int,
    distinct_metrics: int,
    min_day: str,
    max_day: str,
    metrics: list[str],
) -> dict[str, Any]:
    sorted_metrics = sorted(metrics)
    family_metrics: dict[str, list[str]] = defaultdict(list)
    for metric_name in sorted_metrics:
        family_metrics[_family_token(metric_name)].append(metric_name)

    top_level_families = [
        {"family": family, "metric_count": len(names)}
        for family, names in sorted(family_metrics.items(), key=lambda item: (-len(item[1]), item[0]))
    ]

    suffix_counts = Counter()
    for metric_name in sorted_metrics:
        for suffix, _ in KNOWN_SUFFIXES:
            if metric_name.endswith(suffix):
                suffix_counts[suffix] += 1
        percentile_match = PERCENTILE_SUFFIX_RE.search(metric_name)
        if percentile_match:
            suffix_counts[percentile_match.group(0)] += 1

    suffix_registry = []
    descriptions = {suffix: desc for suffix, desc in KNOWN_SUFFIXES}
    for suffix, count in sorted(suffix_counts.items(), key=lambda item: (-item[1], item[0])):
        suffix_registry.append(
            {
                "suffix": suffix,
                "count": count,
                "description": descriptions.get(
                    suffix,
                    "Percentile-style suffix.",
                ),
            }
        )

    namespace_registry = []
    class_to_namespaces: dict[str, list[str]] = defaultdict(list)
    class_metric_counts: Counter[str] = Counter()

    for family, names in sorted(family_metrics.items()):
        semantic_class = classify_family(family, names)
        class_to_namespaces[semantic_class].append(family)
        class_metric_counts[semantic_class] += len(names)
        namespace_registry.append(
            {
                "family": family,
                "semantic_class": semantic_class,
                "metric_count": len(names),
                "pattern_summary": _family_pattern_summary(family, semantic_class, names),
                "example_metrics": names[:8],
                "metrics": names,
            }
        )

    semantic_classes = []
    for class_name in sorted(class_to_namespaces):
        spec = SEMANTIC_CLASS_SPECS[class_name]
        namespaces = sorted(class_to_namespaces[class_name])
        semantic_classes.append(
            {
                "name": class_name,
                "description": spec["description"],
                "matching_rule": spec["matching_rule"],
                "namespace_count": len(namespaces),
                "metric_count": int(class_metric_counts[class_name]),
                "namespaces": namespaces,
            }
        )

    return {
        "dataset_snapshot": {
            "parquet_name": parquet_name,
            "row_count": row_count,
            "distinct_days": distinct_days,
            "distinct_metrics": distinct_metrics,
            "min_day": min_day,
            "max_day": max_day,
            "top_level_family_count": len(family_metrics),
        },
        "physical_schema": list(PHYSICAL_SCHEMA),
        "top_level_families": top_level_families,
        "suffix_registry": suffix_registry,
        "namespace_registry": namespace_registry,
        "semantic_classes": semantic_classes,
    }


def build_taxonomy_from_parquet(parquet_path: Path) -> dict[str, Any]:
    lf = pl.scan_parquet(parquet_path)
    schema = lf.collect_schema()
    if set(schema.names()) != EXPECTED_PHYSICAL_COLUMNS:
        raise ValueError(
            "merged_metrics taxonomy generation requires long-format parquet columns "
            f"{sorted(EXPECTED_PHYSICAL_COLUMNS)}, got {schema.names()}"
        )
    summary = (
        lf.select(
            pl.len().alias("row_count"),
            pl.col("day_utc").n_unique().alias("distinct_days"),
            pl.col("metric").n_unique().alias("distinct_metrics"),
            pl.col("day_utc").min().alias("min_day"),
            pl.col("day_utc").max().alias("max_day"),
        )
        .collect()
        .row(0, named=True)
    )
    metrics = (
        lf.select(pl.col("metric"))
        .unique()
        .collect()
        .get_column("metric")
        .to_list()
    )
    min_day = str(summary["min_day"])
    max_day = str(summary["max_day"])
    return build_taxonomy_from_metrics(
        parquet_name=parquet_path.name,
        row_count=int(summary["row_count"]),
        distinct_days=int(summary["distinct_days"]),
        distinct_metrics=int(summary["distinct_metrics"]),
        min_day=min_day,
        max_day=max_day,
        metrics=metrics,
    )


def render_taxonomy_docs(taxonomy: dict[str, Any]) -> str:
    snapshot = taxonomy["dataset_snapshot"]
    semantic_classes = taxonomy["semantic_classes"]
    suffix_registry = taxonomy["suffix_registry"]
    families = taxonomy["top_level_families"]
    namespace_registry = taxonomy["namespace_registry"]

    lines: list[str] = [
        "---",
        "title: Merged Metrics Taxonomy",
        "description: Generated semantic taxonomy for the BRK merged_metrics parquet metric namespace.",
        "---",
        "",
        "# Merged Metrics Taxonomy",
        "",
        "> Generated from `scripts/generate_merged_metrics_taxonomy.py` against "
        f"`{snapshot['parquet_name']}`.",
        "",
        "This page documents the semantic metric taxonomy for the canonical long-format "
        "`merged_metrics*.parquet` dataset.",
        "",
        "Dataset scale in the current canonical snapshot:",
        "",
        f"- `{snapshot['row_count']:,}` total rows",
        f"- `{snapshot['distinct_days']:,}` daily observations",
        f"- `{snapshot['distinct_metrics']:,}` distinct metric keys",
        f"- `{snapshot['top_level_family_count']:,}` top-level metric families",
        "",
        "Use [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md) for the "
        "physical parquet schema and runtime projection contract.",
        "",
        "Canonical generated artifacts:",
        "",
        "- markdown page: `docs/reference/merged-metrics-taxonomy.md`",
        "- JSON taxonomy: `data/brk_merged_metrics_taxonomy.json`",
        "- refresh command: `python scripts/generate_merged_metrics_taxonomy.py`",
        "",
        "## Snapshot",
        "",
        "| Property | Value |",
        "| --- | --- |",
        f"| Parquet file | `{snapshot['parquet_name']}` |",
        f"| Total rows | `{snapshot['row_count']:,}` |",
        f"| Distinct days | `{snapshot['distinct_days']:,}` |",
        f"| Distinct metrics | `{snapshot['distinct_metrics']:,}` |",
        f"| Day range | `{snapshot['min_day']}` to `{snapshot['max_day']}` |",
        f"| Top-level families | `{snapshot['top_level_family_count']:,}` |",
        "",
        "## Semantic Classes",
        "",
        "| Class | Namespace count | Metric count | Meaning | Rule |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in semantic_classes:
        lines.append(
            f"| `{item['name']}` | `{item['namespace_count']}` | `{item['metric_count']}` | "
            f"{item['description']} | {item['matching_rule']} |"
        )

    lines.extend(
        [
            "",
            "## Suffix And Unit Registry",
            "",
            "| Suffix | Count | Meaning |",
            "| --- | ---: | --- |",
        ]
    )
    for item in suffix_registry:
        lines.append(
            f"| `{item['suffix']}` | `{item['count']}` | {item['description']} |"
        )

    lines.extend(
        [
            "",
            "## Top-Level Families",
            "",
            "| Family | Metric count |",
            "| --- | ---: |",
        ]
    )
    for item in families:
        lines.append(f"| `{item['family']}` | `{item['metric_count']}` |")

    lines.extend(
        [
            "",
            "## Namespace Registry",
            "",
            "This table is exhaustive at the namespace/family level. The JSON artifact at "
            "`data/brk_merged_metrics_taxonomy.json` carries the full metric lists.",
            "",
            "| Family | Class | Metric count | Pattern | Examples |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for item in namespace_registry:
        pattern = item["pattern_summary"].get("pattern", "")
        examples = ", ".join(f"`{name}`" for name in item["example_metrics"][:3])
        lines.append(
            f"| `{item['family']}` | `{item['semantic_class']}` | `{item['metric_count']}` | "
            f"`{pattern}` | {examples} |"
        )

    lines.extend(
        [
            "",
            "## Runtime Projection Notes",
            "",
            "StackSats runtime does not consume the full metric namespace directly. "
            "It projects a small BRK-wide subset into runtime columns such as `date`, "
            "`price_usd`, `mvrv`, and selected overlay features.",
            "",
            "Canonical runtime subset documentation remains on "
            "[Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md) and "
            "[BRK Data Source](../data-source.md).",
            "",
        ]
    )
    return "\n".join(lines)


def render_taxonomy_json(taxonomy: dict[str, Any]) -> str:
    return json.dumps(taxonomy, indent=2, sort_keys=True) + "\n"

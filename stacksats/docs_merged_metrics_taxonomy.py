"""Helpers for generating merged-metrics taxonomy, catalog, and guide artifacts."""

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

RUNTIME_CRITICAL_METRICS = (
    "market_cap",
    "supply_btc",
    "mvrv",
    "adjusted_sopr",
    "adjusted_sopr_7d_ema",
    "realized_cap_growth_rate",
    "market_cap_growth_rate",
)

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

ACCESS_CATEGORY_SPECS: dict[str, dict[str, str]] = {
    "market_and_valuation": {
        "label": "Market and valuation",
        "description": (
            "Price, valuation bands, market value, realized value, and valuation ratios."
        ),
        "typical_use": "Price context, valuation regimes, and long-horizon market state modeling.",
    },
    "profitability_and_sopr": {
        "label": "Profitability and SOPR",
        "description": (
            "Realized and unrealized profit or loss, SOPR-style metrics, and spending pressure."
        ),
        "typical_use": "Profit-taking, capitulation, and spending-behavior analysis.",
    },
    "supply_issuance_and_scarcity": {
        "label": "Supply, issuance, and scarcity",
        "description": "Circulating supply, issuance, subsidy, inflation, and scarcity metrics.",
        "typical_use": "Supply-side modeling and issuance or dilution analysis.",
    },
    "holder_cohorts": {
        "label": "Holder cohorts",
        "description": "Short-term and long-term holder slices and holder-behavior metrics.",
        "typical_use": "Compare STH and LTH behavior across market regimes.",
    },
    "utxo_age_cohorts": {
        "label": "UTXO age cohorts",
        "description": "Age-bucketed UTXO metrics by holding period cohort.",
        "typical_use": "Analyze age-distributed supply, spentness, and conviction.",
    },
    "address_balance_cohorts": {
        "label": "Address balance cohorts",
        "description": "Address cohorts partitioned by balance bucket.",
        "typical_use": "Whale, retail, and cohort-balance distribution analysis.",
    },
    "vintage_and_halving_cohorts": {
        "label": "Vintage and halving cohorts",
        "description": "Year-vintage and halving-epoch cohort metrics.",
        "typical_use": "Cycle-aware cohort comparisons and halving-era analysis.",
    },
    "mining_pools_and_miner_economics": {
        "label": "Mining pools and miner economics",
        "description": "Mining-pool shares plus miner economics such as hash-price and fee flows.",
        "typical_use": "Miner revenue, pool share, and mining-economics monitoring.",
    },
    "script_and_output_types": {
        "label": "Script and output types",
        "description": "Output/script-type cohorts including p2* families, unknown, empty, and OP_RETURN.",
        "typical_use": "Track script adoption, output composition, and script-specific cohorts.",
    },
    "blocks_transactions_and_network_activity": {
        "label": "Blocks, transactions, and network activity",
        "description": "Blocks, transactions, throughput, activity, address counts, and network utilization.",
        "typical_use": "On-chain activity monitoring and network-usage context.",
    },
    "benchmarks_path_metrics_and_technical_indicators": {
        "label": "Benchmarks, path metrics, and technical indicators",
        "description": "Windowed return paths, DCA/lump-sum ladders, and indicator-style metrics.",
        "typical_use": "Benchmarking, path-dependent outcomes, and technical overlays.",
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
WINDOW_TOKEN_RE = re.compile(r"^\d+(?:h|d|w|m|y|sd)$")
YEAR_TOKEN_RE = re.compile(r"^year_(\d{4})_")
EPOCH_TOKEN_RE = re.compile(r"^epoch_(\d+)_")
MINING_POOL_SIGNATURE_STEMS = {
    "blocks_mined",
    "dominance",
    "coinbase",
    "fee",
    "subsidy",
}
STATISTIC_TOKENS = {"average", "median", "min", "max", "sum"}
UNIT_TOKENS = {"usd", "sats", "btc", "cents", "phs", "ths"}
TECHNICAL_FAMILIES = {"dca", "downside", "macd", "pi", "rsi", "sortino", "stoch"}
FAMILY_ALIAS_MAP = {
    "addr": "address",
    "emptyoutput": "empty",
    "unknownoutput": "unknown",
}
DISPLAY_TOKEN_MAP = {
    "addr": "address",
    "addrs": "addresses",
    "btc": "BTC",
    "dca": "DCA",
    "ema": "EMA",
    "eth": "ETH",
    "mvrv": "MVRV",
    "nupl": "NUPL",
    "opreturn": "OP_RETURN",
    "phs": "PH/s",
    "p2a": "P2A",
    "p2ms": "P2MS",
    "p2pk33": "P2PK33",
    "p2pk65": "P2PK65",
    "p2pkh": "P2PKH",
    "p2sh": "P2SH",
    "p2tr": "P2TR",
    "p2wpkh": "P2WPKH",
    "p2wsh": "P2WSH",
    "roi": "ROI",
    "rsi": "RSI",
    "sd": "SD",
    "segwit": "SegWit",
    "sma": "SMA",
    "sopr": "SOPR",
    "sth": "STH",
    "lth": "LTH",
    "ths": "TH/s",
    "usd": "USD",
    "utxos": "UTXOs",
    "vocdd": "VOCDD",
}


def taxonomy_json_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[1]
    return base / "data" / "brk_merged_metrics_taxonomy.json"


def taxonomy_docs_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[1]
    return base / "docs" / "reference" / "merged-metrics-taxonomy.md"


def catalog_json_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[1]
    return base / "data" / "brk_merged_metrics_catalog.json"


def data_guide_docs_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[1]
    return base / "docs" / "reference" / "merged-metrics-data-guide.md"


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


def _normalized_family(family: str) -> str:
    if not family:
        return "root"
    return FAMILY_ALIAS_MAP.get(family, family)


def _normalize_metric_name(metric_name: str) -> str:
    normalized = re.sub(r"_+", "_", metric_name.strip("_"))
    return normalized or metric_name


def _metric_parts(metric_name: str) -> list[str]:
    return [part for part in _normalize_metric_name(metric_name).split("_") if part]


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


def _family_pattern_summary(
    family: str,
    semantic_class: str,
    metrics: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if semantic_class == "windowed_return_and_path_metrics":
        summary["pattern"] = f"{family}_<measure>"
        summary["duration_token"] = family
        return summary
    if semantic_class == "vintage_year_cohorts":
        years = sorted(
            {
                int(match.group(1))
                for metric in metrics
                if (match := YEAR_TOKEN_RE.match(metric))
            }
        )
        summary["pattern"] = "year_<yyyy>_<measure>"
        summary["cohort_values"] = years
        return summary
    if semantic_class == "halving_epoch_cohorts":
        epochs = sorted(
            {
                int(match.group(1))
                for metric in metrics
                if (match := EPOCH_TOKEN_RE.match(metric))
            }
        )
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
            stem
            for stem in MINING_POOL_SIGNATURE_STEMS
            if f"{family}_{stem}" in metrics
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


def _extract_windows(metric_name: str) -> list[str]:
    windows: list[str] = []
    for part in _metric_parts(metric_name):
        if WINDOW_TOKEN_RE.match(part) and part not in windows:
            windows.append(part)
    return windows


def _extract_transforms(metric_name: str) -> list[str]:
    parts = _metric_parts(metric_name)
    transforms: list[str] = []
    ordered_rules = (
        ("change", lambda p: "change" in p),
        ("delta", lambda p: "delta" in p),
        ("growth_rate", lambda p: "growth" in p and "rate" in p),
        ("returns", lambda p: "returns" in p),
        ("cagr", lambda p: "cagr" in p),
        ("ratio", lambda p: "ratio" in p),
        ("cumulative", lambda p: "cumulative" in p),
        ("ema", lambda p: "ema" in p),
        ("sma", lambda p: "sma" in p),
        ("zscore", lambda p: "zscore" in p),
        ("dominance", lambda p: "dominance" in p),
        ("adoption", lambda p: "adoption" in p),
    )
    for label, predicate in ordered_rules:
        if predicate(parts):
            transforms.append(label)
    return transforms


def _detect_unit(metric_name: str) -> str | None:
    for part in reversed(_metric_parts(metric_name)):
        if part in UNIT_TOKENS:
            return part
    return None


def _detect_statistic(metric_name: str) -> str | None:
    normalized = _normalize_metric_name(metric_name)
    if match := PERCENTILE_SUFFIX_RE.search(normalized):
        return match.group(0).removeprefix("_")
    for part in reversed(_metric_parts(metric_name)):
        if part in STATISTIC_TOKENS:
            return part
    return None


def _detect_cohort_scheme(
    family: str,
    semantic_class: str,
) -> str | None:
    if semantic_class == "utxo_cohorts":
        return "utxo_age"
    if semantic_class == "address_balance_cohorts":
        return "address_balance"
    if semantic_class == "vintage_year_cohorts":
        return "vintage_year"
    if semantic_class == "halving_epoch_cohorts":
        return "halving_epoch"
    if semantic_class == "holder_cohorts":
        return "holder"
    if semantic_class == "script_output_type_cohorts":
        return "script_output_type"
    if semantic_class == "mining_pool_metrics":
        return "mining_pool"
    if semantic_class == "windowed_return_and_path_metrics":
        return "windowed_path"
    if family == "dca":
        return "benchmark_class"
    return None


def _detect_entity_scope(
    metric_name: str,
    family: str,
    semantic_class: str,
    access_category_key: str,
) -> str:
    normalized_family = _normalized_family(family)
    if semantic_class == "mining_pool_metrics":
        return f"mining_pool:{family}"
    if (
        access_category_key == "mining_pools_and_miner_economics"
        and family not in {"coinbase", "fee", "hash", "subsidy"}
        and not DURATION_FAMILY_RE.match(family)
    ):
        return f"mining_pool:{family}"
    if semantic_class == "script_output_type_cohorts":
        return f"script_output_type:{normalized_family}"
    if semantic_class == "holder_cohorts":
        if family == "sth":
            return "holder_short_term"
        if family == "lth":
            return "holder_long_term"
    if semantic_class == "utxo_cohorts":
        return "utxo_age_bucket"
    if semantic_class == "address_balance_cohorts":
        return "address_balance_bucket"
    if semantic_class == "vintage_year_cohorts":
        return "vintage_year"
    if semantic_class == "halving_epoch_cohorts":
        return "halving_epoch"
    if semantic_class == "windowed_return_and_path_metrics":
        return f"window:{family}"
    if family == "address":
        return "address_activity"
    if family == "block":
        return "blockchain_blocks"
    return "network_wide"


def _metric_contains_any(metric_name: str, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_metric_name(metric_name)
    return any(pattern in normalized for pattern in patterns)


def _classify_access_category(
    metric_name: str,
    family: str,
    semantic_class: str,
) -> str:
    normalized = _normalize_metric_name(metric_name)

    if semantic_class == "utxo_cohorts":
        return "utxo_age_cohorts"
    if semantic_class == "address_balance_cohorts":
        return "address_balance_cohorts"
    if semantic_class == "holder_cohorts":
        return "holder_cohorts"
    if semantic_class in {"vintage_year_cohorts", "halving_epoch_cohorts"}:
        return "vintage_and_halving_cohorts"
    if semantic_class == "mining_pool_metrics":
        return "mining_pools_and_miner_economics"
    if semantic_class == "script_output_type_cohorts":
        return "script_and_output_types"
    if DURATION_FAMILY_RE.match(family) or family in TECHNICAL_FAMILIES:
        return "benchmarks_path_metrics_and_technical_indicators"

    if _metric_contains_any(
        normalized,
        (
            "hash_price",
            "blocks_mined",
            "coinbase",
            "puell",
            "dominance",
            "subsidy",
        ),
    ):
        return "mining_pools_and_miner_economics"
    if _metric_contains_any(
        normalized,
        (
            "sopr",
            "profit",
            "loss",
            "capitulation",
            "pain",
            "greed",
            "value_created",
            "value_destroyed",
        ),
    ):
        return "profitability_and_sopr"
    if _metric_contains_any(
        normalized,
        (
            "supply",
            "inflation",
            "subsidy",
            "issuance",
            "circulating",
            "unspendable",
            "unclaimed",
        ),
    ):
        return "supply_issuance_and_scarcity"
    if _metric_contains_any(
        normalized,
        (
            "dca",
            "lump_sum",
            "returns",
            "cagr",
            "sortino",
            "macd",
            "stoch",
            "rsi",
        ),
    ):
        return "benchmarks_path_metrics_and_technical_indicators"
    if _metric_contains_any(
        normalized,
        (
            "block_",
            "tx_",
            "hash_rate",
            "difficulty",
            "input_",
            "inputs",
            "output_",
            "outputs",
            "address_activity",
            "addr_count",
            "new_addr",
            "sent",
            "received",
            "liveliness",
            "velocity",
            "segwit",
            "taproot",
            "vocdd",
            "coindays",
            "coinblocks",
        ),
    ):
        return "blocks_transactions_and_network_activity"
    if _metric_contains_any(
        normalized,
        (
            "price",
            "market",
            "mvrv",
            "realized_cap",
            "investor",
            "cost",
            "nvt",
            "nupl",
            "vaulted",
            "active_cap",
            "active_price",
            "true_market_mean",
            "oracle_price",
        ),
    ):
        return "market_and_valuation"
    if semantic_class == "core_market_metrics":
        return "market_and_valuation"
    return "blocks_transactions_and_network_activity"


def _display_token(token: str) -> str:
    if token in DISPLAY_TOKEN_MAP:
        return DISPLAY_TOKEN_MAP[token]
    if WINDOW_TOKEN_RE.match(token):
        return token
    if token.isdigit():
        return token
    return token.title()


def _render_display_label(metric_name: str) -> str:
    return " ".join(_display_token(part) for part in _metric_parts(metric_name))


def _measure_anchor(metric_name: str) -> str:
    parts = _metric_parts(metric_name)
    ignored_tokens = UNIT_TOKENS | STATISTIC_TOKENS | {
        "change",
        "delta",
        "growth",
        "rate",
        "returns",
        "cagr",
        "ratio",
        "cumulative",
        "ema",
        "sma",
        "zscore",
    }
    for part in reversed(parts):
        if part in ignored_tokens:
            continue
        if WINDOW_TOKEN_RE.match(part):
            continue
        if part.isdigit():
            continue
        return part
    return parts[-1] if parts else "metric"


def _catalog_note(
    metric_name: str,
    normalized_name: str,
    family: str,
    coverage_rows: int | None,
    distinct_days: int,
) -> str:
    notes: list[str] = []
    if metric_name != normalized_name:
        notes.append(f"metadata parsing normalizes `{metric_name}` to `{normalized_name}`")
    alias_family = FAMILY_ALIAS_MAP.get(family)
    if alias_family is not None:
        notes.append(f"family is grouped with `{alias_family}` for user-facing access guidance")
    if coverage_rows is not None and coverage_rows < distinct_days:
        notes.append("coverage is partial in the current snapshot")
    return "; ".join(notes)


def _join_or_none(values: list[str]) -> str | None:
    return ", ".join(values) if values else None


def _build_dimension_registry(
    items: list[dict[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for item in items:
        value = item.get(field)
        if value is None:
            continue
        for part in [chunk.strip() for chunk in value.split(",")]:
            if part:
                counts[part] += 1
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    ]


def _build_metric_catalog_entries(
    *,
    metrics: list[str],
    family_metrics: dict[str, list[str]],
    distinct_days: int,
    coverage_by_metric: dict[str, dict[str, str | int]] | None = None,
) -> list[dict[str, Any]]:
    catalog_entries: list[dict[str, Any]] = []
    semantic_class_by_family = {
        family: classify_family(family, names)
        for family, names in family_metrics.items()
    }
    for metric_name in sorted(metrics):
        family = _family_token(metric_name)
        semantic_class = semantic_class_by_family[family]
        normalized_name = _normalize_metric_name(metric_name)
        windows = _extract_windows(metric_name)
        transforms = _extract_transforms(metric_name)
        coverage = (coverage_by_metric or {}).get(metric_name, {})
        access_category_key = _classify_access_category(metric_name, family, semantic_class)
        coverage_rows = coverage.get("coverage_rows")
        first_day = coverage.get("first_day")
        last_day = coverage.get("last_day")
        catalog_entries.append(
            {
                "metric": metric_name,
                "family": family,
                "semantic_class": semantic_class,
                "access_category": ACCESS_CATEGORY_SPECS[access_category_key]["label"],
                "entity_scope": _detect_entity_scope(
                    metric_name, family, semantic_class, access_category_key
                ),
                "unit": _detect_unit(metric_name),
                "statistic": _detect_statistic(metric_name),
                "transform": _join_or_none(transforms),
                "window": _join_or_none(windows),
                "cohort_scheme": _detect_cohort_scheme(family, semantic_class),
                "coverage_rows": int(coverage_rows) if coverage_rows is not None else None,
                "first_day": str(first_day) if first_day is not None else None,
                "last_day": str(last_day) if last_day is not None else None,
                "example_metric_group": f"{_normalized_family(family)}:{_measure_anchor(metric_name)}",
                "display_label": _render_display_label(metric_name),
                "notes": _catalog_note(
                    metric_name,
                    normalized_name,
                    family,
                    int(coverage_rows) if coverage_rows is not None else None,
                    distinct_days,
                ),
            }
        )
    return catalog_entries


def _build_access_category_registry(
    metric_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metric_catalog:
        by_label[item["access_category"]].append(item)

    label_to_key = {
        spec["label"]: key for key, spec in ACCESS_CATEGORY_SPECS.items()
    }
    registry: list[dict[str, Any]] = []
    for label, items in sorted(by_label.items(), key=lambda entry: entry[0]):
        key = label_to_key[label]
        first_days = sorted(
            item["first_day"] for item in items if item["first_day"] is not None
        )
        last_days = sorted(
            item["last_day"] for item in items if item["last_day"] is not None
        )
        registry.append(
            {
                "key": key,
                "label": label,
                "description": ACCESS_CATEGORY_SPECS[key]["description"],
                "typical_use": ACCESS_CATEGORY_SPECS[key]["typical_use"],
                "metric_count": len(items),
                "family_count": len({item["family"] for item in items}),
                "example_metrics": [item["metric"] for item in items[:8]],
                "min_day": first_days[0] if first_days else None,
                "max_day": last_days[-1] if last_days else None,
                "coverage_note": (
                    "Per-metric coverage varies within this domain; inspect the catalog "
                    "for `coverage_rows`, `first_day`, and `last_day`."
                ),
            }
        )
    return registry


def build_taxonomy_from_metrics(
    *,
    parquet_name: str,
    row_count: int,
    distinct_days: int,
    distinct_metrics: int,
    min_day: str,
    max_day: str,
    metrics: list[str],
    coverage_by_metric: dict[str, dict[str, str | int]] | None = None,
) -> dict[str, Any]:
    sorted_metrics = sorted(metrics)
    family_metrics: dict[str, list[str]] = defaultdict(list)
    for metric_name in sorted_metrics:
        family_metrics[_family_token(metric_name)].append(metric_name)

    metric_catalog = _build_metric_catalog_entries(
        metrics=sorted_metrics,
        family_metrics=family_metrics,
        distinct_days=distinct_days,
        coverage_by_metric=coverage_by_metric,
    )
    access_category_registry = _build_access_category_registry(metric_catalog)

    top_level_families = [
        {"family": family, "metric_count": len(names)}
        for family, names in sorted(
            family_metrics.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
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
    for suffix, count in sorted(
        suffix_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        suffix_registry.append(
            {
                "suffix": suffix,
                "count": count,
                "description": descriptions.get(suffix, "Percentile-style suffix."),
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

    dimension_registries = {
        "units": _build_dimension_registry(metric_catalog, field="unit"),
        "statistics": _build_dimension_registry(metric_catalog, field="statistic"),
        "transforms": _build_dimension_registry(metric_catalog, field="transform"),
        "windows": _build_dimension_registry(metric_catalog, field="window"),
        "cohort_schemes": _build_dimension_registry(metric_catalog, field="cohort_scheme"),
        "entity_scopes": _build_dimension_registry(metric_catalog, field="entity_scope"),
    }

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
        "access_category_registry": access_category_registry,
        "dimension_registries": dimension_registries,
        "runtime_critical_metrics": list(RUNTIME_CRITICAL_METRICS),
    }


def build_metric_catalog_from_metrics(
    *,
    parquet_name: str,
    row_count: int,
    distinct_days: int,
    distinct_metrics: int,
    min_day: str,
    max_day: str,
    metrics: list[str],
    coverage_by_metric: dict[str, dict[str, str | int]] | None = None,
) -> dict[str, Any]:
    sorted_metrics = sorted(metrics)
    family_metrics: dict[str, list[str]] = defaultdict(list)
    for metric_name in sorted_metrics:
        family_metrics[_family_token(metric_name)].append(metric_name)
    metric_catalog = _build_metric_catalog_entries(
        metrics=sorted_metrics,
        family_metrics=family_metrics,
        distinct_days=distinct_days,
        coverage_by_metric=coverage_by_metric,
    )
    return {
        "dataset_snapshot": {
            "parquet_name": parquet_name,
            "row_count": row_count,
            "distinct_days": distinct_days,
            "distinct_metrics": distinct_metrics,
            "min_day": min_day,
            "max_day": max_day,
        },
        "fields": [
            "metric",
            "family",
            "semantic_class",
            "access_category",
            "entity_scope",
            "unit",
            "statistic",
            "transform",
            "window",
            "cohort_scheme",
            "coverage_rows",
            "first_day",
            "last_day",
            "example_metric_group",
            "display_label",
            "notes",
        ],
        "metrics": metric_catalog,
    }


def _coverage_by_metric_from_lazy_frame(
    lf: pl.LazyFrame,
) -> dict[str, dict[str, str | int]]:
    coverage_frame = (
        lf.group_by("metric")
        .agg(
            pl.len().alias("coverage_rows"),
            pl.col("day_utc").min().alias("first_day"),
            pl.col("day_utc").max().alias("last_day"),
        )
        .sort("metric")
        .collect()
    )
    coverage: dict[str, dict[str, str | int]] = {}
    for row in coverage_frame.iter_rows(named=True):
        coverage[str(row["metric"])] = {
            "coverage_rows": int(row["coverage_rows"]),
            "first_day": str(row["first_day"]),
            "last_day": str(row["last_day"]),
        }
    return coverage


def build_artifacts_from_parquet(parquet_path: Path) -> dict[str, Any]:
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
    coverage_by_metric = _coverage_by_metric_from_lazy_frame(lf)
    metrics = sorted(coverage_by_metric)
    min_day = str(summary["min_day"])
    max_day = str(summary["max_day"])
    taxonomy = build_taxonomy_from_metrics(
        parquet_name=parquet_path.name,
        row_count=int(summary["row_count"]),
        distinct_days=int(summary["distinct_days"]),
        distinct_metrics=int(summary["distinct_metrics"]),
        min_day=min_day,
        max_day=max_day,
        metrics=metrics,
        coverage_by_metric=coverage_by_metric,
    )
    catalog = build_metric_catalog_from_metrics(
        parquet_name=parquet_path.name,
        row_count=int(summary["row_count"]),
        distinct_days=int(summary["distinct_days"]),
        distinct_metrics=int(summary["distinct_metrics"]),
        min_day=min_day,
        max_day=max_day,
        metrics=metrics,
        coverage_by_metric=coverage_by_metric,
    )
    return {"taxonomy": taxonomy, "catalog": catalog}


def build_taxonomy_from_parquet(parquet_path: Path) -> dict[str, Any]:
    return build_artifacts_from_parquet(parquet_path)["taxonomy"]


def build_metric_catalog_from_parquet(parquet_path: Path) -> dict[str, Any]:
    return build_artifacts_from_parquet(parquet_path)["catalog"]


def render_taxonomy_docs(taxonomy: dict[str, Any]) -> str:
    snapshot = taxonomy["dataset_snapshot"]
    semantic_classes = taxonomy["semantic_classes"]
    suffix_registry = taxonomy["suffix_registry"]
    families = taxonomy["top_level_families"]
    namespace_registry = taxonomy["namespace_registry"]
    access_categories = taxonomy["access_category_registry"]
    dimension_registries = taxonomy["dimension_registries"]

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
        "This page documents the naming structure, compatibility taxonomy, and "
        "user-facing access categories for the canonical long-format "
        "`merged_metrics*.parquet` dataset.",
        "",
        "Read these pages in order:",
        "",
        "1. [Merged Metrics Data Guide](merged-metrics-data-guide.md)",
        "2. [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md)",
        "3. [Merged Metrics Taxonomy](merged-metrics-taxonomy.md)",
        "",
        "Dataset scale in the current canonical snapshot:",
        "",
        f"- `{snapshot['row_count']:,}` total rows",
        f"- `{snapshot['distinct_days']:,}` daily observations",
        f"- `{snapshot['distinct_metrics']:,}` distinct metric keys",
        f"- `{snapshot['top_level_family_count']:,}` top-level metric families",
        "",
        "Canonical generated artifacts:",
        "",
        "- markdown page: `docs/reference/merged-metrics-taxonomy.md`",
        "- data guide: `docs/reference/merged-metrics-data-guide.md`",
        "- JSON taxonomy: `data/brk_merged_metrics_taxonomy.json`",
        "- JSON catalog: `data/brk_merged_metrics_catalog.json`",
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
        "## User-Facing Access Categories",
        "",
        "| Category | Metric count | Family count | Meaning |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in access_categories:
        lines.append(
            f"| {item['label']} | `{item['metric_count']:,}` | `{item['family_count']:,}` | "
            f"{item['description']} |"
        )

    lines.extend(
        [
            "",
            "## Compatibility Semantic Classes",
            "",
            "These classes preserve the existing namespace-oriented grouping model. "
            "Use [Merged Metrics Data Guide](merged-metrics-data-guide.md) for the "
            "main user-facing answer to what data you can access.",
            "",
            "| Class | Namespace count | Metric count | Meaning | Rule |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in semantic_classes:
        lines.append(
            f"| `{item['name']}` | `{item['namespace_count']}` | `{item['metric_count']}` | "
            f"{item['description']} | {item['matching_rule']} |"
        )

    lines.extend(
        [
            "",
            "## Metric Dimension Registry",
            "",
            "The catalog separates units, statistics, transforms, windows, cohort schemes, "
            "and entity scopes instead of relying only on raw suffix buckets.",
            "",
            "### Units",
            "",
            "| Unit | Count |",
            "| --- | ---: |",
        ]
    )
    for item in dimension_registries["units"]:
        lines.append(f"| `{item['name']}` | `{item['count']}` |")

    lines.extend(["", "### Statistics", "", "| Statistic | Count |", "| --- | ---: |"])
    for item in dimension_registries["statistics"]:
        lines.append(f"| `{item['name']}` | `{item['count']}` |")

    lines.extend(["", "### Transforms", "", "| Transform | Count |", "| --- | ---: |"])
    for item in dimension_registries["transforms"]:
        lines.append(f"| `{item['name']}` | `{item['count']}` |")

    lines.extend(["", "### Windows", "", "| Window | Count |", "| --- | ---: |"])
    for item in dimension_registries["windows"]:
        lines.append(f"| `{item['name']}` | `{item['count']}` |")

    lines.extend(
        ["", "### Cohort Schemes", "", "| Cohort scheme | Count |", "| --- | ---: |"]
    )
    for item in dimension_registries["cohort_schemes"]:
        lines.append(f"| `{item['name']}` | `{item['count']}` |")

    lines.extend(
        ["", "### Entity Scopes", "", "| Entity scope | Count |", "| --- | ---: |"]
    )
    for item in dimension_registries["entity_scopes"][:40]:
        lines.append(f"| `{item['name']}` | `{item['count']}` |")

    lines.extend(
        [
            "",
            "## Compatibility Suffix Registry",
            "",
            "This registry is preserved for compatibility with earlier taxonomy outputs.",
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
            "`data/brk_merged_metrics_taxonomy.json` carries the full metric lists, "
            "while `data/brk_merged_metrics_catalog.json` carries per-metric coverage "
            "and access metadata.",
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
            "[Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md), "
            "[Merged Metrics Data Guide](merged-metrics-data-guide.md), and "
            "[BRK Data Source](../data-source.md).",
            "",
        ]
    )
    return "\n".join(lines)


def render_taxonomy_json(taxonomy: dict[str, Any]) -> str:
    return json.dumps(taxonomy, indent=2, sort_keys=True) + "\n"


def render_metric_catalog_json(catalog: dict[str, Any]) -> str:
    return json.dumps(catalog, indent=2, sort_keys=True) + "\n"


def render_data_guide_docs(
    taxonomy: dict[str, Any],
    catalog: dict[str, Any],
) -> str:
    snapshot = taxonomy["dataset_snapshot"]
    access_categories = taxonomy["access_category_registry"]
    catalog_metrics = catalog["metrics"]
    runtime_lookup = {item["metric"]: item for item in catalog_metrics}

    lines: list[str] = [
        "---",
        "title: Merged Metrics Data Guide",
        "description: What data a new user can access in the canonical BRK merged_metrics parquet.",
        "---",
        "",
        "# Merged Metrics Data Guide",
        "",
        "> Generated from `scripts/generate_merged_metrics_taxonomy.py` against "
        f"`{snapshot['parquet_name']}`.",
        "",
        "Use this page first if you want to understand what data you actually have "
        "access to through `merged_metrics*.parquet`.",
        "",
        "Reading order:",
        "",
        "1. [Merged Metrics Data Guide](merged-metrics-data-guide.md)",
        "2. [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md)",
        "3. [Merged Metrics Taxonomy](merged-metrics-taxonomy.md)",
        "",
        "## What This Dataset Is",
        "",
        "The canonical BRK dataset is a long-format daily fact table with exactly three columns:",
        "",
        "- `day_utc`: UTC calendar day",
        "- `metric`: metric key",
        "- `value`: numeric value for that metric on that day",
        "",
        "Each row is one daily numeric observation for one Bitcoin analytics metric. "
        "The dataset gives you access to thousands of derived BTC time series, not a "
        "wide table of fixed columns and not raw transaction-level blockchain records.",
        "",
        "Current snapshot scale:",
        "",
        f"- `{snapshot['row_count']:,}` rows",
        f"- `{snapshot['distinct_days']:,}` daily observations",
        f"- `{snapshot['distinct_metrics']:,}` metric keys",
        f"- `{snapshot['min_day']}` to `{snapshot['max_day']}` coverage in the current snapshot",
        "",
        "## What Data You Can Access",
        "",
        "The metric namespace covers these major user-facing domains:",
        "",
        "| Domain | What it covers | Representative metrics | Typical use | Coverage notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in access_categories:
        examples = ", ".join(f"`{metric}`" for metric in item["example_metrics"][:3])
        lines.append(
            f"| {item['label']} | {item['description']} | {examples} | "
            f"{item['typical_use']} | {item['min_day']} to {item['max_day']} across at least "
            "some metrics; per-metric coverage still varies. |"
        )

    lines.extend(
        [
            "",
            "## Major Data Domains",
            "",
            "High-level examples of what a new user can query from the metric namespace:",
            "",
            "- Market and valuation: `price_*`, `market_*`, `realized_*`, `mvrv`, `investor_*`, `cost_*`",
            "- Profitability and SOPR: `*_sopr*`, `*_profit*`, `*_loss*`, `capitulation_*`, `pain_*`",
            "- Supply and scarcity: `supply_*`, `circulating_*`, `subsidy_*`, `inflation_*`",
            "- Holder cohorts: `sth_*`, `lth_*`",
            "- UTXO age cohorts: `utxos_<age_bucket>_*`",
            "- Address balance cohorts: `addrs_<balance_bucket>_*`",
            "- Vintage and halving cohorts: `year_<yyyy>_*`, `epoch_<n>_*`",
            "- Mining pools and miner economics: `<pool>_blocks_mined`, `<pool>_dominance`, `hash_price_*`, `coinbase_*`, `fee_*`",
            "- Script and output types: `p2*_*`, `unknown_*`, `empty_*`, `opreturn_*`, `segwit_*`, `taproot_*`",
            "- Blocks, transactions, and network activity: `block_*`, `tx_*`, `hash_rate`, `difficulty*`, `sent*`, `received*`",
            "- Benchmarks and path metrics: `1m_*`, `1y_*`, `10y_*`, `dca_*`, `rsi_*`, `macd_*`",
            "",
            "## What This Dataset Does Not Contain",
            "",
            "- It does not expose raw transaction rows, raw block rows, or raw address ledgers.",
            "- It does not provide intraday timestamps; observations are daily.",
            "- It does not make every metric a dedicated parquet column; access happens through metric keys in long format.",
            f"- StackSats runtime does not consume all `{snapshot['distinct_metrics']:,}` metrics directly. Runtime uses a smaller derived BRK-wide projection.",
            "",
            "## Coverage Caveats",
            "",
            f"- Coverage is metric-specific. Some metrics begin much later than `{snapshot['min_day']}`.",
            "- Newer transforms and ratios can have shorter history because they depend on warmup windows or derived inputs.",
            "- Use `data/brk_merged_metrics_catalog.json` to inspect `coverage_rows`, `first_day`, and `last_day` for each metric.",
            "",
            "## Metrics Used By StackSats Runtime",
            "",
            "These runtime-critical metrics are the minimum projection used by built-in strategy audit tooling:",
            "",
            "| Metric | Coverage rows | First day | Last day |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for metric in RUNTIME_CRITICAL_METRICS:
        item = runtime_lookup[metric]
        coverage_rows = (
            f"{item['coverage_rows']:,}"
            if item["coverage_rows"] is not None
            else "n/a"
        )
        first_day = item["first_day"] or "n/a"
        last_day = item["last_day"] or "n/a"
        lines.append(
            f"| `{metric}` | `{coverage_rows}` | `{first_day}` | `{last_day}` |"
        )

    lines.extend(
        [
            "",
            "The runtime projection renames `day_utc` to `date` and derives `price_usd` from "
            "`market_cap / supply_btc`. See [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md) "
            "and [BRK Data Source](../data-source.md).",
            "",
            "## How To Search The Catalog And Taxonomy",
            "",
            "- Use `data/brk_merged_metrics_catalog.json` when you want per-metric access metadata and coverage.",
            "- Use [Merged Metrics Taxonomy](merged-metrics-taxonomy.md) when you want family-level naming patterns.",
            "- Search by `access_category` to find domains, by `family` to find namespaces, and by `display_label` or `metric` when you already know the concept or key.",
            "- If a metric name looks inconsistent, the catalog `notes` field explains metadata-only normalization such as collapsed double underscores or family aliases.",
            "",
            "## Related Pages",
            "",
            "- [Merged Metrics Parquet Schema](merged-metrics-parquet-schema.md)",
            "- [Merged Metrics Taxonomy](merged-metrics-taxonomy.md)",
            "- [BRK Data Source](../data-source.md)",
            "",
        ]
    )
    return "\n".join(lines)

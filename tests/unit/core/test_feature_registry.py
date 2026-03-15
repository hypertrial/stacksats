from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb", reason="duckdb not installed (pip install stacksats[brk])")

from stacksats.feature_registry import DEFAULT_FEATURE_REGISTRY, FeatureRegistry
from stacksats.strategy_types import BaseStrategy


class _ProviderStrategy(BaseStrategy):
    strategy_id = "provider-strategy"

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1", "brk_overlay_v1")

    def propose_weight(self, state):
        return state.uniform_weight


@dataclass(frozen=True, slots=True)
class _ConstantProvider:
    provider_id: str = "constant"

    def required_source_columns(self) -> tuple[str, ...]:
        return ("price_usd",)

    def materialize(self, btc_df, *, start_date, end_date, as_of_date) -> pd.DataFrame:
        idx = pd.date_range(start_date, as_of_date, freq="D")
        return pd.DataFrame({"const": 1.0}, index=idx)


def _btc_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    return pd.DataFrame(
        {
            "price_usd": pd.Series(range(1, 401), index=idx, dtype=float),
            "PriceUSD": pd.Series(range(1, 401), index=idx, dtype=float),
            "mvrv": pd.Series(range(400), index=idx, dtype=float) / 100.0 + 1.0,
        },
        index=idx,
    )


def _create_brk_overlay_duckdb_fixture(path: str) -> None:
    con = duckdb.connect(path)
    try:
        con.execute(
            "CREATE TABLE metrics_distribution (date_day DATE, metric VARCHAR, value DOUBLE)"
        )
        con.execute(
            "CREATE TABLE metrics_supply (date_day DATE, metric VARCHAR, value DOUBLE)"
        )
        con.execute(
            "CREATE TABLE metrics_transactions (date_day DATE, metric VARCHAR, value DOUBLE)"
        )
        con.execute(
            "CREATE TABLE metrics_blocks (date_day DATE, metric VARCHAR, value DOUBLE)"
        )

        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        distribution_metrics = (
            "adjusted_sopr",
            "adjusted_sopr_7d_ema",
            "net_sentiment",
            "greed_index",
            "pain_index",
        )
        supply_metrics = ("realized_cap_growth_rate", "market_cap_growth_rate")
        tx_metrics = ("tx_count_pct10", "annualized_volume_usd")
        block_metrics = ("hash_rate_1y_sma", "subsidy_usd_average")

        rows_distribution: list[tuple[object, str, float]] = []
        rows_supply: list[tuple[object, str, float]] = []
        rows_tx: list[tuple[object, str, float]] = []
        rows_blocks: list[tuple[object, str, float]] = []
        for day_idx, day in enumerate(dates):
            day_factor = float(day_idx + 1)
            for metric_idx, metric in enumerate(distribution_metrics):
                rows_distribution.append(
                    (day.date(), metric, day_factor * float(metric_idx + 1))
                )
            for metric_idx, metric in enumerate(supply_metrics):
                rows_supply.append(
                    (day.date(), metric, day_factor * float(metric_idx + 2))
                )
            for metric_idx, metric in enumerate(tx_metrics):
                rows_tx.append((day.date(), metric, day_factor * float(metric_idx + 3)))
            for metric_idx, metric in enumerate(block_metrics):
                rows_blocks.append(
                    (day.date(), metric, day_factor * float(metric_idx + 4))
                )

        con.executemany("INSERT INTO metrics_distribution VALUES (?, ?, ?)", rows_distribution)
        con.executemany("INSERT INTO metrics_supply VALUES (?, ?, ?)", rows_supply)
        con.executemany("INSERT INTO metrics_transactions VALUES (?, ?, ?)", rows_tx)
        con.executemany("INSERT INTO metrics_blocks VALUES (?, ?, ?)", rows_blocks)
    finally:
        con.close()


@pytest.fixture(autouse=True)
def _overlay_duckdb_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "overlay.duckdb"
    _create_brk_overlay_duckdb_fixture(str(db_path))
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))


def test_feature_registry_rejects_duplicate_registration() -> None:
    registry = FeatureRegistry()
    registry.register(_ConstantProvider())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_ConstantProvider())


def test_feature_registry_rejects_unknown_provider() -> None:
    with pytest.raises(KeyError, match="Unknown feature provider"):
        FeatureRegistry().get("missing")


def test_default_registry_materializes_strategy_features() -> None:
    frame = DEFAULT_FEATURE_REGISTRY.materialize_for_strategy(
        _ProviderStrategy(),
        _btc_df(),
        start_date=pd.Timestamp("2024-06-01"),
        end_date=pd.Timestamp("2024-12-31"),
        current_date=pd.Timestamp("2024-07-01"),
    )

    assert frame.index.min() == pd.Timestamp("2024-06-01")
    assert frame.index.max() == pd.Timestamp("2024-07-01")
    assert "price_vs_ma" in frame.columns
    assert "brk_netflow_fast" in frame.columns


def test_materialization_fingerprint_is_stable() -> None:
    strategy = _ProviderStrategy()
    btc_df = _btc_df()

    _, first = DEFAULT_FEATURE_REGISTRY.materialization_fingerprint(
        strategy,
        btc_df,
        start_date=pd.Timestamp("2024-06-01"),
        end_date=pd.Timestamp("2024-12-31"),
        current_date=pd.Timestamp("2024-07-01"),
    )
    _, second = DEFAULT_FEATURE_REGISTRY.materialization_fingerprint(
        strategy,
        btc_df,
        start_date=pd.Timestamp("2024-06-01"),
        end_date=pd.Timestamp("2024-12-31"),
        current_date=pd.Timestamp("2024-07-01"),
    )

    assert first == second


def test_provider_missing_required_columns_fails() -> None:
    btc_df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=5, freq="D"))
    strategy = _ProviderStrategy()

    with pytest.raises(ValueError, match="missing source columns"):
        DEFAULT_FEATURE_REGISTRY.materialize_for_strategy(
            strategy,
            btc_df,
            start_date=pd.Timestamp("2024-01-01"),
            end_date=pd.Timestamp("2024-01-05"),
            current_date=pd.Timestamp("2024-01-05"),
        )

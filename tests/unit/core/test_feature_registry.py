from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

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

    def materialize(self, btc_df, *, start_date, end_date, as_of_date) -> pl.DataFrame:
        del btc_df, end_date
        dates = pl.datetime_range(start_date, as_of_date, interval="1d", eager=True)
        return pl.DataFrame({"date": dates, "const": [1.0] * len(dates)})


def _btc_df() -> pl.DataFrame:
    dates = pl.datetime_range(
        datetime(2024, 1, 1),
        datetime(2025, 2, 3),
        interval="1d",
        eager=True,
    )
    rows = len(dates)
    return pl.DataFrame(
        {
            "date": dates,
            "price_usd": [float(i) for i in range(1, rows + 1)],
            "PriceUSD": [float(i) for i in range(1, rows + 1)],
            "mvrv": [1.0 + (i / 100.0) for i in range(rows)],
        }
    )


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
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 12, 31),
        current_date=datetime(2024, 7, 1),
    )

    assert frame["date"][0] == datetime(2024, 6, 1)
    assert frame["date"][-1] == datetime(2024, 7, 1)
    assert "price_vs_ma" in frame.columns
    assert "brk_netflow_fast" in frame.columns


def test_materialization_fingerprint_is_stable() -> None:
    strategy = _ProviderStrategy()
    btc_df = _btc_df()

    _, first = DEFAULT_FEATURE_REGISTRY.materialization_fingerprint(
        strategy,
        btc_df,
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 12, 31),
        current_date=datetime(2024, 7, 1),
    )
    _, second = DEFAULT_FEATURE_REGISTRY.materialization_fingerprint(
        strategy,
        btc_df,
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 12, 31),
        current_date=datetime(2024, 7, 1),
    )

    assert first == second


def test_provider_missing_required_columns_fails() -> None:
    strategy = _ProviderStrategy()
    btc_df = pl.DataFrame({"date": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 5), interval="1d", eager=True)})

    with pytest.raises(ValueError, match="missing source columns"):
        DEFAULT_FEATURE_REGISTRY.materialize_for_strategy(
            strategy,
            btc_df,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
            current_date=datetime(2024, 1, 5),
        )

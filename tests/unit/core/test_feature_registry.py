from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from stacksats.feature_registry import DEFAULT_FEATURE_REGISTRY, FeatureRegistry
from stacksats.strategy_types import BaseStrategy


class _ProviderStrategy(BaseStrategy):
    strategy_id = "provider-strategy"

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1", "coinmetrics_overlay_v1")

    def propose_weight(self, state):
        return state.uniform_weight


@dataclass(frozen=True, slots=True)
class _ConstantProvider:
    provider_id: str = "constant"

    def required_source_columns(self) -> tuple[str, ...]:
        return ("PriceUSD_coinmetrics",)

    def materialize(self, btc_df, *, start_date, end_date, as_of_date) -> pd.DataFrame:
        idx = pd.date_range(start_date, as_of_date, freq="D")
        return pd.DataFrame({"const": 1.0}, index=idx)


def _btc_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": pd.Series(range(1, 401), index=idx, dtype=float),
            "PriceUSD": pd.Series(range(1, 401), index=idx, dtype=float),
            "CapMVRVCur": pd.Series(range(400), index=idx, dtype=float) / 100.0 + 1.0,
        },
        index=idx,
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
        start_date=pd.Timestamp("2024-06-01"),
        end_date=pd.Timestamp("2024-12-31"),
        current_date=pd.Timestamp("2024-07-01"),
    )

    assert frame.index.min() == pd.Timestamp("2024-06-01")
    assert frame.index.max() == pd.Timestamp("2024-07-01")
    assert "price_vs_ma" in frame.columns
    assert "cm_netflow_fast" in frame.columns


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

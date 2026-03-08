"""Registry for framework-owned feature providers."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .feature_materialization import build_observed_frame, hash_dataframe, normalize_timestamp
from .feature_providers import (
    BRKOverlayFeatureProvider,
    CoreModelFeatureProvider,
    DuckDBAnalyticsFeatureProvider,
    FeatureProvider,
)


class FeatureRegistry:
    """Registry for deterministic feature provider lookup and materialization."""

    def __init__(self) -> None:
        self._providers: dict[str, FeatureProvider] = {}

    def register(self, provider: FeatureProvider) -> None:
        provider_id = str(provider.provider_id)
        if provider_id in self._providers:
            raise ValueError(f"Feature provider '{provider_id}' is already registered.")
        self._providers[provider_id] = provider

    def get(self, provider_id: str) -> FeatureProvider:
        if provider_id not in self._providers:
            raise KeyError(f"Unknown feature provider '{provider_id}'.")
        return self._providers[provider_id]

    def materialize_for_strategy(
        self,
        strategy,
        btc_df: pd.DataFrame,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.DataFrame:
        observed_end = min(normalize_timestamp(current_date), normalize_timestamp(end_date))
        observed_index = pd.date_range(normalize_timestamp(start_date), observed_end, freq="D")
        provider_ids = tuple(strategy.required_feature_sets())
        if not provider_ids:
            return pd.DataFrame(index=observed_index)

        merged = pd.DataFrame(index=observed_index)
        for provider_id in provider_ids:
            provider = self.get(provider_id)
            frame = provider.materialize(
                btc_df,
                start_date=normalize_timestamp(start_date),
                end_date=normalize_timestamp(end_date),
                as_of_date=observed_end,
            )
            observed = build_observed_frame(
                frame,
                start_date=start_date,
                current_date=observed_end,
            )
            duplicate_columns = sorted(set(merged.columns).intersection(observed.columns))
            if duplicate_columns:
                raise ValueError(
                    f"Feature providers produced duplicate columns for strategy "
                    f"{strategy.metadata().strategy_id}: {duplicate_columns}."
                )
            merged = merged.join(observed, how="left")
        return merged.sort_index()

    def provider_fingerprint(self, strategy) -> str:
        payload = {"provider_ids": list(strategy.required_feature_sets())}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def materialization_fingerprint(
        self,
        strategy,
        btc_df: pd.DataFrame,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, str]:
        frame = self.materialize_for_strategy(
            strategy,
            btc_df,
            start_date=start_date,
            end_date=end_date,
            current_date=current_date,
        )
        payload = {
            "providers": list(strategy.required_feature_sets()),
            "provider_fingerprint": self.provider_fingerprint(strategy),
            "frame_hash": hash_dataframe(frame),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return frame, digest


DEFAULT_FEATURE_REGISTRY = FeatureRegistry()
DEFAULT_FEATURE_REGISTRY.register(CoreModelFeatureProvider())
DEFAULT_FEATURE_REGISTRY.register(BRKOverlayFeatureProvider())
DEFAULT_FEATURE_REGISTRY.register(DuckDBAnalyticsFeatureProvider())

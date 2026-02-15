"""CoinMetrics-enhanced strategy wired to StackSats model_development logic.

The strategy starts from the package's MVRV/MA multiplier and adds lagged
CoinMetrics overlays (flow, activity, liquidity, exchange-supply share, miner
pressure). It still hands only daily intent to the sealed framework allocation
kernel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stacksats import (
    BacktestConfig,
    BaseStrategy,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)
from stacksats import model_development as model_lib


class ExampleMVRVStrategy(BaseStrategy):
    """MVRV strategy with CoinMetrics overlays and regime-aware gating."""

    strategy_id = "example-mvrv"
    version = "4.1.0"
    description = "MVRV + CoinMetrics overlays tuned for strict fold robustness."

    coinmetrics_cache_path: Path = Path(
        os.environ.get("STACKSATS_COINMETRICS_CSV", "~/.stacksats/cache/coinmetrics_btc.csv")
    ).expanduser()

    preference_temperature: float = 0.165
    netflow_weight: float = 0.084
    activity_weight: float = 0.584
    liquidity_weight: float = 0.217
    exchange_share_weight: float = 0.032
    miner_pressure_weight: float = -0.176

    def __init__(self) -> None:
        self._coinmetrics_features: pd.DataFrame | None = None

    @staticmethod
    def _clean_array(values: pd.Series) -> np.ndarray:
        arr = values.to_numpy(dtype=float)
        return np.where(np.isfinite(arr), arr, 0.0)

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        min_periods = max(30, window // 3)
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (series - mean) / std
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _load_coinmetrics_features(self) -> pd.DataFrame:
        if self._coinmetrics_features is not None:
            return self._coinmetrics_features

        path = self.coinmetrics_cache_path
        if not path.exists():
            self._coinmetrics_features = pd.DataFrame()
            return self._coinmetrics_features

        raw = pd.read_csv(path)
        if "time" not in raw.columns:
            self._coinmetrics_features = pd.DataFrame()
            return self._coinmetrics_features

        raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
        raw = raw.dropna(subset=["time"]).set_index("time")
        raw.index = raw.index.normalize().tz_localize(None)
        raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()

        numeric_cols = [
            "PriceUSD",
            "CapMrktCurUSD",
            "FlowInExUSD",
            "FlowOutExUSD",
            "AdrActCnt",
            "TxCnt",
            "volume_reported_spot_usd_1d",
            "SplyExNtv",
            "SplyCur",
            "IssTotUSD",
            "HashRate",
        ]
        for col in numeric_cols:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")

        features = pd.DataFrame(index=raw.index)

        if {"FlowInExUSD", "FlowOutExUSD", "CapMrktCurUSD"}.issubset(raw.columns):
            cap = raw["CapMrktCurUSD"].replace(0.0, np.nan)
            netflow = (raw["FlowInExUSD"] - raw["FlowOutExUSD"]) / cap
            features["cm_netflow_pressure"] = self._rolling_zscore(netflow, 90)
        else:
            features["cm_netflow_pressure"] = 0.0

        if "PriceUSD" in raw.columns:
            price_mom = self._rolling_zscore(np.log(raw["PriceUSD"]).diff(30), 365)
            activity_parts: list[pd.Series] = []
            if "AdrActCnt" in raw.columns:
                activity_parts.append(self._rolling_zscore(np.log1p(raw["AdrActCnt"]), 365))
            if "TxCnt" in raw.columns:
                activity_parts.append(self._rolling_zscore(np.log1p(raw["TxCnt"]), 365))
            if activity_parts:
                activity = sum(activity_parts) / float(len(activity_parts))
                features["cm_activity_divergence"] = activity - price_mom
            else:
                features["cm_activity_divergence"] = 0.0
        else:
            features["cm_activity_divergence"] = 0.0

        if "volume_reported_spot_usd_1d" in raw.columns:
            liquidity = self._rolling_zscore(np.log1p(raw["volume_reported_spot_usd_1d"]), 90)
            features["cm_liquidity_stress"] = liquidity
        else:
            features["cm_liquidity_stress"] = 0.0

        if {"SplyExNtv", "SplyCur"}.issubset(raw.columns):
            share = raw["SplyExNtv"] / raw["SplyCur"].replace(0.0, np.nan)
            features["cm_exchange_share"] = self._rolling_zscore(share, 180)
        else:
            features["cm_exchange_share"] = 0.0

        issuance_pressure: pd.Series | None = None
        hash_momentum: pd.Series | None = None
        if {"IssTotUSD", "CapMrktCurUSD"}.issubset(raw.columns):
            cap = raw["CapMrktCurUSD"].replace(0.0, np.nan)
            issuance_pressure = self._rolling_zscore(raw["IssTotUSD"] / cap, 180)
        if "HashRate" in raw.columns:
            hash_momentum = self._rolling_zscore(
                np.log(raw["HashRate"].replace(0.0, np.nan)).diff(30), 365
            )
        if issuance_pressure is not None and hash_momentum is not None:
            features["cm_miner_pressure"] = issuance_pressure - (0.5 * hash_momentum)
        elif issuance_pressure is not None:
            features["cm_miner_pressure"] = issuance_pressure
        elif hash_momentum is not None:
            features["cm_miner_pressure"] = -hash_momentum
        else:
            features["cm_miner_pressure"] = 0.0

        features = features.shift(1)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-4.0, 4.0)
        self._coinmetrics_features = features
        return features

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        # Runner already passes precomputed model features in ctx.features_df.
        # Add local CoinMetrics overlays (lagged) if cache is available.
        features = ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()
        cm = self._load_coinmetrics_features()
        if cm.empty:
            return features
        overlays = cm.reindex(features.index).fillna(0.0)
        return features.join(overlays, how="left").fillna(0.0)

    def build_signals(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        del ctx, features_df
        return {}

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del ctx, signals
        if features_df.empty:
            return TargetProfile(values=pd.Series(dtype=float), mode="preference")

        price_vs_ma = self._clean_array(features_df["price_vs_ma"])
        mvrv_zscore = self._clean_array(features_df["mvrv_zscore"])
        mvrv_gradient = self._clean_array(features_df["mvrv_gradient"])

        if "mvrv_percentile" in features_df.columns:
            mvrv_percentile = self._clean_array(features_df["mvrv_percentile"])
            mvrv_percentile = np.where(mvrv_percentile == 0.0, 0.5, mvrv_percentile)
        else:
            mvrv_percentile = None

        if "mvrv_acceleration" in features_df.columns:
            mvrv_acceleration = self._clean_array(features_df["mvrv_acceleration"])
        else:
            mvrv_acceleration = None

        if "mvrv_volatility" in features_df.columns:
            mvrv_volatility = self._clean_array(features_df["mvrv_volatility"])
            mvrv_volatility = np.where(mvrv_volatility == 0.0, 0.5, mvrv_volatility)
        else:
            mvrv_volatility = None

        if "signal_confidence" in features_df.columns:
            signal_confidence = self._clean_array(features_df["signal_confidence"])
            signal_confidence = np.where(signal_confidence == 0.0, 0.5, signal_confidence)
        else:
            signal_confidence = None

        multiplier = model_lib.compute_dynamic_multiplier(
            price_vs_ma,
            mvrv_zscore,
            mvrv_gradient,
            mvrv_percentile,
            mvrv_acceleration,
            mvrv_volatility,
            signal_confidence,
        )
        base_preference = np.log(np.clip(multiplier, 1e-12, None))

        netflow = self._clean_array(
            features_df.get("cm_netflow_pressure", pd.Series(0.0, index=features_df.index))
        )
        activity = self._clean_array(
            features_df.get("cm_activity_divergence", pd.Series(0.0, index=features_df.index))
        )
        liquidity = self._clean_array(
            features_df.get("cm_liquidity_stress", pd.Series(0.0, index=features_df.index))
        )
        exchange_share = self._clean_array(
            features_df.get("cm_exchange_share", pd.Series(0.0, index=features_df.index))
        )
        miner_pressure = self._clean_array(
            features_df.get("cm_miner_pressure", pd.Series(0.0, index=features_df.index))
        )

        mvrv_zone = self._clean_array(
            features_df.get("mvrv_zone", pd.Series(0.0, index=features_df.index))
        )
        volatility = self._clean_array(
            features_df.get("mvrv_volatility", pd.Series(0.5, index=features_df.index))
        )
        confidence = self._clean_array(
            features_df.get("signal_confidence", pd.Series(0.5, index=features_df.index))
        )

        value_regime_gate = np.where(
            mvrv_zone <= -1.0,
            1.25,
            np.where(mvrv_zone >= 1.0, 0.70, 1.00),
        )
        risk_regime_gate = np.where(
            mvrv_zone >= 1.0,
            1.25,
            np.where(mvrv_zone <= -1.0, 0.70, 1.00),
        )

        high_vol = np.clip((volatility - 0.5) * 2.0, 0.0, 1.0)
        uncertainty_shrink = 1.0 - (0.30 * high_vol)
        confidence_scale = 0.85 + (0.30 * np.clip(confidence, 0.0, 1.0))

        overlay = (
            -self.netflow_weight * netflow * risk_regime_gate
            + self.activity_weight * activity * value_regime_gate
            - self.liquidity_weight * liquidity * risk_regime_gate
            - self.exchange_share_weight * exchange_share * risk_regime_gate
            - self.miner_pressure_weight * miner_pressure * risk_regime_gate
        )

        preference = (base_preference * self.preference_temperature) + (
            overlay * uncertainty_shrink * confidence_scale
        )
        preference = np.where(np.isfinite(preference), preference, 0.0)
        preference = np.clip(preference, -50.0, 50.0)
        profile = pd.Series(preference, index=features_df.index, dtype=float)
        return TargetProfile(values=profile, mode="preference")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hook-based StackSats example strategy.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--strategy-label", type=str, default="example-mvrv-coinmetrics")
    args = parser.parse_args()

    strategy = ExampleMVRVStrategy()
    validation = strategy.validate(
        ValidationConfig(start_date=args.start_date, end_date=args.end_date)
    )
    print(validation.summary())
    for message in validation.messages:
        print(f"- {message}")

    result = strategy.backtest(
        BacktestConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            strategy_label=args.strategy_label,
        )
    )
    print(result.summary())
    result.plot(output_dir=args.output_dir)
    result.to_json(f"{args.output_dir}/backtest_result.json")


if __name__ == "__main__":
    main()

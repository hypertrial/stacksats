"""Score-focused CoinMetrics-enhanced strategy for StackSats.

This strategy starts from the package MVRV/MA multiplier and adds lagged,
regime-conditional CoinMetrics overlays with multi-horizon features and
interaction terms. It only returns daily intent and leaves allocation mechanics
to the sealed framework kernel.
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
    """MVRV strategy with score-focused CoinMetrics overlays."""

    strategy_id = "example-mvrv"
    version = "4.2.0"
    description = "Score-focused MVRV + multi-horizon CoinMetrics overlays."

    coinmetrics_cache_path: Path = Path(
        os.environ.get("STACKSATS_COINMETRICS_CSV", "~/.stacksats/cache/coinmetrics_btc.csv")
    ).expanduser()

    # Dynamic temperature controls how aggressively baseline preference is used.
    base_temperature: float = 0.58
    temperature_confidence_boost: float = 0.55
    temperature_volatility_penalty: float = 0.35
    temperature_zone_boost: float = 0.14

    # Overlay composition weights.
    overlay_scale: float = 0.75
    netflow_weight: float = 0.22
    activity_weight: float = 0.30
    liquidity_weight: float = 0.10
    exchange_share_weight: float = 0.08
    miner_pressure_weight: float = 0.12
    roi_weight: float = 0.12
    interaction_weight: float = 0.12
    momentum_follow_weight: float = 0.18

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

    @staticmethod
    def _to_numeric(raw: pd.DataFrame, columns: list[str]) -> None:
        for col in columns:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")

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

        self._to_numeric(
            raw,
            [
                "PriceUSD",
                "CapMrktCurUSD",
                "FlowInExUSD",
                "FlowOutExUSD",
                "AdrActCnt",
                "TxCnt",
                "FeeTotNtv",
                "volume_reported_spot_usd_1d",
                "SplyExNtv",
                "SplyCur",
                "IssTotUSD",
                "HashRate",
                "ROI30d",
                "ROI1yr",
            ],
        )

        features = pd.DataFrame(index=raw.index)

        # Exchange flow pressure: fast/slow/slope variants.
        if {"FlowInExUSD", "FlowOutExUSD", "CapMrktCurUSD"}.issubset(raw.columns):
            cap = raw["CapMrktCurUSD"].replace(0.0, np.nan)
            flow_ratio = (raw["FlowInExUSD"] - raw["FlowOutExUSD"]) / cap
            flow_fast = flow_ratio.rolling(7, min_periods=3).mean()
            flow_slow = flow_ratio.rolling(30, min_periods=10).mean()
            features["cm_netflow_fast"] = self._rolling_zscore(flow_fast, 120)
            features["cm_netflow_slow"] = self._rolling_zscore(flow_slow, 240)
            features["cm_netflow_slope"] = self._rolling_zscore(flow_fast - flow_slow, 180)
        else:
            features["cm_netflow_fast"] = 0.0
            features["cm_netflow_slow"] = 0.0
            features["cm_netflow_slope"] = 0.0

        # Activity and divergence vs price momentum.
        activity_parts: list[pd.Series] = []
        if "AdrActCnt" in raw.columns:
            activity_parts.append(self._rolling_zscore(np.log1p(raw["AdrActCnt"]), 365))
        if "TxCnt" in raw.columns:
            activity_parts.append(self._rolling_zscore(np.log1p(raw["TxCnt"]), 365))
        if "FeeTotNtv" in raw.columns:
            activity_parts.append(self._rolling_zscore(np.log1p(raw["FeeTotNtv"]), 365))

        if "PriceUSD" in raw.columns:
            price_log = np.log(raw["PriceUSD"].replace(0.0, np.nan))
            mom_30 = self._rolling_zscore(price_log.diff(30), 365)
            mom_90 = self._rolling_zscore(price_log.diff(90), 365)
        else:
            mom_30 = pd.Series(0.0, index=raw.index)
            mom_90 = pd.Series(0.0, index=raw.index)

        if activity_parts:
            activity = sum(activity_parts) / float(len(activity_parts))
            features["cm_activity_level"] = activity
            features["cm_activity_div_fast"] = activity - mom_30
            features["cm_activity_div_slow"] = activity - mom_90
        else:
            features["cm_activity_level"] = 0.0
            features["cm_activity_div_fast"] = 0.0
            features["cm_activity_div_slow"] = 0.0

        # Liquidity level and short impulse.
        if "volume_reported_spot_usd_1d" in raw.columns:
            vol_log = np.log1p(raw["volume_reported_spot_usd_1d"])
            features["cm_liquidity_level"] = self._rolling_zscore(vol_log, 180)
            features["cm_liquidity_impulse"] = self._rolling_zscore(vol_log.diff(7), 120)
        else:
            features["cm_liquidity_level"] = 0.0
            features["cm_liquidity_impulse"] = 0.0

        # Exchange supply share level + change.
        if {"SplyExNtv", "SplyCur"}.issubset(raw.columns):
            share = raw["SplyExNtv"] / raw["SplyCur"].replace(0.0, np.nan)
            features["cm_exchange_share_level"] = self._rolling_zscore(share, 240)
            features["cm_exchange_share_delta"] = self._rolling_zscore(share.diff(30), 240)
        else:
            features["cm_exchange_share_level"] = 0.0
            features["cm_exchange_share_delta"] = 0.0

        # Miner pressure + hash momentum.
        issuance_pressure: pd.Series | None = None
        if {"IssTotUSD", "CapMrktCurUSD"}.issubset(raw.columns):
            cap = raw["CapMrktCurUSD"].replace(0.0, np.nan)
            issuance_pressure = self._rolling_zscore(raw["IssTotUSD"] / cap, 240)

        hash_momentum: pd.Series | None = None
        if "HashRate" in raw.columns:
            hash_log = np.log(raw["HashRate"].replace(0.0, np.nan))
            hash_fast = self._rolling_zscore(hash_log.diff(30), 365)
            hash_slow = self._rolling_zscore(hash_log.diff(90), 365)
            hash_momentum = (0.6 * hash_fast) + (0.4 * hash_slow)

        if issuance_pressure is not None:
            features["cm_miner_pressure"] = issuance_pressure
        else:
            features["cm_miner_pressure"] = 0.0

        if hash_momentum is not None:
            features["cm_hash_momentum"] = hash_momentum
        else:
            features["cm_hash_momentum"] = 0.0

        # ROI context gates.
        if "ROI30d" in raw.columns:
            features["cm_roi30"] = self._rolling_zscore(raw["ROI30d"], 365)
        else:
            features["cm_roi30"] = mom_30

        if "ROI1yr" in raw.columns:
            features["cm_roi1y"] = self._rolling_zscore(raw["ROI1yr"], 365)
        else:
            features["cm_roi1y"] = mom_90

        features = features.shift(1)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features = features.clip(-6.0, 6.0)

        self._coinmetrics_features = features
        return features

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
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

        def _col(name: str) -> np.ndarray:
            return self._clean_array(features_df.get(name, pd.Series(0.0, index=features_df.index)))

        netflow_fast = _col("cm_netflow_fast")
        netflow_slow = _col("cm_netflow_slow")
        netflow_slope = _col("cm_netflow_slope")

        activity_level = _col("cm_activity_level")
        activity_div_fast = _col("cm_activity_div_fast")
        activity_div_slow = _col("cm_activity_div_slow")

        liquidity_level = _col("cm_liquidity_level")
        liquidity_impulse = _col("cm_liquidity_impulse")

        exchange_level = _col("cm_exchange_share_level")
        exchange_delta = _col("cm_exchange_share_delta")

        miner_pressure = _col("cm_miner_pressure")
        hash_momentum = _col("cm_hash_momentum")

        roi30 = _col("cm_roi30")
        roi1y = _col("cm_roi1y")

        mvrv_zone = _col("mvrv_zone")
        volatility = _col("mvrv_volatility") if "mvrv_volatility" in features_df else np.full(
            len(features_df.index), 0.5
        )
        confidence = _col("signal_confidence") if "signal_confidence" in features_df else np.full(
            len(features_df.index), 0.5
        )

        high_vol = np.clip((volatility - 0.5) / 0.5, 0.0, 1.0)
        value_gate = np.where(mvrv_zone <= -1.0, 1.25, np.where(mvrv_zone >= 1.0, 0.75, 1.00))
        risk_gate = np.where(mvrv_zone >= 1.0, 1.25, np.where(mvrv_zone <= -1.0, 0.75, 1.00))

        zone_temp_boost = np.where(mvrv_zone <= -1.0, self.temperature_zone_boost, -0.05)
        dynamic_temperature = (
            self.base_temperature
            + self.temperature_confidence_boost * (np.clip(confidence, 0.0, 1.0) - 0.5)
            - self.temperature_volatility_penalty * high_vol
            + zone_temp_boost
        )
        dynamic_temperature = np.clip(dynamic_temperature, 0.20, 1.15)

        netflow_component = (
            (-0.55 * netflow_fast) + (-0.30 * netflow_slow) + (-0.15 * netflow_slope)
        ) * risk_gate

        activity_component = (
            (0.40 * activity_level) + (0.40 * activity_div_fast) + (0.20 * activity_div_slow)
        ) * value_gate

        liquidity_component = (
            (-0.65 * liquidity_level) + (-0.35 * liquidity_impulse)
        ) * (0.60 + 0.40 * risk_gate) * (0.70 + 0.30 * high_vol)

        exchange_component = (
            (-0.70 * exchange_level) + (-0.30 * exchange_delta)
        ) * risk_gate

        miner_component = (
            (-0.60 * miner_pressure) + (-0.40 * hash_momentum)
        ) * (0.80 + 0.20 * risk_gate)

        roi_component = ((-0.65 * roi30) + (-0.35 * roi1y)) * np.where(
            mvrv_zone >= 1.0, 1.10, np.where(mvrv_zone <= -1.0, 0.80, 1.00)
        )

        interaction_component = (
            (-netflow_fast * np.clip(roi30, -3.0, 3.0)) * 0.40
            + (activity_div_fast * np.clip(confidence, 0.0, 1.0)) * 0.35
            + (-liquidity_level * high_vol) * 0.25
        )

        momentum_follow = (
            0.60 * np.tanh(roi30 / 2.0) + 0.40 * np.tanh(roi1y / 2.0)
        ) * np.where(mvrv_zone >= 1.0, 0.60, 1.00)

        overlay = (
            self.netflow_weight * netflow_component
            + self.activity_weight * activity_component
            + self.liquidity_weight * liquidity_component
            + self.exchange_share_weight * exchange_component
            + self.miner_pressure_weight * miner_component
            + self.roi_weight * roi_component
            + self.interaction_weight * interaction_component
            + self.momentum_follow_weight * momentum_follow
        )

        uncertainty_shrink = 1.0 - (0.25 * high_vol)
        confidence_scale = 0.75 + (0.50 * np.clip(confidence, 0.0, 1.0))

        preference = (base_preference * dynamic_temperature) + (
            self.overlay_scale * overlay * uncertainty_shrink * confidence_scale
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
    output_root = (
        Path(args.output_dir) / result.strategy_id / result.strategy_version / result.run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    result.plot(output_dir=str(output_root))
    result.to_json(output_root / "backtest_result.json")
    print(f"Saved: {output_root}")


if __name__ == "__main__":
    main()

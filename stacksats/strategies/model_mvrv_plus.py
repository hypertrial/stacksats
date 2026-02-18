"""Improved MVRV strategy with regime and risk gating.

This strategy preserves the package MVRV baseline signal and applies
lightweight, lagged gating to reduce over-allocation in adverse regimes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..model_development import compute_preference_scores
from ..runner import StrategyRunner
from ..strategy_types import (
    BacktestConfig,
    BaseStrategy,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)


class MVRVPlusStrategy(BaseStrategy):
    """MVRV baseline plus regime/risk/disagreement gates."""

    strategy_id = "mvrv-plus"
    version = "0.1.0"
    description = "MVRV baseline with CoinMetrics-aware regime and risk overlays."

    def __init__(
        self,
        deep_value_boost: float = 1.18,
        overheat_dampen: float = 0.62,
        neutral_scale: float = 1.00,
        high_vol_penalty: float = 0.34,
        drawdown_penalty: float = 0.24,
        disagreement_penalty: float = 0.20,
        confidence_boost: float = 0.16,
        overlay_scale: float = 0.20,
        smooth_alpha: float = 0.10,
        risk_budget_base: float = 1.02,
        risk_budget_min: float = 0.72,
        risk_budget_max: float = 1.22,
        miner_pressure_weight: float = 0.20,
        hash_momentum_weight: float = 0.12,
    ) -> None:
        self.deep_value_boost = deep_value_boost
        self.overheat_dampen = overheat_dampen
        self.neutral_scale = neutral_scale
        self.high_vol_penalty = high_vol_penalty
        self.drawdown_penalty = drawdown_penalty
        self.disagreement_penalty = disagreement_penalty
        self.confidence_boost = confidence_boost
        self.overlay_scale = overlay_scale
        self.smooth_alpha = smooth_alpha
        self.risk_budget_base = risk_budget_base
        self.risk_budget_min = risk_budget_min
        self.risk_budget_max = risk_budget_max
        self.miner_pressure_weight = miner_pressure_weight
        self.hash_momentum_weight = hash_momentum_weight
        self.coinmetrics_csv = Path(
            os.environ.get(
                "STACKSATS_COINMETRICS_CSV", "~/.stacksats/cache/coinmetrics_btc.csv"
            )
        ).expanduser()
        self._coinmetrics_overlays: pd.DataFrame | None = None

    @staticmethod
    def _safe_array(values: pd.Series, fill: float = 0.0) -> np.ndarray:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        arr = np.where(np.isfinite(arr), arr, fill)
        return arr

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        min_periods = max(30, window // 3)
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (series - mean) / std
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _adaptive_ewma(values: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        out = np.empty_like(values, dtype=float)
        out[0] = float(values[0])
        for i in range(1, len(values)):
            a = float(np.clip(alpha[i], 0.01, 0.99))
            out[i] = (a * float(values[i])) + ((1.0 - a) * out[i - 1])
        return out

    @staticmethod
    def _to_numeric(df: pd.DataFrame, columns: list[str]) -> None:
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    def _load_coinmetrics_overlays(self) -> pd.DataFrame:
        if self._coinmetrics_overlays is not None:
            return self._coinmetrics_overlays
        if not self.coinmetrics_csv.exists():
            self._coinmetrics_overlays = pd.DataFrame()
            return self._coinmetrics_overlays

        raw = pd.read_csv(self.coinmetrics_csv)
        if "time" not in raw.columns:
            self._coinmetrics_overlays = pd.DataFrame()
            return self._coinmetrics_overlays

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
                "SplyExNtv",
                "SplyCur",
                "AdrActCnt",
                "TxCnt",
                "TxTfrCnt",
                "ROI30d",
                "ROI1yr",
                "volume_reported_spot_usd_1d",
                "IssTotUSD",
                "HashRate",
            ],
        )

        features = pd.DataFrame(index=raw.index)

        if {"FlowInExUSD", "FlowOutExUSD", "CapMrktCurUSD"}.issubset(raw.columns):
            cap = raw["CapMrktCurUSD"].replace(0.0, np.nan)
            netflow = (raw["FlowInExUSD"] - raw["FlowOutExUSD"]) / cap
            features["cm_netflow"] = self._rolling_zscore(
                netflow.rolling(7, min_periods=3).mean(), 180
            )
        else:
            features["cm_netflow"] = 0.0

        if {"SplyExNtv", "SplyCur"}.issubset(raw.columns):
            exchange_share = raw["SplyExNtv"] / raw["SplyCur"].replace(0.0, np.nan)
            features["cm_exchange_share"] = self._rolling_zscore(exchange_share, 365)
            features["cm_exchange_share_delta"] = self._rolling_zscore(
                exchange_share.diff(30), 365
            )
        else:
            features["cm_exchange_share"] = 0.0
            features["cm_exchange_share_delta"] = 0.0

        activity_parts: list[pd.Series] = []
        for col in ("AdrActCnt", "TxCnt", "TxTfrCnt"):
            if col in raw.columns:
                activity_parts.append(self._rolling_zscore(np.log1p(raw[col]), 365))
        if activity_parts:
            activity = sum(activity_parts) / float(len(activity_parts))
            if "PriceUSD" in raw.columns:
                price_log = np.log(raw["PriceUSD"].replace(0.0, np.nan))
                momentum = self._rolling_zscore(price_log.diff(30), 365)
            else:
                momentum = pd.Series(0.0, index=raw.index)
            features["cm_activity_div"] = activity - momentum
        else:
            features["cm_activity_div"] = 0.0

        if "ROI30d" in raw.columns and "ROI1yr" in raw.columns:
            roi30 = self._rolling_zscore(raw["ROI30d"], 365)
            roi1y = self._rolling_zscore(raw["ROI1yr"], 365)
            features["cm_roi_context"] = (-0.65 * roi30) + (-0.35 * roi1y)
        else:
            features["cm_roi_context"] = 0.0

        if "volume_reported_spot_usd_1d" in raw.columns:
            vol = np.log1p(raw["volume_reported_spot_usd_1d"])
            features["cm_liquidity_impulse"] = self._rolling_zscore(vol.diff(7), 180)
        else:
            features["cm_liquidity_impulse"] = 0.0

        if {"IssTotUSD", "CapMrktCurUSD"}.issubset(raw.columns):
            cap = raw["CapMrktCurUSD"].replace(0.0, np.nan)
            issuance = raw["IssTotUSD"] / cap
            features["cm_miner_pressure"] = self._rolling_zscore(issuance, 365)
        else:
            features["cm_miner_pressure"] = 0.0

        if "HashRate" in raw.columns:
            hash_log = np.log(raw["HashRate"].replace(0.0, np.nan))
            hash_mom = self._rolling_zscore(hash_log.diff(30), 365)
            features["cm_hash_momentum"] = hash_mom
        else:
            features["cm_hash_momentum"] = 0.0

        features = features.shift(1)
        features = (
            features.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6.0, 6.0)
        )
        self._coinmetrics_overlays = features
        return self._coinmetrics_overlays

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        window = ctx.features_df.loc[ctx.start_date : ctx.end_date].copy()
        if window.empty:
            return window

        price = pd.to_numeric(window.get("PriceUSD_coinmetrics"), errors="coerce")
        returns = price.pct_change()
        rolling_vol = returns.rolling(21, min_periods=10).std()
        rolling_max = price.rolling(90, min_periods=30).max()
        drawdown = (price / rolling_max) - 1.0

        # Shift by one day to avoid using same-day information.
        window["plus_vol21"] = rolling_vol.shift(1).fillna(0.0)
        window["plus_drawdown90"] = drawdown.shift(1).fillna(0.0)
        overlays = self._load_coinmetrics_overlays()
        if not overlays.empty:
            window = window.join(overlays.reindex(window.index).fillna(0.0), how="left")
        window = window.fillna(0.0)
        return window

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pd.DataFrame,
        signals: dict[str, pd.Series],
    ) -> TargetProfile:
        del signals
        if features_df.empty:
            return TargetProfile(values=pd.Series(dtype=float), mode="preference")

        base_pref = compute_preference_scores(
            features_df=features_df,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
        ).to_numpy(dtype=float)

        zone = self._safe_array(
            features_df.get("mvrv_zone", pd.Series(0.0, index=features_df.index))
        )
        mvrv_zscore = self._safe_array(
            features_df.get("mvrv_zscore", pd.Series(0.0, index=features_df.index))
        )
        price_vs_ma = self._safe_array(
            features_df.get("price_vs_ma", pd.Series(0.0, index=features_df.index))
        )
        confidence = self._safe_array(
            features_df.get(
                "signal_confidence", pd.Series(0.5, index=features_df.index)
            ),
            fill=0.5,
        )
        confidence = np.where(confidence == 0.0, 0.5, confidence)
        confidence = np.clip(confidence, 0.0, 1.0)

        mvrv_volatility = self._safe_array(
            features_df.get("mvrv_volatility", pd.Series(0.5, index=features_df.index)),
            fill=0.5,
        )
        mvrv_volatility = np.where(mvrv_volatility == 0.0, 0.5, mvrv_volatility)
        mvrv_volatility = np.clip(mvrv_volatility, 0.0, 1.0)

        vol21 = self._safe_array(
            features_df.get("plus_vol21", pd.Series(0.0, index=features_df.index))
        )
        drawdown90 = self._safe_array(
            features_df.get("plus_drawdown90", pd.Series(0.0, index=features_df.index))
        )

        regime_scale = np.where(
            zone <= -1.0,
            self.deep_value_boost,
            np.where(zone >= 1.0, self.overheat_dampen, self.neutral_scale),
        )

        high_vol = np.clip((mvrv_volatility - 0.70) / 0.30, 0.0, 1.0)
        vol_scale = 1.0 - (self.high_vol_penalty * high_vol)

        dd_severity = np.clip((-drawdown90 - 0.10) / 0.30, 0.0, 1.0)
        dd_scale = 1.0 - (self.drawdown_penalty * dd_severity)

        # Penalize when value and trend strongly disagree.
        value_signal = -mvrv_zscore
        trend_signal = -price_vs_ma
        disagreement = np.where((value_signal * trend_signal) < 0.0, 1.0, 0.0)
        disagreement_strength = np.clip(
            np.abs(value_signal - trend_signal) / 4.0, 0.0, 1.0
        )
        disagreement_penalty_regime = self.disagreement_penalty * np.where(
            zone >= 1.0,
            1.35,
            np.where(zone <= -1.0, 0.75, 1.00),
        )
        disagreement_scale = 1.0 - (
            disagreement_penalty_regime * disagreement * disagreement_strength
        )

        # Confidence lifts conviction only in medium-low volatility environments.
        conf_centered = np.clip((confidence - 0.5) / 0.5, -1.0, 1.0)
        vol_soft_cap = 1.0 - (0.5 * high_vol)
        confidence_scale = 1.0 + (self.confidence_boost * conf_centered * vol_soft_cap)
        confidence_scale = np.clip(confidence_scale, 0.85, 1.20)

        # Additional small clamp for very high realized short-term volatility.
        vol21_cap = np.clip(vol21 / 0.08, 0.0, 1.0)
        realized_vol_scale = 1.0 - (0.10 * vol21_cap)

        netflow = self._safe_array(
            features_df.get("cm_netflow", pd.Series(0.0, index=features_df.index))
        )
        exchange_share = self._safe_array(
            features_df.get(
                "cm_exchange_share", pd.Series(0.0, index=features_df.index)
            )
        )
        exchange_share_delta = self._safe_array(
            features_df.get(
                "cm_exchange_share_delta", pd.Series(0.0, index=features_df.index)
            )
        )
        activity_div = self._safe_array(
            features_df.get("cm_activity_div", pd.Series(0.0, index=features_df.index))
        )
        roi_context = self._safe_array(
            features_df.get("cm_roi_context", pd.Series(0.0, index=features_df.index))
        )
        liquidity_impulse = self._safe_array(
            features_df.get(
                "cm_liquidity_impulse", pd.Series(0.0, index=features_df.index)
            )
        )
        miner_pressure = self._safe_array(
            features_df.get(
                "cm_miner_pressure", pd.Series(0.0, index=features_df.index)
            )
        )
        hash_momentum = self._safe_array(
            features_df.get("cm_hash_momentum", pd.Series(0.0, index=features_df.index))
        )

        value_gate = np.where(zone <= -1.0, 1.20, np.where(zone >= 1.0, 0.80, 1.00))
        risk_gate = np.where(zone >= 1.0, 1.25, np.where(zone <= -1.0, 0.80, 1.00))

        overlay = (
            (-0.30 * netflow * risk_gate)
            + (-0.22 * exchange_share * risk_gate)
            + (-0.14 * exchange_share_delta * risk_gate)
            + (0.28 * activity_div * value_gate)
            + (0.26 * roi_context * np.where(zone >= 1.0, 1.10, 0.95))
            + (-0.16 * liquidity_impulse * (0.8 + 0.2 * risk_gate))
            + (-self.miner_pressure_weight * miner_pressure * risk_gate)
            + (self.hash_momentum_weight * hash_momentum * value_gate)
        )
        overlay = np.clip(overlay, -6.0, 6.0)

        scale = (
            regime_scale
            * vol_scale
            * dd_scale
            * disagreement_scale
            * confidence_scale
            * realized_vol_scale
        )
        overlay_scale_regime = self.overlay_scale * np.where(
            zone >= 1.0,
            1.25,
            np.where(zone <= -1.0, 1.10, 0.85),
        )

        # Adaptive risk budget scales overall conviction by structural stress.
        risk_pressure = (
            (0.40 * high_vol)
            + (0.30 * dd_severity)
            + (0.20 * np.clip(netflow / 3.0, -1.0, 1.0))
            + (0.10 * np.clip(miner_pressure / 3.0, -1.0, 1.0))
        )
        risk_support = (
            (0.45 * np.clip(activity_div / 3.0, -1.0, 1.0))
            + (0.35 * np.clip(roi_context / 3.0, -1.0, 1.0))
            + (0.20 * np.clip(hash_momentum / 3.0, -1.0, 1.0))
        )
        regime_risk_intensity = np.where(
            zone >= 1.0, 1.25, np.where(zone <= -1.0, 0.90, 1.00)
        )
        risk_budget = self.risk_budget_base + (
            0.20 * (risk_support - risk_pressure) * regime_risk_intensity
        )
        risk_budget = np.clip(risk_budget, self.risk_budget_min, self.risk_budget_max)

        preference = (
            (base_pref * scale) + (overlay_scale_regime * overlay)
        ) * risk_budget
        preference = np.clip(preference, -50.0, 50.0)
        smooth_alpha_regime = np.where(
            np.abs(zone) >= 1.0,
            min(0.35, self.smooth_alpha * 1.35),
            max(0.08, self.smooth_alpha * 0.75),
        )
        preference = self._adaptive_ewma(preference, smooth_alpha_regime)
        preference = np.where(np.isfinite(preference), preference, 0.0)

        return TargetProfile(
            values=pd.Series(preference, index=features_df.index, dtype=float),
            mode="preference",
        )


def _load_coinmetrics_csv(path: str) -> pd.DataFrame:
    """Load local CoinMetrics CSV without network fallback."""
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CoinMetrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("CoinMetrics CSV missing required 'time' column.")
    if "PriceUSD" not in df.columns:
        raise ValueError("CoinMetrics CSV missing required 'PriceUSD' column.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["PriceUSD"] = pd.to_numeric(df["PriceUSD"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time")
    df.index = df.index.normalize().tz_localize(None)
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    df["PriceUSD_coinmetrics"] = df["PriceUSD"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MVRV-plus strategy.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument("--strategy-label", type=str, default="mvrv-plus")
    parser.add_argument(
        "--coinmetrics-csv",
        type=str,
        default=os.environ.get(
            "STACKSATS_COINMETRICS_CSV", "~/.stacksats/cache/coinmetrics_btc.csv"
        ),
        help="Local CoinMetrics BTC CSV path; no network fetch is used.",
    )
    args = parser.parse_args()

    strategy = MVRVPlusStrategy()
    btc_df = _load_coinmetrics_csv(args.coinmetrics_csv)
    csv_end = btc_df.index.max().normalize()

    requested_end = (
        pd.to_datetime(args.end_date).normalize() if args.end_date else csv_end
    )
    if pd.isna(requested_end):
        raise ValueError(f"Invalid --end-date: {args.end_date!r}")
    effective_end = min(requested_end, csv_end)
    effective_start = args.start_date

    runner = StrategyRunner()
    validation = runner.validate(
        strategy,
        ValidationConfig(
            start_date=effective_start, end_date=effective_end.strftime("%Y-%m-%d")
        ),
        btc_df=btc_df,
    )
    print(validation.summary())
    for message in validation.messages:
        print(f"- {message}")

    result = runner.backtest(
        strategy,
        BacktestConfig(
            start_date=effective_start,
            end_date=effective_end.strftime("%Y-%m-%d"),
            strategy_label=args.strategy_label,
            output_dir=args.output_dir,
        ),
        btc_df=btc_df,
    )
    print(result.summary())
    output_root = (
        Path(args.output_dir)
        / result.strategy_id
        / result.strategy_version
        / result.run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    result.plot(output_dir=str(output_root))
    result.to_json(output_root / "backtest_result.json")
    print(f"Saved: {output_root}")


if __name__ == "__main__":
    main()

"""Improved MVRV strategy with regime and risk gating.

This strategy preserves the package MVRV baseline signal and applies
lightweight, lagged gating to reduce over-allocation in adverse regimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..model_development import FEATS, compute_preference_scores
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
    overlay_features: tuple[str, ...] = (
        "cm_netflow",
        "cm_exchange_share",
        "cm_exchange_share_delta",
        "cm_activity_div",
        "cm_roi_context",
        "cm_liquidity_impulse",
        "cm_miner_pressure",
        "cm_hash_momentum",
    )
    derived_features: tuple[str, ...] = (
        "plus_vol21",
        "plus_drawdown90",
    )

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

    def required_feature_columns(self) -> tuple[str, ...]:
        return ("PriceUSD_coinmetrics", *FEATS, *self.overlay_features, *self.derived_features)

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1", "coinmetrics_overlay_v1")

    @staticmethod
    def _safe_array(values: pd.Series, fill: float = 0.0) -> np.ndarray:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        arr = np.where(np.isfinite(arr), arr, fill)
        return arr

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

    def transform_features(self, ctx: StrategyContext) -> pd.DataFrame:
        window = ctx.features_df.copy()
        if window.empty:
            return window
        start_date = pd.Timestamp(ctx.start_date).normalize()
        end_date = pd.Timestamp(ctx.end_date).normalize()
        if start_date > end_date:
            return window.iloc[0:0].copy()
        window = window.loc[(window.index >= start_date) & (window.index <= end_date)].copy()
        if window.empty:
            return window

        price = pd.to_numeric(window.get("PriceUSD_coinmetrics"), errors="coerce")
        returns = price.pct_change()
        rolling_vol = returns.rolling(21, min_periods=10).std()
        rolling_max = price.rolling(90, min_periods=30).max()
        drawdown = (price / rolling_max) - 1.0

        # Shift by one day to avoid using same-day information.
        vol21 = rolling_vol.shift(1).fillna(0.0)
        drawdown90 = drawdown.shift(1).fillna(0.0)

        # Collision-safe behavior: keep provider values when present and only fill gaps.
        if "plus_vol21" in window.columns:
            existing_vol21 = pd.to_numeric(window["plus_vol21"], errors="coerce")
            window["plus_vol21"] = existing_vol21.fillna(vol21)
        else:
            window["plus_vol21"] = vol21
        if "plus_drawdown90" in window.columns:
            existing_drawdown = pd.to_numeric(window["plus_drawdown90"], errors="coerce")
            window["plus_drawdown90"] = existing_drawdown.fillna(drawdown90)
        else:
            window["plus_drawdown90"] = drawdown90
        window = window.replace([np.inf, -np.inf], np.nan).fillna(0.0)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MVRV-plus strategy.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument("--strategy-label", type=str, default="mvrv-plus")
    args = parser.parse_args()

    strategy = MVRVPlusStrategy()
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
        ),
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

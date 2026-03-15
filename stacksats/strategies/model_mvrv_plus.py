"""Improved MVRV strategy with regime and risk gating.

This strategy preserves the package MVRV baseline signal and applies
lightweight, lagged gating to reduce over-allocation in adverse regimes.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from ..model_development import FEATS, compute_preference_scores
from ..strategy_types import (
    BacktestConfig,
    BaseStrategy,
    StrategyContext,
    TargetProfile,
    ValidationConfig,
)


def _to_dt(val) -> dt.datetime:
    if isinstance(val, dt.datetime):
        return val.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.datetime.strptime(str(val)[:10], "%Y-%m-%d")


class MVRVPlusStrategy(BaseStrategy):
    """MVRV baseline plus regime/risk/disagreement gates."""

    strategy_id = "mvrv-plus"
    version = "0.1.0"
    description = "MVRV baseline with BRK-aware regime and risk overlays."
    overlay_features: tuple[str, ...] = (
        "brk_netflow",
        "brk_exchange_share",
        "brk_exchange_share_delta",
        "brk_activity_div",
        "brk_roi_context",
        "brk_liquidity_impulse",
        "brk_miner_pressure",
        "brk_hash_momentum",
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
        # derived_features are created in transform_features(), so only require
        # source/provider columns at context construction time.
        return ("price_usd", *FEATS, *self.overlay_features)

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1", "brk_overlay_v1")

    @staticmethod
    def _safe_array(values: pl.Series, fill: float = 0.0) -> np.ndarray:
        arr = values.cast(pl.Float64, strict=False).fill_null(fill).to_numpy()
        return np.where(np.isfinite(arr), arr, fill)

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

    def transform_features(self, ctx: StrategyContext) -> pl.DataFrame:
        window = ctx.features.data.clone()
        if window.is_empty():
            return window
        start_dt = _to_dt(ctx.start_date)
        end_dt = _to_dt(ctx.end_date)
        if start_dt > end_dt:
            return window.head(0)
        window = window.filter(
            (pl.col("date") >= start_dt) & (pl.col("date") <= end_dt)
        )
        if window.is_empty():
            return window

        price = pl.col("price_usd").cast(pl.Float64, strict=False)
        returns = price.pct_change()
        rolling_vol = returns.rolling_std(window_size=21, min_samples=10)
        rolling_max = price.rolling_max(window_size=90, min_samples=30)
        drawdown = (price / rolling_max) - 1.0
        vol21 = rolling_vol.shift(1).fill_null(0.0)
        drawdown90 = drawdown.shift(1).fill_null(0.0)

        if "plus_vol21" in window.columns:
            existing = pl.col("plus_vol21").cast(pl.Float64, strict=False)
            window = window.with_columns(pl.coalesce(existing, vol21).alias("plus_vol21"))
        else:
            window = window.with_columns(vol21.alias("plus_vol21"))
        if "plus_drawdown90" in window.columns:
            existing = pl.col("plus_drawdown90").cast(pl.Float64, strict=False)
            window = window.with_columns(pl.coalesce(existing, drawdown90).alias("plus_drawdown90"))
        else:
            window = window.with_columns(drawdown90.alias("plus_drawdown90"))
        return window.fill_null(0.0)

    def _safe_get(self, df: pl.DataFrame, col: str, fill: float = 0.0) -> pl.Series:
        if col in df.columns:
            return df[col].cast(pl.Float64, strict=False).fill_null(fill)
        return pl.Series([fill] * df.height)

    def build_target_profile(
        self,
        ctx: StrategyContext,
        features_df: pl.DataFrame,
        signals: dict[str, pl.Series],
    ) -> TargetProfile:
        del signals
        if features_df.is_empty():
            return TargetProfile(
                values=pl.DataFrame(schema={"date": pl.Datetime("us"), "value": pl.Float64}),
                mode="preference",
            )

        base_pref = compute_preference_scores(
            features_df=features_df,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
        )["preference"].to_numpy().astype(float)

        zone = self._safe_array(self._safe_get(features_df, "mvrv_zone"))
        mvrv_zscore = self._safe_array(self._safe_get(features_df, "mvrv_zscore"))
        price_vs_ma = self._safe_array(self._safe_get(features_df, "price_vs_ma"))
        confidence = self._safe_array(self._safe_get(features_df, "signal_confidence", 0.5), fill=0.5)
        confidence = np.where(confidence == 0.0, 0.5, confidence)
        confidence = np.clip(confidence, 0.0, 1.0)

        mvrv_volatility = self._safe_array(
            self._safe_get(features_df, "mvrv_volatility", 0.5), fill=0.5
        )
        mvrv_volatility = np.where(mvrv_volatility == 0.0, 0.5, mvrv_volatility)
        mvrv_volatility = np.clip(mvrv_volatility, 0.0, 1.0)

        vol21 = self._safe_array(self._safe_get(features_df, "plus_vol21"))
        drawdown90 = self._safe_array(self._safe_get(features_df, "plus_drawdown90"))

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

        netflow = self._safe_array(self._safe_get(features_df, "brk_netflow"))
        exchange_share = self._safe_array(self._safe_get(features_df, "brk_exchange_share"))
        exchange_share_delta = self._safe_array(self._safe_get(features_df, "brk_exchange_share_delta"))
        activity_div = self._safe_array(self._safe_get(features_df, "brk_activity_div"))
        roi_context = self._safe_array(self._safe_get(features_df, "brk_roi_context"))
        liquidity_impulse = self._safe_array(self._safe_get(features_df, "brk_liquidity_impulse"))
        miner_pressure = self._safe_array(self._safe_get(features_df, "brk_miner_pressure"))
        hash_momentum = self._safe_array(self._safe_get(features_df, "brk_hash_momentum"))

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

        date_col = features_df["date"] if "date" in features_df.columns else pl.Series(
            "date", pl.datetime_range(_to_dt(ctx.start_date), _to_dt(ctx.end_date), interval="1d", eager=True).to_list()
        )
        values_df = pl.DataFrame({
            "date": date_col,
            "value": pl.Series(preference.astype(float)),
        })
        return TargetProfile(values=values_df, mode="preference")


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

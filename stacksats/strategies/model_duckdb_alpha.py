"""DuckDB factor strategy built on top of the MVRV-plus baseline."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..strategy_types import BacktestConfig, TargetProfile, ValidationConfig
from .model_mvrv_plus import MVRVPlusStrategy


_ARTIFACT_PATH = Path(__file__).with_name("duckdb_alpha_v1.json")


@lru_cache(maxsize=4)
def _load_artifact(path_str: str) -> dict[str, object]:
    path = Path(path_str)
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_columns = payload.get("feature_columns")
    coefficients = payload.get("coefficients")
    scaler = payload.get("scaler")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("DuckDB alpha artifact must provide non-empty feature_columns.")
    if not isinstance(coefficients, dict):
        raise ValueError("DuckDB alpha artifact must provide coefficients.")
    if not isinstance(scaler, dict):
        raise ValueError("DuckDB alpha artifact must provide scaler metadata.")
    return payload


class DuckDBAlphaStrategy(MVRVPlusStrategy):
    """MVRV-plus baseline with frozen DuckDB factor alpha overlay."""

    strategy_id = "duckdb-alpha"
    version = "0.1.0"
    description = "MVRV-plus baseline with frozen DuckDB analytics factor overlay."

    duckdb_alpha_scale: float = 0.04256125797822773
    duckdb_zone_boost: float = 0.15160690277397185
    duckdb_high_vol_penalty: float = 0.19690331633535432
    duckdb_smooth_alpha: float = 0.004376307975407839
    _MVRV_PLUS_TUNED_DEFAULTS: dict[str, float] = {
        "deep_value_boost": 1.4186652720381463,
        "overheat_dampen": 0.43988521932454583,
        "high_vol_penalty": 0.1603325577792499,
        "drawdown_penalty": 0.43209367535941134,
        "disagreement_penalty": 0.41101968369197717,
        "confidence_boost": 0.24713690391856086,
        "overlay_scale": 0.21571523730102354,
        "smooth_alpha": 0.23692852199162623,
        "risk_budget_base": 0.9371657710768919,
        "risk_budget_min": 0.9211554985641621,
        "risk_budget_max": 1.1222844680448187,
        "miner_pressure_weight": 0.08329252692036648,
        "hash_momentum_weight": 0.09759095582004602,
    }

    def __init__(
        self,
        *,
        artifact_path: str | None = None,
        duckdb_alpha_scale: float = 0.04256125797822773,
        duckdb_zone_boost: float = 0.15160690277397185,
        duckdb_high_vol_penalty: float = 0.19690331633535432,
        duckdb_smooth_alpha: float = 0.004376307975407839,
        **kwargs,
    ) -> None:
        for key, value in self._MVRV_PLUS_TUNED_DEFAULTS.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)
        self.duckdb_alpha_scale = duckdb_alpha_scale
        self.duckdb_zone_boost = duckdb_zone_boost
        self.duckdb_high_vol_penalty = duckdb_high_vol_penalty
        self.duckdb_smooth_alpha = duckdb_smooth_alpha
        self._artifact_path = str(Path(artifact_path) if artifact_path else _ARTIFACT_PATH)
        self._artifact = _load_artifact(self._artifact_path)

    @property
    def artifact_feature_columns(self) -> tuple[str, ...]:
        columns = self._artifact.get("feature_columns", [])
        if not isinstance(columns, list):
            return ()
        return tuple(str(column) for column in columns)

    def required_feature_sets(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*super().required_feature_sets(), "duckdb_analytics_factors_v1")
            )
        )

    def required_feature_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*super().required_feature_columns(), *self.artifact_feature_columns)
            )
        )

    @staticmethod
    def _series_values(
        features_df: pd.DataFrame,
        column: str,
    ) -> np.ndarray:
        return (
            pd.to_numeric(
                features_df.get(column, pd.Series(0.0, index=features_df.index)),
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    def _duckdb_alpha_component(self, features_df: pd.DataFrame) -> np.ndarray:
        intercept = float(self._artifact.get("intercept", 0.0))
        coefficients = self._artifact.get("coefficients", {})
        scaler = self._artifact.get("scaler", {})
        means = scaler.get("mean", {}) if isinstance(scaler, dict) else {}
        stds = scaler.get("std", {}) if isinstance(scaler, dict) else {}

        alpha = np.full(len(features_df.index), intercept, dtype=float)
        for column in self.artifact_feature_columns:
            coef = float(coefficients.get(column, 0.0)) if isinstance(coefficients, dict) else 0.0
            mean = float(means.get(column, 0.0)) if isinstance(means, dict) else 0.0
            std = float(stds.get(column, 1.0)) if isinstance(stds, dict) else 1.0
            std = max(abs(std), 1e-8)
            values = self._series_values(features_df, column)
            alpha += coef * ((values - mean) / std)
        return np.tanh(alpha)

    def build_target_profile(
        self,
        ctx,
        features_df: pd.DataFrame,
        signals,
    ) -> TargetProfile:
        base_profile = super().build_target_profile(ctx, features_df, signals)
        if features_df.empty:
            return base_profile

        alpha = self._duckdb_alpha_component(features_df)
        zone = self._series_values(features_df, "mvrv_zone")
        volatility = self._series_values(features_df, "mvrv_volatility")
        confidence = self._series_values(features_df, "signal_confidence")
        confidence = np.where(confidence == 0.0, 0.5, confidence)
        confidence = np.clip(confidence, 0.0, 1.0)

        high_vol = np.clip((volatility - 0.65) / 0.35, 0.0, 1.0)
        zone_scale = np.where(
            zone <= -1.0,
            1.0 + self.duckdb_zone_boost,
            np.where(zone >= 1.0, 1.0 - self.duckdb_zone_boost, 1.0),
        )
        vol_scale = 1.0 - (self.duckdb_high_vol_penalty * high_vol)
        confidence_scale = 0.80 + (0.40 * confidence)
        alpha_scaled = self.duckdb_alpha_scale * alpha * zone_scale * vol_scale * confidence_scale

        base_values = base_profile.values.to_numpy(dtype=float)
        if self.duckdb_smooth_alpha > 0.0:
            smooth_alpha = np.full(len(alpha_scaled), self.duckdb_smooth_alpha, dtype=float)
            alpha_scaled = self._adaptive_ewma(alpha_scaled, smooth_alpha)
        preference = np.clip(base_values + alpha_scaled, -50.0, 50.0)
        preference = np.where(np.isfinite(preference), preference, 0.0)

        return TargetProfile(
            values=pd.Series(preference, index=features_df.index, dtype=float),
            mode="preference",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DuckDB alpha strategy.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--strategy-label", type=str, default="duckdb-alpha")
    args = parser.parse_args()

    strategy = DuckDBAlphaStrategy()
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
    output_root = Path(args.output_dir) / result.strategy_id / result.strategy_version / result.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    result.plot(output_dir=str(output_root))
    result.to_json(output_root / "backtest_result.json")
    print(f"Saved: {output_root}")


if __name__ == "__main__":
    main()

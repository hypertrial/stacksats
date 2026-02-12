"""Example custom strategy using the StackSats package API.

This file is the intended template for user model development.
Users only implement `compute_weights()` and then call package APIs for:
- validation (`validate_strategy`)
- backtesting (`run_backtest`)
- plotting/JSON export (`BacktestResult` helpers)

Available `features_df` columns typically include:
- PriceUSD_coinmetrics
- price_ma
- price_vs_ma
- mvrv_zscore
- mvrv_gradient
- mvrv_percentile
- mvrv_acceleration
- mvrv_zone
- mvrv_volatility
- signal_confidence
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from stacksats import run_backtest, validate_strategy


class ExampleMVRVStrategy:
    """Non-trivial strategy template using package-precomputed features."""

    def compute_weights(
        self,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        del current_date

        window = features_df.loc[start_date:end_date]
        if window.empty:
            full_index = pd.date_range(start=start_date, end=end_date, freq="D")
            return pd.Series(np.full(len(full_index), 1.0 / len(full_index)), index=full_index)

        z = (
            window.get("mvrv_zscore", pd.Series(0.0, index=window.index))
            .fillna(0.0)
            .to_numpy()
        )
        ma = (
            window.get("price_vs_ma", pd.Series(0.0, index=window.index))
            .fillna(0.0)
            .to_numpy()
        )
        pct = (
            window.get("mvrv_percentile", pd.Series(0.5, index=window.index))
            .fillna(0.5)
            .to_numpy()
        )

        # Lower MVRV and below-MA prices get more allocation; percentile adds cycle context.
        signal = (-1.25 * z) + (-0.75 * ma) + (0.6 * (0.5 - pct))
        signal = np.clip(signal, -8, 8)

        raw = np.exp(signal - signal.max())
        raw_sum = raw.sum()
        if not np.isfinite(raw_sum) or raw_sum <= 0:
            return pd.Series(np.full(len(window), 1.0 / len(window)), index=window.index)

        weights = raw / raw_sum
        return pd.Series(weights, index=window.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StackSats example strategy.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--strategy-label",
        type=str,
        default="example-mvrv-strategy",
        help="Label used in backtest reporting",
    )
    args = parser.parse_args()

    strategy = ExampleMVRVStrategy()

    validation = validate_strategy(
        strategy,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(validation.summary())
    for message in validation.messages:
        print(f"- {message}")

    result = run_backtest(
        strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_label=args.strategy_label,
    )
    print(result.summary())
    result.plot(output_dir=args.output_dir)
    result.to_json(f"{args.output_dir}/backtest_result.json")


if __name__ == "__main__":
    main()

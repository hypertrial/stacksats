"""Diagnostic methods for StrategyTimeSeries."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class StrategyTimeSeriesDiagnosticsMixin:
    """Diagnostics and profiling helpers for normalized strategy windows."""

    def profile(self) -> dict[str, Any]:
        """Return an EDA summary of the normalized payload."""
        rows = self._data.height
        columns = list(self._data.columns)
        date_series = self._data["date"]

        column_profiles: dict[str, dict[str, Any]] = {}
        numeric_columns: list[str] = []
        for column in columns:
            raw = self._data[column]
            null_count = raw.null_count()
            profile: dict[str, Any] = {
                "dtype": str(raw.dtype),
                "null_count": null_count,
                "null_fraction": self._native_float(null_count / rows) if rows > 0 else 0.0,
                "non_null_count": int(rows - null_count),
                "unique_non_null": int(raw.n_unique()) if null_count < rows else 0,
            }
            if raw.dtype in (pl.Float64, pl.Int64):
                numeric_columns.append(column)
                numeric = raw.cast(pl.Float64, strict=False)
                profile["numeric_summary"] = self._series_numeric_summary(numeric)
            elif raw.dtype == pl.Datetime or column == "date":
                as_dt = raw.cast(pl.Datetime)
                non_null = as_dt.drop_nulls()
                if non_null.len() > 0:
                    mn, mx = non_null.min(), non_null.max()
                    profile["datetime_min"] = self._native_timestamp(mn)
                    profile["datetime_max"] = self._native_timestamp(mx)
                else:
                    profile["datetime_min"] = None
                    profile["datetime_max"] = None
            column_profiles[column] = profile

        return {
            "row_count": rows,
            "column_count": self._data.width,
            "columns": columns,
            "numeric_columns": numeric_columns,
            "date_start": (
                self._native_timestamp(date_series.min())
                if rows > 0 and date_series.null_count() < rows
                else None
            ),
            "date_end": (
                self._native_timestamp(date_series.max())
                if rows > 0 and date_series.null_count() < rows
                else None
            ),
            "columns_profile": column_profiles,
        }

    def weight_diagnostics(self, top_k: int = 5) -> dict[str, Any]:
        """Return concentration and dispersion diagnostics for `weight`."""
        weights = self._data["weight"].cast(pl.Float64, strict=False)
        non_null = weights.filter(weights.is_finite())
        sample_size = int(non_null.len())
        if sample_size == 0:
            return {
                "sample_size": 0,
                "sum": None,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
                "p10": None,
                "p90": None,
                "hhi": None,
                "effective_n": None,
                "entropy_nats": None,
                "top_weights": [],
            }

        arr = non_null.to_numpy()
        hhi = float(np.sum(np.square(arr)))
        positive = arr[arr > 0]
        entropy = float(-np.sum(positive * np.log(positive))) if positive.size > 0 else 0.0

        top_count = max(int(top_k), 0)
        top_df = (
            self._data.with_columns(weights.alias("_weight"))
            .sort("_weight", descending=True)
            .head(top_count)
            .select(["date", "_weight"])
        )
        top_weights = [
            {
                "date": self._native_timestamp(row["date"]),
                "weight": self._native_float(row["_weight"]),
            }
            for row in top_df.iter_rows(named=True)
        ]

        return {
            "sample_size": sample_size,
            "sum": self._native_float(float(non_null.sum())),
            "mean": self._native_float(float(non_null.mean())),
            "std": self._native_float(float(non_null.std())),
            "min": self._native_float(float(non_null.min())),
            "max": self._native_float(float(non_null.max())),
            "median": self._native_float(float(non_null.median())),
            "p10": self._native_float(float(non_null.quantile(0.10))),
            "p90": self._native_float(float(non_null.quantile(0.90))),
            "hhi": self._native_float(hhi),
            "effective_n": self._native_float((1.0 / hhi) if hhi > 0 else None),
            "entropy_nats": self._native_float(entropy),
            "top_weights": top_weights,
        }

    def returns_diagnostics(self) -> dict[str, Any]:
        """Return basic return/risk diagnostics derived from `price_usd`."""
        prices = self._data["price_usd"].cast(pl.Float64, strict=False)
        p_arr = prices.to_numpy()
        prev_arr = np.roll(p_arr, 1)
        prev_arr[0] = np.nan

        valid_step = np.isfinite(p_arr) & np.isfinite(prev_arr) & (prev_arr != 0)
        simple_returns = np.full(len(p_arr), np.nan)
        simple_returns[valid_step] = (p_arr[valid_step] / prev_arr[valid_step]) - 1.0

        positive_step = valid_step & (p_arr > 0) & (prev_arr > 0)
        log_returns = np.full(len(p_arr), np.nan)
        log_returns[positive_step] = np.log(p_arr[positive_step] / prev_arr[positive_step])

        valid_simple = simple_returns[np.isfinite(simple_returns)]
        valid_log = log_returns[np.isfinite(log_returns)]
        price_valid_mask = np.isfinite(p_arr)
        price_valid = p_arr[price_valid_mask]

        if len(price_valid) == 0:
            return {
                "price_observations": 0,
                "return_observations": 0,
                "periods": self._data.height,
                "cumulative_return": None,
                "mean_simple_return": None,
                "std_simple_return": None,
                "annualized_volatility": None,
                "mean_log_return": None,
                "std_log_return": None,
                "max_drawdown": None,
                "max_drawdown_date": None,
                "best_day_return": None,
                "best_day_date": None,
                "worst_day_return": None,
                "worst_day_date": None,
            }

        running_max = np.maximum.accumulate(price_valid)
        drawdown = (p_arr / running_max) - 1.0
        dd_idx = int(np.argmin(drawdown))

        cumulative_return = (
            self._native_float(float(np.prod(1.0 + valid_simple) - 1.0))
            if len(valid_simple) > 0
            else None
        )
        annualized_vol = (
            self._native_float(float(np.std(valid_simple) * np.sqrt(365.0)))
            if len(valid_simple) >= 2
            else None
        )

        best_idx = int(np.argmax(valid_simple)) if len(valid_simple) > 0 else None
        worst_idx = int(np.argmin(valid_simple)) if len(valid_simple) > 0 else None

        price_valid_indices = np.where(price_valid_mask)[0]
        dd_row_idx = price_valid_indices[dd_idx] if dd_idx < len(price_valid_indices) else 0

        simple_valid_indices = np.where(valid_step)[0]
        best_row_idx = simple_valid_indices[best_idx] if best_idx is not None and best_idx < len(simple_valid_indices) else None
        worst_row_idx = simple_valid_indices[worst_idx] if worst_idx is not None and worst_idx < len(simple_valid_indices) else None

        return {
            "price_observations": int(len(price_valid)),
            "return_observations": int(len(valid_simple)),
            "periods": self._data.height,
            "cumulative_return": cumulative_return,
            "mean_simple_return": (
                self._native_float(float(np.mean(valid_simple))) if len(valid_simple) > 0 else None
            ),
            "std_simple_return": (
                self._native_float(float(np.std(valid_simple)))
                if len(valid_simple) >= 2
                else None
            ),
            "annualized_volatility": annualized_vol,
            "mean_log_return": self._native_float(float(np.mean(valid_log))) if len(valid_log) > 0 else None,
            "std_log_return": (
                self._native_float(float(np.std(valid_log))) if len(valid_log) >= 2 else None
            ),
            "max_drawdown": self._native_float(float(drawdown.min())),
            "max_drawdown_date": self._native_timestamp(self._data["date"][int(dd_row_idx)]),
            "best_day_return": (
                self._native_float(float(valid_simple[best_idx])) if best_idx is not None and len(valid_simple) > 0 else None
            ),
            "best_day_date": (
                self._native_timestamp(self._data["date"][int(best_row_idx)])
                if best_row_idx is not None
                else None
            ),
            "worst_day_return": (
                self._native_float(float(valid_simple[worst_idx])) if worst_idx is not None and len(valid_simple) > 0 else None
            ),
            "worst_day_date": (
                self._native_timestamp(self._data["date"][int(worst_row_idx)])
                if worst_row_idx is not None
                else None
            ),
        }

    def outlier_report(
        self,
        columns: list[str] | None = None,
        *,
        method: str = "mad",
        threshold: float | None = None,
    ) -> pl.DataFrame:
        """Detect numeric outliers and return a tidy report."""
        outlier_method = method.lower()
        if outlier_method not in {"mad", "zscore", "iqr"}:
            raise ValueError("method must be one of: mad, zscore, iqr")

        effective_threshold = threshold
        if effective_threshold is None:
            effective_threshold = 1.5 if outlier_method == "iqr" else 3.5
        if effective_threshold <= 0:
            raise ValueError("threshold must be > 0")

        numeric_columns = [
            col
            for col in self._data.columns
            if self._data[col].dtype in (pl.Float64, pl.Int64) and col != "day_index"
        ]
        selected_columns = columns if columns is not None else numeric_columns
        unknown = [col for col in selected_columns if col not in self._data.columns]
        if unknown:
            raise ValueError("Unknown columns for outlier detection: " + ", ".join(unknown))

        rows: list[dict[str, Any]] = []
        for column in selected_columns:
            numeric = self._data[column].cast(pl.Float64, strict=False)
            valid = numeric.drop_nulls()
            if valid.len() < 2:
                continue

            if outlier_method == "mad":
                median = float(valid.median())
                mad = float((valid - median).abs().median())
                if mad == 0:
                    continue
                scores = pl.Series("s", (0.6745 * (numeric.to_numpy() - median) / mad))
                mask = np.abs(scores.to_numpy()) > effective_threshold
            elif outlier_method == "zscore":
                mean = float(valid.mean())
                std = float(valid.std())
                if std == 0:
                    continue
                scores = pl.Series("s", (numeric.to_numpy() - mean) / std)
                mask = np.abs(scores.to_numpy()) > effective_threshold
            else:
                q1 = float(valid.quantile(0.25))
                q3 = float(valid.quantile(0.75))
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - (effective_threshold * iqr)
                upper = q3 + (effective_threshold * iqr)
                num_arr = numeric.to_numpy()
                below = num_arr < lower
                above = num_arr > upper
                scores_arr = np.zeros(len(num_arr))
                scores_arr[below] = (num_arr[below] - lower) / iqr
                scores_arr[above] = (num_arr[above] - upper) / iqr
                scores = pl.Series("scores", scores_arr)
                mask = below | above

            mask_arr = mask if isinstance(mask, np.ndarray) else np.asarray(mask)
            mask_arr = np.where(np.isnan(mask_arr), False, mask_arr)
            flagged_indices = np.where(mask_arr)[0]
            scores_arr = scores.to_numpy()
            for idx in flagged_indices:
                idx_int = int(idx)
                sc = scores_arr[idx_int] if idx_int < len(scores_arr) else None
                rows.append(
                    {
                        "date": self._data["date"][idx_int],
                        "column": str(column),
                        "value": self._native_float(numeric[idx_int]),
                        "score": self._native_float(sc),
                        "method": outlier_method,
                        "threshold": float(effective_threshold),
                    }
                )

        if not rows:
            return pl.DataFrame(schema={"date": pl.Datetime, "column": pl.Utf8, "value": pl.Float64, "score": pl.Float64, "method": pl.Utf8, "threshold": pl.Float64})
        result = pl.DataFrame(rows)
        return result.sort(["column", "date"])

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

        stats = pl.DataFrame({"weight": non_null}).select(
            (pl.col("weight") ** 2).sum().alias("hhi"),
            (
                -pl.when(pl.col("weight") > 0.0)
                .then(pl.col("weight") * pl.col("weight").log())
                .otherwise(0.0)
                .sum()
            ).alias("entropy"),
        ).row(0, named=True)
        hhi = float(stats["hhi"] or 0.0)
        entropy = float(stats["entropy"] or 0.0)

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
            for row in top_df.to_dicts()
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
        price_frame = self._data.select(
            pl.col("date"),
            pl.col("price_usd").cast(pl.Float64, strict=False).alias("price_usd"),
        )
        valid_prices = price_frame.filter(pl.col("price_usd").is_finite())
        if valid_prices.is_empty():
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
        returns_frame = valid_prices.with_columns(
            pl.col("price_usd").shift(1).alias("_prev_price"),
        ).with_columns(
            pl.when(pl.col("_prev_price").is_finite() & (pl.col("_prev_price") != 0.0))
            .then((pl.col("price_usd") / pl.col("_prev_price")) - 1.0)
            .otherwise(None)
            .alias("simple_return"),
            pl.when(
                pl.col("_prev_price").is_finite()
                & (pl.col("_prev_price") > 0.0)
                & (pl.col("price_usd") > 0.0)
            )
            .then((pl.col("price_usd") / pl.col("_prev_price")).log())
            .otherwise(None)
            .alias("log_return"),
        )
        drawdown_frame = valid_prices.with_columns(
            pl.col("price_usd").cum_max().alias("_running_max"),
        ).with_columns(
            ((pl.col("price_usd") / pl.col("_running_max")) - 1.0).alias("drawdown")
        )
        valid_simple = returns_frame.filter(pl.col("simple_return").is_finite())
        valid_log = returns_frame.filter(pl.col("log_return").is_finite())
        dd_row = drawdown_frame.sort("drawdown").row(0, named=True)
        best_row = (
            valid_simple.sort("simple_return", descending=True).row(0, named=True)
            if not valid_simple.is_empty()
            else None
        )
        worst_row = (
            valid_simple.sort("simple_return").row(0, named=True)
            if not valid_simple.is_empty()
            else None
        )
        cumulative_return = None
        if not valid_simple.is_empty():
            cumulative_return = self._native_float(
                float(
                    valid_simple.select(
                        ((pl.col("simple_return") + 1.0).product() - 1.0).alias("cumulative")
                    ).item()
                )
            )
        annualized_vol = (
            self._native_float(float(valid_simple["simple_return"].std() * np.sqrt(365.0)))
            if valid_simple.height >= 2
            else None
        )

        return {
            "price_observations": int(valid_prices.height),
            "return_observations": int(valid_simple.height),
            "periods": self._data.height,
            "cumulative_return": cumulative_return,
            "mean_simple_return": (
                self._native_float(float(valid_simple["simple_return"].mean()))
                if not valid_simple.is_empty()
                else None
            ),
            "std_simple_return": (
                self._native_float(float(valid_simple["simple_return"].std()))
                if valid_simple.height >= 2
                else None
            ),
            "annualized_volatility": annualized_vol,
            "mean_log_return": (
                self._native_float(float(valid_log["log_return"].mean()))
                if not valid_log.is_empty()
                else None
            ),
            "std_log_return": (
                self._native_float(float(valid_log["log_return"].std()))
                if valid_log.height >= 2
                else None
            ),
            "max_drawdown": self._native_float(float(dd_row["drawdown"])),
            "max_drawdown_date": self._native_timestamp(dd_row["date"]),
            "best_day_return": (
                self._native_float(float(best_row["simple_return"])) if best_row is not None else None
            ),
            "best_day_date": (
                self._native_timestamp(best_row["date"]) if best_row is not None else None
            ),
            "worst_day_return": (
                self._native_float(float(worst_row["simple_return"])) if worst_row is not None else None
            ),
            "worst_day_date": (
                self._native_timestamp(worst_row["date"]) if worst_row is not None else None
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

        reports: list[pl.DataFrame] = []
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
                column_report = self._data.select(
                    "date",
                    numeric.alias("value"),
                ).with_columns(
                    ((pl.col("value") - median) * (0.6745 / mad)).alias("score"),
                ).filter(pl.col("score").abs() > effective_threshold)
            elif outlier_method == "zscore":
                mean = float(valid.mean())
                std = float(valid.std())
                if std == 0:
                    continue
                column_report = self._data.select(
                    "date",
                    numeric.alias("value"),
                ).with_columns(
                    ((pl.col("value") - mean) / std).alias("score"),
                ).filter(pl.col("score").abs() > effective_threshold)
            else:
                q1 = float(valid.quantile(0.25))
                q3 = float(valid.quantile(0.75))
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - (effective_threshold * iqr)
                upper = q3 + (effective_threshold * iqr)
                column_report = self._data.select(
                    "date",
                    numeric.alias("value"),
                ).with_columns(
                    pl.when(pl.col("value") < lower)
                    .then((pl.col("value") - lower) / iqr)
                    .when(pl.col("value") > upper)
                    .then((pl.col("value") - upper) / iqr)
                    .otherwise(None)
                    .alias("score"),
                ).filter(pl.col("score").is_not_null())

            if column_report.is_empty():
                continue
            reports.append(
                column_report.with_columns(
                    pl.lit(str(column)).alias("column"),
                    pl.lit(outlier_method).alias("method"),
                    pl.lit(float(effective_threshold)).alias("threshold"),
                ).select(["date", "column", "value", "score", "method", "threshold"])
            )

        if not reports:
            return pl.DataFrame(schema={"date": pl.Datetime, "column": pl.Utf8, "value": pl.Float64, "score": pl.Float64, "method": pl.Utf8, "threshold": pl.Float64})
        result = pl.concat(reports, how="vertical_relaxed")
        return result.sort(["column", "date"])

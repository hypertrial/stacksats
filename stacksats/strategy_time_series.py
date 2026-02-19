"""Typed time-series output objects for strategy export runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterable

import numpy as np
import pandas as pd

from .strategy_time_series_metadata import StrategySeriesMetadata
from .strategy_time_series_batch import StrategyTimeSeriesBatch
from .strategy_time_series_schema import (
    COINMETRICS_BTC_CSV_COLUMNS as _DEFAULT_COINMETRICS_BTC_CSV_COLUMNS,
    COINMETRICS_LINEAGE as _DEFAULT_COINMETRICS_LINEAGE,
    CoinMetricsLineageSpec,
    ColumnSpec,
    coinmetrics_lineage_markdown as render_coinmetrics_lineage_markdown,
    render_schema_markdown,
    schema_dict as build_schema_dict,
    schema_specs as build_schema_specs,
    validate_coinmetrics_lineage_coverage as validate_lineage_coverage,
)


@dataclass(frozen=True, slots=True)
class StrategyTimeSeries:
    """Single-window normalized strategy output time series."""

    metadata: StrategySeriesMetadata
    data: pd.DataFrame

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = ("date", "weight", "price_usd")
    COINMETRICS_BTC_CSV_COLUMNS: ClassVar[tuple[str, ...]] = _DEFAULT_COINMETRICS_BTC_CSV_COLUMNS
    COINMETRICS_LINEAGE: ClassVar[tuple[CoinMetricsLineageSpec, ...]] = _DEFAULT_COINMETRICS_LINEAGE

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("StrategyTimeSeries.data must be a pandas DataFrame.")

        normalized = self.data.copy(deep=True)
        if "date" in normalized.columns:
            normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
            normalized = normalized.sort_values("date").reset_index(drop=True)
        if "weight" in normalized.columns:
            normalized["weight"] = pd.to_numeric(normalized["weight"], errors="coerce")
        if "price_usd" in normalized.columns:
            normalized["price_usd"] = pd.to_numeric(normalized["price_usd"], errors="coerce")
        if "day_index" in normalized.columns:
            normalized["day_index"] = pd.to_numeric(normalized["day_index"], errors="coerce")

        object.__setattr__(self, "data", normalized)
        self.validate()

    @staticmethod
    def _schema_specs() -> tuple[ColumnSpec, ...]:
        return build_schema_specs()

    def schema(self) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return self.schema_dict()

    @classmethod
    def schema_dict(cls) -> dict[str, ColumnSpec]:
        """Return handwritten column schema specs keyed by column name."""
        return build_schema_dict(cls._schema_specs())

    @classmethod
    def validate_coinmetrics_lineage_coverage(cls) -> None:
        """Ensure lineage mappings target documented StrategyTimeSeries columns."""
        validate_lineage_coverage(
            lineage=cls.COINMETRICS_LINEAGE,
            schema_specs_iter=cls._schema_specs(),
            source_columns=cls.COINMETRICS_BTC_CSV_COLUMNS,
        )

    @classmethod
    def coinmetrics_lineage_markdown(cls) -> str:
        """Render CoinMetrics source-to-schema lineage as a markdown table."""
        cls.validate_coinmetrics_lineage_coverage()
        return render_coinmetrics_lineage_markdown(cls.COINMETRICS_LINEAGE)

    @classmethod
    def schema_markdown_table(cls) -> str:
        """Render StrategyTimeSeries schema specs as a markdown table."""
        cls.validate_coinmetrics_lineage_coverage()
        return cls._render_schema_markdown(cls._schema_specs())

    @staticmethod
    def _render_schema_markdown(specs: Iterable[ColumnSpec]) -> str:
        """Render schema specs as a markdown table."""
        return render_schema_markdown(specs)

    def schema_markdown(self) -> str:
        """Render schema specs as a markdown table."""
        return self._render_schema_markdown(self.schema().values())

    def validate_schema_coverage(self) -> None:
        """Ensure each column has an explicit handwritten schema entry."""
        covered = set(self.schema().keys())
        unknown = [col for col in self.data.columns if col not in covered]
        if unknown:
            raise ValueError(
                "Schema coverage missing for columns: " + ", ".join(str(col) for col in unknown)
            )

    def validate(self) -> None:
        """Validate data and metadata invariants."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing:
            raise ValueError(
                "StrategyTimeSeries missing required columns: "
                + ", ".join(str(col) for col in missing)
            )

        self.validate_schema_coverage()

        dates = pd.to_datetime(self.data["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Column 'date' must contain valid datetimes.")
        if dates.duplicated().any():
            raise ValueError("Column 'date' must not contain duplicates.")
        if not dates.is_monotonic_increasing:
            raise ValueError("Column 'date' must be sorted ascending.")

        if self.metadata.window_start is not None and len(dates) > 0:
            start = pd.Timestamp(self.metadata.window_start)
            if pd.Timestamp(dates.iloc[0]) != start:
                raise ValueError(
                    "Series start date does not match metadata.window_start: "
                    f"{dates.iloc[0]!s} != {start!s}"
                )
        if self.metadata.window_end is not None and len(dates) > 0:
            end = pd.Timestamp(self.metadata.window_end)
            if pd.Timestamp(dates.iloc[-1]) != end:
                raise ValueError(
                    "Series end date does not match metadata.window_end: "
                    f"{dates.iloc[-1]!s} != {end!s}"
                )

        weights = pd.to_numeric(self.data["weight"], errors="coerce")
        if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
            raise ValueError("Column 'weight' must contain finite numeric values.")
        if bool((weights < 0).any()):
            raise ValueError("Column 'weight' must not contain negative values.")
        if len(weights) > 0:
            weight_sum = float(weights.sum())
            if not np.isclose(weight_sum, 1.0, rtol=1e-5, atol=1e-8):
                raise ValueError("Column 'weight' must sum to 1.0 " f"(got {weight_sum:.10f}).")

        raw_price = self.data["price_usd"]
        prices = pd.to_numeric(raw_price, errors="coerce")
        invalid_non_null = raw_price.notna() & prices.isna()
        if invalid_non_null.any():
            raise ValueError("Column 'price_usd' must be numeric when present.")
        finite_mask = prices.notna()
        if finite_mask.any() and not np.isfinite(prices.loc[finite_mask].to_numpy(dtype=float)).all():
            raise ValueError("Column 'price_usd' must be finite when present.")

        if "locked" in self.data.columns:
            locked = self.data["locked"]
            valid_locked = locked.isin([True, False])
            if not bool(valid_locked.all()):
                raise ValueError("Column 'locked' must contain only boolean values.")

        if "day_index" in self.data.columns:
            day_index = pd.to_numeric(self.data["day_index"], errors="coerce")
            if day_index.isna().any():
                raise ValueError("Column 'day_index' must contain integer values.")
            if bool((day_index < 0).any()):
                raise ValueError("Column 'day_index' must be >= 0.")
            if len(day_index) > 0:
                expected = np.arange(len(day_index), dtype=float)
                if not np.array_equal(day_index.to_numpy(dtype=float), expected):
                    raise ValueError("Column 'day_index' must be contiguous starting at 0.")

        for spec in self._schema_specs():
            if spec.source != "coinmetrics":
                continue
            if spec.name not in self.data.columns:
                continue
            if spec.dtype in {"int64", "float64"}:
                self._validate_optional_numeric_column(spec.name)
            elif spec.dtype.startswith("datetime64"):
                self._validate_optional_datetime_column(spec.name)

    def _validate_optional_numeric_column(self, column: str) -> None:
        """Validate optional CoinMetrics passthrough numeric columns."""
        raw = self.data[column]
        values = pd.to_numeric(raw, errors="coerce")
        invalid_non_null = raw.notna() & values.isna()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be numeric when present.")
        finite_mask = values.notna()
        if finite_mask.any() and not np.isfinite(values.loc[finite_mask].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{column}' must be finite when present.")

    def _validate_optional_datetime_column(self, column: str) -> None:
        """Validate optional CoinMetrics passthrough datetime columns."""
        raw = self.data[column]
        values = pd.to_datetime(raw, errors="coerce")
        invalid_non_null = raw.notna() & values.isna()
        if invalid_non_null.any():
            raise ValueError(f"Column '{column}' must be datetime when present.")

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the normalized dataframe payload."""
        return self.data.copy(deep=True)

    @staticmethod
    def _native_float(value: float | np.floating[Any] | int | None) -> float | None:
        if value is None:
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out

    @staticmethod
    def _native_timestamp(value: pd.Timestamp | None) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()

    @staticmethod
    def _series_numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
        non_null = values.dropna()
        if non_null.empty:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
            }
        return {
            "count": int(non_null.shape[0]),
            "mean": StrategyTimeSeries._native_float(non_null.mean()),
            "std": StrategyTimeSeries._native_float(non_null.std(ddof=0)),
            "min": StrategyTimeSeries._native_float(non_null.min()),
            "p25": StrategyTimeSeries._native_float(non_null.quantile(0.25)),
            "median": StrategyTimeSeries._native_float(non_null.median()),
            "p75": StrategyTimeSeries._native_float(non_null.quantile(0.75)),
            "max": StrategyTimeSeries._native_float(non_null.max()),
        }

    def profile(self) -> dict[str, Any]:
        """Return an EDA summary of the normalized payload."""
        rows = int(self.data.shape[0])
        columns = [str(col) for col in self.data.columns]
        date_series = pd.to_datetime(self.data["date"], errors="coerce")

        column_profiles: dict[str, dict[str, Any]] = {}
        numeric_columns: list[str] = []
        for column in columns:
            raw = self.data[column]
            null_count = int(raw.isna().sum())
            profile: dict[str, Any] = {
                "dtype": str(raw.dtype),
                "null_count": null_count,
                "null_fraction": self._native_float(null_count / rows) if rows > 0 else 0.0,
                "non_null_count": int(raw.notna().sum()),
                "unique_non_null": int(raw.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(raw):
                numeric_columns.append(column)
                numeric = pd.to_numeric(raw, errors="coerce")
                profile["numeric_summary"] = self._series_numeric_summary(numeric)
            elif pd.api.types.is_datetime64_any_dtype(raw) or column == "date":
                as_dt = pd.to_datetime(raw, errors="coerce")
                non_null_dt = as_dt.dropna()
                profile["datetime_min"] = (
                    self._native_timestamp(pd.Timestamp(non_null_dt.min()))
                    if not non_null_dt.empty
                    else None
                )
                profile["datetime_max"] = (
                    self._native_timestamp(pd.Timestamp(non_null_dt.max()))
                    if not non_null_dt.empty
                    else None
                )
            column_profiles[column] = profile

        return {
            "row_count": rows,
            "column_count": int(self.data.shape[1]),
            "columns": columns,
            "numeric_columns": numeric_columns,
            "date_start": (
                self._native_timestamp(pd.Timestamp(date_series.min()))
                if rows > 0 and date_series.notna().any()
                else None
            ),
            "date_end": (
                self._native_timestamp(pd.Timestamp(date_series.max()))
                if rows > 0 and date_series.notna().any()
                else None
            ),
            "columns_profile": column_profiles,
        }

    def weight_diagnostics(self, top_k: int = 5) -> dict[str, Any]:
        """Return concentration and dispersion diagnostics for `weight`."""
        weights = pd.to_numeric(self.data["weight"], errors="coerce")
        non_null = weights.dropna()
        sample_size = int(non_null.shape[0])
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

        arr = non_null.to_numpy(dtype=float)
        hhi = float(np.sum(np.square(arr)))
        positive = arr[arr > 0]
        entropy = float(-np.sum(positive * np.log(positive))) if positive.size > 0 else 0.0

        top_count = max(int(top_k), 0)
        top_rows = (
            self.data.assign(_weight=weights)
            .sort_values("_weight", ascending=False)
            .head(top_count)
            .loc[:, ["date", "_weight"]]
        )
        top_weights = [
            {
                "date": self._native_timestamp(pd.Timestamp(row["date"])),
                "weight": self._native_float(row["_weight"]),
            }
            for _, row in top_rows.iterrows()
        ]

        return {
            "sample_size": sample_size,
            "sum": self._native_float(non_null.sum()),
            "mean": self._native_float(non_null.mean()),
            "std": self._native_float(non_null.std(ddof=0)),
            "min": self._native_float(non_null.min()),
            "max": self._native_float(non_null.max()),
            "median": self._native_float(non_null.median()),
            "p10": self._native_float(non_null.quantile(0.10)),
            "p90": self._native_float(non_null.quantile(0.90)),
            "hhi": self._native_float(hhi),
            "effective_n": self._native_float((1.0 / hhi) if hhi > 0 else None),
            "entropy_nats": self._native_float(entropy),
            "top_weights": top_weights,
        }

    def returns_diagnostics(self) -> dict[str, Any]:
        """Return basic return/risk diagnostics derived from `price_usd`."""
        prices = pd.to_numeric(self.data["price_usd"], errors="coerce")
        prev = prices.shift(1)
        valid_step = prices.notna() & prev.notna() & (prev != 0)

        simple_returns = pd.Series(np.nan, index=prices.index, dtype=float)
        simple_returns.loc[valid_step] = (prices.loc[valid_step] / prev.loc[valid_step]) - 1.0

        positive_step = valid_step & (prices > 0) & (prev > 0)
        log_returns = pd.Series(np.nan, index=prices.index, dtype=float)
        log_returns.loc[positive_step] = np.log(prices.loc[positive_step] / prev.loc[positive_step])

        valid_simple = simple_returns.dropna()
        valid_log = log_returns.dropna()

        price_valid = prices.dropna()
        if price_valid.empty:
            return {
                "price_observations": 0,
                "return_observations": 0,
                "periods": int(self.data.shape[0]),
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

        running_max = price_valid.cummax()
        drawdown = (price_valid / running_max) - 1.0
        dd_idx = drawdown.idxmin()
        cumulative_return = (
            self._native_float((1.0 + valid_simple).prod() - 1.0)
            if not valid_simple.empty
            else None
        )
        annualized_vol = (
            self._native_float(valid_simple.std(ddof=1) * np.sqrt(365.0))
            if valid_simple.shape[0] >= 2
            else None
        )

        best_idx = valid_simple.idxmax() if not valid_simple.empty else None
        worst_idx = valid_simple.idxmin() if not valid_simple.empty else None

        return {
            "price_observations": int(price_valid.shape[0]),
            "return_observations": int(valid_simple.shape[0]),
            "periods": int(self.data.shape[0]),
            "cumulative_return": cumulative_return,
            "mean_simple_return": (
                self._native_float(valid_simple.mean()) if not valid_simple.empty else None
            ),
            "std_simple_return": (
                self._native_float(valid_simple.std(ddof=1))
                if valid_simple.shape[0] >= 2
                else None
            ),
            "annualized_volatility": annualized_vol,
            "mean_log_return": self._native_float(valid_log.mean()) if not valid_log.empty else None,
            "std_log_return": (
                self._native_float(valid_log.std(ddof=1)) if valid_log.shape[0] >= 2 else None
            ),
            "max_drawdown": self._native_float(drawdown.min()),
            "max_drawdown_date": self._native_timestamp(pd.Timestamp(self.data.loc[dd_idx, "date"])),
            "best_day_return": (
                self._native_float(valid_simple.loc[best_idx]) if best_idx is not None else None
            ),
            "best_day_date": (
                self._native_timestamp(pd.Timestamp(self.data.loc[best_idx, "date"]))
                if best_idx is not None
                else None
            ),
            "worst_day_return": (
                self._native_float(valid_simple.loc[worst_idx]) if worst_idx is not None else None
            ),
            "worst_day_date": (
                self._native_timestamp(pd.Timestamp(self.data.loc[worst_idx, "date"]))
                if worst_idx is not None
                else None
            ),
        }

    def outlier_report(
        self,
        columns: list[str] | None = None,
        *,
        method: str = "mad",
        threshold: float | None = None,
    ) -> pd.DataFrame:
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
            for col in self.data.columns
            if pd.api.types.is_numeric_dtype(self.data[col]) and col != "day_index"
        ]
        selected_columns = columns if columns is not None else numeric_columns
        unknown = [col for col in selected_columns if col not in self.data.columns]
        if unknown:
            raise ValueError("Unknown columns for outlier detection: " + ", ".join(unknown))

        rows: list[dict[str, Any]] = []
        for column in selected_columns:
            numeric = pd.to_numeric(self.data[column], errors="coerce")
            valid = numeric.dropna()
            if valid.shape[0] < 2:
                continue

            if outlier_method == "mad":
                median = float(valid.median())
                mad = float((valid - median).abs().median())
                if mad == 0:
                    continue
                scores = 0.6745 * (numeric - median) / mad
                mask = scores.abs() > effective_threshold
            elif outlier_method == "zscore":
                mean = float(valid.mean())
                std = float(valid.std(ddof=0))
                if std == 0:
                    continue
                scores = (numeric - mean) / std
                mask = scores.abs() > effective_threshold
            else:
                q1 = float(valid.quantile(0.25))
                q3 = float(valid.quantile(0.75))
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - (effective_threshold * iqr)
                upper = q3 + (effective_threshold * iqr)
                below = numeric < lower
                above = numeric > upper
                scores = pd.Series(0.0, index=numeric.index, dtype=float)
                scores.loc[below] = (numeric.loc[below] - lower) / iqr
                scores.loc[above] = (numeric.loc[above] - upper) / iqr
                mask = below | above

            flagged = mask.fillna(False)
            for idx in numeric.index[flagged]:
                rows.append(
                    {
                        "date": pd.Timestamp(self.data.loc[idx, "date"]),
                        "column": str(column),
                        "value": self._native_float(numeric.loc[idx]),
                        "score": self._native_float(scores.loc[idx]),
                        "method": outlier_method,
                        "threshold": float(effective_threshold),
                    }
                )

        result = pd.DataFrame(rows, columns=["date", "column", "value", "score", "method", "threshold"])
        if result.empty:
            return result
        return result.sort_values(["column", "date"], kind="stable").reset_index(drop=True)

    def _eda_price_series(self, price_col: str = "price_usd") -> pd.Series:
        """Return clean numeric price series indexed by dataframe row."""
        if price_col not in self.data.columns:
            raise ValueError(f"Unknown price column: {price_col}")
        return pd.to_numeric(self.data[price_col], errors="coerce")

    def _eda_value_series(self, series: str, price_col: str = "price_usd") -> pd.Series:
        """Return named EDA series for analysis helpers."""
        key = series.lower()
        if key == "price":
            return self._eda_price_series(price_col=price_col)
        if key in {"returns", "simple_returns"}:
            prices = self._eda_price_series(price_col=price_col)
            prev = prices.shift(1)
            valid_step = prices.notna() & prev.notna() & (prev != 0)
            out = pd.Series(np.nan, index=prices.index, dtype=float)
            out.loc[valid_step] = (prices.loc[valid_step] / prev.loc[valid_step]) - 1.0
            return out
        if key == "log_returns":
            prices = self._eda_price_series(price_col=price_col)
            prev = prices.shift(1)
            positive_step = prices.notna() & prev.notna() & (prices > 0) & (prev > 0)
            out = pd.Series(np.nan, index=prices.index, dtype=float)
            out.loc[positive_step] = np.log(prices.loc[positive_step] / prev.loc[positive_step])
            return out
        if key == "weight":
            return pd.to_numeric(self.data["weight"], errors="coerce")
        raise ValueError("series must be one of: price, returns, simple_returns, log_returns, weight")

    @staticmethod
    def _normalize_positive_ints(values: Iterable[int], field_name: str) -> list[int]:
        normalized: list[int] = []
        for value in values:
            ivalue = int(value)
            if ivalue <= 0:
                raise ValueError(f"{field_name} must contain only positive integers.")
            normalized.append(ivalue)
        return sorted(set(normalized))

    def rolling_statistics(
        self,
        windows: tuple[int, ...] = (7, 30, 90),
        *,
        price_col: str = "price_usd",
    ) -> pd.DataFrame:
        """Return rolling time-series statistics for price and returns."""
        normalized_windows = self._normalize_positive_ints(windows, "windows")
        prices = self._eda_price_series(price_col=price_col)
        returns = self._eda_value_series("returns", price_col=price_col)
        out = pd.DataFrame({"date": pd.to_datetime(self.data["date"], errors="coerce")})
        for window in normalized_windows:
            out[f"{price_col}_mean_{window}"] = prices.rolling(window, min_periods=1).mean()
            out[f"{price_col}_std_{window}"] = prices.rolling(window, min_periods=1).std(ddof=0)
            out[f"return_mean_{window}"] = returns.rolling(window, min_periods=1).mean()
            out[f"return_std_{window}"] = returns.rolling(window, min_periods=1).std(ddof=0)
            out[f"vol_annualized_{window}"] = out[f"return_std_{window}"] * np.sqrt(365.0)
        return out

    def autocorrelation(
        self,
        lags: tuple[int, ...] = (1, 7, 30),
        *,
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> dict[str, Any]:
        """Return autocorrelation values at selected lags."""
        normalized_lags = self._normalize_positive_ints(lags, "lags")
        values = self._eda_value_series(series=series, price_col=price_col).dropna()
        acf: dict[str, float | None] = {}
        for lag in normalized_lags:
            if lag >= int(values.shape[0]):
                acf[str(lag)] = None
            else:
                acf[str(lag)] = self._native_float(values.autocorr(lag=lag))
        return {
            "series": series.lower(),
            "lags": normalized_lags,
            "observations": int(values.shape[0]),
            "autocorrelation": acf,
        }

    def drawdown_table(self, top_n: int = 5, *, price_col: str = "price_usd") -> pd.DataFrame:
        """Return top drawdown episodes ranked by severity."""
        if int(top_n) <= 0:
            raise ValueError("top_n must be > 0")

        prices = self._eda_price_series(price_col=price_col)
        dates = pd.to_datetime(self.data["date"], errors="coerce")
        valid = prices.notna() & dates.notna()
        if not bool(valid.any()):
            return pd.DataFrame(
                columns=[
                    "peak_date",
                    "trough_date",
                    "recovery_date",
                    "max_drawdown",
                    "days_to_trough",
                    "days_to_recovery",
                    "duration_days",
                    "recovered",
                ]
            )

        prices_valid = prices.loc[valid].reset_index(drop=True)
        dates_valid = dates.loc[valid].reset_index(drop=True)
        running_max = prices_valid.cummax()
        drawdown = (prices_valid / running_max) - 1.0

        peak_idx = 0
        episodes: list[dict[str, Any]] = []
        in_drawdown = False
        start_idx = 0
        trough_idx = 0

        for idx in range(len(prices_valid)):
            if prices_valid.iloc[idx] >= running_max.iloc[idx]:
                peak_idx = idx

            dd = float(drawdown.iloc[idx])
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = peak_idx
                trough_idx = idx
            if in_drawdown and dd < float(drawdown.iloc[trough_idx]):
                trough_idx = idx
            if in_drawdown and dd >= 0:
                peak_date = pd.Timestamp(dates_valid.iloc[start_idx])
                trough_date = pd.Timestamp(dates_valid.iloc[trough_idx])
                recovery_date = pd.Timestamp(dates_valid.iloc[idx])
                episodes.append(
                    {
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_date": recovery_date,
                        "max_drawdown": self._native_float(drawdown.iloc[trough_idx]),
                        "days_to_trough": int((trough_date - peak_date).days),
                        "days_to_recovery": int((recovery_date - trough_date).days),
                        "duration_days": int((recovery_date - peak_date).days),
                        "recovered": True,
                    }
                )
                in_drawdown = False

        if in_drawdown:
            peak_date = pd.Timestamp(dates_valid.iloc[start_idx])
            trough_date = pd.Timestamp(dates_valid.iloc[trough_idx])
            end_date = pd.Timestamp(dates_valid.iloc[len(dates_valid) - 1])
            episodes.append(
                {
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "recovery_date": pd.NaT,
                    "max_drawdown": self._native_float(drawdown.iloc[trough_idx]),
                    "days_to_trough": int((trough_date - peak_date).days),
                    "days_to_recovery": None,
                    "duration_days": int((end_date - peak_date).days),
                    "recovered": False,
                }
            )

        if not episodes:
            return pd.DataFrame(
                columns=[
                    "peak_date",
                    "trough_date",
                    "recovery_date",
                    "max_drawdown",
                    "days_to_trough",
                    "days_to_recovery",
                    "duration_days",
                    "recovered",
                ]
            )

        out = pd.DataFrame(episodes)
        out = out.sort_values(["max_drawdown", "peak_date"], ascending=[True, True], kind="stable")
        return out.head(int(top_n)).reset_index(drop=True)

    def seasonality_profile(
        self,
        *,
        freq: str = "weekday",
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> pd.DataFrame:
        """Return calendar-seasonality summary statistics."""
        frequency = freq.lower()
        if frequency not in {"weekday", "month"}:
            raise ValueError("freq must be one of: weekday, month")

        values = self._eda_value_series(series=series, price_col=price_col)
        dates = pd.to_datetime(self.data["date"], errors="coerce")
        frame = pd.DataFrame({"date": dates, "value": values})
        frame = frame.dropna(subset=["date", "value"]).reset_index(drop=True)

        if frequency == "weekday":
            frame["period_id"] = frame["date"].dt.dayofweek
            labels = {
                0: "Mon",
                1: "Tue",
                2: "Wed",
                3: "Thu",
                4: "Fri",
                5: "Sat",
                6: "Sun",
            }
            expected_periods = list(labels.keys())
        else:
            frame["period_id"] = frame["date"].dt.month
            labels = {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            }
            expected_periods = list(labels.keys())

        grouped = frame.groupby("period_id", sort=True)["value"]
        rows: list[dict[str, Any]] = []
        for period_id in expected_periods:
            if period_id in grouped.indices:
                subset = grouped.get_group(period_id)
                rows.append(
                    {
                        "period_id": period_id,
                        "period_label": labels[period_id],
                        "count": int(subset.shape[0]),
                        "mean": self._native_float(subset.mean()),
                        "median": self._native_float(subset.median()),
                        "std": self._native_float(subset.std(ddof=0)),
                        "min": self._native_float(subset.min()),
                        "max": self._native_float(subset.max()),
                    }
                )
            else:
                rows.append(
                    {
                        "period_id": period_id,
                        "period_label": labels[period_id],
                        "count": 0,
                        "mean": None,
                        "median": None,
                        "std": None,
                        "min": None,
                        "max": None,
                    }
                )
        return pd.DataFrame(rows)


__all__ = [
    "ColumnSpec",
    "CoinMetricsLineageSpec",
    "StrategySeriesMetadata",
    "StrategyTimeSeries",
    "StrategyTimeSeriesBatch",
]

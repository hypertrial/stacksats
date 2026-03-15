"""Time-series analysis methods for StrategyTimeSeries."""

from __future__ import annotations

import datetime as dt
from typing import Any, Iterable

import numpy as np
import polars as pl
from scipy.signal import periodogram

def _autocorr_pl(series: pl.Series, lag: int) -> float | None:
    """Compute autocorrelation at lag for a Polars Series."""
    arr = series.drop_nulls().to_numpy()
    if len(arr) <= lag:
        return None
    y = arr[lag:]
    x = arr[:-lag]
    corr = np.corrcoef(y, x)[0, 1]
    return float(corr) if np.isfinite(corr) else None


class StrategyTimeSeriesAnalysisMixin:
    """EDA and statistical analysis helpers for normalized strategy windows."""

    def _eda_price_series(self, price_col: str = "price_usd") -> pl.Series:
        """Return clean numeric price series aligned to dataframe rows."""
        if price_col not in self._data.columns:
            raise ValueError(f"Unknown price column: {price_col}")
        return self._data[price_col].cast(pl.Float64, strict=False)

    def _eda_value_series(self, series: str, price_col: str = "price_usd") -> pl.Series:
        """Return named EDA series for analysis helpers."""
        key = series.lower()
        if key == "price":
            return self._eda_price_series(price_col=price_col)
        if key in {"returns", "simple_returns"}:
            prices = self._eda_price_series(price_col=price_col)
            prev = prices.shift(1)
            p_arr = prices.to_numpy()
            pr_arr = prev.to_numpy()
            out = np.full(len(prices), float("nan"))
            valid = np.isfinite(p_arr) & np.isfinite(pr_arr) & (pr_arr != 0)
            out[valid] = (p_arr[valid] / pr_arr[valid]) - 1.0
            return pl.Series("returns", out)
        if key == "log_returns":
            prices = self._eda_price_series(price_col=price_col)
            prev = prices.shift(1)
            p_arr = prices.to_numpy()
            pr_arr = prev.to_numpy()
            out = np.full(len(prices), float("nan"))
            positive = np.isfinite(p_arr) & np.isfinite(pr_arr) & (p_arr > 0) & (pr_arr > 0)
            out[positive] = np.log(p_arr[positive] / pr_arr[positive])
            return pl.Series("log_returns", out)
        if key == "weight":
            return self._data["weight"].cast(pl.Float64, strict=False)
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
    ) -> pl.DataFrame:
        """Return rolling time-series statistics for price and returns."""
        normalized_windows = self._normalize_positive_ints(windows, "windows")
        prices = self._eda_price_series(price_col=price_col)
        returns_series = self._eda_value_series("returns", price_col=price_col)

        out = self._data.select(["date"])
        for window in normalized_windows:
            out = out.with_columns(
                prices.rolling_mean(window_size=window).alias(f"{price_col}_mean_{window}"),
                prices.rolling_std(window_size=window).alias(f"{price_col}_std_{window}"),
                returns_series.rolling_mean(window_size=window).alias(f"return_mean_{window}"),
                returns_series.rolling_std(window_size=window).alias(f"return_std_{window}"),
            )
        for w in normalized_windows:
            out = out.with_columns(
                (pl.col(f"return_std_{w}") * np.sqrt(365.0)).alias(f"vol_annualized_{w}")
            )
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
        raw = self._eda_value_series(series=series, price_col=price_col)
        values = raw.filter(raw.is_finite())
        acf: dict[str, float | None] = {}
        n = values.len()
        for lag in normalized_lags:
            if lag >= n:
                acf[str(lag)] = None
            else:
                acf[str(lag)] = self._native_float(_autocorr_pl(values, lag))
        return {
            "series": series.lower(),
            "lags": normalized_lags,
            "observations": int(n),
            "autocorrelation": acf,
        }

    def drawdown_table(self, top_n: int = 5, *, price_col: str = "price_usd") -> pl.DataFrame:
        """Return top drawdown episodes ranked by severity."""
        if int(top_n) <= 0:
            raise ValueError("top_n must be > 0")

        prices = self._eda_price_series(price_col=price_col)
        dates = self._data["date"]
        valid = prices.is_not_null() & dates.is_not_null()
        if not valid.any():
            return pl.DataFrame(
                schema={
                    "peak_date": pl.Datetime,
                    "trough_date": pl.Datetime,
                    "recovery_date": pl.Datetime,
                    "max_drawdown": pl.Float64,
                    "days_to_trough": pl.Int64,
                    "days_to_recovery": pl.Int64,
                    "duration_days": pl.Int64,
                    "recovered": pl.Boolean,
                }
            )

        prices_valid = prices.filter(valid).to_numpy()
        dates_valid = dates.filter(valid).to_list()
        running_max = np.maximum.accumulate(prices_valid)
        drawdown = (prices_valid / running_max) - 1.0

        peak_idx = 0
        episodes: list[dict[str, Any]] = []
        in_drawdown = False
        start_idx = 0
        trough_idx = 0

        for idx in range(len(prices_valid)):
            if prices_valid[idx] >= running_max[idx]:
                peak_idx = idx

            dd = float(drawdown[idx])
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = peak_idx
                trough_idx = idx
            if in_drawdown and dd < float(drawdown[trough_idx]):
                trough_idx = idx
            if in_drawdown and dd >= 0:
                peak_date = dates_valid[start_idx]
                trough_date = dates_valid[trough_idx]
                recovery_date = dates_valid[idx]
                peak_dt = peak_date if isinstance(peak_date, dt.datetime) else dt.datetime.fromisoformat(str(peak_date)[:10])
                trough_dt = trough_date if isinstance(trough_date, dt.datetime) else dt.datetime.fromisoformat(str(trough_date)[:10])
                rec_dt = recovery_date if isinstance(recovery_date, dt.datetime) else dt.datetime.fromisoformat(str(recovery_date)[:10])
                episodes.append(
                    {
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_date": recovery_date,
                        "max_drawdown": self._native_float(drawdown[trough_idx]),
                        "days_to_trough": int((trough_dt - peak_dt).days),
                        "days_to_recovery": int((rec_dt - trough_dt).days),
                        "duration_days": int((rec_dt - peak_dt).days),
                        "recovered": True,
                    }
                )
                in_drawdown = False

        if in_drawdown:
            peak_date = dates_valid[start_idx]
            trough_date = dates_valid[trough_idx]
            end_date = dates_valid[-1]

            def _dt(v):
                return v if isinstance(v, dt.datetime) else dt.datetime.fromisoformat(str(v)[:10])

            episodes.append(
                {
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "recovery_date": None,
                    "max_drawdown": self._native_float(drawdown[trough_idx]),
                    "days_to_trough": (_dt(trough_date) - _dt(peak_date)).days,
                    "days_to_recovery": None,
                    "duration_days": (_dt(end_date) - _dt(peak_date)).days,
                    "recovered": False,
                }
            )

        if not episodes:
            return pl.DataFrame(schema={"peak_date": pl.Datetime, "trough_date": pl.Datetime, "recovery_date": pl.Datetime, "max_drawdown": pl.Float64, "days_to_trough": pl.Int64, "days_to_recovery": pl.Int64, "duration_days": pl.Int64, "recovered": pl.Boolean})

        out = pl.DataFrame(episodes)
        out = out.sort(["max_drawdown", "peak_date"])
        return out.head(int(top_n))

    def seasonality_profile(
        self,
        *,
        freq: str = "weekday",
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> pl.DataFrame:
        """Return calendar-seasonality summary statistics."""
        frequency = freq.lower()
        if frequency not in {"weekday", "month"}:
            raise ValueError("freq must be one of: weekday, month")

        values = self._eda_value_series(series=series, price_col=price_col)
        dates = self._data["date"]
        frame = pl.DataFrame({"date": dates, "value": values}).filter(pl.col("value").is_finite())

        if frequency == "weekday":
            frame = frame.with_columns(pl.col("date").dt.weekday().alias("period_id"))
            labels = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
            expected_periods = list(labels.keys())
        else:
            frame = frame.with_columns(pl.col("date").dt.month().alias("period_id"))
            labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            expected_periods = list(labels.keys())

        rows: list[dict[str, Any]] = []
        for period_id in expected_periods:
            subset = frame.filter(pl.col("period_id") == period_id)
            if subset.height > 0:
                v = subset["value"]
                std_val = v.std()
                rows.append(
                    {
                        "period_id": period_id,
                        "period_label": labels[period_id],
                        "count": int(subset.height),
                        "mean": self._native_float(float(v.mean())),
                        "median": self._native_float(float(v.median())),
                        "std": self._native_float(float(std_val)) if std_val is not None else None,
                        "min": self._native_float(float(v.min())),
                        "max": self._native_float(float(v.max())),
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
        return pl.DataFrame(rows)

    @staticmethod
    def _resolve_lags(lags: int | Iterable[int]) -> list[int]:
        if isinstance(lags, int):
            if lags <= 0:
                raise ValueError("lags must be > 0")
            return list(range(1, lags + 1))
        return StrategyTimeSeriesAnalysisMixin._normalize_positive_ints(lags, "lags")

    def _resolve_series_like(
        self,
        series_like: str | pl.Series,
        *,
        default_price_col: str = "price_usd",
    ) -> pl.Series:
        if isinstance(series_like, pl.Series):
            values = series_like.cast(pl.Float64, strict=False)
        else:
            if series_like in self._data.columns:
                return self._data[series_like].cast(pl.Float64, strict=False)
            return self._eda_value_series(series=series_like, price_col=default_price_col)

        target_len = self._data.height
        if values.len() < target_len:
            values = pl.concat([values, pl.Series([float("nan")] * (target_len - values.len()))])
        return values.head(target_len)

    @staticmethod
    def _pacf_at_lag(values: pl.Series, lag: int) -> float | None:
        if lag <= 0:
            return None
        arr = values.drop_nulls().to_numpy()
        if arr.shape[0] <= lag + 1:
            return None
        y = arr[lag:]
        x_lag = arr[:-lag]
        if lag == 1:
            corr = np.corrcoef(y, x_lag)[0, 1]
            return float(corr) if np.isfinite(corr) else None

        rows = y.shape[0]
        z = np.column_stack([arr[lag - i - 1 : arr.shape[0] - i - 1] for i in range(lag - 1)])
        z = np.column_stack([np.ones(rows), z])
        beta_y, *_ = np.linalg.lstsq(z, y, rcond=None)
        beta_x, *_ = np.linalg.lstsq(z, x_lag, rcond=None)
        resid_y = y - (z @ beta_y)
        resid_x = x_lag - (z @ beta_x)
        std_y = float(np.std(resid_y))
        std_x = float(np.std(resid_x))
        if np.isclose(std_y, 0.0) or np.isclose(std_x, 0.0):
            return None
        corr = np.corrcoef(resid_y, resid_x)[0, 1]
        return float(corr) if np.isfinite(corr) else None

    def resample(self, freq: str, agg: str = "mean") -> pl.DataFrame:
        """Resample to a coarser/finer frequency with controlled aggregation."""
        if not isinstance(freq, str) or not freq:
            raise ValueError("freq must be a non-empty Polars interval string.")
        frame = self._data.clone()
        numeric_cols = [c for c in frame.columns if frame[c].dtype in (pl.Float64, pl.Int64)]
        if not numeric_cols:
            return pl.DataFrame(schema={"date": pl.Datetime})
        # Polars: group_by_dynamic
        interval = freq
        if freq == "D":
            interval = "1d"
        elif freq == "W":
            interval = "1w"
        elif freq == "M":
            interval = "1mo"
        out = frame.group_by_dynamic("date", every=interval).agg(
            pl.col(numeric_cols).mean() if agg == "mean" else pl.col(numeric_cols).sum()
        )
        return out.drop_nulls(subset=["date"])

    def decompose(
        self,
        *,
        period: int,
        model: str = "additive",
        series: str = "price",
        price_col: str = "price_usd",
    ) -> pl.DataFrame:
        """Classical seasonal decomposition into trend/seasonal/residual components."""
        model_name = model.lower()
        if model_name not in {"additive", "multiplicative"}:
            raise ValueError("model must be one of: additive, multiplicative")
        if int(period) <= 1:
            raise ValueError("period must be an integer > 1")

        values = self._eda_value_series(series=series, price_col=price_col)
        trend = values.rolling_mean(window_size=int(period), center=True)
        arr = values.to_numpy()
        trend_arr = trend.to_numpy()

        if model_name == "multiplicative" and np.any(arr <= 0):
            raise ValueError("multiplicative decomposition requires strictly positive values.")

        detrended = arr - trend_arr if model_name == "additive" else arr / np.where(trend_arr != 0, trend_arr, np.nan)
        seasonal = np.full(len(arr), np.nan)
        for i in range(int(period)):
            idx = np.arange(i, len(arr), int(period))
            if idx.size > 0:
                seasonal_value = float(np.nanmean(detrended[idx]))
                seasonal[idx] = seasonal_value

        if model_name == "additive":
            seasonal = seasonal - float(np.nanmean(seasonal))
            residual = arr - trend_arr - seasonal
        else:
            seasonal_mean = float(np.nanmean(seasonal))
            if not np.isfinite(seasonal_mean) or np.isclose(seasonal_mean, 0.0):
                seasonal_mean = 1.0
            seasonal = seasonal / seasonal_mean
            residual = arr / (trend_arr * seasonal)

        return pl.DataFrame(
            {
                "date": self._data["date"],
                "observed": values,
                "trend": trend,
                "seasonal": pl.Series("seasonal", seasonal),
                "residual": pl.Series("residual", residual),
                "model": model_name,
                "period": int(period),
                "series": series,
            }
        )

    def detrend(
        self,
        *,
        method: str = "linear",
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Remove trend from numeric columns."""
        method_name = method.lower()
        if method_name not in {"linear", "difference"}:
            raise ValueError("method must be one of: linear, difference")

        out = self._data.select(["date"])
        candidate_columns = columns or [
            col for col in self._data.columns
            if self._data[col].dtype in (pl.Float64, pl.Int64) and col != "day_index"
        ]
        unknown = [col for col in candidate_columns if col not in self._data.columns]
        if unknown:
            raise ValueError("Unknown columns for detrend: " + ", ".join(unknown))

        for column in candidate_columns:
            values = self._data[column].cast(pl.Float64, strict=False)
            if method_name == "difference":
                out = out.with_columns(values.diff().alias(f"{column}_detrended"))
                continue
            mask = values.is_not_null()
            n_valid = mask.sum()
            if n_valid < 2:
                out = out.with_columns(pl.lit(float("nan")).alias(f"{column}_detrended"))
                continue
            x = np.arange(len(values), dtype=float)
            arr = values.to_numpy()
            valid_x = x[mask.to_numpy()]
            valid_y = arr[mask.to_numpy()]
            m, b = np.polyfit(valid_x, valid_y, 1)
            trend = (m * x) + b
            residual = np.where(mask.to_numpy(), arr - trend, np.nan)
            out = out.with_columns(pl.Series(f"{column}_detrended", residual))
        return out

    def difference(
        self,
        order: int = 1,
        *,
        seasonal_order: int = 0,
        seasonal_period: int | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Apply regular and optional seasonal differencing to numeric columns."""
        if int(order) < 0:
            raise ValueError("order must be >= 0")
        if int(seasonal_order) < 0:
            raise ValueError("seasonal_order must be >= 0")
        if seasonal_order > 0 and (seasonal_period is None or int(seasonal_period) <= 0):
            raise ValueError("seasonal_period must be a positive integer when seasonal_order > 0")

        out = self._data.select(["date"])
        candidate_columns = columns or [
            col for col in self._data.columns
            if self._data[col].dtype in (pl.Float64, pl.Int64) and col != "day_index"
        ]
        unknown = [col for col in candidate_columns if col not in self._data.columns]
        if unknown:
            raise ValueError("Unknown columns for difference: " + ", ".join(unknown))

        for column in candidate_columns:
            values = self._data[column].cast(pl.Float64, strict=False)
            transformed = values
            for _ in range(int(order)):
                transformed = transformed.diff()
            if seasonal_order > 0:
                step = int(seasonal_period)
                for _ in range(int(seasonal_order)):
                    transformed = transformed.diff(step)
            out = out.with_columns(transformed.alias(f"{column}_diff"))
        return out

    def acf_pacf(
        self,
        *,
        lags: int | Iterable[int] = 30,
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> pl.DataFrame:
        """Return ACF/PACF diagnostics at selected lags."""
        normalized_lags = self._resolve_lags(lags)
        values = self._eda_value_series(series=series, price_col=price_col).drop_nulls()
        rows: list[dict[str, Any]] = []
        for lag in normalized_lags:
            if lag >= values.len():
                rows.append({"lag": lag, "acf": None, "pacf": None})
                continue
            rows.append(
                {
                    "lag": lag,
                    "acf": self._native_float(_autocorr_pl(values, lag)),
                    "pacf": self._native_float(self._pacf_at_lag(values, lag)),
                }
            )
        out = pl.DataFrame(rows)
        out = out.with_columns(
            pl.lit(series.lower()).alias("series"),
            pl.lit(int(values.len())).alias("observations"),
        )
        return out

    def cross_correlation(
        self,
        other_series: str | pl.Series,
        *,
        max_lag: int = 30,
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> pl.DataFrame:
        """Compute lead/lag cross-correlation between two series."""
        if int(max_lag) < 0:
            raise ValueError("max_lag must be >= 0")
        base = self._eda_value_series(series=series, price_col=price_col)
        other = self._resolve_series_like(other_series, default_price_col=price_col)
        rows: list[dict[str, Any]] = []
        for lag in range(-int(max_lag), int(max_lag) + 1):
            base_shifted = base.shift(-lag)
            aligned = pl.DataFrame({"base": base_shifted, "other": other}).drop_nulls()
            if aligned.height == 0:
                corr = float("nan")
            else:
                corr = aligned["base"].to_numpy()
                oth = aligned["other"].to_numpy()
                valid = np.isfinite(corr) & np.isfinite(oth)
                if valid.sum() < 2:
                    corr = float("nan")
                else:
                    corr = float(np.corrcoef(corr[valid], oth[valid])[0, 1])
            rows.append(
                {
                    "lag": lag,
                    "correlation": self._native_float(corr) if not (corr != corr) else None,
                    "observations": int(aligned.height),
                }
            )
        return pl.DataFrame(rows)

    def spectral_density(
        self,
        *,
        method: str = "periodogram",
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> pl.DataFrame:
        """Estimate frequency-domain power spectral density."""
        method_name = method.lower()
        if method_name != "periodogram":
            raise ValueError("method must be 'periodogram'")

        raw = self._eda_value_series(series=series, price_col=price_col)
        values = raw.filter(raw.is_finite())
        if values.len() == 0:
            return pl.DataFrame(schema={"frequency": pl.Float64, "power": pl.Float64, "series": pl.Utf8, "method": pl.Utf8, "observations": pl.Int64})
        frequencies, power = periodogram(values.to_numpy(), scaling="density")
        return pl.DataFrame(
            {
                "frequency": frequencies,
                "power": power,
                "series": series.lower(),
                "method": method_name,
                "observations": int(values.len()),
            }
        )

    @staticmethod
    def _stationarity_proxy(values: pl.Series, acf_threshold: float) -> bool:
        clean = values.drop_nulls()
        if clean.len() < 3:
            return True
        std = float(clean.std())
        if np.isclose(std, 0.0):
            return True
        lag1 = _autocorr_pl(clean, 1)
        if lag1 is None:
            return True
        return bool(abs(float(lag1)) < float(acf_threshold))

    def integration_order(
        self,
        *,
        columns: list[str] | None = None,
        max_order: int = 2,
        acf_threshold: float = 0.8,
    ) -> pl.DataFrame:
        """Heuristically estimate order of integration per numeric time series."""
        if int(max_order) < 0:
            raise ValueError("max_order must be >= 0")
        if not 0 < float(acf_threshold) < 1:
            raise ValueError("acf_threshold must be between 0 and 1.")

        candidate_columns = columns or [
            col for col in self._data.columns
            if self._data[col].dtype in (pl.Float64, pl.Int64) and col != "day_index"
        ]
        unknown = [col for col in candidate_columns if col not in self._data.columns]
        if unknown:
            raise ValueError("Unknown columns for integration_order: " + ", ".join(unknown))

        rows: list[dict[str, Any]] = []
        for column in candidate_columns:
            base = self._data[column].cast(pl.Float64, strict=False)
            found_order: int | None = None
            lag1_at_found: float | None = None
            for d in range(int(max_order) + 1):
                transformed = base
                for _ in range(d):
                    transformed = transformed.diff()
                clean = transformed.drop_nulls()
                lag1 = _autocorr_pl(clean, 1) if clean.len() >= 2 else None
                if self._stationarity_proxy(clean, acf_threshold=float(acf_threshold)):
                    found_order = d
                    lag1_at_found = self._native_float(lag1) if lag1 is not None else None
                    break
            rows.append(
                {
                    "column": column,
                    "integration_order": found_order,
                    "detected": found_order is not None,
                    "max_order_tested": int(max_order),
                    "lag1_autocorr_at_detected_order": lag1_at_found,
                    "method": "acf_threshold_heuristic",
                    "acf_threshold": float(acf_threshold),
                }
            )
        return pl.DataFrame(rows)

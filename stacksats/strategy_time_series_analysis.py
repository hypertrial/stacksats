"""Time-series analysis methods for StrategyTimeSeries."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import periodogram


class StrategyTimeSeriesAnalysisMixin:
    """EDA and statistical analysis helpers for normalized strategy windows."""

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

    @staticmethod
    def _resolve_lags(lags: int | Iterable[int]) -> list[int]:
        if isinstance(lags, int):
            if lags <= 0:
                raise ValueError("lags must be > 0")
            return list(range(1, lags + 1))
        return StrategyTimeSeriesAnalysisMixin._normalize_positive_ints(lags, "lags")

    def _resolve_series_like(
        self,
        series_like: str | pd.Series,
        *,
        default_price_col: str = "price_usd",
    ) -> pd.Series:
        if isinstance(series_like, pd.Series):
            values = pd.to_numeric(series_like, errors="coerce")
            values = values.reset_index(drop=True)
            target_len = int(self.data.shape[0])
            if values.shape[0] < target_len:
                values = values.reindex(range(target_len))
            return values.iloc[:target_len]

        if series_like in self.data.columns:
            return pd.to_numeric(self.data[series_like], errors="coerce")
        return self._eda_value_series(series=series_like, price_col=default_price_col)

    @staticmethod
    def _pacf_at_lag(values: pd.Series, lag: int) -> float | None:
        if lag <= 0:
            return None
        arr = values.to_numpy(dtype=float)
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

    def resample(self, freq: str, agg: str = "mean") -> pd.DataFrame:
        """Resample to a coarser/finer frequency with controlled aggregation."""
        if not isinstance(freq, str) or not freq:
            raise ValueError("freq must be a non-empty pandas offset alias string.")
        frame = self.data.copy(deep=True)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).set_index("date")
        numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            return pd.DataFrame(columns=["date"])
        out = frame[numeric_columns].resample(freq).agg(agg)
        out = out.dropna(how="all").reset_index()
        return out

    def decompose(
        self,
        *,
        period: int,
        model: str = "additive",
        series: str = "price",
        price_col: str = "price_usd",
    ) -> pd.DataFrame:
        """Classical seasonal decomposition into trend/seasonal/residual components."""
        model_name = model.lower()
        if model_name not in {"additive", "multiplicative"}:
            raise ValueError("model must be one of: additive, multiplicative")
        if int(period) <= 1:
            raise ValueError("period must be an integer > 1")

        values = self._eda_value_series(series=series, price_col=price_col)
        values = values.astype(float)
        trend = values.rolling(window=int(period), center=True, min_periods=1).mean()

        if model_name == "multiplicative" and bool((values <= 0).any()):
            raise ValueError("multiplicative decomposition requires strictly positive values.")

        detrended = values - trend if model_name == "additive" else values / trend.replace(0, np.nan)
        seasonal = pd.Series(np.nan, index=values.index, dtype=float)
        for i in range(int(period)):
            idx = np.arange(i, len(values), int(period))
            if idx.size == 0:
                continue
            seasonal_value = float(np.nanmean(detrended.iloc[idx].to_numpy(dtype=float)))
            seasonal.iloc[idx] = seasonal_value

        if model_name == "additive":
            seasonal = seasonal - float(np.nanmean(seasonal.to_numpy(dtype=float)))
            residual = values - trend - seasonal
        else:
            seasonal_mean = float(np.nanmean(seasonal.to_numpy(dtype=float)))
            if not np.isfinite(seasonal_mean) or np.isclose(seasonal_mean, 0.0):
                seasonal_mean = 1.0
            seasonal = seasonal / seasonal_mean
            residual = values / (trend * seasonal)

        return pd.DataFrame(
            {
                "date": pd.to_datetime(self.data["date"], errors="coerce"),
                "observed": values,
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual,
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
    ) -> pd.DataFrame:
        """Remove trend from numeric columns."""
        method_name = method.lower()
        if method_name not in {"linear", "difference"}:
            raise ValueError("method must be one of: linear, difference")

        out = pd.DataFrame({"date": pd.to_datetime(self.data["date"], errors="coerce")})
        candidate_columns = columns or [
            col
            for col in self.data.columns
            if pd.api.types.is_numeric_dtype(self.data[col]) and col != "day_index"
        ]
        unknown = [col for col in candidate_columns if col not in self.data.columns]
        if unknown:
            raise ValueError("Unknown columns for detrend: " + ", ".join(unknown))

        for column in candidate_columns:
            values = pd.to_numeric(self.data[column], errors="coerce")
            if method_name == "difference":
                out[f"{column}_detrended"] = values.diff()
                continue
            mask = values.notna()
            if int(mask.sum()) < 2:
                out[f"{column}_detrended"] = np.nan
                continue
            x = np.arange(len(values), dtype=float)
            m, b = np.polyfit(x[mask], values.loc[mask].to_numpy(dtype=float), 1)
            trend = (m * x) + b
            residual = pd.Series(np.nan, index=values.index, dtype=float)
            residual.loc[mask] = values.loc[mask] - trend[mask]
            out[f"{column}_detrended"] = residual
        return out

    def difference(
        self,
        order: int = 1,
        *,
        seasonal_order: int = 0,
        seasonal_period: int | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Apply regular and optional seasonal differencing to numeric columns."""
        if int(order) < 0:
            raise ValueError("order must be >= 0")
        if int(seasonal_order) < 0:
            raise ValueError("seasonal_order must be >= 0")
        if seasonal_order > 0 and (seasonal_period is None or int(seasonal_period) <= 0):
            raise ValueError("seasonal_period must be a positive integer when seasonal_order > 0")

        out = pd.DataFrame({"date": pd.to_datetime(self.data["date"], errors="coerce")})
        candidate_columns = columns or [
            col
            for col in self.data.columns
            if pd.api.types.is_numeric_dtype(self.data[col]) and col != "day_index"
        ]
        unknown = [col for col in candidate_columns if col not in self.data.columns]
        if unknown:
            raise ValueError("Unknown columns for difference: " + ", ".join(unknown))

        for column in candidate_columns:
            values = pd.to_numeric(self.data[column], errors="coerce")
            transformed = values.copy()
            for _ in range(int(order)):
                transformed = transformed.diff()
            if seasonal_order > 0:
                step = int(seasonal_period)
                for _ in range(int(seasonal_order)):
                    transformed = transformed.diff(step)
            out[f"{column}_diff"] = transformed
        return out

    def acf_pacf(
        self,
        *,
        lags: int | Iterable[int] = 30,
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> pd.DataFrame:
        """Return ACF/PACF diagnostics at selected lags."""
        normalized_lags = self._resolve_lags(lags)
        values = self._eda_value_series(series=series, price_col=price_col).dropna()
        rows: list[dict[str, Any]] = []
        for lag in normalized_lags:
            if lag >= int(values.shape[0]):
                rows.append({"lag": lag, "acf": None, "pacf": None})
                continue
            rows.append(
                {
                    "lag": lag,
                    "acf": self._native_float(values.autocorr(lag=lag)),
                    "pacf": self._native_float(self._pacf_at_lag(values, lag)),
                }
            )
        out = pd.DataFrame(rows)
        out["series"] = series.lower()
        out["observations"] = int(values.shape[0])
        return out

    def cross_correlation(
        self,
        other_series: str | pd.Series,
        *,
        max_lag: int = 30,
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> pd.DataFrame:
        """Compute lead/lag cross-correlation between two series."""
        if int(max_lag) < 0:
            raise ValueError("max_lag must be >= 0")
        base = self._eda_value_series(series=series, price_col=price_col)
        other = self._resolve_series_like(other_series, default_price_col=price_col)
        rows: list[dict[str, Any]] = []
        for lag in range(-int(max_lag), int(max_lag) + 1):
            aligned = pd.DataFrame({"base": base.shift(lag), "other": other}).dropna()
            corr = aligned["base"].corr(aligned["other"]) if not aligned.empty else np.nan
            rows.append(
                {
                    "lag": lag,
                    "correlation": self._native_float(corr) if pd.notna(corr) else None,
                    "observations": int(aligned.shape[0]),
                }
            )
        return pd.DataFrame(rows)

    def spectral_density(
        self,
        *,
        method: str = "periodogram",
        series: str = "returns",
        price_col: str = "price_usd",
    ) -> pd.DataFrame:
        """Estimate frequency-domain power spectral density."""
        method_name = method.lower()
        if method_name != "periodogram":
            raise ValueError("method must be 'periodogram'")

        values = self._eda_value_series(series=series, price_col=price_col).dropna()
        if values.empty:
            return pd.DataFrame(columns=["frequency", "power", "series", "method", "observations"])
        frequencies, power = periodogram(values.to_numpy(dtype=float), scaling="density")
        return pd.DataFrame(
            {
                "frequency": frequencies,
                "power": power,
                "series": series.lower(),
                "method": method_name,
                "observations": int(values.shape[0]),
            }
        )

    @staticmethod
    def _stationarity_proxy(values: pd.Series, acf_threshold: float) -> bool:
        clean = values.dropna()
        if clean.shape[0] < 3:
            return True
        std = float(clean.std(ddof=0))
        if np.isclose(std, 0.0):
            return True
        lag1 = clean.autocorr(lag=1)
        if pd.isna(lag1):
            return True
        return bool(abs(float(lag1)) < float(acf_threshold))

    def integration_order(
        self,
        *,
        columns: list[str] | None = None,
        max_order: int = 2,
        acf_threshold: float = 0.8,
    ) -> pd.DataFrame:
        """Heuristically estimate order of integration per numeric time series."""
        if int(max_order) < 0:
            raise ValueError("max_order must be >= 0")
        if not 0 < float(acf_threshold) < 1:
            raise ValueError("acf_threshold must be between 0 and 1.")

        candidate_columns = columns or [
            col
            for col in self.data.columns
            if pd.api.types.is_numeric_dtype(self.data[col]) and col != "day_index"
        ]
        unknown = [col for col in candidate_columns if col not in self.data.columns]
        if unknown:
            raise ValueError("Unknown columns for integration_order: " + ", ".join(unknown))

        rows: list[dict[str, Any]] = []
        for column in candidate_columns:
            base = pd.to_numeric(self.data[column], errors="coerce")
            found_order: int | None = None
            lag1_at_found: float | None = None
            for d in range(int(max_order) + 1):
                transformed = base.copy()
                for _ in range(d):
                    transformed = transformed.diff()
                clean = transformed.dropna()
                lag1 = clean.autocorr(lag=1) if clean.shape[0] >= 2 else np.nan
                if self._stationarity_proxy(clean, acf_threshold=float(acf_threshold)):
                    found_order = d
                    lag1_at_found = self._native_float(lag1) if pd.notna(lag1) else None
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
        return pd.DataFrame(rows)

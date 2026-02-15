---
title: Signals and Features
description: Feature engineering and signal composition used by StackSats model intent generation.
---

# Signals and Features

This page covers model-side feature engineering and signal composition before framework allocation.

## Primary signals

| Signal | Weight | Meaning |
| --- | --- | --- |
| MVRV value | 70% | Lower MVRV increases buy intensity |
| Price vs MA | 20% | Below 200-day MA increases buy intensity |
| 4-year percentile | 10% | Cycle-context signal |

## Feature construction

### 200-day moving average

```python
price_ma = price.rolling(200, min_periods=100).mean()
price_vs_ma = (price / price_ma) - 1
```

### MVRV Z-score and gradient

```python
mvrv_zscore = (mvrv - rolling_mean(365)) / rolling_std(365)
mvrv_gradient = tanh(mvrv_zscore.diff(30).ewm(span=30).mean() * 2)
```

### 4-year percentile and volatility

```python
mvrv_percentile = rolling_percentile(mvrv, window=1461)
mvrv_volatility = percentile_rank(mvrv_zscore.rolling(90).std())
```

## Leakage prevention

Signal columns are shifted by one day so weight for day `t` uses only information available through `t-1`.

```python
signal_cols = [
    "price_vs_ma",
    "mvrv_zscore",
    "mvrv_gradient",
    "mvrv_percentile",
    "mvrv_acceleration",
    "mvrv_zone",
    "mvrv_volatility",
]
features[signal_cols] = features[signal_cols].shift(1)
```

## Dynamic multiplier structure

```python
combined = value_signal * 0.70 + ma_signal * 0.20 + pct_signal * 0.10
combined *= acceleration_modifier
combined *= confidence_boost
combined *= volatility_dampening
dynamic = exp(clip(combined * DYNAMIC_STRENGTH, -5, 100))
```

## See also

- [Allocation Kernel](allocation-kernel.md)
- [Backtest Runtime](../model_backtest.md)

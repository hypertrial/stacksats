"""Dynamic DCA weight computation using MVRV + 200-day MA strategy.

This module computes daily investment weights for a Bitcoin DCA strategy
based on two primary signals:
1. MVRV Z-score: Buy more when undervalued (low MVRV)
2. Price vs 200-day MA: Buy more when price is below long-term trend

Enhanced MVRV utilization includes:
- 4-year rolling percentile (captures full halving cycle)
- MVRV acceleration (momentum detection)
- Asymmetric extreme response (aggressive at lows, conservative at highs)
- Zone-based regime classification
- Adaptive thresholds based on volatility
"""

import numpy as np
import pandas as pd
from .framework_contract import (
    ALLOCATION_SPAN_DAYS,
    MAX_DAILY_WEIGHT,
    MIN_DAILY_WEIGHT,
    apply_clipped_weight,
    assert_final_invariants,
    compute_n_past,
    validate_span_length,
    validate_locked_prefix,
)

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"

# Strategy parameters
MIN_W = MIN_DAILY_WEIGHT
MAX_W = MAX_DAILY_WEIGHT
MA_WINDOW = 200  # 200-day simple moving average
MVRV_GRADIENT_WINDOW = 30  # Window for MVRV trend detection
MVRV_ROLLING_WINDOW = 365  # Window for MVRV Z-score normalization
MVRV_CYCLE_WINDOW = 1461  # 4-year window for percentile (halving cycle)
MVRV_ACCEL_WINDOW = 14  # Window for acceleration calculation
DYNAMIC_STRENGTH = 5.0  # Multiplier for weight adjustments

# MVRV Zone thresholds (based on historical distribution)
MVRV_ZONE_DEEP_VALUE = -2.0  # Z-score threshold for deep value
MVRV_ZONE_VALUE = -1.0  # Z-score threshold for value
MVRV_ZONE_CAUTION = 1.5  # Z-score threshold for caution
MVRV_ZONE_DANGER = 2.5  # Z-score threshold for danger

# Volatility adjustment parameters
MVRV_VOLATILITY_WINDOW = 90  # Window for volatility calculation
MVRV_VOLATILITY_DAMPENING = (
    0.2  # How much to dampen signals in high volatility (reduced)
)

# Feature column names used by model computations
FEATS = [
    "price_vs_ma",
    "mvrv_zscore",
    "mvrv_gradient",
    "mvrv_percentile",
    "mvrv_acceleration",
    "mvrv_zone",
    "mvrv_volatility",
    "signal_confidence",
]


# =============================================================================
# Helper Functions
# =============================================================================


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    ex = np.exp(x - x.max())
    return ex / ex.sum()


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling percentile rank (0 to 1).

    Uses a 4-year window by default to capture the full Bitcoin halving cycle.

    Args:
        series: Input time series
        window: Rolling window size (default 1461 = ~4 years)

    Returns:
        Series with percentile ranks in [0, 1]
    """

    def pct_rank(x: np.ndarray) -> float:
        """Compute percentile rank of last value in array."""
        if len(x) < 2:
            return 0.5
        rank = (x[-1] > x[:-1]).sum() / (len(x) - 1)
        return float(rank)

    return series.rolling(window, min_periods=window // 4).apply(pct_rank, raw=True)


def classify_mvrv_zone(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Classify MVRV into discrete zones for regime detection.

    Zones:
    - -2 (deep_value): Z < -2.0 (historically rare, extreme buying opportunity)
    - -1 (value): -2.0 <= Z < -1.0 (undervalued, increase buying)
    -  0 (neutral): -1.0 <= Z < 1.5 (fair value, normal DCA)
    - +1 (caution): 1.5 <= Z < 2.5 (overvalued, reduce buying)
    - +2 (danger): Z >= 2.5 (extreme overvaluation, minimize buying)

    Args:
        mvrv_zscore: Array of MVRV Z-scores

    Returns:
        Array of zone classifications in [-2, -1, 0, 1, 2]
    """
    return np.select(
        [
            mvrv_zscore < MVRV_ZONE_DEEP_VALUE,
            mvrv_zscore < MVRV_ZONE_VALUE,
            mvrv_zscore < MVRV_ZONE_CAUTION,
            mvrv_zscore < MVRV_ZONE_DANGER,
        ],
        [-2, -1, 0, 1],
        default=2,
    )


def compute_mvrv_volatility(mvrv_zscore: pd.Series, window: int) -> pd.Series:
    """Compute rolling volatility of MVRV Z-score.

    High volatility periods suggest uncertainty - signals should be dampened.
    Low volatility periods suggest conviction - signals can be amplified.

    Args:
        mvrv_zscore: MVRV Z-score series
        window: Rolling window for volatility calculation

    Returns:
        Normalized volatility in [0, 1] where 1 = high volatility
    """
    vol = mvrv_zscore.rolling(window, min_periods=window // 4).std()
    # Normalize to [0, 1] using historical quantiles
    vol_pct = vol.rolling(window * 4, min_periods=window).apply(
        lambda x: (x.iloc[-1] > x[:-1]).sum() / max(len(x) - 1, 1)
        if len(x) > 1
        else 0.5,
        raw=False,
    )
    return vol_pct.fillna(0.5)


def compute_signal_confidence(
    mvrv_zscore: np.ndarray,
    mvrv_percentile: np.ndarray,
    mvrv_gradient: np.ndarray,
    price_vs_ma: np.ndarray,
) -> np.ndarray:
    """Compute confidence score based on signal agreement.

    When multiple signals agree, confidence is high:
    - Low Z-score + Low percentile + Rising gradient = High confidence buy
    - High Z-score + High percentile + Falling gradient = High confidence reduce

    Args:
        mvrv_zscore: MVRV Z-score in [-4, 4]
        mvrv_percentile: 4-year percentile in [0, 1]
        mvrv_gradient: Trend direction in [-1, 1]
        price_vs_ma: Price vs MA in [-1, 1]

    Returns:
        Confidence score in [0, 1] where 1 = all signals strongly agree
    """
    # Normalize all signals to [-1, 1] where negative = buy signal
    z_signal = -mvrv_zscore / 4  # Normalize to [-1, 1]
    pct_signal = 0.5 - mvrv_percentile  # [0,1] -> [-0.5, 0.5] -> scale to [-1, 1]
    pct_signal = pct_signal * 2
    ma_signal = -price_vs_ma  # Below MA = buy signal

    # Gradient indicates momentum direction
    # Positive gradient with buy signals = confirmation
    # Negative gradient with buy signals = divergence (lower confidence)
    gradient_alignment = np.where(
        z_signal < 0,  # Buy signal from Z-score
        np.where(mvrv_gradient > 0, 1.0, 0.5),  # Rising = confirmation
        np.where(mvrv_gradient < 0, 1.0, 0.5),  # Falling = confirmation for sell
    )

    # Calculate agreement: how many signals point the same direction?
    signals = np.stack([z_signal, pct_signal, ma_signal], axis=0)
    signal_std = signals.std(axis=0)

    # Low std = high agreement, high std = disagreement
    # Transform std to confidence: confidence = 1 - normalized_std
    max_std = 1.0  # Maximum possible std when signals fully disagree
    agreement = 1.0 - np.clip(signal_std / max_std, 0, 1)

    # Combine agreement with gradient alignment
    confidence = agreement * 0.7 + gradient_alignment * 0.3

    return np.clip(confidence, 0, 1)


def compute_mean_reversion_pressure(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Compute mean reversion pressure based on distance from equilibrium.

    MVRV tends to revert to the mean. The further from equilibrium,
    the stronger the expected reversion pressure.

    Uses a sigmoid-like function to model increasing pressure at extremes.

    Args:
        mvrv_zscore: MVRV Z-score in [-4, 4]

    Returns:
        Reversion pressure in [-1, 1] where:
        - Positive = pressure to decrease (price likely to fall)
        - Negative = pressure to increase (price likely to rise)
    """
    # Sigmoid-like pressure: increases non-linearly at extremes
    pressure = np.tanh(mvrv_zscore * 0.5)

    # Extra pressure at extremes (beyond ±2 std)
    extreme_pressure = np.where(
        np.abs(mvrv_zscore) > 2,
        np.sign(mvrv_zscore) * 0.3 * (np.abs(mvrv_zscore) - 2),
        0,
    )

    return np.clip(pressure + extreme_pressure, -1, 1)


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MVRV and MA features for weight calculation.

    Features (all lagged 1 day to prevent look-ahead bias):
    - price_vs_ma: Normalized distance from 200-day MA, clipped to [-1, 1]
    - mvrv_zscore: MVRV Z-score (365-day window), clipped to [-4, 4]
    - mvrv_gradient: Smoothed MVRV trend direction in [-1, 1]
    - mvrv_percentile: 4-year rolling percentile [0, 1] (halving cycle context)
    - mvrv_acceleration: Second derivative of MVRV gradient (momentum)
    - mvrv_zone: Discrete zone classification [-2, -1, 0, 1, 2]

    Args:
        df: DataFrame with price and MVRV columns

    Returns:
        DataFrame with price and computed features
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # Filter to valid date range
    price = df[PRICE_COL].loc["2010-07-18":].copy()

    # 200-day MA and distance
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    # MVRV features
    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]

        # Core Z-score (365-day window)
        mvrv_z = zscore(mvrv, MVRV_ROLLING_WINDOW).clip(-4, 4)

        # 4-year rolling percentile (captures full halving cycle)
        mvrv_pct = rolling_percentile(mvrv, MVRV_CYCLE_WINDOW).fillna(0.5)

        # Smoothed gradient using EMA
        gradient_raw = mvrv_z.diff(MVRV_GRADIENT_WINDOW)
        gradient_smooth = gradient_raw.ewm(
            span=MVRV_GRADIENT_WINDOW, adjust=False
        ).mean()
        mvrv_gradient = np.tanh(gradient_smooth * 2).fillna(0)

        # MVRV acceleration (second derivative - momentum detection)
        accel_raw = mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
        mvrv_acceleration = accel_raw.ewm(span=MVRV_ACCEL_WINDOW, adjust=False).mean()
        mvrv_acceleration = np.tanh(mvrv_acceleration * 3).fillna(0)

        # Zone classification
        mvrv_zone = pd.Series(
            classify_mvrv_zone(mvrv_z.values),
            index=mvrv_z.index,
        )

        # MVRV volatility (for signal dampening in uncertain periods)
        mvrv_volatility = compute_mvrv_volatility(mvrv_z, MVRV_VOLATILITY_WINDOW)

        # Signal confidence (computed after lag, using lagged values)
        # Will be computed after lag is applied
        signal_confidence = pd.Series(0.5, index=price.index)
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_pct = pd.Series(0.5, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_zone = pd.Series(0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)
        signal_confidence = pd.Series(0.5, index=price.index)

    # Build and lag features
    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma": ma,
            "price_vs_ma": price_vs_ma,
            "mvrv_zscore": mvrv_z,
            "mvrv_gradient": mvrv_gradient,
            "mvrv_percentile": mvrv_pct,
            "mvrv_acceleration": mvrv_acceleration,
            "mvrv_zone": mvrv_zone,
            "mvrv_volatility": mvrv_volatility,
            "signal_confidence": signal_confidence,
        },
        index=price.index,
    )

    # Lag signals by 1 day to prevent look-ahead bias
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

    # Fill NaN values with appropriate defaults
    features["mvrv_percentile"] = features["mvrv_percentile"].fillna(0.5)
    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    features = features.fillna(0)

    # Compute signal confidence using lagged values (no look-ahead)
    features["signal_confidence"] = compute_signal_confidence(
        features["mvrv_zscore"].values,
        features["mvrv_percentile"].values,
        features["mvrv_gradient"].values,
        features["price_vs_ma"].values,
    )

    return features


# =============================================================================
# Weight Allocation
# =============================================================================


def _compute_stable_signal(raw: np.ndarray) -> np.ndarray:
    """Compute stable signal weights using cumulative mean normalization.

    signal[i] = raw[i] / mean(raw[0:i+1])

    This ensures weights only depend on past data.
    """
    n = len(raw)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    cumsum = np.cumsum(raw)
    running_mean = cumsum / np.arange(1, n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = raw / running_mean
    return np.where(np.isfinite(signal), signal, 1.0)


def allocate_sequential_stable(
    raw: np.ndarray,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate weights with lock-on-compute stability.

    Past weights are locked and never change. Future days absorb remainder.

    Args:
        raw: Raw weight values for all dates
        n_past: Number of past/current dates (locked)
        locked_weights: Optional pre-computed locked weights from database

    Returns:
        Weights summing to 1.0
    """
    n = len(raw)
    if n == 0:
        return np.array([])
    enforce_contract_bounds = n == ALLOCATION_SPAN_DAYS
    if n_past <= 0:
        out = np.full(n, 1.0 / n, dtype=float)
        assert_final_invariants(out)
        return out

    n_past = min(n_past, n)
    raw_arr = np.asarray(raw, dtype=float)
    w = np.zeros(n, dtype=float)
    base_weight = 1.0 / n
    locked_prefix = validate_locked_prefix(locked_weights, n_past)
    prefix_len = len(locked_prefix)
    if prefix_len > 0:
        w[:prefix_len] = locked_prefix

    remaining_budget = 1.0 - float(w[:prefix_len].sum())
    for i in range(prefix_len, n_past):
        signal = float(_compute_stable_signal(raw_arr[: i + 1])[-1])
        proposed = signal * base_weight
        clipped, remaining_budget = apply_clipped_weight(
            proposed,
            remaining_budget,
            n - i,
            enforce_contract_bounds=enforce_contract_bounds,
        )
        w[i] = clipped

    # Future days are reinitialized uniformly from remaining budget.
    n_future = n - n_past
    if n_future > 0:
        uniform_future = max(remaining_budget, 0.0) / n_future
        w[n_past:] = uniform_future
    else:
        w[n - 1] += max(remaining_budget, 0.0)

    assert_final_invariants(w)
    return w


def allocate_from_proposals(
    proposals: np.ndarray,
    n_past: int,
    n_total: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate final weights from user-proposed per-day values.

    Past/current proposals are clipped into feasible bounds and locked.
    Future days receive uniform allocation of the remaining budget.
    """
    if n_total == 0:
        return np.array([], dtype=float)
    enforce_contract_bounds = n_total == ALLOCATION_SPAN_DAYS
    if n_past <= 0:
        out = np.full(n_total, 1.0 / n_total, dtype=float)
        assert_final_invariants(out)
        return out

    n_past = min(n_past, n_total)
    proposals_arr = np.asarray(proposals, dtype=float)
    w = np.zeros(n_total, dtype=float)
    locked_prefix = validate_locked_prefix(locked_weights, n_past)
    prefix_len = len(locked_prefix)
    if prefix_len > 0:
        w[:prefix_len] = locked_prefix

    remaining_budget = 1.0 - float(w[:prefix_len].sum())
    for i in range(prefix_len, n_past):
        proposed = float(proposals_arr[i]) if i < len(proposals_arr) else 0.0
        clipped, remaining_budget = apply_clipped_weight(
            proposed,
            remaining_budget,
            n_total - i,
            enforce_contract_bounds=enforce_contract_bounds,
        )
        w[i] = clipped

    n_future = n_total - n_past
    if n_future > 0:
        uniform_future = max(remaining_budget, 0.0) / n_future
        w[n_past:] = uniform_future
    else:
        # Entire window is in the past/current segment.
        w[n_total - 1] += max(remaining_budget, 0.0)

    assert_final_invariants(w)
    return w


# =============================================================================
# Dynamic Multiplier
# =============================================================================


def compute_asymmetric_extreme_boost(mvrv_zscore: np.ndarray) -> np.ndarray:
    """Compute asymmetric boost for extreme MVRV values.

    Key insight: Bitcoin's MVRV is asymmetric - extreme lows are rare buying
    opportunities, while extreme highs often precede corrections.

    Behavior:
    - Z < -2: Strong positive boost (aggressive accumulation)
    - Z < -1: Moderate positive boost (value buying)
    - -1 <= Z <= 1.5: No boost (neutral zone)
    - Z > 1.5: Negative boost (reduce buying)
    - Z > 2.5: Strong negative boost (minimize buying)

    Args:
        mvrv_zscore: Array of MVRV Z-scores in [-4, 4]

    Returns:
        Boost values (positive = increase buying, negative = reduce buying)
    """
    boost = np.zeros_like(mvrv_zscore)

    # Deep undervaluation: strong positive boost (quadratic increase)
    deep_value_mask = mvrv_zscore < MVRV_ZONE_DEEP_VALUE
    boost = np.where(
        deep_value_mask,
        0.8 * (mvrv_zscore - MVRV_ZONE_DEEP_VALUE) ** 2 + 0.5,
        boost,
    )

    # Moderate undervaluation: linear positive boost
    value_mask = (mvrv_zscore >= MVRV_ZONE_DEEP_VALUE) & (mvrv_zscore < MVRV_ZONE_VALUE)
    boost = np.where(
        value_mask,
        -0.5 * mvrv_zscore,  # Linear boost proportional to undervaluation
        boost,
    )

    # Caution zone: moderate negative boost
    caution_mask = (mvrv_zscore >= MVRV_ZONE_CAUTION) & (mvrv_zscore < MVRV_ZONE_DANGER)
    boost = np.where(
        caution_mask,
        -0.3 * (mvrv_zscore - MVRV_ZONE_CAUTION),
        boost,
    )

    # Danger zone: strong negative boost (quadratic decrease)
    danger_mask = mvrv_zscore >= MVRV_ZONE_DANGER
    boost = np.where(
        danger_mask,
        -0.5 * (mvrv_zscore - MVRV_ZONE_DANGER) ** 2 - 0.3,
        boost,
    )

    return boost


def compute_percentile_signal(mvrv_percentile: np.ndarray) -> np.ndarray:
    """Convert MVRV percentile to buy signal.

    Uses 4-year rolling percentile to capture halving cycle context.

    Args:
        mvrv_percentile: Array of percentile values in [0, 1]

    Returns:
        Signal values (positive = buy more, negative = buy less)
    """
    # Transform [0, 1] to [-1, 1] with emphasis on extremes
    # Percentile < 0.25 -> strong buy signal
    # Percentile > 0.75 -> strong sell signal
    centered = 0.5 - mvrv_percentile  # Now in [-0.5, 0.5], positive = undervalued

    # Non-linear scaling to emphasize extremes
    signal = np.sign(centered) * (2 * np.abs(centered)) ** 1.5

    return np.clip(signal, -1, 1)


def compute_acceleration_modifier(
    mvrv_acceleration: np.ndarray,
    mvrv_gradient: np.ndarray,
) -> np.ndarray:
    """Compute modifier based on MVRV acceleration (momentum).

    Acceleration helps identify:
    - Momentum building in current direction
    - Potential trend reversals

    Args:
        mvrv_acceleration: Second derivative of MVRV gradient
        mvrv_gradient: First derivative (trend direction)

    Returns:
        Modifier in [0.5, 1.5] to scale other signals
    """
    # Same-direction acceleration: momentum building
    # Opposite-direction acceleration: potential reversal
    same_direction = (mvrv_acceleration * mvrv_gradient) > 0

    modifier = np.where(
        same_direction,
        1.0 + 0.3 * np.abs(mvrv_acceleration),  # Amplify if momentum building
        1.0 - 0.2 * np.abs(mvrv_acceleration),  # Dampen if potential reversal
    )

    return np.clip(modifier, 0.5, 1.5)


def compute_adaptive_trend_modifier(
    mvrv_gradient: np.ndarray,
    mvrv_zscore: np.ndarray,
) -> np.ndarray:
    """Compute trend modifier with adaptive thresholds.

    Instead of fixed 0.2/-0.2 thresholds, adapts based on current MVRV level:
    - In deep value: be more aggressive with dip buying
    - In overvalued territory: be more conservative

    Args:
        mvrv_gradient: MVRV trend direction in [-1, 1]
        mvrv_zscore: Current MVRV Z-score

    Returns:
        Trend modifier for MA signal
    """
    # Adaptive thresholds based on MVRV level
    # Lower threshold in value zone (more sensitive to reversals)
    # Higher threshold in danger zone (require stronger confirmation)
    threshold = np.where(
        mvrv_zscore < -1,
        0.1,  # Low threshold in value zone
        np.where(mvrv_zscore > 1.5, 0.4, 0.2),  # High threshold in danger zone
    )

    # Compute modifier
    modifier = np.where(
        mvrv_gradient > threshold,
        1.0 + 0.5 * np.minimum(mvrv_gradient, 1.0),  # Bull: up to 1.5x
        np.where(
            mvrv_gradient < -threshold,
            0.3 + 0.2 * (1 + mvrv_gradient),  # Bear: down to 0.3x
            1.0,  # Neutral
        ),
    )

    return np.clip(modifier, 0.3, 1.5)


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    mvrv_percentile: np.ndarray | None = None,
    mvrv_acceleration: np.ndarray | None = None,
    mvrv_volatility: np.ndarray | None = None,
    signal_confidence: np.ndarray | None = None,
) -> np.ndarray:
    """Compute weight multiplier from MVRV and MA signals.

    Enhanced strategy with multiple MVRV signals:
    - Primary (55%): MVRV value signal with asymmetric extreme boost
    - Secondary (25%): MA signal with adaptive trend modulation
    - Tertiary (15%): 4-year percentile context (halving cycle)
    - Quaternary (5%): Mean reversion pressure

    Modulated by:
    - Signal confidence: Amplify when signals agree
    - Volatility: Dampen in high uncertainty periods

    Args:
        price_vs_ma: Distance from 200-day MA in [-1, 1]
        mvrv_zscore: MVRV Z-score in [-4, 4]
        mvrv_gradient: MVRV trend direction in [-1, 1]
        mvrv_percentile: Optional 4-year rolling percentile [0, 1]
        mvrv_acceleration: Optional MVRV acceleration [-1, 1]
        mvrv_volatility: Optional volatility percentile [0, 1]
        signal_confidence: Optional confidence score [0, 1]

    Returns:
        Multipliers centered around 1.0
    """
    # Default to neutral if not provided
    if mvrv_percentile is None:
        mvrv_percentile = np.full_like(mvrv_zscore, 0.5)
    if mvrv_acceleration is None:
        mvrv_acceleration = np.zeros_like(mvrv_zscore)
    if mvrv_volatility is None:
        mvrv_volatility = np.full_like(mvrv_zscore, 0.5)
    if signal_confidence is None:
        signal_confidence = np.full_like(mvrv_zscore, 0.5)

    # 1. MVRV value signal: low MVRV = buy more
    value_signal = -mvrv_zscore

    # 2. Asymmetric extreme boost (corrected sign logic)
    extreme_boost = compute_asymmetric_extreme_boost(mvrv_zscore)
    value_signal = value_signal + extreme_boost

    # 3. MA signal: buy when below MA, with adaptive trend modulation
    ma_signal = -price_vs_ma
    trend_modifier = compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
    ma_signal = ma_signal * trend_modifier

    # 4. Percentile signal: 4-year context
    pct_signal = compute_percentile_signal(mvrv_percentile)

    # 5. Acceleration modifier: momentum detection
    accel_modifier = compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)

    # Combine signals with weights
    # Primary: MVRV value (70%), Secondary: MA (20%), Tertiary: Percentile (10%)
    # Focus on core MVRV signal with asymmetric boost
    combined = value_signal * 0.70 + ma_signal * 0.20 + pct_signal * 0.10

    # Apply acceleration modifier (subtle: range [0.85, 1.15])
    accel_modifier_subtle = 0.85 + 0.30 * (accel_modifier - 0.5) / 0.5
    accel_modifier_subtle = np.clip(accel_modifier_subtle, 0.85, 1.15)
    combined = combined * accel_modifier_subtle

    # Confidence boost only when very high (> 0.7), otherwise neutral
    # This prevents dampening when signals disagree
    confidence_boost = np.where(
        signal_confidence > 0.7,
        1.0 + 0.15 * (signal_confidence - 0.7) / 0.3,  # Up to 1.15x
        1.0,  # Neutral otherwise
    )
    combined = combined * confidence_boost

    # Volatility dampening only in extreme volatility (top 20%)
    # This prevents over-dampening in normal conditions
    volatility_dampening = np.where(
        mvrv_volatility > 0.8,
        1.0 - MVRV_VOLATILITY_DAMPENING * (mvrv_volatility - 0.8) / 0.2,
        1.0,  # No dampening for normal volatility
    )
    combined = combined * volatility_dampening

    # Scale and clip
    adjustment = combined * DYNAMIC_STRENGTH
    adjustment = np.clip(adjustment, -5, 100)

    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
# =============================================================================


def _clean_array(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    return np.where(np.isfinite(arr), arr, 0)


def compute_preference_scores(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Compute daily preference scores from model features.

    Preference scores are framework-internal logits; higher means more preferred.
    """
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    # Extract and clean features
    price_vs_ma = _clean_array(df["price_vs_ma"].values)
    mvrv_zscore = _clean_array(df["mvrv_zscore"].values)
    mvrv_gradient = _clean_array(df["mvrv_gradient"].values)

    # Optional features
    if "mvrv_percentile" in df.columns:
        mvrv_percentile = _clean_array(df["mvrv_percentile"].values)
        mvrv_percentile = np.where(mvrv_percentile == 0, 0.5, mvrv_percentile)
    else:
        mvrv_percentile = None

    if "mvrv_acceleration" in df.columns:
        mvrv_acceleration = _clean_array(df["mvrv_acceleration"].values)
    else:
        mvrv_acceleration = None

    if "mvrv_volatility" in df.columns:
        mvrv_volatility = _clean_array(df["mvrv_volatility"].values)
        mvrv_volatility = np.where(mvrv_volatility == 0, 0.5, mvrv_volatility)
    else:
        mvrv_volatility = None

    if "signal_confidence" in df.columns:
        signal_confidence = _clean_array(df["signal_confidence"].values)
        signal_confidence = np.where(signal_confidence == 0, 0.5, signal_confidence)
    else:
        signal_confidence = None

    multiplier = compute_dynamic_multiplier(
        price_vs_ma,
        mvrv_zscore,
        mvrv_gradient,
        mvrv_percentile,
        mvrv_acceleration,
        mvrv_volatility,
        signal_confidence,
    )
    safe_multiplier = np.clip(multiplier, 1e-12, None)
    preference = np.log(safe_multiplier)
    return pd.Series(preference, index=df.index, dtype=float)


def compute_weights_from_target_profile(
    *,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    target_profile: pd.Series,
    mode: str = "preference",
    locked_weights: np.ndarray | None = None,
    n_past: int | None = None,
) -> pd.Series:
    """Convert a target profile into final iterative stable allocation weights."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(full_range) == 0:
        return pd.Series(dtype=float)

    target = target_profile.reindex(full_range)
    target = pd.to_numeric(target, errors="coerce")

    n = len(full_range)
    base = np.ones(n, dtype=float) / n
    if mode == "absolute":
        absolute = target.fillna(0.0).to_numpy(dtype=float)
        absolute = np.where(np.isfinite(absolute), absolute, 0.0)
        absolute = np.clip(absolute, 0.0, None)
        if absolute.sum() <= 0:
            raw = base
        else:
            raw = absolute / absolute.sum()
    elif mode == "preference":
        preference = target.fillna(0.0).to_numpy(dtype=float)
        preference = np.where(np.isfinite(preference), preference, 0.0)
        preference = np.clip(preference, -50, 50)
        raw = base * np.exp(preference)
    else:
        raise ValueError(f"Unsupported target profile mode '{mode}'.")

    if n_past is None:
        n_past = compute_n_past(full_range, current_date)
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    assert_final_invariants(weights)
    return pd.Series(weights, index=full_range, dtype=float)


def compute_weights_from_proposals(
    *,
    proposals: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Convert per-day user proposals into final framework weights."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(full_range) == 0:
        return pd.Series(dtype=float)

    proposed = pd.to_numeric(proposals.reindex(full_range), errors="coerce")
    proposed_arr = proposed.fillna(0.0).to_numpy(dtype=float)
    weights = allocate_from_proposals(
        proposals=proposed_arr,
        n_past=n_past,
        n_total=len(full_range),
        locked_weights=locked_weights,
    )
    assert_final_invariants(weights)
    return pd.Series(weights, index=full_range, dtype=float)


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features.

    Internal helper retained for low-level model experiments.
    Canonical strategy execution is hook-driven:
    `propose_weight`/`build_target_profile` -> framework allocation kernel.

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Window start
        end_date: Window end
        n_past: Number of past days (for stable allocation)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights indexed by date
    """
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    preference = compute_preference_scores(features_df, start_date, end_date)
    raw = (np.ones(n, dtype=float) / n) * np.exp(np.clip(preference.to_numpy(), -50, 50))

    # Allocate with stability
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability.

    Two modes:
    1. BACKTEST (locked_weights=None): Signal-based allocation
    2. PRODUCTION (locked_weights provided): DB-backed stability

    Args:
        features_df: DataFrame from precompute_features()
        start_date: Investment window start
        end_date: Investment window end
        current_date: Current date (past/future boundary)
        locked_weights: Optional locked weights from database

    Returns:
        Series of weights summing to 1.0
    """
    validate_span_length(start_date, end_date)
    preference = compute_preference_scores(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
    )
    weights = compute_weights_from_target_profile(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
        current_date=current_date,
        target_profile=preference,
        mode="preference",
        locked_weights=locked_weights,
    )
    assert_final_invariants(weights.to_numpy(dtype=float))
    return weights

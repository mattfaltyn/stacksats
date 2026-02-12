import numpy as np
import pandas as pd
import scipy

PRICE_COL = "PriceUSD_coinmetrics"

# ----------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------
MIN_W = 1e-6
WINS = [30, 90, 180, 365, 1461]
FEATS = [f"z{w}" for w in WINS]
PROTOS = [(0.5, 5.0), (1.0, 1.0), (5.0, 0.5)]

THETA = np.array(
    [
        1.3742,
        1.0547,
        -1.2346,
        2.6553,
        2.9991,
        -0.4332,
        -0.1736,
        -0.667,
        0.4097,
        -0.6316,
        -2.9907,
        -2.999,
        -1.2846,
        -0.423,
        0.8559,
        -1.9027,
        -1.9168,
        2.9988,
        0.5724,
        0.0001,
        0.8663,
        1.2674,
        4.9999,
    ]
)


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------


def softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - x.max())
    return ex / ex.sum()


# ----------------------------------------------------------------------------
# Global Locked Weights Cache
# ----------------------------------------------------------------------------
# Cache for precomputed locked weights indexed by (raw_hash, day_index)
_LOCKED_WEIGHT_CACHE: dict[int, np.ndarray] = {}


def compute_stable_signal_weights(raw: np.ndarray) -> np.ndarray:
    """Compute stable signal weights that only depend on past data.

    Formula: signal[i] = raw[i] / mean(raw[0:i+1])

    This gives:
    - Average days: signal ≈ 1.0
    - Above-average days: signal > 1.0
    - Below-average days: signal < 1.0

    The signal is stable because it only depends on data up to day i.

    Args:
        raw: Raw weight values for all days

    Returns:
        Array of stable signal weights (unbounded, average ~1.0)
    """
    n = len(raw)
    if n == 0:
        return np.array([])

    if n == 1:
        return np.array([1.0])

    # Cumulative sum and count for running mean
    cumsum_raw = np.cumsum(raw)
    counts = np.arange(1, n + 1)

    # Running mean: cumsum / count
    running_mean = cumsum_raw / counts

    # Signal: raw / running_mean (1.0 for average)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal_weights = raw / running_mean
    signal_weights = np.where(np.isfinite(signal_weights), signal_weights, 1.0)

    return signal_weights


def get_or_compute_locked_weights(raw: np.ndarray) -> np.ndarray:
    """Compute stable signal weights using fast vectorized approach.

    No caching needed since the fast method is O(n).

    Args:
        raw: Raw weight values

    Returns:
        Stable signal weights array (NOT normalized to sum=1)
    """
    return compute_stable_signal_weights(raw)


def clear_locked_weight_cache() -> None:
    """Clear the locked weights cache."""
    global _LOCKED_WEIGHT_CACHE
    _LOCKED_WEIGHT_CACHE = {}


def allocate_sequential_stable(
    raw: np.ndarray, n_past: int, locked_weights: np.ndarray | None = None
) -> np.ndarray:
    """Allocate weights with TRUE lock-on-compute stability.

    ABSOLUTE stability guarantee:
        weight[i] at time T1 == weight[i] at time T2
        for all i < n_past at both T1 and T2

    Formula for all past/current days (0 to n_past-1):
        locked[i] = signal[i] / n_total
        where signal[i] = raw[i] / mean(raw[0:i+1])

    Future days absorb any discrepancy to ensure sum = 1.0.

    Args:
        raw: Raw weight values (base * dynamic) for all dates
        n_past: Number of past/current dates (indices 0 to n_past-1)
        locked_weights: Optional pre-computed locked weights from database

    Returns:
        Array of weights summing to 1.0
    """
    n_total = len(raw)

    if n_total == 0:
        return np.array([])

    if n_past <= 0:
        # All future: uniform distribution
        return np.full(n_total, 1.0 / n_total)

    # Clamp n_past to n_total
    n_past = min(n_past, n_total)
    n_future = n_total - n_past

    w = np.zeros(n_total)

    # Compute locked weights for all past days (0 to n_past-1)
    # Signal averages ~1.0, so weight = signal / n_total gives average of 1/n_total
    base_weight = 1.0 / n_total

    if locked_weights is not None and len(locked_weights) >= n_past:
        w[:n_past] = locked_weights[:n_past]
    else:
        for i in range(n_past):
            signal = compute_stable_signal_weights(raw[: i + 1])[-1]
            # Signal averages ~1.0
            w[i] = signal * base_weight

    # Check if past weights exceed budget
    past_sum = w[:n_past].sum()
    target_past_budget = n_past / n_total

    # If past weights exceed their target budget, scale them down proportionally
    # This preserves relative proportions (stability) while fitting budget
    if past_sum > target_past_budget + 1e-10:
        scale = target_past_budget / past_sum
        w[:n_past] = w[:n_past] * scale

    # Future days (except the last): uniform
    if n_future > 1:
        w[n_past : n_total - 1] = base_weight

    # The very last day of window absorbs remainder to ensure sum = 1.0
    other_sum = w[: n_total - 1].sum()
    w[n_total - 1] = max(1.0 - other_sum, 0)

    return w


def beta_mix_pdf(n: int, mix: np.ndarray) -> np.ndarray:
    t = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    pdf = (
        mix[0] * scipy.stats.beta.pdf(t, *PROTOS[0])
        + mix[1] * scipy.stats.beta.pdf(t, *PROTOS[1])
        + mix[2] * scipy.stats.beta.pdf(t, *PROTOS[2])
    )
    return pdf / n


def zscore(series: pd.Series, win: int) -> pd.Series:
    m = series.rolling(win, win // 2).mean()
    sd = series.rolling(win, win // 2).std()
    return ((series - m) / sd).fillna(0)


# ----------------------------------------------------------------------------
# Feature Construction
# ----------------------------------------------------------------------------


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute all features for the entire dataset.

    Args:
        df: DataFrame containing price data

    Returns:
        DataFrame with price and all z-score features
    """
    if PRICE_COL not in df.columns:
        raise KeyError(
            f"Price column '{PRICE_COL}' not found in DataFrame columns: {list(df.columns)}"
        )
    price_df = df[[PRICE_COL]].copy()

    # Ensure history for rolling calculations
    # Use start of data or specific date if needed, but generally just use full history available
    # Using loc from specific date as in original code to match logic
    price_df = price_df.loc["2010-07-18":]
    log_p = np.log(price_df[PRICE_COL])

    z_all = pd.DataFrame(
        {f"z{w}": zscore(log_p, w).clip(-4, 4) for w in WINS}, index=log_p.index
    )
    z_lag = z_all.shift(1).fillna(0)

    return price_df.join(z_lag).fillna(0)


# ----------------------------------------------------------------------------
# Weight Computation
# ----------------------------------------------------------------------------


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Optimized weight computation using precomputed features.

    Args:
        features_df: DataFrame with precomputed features (from precompute_features)
        start_date: Start date for the window
        end_date: End date for the window
        n_past: If provided, use stable allocation with this many past days.
                If None, use legacy allocate_sequential.
        locked_weights: Optional array of locked weights from database.
                       Used for production mode with DB-backed stability.

    Returns:
        Series of weights indexed by date
    """
    # Slice the precomputed features
    # Note: Slice includes start and end dates
    df_window = features_df.loc[start_date:end_date]

    if df_window.empty:
        return pd.Series(dtype=float)

    alpha, beta_v = THETA[:18].reshape(3, 6), THETA[18:]

    # Get features for first day of window to determine mixture weights
    # Use .iloc[0] because df_window starts at start_date
    first_feats = df_window[FEATS].iloc[0].values
    mix = softmax(alpha @ np.r_[1, first_feats])

    n = len(df_window)
    base = beta_mix_pdf(n, mix)

    # Compute dynamic component with numerical stability
    feat_values = df_window[FEATS].values
    # Clean features: replace NaN/Inf with 0
    feat_values = np.where(np.isfinite(feat_values), feat_values, 0)

    # Suppress warnings during matrix multiplication (we handle edge cases afterwards)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dot_product = feat_values @ beta_v

    # Clip to prevent overflow/underflow in exp
    dot_product = np.clip(dot_product, -700, 700)  # exp(-700) ≈ 0, exp(700) ≈ inf
    dyn = np.exp(-dot_product)
    # Replace any NaN/Inf with small positive value
    dyn = np.where(np.isfinite(dyn), dyn, 1e-10)

    raw = base * dyn

    # Always use stable allocation for consistent behavior
    # If n_past is None, treat all days as past (full window)
    if n_past is None:
        n_past = len(raw)

    weights = allocate_sequential_stable(raw, n_past, locked_weights)

    return pd.Series(weights, index=df_window.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute approach.

    This is the canonical weight computation function used by both backtest.py
    and export_weights.py.

    Two modes of operation:
    1. BACKTEST MODE (locked_weights=None):
       - Uses allocate_sequential for signal-based weights
       - All weights computed together (optimal for backtesting performance)
       - Past weights may change if recomputed with different current_date

    2. PRODUCTION MODE (locked_weights provided):
       - Uses locked weights from database for past days (NEVER recomputed)
       - Only current day's weight is signal-based and adjusted
       - Future days get uniform distribution
       - Perfect stability: past weights never change

    Args:
        features_df: DataFrame with precomputed features (from precompute_features)
        start_date: Start date of the investment window
        end_date: End date of the investment window
        current_date: Current date (determines past/future boundary)
        locked_weights: Optional array of locked weights from database.
                       If provided, uses DB-backed stability mode.

    Returns:
        Series of weights indexed by date, summing to 1.0
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features_df with placeholder rows for future dates
    missing_dates = full_range.difference(features_df.index)
    if len(missing_dates) > 0:
        # Use zeros for features (neutral values for future dates)
        placeholder_data = {col: 0.0 for col in features_df.columns}
        placeholder_df = pd.DataFrame(placeholder_data, index=missing_dates)
        extended_features_df = pd.concat([features_df, placeholder_df]).sort_index()
    else:
        extended_features_df = features_df

    # Determine n_past (number of days <= current_date)
    past_end = min(current_date, end_date)
    if start_date <= past_end:
        past_range = pd.date_range(start=start_date, end=past_end, freq="D")
        n_past = len(past_range)
    else:
        # All dates are in the future (start_date > current_date)
        n_past = 0

    # Compute weights
    # If locked_weights is None (backtest mode), uses allocate_sequential
    # If locked_weights is provided (production mode), uses DB-backed stability
    weights = compute_weights_fast(
        extended_features_df,
        start_date,
        end_date,
        n_past=n_past,
        locked_weights=locked_weights,
    )

    return weights.reindex(full_range, fill_value=0.0)

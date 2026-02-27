"""
Backtest Engine — Indicator Computation Engine
=================================================
Computes ALL technical indicators used by the 400+ MVP strategies.
Uses numpy vectorization for speed (no loops over bars).
Implements caching: same data hash → cached result.

Categories:
  1. Moving Averages (SMA, EMA, WMA, DEMA, TEMA, KAMA, Hull MA)
  2. Oscillators (RSI, Stochastic, StochRSI, Williams %R, CCI, ROC, Momentum, UO)
  3. MACD Family (MACD configs, TRIX, TSI, Awesome Oscillator)
  4. Volatility (ATR, Bollinger, Keltner, Donchian, SuperTrend, Parabolic SAR, Squeeze)
  5. Volume (OBV, MFI, CMF, Chaikin Osc, Volume Osc, VWAP, Volume SMA)
  6. Trend (ADX/DMI, Aroon, Vortex, Ichimoku, Heikin-Ashi)
  7. Structure (Pivot Points, Linear Regression, Z-Score, Hurst, Fibonacci)
  8. Candlestick Patterns (Doji, Hammer, Engulfing, Pin Bar, Inside Bar, etc.)
"""

import hashlib
import time as _time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════

_CACHE: Dict[str, Dict[str, Any]] = {}


def _data_hash(df: pd.DataFrame) -> str:
    """Fast hash of DataFrame for cache key."""
    raw = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.md5(raw).hexdigest()


def _clear_cache():
    """Clear indicator cache."""
    _CACHE.clear()


# ═══════════════════════════════════════════════════════════════
# 1. MOVING AVERAGES
# ═══════════════════════════════════════════════════════════════

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=np.float64)
    return series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def dema(series: pd.Series, period: int) -> pd.Series:
    """Double Exponential Moving Average."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2.0 * e1 - e2


def tema(series: pd.Series, period: int) -> pd.Series:
    """Triple Exponential Moving Average."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3.0 * e1 - 3.0 * e2 + e3


def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    vals = series.values.astype(np.float64)
    n = len(vals)
    result = np.full(n, np.nan)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    if n <= period:
        return pd.Series(result, index=series.index)

    result[period - 1] = vals[period - 1]

    for i in range(period, n):
        direction = abs(vals[i] - vals[i - period])
        volatility = np.sum(np.abs(np.diff(vals[i - period:i + 1])))
        if volatility == 0:
            er = 0.0
        else:
            er = direction / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result[i] = result[i - 1] + sc * (vals[i] - result[i - 1])

    return pd.Series(result, index=series.index)


def hull_ma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    w1 = wma(series, half)
    w2 = wma(series, period)
    diff = 2.0 * w1 - w2
    return wma(diff, sqrt_p)


# ═══════════════════════════════════════════════════════════════
# 2. OSCILLATORS
# ═══════════════════════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """Stochastic Oscillator (%K and %D)."""
    lowest = low.rolling(window=k_period, min_periods=k_period).min()
    highest = high.rolling(window=k_period, min_periods=k_period).max()
    denom = highest - lowest
    k = 100.0 * (close - lowest) / denom.replace(0, np.nan)
    d = sma(k, d_period)
    return {"k": k, "d": d}


def stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
              k_smooth: int = 3, d_smooth: int = 3) -> Dict[str, pd.Series]:
    """Stochastic RSI."""
    rsi_vals = rsi(series, rsi_period)
    lowest = rsi_vals.rolling(window=stoch_period, min_periods=stoch_period).min()
    highest = rsi_vals.rolling(window=stoch_period, min_periods=stoch_period).max()
    denom = highest - lowest
    stoch_rsi_raw = (rsi_vals - lowest) / denom.replace(0, np.nan)
    k = sma(stoch_rsi_raw * 100.0, k_smooth)
    d = sma(k, d_smooth)
    return {"k": k, "d": d}


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Williams %R."""
    highest = high.rolling(window=period, min_periods=period).max()
    lowest = low.rolling(window=period, min_periods=period).min()
    denom = highest - lowest
    return -100.0 * (highest - close) / denom.replace(0, np.nan)


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    tp_sma = sma(tp, period)
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change (percentage)."""
    prev = series.shift(period)
    return 100.0 * (series - prev) / prev.replace(0, np.nan)


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Momentum (absolute difference)."""
    return series - series.shift(period)


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                        p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    """Ultimate Oscillator."""
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum().replace(0, np.nan)
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum().replace(0, np.nan)
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum().replace(0, np.nan)

    return 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0


# ═══════════════════════════════════════════════════════════════
# 3. MACD FAMILY
# ═══════════════════════════════════════════════════════════════

def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Dict[str, pd.Series]:
    """MACD: line, signal, histogram."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"line": macd_line, "signal": signal_line, "histogram": histogram}


def trix(series: pd.Series, period: int = 15, signal_period: int = 9) -> Dict[str, pd.Series]:
    """TRIX: triple-smoothed EMA rate of change."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    trix_line = e3.pct_change() * 100.0
    signal_line = ema(trix_line, signal_period)
    return {"line": trix_line, "signal": signal_line}


def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13,
        signal_period: int = 7) -> Dict[str, pd.Series]:
    """True Strength Index."""
    diff = series.diff()
    double_smooth_diff = ema(ema(diff, long_period), short_period)
    double_smooth_abs = ema(ema(diff.abs(), long_period), short_period)
    tsi_line = 100.0 * double_smooth_diff / double_smooth_abs.replace(0, np.nan)
    signal_line = ema(tsi_line, signal_period)
    return {"line": tsi_line, "signal": signal_line}


def awesome_oscillator(high: pd.Series, low: pd.Series,
                       fast: int = 5, slow: int = 34) -> pd.Series:
    """Awesome Oscillator."""
    midpoint = (high + low) / 2.0
    return sma(midpoint, fast) - sma(midpoint, slow)


# ═══════════════════════════════════════════════════════════════
# 4. VOLATILITY
# ═══════════════════════════════════════════════════════════════

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Bollinger Bands: upper, middle, lower, %B, bandwidth."""
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    band_width = (upper - lower) / middle.replace(0, np.nan) * 100.0
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return {
        "upper": upper, "middle": middle, "lower": lower,
        "pct_b": pct_b, "bandwidth": band_width,
    }


def keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                    ema_period: int = 20, atr_period: int = 14,
                    multiplier: float = 2.0) -> Dict[str, pd.Series]:
    """Keltner Channel."""
    middle = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return {"upper": upper, "middle": middle, "lower": lower}


def donchian_channel(high: pd.Series, low: pd.Series,
                     period: int = 20) -> Dict[str, pd.Series]:
    """Donchian Channel."""
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    return {"upper": upper, "middle": middle, "lower": lower}


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
    """SuperTrend indicator."""
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    n = len(close)
    st = np.full(n, np.nan)
    direction = np.zeros(n)  # 1 = up (bullish), -1 = down (bearish)

    ub = upper_band.values.copy()
    lb = lower_band.values.copy()
    c = close.values

    # Find first valid index
    first_valid = period
    while first_valid < n and (np.isnan(ub[first_valid]) or np.isnan(lb[first_valid])):
        first_valid += 1

    if first_valid >= n:
        return {"supertrend": pd.Series(st, index=close.index),
                "direction": pd.Series(direction, index=close.index)}

    direction[first_valid] = 1
    st[first_valid] = lb[first_valid]

    for i in range(first_valid + 1, n):
        if np.isnan(ub[i]) or np.isnan(lb[i]):
            st[i] = st[i - 1]
            direction[i] = direction[i - 1]
            continue

        # Adjust bands
        if lb[i] > lb[i - 1] or c[i - 1] < lb[i - 1]:
            pass
        else:
            lb[i] = lb[i - 1]

        if ub[i] < ub[i - 1] or c[i - 1] > ub[i - 1]:
            pass
        else:
            ub[i] = ub[i - 1]

        if direction[i - 1] == 1:  # was bullish
            if c[i] < lb[i]:
                direction[i] = -1
                st[i] = ub[i]
            else:
                direction[i] = 1
                st[i] = lb[i]
        else:  # was bearish
            if c[i] > ub[i]:
                direction[i] = 1
                st[i] = lb[i]
            else:
                direction[i] = -1
                st[i] = ub[i]

    return {
        "supertrend": pd.Series(st, index=close.index),
        "direction": pd.Series(direction, index=close.index),
    }


def parabolic_sar(high: pd.Series, low: pd.Series,
                  af_start: float = 0.02, af_step: float = 0.02,
                  af_max: float = 0.20) -> Dict[str, pd.Series]:
    """Parabolic SAR."""
    n = len(high)
    h = high.values.astype(np.float64)
    l = low.values.astype(np.float64)

    sar = np.full(n, np.nan)
    trend = np.zeros(n)  # 1 = up, -1 = down

    if n < 2:
        return {"sar": pd.Series(sar, index=high.index),
                "trend": pd.Series(trend, index=high.index)}

    # Initialize
    if h[1] > h[0]:
        trend[0] = trend[1] = 1
        sar[0] = sar[1] = l[0]
        ep = h[1]
    else:
        trend[0] = trend[1] = -1
        sar[0] = sar[1] = h[0]
        ep = l[1]
    af = af_start

    for i in range(2, n):
        prev_sar = sar[i - 1]

        if trend[i - 1] == 1:  # uptrend
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], l[i - 1], l[i - 2])

            if l[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                ep = l[i]
                af = af_start
            else:
                trend[i] = 1
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:  # downtrend
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], h[i - 1], h[i - 2])

            if h[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                ep = h[i]
                af = af_start
            else:
                trend[i] = -1
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)

    return {
        "sar": pd.Series(sar, index=high.index),
        "trend": pd.Series(trend, index=high.index),
    }


def squeeze_momentum(high: pd.Series, low: pd.Series, close: pd.Series,
                     bb_period: int = 20, bb_mult: float = 2.0,
                     kc_period: int = 20, kc_mult: float = 1.5) -> Dict[str, pd.Series]:
    """Squeeze Momentum (BB inside KC = squeeze on)."""
    bb = bollinger_bands(close, bb_period, bb_mult)
    kc = keltner_channel(high, low, close, kc_period, kc_period, kc_mult)
    squeeze_on = (bb["lower"] > kc["lower"]) & (bb["upper"] < kc["upper"])

    # Momentum component: linear regression of (close - midline of KC+DC)
    dc = donchian_channel(high, low, bb_period)
    mid = (kc["middle"] + dc["middle"]) / 2.0
    val = close - mid
    # Use SMA of the value as a simple momentum proxy
    mom = sma(val, bb_period)

    return {"squeeze_on": squeeze_on.astype(int), "momentum": mom}


# ═══════════════════════════════════════════════════════════════
# 5. VOLUME
# ═══════════════════════════════════════════════════════════════

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    tp_diff = tp.diff()

    pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(period, min_periods=period).sum()
    neg_mf = raw_mf.where(tp_diff <= 0, 0.0).rolling(period, min_periods=period).sum()

    mf_ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + mf_ratio))


def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    hl_range = high - low
    clv = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    mf_volume = clv * volume
    return mf_volume.rolling(period, min_periods=period).sum() / \
           volume.rolling(period, min_periods=period).sum().replace(0, np.nan)


def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                       volume: pd.Series, fast: int = 3, slow: int = 10) -> pd.Series:
    """Chaikin Oscillator = EMA(ADL, fast) - EMA(ADL, slow)."""
    hl_range = high - low
    clv = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    adl = (clv * volume).cumsum()
    return ema(adl, fast) - ema(adl, slow)


def volume_oscillator(volume: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """Volume Oscillator = (fast_vol_sma - slow_vol_sma) / slow_vol_sma * 100."""
    fast_sma = sma(volume, fast)
    slow_sma = sma(volume, slow)
    return (fast_sma - slow_sma) / slow_sma.replace(0, np.nan) * 100.0


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price (cumulative)."""
    tp = (high + low + close) / 3.0
    cum_tp_vol = (tp * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average."""
    return sma(volume, period)


# ═══════════════════════════════════════════════════════════════
# 6. TREND
# ═══════════════════════════════════════════════════════════════

def adx_dmi(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> Dict[str, pd.Series]:
    """ADX with +DI and -DI."""
    tr = true_range(high, low, close)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha = 1.0 / period
    atr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return {"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di}


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> Dict[str, pd.Series]:
    """Aroon Up, Down, and Oscillator."""
    n = len(high)
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)

    h = high.values
    l = low.values

    for i in range(period, n):
        window_h = h[i - period:i + 1]
        window_l = l[i - period:i + 1]
        high_idx = np.argmax(window_h)
        low_idx = np.argmin(window_l)
        up[i] = 100.0 * high_idx / period
        down[i] = 100.0 * low_idx / period

    aroon_up = pd.Series(up, index=high.index)
    aroon_down = pd.Series(down, index=high.index)
    aroon_osc = aroon_up - aroon_down

    return {"up": aroon_up, "down": aroon_down, "oscillator": aroon_osc}


def vortex(high: pd.Series, low: pd.Series, close: pd.Series,
           period: int = 14) -> Dict[str, pd.Series]:
    """Vortex Indicator (+VI, -VI)."""
    tr = true_range(high, low, close)

    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    tr_sum = tr.rolling(period, min_periods=period).sum()
    vip = vm_plus.rolling(period, min_periods=period).sum() / tr_sum.replace(0, np.nan)
    vim = vm_minus.rolling(period, min_periods=period).sum() / tr_sum.replace(0, np.nan)

    return {"plus": vip, "minus": vim}


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
             displacement: int = 26) -> Dict[str, pd.Series]:
    """Ichimoku Cloud (5 lines)."""
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(displacement)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2.0).shift(displacement)
    chikou = close.shift(-displacement)

    return {
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_line,
        "chikou": chikou,
    }


def heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series,
                close: pd.Series) -> Dict[str, pd.Series]:
    """Heikin-Ashi candles."""
    ha_close = (open_ + high + low + close) / 4.0

    n = len(open_)
    ha_open = np.full(n, np.nan, dtype=np.float64)
    ha_open[0] = (open_.iloc[0] + close.iloc[0]) / 2.0

    o = open_.values.astype(np.float64)
    c = close.values.astype(np.float64)
    hac = ha_close.values

    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + hac[i - 1]) / 2.0

    ha_open_s = pd.Series(ha_open, index=open_.index)
    ha_high = pd.concat([high, ha_open_s, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open_s, ha_close], axis=1).min(axis=1)

    return {"open": ha_open_s, "high": ha_high, "low": ha_low, "close": ha_close}


# ═══════════════════════════════════════════════════════════════
# 7. STRUCTURE / STATISTICAL
# ═══════════════════════════════════════════════════════════════

def pivot_points_classic(high: float, low: float, close: float) -> Dict[str, float]:
    """Classic Pivot Points from previous period's HLC."""
    pp = (high + low + close) / 3.0
    r1 = 2.0 * pp - low
    s1 = 2.0 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2.0 * (pp - low)
    s3 = low - 2.0 * (high - pp)
    return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


def pivot_points_fibonacci(high: float, low: float, close: float) -> Dict[str, float]:
    """Fibonacci Pivot Points."""
    pp = (high + low + close) / 3.0
    diff = high - low
    r1 = pp + 0.382 * diff
    r2 = pp + 0.618 * diff
    r3 = pp + diff
    s1 = pp - 0.382 * diff
    s2 = pp - 0.618 * diff
    s3 = pp - diff
    return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


def pivot_points_camarilla(high: float, low: float, close: float) -> Dict[str, float]:
    """Camarilla Pivot Points."""
    diff = high - low
    r1 = close + diff * 1.1 / 12.0
    r2 = close + diff * 1.1 / 6.0
    r3 = close + diff * 1.1 / 4.0
    r4 = close + diff * 1.1 / 2.0
    s1 = close - diff * 1.1 / 12.0
    s2 = close - diff * 1.1 / 6.0
    s3 = close - diff * 1.1 / 4.0
    s4 = close - diff * 1.1 / 2.0
    return {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "s1": s1, "s2": s2, "s3": s3, "s4": s4}


def pivot_points_woodie(high: float, low: float, close: float) -> Dict[str, float]:
    """Woodie Pivot Points."""
    pp = (high + low + 2.0 * close) / 4.0
    r1 = 2.0 * pp - low
    r2 = pp + high - low
    s1 = 2.0 * pp - high
    s2 = pp - high + low
    return {"pp": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2}


def linear_regression(series: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    """Linear Regression: slope, intercept, upper/lower channel, R-squared."""
    n = len(series)
    vals = series.values.astype(np.float64)

    slope_arr = np.full(n, np.nan)
    intercept_arr = np.full(n, np.nan)
    r_squared_arr = np.full(n, np.nan)
    value_arr = np.full(n, np.nan)
    upper_arr = np.full(n, np.nan)
    lower_arr = np.full(n, np.nan)

    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for i in range(period - 1, n):
        y = vals[i - period + 1:i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        slope = cov / x_var if x_var != 0 else 0.0
        intercept = y_mean - slope * x_mean

        y_pred = slope * x + intercept
        residuals = y - y_pred
        ss_res = (residuals ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        std_err = np.std(residuals)

        slope_arr[i] = slope
        intercept_arr[i] = intercept
        r_squared_arr[i] = r2
        value_arr[i] = y_pred[-1]
        upper_arr[i] = y_pred[-1] + 2.0 * std_err
        lower_arr[i] = y_pred[-1] - 2.0 * std_err

    idx = series.index
    return {
        "slope": pd.Series(slope_arr, index=idx),
        "intercept": pd.Series(intercept_arr, index=idx),
        "r_squared": pd.Series(r_squared_arr, index=idx),
        "value": pd.Series(value_arr, index=idx),
        "upper": pd.Series(upper_arr, index=idx),
        "lower": pd.Series(lower_arr, index=idx),
    }


def z_score(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-Score: (value - mean) / std."""
    mean = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    return (series - mean) / std.replace(0, np.nan)


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> pd.Series:
    """Hurst Exponent (rolling estimate). >0.5=trending, <0.5=mean-reverting."""
    n = len(series)
    vals = series.values.astype(np.float64)
    result = np.full(n, np.nan)
    window = max(max_lag * 2, 200)

    for i in range(window - 1, n):
        segment = vals[i - window + 1:i + 1]
        if np.any(np.isnan(segment)):
            continue

        lags = range(2, min(max_lag, len(segment) // 2))
        tau = []
        for lag in lags:
            diff = segment[lag:] - segment[:-lag]
            tau.append(np.sqrt(np.mean(diff ** 2)))

        if len(tau) < 4:
            continue

        log_lags = np.log(list(lags))
        log_tau = np.log(tau)

        # Linear fit
        coeffs = np.polyfit(log_lags, log_tau, 1)
        result[i] = coeffs[0]

    return pd.Series(result, index=series.index)


def fibonacci_retracement(swing_high: float, swing_low: float) -> Dict[str, float]:
    """Fibonacci Retracement levels."""
    diff = swing_high - swing_low
    return {
        "0.0": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0": swing_low,
    }


def fibonacci_extension(swing_high: float, swing_low: float,
                        retracement_point: float) -> Dict[str, float]:
    """Fibonacci Extension levels."""
    diff = swing_high - swing_low
    return {
        "1.0": retracement_point + diff,
        "1.272": retracement_point + 1.272 * diff,
        "1.618": retracement_point + 1.618 * diff,
        "2.0": retracement_point + 2.0 * diff,
        "2.618": retracement_point + 2.618 * diff,
    }


# ═══════════════════════════════════════════════════════════════
# 8. CANDLESTICK PATTERNS
# ═══════════════════════════════════════════════════════════════

def _body(open_: pd.Series, close: pd.Series) -> pd.Series:
    return (close - open_).abs()


def _upper_shadow(open_: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
    return high - pd.concat([open_, close], axis=1).max(axis=1)


def _lower_shadow(open_: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat([open_, close], axis=1).min(axis=1) - low


def _candle_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - low


def pattern_doji(open_: pd.Series, high: pd.Series, low: pd.Series,
                 close: pd.Series, threshold: float = 0.05) -> pd.Series:
    """Doji: body < threshold * range."""
    body = _body(open_, close)
    cr = _candle_range(high, low).replace(0, np.nan)
    return (body / cr < threshold).astype(int)


def pattern_hammer(open_: pd.Series, high: pd.Series, low: pd.Series,
                   close: pd.Series) -> pd.Series:
    """Hammer: small body at top, long lower shadow >= 2x body."""
    body = _body(open_, close)
    ls = _lower_shadow(open_, low, close)
    us = _upper_shadow(open_, high, close)
    cr = _candle_range(high, low)
    body_safe = body.replace(0, np.nan)
    return ((ls >= 2.0 * body) & (us < body) & (body > 0) & (cr > 0)).astype(int)


def pattern_shooting_star(open_: pd.Series, high: pd.Series, low: pd.Series,
                          close: pd.Series) -> pd.Series:
    """Shooting Star: small body at bottom, long upper shadow >= 2x body."""
    body = _body(open_, close)
    ls = _lower_shadow(open_, low, close)
    us = _upper_shadow(open_, high, close)
    return ((us >= 2.0 * body) & (ls < body) & (body > 0)).astype(int)


def pattern_engulfing(open_: pd.Series, high: pd.Series, low: pd.Series,
                      close: pd.Series) -> pd.Series:
    """Engulfing: +1 = bullish engulfing, -1 = bearish engulfing, 0 = none."""
    n = len(open_)
    result = np.zeros(n, dtype=int)
    o = open_.values
    c = close.values

    for i in range(1, n):
        prev_body = c[i - 1] - o[i - 1]
        curr_body = c[i] - o[i]

        # Bullish engulfing: prev bearish, curr bullish, curr body engulfs prev
        if prev_body < 0 and curr_body > 0:
            if o[i] <= c[i - 1] and c[i] >= o[i - 1]:
                result[i] = 1

        # Bearish engulfing: prev bullish, curr bearish, curr body engulfs prev
        if prev_body > 0 and curr_body < 0:
            if o[i] >= c[i - 1] and c[i] <= o[i - 1]:
                result[i] = -1

    return pd.Series(result, index=open_.index)


def pattern_pin_bar(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series, shadow_ratio: float = 2.5) -> pd.Series:
    """Pin Bar: +1 = bullish, -1 = bearish. Long wick >= shadow_ratio * body."""
    body = _body(open_, close)
    us = _upper_shadow(open_, high, close)
    ls = _lower_shadow(open_, low, close)
    cr = _candle_range(high, low)

    bullish = (ls >= shadow_ratio * body) & (us < 0.3 * cr) & (body > 0) & (cr > 0)
    bearish = (us >= shadow_ratio * body) & (ls < 0.3 * cr) & (body > 0) & (cr > 0)

    result = pd.Series(0, index=open_.index)
    result[bullish] = 1
    result[bearish] = -1
    return result


def pattern_inside_bar(high: pd.Series, low: pd.Series) -> pd.Series:
    """Inside Bar: current bar's range is within previous bar."""
    return ((high <= high.shift(1)) & (low >= low.shift(1))).astype(int)


def pattern_outside_bar(high: pd.Series, low: pd.Series) -> pd.Series:
    """Outside Bar: current bar engulfs previous bar's range."""
    return ((high > high.shift(1)) & (low < low.shift(1))).astype(int)


def pattern_morning_star(open_: pd.Series, high: pd.Series, low: pd.Series,
                         close: pd.Series) -> pd.Series:
    """Morning Star (3-bar bullish reversal)."""
    n = len(open_)
    result = np.zeros(n, dtype=int)
    o = open_.values
    c = close.values
    h = high.values
    l = low.values

    for i in range(2, n):
        bar1_bearish = c[i - 2] < o[i - 2]
        bar1_body = abs(c[i - 2] - o[i - 2])
        bar2_small = abs(c[i - 1] - o[i - 1]) < bar1_body * 0.3
        bar3_bullish = c[i] > o[i]
        bar3_closes_above = c[i] > (o[i - 2] + c[i - 2]) / 2.0

        if bar1_bearish and bar2_small and bar3_bullish and bar3_closes_above:
            result[i] = 1

    return pd.Series(result, index=open_.index)


def pattern_evening_star(open_: pd.Series, high: pd.Series, low: pd.Series,
                         close: pd.Series) -> pd.Series:
    """Evening Star (3-bar bearish reversal)."""
    n = len(open_)
    result = np.zeros(n, dtype=int)
    o = open_.values
    c = close.values

    for i in range(2, n):
        bar1_bullish = c[i - 2] > o[i - 2]
        bar1_body = abs(c[i - 2] - o[i - 2])
        bar2_small = abs(c[i - 1] - o[i - 1]) < bar1_body * 0.3
        bar3_bearish = c[i] < o[i]
        bar3_closes_below = c[i] < (o[i - 2] + c[i - 2]) / 2.0

        if bar1_bullish and bar2_small and bar3_bearish and bar3_closes_below:
            result[i] = 1

    return pd.Series(result, index=open_.index)


def pattern_three_white_soldiers(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Three White Soldiers (3 consecutive bullish bars with higher closes)."""
    n = len(open_)
    result = np.zeros(n, dtype=int)
    o = open_.values
    c = close.values

    for i in range(2, n):
        bull1 = c[i - 2] > o[i - 2]
        bull2 = c[i - 1] > o[i - 1]
        bull3 = c[i] > o[i]
        higher = c[i] > c[i - 1] > c[i - 2]
        opens_in_body = o[i - 1] > o[i - 2] and o[i] > o[i - 1]

        if bull1 and bull2 and bull3 and higher and opens_in_body:
            result[i] = 1

    return pd.Series(result, index=open_.index)


def pattern_three_black_crows(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Three Black Crows (3 consecutive bearish bars with lower closes)."""
    n = len(open_)
    result = np.zeros(n, dtype=int)
    o = open_.values
    c = close.values

    for i in range(2, n):
        bear1 = c[i - 2] < o[i - 2]
        bear2 = c[i - 1] < o[i - 1]
        bear3 = c[i] < o[i]
        lower = c[i] < c[i - 1] < c[i - 2]
        opens_in_body = o[i - 1] < o[i - 2] and o[i] < o[i - 1]

        if bear1 and bear2 and bear3 and lower and opens_in_body:
            result[i] = 1

    return pd.Series(result, index=open_.index)


def pattern_tweezer(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series, tolerance: float = 0.001) -> pd.Series:
    """Tweezer Top (+1) / Tweezer Bottom (-1). Equal highs/lows within tolerance."""
    n = len(open_)
    result = np.zeros(n, dtype=int)
    h = high.values
    l = low.values
    o = open_.values
    c = close.values

    for i in range(1, n):
        price_avg = (h[i] + l[i]) / 2.0
        tol = price_avg * tolerance

        # Tweezer Top: equal highs, first bullish second bearish
        if abs(h[i] - h[i - 1]) < tol:
            if c[i - 1] > o[i - 1] and c[i] < o[i]:
                result[i] = 1

        # Tweezer Bottom: equal lows, first bearish second bullish
        if abs(l[i] - l[i - 1]) < tol:
            if c[i - 1] < o[i - 1] and c[i] > o[i]:
                result[i] = -1

    return pd.Series(result, index=open_.index)


# ═══════════════════════════════════════════════════════════════
# COMPUTE ALL — Master function
# ═══════════════════════════════════════════════════════════════

def compute_all(df: pd.DataFrame, timeframe: str = "H1") -> Dict[str, Any]:
    """
    Compute ALL indicators in one pass.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, tick_volume
        timeframe: Timeframe name (for logging only)

    Returns:
        Dict mapping indicator names to Series/dicts:
            "rsi_14": Series, "macd_12_26_9": dict, etc.

    Uses caching: if called with same data hash, returns cached result.
    """
    # --- Cache check ---
    cache_key = _data_hash(df)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    t0 = _time.time()

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df.get("tick_volume", df.get("volume", pd.Series(0, index=df.index)))

    result: Dict[str, Any] = {}

    # ─── 1. Moving Averages ───
    for p in [5, 8, 10, 13, 20, 21, 26, 34, 50, 100, 200]:
        result[f"sma_{p}"] = sma(c, p)
        result[f"ema_{p}"] = ema(c, p)

    for p in [10, 20, 50]:
        result[f"wma_{p}"] = wma(c, p)

    result["dema_20"] = dema(c, 20)
    result["tema_20"] = tema(c, 20)
    result["kama_10"] = kama(c, 10)
    result["hull_ma_20"] = hull_ma(c, 20)
    result["hull_ma_9"] = hull_ma(c, 9)

    # ─── 2. Oscillators ───
    for p in [7, 14, 21]:
        result[f"rsi_{p}"] = rsi(c, p)

    result["stoch_14_3"] = stochastic(h, l, c, 14, 3)
    result["stoch_5_3"] = stochastic(h, l, c, 5, 3)
    result["stoch_rsi_14"] = stoch_rsi(c, 14, 14, 3, 3)
    result["williams_r_14"] = williams_r(h, l, c, 14)
    result["cci_20"] = cci(h, l, c, 20)
    result["cci_14"] = cci(h, l, c, 14)
    result["roc_12"] = roc(c, 12)
    result["roc_9"] = roc(c, 9)
    result["momentum_10"] = momentum(c, 10)
    result["momentum_14"] = momentum(c, 14)
    result["ultimate_osc_7_14_28"] = ultimate_oscillator(h, l, c, 7, 14, 28)

    # ─── 3. MACD Family ───
    result["macd_12_26_9"] = macd(c, 12, 26, 9)
    result["macd_8_21_5"] = macd(c, 8, 21, 5)
    result["macd_5_13_3"] = macd(c, 5, 13, 3)
    result["trix_15_9"] = trix(c, 15, 9)
    result["tsi_25_13_7"] = tsi(c, 25, 13, 7)
    result["awesome_osc_5_34"] = awesome_oscillator(h, l, 5, 34)

    # ─── 4. Volatility ───
    result["atr_14"] = atr(h, l, c, 14)
    result["atr_7"] = atr(h, l, c, 7)
    result["atr_20"] = atr(h, l, c, 20)
    result["true_range"] = true_range(h, l, c)

    result["bb_20_2"] = bollinger_bands(c, 20, 2.0)
    result["bb_20_1"] = bollinger_bands(c, 20, 1.0)

    result["keltner_20_14_2"] = keltner_channel(h, l, c, 20, 14, 2.0)
    result["donchian_20"] = donchian_channel(h, l, 20)
    result["donchian_10"] = donchian_channel(h, l, 10)

    result["supertrend_10_3"] = supertrend(h, l, c, 10, 3.0)
    result["supertrend_7_2"] = supertrend(h, l, c, 7, 2.0)

    result["psar_002"] = parabolic_sar(h, l, 0.02, 0.02, 0.20)

    result["squeeze_20"] = squeeze_momentum(h, l, c, 20, 2.0, 20, 1.5)

    # ─── 5. Volume ───
    result["obv"] = obv(c, v)
    result["mfi_14"] = mfi(h, l, c, v, 14)
    result["cmf_20"] = cmf(h, l, c, v, 20)
    result["chaikin_osc_3_10"] = chaikin_oscillator(h, l, c, v, 3, 10)
    result["vol_osc_5_20"] = volume_oscillator(v, 5, 20)
    result["vwap"] = vwap(h, l, c, v)
    result["vol_sma_20"] = volume_sma(v, 20)
    result["vol_sma_50"] = volume_sma(v, 50)

    # ─── 6. Trend ───
    result["adx_14"] = adx_dmi(h, l, c, 14)
    result["adx_7"] = adx_dmi(h, l, c, 7)
    result["aroon_25"] = aroon(h, l, 25)
    result["aroon_14"] = aroon(h, l, 14)
    result["vortex_14"] = vortex(h, l, c, 14)

    result["ichimoku"] = ichimoku(h, l, c, 9, 26, 52, 26)
    result["heikin_ashi"] = heikin_ashi(o, h, l, c)

    # ─── 7. Structure / Statistical ───
    result["linreg_20"] = linear_regression(c, 20)
    result["linreg_50"] = linear_regression(c, 50)
    result["z_score_20"] = z_score(c, 20)
    result["z_score_50"] = z_score(c, 50)

    # Hurst is expensive; compute on close only with reasonable window
    result["hurst_100"] = hurst_exponent(c, 100)

    # Pivot points from previous day (use last bar of daily-equivalent data)
    # For H1, we use previous 24-bar window's HLC
    pivot_period = 24 if timeframe == "H1" else 1
    if len(df) > pivot_period:
        ph = h.iloc[-pivot_period - 1] if len(df) > pivot_period else h.iloc[-1]
        pl = l.iloc[-pivot_period - 1] if len(df) > pivot_period else l.iloc[-1]
        pc = c.iloc[-pivot_period - 1] if len(df) > pivot_period else c.iloc[-1]
        result["pivot_classic"] = pivot_points_classic(ph, pl, pc)
        result["pivot_fibonacci"] = pivot_points_fibonacci(ph, pl, pc)
        result["pivot_camarilla"] = pivot_points_camarilla(ph, pl, pc)
        result["pivot_woodie"] = pivot_points_woodie(ph, pl, pc)

    # Fibonacci retracements from recent swing high/low
    lookback = min(100, len(df))
    recent_high = h.iloc[-lookback:].max()
    recent_low = l.iloc[-lookback:].min()
    result["fib_retracement"] = fibonacci_retracement(recent_high, recent_low)
    result["fib_extension"] = fibonacci_extension(recent_high, recent_low, recent_low)

    # ─── 8. Candlestick Patterns ───
    result["pattern_doji"] = pattern_doji(o, h, l, c)
    result["pattern_hammer"] = pattern_hammer(o, h, l, c)
    result["pattern_shooting_star"] = pattern_shooting_star(o, h, l, c)
    result["pattern_engulfing"] = pattern_engulfing(o, h, l, c)
    result["pattern_pin_bar"] = pattern_pin_bar(o, h, l, c)
    result["pattern_inside_bar"] = pattern_inside_bar(h, l)
    result["pattern_outside_bar"] = pattern_outside_bar(h, l)
    result["pattern_morning_star"] = pattern_morning_star(o, h, l, c)
    result["pattern_evening_star"] = pattern_evening_star(o, h, l, c)
    result["pattern_three_white_soldiers"] = pattern_three_white_soldiers(o, c)
    result["pattern_three_black_crows"] = pattern_three_black_crows(o, c)
    result["pattern_tweezer"] = pattern_tweezer(o, h, l, c)

    elapsed = _time.time() - t0

    # Count total indicators
    count = 0
    for key, val in result.items():
        if isinstance(val, dict):
            count += len(val)
        else:
            count += 1

    result["_meta"] = {
        "indicator_count": count,
        "computation_time_sec": round(elapsed, 3),
        "timeframe": timeframe,
        "bars": len(df),
        "cache_key": cache_key,
    }

    # Store in cache
    _CACHE[cache_key] = result

    return result

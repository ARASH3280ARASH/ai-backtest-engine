"""
Phase 8 — Parameter Optimizer with Walk-Forward Analysis
==========================================================
Optimizes entry-signal parameters for the top 50 strategies from Phase 5.
Uses anchored expanding-window walk-forward to avoid overfitting.

Two tiers:
  Tier 1 — Full optimization via parameterized signal generators
  Tier 2 — SL/TP-only optimization using cached Phase 5 signals

Search methods: Grid (small spaces), Random (large spaces), Bayesian (optuna).
"""

import json
import math
import os
import sys
import time
import itertools
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Project imports
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.broker import BTCUSD_CONFIG
from engine.costs import compute_variable_cost
from config.settings import (
    TRAIN_DIR, RESULTS_DIR, INDIVIDUAL_DIR, OPTIMIZED_DIR,
)
from indicators import compute as ind

logger = logging.getLogger("phase8")

# ═══════════════════════════════════════════════════════════════
#  A) PARAMETER SPACES
# ═══════════════════════════════════════════════════════════════

PARAM_SPACES: Dict[str, List[Dict]] = {
    "RSI": [
        {"name": "period", "type": "int", "low": 7, "high": 28, "step": 3, "default": 14},
        {"name": "ob", "type": "int", "low": 60, "high": 85, "step": 5, "default": 70},
        {"name": "os", "type": "int", "low": 15, "high": 40, "step": 5, "default": 30},
    ],
    "MACD": [
        {"name": "fast", "type": "int", "low": 5, "high": 20, "step": 3, "default": 12},
        {"name": "slow", "type": "int", "low": 15, "high": 40, "step": 5, "default": 26},
        {"name": "signal", "type": "int", "low": 3, "high": 15, "step": 2, "default": 9},
    ],
    "BB": [
        {"name": "period", "type": "int", "low": 10, "high": 40, "step": 5, "default": 20},
        {"name": "std_dev", "type": "float", "low": 1.0, "high": 3.0, "step": 0.25, "default": 2.0},
    ],
    "MA": [
        {"name": "fast", "type": "int", "low": 5, "high": 30, "step": 5, "default": 10},
        {"name": "slow", "type": "int", "low": 20, "high": 100, "step": 10, "default": 50},
    ],
    "ADX": [
        {"name": "period", "type": "int", "low": 7, "high": 28, "step": 3, "default": 14},
        {"name": "threshold", "type": "int", "low": 15, "high": 35, "step": 5, "default": 25},
    ],
    "WR": [
        {"name": "period", "type": "int", "low": 7, "high": 28, "step": 3, "default": 14},
        {"name": "ob", "type": "int", "low": -30, "high": -10, "step": 5, "default": -20},
        {"name": "os", "type": "int", "low": -90, "high": -70, "step": 5, "default": -80},
    ],
    "STOCH": [
        {"name": "k_period", "type": "int", "low": 5, "high": 21, "step": 4, "default": 14},
        {"name": "d_period", "type": "int", "low": 3, "high": 7, "step": 2, "default": 3},
        {"name": "ob", "type": "int", "low": 70, "high": 85, "step": 5, "default": 80},
        {"name": "os", "type": "int", "low": 15, "high": 30, "step": 5, "default": 20},
    ],
    "CCI": [
        {"name": "period", "type": "int", "low": 10, "high": 30, "step": 5, "default": 20},
        {"name": "ob", "type": "int", "low": 80, "high": 150, "step": 10, "default": 100},
        {"name": "os", "type": "int", "low": -150, "high": -80, "step": 10, "default": -100},
    ],
    "SRSI": [
        {"name": "rsi_period", "type": "int", "low": 7, "high": 21, "step": 7, "default": 14},
        {"name": "stoch_period", "type": "int", "low": 7, "high": 21, "step": 7, "default": 14},
        {"name": "ob", "type": "int", "low": 70, "high": 85, "step": 5, "default": 80},
        {"name": "os", "type": "int", "low": 15, "high": 30, "step": 5, "default": 20},
    ],
    "KC": [
        {"name": "ema_period", "type": "int", "low": 10, "high": 30, "step": 5, "default": 20},
        {"name": "atr_period", "type": "int", "low": 7, "high": 21, "step": 7, "default": 14},
        {"name": "mult", "type": "float", "low": 1.0, "high": 3.0, "step": 0.5, "default": 2.0},
    ],
    "DON": [
        {"name": "period", "type": "int", "low": 10, "high": 40, "step": 5, "default": 20},
    ],
    "ENV": [
        {"name": "period", "type": "int", "low": 10, "high": 40, "step": 5, "default": 20},
        {"name": "pct", "type": "float", "low": 0.5, "high": 3.0, "step": 0.5, "default": 1.0},
    ],
    "OBV": [
        {"name": "fast", "type": "int", "low": 5, "high": 15, "step": 5, "default": 10},
        {"name": "slow", "type": "int", "low": 20, "high": 50, "step": 10, "default": 30},
    ],
    "TSI": [
        {"name": "long_p", "type": "int", "low": 15, "high": 35, "step": 5, "default": 25},
        {"name": "short_p", "type": "int", "low": 7, "high": 19, "step": 3, "default": 13},
        {"name": "signal_p", "type": "int", "low": 5, "high": 11, "step": 3, "default": 7},
    ],
    "ULT": [
        {"name": "p1", "type": "int", "low": 5, "high": 10, "step": 1, "default": 7},
        {"name": "p2", "type": "int", "low": 10, "high": 20, "step": 2, "default": 14},
        {"name": "p3", "type": "int", "low": 20, "high": 35, "step": 5, "default": 28},
    ],
    "FISHER": [
        {"name": "period", "type": "int", "low": 5, "high": 21, "step": 4, "default": 9},
    ],
    "ARN": [
        {"name": "period", "type": "int", "low": 10, "high": 40, "step": 5, "default": 25},
    ],
    "VTX": [
        {"name": "period", "type": "int", "low": 7, "high": 28, "step": 3, "default": 14},
    ],
    "RVI": [
        {"name": "period", "type": "int", "low": 7, "high": 21, "step": 7, "default": 10},
    ],
    "REG": [
        {"name": "period", "type": "int", "low": 10, "high": 40, "step": 5, "default": 20},
    ],
    "ROC": [
        {"name": "period", "type": "int", "low": 5, "high": 20, "step": 3, "default": 12},
    ],
    "KST": [
        {"name": "r1", "type": "int", "low": 5, "high": 15, "step": 5, "default": 10},
        {"name": "r2", "type": "int", "low": 10, "high": 20, "step": 5, "default": 15},
        {"name": "r3", "type": "int", "low": 15, "high": 25, "step": 5, "default": 20},
        {"name": "r4", "type": "int", "low": 25, "high": 35, "step": 5, "default": 30},
    ],
    "PSAR": [
        {"name": "af_start", "type": "float", "low": 0.01, "high": 0.04, "step": 0.01, "default": 0.02},
        {"name": "af_step", "type": "float", "low": 0.01, "high": 0.04, "step": 0.01, "default": 0.02},
        {"name": "af_max", "type": "float", "low": 0.10, "high": 0.30, "step": 0.05, "default": 0.20},
    ],
    # SL/TP-only space for Tier 2
    "SLTP": [
        {"name": "sl_mult", "type": "float", "low": 1.0, "high": 3.0, "step": 0.25, "default": 1.5},
        {"name": "tp_mult", "type": "float", "low": 1.5, "high": 5.0, "step": 0.5, "default": 2.25},
        {"name": "atr_period", "type": "int", "low": 7, "high": 28, "step": 7, "default": 14},
    ],
}

# Categories that get full Tier 1 optimization
TIER1_CATEGORIES = {
    "RSI", "MACD", "BB", "MA", "ADX", "WR", "FISHER", "ARN", "VTX",
    "ULT", "KC", "DON", "ENV", "OBV", "TSI", "RVI", "REG", "ROC",
    "KST", "STOCH", "CCI", "SRSI", "PSAR",
}

# Categories that get Tier 2 (SL/TP only) optimization
TIER2_CATEGORIES = {
    "SNT", "EW", "CH", "COMBO", "CDL", "MASS", "SM", "WYC", "FI",
    "AD", "RANGE", "VOL", "HEIKIN", "ELDER", "STAT", "PA", "GAPS",
    "PIVOT_ADV", "ICH", "DIV", "MFI", "VORTEX", "CP", "PPO", "MS",
    "CHOP", "CMO", "ADP", "AIC", "ALLI", "ATR",
}


def _get_category(strategy_id: str) -> str:
    """Extract category prefix from strategy id (e.g. 'RSI_05' -> 'RSI')."""
    parts = strategy_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    # Handle multi-part like ADX_ADV_01
    for i in range(len(strategy_id) - 1, 0, -1):
        if strategy_id[i] == "_" and strategy_id[i + 1:].isdigit():
            return strategy_id[:i]
    return strategy_id


def get_param_space(strategy_id: str) -> Tuple[str, List[Dict]]:
    """Return (tier, param_space) for a strategy. tier='tier1' or 'tier2'."""
    cat = _get_category(strategy_id)
    if cat in TIER1_CATEGORIES and cat in PARAM_SPACES:
        return "tier1", PARAM_SPACES[cat]
    return "tier2", PARAM_SPACES["SLTP"]


# ═══════════════════════════════════════════════════════════════
#  B) SIGNAL GENERATORS (Tier 1)
# ═══════════════════════════════════════════════════════════════

def gen_rsi_signals(closes: pd.Series, period=14, ob=70, os_=30) -> np.ndarray:
    """RSI threshold cross: buy when RSI crosses below os, sell above ob."""
    r = ind.rsi(closes, period)
    vals = r.values
    n = len(vals)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]):
            continue
        if vals[i - 1] >= os_ and vals[i] < os_:
            signals[i] = 1
        elif vals[i - 1] <= ob and vals[i] > ob:
            signals[i] = -1
    return signals


def gen_rsi_midline_signals(closes: pd.Series, period=14) -> np.ndarray:
    """RSI midline cross: buy above 50, sell below 50."""
    r = ind.rsi(closes, period)
    vals = r.values
    n = len(vals)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]):
            continue
        if vals[i - 1] <= 50 and vals[i] > 50:
            signals[i] = 1
        elif vals[i - 1] >= 50 and vals[i] < 50:
            signals[i] = -1
    return signals


def gen_macd_signals(closes: pd.Series, fast=12, slow=26, signal=9) -> np.ndarray:
    """MACD signal-line crossover."""
    m = ind.macd(closes, fast, slow, signal)
    ml = m["line"].values
    ms = m["signal"].values
    n = len(ml)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(ml[i]) or np.isnan(ms[i]) or np.isnan(ml[i - 1]) or np.isnan(ms[i - 1]):
            continue
        if ml[i - 1] <= ms[i - 1] and ml[i] > ms[i]:
            signals[i] = 1
        elif ml[i - 1] >= ms[i - 1] and ml[i] < ms[i]:
            signals[i] = -1
    return signals


def gen_bb_signals(closes: pd.Series, period=20, std_dev=2.0) -> np.ndarray:
    """Bollinger Band bounce: buy at lower, sell at upper."""
    bb = ind.bollinger_bands(closes, period, std_dev)
    upper = bb["upper"].values
    lower = bb["lower"].values
    c = closes.values
    n = len(c)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if c[i - 1] >= lower[i - 1] and c[i] < lower[i]:
            signals[i] = 1
        elif c[i - 1] <= upper[i - 1] and c[i] > upper[i]:
            signals[i] = -1
    return signals


def gen_ma_signals(closes: pd.Series, fast=10, slow=50) -> np.ndarray:
    """MA crossover: buy when fast crosses above slow."""
    f = ind.ema(closes, fast).values
    s = ind.ema(closes, slow).values
    n = len(f)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(f[i]) or np.isnan(s[i]) or np.isnan(f[i - 1]) or np.isnan(s[i - 1]):
            continue
        if f[i - 1] <= s[i - 1] and f[i] > s[i]:
            signals[i] = 1
        elif f[i - 1] >= s[i - 1] and f[i] < s[i]:
            signals[i] = -1
    return signals


def gen_adx_signals(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                    period=14, threshold=25) -> np.ndarray:
    """ADX + DI crossover: buy +DI > -DI when ADX > threshold."""
    dmi = ind.adx_dmi(highs, lows, closes, period)
    adx = dmi["adx"].values
    pdi = dmi["plus_di"].values
    mdi = dmi["minus_di"].values
    n = len(adx)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(adx[i]) or np.isnan(pdi[i]) or np.isnan(mdi[i]):
            continue
        if adx[i] < threshold:
            continue
        if pdi[i - 1] <= mdi[i - 1] and pdi[i] > mdi[i]:
            signals[i] = 1
        elif pdi[i - 1] >= mdi[i - 1] and pdi[i] < mdi[i]:
            signals[i] = -1
    return signals


def gen_wr_signals(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                   period=14, ob=-20, os_=-80) -> np.ndarray:
    """Williams %R threshold cross."""
    wr = ind.williams_r(highs, lows, closes, period).values
    n = len(wr)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(wr[i]) or np.isnan(wr[i - 1]):
            continue
        if wr[i - 1] >= os_ and wr[i] < os_:
            signals[i] = 1
        elif wr[i - 1] <= ob and wr[i] > ob:
            signals[i] = -1
    return signals


def gen_stoch_signals(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                      k_period=14, d_period=3, ob=80, os_=20) -> np.ndarray:
    """Stochastic %K/%D crossover in OB/OS zones."""
    st = ind.stochastic(highs, lows, closes, k_period, d_period)
    k = st["k"].values
    d = st["d"].values
    n = len(k)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(k[i - 1]) or np.isnan(d[i - 1]):
            continue
        if k[i - 1] <= d[i - 1] and k[i] > d[i] and k[i] < os_:
            signals[i] = 1
        elif k[i - 1] >= d[i - 1] and k[i] < d[i] and k[i] > ob:
            signals[i] = -1
    return signals


def gen_cci_signals(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                    period=20, ob=100, os_=-100) -> np.ndarray:
    """CCI threshold cross."""
    c = ind.cci(highs, lows, closes, period).values
    n = len(c)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(c[i]) or np.isnan(c[i - 1]):
            continue
        if c[i - 1] >= os_ and c[i] < os_:
            signals[i] = 1
        elif c[i - 1] <= ob and c[i] > ob:
            signals[i] = -1
    return signals


def gen_srsi_signals(closes: pd.Series, rsi_period=14, stoch_period=14,
                     ob=80, os_=20) -> np.ndarray:
    """Stochastic RSI crossover in OB/OS zones."""
    sr = ind.stoch_rsi(closes, rsi_period, stoch_period)
    k = sr["k"].values
    d = sr["d"].values
    n = len(k)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(k[i - 1]) or np.isnan(d[i - 1]):
            continue
        if k[i - 1] <= d[i - 1] and k[i] > d[i] and k[i] < os_:
            signals[i] = 1
        elif k[i - 1] >= d[i - 1] and k[i] < d[i] and k[i] > ob:
            signals[i] = -1
    return signals


def gen_kc_signals(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                   ema_period=20, atr_period=14, mult=2.0) -> np.ndarray:
    """Keltner Channel breakout: buy above upper, sell below lower."""
    kc = ind.keltner_channel(highs, lows, closes, ema_period, atr_period, mult)
    upper = kc["upper"].values
    lower = kc["lower"].values
    c = closes.values
    n = len(c)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if c[i - 1] <= upper[i - 1] and c[i] > upper[i]:
            signals[i] = 1
        elif c[i - 1] >= lower[i - 1] and c[i] < lower[i]:
            signals[i] = -1
    return signals


def gen_don_signals(highs: pd.Series, lows: pd.Series, period=20) -> np.ndarray:
    """Donchian Channel breakout."""
    dc = ind.donchian_channel(highs, lows, period)
    upper = dc["upper"].values
    lower = dc["lower"].values
    c = highs.values  # Use close for comparison via high/low for breakout
    cl = lows.values
    n = len(upper)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if np.isnan(upper[i - 1]) or np.isnan(lower[i - 1]):
            continue
        if c[i] > upper[i - 1]:
            signals[i] = 1
        elif cl[i] < lower[i - 1]:
            signals[i] = -1
    return signals


def gen_env_signals(closes: pd.Series, period=20, pct=1.0) -> np.ndarray:
    """Envelope (MA ± pct%) bounce."""
    ma = ind.sma(closes, period).values
    c = closes.values
    n = len(c)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(ma[i]):
            continue
        upper = ma[i] * (1 + pct / 100.0)
        lower = ma[i] * (1 - pct / 100.0)
        if c[i] < lower and c[i - 1] >= ma[i - 1] * (1 - pct / 100.0) if not np.isnan(ma[i - 1]) else False:
            signals[i] = 1
        elif c[i] > upper and c[i - 1] <= ma[i - 1] * (1 + pct / 100.0) if not np.isnan(ma[i - 1]) else False:
            signals[i] = -1
    return signals


def gen_obv_signals(closes: pd.Series, volume: pd.Series,
                    fast=10, slow=30) -> np.ndarray:
    """OBV MA crossover."""
    obv = ind.obv(closes, volume)
    f = ind.ema(obv, fast).values
    s = ind.ema(obv, slow).values
    n = len(f)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(f[i]) or np.isnan(s[i]) or np.isnan(f[i - 1]) or np.isnan(s[i - 1]):
            continue
        if f[i - 1] <= s[i - 1] and f[i] > s[i]:
            signals[i] = 1
        elif f[i - 1] >= s[i - 1] and f[i] < s[i]:
            signals[i] = -1
    return signals


def gen_tsi_signals(closes: pd.Series, long_p=25, short_p=13,
                    signal_p=7) -> np.ndarray:
    """TSI signal-line crossover."""
    t = ind.tsi(closes, long_p, short_p, signal_p)
    tl = t["line"].values
    ts = t["signal"].values
    n = len(tl)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(tl[i]) or np.isnan(ts[i]) or np.isnan(tl[i - 1]) or np.isnan(ts[i - 1]):
            continue
        if tl[i - 1] <= ts[i - 1] and tl[i] > ts[i]:
            signals[i] = 1
        elif tl[i - 1] >= ts[i - 1] and tl[i] < ts[i]:
            signals[i] = -1
    return signals


def gen_ult_signals(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                    p1=7, p2=14, p3=28) -> np.ndarray:
    """Ultimate Oscillator cross 50."""
    uo = ind.ultimate_oscillator(highs, lows, closes, p1, p2, p3).values
    n = len(uo)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(uo[i]) or np.isnan(uo[i - 1]):
            continue
        if uo[i - 1] <= 50 and uo[i] > 50:
            signals[i] = 1
        elif uo[i - 1] >= 50 and uo[i] < 50:
            signals[i] = -1
    return signals


def gen_fisher_signals(highs: pd.Series, lows: pd.Series,
                       period=9) -> np.ndarray:
    """Fisher Transform cross (using Aroon as proxy for fisher logic)."""
    # Fisher Transform: compute manually
    h = highs.values.astype(np.float64)
    lo = lows.values.astype(np.float64)
    n = len(h)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    val = 0.0
    fish_prev = 0.0

    for i in range(period, n):
        hh = np.max(h[i - period + 1:i + 1])
        ll = np.min(lo[i - period + 1:i + 1])
        denom = hh - ll
        if denom == 0:
            raw = 0.0
        else:
            raw = 2.0 * ((h[i] + lo[i]) / 2.0 - ll) / denom - 1.0
        raw = max(-0.999, min(0.999, raw))
        val = 0.5 * val + 0.5 * raw
        val = max(-0.999, min(0.999, val))
        fish = 0.5 * math.log((1 + val) / (1 - val)) if abs(val) < 0.999 else fish_prev
        trigger[i] = fish_prev
        fisher[i] = fish
        fish_prev = fish

    signals = np.zeros(n, dtype=np.int8)
    for i in range(period + 1, n):
        if fisher[i - 1] <= trigger[i - 1] and fisher[i] > trigger[i]:
            signals[i] = 1
        elif fisher[i - 1] >= trigger[i - 1] and fisher[i] < trigger[i]:
            signals[i] = -1
    return signals


def gen_aroon_signals(highs: pd.Series, lows: pd.Series,
                      period=25) -> np.ndarray:
    """Aroon crossover."""
    ar = ind.aroon(highs, lows, period)
    up = ar["up"].values
    dn = ar["down"].values
    n = len(up)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(up[i]) or np.isnan(dn[i]) or np.isnan(up[i - 1]) or np.isnan(dn[i - 1]):
            continue
        if up[i - 1] <= dn[i - 1] and up[i] > dn[i]:
            signals[i] = 1
        elif up[i - 1] >= dn[i - 1] and up[i] < dn[i]:
            signals[i] = -1
    return signals


def gen_vtx_signals(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                    period=14) -> np.ndarray:
    """Vortex crossover."""
    vx = ind.vortex(highs, lows, closes, period)
    vp = vx["plus"].values
    vm = vx["minus"].values
    n = len(vp)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(vp[i]) or np.isnan(vm[i]) or np.isnan(vp[i - 1]) or np.isnan(vm[i - 1]):
            continue
        if vp[i - 1] <= vm[i - 1] and vp[i] > vm[i]:
            signals[i] = 1
        elif vp[i - 1] >= vm[i - 1] and vp[i] < vm[i]:
            signals[i] = -1
    return signals


def gen_rvi_signals(opens: pd.Series, highs: pd.Series, lows: pd.Series,
                    closes: pd.Series, period=10) -> np.ndarray:
    """RVI: (close-open)/(high-low) smoothed, cross zero."""
    hl = (highs - lows).replace(0, np.nan)
    co = closes - opens
    rvi_raw = co / hl
    rvi_smooth = ind.sma(rvi_raw, period).values
    n = len(rvi_smooth)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(rvi_smooth[i]) or np.isnan(rvi_smooth[i - 1]):
            continue
        if rvi_smooth[i - 1] <= 0 and rvi_smooth[i] > 0:
            signals[i] = 1
        elif rvi_smooth[i - 1] >= 0 and rvi_smooth[i] < 0:
            signals[i] = -1
    return signals


def gen_reg_signals(closes: pd.Series, period=20) -> np.ndarray:
    """Linear Regression: buy when slope turns positive, sell negative."""
    lr = ind.linear_regression(closes, period)
    slope = lr["slope"].values
    n = len(slope)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(slope[i]) or np.isnan(slope[i - 1]):
            continue
        if slope[i - 1] <= 0 and slope[i] > 0:
            signals[i] = 1
        elif slope[i - 1] >= 0 and slope[i] < 0:
            signals[i] = -1
    return signals


def gen_roc_signals(closes: pd.Series, period=12) -> np.ndarray:
    """ROC zero cross."""
    r = ind.roc(closes, period).values
    n = len(r)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(r[i]) or np.isnan(r[i - 1]):
            continue
        if r[i - 1] <= 0 and r[i] > 0:
            signals[i] = 1
        elif r[i - 1] >= 0 and r[i] < 0:
            signals[i] = -1
    return signals


def gen_kst_signals(closes: pd.Series, r1=10, r2=15, r3=20, r4=30) -> np.ndarray:
    """KST (sum of weighted ROCs) signal-line cross."""
    roc1 = ind.roc(closes, r1)
    roc2 = ind.roc(closes, r2)
    roc3 = ind.roc(closes, r3)
    roc4 = ind.roc(closes, r4)
    kst = ind.sma(roc1, 10) * 1 + ind.sma(roc2, 10) * 2 + \
          ind.sma(roc3, 10) * 3 + ind.sma(roc4, 15) * 4
    sig = ind.sma(kst, 9)
    kv = kst.values
    sv = sig.values
    n = len(kv)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(kv[i]) or np.isnan(sv[i]) or np.isnan(kv[i - 1]) or np.isnan(sv[i - 1]):
            continue
        if kv[i - 1] <= sv[i - 1] and kv[i] > sv[i]:
            signals[i] = 1
        elif kv[i - 1] >= sv[i - 1] and kv[i] < sv[i]:
            signals[i] = -1
    return signals


def gen_psar_signals(highs: pd.Series, lows: pd.Series,
                     af_start=0.02, af_step=0.02, af_max=0.20) -> np.ndarray:
    """Parabolic SAR trend flip."""
    ps = ind.parabolic_sar(highs, lows, af_start, af_step, af_max)
    trend = ps["trend"].values
    n = len(trend)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if trend[i - 1] == -1 and trend[i] == 1:
            signals[i] = 1
        elif trend[i - 1] == 1 and trend[i] == -1:
            signals[i] = -1
    return signals


def generate_signals(category: str, df: pd.DataFrame, params: Dict) -> np.ndarray:
    """Dispatch to the appropriate signal generator based on category."""
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    c = df["close"]
    vol = df.get("tick_volume", df.get("volume", pd.Series(np.zeros(len(df)), index=df.index)))

    gen_map = {
        "RSI": lambda: gen_rsi_signals(c, params.get("period", 14), params.get("ob", 70), params.get("os", 30)),
        "MACD": lambda: gen_macd_signals(c, params.get("fast", 12), params.get("slow", 26), params.get("signal", 9)),
        "BB": lambda: gen_bb_signals(c, params.get("period", 20), params.get("std_dev", 2.0)),
        "MA": lambda: gen_ma_signals(c, params.get("fast", 10), params.get("slow", 50)),
        "ADX": lambda: gen_adx_signals(h, lo, c, params.get("period", 14), params.get("threshold", 25)),
        "WR": lambda: gen_wr_signals(h, lo, c, params.get("period", 14), params.get("ob", -20), params.get("os", -80)),
        "STOCH": lambda: gen_stoch_signals(h, lo, c, params.get("k_period", 14), params.get("d_period", 3), params.get("ob", 80), params.get("os", 20)),
        "CCI": lambda: gen_cci_signals(h, lo, c, params.get("period", 20), params.get("ob", 100), params.get("os", -100)),
        "SRSI": lambda: gen_srsi_signals(c, params.get("rsi_period", 14), params.get("stoch_period", 14), params.get("ob", 80), params.get("os", 20)),
        "KC": lambda: gen_kc_signals(h, lo, c, params.get("ema_period", 20), params.get("atr_period", 14), params.get("mult", 2.0)),
        "DON": lambda: gen_don_signals(h, lo, params.get("period", 20)),
        "ENV": lambda: gen_env_signals(c, params.get("period", 20), params.get("pct", 1.0)),
        "OBV": lambda: gen_obv_signals(c, vol, params.get("fast", 10), params.get("slow", 30)),
        "TSI": lambda: gen_tsi_signals(c, params.get("long_p", 25), params.get("short_p", 13), params.get("signal_p", 7)),
        "ULT": lambda: gen_ult_signals(h, lo, c, params.get("p1", 7), params.get("p2", 14), params.get("p3", 28)),
        "FISHER": lambda: gen_fisher_signals(h, lo, params.get("period", 9)),
        "ARN": lambda: gen_aroon_signals(h, lo, params.get("period", 25)),
        "VTX": lambda: gen_vtx_signals(h, lo, c, params.get("period", 14)),
        "RVI": lambda: gen_rvi_signals(o, h, lo, c, params.get("period", 10)),
        "REG": lambda: gen_reg_signals(c, params.get("period", 20)),
        "ROC": lambda: gen_roc_signals(c, params.get("period", 12)),
        "KST": lambda: gen_kst_signals(c, params.get("r1", 10), params.get("r2", 15), params.get("r3", 20), params.get("r4", 30)),
        "PSAR": lambda: gen_psar_signals(h, lo, params.get("af_start", 0.02), params.get("af_step", 0.02), params.get("af_max", 0.20)),
    }

    if category in gen_map:
        return gen_map[category]()

    return np.zeros(len(df), dtype=np.int8)


# ═══════════════════════════════════════════════════════════════
#  C) FAST BACKTESTER
# ═══════════════════════════════════════════════════════════════

def fast_backtest(signal_array: np.ndarray,
                  opens: np.ndarray, highs: np.ndarray,
                  lows: np.ndarray, closes: np.ndarray,
                  atr_arr: np.ndarray,
                  sl_mult: float = 1.5, tp_mult: float = 2.25,
                  start_bar: int = 50, end_bar: int = -1) -> Dict:
    """
    Fast bar-by-bar backtest using pre-computed signal array and numpy arrays.
    Mirrors combo_optimizer.backtest_combination() logic.
    Returns metrics dict.
    """
    cfg = BTCUSD_CONFIG
    pip = cfg["pip_size"]
    pip_value = cfg["pip_value_per_lot"]
    lot = cfg["backtest_lot"]
    spread_pips = cfg["spread_points"] * cfg["point"]
    half_spread = spread_pips / 2.0

    if end_bar < 0:
        end_bar = len(opens)
    n_bars = end_bar

    _cost_rng = np.random.RandomState(42)

    balance = 10000.0
    peak = balance
    max_dd_pct = 0.0
    max_dd_dollars = 0.0
    trades = []

    in_trade = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    sl_dist = 0.0
    direction = 0
    entry_bar = 0
    total_cost = 0.0
    be_moved = False
    pending_dir = 0
    pending_bar = -1

    for bar_idx in range(start_bar, n_bars):
        o = opens[bar_idx]
        h = highs[bar_idx]
        lo = lows[bar_idx]
        c = closes[bar_idx]
        atr_val = atr_arr[bar_idx] if bar_idx < len(atr_arr) else 0.0

        # Execute pending entry
        if pending_dir != 0 and not in_trade:
            if bar_idx == pending_bar + 1 and atr_val > 0:
                atr_pips = atr_val / pip
                if pending_dir == 1:
                    ep = o + half_spread
                else:
                    ep = o - half_spread

                sd = max(atr_pips * sl_mult, 20)
                td = max(atr_pips * tp_mult, sd * 1.5)

                if sd >= 20 and td >= 20:
                    if pending_dir == 1:
                        sp = ep - sd * pip
                        tp = ep + td * pip
                    else:
                        sp = ep + sd * pip
                        tp = ep - td * pip

                    tc = compute_variable_cost(atr_arr, bar_idx, _cost_rng)
                    balance -= tc
                    in_trade = True
                    entry_price = ep
                    sl_price = sp
                    tp_price = tp
                    sl_dist = sd
                    direction = pending_dir
                    entry_bar = bar_idx
                    total_cost = tc
                    be_moved = False

            pending_dir = 0
            pending_bar = -1

        # Check exits
        if in_trade:
            closed = False
            exit_price = 0.0

            if direction == 1:
                sl_hit = lo <= sl_price
                tp_hit = h >= tp_price
            else:
                sl_hit = h >= sl_price
                tp_hit = lo <= tp_price

            if sl_hit:
                exit_price = sl_price
                closed = True
            elif tp_hit:
                exit_price = tp_price
                closed = True
            elif bar_idx - entry_bar >= 200:
                exit_price = c
                closed = True

            # Breakeven check (only after entry bar)
            if not closed and not be_moved and bar_idx > entry_bar:
                if direction == 1:
                    profit_pips = (c - entry_price) / pip
                else:
                    profit_pips = (entry_price - c) / pip
                if profit_pips >= sl_dist * 0.5:
                    sl_price = entry_price
                    be_moved = True

            if closed:
                if direction == 1:
                    pnl_pips = (exit_price - entry_price) / pip
                else:
                    pnl_pips = (entry_price - exit_price) / pip
                gross_pnl = pnl_pips * pip_value * lot
                net_pnl = gross_pnl - total_cost
                balance += gross_pnl
                trades.append((net_pnl, bar_idx - entry_bar))
                in_trade = False

        # Check for new signals
        if not in_trade and pending_dir == 0:
            if bar_idx < n_bars - 1:
                sig = signal_array[bar_idx]
                if sig != 0:
                    pending_dir = int(sig)
                    pending_bar = bar_idx

        # Drawdown tracking
        if balance > peak:
            peak = balance
        dd = peak - balance
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        if dd > max_dd_dollars:
            max_dd_dollars = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    # Force close
    if in_trade:
        c = closes[min(n_bars - 1, len(closes) - 1)]
        if direction == 1:
            pnl_pips = (c - entry_price) / pip
        else:
            pnl_pips = (entry_price - c) / pip
        gross_pnl = pnl_pips * pip_value * lot
        net_pnl = gross_pnl - total_cost
        balance += gross_pnl
        trades.append((net_pnl, n_bars - 1 - entry_bar))

    return _compute_fast_metrics(trades, max_dd_pct, max_dd_dollars, balance)


def _compute_fast_metrics(trades_data, max_dd_pct, max_dd_dollars, final_balance):
    """Compute metrics from list of (net_pnl, bars_held) tuples."""
    n = len(trades_data)
    if n == 0:
        return _empty_metrics()

    pnls = [t[0] for t in trades_data]
    bars_held = [t[1] for t in trades_data]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net = sum(pnls)
    wr = (len(wins) / n * 100) if n > 0 else 0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (
        999.0 if gross_profit > 0 else 0.0
    )

    sharpe = 0.0
    if n >= 2:
        arr = np.array(pnls, dtype=np.float64)
        mean_r = np.mean(arr)
        std_r = np.std(arr, ddof=1)
        if std_r > 0:
            sharpe = float(mean_r / std_r * np.sqrt(min(n * 4, 200)))

    avg_bars = round(sum(bars_held) / n, 1) if n > 0 else 0

    return {
        "total_trades": n,
        "win_rate": round(wr, 2),
        "net_profit": round(net, 4),
        "profit_factor": round(min(pf, 999.0), 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "max_drawdown_dollars": round(max_dd_dollars, 4),
        "sharpe_ratio": round(sharpe, 4) if abs(sharpe) < 1e6 else 0.0,
        "expectancy": round(net / n, 4) if n > 0 else 0.0,
        "avg_bars_held": avg_bars,
        "total_costs": round(n * 0.65, 4),
    }


def _empty_metrics():
    return {
        "total_trades": 0, "win_rate": 0, "net_profit": 0,
        "profit_factor": 0, "max_drawdown_pct": 0,
        "max_drawdown_dollars": 0, "sharpe_ratio": 0,
        "expectancy": 0, "avg_bars_held": 0, "total_costs": 0,
    }


# ═══════════════════════════════════════════════════════════════
#  D) WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════

def create_wf_windows(n_bars: int, n_folds: int = 5,
                      warmup: int = 50) -> List[Dict]:
    """
    Create anchored expanding walk-forward windows.
    Each fold has expanding IS (in-sample) and fixed-width OOS (out-of-sample).
    """
    usable = n_bars - warmup
    oos_size = usable // (n_folds + 1)  # ~1000 bars per fold for 5951 bars
    windows = []

    for fold in range(n_folds):
        is_start = warmup
        is_end = warmup + (fold + 1) * oos_size - 1
        oos_start = is_end + 1
        if fold == n_folds - 1:
            oos_end = n_bars - 1
        else:
            oos_end = oos_start + oos_size - 1
            oos_end = min(oos_end, n_bars - 1)

        windows.append({
            "fold": fold + 1,
            "is_start": is_start,
            "is_end": is_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "is_bars": is_end - is_start + 1,
            "oos_bars": oos_end - oos_start + 1,
        })

    return windows


def objective_function(metrics: Dict) -> float:
    """
    Optimization objective:
    objective = profit_factor * sqrt(total_trades) * (1 - max_drawdown_pct / 100)
    """
    pf = metrics.get("profit_factor", 0)
    n_trades = metrics.get("total_trades", 0)
    dd = metrics.get("max_drawdown_pct", 0)

    if pf <= 0 or n_trades == 0:
        return 0.0

    return pf * math.sqrt(n_trades) * (1.0 - dd / 100.0)


# ═══════════════════════════════════════════════════════════════
#  GRID / RANDOM / BAYESIAN SEARCH
# ═══════════════════════════════════════════════════════════════

def _generate_grid(param_space: List[Dict]) -> List[Dict]:
    """Generate all grid points for a parameter space."""
    axes = []
    names = []
    for p in param_space:
        names.append(p["name"])
        if p["type"] == "int":
            vals = list(range(p["low"], p["high"] + 1, p["step"]))
        else:
            vals = []
            v = p["low"]
            while v <= p["high"] + 1e-9:
                vals.append(round(v, 6))
                v += p["step"]
        axes.append(vals)

    grid = []
    for combo in itertools.product(*axes):
        grid.append(dict(zip(names, combo)))
    return grid


def _random_samples(param_space: List[Dict], n_samples: int = 500,
                    rng: Optional[np.random.RandomState] = None) -> List[Dict]:
    """Generate random parameter samples within bounds."""
    if rng is None:
        rng = np.random.RandomState(42)
    samples = []
    for _ in range(n_samples):
        point = {}
        for p in param_space:
            if p["type"] == "int":
                vals = list(range(p["low"], p["high"] + 1, p["step"]))
                point[p["name"]] = int(rng.choice(vals))
            else:
                vals = []
                v = p["low"]
                while v <= p["high"] + 1e-9:
                    vals.append(round(v, 6))
                    v += p["step"]
                point[p["name"]] = float(rng.choice(vals))
        samples.append(point)
    return samples


def _bayesian_search(param_space: List[Dict], eval_fn, n_iter: int = 100) -> List[Tuple[Dict, float]]:
    """Bayesian optimization via optuna. Falls back to random if unavailable."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed, falling back to random search")
        samples = _random_samples(param_space, n_iter)
        results = []
        for s in samples:
            score = eval_fn(s)
            results.append((s, score))
        return results

    results = []

    def _objective(trial):
        params = {}
        for p in param_space:
            if p["type"] == "int":
                params[p["name"]] = trial.suggest_int(
                    p["name"], p["low"], p["high"], step=p["step"])
            else:
                # Discretize float params to step grid
                vals = []
                v = p["low"]
                while v <= p["high"] + 1e-9:
                    vals.append(round(v, 6))
                    v += p["step"]
                params[p["name"]] = trial.suggest_categorical(p["name"], vals)
        score = eval_fn(params)
        results.append((params, score))
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_objective, n_trials=n_iter, show_progress_bar=False)
    return results


def _get_defaults(param_space: List[Dict]) -> Dict:
    """Get default parameter values."""
    return {p["name"]: p["default"] for p in param_space}


def _choose_search_method(param_space: List[Dict]) -> str:
    """Choose grid vs random based on space size."""
    grid = _generate_grid(param_space)
    n_params = len(param_space)
    if n_params <= 3 and len(grid) <= 5000:
        return "grid"
    return "random"


# ═══════════════════════════════════════════════════════════════
#  SIGNAL RECONSTRUCTION FOR TIER 2
# ═══════════════════════════════════════════════════════════════

def reconstruct_signals_from_trades(strategy_id: str, n_bars: int) -> np.ndarray:
    """
    Reconstruct signal array from cached Phase 5 individual results.
    Reads trade signal_bar_index + direction from the JSON file.
    """
    fpath = os.path.join(INDIVIDUAL_DIR, f"{strategy_id}.json")
    signals = np.zeros(n_bars, dtype=np.int8)

    if not os.path.exists(fpath):
        logger.warning(f"No cached result for {strategy_id}")
        return signals

    with open(fpath, "r") as f:
        data = json.load(f)

    for trade in data.get("trades", []):
        bar_idx = trade.get("signal_bar_index", -1)
        direction = trade.get("direction", "")
        if 0 <= bar_idx < n_bars:
            if direction == "BUY":
                signals[bar_idx] = 1
            elif direction == "SELL":
                signals[bar_idx] = -1

    return signals


# ═══════════════════════════════════════════════════════════════
#  E) MAIN OPTIMIZER
# ═══════════════════════════════════════════════════════════════

def optimize_single_strategy(
    strategy_id: str,
    df: pd.DataFrame,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr_arr: np.ndarray,
    n_folds: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run walk-forward optimization for a single strategy.
    Returns result dict with default metrics, optimized metrics, params, fold details.
    """
    t0 = time.time()
    sid = strategy_id
    cat = _get_category(sid)
    tier, param_space = get_param_space(sid)
    n_bars = len(df)
    windows = create_wf_windows(n_bars, n_folds)

    if verbose:
        logger.info(f"  Optimizing {sid} (cat={cat}, tier={tier}, "
                     f"params={len(param_space)}, folds={n_folds})")

    # Get default params
    defaults = _get_defaults(param_space)

    # Generate default signal array
    if tier == "tier1":
        default_signals = generate_signals(cat, df, defaults)
    else:
        default_signals = reconstruct_signals_from_trades(sid, n_bars)

    # Run default backtest on full data
    default_metrics = fast_backtest(
        default_signals, opens, highs, lows, closes, atr_arr,
        sl_mult=defaults.get("sl_mult", 1.5),
        tp_mult=defaults.get("tp_mult", 2.25),
    )

    # Choose search method
    method = _choose_search_method(param_space)

    # Walk-forward optimization
    fold_results = []
    best_params_per_fold = []

    for win in windows:
        is_start = win["is_start"]
        is_end = win["is_end"] + 1  # exclusive
        oos_start = win["oos_start"]
        oos_end = win["oos_end"] + 1  # exclusive

        # Evaluation function for IS period
        def eval_params(params, _is_start=is_start, _is_end=is_end):
            if tier == "tier1":
                sig = generate_signals(cat, df, params)
            else:
                sig = default_signals.copy()
            m = fast_backtest(
                sig, opens, highs, lows, closes, atr_arr,
                sl_mult=params.get("sl_mult", 1.5),
                tp_mult=params.get("tp_mult", 2.25),
                start_bar=_is_start, end_bar=_is_end,
            )
            return objective_function(m)

        # Search IS
        if method == "grid":
            candidates = _generate_grid(param_space)
        else:
            candidates = _random_samples(param_space, 500)

        best_score = -1
        best_params = defaults.copy()

        for params in candidates:
            score = eval_params(params)
            if score > best_score:
                best_score = score
                best_params = params.copy()

        # Evaluate best params on OOS
        if tier == "tier1":
            oos_sig = generate_signals(cat, df, best_params)
        else:
            oos_sig = default_signals.copy()

        is_metrics = fast_backtest(
            oos_sig if tier == "tier1" else default_signals,
            opens, highs, lows, closes, atr_arr,
            sl_mult=best_params.get("sl_mult", 1.5),
            tp_mult=best_params.get("tp_mult", 2.25),
            start_bar=is_start, end_bar=is_end,
        )
        oos_metrics = fast_backtest(
            oos_sig, opens, highs, lows, closes, atr_arr,
            sl_mult=best_params.get("sl_mult", 1.5),
            tp_mult=best_params.get("tp_mult", 2.25),
            start_bar=oos_start, end_bar=oos_end,
        )

        fold_results.append({
            "fold": win["fold"],
            "is_start": is_start,
            "is_end": win["is_end"],
            "oos_start": oos_start,
            "oos_end": win["oos_end"],
            "best_params": best_params,
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "is_objective": objective_function(is_metrics),
            "oos_objective": objective_function(oos_metrics),
        })
        best_params_per_fold.append(best_params)

    # Walk-forward acceptance criteria
    all_oos_pf = [fr["oos_metrics"]["profit_factor"] for fr in fold_results]
    all_is_pf = [fr["is_metrics"]["profit_factor"] for fr in fold_results]
    all_oos_trades = [fr["oos_metrics"]["total_trades"] for fr in fold_results]

    any_negative_pf = any(pf < 1.0 for pf in all_oos_pf)
    avg_oos_pf = np.mean(all_oos_pf) if all_oos_pf else 0
    avg_is_pf = np.mean(all_is_pf) if all_is_pf else 0
    min_trades_ok = all(t >= 10 for t in all_oos_trades)
    degradation_ok = avg_oos_pf > 0.5 * avg_is_pf if avg_is_pf > 0 else False

    passed_wf = (not any_negative_pf) and degradation_ok and min_trades_ok

    # Use most recent fold's best params as the optimized params
    optimized_params = best_params_per_fold[-1] if best_params_per_fold else defaults

    # Run optimized on full data for comparison
    if tier == "tier1":
        opt_signals = generate_signals(cat, df, optimized_params)
    else:
        opt_signals = default_signals.copy()

    optimized_metrics = fast_backtest(
        opt_signals, opens, highs, lows, closes, atr_arr,
        sl_mult=optimized_params.get("sl_mult", 1.5),
        tp_mult=optimized_params.get("tp_mult", 2.25),
    )

    # Improvement
    default_obj = objective_function(default_metrics)
    optimized_obj = objective_function(optimized_metrics)
    if default_obj > 0:
        improvement_pct = ((optimized_obj - default_obj) / default_obj) * 100
    else:
        improvement_pct = 0 if optimized_obj <= 0 else 100.0

    # Overfit detection
    overfit_flags = []
    if improvement_pct > 100:
        overfit_flags.append("improvement_over_100pct")
    # Check boundary params
    for p in param_space:
        val = optimized_params.get(p["name"])
        if val is not None and (val == p["low"] or val == p["high"]):
            overfit_flags.append(f"boundary_{p['name']}")
    # Check fold variance
    if len(all_oos_pf) >= 2:
        fold_std = np.std(all_oos_pf)
        fold_mean = np.mean(all_oos_pf)
        if fold_mean > 0 and fold_std / fold_mean > 1.0:
            overfit_flags.append("high_fold_variance")

    elapsed = time.time() - t0

    result = {
        "strategy_id": sid,
        "category": cat,
        "tier": tier,
        "search_method": method,
        "default_params": defaults,
        "optimized_params": optimized_params,
        "default_metrics": default_metrics,
        "optimized_metrics": optimized_metrics,
        "default_objective": round(default_obj, 4),
        "optimized_objective": round(optimized_obj, 4),
        "improvement_pct": round(improvement_pct, 2),
        "walk_forward_passed": passed_wf,
        "wf_acceptance": {
            "all_folds_positive_pf": not any_negative_pf,
            "avg_oos_pf": round(avg_oos_pf, 4),
            "avg_is_pf": round(avg_is_pf, 4),
            "degradation_ok": degradation_ok,
            "min_trades_all_folds": min_trades_ok,
        },
        "fold_results": fold_results,
        "overfit_flags": overfit_flags,
        "elapsed_sec": round(elapsed, 2),
    }

    if verbose:
        status = "PASS" if passed_wf else "FAIL"
        logger.info(f"    {sid}: {status} | default_obj={default_obj:.2f} "
                     f"opt_obj={optimized_obj:.2f} imp={improvement_pct:.1f}% "
                     f"({elapsed:.1f}s)")

    return result


def run_optimization(
    top_n: int = 50,
    n_folds: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run Phase 8 optimization on top N strategies from Phase 5.
    Returns summary dict.
    """
    t0 = time.time()

    # Load training data
    train_file = os.path.join(TRAIN_DIR, "BTCUSD_H1.csv")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")

    df = pd.read_csv(train_file)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    n_bars = len(df)
    logger.info(f"Loaded training data: {n_bars} bars")

    # Pre-extract numpy arrays
    opens = df["open"].values.astype(np.float64)
    highs_arr = df["high"].values.astype(np.float64)
    lows_arr = df["low"].values.astype(np.float64)
    closes_arr = df["close"].values.astype(np.float64)

    # Compute ATR
    atr_series = ind.atr(df["high"], df["low"], df["close"], 14)
    atr_arr = np.array(atr_series, dtype=np.float64)
    atr_arr = np.nan_to_num(atr_arr, nan=0.0)

    # Load top strategies from Phase 5
    summary_path = os.path.join(RESULTS_DIR, "individual_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Individual summary not found: {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    top_strategies = summary.get("top_100", [])[:top_n]
    strategy_ids = [s["strategy_id"] for s in top_strategies]
    logger.info(f"Optimizing top {len(strategy_ids)} strategies")

    # Ensure output dir
    os.makedirs(OPTIMIZED_DIR, exist_ok=True)

    # Run optimization for each strategy
    results = []
    passed = 0
    failed = 0

    for i, sid in enumerate(strategy_ids):
        logger.info(f"[{i + 1}/{len(strategy_ids)}] {sid}")
        try:
            result = optimize_single_strategy(
                sid, df, opens, highs_arr, lows_arr, closes_arr,
                atr_arr, n_folds=n_folds, verbose=verbose,
            )
            results.append(result)

            # Save individual result
            out_path = os.path.join(OPTIMIZED_DIR, f"{sid}_params.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            if result["walk_forward_passed"]:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"  Error optimizing {sid}: {e}")
            results.append({
                "strategy_id": sid,
                "error": str(e),
                "walk_forward_passed": False,
            })
            failed += 1

    # Build summary
    elapsed = time.time() - t0

    # Compute aggregate stats
    improvements = [r["improvement_pct"] for r in results if "improvement_pct" in r]
    avg_improvement = np.mean(improvements) if improvements else 0

    optimization_summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "total_strategies": len(strategy_ids),
        "passed_walk_forward": passed,
        "failed_walk_forward": failed,
        "avg_improvement_pct": round(avg_improvement, 2),
        "n_folds": n_folds,
        "total_bars": n_bars,
        "elapsed_sec": round(elapsed, 2),
        "strategies": [
            {
                "strategy_id": r["strategy_id"],
                "category": r.get("category", ""),
                "tier": r.get("tier", ""),
                "walk_forward_passed": r.get("walk_forward_passed", False),
                "improvement_pct": r.get("improvement_pct", 0),
                "default_objective": r.get("default_objective", 0),
                "optimized_objective": r.get("optimized_objective", 0),
                "overfit_flags": r.get("overfit_flags", []),
            }
            for r in results
        ],
    }

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(optimization_summary, f, indent=2)

    logger.info(f"\nOptimization complete in {elapsed:.1f}s")
    logger.info(f"  Passed WF: {passed}/{len(strategy_ids)}")
    logger.info(f"  Failed WF: {failed}/{len(strategy_ids)}")
    logger.info(f"  Avg improvement: {avg_improvement:.1f}%")
    logger.info(f"  Results saved to: {OPTIMIZED_DIR}")

    return optimization_summary

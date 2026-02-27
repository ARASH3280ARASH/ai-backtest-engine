"""
Backtest Engine — Realistic Cost Model
=========================================
Models spread, commission, and slippage for BTCUSD backtesting.
All functions return dollar amounts.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.broker import BTCUSD_CONFIG

_CFG = BTCUSD_CONFIG


def calculate_spread_cost(lot_size: float) -> float:
    """
    Calculate spread cost in dollars. Always deducted at entry.

    Spread = spread_points * point * contract_size * lot_size
    For BTCUSD: 1700 * 0.01 * 1 * lot = 17.0 * lot
    At 0.01 lot: $0.17
    """
    return _CFG["spread_dollars_per_lot"] * lot_size


def calculate_commission(lot_size: float) -> float:
    """
    Calculate commission in dollars. Applied at BOTH open and close.

    Commission per side = $6/lot
    Round trip = $12/lot
    At 0.01 lot: $0.12 round trip
    """
    return _CFG["commission_round_trip_per_lot"] * lot_size


def calculate_slippage(direction: str, atr_value: float = 0.0) -> float:
    """
    Estimate slippage in pips, scaled by volatility.

    Base slippage: 2 pips.
    If ATR is provided, scale: higher ATR = more slippage (1-3 pip range).
    Returns slippage in PIPS (not dollars).
    """
    base_slippage = _CFG["slippage_estimate_pips"]

    if atr_value <= 0:
        return float(base_slippage)

    # ATR-based scaling: normalize ATR relative to typical BTCUSD ATR (~500-2000 pips)
    # Low volatility (ATR < 500): 1 pip slippage
    # Normal volatility (ATR ~1000): 2 pips
    # High volatility (ATR > 2000): 3 pips
    typical_atr = 1000.0
    ratio = atr_value / typical_atr
    scaled = base_slippage * max(0.5, min(1.5, ratio))
    return round(scaled, 1)


def calculate_slippage_dollars(lot_size: float, direction: str, atr_value: float = 0.0) -> float:
    """
    Calculate slippage cost in dollars.

    slippage_pips * pip_value_per_lot * lot_size
    """
    slippage_pips = calculate_slippage(direction, atr_value)
    return slippage_pips * _CFG["pip_value_per_lot"] * lot_size


def calculate_total_cost(lot_size: float, atr_value: float = 0.0) -> float:
    """
    Calculate total trade cost in dollars (spread + commission + slippage).

    At 0.01 lot with default ATR:
        Spread:     $0.17
        Commission: $0.12
        Slippage:   $0.02 (2 pips * $0.01/pip)
        Total:      $0.31
    """
    spread = calculate_spread_cost(lot_size)
    commission = calculate_commission(lot_size)
    slippage = calculate_slippage_dollars(lot_size, "BUY", atr_value)
    return spread + commission + slippage


def compute_variable_cost(atr_arr, bar_idx, rng=None):
    """
    Compute trade cost with ATR-based variable slippage.

    Spread:     $0.17 fixed (1700 pts x $0.01/pip x 0.01 lot)
    Commission: $0.12 fixed ($6/side x 0.01 lot x 2 sides)
    Slippage:   variable 1-3 pips based on ATR percentile of last 200 bars
                plus random +/-0.5 pip jitter
    Total range: ~$0.30 to $0.32
    """
    SPREAD = 0.17
    COMMISSION = 0.12
    PIP_VALUE_001 = 0.01  # pip value at 0.01 lot

    atr_val = float(atr_arr[bar_idx]) if bar_idx < len(atr_arr) else 0.0

    # ATR percentile of last 200 bars
    lb_start = max(0, bar_idx - 200)
    atr_window = atr_arr[lb_start:bar_idx + 1]
    atr_window = atr_window[np.isfinite(atr_window) & (atr_window > 0)]

    if len(atr_window) > 0 and atr_val > 0:
        p80 = float(np.percentile(atr_window, 80))
        p60 = float(np.percentile(atr_window, 60))
        if atr_val > p80:
            slip_pips = 3.0
        elif atr_val > p60:
            slip_pips = 2.0
        else:
            slip_pips = 1.0
    else:
        slip_pips = 1.0

    # Jitter +/-0.5 pip
    if rng is not None:
        slip_pips += rng.uniform(-0.5, 0.5)
    slip_pips = max(0.5, slip_pips)

    return round(SPREAD + COMMISSION + slip_pips * PIP_VALUE_001, 4)


def get_cost_summary(lot_size: float = None, atr_value: float = 0.0) -> dict:
    """Return a breakdown of all costs for display/logging."""
    if lot_size is None:
        lot_size = _CFG["backtest_lot"]

    spread = calculate_spread_cost(lot_size)
    commission = calculate_commission(lot_size)
    slippage_pips = calculate_slippage("BUY", atr_value)
    slippage_usd = calculate_slippage_dollars(lot_size, "BUY", atr_value)
    total = spread + commission + slippage_usd

    return {
        "lot_size": lot_size,
        "spread_usd": round(spread, 4),
        "commission_usd": round(commission, 4),
        "slippage_pips": slippage_pips,
        "slippage_usd": round(slippage_usd, 4),
        "total_cost_usd": round(total, 4),
    }

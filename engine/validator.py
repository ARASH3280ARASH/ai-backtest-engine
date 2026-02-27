"""
Backtest Engine — Trade Validator
====================================
Validates EVERY trade before execution to ensure realism.

Checks:
1. SL distance >= stop_level (20 pips from entry)
2. TP distance >= stop_level (20 pips from entry)
3. Both SL and TP must be set
4. Entry price within spread of the candle's OHLC
5. No lookahead bias: entry on NEXT bar after signal
6. R:R ratio calculated and stored
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.broker import BTCUSD_CONFIG
from engine.trade import Trade, ValidatedTrade
from engine.costs import calculate_spread_cost, calculate_commission, calculate_slippage_dollars

_CFG = BTCUSD_CONFIG
_PIP = _CFG["pip_size"]           # 1.0 for BTCUSD
_STOP_LEVEL = _CFG["stop_level_pips"]  # 20 pips
_POINT = _CFG["point"]            # 0.01


def validate_trade(
    strategy_id: str,
    direction: str,
    signal_bar_index: int,
    entry_bar_index: int,
    entry_bar_open: float,
    entry_bar_high: float,
    entry_bar_low: float,
    entry_bar_close: float,
    entry_time: str,
    sl_price: float,
    tp_price: float,
    lot_size: float = None,
    atr_value: float = 0.0,
    confidence: float = 0.0,
) -> ValidatedTrade:
    """
    Validate a proposed trade. Returns ValidatedTrade (accepted or rejected).

    Args:
        strategy_id: Strategy identifier
        direction: "BUY" or "SELL"
        signal_bar_index: Bar index where signal was generated
        entry_bar_index: Bar index for execution (must be signal_bar + 1)
        entry_bar_open/high/low/close: OHLC of the entry bar
        entry_time: Timestamp of the entry bar
        sl_price: Stop loss price
        tp_price: Take profit price
        lot_size: Lot size (defaults to broker config)
        atr_value: ATR for slippage calculation
        confidence: Signal confidence (0-100)
    """
    if lot_size is None:
        lot_size = _CFG["backtest_lot"]

    # --- CHECK 1: Direction valid ---
    if direction not in ("BUY", "SELL"):
        return ValidatedTrade.rejected(f"Invalid direction: {direction}")

    # --- CHECK 2: No lookahead bias ---
    if entry_bar_index <= signal_bar_index:
        return ValidatedTrade.rejected(
            f"Lookahead bias: entry_bar={entry_bar_index} <= signal_bar={signal_bar_index}"
        )

    # --- CHECK 3: SL and TP must be set ---
    if sl_price <= 0:
        return ValidatedTrade.rejected("SL not set (sl_price <= 0)")
    if tp_price <= 0:
        return ValidatedTrade.rejected("TP not set (tp_price <= 0)")

    # --- CHECK 4: Entry price realistic ---
    # BUY enters at ask (open + half spread), SELL enters at bid (open - half spread)
    spread_pips = _CFG["spread_points"] * _POINT  # 1700 * 0.01 = 17.0
    half_spread = spread_pips / 2.0

    if direction == "BUY":
        entry_price = entry_bar_open + half_spread
    else:
        entry_price = entry_bar_open - half_spread

    # Verify entry is within the bar's range (with spread tolerance)
    bar_low_adj = entry_bar_low - spread_pips
    bar_high_adj = entry_bar_high + spread_pips
    if entry_price < bar_low_adj or entry_price > bar_high_adj:
        return ValidatedTrade.rejected(
            f"Entry {entry_price:.2f} outside bar range [{bar_low_adj:.2f}, {bar_high_adj:.2f}]"
        )

    # --- CHECK 5: SL on correct side ---
    if direction == "BUY":
        if sl_price >= entry_price:
            return ValidatedTrade.rejected(f"BUY but SL({sl_price:.2f}) >= entry({entry_price:.2f})")
        if tp_price <= entry_price:
            return ValidatedTrade.rejected(f"BUY but TP({tp_price:.2f}) <= entry({entry_price:.2f})")
    else:
        if sl_price <= entry_price:
            return ValidatedTrade.rejected(f"SELL but SL({sl_price:.2f}) <= entry({entry_price:.2f})")
        if tp_price >= entry_price:
            return ValidatedTrade.rejected(f"SELL but TP({tp_price:.2f}) >= entry({entry_price:.2f})")

    # --- CHECK 6: SL distance >= stop_level ---
    sl_distance_pips = abs(entry_price - sl_price) / _PIP
    if sl_distance_pips < _STOP_LEVEL:
        return ValidatedTrade.rejected(
            f"SL too close: {sl_distance_pips:.1f} pips < {_STOP_LEVEL} pips minimum"
        )

    # --- CHECK 7: TP distance >= stop_level ---
    tp_distance_pips = abs(tp_price - entry_price) / _PIP
    if tp_distance_pips < _STOP_LEVEL:
        return ValidatedTrade.rejected(
            f"TP too close: {tp_distance_pips:.1f} pips < {_STOP_LEVEL} pips minimum"
        )

    # --- CHECK 8: Calculate R:R ---
    risk = sl_distance_pips
    reward = tp_distance_pips
    rr_ratio = reward / risk if risk > 0 else 0.0

    # --- CALCULATE COSTS ---
    spread_cost = calculate_spread_cost(lot_size)
    commission_cost = calculate_commission(lot_size)
    slippage_cost = calculate_slippage_dollars(lot_size, direction, atr_value)
    total_cost = spread_cost + commission_cost + slippage_cost

    # --- BUILD TRADE ---
    trade = Trade(
        trade_id=f"{strategy_id}_{entry_bar_index}",
        strategy_id=strategy_id,
        symbol=_CFG["symbol"],
        direction=direction,
        entry_price=round(entry_price, _CFG["digits"]),
        sl_price=round(sl_price, _CFG["digits"]),
        tp1_price=round(tp_price, _CFG["digits"]),
        lot_size=lot_size,
        entry_time=entry_time,
        entry_bar_index=entry_bar_index,
        signal_bar_index=signal_bar_index,
        sl_distance_pips=round(sl_distance_pips, 1),
        tp_distance_pips=round(tp_distance_pips, 1),
        rr_ratio=round(rr_ratio, 2),
        planned_rr=round(rr_ratio, 2),
        spread_cost=round(spread_cost, 4),
        commission_cost=round(commission_cost, 4),
        slippage_cost=round(slippage_cost, 4),
        total_cost=round(total_cost, 4),
        confidence=confidence,
        is_valid=True,
    )

    return ValidatedTrade.accepted(trade)

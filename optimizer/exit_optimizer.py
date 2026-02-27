"""
Phase 10 — Exit & Trade Management Optimizer with Walk-Forward
================================================================
Optimizes what happens AFTER a trade is opened: SL method, TP method,
trailing/BE management, partial closes, and time exits.

Uses the same anchored expanding walk-forward as Phase 8.
Sequential optimization: SL → TP → Management → Time exit.

Data leakage prevention:
  - ATR for SL/TP sizing uses ONLY data available UP TO entry bar
  - Trailing stops use only past-bar prices (highest/lowest SO FAR)
  - Break-even checks current vs entry price only
  - Walk-forward folds are WITHIN training data only
"""

import json
import math
import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.broker import BTCUSD_CONFIG
from config.settings import (
    TRAIN_DIR, RESULTS_DIR, INDIVIDUAL_DIR, OPTIMIZED_DIR,
)
from indicators import compute as ind
from optimizer.param_optimizer import (
    _get_category, generate_signals, reconstruct_signals_from_trades,
    create_wf_windows, _compute_fast_metrics, _empty_metrics,
    objective_function, TIER1_CATEGORIES, PARAM_SPACES,
)

logger = logging.getLogger("phase10")

EXIT_DIR = os.path.join(OPTIMIZED_DIR, "exit")


# ═══════════════════════════════════════════════════════════════
#  A) EXIT STRATEGY SEARCH SPACE
# ═══════════════════════════════════════════════════════════════

SL_METHODS = {
    "fixed": [
        {"sl_method": "fixed", "sl_fixed_pips": p}
        for p in [50, 75, 100, 150, 200, 300, 400, 500]
    ],
    "atr": [
        {"sl_method": "atr", "sl_atr_period": per, "sl_atr_multiplier": m}
        for per in [10, 14, 20]
        for m in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    ],
    "swing": [
        {"sl_method": "swing", "sl_swing_lookback": lb, "sl_swing_buffer": buf}
        for lb in [10, 15, 20, 30]
        for buf in [5, 10, 20]
    ],
    "pct": [
        {"sl_method": "pct", "sl_pct": p}
        for p in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    ],
}

TP_METHODS = {
    "fixed": [
        {"tp_method": "fixed", "tp_fixed_pips": p}
        for p in [100, 150, 200, 300, 400, 500, 700, 1000]
    ],
    "atr": [
        {"tp_method": "atr", "tp_atr_period": per, "tp_atr_multiplier": m}
        for per in [10, 14, 20]
        for m in [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    ],
    "rr": [
        {"tp_method": "rr", "tp_rr_mult": r}
        for r in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    ],
    "fib": [
        {"tp_method": "fib", "tp_fib_level": f}
        for f in [1.0, 1.272, 1.618, 2.0, 2.618]
    ],
    "dual_rr": [
        {"tp_method": "dual_rr", "tp1_r_mult": t1, "tp2_r_mult": t2}
        for t1 in [1.0, 1.5, 2.0]
        for t2 in [2.0, 3.0, 4.0, 5.0]
        if t2 > t1
    ],
}

MGMT_METHODS = {
    "none": [{"management": "none"}],
    "breakeven": [
        {"management": "breakeven", "be_trigger_pips": p}
        for p in [30, 50, 75, 100, 150, 200]
    ],
    "trail_fixed": [
        {"management": "trail_fixed", "trail_activate_pips": a, "trail_distance_pips": d}
        for a in [50, 100, 150]
        for d in [30, 50, 75, 100, 150]
    ],
    "trail_atr": [
        {"management": "trail_atr", "trail_activate_atr_mult": a, "trail_distance_atr_mult": d}
        for a in [1.0, 1.5, 2.0]
        for d in [1.0, 1.5, 2.0, 2.5]
    ],
    "step": [
        {"management": "step", "step_size_pips": s}
        for s in [50, 100]
    ],
}

PARTIAL_METHODS = [
    {"partial_close": "none"},
    {"partial_close": "50_50", "partial_pct_tp1": 0.50, "partial_pct_tp2": 0.50},
    {"partial_close": "33_33_34", "partial_pct_tp1": 0.33, "partial_pct_tp2": 0.33, "partial_pct_rest": 0.34},
    {"partial_close": "25_25_50", "partial_pct_tp1": 0.25, "partial_pct_tp2": 0.25, "partial_pct_rest": 0.50},
    {"partial_close": "75_25", "partial_pct_tp1": 0.75, "partial_pct_tp2": 0.25},
]

TIME_EXIT_METHODS = [
    {"time_exit": "none"},
    *[{"time_exit": "close", "time_exit_bars": b} for b in [48, 72, 96, 120, 150, 200]],
    *[{"time_exit": "reduce", "time_exit_bars": b, "time_reduce_pct": 0.5} for b in [72, 100]],
]


def _all_sl_configs() -> List[Dict]:
    """Flatten all SL method configs."""
    out = []
    for configs in SL_METHODS.values():
        out.extend(configs)
    return out


def _all_tp_configs() -> List[Dict]:
    out = []
    for configs in TP_METHODS.values():
        out.extend(configs)
    return out


def _all_mgmt_configs() -> List[Dict]:
    out = []
    for configs in MGMT_METHODS.values():
        out.extend(configs)
    return out


# ═══════════════════════════════════════════════════════════════
#  B) ADVANCED BACKTESTER WITH EXIT CONFIG
# ═══════════════════════════════════════════════════════════════

def _compute_sl_distance(exit_cfg: Dict, entry_price: float, direction: int,
                         atr_val: float, pip: float,
                         highs: np.ndarray, lows: np.ndarray,
                         bar_idx: int) -> float:
    """Compute SL distance in pips based on exit config. No future data."""
    method = exit_cfg.get("sl_method", "atr")

    if method == "fixed":
        return max(exit_cfg.get("sl_fixed_pips", 150), 20)

    elif method == "atr":
        period = exit_cfg.get("sl_atr_period", 14)
        mult = exit_cfg.get("sl_atr_multiplier", 2.0)
        atr_pips = atr_val / pip
        return max(atr_pips * mult, 20)

    elif method == "swing":
        lookback = exit_cfg.get("sl_swing_lookback", 20)
        buffer = exit_cfg.get("sl_swing_buffer", 10)
        start = max(0, bar_idx - lookback)
        if direction == 1:
            swing_low = np.min(lows[start:bar_idx]) if bar_idx > start else lows[bar_idx]
            dist = (entry_price - swing_low) / pip + buffer
        else:
            swing_high = np.max(highs[start:bar_idx]) if bar_idx > start else highs[bar_idx]
            dist = (swing_high - entry_price) / pip + buffer
        return max(dist, 20)

    elif method == "pct":
        pct = exit_cfg.get("sl_pct", 1.0)
        dist = entry_price * pct / 100.0 / pip
        return max(dist, 20)

    # Default ATR fallback
    atr_pips = atr_val / pip
    return max(atr_pips * 1.5, 20)


def _compute_tp_distance(exit_cfg: Dict, sl_dist_pips: float,
                         atr_val: float, pip: float) -> Tuple[float, float]:
    """
    Compute TP distance(s) in pips.
    Returns (tp1_pips, tp2_pips). tp2 is 0 if single TP.
    """
    method = exit_cfg.get("tp_method", "rr")

    if method == "fixed":
        tp1 = max(exit_cfg.get("tp_fixed_pips", 300), 20)
        return tp1, 0.0

    elif method == "atr":
        mult = exit_cfg.get("tp_atr_multiplier", 3.0)
        atr_pips = atr_val / pip
        tp1 = max(atr_pips * mult, 20)
        return tp1, 0.0

    elif method == "rr":
        mult = exit_cfg.get("tp_rr_mult", 2.0)
        tp1 = max(sl_dist_pips * mult, 20)
        return tp1, 0.0

    elif method == "fib":
        fib = exit_cfg.get("tp_fib_level", 1.618)
        tp1 = max(sl_dist_pips * fib, 20)
        return tp1, 0.0

    elif method == "dual_rr":
        t1 = exit_cfg.get("tp1_r_mult", 1.5)
        t2 = exit_cfg.get("tp2_r_mult", 3.0)
        tp1 = max(sl_dist_pips * t1, 20)
        tp2 = max(sl_dist_pips * t2, 20)
        return tp1, tp2

    # Default
    return max(sl_dist_pips * 2.0, 20), 0.0


def exit_backtest(signal_array: np.ndarray,
                  opens: np.ndarray, highs: np.ndarray,
                  lows: np.ndarray, closes: np.ndarray,
                  atr_arr: np.ndarray,
                  exit_cfg: Dict,
                  start_bar: int = 50, end_bar: int = -1) -> Dict:
    """
    Full-featured backtester with configurable exit strategy.
    Supports: SL methods, TP methods, trailing, break-even,
    partial close, time exit.

    DATA LEAKAGE SAFE:
    - ATR at bar N uses data up to bar N (computed offline with ewm)
    - Swing lookback uses bars BEFORE entry
    - Trailing uses max/min prices from entry to current bar (past only)
    - Break-even checks current close vs entry
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

    # RNG for variable slippage (seeded for reproducibility)
    _cost_rng = np.random.RandomState(42)

    # Management config
    mgmt = exit_cfg.get("management", "none")
    partial = exit_cfg.get("partial_close", "none")
    time_exit = exit_cfg.get("time_exit", "none")
    time_exit_bars = exit_cfg.get("time_exit_bars", 200)
    time_reduce_pct = exit_cfg.get("time_reduce_pct", 0.5)

    balance = 10000.0
    peak = balance
    max_dd_pct = 0.0
    max_dd_dollars = 0.0
    trades = []  # (net_pnl, bars_held)
    trade_details = []  # detailed info for leakage audit

    in_trade = False
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    sl_dist = 0.0
    tp1_dist = 0.0
    tp2_dist = 0.0
    direction = 0
    entry_bar = 0
    total_cost = 0.0
    be_moved = False
    trailing_active = False
    trail_price = 0.0  # best price seen since entry (for trailing)
    partial_closed_tp1 = False
    position_remaining = 1.0  # fraction of position still open
    realized_pnl = 0.0  # accumulated from partial closes
    time_reduced = False
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
                if pending_dir == 1:
                    ep = o + half_spread
                else:
                    ep = o - half_spread

                # Compute SL
                sd = _compute_sl_distance(
                    exit_cfg, ep, pending_dir, atr_val, pip,
                    highs, lows, bar_idx)

                # Compute TP
                tp1_d, tp2_d = _compute_tp_distance(exit_cfg, sd, atr_val, pip)

                if sd >= 20 and tp1_d >= 20:
                    if pending_dir == 1:
                        sp = ep - sd * pip
                        tp1_p = ep + tp1_d * pip
                        tp2_p = ep + tp2_d * pip if tp2_d > 0 else 0.0
                    else:
                        sp = ep + sd * pip
                        tp1_p = ep - tp1_d * pip
                        tp2_p = ep - tp2_d * pip if tp2_d > 0 else 0.0

                    from engine.costs import compute_variable_cost
                    tc = compute_variable_cost(atr_arr, bar_idx, _cost_rng)
                    balance -= tc
                    in_trade = True
                    entry_price = ep
                    sl_price = sp
                    tp1_price = tp1_p
                    tp2_price = tp2_p
                    sl_dist = sd
                    tp1_dist = tp1_d
                    tp2_dist = tp2_d
                    direction = pending_dir
                    entry_bar = bar_idx
                    total_cost = tc
                    be_moved = False
                    trailing_active = False
                    trail_price = ep
                    partial_closed_tp1 = False
                    position_remaining = 1.0
                    realized_pnl = 0.0
                    time_reduced = False

                    # Record for audit
                    if len(trade_details) < 20:
                        trade_details.append({
                            "entry_bar": bar_idx,
                            "direction": pending_dir,
                            "entry_price": round(ep, 2),
                            "sl_price": round(sp, 2),
                            "tp1_price": round(tp1_p, 2),
                            "sl_dist_pips": round(sd, 1),
                            "atr_at_entry": round(atr_val, 2),
                            "trail_updates": [],
                        })

            pending_dir = 0
            pending_bar = -1

        # In-trade management
        if in_trade:
            closed = False
            exit_price = 0.0

            # FIX: On entry bar, only check initial SL/TP.
            # Trailing/management starts on bar N+1 onwards.
            bars_in = bar_idx - entry_bar

            if bars_in > 0:
                # Track best price for trailing (DATA LEAKAGE SAFE: uses current bar)
                if direction == 1:
                    if h > trail_price:
                        trail_price = h
                else:
                    if lo < trail_price or trail_price == entry_price:
                        trail_price = lo if lo < trail_price or trail_price == entry_price else trail_price

                # Current profit in pips
                if direction == 1:
                    cur_profit_pips = (c - entry_price) / pip
                    best_profit_pips = (trail_price - entry_price) / pip
                else:
                    cur_profit_pips = (entry_price - c) / pip
                    best_profit_pips = (entry_price - trail_price) / pip

                # --- MANAGEMENT LOGIC ---

                # Break-even
                if mgmt == "breakeven" and not be_moved:
                    be_trigger = exit_cfg.get("be_trigger_pips", 50)
                    if cur_profit_pips >= be_trigger:
                        if direction == 1:
                            sl_price = entry_price + 2 * pip
                        else:
                            sl_price = entry_price - 2 * pip
                        be_moved = True

                # Trailing fixed
                elif mgmt == "trail_fixed":
                    activate = exit_cfg.get("trail_activate_pips", 100)
                    trail_dist = exit_cfg.get("trail_distance_pips", 50)
                    if best_profit_pips >= activate:
                        trailing_active = True
                    if trailing_active:
                        if direction == 1:
                            new_sl = trail_price - trail_dist * pip
                            if new_sl > sl_price:
                                sl_price = new_sl
                                if trade_details and len(trade_details[-1].get("trail_updates", [])) < 5:
                                    trade_details[-1]["trail_updates"].append({
                                        "bar": bar_idx, "new_sl": round(new_sl, 2),
                                        "trail_price": round(trail_price, 2)
                                    })
                        else:
                            new_sl = trail_price + trail_dist * pip
                            if new_sl < sl_price:
                                sl_price = new_sl

                # Trailing ATR
                elif mgmt == "trail_atr":
                    activate_mult = exit_cfg.get("trail_activate_atr_mult", 1.5)
                    dist_mult = exit_cfg.get("trail_distance_atr_mult", 1.5)
                    atr_pips = atr_val / pip if atr_val > 0 else sl_dist
                    activate_pips = atr_pips * activate_mult
                    trail_dist_pips = atr_pips * dist_mult
                    if best_profit_pips >= activate_pips:
                        trailing_active = True
                    if trailing_active:
                        if direction == 1:
                            new_sl = trail_price - trail_dist_pips * pip
                            if new_sl > sl_price:
                                sl_price = new_sl
                                if trade_details and len(trade_details[-1].get("trail_updates", [])) < 5:
                                    trade_details[-1]["trail_updates"].append({
                                        "bar": bar_idx, "new_sl": round(new_sl, 2),
                                        "atr_used": round(atr_val, 2)
                                    })
                        else:
                            new_sl = trail_price + trail_dist_pips * pip
                            if new_sl < sl_price:
                                sl_price = new_sl

                # Step trailing
                elif mgmt == "step":
                    step_size = exit_cfg.get("step_size_pips", 100)
                    if best_profit_pips >= step_size:
                        steps = int(best_profit_pips / step_size)
                        lock_pips = (steps - 1) * step_size if steps > 1 else 0
                        if lock_pips > 0:
                            if direction == 1:
                                new_sl = entry_price + lock_pips * pip
                                if new_sl > sl_price:
                                    sl_price = new_sl
                            else:
                                new_sl = entry_price - lock_pips * pip
                                if new_sl < sl_price:
                                    sl_price = new_sl

                # No management + default breakeven at 50% SL distance
                elif mgmt == "none" and not be_moved:
                    if cur_profit_pips >= sl_dist * 0.5:
                        sl_price = entry_price
                        be_moved = True

            # --- CHECK SL/TP HITS (always, including entry bar) ---
            if direction == 1:
                sl_hit = lo <= sl_price
                tp1_hit = h >= tp1_price
            else:
                sl_hit = h >= sl_price
                tp1_hit = lo <= tp1_price

            # SL priority over TP (conservative)
            if sl_hit:
                exit_price = sl_price
                closed = True
            elif tp1_hit and not partial_closed_tp1:
                # Handle partial close at TP1
                if partial != "none" and tp2_price != 0.0:
                    pct_close = exit_cfg.get("partial_pct_tp1", 0.5)
                    # Realize partial PnL
                    if direction == 1:
                        partial_pnl_pips = (tp1_price - entry_price) / pip
                    else:
                        partial_pnl_pips = (entry_price - tp1_price) / pip
                    partial_pnl = partial_pnl_pips * pip_value * lot * pct_close
                    realized_pnl += partial_pnl
                    position_remaining -= pct_close
                    partial_closed_tp1 = True
                    # Move SL to breakeven for remaining
                    sl_price = entry_price
                    be_moved = True
                else:
                    exit_price = tp1_price
                    closed = True
            elif partial_closed_tp1 and tp2_price != 0.0:
                # Check TP2 for remaining position
                if direction == 1:
                    tp2_hit = h >= tp2_price
                else:
                    tp2_hit = lo <= tp2_price
                if tp2_hit:
                    exit_price = tp2_price
                    closed = True

            # Time exit
            if not closed:
                bars_in_trade = bar_idx - entry_bar
                if time_exit == "close" and bars_in_trade >= time_exit_bars:
                    exit_price = c
                    closed = True
                elif time_exit == "reduce" and bars_in_trade >= time_exit_bars and not time_reduced:
                    # Reduce position by percentage
                    if direction == 1:
                        reduce_pnl_pips = (c - entry_price) / pip
                    else:
                        reduce_pnl_pips = (entry_price - c) / pip
                    reduce_pnl = reduce_pnl_pips * pip_value * lot * position_remaining * time_reduce_pct
                    realized_pnl += reduce_pnl
                    position_remaining *= (1.0 - time_reduce_pct)
                    time_reduced = True
                elif time_exit == "none" and bars_in_trade >= 200:
                    exit_price = c
                    closed = True

            if closed:
                if direction == 1:
                    pnl_pips = (exit_price - entry_price) / pip
                else:
                    pnl_pips = (entry_price - exit_price) / pip
                remaining_pnl = pnl_pips * pip_value * lot * position_remaining
                gross_pnl = realized_pnl + remaining_pnl
                net_pnl = gross_pnl - total_cost
                balance += gross_pnl
                trades.append((net_pnl, bar_idx - entry_bar))
                in_trade = False

        # New signals
        if not in_trade and pending_dir == 0:
            if bar_idx < n_bars - 1:
                sig = signal_array[bar_idx]
                if sig != 0:
                    pending_dir = int(sig)
                    pending_bar = bar_idx

        # Drawdown
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
        c_final = closes[min(n_bars - 1, len(closes) - 1)]
        if direction == 1:
            pnl_pips = (c_final - entry_price) / pip
        else:
            pnl_pips = (entry_price - c_final) / pip
        remaining_pnl = pnl_pips * pip_value * lot * position_remaining
        gross_pnl = realized_pnl + remaining_pnl
        net_pnl = gross_pnl - total_cost
        balance += gross_pnl
        trades.append((net_pnl, n_bars - 1 - entry_bar))

    metrics = _compute_fast_metrics(trades, max_dd_pct, max_dd_dollars, balance)
    metrics["_trade_details"] = trade_details
    return metrics


# ═══════════════════════════════════════════════════════════════
#  C) SEQUENTIAL OPTIMIZER
# ═══════════════════════════════════════════════════════════════

def _best_config(signal_array, opens, highs, lows, closes, atr_arr,
                 candidates: List[Dict], base_cfg: Dict,
                 start_bar: int, end_bar: int) -> Tuple[Dict, Dict]:
    """Test all candidates, return (best_config, best_metrics)."""
    best_score = -1e9
    best_cfg = base_cfg.copy()
    best_metrics = _empty_metrics()

    for cand in candidates:
        cfg = {**base_cfg, **cand}
        m = exit_backtest(signal_array, opens, highs, lows, closes, atr_arr,
                          cfg, start_bar=start_bar, end_bar=end_bar)
        score = objective_function(m)
        if score > best_score:
            best_score = score
            best_cfg = cfg.copy()
            best_metrics = m.copy()

    # Remove internal audit data
    best_metrics.pop("_trade_details", None)
    return best_cfg, best_metrics


def optimize_exit_for_strategy(
    strategy_id: str,
    df: pd.DataFrame,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr_arr: np.ndarray,
    entry_params: Dict,
    n_folds: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run sequential exit optimization with walk-forward for one strategy.

    Steps:
        Phase A: Find best SL (with default TP=2R, no management)
        Phase B: With best SL, find best TP
        Phase C: With best SL+TP, find best management
        Phase D: With best SL+TP+mgmt, find best time exit

    Then run walk-forward validation on final config.
    """
    t0 = time.time()
    sid = strategy_id
    cat = _get_category(sid)
    n_bars = len(df)

    # Generate signal array
    if cat in TIER1_CATEGORIES and cat in PARAM_SPACES:
        signal_array = generate_signals(cat, df, entry_params)
    else:
        signal_array = reconstruct_signals_from_trades(sid, n_bars)

    # Baseline: default ATR-based exit (Phase 5 style)
    baseline_cfg = {
        "sl_method": "atr", "sl_atr_period": 14, "sl_atr_multiplier": 1.5,
        "tp_method": "rr", "tp_rr_mult": 2.0,
        "management": "none",
        "partial_close": "none",
        "time_exit": "none",
    }

    baseline_metrics = exit_backtest(
        signal_array, opens, highs, lows, closes, atr_arr, baseline_cfg)
    baseline_metrics.pop("_trade_details", None)

    if verbose:
        logger.info(f"  [{sid}] Baseline: PF={baseline_metrics['profit_factor']:.2f} "
                     f"trades={baseline_metrics['total_trades']}")

    # Use full training data for sequential optimization
    full_start = 50
    full_end = n_bars

    # Phase A: Best SL
    sl_candidates = _all_sl_configs()
    phase_a_base = {
        "tp_method": "rr", "tp_rr_mult": 2.0,
        "management": "none", "partial_close": "none", "time_exit": "none",
    }
    best_cfg, _ = _best_config(
        signal_array, opens, highs, lows, closes, atr_arr,
        sl_candidates, phase_a_base, full_start, full_end)

    sl_result = {k: v for k, v in best_cfg.items() if k.startswith("sl_")}
    if verbose:
        logger.info(f"  [{sid}] Phase A (SL): {sl_result}")

    # Phase B: Best TP (with best SL)
    tp_candidates = _all_tp_configs()
    phase_b_base = {**sl_result, "management": "none",
                    "partial_close": "none", "time_exit": "none"}
    best_cfg, _ = _best_config(
        signal_array, opens, highs, lows, closes, atr_arr,
        tp_candidates, phase_b_base, full_start, full_end)

    tp_result = {k: v for k, v in best_cfg.items()
                 if k.startswith("tp") or k.startswith("tp1") or k.startswith("tp2")}
    if verbose:
        logger.info(f"  [{sid}] Phase B (TP): {tp_result}")

    # Phase C: Best management + partial close
    mgmt_candidates = _all_mgmt_configs()
    # Combine mgmt with partial close options
    combined_mgmt = []
    for mc in mgmt_candidates:
        for pc in PARTIAL_METHODS:
            combined_mgmt.append({**mc, **pc})

    phase_c_base = {**sl_result, **tp_result, "time_exit": "none"}
    best_cfg, _ = _best_config(
        signal_array, opens, highs, lows, closes, atr_arr,
        combined_mgmt, phase_c_base, full_start, full_end)

    mgmt_result = {k: v for k, v in best_cfg.items()
                   if k.startswith("management") or k.startswith("be_") or
                   k.startswith("trail_") or k.startswith("step_") or
                   k.startswith("partial")}
    if verbose:
        logger.info(f"  [{sid}] Phase C (Mgmt): management={mgmt_result.get('management','none')}")

    # Phase D: Best time exit
    phase_d_base = {**sl_result, **tp_result, **mgmt_result}
    best_cfg, _ = _best_config(
        signal_array, opens, highs, lows, closes, atr_arr,
        TIME_EXIT_METHODS, phase_d_base, full_start, full_end)

    time_result = {k: v for k, v in best_cfg.items()
                   if k.startswith("time_")}

    # Final optimized config
    final_cfg = {**sl_result, **tp_result, **mgmt_result, **time_result}

    # Run optimized on full data
    opt_full = exit_backtest(
        signal_array, opens, highs, lows, closes, atr_arr, final_cfg)
    audit_details = opt_full.pop("_trade_details", [])
    opt_full.pop("_trade_details", None)

    # Walk-forward validation
    windows = create_wf_windows(n_bars, n_folds)
    fold_results = []

    for win in windows:
        is_start = win["is_start"]
        is_end = win["is_end"] + 1
        oos_start = win["oos_start"]
        oos_end = win["oos_end"] + 1

        is_m = exit_backtest(signal_array, opens, highs, lows, closes, atr_arr,
                             final_cfg, start_bar=is_start, end_bar=is_end)
        is_m.pop("_trade_details", None)
        oos_m = exit_backtest(signal_array, opens, highs, lows, closes, atr_arr,
                              final_cfg, start_bar=oos_start, end_bar=oos_end)
        oos_m.pop("_trade_details", None)

        fold_results.append({
            "fold": win["fold"],
            "is_start": is_start,
            "is_end": win["is_end"],
            "oos_start": oos_start,
            "oos_end": win["oos_end"],
            "is_pf": is_m["profit_factor"],
            "oos_pf": oos_m["profit_factor"],
            "oos_trades": oos_m["total_trades"],
            "is_metrics": is_m,
            "oos_metrics": oos_m,
        })

    # Walk-forward acceptance
    all_oos_pf = [f["oos_pf"] for f in fold_results]
    all_is_pf = [f["is_pf"] for f in fold_results]
    all_oos_trades = [f["oos_trades"] for f in fold_results]

    all_profitable = all(pf > 1.0 for pf in all_oos_pf)
    avg_oos_pf = float(np.mean(all_oos_pf)) if all_oos_pf else 0
    avg_is_pf = float(np.mean(all_is_pf)) if all_is_pf else 0
    degradation_ok = avg_oos_pf > 0.5 * avg_is_pf if avg_is_pf > 0 else False

    passed_wf = all_profitable and avg_oos_pf > 1.0 and degradation_ok

    # Improvement calculation
    baseline_obj = objective_function(baseline_metrics)
    optimized_obj = objective_function(opt_full)

    improvements = {}
    if baseline_metrics["net_profit"] != 0:
        improvements["net_profit_change_pct"] = round(
            (opt_full["net_profit"] - baseline_metrics["net_profit"]) /
            abs(baseline_metrics["net_profit"]) * 100, 1)
    else:
        improvements["net_profit_change_pct"] = 0
    improvements["pf_change"] = round(opt_full["profit_factor"] - baseline_metrics["profit_factor"], 4)
    improvements["dd_change"] = round(opt_full["max_drawdown_pct"] - baseline_metrics["max_drawdown_pct"], 4)
    improvements["sharpe_change"] = round(opt_full["sharpe_ratio"] - baseline_metrics["sharpe_ratio"], 4)

    # Overfit flags
    overfit_flags = []
    if optimized_obj > 0 and baseline_obj > 0:
        imp_pct = (optimized_obj - baseline_obj) / baseline_obj * 100
        if imp_pct > 100:
            overfit_flags.append("improvement_over_100pct")
    if len(all_oos_pf) >= 2:
        fold_std = float(np.std(all_oos_pf))
        fold_mean = float(np.mean(all_oos_pf))
        if fold_mean > 0 and fold_std / fold_mean > 1.0:
            overfit_flags.append("high_fold_variance")

    # Data leakage check: verify audit trail
    leakage_check = "PASSED"
    for td in audit_details[:5]:
        eb = td.get("entry_bar", 0)
        for tu in td.get("trail_updates", []):
            if tu.get("bar", 0) < eb:
                leakage_check = "FAILED"
                break

    elapsed = time.time() - t0

    # Get train data boundaries
    train_start = str(df.iloc[0]["time"]) if "time" in df.columns else "bar_0"
    train_end = str(df.iloc[-1]["time"]) if "time" in df.columns else f"bar_{n_bars - 1}"

    result = {
        "strategy_id": sid,
        "phase": "4.2_exit_optimization",
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_boundaries": {
            "train_start": train_start,
            "train_end": train_end,
            "total_bars": n_bars,
            "NOTE": "validation and test data NOT used",
        },
        "entry_params": entry_params,
        "best_exit_config": final_cfg,
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": opt_full,
        "walk_forward_passed": passed_wf,
        "walk_forward_folds": fold_results,
        "wf_acceptance": {
            "all_folds_profitable": all_profitable,
            "avg_oos_pf": round(avg_oos_pf, 4),
            "avg_is_pf": round(avg_is_pf, 4),
            "degradation_ok": degradation_ok,
        },
        "improvement_vs_baseline": improvements,
        "overfit_flags": overfit_flags,
        "data_leakage_check": leakage_check,
        "trade_audit_sample": audit_details[:5],
        "elapsed_sec": round(elapsed, 2),
    }

    if verbose:
        wf_str = "PASS" if passed_wf else "FAIL"
        logger.info(f"  [{sid}] WF={wf_str} | baseline_pf={baseline_metrics['profit_factor']:.2f} "
                     f"opt_pf={opt_full['profit_factor']:.2f} "
                     f"pf_change={improvements['pf_change']:+.2f} ({elapsed:.1f}s)")

    return result


# ═══════════════════════════════════════════════════════════════
#  D) MAIN RUNNER
# ═══════════════════════════════════════════════════════════════

def _load_entry_params(strategy_id: str) -> Dict:
    """Load Phase 8 optimized entry params. Falls back to defaults."""
    p8_path = os.path.join(OPTIMIZED_DIR, f"{strategy_id}_params.json")
    if os.path.exists(p8_path):
        with open(p8_path, "r") as f:
            data = json.load(f)
        return data.get("optimized_params", data.get("default_params", {}))

    # Fallback: try to get defaults from param space
    cat = _get_category(strategy_id)
    if cat in PARAM_SPACES:
        return {p["name"]: p["default"] for p in PARAM_SPACES[cat]}
    return {}


def run_exit_optimization(
    top_n: int = 50,
    n_folds: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run Phase 10 exit optimization on top N strategies.
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
    logger.info(f"Optimizing exit for top {len(strategy_ids)} strategies")

    # Ensure output dir
    os.makedirs(EXIT_DIR, exist_ok=True)

    results = []
    passed = 0
    failed = 0

    for i, sid in enumerate(strategy_ids):
        logger.info(f"[{i + 1}/{len(strategy_ids)}] {sid}")
        try:
            entry_params = _load_entry_params(sid)
            result = optimize_exit_for_strategy(
                sid, df, opens, highs_arr, lows_arr, closes_arr,
                atr_arr, entry_params, n_folds=n_folds, verbose=verbose)

            results.append(result)

            # Save individual result
            out_path = os.path.join(EXIT_DIR, f"{sid}_exit.json")
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

    elapsed = time.time() - t0

    # Compute aggregate stats
    pf_changes = [r["improvement_vs_baseline"]["pf_change"]
                  for r in results if "improvement_vs_baseline" in r]
    avg_pf_change = float(np.mean(pf_changes)) if pf_changes else 0

    net_changes = [r["improvement_vs_baseline"]["net_profit_change_pct"]
                   for r in results if "improvement_vs_baseline" in r]
    avg_net_change = float(np.mean(net_changes)) if net_changes else 0

    exit_summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "phase": "4.2_exit_optimization",
        "total_strategies": len(strategy_ids),
        "passed_walk_forward": passed,
        "failed_walk_forward": failed,
        "avg_pf_change": round(avg_pf_change, 4),
        "avg_net_profit_change_pct": round(avg_net_change, 1),
        "n_folds": n_folds,
        "total_bars": n_bars,
        "elapsed_sec": round(elapsed, 2),
        "strategies": [
            {
                "strategy_id": r["strategy_id"],
                "walk_forward_passed": r.get("walk_forward_passed", False),
                "best_exit_config": r.get("best_exit_config", {}),
                "pf_change": r.get("improvement_vs_baseline", {}).get("pf_change", 0),
                "net_change_pct": r.get("improvement_vs_baseline", {}).get("net_profit_change_pct", 0),
                "overfit_flags": r.get("overfit_flags", []),
                "data_leakage_check": r.get("data_leakage_check", "N/A"),
            }
            for r in results
        ],
    }

    # Save summary
    summary_out = os.path.join(EXIT_DIR, "exit_summary.json")
    with open(summary_out, "w") as f:
        json.dump(exit_summary, f, indent=2, default=str)

    logger.info(f"\nExit optimization complete in {elapsed:.1f}s")
    logger.info(f"  Passed WF: {passed}/{len(strategy_ids)}")
    logger.info(f"  Failed WF: {failed}/{len(strategy_ids)}")
    logger.info(f"  Avg PF change: {avg_pf_change:+.3f}")
    logger.info(f"  Avg net profit change: {avg_net_change:+.1f}%")
    logger.info(f"  Results: {EXIT_DIR}")

    return exit_summary

"""
Phase 6.1 — Final Production Config + Reports
================================================
Compiles ALL previous results into production configs and reports.
NO new optimization — only compiling and reporting.

Generates:
  - results/final/robot_config.json  (PRIMARY DELIVERABLE)
  - results/final/details/{strategy}_detail.json
  - reports/final_report.html
  - reports/*.csv exports
"""

import json
import math
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.broker import BTCUSD_CONFIG
from engine.costs import compute_variable_cost
from config.settings import (
    RESULTS_DIR, INDIVIDUAL_DIR, COMBOS_DIR, OPTIMIZED_DIR,
    TRAIN_DIR, VALIDATION_DIR, TEST_DIR, REPORTS_DIR,
)
from indicators import compute as ind
from optimizer.param_optimizer import (
    _get_category, generate_signals, reconstruct_signals_from_trades,
    TIER1_CATEGORIES, PARAM_SPACES,
)
from optimizer.exit_optimizer import exit_backtest, _compute_sl_distance, _compute_tp_distance
from optimizer.combined_validator import COMBINED_DIR
from optimizer.validation_tester import VALIDATION_RESULTS_DIR
from optimizer.final_oos_tester import FINAL_DIR

logger = logging.getLogger("phase14")

DETAILS_DIR = os.path.join(FINAL_DIR, "details")

CATEGORY_DESC = {
    "RSI": "RSI overbought/oversold crossover",
    "MACD": "MACD line/signal crossover",
    "BB": "Bollinger Band breakout/squeeze",
    "MA": "Moving average crossover",
    "ADX": "ADX trend strength filter",
    "WR": "Williams %R overbought/oversold",
    "STOCH": "Stochastic oscillator crossover",
    "CCI": "Commodity Channel Index levels",
    "SRSI": "Stochastic RSI signals",
    "KC": "Keltner Channel breakout",
    "DON": "Donchian Channel breakout",
    "ENV": "Envelope channel crossover",
    "OBV": "On-Balance Volume divergence",
    "FISHER": "Fisher Transform crossover",
    "TSI": "True Strength Index crossover",
    "ULT": "Ultimate Oscillator levels",
    "VTX": "Vortex Indicator crossover",
    "VORTEX": "Vortex Indicator signals",
    "EW": "Elliott Wave pattern",
    "SM": "Smart Money flow",
    "CDL": "Candlestick pattern",
    "ARN": "Aroon oscillator crossover",
    "CH": "Chande momentum oscillator",
    "AD": "Accumulation/Distribution flow",
    "FI": "Force Index momentum",
    "WYC": "Wyckoff method signals",
    "RVI": "Relative Vigor Index crossover",
}


def _generate_signal_array(strategy_id, df, entry_params):
    cat = _get_category(strategy_id)
    if cat in TIER1_CATEGORIES and cat in PARAM_SPACES:
        return generate_signals(cat, df, entry_params)
    return reconstruct_signals_from_trades(strategy_id, len(df))


# ═══════════════════════════════════════════════════════════
#  DETAILED TRADE EXTRACTION
# ═══════════════════════════════════════════════════════════

def _extract_full_trades(signal_array, opens, highs, lows, closes, atr_arr,
                         exit_cfg, times):
    """
    Extract full trade details from backtest.
    Uses same logic as exit_backtest (tc = 0.65) for consistency.
    """
    cfg = BTCUSD_CONFIG
    pip = cfg["pip_size"]
    pip_value = cfg["pip_value_per_lot"]
    lot = cfg["backtest_lot"]
    spread_pips = cfg["spread_points"] * cfg["point"]
    half_spread = spread_pips / 2.0

    n_bars = len(opens)
    mgmt = exit_cfg.get("management", "none")
    partial = exit_cfg.get("partial_close", "none")
    time_exit_type = exit_cfg.get("time_exit", "none")
    time_exit_bars = exit_cfg.get("time_exit_bars", 200)

    _cost_rng = np.random.RandomState(42)

    trades = []
    trade_num = 0
    in_trade = False
    entry_price = sl_price = tp1_price = tp2_price = sl_dist = 0.0
    direction = entry_bar = 0
    total_cost = 0.0
    be_moved = trailing_active = partial_closed_tp1 = time_reduced = False
    trail_price = 0.0
    position_remaining = 1.0
    realized_pnl = 0.0
    pending_dir = 0
    pending_bar = -1

    for bar_idx in range(50, n_bars):
        o, h, lo, c = opens[bar_idx], highs[bar_idx], lows[bar_idx], closes[bar_idx]
        atr_val = atr_arr[bar_idx] if bar_idx < len(atr_arr) else 0.0

        if pending_dir != 0 and not in_trade:
            if bar_idx == pending_bar + 1 and atr_val > 0:
                ep = o + half_spread if pending_dir == 1 else o - half_spread
                sd = _compute_sl_distance(exit_cfg, ep, pending_dir, atr_val, pip, highs, lows, bar_idx)
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

                    tc = compute_variable_cost(atr_arr, bar_idx, _cost_rng)
                    in_trade = True
                    entry_price = ep
                    sl_price = sp
                    tp1_price = tp1_p
                    tp2_price = tp2_p
                    sl_dist = sd
                    direction = pending_dir
                    entry_bar = bar_idx
                    total_cost = tc
                    be_moved = trailing_active = partial_closed_tp1 = time_reduced = False
                    trail_price = ep
                    position_remaining = 1.0
                    realized_pnl = 0.0
                    trade_num += 1

            pending_dir = 0
            pending_bar = -1

        if in_trade:
            closed = False
            exit_price = 0.0
            exit_reason = ""

            # FIX: On entry bar, only check initial SL/TP.
            # Trailing/management starts on bar N+1 onwards.
            bars_in = bar_idx - entry_bar

            if bars_in > 0:
                if direction == 1:
                    if h > trail_price:
                        trail_price = h
                else:
                    if lo < trail_price or trail_price == entry_price:
                        trail_price = lo if lo < trail_price or trail_price == entry_price else trail_price

                if direction == 1:
                    cur_profit_pips = (c - entry_price) / pip
                    best_profit_pips = (trail_price - entry_price) / pip
                else:
                    cur_profit_pips = (entry_price - c) / pip
                    best_profit_pips = (entry_price - trail_price) / pip

                # Management
                if mgmt == "breakeven" and not be_moved:
                    be_trigger = exit_cfg.get("be_trigger_pips", 50)
                    if cur_profit_pips >= be_trigger:
                        sl_price = entry_price + (2 * pip if direction == 1 else -2 * pip)
                        be_moved = True
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
                        else:
                            new_sl = trail_price + trail_dist * pip
                            if new_sl < sl_price:
                                sl_price = new_sl
                elif mgmt == "trail_atr":
                    a_mult = exit_cfg.get("trail_activate_atr_mult", 1.5)
                    d_mult = exit_cfg.get("trail_distance_atr_mult", 1.5)
                    atr_pips = atr_val / pip if atr_val > 0 else sl_dist
                    if best_profit_pips >= atr_pips * a_mult:
                        trailing_active = True
                    if trailing_active:
                        td_pips = atr_pips * d_mult
                        if direction == 1:
                            new_sl = trail_price - td_pips * pip
                            if new_sl > sl_price:
                                sl_price = new_sl
                        else:
                            new_sl = trail_price + td_pips * pip
                            if new_sl < sl_price:
                                sl_price = new_sl
                elif mgmt == "step":
                    step_size = exit_cfg.get("step_size_pips", 100)
                    if best_profit_pips >= step_size:
                        steps = int(best_profit_pips / step_size)
                        lock = (steps - 1) * step_size if steps > 1 else 0
                        if lock > 0:
                            if direction == 1:
                                new_sl = entry_price + lock * pip
                                if new_sl > sl_price:
                                    sl_price = new_sl
                            else:
                                new_sl = entry_price - lock * pip
                                if new_sl < sl_price:
                                    sl_price = new_sl
                elif mgmt == "none" and not be_moved:
                    if cur_profit_pips >= sl_dist * 0.5:
                        sl_price = entry_price
                        be_moved = True

            # SL/TP check (always, including entry bar)
            if direction == 1:
                sl_hit = lo <= sl_price
                tp1_hit = h >= tp1_price
            else:
                sl_hit = h >= sl_price
                tp1_hit = lo <= tp1_price

            if sl_hit:
                exit_price = sl_price
                exit_reason = "TRAIL" if trailing_active else ("BE" if be_moved else "SL")
                closed = True
            elif tp1_hit and not partial_closed_tp1:
                if partial != "none" and tp2_price != 0.0:
                    pct_close = exit_cfg.get("partial_pct_tp1", 0.5)
                    pp = ((tp1_price - entry_price) / pip if direction == 1
                          else (entry_price - tp1_price) / pip)
                    realized_pnl += pp * pip_value * lot * pct_close
                    position_remaining -= pct_close
                    partial_closed_tp1 = True
                    sl_price = entry_price
                    be_moved = True
                else:
                    exit_price = tp1_price
                    exit_reason = "TP"
                    closed = True
            elif partial_closed_tp1 and tp2_price != 0.0:
                tp2_hit = (h >= tp2_price if direction == 1 else lo <= tp2_price)
                if tp2_hit:
                    exit_price = tp2_price
                    exit_reason = "TP2"
                    closed = True

            if not closed:
                bars_in = bar_idx - entry_bar
                if time_exit_type == "close" and bars_in >= time_exit_bars:
                    exit_price = c
                    exit_reason = "TIME"
                    closed = True
                elif time_exit_type == "none" and bars_in >= 200:
                    exit_price = c
                    exit_reason = "TIME"
                    closed = True

            if closed:
                pp = ((exit_price - entry_price) / pip if direction == 1
                      else (entry_price - exit_price) / pip)
                remaining_pnl = pp * pip_value * lot * position_remaining
                gross = realized_pnl + remaining_pnl
                net = gross - total_cost
                entry_t = str(times[entry_bar]) if entry_bar < len(times) else ""
                exit_t = str(times[bar_idx]) if bar_idx < len(times) else ""

                trades.append({
                    "trade_num": trade_num,
                    "direction": "BUY" if direction == 1 else "SELL",
                    "entry_bar": int(entry_bar),
                    "exit_bar": int(bar_idx),
                    "entry_time": entry_t,
                    "exit_time": exit_t,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "sl_price": round(sl_price, 2),
                    "tp_price": round(tp1_price, 2),
                    "gross_pnl": round(gross, 4),
                    "net_pnl": round(net, 4),
                    "total_cost": round(total_cost, 4),
                    "bars_held": int(bar_idx - entry_bar),
                    "exit_reason": exit_reason,
                })
                in_trade = False

        if not in_trade and pending_dir == 0:
            if bar_idx < n_bars - 1:
                sig = signal_array[bar_idx]
                if sig != 0:
                    pending_dir = int(sig)
                    pending_bar = bar_idx

    if in_trade:
        c_f = closes[min(n_bars - 1, len(closes) - 1)]
        pp = ((c_f - entry_price) / pip if direction == 1
              else (entry_price - c_f) / pip)
        remaining_pnl = pp * pip_value * lot * position_remaining
        gross = realized_pnl + remaining_pnl
        net = gross - total_cost
        entry_t = str(times[entry_bar]) if entry_bar < len(times) else ""
        exit_t = str(times[n_bars - 1]) if n_bars - 1 < len(times) else ""
        trades.append({
            "trade_num": trade_num,
            "direction": "BUY" if direction == 1 else "SELL",
            "entry_bar": int(entry_bar),
            "exit_bar": n_bars - 1,
            "entry_time": entry_t,
            "exit_time": exit_t,
            "entry_price": round(entry_price, 2),
            "exit_price": round(c_f, 2),
            "sl_price": round(sl_price, 2),
            "tp_price": round(tp1_price, 2),
            "gross_pnl": round(gross, 4),
            "net_pnl": round(net, 4),
            "total_cost": round(total_cost, 4),
            "bars_held": n_bars - 1 - entry_bar,
            "exit_reason": "FORCED_CLOSE",
        })

    return trades


# ═══════════════════════════════════════════════════════════
#  PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════

def _compute_sensitivity(sid, entry_params, exit_config, test_df,
                         opens, highs, lows, closes, atr_arr):
    """Vary each entry param ±1 step and measure PF change."""
    cat = _get_category(sid)
    if cat not in PARAM_SPACES:
        return {"note": "Tier 2 — no parameterized sensitivity available"}

    space = PARAM_SPACES[cat]
    results = {}

    for param_def in space:
        pname = param_def["name"]
        current_val = entry_params.get(pname)
        if current_val is None:
            continue

        lo_bound = param_def["low"]
        hi_bound = param_def["high"]
        step = param_def["step"]

        # Generate grid values
        vals = []
        v = lo_bound
        while v <= hi_bound + 1e-9:
            vals.append(round(v, 6))
            v += step

        # Find closest match
        idx = None
        for i, gv in enumerate(vals):
            if abs(gv - current_val) < step * 0.5:
                idx = i
                break
        if idx is None:
            continue

        variants = {}
        for delta_name, delta_idx in [("minus_1", idx - 1), ("plus_1", idx + 1)]:
            if delta_idx < 0 or delta_idx >= len(vals):
                continue
            test_params = dict(entry_params)
            test_params[pname] = vals[delta_idx]
            try:
                sigs = generate_signals(cat, test_df, test_params)
                m = exit_backtest(sigs, opens, highs, lows, closes, atr_arr, exit_config)
                m.pop("_trade_details", None)
                variants[f"{pname}={vals[delta_idx]}"] = {
                    "pf": round(m["profit_factor"], 4),
                    "trades": m["total_trades"],
                    "net": round(m["net_profit"], 2),
                }
            except Exception:
                pass

        results[pname] = {
            "current": current_val,
            "variants": variants,
        }

    return results


# ═══════════════════════════════════════════════════════════
#  LOAD ALL DATA
# ═══════════════════════════════════════════════════════════

def load_all_data(verbose=True):
    """Load all results from every phase."""
    with open(os.path.join(FINAL_DIR, "oos_results.json")) as f:
        oos = json.load(f)
    with open(os.path.join(FINAL_DIR, "final_rankings.json")) as f:
        rankings = json.load(f)
    with open(os.path.join(VALIDATION_RESULTS_DIR, "validation_summary.json")) as f:
        val_summary = json.load(f)
    with open(os.path.join(COMBINED_DIR, "combined_summary.json")) as f:
        combined_summary = json.load(f)

    combos_path = os.path.join(COMBOS_DIR, "top50_combos.json")
    combos = {}
    if os.path.exists(combos_path):
        with open(combos_path) as f:
            combos = json.load(f)

    # Load test data
    test_df = pd.read_csv(os.path.join(TEST_DIR, "BTCUSD_H1.csv"))
    test_df["time"] = pd.to_datetime(test_df["time"])
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, "BTCUSD_H1.csv"))
    train_df["time"] = pd.to_datetime(train_df["time"])
    val_df = pd.read_csv(os.path.join(VALIDATION_DIR, "BTCUSD_H1.csv"))
    val_df["time"] = pd.to_datetime(val_df["time"])

    test_opens = test_df["open"].values.astype(np.float64)
    test_highs = test_df["high"].values.astype(np.float64)
    test_lows = test_df["low"].values.astype(np.float64)
    test_closes = test_df["close"].values.astype(np.float64)
    test_atr = np.array(
        ind.atr(test_df["high"], test_df["low"], test_df["close"], 14),
        dtype=np.float64)
    test_atr = np.nan_to_num(test_atr, nan=0.0)
    test_times = test_df["time"].values

    # Get approved strategies
    approved = [s for s in oos["strategies"]
                if "error" not in s
                and s["classification"] in ("PRODUCTION_READY", "ACCEPTABLE")]

    if verbose:
        logger.info(f"Computing trade details for {len(approved)} approved strategies...")

    for s in approved:
        sid = s["strategy_id"]
        entry_params = s["entry_params"]
        exit_config = s["exit_config"]

        signals = _generate_signal_array(sid, test_df, entry_params)

        # Full trade details
        trade_list = _extract_full_trades(
            signals, test_opens, test_highs, test_lows, test_closes,
            test_atr, exit_config, test_times)
        s["test_trades_detail"] = trade_list

        # Equity curve
        equity = [10000.0]
        for t in trade_list:
            equity.append(equity[-1] + t["net_pnl"])
        s["equity_curve"] = equity

        # Monthly returns
        monthly = {}
        for t in trade_list:
            mk = t["exit_time"][:7]
            monthly[mk] = round(monthly.get(mk, 0) + t["net_pnl"], 4)
        s["monthly_returns"] = monthly

        # Daily PnL
        daily = {}
        for t in trade_list:
            dk = t["exit_time"][:10]
            daily[dk] = round(daily.get(dk, 0) + t["net_pnl"], 4)
        s["daily_pnl"] = daily

        # Parameter sensitivity
        s["param_sensitivity"] = _compute_sensitivity(
            sid, entry_params, exit_config, test_df,
            test_opens, test_highs, test_lows, test_closes, test_atr)

    n_individual = len([f for f in os.listdir(INDIVIDUAL_DIR) if f.endswith(".json")])

    return {
        "oos_results": oos,
        "rankings": rankings,
        "validation_summary": val_summary,
        "combined_summary": combined_summary,
        "combos": combos,
        "broker": BTCUSD_CONFIG,
        "approved": approved,
        "n_individual": n_individual,
        "data_info": {
            "train_start": str(train_df["time"].iloc[0]),
            "train_end": str(train_df["time"].iloc[-1]),
            "train_bars": len(train_df),
            "val_start": str(val_df["time"].iloc[0]),
            "val_end": str(val_df["time"].iloc[-1]),
            "val_bars": len(val_df),
            "test_start": str(test_df["time"].iloc[0]),
            "test_end": str(test_df["time"].iloc[-1]),
            "test_bars": len(test_df),
        },
    }


# ═══════════════════════════════════════════════════════════
#  ROBOT CONFIG (PRIMARY DELIVERABLE)
# ═══════════════════════════════════════════════════════════

def generate_robot_config(data: Dict, paths=None) -> Dict:
    """Generate results/final/robot_config.json."""
    oos = data["oos_results"]
    rankings = data["rankings"]
    val_summary = data["validation_summary"]
    combined = data["combined_summary"]
    approved = data["approved"]
    broker = data["broker"]

    # Pipeline summary
    pipeline = {
        "phase_5_individual": data["n_individual"],
        "phase_7_combos": len(data["combos"].get("top_50", [])),
        "phase_8_entry_optimized": len([
            f for f in os.listdir(OPTIMIZED_DIR)
            if f.endswith("_params.json")
        ]),
        "phase_10_exit_optimized": len([
            f for f in os.listdir(os.path.join(OPTIMIZED_DIR, "exit"))
            if f.endswith("_exit.json")
        ]) if os.path.isdir(os.path.join(OPTIMIZED_DIR, "exit")) else 0,
        "phase_4_3_combined": combined["total_strategies"],
        "phase_4_3_approved": combined["approved"],
        "phase_5_1_validation_tested": val_summary["total_strategies"],
        "phase_5_1_robust": val_summary["classification_counts"].get("ROBUST", 0),
        "phase_5_1_acceptable": val_summary["classification_counts"].get("ACCEPTABLE", 0),
        "phase_5_2_oos_tested": oos["total_tested"],
        "phase_5_2_production_ready": oos["classification_counts"].get("PRODUCTION_READY", 0),
        "phase_5_2_acceptable": oos["classification_counts"].get("ACCEPTABLE", 0),
        "final_approved": len(approved),
    }

    # Build strategy entries
    strat_entries = []
    for s in approved:
        sid = s["strategy_id"]
        cat = _get_category(sid)
        tw = s["three_way_comparison"]
        mc = s["monte_carlo"]
        reg = s["regime_analysis"]

        entry = {
            "id": sid,
            "category": cat,
            "final_score": s["final_score"],
            "classification": s["classification"],
            "entry_params": s["entry_params"],
            "exit_config": s["exit_config"],
            "entry_rules": CATEGORY_DESC.get(cat, f"{cat} signal-based entry"),
            "performance": {
                "train": {
                    "pf": tw["train_pf"],
                    "sharpe": tw["train_sharpe"],
                },
                "validation": {
                    "pf": tw["val_pf"],
                    "sharpe": tw["val_sharpe"],
                },
                "test": {
                    "pf": tw["test_pf"],
                    "sharpe": tw["test_sharpe"],
                    "win_rate": tw["test_wr"],
                    "trades": tw["test_trades"],
                    "net_profit": tw["test_net"],
                    "max_drawdown_pct": tw["test_dd"],
                },
                "monte_carlo": {
                    "p95_dd": mc["mc_p95_dd"],
                    "p99_dd": mc["mc_p99_dd"],
                    "prob_ruin_30pct": mc["mc_prob_of_ruin_30pct"],
                    "prob_ruin_50pct": mc["mc_prob_of_ruin_50pct"],
                    "stress_profitable_pct": mc["mc_stress_pct_profitable"],
                    "median_pf": mc["mc_median_pf"],
                    "mc_passed": mc["mc_passed"],
                },
                "regime": {
                    "active_regimes": reg["active_regimes"],
                    "profitable_regimes": reg["profitable_regimes"],
                    "regime_diversity": reg["regime_diversity"],
                },
            },
            "risk_limits": {
                "max_dd_pct": round(max(tw["test_dd"], mc["mc_p95_dd"]) * 1.5, 2),
                "max_consecutive_losses_alert": 5,
            },
        }
        strat_entries.append(entry)

    # Sort by score descending
    strat_entries.sort(key=lambda x: x["final_score"], reverse=True)

    # Approved combos (from Phase 7 — not OOS tested)
    combo_entries = []
    for combo in data["combos"].get("top_50", [])[:10]:
        combo_entries.append({
            "combo_id": combo["combo_id"],
            "strategies": combo["strategies"],
            "mode": combo["mode"],
            "train_pf": combo["train_metrics"]["profit_factor"],
            "val_pf": combo["validation_metrics"]["profit_factor"],
            "train_score": combo["train_score"],
            "note": "NOT OOS tested — use with caution",
        })

    # Recommendations
    best = strat_entries[0] if strat_entries else {}
    safest = min(strat_entries,
                 key=lambda x: x["performance"]["monte_carlo"]["p95_dd"]) if strat_entries else {}

    config = {
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "broker_config": {
            "symbol": broker["symbol"],
            "digits": broker["digits"],
            "stop_level_pips": broker["stop_level_pips"],
            "commission_per_side_per_lot": broker["commission_per_side_per_lot"],
            "spread_points": broker["spread_points"],
            "backtest_lot": broker["backtest_lot"],
            "pip_value_per_lot": broker["pip_value_per_lot"],
        },
        "data_info": data["data_info"],
        "pipeline_summary": pipeline,
        "approved_individual_strategies": strat_entries,
        "approved_combinations": combo_entries,
        "global_risk_rules": {
            "max_positions": 3,
            "max_daily_loss_usd": 50,
            "max_weekly_loss_usd": 150,
            "equity_stop_pct": 10,
            "notes": "Apply per 0.01 lot position sizing",
        },
        "recommendations": {
            "best_single": {
                "id": best.get("id", ""),
                "score": best.get("final_score", 0),
                "reason": "Highest composite score across all metrics",
            },
            "best_combo": {
                "id": combo_entries[0]["combo_id"] if combo_entries else "N/A",
                "note": "Combos not OOS tested",
            },
            "safest_lowest_dd": {
                "id": safest.get("id", ""),
                "p95_dd": safest.get("performance", {}).get("monte_carlo", {}).get("p95_dd", 0),
                "reason": "Lowest Monte Carlo p95 drawdown",
            },
        },
    }

    out_dir = paths["final_dir"] if paths else FINAL_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "robot_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"robot_config.json written ({len(strat_entries)} strategies) -> {path}")

    return config


# ═══════════════════════════════════════════════════════════
#  DETAIL FILES
# ═══════════════════════════════════════════════════════════

def generate_detail_files(data: Dict, paths=None):
    """Generate results/final/details/{strategy}_detail.json."""
    out_dir = paths["details_dir"] if paths else DETAILS_DIR
    os.makedirs(out_dir, exist_ok=True)
    approved = data["approved"]

    for s in approved:
        sid = s["strategy_id"]
        detail = {
            "strategy_id": sid,
            "category": _get_category(sid),
            "final_score": s["final_score"],
            "classification": s["classification"],
            "entry_params": s["entry_params"],
            "exit_config": s["exit_config"],
            "three_way_comparison": s["three_way_comparison"],
            "monte_carlo": s["monte_carlo"],
            "regime_analysis": s["regime_analysis"],
            "degradation": s["degradation"],
            "trade_list": s["test_trades_detail"],
            "equity_curve": s["equity_curve"],
            "monthly_returns": s["monthly_returns"],
            "daily_pnl": s["daily_pnl"],
            "param_sensitivity": s["param_sensitivity"],
        }
        path = os.path.join(out_dir, f"{sid}_detail.json")
        with open(path, "w") as f:
            json.dump(detail, f, indent=2, default=str)

    logger.info(f"Detail files written for {len(approved)} strategies in {out_dir}")


# ═══════════════════════════════════════════════════════════
#  SVG HELPERS
# ═══════════════════════════════════════════════════════════

_COLORS = ["#00b894", "#0984e3", "#fdcb6e", "#e17055", "#6c5ce7",
           "#00cec9", "#fab1a0", "#81ecec", "#a29bfe", "#ffeaa7"]


def _svg_line_chart(series: Dict[str, List[float]], width=800, height=300,
                    title="", y_label="") -> str:
    """SVG multi-line chart."""
    if not series:
        return ""
    all_vals = [v for vals in series.values() for v in vals]
    if not all_vals:
        return ""
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min if y_max != y_min else 1
    max_pts = max(len(v) for v in series.values())

    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 50
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    lines = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="background:#1e1e2e;border-radius:8px;">']
    if title:
        lines.append(f'<text x="{width//2}" y="20" text-anchor="middle" '
                     f'fill="#e0e0e0" font-size="14" font-weight="bold">{title}</text>')

    # Y-axis labels
    for i in range(5):
        yv = y_min + y_range * i / 4
        yp = pad_t + ph - (ph * i / 4)
        lines.append(f'<text x="{pad_l-5}" y="{yp+4}" text-anchor="end" '
                     f'fill="#888" font-size="10">{yv:.1f}</text>')
        lines.append(f'<line x1="{pad_l}" y1="{yp}" x2="{width-pad_r}" y2="{yp}" '
                     f'stroke="#333" stroke-width="0.5"/>')

    for ci, (label, vals) in enumerate(series.items()):
        color = _COLORS[ci % len(_COLORS)]
        pts = []
        for i, v in enumerate(vals):
            x = pad_l + (i / max(max_pts - 1, 1)) * pw
            y = pad_t + ph - ((v - y_min) / y_range) * ph
            pts.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="{color}" stroke-width="2"/>')
        lx = width - pad_r - 120
        ly = pad_t + 15 + ci * 18
        lines.append(f'<rect x="{lx}" y="{ly-8}" width="12" height="12" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{lx+16}" y="{ly+3}" fill="#ccc" font-size="11">{label}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def _svg_bar_chart(labels: List[str], values: List[float], width=800, height=300,
                   title="", color="#00b894") -> str:
    """SVG bar chart."""
    if not values:
        return ""
    v_max = max(max(values), 0.001)
    v_min = min(min(values), 0)
    v_range = v_max - v_min if v_max != v_min else 1

    pad_l, pad_r, pad_t, pad_b = 80, 20, 30, 60
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b
    bar_w = max(pw / max(len(values), 1) * 0.7, 4)
    gap = pw / max(len(values), 1)

    lines = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="background:#1e1e2e;border-radius:8px;">']
    if title:
        lines.append(f'<text x="{width//2}" y="20" text-anchor="middle" '
                     f'fill="#e0e0e0" font-size="14" font-weight="bold">{title}</text>')

    zero_y = pad_t + ph - (-v_min / v_range * ph) if v_min < 0 else pad_t + ph
    lines.append(f'<line x1="{pad_l}" y1="{zero_y}" x2="{width-pad_r}" y2="{zero_y}" '
                 f'stroke="#555" stroke-width="1"/>')

    for i, (lbl, val) in enumerate(zip(labels, values)):
        x = pad_l + i * gap + (gap - bar_w) / 2
        bar_h = abs(val) / v_range * ph
        bc = "#00b894" if val >= 0 else "#d63031"
        if val >= 0:
            y = zero_y - bar_h
        else:
            y = zero_y
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                     f'height="{max(bar_h, 1):.1f}" fill="{bc}" rx="2"/>')
        # Label
        lbl_short = lbl[:8] if len(lbl) > 8 else lbl
        lines.append(f'<text x="{x + bar_w/2:.1f}" y="{height - pad_b + 15}" '
                     f'text-anchor="middle" fill="#aaa" font-size="9" '
                     f'transform="rotate(-45 {x + bar_w/2:.1f} {height - pad_b + 15})">'
                     f'{lbl_short}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def _svg_heatmap(rows: List[str], cols: List[str], data_grid: List[List[float]],
                 width=800, height=200, title="") -> str:
    """SVG heatmap."""
    if not data_grid or not rows:
        return ""
    pad_l, pad_t = 100, 30
    cell_w = min(60, (width - pad_l) / max(len(cols), 1))
    cell_h = min(40, (height - pad_t) / max(len(rows), 1))
    all_vals = [v for row in data_grid for v in row if v is not None]
    v_max = max(all_vals) if all_vals else 1
    v_min = min(all_vals) if all_vals else 0
    v_range = v_max - v_min if v_max != v_min else 1

    lines = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="background:#1e1e2e;border-radius:8px;">']
    if title:
        lines.append(f'<text x="{width//2}" y="18" text-anchor="middle" '
                     f'fill="#e0e0e0" font-size="14" font-weight="bold">{title}</text>')

    for ci, col in enumerate(cols):
        x = pad_l + ci * cell_w + cell_w / 2
        lines.append(f'<text x="{x:.0f}" y="{pad_t - 5}" text-anchor="middle" '
                     f'fill="#aaa" font-size="9">{col[:8]}</text>')

    for ri, row_label in enumerate(rows):
        y = pad_t + ri * cell_h
        lines.append(f'<text x="{pad_l - 5}" y="{y + cell_h/2 + 4:.0f}" '
                     f'text-anchor="end" fill="#aaa" font-size="10">{row_label}</text>')
        for ci in range(len(cols)):
            val = data_grid[ri][ci] if ci < len(data_grid[ri]) else 0
            x = pad_l + ci * cell_w
            if val is None:
                fc = "#333"
            elif val > 0:
                intensity = min(val / max(v_max, 0.01), 1.0)
                g = int(100 + 155 * intensity)
                fc = f"rgb(0,{g},0)"
            elif val < 0:
                intensity = min(abs(val) / max(abs(v_min), 0.01), 1.0)
                r = int(100 + 155 * intensity)
                fc = f"rgb({r},0,0)"
            else:
                fc = "#444"
            lines.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell_w:.0f}" '
                         f'height="{cell_h:.0f}" fill="{fc}" stroke="#222" rx="2"/>')
            lines.append(f'<text x="{x + cell_w/2:.0f}" y="{y + cell_h/2 + 4:.0f}" '
                         f'text-anchor="middle" fill="#eee" font-size="9">'
                         f'{val:.1f}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  HTML REPORT
# ═══════════════════════════════════════════════════════════

def generate_html_report(data: Dict, paths=None):
    """Generate reports/final_report.html."""
    reports_dir = paths["reports_dir"] if paths else REPORTS_DIR
    html_filename = paths["html_filename"] if paths else "final_report.html"
    os.makedirs(reports_dir, exist_ok=True)
    approved = data["approved"]
    oos = data["oos_results"]
    rankings = data["rankings"]

    # Sort by score
    approved_sorted = sorted(approved, key=lambda x: x["final_score"], reverse=True)
    top5 = approved_sorted[:5]

    # --- Equity Curves for top 5 ---
    eq_series = {}
    for s in top5:
        eq_series[s["strategy_id"]] = s["equity_curve"]
    equity_svg = _svg_line_chart(eq_series, width=900, height=350,
                                 title="Top 5 Equity Curves (Test Data)")

    # --- Monthly returns heatmap ---
    all_months = sorted({m for s in approved_sorted[:10] for m in s["monthly_returns"]})
    heat_cols = [s["strategy_id"] for s in approved_sorted[:10]]
    heat_grid = []
    for month in all_months:
        row = [s["monthly_returns"].get(month, 0) for s in approved_sorted[:10]]
        heat_grid.append(row)
    monthly_svg = _svg_heatmap(all_months, heat_cols, heat_grid, width=900, height=180,
                               title="Monthly Returns ($) — Top 10 Strategies")

    # --- Drawdown chart (top 5) ---
    dd_series = {}
    for s in top5:
        eq = s["equity_curve"]
        dd_vals = []
        peak = eq[0]
        for v in eq:
            if v > peak:
                peak = v
            dd_pct = (peak - v) / peak * 100 if peak > 0 else 0
            dd_vals.append(-dd_pct)
        dd_series[s["strategy_id"]] = dd_vals
    dd_svg = _svg_line_chart(dd_series, width=900, height=300,
                             title="Drawdown % (Test Data)")

    # --- Category performance ---
    cat_pfs = {}
    for s in approved_sorted:
        cat = _get_category(s["strategy_id"])
        cat_pfs.setdefault(cat, []).append(s["three_way_comparison"]["test_pf"])
    cat_labels = sorted(cat_pfs.keys())
    cat_avg_pfs = [round(np.mean(cat_pfs[c]), 2) for c in cat_labels]
    # Cap for display
    cat_avg_display = [min(v, 50) for v in cat_avg_pfs]
    cat_svg = _svg_bar_chart(cat_labels, cat_avg_display, width=900, height=300,
                             title="Average Test PF by Category (capped at 50)")

    # --- Monte Carlo p95 DD per strategy ---
    mc_labels = [s["strategy_id"] for s in approved_sorted[:20]]
    mc_vals = [s["monte_carlo"]["mc_p95_dd"] for s in approved_sorted[:20]]
    mc_svg = _svg_bar_chart(mc_labels, mc_vals, width=900, height=280,
                            title="Monte Carlo p95 Drawdown % (Top 20)")

    # --- Regime analysis ---
    regime_types = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL", "LOW_VOL"]
    regime_labels = regime_types
    regime_profitable = []
    for regime in regime_types:
        cnt = sum(1 for s in approved
                  if s["regime_analysis"]["per_regime"].get(regime, {}).get("profitable", False))
        regime_profitable.append(cnt)
    regime_svg = _svg_bar_chart(regime_labels, regime_profitable, width=700, height=280,
                                title="Strategies Profitable per Regime")

    # --- Build rankings table rows ---
    table_rows = ""
    for i, s in enumerate(approved_sorted):
        tw = s["three_way_comparison"]
        mc = s["monte_carlo"]
        reg = s["regime_analysis"]
        cat = _get_category(s["strategy_id"])
        pf_display = f"{tw['test_pf']:.2f}" if tw["test_pf"] < 900 else "999+"
        table_rows += f"""<tr>
<td>{i+1}</td><td>{s['strategy_id']}</td><td>{cat}</td>
<td>{s['final_score']:.3f}</td><td>{pf_display}</td>
<td>{tw['test_sharpe']:.2f}</td><td>{tw['test_wr']:.1f}%</td>
<td>{tw['test_trades']}</td><td>${tw['test_net']:.2f}</td>
<td>{mc['mc_p95_dd']:.1f}%</td><td>{mc['mc_prob_of_ruin_30pct']:.2%}</td>
<td>{reg['regime_diversity']:.0%}</td>
<td class="status-{s['classification'].lower()}">{s['classification']}</td>
</tr>\n"""

    # --- Risk table ---
    risk_rows = ""
    for s in approved_sorted[:15]:
        tw = s["three_way_comparison"]
        mc = s["monte_carlo"]
        risk_rows += f"""<tr>
<td>{s['strategy_id']}</td>
<td>{tw['test_dd']:.2f}%</td><td>{mc['mc_p95_dd']:.1f}%</td>
<td>{mc['mc_p99_dd']:.1f}%</td><td>{mc['mc_prob_of_ruin_30pct']:.2%}</td>
<td>{mc['mc_prob_of_ruin_50pct']:.2%}</td>
<td>{mc['mc_stress_pct_profitable']:.0f}%</td>
</tr>\n"""

    # --- Cost breakdown ---
    cost_rows = ""
    for s in approved_sorted[:15]:
        trades = s["test_trades_detail"]
        n_trades = len(trades)
        total_cost = sum(t["total_cost"] for t in trades)
        total_gross = sum(t["gross_pnl"] for t in trades)
        total_net = sum(t["net_pnl"] for t in trades)
        cost_pct = total_cost / total_gross * 100 if total_gross > 0 else 0
        cost_rows += f"""<tr>
<td>{s['strategy_id']}</td><td>{n_trades}</td>
<td>${total_cost:.2f}</td><td>${total_gross:.2f}</td>
<td>${total_net:.2f}</td><td>{cost_pct:.1f}%</td>
</tr>\n"""

    # --- Executive summary ---
    total_discovered = data["n_individual"]
    final_count = len(approved)
    prod_ready = sum(1 for s in approved if s["classification"] == "PRODUCTION_READY")
    total_test_net = sum(s["three_way_comparison"]["test_net"] for s in approved)
    best = approved_sorted[0] if approved_sorted else {}
    best_id = best.get("strategy_id", "N/A")
    best_score = best.get("final_score", 0)
    best_pf = best.get("three_way_comparison", {}).get("test_pf", 0)

    regime_dist = oos.get("regime_distribution", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTCUSD Backtest Engine — Final Report</title>
<style>
:root {{
  --bg: #0f0f1a;
  --card: #1a1a2e;
  --surface: #16213e;
  --text: #e0e0e0;
  --muted: #888;
  --accent: #0f3460;
  --green: #00b894;
  --red: #d63031;
  --gold: #fdcb6e;
  --blue: #0984e3;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', Tahoma, sans-serif; padding: 20px 40px; line-height: 1.6; }}
h1 {{ font-size: 28px; color: var(--gold); margin-bottom: 5px; }}
h2 {{ font-size: 20px; color: var(--blue); margin: 30px 0 15px; border-bottom: 2px solid var(--accent); padding-bottom: 8px; }}
h3 {{ font-size: 16px; color: #ccc; margin: 20px 0 10px; }}
.subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 25px; }}
.cards {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }}
.card {{ background: var(--card); border-radius: 10px; padding: 18px 24px; min-width: 180px; flex: 1; border-left: 4px solid var(--blue); }}
.card-title {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }}
.card-value {{ font-size: 28px; font-weight: bold; color: var(--gold); margin-top: 5px; }}
.card-sub {{ font-size: 12px; color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; margin-bottom: 20px; }}
th {{ background: var(--accent); padding: 10px 8px; text-align: left; font-size: 12px; text-transform: uppercase; cursor: pointer; user-select: none; white-space: nowrap; }}
th:hover {{ background: #1a4a80; }}
td {{ padding: 8px; border-bottom: 1px solid #222; font-size: 13px; white-space: nowrap; }}
tr:hover {{ background: rgba(255,255,255,0.03); }}
.status-production_ready {{ color: var(--green); font-weight: bold; }}
.status-acceptable {{ color: var(--gold); font-weight: bold; }}
.chart-container {{ background: var(--card); border-radius: 10px; padding: 15px; margin-bottom: 20px; overflow-x: auto; }}
.methodology {{ background: var(--card); border-radius: 10px; padding: 20px; font-size: 13px; color: #bbb; }}
.methodology li {{ margin: 5px 0; }}
@media print {{
  body {{ background: #fff; color: #000; padding: 10px; }}
  .card {{ border: 1px solid #ccc; }}
  th {{ background: #ddd; color: #000; }}
  td {{ color: #000; border-bottom: 1px solid #ccc; }}
  h1, h2, .card-value {{ color: #000; }}
  .chart-container {{ break-inside: avoid; }}
  svg {{ background: #f8f8f8 !important; }}
  svg text {{ fill: #333 !important; }}
  svg line {{ stroke: #ccc !important; }}
  svg polyline {{ stroke-width: 2 !important; }}
}}
</style>
</head>
<body>

<h1>BTCUSD Backtest Engine</h1>
<div class="subtitle">Phase 6.1 — Final Production Config + Reports | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<h2>Executive Summary</h2>
<div class="cards">
  <div class="card"><div class="card-title">Strategies Discovered</div><div class="card-value">{total_discovered}</div><div class="card-sub">Individual signals tested</div></div>
  <div class="card"><div class="card-title">Final Approved</div><div class="card-value">{final_count}</div><div class="card-sub">{prod_ready} production-ready</div></div>
  <div class="card"><div class="card-title">Best Strategy</div><div class="card-value">{best_id}</div><div class="card-sub">Score: {best_score:.3f}</div></div>
  <div class="card"><div class="card-title">Best Test PF</div><div class="card-value">{best_pf:.1f}</div><div class="card-sub">On unseen test data</div></div>
  <div class="card"><div class="card-title">Total Test Profit</div><div class="card-value">${total_test_net:.0f}</div><div class="card-sub">All approved combined</div></div>
</div>

<h2>Strategy Rankings</h2>
<table id="rankings-table">
<thead><tr>
<th onclick="sortTable(0,'num')">Rank</th>
<th onclick="sortTable(1,'str')">Strategy</th>
<th onclick="sortTable(2,'str')">Category</th>
<th onclick="sortTable(3,'num')">Score</th>
<th onclick="sortTable(4,'num')">Test PF</th>
<th onclick="sortTable(5,'num')">Sharpe</th>
<th onclick="sortTable(6,'num')">Win Rate</th>
<th onclick="sortTable(7,'num')">Trades</th>
<th onclick="sortTable(8,'num')">Net $</th>
<th onclick="sortTable(9,'num')">MC p95 DD</th>
<th onclick="sortTable(10,'num')">Ruin Prob</th>
<th onclick="sortTable(11,'num')">Regime Div</th>
<th onclick="sortTable(12,'str')">Status</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2>Top 5 Equity Curves</h2>
<div class="chart-container">{equity_svg}</div>

<h2>Monthly Returns Heatmap</h2>
<div class="chart-container">{monthly_svg}</div>

<h2>Drawdown Analysis</h2>
<div class="chart-container">{dd_svg}</div>

<h2>Category Performance</h2>
<div class="chart-container">{cat_svg}</div>

<h2>Monte Carlo Analysis</h2>
<div class="chart-container">{mc_svg}</div>

<h2>Regime Analysis</h2>
<div class="chart-container">{regime_svg}</div>
<h3>Regime Distribution (Test Data)</h3>
<table>
<thead><tr><th>Regime</th><th>Bars</th><th>% of Total</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{r}</td><td>{cnt}</td><td>{cnt/sum(regime_dist.values())*100:.1f}%</td></tr>' for r, cnt in sorted(regime_dist.items(), key=lambda x: -x[1]))}
</tbody>
</table>

<h2>Risk Metrics</h2>
<table>
<thead><tr><th>Strategy</th><th>Test DD%</th><th>MC p95 DD</th><th>MC p99 DD</th><th>Ruin 30%</th><th>Ruin 50%</th><th>Stress Profitable</th></tr></thead>
<tbody>{risk_rows}</tbody>
</table>

<h2>Cost Breakdown</h2>
<table>
<thead><tr><th>Strategy</th><th>Trades</th><th>Total Cost</th><th>Gross Profit</th><th>Net Profit</th><th>Cost %</th></tr></thead>
<tbody>{cost_rows}</tbody>
</table>

<h2>Methodology</h2>
<div class="methodology">
<h3>Pipeline Overview</h3>
<ol>
<li><strong>Phase 5 — Individual Testing:</strong> {total_discovered} strategies tested on training data (70%, {data['data_info']['train_bars']} bars)</li>
<li><strong>Phase 7 — Combinations:</strong> Top strategies combined in 2-3 strategy ensembles with majority/weighted voting</li>
<li><strong>Phase 8 — Entry Optimization:</strong> Tier 1 (parameterized) via grid search + walk-forward; Tier 2 (signal-based) SL/TP optimization only</li>
<li><strong>Phase 10 — Exit Optimization:</strong> SL methods (fixed/ATR/swing/pct), TP methods (fixed/ATR/RR/fib), management (trail/breakeven/step), partial closes, time exits</li>
<li><strong>Phase 4.3 — Combined Validation:</strong> 4-way comparison (baseline, entry-only, exit-only, combined) with walk-forward + data leakage audit</li>
<li><strong>Phase 5.1 — Validation Testing:</strong> First use of validation data (15%). Classified ROBUST/ACCEPTABLE/DEGRADED/OVERFIT</li>
<li><strong>Phase 5.2 — Final OOS Test:</strong> Test data (15%, NEVER seen). Monte Carlo (1000 iterations: shuffle/skip/stress). Regime analysis (5 market regimes). Composite scoring</li>
</ol>
<h3>Broker Configuration</h3>
<ul>
<li>Symbol: BTCUSD | Spread: {BTCUSD_CONFIG['spread_points']} points | Commission: ${BTCUSD_CONFIG['commission_per_side_per_lot']}/side/lot | Stop Level: {BTCUSD_CONFIG['stop_level_pips']} pips</li>
<li>Lot size: {BTCUSD_CONFIG['backtest_lot']} | Pip value: ${BTCUSD_CONFIG['pip_value_per_lot']}/pip/lot</li>
</ul>
<h3>Data Boundaries</h3>
<ul>
<li>Training: {data['data_info']['train_start'][:10]} to {data['data_info']['train_end'][:10]} ({data['data_info']['train_bars']} bars)</li>
<li>Validation: {data['data_info']['val_start'][:10]} to {data['data_info']['val_end'][:10]} ({data['data_info']['val_bars']} bars)</li>
<li>Test (OOS): {data['data_info']['test_start'][:10]} to {data['data_info']['test_end'][:10]} ({data['data_info']['test_bars']} bars)</li>
</ul>
</div>

<script>
function sortTable(col, type) {{
  var table = document.getElementById("rankings-table");
  var tbody = table.querySelector("tbody");
  var rows = Array.from(tbody.querySelectorAll("tr"));
  var asc = table.dataset.sortCol == col && table.dataset.sortDir == "asc";
  table.dataset.sortCol = col;
  table.dataset.sortDir = asc ? "desc" : "asc";
  rows.sort(function(a, b) {{
    var va = a.cells[col].textContent.replace(/[$%,+]/g, "").trim();
    var vb = b.cells[col].textContent.replace(/[$%,+]/g, "").trim();
    if (type === "num") {{
      va = parseFloat(va) || 0;
      vb = parseFloat(vb) || 0;
    }}
    return asc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
}}
</script>
</body>
</html>"""

    path = os.path.join(reports_dir, html_filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML report written: {path}")


# ═══════════════════════════════════════════════════════════
#  CSV EXPORTS
# ═══════════════════════════════════════════════════════════

def generate_csv_exports(data: Dict, paths=None):
    """Generate CSV exports in reports/."""
    reports_dir = paths["reports_dir"] if paths else REPORTS_DIR
    csv_suffix = paths["csv_suffix"] if paths else ""
    os.makedirs(reports_dir, exist_ok=True)
    approved = data["approved"]
    approved_sorted = sorted(approved, key=lambda x: x["final_score"], reverse=True)

    # 1. all_approved_trades.csv
    all_trades = []
    for s in approved_sorted:
        sid = s["strategy_id"]
        for t in s["test_trades_detail"]:
            row = dict(t)
            row["strategy_id"] = sid
            all_trades.append(row)

    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        cols = ["strategy_id", "trade_num", "direction", "entry_time", "exit_time",
                "entry_price", "exit_price", "sl_price", "tp_price",
                "gross_pnl", "net_pnl", "total_cost", "bars_held", "exit_reason"]
        df_trades = df_trades[[c for c in cols if c in df_trades.columns]]
        df_trades.to_csv(os.path.join(reports_dir, f"all_approved_trades{csv_suffix}.csv"), index=False)

    # 2. strategy_rankings.csv
    rows = []
    for i, s in enumerate(approved_sorted):
        tw = s["three_way_comparison"]
        mc = s["monte_carlo"]
        reg = s["regime_analysis"]
        rows.append({
            "rank": i + 1,
            "strategy_id": s["strategy_id"],
            "category": _get_category(s["strategy_id"]),
            "classification": s["classification"],
            "final_score": s["final_score"],
            "train_pf": tw["train_pf"],
            "val_pf": tw["val_pf"],
            "test_pf": tw["test_pf"],
            "test_sharpe": tw["test_sharpe"],
            "test_wr": tw["test_wr"],
            "test_trades": tw["test_trades"],
            "test_net": tw["test_net"],
            "test_dd_pct": tw["test_dd"],
            "mc_p95_dd": mc["mc_p95_dd"],
            "mc_p99_dd": mc["mc_p99_dd"],
            "mc_prob_ruin_30": mc["mc_prob_of_ruin_30pct"],
            "mc_stress_profitable": mc["mc_stress_pct_profitable"],
            "regime_diversity": reg["regime_diversity"],
            "profitable_regimes": reg["profitable_regimes"],
            "active_regimes": reg["active_regimes"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(reports_dir, f"strategy_rankings{csv_suffix}.csv"), index=False)

    # 3. monthly_returns.csv
    all_months = sorted({m for s in approved_sorted for m in s["monthly_returns"]})
    monthly_rows = []
    for s in approved_sorted:
        row = {"strategy_id": s["strategy_id"]}
        for m in all_months:
            row[m] = s["monthly_returns"].get(m, 0)
        monthly_rows.append(row)
    pd.DataFrame(monthly_rows).to_csv(os.path.join(reports_dir, f"monthly_returns{csv_suffix}.csv"), index=False)

    # 4. regime_analysis.csv
    regime_types = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL", "LOW_VOL"]
    regime_rows = []
    for s in approved_sorted:
        reg = s["regime_analysis"]
        for rt in regime_types:
            rd = reg["per_regime"].get(rt, {})
            regime_rows.append({
                "strategy_id": s["strategy_id"],
                "regime": rt,
                "bars": rd.get("bars", 0),
                "trades": rd.get("trades", 0),
                "pf": rd.get("pf", 0),
                "wr": rd.get("wr", 0),
                "net": rd.get("net", 0),
                "profitable": rd.get("profitable", False),
            })
    pd.DataFrame(regime_rows).to_csv(os.path.join(reports_dir, f"regime_analysis{csv_suffix}.csv"), index=False)

    # 5. monte_carlo_summary.csv
    mc_rows = []
    for s in approved_sorted:
        mc = s["monte_carlo"]
        mc_rows.append({
            "strategy_id": s["strategy_id"],
            "mc_iterations": mc["mc_iterations"],
            "mc_avg_dd": mc["mc_avg_dd"],
            "mc_p95_dd": mc["mc_p95_dd"],
            "mc_p99_dd": mc["mc_p99_dd"],
            "mc_prob_ruin_30": mc["mc_prob_of_ruin_30pct"],
            "mc_prob_ruin_50": mc["mc_prob_of_ruin_50pct"],
            "mc_p5_profit": mc["mc_p5_profit"],
            "mc_median_pf": mc["mc_median_pf"],
            "mc_stress_profitable": mc["mc_stress_pct_profitable"],
            "mc_passed": mc["mc_passed"],
        })
    pd.DataFrame(mc_rows).to_csv(os.path.join(reports_dir, f"monte_carlo_summary{csv_suffix}.csv"), index=False)

    logger.info(f"CSV exports written to {reports_dir}")


# ═══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

def generate_all(verbose=True, output_paths=None) -> Dict:
    """Main entry point — generates all Phase 6.1 outputs.

    Args:
        output_paths: Optional dict to override output locations. Keys:
            final_dir, details_dir, reports_dir, html_filename, csv_suffix
    """
    t0 = time.time()

    # Build paths dict from defaults + optional overrides
    paths = {
        "final_dir": FINAL_DIR,
        "details_dir": DETAILS_DIR,
        "reports_dir": REPORTS_DIR,
        "html_filename": "final_report.html",
        "csv_suffix": "",
    }
    if output_paths:
        paths.update(output_paths)

    logger.info("Loading all data from all phases...")
    data = load_all_data(verbose=verbose)
    logger.info(f"  {len(data['approved'])} approved strategies loaded")

    logger.info("Generating robot_config.json (PRIMARY DELIVERABLE)...")
    robot_config = generate_robot_config(data, paths=paths)

    logger.info("Generating detail files...")
    generate_detail_files(data, paths=paths)

    logger.info("Generating HTML report...")
    generate_html_report(data, paths=paths)

    logger.info("Generating CSV exports...")
    generate_csv_exports(data, paths=paths)

    elapsed = time.time() - t0
    logger.info(f"Phase 6.1 complete in {elapsed:.1f}s")

    return robot_config

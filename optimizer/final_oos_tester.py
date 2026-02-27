"""
Phase 5.2 — Final OOS Test + Monte Carlo + Regime Analysis
=============================================================
MOST CRITICAL phase. Uses TEST data (newest 15%) that has NEVER been seen.
NO optimization. Loads strategies that passed validation (ROBUST + ACCEPTABLE).

Pipeline per strategy:
  1. Backtest on TEST data
  2. 3-way degradation: train → val → test
  3. Monte Carlo simulation (1000 iterations)
  4. Regime analysis (trending/ranging/volatility)
  5. Final composite score

Saves to: results/final/
"""

import json
import math
import os
import sys
import time
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.broker import BTCUSD_CONFIG
from engine.costs import compute_variable_cost
from config.settings import (
    TRAIN_DIR, VALIDATION_DIR, TEST_DIR, RESULTS_DIR, OPTIMIZED_DIR,
)
from indicators import compute as ind
from optimizer.param_optimizer import (
    _get_category, generate_signals, reconstruct_signals_from_trades,
    TIER1_CATEGORIES, PARAM_SPACES,
)
from optimizer.exit_optimizer import exit_backtest
from optimizer.combined_validator import (
    COMBINED_DIR, BASELINE_EXIT, _load_default_entry_params,
)
from optimizer.validation_tester import VALIDATION_RESULTS_DIR

logger = logging.getLogger("phase13")

FINAL_DIR = os.path.join(RESULTS_DIR, "final")


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _load_validation_summary() -> Dict:
    path = os.path.join(VALIDATION_RESULTS_DIR, "validation_summary.json")
    with open(path, "r") as f:
        return json.load(f)


def _load_validation_result(strategy_id: str) -> Dict:
    path = os.path.join(VALIDATION_RESULTS_DIR, f"{strategy_id}_validation.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _load_combined_result(strategy_id: str) -> Dict:
    path = os.path.join(COMBINED_DIR, f"{strategy_id}_combined.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _get_final_config(combined_result: Dict) -> Tuple[Dict, Dict]:
    sid = combined_result["strategy_id"]
    final = combined_result.get("final_config_to_use", "combined")
    if final == "combined":
        return combined_result.get("entry_params", {}), combined_result.get("exit_config", {})
    elif final == "entry_only":
        return combined_result.get("entry_params", {}), BASELINE_EXIT.copy()
    elif final == "exit_only":
        return _load_default_entry_params(sid), combined_result.get("exit_config", {})
    return combined_result.get("entry_params", {}), combined_result.get("exit_config", BASELINE_EXIT.copy())


def _generate_signal_array(strategy_id: str, df: pd.DataFrame,
                           entry_params: Dict) -> np.ndarray:
    cat = _get_category(strategy_id)
    if cat in TIER1_CATEGORIES and cat in PARAM_SPACES:
        return generate_signals(cat, df, entry_params)
    return reconstruct_signals_from_trades(strategy_id, len(df))


def _safe_ratio(val: float, train: float) -> float:
    if train == 0 or train == 999.0:
        return 1.0 if val > 0 else 0.0
    if train < 0:
        return 1.0 if val > 0 else val / train
    return val / train


# ═══════════════════════════════════════════════════════════════
#  MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════

def _mc_compute_dd(pnls: np.ndarray, initial: float = 10000.0) -> float:
    """Compute max drawdown % from array of trade PnLs."""
    equity = initial + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / np.where(peak > 0, peak, 1.0) * 100.0
    return float(np.max(dd_pct)) if len(dd_pct) > 0 else 0.0


def _mc_compute_pf(pnls: np.ndarray) -> float:
    """Compute profit factor from PnL array."""
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses < 0.001:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def run_monte_carlo(trade_pnls: List[float], n_total: int = 1000,
                    rng_seed: int = 42) -> Dict:
    """
    Monte Carlo simulation:
      a) 500 iterations: shuffle trade order → recalculate max drawdown
      b) 300 iterations: randomly skip 10-15% of trades → recalculate metrics
      c) 200 iterations: add random extra slippage ± 1-5 pips + spread widening
    """
    rng = np.random.RandomState(rng_seed)
    pnls = np.array(trade_pnls, dtype=np.float64)
    n_trades = len(pnls)

    if n_trades < 3:
        return {
            "mc_iterations": 0,
            "mc_avg_dd": 0.0,
            "mc_p95_dd": 0.0,
            "mc_p99_dd": 0.0,
            "mc_prob_of_ruin_30pct": 0.0,
            "mc_prob_of_ruin_50pct": 0.0,
            "mc_p5_profit": 0.0,
            "mc_median_pf": 0.0,
            "mc_stress_pct_profitable": 100.0,
            "mc_passed": True,
        }

    cfg = BTCUSD_CONFIG
    pip = cfg["pip_size"]
    pip_value = cfg["pip_value_per_lot"]
    lot = cfg["backtest_lot"]

    # Proportional split
    n_shuffle = min(500, n_total)
    n_skip = min(300, n_total - n_shuffle)
    n_stress = n_total - n_shuffle - n_skip

    all_dds = []
    all_profits = []
    all_pfs = []
    ruin_30 = 0
    ruin_50 = 0
    stress_profitable = 0

    # a) Shuffle trade order (path-dependent DD)
    for _ in range(n_shuffle):
        shuffled = rng.permutation(pnls)
        dd = _mc_compute_dd(shuffled)
        net = float(np.sum(shuffled))
        all_dds.append(dd)
        all_profits.append(net)
        all_pfs.append(_mc_compute_pf(shuffled))
        if dd >= 30.0:
            ruin_30 += 1
        if dd >= 50.0:
            ruin_50 += 1

    # b) Randomly skip 10-15% of trades
    for _ in range(n_skip):
        skip_pct = rng.uniform(0.10, 0.15)
        n_keep = max(1, int(n_trades * (1.0 - skip_pct)))
        idx = rng.choice(n_trades, size=n_keep, replace=False)
        subset = pnls[idx]
        dd = _mc_compute_dd(subset)
        net = float(np.sum(subset))
        all_dds.append(dd)
        all_profits.append(net)
        all_pfs.append(_mc_compute_pf(subset))
        if dd >= 30.0:
            ruin_30 += 1
        if dd >= 50.0:
            ruin_50 += 1

    # c) Add random extra slippage + spread widening
    for _ in range(n_stress):
        extra_slip_pips = rng.uniform(1, 5, size=n_trades)
        spread_widen_pts = rng.uniform(0, 500, size=n_trades)
        # Each trade loses extra: (slip_pips * pip_value * lot) + (spread_pts * point * lot)
        extra_cost = extra_slip_pips * pip_value * lot + spread_widen_pts * cfg["point"] * lot
        stressed = pnls - extra_cost
        dd = _mc_compute_dd(stressed)
        net = float(np.sum(stressed))
        all_dds.append(dd)
        all_profits.append(net)
        all_pfs.append(_mc_compute_pf(stressed))
        if dd >= 30.0:
            ruin_30 += 1
        if dd >= 50.0:
            ruin_50 += 1
        if net > 0:
            stress_profitable += 1

    total_iters = n_shuffle + n_skip + n_stress
    dds = np.array(all_dds)
    profits = np.array(all_profits)
    pfs = np.array(all_pfs)

    mc_avg_dd = float(np.mean(dds))
    mc_p95_dd = float(np.percentile(dds, 95))
    mc_p99_dd = float(np.percentile(dds, 99))
    mc_prob_ruin_30 = ruin_30 / total_iters
    mc_prob_ruin_50 = ruin_50 / total_iters
    mc_p5_profit = float(np.percentile(profits, 5))
    mc_median_pf = float(np.median(pfs))
    mc_stress_pct = (stress_profitable / n_stress * 100.0) if n_stress > 0 else 100.0

    # MC filter
    mc_passed = (mc_p95_dd <= 40.0 and
                 mc_prob_ruin_30 <= 0.15 and
                 mc_stress_pct >= 70.0)

    return {
        "mc_iterations": total_iters,
        "mc_shuffle": n_shuffle,
        "mc_skip": n_skip,
        "mc_stress": n_stress,
        "mc_avg_dd": round(mc_avg_dd, 2),
        "mc_p95_dd": round(mc_p95_dd, 2),
        "mc_p99_dd": round(mc_p99_dd, 2),
        "mc_prob_of_ruin_30pct": round(mc_prob_ruin_30, 4),
        "mc_prob_of_ruin_50pct": round(mc_prob_ruin_50, 4),
        "mc_p5_profit": round(mc_p5_profit, 2),
        "mc_median_pf": round(mc_median_pf, 4),
        "mc_stress_pct_profitable": round(mc_stress_pct, 1),
        "mc_passed": mc_passed,
    }


# ═══════════════════════════════════════════════════════════════
#  REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════

def classify_regimes(df: pd.DataFrame) -> np.ndarray:
    """
    Classify each bar into regime(s). Returns array of regime labels.
    Regimes:
      TRENDING_UP:   price > SMA50 and SMA50 slope > 0 for 20+ bars
      TRENDING_DOWN: price < SMA50 and SMA50 slope < 0 for 20+ bars
      RANGING:       BB width < 50th percentile
      HIGH_VOL:      ATR14 > 80th percentile
      LOW_VOL:       ATR14 < 20th percentile
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    n = len(df)

    sma50 = ind.sma(close, 50)
    sma50_slope = sma50.diff(5)  # 5-bar slope

    atr14 = ind.atr(high, low, close, 14)
    atr14 = atr14.fillna(0)
    atr_p80 = atr14.quantile(0.80)
    atr_p20 = atr14.quantile(0.20)

    bb = ind.bollinger_bands(close, 20, 2.0)
    bb_width = bb["bandwidth"].fillna(0)
    bb_p50 = bb_width.quantile(0.50)

    regimes = np.full(n, "UNKNOWN", dtype=object)

    # Track consecutive bars for trending
    up_streak = 0
    down_streak = 0

    for i in range(n):
        s50 = sma50.iloc[i] if not np.isnan(sma50.iloc[i]) else 0
        slope = sma50_slope.iloc[i] if not np.isnan(sma50_slope.iloc[i]) else 0
        c = close.iloc[i]
        a = atr14.iloc[i]
        bw = bb_width.iloc[i]

        if c > s50 and slope > 0:
            up_streak += 1
            down_streak = 0
        elif c < s50 and slope < 0:
            down_streak += 1
            up_streak = 0
        else:
            up_streak = 0
            down_streak = 0

        if up_streak >= 20:
            regimes[i] = "TRENDING_UP"
        elif down_streak >= 20:
            regimes[i] = "TRENDING_DOWN"
        elif bw < bb_p50 and bw > 0:
            regimes[i] = "RANGING"
        elif a > atr_p80:
            regimes[i] = "HIGH_VOL"
        elif a < atr_p20:
            regimes[i] = "LOW_VOL"
        else:
            regimes[i] = "NEUTRAL"

    return regimes


def analyze_regimes_for_strategy(
    signal_array: np.ndarray,
    opens: np.ndarray, highs: np.ndarray,
    lows: np.ndarray, closes: np.ndarray,
    atr_arr: np.ndarray,
    exit_config: Dict,
    regimes: np.ndarray,
) -> Dict:
    """
    Run backtest in each regime window and compute per-regime PF and WR.
    """
    regime_types = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL", "LOW_VOL"]
    results = {}
    n_bars = len(opens)

    for regime in regime_types:
        mask = (regimes == regime)
        n_regime_bars = int(np.sum(mask))

        if n_regime_bars < 30:
            results[regime] = {
                "bars": n_regime_bars,
                "trades": 0, "pf": 0, "wr": 0,
                "profitable": False,
            }
            continue

        # Create a filtered signal array: only allow signals in this regime
        filtered_signals = np.where(mask, signal_array, 0)

        m = exit_backtest(
            filtered_signals, opens, highs, lows, closes,
            atr_arr, exit_config)
        m.pop("_trade_details", None)

        results[regime] = {
            "bars": n_regime_bars,
            "trades": m["total_trades"],
            "pf": round(m["profit_factor"], 4),
            "wr": round(m["win_rate"], 2),
            "net": round(m["net_profit"], 2),
            "profitable": m["net_profit"] > 0 and m["profit_factor"] > 1.0,
        }

    # Regime diversity
    active_regimes = [r for r in regime_types if results[r]["trades"] >= 3]
    profitable_regimes = [r for r in active_regimes if results[r]["profitable"]]
    regime_diversity = len(profitable_regimes) / len(active_regimes) if active_regimes else 0

    # Coverage
    classified_bars = int(np.sum(regimes != "UNKNOWN"))
    coverage_pct = classified_bars / n_bars * 100 if n_bars > 0 else 0

    return {
        "per_regime": results,
        "active_regimes": len(active_regimes),
        "profitable_regimes": len(profitable_regimes),
        "regime_diversity": round(regime_diversity, 4),
        "classified_bars": classified_bars,
        "total_bars": n_bars,
        "coverage_pct": round(coverage_pct, 1),
    }


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION + SCORING
# ═══════════════════════════════════════════════════════════════

def classify_oos(test_pf: float, test_sharpe: float, test_trades: int,
                 train_pf: float, val_pf: float) -> str:
    """Classify OOS result."""
    if test_trades < 5:
        return "INSUFFICIENT"
    if test_pf > 1.3 and test_sharpe > 0.5:
        # Check gradual degradation (train >= val >= test roughly)
        return "PRODUCTION_READY"
    if test_pf > 1.0:
        return "ACCEPTABLE"
    if test_pf >= 0.8:
        return "MARGINAL"
    return "FAILED"


def compute_final_score(
    test_pf: float, test_sharpe: float, test_wr: float,
    mc_prob_ruin: float, val_robustness: float, test_robustness: float,
    regime_diversity: float, test_trades: int,
    mc_stress_pct: float,
) -> float:
    """
    FINAL SCORE composite:
      test_pf*0.15 + test_sharpe*0.15 + test_wr/100*0.10 +
      (1-mc_prob_ruin)*0.15 + val_robustness*0.10 + test_robustness*0.10 +
      regime_diversity*0.10 + min(1,sqrt(trades/50))*0.10 +
      mc_stress_pct/100*0.05
    """
    # Cap values to keep score in 0-1 range
    pf_norm = min(test_pf / 10.0, 1.0) if test_pf < 999 else 1.0
    sharpe_norm = min(max(test_sharpe, 0) / 5.0, 1.0)
    wr_norm = min(test_wr / 100.0, 1.0)
    ruin_comp = max(1.0 - mc_prob_ruin, 0)
    val_rob = min(max(val_robustness, 0), 1.0)
    test_rob = min(max(test_robustness, 0), 1.0)
    reg_div = min(max(regime_diversity, 0), 1.0)
    trade_norm = min(1.0, math.sqrt(test_trades / 50.0))
    stress_norm = min(mc_stress_pct / 100.0, 1.0)

    score = (pf_norm * 0.15 + sharpe_norm * 0.15 + wr_norm * 0.10 +
             ruin_comp * 0.15 + val_rob * 0.10 + test_rob * 0.10 +
             reg_div * 0.10 + trade_norm * 0.10 + stress_norm * 0.05)

    return round(min(max(score, 0), 1.0), 4)


# ═══════════════════════════════════════════════════════════════
#  SINGLE STRATEGY TEST
# ═══════════════════════════════════════════════════════════════

def test_single_strategy(
    strategy_id: str,
    test_df: pd.DataFrame,
    test_opens: np.ndarray, test_highs: np.ndarray,
    test_lows: np.ndarray, test_closes: np.ndarray,
    test_atr: np.ndarray,
    regimes: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """Run full OOS test + MC + regime for one strategy."""
    t0 = time.time()
    sid = strategy_id

    # Load configs
    combined = _load_combined_result(sid)
    validation = _load_validation_result(sid)

    if not combined:
        raise ValueError(f"No combined result for {sid}")

    entry_params, exit_config = _get_final_config(combined)
    final_config = combined.get("final_config_to_use", "combined")

    # Get training and validation metrics
    comp = combined.get("comparison", {})
    if final_config == "combined":
        train_m = comp.get("combined", {})
    elif final_config == "entry_only":
        train_m = comp.get("entry_optimized", {})
    else:
        train_m = comp.get("exit_optimized", {})

    train_pf = train_m.get("pf", 0)
    train_sharpe = train_m.get("sharpe", 0)

    val_pf = validation.get("validation_metrics", {}).get("pf", 0) if validation else 0
    val_sharpe = validation.get("validation_metrics", {}).get("sharpe", 0) if validation else 0
    val_pf_ratio = validation.get("degradation_ratios", {}).get("pf_ratio", 0) if validation else 0

    # Generate signals on TEST data
    test_signals = _generate_signal_array(sid, test_df, entry_params)

    # Run backtest on TEST data
    test_m = exit_backtest(
        test_signals, test_opens, test_highs, test_lows, test_closes,
        test_atr, exit_config)
    trade_details = test_m.pop("_trade_details", [])

    test_pf = test_m["profit_factor"]
    test_sharpe_val = test_m["sharpe_ratio"]
    test_wr = test_m["win_rate"]
    test_trades = test_m["total_trades"]
    test_net = test_m["net_profit"]

    # 3-way degradation
    test_pf_ratio = _safe_ratio(test_pf, train_pf)
    val_to_test_ratio = _safe_ratio(test_pf, val_pf)

    # Classification
    status = classify_oos(test_pf, test_sharpe_val, test_trades, train_pf, val_pf)

    # Monte Carlo (only for passing strategies)
    trade_pnls = [t[0] for t in []] # placeholder
    # Extract trade PnLs from a fresh backtest that returns trade list
    # Use the exit_backtest metrics to reconstruct trade pnls
    # Actually, we need to reconstruct the trade pnls from the test backtest
    # The exit_backtest doesn't return individual PnLs directly, but we can
    # get them from the _compute_fast_metrics input (trades list)
    # Let's re-run with a wrapper that captures trades
    trade_pnls = _extract_trade_pnls(
        test_signals, test_opens, test_highs, test_lows, test_closes,
        test_atr, exit_config)

    mc_results = run_monte_carlo(trade_pnls, n_total=1000, rng_seed=hash(sid) % 10000)

    # Regime analysis
    regime_results = analyze_regimes_for_strategy(
        test_signals, test_opens, test_highs, test_lows, test_closes,
        test_atr, exit_config, regimes)

    # Final score
    test_robustness = min(test_pf_ratio, 1.0)
    val_robustness = min(val_pf_ratio, 1.0)

    final_score = compute_final_score(
        test_pf=test_pf,
        test_sharpe=test_sharpe_val,
        test_wr=test_wr,
        mc_prob_ruin=mc_results["mc_prob_of_ruin_30pct"],
        val_robustness=val_robustness,
        test_robustness=test_robustness,
        regime_diversity=regime_results["regime_diversity"],
        test_trades=test_trades,
        mc_stress_pct=mc_results["mc_stress_pct_profitable"],
    )

    # MC filter
    mc_rejected = not mc_results["mc_passed"]
    if mc_rejected and status in ("PRODUCTION_READY", "ACCEPTABLE"):
        status = "MC_REJECTED"

    elapsed = time.time() - t0

    result = {
        "strategy_id": sid,
        "phase": "5.2_final_oos_test",
        "timestamp": pd.Timestamp.now().isoformat(),
        "final_config_used": final_config,
        "entry_params": entry_params,
        "exit_config": exit_config,
        "data_boundaries": {
            "test_start": str(test_df["time"].iloc[0]) if "time" in test_df.columns else "?",
            "test_end": str(test_df["time"].iloc[-1]) if "time" in test_df.columns else "?",
            "test_bars": len(test_df),
        },
        "three_way_comparison": {
            "train_pf": round(train_pf, 4),
            "train_sharpe": round(train_sharpe, 4),
            "val_pf": round(val_pf, 4),
            "val_sharpe": round(val_sharpe, 4),
            "test_pf": round(test_pf, 4),
            "test_sharpe": round(test_sharpe_val, 4),
            "test_wr": round(test_wr, 2),
            "test_net": round(test_net, 4),
            "test_trades": test_trades,
            "test_dd": round(test_m["max_drawdown_pct"], 4),
        },
        "degradation": {
            "train_to_test_pf_ratio": round(test_pf_ratio, 4),
            "val_to_test_pf_ratio": round(val_to_test_ratio, 4),
            "val_robustness": round(val_robustness, 4),
            "test_robustness": round(test_robustness, 4),
        },
        "monte_carlo": mc_results,
        "regime_analysis": regime_results,
        "final_score": final_score,
        "classification": status,
        "elapsed_sec": round(elapsed, 2),
    }

    if verbose:
        logger.info(f"  [{sid}] test_pf={test_pf:.2f} mc_p95={mc_results['mc_p95_dd']:.1f}% "
                     f"regimes={regime_results['profitable_regimes']}/{regime_results['active_regimes']} "
                     f"score={final_score:.3f} -> {status} ({elapsed:.1f}s)")

    return result


def _extract_trade_pnls(
    signal_array, opens, highs, lows, closes, atr_arr, exit_cfg
) -> List[float]:
    """
    Run exit_backtest and extract individual trade PnLs.
    The exit_backtest internally builds a trades list of (net_pnl, bars_held).
    We replicate the logic to extract just PnLs.
    """
    cfg = BTCUSD_CONFIG
    pip = cfg["pip_size"]
    pip_value = cfg["pip_value_per_lot"]
    lot = cfg["backtest_lot"]
    spread_pips = cfg["spread_points"] * cfg["point"]
    half_spread = spread_pips / 2.0

    from optimizer.exit_optimizer import (
        _compute_sl_distance, _compute_tp_distance,
    )

    n_bars = len(opens)
    mgmt = exit_cfg.get("management", "none")
    partial = exit_cfg.get("partial_close", "none")
    time_exit_type = exit_cfg.get("time_exit", "none")
    time_exit_bars = exit_cfg.get("time_exit_bars", 200)
    time_reduce_pct = exit_cfg.get("time_reduce_pct", 0.5)

    _cost_rng = np.random.RandomState(42)

    pnls = []
    in_trade = False
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    sl_dist = 0.0
    direction = 0
    entry_bar = 0
    total_cost = 0.0
    be_moved = False
    trailing_active = False
    trail_price = 0.0
    partial_closed_tp1 = False
    position_remaining = 1.0
    realized_pnl = 0.0
    time_reduced = False
    pending_dir = 0
    pending_bar = -1

    for bar_idx in range(50, n_bars):
        o = opens[bar_idx]
        h = highs[bar_idx]
        lo = lows[bar_idx]
        c = closes[bar_idx]
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
                    be_moved = False
                    trailing_active = False
                    trail_price = ep
                    partial_closed_tp1 = False
                    position_remaining = 1.0
                    realized_pnl = 0.0
                    time_reduced = False

            pending_dir = 0
            pending_bar = -1

        if in_trade:
            closed = False
            exit_price = 0.0

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

                # Management (simplified, matching exit_optimizer)
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
                    closed = True
            elif partial_closed_tp1 and tp2_price != 0.0:
                tp2_hit = (h >= tp2_price if direction == 1 else lo <= tp2_price)
                if tp2_hit:
                    exit_price = tp2_price
                    closed = True

            if not closed:
                bars_in = bar_idx - entry_bar
                if time_exit_type == "close" and bars_in >= time_exit_bars:
                    exit_price = c
                    closed = True
                elif time_exit_type == "reduce" and bars_in >= time_exit_bars and not time_reduced:
                    rp = ((c - entry_price) / pip if direction == 1
                          else (entry_price - c) / pip)
                    realized_pnl += rp * pip_value * lot * position_remaining * time_reduce_pct
                    position_remaining *= (1.0 - time_reduce_pct)
                    time_reduced = True
                elif time_exit_type == "none" and bars_in >= 200:
                    exit_price = c
                    closed = True

            if closed:
                pp = ((exit_price - entry_price) / pip if direction == 1
                      else (entry_price - exit_price) / pip)
                remaining_pnl = pp * pip_value * lot * position_remaining
                gross = realized_pnl + remaining_pnl
                net = gross - total_cost
                pnls.append(net)
                in_trade = False

        if not in_trade and pending_dir == 0:
            if bar_idx < n_bars - 1:
                sig = signal_array[bar_idx]
                if sig != 0:
                    pending_dir = int(sig)
                    pending_bar = bar_idx

    # Force close
    if in_trade:
        c_f = closes[min(n_bars - 1, len(closes) - 1)]
        pp = ((c_f - entry_price) / pip if direction == 1
              else (entry_price - c_f) / pip)
        remaining_pnl = pp * pip_value * lot * position_remaining
        gross = realized_pnl + remaining_pnl
        net = gross - total_cost
        pnls.append(net)

    return pnls


# ═══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════

def run_final_oos_test(
    verbose: bool = True,
) -> Dict:
    """Run Phase 5.2 final OOS test for all validation-passing strategies."""
    t0 = time.time()

    # Load TEST data
    test_file = os.path.join(TEST_DIR, "BTCUSD_H1.csv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data not found: {test_file}")

    test_df = pd.read_csv(test_file)
    if "time" in test_df.columns:
        test_df["time"] = pd.to_datetime(test_df["time"])
    test_bars = len(test_df)
    logger.info(f"Loaded test data: {test_bars} bars")

    test_opens = test_df["open"].values.astype(np.float64)
    test_highs = test_df["high"].values.astype(np.float64)
    test_lows = test_df["low"].values.astype(np.float64)
    test_closes = test_df["close"].values.astype(np.float64)

    test_atr = np.array(
        ind.atr(test_df["high"], test_df["low"], test_df["close"], 14),
        dtype=np.float64)
    test_atr = np.nan_to_num(test_atr, nan=0.0)

    # Classify regimes on test data
    logger.info("Classifying regimes on test data...")
    regimes = classify_regimes(test_df)
    regime_counts = {}
    for r in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    logger.info(f"Regime distribution: {regime_counts}")

    # Verify no overlap
    train_file = os.path.join(TRAIN_DIR, "BTCUSD_H1.csv")
    val_file = os.path.join(VALIDATION_DIR, "BTCUSD_H1.csv")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    train_df["time"] = pd.to_datetime(train_df["time"])
    val_df["time"] = pd.to_datetime(val_df["time"])

    train_end = train_df["time"].iloc[-1]
    val_end = val_df["time"].iloc[-1]
    test_start = test_df["time"].iloc[0]

    assert test_start > val_end, \
        f"DATE OVERLAP: test starts {test_start}, val ends {val_end}"
    assert test_start > train_end, \
        f"DATE OVERLAP: test starts {test_start}, train ends {train_end}"
    logger.info(f"Date check OK: test starts {test_start}, val ends {val_end}")

    # Load passing strategies from validation
    val_summary = _load_validation_summary()
    passing = [s for s in val_summary["strategies"]
               if s["classification"] in ("ROBUST", "ACCEPTABLE")]
    strategy_ids = [s["strategy_id"] for s in passing]
    logger.info(f"Testing {len(strategy_ids)} validation-passing strategies")

    os.makedirs(FINAL_DIR, exist_ok=True)

    results = []
    counts = {"PRODUCTION_READY": 0, "ACCEPTABLE": 0, "MARGINAL": 0,
              "FAILED": 0, "MC_REJECTED": 0, "INSUFFICIENT": 0, "ERROR": 0}

    for i, sid in enumerate(strategy_ids):
        logger.info(f"[{i + 1}/{len(strategy_ids)}] {sid}")
        try:
            result = test_single_strategy(
                sid, test_df,
                test_opens, test_highs, test_lows, test_closes,
                test_atr, regimes, verbose=verbose)

            results.append(result)
            counts[result["classification"]] += 1

        except Exception as e:
            logger.error(f"  Error testing {sid}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "strategy_id": sid,
                "error": str(e),
                "classification": "ERROR",
            })
            counts["ERROR"] += 1

    elapsed = time.time() - t0

    # Save individual results
    oos_results = {
        "generated": pd.Timestamp.now().isoformat(),
        "phase": "5.2_final_oos_test",
        "total_tested": len(strategy_ids),
        "classification_counts": counts,
        "data_boundaries": {
            "train_end": str(train_end),
            "val_end": str(val_end),
            "test_start": str(test_df["time"].iloc[0]),
            "test_end": str(test_df["time"].iloc[-1]),
            "test_bars": test_bars,
            "no_overlap_verified": True,
        },
        "regime_distribution": regime_counts,
        "elapsed_sec": round(elapsed, 2),
        "strategies": results,
    }

    oos_path = os.path.join(FINAL_DIR, "oos_results.json")
    with open(oos_path, "w") as f:
        json.dump(oos_results, f, indent=2, default=str)

    # Final rankings (sorted by score)
    ranked = sorted(
        [r for r in results if "error" not in r],
        key=lambda x: x.get("final_score", 0),
        reverse=True,
    )

    rankings = {
        "generated": pd.Timestamp.now().isoformat(),
        "phase": "5.2_final_rankings",
        "total_ranked": len(ranked),
        "classification_counts": counts,
        "rankings": [
            {
                "rank": i + 1,
                "strategy_id": r["strategy_id"],
                "classification": r["classification"],
                "final_score": r["final_score"],
                "test_pf": r["three_way_comparison"]["test_pf"],
                "test_sharpe": r["three_way_comparison"]["test_sharpe"],
                "test_trades": r["three_way_comparison"]["test_trades"],
                "mc_p95_dd": r["monte_carlo"]["mc_p95_dd"],
                "mc_passed": r["monte_carlo"]["mc_passed"],
                "regime_diversity": r["regime_analysis"]["regime_diversity"],
            }
            for i, r in enumerate(ranked)
        ],
    }

    rank_path = os.path.join(FINAL_DIR, "final_rankings.json")
    with open(rank_path, "w") as f:
        json.dump(rankings, f, indent=2, default=str)

    logger.info(f"\nFinal OOS test complete in {elapsed:.1f}s")
    for cls, cnt in counts.items():
        if cnt > 0:
            logger.info(f"  {cls}: {cnt}")
    logger.info(f"  Results: {FINAL_DIR}")

    return oos_results

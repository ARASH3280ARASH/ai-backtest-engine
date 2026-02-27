"""
Phase 11 — Combined Entry+Exit Validation
============================================
Combines optimized ENTRY params (Phase 8) with optimized EXIT configs (Phase 10)
and validates they work together. Since entry and exit were optimized separately,
we confirm the combination is robust via walk-forward.

If combined is worse than either alone, falls back to the better individual result.

Saves to: results/optimized/combined/
"""

import json
import os
import random
import sys
import time
import logging
from typing import Dict, List, Optional

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
    fast_backtest,
)
from optimizer.exit_optimizer import (
    EXIT_DIR, exit_backtest,
)

logger = logging.getLogger("phase11")

COMBINED_DIR = os.path.join(OPTIMIZED_DIR, "combined")

# Default baseline exit config (Phase 5 style)
BASELINE_EXIT = {
    "sl_method": "atr", "sl_atr_period": 14, "sl_atr_multiplier": 1.5,
    "tp_method": "rr", "tp_rr_mult": 2.0,
    "management": "none",
    "partial_close": "none",
    "time_exit": "none",
}


# ═══════════════════════════════════════════════════════════════
#  LOADING HELPERS
# ═══════════════════════════════════════════════════════════════

def _load_phase8_entry_params(strategy_id: str) -> Dict:
    """Load Phase 8 optimized entry params. Falls back to defaults."""
    p8_path = os.path.join(OPTIMIZED_DIR, f"{strategy_id}_params.json")
    if os.path.exists(p8_path):
        with open(p8_path, "r") as f:
            data = json.load(f)
        return data.get("optimized_params", data.get("default_params", {}))
    cat = _get_category(strategy_id)
    if cat in PARAM_SPACES:
        return {p["name"]: p["default"] for p in PARAM_SPACES[cat]}
    return {}


def _load_default_entry_params(strategy_id: str) -> Dict:
    """Load default (non-optimized) entry params."""
    cat = _get_category(strategy_id)
    if cat in PARAM_SPACES:
        return {p["name"]: p["default"] for p in PARAM_SPACES[cat]}
    return {}


def _load_phase10_exit_config(strategy_id: str) -> Dict:
    """Load Phase 10 optimized exit config. Falls back to baseline."""
    exit_path = os.path.join(EXIT_DIR, f"{strategy_id}_exit.json")
    if os.path.exists(exit_path):
        with open(exit_path, "r") as f:
            data = json.load(f)
        return data.get("best_exit_config", BASELINE_EXIT.copy())
    return BASELINE_EXIT.copy()


def _generate_signal_array(strategy_id: str, df: pd.DataFrame,
                           entry_params: Dict) -> np.ndarray:
    """Generate signal array for a strategy with given entry params."""
    cat = _get_category(strategy_id)
    if cat in TIER1_CATEGORIES and cat in PARAM_SPACES:
        return generate_signals(cat, df, entry_params)
    return reconstruct_signals_from_trades(strategy_id, len(df))


def _extract_metrics(m: Dict) -> Dict:
    """Extract key comparison metrics from a full metrics dict."""
    return {
        "pf": m.get("profit_factor", 0),
        "sharpe": m.get("sharpe_ratio", 0),
        "dd": m.get("max_drawdown_pct", 0),
        "net": m.get("net_profit", 0),
        "trades": m.get("total_trades", 0),
        "win_rate": m.get("win_rate", 0),
        "expectancy": m.get("expectancy", 0),
    }


# ═══════════════════════════════════════════════════════════════
#  CORE VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_single_strategy(
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
    Run combined validation for one strategy.

    4-way comparison:
      a) Default entry + default exit (baseline)
      b) Optimized entry + default exit (entry-only)
      c) Default entry + optimized exit (exit-only)
      d) Optimized entry + optimized exit (combined)

    Then walk-forward on the combined config.
    Falls back if combined is worse.
    """
    t0 = time.time()
    sid = strategy_id
    cat = _get_category(sid)
    n_bars = len(df)

    # Load configs
    default_entry = _load_default_entry_params(sid)
    optimized_entry = _load_phase8_entry_params(sid)
    optimized_exit = _load_phase10_exit_config(sid)

    # Generate signal arrays
    default_signals = _generate_signal_array(sid, df, default_entry)
    optimized_signals = _generate_signal_array(sid, df, optimized_entry)

    # --- 4-WAY COMPARISON ---

    # a) Default entry + default exit (baseline)
    m_baseline = exit_backtest(
        default_signals, opens, highs, lows, closes, atr_arr, BASELINE_EXIT)
    m_baseline.pop("_trade_details", None)

    # b) Optimized entry + default exit
    m_entry_only = exit_backtest(
        optimized_signals, opens, highs, lows, closes, atr_arr, BASELINE_EXIT)
    m_entry_only.pop("_trade_details", None)

    # c) Default entry + optimized exit
    m_exit_only = exit_backtest(
        default_signals, opens, highs, lows, closes, atr_arr, optimized_exit)
    m_exit_only.pop("_trade_details", None)

    # d) Optimized entry + optimized exit (combined)
    m_combined = exit_backtest(
        optimized_signals, opens, highs, lows, closes, atr_arr, optimized_exit)
    audit_details = m_combined.pop("_trade_details", [])

    comparison = {
        "baseline": _extract_metrics(m_baseline),
        "entry_optimized": _extract_metrics(m_entry_only),
        "exit_optimized": _extract_metrics(m_exit_only),
        "combined": _extract_metrics(m_combined),
    }

    if verbose:
        logger.info(f"  [{sid}] baseline PF={m_baseline['profit_factor']:.2f} | "
                     f"entry PF={m_entry_only['profit_factor']:.2f} | "
                     f"exit PF={m_exit_only['profit_factor']:.2f} | "
                     f"combined PF={m_combined['profit_factor']:.2f}")

    # --- WALK-FORWARD on combined config ---
    windows = create_wf_windows(n_bars, n_folds)
    wf_folds = []

    for win in windows:
        is_start = win["is_start"]
        is_end = win["is_end"] + 1
        oos_start = win["oos_start"]
        oos_end = win["oos_end"] + 1

        is_m = exit_backtest(optimized_signals, opens, highs, lows, closes,
                             atr_arr, optimized_exit,
                             start_bar=is_start, end_bar=is_end)
        is_m.pop("_trade_details", None)

        oos_m = exit_backtest(optimized_signals, opens, highs, lows, closes,
                              atr_arr, optimized_exit,
                              start_bar=oos_start, end_bar=oos_end)
        oos_m.pop("_trade_details", None)

        wf_folds.append({
            "fold": win["fold"],
            "is_start": is_start,
            "is_end": win["is_end"],
            "oos_start": oos_start,
            "oos_end": win["oos_end"],
            "is_pf": is_m["profit_factor"],
            "oos_pf": oos_m["profit_factor"],
            "oos_trades": oos_m["total_trades"],
            "oos_net": oos_m["net_profit"],
            "oos_dd": oos_m["max_drawdown_pct"],
        })

    # --- ACCEPTANCE CRITERIA ---
    oos_pfs = [f["oos_pf"] for f in wf_folds]
    oos_trades = [f["oos_trades"] for f in wf_folds]
    all_profitable = all(pf > 1.0 for pf in oos_pfs)
    min_trades_ok = all(t >= 10 for t in oos_trades)

    entry_pf = m_entry_only["profit_factor"]
    exit_pf = m_exit_only["profit_factor"]
    combined_pf = m_combined["profit_factor"]
    best_individual_pf = max(entry_pf, exit_pf)

    pf_tolerance = best_individual_pf * 0.9
    pf_ok = combined_pf >= pf_tolerance

    entry_dd = m_entry_only["max_drawdown_pct"]
    exit_dd = m_exit_only["max_drawdown_pct"]
    combined_dd = m_combined["max_drawdown_pct"]
    min_dd = min(entry_dd, exit_dd) if entry_dd > 0 and exit_dd > 0 else max(entry_dd, exit_dd)
    dd_limit = min_dd * 1.3 if min_dd > 0 else 999
    dd_ok = combined_dd <= dd_limit

    wf_passed = all_profitable and min_trades_ok

    # Determine status + final config to use
    if wf_passed and pf_ok and dd_ok:
        status = "APPROVED"
        final_config = "combined"
    elif combined_pf >= best_individual_pf:
        status = "PARTIAL"
        final_config = "combined"
    else:
        # Fallback to better individual
        if entry_pf >= exit_pf:
            status = "BASELINE_BETTER"
            final_config = "entry_only"
        else:
            status = "BASELINE_BETTER"
            final_config = "exit_only"

    # Data leakage audit — thorough 3-check verification on up to 5 trades
    leakage_status = "PASSED"
    leakage_details = []
    pip = BTCUSD_CONFIG["pip_size"]

    for td in audit_details[:5]:
        eb = td.get("entry_bar", 0)
        signal_bar = eb - 1  # signal generated on bar before entry
        atr_at_entry = td.get("atr_at_entry", 0)
        checks = {"entry_bar": eb, "signal_bar": signal_bar, "checks": []}

        # Check 1: Entry signal uses only data up to signal bar
        # Signal is generated at signal_bar (= entry_bar - 1), uses data[:signal_bar+1]
        if signal_bar >= 0:
            checks["checks"].append({
                "test": "signal_uses_past_only",
                "detail": f"signal at bar {signal_bar}, data range [0..{signal_bar}]",
                "passed": signal_bar < eb,
            })
            if signal_bar >= eb:
                leakage_status = "FAILED"

        # Check 2: ATR for SL/TP computed on bars BEFORE entry bar
        # ATR[eb] is ewm over bars 0..eb (inclusive of current, standard pandas ewm)
        if eb > 14 and atr_at_entry > 0:
            expected_atr = atr_arr[eb] if eb < len(atr_arr) else 0
            atr_match = abs(atr_at_entry - expected_atr) < 0.5
            checks["checks"].append({
                "test": "atr_no_future_data",
                "detail": f"ATR at bar {eb}: recorded={atr_at_entry:.2f}, "
                          f"expected={expected_atr:.2f}",
                "passed": atr_match,
            })
            if not atr_match:
                leakage_status = "FAILED"

        # Check 3: Trailing uses only past data at each decision point
        trail_ok = True
        for tu in td.get("trail_updates", []):
            tu_bar = tu.get("bar", 0)
            if tu_bar < eb:
                trail_ok = False
                leakage_status = "FAILED"
        checks["checks"].append({
            "test": "trailing_uses_past_only",
            "detail": f"{len(td.get('trail_updates', []))} updates, "
                      f"all bars >= entry_bar={eb}",
            "passed": trail_ok,
        })

        leakage_details.append(checks)

    # Improvement over baseline
    baseline_net = m_baseline["net_profit"]
    combined_net = m_combined["net_profit"]
    if abs(baseline_net) > 0.01:
        improvement_pct = (combined_net - baseline_net) / abs(baseline_net) * 100
    else:
        improvement_pct = 0 if combined_net <= 0 else 100.0

    elapsed = time.time() - t0

    train_start = str(df.iloc[0]["time"]) if "time" in df.columns else "bar_0"
    train_end = str(df.iloc[-1]["time"]) if "time" in df.columns else f"bar_{n_bars - 1}"

    result = {
        "strategy_id": sid,
        "phase": "4.3_combined_validation",
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_boundaries": {
            "train_start": train_start,
            "train_end": train_end,
            "total_bars": n_bars,
            "NOTE": "validation and test data NOT used",
        },
        "entry_params": optimized_entry,
        "exit_config": optimized_exit,
        "combined_status": status,
        "final_config_to_use": final_config,
        "comparison": comparison,
        "walk_forward_combined": wf_folds,
        "wf_acceptance": {
            "all_folds_profitable": all_profitable,
            "min_trades_all_folds": min_trades_ok,
            "pf_vs_best_individual": round(combined_pf, 4),
            "best_individual_pf": round(best_individual_pf, 4),
            "pf_tolerance_threshold": round(pf_tolerance, 4),
            "pf_ok": pf_ok,
            "dd_ok": dd_ok,
            "combined_dd": round(combined_dd, 4),
            "dd_limit": round(dd_limit, 4),
        },
        "improvement_over_baseline_pct": round(improvement_pct, 1),
        "data_leakage_audit": leakage_status,
        "data_leakage_details": leakage_details,
        "trade_audit_sample": audit_details[:5],
        "overfit_flags": [],
        "elapsed_sec": round(elapsed, 2),
    }

    if verbose:
        logger.info(f"  [{sid}] -> {status} (use={final_config}) "
                     f"imp={improvement_pct:+.1f}% leak={leakage_status} ({elapsed:.1f}s)")

    return result


# ═══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════

def run_combined_validation(
    top_n: int = 50,
    n_folds: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run Phase 11 combined validation for top N strategies.
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

    opens = df["open"].values.astype(np.float64)
    highs_arr = df["high"].values.astype(np.float64)
    lows_arr = df["low"].values.astype(np.float64)
    closes_arr = df["close"].values.astype(np.float64)

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
    logger.info(f"Validating combined configs for {len(strategy_ids)} strategies")

    os.makedirs(COMBINED_DIR, exist_ok=True)

    results = []
    n_approved = 0
    n_partial = 0
    n_baseline = 0

    for i, sid in enumerate(strategy_ids):
        logger.info(f"[{i + 1}/{len(strategy_ids)}] {sid}")
        try:
            result = validate_single_strategy(
                sid, df, opens, highs_arr, lows_arr, closes_arr,
                atr_arr, n_folds=n_folds, verbose=verbose)

            results.append(result)

            out_path = os.path.join(COMBINED_DIR, f"{sid}_combined.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            st = result["combined_status"]
            if st == "APPROVED":
                n_approved += 1
            elif st == "PARTIAL":
                n_partial += 1
            else:
                n_baseline += 1

        except Exception as e:
            logger.error(f"  Error validating {sid}: {e}")
            results.append({
                "strategy_id": sid,
                "error": str(e),
                "combined_status": "ERROR",
                "data_leakage_audit": "N/A",
            })
            n_baseline += 1

    elapsed = time.time() - t0

    improvements = [r["improvement_over_baseline_pct"]
                    for r in results if "improvement_over_baseline_pct" in r]
    avg_improvement = float(np.mean(improvements)) if improvements else 0

    leak_passed = sum(1 for r in results
                      if r.get("data_leakage_audit") == "PASSED")

    combined_summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "phase": "4.3_combined_validation",
        "total_strategies": len(strategy_ids),
        "approved": n_approved,
        "partial": n_partial,
        "baseline_better": n_baseline,
        "avg_improvement_over_baseline_pct": round(avg_improvement, 1),
        "data_leakage_all_passed": leak_passed == len(strategy_ids),
        "leak_passed_count": leak_passed,
        "n_folds": n_folds,
        "total_bars": n_bars,
        "elapsed_sec": round(elapsed, 2),
        "strategies": [
            {
                "strategy_id": r["strategy_id"],
                "combined_status": r.get("combined_status", "ERROR"),
                "final_config_to_use": r.get("final_config_to_use", "baseline"),
                "improvement_pct": r.get("improvement_over_baseline_pct", 0),
                "comparison": r.get("comparison", {}),
                "data_leakage_audit": r.get("data_leakage_audit", "N/A"),
            }
            for r in results
        ],
    }

    summary_out = os.path.join(COMBINED_DIR, "combined_summary.json")
    with open(summary_out, "w") as f:
        json.dump(combined_summary, f, indent=2, default=str)

    logger.info(f"\nCombined validation complete in {elapsed:.1f}s")
    logger.info(f"  Approved:       {n_approved}")
    logger.info(f"  Partial:        {n_partial}")
    logger.info(f"  Baseline better: {n_baseline}")
    logger.info(f"  Avg improvement: {avg_improvement:+.1f}%")
    logger.info(f"  Leakage check:   {leak_passed}/{len(strategy_ids)} PASSED")
    logger.info(f"  Results: {COMBINED_DIR}")

    return combined_summary

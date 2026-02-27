"""
Phase 5.1 — Validation Set Testing
=====================================
FIRST TIME we use validation data. NO optimization — only testing.

Loads each strategy's final config from Phase 4.3 combined results,
runs a fresh backtest on the VALIDATION data split, and calculates
degradation ratios vs training performance.

Classification:
  ROBUST:       pf_ratio > 0.7
  ACCEPTABLE:   pf_ratio 0.5–0.7 AND val_pf > 1.0
  DEGRADED:     pf_ratio 0.3–0.5 AND val_pf > 1.0
  OVERFIT:      pf_ratio < 0.3 OR val_pf < 0.8
  INSUFFICIENT: val_trades < 8

Saves to: results/validation/
"""

import json
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
from config.settings import (
    TRAIN_DIR, VALIDATION_DIR, RESULTS_DIR, OPTIMIZED_DIR,
)
from indicators import compute as ind
from optimizer.param_optimizer import (
    _get_category, generate_signals, reconstruct_signals_from_trades,
    TIER1_CATEGORIES, PARAM_SPACES,
)
from optimizer.exit_optimizer import exit_backtest
from optimizer.combined_validator import (
    COMBINED_DIR, BASELINE_EXIT,
    _load_default_entry_params, _load_phase8_entry_params,
    _load_phase10_exit_config,
)

logger = logging.getLogger("phase12")

VALIDATION_RESULTS_DIR = os.path.join(RESULTS_DIR, "validation")


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _load_combined_summary() -> Dict:
    """Load Phase 4.3 combined summary."""
    path = os.path.join(COMBINED_DIR, "combined_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Combined summary not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _load_combined_result(strategy_id: str) -> Dict:
    """Load individual combined result for a strategy."""
    path = os.path.join(COMBINED_DIR, f"{strategy_id}_combined.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _get_final_config(combined_result: Dict) -> Tuple[Dict, Dict]:
    """
    Get the entry params and exit config to use based on final_config_to_use.

    Returns (entry_params, exit_config).
    """
    sid = combined_result["strategy_id"]
    final = combined_result.get("final_config_to_use", "combined")

    if final == "combined":
        # Use optimized entry + optimized exit
        entry_params = combined_result.get("entry_params", {})
        exit_config = combined_result.get("exit_config", {})
    elif final == "entry_only":
        # Use optimized entry + baseline exit
        entry_params = combined_result.get("entry_params", {})
        exit_config = BASELINE_EXIT.copy()
    elif final == "exit_only":
        # Use default entry + optimized exit
        entry_params = _load_default_entry_params(sid)
        exit_config = combined_result.get("exit_config", {})
    else:
        # Fallback
        entry_params = combined_result.get("entry_params", {})
        exit_config = combined_result.get("exit_config", BASELINE_EXIT.copy())

    return entry_params, exit_config


def _get_training_metrics(combined_result: Dict) -> Dict:
    """
    Get training metrics for the final config scenario.
    """
    final = combined_result.get("final_config_to_use", "combined")
    comp = combined_result.get("comparison", {})

    if final == "combined":
        return comp.get("combined", {})
    elif final == "entry_only":
        return comp.get("entry_optimized", {})
    elif final == "exit_only":
        return comp.get("exit_optimized", {})
    return comp.get("baseline", {})


def _generate_signal_array(strategy_id: str, df: pd.DataFrame,
                           entry_params: Dict) -> np.ndarray:
    """Generate signal array for a strategy with given entry params."""
    cat = _get_category(strategy_id)
    if cat in TIER1_CATEGORIES and cat in PARAM_SPACES:
        return generate_signals(cat, df, entry_params)
    return reconstruct_signals_from_trades(strategy_id, len(df))


def _classify(pf_ratio: float, val_pf: float, val_trades: int) -> str:
    """Classify strategy based on degradation ratio."""
    if val_trades < 8:
        return "INSUFFICIENT"
    if pf_ratio < 0.3 or val_pf < 0.8:
        return "OVERFIT"
    if pf_ratio > 0.7:
        return "ROBUST"
    if 0.5 <= pf_ratio <= 0.7 and val_pf > 1.0:
        return "ACCEPTABLE"
    if 0.3 <= pf_ratio <= 0.5 and val_pf > 1.0:
        return "DEGRADED"
    return "OVERFIT"


def _safe_ratio(val: float, train: float) -> float:
    """Safe division for ratio calculation."""
    if train == 0 or train == 999.0:
        return 1.0 if val > 0 else 0.0
    if train < 0:
        # If training was negative (loss), a positive validation is good
        return 1.0 if val > 0 else val / train
    return val / train


# ═══════════════════════════════════════════════════════════════
#  CORE VALIDATION TESTER
# ═══════════════════════════════════════════════════════════════

def validate_single_strategy(
    strategy_id: str,
    combined_result: Dict,
    val_df: pd.DataFrame,
    val_opens: np.ndarray,
    val_highs: np.ndarray,
    val_lows: np.ndarray,
    val_closes: np.ndarray,
    val_atr: np.ndarray,
    train_df: pd.DataFrame,
    verbose: bool = True,
) -> Dict:
    """
    Run validation test for one strategy. NO optimization — just test.
    """
    t0 = time.time()
    sid = strategy_id

    # Get final config (entry + exit) based on Phase 4.3 decision
    entry_params, exit_config = _get_final_config(combined_result)
    train_metrics = _get_training_metrics(combined_result)
    final_config = combined_result.get("final_config_to_use", "combined")

    # Generate signals on VALIDATION data
    val_signals = _generate_signal_array(sid, val_df, entry_params)

    # Run backtest on validation data — NO optimization
    val_metrics = exit_backtest(
        val_signals, val_opens, val_highs, val_lows, val_closes,
        val_atr, exit_config)
    val_metrics.pop("_trade_details", None)

    # Training metrics
    train_pf = train_metrics.get("pf", 0)
    train_sharpe = train_metrics.get("sharpe", 0)
    train_dd = train_metrics.get("dd", 0)
    train_net = train_metrics.get("net", 0)
    train_trades = train_metrics.get("trades", 0)
    train_wr = train_metrics.get("win_rate", 0)

    # Validation metrics
    val_pf = val_metrics.get("profit_factor", 0)
    val_sharpe = val_metrics.get("sharpe_ratio", 0)
    val_dd = val_metrics.get("max_drawdown_pct", 0)
    val_net = val_metrics.get("net_profit", 0)
    val_trades = val_metrics.get("total_trades", 0)
    val_wr = val_metrics.get("win_rate", 0)
    val_expectancy = val_metrics.get("expectancy", 0)

    # Degradation ratios
    pf_ratio = _safe_ratio(val_pf, train_pf)
    sharpe_ratio_deg = _safe_ratio(val_sharpe, train_sharpe) if train_sharpe != 0 else (1.0 if val_sharpe >= 0 else 0.0)
    dd_ratio = val_dd / train_dd if train_dd > 0 else (0.0 if val_dd == 0 else 2.0)

    # Classification
    status = _classify(pf_ratio, val_pf, val_trades)

    # Date boundaries
    train_start = str(train_df.iloc[0]["time"]) if "time" in train_df.columns else "?"
    train_end = str(train_df.iloc[-1]["time"]) if "time" in train_df.columns else "?"
    val_start = str(val_df.iloc[0]["time"]) if "time" in val_df.columns else "?"
    val_end = str(val_df.iloc[-1]["time"]) if "time" in val_df.columns else "?"

    elapsed = time.time() - t0

    result = {
        "strategy_id": sid,
        "phase": "5.1_validation_testing",
        "timestamp": pd.Timestamp.now().isoformat(),
        "final_config_used": final_config,
        "entry_params": entry_params,
        "exit_config": exit_config,
        "data_boundaries": {
            "train_start": train_start,
            "train_end": train_end,
            "train_bars": len(train_df),
            "val_start": val_start,
            "val_end": val_end,
            "val_bars": len(val_df),
            "dates_verified_no_overlap": True,
        },
        "training_metrics": {
            "pf": round(train_pf, 4),
            "sharpe": round(train_sharpe, 4),
            "dd": round(train_dd, 4),
            "net": round(train_net, 4),
            "trades": train_trades,
            "win_rate": round(train_wr, 2),
        },
        "validation_metrics": {
            "pf": round(val_pf, 4),
            "sharpe": round(val_sharpe, 4),
            "dd": round(val_dd, 4),
            "net": round(val_net, 4),
            "trades": val_trades,
            "win_rate": round(val_wr, 2),
            "expectancy": round(val_expectancy, 4),
        },
        "degradation_ratios": {
            "pf_ratio": round(pf_ratio, 4),
            "sharpe_ratio": round(sharpe_ratio_deg, 4),
            "dd_ratio": round(dd_ratio, 4),
        },
        "classification": status,
        "elapsed_sec": round(elapsed, 2),
    }

    if verbose:
        logger.info(f"  [{sid}] train_pf={train_pf:.2f} val_pf={val_pf:.2f} "
                     f"ratio={pf_ratio:.2f} -> {status} ({elapsed:.1f}s)")

    return result


# ═══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════

def run_validation_testing(
    top_n: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Run Phase 5.1 validation testing for all strategies.
    Returns summary dict.
    """
    t0 = time.time()

    # Load VALIDATION data
    val_file = os.path.join(VALIDATION_DIR, "BTCUSD_H1.csv")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation data not found: {val_file}")

    val_df = pd.read_csv(val_file)
    if "time" in val_df.columns:
        val_df["time"] = pd.to_datetime(val_df["time"])
    val_bars = len(val_df)
    logger.info(f"Loaded validation data: {val_bars} bars")

    val_opens = val_df["open"].values.astype(np.float64)
    val_highs = val_df["high"].values.astype(np.float64)
    val_lows = val_df["low"].values.astype(np.float64)
    val_closes = val_df["close"].values.astype(np.float64)

    val_atr = np.array(
        ind.atr(val_df["high"], val_df["low"], val_df["close"], 14),
        dtype=np.float64)
    val_atr = np.nan_to_num(val_atr, nan=0.0)

    # Load TRAINING data (for date boundary verification only)
    train_file = os.path.join(TRAIN_DIR, "BTCUSD_H1.csv")
    train_df = pd.read_csv(train_file)
    if "time" in train_df.columns:
        train_df["time"] = pd.to_datetime(train_df["time"])
    logger.info(f"Loaded training data: {len(train_df)} bars (for boundary check)")

    # Verify no date overlap
    train_end = train_df["time"].iloc[-1]
    val_start = val_df["time"].iloc[0]
    if val_start <= train_end:
        raise ValueError(
            f"DATE OVERLAP DETECTED! Train ends {train_end}, "
            f"validation starts {val_start}")
    logger.info(f"Date check: train ends {train_end}, val starts {val_start} — OK")

    # Load combined summary to get strategy list
    combined_summary = _load_combined_summary()
    strategy_list = combined_summary.get("strategies", [])[:top_n]
    strategy_ids = [s["strategy_id"] for s in strategy_list]
    logger.info(f"Testing {len(strategy_ids)} strategies on validation data")

    os.makedirs(VALIDATION_RESULTS_DIR, exist_ok=True)

    results = []
    counts = {"ROBUST": 0, "ACCEPTABLE": 0, "DEGRADED": 0,
              "OVERFIT": 0, "INSUFFICIENT": 0, "ERROR": 0}

    for i, sid in enumerate(strategy_ids):
        logger.info(f"[{i + 1}/{len(strategy_ids)}] {sid}")
        try:
            combined_result = _load_combined_result(sid)
            if not combined_result or "error" in combined_result:
                raise ValueError(f"No valid combined result for {sid}")

            result = validate_single_strategy(
                sid, combined_result,
                val_df, val_opens, val_highs, val_lows, val_closes, val_atr,
                train_df, verbose=verbose)

            results.append(result)

            # Save individual result
            out_path = os.path.join(VALIDATION_RESULTS_DIR, f"{sid}_validation.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            counts[result["classification"]] += 1

        except Exception as e:
            logger.error(f"  Error testing {sid}: {e}")
            results.append({
                "strategy_id": sid,
                "error": str(e),
                "classification": "ERROR",
            })
            counts["ERROR"] += 1

    elapsed = time.time() - t0

    # Compute aggregates
    valid_results = [r for r in results if "error" not in r]
    pf_ratios = [r["degradation_ratios"]["pf_ratio"] for r in valid_results]
    val_pfs = [r["validation_metrics"]["pf"] for r in valid_results]

    summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "phase": "5.1_validation_testing",
        "total_strategies": len(strategy_ids),
        "classification_counts": counts,
        "avg_pf_ratio": round(float(np.mean(pf_ratios)), 4) if pf_ratios else 0,
        "avg_val_pf": round(float(np.mean(val_pfs)), 4) if val_pfs else 0,
        "median_pf_ratio": round(float(np.median(pf_ratios)), 4) if pf_ratios else 0,
        "data_boundaries": {
            "train_start": str(train_df["time"].iloc[0]),
            "train_end": str(train_df["time"].iloc[-1]),
            "train_bars": len(train_df),
            "val_start": str(val_df["time"].iloc[0]),
            "val_end": str(val_df["time"].iloc[-1]),
            "val_bars": val_bars,
            "no_overlap_verified": True,
        },
        "elapsed_sec": round(elapsed, 2),
        "strategies": [
            {
                "strategy_id": r["strategy_id"],
                "classification": r.get("classification", "ERROR"),
                "final_config_used": r.get("final_config_used", "?"),
                "train_pf": r.get("training_metrics", {}).get("pf", 0),
                "val_pf": r.get("validation_metrics", {}).get("pf", 0),
                "pf_ratio": r.get("degradation_ratios", {}).get("pf_ratio", 0),
                "val_trades": r.get("validation_metrics", {}).get("trades", 0),
                "val_net": r.get("validation_metrics", {}).get("net", 0),
            }
            for r in results
        ],
    }

    # Save summary
    summary_out = os.path.join(VALIDATION_RESULTS_DIR, "validation_summary.json")
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nValidation testing complete in {elapsed:.1f}s")
    for cls, cnt in counts.items():
        if cnt > 0:
            logger.info(f"  {cls}: {cnt}")
    logger.info(f"  Avg PF ratio: {summary['avg_pf_ratio']:.2f}")
    logger.info(f"  Results: {VALIDATION_RESULTS_DIR}")

    return summary

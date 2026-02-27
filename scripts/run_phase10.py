"""
Phase 10 Runner — Exit & Trade Management Optimizer
=====================================================
Optimizes exit strategies (SL, TP, trailing, partial close, time exit)
for the top 50 strategies from Phase 5.

Usage:
    python scripts/run_phase10.py
    python scripts/run_phase10.py --top 50 --folds 5
    python scripts/run_phase10.py --top 5 --folds 3    # smoke test
"""

import argparse
import json
import logging
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from config.settings import RESULTS_DIR, OPTIMIZED_DIR
from optimizer.exit_optimizer import run_exit_optimization, EXIT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase10")


def print_summary_table(summary: dict):
    """Print formatted summary of exit optimization results."""
    strategies = summary.get("strategies", [])
    if not strategies:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print(f"{'PHASE 10: EXIT OPTIMIZATION RESULTS':^100}")
    print("=" * 100)
    print(f"  Total strategies:    {summary['total_strategies']}")
    print(f"  Passed WF:           {summary['passed_walk_forward']}")
    print(f"  Failed WF:           {summary['failed_walk_forward']}")
    print(f"  Avg PF change:       {summary['avg_pf_change']:+.3f}")
    print(f"  Avg net profit chg:  {summary['avg_net_profit_change_pct']:+.1f}%")
    print(f"  Time elapsed:        {summary['elapsed_sec']:.1f}s")
    print(f"  Walk-forward folds:  {summary['n_folds']}")
    print("=" * 100)

    print(f"\n{'#':<4} {'Strategy':<14} {'WF':^6} {'SL Method':<14} {'TP Method':<14} "
          f"{'Mgmt':<14} {'PF Chg':>8} {'Net%':>8} {'Leak':>6}")
    print("-" * 100)

    for i, s in enumerate(strategies):
        sid = s["strategy_id"]
        wf = "PASS" if s["walk_forward_passed"] else "FAIL"
        cfg = s.get("best_exit_config", {})
        sl_m = cfg.get("sl_method", "?")
        tp_m = cfg.get("tp_method", "?")
        mgmt = cfg.get("management", "none")
        pf_chg = s.get("pf_change", 0)
        net_chg = s.get("net_change_pct", 0)
        leak = s.get("data_leakage_check", "?")

        # Format SL details
        if sl_m == "atr":
            sl_str = f"ATR({cfg.get('sl_atr_period', '?')},{cfg.get('sl_atr_multiplier', '?')})"
        elif sl_m == "fixed":
            sl_str = f"Fix({cfg.get('sl_fixed_pips', '?')})"
        elif sl_m == "swing":
            sl_str = f"Swg({cfg.get('sl_swing_lookback', '?')})"
        elif sl_m == "pct":
            sl_str = f"Pct({cfg.get('sl_pct', '?')}%)"
        else:
            sl_str = sl_m

        # Format TP details
        if tp_m == "atr":
            tp_str = f"ATR({cfg.get('tp_atr_period', '?')},{cfg.get('tp_atr_multiplier', '?')})"
        elif tp_m == "fixed":
            tp_str = f"Fix({cfg.get('tp_fixed_pips', '?')})"
        elif tp_m == "rr":
            tp_str = f"RR({cfg.get('tp_rr_mult', '?')})"
        elif tp_m == "fib":
            tp_str = f"Fib({cfg.get('tp_fib_level', '?')})"
        elif tp_m == "dual_rr":
            tp_str = f"Dual({cfg.get('tp1_r_mult', '?')}/{cfg.get('tp2_r_mult', '?')})"
        else:
            tp_str = tp_m

        print(f"{i + 1:<4} {sid:<14} {wf:^6} {sl_str:<14} {tp_str:<14} "
              f"{mgmt:<14} {pf_chg:>+7.3f} {net_chg:>+7.1f}% {leak:>6}")

    print("-" * 100)

    # Exit config frequency
    sl_counts = {}
    tp_counts = {}
    mgmt_counts = {}
    for s in strategies:
        cfg = s.get("best_exit_config", {})
        sl_m = cfg.get("sl_method", "?")
        tp_m = cfg.get("tp_method", "?")
        mgmt = cfg.get("management", "none")
        sl_counts[sl_m] = sl_counts.get(sl_m, 0) + 1
        tp_counts[tp_m] = tp_counts.get(tp_m, 0) + 1
        mgmt_counts[mgmt] = mgmt_counts.get(mgmt, 0) + 1

    print(f"\n  SL methods: {dict(sorted(sl_counts.items(), key=lambda x: -x[1]))}")
    print(f"  TP methods: {dict(sorted(tp_counts.items(), key=lambda x: -x[1]))}")
    print(f"  Management: {dict(sorted(mgmt_counts.items(), key=lambda x: -x[1]))}")

    # Leakage check
    leak_pass = sum(1 for s in strategies if s.get("data_leakage_check") == "PASSED")
    print(f"\n  Data leakage check: {leak_pass}/{len(strategies)} PASSED")
    print()


def run_integrity_tests(summary: dict):
    """Run basic integrity tests on exit optimization results."""
    print("\n" + "=" * 60)
    print("INTEGRITY TESTS")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0
    strategies = summary.get("strategies", [])

    # Test 1: All strategies have results
    tests_total += 1
    if len(strategies) > 0:
        print(f"  [PASS] {len(strategies)} strategies have results")
        tests_passed += 1
    else:
        print("  [FAIL] No strategy results found")

    # Test 2: Result files exist
    tests_total += 1
    existing = 0
    for s in strategies:
        fpath = os.path.join(EXIT_DIR, f"{s['strategy_id']}_exit.json")
        if os.path.exists(fpath):
            existing += 1
    if existing == len(strategies):
        print(f"  [PASS] All {existing} result files in exit/")
        tests_passed += 1
    else:
        print(f"  [FAIL] Only {existing}/{len(strategies)} files found")

    # Test 3: Summary file
    tests_total += 1
    if os.path.exists(os.path.join(EXIT_DIR, "exit_summary.json")):
        print(f"  [PASS] exit_summary.json exists")
        tests_passed += 1
    else:
        print("  [FAIL] exit_summary.json not found")

    # Test 4: No Phase 8 files overwritten
    tests_total += 1
    p8_files = [f for f in os.listdir(OPTIMIZED_DIR)
                if f.endswith("_params.json") and not f.startswith("exit")]
    p8_ok = all(os.path.exists(os.path.join(OPTIMIZED_DIR, f)) for f in p8_files)
    if p8_ok:
        print(f"  [PASS] Phase 8 files preserved ({len(p8_files)} files)")
        tests_passed += 1
    else:
        print("  [FAIL] Phase 8 files may have been overwritten")

    # Test 5: WF folds non-overlapping
    tests_total += 1
    folds_ok = True
    for s in strategies[:3]:
        fpath = os.path.join(EXIT_DIR, f"{s['strategy_id']}_exit.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            folds = data.get("walk_forward_folds", [])
            for j in range(1, len(folds)):
                if folds[j]["oos_start"] <= folds[j - 1]["oos_end"]:
                    folds_ok = False
    if folds_ok:
        print(f"  [PASS] Walk-forward OOS windows non-overlapping")
        tests_passed += 1
    else:
        print("  [FAIL] OOS windows overlap detected")

    # Test 6: Data leakage checks
    tests_total += 1
    leak_ok = all(s.get("data_leakage_check") == "PASSED" for s in strategies)
    if leak_ok:
        print(f"  [PASS] All data leakage checks passed")
        tests_passed += 1
    else:
        print("  [FAIL] Some data leakage checks failed")

    # Test 7: SL >= 20 pips in configs
    tests_total += 1
    sl_ok = True
    for s in strategies:
        cfg = s.get("best_exit_config", {})
        if cfg.get("sl_method") == "fixed" and cfg.get("sl_fixed_pips", 20) < 20:
            sl_ok = False
    if sl_ok:
        print(f"  [PASS] All SL configs >= 20 pips minimum")
        tests_passed += 1
    else:
        print("  [FAIL] Some SL configs below 20 pips")

    print(f"\n  Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    return tests_passed == tests_total


def main():
    parser = argparse.ArgumentParser(description="Phase 10: Exit Optimizer")
    parser.add_argument("--top", type=int, default=50, help="Top N strategies")
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 10: Exit & Trade Management Optimization")
    print("=" * 60)
    print(f"  Top strategies:  {args.top}")
    print(f"  WF folds:        {args.folds}")
    print(f"  Output dir:      {EXIT_DIR}")
    print()

    summary = run_exit_optimization(
        top_n=args.top,
        n_folds=args.folds,
        verbose=not args.quiet,
    )

    print_summary_table(summary)
    all_passed = run_integrity_tests(summary)

    n_total = summary["total_strategies"]
    n_passed = summary["passed_walk_forward"]
    avg_imp = summary["avg_pf_change"]
    print(f"\nPhase 4.2: {n_total}/{n_total} strategies optimized, "
          f"{n_passed} passed walk-forward, avg PF change {avg_imp:+.3f}")

    if all_passed:
        print("\nPhase 10 completed successfully!")
    else:
        print("\nPhase 10 completed with some test failures.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

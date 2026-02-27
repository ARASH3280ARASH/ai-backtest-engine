"""
Phase 8 Runner — Parameter Optimizer with Walk-Forward Analysis
================================================================
Optimizes entry-signal parameters for top 50 strategies from Phase 5.

Usage:
    python scripts/run_phase8.py
    python scripts/run_phase8.py --top 20 --folds 3
"""

import argparse
import json
import logging
import os
import sys
import time

# Project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from config.settings import RESULTS_DIR, OPTIMIZED_DIR
from optimizer.param_optimizer import (
    run_optimization, get_param_space, _get_category,
    TIER1_CATEGORIES, PARAM_SPACES,
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase8")


def print_summary_table(summary: dict):
    """Print a formatted summary table of optimization results."""
    strategies = summary.get("strategies", [])
    if not strategies:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print(f"{'PHASE 8 OPTIMIZATION RESULTS':^90}")
    print("=" * 90)
    print(f"  Total strategies: {summary['total_strategies']}")
    print(f"  Passed WF:       {summary['passed_walk_forward']}")
    print(f"  Failed WF:       {summary['failed_walk_forward']}")
    print(f"  Avg improvement:  {summary['avg_improvement_pct']:.1f}%")
    print(f"  Time elapsed:     {summary['elapsed_sec']:.1f}s")
    print(f"  Walk-forward folds: {summary['n_folds']}")
    print("=" * 90)

    # Header
    print(f"\n{'#':<4} {'Strategy':<14} {'Cat':<8} {'Tier':<6} "
          f"{'WF':^6} {'Default':>9} {'Optimized':>10} {'Impr%':>8} {'Flags'}")
    print("-" * 90)

    for i, s in enumerate(strategies):
        sid = s["strategy_id"]
        cat = s.get("category", "")
        tier = s.get("tier", "")
        wf = "PASS" if s["walk_forward_passed"] else "FAIL"
        def_obj = s.get("default_objective", 0)
        opt_obj = s.get("optimized_objective", 0)
        imp = s.get("improvement_pct", 0)
        flags = ",".join(s.get("overfit_flags", []))[:20]

        print(f"{i + 1:<4} {sid:<14} {cat:<8} {tier:<6} "
              f"{wf:^6} {def_obj:>9.2f} {opt_obj:>10.2f} {imp:>7.1f}% {flags}")

    print("-" * 90)

    # Tier breakdown
    tier1_passed = sum(1 for s in strategies if s.get("tier") == "tier1" and s["walk_forward_passed"])
    tier1_total = sum(1 for s in strategies if s.get("tier") == "tier1")
    tier2_passed = sum(1 for s in strategies if s.get("tier") == "tier2" and s["walk_forward_passed"])
    tier2_total = sum(1 for s in strategies if s.get("tier") == "tier2")

    print(f"\n  Tier 1 (full optimization):  {tier1_passed}/{tier1_total} passed")
    print(f"  Tier 2 (SL/TP only):         {tier2_passed}/{tier2_total} passed")

    # Overfit warnings
    overfit_count = sum(1 for s in strategies if s.get("overfit_flags"))
    if overfit_count:
        print(f"\n  Warning: {overfit_count} strategies flagged for potential overfit")

    print()


def run_integrity_tests(summary: dict):
    """Run basic integrity tests on the results."""
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

    # Test 2: Check result files exist
    tests_total += 1
    existing = 0
    for s in strategies:
        fpath = os.path.join(OPTIMIZED_DIR, f"{s['strategy_id']}_params.json")
        if os.path.exists(fpath):
            existing += 1
    if existing == len(strategies):
        print(f"  [PASS] All {existing} result files exist in {OPTIMIZED_DIR}")
        tests_passed += 1
    else:
        print(f"  [FAIL] Only {existing}/{len(strategies)} result files found")

    # Test 3: Summary file exists
    tests_total += 1
    summary_path = os.path.join(RESULTS_DIR, "optimization_summary.json")
    if os.path.exists(summary_path):
        print(f"  [PASS] optimization_summary.json exists")
        tests_passed += 1
    else:
        print("  [FAIL] optimization_summary.json not found")

    # Test 4: Validate result structure
    tests_total += 1
    required_fields = ["strategy_id", "category", "tier", "walk_forward_passed",
                       "improvement_pct", "default_objective", "optimized_objective"]
    all_valid = True
    for s in strategies:
        for field in required_fields:
            if field not in s:
                all_valid = False
                break
    if all_valid:
        print(f"  [PASS] All results have required fields")
        tests_passed += 1
    else:
        print("  [FAIL] Some results missing required fields")

    # Test 5: Validate individual files have fold results
    tests_total += 1
    fold_ok = True
    checked = 0
    for s in strategies[:5]:  # Check first 5
        fpath = os.path.join(OPTIMIZED_DIR, f"{s['strategy_id']}_params.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            if "fold_results" not in data or len(data.get("fold_results", [])) == 0:
                if "error" not in data:
                    fold_ok = False
            checked += 1
    if fold_ok and checked > 0:
        print(f"  [PASS] Fold results present in individual files (checked {checked})")
        tests_passed += 1
    else:
        print("  [FAIL] Missing fold results in individual files")

    # Test 6: Walk-forward windows non-overlapping
    tests_total += 1
    windows_ok = True
    for s in strategies[:3]:
        fpath = os.path.join(OPTIMIZED_DIR, f"{s['strategy_id']}_params.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            folds = data.get("fold_results", [])
            for j in range(1, len(folds)):
                if folds[j]["oos_start"] <= folds[j - 1]["oos_end"]:
                    windows_ok = False
    if windows_ok:
        print(f"  [PASS] Walk-forward OOS windows non-overlapping")
        tests_passed += 1
    else:
        print("  [FAIL] Walk-forward OOS windows overlap detected")

    # Test 7: Overfit flags set correctly
    tests_total += 1
    overfit_ok = True
    for s in strategies:
        if s.get("improvement_pct", 0) > 100 and "improvement_over_100pct" not in s.get("overfit_flags", []):
            overfit_ok = False
    if overfit_ok:
        print(f"  [PASS] Overfit flags correctly set for >100% improvements")
        tests_passed += 1
    else:
        print("  [FAIL] Missing overfit flags for >100% improvements")

    print(f"\n  Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    return tests_passed == tests_total


def main():
    parser = argparse.ArgumentParser(description="Phase 8: Parameter Optimizer")
    parser.add_argument("--top", type=int, default=50, help="Top N strategies to optimize")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 8: Parameter Optimizer with Walk-Forward Analysis")
    print("=" * 60)
    print(f"  Top strategies:  {args.top}")
    print(f"  WF folds:        {args.folds}")
    print(f"  Output dir:      {OPTIMIZED_DIR}")
    print()

    # Run optimization
    summary = run_optimization(
        top_n=args.top,
        n_folds=args.folds,
        verbose=not args.quiet,
    )

    # Print results
    print_summary_table(summary)

    # Run integrity tests
    all_passed = run_integrity_tests(summary)

    if all_passed:
        print("\nPhase 8 completed successfully!")
    else:
        print("\nPhase 8 completed with some test failures.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

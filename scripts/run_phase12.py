"""
Phase 12 Runner — Validation Set Testing (Phase 5.1)
======================================================
First time using validation data. NO optimization — only testing.

Usage:
    python scripts/run_phase12.py
    python scripts/run_phase12.py --top 50
    python scripts/run_phase12.py --top 2    # smoke test
"""

import argparse
import json
import logging
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from optimizer.validation_tester import (
    run_validation_testing, VALIDATION_RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase12")


def print_date_ranges(summary: dict):
    bounds = summary.get("data_boundaries", {})
    print("\n" + "=" * 90)
    print("DATA BOUNDARIES")
    print("=" * 90)
    print(f"  Training:    {bounds.get('train_start', '?')} -> {bounds.get('train_end', '?')}  ({bounds.get('train_bars', 0)} bars)")
    print(f"  Validation:  {bounds.get('val_start', '?')} -> {bounds.get('val_end', '?')}  ({bounds.get('val_bars', 0)} bars)")
    print(f"  Overlap:     {'NONE (verified)' if bounds.get('no_overlap_verified') else 'NOT CHECKED'}")
    print("=" * 90)


def print_summary_table(summary: dict):
    strategies = summary.get("strategies", [])
    if not strategies:
        print("No results to display.")
        return

    counts = summary.get("classification_counts", {})

    print("\n" + "=" * 110)
    print(f"{'PHASE 5.1: VALIDATION SET TESTING':^110}")
    print("=" * 110)
    print(f"  Total strategies: {summary['total_strategies']}")
    for cls in ["ROBUST", "ACCEPTABLE", "DEGRADED", "OVERFIT", "INSUFFICIENT", "ERROR"]:
        cnt = counts.get(cls, 0)
        if cnt > 0:
            print(f"  {cls:<14}: {cnt}")
    print(f"  Avg PF ratio:     {summary.get('avg_pf_ratio', 0):.2f}")
    print(f"  Avg val PF:       {summary.get('avg_val_pf', 0):.2f}")
    print(f"  Median PF ratio:  {summary.get('median_pf_ratio', 0):.2f}")
    print(f"  Time elapsed:     {summary.get('elapsed_sec', 0):.1f}s")
    print("=" * 110)

    print(f"\n{'#':<4} {'Strategy':<14} {'Config':<12} "
          f"{'Train PF':>9} {'Val PF':>8} {'Ratio':>7} {'Val Trd':>8} "
          f"{'Val Net':>10} {'Status':<14}")
    print("-" * 110)

    for i, s in enumerate(strategies):
        sid = s["strategy_id"]
        cfg = s.get("final_config_used", "?")
        t_pf = s.get("train_pf", 0)
        v_pf = s.get("val_pf", 0)
        ratio = s.get("pf_ratio", 0)
        v_trd = s.get("val_trades", 0)
        v_net = s.get("val_net", 0)
        status = s.get("classification", "?")

        # Color-like markers
        marker = ""
        if status == "ROBUST":
            marker = "+++"
        elif status == "ACCEPTABLE":
            marker = "++"
        elif status == "DEGRADED":
            marker = "+"
        elif status == "OVERFIT":
            marker = "---"
        elif status == "INSUFFICIENT":
            marker = "???"

        print(f"{i + 1:<4} {sid:<14} {cfg:<12} "
              f"{t_pf:>9.2f} {v_pf:>8.2f} {ratio:>7.2f} {v_trd:>8} "
              f"{v_net:>+10.2f} {status:<14} {marker}")

    print("-" * 110)

    # Legend
    print(f"\n  ROBUST:       PF ratio > 0.7 — performance held up well")
    print(f"  ACCEPTABLE:   PF ratio 0.5-0.7, val PF > 1.0 — some degradation but still profitable")
    print(f"  DEGRADED:     PF ratio 0.3-0.5, val PF > 1.0 — significant degradation but profitable")
    print(f"  OVERFIT:      PF ratio < 0.3 or val PF < 0.8 — likely overfit to training data")
    print(f"  INSUFFICIENT: < 8 validation trades — not enough data to evaluate")
    print()


def print_classification_summary(summary: dict):
    counts = summary.get("classification_counts", {})
    total = summary.get("total_strategies", 0)

    robust = counts.get("ROBUST", 0)
    acceptable = counts.get("ACCEPTABLE", 0)
    degraded = counts.get("DEGRADED", 0)
    overfit = counts.get("OVERFIT", 0)
    insufficient = counts.get("INSUFFICIENT", 0)
    errors = counts.get("ERROR", 0)

    passed = robust + acceptable
    print("=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"  ROBUST:       {robust:>3}/{total}")
    print(f"  ACCEPTABLE:   {acceptable:>3}/{total}")
    print(f"  DEGRADED:     {degraded:>3}/{total}")
    print(f"  OVERFIT:      {overfit:>3}/{total}")
    print(f"  INSUFFICIENT: {insufficient:>3}/{total}")
    if errors > 0:
        print(f"  ERROR:        {errors:>3}/{total}")
    print(f"  -------------------------")
    print(f"  Passed (R+A): {passed:>3}/{total}  ({passed/total*100:.0f}%)" if total > 0 else "")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Phase 5.1: Validation Testing")
    parser.add_argument("--top", type=int, default=50, help="Top N strategies")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 5.1: Validation Set Testing")
    print("=" * 60)
    print(f"  Top strategies:  {args.top}")
    print(f"  Output dir:      {VALIDATION_RESULTS_DIR}")
    print(f"  NOTE: NO optimization — testing only!")
    print()

    summary = run_validation_testing(
        top_n=args.top,
        verbose=not args.quiet,
    )

    print_date_ranges(summary)
    print_summary_table(summary)
    print_classification_summary(summary)

    counts = summary.get("classification_counts", {})
    robust = counts.get("ROBUST", 0)
    acceptable = counts.get("ACCEPTABLE", 0)
    degraded = counts.get("DEGRADED", 0)
    overfit = counts.get("OVERFIT", 0)
    insufficient = counts.get("INSUFFICIENT", 0)
    total = summary["total_strategies"]

    print(f"\nPhase 5.1: {robust} robust, {acceptable} acceptable, "
          f"{degraded} degraded, {overfit} overfit, {insufficient} insufficient "
          f"(out of {total})")

    print("\nPhase 5.1 completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

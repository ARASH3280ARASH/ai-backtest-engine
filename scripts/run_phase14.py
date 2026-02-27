"""
Phase 14 Runner — Final Production Config + Reports (Phase 6.1)
================================================================
THE LAST PHASE. Compiles all previous results into production deliverables.
NO new optimizations — only compiling, verifying, and reporting.

Usage:
    python scripts/run_phase14.py
"""

import json
import math
import os
import sys
import logging

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from optimizer.report_generator import (
    generate_all, DETAILS_DIR, FINAL_DIR, REPORTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase14")


def verify_integrity(robot_config: dict) -> list:
    """Run integrity checks on all generated outputs."""
    issues = []

    # 1. robot_config.json is valid JSON (already parsed)
    config_path = os.path.join(FINAL_DIR, "robot_config.json")
    if not os.path.exists(config_path):
        issues.append("robot_config.json missing")
    else:
        try:
            with open(config_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            issues.append(f"robot_config.json invalid JSON: {e}")

    # 2. All strategy IDs exist in detail files
    strategies = robot_config.get("approved_individual_strategies", [])
    for s in strategies:
        sid = s["id"]
        detail_path = os.path.join(DETAILS_DIR, f"{sid}_detail.json")
        if not os.path.exists(detail_path):
            issues.append(f"Detail file missing: {sid}")
        else:
            with open(detail_path) as f:
                detail = json.load(f)
            # Check no NaN/null in critical fields
            for field in ["final_score", "strategy_id", "classification"]:
                val = detail.get(field)
                if val is None:
                    issues.append(f"{sid}: null {field}")
                elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    issues.append(f"{sid}: NaN/Inf {field}")

    # 3. No NaN/null in critical robot_config fields
    for s in strategies:
        for field in ["id", "final_score", "classification"]:
            val = s.get(field)
            if val is None:
                issues.append(f"robot_config {s.get('id', '?')}: null {field}")
            elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                issues.append(f"robot_config {s.get('id', '?')}: NaN/Inf {field}")

    # 4. All approved have test_pf > 1.0
    for s in strategies:
        test_pf = s.get("performance", {}).get("test", {}).get("pf", 0)
        if test_pf < 1.0:
            issues.append(f"{s['id']}: test_pf={test_pf} < 1.0")

    # 5. HTML report exists
    html_path = os.path.join(REPORTS_DIR, "final_report.html")
    if not os.path.exists(html_path):
        issues.append("final_report.html missing")

    # 6. CSV files exist
    csv_files = [
        "all_approved_trades.csv", "strategy_rankings.csv",
        "monthly_returns.csv", "regime_analysis.csv",
        "monte_carlo_summary.csv",
    ]
    for csv_file in csv_files:
        if not os.path.exists(os.path.join(REPORTS_DIR, csv_file)):
            issues.append(f"CSV missing: {csv_file}")

    return issues


def print_final_summary(robot_config: dict, issues: list):
    """Print the COMPLETE final summary box."""
    strategies = robot_config.get("approved_individual_strategies", [])
    pipeline = robot_config.get("pipeline_summary", {})

    # Sort by score
    sorted_strats = sorted(strategies, key=lambda x: x["final_score"], reverse=True)
    best = sorted_strats[0] if sorted_strats else {}
    safest = min(sorted_strats,
                 key=lambda x: x["performance"]["monte_carlo"]["p95_dd"]) if sorted_strats else {}

    prod_ready = sum(1 for s in strategies if s["classification"] == "PRODUCTION_READY")
    acceptable = sum(1 for s in strategies if s["classification"] == "ACCEPTABLE")
    total_net = sum(s["performance"]["test"]["net_profit"] for s in strategies)

    print("\n" + "=" * 70)
    print(f"{'FINAL SUMMARY — BTCUSD BACKTEST ENGINE':^70}")
    print("=" * 70)

    print(f"\n  {'Pipeline Results':}")
    print(f"  {'-' * 50}")
    print(f"  Total strategies discovered:     {pipeline.get('phase_5_individual', 0):>6}")
    print(f"  Combos tested:                   {pipeline.get('phase_7_combos', 0):>6}")
    print(f"  Entry optimized:                 {pipeline.get('phase_8_entry_optimized', 0):>6}")
    print(f"  Exit optimized:                  {pipeline.get('phase_10_exit_optimized', 0):>6}")
    print(f"  Combined validated:              {pipeline.get('phase_4_3_combined', 0):>6}")
    print(f"  Combined approved:               {pipeline.get('phase_4_3_approved', 0):>6}")
    print(f"  Validation tested:               {pipeline.get('phase_5_1_validation_tested', 0):>6}")
    print(f"  Validation ROBUST:               {pipeline.get('phase_5_1_robust', 0):>6}")
    print(f"  Validation ACCEPTABLE:           {pipeline.get('phase_5_1_acceptable', 0):>6}")
    print(f"  OOS tested:                      {pipeline.get('phase_5_2_oos_tested', 0):>6}")
    print(f"  PRODUCTION_READY:                {pipeline.get('phase_5_2_production_ready', 0):>6}")
    print(f"  ACCEPTABLE:                      {pipeline.get('phase_5_2_acceptable', 0):>6}")
    print(f"  {'-' * 50}")
    print(f"  FINAL APPROVED:                  {len(strategies):>6}")
    print(f"    Production-ready:              {prod_ready:>6}")
    print(f"    Acceptable:                    {acceptable:>6}")

    print(f"\n  {'Performance':}")
    print(f"  {'-' * 50}")
    print(f"  Best strategy:                   {best.get('id', 'N/A')}")
    print(f"  Best score:                      {best.get('final_score', 0):.4f}")
    best_pf = best.get('performance', {}).get('test', {}).get('pf', 0)
    print(f"  Best test PF:                    {best_pf:.2f}")
    print(f"  Safest strategy (lowest DD):     {safest.get('id', 'N/A')}")
    safest_dd = safest.get('performance', {}).get('monte_carlo', {}).get('p95_dd', 0)
    print(f"  Safest MC p95 DD:                {safest_dd:.1f}%")
    print(f"  Total test profit (all):         ${total_net:.2f}")

    print(f"\n  {'Top 10 Rankings':}")
    print(f"  {'-' * 50}")
    print(f"  {'#':>4} {'Strategy':<14} {'Score':>7} {'Test PF':>9} {'Status':<18}")
    for i, s in enumerate(sorted_strats[:10]):
        pf = s["performance"]["test"]["pf"]
        pf_str = f"{pf:.2f}" if pf < 900 else "999+"
        print(f"  {i+1:>4} {s['id']:<14} {s['final_score']:>7.3f} {pf_str:>9} {s['classification']:<18}")

    print(f"\n  {'Output Files':}")
    print(f"  {'-' * 50}")
    print(f"  robot_config.json:     {os.path.join(FINAL_DIR, 'robot_config.json')}")
    print(f"  Detail files:          {DETAILS_DIR} ({len(strategies)} files)")
    print(f"  HTML report:           {os.path.join(REPORTS_DIR, 'final_report.html')}")
    print(f"  CSV exports:           {REPORTS_DIR}")

    print(f"\n  {'Integrity Verification':}")
    print(f"  {'-' * 50}")
    if not issues:
        print(f"  ALL CHECKS PASSED")
    else:
        for issue in issues:
            print(f"  WARNING: {issue}")

    print("\n" + "=" * 70)
    if not issues:
        print(f"  Phase 6.1 COMPLETE — {len(strategies)} strategies ready for production")
    else:
        print(f"  Phase 6.1 COMPLETE with {len(issues)} warnings")
    print("=" * 70)


def main():
    print("=" * 60)
    print("PHASE 6.1: Final Production Config + Reports")
    print("=" * 60)
    print(f"  Output dirs: {FINAL_DIR}")
    print(f"               {REPORTS_DIR}")
    print(f"  NOTE: NO optimization — compiling and reporting only!")
    print()

    robot_config = generate_all(verbose=True)

    logger.info("Running integrity verification...")
    issues = verify_integrity(robot_config)

    print_final_summary(robot_config, issues)

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())

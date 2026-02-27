"""
Phase 13 Runner — Final OOS Test + Monte Carlo + Regime Analysis (Phase 5.2)
==============================================================================
Most critical phase. Uses TEST data (newest 15%) never seen before.
NO optimization. Tests validation-passing strategies only.

Usage:
    python scripts/run_phase13.py
"""

import argparse
import json
import logging
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from optimizer.final_oos_tester import run_final_oos_test, FINAL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase13")


def print_three_way_table(oos_results: dict):
    strategies = oos_results.get("strategies", [])
    if not strategies:
        print("No results to display.")
        return

    counts = oos_results.get("classification_counts", {})

    print("\n" + "=" * 140)
    print(f"{'PHASE 5.2: FINAL OOS TEST + MONTE CARLO + REGIME ANALYSIS':^140}")
    print("=" * 140)
    print(f"  Strategies tested: {oos_results['total_tested']}")
    for cls in ["PRODUCTION_READY", "ACCEPTABLE", "MARGINAL", "FAILED", "MC_REJECTED", "INSUFFICIENT"]:
        cnt = counts.get(cls, 0)
        if cnt > 0:
            print(f"  {cls:<18}: {cnt}")
    print("=" * 140)

    print(f"\n{'#':<4} {'Strategy':<14} {'Train PF':>9} {'Val PF':>8} {'Test PF':>8} "
          f"{'Path':>12} {'MC p95 DD':>10} {'MC Ruin':>8} {'Regimes':>8} "
          f"{'Score':>7} {'Status':<18}")
    print("-" * 140)

    for i, s in enumerate(strategies):
        if "error" in s:
            print(f"{i+1:<4} {s['strategy_id']:<14} {'ERROR':>9}")
            continue

        sid = s["strategy_id"]
        tw = s.get("three_way_comparison", {})
        mc = s.get("monte_carlo", {})
        reg = s.get("regime_analysis", {})

        t_pf = tw.get("train_pf", 0)
        v_pf = tw.get("val_pf", 0)
        x_pf = tw.get("test_pf", 0)

        # Degradation path arrow
        if x_pf >= v_pf * 0.9:
            path = "stable"
        elif x_pf >= v_pf * 0.5:
            path = "gradual"
        else:
            path = "steep"

        mc_p95 = mc.get("mc_p95_dd", 0)
        mc_ruin = mc.get("mc_prob_of_ruin_30pct", 0)
        n_profitable = reg.get("profitable_regimes", 0)
        n_active = reg.get("active_regimes", 0)
        regime_str = f"{n_profitable}/{n_active}"

        score = s.get("final_score", 0)
        status = s.get("classification", "?")

        print(f"{i+1:<4} {sid:<14} {t_pf:>9.2f} {v_pf:>8.2f} {x_pf:>8.2f} "
              f"{path:>12} {mc_p95:>9.1f}% {mc_ruin:>8.2%} {regime_str:>8} "
              f"{score:>7.3f} {status:<18}")

    print("-" * 140)


def print_mc_summary(oos_results: dict):
    strategies = [s for s in oos_results.get("strategies", []) if "error" not in s]
    if not strategies:
        return

    print("\n" + "=" * 80)
    print("MONTE CARLO SUMMARY")
    print("=" * 80)

    mc_passed = sum(1 for s in strategies if s["monte_carlo"]["mc_passed"])
    mc_rejected = sum(1 for s in strategies if not s["monte_carlo"]["mc_passed"])

    all_p95 = [s["monte_carlo"]["mc_p95_dd"] for s in strategies]
    all_ruin = [s["monte_carlo"]["mc_prob_of_ruin_30pct"] for s in strategies]
    all_stress = [s["monte_carlo"]["mc_stress_pct_profitable"] for s in strategies]

    import numpy as np

    print(f"  MC passed:              {mc_passed}/{len(strategies)}")
    print(f"  MC rejected:            {mc_rejected}/{len(strategies)}")
    print(f"  Avg p95 drawdown:       {np.mean(all_p95):.1f}%")
    print(f"  Max p95 drawdown:       {np.max(all_p95):.1f}%")
    print(f"  Avg ruin probability:   {np.mean(all_ruin):.2%}")
    print(f"  Avg stress profitable:  {np.mean(all_stress):.1f}%")
    print("=" * 80)


def print_regime_breakdown(oos_results: dict):
    strategies = [s for s in oos_results.get("strategies", []) if "error" not in s]
    if not strategies:
        return

    print("\n" + "=" * 80)
    print("REGIME BREAKDOWN")
    print("=" * 80)

    regime_dist = oos_results.get("regime_distribution", {})
    total_bars = sum(regime_dist.values())
    for regime, count in sorted(regime_dist.items(), key=lambda x: -x[1]):
        pct = count / total_bars * 100 if total_bars > 0 else 0
        print(f"  {regime:<16}: {count:>6} bars ({pct:.1f}%)")

    print(f"\n  Per-strategy regime profitability:")
    regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL", "LOW_VOL"]
    for regime in regimes:
        profitable = sum(1 for s in strategies
                        if s["regime_analysis"]["per_regime"].get(regime, {}).get("profitable", False))
        active = sum(1 for s in strategies
                    if s["regime_analysis"]["per_regime"].get(regime, {}).get("trades", 0) >= 3)
        if active > 0:
            print(f"    {regime:<16}: {profitable}/{active} strategies profitable")

    print("=" * 80)


def print_final_rankings(oos_results: dict):
    strategies = [s for s in oos_results.get("strategies", []) if "error" not in s]
    if not strategies:
        return

    ranked = sorted(strategies, key=lambda x: x.get("final_score", 0), reverse=True)

    print("\n" + "=" * 80)
    print("FINAL RANKINGS (sorted by composite score)")
    print("=" * 80)

    for i, s in enumerate(ranked[:20]):
        sid = s["strategy_id"]
        score = s["final_score"]
        status = s["classification"]
        pf = s["three_way_comparison"]["test_pf"]
        print(f"  #{i+1:>2}  {sid:<14} score={score:.3f}  test_pf={pf:.2f}  {status}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Phase 5.2: Final OOS Test")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 5.2: Final OOS Test + Monte Carlo + Regime Analysis")
    print("=" * 60)
    print(f"  Output dir: {FINAL_DIR}")
    print(f"  NOTE: NO optimization — testing only on UNSEEN data!")
    print()

    oos_results = run_final_oos_test(verbose=not args.quiet)

    print_three_way_table(oos_results)
    print_mc_summary(oos_results)
    print_regime_breakdown(oos_results)
    print_final_rankings(oos_results)

    counts = oos_results.get("classification_counts", {})
    prod = counts.get("PRODUCTION_READY", 0)
    accept = counts.get("ACCEPTABLE", 0)
    failed = counts.get("FAILED", 0) + counts.get("MARGINAL", 0)
    mc_rej = counts.get("MC_REJECTED", 0)
    total = oos_results["total_tested"]

    print(f"\nPhase 5.2: {prod} production-ready, {accept} acceptable, "
          f"{failed} failed, {mc_rej} rejected by MC (out of {total})")

    print("\nPhase 5.2 completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

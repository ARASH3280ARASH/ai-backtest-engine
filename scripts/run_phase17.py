"""
Phase 17 Runner — Final OOS Test + V2 Reports + Quality Verification
======================================================================
Completes the full pipeline after Phase 16 fixes:
  STEP 1: Run Phase 5.2 (OOS test on unseen test data)
  STEP 2: Generate v2 reports with output_paths overrides
  STEP 3: Run 8 quality checks on v2 detail files
  STEP 4: Print final summary

Usage:
    python scripts/run_phase17.py
    python scripts/run_phase17.py --skip-oos   # skip 5+ min OOS test, reuse existing results
"""

import argparse
import json
import math
import os
import sys
import time
import logging
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from config.settings import RESULTS_DIR, REPORTS_DIR
from optimizer.final_oos_tester import run_final_oos_test, FINAL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase17")

# V2 output paths
FINAL_V2_DIR = os.path.join(RESULTS_DIR, "final_v2")
DETAILS_V2_DIR = os.path.join(FINAL_V2_DIR, "strategy_details")
V2_PATHS = {
    "final_dir": FINAL_V2_DIR,
    "details_dir": DETAILS_V2_DIR,
    "reports_dir": REPORTS_DIR,
    "html_filename": "final_report_v2.html",
    "csv_suffix": "_v2",
}


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — OOS TEST
# ═══════════════════════════════════════════════════════════════

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

        tw = s.get("three_way_comparison", {})
        mc = s.get("monte_carlo", {})
        reg = s.get("regime_analysis", {})

        t_pf = tw.get("train_pf", 0)
        v_pf = tw.get("val_pf", 0)
        x_pf = tw.get("test_pf", 0)

        if x_pf >= v_pf * 0.9:
            path = "stable"
        elif x_pf >= v_pf * 0.5:
            path = "gradual"
        else:
            path = "steep"

        mc_p95 = mc.get("mc_p95_dd", 0)
        mc_ruin = mc.get("mc_prob_of_ruin_30pct", 0)
        regime_str = f"{reg.get('profitable_regimes', 0)}/{reg.get('active_regimes', 0)}"
        score = s.get("final_score", 0)
        status = s.get("classification", "?")

        print(f"{i+1:<4} {s['strategy_id']:<14} {t_pf:>9.2f} {v_pf:>8.2f} {x_pf:>8.2f} "
              f"{path:>12} {mc_p95:>9.1f}% {mc_ruin:>8.2%} {regime_str:>8} "
              f"{score:>7.3f} {status:<18}")

    print("-" * 140)


def print_mc_summary(oos_results: dict):
    import numpy as np
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


def run_step1_oos_test(skip_oos: bool = False):
    """STEP 1: Run Phase 5.2 OOS test."""
    print("\n" + "=" * 70)
    print("STEP 1: Final OOS Test (Phase 5.2)")
    print("=" * 70)

    if skip_oos:
        oos_path = os.path.join(FINAL_DIR, "oos_results.json")
        if not os.path.exists(oos_path):
            print("ERROR: --skip-oos specified but no existing oos_results.json found!")
            sys.exit(1)
        print(f"  [SKIP] Reusing existing results from {oos_path}")
        with open(oos_path) as f:
            oos_results = json.load(f)
    else:
        print(f"  Output dir: {FINAL_DIR}")
        print(f"  NOTE: NO optimization — testing only on UNSEEN data!")
        print()
        oos_results = run_final_oos_test(verbose=True)

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

    print(f"\n  STEP 1 result: {prod} production-ready, {accept} acceptable, "
          f"{failed} failed, {mc_rej} rejected by MC (out of {total})")

    return oos_results


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — V2 REPORTS
# ═══════════════════════════════════════════════════════════════

def _build_fix_comparison_html(details_dir: str) -> str:
    """Build 'Before Fix vs After Fix' HTML section from v2 detail files."""
    # OLD values (hardcoded from pre-fix Phase 16 analysis)
    old_approved = 29
    old_same_bar_pct = 82.6
    old_max_pf = 401.46
    old_cost = "$0.65 (fixed)"

    # NEW values (computed from v2 detail files)
    detail_files = [f for f in os.listdir(details_dir) if f.endswith("_detail.json")]
    new_approved = len(detail_files)

    total_trades = 0
    same_bar_exits = 0
    all_pfs = []
    all_costs = []

    for fname in detail_files:
        fpath = os.path.join(details_dir, fname)
        with open(fpath) as f:
            detail = json.load(f)

        trades = detail.get("trade_list", [])
        tw = detail.get("three_way_comparison", {})
        pf = tw.get("test_pf", 0)
        all_pfs.append(pf)

        for t in trades:
            total_trades += 1
            bars_held = t.get("bars_held", -1)
            if bars_held == 0:
                same_bar_exits += 1
            cost = t.get("total_cost", 0)
            all_costs.append(cost)

    new_same_bar_pct = (same_bar_exits / total_trades * 100) if total_trades > 0 else 0.0
    new_max_pf = max(all_pfs) if all_pfs else 0
    if all_costs:
        cost_min = min(all_costs)
        cost_max = max(all_costs)
        new_cost_str = f"${cost_min:.2f} - ${cost_max:.2f}"
    else:
        new_cost_str = "N/A"

    new_max_pf_str = f"{new_max_pf:.2f}" if new_max_pf < 900 else "999+"
    old_max_pf_str = f"{old_max_pf:.2f}"

    html = f"""
<h2>Before Fix vs After Fix (Phase 16 Critical Fixes)</h2>
<div style="display:flex;gap:20px;margin-bottom:20px;">
  <div style="flex:1;background:#2d1515;border:2px solid #d63031;border-radius:12px;padding:20px;">
    <h3 style="color:#d63031;margin-bottom:15px;font-size:18px;">BEFORE (v1 — 3 Bugs)</h3>
    <table style="width:100%;background:transparent;">
      <tr><td style="color:#aaa;border:none;padding:6px 0;">Approved strategies</td><td style="color:#ff7675;border:none;padding:6px 0;text-align:right;font-weight:bold;">{old_approved}</td></tr>
      <tr><td style="color:#aaa;border:none;padding:6px 0;">Same-bar exits</td><td style="color:#ff7675;border:none;padding:6px 0;text-align:right;font-weight:bold;">{old_same_bar_pct:.1f}%</td></tr>
      <tr><td style="color:#aaa;border:none;padding:6px 0;">Max profit factor</td><td style="color:#ff7675;border:none;padding:6px 0;text-align:right;font-weight:bold;">{old_max_pf_str}</td></tr>
      <tr><td style="color:#aaa;border:none;padding:6px 0;">Transaction cost</td><td style="color:#ff7675;border:none;padding:6px 0;text-align:right;font-weight:bold;">{old_cost}</td></tr>
    </table>
  </div>
  <div style="flex:1;background:#152d15;border:2px solid #00b894;border-radius:12px;padding:20px;">
    <h3 style="color:#00b894;margin-bottom:15px;font-size:18px;">AFTER (v2 — All Fixed)</h3>
    <table style="width:100%;background:transparent;">
      <tr><td style="color:#aaa;border:none;padding:6px 0;">Approved strategies</td><td style="color:#55efc4;border:none;padding:6px 0;text-align:right;font-weight:bold;">{new_approved}</td></tr>
      <tr><td style="color:#aaa;border:none;padding:6px 0;">Same-bar exits</td><td style="color:#55efc4;border:none;padding:6px 0;text-align:right;font-weight:bold;">{new_same_bar_pct:.1f}%</td></tr>
      <tr><td style="color:#aaa;border:none;padding:6px 0;">Max profit factor</td><td style="color:#55efc4;border:none;padding:6px 0;text-align:right;font-weight:bold;">{new_max_pf_str}</td></tr>
      <tr><td style="color:#aaa;border:none;padding:6px 0;">Transaction cost</td><td style="color:#55efc4;border:none;padding:6px 0;text-align:right;font-weight:bold;">{new_cost_str}</td></tr>
    </table>
  </div>
</div>

<h3>Comparison Table</h3>
<table>
<thead><tr><th>Metric</th><th>Before (v1)</th><th>After (v2)</th><th>Change</th></tr></thead>
<tbody>
<tr><td>Approved strategies</td><td>{old_approved}</td><td>{new_approved}</td>
    <td style="color:{'#00b894' if new_approved != old_approved else '#888'}">{new_approved - old_approved:+d}</td></tr>
<tr><td>Same-bar exit %</td><td>{old_same_bar_pct:.1f}%</td><td>{new_same_bar_pct:.1f}%</td>
    <td style="color:#00b894">{new_same_bar_pct - old_same_bar_pct:+.1f}%</td></tr>
<tr><td>Max profit factor</td><td>{old_max_pf_str}</td><td>{new_max_pf_str}</td>
    <td style="color:#00b894">Fixed</td></tr>
<tr><td>Transaction cost</td><td>{old_cost}</td><td>{new_cost_str}</td>
    <td style="color:#00b894">Variable</td></tr>
<tr><td>Total test trades</td><td>—</td><td>{total_trades}</td><td>—</td></tr>
</tbody>
</table>

<div style="background:var(--card);border-radius:10px;padding:15px;margin:15px 0;font-size:13px;color:#bbb;">
<strong>Phase 16 Fixes Applied:</strong>
<ol style="margin:8px 0 0 20px;">
<li><strong>Same-bar exit bug:</strong> Trailing stop / management now only activates on bar N+1, not the entry bar</li>
<li><strong>Strategy deduplication:</strong> Removed duplicate strategies sharing identical trade fingerprints</li>
<li><strong>Variable slippage:</strong> Transaction costs now vary by ATR regime ($0.52-$0.85) instead of fixed $0.65</li>
</ol>
</div>
"""
    return html


def _clean_v2_dirs():
    """Remove stale files from v2 output directories before regeneration."""
    import shutil
    if os.path.isdir(DETAILS_V2_DIR):
        shutil.rmtree(DETAILS_V2_DIR)
    if os.path.isdir(FINAL_V2_DIR):
        for f in os.listdir(FINAL_V2_DIR):
            fpath = os.path.join(FINAL_V2_DIR, f)
            if os.path.isfile(fpath):
                os.remove(fpath)


def run_step2_generate_reports():
    """STEP 2: Generate v2 reports with output_paths overrides."""
    print("\n" + "=" * 70)
    print("STEP 2: Generate V2 Reports")
    print("=" * 70)
    print(f"  robot_config -> {FINAL_V2_DIR}")
    print(f"  details      -> {DETAILS_V2_DIR}")
    print(f"  HTML report  -> {os.path.join(REPORTS_DIR, 'final_report_v2.html')}")
    print(f"  CSV suffix   -> _v2")
    print()

    # Clean stale files from previous runs
    _clean_v2_dirs()

    from optimizer.report_generator import generate_all

    t0 = time.time()
    robot_config = generate_all(verbose=True, output_paths=V2_PATHS)
    elapsed = time.time() - t0

    # Inject "Before Fix vs After Fix" section into HTML report
    html_path = os.path.join(REPORTS_DIR, "final_report_v2.html")
    if os.path.exists(html_path) and os.path.isdir(DETAILS_V2_DIR):
        print("  Injecting 'Before Fix vs After Fix' section...")
        fix_html = _build_fix_comparison_html(DETAILS_V2_DIR)

        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Insert before the <script> tag
        insert_point = content.find("<script>")
        if insert_point != -1:
            content = content[:insert_point] + fix_html + "\n" + content[insert_point:]
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("  Fix comparison section injected successfully.")

    print(f"\n  STEP 2 complete in {elapsed:.1f}s")
    return robot_config


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════

def _load_all_detail_files(details_dir: str) -> list:
    """Load all v2 detail JSON files."""
    details = []
    if not os.path.isdir(details_dir):
        return details
    for fname in sorted(os.listdir(details_dir)):
        if fname.endswith("_detail.json"):
            fpath = os.path.join(details_dir, fname)
            with open(fpath) as f:
                details.append(json.load(f))
    return details


def check_a_same_bar_exits(details: list) -> dict:
    """CHECK A (CRITICAL): TRAIL exits with bars_held=0 -> expect 0."""
    violations = []
    for d in details:
        sid = d["strategy_id"]
        for t in d.get("trade_list", []):
            if t.get("exit_reason") == "TRAIL" and t.get("bars_held", -1) == 0:
                violations.append(f"{sid} trade#{t.get('trade_num', '?')}")

    passed = len(violations) == 0
    return {
        "name": "A: Same-bar TRAIL exits",
        "level": "CRITICAL",
        "passed": passed,
        "count": len(violations),
        "detail": f"{len(violations)} TRAIL exits with bars_held=0" + (
            f" [{', '.join(violations[:5])}]" if violations else ""),
    }


def check_b_duplicates(details: list) -> dict:
    """CHECK B (CRITICAL): Duplicate strategies by trade fingerprint -> expect 0."""
    fingerprints = {}
    for d in details:
        sid = d["strategy_id"]
        trades = d.get("trade_list", [])
        # Fingerprint: tuple of entry_bar values
        fp = tuple(t.get("entry_bar", -1) for t in trades)
        if fp in fingerprints:
            fingerprints[fp].append(sid)
        else:
            fingerprints[fp] = [sid]

    dupes = {fp: sids for fp, sids in fingerprints.items() if len(sids) > 1}
    dupe_count = sum(len(sids) - 1 for sids in dupes.values())

    detail_str = f"{dupe_count} duplicate strategies"
    if dupes:
        groups = [f"{{{', '.join(sids)}}}" for sids in list(dupes.values())[:3]]
        detail_str += f" [{'; '.join(groups)}]"

    return {
        "name": "B: Duplicate strategies",
        "level": "CRITICAL",
        "passed": dupe_count == 0,
        "count": dupe_count,
        "detail": detail_str,
    }


def check_c_variable_cost(details: list) -> dict:
    """CHECK C (CRITICAL): Variable cost std > 0 -> print min/max/avg/std."""
    import numpy as np
    all_costs = []
    for d in details:
        for t in d.get("trade_list", []):
            c = t.get("total_cost", 0)
            all_costs.append(c)

    if not all_costs:
        return {
            "name": "C: Variable transaction costs",
            "level": "CRITICAL",
            "passed": False,
            "count": 0,
            "detail": "No trades found",
        }

    costs = np.array(all_costs)
    cost_std = float(np.std(costs))
    cost_min = float(np.min(costs))
    cost_max = float(np.max(costs))
    cost_avg = float(np.mean(costs))

    passed = cost_std > 0
    return {
        "name": "C: Variable transaction costs",
        "level": "CRITICAL",
        "passed": passed,
        "count": len(all_costs),
        "detail": f"min=${cost_min:.4f} max=${cost_max:.4f} avg=${cost_avg:.4f} std=${cost_std:.4f}" + (
            "" if passed else " (FIXED COST DETECTED)"),
    }


def check_d_high_pf(details: list) -> dict:
    """CHECK D (FLAG): PF > 10 -> list flagged strategies."""
    flagged = []
    for d in details:
        pf = d.get("three_way_comparison", {}).get("test_pf", 0)
        if pf > 10:
            pf_str = f"{pf:.2f}" if pf < 900 else "999+"
            flagged.append(f"{d['strategy_id']}(PF={pf_str})")

    return {
        "name": "D: High PF (>10) strategies",
        "level": "FLAG",
        "passed": True,  # Flags don't fail the pipeline
        "count": len(flagged),
        "detail": f"{len(flagged)} flagged" + (f" [{', '.join(flagged[:5])}]" if flagged else ""),
    }


def check_e_high_winrate(details: list) -> dict:
    """CHECK E (FLAG): Win rate > 90% -> list flagged."""
    flagged = []
    for d in details:
        wr = d.get("three_way_comparison", {}).get("test_wr", 0)
        if wr > 90:
            flagged.append(f"{d['strategy_id']}(WR={wr:.1f}%)")

    return {
        "name": "E: High win rate (>90%) strategies",
        "level": "FLAG",
        "passed": True,
        "count": len(flagged),
        "detail": f"{len(flagged)} flagged" + (f" [{', '.join(flagged[:5])}]" if flagged else ""),
    }


def check_f_trail_bars_held(details: list) -> dict:
    """CHECK F (CRITICAL): TRAIL bars_held min >= 1 -> print distribution."""
    trail_bars = []
    for d in details:
        for t in d.get("trade_list", []):
            if t.get("exit_reason") == "TRAIL":
                trail_bars.append(t.get("bars_held", 0))

    if not trail_bars:
        return {
            "name": "F: TRAIL bars_held >= 1",
            "level": "CRITICAL",
            "passed": True,
            "count": 0,
            "detail": "No TRAIL exits found (OK)",
        }

    import numpy as np
    bars = np.array(trail_bars)
    min_bars = int(np.min(bars))
    passed = min_bars >= 1

    return {
        "name": "F: TRAIL bars_held >= 1",
        "level": "CRITICAL",
        "passed": passed,
        "count": len(trail_bars),
        "detail": f"{len(trail_bars)} TRAIL exits: min={min_bars} max={int(np.max(bars))} "
                  f"avg={float(np.mean(bars)):.1f} median={float(np.median(bars)):.0f}",
    }


def check_g_low_trades(details: list) -> dict:
    """CHECK G (FLAG): < 15 test trades -> list low-confidence."""
    flagged = []
    for d in details:
        n_trades = d.get("three_way_comparison", {}).get("test_trades", 0)
        if n_trades < 15:
            flagged.append(f"{d['strategy_id']}(n={n_trades})")

    return {
        "name": "G: Low trade count (<15 test trades)",
        "level": "FLAG",
        "passed": True,
        "count": len(flagged),
        "detail": f"{len(flagged)} low-confidence" + (f" [{', '.join(flagged[:5])}]" if flagged else ""),
    }


def check_h_data_integrity(details: list, robot_config: dict) -> dict:
    """CHECK H (CRITICAL): Data integrity — valid JSON, IDs match, no NaN/null."""
    issues = []

    # Get strategy IDs from robot_config
    config_ids = set()
    for s in robot_config.get("approved_individual_strategies", []):
        config_ids.add(s["id"])

    detail_ids = set()
    for d in details:
        sid = d.get("strategy_id")
        if sid is None:
            issues.append("Detail file missing strategy_id")
            continue
        detail_ids.add(sid)

        # Check critical fields for NaN/null
        for field in ["final_score", "classification"]:
            val = d.get(field)
            if val is None:
                issues.append(f"{sid}: null {field}")
            elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                issues.append(f"{sid}: NaN/Inf {field}")

        # Check three_way_comparison fields
        tw = d.get("three_way_comparison", {})
        for k in ["test_pf", "test_trades", "test_net"]:
            val = tw.get(k)
            if val is None:
                issues.append(f"{sid}: null {k}")
            elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                issues.append(f"{sid}: NaN/Inf {k}")

    # Check IDs match between config and details
    missing_in_details = config_ids - detail_ids
    missing_in_config = detail_ids - config_ids
    if missing_in_details:
        issues.append(f"IDs in config but not in details: {missing_in_details}")
    if missing_in_config:
        issues.append(f"IDs in details but not in config: {missing_in_config}")

    passed = len(issues) == 0
    return {
        "name": "H: Data integrity",
        "level": "CRITICAL",
        "passed": passed,
        "count": len(issues),
        "detail": f"{len(issues)} issues" + (f" [{'; '.join(issues[:3])}]" if issues else ""),
    }


def run_step3_quality_checks(robot_config: dict) -> list:
    """STEP 3: Run 8 quality checks on v2 detail files."""
    print("\n" + "=" * 70)
    print("STEP 3: Quality Checks (8 checks on v2 detail files)")
    print("=" * 70)

    details = _load_all_detail_files(DETAILS_V2_DIR)
    print(f"  Loaded {len(details)} detail files from {DETAILS_V2_DIR}")
    print()

    checks = [
        check_a_same_bar_exits(details),
        check_b_duplicates(details),
        check_c_variable_cost(details),
        check_d_high_pf(details),
        check_e_high_winrate(details),
        check_f_trail_bars_held(details),
        check_g_low_trades(details),
        check_h_data_integrity(details, robot_config),
    ]

    for c in checks:
        icon = "PASS" if c["passed"] else "FAIL"
        level_str = f"[{c['level']}]"
        print(f"  {icon:>4} {level_str:<10} {c['name']}")
        print(f"         {c['detail']}")
        print()

    # Summary
    critical_checks = [c for c in checks if c["level"] == "CRITICAL"]
    critical_passed = sum(1 for c in critical_checks if c["passed"])
    critical_total = len(critical_checks)
    flag_checks = [c for c in checks if c["level"] == "FLAG"]
    flag_count = sum(c["count"] for c in flag_checks)

    print(f"  Critical checks: {critical_passed}/{critical_total} passed")
    print(f"  Flags raised: {flag_count}")

    return checks


# ═══════════════════════════════════════════════════════════════
#  STEP 4 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

def run_step4_summary(robot_config: dict, checks: list):
    """STEP 4: Print final summary box."""
    strategies = robot_config.get("approved_individual_strategies", [])
    pipeline = robot_config.get("pipeline_summary", {})

    sorted_strats = sorted(strategies, key=lambda x: x["final_score"], reverse=True)
    best = sorted_strats[0] if sorted_strats else {}
    safest = min(sorted_strats,
                 key=lambda x: x["performance"]["monte_carlo"]["p95_dd"]) if sorted_strats else {}

    prod_ready = sum(1 for s in strategies if s["classification"] == "PRODUCTION_READY")
    acceptable = sum(1 for s in strategies if s["classification"] == "ACCEPTABLE")
    total_net = sum(s["performance"]["test"]["net_profit"] for s in strategies)

    critical_checks = [c for c in checks if c["level"] == "CRITICAL"]
    all_critical_pass = all(c["passed"] for c in critical_checks)

    print("\n" + "=" * 70)
    print(f"{'FINAL SUMMARY — BTCUSD BACKTEST ENGINE (V2)':^70}")
    print("=" * 70)

    print(f"\n  {'Pipeline Funnel':}")
    print(f"  {'-' * 55}")
    print(f"  Total strategies discovered:       {pipeline.get('phase_5_individual', 0):>6}")
    print(f"  Entry optimized:                   {pipeline.get('phase_8_entry_optimized', 0):>6}")
    print(f"  Exit optimized:                    {pipeline.get('phase_10_exit_optimized', 0):>6}")
    print(f"  Combined validated:                {pipeline.get('phase_4_3_combined', 0):>6}")
    print(f"  Combined approved:                 {pipeline.get('phase_4_3_approved', 0):>6}")
    print(f"  Validation ROBUST:                 {pipeline.get('phase_5_1_robust', 0):>6}")
    print(f"  Validation ACCEPTABLE:             {pipeline.get('phase_5_1_acceptable', 0):>6}")
    print(f"  OOS tested (Phase 5.2):            {pipeline.get('phase_5_2_oos_tested', 0):>6}")
    print(f"  PRODUCTION_READY:                  {pipeline.get('phase_5_2_production_ready', 0):>6}")
    print(f"  ACCEPTABLE:                        {pipeline.get('phase_5_2_acceptable', 0):>6}")
    print(f"  {'-' * 55}")
    print(f"  FINAL APPROVED:                    {len(strategies):>6}")
    print(f"    Production-ready:                {prod_ready:>6}")
    print(f"    Acceptable:                      {acceptable:>6}")

    print(f"\n  {'Best & Safest':}")
    print(f"  {'-' * 55}")
    best_pf = best.get('performance', {}).get('test', {}).get('pf', 0)
    best_pf_str = f"{best_pf:.2f}" if best_pf < 900 else "999+"
    print(f"  Best strategy:                     {best.get('id', 'N/A')}")
    print(f"  Best score:                        {best.get('final_score', 0):.4f}")
    print(f"  Best test PF:                      {best_pf_str}")
    safest_dd = safest.get('performance', {}).get('monte_carlo', {}).get('p95_dd', 0)
    print(f"  Safest (lowest DD):                {safest.get('id', 'N/A')}")
    print(f"  Safest MC p95 DD:                  {safest_dd:.1f}%")
    print(f"  Total test profit (all):           ${total_net:.2f}")

    print(f"\n  {'Top 10 Rankings':}")
    print(f"  {'-' * 55}")
    print(f"  {'#':>4} {'Strategy':<14} {'Score':>7} {'Test PF':>9} {'Status':<18}")
    for i, s in enumerate(sorted_strats[:10]):
        pf = s["performance"]["test"]["pf"]
        pf_str = f"{pf:.2f}" if pf < 900 else "999+"
        print(f"  {i+1:>4} {s['id']:<14} {s['final_score']:>7.3f} {pf_str:>9} {s['classification']:<18}")

    print(f"\n  {'Quality Checks':}")
    print(f"  {'-' * 55}")
    for c in checks:
        icon = "PASS" if c["passed"] else "FAIL"
        print(f"  {icon:>4}  {c['name']}")

    print(f"\n  {'Output Files':}")
    print(f"  {'-' * 55}")
    print(f"  robot_config.json:     {os.path.join(FINAL_V2_DIR, 'robot_config.json')}")
    print(f"  Detail files:          {DETAILS_V2_DIR} ({len(strategies)} files)")
    print(f"  HTML report:           {os.path.join(REPORTS_DIR, 'final_report_v2.html')}")
    print(f"  CSV exports:           {REPORTS_DIR} (*_v2.csv)")

    print("\n" + "=" * 70)
    if all_critical_pass:
        print(f"  Phase 17 COMPLETE — {len(strategies)} strategies ready for production (V2)")
        print(f"  All {len(critical_checks)} critical checks PASSED")
    else:
        failed_critical = [c["name"] for c in critical_checks if not c["passed"]]
        print(f"  Phase 17 COMPLETE with {len(failed_critical)} CRITICAL failures:")
        for name in failed_critical:
            print(f"    FAIL: {name}")
    print("=" * 70)

    return all_critical_pass


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 17: Final OOS Test + V2 Reports + Quality")
    parser.add_argument("--skip-oos", action="store_true",
                        help="Skip OOS test (reuse existing results)")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 17: Final OOS Test + V2 Reports + Quality Verification")
    print("=" * 70)
    print(f"  V2 output dir:  {FINAL_V2_DIR}")
    print(f"  Reports dir:    {REPORTS_DIR}")
    print(f"  Skip OOS:       {args.skip_oos}")
    print()

    t0 = time.time()

    # STEP 1: OOS Test
    oos_results = run_step1_oos_test(skip_oos=args.skip_oos)

    # STEP 2: Generate V2 Reports
    robot_config = run_step2_generate_reports()

    # STEP 3: Quality Checks
    checks = run_step3_quality_checks(robot_config)

    # STEP 4: Final Summary
    all_critical_pass = run_step4_summary(robot_config, checks)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    return 0 if all_critical_pass else 1


if __name__ == "__main__":
    sys.exit(main())

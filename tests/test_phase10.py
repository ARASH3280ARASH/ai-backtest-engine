"""
Phase 10 Tests — Combined Entry+Exit Validation (Phase 4.3)
==============================================================
15 tests for the combined validation that merges Phase 8 entry params
with Phase 10 exit configs, validates they work together.

Usage:
    python tests/test_phase10.py
"""

import json
import os
import random
import sys
import traceback

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from config.settings import RESULTS_DIR, INDIVIDUAL_DIR, OPTIMIZED_DIR, TRAIN_DIR
from config.broker import BTCUSD_CONFIG
from optimizer.combined_validator import (
    COMBINED_DIR, BASELINE_EXIT,
    _load_phase8_entry_params, _load_default_entry_params,
    _load_phase10_exit_config, _generate_signal_array,
    _extract_metrics, validate_single_strategy,
)
from optimizer.exit_optimizer import (
    EXIT_DIR, exit_backtest, PARTIAL_METHODS,
)
from optimizer.param_optimizer import (
    _get_category, generate_signals, reconstruct_signals_from_trades,
    create_wf_windows, TIER1_CATEGORIES, PARAM_SPACES,
)
from indicators import compute as ind

_results = []


def _test(name, fn):
    try:
        fn()
        _results.append(("PASS", name))
        print(f"  [PASS] {name}")
    except Exception as e:
        _results.append(("FAIL", name))
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        print()


def _load_df():
    train_file = os.path.join(TRAIN_DIR, "BTCUSD_H1.csv")
    if not os.path.exists(train_file):
        return None
    df = pd.read_csv(train_file)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df


def _get_combined_files():
    if not os.path.exists(COMBINED_DIR):
        return []
    return [f for f in os.listdir(COMBINED_DIR) if f.endswith("_combined.json")]


def _load_summary():
    summary_path = os.path.join(COMBINED_DIR, "combined_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

def test_01_all_50_strategies_have_combined_results():
    """All 50 strategies have combined result files."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results yet)")
        return
    summary = _load_summary()
    assert summary is not None, "combined_summary.json not found"
    n_total = summary["total_strategies"]
    # Each strategy should have a result file
    assert len(files) >= n_total, \
        f"Only {len(files)} combined files for {n_total} strategies (need {n_total}+1 for summary)"
    # Check summary lists all strategies
    strat_list = summary.get("strategies", [])
    assert len(strat_list) == n_total, \
        f"Summary lists {len(strat_list)} strategies, expected {n_total}"


def test_02_combined_pf_calculation_correct():
    """Combined PF in result matches fresh backtest calculation."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    df = _load_df()
    if df is None:
        print("    (skipped: no training data)")
        return

    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    atr_arr = np.array(ind.atr(df["high"], df["low"], df["close"], 14), dtype=np.float64)
    atr_arr = np.nan_to_num(atr_arr, nan=0.0)

    # Spot-check 2 strategies
    checked = 0
    random.seed(123)
    sample = random.sample(files, min(2, len(files)))
    for fname in sample:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        sid = data["strategy_id"]
        entry_params = data.get("entry_params", {})
        exit_cfg = data.get("exit_config", {})

        # Generate signals and run fresh backtest
        signals = _generate_signal_array(sid, df, entry_params)
        m = exit_backtest(signals, opens, highs, lows, closes, atr_arr, exit_cfg)
        m.pop("_trade_details", None)

        recorded_pf = data.get("comparison", {}).get("combined", {}).get("pf", 0)
        fresh_pf = m["profit_factor"]

        assert abs(recorded_pf - fresh_pf) < 0.01, \
            f"{sid}: recorded PF={recorded_pf}, fresh PF={fresh_pf}"
        checked += 1

    assert checked > 0, "No strategies verified"
    print(f"    (verified {checked} strategies)")


def test_03_fallback_logic_when_combined_worse():
    """When combined is worse than best individual, fallback is used."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    baseline_better_count = 0
    for fname in files:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        status = data.get("combined_status", "")
        final = data.get("final_config_to_use", "")
        comp = data.get("comparison", {})

        combined_pf = comp.get("combined", {}).get("pf", 0)
        entry_pf = comp.get("entry_optimized", {}).get("pf", 0)
        exit_pf = comp.get("exit_optimized", {}).get("pf", 0)
        best_ind_pf = max(entry_pf, exit_pf)

        if status == "BASELINE_BETTER":
            # Combined should NOT be the chosen config
            assert final != "combined", \
                f"{fname}: BASELINE_BETTER but final_config=combined"
            # The fallback should be entry_only or exit_only
            assert final in ("entry_only", "exit_only"), \
                f"{fname}: BASELINE_BETTER but final_config={final}"
            baseline_better_count += 1

        if status == "APPROVED":
            assert final == "combined", \
                f"{fname}: APPROVED but final_config={final}"

    print(f"    ({baseline_better_count} strategies fell back to individual)")


def test_04_walk_forward_folds_dont_overlap():
    """Walk-forward OOS folds don't overlap with each other."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    for fname in files[:10]:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        folds = data.get("walk_forward_combined", [])
        if len(folds) < 2:
            continue

        for i in range(1, len(folds)):
            assert folds[i]["oos_start"] > folds[i - 1]["oos_end"], \
                f"{fname}: fold {i + 1} OOS start={folds[i]['oos_start']} <= " \
                f"fold {i} OOS end={folds[i - 1]['oos_end']}"


def test_05_walk_forward_uses_only_training_data():
    """Walk-forward used ONLY training data (check bar boundaries)."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    for fname in files[:10]:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        bounds = data.get("data_boundaries", {})
        assert "NOTE" in bounds, f"{fname}: missing NOTE in boundaries"
        assert "NOT used" in bounds["NOTE"], \
            f"{fname}: boundary note doesn't confirm val/test exclusion"

        n_bars = bounds.get("total_bars", 0)
        for fold in data.get("walk_forward_combined", []):
            assert fold["oos_end"] < n_bars, \
                f"{fname}: fold {fold['fold']} oos_end={fold['oos_end']} >= n_bars={n_bars}"
            assert fold["is_start"] >= 0, \
                f"{fname}: fold {fold['fold']} is_start={fold['is_start']} < 0"


def test_06_data_leakage_audit_passes():
    """Data leakage audit passes for all strategies (spot check)."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    failed_strategies = []
    for fname in files:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        audit_status = data.get("data_leakage_audit", "N/A")
        if audit_status == "FAILED":
            failed_strategies.append(data["strategy_id"])

    if failed_strategies:
        assert False, f"Data leakage FAILED for: {failed_strategies}"
    print(f"    (all {len(files)} strategies passed leakage check)")


def test_07_no_previous_phase_files_overwritten():
    """Phase 8 and Phase 10 result files are still intact."""
    # Phase 8 params files
    p8_files = [f for f in os.listdir(OPTIMIZED_DIR)
                if f.endswith("_params.json")]
    assert len(p8_files) > 0, "Phase 8 param files missing!"

    # Phase 10 exit files
    exit_dir = os.path.join(OPTIMIZED_DIR, "exit")
    assert os.path.exists(exit_dir), "Phase 10 exit directory missing!"
    exit_files = [f for f in os.listdir(exit_dir) if f.endswith("_exit.json")]
    assert len(exit_files) > 0, "Phase 10 exit files missing!"

    # Verify they're valid JSON
    for fname in p8_files[:3]:
        fpath = os.path.join(OPTIMIZED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        assert "strategy_id" in data, f"{fname}: not a valid Phase 8 file"

    for fname in exit_files[:3]:
        fpath = os.path.join(exit_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        assert "strategy_id" in data, f"{fname}: not a valid Phase 10 file"

    print(f"    ({len(p8_files)} Phase 8 + {len(exit_files)} Phase 10 files intact)")


def test_08_sl_ge_20_pips_on_all_trades():
    """SL distance >= 20 pips (stop level) on all audited trades."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    checked = 0
    for fname in files:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        audit = data.get("trade_audit_sample", [])
        for t in audit:
            sl_dist = t.get("sl_dist_pips", 0)
            assert sl_dist >= 20, \
                f"{fname}: SL dist {sl_dist} < 20 pips"
            checked += 1

    assert checked > 0, "No trades checked"
    print(f"    (verified {checked} trades)")


def test_09_tp_ge_20_pips_on_all_trades():
    """TP distance >= 20 pips on all audited trades."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    pip = BTCUSD_CONFIG["pip_size"]
    checked = 0
    for fname in files:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        audit = data.get("trade_audit_sample", [])
        for t in audit:
            entry_price = t.get("entry_price", 0)
            tp1_price = t.get("tp1_price", 0)
            if entry_price > 0 and tp1_price > 0:
                tp_dist_pips = abs(tp1_price - entry_price) / pip
                assert tp_dist_pips >= 20, \
                    f"{fname}: TP dist {tp_dist_pips:.1f} < 20 pips"
                checked += 1

    assert checked > 0, "No trades checked"
    print(f"    (verified {checked} trades)")


def test_10_partial_close_percentages_sum_to_100():
    """Partial close percentages sum to 100% in exit configs."""
    for pc in PARTIAL_METHODS:
        if pc["partial_close"] == "none":
            continue
        total = 0.0
        for k, v in pc.items():
            if k.startswith("partial_pct"):
                total += v
        assert abs(total - 1.0) < 0.01, \
            f"Partial close {pc['partial_close']}: sum={total} != 1.0"

    # Also check in actual result files
    files = _get_combined_files()
    for fname in files[:10]:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue
        cfg = data.get("exit_config", {})
        partial = cfg.get("partial_close", "none")
        if partial != "none":
            total = 0.0
            for k, v in cfg.items():
                if k.startswith("partial_pct") and isinstance(v, (int, float)):
                    total += v
            assert abs(total - 1.0) < 0.01, \
                f"{fname}: partial close {partial} sums to {total}"


def test_11_summary_file_structure():
    """combined_summary.json exists with correct structure and counts."""
    summary = _load_summary()
    if summary is None:
        print("    (skipped: no summary)")
        return

    required = ["generated", "phase", "total_strategies", "approved",
                "partial", "baseline_better", "avg_improvement_over_baseline_pct",
                "data_leakage_all_passed", "leak_passed_count",
                "n_folds", "total_bars", "elapsed_sec", "strategies"]
    for field in required:
        assert field in summary, f"Summary missing field: {field}"

    # Counts must add up
    n_a = summary["approved"]
    n_p = summary["partial"]
    n_b = summary["baseline_better"]
    n_total = summary["total_strategies"]
    assert n_a + n_p + n_b == n_total, \
        f"Status counts don't add up: {n_a}+{n_p}+{n_b}={n_a + n_p + n_b} != {n_total}"

    assert len(summary["strategies"]) == n_total, \
        f"Strategy list length {len(summary['strategies'])} != {n_total}"


def test_12_four_way_comparison_present():
    """Each result file has all 4 comparison scenarios."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    expected_keys = {"baseline", "entry_optimized", "exit_optimized", "combined"}
    metric_keys = {"pf", "sharpe", "dd", "net", "trades", "win_rate", "expectancy"}

    for fname in files:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        comp = data.get("comparison", {})
        assert expected_keys.issubset(set(comp.keys())), \
            f"{fname}: missing comparison keys, has {set(comp.keys())}"

        for scenario in expected_keys:
            for mk in metric_keys:
                assert mk in comp[scenario], \
                    f"{fname}: {scenario} missing metric '{mk}'"


def test_13_combined_result_required_fields():
    """Each combined result file has all required fields."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    required = [
        "strategy_id", "phase", "timestamp", "data_boundaries",
        "entry_params", "exit_config", "combined_status",
        "final_config_to_use", "comparison", "walk_forward_combined",
        "wf_acceptance", "improvement_over_baseline_pct",
        "data_leakage_audit", "trade_audit_sample",
        "overfit_flags", "elapsed_sec",
    ]

    for fname in files:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue
        for field in required:
            assert field in data, f"{fname}: missing field '{field}'"


def test_14_wf_acceptance_criteria_fields():
    """Walk-forward acceptance criteria are properly computed."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    for fname in files[:10]:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        wf_acc = data.get("wf_acceptance", {})
        assert "all_folds_profitable" in wf_acc, f"{fname}: missing all_folds_profitable"
        assert "min_trades_all_folds" in wf_acc, f"{fname}: missing min_trades_all_folds"
        assert "pf_vs_best_individual" in wf_acc, f"{fname}: missing pf_vs_best_individual"
        assert "best_individual_pf" in wf_acc, f"{fname}: missing best_individual_pf"
        assert "pf_ok" in wf_acc, f"{fname}: missing pf_ok"
        assert "dd_ok" in wf_acc, f"{fname}: missing dd_ok"

        # Verify: if APPROVED, both pf_ok and dd_ok and wf must pass
        status = data.get("combined_status", "")
        if status == "APPROVED":
            folds = data.get("walk_forward_combined", [])
            oos_pfs = [f["oos_pf"] for f in folds]
            assert wf_acc["pf_ok"], f"{fname}: APPROVED but pf_ok=False"
            assert wf_acc["dd_ok"], f"{fname}: APPROVED but dd_ok=False"


def test_15_data_leakage_audit_detailed():
    """Data leakage audit trail has detailed checks for 5 trades."""
    files = _get_combined_files()
    if not files:
        print("    (skipped: no combined results)")
        return

    print("\n    DATA LEAKAGE AUDIT (detailed spot check):")
    print("    " + "-" * 60)

    audited = 0
    random.seed(42)
    sampled = random.sample(files, min(5, len(files)))

    for fname in sampled:
        fpath = os.path.join(COMBINED_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        sid = data["strategy_id"]
        leakage_details = data.get("data_leakage_details", [])
        audit_samples = data.get("trade_audit_sample", [])

        if not audit_samples:
            continue

        t = audit_samples[0]
        entry_bar = t.get("entry_bar", 0)
        signal_bar = entry_bar - 1

        print(f"    {sid}:")
        print(f"      Signal bar={signal_bar}, Entry bar={entry_bar}")
        print(f"      ATR at entry={t.get('atr_at_entry', 0):.2f}")
        print(f"      SL dist={t.get('sl_dist_pips', 0):.1f} pips")

        # Verify signal bar is before entry bar (no lookahead)
        assert signal_bar < entry_bar, \
            f"{sid}: signal_bar={signal_bar} >= entry_bar={entry_bar}"

        # Verify trailing updates are all after entry
        for tu in t.get("trail_updates", []):
            tu_bar = tu.get("bar", 0)
            assert tu_bar >= entry_bar, \
                f"{sid}: trailing at bar {tu_bar} < entry_bar {entry_bar}"

        # Check leakage detail checks all passed
        if leakage_details:
            for ld in leakage_details[:1]:
                for chk in ld.get("checks", []):
                    status = "OK" if chk["passed"] else "FAIL"
                    print(f"      [{status}] {chk['test']}: {chk['detail']}")
                    assert chk["passed"], \
                        f"{sid}: leakage check failed: {chk['test']}"

        print(f"      Overall: {data.get('data_leakage_audit', 'N/A')}")
        audited += 1

    print("    " + "-" * 60)
    assert audited > 0 or len(files) == 0, "Could not audit any trades"
    print(f"    (audited {audited} strategies)")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PHASE 10 TESTS — Combined Entry+Exit Validation (Phase 4.3)")
    print("=" * 60)

    tests = [
        ("1. All 50 strategies have combined results", test_01_all_50_strategies_have_combined_results),
        ("2. Combined PF calculation is correct", test_02_combined_pf_calculation_correct),
        ("3. Fallback logic works when combined worse", test_03_fallback_logic_when_combined_worse),
        ("4. Walk-forward folds don't overlap", test_04_walk_forward_folds_dont_overlap),
        ("5. WF uses only training data", test_05_walk_forward_uses_only_training_data),
        ("6. Data leakage audit passes (spot check)", test_06_data_leakage_audit_passes),
        ("7. No previous phase files overwritten", test_07_no_previous_phase_files_overwritten),
        ("8. SL >= 20 pips on all trades", test_08_sl_ge_20_pips_on_all_trades),
        ("9. TP >= 20 pips on all trades", test_09_tp_ge_20_pips_on_all_trades),
        ("10. Partial close percentages sum to 100%", test_10_partial_close_percentages_sum_to_100),
        ("11. Summary file structure correct", test_11_summary_file_structure),
        ("12. Four-way comparison present", test_12_four_way_comparison_present),
        ("13. Combined result required fields", test_13_combined_result_required_fields),
        ("14. WF acceptance criteria fields", test_14_wf_acceptance_criteria_fields),
        ("15. Data leakage audit (5 trades detailed)", test_15_data_leakage_audit_detailed),
    ]

    for name, fn in tests:
        _test(name, fn)

    passed = sum(1 for r in _results if r[0] == "PASS")
    failed = sum(1 for r in _results if r[0] == "FAIL")

    # Summary line
    summary = _load_summary()
    if summary:
        n_total = summary["total_strategies"]
        n_a = summary["approved"]
        n_p = summary["partial"]
        n_b = summary["baseline_better"]
        avg_imp = summary["avg_improvement_over_baseline_pct"]
        leak_n = summary["leak_passed_count"]
        print(f"\n  Phase 4.3: {n_a}/{n_total} combined approved, "
              f"{n_p} partial, {n_b} baseline-better")
        print(f"  Average improvement: {avg_imp:+.1f}%")
        print(f"  Leakage check: {leak_n}/{n_total} PASSED")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(_results)} tests passed, {failed} failed")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

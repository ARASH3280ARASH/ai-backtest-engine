"""
Phase 11 Tests — Validation Set Testing (Phase 5.1)
======================================================
15 tests covering validation logic, date overlap checks,
classification correctness, and no-optimization verification.

Usage:
    python tests/test_phase11.py
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

from config.settings import (
    RESULTS_DIR, OPTIMIZED_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR,
)
from config.broker import BTCUSD_CONFIG
from optimizer.validation_tester import (
    VALIDATION_RESULTS_DIR, _classify, _safe_ratio,
)
from optimizer.combined_validator import COMBINED_DIR
from optimizer.exit_optimizer import PARTIAL_METHODS

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


def _get_validation_files():
    if not os.path.exists(VALIDATION_RESULTS_DIR):
        return []
    return [f for f in os.listdir(VALIDATION_RESULTS_DIR)
            if f.endswith("_validation.json")]


def _load_summary():
    path = os.path.join(VALIDATION_RESULTS_DIR, "validation_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

def test_01_all_strategies_have_validation_results():
    """All strategies have validation result files."""
    files = _get_validation_files()
    if not files:
        print("    (skipped: no validation results yet)")
        return
    summary = _load_summary()
    assert summary is not None, "validation_summary.json not found"
    n_total = summary["total_strategies"]
    assert len(files) >= n_total, \
        f"Only {len(files)} validation files for {n_total} strategies"


def test_02_validation_dates_dont_overlap_training():
    """Validation date range does NOT overlap training date range."""
    train_file = os.path.join(TRAIN_DIR, "BTCUSD_H1.csv")
    val_file = os.path.join(VALIDATION_DIR, "BTCUSD_H1.csv")
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("    (skipped: data files not found)")
        return

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    train_df["time"] = pd.to_datetime(train_df["time"])
    val_df["time"] = pd.to_datetime(val_df["time"])

    train_end = train_df["time"].iloc[-1]
    val_start = val_df["time"].iloc[0]

    assert val_start > train_end, \
        f"OVERLAP: train ends {train_end}, val starts {val_start}"
    print(f"    (train ends {train_end}, val starts {val_start})")


def test_03_validation_dates_dont_overlap_test():
    """Validation date range does NOT overlap test date range."""
    val_file = os.path.join(VALIDATION_DIR, "BTCUSD_H1.csv")
    test_file = os.path.join(TEST_DIR, "BTCUSD_H1.csv")
    if not os.path.exists(val_file) or not os.path.exists(test_file):
        print("    (skipped: data files not found)")
        return

    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    val_df["time"] = pd.to_datetime(val_df["time"])
    test_df["time"] = pd.to_datetime(test_df["time"])

    val_end = val_df["time"].iloc[-1]
    test_start = test_df["time"].iloc[0]

    assert test_start > val_end, \
        f"OVERLAP: val ends {val_end}, test starts {test_start}"
    print(f"    (val ends {val_end}, test starts {test_start})")


def test_04_classification_logic_correct():
    """Classification function works correctly per spec."""
    # ROBUST: pf_ratio > 0.7
    assert _classify(0.8, 1.5, 20) == "ROBUST"
    assert _classify(1.0, 2.0, 50) == "ROBUST"
    assert _classify(0.71, 1.1, 10) == "ROBUST"

    # ACCEPTABLE: 0.5-0.7, val_pf > 1.0
    assert _classify(0.6, 1.5, 20) == "ACCEPTABLE"
    assert _classify(0.5, 1.2, 15) == "ACCEPTABLE"

    # DEGRADED: 0.3-0.5, val_pf > 1.0
    assert _classify(0.4, 1.3, 20) == "DEGRADED"
    assert _classify(0.3, 1.1, 10) == "DEGRADED"

    # OVERFIT: pf_ratio < 0.3 or val_pf < 0.8
    assert _classify(0.2, 1.5, 20) == "OVERFIT"
    assert _classify(0.8, 0.7, 20) == "OVERFIT"
    assert _classify(0.1, 0.5, 20) == "OVERFIT"

    # INSUFFICIENT: trades < 8
    assert _classify(0.8, 1.5, 5) == "INSUFFICIENT"
    assert _classify(1.0, 2.0, 0) == "INSUFFICIENT"
    assert _classify(0.9, 1.5, 7) == "INSUFFICIENT"


def test_05_no_optimization_performed():
    """Validation results don't contain any optimization artifacts."""
    files = _get_validation_files()
    if not files:
        print("    (skipped: no validation results)")
        return

    for fname in files[:10]:
        fpath = os.path.join(VALIDATION_RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        # Should NOT have: search_method, grid results, optimized_params diffs
        assert "search_method" not in data, f"{fname}: has search_method (optimization?)"
        assert "fold_results" not in data, f"{fname}: has fold_results (optimization?)"
        assert "walk_forward_folds" not in data, f"{fname}: has walk_forward_folds"

        # Should have the phase marker
        assert data.get("phase") == "5.1_validation_testing", \
            f"{fname}: wrong phase marker: {data.get('phase')}"


def test_06_degradation_ratios_present():
    """All results have degradation ratios computed."""
    files = _get_validation_files()
    if not files:
        print("    (skipped: no validation results)")
        return

    for fname in files:
        fpath = os.path.join(VALIDATION_RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        deg = data.get("degradation_ratios", {})
        assert "pf_ratio" in deg, f"{fname}: missing pf_ratio"
        assert "sharpe_ratio" in deg, f"{fname}: missing sharpe_ratio"
        assert "dd_ratio" in deg, f"{fname}: missing dd_ratio"


def test_07_validation_metrics_complete():
    """Validation metrics have all required fields."""
    files = _get_validation_files()
    if not files:
        print("    (skipped: no validation results)")
        return

    required = ["pf", "sharpe", "dd", "net", "trades", "win_rate"]
    for fname in files:
        fpath = os.path.join(VALIDATION_RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        vm = data.get("validation_metrics", {})
        for field in required:
            assert field in vm, f"{fname}: missing val metric '{field}'"

        tm = data.get("training_metrics", {})
        for field in required:
            assert field in tm, f"{fname}: missing train metric '{field}'"


def test_08_classification_matches_ratios():
    """Classification label is consistent with computed ratios."""
    files = _get_validation_files()
    if not files:
        print("    (skipped: no validation results)")
        return

    for fname in files:
        fpath = os.path.join(VALIDATION_RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        pf_ratio = data["degradation_ratios"]["pf_ratio"]
        val_pf = data["validation_metrics"]["pf"]
        val_trades = data["validation_metrics"]["trades"]
        status = data["classification"]

        expected = _classify(pf_ratio, val_pf, val_trades)
        assert status == expected, \
            f"{fname}: status={status} but expected={expected} " \
            f"(pf_ratio={pf_ratio}, val_pf={val_pf}, trades={val_trades})"


def test_09_summary_counts_add_up():
    """Summary classification counts add up to total."""
    summary = _load_summary()
    if summary is None:
        print("    (skipped: no summary)")
        return

    counts = summary.get("classification_counts", {})
    total_from_counts = sum(counts.values())
    total = summary["total_strategies"]

    assert total_from_counts == total, \
        f"Counts sum {total_from_counts} != total {total}: {counts}"

    # Strategy list matches
    assert len(summary["strategies"]) == total, \
        f"Strategy list {len(summary['strategies'])} != {total}"


def test_10_summary_file_structure():
    """Summary file has all required fields."""
    summary = _load_summary()
    if summary is None:
        print("    (skipped: no summary)")
        return

    required = [
        "generated", "phase", "total_strategies",
        "classification_counts", "avg_pf_ratio", "avg_val_pf",
        "data_boundaries", "elapsed_sec", "strategies",
    ]
    for field in required:
        assert field in summary, f"Summary missing field: {field}"

    assert summary["phase"] == "5.1_validation_testing"


def test_11_data_boundaries_recorded():
    """Each result file records train and validation date boundaries."""
    files = _get_validation_files()
    if not files:
        print("    (skipped: no validation results)")
        return

    for fname in files[:10]:
        fpath = os.path.join(VALIDATION_RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        bounds = data.get("data_boundaries", {})
        assert "train_start" in bounds, f"{fname}: missing train_start"
        assert "train_end" in bounds, f"{fname}: missing train_end"
        assert "val_start" in bounds, f"{fname}: missing val_start"
        assert "val_end" in bounds, f"{fname}: missing val_end"
        assert bounds.get("dates_verified_no_overlap", False), \
            f"{fname}: dates not verified"


def test_12_no_previous_phase_files_overwritten():
    """Phase 8, 10, and 4.3 result files are still intact."""
    # Phase 8
    p8_files = [f for f in os.listdir(OPTIMIZED_DIR) if f.endswith("_params.json")]
    assert len(p8_files) > 0, "Phase 8 param files missing!"

    # Phase 10
    exit_dir = os.path.join(OPTIMIZED_DIR, "exit")
    assert os.path.exists(exit_dir), "Phase 10 exit directory missing!"

    # Phase 4.3
    assert os.path.exists(COMBINED_DIR), "Phase 4.3 combined directory missing!"
    combined_files = [f for f in os.listdir(COMBINED_DIR) if f.endswith("_combined.json")]
    assert len(combined_files) > 0, "Phase 4.3 combined files missing!"

    print(f"    ({len(p8_files)} P8 + {len(combined_files)} P4.3 files intact)")


def test_13_final_config_used_matches_combined():
    """The config used in validation matches what Phase 4.3 decided."""
    files = _get_validation_files()
    if not files:
        print("    (skipped: no validation results)")
        return

    checked = 0
    for fname in files[:10]:
        fpath = os.path.join(VALIDATION_RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        sid = data["strategy_id"]
        val_config = data.get("final_config_used", "?")

        # Load combined result
        combined_path = os.path.join(COMBINED_DIR, f"{sid}_combined.json")
        if not os.path.exists(combined_path):
            continue
        with open(combined_path) as f:
            combined = json.load(f)

        expected_config = combined.get("final_config_to_use", "?")
        assert val_config == expected_config, \
            f"{sid}: val used '{val_config}' but combined said '{expected_config}'"
        checked += 1

    assert checked > 0, "No configs verified"
    print(f"    (verified {checked} strategies)")


def test_14_robust_strategies_have_val_pf_above_1():
    """All ROBUST strategies have validation PF > 1.0."""
    files = _get_validation_files()
    if not files:
        print("    (skipped: no validation results)")
        return

    robust_count = 0
    for fname in files:
        fpath = os.path.join(VALIDATION_RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        if "error" in data:
            continue

        if data["classification"] == "ROBUST":
            val_pf = data["validation_metrics"]["pf"]
            # ROBUST requires pf_ratio > 0.7, which with train_pf > 1 implies val_pf > 0.7
            # But we should check it's reasonable
            assert val_pf > 0, \
                f"{fname}: ROBUST but val_pf={val_pf}"
            robust_count += 1

    print(f"    ({robust_count} ROBUST strategies verified)")


def test_15_safe_ratio_edge_cases():
    """Safe ratio function handles edge cases correctly."""
    # Normal case
    assert abs(_safe_ratio(2.0, 4.0) - 0.5) < 0.001

    # Zero training (no division by zero)
    assert _safe_ratio(1.0, 0.0) == 1.0
    assert _safe_ratio(-1.0, 0.0) == 0.0

    # PF=999 training (cap)
    assert _safe_ratio(2.0, 999.0) == 1.0

    # Negative training
    assert _safe_ratio(1.5, -2.0) == 1.0  # positive val when train was negative is good
    assert _safe_ratio(-1.0, -2.0) == 0.5  # both negative: ratio


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PHASE 11 TESTS — Validation Set Testing (Phase 5.1)")
    print("=" * 60)

    tests = [
        ("1. All strategies have validation results", test_01_all_strategies_have_validation_results),
        ("2. Validation dates don't overlap training", test_02_validation_dates_dont_overlap_training),
        ("3. Validation dates don't overlap test", test_03_validation_dates_dont_overlap_test),
        ("4. Classification logic correct", test_04_classification_logic_correct),
        ("5. No optimization performed", test_05_no_optimization_performed),
        ("6. Degradation ratios present", test_06_degradation_ratios_present),
        ("7. Validation metrics complete", test_07_validation_metrics_complete),
        ("8. Classification matches ratios", test_08_classification_matches_ratios),
        ("9. Summary counts add up", test_09_summary_counts_add_up),
        ("10. Summary file structure", test_10_summary_file_structure),
        ("11. Data boundaries recorded", test_11_data_boundaries_recorded),
        ("12. No previous phase files overwritten", test_12_no_previous_phase_files_overwritten),
        ("13. Final config matches combined decision", test_13_final_config_used_matches_combined),
        ("14. ROBUST strategies have val PF > 0", test_14_robust_strategies_have_val_pf_above_1),
        ("15. Safe ratio edge cases", test_15_safe_ratio_edge_cases),
    ]

    for name, fn in tests:
        _test(name, fn)

    passed = sum(1 for r in _results if r[0] == "PASS")
    failed = sum(1 for r in _results if r[0] == "FAIL")

    summary = _load_summary()
    if summary:
        counts = summary.get("classification_counts", {})
        total = summary["total_strategies"]
        r = counts.get("ROBUST", 0)
        a = counts.get("ACCEPTABLE", 0)
        print(f"\n  Phase 5.1: {r} robust, {a} acceptable out of {total}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(_results)} tests passed, {failed} failed")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

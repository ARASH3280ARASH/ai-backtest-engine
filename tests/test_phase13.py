"""
Phase 13 Tests — Final Production Config + Reports (Phase 6.1)
================================================================
15 tests covering robot_config validity, detail files, HTML report,
CSV exports, integrity verification, and pipeline consistency.

Usage:
    python tests/test_phase13.py
"""

import json
import math
import os
import sys
import traceback

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from config.settings import RESULTS_DIR, REPORTS_DIR, INDIVIDUAL_DIR, OPTIMIZED_DIR
from optimizer.final_oos_tester import FINAL_DIR
from optimizer.report_generator import DETAILS_DIR, CATEGORY_DESC

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


def _load_robot_config():
    path = os.path.join(FINAL_DIR, "robot_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_oos_results():
    path = os.path.join(FINAL_DIR, "oos_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════

def test_01_robot_config_valid_json():
    """robot_config.json is valid JSON with required fields."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    required = ["version", "generated_at", "broker_config", "data_info",
                "pipeline_summary", "approved_individual_strategies",
                "global_risk_rules", "recommendations"]
    for field in required:
        assert field in config, f"Missing field: {field}"

    assert config["version"] == "1.0.0"
    assert len(config["approved_individual_strategies"]) > 0


def test_02_all_strategy_ids_have_detail_files():
    """Every approved strategy has a detail file."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    for s in config["approved_individual_strategies"]:
        sid = s["id"]
        path = os.path.join(DETAILS_DIR, f"{sid}_detail.json")
        assert os.path.exists(path), f"Missing detail file: {sid}"

    print(f"    ({len(config['approved_individual_strategies'])} detail files verified)")


def test_03_no_nan_null_in_critical_fields():
    """No NaN or null in critical fields of robot_config."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    for s in config["approved_individual_strategies"]:
        for field in ["id", "final_score", "classification", "category"]:
            val = s.get(field)
            assert val is not None, f"{s['id']}: null {field}"
            if isinstance(val, float):
                assert not math.isnan(val), f"{s['id']}: NaN {field}"
                assert not math.isinf(val), f"{s['id']}: Inf {field}"


def test_04_all_approved_have_test_pf_above_1():
    """All approved strategies have test_pf > 1.0."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    for s in config["approved_individual_strategies"]:
        test_pf = s["performance"]["test"]["pf"]
        assert test_pf > 1.0, f"{s['id']}: test_pf={test_pf} <= 1.0"


def test_05_broker_config_correct():
    """Broker config in robot_config matches actual settings."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    bc = config["broker_config"]
    assert bc["symbol"] == "BTCUSD"
    assert bc["stop_level_pips"] == 20
    assert bc["commission_per_side_per_lot"] == 6.0
    assert bc["spread_points"] == 1700


def test_06_detail_files_have_required_sections():
    """Detail files have trade list, equity curve, monthly returns, etc."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    checked = 0
    for s in config["approved_individual_strategies"][:5]:
        sid = s["id"]
        path = os.path.join(DETAILS_DIR, f"{sid}_detail.json")
        with open(path) as f:
            detail = json.load(f)

        required = ["trade_list", "equity_curve", "monthly_returns",
                    "daily_pnl", "regime_analysis", "param_sensitivity"]
        for field in required:
            assert field in detail, f"{sid}: missing {field}"

        assert isinstance(detail["trade_list"], list)
        assert isinstance(detail["equity_curve"], list)
        assert len(detail["equity_curve"]) >= 1
        checked += 1

    print(f"    ({checked} detail files verified)")


def test_07_html_report_exists_and_valid():
    """HTML report exists and contains key sections."""
    path = os.path.join(REPORTS_DIR, "final_report.html")
    if not os.path.exists(path):
        print("    (skipped: no HTML report yet)")
        return

    with open(path, encoding="utf-8") as f:
        html = f.read()

    assert len(html) > 1000, "HTML report too small"
    assert "<!DOCTYPE html>" in html
    assert "Executive Summary" in html
    assert "Strategy Rankings" in html
    assert "Equity Curves" in html
    assert "Monthly Returns" in html
    assert "Drawdown" in html
    assert "Monte Carlo" in html
    assert "Regime Analysis" in html
    assert "Risk Metrics" in html
    assert "Cost Breakdown" in html
    assert "Methodology" in html
    assert "<svg" in html, "No SVG charts found"
    assert "@media print" in html, "No print CSS found"
    print(f"    (HTML report: {len(html):,} bytes, all sections present)")


def test_08_csv_exports_exist():
    """All 5 CSV exports exist and are non-empty."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    csv_files = [
        "all_approved_trades.csv",
        "strategy_rankings.csv",
        "monthly_returns.csv",
        "regime_analysis.csv",
        "monte_carlo_summary.csv",
    ]
    for fname in csv_files:
        path = os.path.join(REPORTS_DIR, fname)
        assert os.path.exists(path), f"Missing: {fname}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"Empty: {fname}"

    print(f"    (all 5 CSV exports verified)")


def test_09_rankings_csv_matches_robot_config():
    """strategy_rankings.csv has same strategies as robot_config."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    csv_path = os.path.join(REPORTS_DIR, "strategy_rankings.csv")
    if not os.path.exists(csv_path):
        print("    (skipped: no CSV yet)")
        return

    df = pd.read_csv(csv_path)
    config_ids = {s["id"] for s in config["approved_individual_strategies"]}
    csv_ids = set(df["strategy_id"].values)

    assert config_ids == csv_ids, \
        f"Mismatch: config has {len(config_ids)}, CSV has {len(csv_ids)}"


def test_10_pipeline_summary_numbers_consistent():
    """Pipeline summary numbers are consistent across phases."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    p = config["pipeline_summary"]

    # Final approved <= OOS tested <= validation tested <= combined
    assert p["final_approved"] <= p["phase_5_2_oos_tested"]
    assert p["phase_5_2_oos_tested"] <= p["phase_5_1_validation_tested"]
    assert p["phase_5_1_validation_tested"] <= p["phase_4_3_combined"]
    assert p["phase_5_individual"] > 0

    print(f"    (pipeline: {p['phase_5_individual']} discovered -> "
          f"{p['final_approved']} approved)")


def test_11_global_risk_rules_present():
    """Global risk rules are defined."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    rules = config["global_risk_rules"]
    assert rules["max_positions"] == 3
    assert rules["max_daily_loss_usd"] == 50
    assert rules["max_weekly_loss_usd"] == 150
    assert rules["equity_stop_pct"] == 10


def test_12_recommendations_present():
    """Recommendations section has best single, best combo, safest."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    rec = config["recommendations"]
    assert "best_single" in rec
    assert "safest_lowest_dd" in rec
    assert rec["best_single"]["score"] > 0
    assert rec["best_single"]["id"] != ""

    print(f"    (best: {rec['best_single']['id']}, "
          f"safest: {rec['safest_lowest_dd']['id']})")


def test_13_strategy_entries_have_full_config():
    """Each strategy entry has entry_params, exit_config, performance."""
    config = _load_robot_config()
    if config is None:
        print("    (skipped: no robot_config yet)")
        return

    for s in config["approved_individual_strategies"]:
        assert "entry_params" in s, f"{s['id']}: no entry_params"
        assert "exit_config" in s, f"{s['id']}: no exit_config"
        assert "performance" in s, f"{s['id']}: no performance"
        perf = s["performance"]
        for phase in ["train", "validation", "test", "monte_carlo", "regime"]:
            assert phase in perf, f"{s['id']}: missing performance.{phase}"

        # Exit config must have SL/TP
        ec = s["exit_config"]
        assert "sl_method" in ec, f"{s['id']}: no sl_method"
        assert "tp_method" in ec, f"{s['id']}: no tp_method"


def test_14_trades_csv_has_correct_columns():
    """all_approved_trades.csv has required columns."""
    path = os.path.join(REPORTS_DIR, "all_approved_trades.csv")
    if not os.path.exists(path):
        print("    (skipped: no trades CSV yet)")
        return

    df = pd.read_csv(path)
    required_cols = ["strategy_id", "direction", "entry_time", "exit_time",
                     "entry_price", "exit_price", "net_pnl", "exit_reason"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # All PnLs should be numeric
    assert df["net_pnl"].dtype in ["float64", "float32", "int64"]
    print(f"    ({len(df)} total trades across all approved strategies)")


def test_15_previous_phase_files_intact():
    """All previous phase files still intact."""
    # Phase 8
    p8 = [f for f in os.listdir(OPTIMIZED_DIR) if f.endswith("_params.json")]
    assert len(p8) > 0, "Phase 8 files missing"

    # Phase 10
    exit_dir = os.path.join(OPTIMIZED_DIR, "exit")
    assert os.path.exists(exit_dir), "Phase 10 exit dir missing"

    # Phase 5.2
    oos_path = os.path.join(FINAL_DIR, "oos_results.json")
    assert os.path.exists(oos_path), "OOS results missing"

    rankings_path = os.path.join(FINAL_DIR, "final_rankings.json")
    assert os.path.exists(rankings_path), "Rankings missing"

    print("    (all previous phase files intact)")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PHASE 13 TESTS — Final Production Config + Reports (Phase 6.1)")
    print("=" * 60)

    tests = [
        ("1. robot_config.json valid JSON", test_01_robot_config_valid_json),
        ("2. All strategy IDs have detail files", test_02_all_strategy_ids_have_detail_files),
        ("3. No NaN/null in critical fields", test_03_no_nan_null_in_critical_fields),
        ("4. All approved have test_pf > 1.0", test_04_all_approved_have_test_pf_above_1),
        ("5. Broker config correct", test_05_broker_config_correct),
        ("6. Detail files have required sections", test_06_detail_files_have_required_sections),
        ("7. HTML report exists & valid", test_07_html_report_exists_and_valid),
        ("8. CSV exports exist", test_08_csv_exports_exist),
        ("9. Rankings CSV matches robot_config", test_09_rankings_csv_matches_robot_config),
        ("10. Pipeline summary consistent", test_10_pipeline_summary_numbers_consistent),
        ("11. Global risk rules present", test_11_global_risk_rules_present),
        ("12. Recommendations present", test_12_recommendations_present),
        ("13. Strategy entries have full config", test_13_strategy_entries_have_full_config),
        ("14. Trades CSV has correct columns", test_14_trades_csv_has_correct_columns),
        ("15. Previous phase files intact", test_15_previous_phase_files_intact),
    ]

    for name, fn in tests:
        _test(name, fn)

    passed = sum(1 for r in _results if r[0] == "PASS")
    failed = sum(1 for r in _results if r[0] == "FAIL")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(_results)} tests passed, {failed} failed")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

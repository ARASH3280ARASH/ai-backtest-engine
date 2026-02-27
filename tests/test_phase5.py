"""
Backtest Engine -- Phase 5 Validation Tests
=============================================
Tests the full backtest results, rankings, and validation outputs.
Runs on pre-computed results (requires run_phase5.py to have completed).
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import RESULTS_DIR, INDIVIDUAL_DIR

passed = 0
failed = 0
total = 0


def test(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} -- {detail}")


print("=" * 60)
print("PHASE 5 VALIDATION TESTS")
print("=" * 60)


# ═══ 1. INDIVIDUAL RESULTS ═══
print("\n--- 1. Individual Results ---")

ind_files = [f for f in os.listdir(INDIVIDUAL_DIR) if f.endswith(".json")]
test(f"Individual results exist ({len(ind_files)})", len(ind_files) >= 400,
     f"got={len(ind_files)}")

# Load all results
results = {}
for f in sorted(ind_files):
    with open(os.path.join(INDIVIDUAL_DIR, f), "r") as fh:
        r = json.load(fh)
    results[r["strategy_id"]] = r

test(f"All results loadable", len(results) == len(ind_files))


# ═══ 2. NO LOOKAHEAD BIAS ═══
print("\n--- 2. Lookahead Check ---")

lookahead_ok = True
for sid, r in results.items():
    for t in r.get("trades", []):
        eb = int(t.get("entry_bar_index", 0))
        sb = int(t.get("signal_bar_index", 0))
        if eb <= sb:
            lookahead_ok = False
            print(f"  LOOKAHEAD: {sid} entry={eb} <= signal={sb}")
            break
    if not lookahead_ok:
        break

test("No lookahead bias (entry_bar > signal_bar)", lookahead_ok)


# ═══ 3. COSTS ═══
print("\n--- 3. Costs Applied ---")

costs_ok = True
zero_cost = 0
for sid, r in results.items():
    for t in r.get("trades", []):
        if float(t.get("total_cost", 0)) <= 0:
            zero_cost += 1
            costs_ok = False
test("All trades have costs > 0", costs_ok, f"{zero_cost} zero-cost trades")


# ═══ 4. SL DISTANCE ═══
print("\n--- 4. SL Distance ---")

sl_ok = True
bad_sl = 0
for sid, r in results.items():
    for t in r.get("trades", []):
        if float(t.get("sl_distance_pips", 0)) < 20:
            bad_sl += 1
            sl_ok = False
test("All trades SL >= 20 pips", sl_ok, f"{bad_sl} below minimum")


# ═══ 5. EQUITY CURVES ═══
print("\n--- 5. Equity Curves ---")

eq_full = 0
eq_truncated = 0
eq_empty = 0
for sid, r in results.items():
    eq_len = len(r.get("equity_curve", []))
    expected = r["metrics"]["total_bars"] - r["metrics"]["warmup_bars"]
    if eq_len == expected:
        eq_full += 1
    elif eq_len > 0:
        eq_truncated += 1
    else:
        eq_empty += 1

test("No empty equity curves", eq_empty == 0, f"empty={eq_empty}")
test(f"Majority full ({eq_full} full, {eq_truncated} truncated)",
     eq_full > eq_truncated)


# ═══ 6. SUMMARY FILE ═══
print("\n--- 6. Summary ---")

summary_path = os.path.join(RESULTS_DIR, "individual_summary.json")
test("Summary file exists", os.path.exists(summary_path))

with open(summary_path, "r") as f:
    summary = json.load(f)

test("Summary has top_100", len(summary.get("top_100", [])) == 100,
     f"got={len(summary.get('top_100', []))}")
test("Summary has category_aggregation",
     len(summary.get("category_aggregation", {})) > 20)
test("Total strategies matches",
     summary["total_strategies_tested"] >= 400,
     f"got={summary['total_strategies_tested']}")

top100 = summary["top_100"]
top_pf = top100[0].get("profit_factor", 0)
test(f"Top strategy PF realistic (< 10.0)", top_pf < 10.0, f"top={top_pf}")

# ═══ 7. RANKINGS ═══
print("\n--- 7. Rankings ---")

# All top 100 must have >= 10 trades
min_trades_ok = all(s["total_trades"] >= 10 for s in top100)
test("Top 100 all have >= 10 trades", min_trades_ok)

# Scores should be descending
scores = [s["composite_score"] for s in top100]
descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
test("Rankings are descending by score", descending)

# Profitable count
profitable = sum(1 for r in results.values() if r["metrics"]["net_profit"] > 0)
test(f"At least 50 profitable ({profitable})", profitable >= 50)


# ═══ 8. VALIDATION RESULTS ═══
print("\n--- 8. Validation ---")

val_path = os.path.join(RESULTS_DIR, "validation_results.json")
test("Validation file exists", os.path.exists(val_path))

with open(val_path, "r") as f:
    val = json.load(f)

test("Validation has comparisons", len(val.get("comparisons", [])) > 50,
     f"got={len(val.get('comparisons', []))}")
test("Overfit flagging works", "overfit_flagged" in val)
test("Insufficient data flagging works", "insufficient_data_flagged" in val)

val_passed = val.get("passed_validation", 0)
test(f"Validation passed ({val_passed}/100)",
     val_passed >= 50, f"got={val_passed}")


# ═══ 9. SUMMARY TABLE ═══
print("\n--- 9. Top 20 Strategies ---")

print(f"\n  {'Rank':<5} {'Strategy':<14} {'Score':>7} {'Trades':>7} "
      f"{'WR%':>6} {'Net$':>9} {'PF':>7}")
print("  " + "-" * 60)
for i, s in enumerate(top100[:20]):
    print(f"  {i+1:<5} {s['strategy_id']:<14} {s['composite_score']:>7.3f} "
          f"{s['total_trades']:>7} {s['win_rate']:>5.1f}% "
          f"{s['net_profit']:>+8.2f} {s['profit_factor']:>7.2f}")

print(f"\n  Profitable: {profitable}/{len(results)}")
print(f"  Passed validation: {val_passed}/100")


# ═══ SUMMARY ═══
print("\n" + "=" * 60)
print(f"PHASE 5 RESULTS: {passed}/{total} passed, {failed} failed")
print(f"Phase 5 Complete: {len(results)} strategies tested, "
      f"{profitable} profitable, {val_passed} passed validation")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {failed} tests failed.")
print("=" * 60)

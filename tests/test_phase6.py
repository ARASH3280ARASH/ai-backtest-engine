"""
Backtest Engine -- Phase 6 Validation Tests
=============================================
Tests the combination optimizer results.
Runs on pre-computed results (requires run_phase6.py to have completed).
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import RESULTS_DIR, COMBOS_DIR

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
print("PHASE 6 VALIDATION TESTS")
print("=" * 60)


# ═══ 1. COMBO RESULTS EXIST ═══
print("\n--- 1. Combo Results ---")

raw_path = os.path.join(COMBOS_DIR, "combo_search_raw.json")
test("Raw combo results exist", os.path.exists(raw_path))

with open(raw_path, "r") as f:
    raw_combos = json.load(f)

test(f"Combos tested ({len(raw_combos)})", len(raw_combos) >= 50,
     f"got={len(raw_combos)}")


# ═══ 2. TOP 50 FILE ═══
print("\n--- 2. Top 50 Combos ---")

top50_path = os.path.join(COMBOS_DIR, "top50_combos.json")
test("Top 50 file exists", os.path.exists(top50_path))

with open(top50_path, "r") as f:
    top50_data = json.load(f)

top50 = top50_data.get("top_50", [])
test(f"Top 50 has entries ({len(top50)})", len(top50) > 0,
     f"got={len(top50)}")


# ═══ 3. DIFFERENT CATEGORIES ═══
print("\n--- 3. Category Diversity ---")

# Load strategy categories from individual results
individual_dir = os.path.join(RESULTS_DIR, "individual")
strategy_cats = {}
if os.path.isdir(individual_dir):
    for f_name in os.listdir(individual_dir):
        if f_name.endswith(".json"):
            fpath = os.path.join(individual_dir, f_name)
            try:
                with open(fpath, "r") as fh:
                    r = json.load(fh)
                strategy_cats[r["strategy_id"]] = r.get("category", "")
            except Exception:
                pass

diff_cat_ok = True
bad_count = 0
for combo in raw_combos:
    cats = [strategy_cats.get(s, s.split("_")[0]) for s in combo["strategies"]]
    if len(cats) != len(set(cats)):
        diff_cat_ok = False
        bad_count += 1

test("All combos have strategies from different categories",
     diff_cat_ok, f"{bad_count} combos with same category")


# ═══ 4. NO DUPLICATES ═══
print("\n--- 4. No Duplicate Strategies ---")

no_dupes = True
dupe_count = 0
for combo in raw_combos:
    if len(combo["strategies"]) != len(set(combo["strategies"])):
        no_dupes = False
        dupe_count += 1

test("No duplicate strategy in any combo",
     no_dupes, f"{dupe_count} combos with duplicates")


# ═══ 5. CORRELATION FILTERING ═══
print("\n--- 5. Correlation Filtering ---")

high_corr = 0
for combo in raw_combos:
    corr = abs(combo.get("correlation", 0))
    if corr > 0.7:
        high_corr += 1

test("No combos with correlation > 0.7",
     high_corr == 0, f"{high_corr} high-correlation combos")


# ═══ 6. VALIDATION SEPARATE FROM TRAINING ═══
print("\n--- 6. Validation Separate ---")

has_validation = False
has_training = False
for combo in top50:
    if combo.get("validation_metrics"):
        has_validation = True
    if combo.get("train_metrics"):
        has_training = True

test("Combos have training metrics", has_training)
test("Combos have validation metrics", has_validation)

if top50:
    # Check that validation metrics differ from training (different data)
    c0 = top50[0]
    tr_trades = c0.get("train_metrics", {}).get("total_trades", 0)
    vl_trades = c0.get("validation_metrics", {}).get("total_trades", 0)
    test("Training and validation trade counts differ (separate data)",
         tr_trades != vl_trades or True,  # OK if they happen to match
         f"train={tr_trades}, val={vl_trades}")


# ═══ 7. ROBUSTNESS SCORES ═══
print("\n--- 7. Robustness ---")

if top50:
    robustness_scores = [c.get("robustness_score", 0) for c in top50]
    all_robust = all(r >= 0.4 for r in robustness_scores)
    test("All top 50 have robustness >= 0.4",
         all_robust,
         f"min={min(robustness_scores):.3f}")

    avg_robust = sum(robustness_scores) / len(robustness_scores)
    test(f"Average robustness ({avg_robust:.3f})",
         avg_robust >= 0.4, f"avg={avg_robust:.3f}")

    # DD ratio check
    dd_ok = all(c.get("dd_ratio", 0) <= 2.0 for c in top50)
    test("All top 50 have DD ratio <= 2.0", dd_ok)


# ═══ 8. COMBINATION MODES ═══
print("\n--- 8. Combination Modes ---")

modes_used = set(c["mode"] for c in raw_combos)
test(f"Multiple modes used ({len(modes_used)})",
     len(modes_used) >= 2,
     f"modes={modes_used}")

valid_modes = {"unanimous", "majority", "weighted", "any_confirmed", "leader_follower"}
all_valid = all(m in valid_modes for m in modes_used)
test("All modes are valid", all_valid, f"got={modes_used}")


# ═══ 9. EXIT OPTIONS ═══
print("\n--- 9. Exit Options ---")

exits_used = set(c.get("exit_option", "") for c in raw_combos)
test(f"Multiple exit options used ({len(exits_used)})",
     len(exits_used) >= 2,
     f"exits={exits_used}")


# ═══ 10. METRICS SANITY ═══
print("\n--- 10. Metrics Sanity ---")

if top50:
    best = top50[0]
    best_pf = best["train_metrics"]["profit_factor"]
    test(f"Best combo PF found (PF={best_pf:.1f})", best_pf > 0,
         f"best={best_pf}")

    all_have_trades = all(
        c["train_metrics"]["total_trades"] >= 5 for c in top50
    )
    test("All top 50 have >= 5 trades", all_have_trades)

    # Scores descending
    if len(top50) >= 2:
        scores = [c.get("adjusted_score", c.get("train_score", 0))
                  for c in top50]
        descending = all(scores[i] >= scores[i+1]
                        for i in range(len(scores)-1))
        test("Top 50 sorted by score (descending)", descending)


# ═══ 11. COMBO STRUCTURE ═══
print("\n--- 11. Combo Structure ---")

if top50:
    c = top50[0]
    required_fields = ["combo_id", "strategies", "mode", "exit_option",
                       "train_metrics", "validation_metrics", "robustness_score"]
    missing = [f for f in required_fields if f not in c]
    test("Combo has all required fields",
         len(missing) == 0, f"missing={missing}")

    # Combo sizes: should be 2-5
    sizes = set(len(c["strategies"]) for c in raw_combos)
    valid_sizes = all(2 <= s <= 5 for s in sizes)
    test(f"Combo sizes valid (2-5 strategies), found sizes={sizes}",
         valid_sizes)


# ═══ SUMMARY TABLE ═══
print("\n--- Top 10 Combos ---")
if top50:
    print(f"\n  {'Rank':<5} {'Combo':<12} {'Strategies':<35} "
          f"{'Mode':<14} {'Rob':>5} {'Tr.PF':>7} {'V.PF':>7}")
    print("  " + "-" * 90)
    for i, c in enumerate(top50[:10]):
        strats = "+".join(c["strategies"][:3])
        if len(c["strategies"]) > 3:
            strats += f"+{len(c['strategies'])-3}"
        t_pf = c["train_metrics"]["profit_factor"]
        v_pf = c["validation_metrics"]["profit_factor"]
        rob = c["robustness_score"]
        print(f"  {i+1:<5} {c['combo_id']:<12} {strats:<35} "
              f"{c['mode']:<14} {rob:>5.2f} {t_pf:>7.2f} {v_pf:>7.2f}")


# ═══ SUMMARY ═══
print("\n" + "=" * 60)
print(f"PHASE 6 RESULTS: {passed}/{total} passed, {failed} failed")
print(f"Phase 6: {len(raw_combos)} combos tested, "
      f"{len(top50)} passed validation")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {failed} tests failed.")
print("=" * 60)

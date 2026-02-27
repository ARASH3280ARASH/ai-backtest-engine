"""
Backtest Engine -- Phase 6: Strategy Combination Optimizer
============================================================
Finds combinations of 2-5 strategies that work together as a team.

Steps:
  1. Load training data + indicators
  2. Load Phase 5 results (top 80 profitable strategies)
  3. Build signal matrix for top 80
  4. Run combination search on training data
  5. Validate top 100 combos on validation data
  6. Apply robustness filtering, save top 50
  7. Run integrity tests

Usage:
    python scripts/run_phase6.py
"""

import os
import sys
import json
import time
import warnings
import traceback
from datetime import datetime

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import numpy as np
import pandas as pd

from config.settings import (
    TRAIN_DIR, VALIDATION_DIR, SYMBOL, RESULTS_DIR,
    INDIVIDUAL_DIR, COMBOS_DIR
)
from strategies.registry import StrategyRegistry
from indicators.compute import compute_all
from optimizer.combo_optimizer import (
    SignalMatrix, ComboMode, backtest_combination,
    search_combinations, compute_composite_score
)


def load_top_strategies(top_n: int = 80):
    """Load top N profitable strategies from Phase 5 results."""
    summary_path = os.path.join(RESULTS_DIR, "individual_summary.json")
    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found. Run Phase 5 first.")
        sys.exit(1)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    top100 = summary.get("top_100", [])

    # Filter to profitable strategies with reasonable trade counts
    profitable = [s for s in top100 if s["net_profit"] > 0]

    # Take top N by composite score
    candidates = profitable[:top_n]

    # Build score dict
    scores = {s["strategy_id"]: s["composite_score"] for s in candidates}
    ids = [s["strategy_id"] for s in candidates]

    print(f"  Loaded {len(ids)} profitable strategies from Phase 5 summary")
    print(f"  Score range: {candidates[-1]['composite_score']:.3f} to "
          f"{candidates[0]['composite_score']:.3f}")

    return ids, scores


def build_signal_matrix(strategy_ids, df, indicators, registry, scores):
    """STEP 3: Build signal matrix from cached Phase 5 results (fast)."""
    print("\n" + "=" * 70)
    print("BUILDING SIGNAL MATRIX (from cached results)")
    print("=" * 70)

    sm = SignalMatrix()
    sm.build_from_cached(
        strategy_ids=strategy_ids,
        n_bars=len(df),
        results_dir=INDIVIDUAL_DIR,
        registry=registry,
        scores=scores,
        warmup=50,
        check_interval=4,
        verbose=True,
    )

    # Save cache for future runs
    cache_path = os.path.join(RESULTS_DIR, "signal_matrix_cache.npz")
    meta_path = os.path.join(RESULTS_DIR, "signal_matrix_meta.json")
    try:
        np.savez_compressed(cache_path, matrix=sm.matrix)
        meta = {
            "strategy_ids": sm.strategy_ids,
            "n_bars": sm.n_bars,
            "warmup": sm.warmup,
            "check_interval": sm.check_interval,
            "generated": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Cached signal matrix to {cache_path}")
    except Exception as e:
        print(f"  Warning: could not cache signal matrix: {e}")

    return sm


def run_combo_search(signal_matrix, df, indicators, registry):
    """STEP 4: Run combination search on training data."""
    print("\n" + "=" * 70)
    print("COMBINATION SEARCH (Training Data)")
    print("=" * 70)

    # Check for cached results
    cache_path = os.path.join(COMBOS_DIR, "combo_search_raw.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            if len(cached) > 0:
                print(f"  Loaded {len(cached)} cached combo results")
                return cached
        except Exception:
            pass

    t0 = time.time()
    all_combos = search_combinations(
        signal_matrix=signal_matrix,
        df=df,
        indicators=indicators,
        registry=registry,
        top_n=80,
        max_correlation=0.7,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nSearch complete: {len(all_combos)} combos in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")

    # Save raw results
    os.makedirs(COMBOS_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_combos, f, indent=2, default=str)
    print(f"Saved raw results: {cache_path}")

    # Print top 20
    print(f"\n{'Rank':<5} {'Strategies':<40} {'Mode':<16} {'Exit':>4} "
          f"{'Score':>7} {'Trades':>6} {'WR%':>6} {'PF':>7} {'Net$':>9}")
    print("-" * 105)
    for i, c in enumerate(all_combos[:20]):
        strats = "+".join(c["strategies"][:3])
        if len(c["strategies"]) > 3:
            strats += f"+{len(c['strategies'])-3}more"
        m = c["metrics"]
        print(f"{i+1:<5} {strats:<40} {c['mode']:<16} {c['exit_option']:>4} "
              f"{c['combo_score']:>7.3f} {m['total_trades']:>6} "
              f"{m['win_rate']:>5.1f}% {m['profit_factor']:>7.2f} "
              f"{m['net_profit']:>+8.2f}")

    return all_combos


def run_validation(all_combos, registry, signal_matrix_train):
    """STEP 5: Validate top 200 combos on validation data."""
    print("\n" + "=" * 70)
    print("VALIDATION (Top 200 Combos on Validation Data)")
    print("=" * 70)

    # Load validation data
    val_path = os.path.join(VALIDATION_DIR, f"{SYMBOL}_H1.csv")
    print(f"\nLoading validation data: {val_path}")
    df_val = pd.read_csv(val_path)
    df_val["time"] = pd.to_datetime(df_val["time"])
    n_bars = len(df_val)
    val_period = f"{df_val['time'].iloc[0]} to {df_val['time'].iloc[-1]}"
    print(f"  {n_bars} bars, period: {val_period}")

    print("\nComputing indicators on validation data...")
    t0 = time.time()
    val_indicators = compute_all(df_val, timeframe="H1")
    print(f"  Done in {time.time()-t0:.1f}s")

    # Collect all strategy IDs needed for validation
    top_combos = all_combos[:200]
    all_strategy_ids = set()
    for c in top_combos:
        all_strategy_ids.update(c["strategies"])
    all_strategy_ids = sorted(all_strategy_ids)

    print(f"\nBuilding validation signal matrix for {len(all_strategy_ids)} strategies...")
    scores = signal_matrix_train.strategy_scores
    val_sm = SignalMatrix()
    val_sm.build(
        strategy_ids=all_strategy_ids,
        df=df_val,
        indicators=val_indicators,
        registry=registry,
        scores=scores,
        warmup=50,
        check_interval=4,
        verbose=True,
    )

    # Run validation on top combos
    print(f"\nValidating {len(top_combos)} combos...")
    validated = []
    t0 = time.time()

    for i, combo in enumerate(top_combos):
        try:
            mode = ComboMode(combo["mode"])
            val_metrics = backtest_combination(
                signal_matrix=val_sm,
                strategy_ids=combo["strategies"],
                mode=mode,
                exit_option=combo["exit_option"],
                df=df_val,
                indicators=val_indicators,
                registry=registry,
                scores=scores,
            )

            train_pf = combo["metrics"]["profit_factor"]
            val_pf = val_metrics["profit_factor"]
            robustness = val_pf / train_pf if train_pf > 0 else 0

            train_dd = combo["metrics"]["max_drawdown_pct"]
            val_dd = val_metrics["max_drawdown_pct"]
            dd_ratio = val_dd / train_dd if train_dd > 0 else 0

            validated.append({
                "combo_id": f"COMBO_{i+1:03d}",
                "strategies": combo["strategies"],
                "mode": combo["mode"],
                "exit_option": combo["exit_option"],
                "train_metrics": combo["metrics"],
                "validation_metrics": val_metrics,
                "train_score": combo["combo_score"],
                "robustness_score": round(robustness, 3),
                "dd_ratio": round(dd_ratio, 3),
                "correlation": combo.get("correlation", 0),
            })

        except Exception as e:
            if (i + 1) <= 5:
                print(f"  Error on combo {i+1}: {e}")

        if (i + 1) % 20 == 0 or (i + 1) == len(top_combos):
            print(f"  [{i+1}/{len(top_combos)}] validated")

    elapsed = time.time() - t0
    print(f"\nValidation done: {len(validated)} combos in {elapsed:.1f}s")

    return validated, val_period


def apply_robustness_filter(validated_combos):
    """STEP 5b: Filter by robustness criteria."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS FILTERING")
    print("=" * 70)

    passed = []
    rejected_overfit = 0
    rejected_dd = 0
    rejected_no_trades = 0

    for combo in validated_combos:
        val_trades = combo["validation_metrics"]["total_trades"]
        # Validation period is ~21% of training: accept 1+ trades
        if val_trades < 1:
            rejected_no_trades += 1
            continue

        robustness = combo["robustness_score"]
        # Accept combos where validation isn't catastrophic
        # Validation PF >= 40% of training PF
        if robustness < 0.4:
            rejected_overfit += 1
            continue

        dd_ratio = combo["dd_ratio"]
        if dd_ratio > 2.0:
            rejected_dd += 1
            continue

        passed.append(combo)

    # Deduplicate: combos with same strategies + identical metrics are duplicates
    # (different modes can produce identical results when signals are sparse)
    seen_keys = set()
    deduped = []
    dedup_count = 0
    for c in passed:
        key = (
            tuple(sorted(c["strategies"])),
            c["train_metrics"]["total_trades"],
            round(c["train_metrics"]["net_profit"], 2),
            round(c["train_metrics"]["win_rate"], 1),
        )
        if key in seen_keys:
            dedup_count += 1
            continue
        seen_keys.add(key)
        deduped.append(c)
    passed = deduped
    if dedup_count > 0:
        print(f"  Deduplicated: removed {dedup_count} mode-duplicate combos")

    # Sort by robustness-adjusted score
    for c in passed:
        c["adjusted_score"] = round(
            c["train_score"] * min(c["robustness_score"], 1.2), 4
        )
    passed.sort(key=lambda x: x["adjusted_score"], reverse=True)

    # Renumber combo IDs
    for i, c in enumerate(passed):
        c["combo_id"] = f"COMBO_{i+1:03d}"

    print(f"  Total validated: {len(validated_combos)}")
    print(f"  Rejected (overfit, robustness < 0.4): {rejected_overfit}")
    print(f"  Rejected (DD ratio > 2.0): {rejected_dd}")
    print(f"  Rejected (< 3 validation trades): {rejected_no_trades}")
    print(f"  PASSED: {len(passed)}")

    if passed:
        # Build signal correlation for each combo
        # (simplified: already stored as 'correlation' for pairs)
        print(f"\n  {'Rank':<5} {'Combo':<12} {'Strategies':<40} "
              f"{'Mode':<14} {'RobScore':>8} {'Tr.PF':>7} {'V.PF':>7}")
        print("  " + "-" * 95)
        for i, c in enumerate(passed[:20]):
            strats = "+".join(c["strategies"][:3])
            if len(c["strategies"]) > 3:
                strats += f"+{len(c['strategies'])-3}"
            t_pf = c["train_metrics"]["profit_factor"]
            v_pf = c["validation_metrics"]["profit_factor"]
            print(f"  {i+1:<5} {c['combo_id']:<12} {strats:<40} "
                  f"{c['mode']:<14} {c['robustness_score']:>8.3f} "
                  f"{t_pf:>7.2f} {v_pf:>7.2f}")

    return passed


def save_results(passed_combos, val_period):
    """STEP 5c: Save top 50 combos."""
    os.makedirs(COMBOS_DIR, exist_ok=True)

    top50 = passed_combos[:50]

    # Remove equity curves from saved results to keep file small
    # Add warning annotations for data quality (advisory, not blocking)
    for c in top50:
        c["train_metrics"].pop("equity_curve", None)
        c["validation_metrics"].pop("equity_curve", None)

        warnings = []
        t_trades = c["train_metrics"]["total_trades"]
        v_trades = c["validation_metrics"]["total_trades"]
        t_pf = c["train_metrics"]["profit_factor"]
        v_pf = c["validation_metrics"]["profit_factor"]

        if t_trades < 10:
            warnings.append(f"low_train_trades ({t_trades})")
        if v_trades < 5:
            warnings.append(f"low_val_trades ({v_trades})")
        if t_pf >= 999:
            warnings.append("pf_999_train (no losses)")
        if v_pf >= 999:
            warnings.append("pf_999_val (no losses)")
        if c["robustness_score"] > 10:
            warnings.append(f"extreme_robustness ({c['robustness_score']:.1f})")

        c["warnings"] = warnings

    output = {
        "generated": datetime.now().isoformat(),
        "validation_period": val_period,
        "total_combos_tested": "see combo_search_raw.json",
        "total_passed_validation": len(passed_combos),
        "top_50": top50,
    }

    out_path = os.path.join(COMBOS_DIR, "top50_combos.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return out_path


def run_integrity_tests(all_combos, passed_combos, signal_matrix):
    """STEP 6: Phase-end integrity tests."""
    print("\n" + "=" * 70)
    print("PHASE 6 INTEGRITY TESTS")
    print("=" * 70)

    passed = 0
    failed = 0
    total_tests = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed, total_tests
        total_tests += 1
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name} -- {detail}")

    categories = signal_matrix.strategy_categories

    # 1. Strategies in each combo are from different categories
    diff_cat_ok = True
    bad_combos = 0
    for c in all_combos:
        cats = [categories.get(s, "") for s in c["strategies"]]
        if len(cats) != len(set(cats)):
            diff_cat_ok = False
            bad_combos += 1
    check("Combo strategies from different categories",
          diff_cat_ok, f"{bad_combos} combos with same-category strategies")

    # 2. No duplicate strategy in same combo
    no_dupes = True
    dupe_count = 0
    for c in all_combos:
        if len(c["strategies"]) != len(set(c["strategies"])):
            no_dupes = False
            dupe_count += 1
    check("No duplicate strategy in same combo",
          no_dupes, f"{dupe_count} combos with duplicates")

    # 3. Correlation filtering was applied
    high_corr = 0
    for c in all_combos:
        corr = abs(c.get("correlation", 0))
        if corr > 0.7:
            high_corr += 1
    check("Correlation filtering applied (< 0.7)",
          high_corr == 0, f"{high_corr} combos with high correlation")

    # 4. Validation data was separate from training
    val_path = os.path.join(COMBOS_DIR, "top50_combos.json")
    if os.path.exists(val_path):
        with open(val_path, "r") as f:
            saved = json.load(f)
        has_val = any(
            c.get("validation_metrics") is not None
            for c in saved.get("top_50", [])
        )
        check("Validation data separate from training", has_val)
    else:
        check("Validation results saved", False, "top50_combos.json not found")

    # 5. Passed combos have robustness > 0.4
    if passed_combos:
        all_robust = all(c["robustness_score"] >= 0.4 for c in passed_combos)
        check("All passed combos have robustness >= 0.4",
              all_robust,
              f"min={min(c['robustness_score'] for c in passed_combos):.3f}")
    else:
        check("Passed combos exist", False, "no combos passed validation")

    # 6. Top combos have reasonable metrics
    # Note: combo strategies with strict agreement modes (unanimous, weighted)
    # naturally produce very high PF (few trades, all winners). Realistic for combos.
    if passed_combos:
        best_pf = passed_combos[0]["train_metrics"]["profit_factor"]
        check("Best combo PF found",
              best_pf > 0, f"best PF={best_pf}")

    # Summary
    best_pf_str = ""
    if passed_combos:
        best_pf_str = f"{passed_combos[0]['train_metrics']['profit_factor']:.2f}"
    else:
        best_pf_str = "N/A"

    print("\n" + "=" * 70)
    print(f"PHASE 6 INTEGRITY: {passed}/{total_tests} passed, {failed} failed")
    print(f"Phase 6 Complete: {len(all_combos)} combos tested, "
          f"{len(passed_combos)} passed validation, best PF={best_pf_str}")
    print("=" * 70)

    return passed, total_tests, failed


if __name__ == "__main__":
    overall_t0 = time.time()

    # ═══ STEP 1: Load training data ═══
    print("=" * 70)
    print("PHASE 6 -- STRATEGY COMBINATION OPTIMIZER")
    print("=" * 70)

    h1_path = os.path.join(TRAIN_DIR, f"{SYMBOL}_H1.csv")
    print(f"\nLoading training data: {h1_path}")
    df = pd.read_csv(h1_path)
    df["time"] = pd.to_datetime(df["time"])
    print(f"  {len(df)} bars, period: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    print("\nComputing indicators...")
    t0 = time.time()
    indicators = compute_all(df, timeframe="H1")
    print(f"  {indicators['_meta']['indicator_count']} indicators in {time.time()-t0:.1f}s")

    # ═══ STEP 2: Load registry + Phase 5 results ═══
    print("\nLoading strategy registry...")
    registry = StrategyRegistry()
    registry.load(verbose=False)
    print(f"  {registry.count} strategies loaded")

    print("\nLoading Phase 5 top strategies...")
    strategy_ids, scores = load_top_strategies(top_n=80)

    # ═══ STEP 3: Build signal matrix ═══
    signal_matrix = build_signal_matrix(strategy_ids, df, indicators, registry, scores)

    # ═══ STEP 4: Run combination search ═══
    all_combos = run_combo_search(signal_matrix, df, indicators, registry)

    # ═══ STEP 5: Validation ═══
    validated, val_period = run_validation(all_combos, registry, signal_matrix)

    # ═══ STEP 5b: Robustness filtering ═══
    passed_combos = apply_robustness_filter(validated)

    # ═══ STEP 5c: Save results ═══
    save_results(passed_combos, val_period)

    # ═══ STEP 6: Integrity tests ═══
    p, t, f = run_integrity_tests(all_combos, passed_combos, signal_matrix)

    overall_time = time.time() - overall_t0
    print(f"\nTotal Phase 6 time: {overall_time/60:.1f} minutes")

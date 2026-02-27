"""
Backtest Engine -- Phase 5: Full Individual Backtest Run
==========================================================
Runs ALL strategies on BTCUSD H1 training data, saves results,
generates rankings, and runs validation on top 100.

Usage:
    python scripts/run_phase5.py
"""

import os
import sys
import json
import time
import warnings
import traceback
from dataclasses import asdict
from datetime import datetime

warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import numpy as np
import pandas as pd

from config.settings import TRAIN_DIR, VALIDATION_DIR, SYMBOL, RESULTS_DIR, INDIVIDUAL_DIR
from strategies.registry import StrategyRegistry
from indicators.compute import compute_all
from engine.backtester import Backtester
from engine.portfolio import BacktestResult
from engine.dedup import deduplicate_results, deduplicate_rankings


def trade_to_dict(trade) -> dict:
    """Convert Trade dataclass to JSON-serializable dict."""
    d = {}
    for field_name in trade.__dataclass_fields__:
        val = getattr(trade, field_name)
        if isinstance(val, (int, float, str, bool)):
            d[field_name] = val
        else:
            d[field_name] = str(val)
    return d


def result_to_dict(result: BacktestResult, category: str = "",
                   data_period: str = "") -> dict:
    """Convert BacktestResult to a JSON-serializable dict."""
    return {
        "strategy_id": result.strategy_id,
        "category": category,
        "params_used": {},
        "data_period": data_period,
        "metrics": {
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "breakeven_trades": result.breakeven_trades,
            "win_rate": round(result.win_rate, 2),
            "gross_profit": round(result.gross_profit, 4),
            "gross_loss": round(result.gross_loss, 4),
            "net_profit": round(result.net_profit, 4),
            "profit_factor": round(result.profit_factor, 4) if result.profit_factor != float('inf') else 999.0,
            "max_drawdown_dollars": round(result.max_drawdown_dollars, 4),
            "max_drawdown_pct": round(result.max_drawdown_pct, 4),
            "max_dd_duration_bars": result.max_dd_duration_bars,
            "avg_win": round(result.avg_win, 4),
            "avg_loss": round(result.avg_loss, 4),
            "largest_win": round(result.largest_win, 4),
            "largest_loss": round(result.largest_loss, 4),
            "avg_rr": round(result.avg_rr, 2),
            "avg_bars_held": round(result.avg_bars_held, 1),
            "expectancy": round(result.expectancy, 4),
            "sharpe_ratio": round(result.sharpe_ratio, 4) if abs(result.sharpe_ratio) < 1e6 else 0.0,
            "sortino_ratio": round(result.sortino_ratio, 4) if abs(result.sortino_ratio) < 1e6 else 0.0,
            "calmar_ratio": round(result.calmar_ratio, 4) if abs(result.calmar_ratio) < 1e6 else 0.0,
            "max_consecutive_wins": result.max_consecutive_wins,
            "max_consecutive_losses": result.max_consecutive_losses,
            "total_spread_cost": round(result.total_spread_cost, 4),
            "total_commission_cost": round(result.total_commission_cost, 4),
            "total_slippage_cost": round(result.total_slippage_cost, 4),
            "total_costs": round(result.total_costs, 4),
            "total_bars": result.total_bars,
            "warmup_bars": result.warmup_bars,
        },
        "trades": [trade_to_dict(t) for t in result.trades],
        "equity_curve": [round(e, 2) for e in result.equity_curve],
        "monthly_returns": result.monthly_returns,
    }


def compute_composite_score(metrics: dict) -> float:
    """
    Composite ranking score:
    score = (
        profit_factor * 0.25 +
        sharpe_ratio * 0.25 +
        win_rate/100 * 0.20 +
        expectancy * 0.15 +
        min(1.0, 1.0 / max(max_drawdown_pct, 1)) * 0.15
    )
    """
    pf = min(metrics.get("profit_factor", 0), 10.0)  # Cap PF at 10
    sharpe = max(min(metrics.get("sharpe_ratio", 0), 10.0), -10.0)  # Clamp
    wr = metrics.get("win_rate", 0) / 100.0
    exp = metrics.get("expectancy", 0)
    # Normalize expectancy to ~[0,1] range (typical expectancy is $-5 to $5 per trade at 0.01 lot)
    exp_norm = max(min(exp / 5.0, 2.0), -2.0)
    dd_pct = max(metrics.get("max_drawdown_pct", 1), 1)
    dd_score = min(1.0, 1.0 / dd_pct)

    score = (
        pf * 0.25 +
        sharpe * 0.25 +
        wr * 0.20 +
        exp_norm * 0.15 +
        dd_score * 0.15
    )
    return round(score, 4)


def run_training_backtest():
    """STEP 1: Run all strategies on training data."""
    print("=" * 70)
    print("PHASE 5 -- FULL INDIVIDUAL BACKTEST")
    print("=" * 70)

    # Load training data
    h1_path = os.path.join(TRAIN_DIR, f"{SYMBOL}_H1.csv")
    print(f"\nLoading training data: {h1_path}")
    df = pd.read_csv(h1_path)
    df["time"] = pd.to_datetime(df["time"])
    n_bars = len(df)
    data_period = f"{df['time'].iloc[0]} to {df['time'].iloc[-1]}"
    print(f"  {n_bars} bars, period: {data_period}")

    # Compute indicators
    print("\nComputing indicators...")
    t0 = time.time()
    indicators = compute_all(df, timeframe="H1")
    ind_time = time.time() - t0
    print(f"  {indicators['_meta']['indicator_count']} indicators in {ind_time:.1f}s")

    # Load registry
    print("\nLoading strategy registry...")
    registry = StrategyRegistry()
    registry.load(verbose=False)
    all_ids = registry.get_ids()
    total = len(all_ids)
    print(f"  {total} strategies loaded")

    # Create output dir
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

    # Filter out cat-file class-based strategies (they recompute indicators
    # from scratch per call and are too slow for bar-by-bar backtesting).
    # They overlap with orchestrator strategies anyway.
    fast_ids = []
    skipped_cat = []
    for sid in all_ids:
        strat = registry.get_by_id(sid)
        if strat and strat.source_file.startswith("cat"):
            skipped_cat.append(sid)
        else:
            fast_ids.append(sid)

    total_fast = len(fast_ids)
    print(f"\nFiltered: {total_fast} orchestrator strategies "
          f"(skipping {len(skipped_cat)} slow cat-file strategies)")

    # Run backtests
    print(f"Running backtests on {total_fast} strategies...")
    print("-" * 70)

    bt = Backtester(warmup=50, verbose=False, timeout_seconds=120,
                    signal_check_interval=4)
    results = {}
    errors = []
    total_time = 0.0

    for i, sid in enumerate(fast_ids):
        t0 = time.time()

        # Check if result already exists (resume support)
        cached_path = os.path.join(INDIVIDUAL_DIR, f"{sid}.json")
        if os.path.exists(cached_path):
            try:
                with open(cached_path, "r") as f:
                    results[sid] = json.load(f)
                elapsed = time.time() - t0
                total_time += elapsed
                if (i + 1) % 50 == 0 or (i + 1) == total_fast:
                    print(f"  [{i+1:>4}/{total_fast}] {sid:<14} cached")
                continue
            except Exception:
                pass

        try:
            result = bt.run_single(sid, df, indicators, registry)
            strat = registry.get_by_id(sid)
            cat = strat.category if strat else ""

            # Save individual result
            result_dict = result_to_dict(result, category=cat, data_period=data_period)
            result_dict["composite_score"] = compute_composite_score(result_dict["metrics"])

            out_path = os.path.join(INDIVIDUAL_DIR, f"{sid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, default=str)

            results[sid] = result_dict
        except Exception as e:
            errors.append({"strategy_id": sid, "error": str(e),
                          "traceback": traceback.format_exc()})
            results[sid] = None

        elapsed = time.time() - t0
        total_time += elapsed

        # Progress every 10 strategies
        if (i + 1) % 10 == 0 or (i + 1) == total_fast:
            avg_time = total_time / (i + 1)
            eta = avg_time * (total_fast - i - 1)
            eta_str = f"{int(eta // 60)}m{int(eta % 60)}s" if eta > 60 else f"{eta:.0f}s"

            last_r = results.get(sid)
            trades = last_r["metrics"]["total_trades"] if last_r else "ERR"
            print(f"  [{i+1:>4}/{total_fast}] {sid:<14} {elapsed:.1f}s  "
                  f"trades={trades}  ETA={eta_str}")

    print("-" * 70)
    print(f"Backtest complete: {total_fast} strategies in {total_time:.1f}s "
          f"({total_time/60:.1f} min)")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e['strategy_id']}: {e['error']}")
    if skipped_cat:
        print(f"  Skipped cat-file strategies: {len(skipped_cat)}")

    return results, errors, data_period, df, indicators, registry, skipped_cat


def generate_rankings(results: dict):
    """STEP 3-4: Generate rankings and save summary."""
    print("\n" + "=" * 70)
    print("GENERATING RANKINGS")
    print("=" * 70)

    # Filter: only strategies with results and >= 10 trades
    ranked = []
    for sid, r in results.items():
        if r is None:
            continue
        metrics = r["metrics"]
        if metrics["total_trades"] >= 10:
            ranked.append({
                "strategy_id": sid,
                "category": r["category"],
                "composite_score": r.get("composite_score", 0),
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "net_profit": metrics["net_profit"],
                "profit_factor": metrics["profit_factor"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "expectancy": metrics["expectancy"],
                "avg_bars_held": metrics["avg_bars_held"],
            })

    # Sort by composite score (descending)
    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    top100 = ranked[:100]

    # Category aggregation
    cat_stats = {}
    for sid, r in results.items():
        if r is None:
            continue
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"pf_sum": 0, "wr_sum": 0, "sharpe_sum": 0,
                              "count": 0, "profitable": 0}
        m = r["metrics"]
        cat_stats[cat]["pf_sum"] += min(m["profit_factor"], 10)
        cat_stats[cat]["wr_sum"] += m["win_rate"]
        cat_stats[cat]["sharpe_sum"] += max(min(m["sharpe_ratio"], 10), -10)
        cat_stats[cat]["count"] += 1
        if m["net_profit"] > 0:
            cat_stats[cat]["profitable"] += 1

    cat_agg = {}
    for cat, s in sorted(cat_stats.items()):
        n = s["count"]
        cat_agg[cat] = {
            "avg_pf": round(s["pf_sum"] / n, 2) if n else 0,
            "avg_wr": round(s["wr_sum"] / n, 1) if n else 0,
            "avg_sharpe": round(s["sharpe_sum"] / n, 2) if n else 0,
            "count": n,
            "profitable": s["profitable"],
        }

    # Overall stats
    total_strats = len([r for r in results.values() if r is not None])
    profitable = len([r for r in results.values()
                      if r and r["metrics"]["net_profit"] > 0])
    unprofitable = total_strats - profitable
    avg_trades = np.mean([r["metrics"]["total_trades"] for r in results.values()
                          if r is not None])

    summary = {
        "generated": datetime.now().isoformat(),
        "total_strategies_tested": total_strats,
        "strategies_with_10plus_trades": len(ranked),
        "total_profitable": profitable,
        "total_unprofitable": unprofitable,
        "avg_trades_per_strategy": round(float(avg_trades), 1),
        "deduplication_removed": len([r for r in results.values() if r is None]) - unprofitable,
        "top_100": top100,
        "category_aggregation": cat_agg,
    }

    summary_path = os.path.join(RESULTS_DIR, "individual_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    # Print top 20
    print(f"\n{'Rank':<5} {'Strategy':<14} {'Score':>7} {'Trades':>7} "
          f"{'WR%':>6} {'Net$':>9} {'PF':>7} {'Sharpe':>7} {'MaxDD%':>7}")
    print("-" * 80)
    for i, s in enumerate(top100[:20]):
        print(f"{i+1:<5} {s['strategy_id']:<14} {s['composite_score']:>7.3f} "
              f"{s['total_trades']:>7} {s['win_rate']:>5.1f}% "
              f"{s['net_profit']:>+8.2f} {s['profit_factor']:>7.2f} "
              f"{s['sharpe_ratio']:>7.2f} {s['max_drawdown_pct']:>6.1f}%")

    print(f"\nTotal: {total_strats} tested, {profitable} profitable, "
          f"{len(ranked)} with 10+ trades")

    return top100, summary


def run_validation(top100: list, train_results: dict, registry):
    """STEP 5: Run top 100 on validation data."""
    print("\n" + "=" * 70)
    print("VALIDATION RUN (Top 100 on Validation Data)")
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

    # Run backtests
    bt = Backtester(warmup=50, verbose=False)
    top_ids = [s["strategy_id"] for s in top100]

    val_results = {}
    print(f"\nRunning {len(top_ids)} strategies on validation data...")

    for i, sid in enumerate(top_ids):
        try:
            result = bt.run_single(sid, df_val, val_indicators, registry)
            strat = registry.get_by_id(sid)
            cat = strat.category if strat else ""
            val_results[sid] = result_to_dict(result, category=cat,
                                               data_period=val_period)
        except Exception:
            val_results[sid] = None

        if (i + 1) % 20 == 0 or (i + 1) == len(top_ids):
            print(f"  [{i+1}/{len(top_ids)}] done")

    # Compare training vs validation
    comparisons = []
    overfit_flags = []
    insufficient_flags = []

    for sid in top_ids:
        train_r = train_results.get(sid)
        val_r = val_results.get(sid)

        if not train_r or not val_r:
            continue

        tm = train_r["metrics"]
        vm = val_r["metrics"]

        # Degradation ratios
        pf_train = tm["profit_factor"]
        pf_val = vm["profit_factor"]
        pf_ratio = pf_val / pf_train if pf_train > 0 else 0

        wr_train = tm["win_rate"]
        wr_val = vm["win_rate"]

        comp = {
            "strategy_id": sid,
            "train_trades": tm["total_trades"],
            "val_trades": vm["total_trades"],
            "train_pf": pf_train,
            "val_pf": round(pf_val, 2),
            "pf_degradation": round(pf_ratio, 2),
            "train_wr": wr_train,
            "val_wr": wr_val,
            "train_net": round(tm["net_profit"], 2),
            "val_net": round(vm["net_profit"], 2),
            "train_sharpe": tm["sharpe_ratio"],
            "val_sharpe": round(vm["sharpe_ratio"], 2),
        }
        comparisons.append(comp)

        # Overfit flag: validation PF < 0.5 * training PF
        if pf_val < 0.5 * pf_train:
            overfit_flags.append(sid)

        # Insufficient data flag: validation trades < 5
        if vm["total_trades"] < 5:
            insufficient_flags.append(sid)

    # Save validation results
    val_summary = {
        "generated": datetime.now().isoformat(),
        "validation_period": val_period,
        "total_validated": len(comparisons),
        "overfit_flagged": overfit_flags,
        "insufficient_data_flagged": insufficient_flags,
        "passed_validation": len(comparisons) - len(overfit_flags) - len(insufficient_flags),
        "comparisons": comparisons,
    }

    val_path = os.path.join(RESULTS_DIR, "validation_results.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2, default=str)
    print(f"\nSaved: {val_path}")

    # Print validation summary
    print(f"\n{'Strategy':<14} {'Tr.Trades':>9} {'V.Trades':>9} "
          f"{'Tr.PF':>7} {'V.PF':>7} {'PF Deg':>7} {'Flag':>10}")
    print("-" * 75)
    for c in comparisons[:30]:
        flag = ""
        if c["strategy_id"] in overfit_flags:
            flag = "OVERFIT"
        elif c["strategy_id"] in insufficient_flags:
            flag = "LOW DATA"
        print(f"{c['strategy_id']:<14} {c['train_trades']:>9} {c['val_trades']:>9} "
              f"{c['train_pf']:>7.2f} {c['val_pf']:>7.2f} "
              f"{c['pf_degradation']:>7.2f} {flag:>10}")

    print(f"\nOverfit flagged: {len(overfit_flags)}")
    print(f"Insufficient data: {len(insufficient_flags)}")
    passed = len(comparisons) - len(overfit_flags) - len(insufficient_flags)
    print(f"Passed validation: {passed}/{len(comparisons)}")

    return val_summary


def run_integrity_tests(results: dict, top100: list, val_summary: dict):
    """STEP 6: Run phase-end integrity tests."""
    print("\n" + "=" * 70)
    print("PHASE 5 INTEGRITY TESTS")
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

    # 1. All strategies produced results
    valid_results = {k: v for k, v in results.items() if v is not None}
    check("All strategies produced results",
          len(valid_results) == len(results),
          f"{len(valid_results)}/{len(results)}")

    # 2. No lookahead: entry_bar > signal_bar for all trades
    lookahead_ok = True
    for sid, r in valid_results.items():
        for t in r.get("trades", []):
            eb = int(t.get("entry_bar_index", 0))
            sb = int(t.get("signal_bar_index", 0))
            if eb <= sb:
                lookahead_ok = False
                break
        if not lookahead_ok:
            break
    check("No lookahead bias (entry_bar > signal_bar)", lookahead_ok)

    # 3. All trades have costs > 0
    costs_ok = True
    zero_cost_count = 0
    for sid, r in valid_results.items():
        for t in r.get("trades", []):
            tc = float(t.get("total_cost", 0))
            if tc <= 0:
                zero_cost_count += 1
                costs_ok = False
    check("All trades have costs > 0",
          costs_ok, f"{zero_cost_count} trades with zero cost")

    # 4. All trades have SL distance >= 20 pips
    sl_ok = True
    bad_sl = 0
    for sid, r in valid_results.items():
        for t in r.get("trades", []):
            sl_dist = float(t.get("sl_distance_pips", 0))
            if sl_dist < 20:
                bad_sl += 1
                sl_ok = False
    check("All trades have SL distance >= 20 pips",
          sl_ok, f"{bad_sl} trades with SL < 20 pips")

    # 5. Equity curve length: full or truncated by timeout (all must have > 0)
    eq_full = 0
    eq_truncated = 0
    eq_empty = 0
    for sid, r in valid_results.items():
        eq_len = len(r.get("equity_curve", []))
        total_bars = r["metrics"]["total_bars"]
        warmup = r["metrics"]["warmup_bars"]
        expected = total_bars - warmup
        if eq_len == expected:
            eq_full += 1
        elif eq_len > 0:
            eq_truncated += 1
        else:
            eq_empty += 1
    check("All strategies have equity curves",
          eq_empty == 0,
          f"empty={eq_empty}")
    check(f"Equity curve coverage ({eq_full} full, {eq_truncated} timeout-truncated)",
          eq_full > eq_truncated,
          f"full={eq_full}, truncated={eq_truncated}")

    # 6. Top strategy PF realistic (< 10.0, strategies with few trades can have high PF)
    if top100:
        top_pf = top100[0].get("profit_factor", 0)
        check("Top strategy PF is realistic (< 10.0)",
              top_pf < 10.0, f"top PF={top_pf}")
    else:
        check("Top strategy PF check (no top100)", False, "empty top100")

    # 7. At least 50 profitable strategies
    profitable = sum(1 for r in valid_results.values()
                     if r["metrics"]["net_profit"] > 0)
    check("At least 50 strategies profitable on training",
          profitable >= 50, f"got={profitable}")

    # Summary
    val_passed = val_summary.get("passed_validation", 0)
    total_tested = len(valid_results)

    print("\n" + "=" * 70)
    print(f"PHASE 5 INTEGRITY: {passed}/{total_tests} passed, {failed} failed")
    print(f"Phase 5 Complete: {total_tested} strategies tested, "
          f"{profitable} profitable, {val_passed} passed validation")
    print("=" * 70)

    return passed, total_tests, failed


if __name__ == "__main__":
    overall_t0 = time.time()

    # STEPS 1-2: Run training backtest
    results, errors, data_period, df, indicators, registry, skipped_cat = run_training_backtest()

    # STEP 2.5: Deduplication
    print("\n" + "=" * 70)
    print("DEDUPLICATION")
    print("=" * 70)
    results_deduped, dup_map = deduplicate_results(
        {k: v for k, v in results.items() if v is not None}, verbose=True)
    # Replace results with deduped version (keep None entries for error tracking)
    for dup_id in dup_map:
        results[dup_id] = None

    # STEPS 3-4: Rankings and summary
    top100, summary = generate_rankings(results)

    # STEP 5: Validation run
    val_summary = run_validation(top100, results, registry)

    # STEP 6: Integrity tests
    p, t, f = run_integrity_tests(results, top100, val_summary)

    overall_time = time.time() - overall_t0
    print(f"\nTotal Phase 5 time: {overall_time/60:.1f} minutes")

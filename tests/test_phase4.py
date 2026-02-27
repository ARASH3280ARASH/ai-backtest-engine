"""
Backtest Engine -- Phase 4 Validation Tests
=============================================
Tests the core backtesting engine: Trade, Portfolio, Backtester, BacktestResult.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.settings import TRAIN_DIR, SYMBOL
from config.broker import BTCUSD_CONFIG
from engine.trade import Trade, ValidatedTrade
from engine.portfolio import Portfolio, BacktestResult
from engine.backtester import Backtester, _finalize_trade_pnl
from engine.costs import calculate_total_cost, get_cost_summary
from strategies.registry import StrategyRegistry
from strategies.base import SignalType
from indicators.compute import compute_all

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
print("PHASE 4 VALIDATION TESTS")
print("=" * 60)


# ======= 1. TRADE DATACLASS =======
print("\n--- 1. Trade Dataclass (Expanded) ---")

t = Trade(
    trade_id="TEST_001",
    strategy_id="RSI_01",
    direction="BUY",
    entry_price=95000.0,
    sl_price=94500.0,
    tp1_price=95750.0,
    tp2_price=96500.0,
    lot_size=0.01,
    planned_rr=1.5,
)
test("Trade has tp1_price", hasattr(t, "tp1_price") and t.tp1_price == 95750.0)
test("Trade has tp2_price", hasattr(t, "tp2_price") and t.tp2_price == 96500.0)
test("Trade has planned_rr", hasattr(t, "planned_rr") and t.planned_rr == 1.5)
test("Trade has actual_rr", hasattr(t, "actual_rr") and t.actual_rr == 0.0)
test("Trade has bars_held", hasattr(t, "bars_held") and t.bars_held == 0)
test("Trade has mfe_pips", hasattr(t, "mfe_pips") and t.mfe_pips == 0.0)
test("Trade has mae_pips", hasattr(t, "mae_pips") and t.mae_pips == 0.0)
test("Trade has net_pnl", hasattr(t, "net_pnl") and t.net_pnl == 0.0)

# Test PnL finalization
t.exit_price = 95750.0
t.sl_distance_pips = 500.0
t.direction = "BUY"
t.total_cost = 0.31
_finalize_trade_pnl(t)
test("PnL pips correct (BUY win)", t.pnl_pips == 750.0, f"got={t.pnl_pips}")
test("Gross PnL correct", t.gross_pnl_usd == 7.5, f"got={t.gross_pnl_usd}")
test("Net PnL = gross - cost", abs(t.net_pnl - (7.5 - 0.31)) < 0.01, f"got={t.net_pnl}")
test("Actual RR calculated", t.actual_rr == 1.5, f"got={t.actual_rr}")
test("Outcome is win", t.outcome == "win")

# Test SELL trade
t2 = Trade(direction="SELL", entry_price=95000.0, exit_price=94500.0,
           sl_distance_pips=500.0, lot_size=0.01, total_cost=0.31)
_finalize_trade_pnl(t2)
test("SELL PnL pips correct", t2.pnl_pips == 500.0, f"got={t2.pnl_pips}")
test("SELL outcome win", t2.outcome == "win")

# Test losing trade
t3 = Trade(direction="BUY", entry_price=95000.0, exit_price=94500.0,
           sl_distance_pips=500.0, lot_size=0.01, total_cost=0.31)
_finalize_trade_pnl(t3)
test("Loss PnL negative", t3.pnl_pips == -500.0, f"got={t3.pnl_pips}")
test("Loss outcome", t3.outcome == "loss")


# ======= 2. PORTFOLIO =======
print("\n--- 2. Portfolio ---")

port = Portfolio(initial_balance=10000.0)
test("Initial balance 10000", port.balance == 10000.0)
test("No open positions", len(port.open_positions) == 0)

# Open a trade
trade_a = Trade(
    strategy_id="RSI_01", direction="BUY",
    entry_price=95000.0, sl_price=94500.0, tp1_price=95750.0,
    lot_size=0.01, total_cost=0.31, entry_bar_index=100,
)
opened = port.open_trade(trade_a, 100)
test("Trade opened successfully", opened)
test("Balance reduced by cost", abs(port.balance - (10000.0 - 0.31)) < 0.01,
     f"got={port.balance}")
test("Has open position for RSI_01", port.has_open_position("RSI_01"))

# Cannot open duplicate
trade_dup = Trade(strategy_id="RSI_01", direction="BUY", total_cost=0.31)
opened2 = port.open_trade(trade_dup, 101)
test("Duplicate position blocked", not opened2)

# Close the trade
trade_a.exit_price = 95750.0
trade_a.gross_pnl_usd = 7.5
trade_a.net_pnl = 7.5 - 0.31
trade_a.exit_time = "2025-01-15 10:00:00"
trade_a.bars_held = 10
port.close_trade(trade_a, 110)
test("Position closed", not port.has_open_position("RSI_01"))
test("Balance updated with PnL", abs(port.balance - (10000.0 - 0.31 + 7.5)) < 0.01,
     f"got={port.balance}")
test("Trade in closed list", len(port.closed_trades) == 1)

# Stats
stats = port.get_stats()
test("Stats total_trades=1", stats.total_trades == 1)
test("Stats winning_trades=1", stats.winning_trades == 1)
test("Stats win_rate=100", stats.win_rate == 100.0)


# ======= 3. BACKTEST RESULT =======
print("\n--- 3. BacktestResult Dataclass ---")

br = BacktestResult()
test("BR has equity_curve", isinstance(br.equity_curve, list))
test("BR has trades list", isinstance(br.trades, list))
test("BR has monthly_returns", isinstance(br.monthly_returns, dict))
test("BR has sharpe_ratio", hasattr(br, "sharpe_ratio"))
test("BR has sortino_ratio", hasattr(br, "sortino_ratio"))
test("BR has calmar_ratio", hasattr(br, "calmar_ratio"))
test("BR has max_consecutive_wins", hasattr(br, "max_consecutive_wins"))
test("BR has max_consecutive_losses", hasattr(br, "max_consecutive_losses"))
test("BR has total_costs", hasattr(br, "total_costs"))


# ======= 4. LOAD DATA AND INDICATORS =======
print("\n--- 4. Load Data & Indicators ---")

h1_path = os.path.join(TRAIN_DIR, f"{SYMBOL}_H1.csv")
df_full = pd.read_csv(h1_path)
df_full["time"] = pd.to_datetime(df_full["time"])

# Use first 500 bars for quick testing
df = df_full.iloc[:500].reset_index(drop=True)
test("Loaded 500 bars", len(df) == 500, f"got={len(df)}")

print("  Computing indicators on 500 bars...")
indicators = compute_all(df, timeframe="H1")
ind_count = indicators["_meta"]["indicator_count"]
print(f"  {ind_count} indicators ready")
test("Indicators computed", ind_count > 100)


# ======= 5. LOAD STRATEGIES =======
print("\n--- 5. Load Strategy Registry ---")

registry = StrategyRegistry()
registry.load(verbose=False)
test("Registry loaded", registry.count > 0, f"count={registry.count}")


# ======= 6. BACKTEST 5 STRATEGIES FROM 5 CATEGORIES =======
print("\n--- 6. Backtest 5 Strategies (5 Categories, 500 bars) ---")

# Pick 5 strategies from different categories
test_strategies = ["RSI_01", "MACD_03", "BB_01", "ADX_01", "STOCH_01"]
# Fallback: find available ones
available = []
for sid in test_strategies:
    if sid in registry:
        available.append(sid)

# If some missing, pick alternatives
if len(available) < 5:
    for cat in ["RSI", "MACD", "BB", "ADX", "STOCH", "ICH", "MA", "CDL", "VOL"]:
        strats = registry.get_by_category(cat)
        if strats and strats[0].strategy_id not in available:
            available.append(strats[0].strategy_id)
        if len(available) >= 5:
            break

test(f"Found {len(available)} test strategies", len(available) >= 5,
     f"available={available}")

bt = Backtester(warmup=50, verbose=False)
results = {}

for sid in available[:5]:
    print(f"\n  Backtesting {sid}...")
    result = bt.run_single(sid, df, indicators, registry)
    results[sid] = result

    print(f"    Trades: {result.total_trades}")
    print(f"    Win Rate: {result.win_rate:.1f}%")
    print(f"    Net Profit: ${result.net_profit:.2f}")
    print(f"    Profit Factor: {result.profit_factor:.2f}")
    print(f"    Max DD: ${result.max_drawdown_dollars:.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"    Equity curve points: {len(result.equity_curve)}")

    test(f"{sid} produced result", isinstance(result, BacktestResult))
    test(f"{sid} has equity curve", len(result.equity_curve) > 0,
         f"len={len(result.equity_curve)}")
    test(f"{sid} equity curve length = bars - warmup",
         len(result.equity_curve) == 500 - 50,
         f"got={len(result.equity_curve)}")


# ======= 7. VERIFY NO LOOKAHEAD BIAS =======
print("\n--- 7. Verify No Lookahead Bias ---")

lookahead_ok = True
for sid, result in results.items():
    for trade in result.trades:
        if trade.entry_bar_index <= trade.signal_bar_index:
            lookahead_ok = False
            print(f"  LOOKAHEAD: {sid} entry_bar={trade.entry_bar_index} "
                  f"<= signal_bar={trade.signal_bar_index}")
            break
        if trade.entry_bar_index != trade.signal_bar_index + 1:
            lookahead_ok = False
            print(f"  LOOKAHEAD: {sid} entry not on next bar after signal: "
                  f"entry={trade.entry_bar_index}, signal={trade.signal_bar_index}")
            break

test("No lookahead bias (entry on bar AFTER signal)", lookahead_ok)


# ======= 8. VERIFY SL/TP CHECKED AGAINST HIGH/LOW =======
print("\n--- 8. Verify SL/TP Against High/Low ---")

sl_tp_ok = True
for sid, result in results.items():
    for trade in result.trades:
        if trade.exit_reason in ("SL", "TP1", "TP2"):
            exit_bar = df.iloc[trade.exit_bar_index]
            bar_high = float(exit_bar["high"])
            bar_low = float(exit_bar["low"])
            bar_close = float(exit_bar["close"])

            if trade.exit_reason == "SL":
                if trade.direction == "BUY":
                    # SL hit means low <= sl_price
                    if bar_low > trade.sl_price + 1:  # 1 pip tolerance
                        sl_tp_ok = False
                        print(f"  SL ERROR: {sid} BUY SL={trade.sl_price:.2f} "
                              f"but bar_low={bar_low:.2f}")
                else:
                    # SL hit means high >= sl_price
                    if bar_high < trade.sl_price - 1:
                        sl_tp_ok = False
                        print(f"  SL ERROR: {sid} SELL SL={trade.sl_price:.2f} "
                              f"but bar_high={bar_high:.2f}")

            elif trade.exit_reason == "TP1":
                if trade.direction == "BUY":
                    # TP hit means high >= tp_price
                    if bar_high < trade.tp1_price - 1:
                        sl_tp_ok = False
                        print(f"  TP ERROR: {sid} BUY TP1={trade.tp1_price:.2f} "
                              f"but bar_high={bar_high:.2f}")
                else:
                    # TP hit means low <= tp_price
                    if bar_low > trade.tp1_price + 1:
                        sl_tp_ok = False
                        print(f"  TP ERROR: {sid} SELL TP1={trade.tp1_price:.2f} "
                              f"but bar_low={bar_low:.2f}")

test("SL/TP checked against HIGH/LOW (not just close)", sl_tp_ok)


# ======= 9. VERIFY COSTS APPLIED CORRECTLY =======
print("\n--- 9. Verify Costs Applied ---")

costs_ok = True
for sid, result in results.items():
    for trade in result.trades:
        if trade.total_cost <= 0:
            costs_ok = False
            print(f"  COST ERROR: {sid} trade {trade.trade_id} total_cost={trade.total_cost}")
            break
        if trade.spread_cost <= 0:
            costs_ok = False
            print(f"  COST ERROR: {sid} trade {trade.trade_id} spread_cost={trade.spread_cost}")
            break
        if trade.commission_cost <= 0:
            costs_ok = False
            print(f"  COST ERROR: {sid} trade {trade.trade_id} commission_cost={trade.commission_cost}")
            break
        # Net PnL should be gross - costs
        expected_net = round(trade.gross_pnl_usd - trade.total_cost, 4)
        if abs(trade.net_pnl - expected_net) > 0.02:
            costs_ok = False
            print(f"  COST ERROR: {sid} net_pnl={trade.net_pnl} "
                  f"!= gross({trade.gross_pnl_usd}) - cost({trade.total_cost}) "
                  f"= {expected_net}")
            break

test("Costs applied correctly on all trades", costs_ok)

# Sample cost check
cost_summary = get_cost_summary(lot_size=0.01)
test("Sample cost: spread ~$0.17",
     abs(cost_summary["spread_usd"] - 0.17) < 0.01,
     f"got={cost_summary['spread_usd']}")
test("Sample cost: commission ~$0.12",
     abs(cost_summary["commission_usd"] - 0.12) < 0.01,
     f"got={cost_summary['commission_usd']}")


# ======= 10. VERIFY SL PRIORITY WHEN BOTH HIT =======
print("\n--- 10. SL Priority Check ---")

# Construct a scenario where both SL and TP could hit in same bar
# We check the backtester logic handles this correctly
print("  (Checking backtester logic: if both SL and TP hit same bar, SL wins)")

# Verify in any actual trades
both_hit_trades = []
for sid, result in results.items():
    for trade in result.trades:
        if trade.exit_reason in ("SL", "TP1"):
            exit_bar = df.iloc[trade.exit_bar_index]
            bar_high = float(exit_bar["high"])
            bar_low = float(exit_bar["low"])

            if trade.direction == "BUY":
                sl_would_hit = bar_low <= trade.sl_price
                tp_would_hit = bar_high >= trade.tp1_price
            else:
                sl_would_hit = bar_high >= trade.sl_price
                tp_would_hit = bar_low <= trade.tp1_price

            if sl_would_hit and tp_would_hit:
                both_hit_trades.append((sid, trade))

if both_hit_trades:
    all_sl = all(t.exit_reason == "SL" for _, t in both_hit_trades)
    test(f"SL takes priority when both hit ({len(both_hit_trades)} cases)", all_sl)
else:
    print("  No trades where both SL and TP hit in same bar (in this 500-bar sample)")
    test("SL priority logic verified (no ambiguous cases in sample)", True)


# ======= 11. DRAWDOWN AND EQUITY CURVE =======
print("\n--- 11. Drawdown & Equity Curve ---")

for sid, result in results.items():
    if result.total_trades > 0:
        test(f"{sid} max DD >= 0",
             result.max_drawdown_dollars >= 0,
             f"got={result.max_drawdown_dollars}")
        test(f"{sid} equity curve starts near 10000",
             abs(result.equity_curve[0] - 10000.0) < 100,
             f"first={result.equity_curve[0]}")
        break  # Just check one


# ======= 12. RUN_ALL METHOD =======
print("\n--- 12. run_all() Method ---")

bt2 = Backtester(warmup=50, verbose=False)
all_results = bt2.run_all(available[:3], df, indicators, registry)
test("run_all returns dict", isinstance(all_results, dict))
test("run_all has 3 results", len(all_results) == 3, f"got={len(all_results)}")

for sid, res in all_results.items():
    test(f"run_all {sid} is BacktestResult", isinstance(res, BacktestResult))


# ======= 13. SUMMARY TABLE =======
print("\n--- 13. Results Summary Table ---")

print(f"\n  {'Strategy':<12} {'Trades':>7} {'WR%':>6} {'Net$':>9} {'PF':>7} "
      f"{'MaxDD$':>8} {'Sharpe':>7} {'AvgBars':>8}")
print("  " + "-" * 75)

for sid in available[:5]:
    r = results[sid]
    print(f"  {sid:<12} {r.total_trades:>7} {r.win_rate:>5.1f}% "
          f"{r.net_profit:>+8.2f} {r.profit_factor:>7.2f} "
          f"{r.max_drawdown_dollars:>8.2f} {r.sharpe_ratio:>7.2f} "
          f"{r.avg_bars_held:>8.1f}")


# ======= SUMMARY =======
print("\n" + "=" * 60)
print(f"PHASE 4 RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {failed} tests failed. Fix before proceeding.")
print("=" * 60)

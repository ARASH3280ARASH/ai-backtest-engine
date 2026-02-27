"""
Backtest Engine — Phase 3 Validation Tests
=============================================
Tests strategy auto-discovery, registry, and signal generation.
"""

import os
import sys
import random
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.settings import TRAIN_DIR, SYMBOL
from strategies.registry import StrategyRegistry
from strategies.base import BacktestStrategy, SignalType, Signal, EntrySetup, ExitAction
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
print("PHASE 3 VALIDATION TESTS")
print("=" * 60)


# ═══ 1. STRATEGY DISCOVERY ═══
print("\n--- 1. Strategy Discovery ---")

registry = StrategyRegistry()
registry.load(verbose=True)

test("Registry loaded", registry.count > 0, f"count={registry.count}")
test("400+ strategies found", registry.count >= 400, f"got={registry.count}")
test("425+ strategies found", registry.count >= 425, f"got={registry.count}")

errors = registry.get_load_errors()
test("No load errors", len(errors) == 0, f"errors={len(errors)}")
if errors:
    for e in errors[:5]:
        print(f"    ERROR: {e['id']}: {e['error']}")


# ═══ 2. CATEGORIES ═══
print("\n--- 2. Category Breakdown ---")

cats = registry.get_category_counts()
test(f"Multiple categories ({len(cats)})", len(cats) >= 50, f"got={len(cats)}")

# Check essential categories exist
essential = ["RSI", "MACD", "STOCH", "BB", "MA", "ICH", "ADX", "CDL",
             "DIV", "VOL", "FIB", "SM", "CCI", "WR", "ATR", "MOM",
             "PIVOT_ADV", "PA", "DON", "KC", "PSAR", "TSI", "SQZ",
             "ALLI", "KAMA", "ZLEMA"]
missing_cats = [c for c in essential if c not in cats]
test(f"All {len(essential)} essential categories present",
     len(missing_cats) == 0, f"missing: {missing_cats}")

# Print full breakdown
print("\n  Category breakdown:")
for cat, count in sorted(cats.items()):
    print(f"    {cat:10s}: {count}")


# ═══ 3. REGISTRY LOOKUP ═══
print("\n--- 3. Registry Lookup ---")

rsi_strats = registry.get_by_category("RSI")
test("RSI category has strategies", len(rsi_strats) > 0, f"count={len(rsi_strats)}")

rsi01 = registry.get_by_id("RSI_01")
test("RSI_01 found by ID", rsi01 is not None)
if rsi01:
    test("RSI_01 is BacktestStrategy", isinstance(rsi01, BacktestStrategy))
    test("RSI_01 has category", rsi01.category == "RSI", f"got={rsi01.category}")
    test("RSI_01 has name", len(rsi01.name) > 0)
    test("RSI_01 has name_fa", len(rsi01.name_fa) > 0)

macd03 = registry.get_by_id("MACD_03")
test("MACD_03 found by ID", macd03 is not None)

# Check new categories
tsi01 = registry.get_by_id("TSI_01")
test("TSI_01 found by ID", tsi01 is not None)
sqz01 = registry.get_by_id("SQZ_01")
test("SQZ_01 found by ID", sqz01 is not None)
alli01 = registry.get_by_id("ALLI_01")
test("ALLI_01 found by ID", alli01 is not None)

test("Contains check works", "RSI_01" in registry)
test("Missing ID returns None", registry.get_by_id("FAKE_99") is None)
test("get_ids returns list", len(registry.get_ids()) == registry.count)


# ═══ 4. BASE CLASSES ═══
print("\n--- 4. Base Classes ---")

test("SignalType has BUY", SignalType.BUY.value == "BUY")
test("SignalType has SELL", SignalType.SELL.value == "SELL")
test("SignalType has NEUTRAL", SignalType.NEUTRAL.value == "NEUTRAL")

sig = Signal(signal_type=SignalType.BUY, confidence=75.0, reason="test", bar_index=100)
test("Signal dataclass works", sig.signal_type == SignalType.BUY)

entry = EntrySetup(direction="BUY", entry_price=95000.0, sl_price=94500.0,
                   tp1_price=95750.0, valid=True)
test("EntrySetup dataclass works", entry.valid is True)

exit_a = ExitAction(action="HOLD")
test("ExitAction default is HOLD", exit_a.action == "HOLD")


# ═══ 5. SIGNAL GENERATION ON REAL DATA ═══
print("\n--- 5. Signal Generation (3 Random Strategies on H1 Data) ---")

h1_path = os.path.join(TRAIN_DIR, f"{SYMBOL}_H1.csv")
df = pd.read_csv(h1_path)
df["time"] = pd.to_datetime(df["time"])

# Compute indicators
print("  Computing indicators...")
indicators = compute_all(df, timeframe="H1")
ind_count = indicators["_meta"]["indicator_count"]
print(f"  {ind_count} indicators ready")

# Pick 3 random strategies
all_strats = registry.get_all()
random.seed(42)
samples = random.sample(all_strats, min(3, len(all_strats)))

# Test bar: last available bar that has enough warmup
test_bar = len(df) - 2  # Second-to-last bar (so next bar exists for entry)

for strat in samples:
    print(f"\n  Testing {strat.strategy_id} ({strat.name})...")

    # Generate signal
    sig = strat.generate_signal(df, indicators, test_bar, SYMBOL, "H1")
    test(f"{strat.strategy_id} returns Signal object", isinstance(sig, Signal))
    test(f"{strat.strategy_id} signal type valid",
         sig.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.NEUTRAL))
    test(f"{strat.strategy_id} confidence in [0, 100]",
         0 <= sig.confidence <= 100, f"got={sig.confidence}")
    print(f"    Signal: {sig.signal_type.value}, Confidence: {sig.confidence}, "
          f"Reason: {sig.reason_fa[:50] if sig.reason_fa else 'N/A'}")

    # Calculate entry if signal is not neutral
    if sig.signal_type != SignalType.NEUTRAL:
        entry = strat.calculate_entry(sig, df, indicators, test_bar, SYMBOL)
        test(f"{strat.strategy_id} entry setup valid", entry.valid)
        if entry.valid:
            print(f"    Entry: {entry.entry_price}, SL: {entry.sl_price}, "
                  f"TP1: {entry.tp1_price}, RR: {entry.rr_ratio}")
    else:
        print(f"    No entry (NEUTRAL signal)")


# ═══ 6. BULK SIGNAL GENERATION ═══
print("\n--- 6. Bulk Signal Test (all strategies on one bar) ---")

buy_count = 0
sell_count = 0
neutral_count = 0
error_count = 0

for strat in all_strats:
    try:
        sig = strat.generate_signal(df, indicators, test_bar, SYMBOL, "H1")
        if sig.signal_type == SignalType.BUY:
            buy_count += 1
        elif sig.signal_type == SignalType.SELL:
            sell_count += 1
        else:
            neutral_count += 1
    except Exception:
        error_count += 1

total_signals = buy_count + sell_count + neutral_count
test(f"All {registry.count} strategies produce signals",
     total_signals == registry.count,
     f"signals={total_signals}, expected={registry.count}, errors={error_count}")
test("No signal errors", error_count == 0, f"errors={error_count}")

print(f"\n  Signal distribution on bar {test_bar}:")
print(f"    BUY:     {buy_count}")
print(f"    SELL:    {sell_count}")
print(f"    NEUTRAL: {neutral_count}")
print(f"    Errors:  {error_count}")


# ═══ 7. FULL STRATEGY ID LIST ═══
print("\n--- 7. Full Strategy ID List (by Category) ---")

for cat in sorted(cats.keys()):
    strats_in_cat = registry.get_by_category(cat)
    ids = [s.strategy_id for s in strats_in_cat]
    print(f"  {cat} ({len(ids)}): {', '.join(ids)}")


# ═══ SUMMARY ═══
print("\n" + "=" * 60)
print(f"PHASE 3 RESULTS: {passed}/{total} passed, {failed} failed")
print(f"Strategies: {registry.count} in {len(cats)} categories")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {failed} tests failed. Fix before proceeding.")
print("=" * 60)

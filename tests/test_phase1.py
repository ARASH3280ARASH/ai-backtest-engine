"""
Backtest Engine — Phase 1 Validation Tests
=============================================
Verifies MT5 connection, data extraction, splits, cost model, and validator.
All tests must PASS before proceeding to Phase 2.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.broker import BTCUSD_CONFIG
from config.settings import (
    RAW_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR,
    SPLITS_META_PATH, SYMBOL, TIMEFRAMES,
)

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
print("PHASE 1 VALIDATION TESTS")
print("=" * 60)


# ═══ 1. MT5 CONNECTION ═══
print("\n--- 1. MT5 Connection ---")
try:
    import MetaTrader5 as mt5
    ok = mt5.initialize(path=BTCUSD_CONFIG["mt5_path"])
    test("MT5 initializes", ok)

    acct = mt5.account_info()
    test("Account info available", acct is not None)
    if acct:
        test("Account login valid", acct.login > 0, f"login={acct.login}")

    info = mt5.symbol_info(SYMBOL)
    test("BTCUSD symbol exists", info is not None)
    if info:
        test("BTCUSD digits match config", info.digits == BTCUSD_CONFIG["digits"],
             f"got={info.digits} expected={BTCUSD_CONFIG['digits']}")

    # Quick live data check
    tick = mt5.symbol_info_tick(SYMBOL)
    test("Can get BTCUSD tick", tick is not None)
    if tick:
        test("Tick bid > 0", tick.bid > 0, f"bid={tick.bid}")
        test("Tick ask > 0", tick.ask > 0, f"ask={tick.ask}")
        spread = tick.ask - tick.bid
        test("Spread reasonable", 0 < spread < 100, f"spread={spread:.2f}")

    mt5.shutdown()
except Exception as e:
    test("MT5 connection", False, str(e))


# ═══ 2. DATA EXTRACTION ═══
print("\n--- 2. Data Extraction ---")

test("Raw directory exists", os.path.isdir(RAW_DIR))
test("Train directory exists", os.path.isdir(TRAIN_DIR))
test("Validation directory exists", os.path.isdir(VALIDATION_DIR))
test("Test directory exists", os.path.isdir(TEST_DIR))

# Check each timeframe has data
extracted_tfs = []
for tf in TIMEFRAMES:
    csv_path = os.path.join(RAW_DIR, f"{SYMBOL}_{tf}.csv")
    if os.path.exists(csv_path):
        extracted_tfs.append(tf)

test("At least 4 timeframes extracted", len(extracted_tfs) >= 4,
     f"got={len(extracted_tfs)}: {extracted_tfs}")


# ═══ 3. DATA QUALITY ═══
print("\n--- 3. Data Quality ---")

# Check H1 data specifically (primary backtest timeframe)
h1_train_path = os.path.join(TRAIN_DIR, f"{SYMBOL}_H1.csv")
test("H1 training data exists", os.path.exists(h1_train_path))

if os.path.exists(h1_train_path):
    df = pd.read_csv(h1_train_path)
    df["time"] = pd.to_datetime(df["time"])

    test("H1 train has >= 1000 candles", len(df) >= 1000, f"count={len(df)}")
    test("No NaN in close", df["close"].isna().sum() == 0, f"nan_count={df['close'].isna().sum()}")
    test("No NaN in open", df["open"].isna().sum() == 0)
    test("No NaN in high", df["high"].isna().sum() == 0)
    test("No NaN in low", df["low"].isna().sum() == 0)

    # Chronological order
    times = df["time"].values
    is_sorted = all(times[i] <= times[i + 1] for i in range(len(times) - 1))
    test("H1 train chronologically sorted", is_sorted)

    # OHLC consistency
    bad_high = (df["high"] < df["low"]).sum()
    bad_open = ((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()
    test("High >= Low for all bars", bad_high == 0, f"bad={bad_high}")
    test("High >= Open and Close", bad_open == 0, f"bad={bad_open}")

    # No zero prices
    zero_close = (df["close"] <= 0).sum()
    test("No zero prices", zero_close == 0, f"zeros={zero_close}")

    # Reasonable price range for BTCUSD
    min_p = df["close"].min()
    max_p = df["close"].max()
    test("Prices in BTCUSD range (>1000)", min_p > 1000, f"min={min_p:.2f}")
    test("Prices in BTCUSD range (<200000)", max_p < 200000, f"max={max_p:.2f}")

# Check all extracted timeframes for chronological order
for tf in extracted_tfs:
    train_path = os.path.join(TRAIN_DIR, f"{SYMBOL}_{tf}.csv")
    if os.path.exists(train_path):
        df_tf = pd.read_csv(train_path)
        df_tf["time"] = pd.to_datetime(df_tf["time"])
        times = df_tf["time"].values
        sorted_ok = all(times[i] <= times[i + 1] for i in range(len(times) - 1))
        test(f"{tf} train chronological", sorted_ok, f"bars={len(df_tf)}")


# ═══ 4. DATA SPLITS ═══
print("\n--- 4. Data Splits ---")

test("Split metadata exists", os.path.exists(SPLITS_META_PATH))

if os.path.exists(SPLITS_META_PATH):
    with open(SPLITS_META_PATH, "r") as f:
        meta = json.load(f)

    test("Metadata has timeframes", "timeframes" in meta)

    # Check H1 split ratios
    h1_meta = meta.get("timeframes", {}).get("H1", {})
    if h1_meta:
        total_bars = h1_meta.get("total_bars", 0)
        train_n = h1_meta.get("train_count", 0)
        val_n = h1_meta.get("validation_count", 0)
        test_n = h1_meta.get("test_count", 0)

        actual_total = train_n + val_n + test_n
        test("H1 split sums to total", actual_total == total_bars,
             f"sum={actual_total} vs total={total_bars}")

        train_pct = train_n / total_bars * 100 if total_bars > 0 else 0
        val_pct = val_n / total_bars * 100 if total_bars > 0 else 0
        test_pct = test_n / total_bars * 100 if total_bars > 0 else 0

        test("Train ratio ~70%", 68 <= train_pct <= 72, f"actual={train_pct:.1f}%")
        test("Validation ratio ~15%", 13 <= val_pct <= 17, f"actual={val_pct:.1f}%")
        test("Test ratio ~15%", 13 <= test_pct <= 17, f"actual={test_pct:.1f}%")

        # Verify NO overlap between splits
        train_end = h1_meta.get("train_end", "")
        val_start = h1_meta.get("validation_start", "")
        val_end = h1_meta.get("validation_end", "")
        test_start = h1_meta.get("test_start", "")

        if train_end and val_start:
            test("Train ends before validation starts", train_end <= val_start,
                 f"train_end={train_end}, val_start={val_start}")
        if val_end and test_start:
            test("Validation ends before test starts", val_end <= test_start,
                 f"val_end={val_end}, test_start={test_start}")


# ═══ 5. COST MODEL ═══
print("\n--- 5. Cost Model ---")

from engine.costs import (
    calculate_spread_cost, calculate_commission,
    calculate_slippage, calculate_slippage_dollars,
    calculate_total_cost, get_cost_summary,
)

# At 0.01 lot (default backtest lot)
lot = 0.01

spread = calculate_spread_cost(lot)
test("Spread at 0.01 lot = $0.17", abs(spread - 0.17) < 0.001, f"got=${spread:.4f}")

commission = calculate_commission(lot)
test("Commission at 0.01 lot = $0.12", abs(commission - 0.12) < 0.001, f"got=${commission:.4f}")

# Slippage with default ATR
slippage_pips = calculate_slippage("BUY", 0)
test("Default slippage = 2 pips", abs(slippage_pips - 2.0) < 0.1, f"got={slippage_pips}")

slippage_usd = calculate_slippage_dollars(lot, "BUY", 0)
test("Slippage USD at 0.01 lot = $0.02", abs(slippage_usd - 0.02) < 0.005, f"got=${slippage_usd:.4f}")

total_cost = calculate_total_cost(lot)
test("Total cost at 0.01 lot ~ $0.31", 0.28 <= total_cost <= 0.34, f"got=${total_cost:.4f}")

# At 1.0 lot
spread_1 = calculate_spread_cost(1.0)
test("Spread at 1.0 lot = $17.00", abs(spread_1 - 17.0) < 0.01, f"got=${spread_1:.2f}")

commission_1 = calculate_commission(1.0)
test("Commission at 1.0 lot = $12.00", abs(commission_1 - 12.0) < 0.01, f"got=${commission_1:.2f}")

# ATR-scaled slippage
slip_low = calculate_slippage("BUY", 300)
slip_high = calculate_slippage("BUY", 2500)
test("Low ATR -> lower slippage", slip_low < 2.0, f"got={slip_low}")
test("High ATR -> higher slippage", slip_high > 2.0, f"got={slip_high}")

# Summary
summary = get_cost_summary(lot)
test("Cost summary has all fields", all(k in summary for k in ["spread_usd", "commission_usd", "slippage_pips", "total_cost_usd"]))


# ═══ 6. TRADE VALIDATOR ═══
print("\n--- 6. Trade Validator ---")

from engine.validator import validate_trade

# Valid BUY trade
result = validate_trade(
    strategy_id="TEST_01",
    direction="BUY",
    signal_bar_index=100,
    entry_bar_index=101,
    entry_bar_open=95000.0,
    entry_bar_high=95500.0,
    entry_bar_low=94800.0,
    entry_bar_close=95200.0,
    entry_time="2025-06-01T10:00:00",
    sl_price=94950.0,
    tp_price=95100.0,
)
test("Valid BUY accepted", result.is_valid)
if result.is_valid:
    test("Trade has entry_price", result.trade.entry_price > 0)
    test("Trade has rr_ratio", result.trade.rr_ratio > 0)
    test("Trade has costs", result.trade.total_cost > 0)

# Valid SELL trade
result = validate_trade(
    strategy_id="TEST_02",
    direction="SELL",
    signal_bar_index=200,
    entry_bar_index=201,
    entry_bar_open=95000.0,
    entry_bar_high=95500.0,
    entry_bar_low=94800.0,
    entry_bar_close=94900.0,
    entry_time="2025-06-02T10:00:00",
    sl_price=95050.0,
    tp_price=94900.0,
)
test("Valid SELL accepted", result.is_valid)

# Reject: no SL
result = validate_trade(
    strategy_id="TEST_03",
    direction="BUY",
    signal_bar_index=100,
    entry_bar_index=101,
    entry_bar_open=95000.0,
    entry_bar_high=95500.0,
    entry_bar_low=94800.0,
    entry_bar_close=95200.0,
    entry_time="2025-06-01T10:00:00",
    sl_price=0,
    tp_price=95100.0,
)
test("Reject: no SL", not result.is_valid)

# Reject: no TP
result = validate_trade(
    strategy_id="TEST_04",
    direction="BUY",
    signal_bar_index=100,
    entry_bar_index=101,
    entry_bar_open=95000.0,
    entry_bar_high=95500.0,
    entry_bar_low=94800.0,
    entry_bar_close=95200.0,
    entry_time="2025-06-01T10:00:00",
    sl_price=94950.0,
    tp_price=0,
)
test("Reject: no TP", not result.is_valid)

# Reject: SL too close (< 20 pips)
result = validate_trade(
    strategy_id="TEST_05",
    direction="BUY",
    signal_bar_index=100,
    entry_bar_index=101,
    entry_bar_open=95000.0,
    entry_bar_high=95500.0,
    entry_bar_low=94800.0,
    entry_bar_close=95200.0,
    entry_time="2025-06-01T10:00:00",
    sl_price=94995.0,     # Only ~5 pips from entry
    tp_price=95100.0,
)
test("Reject: SL < 20 pips", not result.is_valid)
if not result.is_valid:
    test("Rejection mentions stop level", "too close" in result.rejection_reason.lower(),
         result.rejection_reason)

# Reject: TP too close (< 20 pips)
result = validate_trade(
    strategy_id="TEST_06",
    direction="BUY",
    signal_bar_index=100,
    entry_bar_index=101,
    entry_bar_open=95000.0,
    entry_bar_high=95500.0,
    entry_bar_low=94800.0,
    entry_bar_close=95200.0,
    entry_time="2025-06-01T10:00:00",
    sl_price=94950.0,
    tp_price=95015.0,     # Only ~15 pips from entry
)
test("Reject: TP < 20 pips", not result.is_valid)

# Reject: lookahead bias
result = validate_trade(
    strategy_id="TEST_07",
    direction="BUY",
    signal_bar_index=101,
    entry_bar_index=101,    # Same bar = lookahead!
    entry_bar_open=95000.0,
    entry_bar_high=95500.0,
    entry_bar_low=94800.0,
    entry_bar_close=95200.0,
    entry_time="2025-06-01T10:00:00",
    sl_price=94950.0,
    tp_price=95100.0,
)
test("Reject: lookahead bias", not result.is_valid)
if not result.is_valid:
    test("Rejection mentions lookahead", "lookahead" in result.rejection_reason.lower(),
         result.rejection_reason)

# Reject: SL on wrong side (BUY with SL above entry)
result = validate_trade(
    strategy_id="TEST_08",
    direction="BUY",
    signal_bar_index=100,
    entry_bar_index=101,
    entry_bar_open=95000.0,
    entry_bar_high=95500.0,
    entry_bar_low=94800.0,
    entry_bar_close=95200.0,
    entry_time="2025-06-01T10:00:00",
    sl_price=95100.0,     # SL above entry for a BUY
    tp_price=95200.0,
)
test("Reject: SL wrong side (BUY)", not result.is_valid)


# ═══ SUMMARY ═══
print("\n" + "=" * 60)
print(f"PHASE 1 RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED! Ready for Phase 2.")
else:
    print(f"WARNING: {failed} tests failed. Fix before proceeding.")
print("=" * 60)

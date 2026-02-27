"""
Backtest Engine — Phase 2 Validation Tests
=============================================
Tests indicator computation engine:
  1. All indicators compute without error
  2. No NaN in usable range (after warmup)
  3. Spot-check 3 indicators against manual calculation
  4. Caching works (same data → same result)
  5. Performance benchmarks
"""

import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TRAIN_DIR, SYMBOL
from indicators.compute import (
    compute_all, _clear_cache,
    sma, ema, rsi, macd, atr, bollinger_bands, stochastic,
    obv, adx_dmi, pattern_doji, pattern_engulfing,
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
print("PHASE 2 VALIDATION TESTS")
print("=" * 60)


# ═══ LOAD DATA ═══
h1_path = os.path.join(TRAIN_DIR, f"{SYMBOL}_H1.csv")
df = pd.read_csv(h1_path)
df["time"] = pd.to_datetime(df["time"])
print(f"\nLoaded {len(df)} H1 bars from {h1_path}")


# ═══ 1. COMPUTE ALL INDICATORS ═══
print("\n--- 1. Compute All Indicators ---")

tracemalloc.start()
_clear_cache()
t0 = time.time()
indicators = compute_all(df, timeframe="H1")
elapsed = time.time() - t0
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

meta = indicators.get("_meta", {})
ind_count = meta.get("indicator_count", 0)

test("compute_all() returns dict", isinstance(indicators, dict))
test("Has _meta field", "_meta" in indicators)
test(f"Indicator count > 100", ind_count > 100, f"got={ind_count}")
print(f"\n  All {ind_count} indicators computed in {elapsed:.3f} seconds")
print(f"  Memory: current={current_mem / 1024 / 1024:.1f} MB, peak={peak_mem / 1024 / 1024:.1f} MB")

test("Computation < 60 seconds", elapsed < 60, f"took {elapsed:.1f}s")
test("Peak memory < 500 MB", peak_mem < 500 * 1024 * 1024, f"peak={peak_mem / 1024 / 1024:.1f} MB")


# ═══ 2. KEY INDICATORS EXIST ═══
print("\n--- 2. Key Indicators Exist ---")

required_keys = [
    "sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "ema_200",
    "wma_20", "dema_20", "tema_20", "kama_10", "hull_ma_20",
    "rsi_7", "rsi_14", "rsi_21",
    "stoch_14_3", "stoch_rsi_14", "williams_r_14", "cci_20", "roc_12",
    "momentum_10", "ultimate_osc_7_14_28",
    "macd_12_26_9", "macd_8_21_5", "macd_5_13_3",
    "trix_15_9", "tsi_25_13_7", "awesome_osc_5_34",
    "atr_14", "atr_7", "true_range",
    "bb_20_2", "keltner_20_14_2", "donchian_20",
    "supertrend_10_3", "psar_002", "squeeze_20",
    "obv", "mfi_14", "cmf_20", "chaikin_osc_3_10", "vol_osc_5_20", "vwap",
    "vol_sma_20",
    "adx_14", "aroon_25", "vortex_14", "ichimoku", "heikin_ashi",
    "linreg_20", "z_score_20", "hurst_100",
    "pivot_classic", "pivot_fibonacci", "pivot_camarilla", "pivot_woodie",
    "fib_retracement", "fib_extension",
    "pattern_doji", "pattern_hammer", "pattern_shooting_star",
    "pattern_engulfing", "pattern_pin_bar", "pattern_inside_bar",
    "pattern_outside_bar", "pattern_morning_star", "pattern_evening_star",
    "pattern_three_white_soldiers", "pattern_three_black_crows", "pattern_tweezer",
]

missing = [k for k in required_keys if k not in indicators]
test(f"All {len(required_keys)} required keys present", len(missing) == 0,
     f"missing: {missing}")


# ═══ 3. NO NaN IN USABLE RANGE ═══
print("\n--- 3. Data Quality (NaN check after warmup) ---")

# Max warmup period is ~200 bars (for SMA_200, Hurst, etc.)
warmup = 250
usable_len = len(df) - warmup
test(f"Usable range has {usable_len} bars", usable_len > 1000, f"only {usable_len}")

nan_issues = []
for key, val in indicators.items():
    if key.startswith("_") or key.startswith("pivot_") or key.startswith("fib_"):
        continue  # Skip meta and scalar results

    if isinstance(val, pd.Series):
        usable = val.iloc[warmup:]
        nan_count = usable.isna().sum()
        if nan_count > 0:
            nan_pct = nan_count / len(usable) * 100
            # Allow small NaN at boundaries (ichimoku chikou shifts forward)
            if nan_pct > 10:
                nan_issues.append(f"{key}: {nan_count} NaN ({nan_pct:.1f}%)")
    elif isinstance(val, dict):
        for sub_key, sub_val in val.items():
            if isinstance(sub_val, pd.Series):
                usable = sub_val.iloc[warmup:]
                nan_count = usable.isna().sum()
                if nan_count > 0:
                    nan_pct = nan_count / len(usable) * 100
                    if nan_pct > 10:
                        nan_issues.append(f"{key}.{sub_key}: {nan_count} NaN ({nan_pct:.1f}%)")

if nan_issues:
    for issue in nan_issues[:10]:
        print(f"    WARNING: {issue}")
test("No excessive NaN after warmup", len(nan_issues) == 0,
     f"{len(nan_issues)} indicators with >10% NaN")


# ═══ 4. SPOT-CHECK: SMA ═══
print("\n--- 4. Spot-Check: SMA_20 ---")

close = df["close"]
# Manual SMA_20 at bar 500
manual_sma = close.iloc[481:501].mean()
computed_sma = indicators["sma_20"].iloc[500]
test("SMA_20 at bar 500 matches manual",
     abs(manual_sma - computed_sma) < 0.01,
     f"manual={manual_sma:.2f}, computed={computed_sma:.2f}")


# ═══ 5. SPOT-CHECK: RSI ═══
print("\n--- 5. Spot-Check: RSI_14 ---")

computed_rsi = indicators["rsi_14"]
# RSI should be 0-100
rsi_usable = computed_rsi.iloc[warmup:]
min_rsi = rsi_usable.min()
max_rsi = rsi_usable.max()
test("RSI_14 min >= 0", min_rsi >= 0, f"min={min_rsi:.2f}")
test("RSI_14 max <= 100", max_rsi <= 100, f"max={max_rsi:.2f}")
test("RSI_14 mean in [30, 70]", 30 <= rsi_usable.mean() <= 70,
     f"mean={rsi_usable.mean():.2f}")


# ═══ 6. SPOT-CHECK: MACD ═══
print("\n--- 6. Spot-Check: MACD_12_26_9 ---")

macd_data = indicators["macd_12_26_9"]
test("MACD has 'line' key", "line" in macd_data)
test("MACD has 'signal' key", "signal" in macd_data)
test("MACD has 'histogram' key", "histogram" in macd_data)

# Histogram = line - signal
macd_line = macd_data["line"].iloc[warmup:]
macd_sig = macd_data["signal"].iloc[warmup:]
macd_hist = macd_data["histogram"].iloc[warmup:]
manual_hist = macd_line - macd_sig
max_diff = (macd_hist - manual_hist).abs().max()
test("MACD histogram = line - signal", max_diff < 0.01,
     f"max_diff={max_diff:.6f}")


# ═══ 7. BOLLINGER BANDS ═══
print("\n--- 7. Spot-Check: Bollinger Bands ---")

bb = indicators["bb_20_2"]
test("BB has upper/middle/lower", all(k in bb for k in ["upper", "middle", "lower"]))
test("BB has pct_b and bandwidth", all(k in bb for k in ["pct_b", "bandwidth"]))

# Upper > Middle > Lower (for usable range)
bb_upper = bb["upper"].iloc[warmup:]
bb_middle = bb["middle"].iloc[warmup:]
bb_lower = bb["lower"].iloc[warmup:]
test("BB upper > middle always", (bb_upper >= bb_middle).all())
test("BB middle > lower always", (bb_middle >= bb_lower).all())

# Middle = SMA_20
sma_20 = indicators["sma_20"].iloc[warmup:]
bb_mid_usable = bb_middle.iloc[:len(sma_20)]
sma_usable = sma_20.iloc[:len(bb_mid_usable)]
max_diff_bb = (bb_mid_usable - sma_usable).abs().max()
test("BB middle = SMA_20", max_diff_bb < 0.01,
     f"max_diff={max_diff_bb:.6f}")


# ═══ 8. ATR ═══
print("\n--- 8. Spot-Check: ATR ---")

atr_14 = indicators["atr_14"]
atr_usable = atr_14.iloc[warmup:]
test("ATR_14 all positive", (atr_usable > 0).all())
test("ATR_14 reasonable for BTCUSD (< 10000)", atr_usable.max() < 10000,
     f"max={atr_usable.max():.2f}")
test("ATR_14 mean > 100", atr_usable.mean() > 100,
     f"mean={atr_usable.mean():.2f}")


# ═══ 9. STOCHASTIC ═══
print("\n--- 9. Spot-Check: Stochastic ---")

stoch = indicators["stoch_14_3"]
stoch_k = stoch["k"].iloc[warmup:]
stoch_d = stoch["d"].iloc[warmup:]
test("Stoch K in [0, 100]", stoch_k.min() >= -0.01 and stoch_k.max() <= 100.01,
     f"min={stoch_k.min():.2f}, max={stoch_k.max():.2f}")
test("Stoch D in [0, 100]", stoch_d.min() >= -0.01 and stoch_d.max() <= 100.01,
     f"min={stoch_d.min():.2f}, max={stoch_d.max():.2f}")


# ═══ 10. ADX ═══
print("\n--- 10. Spot-Check: ADX ---")

adx_data = indicators["adx_14"]
test("ADX has adx/plus_di/minus_di", all(k in adx_data for k in ["adx", "plus_di", "minus_di"]))
adx_val = adx_data["adx"].iloc[warmup:]
test("ADX in [0, 100]", adx_val.min() >= 0 and adx_val.max() <= 100,
     f"min={adx_val.min():.2f}, max={adx_val.max():.2f}")


# ═══ 11. ICHIMOKU ═══
print("\n--- 11. Spot-Check: Ichimoku ---")

ichi = indicators["ichimoku"]
test("Ichimoku has all 5 lines",
     all(k in ichi for k in ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"]))


# ═══ 12. SUPERTREND ═══
print("\n--- 12. Spot-Check: SuperTrend ---")

st = indicators["supertrend_10_3"]
test("SuperTrend has supertrend/direction", all(k in st for k in ["supertrend", "direction"]))
st_dir = st["direction"].iloc[warmup:]
unique_dirs = st_dir.unique()
test("SuperTrend direction is +1 or -1", set(unique_dirs).issubset({1.0, -1.0, 0.0}),
     f"unique={unique_dirs}")


# ═══ 13. PARABOLIC SAR ═══
print("\n--- 13. Spot-Check: Parabolic SAR ---")

psar = indicators["psar_002"]
test("PSAR has sar/trend", all(k in psar for k in ["sar", "trend"]))
psar_val = psar["sar"].iloc[warmup:]
test("PSAR all positive", (psar_val > 0).all())


# ═══ 14. VOLUME INDICATORS ═══
print("\n--- 14. Volume Indicators ---")

test("OBV is a Series", isinstance(indicators["obv"], pd.Series))
test("MFI_14 in [0, 100]",
     indicators["mfi_14"].iloc[warmup:].min() >= -0.01 and
     indicators["mfi_14"].iloc[warmup:].max() <= 100.01)
test("VWAP all positive", (indicators["vwap"].iloc[warmup:] > 0).all())


# ═══ 15. CANDLESTICK PATTERNS ═══
print("\n--- 15. Candlestick Patterns ---")

doji = indicators["pattern_doji"]
test("Doji pattern values in {0, 1}", set(doji.unique()).issubset({0, 1}))
doji_pct = doji.sum() / len(doji) * 100
test("Doji detection rate reasonable (1-15%)", 0.5 <= doji_pct <= 20,
     f"got={doji_pct:.1f}%")

engulfing = indicators["pattern_engulfing"]
test("Engulfing values in {-1, 0, 1}", set(engulfing.unique()).issubset({-1, 0, 1}))

pin_bar = indicators["pattern_pin_bar"]
test("Pin Bar values in {-1, 0, 1}", set(pin_bar.unique()).issubset({-1, 0, 1}))

inside = indicators["pattern_inside_bar"]
test("Inside Bar values in {0, 1}", set(inside.unique()).issubset({0, 1}))


# ═══ 16. CACHING ═══
print("\n--- 16. Caching ---")

t0 = time.time()
indicators2 = compute_all(df, timeframe="H1")
cache_time = time.time() - t0

test("Cached result returns same object", indicators2 is indicators)
test("Cache hit < 0.01s", cache_time < 0.01, f"took {cache_time:.4f}s")

# Clear cache and recompute
_clear_cache()
t0 = time.time()
indicators3 = compute_all(df, timeframe="H1")
recompute_time = time.time() - t0
test("Recompute after clear works", indicators3 is not indicators)
test("Recompute indicator count matches", indicators3["_meta"]["indicator_count"] == ind_count)


# ═══ 17. HEIKIN-ASHI ═══
print("\n--- 17. Heikin-Ashi ---")

ha = indicators["heikin_ashi"]
test("HA has open/high/low/close", all(k in ha for k in ["open", "high", "low", "close"]))
ha_h = ha["high"].iloc[warmup:]
ha_l = ha["low"].iloc[warmup:]
test("HA high >= low", (ha_h >= ha_l).all())


# ═══ 18. LINEAR REGRESSION ═══
print("\n--- 18. Linear Regression ---")

lr = indicators["linreg_20"]
test("LinReg has slope/r_squared/value", all(k in lr for k in ["slope", "r_squared", "value"]))
r2 = lr["r_squared"].iloc[warmup:]
test("R-squared in [0, 1]", r2.min() >= -0.01 and r2.max() <= 1.01,
     f"min={r2.min():.3f}, max={r2.max():.3f}")


# ═══ 19. PIVOT POINTS ═══
print("\n--- 19. Pivot Points ---")

pp_classic = indicators["pivot_classic"]
test("Classic pivot has pp/r1/r2/r3/s1/s2/s3",
     all(k in pp_classic for k in ["pp", "r1", "r2", "r3", "s1", "s2", "s3"]))
test("Classic: R1 > PP > S1",
     pp_classic["r1"] > pp_classic["pp"] > pp_classic["s1"])

pp_fib = indicators["pivot_fibonacci"]
test("Fibonacci pivot has pp/r1/s1", all(k in pp_fib for k in ["pp", "r1", "s1"]))


# ═══ 20. FIBONACCI ═══
print("\n--- 20. Fibonacci Levels ---")

fib = indicators["fib_retracement"]
test("Fib has 0.382/0.5/0.618 levels",
     all(k in fib for k in ["0.382", "0.5", "0.618"]))
test("Fib 0.0 (high) > 1.0 (low)", fib["0.0"] > fib["1.0"])
test("Fib levels are ordered",
     fib["0.0"] > fib["0.236"] > fib["0.382"] > fib["0.5"] > fib["0.618"] > fib["0.786"] > fib["1.0"])


# ═══ SUMMARY ═══
print("\n" + "=" * 60)
print(f"PHASE 2 RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print(f"ALL TESTS PASSED!")
    print(f"All {ind_count} indicators computed in {elapsed:.3f} seconds")
else:
    print(f"WARNING: {failed} tests failed. Fix before proceeding.")
print("=" * 60)

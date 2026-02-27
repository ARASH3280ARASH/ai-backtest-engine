"""
Backtest Engine — MT5 Data Extractor
=======================================
Extracts BTCUSD tick and candle data from MetaTrader 5.
Saves raw data, then splits chronologically into train/validation/test sets.

Usage:
    python data/mt5_extractor.py
"""

import json
import os
import sys
import time
import pickle
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.broker import BTCUSD_CONFIG
from config.settings import (
    RAW_DIR, PROCESSED_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR,
    TIMEFRAMES, DATA_SPLITS, SPLITS_META_PATH, SYMBOL,
    MIN_MONTHS_HISTORY, TARGET_MONTHS_HISTORY,
)

# MT5 timeframe constants
MT5_TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def connect_mt5() -> bool:
    """Connect to MT5 terminal."""
    if mt5.terminal_info() is not None:
        info = mt5.terminal_info()
        if info.connected:
            print(f"  MT5 already connected")
            return True

    path = BTCUSD_CONFIG["mt5_path"]
    ok = mt5.initialize(path=path)
    if not ok:
        print(f"  MT5 init failed: {mt5.last_error()}")
        return False

    acct = mt5.account_info()
    if acct:
        print(f"  MT5 connected — Account {acct.login}")
    return True


def extract_candles(symbol: str, timeframe_name: str, months: int) -> pd.DataFrame:
    """Extract candle data for a specific timeframe."""
    tf = MT5_TF_MAP.get(timeframe_name)
    if tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe_name}")

    now = datetime.now(timezone.utc)
    from_date = now - timedelta(days=months * 30)

    print(f"  Extracting {symbol} {timeframe_name} from {from_date.date()} to {now.date()}...")

    for attempt in range(3):
        rates = mt5.copy_rates_range(symbol, tf, from_date, now)
        if rates is not None and len(rates) > 0:
            break
        print(f"    Retry {attempt + 1}/3...")
        time.sleep(2)
    else:
        print(f"    FAILED to get {timeframe_name} data")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "tick_volume", "real_volume": "real_volume"})

    # Validate
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("time").reset_index(drop=True)

    # Remove zero-price rows
    df = df[(df["close"] > 0) & (df["open"] > 0)]

    # Validate OHLC consistency
    df = df[(df["high"] >= df["low"]) & (df["high"] >= df["open"]) & (df["high"] >= df["close"])]
    df = df[(df["low"] <= df["open"]) & (df["low"] <= df["close"])]

    print(f"    Got {len(df)} candles ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")
    return df


def extract_ticks(symbol: str, months: int) -> pd.DataFrame:
    """Extract tick data. Limited by MT5's tick history depth."""
    now = datetime.now(timezone.utc)

    # MT5 typically limits tick history to ~1 month for most brokers.
    # Try progressively shorter ranges if full range fails.
    ranges_to_try = [months * 30, 90, 30, 14, 7]

    for days in ranges_to_try:
        from_date = now - timedelta(days=days)
        print(f"  Extracting {symbol} ticks ({days} days)...")

        ticks = mt5.copy_ticks_range(symbol, from_date, now, mt5.COPY_TICKS_ALL)
        if ticks is not None and len(ticks) > 0:
            df = pd.DataFrame(ticks)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            df = df.dropna(subset=["bid", "ask"])
            df = df[(df["bid"] > 0) & (df["ask"] > 0)]
            print(f"    Got {len(df):,} ticks ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")
            return df
        print(f"    No tick data for {days} days range")

    print("    WARNING: No tick data available")
    return pd.DataFrame()


def save_raw_data(df: pd.DataFrame, name: str):
    """Save DataFrame as both CSV and pickle."""
    if df.empty:
        print(f"    Skipping save for empty {name}")
        return

    csv_path = os.path.join(RAW_DIR, f"{name}.csv")
    pkl_path = os.path.join(RAW_DIR, f"{name}.pkl")

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    size_mb = os.path.getsize(csv_path) / 1024 / 1024
    print(f"    Saved {name}: {len(df):,} rows ({size_mb:.1f} MB CSV)")


def split_chronological(df: pd.DataFrame, time_col: str = "time") -> dict:
    """
    Split DataFrame chronologically: 70% train, 15% validation, 15% test.
    Returns dict with DataFrames and boundary dates.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)

    train_end = int(n * DATA_SPLITS["train"])
    val_end = int(n * (DATA_SPLITS["train"] + DATA_SPLITS["validation"]))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "boundaries": {
            "train_start": str(train_df[time_col].iloc[0]) if len(train_df) > 0 else "",
            "train_end": str(train_df[time_col].iloc[-1]) if len(train_df) > 0 else "",
            "validation_start": str(val_df[time_col].iloc[0]) if len(val_df) > 0 else "",
            "validation_end": str(val_df[time_col].iloc[-1]) if len(val_df) > 0 else "",
            "test_start": str(test_df[time_col].iloc[0]) if len(test_df) > 0 else "",
            "test_end": str(test_df[time_col].iloc[-1]) if len(test_df) > 0 else "",
            "train_count": len(train_df),
            "validation_count": len(val_df),
            "test_count": len(test_df),
        },
    }


def save_split(split_data: dict, tf_name: str, time_col: str = "time"):
    """Save split DataFrames to processed directories."""
    for split_name, dir_path in [("train", TRAIN_DIR), ("validation", VALIDATION_DIR), ("test", TEST_DIR)]:
        df = split_data[split_name]
        if df.empty:
            continue
        csv_path = os.path.join(dir_path, f"{SYMBOL}_{tf_name}.csv")
        pkl_path = os.path.join(dir_path, f"{SYMBOL}_{tf_name}.pkl")
        df.to_csv(csv_path, index=False)
        df.to_pickle(pkl_path)

    print(f"    Split {tf_name}: train={len(split_data['train'])}, val={len(split_data['validation'])}, test={len(split_data['test'])}")


def run_extraction():
    """Main extraction pipeline."""
    print("=" * 60)
    print("BTCUSD DATA EXTRACTION")
    print("=" * 60)

    # Ensure directories exist
    for d in [RAW_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    # Connect
    print("\n--- Connecting to MT5 ---")
    if not connect_mt5():
        print("FATAL: Cannot connect to MT5")
        sys.exit(1)

    # Check symbol
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        print(f"FATAL: Symbol {SYMBOL} not found on MT5")
        mt5.shutdown()
        sys.exit(1)
    if not info.visible:
        mt5.symbol_select(SYMBOL, True)
    print(f"  Symbol: {SYMBOL} | digits={info.digits} | point={info.point}")

    splits_meta = {"symbol": SYMBOL, "extracted_at": datetime.now(timezone.utc).isoformat(), "timeframes": {}}

    # Extract candle data for each timeframe
    print("\n--- Extracting Candle Data ---")
    for tf_name in TIMEFRAMES:
        df = extract_candles(SYMBOL, tf_name, TARGET_MONTHS_HISTORY)
        if df.empty:
            # Fallback to minimum
            df = extract_candles(SYMBOL, tf_name, MIN_MONTHS_HISTORY)

        if df.empty:
            print(f"  WARNING: No data for {tf_name}")
            continue

        # Save raw
        save_raw_data(df, f"{SYMBOL}_{tf_name}")

        # Split and save
        split = split_chronological(df, "time")
        save_split(split, tf_name)

        splits_meta["timeframes"][tf_name] = {
            "total_bars": len(df),
            "date_range": f"{df['time'].iloc[0]} to {df['time'].iloc[-1]}",
            **split["boundaries"],
        }

    # Extract tick data
    print("\n--- Extracting Tick Data ---")
    tick_df = extract_ticks(SYMBOL, TARGET_MONTHS_HISTORY)
    if not tick_df.empty:
        save_raw_data(tick_df, f"{SYMBOL}_ticks")
        tick_split = split_chronological(tick_df, "time")
        save_split(tick_split, "ticks")
        splits_meta["timeframes"]["ticks"] = {
            "total_ticks": len(tick_df),
            "date_range": f"{tick_df['time'].iloc[0]} to {tick_df['time'].iloc[-1]}",
            **tick_split["boundaries"],
        }

    # Save split metadata
    os.makedirs(os.path.dirname(SPLITS_META_PATH), exist_ok=True)
    with open(SPLITS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(splits_meta, f, indent=2, default=str)
    print(f"\n  Split metadata saved to {SPLITS_META_PATH}")

    mt5.shutdown()

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for tf, info in splits_meta.get("timeframes", {}).items():
        total = info.get("total_bars", info.get("total_ticks", 0))
        train = info.get("train_count", 0)
        val = info.get("validation_count", 0)
        test = info.get("test_count", 0)
        print(f"  {tf:6s}: {total:>8,} total | train={train:>7,} val={val:>6,} test={test:>6,}")
    print("=" * 60)


if __name__ == "__main__":
    run_extraction()

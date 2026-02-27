"""
Backtest Engine — Global Settings
===================================
Paths, data split ratios, timeframes, and defaults.
"""

import os

# Project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DIR, "validation")
TEST_DIR = os.path.join(PROCESSED_DIR, "test")
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "cache")

# Results directories
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
INDIVIDUAL_DIR = os.path.join(RESULTS_DIR, "individual")
COMBOS_DIR = os.path.join(RESULTS_DIR, "combos")
OPTIMIZED_DIR = os.path.join(RESULTS_DIR, "optimized")
FINAL_DIR = os.path.join(RESULTS_DIR, "final")

# Reports
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")

# Data split configuration (chronological)
DATA_SPLITS = {
    "train": 0.70,       # 70% oldest data
    "validation": 0.15,  # 15% middle data
    "test": 0.15,        # 15% newest data (NEVER touch until Phase 5)
}

# Timeframes to extract
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

# MT5 timeframe mapping
MT5_TIMEFRAMES = {
    "M1": 1,    # TIMEFRAME_M1
    "M5": 5,    # TIMEFRAME_M5
    "M15": 15,  # TIMEFRAME_M15
    "M30": 30,  # TIMEFRAME_M30
    "H1": 16385,  # TIMEFRAME_H1
    "H4": 16388,  # TIMEFRAME_H4
    "D1": 16408,  # TIMEFRAME_D1
}

# Data extraction settings
MIN_MONTHS_HISTORY = 6
TARGET_MONTHS_HISTORY = 12

# Split metadata file
SPLITS_META_PATH = os.path.join(PROJECT_DIR, "config", "data_splits.json")

# Symbol
SYMBOL = "BTCUSD"

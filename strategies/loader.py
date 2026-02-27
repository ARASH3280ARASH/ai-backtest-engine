"""
Backtest Engine — Strategy Auto-Discovery Loader
===================================================
Aggressively scans the MVP project for ALL strategy definitions.

Scan sources:
  1. *_strategies.py files → look for *_STRATEGIES lists
  2. cat*.py files → look for BaseStrategy classes
  3. orchestrator.py → import ALL_STRATEGIES directly
  4. advanced_oscillators.py, ai_combo_strategies.py, etc.
  5. Any .py file with dicts containing "id" + "func" keys

For each strategy found, extracts:
  - id, name, name_fa, category, source_file, func, func_type
"""

import os
import sys
import ast
import importlib
import inspect
import re
from typing import List, Dict, Any, Optional

# MVP project root
MVP_ROOT = r"C:\Users\Administrator\Desktop\mvp"
MVP_STRATEGIES_DIR = os.path.join(MVP_ROOT, "backend", "strategies")

# Ensure MVP is importable
if MVP_ROOT not in sys.path:
    sys.path.insert(0, MVP_ROOT)


def _detect_func_type(func) -> str:
    """Detect whether a function uses (df, context) or (df, ind, symbol, tf) signature."""
    try:
        params = list(inspect.signature(func).parameters.keys())
        if len(params) >= 4:
            return "extended"
        return "simple"
    except (ValueError, TypeError):
        return "simple"


def _extract_category(strategy_id: str) -> str:
    """Extract category prefix from strategy ID like RSI_01 → RSI."""
    if "_" in strategy_id:
        return strategy_id.rsplit("_", 1)[0]
    return strategy_id


def _guess_required_indicators(strategy_id: str) -> List[str]:
    """Guess what indicators a strategy needs based on its category."""
    cat = _extract_category(strategy_id).upper()
    mapping = {
        "RSI": ["rsi_14", "rsi_7", "rsi_21"],
        "MACD": ["macd_12_26_9"],
        "STOCH": ["stoch_14_3"],
        "BB": ["bb_20_2"],
        "MA": ["sma_20", "sma_50", "ema_20", "ema_50", "ema_200"],
        "ICH": ["ichimoku"],
        "ADX": ["adx_14"],
        "CDL": [],  # candlestick — uses raw OHLC
        "DIV": ["rsi_14", "macd_12_26_9"],
        "VOL": ["obv", "mfi_14"],
        "FIB": ["atr_14"],
        "SM": ["atr_14"],
        "CCI": ["cci_20"],
        "WR": ["williams_r_14"],
        "ATR": ["atr_14"],
        "MOM": ["momentum_10"],
        "PIVOT": ["atr_14"],
        "MTF": ["ema_50", "rsi_14"],
        "PA": ["atr_14"],
        "DON": ["donchian_20"],
        "KC": ["keltner_20_14_2"],
        "ENV": ["ema_20"],
        "PSAR": ["psar_002"],
        "REG": ["linreg_20"],
        "TRIX": ["trix_15_9"],
        "ROC": ["roc_12"],
        "CMO": [],
        "RVI": [],
        "PPO": [],
        "OBV": ["obv"],
        "MFI": ["mfi_14"],
        "VWAP": ["vwap"],
        "AD": ["obv"],
        "FI": [],
        "STREND": ["supertrend_10_3"],
        "AROON": ["aroon_25"],
        "DPO": [],
        "VORTEX": ["vortex_14"],
        "ELDER": ["atr_14"],
        "FISHER": [],
        "HEIKIN": ["heikin_ashi"],
        "CHOP": [],
        "MASS": [],
        "GAPS": ["atr_14"],
        "RANGE": ["atr_14", "bb_20_2"],
        "SWING": ["atr_14"],
        "HARMONIC": ["atr_14"],
        "COMBO": ["rsi_14", "macd_12_26_9", "bb_20_2", "atr_14"],
        "SRSI": ["stoch_rsi_14"],
        "ARN": ["aroon_25"],
        "VTX": ["vortex_14"],
        "ULT": ["ultimate_osc_7_14_28"],
        "KST": [],
        "CH": [],
        "GAP": ["atr_14"],
        "MS": ["atr_14"],
        "WYC": [],
        "SNT": [],
        "COR": [],
        "AIC": ["rsi_14", "macd_12_26_9", "atr_14"],
        "ADP": ["atr_14", "rsi_14"],
        "CP": ["rsi_14", "atr_14"],
        "EW": [],
        "TSI": [],
        "SQZ": ["bb_20_2", "keltner_20_14_2"],
        "ALLI": [],
        "KAMA": ["kama_10"],
        "ZLEMA": [],
    }
    return mapping.get(cat, ["atr_14"])


def scan_strategies_from_orchestrator() -> List[Dict[str, Any]]:
    """
    Primary method: Import ALL_STRATEGIES from the MVP orchestrator.
    This is the most reliable source — the orchestrator already aggregates everything.
    """
    strategies = []
    try:
        from backend.strategies.orchestrator import ALL_STRATEGIES
        for s in ALL_STRATEGIES:
            sid = s.get("id", "")
            func = s.get("func")
            if not sid or not func:
                continue
            cat = _extract_category(sid)
            ft = _detect_func_type(func)
            strategies.append({
                "id": sid,
                "name": s.get("name", sid),
                "name_fa": s.get("name_fa", s.get("name", sid)),
                "category": cat,
                "source_file": "orchestrator.py",
                "params": {},
                "required_indicators": _guess_required_indicators(sid),
                "has_buy_signal": True,
                "has_sell_signal": True,
                "analyze_function": func,
                "func_type": ft,
            })
    except Exception as e:
        print(f"[LOADER] Warning: orchestrator import failed: {e}")

    return strategies


def scan_strategy_files() -> List[Dict[str, Any]]:
    """
    Secondary method: Scan individual *_strategies.py files for *_STRATEGIES lists.
    Falls back to this if orchestrator import has issues.
    """
    strategies = []
    seen_ids = set()

    if not os.path.isdir(MVP_STRATEGIES_DIR):
        print(f"[LOADER] Warning: {MVP_STRATEGIES_DIR} not found")
        return strategies

    for fname in os.listdir(MVP_STRATEGIES_DIR):
        if not fname.endswith(".py"):
            continue
        if fname.startswith("__"):
            continue

        fpath = os.path.join(MVP_STRATEGIES_DIR, fname)

        # Read file and look for *_STRATEGIES list variables
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        # Find variable names ending with _STRATEGIES
        pattern = re.compile(r'^([A-Z_]+_STRATEGIES)\s*=\s*\[', re.MULTILINE)
        matches = pattern.findall(content)

        if not matches:
            continue

        # Import the module
        module_name = f"backend.strategies.{fname[:-3]}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"[LOADER] Warning: Cannot import {module_name}: {e}")
            continue

        for var_name in matches:
            strat_list = getattr(mod, var_name, None)
            if not isinstance(strat_list, list):
                continue

            for s in strat_list:
                if not isinstance(s, dict):
                    continue
                sid = s.get("id", "")
                func = s.get("func")
                if not sid or not func:
                    continue
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)

                cat = _extract_category(sid)
                ft = _detect_func_type(func)
                strategies.append({
                    "id": sid,
                    "name": s.get("name", sid),
                    "name_fa": s.get("name_fa", s.get("name", sid)),
                    "category": cat,
                    "source_file": fname,
                    "params": {},
                    "required_indicators": _guess_required_indicators(sid),
                    "has_buy_signal": True,
                    "has_sell_signal": True,
                    "analyze_function": func,
                    "func_type": ft,
                })

    return strategies


def scan_cat_files() -> List[Dict[str, Any]]:
    """
    Tertiary method: Scan cat*.py files for class-based strategies.
    These use BaseStrategy pattern with analyze() method.
    """
    strategies = []

    if not os.path.isdir(MVP_STRATEGIES_DIR):
        return strategies

    for fname in os.listdir(MVP_STRATEGIES_DIR):
        if not fname.startswith("cat") or not fname.endswith(".py"):
            continue

        fpath = os.path.join(MVP_STRATEGIES_DIR, fname)
        module_name = f"backend.strategies.{fname[:-3]}"

        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        # Look for classes with STRATEGY_ID attribute or analyze method
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if not inspect.isclass(obj):
                continue
            if hasattr(obj, "STRATEGY_ID") and hasattr(obj, "analyze"):
                sid = getattr(obj, "STRATEGY_ID", "")
                if not sid:
                    continue
                cat = _extract_category(sid)

                # Create wrapper function
                instance = None
                try:
                    instance = obj()
                except Exception:
                    continue

                def _make_wrapper(inst):
                    def wrapper(df, context=None):
                        return inst.analyze(df, context or {})
                    return wrapper

                strategies.append({
                    "id": sid,
                    "name": getattr(obj, "STRATEGY_NAME_EN", sid),
                    "name_fa": getattr(obj, "STRATEGY_NAME_FA", sid),
                    "category": cat,
                    "source_file": fname,
                    "params": {},
                    "required_indicators": getattr(obj, "REQUIRED_INDICATORS", []),
                    "has_buy_signal": True,
                    "has_sell_signal": True,
                    "analyze_function": _make_wrapper(instance),
                    "func_type": "simple",
                })

    return strategies


def discover_all_strategies(verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Master discovery function. Tries all scan methods and deduplicates.

    Priority:
      1. Orchestrator (most authoritative — already aggregates everything)
      2. Individual strategy files (catches any not in orchestrator)
      3. Cat files (class-based strategies)

    Returns:
        List of strategy dicts with all metadata.
    """
    all_strats = []
    seen_ids = set()

    # Method 1: Orchestrator
    if verbose:
        print("[LOADER] Scanning orchestrator.py ...")
    orch_strats = scan_strategies_from_orchestrator()
    for s in orch_strats:
        if s["id"] not in seen_ids:
            seen_ids.add(s["id"])
            all_strats.append(s)
    if verbose:
        print(f"  Found {len(orch_strats)} from orchestrator")

    # Method 2: Individual files
    if verbose:
        print("[LOADER] Scanning *_strategies.py files ...")
    file_strats = scan_strategy_files()
    added = 0
    for s in file_strats:
        if s["id"] not in seen_ids:
            seen_ids.add(s["id"])
            all_strats.append(s)
            added += 1
    if verbose:
        print(f"  Found {len(file_strats)} in files, {added} new (not in orchestrator)")

    # Method 3: Cat files
    if verbose:
        print("[LOADER] Scanning cat*.py files ...")
    cat_strats = scan_cat_files()
    added = 0
    for s in cat_strats:
        if s["id"] not in seen_ids:
            seen_ids.add(s["id"])
            all_strats.append(s)
            added += 1
    if verbose:
        print(f"  Found {len(cat_strats)} in cat files, {added} new")

    # Sort by ID
    all_strats.sort(key=lambda x: x["id"])

    if verbose:
        print(f"\n[LOADER] Total unique strategies discovered: {len(all_strats)}")

    return all_strats


if __name__ == "__main__":
    strats = discover_all_strategies(verbose=True)

    # Breakdown by category
    from collections import Counter
    cats = Counter(s["category"] for s in strats)
    print(f"\n{'=' * 60}")
    print(f"STRATEGY DISCOVERY REPORT")
    print(f"{'=' * 60}")
    print(f"Total: {len(strats)} strategies in {len(cats)} categories\n")

    for cat, count in sorted(cats.items()):
        ids = [s["id"] for s in strats if s["category"] == cat]
        print(f"  {cat} ({count}): {', '.join(ids)}")

    print(f"\n{'=' * 60}")

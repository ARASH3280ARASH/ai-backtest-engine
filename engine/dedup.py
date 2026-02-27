"""
Strategy Deduplication Utility
================================
Detects and removes duplicate strategies that produce identical trades.

Two strategies are duplicates if they have:
  - Same number of trades
  - Identical entry_bar_index sequences
  - Identical direction sequences
  - Net PnL within $0.01 per trade

Keeps the first occurrence (lowest ID number).
"""

import json
import os
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _trade_fingerprint(trades: list) -> str:
    """Create a fingerprint from a trade list for fast comparison."""
    if not trades:
        return "EMPTY"
    parts = []
    for t in trades:
        eb = t.get("entry_bar_index", 0)
        d = t.get("direction", "")
        pnl = round(float(t.get("net_pnl", 0)), 2)
        parts.append(f"{eb}:{d}:{pnl}")
    return "|".join(parts)


def _are_duplicates(trades_a: list, trades_b: list, tol: float = 0.01) -> bool:
    """Check if two trade lists are duplicates within tolerance."""
    if len(trades_a) != len(trades_b):
        return False
    if not trades_a:
        return True

    for ta, tb in zip(trades_a, trades_b):
        if ta.get("entry_bar_index") != tb.get("entry_bar_index"):
            return False
        if ta.get("direction") != tb.get("direction"):
            return False
        pnl_a = float(ta.get("net_pnl", 0))
        pnl_b = float(tb.get("net_pnl", 0))
        if abs(pnl_a - pnl_b) > tol:
            return False
    return True


def find_duplicates(results: Dict[str, dict]) -> Dict[str, str]:
    """
    Find duplicate strategies in results dict.

    Args:
        results: {strategy_id: result_dict} where result_dict has "trades" key

    Returns:
        {duplicate_id: kept_id} mapping duplicates to the strategy they duplicate
    """
    duplicates = {}
    fingerprints = {}  # fingerprint -> (strategy_id, trades)

    # Sort by ID to ensure lowest ID is kept
    sorted_ids = sorted(results.keys())

    for sid in sorted_ids:
        r = results[sid]
        if r is None:
            continue
        trades = r.get("trades", [])
        fp = _trade_fingerprint(trades)

        if fp in fingerprints:
            kept_id = fingerprints[fp][0]
            # Double-check with full comparison
            if _are_duplicates(trades, fingerprints[fp][1]):
                duplicates[sid] = kept_id
                logger.info(f"  DUPLICATE: {sid} == {kept_id} "
                           f"({len(trades)} trades, same entry/dir/pnl)")
        else:
            fingerprints[fp] = (sid, trades)

    return duplicates


def deduplicate_results(results: Dict[str, dict],
                        verbose: bool = True) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Remove duplicate strategies from results.

    Returns:
        (filtered_results, duplicates_map)
    """
    duplicates = find_duplicates(results)

    if verbose and duplicates:
        # Group by kept strategy
        groups = {}
        for dup_id, kept_id in duplicates.items():
            groups.setdefault(kept_id, []).append(dup_id)

        print(f"\n  Deduplication: {len(duplicates)} duplicates found")
        for kept_id, dup_ids in groups.items():
            trades_n = len(results[kept_id].get("trades", []))
            net = results[kept_id].get("metrics", {}).get("net_profit", 0)
            print(f"    KEEP {kept_id} ({trades_n} trades, ${net:.2f}), "
                  f"REMOVE: {', '.join(dup_ids)}")

    filtered = {k: v for k, v in results.items() if k not in duplicates}

    if verbose:
        print(f"  Result: {len(results)} -> {len(filtered)} strategies "
              f"({len(duplicates)} removed)")

    return filtered, duplicates


def deduplicate_rankings(rankings: list, duplicates: Dict[str, str]) -> list:
    """Remove duplicates from a rankings list."""
    return [r for r in rankings if r.get("strategy_id") not in duplicates]


def deduplicate_directory(results_dir: str, verbose: bool = True) -> Dict[str, str]:
    """
    Find duplicates by scanning individual result JSON files in a directory.

    Returns:
        {duplicate_id: kept_id} mapping
    """
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        sid = fname.replace(".json", "")
        path = os.path.join(results_dir, fname)
        try:
            with open(path) as f:
                results[sid] = json.load(f)
        except Exception:
            continue

    return find_duplicates(results)

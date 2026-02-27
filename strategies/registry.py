"""
Backtest Engine — Strategy Registry
======================================
Central registry that loads all discovered strategies, wraps them in
BacktestStrategy interface, and provides lookup methods.

Usage:
    registry = StrategyRegistry()
    registry.load()

    all_strats = registry.get_all()
    rsi_strats = registry.get_by_category("RSI")
    one = registry.get_by_id("RSI_01")
"""

import sys
import os
from typing import List, Dict, Optional
from collections import OrderedDict

# Ensure project root on path
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from strategies.base import BacktestStrategy
from strategies.loader import discover_all_strategies


class StrategyRegistry:
    """
    Central registry for all backtest strategies.

    Loads strategies from MVP via the loader, wraps each in BacktestStrategy,
    and provides fast lookup by id/category.
    """

    def __init__(self):
        self._strategies: Dict[str, BacktestStrategy] = OrderedDict()
        self._by_category: Dict[str, List[BacktestStrategy]] = {}
        self._load_errors: List[Dict] = []
        self._loaded = False

    def load(self, verbose: bool = False) -> "StrategyRegistry":
        """
        Discover and load all strategies.
        Call once at startup. Safe to call multiple times (idempotent).
        """
        if self._loaded:
            return self

        raw_strats = discover_all_strategies(verbose=verbose)

        for raw in raw_strats:
            try:
                strat = BacktestStrategy(
                    strategy_id=raw["id"],
                    name=raw.get("name", ""),
                    name_fa=raw.get("name_fa", ""),
                    category=raw.get("category", ""),
                    source_file=raw.get("source_file", ""),
                    params=raw.get("params", {}),
                    required_indicators=raw.get("required_indicators", []),
                    analyze_func=raw.get("analyze_function"),
                    func_type=raw.get("func_type", "simple"),
                )

                self._strategies[strat.strategy_id] = strat

                cat = strat.category
                if cat not in self._by_category:
                    self._by_category[cat] = []
                self._by_category[cat].append(strat)

            except Exception as e:
                self._load_errors.append({
                    "id": raw.get("id", "?"),
                    "error": str(e),
                })

        self._loaded = True

        if verbose:
            print(f"\n[REGISTRY] Loaded {len(self._strategies)} strategies "
                  f"in {len(self._by_category)} categories")
            if self._load_errors:
                print(f"[REGISTRY] {len(self._load_errors)} load errors:")
                for err in self._load_errors:
                    print(f"  {err['id']}: {err['error']}")

        return self

    def get_all(self) -> List[BacktestStrategy]:
        """Return all loaded strategies (ordered by ID)."""
        return list(self._strategies.values())

    def get_by_id(self, strategy_id: str) -> Optional[BacktestStrategy]:
        """Lookup a single strategy by ID."""
        return self._strategies.get(strategy_id)

    def get_by_category(self, category: str) -> List[BacktestStrategy]:
        """Return all strategies in a category (e.g., 'RSI', 'MACD')."""
        return self._by_category.get(category, [])

    def get_categories(self) -> List[str]:
        """Return all category names."""
        return sorted(self._by_category.keys())

    def get_category_counts(self) -> Dict[str, int]:
        """Return {category: count} mapping."""
        return {cat: len(strats) for cat, strats in sorted(self._by_category.items())}

    def get_ids(self) -> List[str]:
        """Return all strategy IDs."""
        return list(self._strategies.keys())

    def get_load_errors(self) -> List[Dict]:
        """Return any strategies that failed to load."""
        return self._load_errors

    @property
    def count(self) -> int:
        return len(self._strategies)

    @property
    def category_count(self) -> int:
        return len(self._by_category)

    def __len__(self):
        return len(self._strategies)

    def __contains__(self, strategy_id: str):
        return strategy_id in self._strategies

    def __repr__(self):
        return f"StrategyRegistry({self.count} strategies, {self.category_count} categories)"

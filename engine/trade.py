"""
Backtest Engine -- Trade Dataclass
====================================
Represents a single trade with all relevant fields.
Phase 4: expanded with TP1/TP2, MFE/MAE, planned/actual RR, net PnL.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Trade:
    """A completed or in-progress trade."""
    trade_id: str = ""
    strategy_id: str = ""
    symbol: str = "BTCUSD"
    direction: str = ""           # "BUY" or "SELL"
    lot_size: float = 0.01

    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0       # Primary take profit
    tp2_price: float = 0.0       # Extended take profit

    # Timing
    entry_time: str = ""
    exit_time: str = ""
    entry_bar_index: int = 0
    exit_bar_index: int = 0
    signal_bar_index: int = 0     # Bar where signal was generated
    bars_held: int = 0            # Number of bars position was open

    # PnL
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    gross_pnl_usd: float = 0.0   # Before costs
    net_pnl: float = 0.0         # After all costs

    # Costs
    spread_cost: float = 0.0
    commission_cost: float = 0.0
    slippage_cost: float = 0.0
    total_cost: float = 0.0

    # Risk metrics
    planned_rr: float = 0.0      # Planned risk:reward at entry
    actual_rr: float = 0.0       # Achieved risk:reward at exit
    rr_ratio: float = 0.0        # Alias for planned_rr (backward compat)
    sl_distance_pips: float = 0.0
    tp_distance_pips: float = 0.0

    # Excursion
    mfe_pips: float = 0.0        # Max Favorable Excursion (best unrealized P&L)
    mae_pips: float = 0.0        # Max Adverse Excursion (worst unrealized P&L)

    # Outcome
    outcome: str = ""             # "win", "loss", "be"
    exit_reason: str = ""         # "TP1", "TP2", "SL", "TRAILING", "SIGNAL", "TIMEOUT", "BE"

    # Validation
    is_valid: bool = True
    rejection_reason: str = ""

    # Metadata
    confidence: float = 0.0


@dataclass
class ValidatedTrade:
    """Result of trade validation. Either valid trade or rejection."""
    is_valid: bool
    trade: Optional[Trade] = None
    rejection_reason: str = ""

    @classmethod
    def accepted(cls, trade: Trade) -> "ValidatedTrade":
        return cls(is_valid=True, trade=trade)

    @classmethod
    def rejected(cls, reason: str) -> "ValidatedTrade":
        return cls(is_valid=False, rejection_reason=reason)

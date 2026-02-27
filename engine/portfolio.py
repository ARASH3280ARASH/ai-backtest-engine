"""
Backtest Engine -- Portfolio Manager
=======================================
Tracks equity, drawdowns, open positions, and running statistics
throughout a backtest run.

Usage:
    portfolio = Portfolio(initial_balance=10000.0)
    portfolio.open_trade(trade, bar_idx)
    portfolio.close_trade(trade, bar_idx)
    portfolio.update_equity(bar_idx, unrealized_pnl)
    stats = portfolio.get_stats()
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.trade import Trade


@dataclass
class BacktestResult:
    """Complete results from a single strategy backtest."""
    strategy_id: str = ""

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0

    # PnL
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    # Drawdown
    max_drawdown_dollars: float = 0.0
    max_drawdown_pct: float = 0.0
    max_dd_duration_bars: int = 0

    # Trade statistics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_rr: float = 0.0
    avg_bars_held: float = 0.0
    expectancy: float = 0.0       # Average $ per trade

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Costs
    total_spread_cost: float = 0.0
    total_commission_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_costs: float = 0.0

    # Curves and data
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    # Metadata
    total_bars: int = 0
    warmup_bars: int = 0


class Portfolio:
    """
    Tracks equity and positions during a backtest.

    Rules:
    - Max 1 position per strategy at a time
    - Fixed position sizing: 0.01 lot per trade
    - Equity updated bar-by-bar
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance       # Cash balance (closed P&L applied)
        self.equity = initial_balance        # Balance + unrealized P&L

        # Open positions: strategy_id -> Trade
        self.open_positions: Dict[str, Trade] = {}

        # Equity curve (one entry per bar)
        self.equity_curve: List[float] = []

        # Drawdown tracking
        self.peak_equity = initial_balance
        self.current_dd = 0.0
        self.max_dd_dollars = 0.0
        self.max_dd_pct = 0.0
        self.dd_start_bar = 0
        self.max_dd_duration = 0
        self._in_drawdown = False
        self._dd_bar_count = 0

        # Closed trades
        self.closed_trades: List[Trade] = []

        # Daily P&L tracking
        self.daily_pnl: Dict[str, float] = {}

        # Running stats
        self._wins = 0
        self._losses = 0
        self._total_profit = 0.0
        self._total_loss = 0.0
        self._total_costs = 0.0

    def has_open_position(self, strategy_id: str) -> bool:
        """Check if a strategy already has an open position."""
        return strategy_id in self.open_positions

    def open_trade(self, trade: Trade, bar_idx: int) -> bool:
        """
        Open a new trade. Returns False if strategy already has an open position.

        Costs are deducted from balance immediately at entry.
        """
        if trade.strategy_id in self.open_positions:
            return False

        # Deduct costs at entry
        self.balance -= trade.total_cost
        self._total_costs += trade.total_cost

        self.open_positions[trade.strategy_id] = trade
        return True

    def close_trade(self, trade: Trade, bar_idx: int) -> None:
        """
        Close an open trade. Updates balance with gross P&L.
        Costs were already deducted at open_trade().
        """
        if trade.strategy_id in self.open_positions:
            del self.open_positions[trade.strategy_id]

        # Apply gross P&L to balance (costs already deducted at entry)
        self.balance += trade.gross_pnl_usd

        # Track daily P&L
        day_key = trade.exit_time[:10] if trade.exit_time else ""
        if day_key:
            self.daily_pnl[day_key] = self.daily_pnl.get(day_key, 0.0) + trade.net_pnl

        # Running stats
        if trade.net_pnl > 0:
            self._wins += 1
            self._total_profit += trade.net_pnl
        elif trade.net_pnl < 0:
            self._losses += 1
            self._total_loss += abs(trade.net_pnl)

        self.closed_trades.append(trade)

    def update_equity(self, bar_idx: int, df=None, pip_size: float = 1.0,
                      pip_value_per_lot: float = 1.0) -> None:
        """
        Update equity at end of each bar. Calculates unrealized P&L for open positions.

        Args:
            bar_idx: Current bar index
            df: DataFrame with OHLCV data (to get current close for unrealized P&L)
            pip_size: Pip size for the symbol
            pip_value_per_lot: Dollar value per pip per lot
        """
        unrealized = 0.0
        if df is not None:
            close = float(df.iloc[bar_idx]["close"])
            for strat_id, trade in self.open_positions.items():
                if trade.direction == "BUY":
                    pnl_pips = (close - trade.entry_price) / pip_size
                else:
                    pnl_pips = (trade.entry_price - close) / pip_size
                unrealized += pnl_pips * pip_value_per_lot * trade.lot_size

        self.equity = self.balance + unrealized
        self.equity_curve.append(self.equity)

        # Drawdown tracking
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            if self._in_drawdown:
                # Exiting drawdown
                self.max_dd_duration = max(self.max_dd_duration, self._dd_bar_count)
                self._in_drawdown = False
                self._dd_bar_count = 0
        else:
            dd = self.peak_equity - self.equity
            dd_pct = (dd / self.peak_equity * 100) if self.peak_equity > 0 else 0
            if dd > self.max_dd_dollars:
                self.max_dd_dollars = dd
            if dd_pct > self.max_dd_pct:
                self.max_dd_pct = dd_pct
            if not self._in_drawdown:
                self._in_drawdown = True
                self._dd_bar_count = 0
            self._dd_bar_count += 1

    def get_stats(self) -> BacktestResult:
        """Calculate and return comprehensive backtest statistics."""
        trades = self.closed_trades
        n = len(trades)

        result = BacktestResult()
        result.total_trades = n
        result.trades = trades
        result.equity_curve = list(self.equity_curve)

        if n == 0:
            return result

        # Win/loss counts
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl < 0]
        be_trades = [t for t in trades if t.net_pnl == 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.breakeven_trades = len(be_trades)
        result.win_rate = (len(wins) / n * 100) if n > 0 else 0

        # PnL
        result.gross_profit = sum(t.net_pnl for t in wins)
        result.gross_loss = abs(sum(t.net_pnl for t in losses))
        result.net_profit = sum(t.net_pnl for t in trades)
        result.profit_factor = (result.gross_profit / result.gross_loss) if result.gross_loss > 0 else (
            float('inf') if result.gross_profit > 0 else 0.0
        )

        # Trade statistics
        result.avg_win = (result.gross_profit / len(wins)) if wins else 0
        result.avg_loss = (result.gross_loss / len(losses)) if losses else 0
        result.largest_win = max((t.net_pnl for t in wins), default=0)
        result.largest_loss = min((t.net_pnl for t in losses), default=0)

        rr_values = [t.actual_rr for t in trades if t.actual_rr != 0]
        result.avg_rr = (sum(rr_values) / len(rr_values)) if rr_values else 0

        bars_held = [t.bars_held for t in trades if t.bars_held > 0]
        result.avg_bars_held = (sum(bars_held) / len(bars_held)) if bars_held else 0

        result.expectancy = result.net_profit / n if n > 0 else 0

        # Drawdown
        result.max_drawdown_dollars = round(self.max_dd_dollars, 2)
        result.max_drawdown_pct = round(self.max_dd_pct, 2)
        # Finalize DD duration if still in drawdown
        if self._in_drawdown:
            self.max_dd_duration = max(self.max_dd_duration, self._dd_bar_count)
        result.max_dd_duration_bars = self.max_dd_duration

        # Consecutive wins/losses
        result.max_consecutive_wins = _max_streak(trades, win=True)
        result.max_consecutive_losses = _max_streak(trades, win=False)

        # Costs
        result.total_spread_cost = sum(t.spread_cost for t in trades)
        result.total_commission_cost = sum(t.commission_cost for t in trades)
        result.total_slippage_cost = sum(t.slippage_cost for t in trades)
        result.total_costs = sum(t.total_cost for t in trades)

        # Risk-adjusted ratios
        result.sharpe_ratio = _calc_sharpe(trades)
        result.sortino_ratio = _calc_sortino(trades)
        result.calmar_ratio = _calc_calmar(result.net_profit, self.max_dd_dollars,
                                            len(self.equity_curve))

        # Monthly returns
        result.monthly_returns = _calc_monthly_returns(trades)

        return result


def _max_streak(trades: List[Trade], win: bool) -> int:
    """Calculate max consecutive wins or losses."""
    max_streak = 0
    current = 0
    for t in trades:
        if win and t.net_pnl > 0:
            current += 1
        elif not win and t.net_pnl < 0:
            current += 1
        else:
            max_streak = max(max_streak, current)
            current = 0
    return max(max_streak, current)


def _calc_sharpe(trades: List[Trade], annual_bars: int = 8760) -> float:
    """
    Annualized Sharpe ratio from per-trade returns.
    Assumes H1 bars: 8760 bars/year.
    """
    if len(trades) < 2:
        return 0.0
    returns = [t.net_pnl for t in trades]
    import numpy as np
    arr = np.array(returns, dtype=np.float64)
    mean_r = np.mean(arr)
    std_r = np.std(arr, ddof=1)
    if std_r == 0:
        return 0.0
    # Annualize: multiply by sqrt(trades_per_year)
    # Approximate trades_per_year from avg_bars_held
    avg_bars = np.mean([t.bars_held for t in trades]) if trades else 1
    if avg_bars <= 0:
        avg_bars = 1
    trades_per_year = annual_bars / avg_bars
    return float((mean_r / std_r) * np.sqrt(trades_per_year))


def _calc_sortino(trades: List[Trade], annual_bars: int = 8760) -> float:
    """Annualized Sortino ratio (only downside deviation)."""
    if len(trades) < 2:
        return 0.0
    import numpy as np
    returns = np.array([t.net_pnl for t in trades], dtype=np.float64)
    mean_r = np.mean(returns)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float('inf') if mean_r > 0 else 0.0
    down_std = np.std(downside, ddof=1)
    if down_std == 0:
        return 0.0
    avg_bars = np.mean([t.bars_held for t in trades]) if trades else 1
    if avg_bars <= 0:
        avg_bars = 1
    trades_per_year = annual_bars / avg_bars
    return float((mean_r / down_std) * np.sqrt(trades_per_year))


def _calc_calmar(net_profit: float, max_dd: float, total_bars: int,
                 annual_bars: int = 8760) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    if max_dd <= 0 or total_bars <= 0:
        return 0.0
    annual_factor = annual_bars / total_bars
    annual_return = net_profit * annual_factor
    return round(annual_return / max_dd, 2)


def _calc_monthly_returns(trades: List[Trade]) -> Dict[str, float]:
    """Aggregate net PnL by month (YYYY-MM)."""
    monthly = {}
    for t in trades:
        if t.exit_time and len(t.exit_time) >= 7:
            month_key = t.exit_time[:7]  # "YYYY-MM"
            monthly[month_key] = monthly.get(month_key, 0.0) + t.net_pnl
    return {k: round(v, 2) for k, v in sorted(monthly.items())}

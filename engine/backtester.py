"""
Backtest Engine -- Core Backtester
=====================================
Bar-by-bar backtesting loop with realistic execution model.

Key rules:
  - Signal on bar N, entry on bar N+1's OPEN
  - SL/TP checked against bar HIGH/LOW (not just close)
  - If both SL and TP hit in same bar: SL takes priority (conservative)
  - Costs deducted at entry
  - Max 1 position per strategy at a time
  - Timeout: close after 200 bars

Usage:
    bt = Backtester()
    result = bt.run_single("RSI_01", df, indicators, registry)
    results = bt.run_all(["RSI_01", "MACD_03"], df, indicators, registry)
"""

import sys
import os
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.trade import Trade
from engine.portfolio import Portfolio, BacktestResult
from engine.costs import (
    calculate_spread_cost,
    calculate_commission,
    calculate_slippage_dollars,
    calculate_slippage,
)
from engine.validator import validate_trade
from config.broker import BTCUSD_CONFIG
from strategies.base import SignalType

_CFG = BTCUSD_CONFIG
_PIP = _CFG["pip_size"]           # 1.0 for BTCUSD
_POINT = _CFG["point"]            # 0.01
_PIP_VALUE = _CFG["pip_value_per_lot"]  # 1.0
_LOT = _CFG["backtest_lot"]       # 0.01
_STOP_LEVEL = _CFG["stop_level_pips"]  # 20

# Trade management defaults
TIMEOUT_BARS = 200
BE_TRIGGER_PCT = 0.5   # Move SL to breakeven when profit >= 50% of SL distance


class Backtester:
    """
    Core backtesting engine.

    Iterates bar-by-bar over a DataFrame, checking exits, managing trades,
    and generating new entries from strategy signals.
    """

    def __init__(self, warmup: int = 50, verbose: bool = False,
                 timeout_seconds: float = 120.0, signal_check_interval: int = 1):
        """
        Args:
            warmup: Number of initial bars to skip (indicator warmup)
            verbose: Print trade details during backtest
            timeout_seconds: Max time per strategy backtest (0 = no limit)
            signal_check_interval: Check for new signals every N bars (1=every bar,
                                   4=every 4th bar). Higher values speed up backtesting.
        """
        self.warmup = warmup
        self.verbose = verbose
        self.timeout_seconds = timeout_seconds
        self.signal_check_interval = max(1, signal_check_interval)

    def run_single(
        self,
        strategy_id: str,
        df,
        indicators: Dict,
        registry,
        params: Optional[Dict] = None,
    ) -> BacktestResult:
        """
        Run backtest for a single strategy.

        Args:
            strategy_id: Strategy ID (e.g., "RSI_01")
            df: OHLCV DataFrame with columns: time, open, high, low, close, tick_volume
            indicators: Pre-computed indicators from compute_all()
            registry: StrategyRegistry with loaded strategies
            params: Optional override params

        Returns:
            BacktestResult with all metrics
        """
        strategy = registry.get_by_id(strategy_id)
        if strategy is None:
            result = BacktestResult(strategy_id=strategy_id)
            return result

        portfolio = Portfolio(initial_balance=10000.0)
        n_bars = len(df)

        # Track pending entry (signal on bar N, enter on bar N+1)
        pending_entry = None  # Will hold (signal, entry_setup, signal_bar_idx)

        # Track open trade + excursion
        open_trade: Optional[Trade] = None
        mfe_pips = 0.0
        mae_pips = 0.0

        # Get ATR series for cost calculation
        atr_series = indicators.get("atr_14", None)

        # Timeout tracking
        _start_time = time.time()

        for bar_idx in range(self.warmup, n_bars):
            # Timeout check (every 100 bars to avoid overhead)
            if self.timeout_seconds > 0 and bar_idx % 100 == 0:
                if time.time() - _start_time > self.timeout_seconds:
                    if self.verbose:
                        print(f"  [TIMEOUT] {strategy_id} at bar {bar_idx}/{n_bars}")
                    break

            bar = df.iloc[bar_idx]
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            bar_time = str(bar["time"])

            # Get ATR at this bar
            atr_val = 0.0
            if atr_series is not None and hasattr(atr_series, 'iloc') and bar_idx < len(atr_series):
                v = float(atr_series.iloc[bar_idx])
                if v == v:  # NaN check
                    atr_val = v

            # ═══ PHASE A: EXECUTE PENDING ENTRY ═══
            if pending_entry is not None and open_trade is None:
                sig, entry_setup, sig_bar = pending_entry
                pending_entry = None

                if bar_idx == sig_bar + 1 and entry_setup.valid:
                    # Validate the trade
                    validated = validate_trade(
                        strategy_id=strategy_id,
                        direction=entry_setup.direction,
                        signal_bar_index=sig_bar,
                        entry_bar_index=bar_idx,
                        entry_bar_open=bar_open,
                        entry_bar_high=bar_high,
                        entry_bar_low=bar_low,
                        entry_bar_close=bar_close,
                        entry_time=bar_time,
                        sl_price=entry_setup.sl_price,
                        tp_price=entry_setup.tp1_price,
                        lot_size=_LOT,
                        atr_value=atr_val,
                        confidence=entry_setup.confidence,
                    )

                    if validated.is_valid:
                        trade = validated.trade
                        # Set TP1/TP2
                        trade.tp1_price = entry_setup.tp1_price
                        trade.tp2_price = entry_setup.tp2_price
                        trade.planned_rr = entry_setup.rr_ratio

                        if portfolio.open_trade(trade, bar_idx):
                            open_trade = trade
                            mfe_pips = 0.0
                            mae_pips = 0.0
                            if self.verbose:
                                print(f"  [BAR {bar_idx}] OPEN {trade.direction} "
                                      f"@ {trade.entry_price:.2f} "
                                      f"SL={trade.sl_price:.2f} "
                                      f"TP1={trade.tp1_price:.2f}")

            # ═══ PHASE B: CHECK EXITS for open position ═══
            if open_trade is not None:
                closed = False
                exit_reason = ""
                exit_price = 0.0

                entry = open_trade.entry_price
                sl = open_trade.sl_price
                tp1 = open_trade.tp1_price
                direction = open_trade.direction

                # Track excursion
                if direction == "BUY":
                    favorable = (bar_high - entry) / _PIP
                    adverse = (entry - bar_low) / _PIP
                else:
                    favorable = (entry - bar_low) / _PIP
                    adverse = (bar_high - entry) / _PIP

                mfe_pips = max(mfe_pips, favorable)
                mae_pips = max(mae_pips, adverse)

                # Check SL and TP hits
                sl_hit = False
                tp_hit = False

                if direction == "BUY":
                    sl_hit = bar_low <= sl
                    tp_hit = bar_high >= tp1
                else:  # SELL
                    sl_hit = bar_high >= sl
                    tp_hit = bar_low <= tp1

                # If BOTH hit in same bar: SL takes priority (conservative)
                if sl_hit and tp_hit:
                    exit_price = sl
                    exit_reason = "SL"
                    closed = True
                elif sl_hit:
                    exit_price = sl
                    exit_reason = "SL"
                    closed = True
                elif tp_hit:
                    exit_price = tp1
                    exit_reason = "TP1"
                    closed = True

                # Timeout check
                bars_in_trade = bar_idx - open_trade.entry_bar_index
                if not closed and bars_in_trade >= TIMEOUT_BARS:
                    exit_price = bar_close
                    exit_reason = "TIMEOUT"
                    closed = True

                # ═══ PHASE C: TRADE MANAGEMENT (if not closed, after entry bar) ═══
                if not closed and bars_in_trade > 0:
                    # Break-even: move SL to entry when profit >= BE_TRIGGER_PCT of SL distance
                    sl_dist_pips = open_trade.sl_distance_pips
                    if direction == "BUY":
                        current_profit_pips = (bar_close - entry) / _PIP
                    else:
                        current_profit_pips = (entry - bar_close) / _PIP

                    be_threshold = sl_dist_pips * BE_TRIGGER_PCT
                    if current_profit_pips >= be_threshold and sl != entry:
                        # Move SL to breakeven (entry price)
                        open_trade.sl_price = entry
                        sl = entry

                # ═══ CLOSE TRADE if triggered ═══
                if closed:
                    open_trade.exit_price = round(exit_price, _CFG["digits"])
                    open_trade.exit_bar_index = bar_idx
                    open_trade.exit_time = bar_time
                    open_trade.exit_reason = exit_reason
                    open_trade.bars_held = bar_idx - open_trade.entry_bar_index
                    open_trade.mfe_pips = round(mfe_pips, 1)
                    open_trade.mae_pips = round(mae_pips, 1)

                    # Calculate P&L
                    _finalize_trade_pnl(open_trade)

                    portfolio.close_trade(open_trade, bar_idx)

                    if self.verbose:
                        print(f"  [BAR {bar_idx}] CLOSE {exit_reason} "
                              f"@ {exit_price:.2f} "
                              f"PnL={open_trade.net_pnl:.2f} "
                              f"({open_trade.pnl_pips:.1f} pips)")

                    open_trade = None

            # ═══ PHASE D: CHECK FOR NEW SIGNALS (only if no open position) ═══
            if open_trade is None and pending_entry is None:
                # Only check signals at the configured interval (speedup)
                if (bar_idx - self.warmup) % self.signal_check_interval != 0:
                    portfolio.update_equity(bar_idx, df, _PIP, _PIP_VALUE)
                    continue

                # Don't generate signal on last bar (need next bar for entry)
                if bar_idx < n_bars - 1:
                    sig = strategy.generate_signal(
                        df, indicators, bar_idx, _CFG["symbol"], "H1"
                    )

                    if sig.signal_type in (SignalType.BUY, SignalType.SELL):
                        entry_setup = strategy.calculate_entry(
                            sig, df, indicators, bar_idx, _CFG["symbol"]
                        )
                        if entry_setup.valid:
                            pending_entry = (sig, entry_setup, bar_idx)

            # ═══ PHASE E: UPDATE EQUITY ═══
            portfolio.update_equity(bar_idx, df, _PIP, _PIP_VALUE)

        # Force-close any remaining open position at last bar's close
        if open_trade is not None:
            last_bar = df.iloc[n_bars - 1]
            open_trade.exit_price = round(float(last_bar["close"]), _CFG["digits"])
            open_trade.exit_bar_index = n_bars - 1
            open_trade.exit_time = str(last_bar["time"])
            open_trade.exit_reason = "END"
            open_trade.bars_held = (n_bars - 1) - open_trade.entry_bar_index
            open_trade.mfe_pips = round(mfe_pips, 1)
            open_trade.mae_pips = round(mae_pips, 1)
            _finalize_trade_pnl(open_trade)
            portfolio.close_trade(open_trade, n_bars - 1)

        # Build result
        result = portfolio.get_stats()
        result.strategy_id = strategy_id
        result.total_bars = n_bars
        result.warmup_bars = self.warmup

        return result

    def run_all(
        self,
        strategy_ids: List[str],
        df,
        indicators: Dict,
        registry,
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest for multiple strategies.

        Args:
            strategy_ids: List of strategy IDs to backtest
            df: OHLCV DataFrame
            indicators: Pre-computed indicators
            registry: StrategyRegistry

        Returns:
            Dict mapping strategy_id -> BacktestResult
        """
        results = {}
        total = len(strategy_ids)

        for i, sid in enumerate(strategy_ids):
            t0 = time.time()
            result = self.run_single(sid, df, indicators, registry)
            elapsed = time.time() - t0

            results[sid] = result

            if self.verbose:
                print(f"[{i+1}/{total}] {sid}: "
                      f"{result.total_trades} trades, "
                      f"WR={result.win_rate:.1f}%, "
                      f"Net={result.net_profit:.2f}, "
                      f"PF={result.profit_factor:.2f} "
                      f"({elapsed:.1f}s)")

        return results


def _finalize_trade_pnl(trade: Trade) -> None:
    """Calculate all PnL fields for a closed trade."""
    pip = _PIP
    pip_value = _PIP_VALUE
    lot = trade.lot_size

    # PnL in pips
    if trade.direction == "BUY":
        trade.pnl_pips = round((trade.exit_price - trade.entry_price) / pip, 1)
    else:
        trade.pnl_pips = round((trade.entry_price - trade.exit_price) / pip, 1)

    # PnL in dollars (gross, before costs)
    trade.gross_pnl_usd = round(trade.pnl_pips * pip_value * lot, 4)

    # Net PnL (gross minus costs, which were already deducted from balance)
    trade.pnl_usd = trade.gross_pnl_usd
    trade.net_pnl = round(trade.gross_pnl_usd - trade.total_cost, 4)

    # Actual R:R
    sl_dist = trade.sl_distance_pips
    if sl_dist > 0:
        trade.actual_rr = round(trade.pnl_pips / sl_dist, 2)
    else:
        trade.actual_rr = 0.0

    # Outcome
    if trade.net_pnl > 0:
        trade.outcome = "win"
    elif trade.net_pnl < 0:
        trade.outcome = "loss"
    else:
        trade.outcome = "be"

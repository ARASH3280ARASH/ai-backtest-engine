"""
Backtest Engine -- Strategy Combination Optimizer
====================================================
Finds combinations of 2-5 strategies that work together as a team.

Approach:
  1. Pre-compute signal matrices for all candidate strategies (one-time cost)
  2. Test combinations using fast array operations
  3. Run full backtest only on promising combos

Combination Modes:
  UNANIMOUS:        All strategies must agree
  MAJORITY:         >50% must agree on direction
  WEIGHTED:         Weighted by performance score, threshold > 0.6
  ANY_CONFIRMED:    Any signal + at least 1 other confirms
  LEADER_FOLLOWER:  Best strategy leads, others must confirm

Exit Options:
  A: Use primary strategy's SL/TP (highest score)
  B: Average SL/TP across components
  C: Most conservative SL + most aggressive TP
  D: Dynamic exit when majority flips
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.broker import BTCUSD_CONFIG
from engine.costs import compute_variable_cost
from config.settings import RESULTS_DIR

_PIP = BTCUSD_CONFIG["pip_size"]
_LOT = BTCUSD_CONFIG["backtest_lot"]


class ComboMode(Enum):
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    ANY_CONFIRMED = "any_confirmed"
    LEADER_FOLLOWER = "leader_follower"


# ═══════════════════════════════════════════════════════════════
#  SIGNAL MATRIX: Pre-compute signals for all strategies
# ═══════════════════════════════════════════════════════════════

class SignalMatrix:
    """
    Pre-computes and stores signals for all candidate strategies.

    signals[strategy_idx, bar_idx] = +1 (BUY), -1 (SELL), 0 (NEUTRAL)

    This avoids calling generate_signal() repeatedly during combo search.
    """

    def __init__(self):
        self.strategy_ids: List[str] = []
        self.strategy_scores: Dict[str, float] = {}
        self.strategy_categories: Dict[str, str] = {}
        self.matrix: Optional[np.ndarray] = None  # shape: (n_strategies, n_bars)
        self.n_bars = 0
        self.warmup = 50
        self.check_interval = 4

    def build(self, strategy_ids: List[str], df, indicators: Dict,
              registry, scores: Dict[str, float] = None,
              warmup: int = 50, check_interval: int = 4,
              verbose: bool = True) -> "SignalMatrix":
        """
        Pre-compute signal matrix for all candidate strategies.

        Args:
            strategy_ids: List of strategy IDs to pre-compute
            df: OHLCV DataFrame
            indicators: Pre-computed indicators
            registry: StrategyRegistry
            scores: {strategy_id: composite_score} for weighting
            warmup: Bars to skip at start
            check_interval: Check signals every N bars
            verbose: Print progress
        """
        from strategies.base import SignalType

        self.strategy_ids = list(strategy_ids)
        self.strategy_scores = scores or {}
        self.n_bars = len(df)
        self.warmup = warmup
        self.check_interval = check_interval

        n_strats = len(self.strategy_ids)
        self.matrix = np.zeros((n_strats, self.n_bars), dtype=np.int8)

        # Store categories
        for sid in self.strategy_ids:
            strat = registry.get_by_id(sid)
            if strat:
                self.strategy_categories[sid] = strat.category

        symbol = BTCUSD_CONFIG["symbol"]
        t0 = time.time()
        timeout_per_strat = 90  # seconds per strategy
        skipped = []

        for i, sid in enumerate(self.strategy_ids):
            strategy = registry.get_by_id(sid)
            if strategy is None:
                skipped.append(sid)
                continue

            strat_t0 = time.time()
            timed_out = False

            for bar_idx in range(warmup, self.n_bars):
                if (bar_idx - warmup) % check_interval != 0:
                    continue

                # Per-strategy timeout check every 200 bars
                if bar_idx % 200 == 0:
                    if time.time() - strat_t0 > timeout_per_strat:
                        timed_out = True
                        break

                try:
                    sig = strategy.generate_signal(
                        df, indicators, bar_idx, symbol, "H1"
                    )
                    if sig.signal_type == SignalType.BUY:
                        self.matrix[i, bar_idx] = 1
                    elif sig.signal_type == SignalType.SELL:
                        self.matrix[i, bar_idx] = -1
                except Exception:
                    pass

            elapsed = time.time() - strat_t0
            if timed_out:
                skipped.append(sid)
                if verbose:
                    print(f"  [{i+1:>3}/{n_strats}] {sid:<14} TIMEOUT ({elapsed:.0f}s)",
                          flush=True)
            elif verbose and ((i + 1) % 5 == 0 or (i + 1) == n_strats):
                total_elapsed = time.time() - t0
                avg = total_elapsed / (i + 1)
                eta = avg * (n_strats - i - 1)
                print(f"  [{i+1:>3}/{n_strats}] {sid:<14} {elapsed:.1f}s  "
                      f"ETA={eta:.0f}s", flush=True)

        # Remove timed-out strategies from the matrix
        if skipped:
            keep_mask = [sid not in skipped for sid in self.strategy_ids]
            self.matrix = self.matrix[keep_mask]
            self.strategy_ids = [sid for sid in self.strategy_ids
                                 if sid not in skipped]
            for sid in skipped:
                self.strategy_categories.pop(sid, None)
                self.strategy_scores.pop(sid, None)

        if verbose:
            total = time.time() - t0
            print(f"  Signal matrix built: {len(self.strategy_ids)} strategies x "
                  f"{self.n_bars} bars in {total:.1f}s", flush=True)
            if skipped:
                print(f"  Skipped (timeout/missing): {len(skipped)}: "
                      f"{', '.join(skipped[:10])}", flush=True)

        return self

    def build_from_cached(self, strategy_ids: List[str], n_bars: int,
                          results_dir: str, registry=None,
                          scores: Dict[str, float] = None,
                          warmup: int = 50, check_interval: int = 4,
                          verbose: bool = True) -> "SignalMatrix":
        """
        Build signal matrix from cached Phase 5 individual results.
        Instead of re-running generate_signal(), reads trade signal_bar_index
        and direction from the saved JSON files. ~1000x faster.

        Note: Only captures signals that resulted in trades. Signals that
        occurred while a trade was open are missed. This is a reasonable
        approximation for combo search.
        """
        t0 = time.time()
        self.strategy_ids = []
        self.strategy_scores = scores or {}
        self.n_bars = n_bars
        self.warmup = warmup
        self.check_interval = check_interval

        # First pass: load all cached results to find valid strategies
        loaded = {}
        skipped = []
        for sid in strategy_ids:
            fpath = os.path.join(results_dir, f"{sid}.json")
            if not os.path.exists(fpath):
                skipped.append(sid)
                continue
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                trades = data.get("trades", [])
                if len(trades) < 3:
                    skipped.append(sid)
                    continue
                loaded[sid] = data
                self.strategy_ids.append(sid)
                cat = data.get("category", "")
                if not cat and registry:
                    strat = registry.get_by_id(sid)
                    if strat:
                        cat = strat.category
                self.strategy_categories[sid] = cat
            except Exception:
                skipped.append(sid)

        n_strats = len(self.strategy_ids)
        self.matrix = np.zeros((n_strats, n_bars), dtype=np.int8)

        # Build signal matrix from trade data
        for i, sid in enumerate(self.strategy_ids):
            data = loaded[sid]
            for trade in data.get("trades", []):
                sig_bar = int(trade.get("signal_bar_index", 0))
                direction = trade.get("direction", "")
                if 0 <= sig_bar < n_bars:
                    if direction == "BUY":
                        self.matrix[i, sig_bar] = 1
                    elif direction == "SELL":
                        self.matrix[i, sig_bar] = -1

        elapsed = time.time() - t0
        if verbose:
            total_signals = int(np.count_nonzero(self.matrix))
            print(f"  Signal matrix from cache: {n_strats} strategies x "
                  f"{n_bars} bars, {total_signals} signals in {elapsed:.1f}s",
                  flush=True)
            if skipped:
                print(f"  Skipped: {len(skipped)} (no cache/few trades)",
                      flush=True)

        return self

    def get_signals(self, strategy_id: str) -> np.ndarray:
        """Get signal array for a single strategy."""
        idx = self.strategy_ids.index(strategy_id)
        return self.matrix[idx]

    def get_correlation(self, sid_a: str, sid_b: str) -> float:
        """Compute signal correlation between two strategies."""
        try:
            idx_a = self.strategy_ids.index(sid_a)
            idx_b = self.strategy_ids.index(sid_b)
        except ValueError:
            return 0.0

        a = self.matrix[idx_a].astype(np.float64)
        b = self.matrix[idx_b].astype(np.float64)

        # Only compare bars where at least one has a signal
        mask = (a != 0) | (b != 0)
        if mask.sum() < 10:
            return 0.0

        a_m = a[mask]
        b_m = b[mask]

        if np.std(a_m) == 0 or np.std(b_m) == 0:
            return 0.0

        corr = np.corrcoef(a_m, b_m)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def build_correlation_matrix(self) -> np.ndarray:
        """Build full NxN correlation matrix."""
        n = len(self.strategy_ids)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.get_correlation(
                    self.strategy_ids[i], self.strategy_ids[j]
                )
                corr[i, j] = c
                corr[j, i] = c
        return corr


# ═══════════════════════════════════════════════════════════════
#  COMBINATION SIGNAL LOGIC
# ═══════════════════════════════════════════════════════════════

def combine_signals(
    signal_matrix: SignalMatrix,
    strategy_ids: List[str],
    mode: ComboMode,
    bar_idx: int,
    scores: Dict[str, float] = None,
) -> int:
    """
    Combine signals from multiple strategies at a given bar.

    Returns: +1 (BUY), -1 (SELL), 0 (NEUTRAL)
    """
    signals = []
    for sid in strategy_ids:
        try:
            idx = signal_matrix.strategy_ids.index(sid)
            signals.append(int(signal_matrix.matrix[idx, bar_idx]))
        except (ValueError, IndexError):
            signals.append(0)

    n = len(signals)
    if n == 0:
        return 0

    buys = sum(1 for s in signals if s == 1)
    sells = sum(1 for s in signals if s == -1)

    if mode == ComboMode.UNANIMOUS:
        if buys == n:
            return 1
        if sells == n:
            return -1
        return 0

    elif mode == ComboMode.MAJORITY:
        threshold = n / 2.0
        if buys > threshold:
            return 1
        if sells > threshold:
            return -1
        return 0

    elif mode == ComboMode.WEIGHTED:
        if scores is None:
            scores = {}
        total_weight = 0.0
        weighted_signal = 0.0
        for sid, sig in zip(strategy_ids, signals):
            w = scores.get(sid, 1.0)
            total_weight += w
            weighted_signal += sig * w
        if total_weight == 0:
            return 0
        normalized = weighted_signal / total_weight
        if normalized > 0.6:
            return 1
        if normalized < -0.6:
            return -1
        return 0

    elif mode == ComboMode.ANY_CONFIRMED:
        if buys >= 2:
            return 1
        if sells >= 2:
            return -1
        return 0

    elif mode == ComboMode.LEADER_FOLLOWER:
        # Leader = first strategy (highest score)
        leader_sig = signals[0]
        if leader_sig == 0:
            return 0
        # At least 1 follower must confirm
        confirmations = sum(1 for s in signals[1:] if s == leader_sig)
        if confirmations >= 1:
            return leader_sig
        return 0

    return 0


# ═══════════════════════════════════════════════════════════════
#  COMBINATION BACKTESTER
# ═══════════════════════════════════════════════════════════════

def backtest_combination(
    signal_matrix: SignalMatrix,
    strategy_ids: List[str],
    mode: ComboMode,
    exit_option: str,
    df,
    indicators: Dict,
    registry,
    scores: Dict[str, float] = None,
    _arrays_cache: Dict = None,
) -> Dict:
    """
    Fast backtest for a strategy combination using pre-extracted numpy arrays.
    Avoids pandas .iloc access in the hot loop for ~10x speedup.
    """
    cfg = BTCUSD_CONFIG
    pip = cfg["pip_size"]
    pip_value = cfg["pip_value_per_lot"]
    lot = cfg["backtest_lot"]
    warmup = signal_matrix.warmup
    n_bars = len(df)

    # Pre-extract numpy arrays (cache across calls)
    if _arrays_cache is not None and "opens" in _arrays_cache:
        opens = _arrays_cache["opens"]
        highs = _arrays_cache["highs"]
        lows = _arrays_cache["lows"]
        closes = _arrays_cache["closes"]
        atr_arr = _arrays_cache["atr"]
    else:
        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        atr_series = indicators.get("atr_14", None)
        if atr_series is not None:
            atr_arr = np.array(atr_series, dtype=np.float64)
            atr_arr = np.nan_to_num(atr_arr, nan=0.0)
        else:
            atr_arr = np.zeros(n_bars, dtype=np.float64)
        if _arrays_cache is not None:
            _arrays_cache["opens"] = opens
            _arrays_cache["highs"] = highs
            _arrays_cache["lows"] = lows
            _arrays_cache["closes"] = closes
            _arrays_cache["atr"] = atr_arr

    # Pre-compute combined signal array for this combo
    strat_indices = []
    for sid in strategy_ids:
        try:
            idx = signal_matrix.strategy_ids.index(sid)
            strat_indices.append(idx)
        except ValueError:
            pass
    if not strat_indices:
        return _empty_metrics()

    combo_matrix = signal_matrix.matrix[strat_indices]  # shape: (n_strats, n_bars)
    n_strats = len(strat_indices)

    # Pre-compute combined signals for all bars using vectorized ops
    combo_signals = np.zeros(n_bars, dtype=np.int8)
    buys = (combo_matrix == 1).sum(axis=0)
    sells = (combo_matrix == -1).sum(axis=0)

    if mode == ComboMode.UNANIMOUS:
        combo_signals[buys == n_strats] = 1
        combo_signals[sells == n_strats] = -1
    elif mode == ComboMode.MAJORITY:
        threshold = n_strats / 2.0
        combo_signals[buys > threshold] = 1
        combo_signals[sells > threshold] = -1
    elif mode == ComboMode.WEIGHTED:
        weights = np.array([
            (scores or {}).get(strategy_ids[i], 1.0) for i in range(n_strats)
        ], dtype=np.float64).reshape(-1, 1)
        total_w = weights.sum()
        if total_w > 0:
            weighted = (combo_matrix.astype(np.float64) * weights).sum(axis=0) / total_w
            combo_signals[weighted > 0.6] = 1
            combo_signals[weighted < -0.6] = -1
    elif mode == ComboMode.ANY_CONFIRMED:
        combo_signals[buys >= 2] = 1
        combo_signals[sells >= 2] = -1
    elif mode == ComboMode.LEADER_FOLLOWER:
        leader = combo_matrix[0]
        for i in range(n_bars):
            if leader[i] == 0:
                continue
            confirmations = 0
            for j in range(1, n_strats):
                if combo_matrix[j, i] == leader[i]:
                    confirmations += 1
            if confirmations >= 1:
                combo_signals[i] = leader[i]

    # Spread cost constants
    spread_pips = cfg["spread_points"] * cfg["point"]
    half_spread = spread_pips / 2.0

    # Fast bar-by-bar backtest using numpy arrays
    _cost_rng = np.random.RandomState(42)

    balance = 10000.0
    peak = balance
    max_dd_pct = 0.0
    max_dd_dollars = 0.0
    trades = []
    in_trade = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    sl_dist = 0.0
    direction = 0  # 1=BUY, -1=SELL
    entry_bar = 0
    total_cost = 0.0
    be_moved = False
    pending_dir = 0
    pending_bar = -1

    for bar_idx in range(warmup, n_bars):
        o = opens[bar_idx]
        h = highs[bar_idx]
        lo = lows[bar_idx]
        c = closes[bar_idx]
        atr_val = atr_arr[bar_idx] if bar_idx < len(atr_arr) else 0.0

        # Execute pending entry
        if pending_dir != 0 and not in_trade:
            if bar_idx == pending_bar + 1 and atr_val > 0:
                atr_pips = atr_val / pip
                if pending_dir == 1:
                    ep = o + half_spread
                else:
                    ep = o - half_spread

                if exit_option == "A":
                    sd = max(atr_pips * 1.5, 20)
                    td = max(atr_pips * 2.0, sd * 1.5)
                elif exit_option == "C":
                    sd = max(atr_pips * 1.0, 20)
                    td = max(atr_pips * 3.0, sd * 2.0)
                else:
                    sd = max(atr_pips * 2.0, 20)
                    td = max(atr_pips * 4.0, sd * 1.5)

                if sd >= 20 and td >= 20:
                    if pending_dir == 1:
                        sp = ep - sd * pip
                        tp = ep + td * pip
                    else:
                        sp = ep + sd * pip
                        tp = ep - td * pip

                    tc = compute_variable_cost(atr_arr, bar_idx, _cost_rng)
                    balance -= tc
                    in_trade = True
                    entry_price = ep
                    sl_price = sp
                    tp_price = tp
                    sl_dist = sd
                    direction = pending_dir
                    entry_bar = bar_idx
                    total_cost = tc
                    be_moved = False

            pending_dir = 0
            pending_bar = -1

        # Check exits
        if in_trade:
            closed = False
            exit_price = 0.0

            if direction == 1:
                sl_hit = lo <= sl_price
                tp_hit = h >= tp_price
            else:
                sl_hit = h >= sl_price
                tp_hit = lo <= tp_price

            if sl_hit:
                exit_price = sl_price
                closed = True
            elif tp_hit:
                exit_price = tp_price
                closed = True
            elif bar_idx - entry_bar >= 200:
                exit_price = c
                closed = True

            # Breakeven check (only after entry bar)
            if not closed and not be_moved and bar_idx > entry_bar:
                if direction == 1:
                    profit_pips = (c - entry_price) / pip
                else:
                    profit_pips = (entry_price - c) / pip
                if profit_pips >= sl_dist * 0.5:
                    sl_price = entry_price
                    be_moved = True

            if closed:
                if direction == 1:
                    pnl_pips = (exit_price - entry_price) / pip
                else:
                    pnl_pips = (entry_price - exit_price) / pip
                gross_pnl = pnl_pips * pip_value * lot
                net_pnl = gross_pnl - total_cost
                balance += gross_pnl
                trades.append((net_pnl, bar_idx - entry_bar))
                in_trade = False

        # Check for new signals
        if not in_trade and pending_dir == 0:
            if bar_idx < n_bars - 1:
                sig = combo_signals[bar_idx]
                if sig != 0:
                    pending_dir = sig
                    pending_bar = bar_idx

        # Drawdown tracking
        if balance > peak:
            peak = balance
        dd = peak - balance
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        if dd > max_dd_dollars:
            max_dd_dollars = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    # Force close
    if in_trade:
        c = closes[n_bars - 1]
        if direction == 1:
            pnl_pips = (c - entry_price) / pip
        else:
            pnl_pips = (entry_price - c) / pip
        gross_pnl = pnl_pips * pip_value * lot
        net_pnl = gross_pnl - total_cost
        balance += gross_pnl
        trades.append((net_pnl, n_bars - 1 - entry_bar))

    return _compute_fast_metrics(trades, max_dd_pct, max_dd_dollars, balance)


def _empty_metrics():
    return {
        "total_trades": 0, "win_rate": 0, "net_profit": 0,
        "profit_factor": 0, "max_drawdown_pct": 0,
        "max_drawdown_dollars": 0, "sharpe_ratio": 0,
        "expectancy": 0, "avg_bars_held": 0, "total_costs": 0,
    }


def _compute_fast_metrics(trades_data, max_dd_pct, max_dd_dollars, final_balance):
    """Compute metrics from list of (net_pnl, bars_held) tuples."""
    n = len(trades_data)
    if n == 0:
        return _empty_metrics()

    # Unpack tuples
    pnls = [t[0] for t in trades_data]
    bars_held = [t[1] for t in trades_data]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net = sum(pnls)
    wr = (len(wins) / n * 100) if n > 0 else 0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (
        999.0 if gross_profit > 0 else 0.0
    )

    # Sharpe approximation
    sharpe = 0.0
    if n >= 2:
        arr = np.array(pnls, dtype=np.float64)
        mean_r = np.mean(arr)
        std_r = np.std(arr, ddof=1)
        if std_r > 0:
            sharpe = float(mean_r / std_r * np.sqrt(min(n * 4, 200)))

    avg_bars = round(sum(bars_held) / n, 1) if n > 0 else 0

    return {
        "total_trades": n,
        "win_rate": round(wr, 2),
        "net_profit": round(net, 4),
        "profit_factor": round(min(pf, 999.0), 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "max_drawdown_dollars": round(max_dd_dollars, 4),
        "sharpe_ratio": round(sharpe, 4) if abs(sharpe) < 1e6 else 0.0,
        "expectancy": round(net / n, 4) if n > 0 else 0.0,
        "avg_bars_held": avg_bars,
        "total_costs": round(n * 0.65, 4),
    }


def _calc_sl_tp(exit_option, direction, entry_price, atr_val,
                strategy_ids, df, indicators, signal_bar, registry, scores):
    """Calculate SL/TP distances in pips based on exit option.

    Uses ATR-based calculations for speed (avoids calling strategy.calculate_entry
    which is slow for combo search with thousands of combos).
    """
    pip = BTCUSD_CONFIG["pip_size"]

    # ATR-based distances in pips
    atr_pips = atr_val / pip

    if exit_option == "A":
        # Standard: 1.5x ATR SL, 2.0x ATR TP (R:R = 1.33)
        sl = max(atr_pips * 1.5, 20)
        tp = atr_pips * 2.0
        return sl, max(tp, sl * 1.5)

    elif exit_option == "C":
        # Conservative SL (tighter) + aggressive TP (wider)
        sl = max(atr_pips * 1.0, 20)
        tp = atr_pips * 3.0
        return sl, max(tp, sl * 2.0)

    else:  # "D" — dynamic exit, wider TP since we exit on signal flip
        sl = max(atr_pips * 2.0, 20)
        tp = atr_pips * 4.0
        return sl, max(tp, sl * 1.5)


def _finalize_combo_pnl(trade):
    """Calculate PnL fields for a closed combo trade."""
    pip = BTCUSD_CONFIG["pip_size"]
    pip_value = BTCUSD_CONFIG["pip_value_per_lot"]
    lot = trade.lot_size

    if trade.direction == "BUY":
        trade.pnl_pips = round((trade.exit_price - trade.entry_price) / pip, 1)
    else:
        trade.pnl_pips = round((trade.entry_price - trade.exit_price) / pip, 1)

    trade.gross_pnl_usd = round(trade.pnl_pips * pip_value * lot, 4)
    trade.pnl_usd = trade.gross_pnl_usd
    trade.net_pnl = round(trade.gross_pnl_usd - trade.total_cost, 4)

    sl_dist = trade.sl_distance_pips
    if sl_dist > 0:
        trade.actual_rr = round(trade.pnl_pips / sl_dist, 2)

    if trade.net_pnl > 0:
        trade.outcome = "win"
    elif trade.net_pnl < 0:
        trade.outcome = "loss"
    else:
        trade.outcome = "be"


# ═══════════════════════════════════════════════════════════════
#  SMART COMBINATION SEARCH
# ═══════════════════════════════════════════════════════════════

def compute_composite_score(metrics: dict) -> float:
    """Same scoring as Phase 5."""
    pf = min(metrics.get("profit_factor", 0), 10.0)
    sharpe = max(min(metrics.get("sharpe_ratio", 0), 10.0), -10.0)
    wr = metrics.get("win_rate", 0) / 100.0
    exp = metrics.get("expectancy", 0)
    exp_norm = max(min(exp / 5.0, 2.0), -2.0)
    dd_pct = max(metrics.get("max_drawdown_pct", 1), 1)
    dd_score = min(1.0, 1.0 / dd_pct)
    return round(pf * 0.25 + sharpe * 0.25 + wr * 0.20 +
                 exp_norm * 0.15 + dd_score * 0.15, 4)


def search_combinations(
    signal_matrix: SignalMatrix,
    df, indicators, registry,
    top_n: int = 80,
    max_correlation: float = 0.7,
    verbose: bool = True,
) -> List[Dict]:
    """
    Smart combination search with correlation filtering.

    Steps:
      1. Filter to top_n individually profitable strategies
      2. Build correlation matrix, filter pairs with corr < max_correlation
      3. Test all valid pairs with all 5 modes + 4 exit options
      4. Top pairs → try triples
      5. Top triples → try quads
    """
    scores = signal_matrix.strategy_scores
    categories = signal_matrix.strategy_categories

    # Get top N strategies sorted by score
    candidates = sorted(
        [(sid, scores.get(sid, 0)) for sid in signal_matrix.strategy_ids],
        key=lambda x: x[1], reverse=True
    )[:top_n]
    cand_ids = [c[0] for c in candidates]

    if verbose:
        print(f"\nCandidate strategies: {len(cand_ids)}")
        print(f"Building correlation matrix...")

    # Build correlation lookup
    corr_cache = {}
    for i, a in enumerate(cand_ids):
        for j, b in enumerate(cand_ids):
            if i < j:
                c = signal_matrix.get_correlation(a, b)
                corr_cache[(a, b)] = c
                corr_cache[(b, a)] = c

    # ═══ STEP A: Generate valid pairs ═══
    modes = [ComboMode.UNANIMOUS, ComboMode.MAJORITY, ComboMode.WEIGHTED,
             ComboMode.ANY_CONFIRMED, ComboMode.LEADER_FOLLOWER]
    exit_options = ["A", "C"]  # Test A and C (most distinct); B and D as secondary

    valid_pairs = []
    for a, b in combinations(cand_ids, 2):
        # Rule 1: Different categories
        cat_a = categories.get(a, "")
        cat_b = categories.get(b, "")
        if cat_a == cat_b:
            continue
        # Rule 3: Correlation < threshold
        corr = corr_cache.get((a, b), 0)
        if abs(corr) > max_correlation:
            continue
        valid_pairs.append((a, b, corr))

    if verbose:
        print(f"Valid pairs (diff category, corr < {max_correlation}): {len(valid_pairs)}", flush=True)

    # ═══ STEP B: Test all pairs ═══
    all_results = []
    total_tests = len(valid_pairs) * len(modes) * len(exit_options)
    tested = 0
    t0 = time.time()

    # Pre-extract numpy arrays once for reuse across all combos
    arrays_cache = {}

    for a, b, corr in valid_pairs:
        pair_ids = [a, b]
        # Sort by score (leader first for LEADER_FOLLOWER mode)
        pair_ids.sort(key=lambda s: scores.get(s, 0), reverse=True)

        for mode in modes:
            for exit_opt in exit_options:
                tested += 1
                try:
                    metrics = backtest_combination(
                        signal_matrix, pair_ids, mode, exit_opt,
                        df, indicators, registry, scores,
                        _arrays_cache=arrays_cache,
                    )

                    if metrics["total_trades"] >= 5:
                        combo_score = compute_composite_score(metrics)
                        all_results.append({
                            "strategies": list(pair_ids),
                            "mode": mode.value,
                            "exit_option": exit_opt,
                            "combo_score": combo_score,
                            "correlation": round(corr, 3),
                            "metrics": metrics,
                        })
                except Exception:
                    pass

                if verbose and tested % 200 == 0:
                    elapsed = time.time() - t0
                    rate = tested / elapsed if elapsed > 0 else 0
                    eta = (total_tests - tested) / rate if rate > 0 else 0
                    print(f"  Pairs: {tested}/{total_tests} "
                          f"({rate:.0f}/s, ETA={eta:.0f}s) "
                          f"found={len(all_results)}", flush=True)

    if verbose:
        elapsed = time.time() - t0
        print(f"  Pair search done: {tested} tests in {elapsed:.1f}s, "
              f"{len(all_results)} combos with 5+ trades", flush=True)

    # ═══ STEP C: Top pairs → try triples ═══
    all_results.sort(key=lambda x: x["combo_score"], reverse=True)
    top_pairs = all_results[:200]

    if verbose:
        print(f"\nTesting triples from top {len(top_pairs)} pairs...")

    triple_results = []
    triple_tested = 0

    for pair_result in top_pairs[:100]:
        pair = pair_result["strategies"]
        best_mode = ComboMode(pair_result["mode"])
        best_exit = pair_result["exit_option"]

        pair_cats = {categories.get(s, "") for s in pair}

        for third in cand_ids:
            if third in pair:
                continue
            cat_third = categories.get(third, "")
            if cat_third in pair_cats:
                continue
            # Check correlation with both pair members
            corr_ok = True
            for p in pair:
                c = corr_cache.get((p, third), corr_cache.get((third, p), 0))
                if abs(c) > max_correlation:
                    corr_ok = False
                    break
            if not corr_ok:
                continue

            triple_ids = sorted(pair + [third],
                                key=lambda s: scores.get(s, 0), reverse=True)

            try:
                metrics = backtest_combination(
                    signal_matrix, triple_ids, best_mode, best_exit,
                    df, indicators, registry, scores,
                    _arrays_cache=arrays_cache,
                )
                if metrics["total_trades"] >= 5:
                    combo_score = compute_composite_score(metrics)
                    if combo_score > pair_result["combo_score"] * 0.8:
                        triple_results.append({
                            "strategies": list(triple_ids),
                            "mode": best_mode.value,
                            "exit_option": best_exit,
                            "combo_score": combo_score,
                            "metrics": metrics,
                        })
            except Exception:
                pass

            triple_tested += 1

        if verbose and (top_pairs.index(pair_result) + 1) % 20 == 0:
            print(f"  Triples: {triple_tested} tested, {len(triple_results)} found", flush=True)

    if verbose:
        print(f"  Triple search done: {triple_tested} tests, "
              f"{len(triple_results)} combos")

    # Merge all results
    all_combos = all_results + triple_results
    all_combos.sort(key=lambda x: x["combo_score"], reverse=True)

    if verbose:
        print(f"\nTotal combos found: {len(all_combos)}")

    return all_combos

"""
Backtest Engine — Strategy Base Classes
==========================================
Unified interface for all backtest strategies.
Adapts the various MVP strategy formats into a single BacktestStrategy interface.

Signal flow:
  1. generate_signal(df, indicators, bar_index) → Signal
  2. calculate_entry(signal, df, indicators, bar_index) → EntrySetup
  3. calculate_exit(open_trade, df, indicators, bar_index) → ExitAction
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    """Output of generate_signal()."""
    signal_type: SignalType = SignalType.NEUTRAL
    confidence: float = 0.0
    reason: str = ""
    reason_fa: str = ""
    bar_index: int = 0


@dataclass
class EntrySetup:
    """Output of calculate_entry(). Defines where to enter, SL, TP."""
    direction: str = ""           # "BUY" or "SELL"
    entry_price: float = 0.0     # Filled at next-bar open (adjusted for spread)
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    rr_ratio: float = 0.0
    sl_pips: float = 0.0
    tp1_pips: float = 0.0
    confidence: float = 0.0
    valid: bool = False
    reason: str = ""


@dataclass
class ExitAction:
    """Output of calculate_exit(). Says what to do with an open trade."""
    action: str = "HOLD"         # "HOLD", "CLOSE", "MOVE_SL", "MOVE_TP"
    new_sl: float = 0.0
    new_tp: float = 0.0
    reason: str = ""


class BacktestStrategy:
    """
    Unified backtest strategy interface.

    Every strategy (whether from MVP function-dict or class-based) is wrapped
    into this interface. The wrapper adapts the original function signature.

    Attributes:
        strategy_id: Unique ID like "RSI_01", "MACD_03"
        name: English name
        name_fa: Farsi name
        category: Category prefix like "RSI", "MACD"
        source_file: Original .py file
        params: Strategy parameters
        required_indicators: List of indicator keys needed
        _analyze_func: Reference to the original MVP function
        _func_type: "simple" (df,ctx), "extended" (df,ind,sym,tf), "class"
    """

    def __init__(
        self,
        strategy_id: str,
        name: str = "",
        name_fa: str = "",
        category: str = "",
        source_file: str = "",
        params: Dict[str, Any] = None,
        required_indicators: list = None,
        analyze_func: Callable = None,
        func_type: str = "simple",
    ):
        self.strategy_id = strategy_id
        self.name = name
        self.name_fa = name_fa
        self.category = category
        self.source_file = source_file
        self.params = params or {}
        self.required_indicators = required_indicators or []
        self._analyze_func = analyze_func
        self._func_type = func_type

    def generate_signal(self, df, indicators: Dict, bar_index: int,
                        symbol: str = "BTCUSD", timeframe: str = "H1") -> Signal:
        """
        Generate signal for the given bar. NO lookahead allowed.

        Args:
            df: OHLCV DataFrame (full history up to bar_index)
            indicators: Pre-computed indicator dict from compute_all()
            bar_index: Current bar index (signal generated here, entry on bar_index+1)
            symbol: Symbol being analyzed
            timeframe: Timeframe

        Returns:
            Signal with type, confidence, reason
        """
        if self._analyze_func is None:
            return Signal()

        # Use a view (no copy) up to bar_index to prevent lookahead.
        # iloc slice returns a view when no copy is requested — fast.
        df_slice = df.iloc[:bar_index + 1]

        try:
            if self._func_type == "simple":
                # Pattern: func(df, context=None)
                ctx = self._build_context(df_slice, indicators, bar_index)
                result = self._analyze_func(df_slice, ctx)
            elif self._func_type == "extended":
                # Pattern: func(df, indicators, symbol, timeframe)
                ctx = self._build_context(df_slice, indicators, bar_index)
                result = self._analyze_func(df_slice, ctx, symbol, timeframe)
            else:
                result = {"signal": "NEUTRAL", "confidence": 0}
        except Exception:
            return Signal()

        if not isinstance(result, dict):
            return Signal()

        sig_str = result.get("signal", "NEUTRAL").upper()
        sig_type = SignalType.BUY if sig_str == "BUY" else \
                   SignalType.SELL if sig_str == "SELL" else SignalType.NEUTRAL

        return Signal(
            signal_type=sig_type,
            confidence=float(result.get("confidence", 0)),
            reason=result.get("reason", ""),
            reason_fa=result.get("reason_fa", ""),
            bar_index=bar_index,
        )

    def calculate_entry(self, signal: Signal, df, indicators: Dict,
                        bar_index: int, symbol: str = "BTCUSD") -> EntrySetup:
        """
        Calculate entry/SL/TP for next bar after signal.

        Uses ATR-based SL/TP if the original strategy provided setup data,
        otherwise calculates a default setup.

        Args:
            signal: The signal from generate_signal()
            df: Full OHLCV DataFrame
            indicators: Pre-computed indicators
            bar_index: Signal bar index (entry will be on bar_index + 1)
            symbol: Symbol

        Returns:
            EntrySetup with entry, SL, TP
        """
        if signal.signal_type == SignalType.NEUTRAL:
            return EntrySetup(valid=False, reason="No signal")

        if bar_index + 1 >= len(df):
            return EntrySetup(valid=False, reason="No next bar for entry")

        next_bar = df.iloc[bar_index + 1]
        entry_price = next_bar["open"]

        # Get ATR for SL/TP calculation
        atr_key = "atr_14"
        atr_val = 0.0
        if atr_key in indicators and hasattr(indicators[atr_key], 'iloc'):
            atr_val = float(indicators[atr_key].iloc[bar_index])
            if atr_val != atr_val:  # NaN check
                atr_val = 0.0

        if atr_val <= 0:
            return EntrySetup(valid=False, reason="ATR not available")

        # Default: SL = 1.5 * ATR, TP1 = 1.5 * SL (RR=1.5), TP2 = 3 * SL
        sl_dist = atr_val * 1.5
        tp1_dist = sl_dist * 1.5
        tp2_dist = sl_dist * 3.0

        direction = signal.signal_type.value
        if direction == "BUY":
            sl = entry_price - sl_dist
            tp1 = entry_price + tp1_dist
            tp2 = entry_price + tp2_dist
        else:
            sl = entry_price + sl_dist
            tp1 = entry_price - tp1_dist
            tp2 = entry_price - tp2_dist

        # Pip size for the symbol
        pip = _pip_size(symbol)
        rr = tp1_dist / sl_dist if sl_dist > 0 else 0.0

        return EntrySetup(
            direction=direction,
            entry_price=round(entry_price, 2),
            sl_price=round(sl, 2),
            tp1_price=round(tp1, 2),
            tp2_price=round(tp2, 2),
            rr_ratio=round(rr, 2),
            sl_pips=round(sl_dist / pip, 1) if pip > 0 else 0,
            tp1_pips=round(tp1_dist / pip, 1) if pip > 0 else 0,
            confidence=signal.confidence,
            valid=True,
        )

    def calculate_exit(self, open_trade, df, indicators: Dict,
                       bar_index: int) -> ExitAction:
        """
        Check if an open trade should be modified or closed.
        Default: no action (hold). Subclasses can override.
        """
        return ExitAction(action="HOLD")

    def _build_context(self, df_slice, indicators: Dict, bar_index: int) -> Dict:
        """Build a context dict from pre-computed indicators for MVP strategy funcs."""
        ctx = {}
        try:
            close = df_slice["close"]
            high = df_slice["high"]
            low = df_slice["low"]

            # Populate context keys that MVP strategies expect
            for key in ["ema_9", "ema_21", "ema_50", "ema_200"]:
                if key in indicators and hasattr(indicators[key], 'iloc') and bar_index < len(indicators[key]):
                    v = float(indicators[key].iloc[bar_index])
                    ctx[key] = v if v == v else None
                else:
                    ctx[key] = None

            if "rsi_14" in indicators and hasattr(indicators["rsi_14"], 'iloc') and bar_index < len(indicators["rsi_14"]):
                ctx["rsi_14"] = float(indicators["rsi_14"].iloc[bar_index])

            if "stoch_14_3" in indicators:
                stoch = indicators["stoch_14_3"]
                if isinstance(stoch, dict) and "k" in stoch and bar_index < len(stoch["k"]):
                    ctx["stoch_k"] = float(stoch["k"].iloc[bar_index])

            if "atr_14" in indicators and hasattr(indicators["atr_14"], 'iloc') and bar_index < len(indicators["atr_14"]):
                atr_v = float(indicators["atr_14"].iloc[bar_index])
                ctx["atr_14"] = atr_v
                price = float(close.iloc[-1]) if len(close) > 0 else 1
                ctx["atr_percent"] = (atr_v / price) * 100 if price > 0 else 0

            if "bb_20_2" in indicators:
                bb = indicators["bb_20_2"]
                if isinstance(bb, dict):
                    for bk in ["upper", "lower"]:
                        if bk in bb and bar_index < len(bb[bk]):
                            ctx[f"bb_{bk}"] = float(bb[bk].iloc[bar_index])
                    if "upper" in ctx and "lower" in ctx:
                        price = float(close.iloc[-1])
                        bw = ctx.get("bb_upper", 0) - ctx.get("bb_lower", 0)
                        ctx["bb_percent_b"] = ((price - ctx["bb_lower"]) / bw * 100) if bw > 0 else 50

            if "adx_14" in indicators:
                adx_d = indicators["adx_14"]
                if isinstance(adx_d, dict):
                    for ak in ["adx", "plus_di", "minus_di"]:
                        if ak in adx_d and bar_index < len(adx_d[ak]):
                            ctx[ak] = float(adx_d[ak].iloc[bar_index])

            # Regime
            adx_v = ctx.get("adx", 0)
            if adx_v > 25:
                ctx["regime"] = "trending"
            else:
                ctx["regime"] = "ranging"

            # MA stack
            e9 = ctx.get("ema_9")
            e21 = ctx.get("ema_21")
            e50 = ctx.get("ema_50")
            e200 = ctx.get("ema_200")
            if all(v is not None for v in [e9, e21, e50, e200]):
                if e9 > e21 > e50 > e200:
                    ctx["ma_stack"] = 1
                elif e9 < e21 < e50 < e200:
                    ctx["ma_stack"] = -1
                else:
                    ctx["ma_stack"] = 0
            else:
                ctx["ma_stack"] = 0

            # SuperTrend direction
            if "supertrend_10_3" in indicators:
                st = indicators["supertrend_10_3"]
                if isinstance(st, dict) and "direction" in st and bar_index < len(st["direction"]):
                    ctx["supertrend_dir"] = float(st["direction"].iloc[bar_index])

        except Exception:
            pass

        return ctx

    def __repr__(self):
        return f"BacktestStrategy({self.strategy_id}, {self.category})"


def _pip_size(symbol: str) -> float:
    """Pip size for a given symbol."""
    s = symbol.upper()
    if "JPY" in s:
        return 0.01
    if "XAU" in s:
        return 0.1
    if "XAG" in s:
        return 0.01
    if "BTC" in s:
        return 1.0
    if s in ("NAS100", "US30", "SPX500", "GER40", "UK100"):
        return 1.0
    return 0.0001

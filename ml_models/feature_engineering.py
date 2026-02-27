"""
Feature Engineering Pipeline for ML-driven trading signal generation.

Transforms raw OHLCV data and pre-computed technical indicators into a
structured feature matrix suitable for supervised learning. All features
are computed using only past data to prevent look-ahead bias.

Typical usage:
    >>> fe = FeatureEngineer(lookback=20)
    >>> X, feature_names = fe.build_features(df, indicators)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""

    lookback: int = 20
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    momentum_periods: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    include_time_features: bool = True
    include_volume_features: bool = True
    include_volatility_regime: bool = True
    drop_na: bool = True


class FeatureEngineer:
    """Build ML-ready feature matrices from OHLCV data and indicators.

    The pipeline computes five categories of features:
        1. Technical indicator values (RSI, MACD, Bollinger, ATR, …)
        2. Rolling statistics (mean, std, skewness, kurtosis)
        3. Lag / momentum features (returns over multiple horizons)
        4. Volume-based features (relative volume, OBV slope, MFI)
        5. Volatility-regime features (ATR percentile, squeeze)

    Parameters
    ----------
    config : FeatureConfig, optional
        Feature generation configuration.  Uses defaults when omitted.
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        """Return the ordered list of feature names from the last build."""
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features(
        self,
        df: pd.DataFrame,
        indicators: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Construct the full feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns ``open, high, low, close, tick_volume``.
        indicators : dict, optional
            Pre-computed indicator dict (from ``indicators.compute.compute_all``).

        Returns
        -------
        features : pd.DataFrame
            Feature matrix aligned to *df* index, NaN rows dropped.
        feature_names : list[str]
            Ordered column names.
        """
        logger.info("Building feature matrix (rows=%d)", len(df))
        frames: List[pd.DataFrame] = []

        frames.append(self._price_features(df))
        frames.append(self._rolling_statistics(df))
        frames.append(self._lag_features(df))
        frames.append(self._momentum_features(df))

        if indicators:
            frames.append(self._indicator_features(df, indicators))

        if self.config.include_volume_features:
            frames.append(self._volume_features(df, indicators))

        if self.config.include_volatility_regime and indicators:
            frames.append(self._volatility_regime_features(df, indicators))

        if self.config.include_time_features:
            frames.append(self._time_features(df))

        features = pd.concat(frames, axis=1)

        if self.config.drop_na:
            features = features.dropna()

        # Replace infinities that can break tree models
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0.0, inplace=True)

        self._feature_names = features.columns.tolist()
        logger.info(
            "Feature matrix ready: %d rows × %d features",
            len(features),
            len(self._feature_names),
        )
        return features, self._feature_names

    def build_target(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        threshold_pct: float = 0.3,
    ) -> pd.Series:
        """Create a classification target: 1 (buy), 0 (hold), -1 (sell).

        The target is based on the *future* return over ``horizon`` bars
        compared to a neutral ``threshold_pct`` band.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        horizon : int
            Number of bars to look ahead for return calculation.
        threshold_pct : float
            Minimum absolute percent move to classify as buy/sell.

        Returns
        -------
        pd.Series
            Target labels aligned to the DataFrame index.
        """
        future_return = df["close"].pct_change(horizon).shift(-horizon) * 100.0

        target = pd.Series(0, index=df.index, name="target")
        target[future_return > threshold_pct] = 1
        target[future_return < -threshold_pct] = -1
        return target

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core price-derived features."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        feats: Dict[str, pd.Series] = {}

        # Bar structure
        feats["body_pct"] = (close - open_) / open_ * 100
        feats["upper_shadow_pct"] = (high - np.maximum(close, open_)) / open_ * 100
        feats["lower_shadow_pct"] = (np.minimum(close, open_) - low) / open_ * 100
        feats["bar_range_pct"] = (high - low) / open_ * 100

        # Distance from recent extremes
        for w in [10, 20, 50]:
            feats[f"dist_high_{w}"] = (close - high.rolling(w).max()) / close * 100
            feats[f"dist_low_{w}"] = (close - low.rolling(w).min()) / close * 100

        return pd.DataFrame(feats, index=df.index)

    def _rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean, std, skewness, and kurtosis of returns."""
        returns = df["close"].pct_change()
        feats: Dict[str, pd.Series] = {}

        for w in self.config.rolling_windows:
            roll = returns.rolling(w)
            feats[f"ret_mean_{w}"] = roll.mean()
            feats[f"ret_std_{w}"] = roll.std()
            feats[f"ret_skew_{w}"] = roll.skew()
            feats[f"ret_kurt_{w}"] = roll.apply(
                lambda x: x.kurt() if len(x) >= 4 else 0.0, raw=False
            )

        return pd.DataFrame(feats, index=df.index)

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged close prices and returns."""
        close = df["close"]
        returns = close.pct_change()
        feats: Dict[str, pd.Series] = {}

        for lag in self.config.lag_periods:
            feats[f"close_lag_{lag}"] = close.shift(lag)
            feats[f"return_lag_{lag}"] = returns.shift(lag)

        return pd.DataFrame(feats, index=df.index)

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-horizon momentum and rate-of-change."""
        close = df["close"]
        feats: Dict[str, pd.Series] = {}

        for p in self.config.momentum_periods:
            feats[f"momentum_{p}"] = close.pct_change(p) * 100
            feats[f"roc_{p}"] = (close / close.shift(p) - 1.0) * 100

        # Acceleration (momentum of momentum)
        mom_5 = close.pct_change(5)
        feats["momentum_accel"] = mom_5 - mom_5.shift(5)

        return pd.DataFrame(feats, index=df.index)

    def _indicator_features(
        self, df: pd.DataFrame, indicators: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Extract pre-computed technical indicators as features.

        Automatically discovers available indicators from the dict keys
        and converts them into aligned DataFrame columns.
        """
        feats: Dict[str, pd.Series] = {}

        # Direct indicator values
        indicator_keys = [
            "rsi_14", "rsi_7", "rsi_21",
            "macd_line", "macd_signal", "macd_histogram",
            "bb_upper_20", "bb_middle_20", "bb_lower_20",
            "atr_14", "atr_7",
            "adx_14", "plus_di_14", "minus_di_14",
            "cci_14", "cci_20",
            "stoch_k_14", "stoch_d_14",
            "williams_r_14",
            "obv", "mfi_14", "cmf_20",
            "trix_15",
            "supertrend_10_3",
            "squeeze_momentum",
        ]

        for key in indicator_keys:
            if key in indicators:
                arr = indicators[key]
                if isinstance(arr, np.ndarray) and arr.shape[0] == len(df):
                    feats[f"ind_{key}"] = pd.Series(arr, index=df.index)

        # Derived indicator features
        close = df["close"]

        if "bb_upper_20" in indicators and "bb_lower_20" in indicators:
            bb_u = pd.Series(indicators["bb_upper_20"], index=df.index)
            bb_l = pd.Series(indicators["bb_lower_20"], index=df.index)
            bb_width = bb_u - bb_l
            feats["bb_width_pct"] = bb_width / close * 100
            feats["bb_position"] = (close - bb_l) / bb_width.replace(0, np.nan)

        if "atr_14" in indicators:
            atr = pd.Series(indicators["atr_14"], index=df.index)
            feats["atr_pct"] = atr / close * 100
            feats["atr_ratio_7_14"] = (
                pd.Series(indicators.get("atr_7", atr.values), index=df.index) / atr
            )

        if "rsi_14" in indicators:
            rsi = pd.Series(indicators["rsi_14"], index=df.index)
            feats["rsi_slope_5"] = rsi - rsi.shift(5)
            feats["rsi_overbought"] = (rsi > 70).astype(float)
            feats["rsi_oversold"] = (rsi < 30).astype(float)

        if "macd_histogram" in indicators:
            hist = pd.Series(indicators["macd_histogram"], index=df.index)
            feats["macd_hist_slope"] = hist - hist.shift(1)
            feats["macd_hist_accel"] = feats["macd_hist_slope"] - feats["macd_hist_slope"].shift(1)

        # Moving-average cross features
        for period in [20, 50, 200]:
            sma_key = f"sma_{period}"
            if sma_key in indicators:
                sma = pd.Series(indicators[sma_key], index=df.index)
                feats[f"close_vs_sma_{period}"] = (close - sma) / sma * 100

        if "ema_20" in indicators and "ema_50" in indicators:
            ema20 = pd.Series(indicators["ema_20"], index=df.index)
            ema50 = pd.Series(indicators["ema_50"], index=df.index)
            feats["ema_cross_20_50"] = (ema20 - ema50) / ema50 * 100

        return pd.DataFrame(feats, index=df.index)

    def _volume_features(
        self, df: pd.DataFrame, indicators: Optional[Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """Volume-based features."""
        feats: Dict[str, pd.Series] = {}
        vol = df.get("tick_volume", df.get("volume", pd.Series(0, index=df.index)))

        if vol.sum() == 0:
            return pd.DataFrame(index=df.index)

        for w in [5, 10, 20]:
            vol_ma = vol.rolling(w).mean()
            feats[f"vol_ratio_{w}"] = vol / vol_ma.replace(0, np.nan)

        feats["vol_change"] = vol.pct_change()
        feats["vol_std_20"] = vol.rolling(20).std() / vol.rolling(20).mean()

        # Price-volume divergence
        close_chg = df["close"].pct_change(5)
        vol_chg = vol.pct_change(5)
        feats["price_vol_corr_20"] = close_chg.rolling(20).corr(vol_chg)

        if indicators and "obv" in indicators:
            obv = pd.Series(indicators["obv"], index=df.index)
            feats["obv_slope_10"] = (obv - obv.shift(10)) / (obv.shift(10).abs() + 1e-8)

        return pd.DataFrame(feats, index=df.index)

    def _volatility_regime_features(
        self, df: pd.DataFrame, indicators: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Volatility regime classification features."""
        feats: Dict[str, pd.Series] = {}

        if "atr_14" in indicators:
            atr = pd.Series(indicators["atr_14"], index=df.index)
            feats["atr_percentile_50"] = atr.rolling(50).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            feats["atr_expanding"] = (atr > atr.rolling(20).mean()).astype(float)

        # Realized volatility
        log_ret = np.log(df["close"] / df["close"].shift(1))
        for w in [10, 20]:
            feats[f"realized_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

        # Range contraction / expansion
        bar_range = df["high"] - df["low"]
        feats["range_ratio_5_20"] = (
            bar_range.rolling(5).mean() / bar_range.rolling(20).mean()
        )

        if "squeeze_momentum" in indicators:
            sq = pd.Series(indicators["squeeze_momentum"], index=df.index)
            feats["squeeze_on"] = (sq == 0).astype(float)

        return pd.DataFrame(feats, index=df.index)

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calendar / session features (cyclic encoding)."""
        feats: Dict[str, pd.Series] = {}

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(df.index)
            except Exception:
                return pd.DataFrame(index=df.index)
        else:
            idx = df.index

        # Cyclic encoding avoids discontinuities at midnight / Sunday
        feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        feats["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        feats["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        feats["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

        return pd.DataFrame(feats, index=df.index)

"""
AI-Powered Performance Analyzer — Risk metrics, Monte Carlo, and statistical testing.

Provides a comprehensive suite of risk-adjusted performance metrics,
Monte Carlo simulation for drawdown estimation, and statistical
significance tests for strategy evaluation.

Typical usage:
    >>> analyzer = PerformanceAnalyzer()
    >>> metrics = analyzer.compute_metrics(equity_curve, trades)
    >>> mc = analyzer.monte_carlo_simulation(returns, n_simulations=10000)
    >>> report = analyzer.generate_report(metrics, mc)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    """Comprehensive strategy performance metrics."""

    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    cagr_pct: float = 0.0

    # Risk-adjusted ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0
    avg_drawdown_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0

    # Tail risk
    var_95: float = 0.0
    cvar_95: float = 0.0
    var_99: float = 0.0
    cvar_99: float = 0.0

    # Distribution
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Statistical tests
    t_stat: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation output."""

    n_simulations: int = 0
    median_final_equity: float = 0.0
    p5_final_equity: float = 0.0
    p95_final_equity: float = 0.0
    median_max_drawdown: float = 0.0
    p95_max_drawdown: float = 0.0
    p99_max_drawdown: float = 0.0
    probability_of_profit: float = 0.0
    probability_of_ruin: float = 0.0
    ruin_threshold_pct: float = -50.0
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────

class PerformanceAnalyzer:
    """Compute risk-adjusted performance metrics and run Monte Carlo simulations.

    Parameters
    ----------
    risk_free_rate : float
        Annualized risk-free rate for Sharpe/Sortino (default 0.04 = 4%).
    trading_periods_per_year : int
        Number of trading bars per year (e.g. 252 for daily, 6300 for H1).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        trading_periods_per_year: int = 6300,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = trading_periods_per_year

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        equity_curve: np.ndarray,
        trade_pnls: Optional[np.ndarray] = None,
    ) -> PerformanceMetrics:
        """Compute all performance metrics from an equity curve.

        Parameters
        ----------
        equity_curve : np.ndarray
            Time-ordered equity values (starting capital → final).
        trade_pnls : np.ndarray, optional
            Per-trade P&L values.  If provided, trade-level stats are computed.

        Returns
        -------
        PerformanceMetrics
            Populated metrics dataclass.
        """
        eq = np.asarray(equity_curve, dtype=np.float64)
        returns = np.diff(eq) / eq[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2:
            logger.warning("Insufficient data for metrics (%d returns)", len(returns))
            return PerformanceMetrics()

        m = PerformanceMetrics()

        # ── Returns ──
        m.total_return_pct = (eq[-1] / eq[0] - 1.0) * 100
        n_periods = len(returns)
        m.annualized_return_pct = float(np.mean(returns) * self.periods_per_year * 100)
        years = n_periods / self.periods_per_year
        if years > 0 and eq[0] > 0:
            m.cagr_pct = ((eq[-1] / eq[0]) ** (1.0 / max(years, 1e-9)) - 1.0) * 100

        # ── Risk-adjusted ratios ──
        m.sharpe_ratio = self._sharpe(returns)
        m.sortino_ratio = self._sortino(returns)
        m.calmar_ratio = self._calmar(returns, eq)
        m.omega_ratio = self._omega(returns)

        # ── Drawdown ──
        dd = self._drawdown_series(eq)
        m.max_drawdown_pct = float(np.min(dd) * 100) if len(dd) > 0 else 0.0
        m.max_drawdown_duration_bars = self._max_dd_duration(eq)
        m.avg_drawdown_pct = float(np.mean(dd[dd < 0]) * 100) if np.any(dd < 0) else 0.0

        # ── Tail risk ──
        m.var_95 = float(np.percentile(returns, 5) * 100)
        m.cvar_95 = float(np.mean(returns[returns <= np.percentile(returns, 5)]) * 100)
        m.var_99 = float(np.percentile(returns, 1) * 100)
        m.cvar_99 = float(np.mean(returns[returns <= np.percentile(returns, 1)]) * 100)

        # ── Distribution ──
        m.skewness = float(sp_stats.skew(returns))
        m.kurtosis = float(sp_stats.kurtosis(returns))

        # ── Statistical significance ──
        t_stat, p_value = sp_stats.ttest_1samp(returns, 0)
        m.t_stat = float(t_stat)
        m.p_value = float(p_value)
        m.is_significant = p_value < 0.05 and np.mean(returns) > 0

        # ── Trade-level stats ──
        if trade_pnls is not None and len(trade_pnls) > 0:
            pnl = np.asarray(trade_pnls, dtype=np.float64)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]

            m.total_trades = len(pnl)
            m.win_rate = len(wins) / len(pnl) * 100 if len(pnl) > 0 else 0.0
            m.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            m.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            m.payoff_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss != 0 else 0.0
            m.profit_factor = (
                float(np.sum(wins) / abs(np.sum(losses)))
                if np.sum(losses) != 0 else float("inf")
            )
            m.expectancy = float(np.mean(pnl))

        return m

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self,
        returns: np.ndarray,
        n_simulations: int = 10_000,
        n_periods: Optional[int] = None,
        initial_equity: float = 10_000.0,
        ruin_threshold_pct: float = -50.0,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation by resampling historical returns.

        Randomly shuffles the return sequence to generate alternative
        equity paths, estimating the distribution of outcomes.

        Parameters
        ----------
        returns : np.ndarray
            Historical period returns (e.g. bar-by-bar).
        n_simulations : int
            Number of random paths to simulate.
        n_periods : int, optional
            Path length (defaults to len(returns)).
        initial_equity : float
            Starting capital.
        ruin_threshold_pct : float
            Drawdown threshold to count as "ruin" (e.g. -50%).

        Returns
        -------
        MonteCarloResult
            Simulation statistics.
        """
        ret = np.asarray(returns, dtype=np.float64)
        ret = ret[np.isfinite(ret)]

        if len(ret) < 10:
            logger.warning("Too few returns for Monte Carlo (%d)", len(ret))
            return MonteCarloResult()

        n_per = n_periods or len(ret)
        rng = np.random.default_rng(42)

        final_equities = np.zeros(n_simulations)
        max_drawdowns = np.zeros(n_simulations)

        for i in range(n_simulations):
            sampled = rng.choice(ret, size=n_per, replace=True)
            equity = initial_equity * np.cumprod(1.0 + sampled)
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            final_equities[i] = equity[-1]
            max_drawdowns[i] = np.min(dd)

        ruin_count = np.sum(max_drawdowns * 100 <= ruin_threshold_pct)

        result = MonteCarloResult(
            n_simulations=n_simulations,
            median_final_equity=float(np.median(final_equities)),
            p5_final_equity=float(np.percentile(final_equities, 5)),
            p95_final_equity=float(np.percentile(final_equities, 95)),
            median_max_drawdown=float(np.median(max_drawdowns) * 100),
            p95_max_drawdown=float(np.percentile(max_drawdowns, 5) * 100),
            p99_max_drawdown=float(np.percentile(max_drawdowns, 1) * 100),
            probability_of_profit=float(np.mean(final_equities > initial_equity)),
            probability_of_ruin=float(ruin_count / n_simulations),
            ruin_threshold_pct=ruin_threshold_pct,
            confidence_intervals={
                "90%": (
                    float(np.percentile(final_equities, 5)),
                    float(np.percentile(final_equities, 95)),
                ),
                "95%": (
                    float(np.percentile(final_equities, 2.5)),
                    float(np.percentile(final_equities, 97.5)),
                ),
            },
        )

        logger.info(
            "Monte Carlo (%d sims): median=%.2f, P95 DD=%.2f%%, ruin=%.2f%%",
            n_simulations,
            result.median_final_equity,
            result.p95_max_drawdown,
            result.probability_of_ruin * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Statistical testing
    # ------------------------------------------------------------------

    def test_strategy_significance(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Statistical significance tests for strategy returns.

        Parameters
        ----------
        strategy_returns : np.ndarray
            Strategy period returns.
        benchmark_returns : np.ndarray, optional
            Benchmark returns for paired comparison.

        Returns
        -------
        dict
            Test results including t-statistics and p-values.
        """
        strat = np.asarray(strategy_returns, dtype=np.float64)
        strat = strat[np.isfinite(strat)]

        results: Dict[str, Any] = {}

        # 1. One-sample t-test: are returns significantly > 0?
        t_stat, p_val = sp_stats.ttest_1samp(strat, 0)
        results["one_sample_t"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_5pct": bool(p_val < 0.05 and np.mean(strat) > 0),
        }

        # 2. Jarque-Bera normality test
        jb_stat, jb_p = sp_stats.jarque_bera(strat)
        results["normality_jb"] = {
            "statistic": float(jb_stat),
            "p_value": float(jb_p),
            "is_normal": bool(jb_p > 0.05),
        }

        # 3. Runs test for randomness
        median = np.median(strat)
        runs = np.diff(np.sign(strat - median))
        n_runs = np.sum(runs != 0) + 1
        n_pos = np.sum(strat > median)
        n_neg = np.sum(strat <= median)
        n = n_pos + n_neg
        if n > 0 and n_pos > 0 and n_neg > 0:
            expected_runs = 1 + (2 * n_pos * n_neg) / n
            var_runs = (
                (2 * n_pos * n_neg * (2 * n_pos * n_neg - n))
                / (n**2 * (n - 1))
            )
            z_runs = (n_runs - expected_runs) / (np.sqrt(var_runs) + 1e-10)
            results["runs_test"] = {
                "n_runs": int(n_runs),
                "expected_runs": float(expected_runs),
                "z_statistic": float(z_runs),
                "is_random": bool(abs(z_runs) < 1.96),
            }

        # 4. Paired comparison vs benchmark
        if benchmark_returns is not None:
            bench = np.asarray(benchmark_returns, dtype=np.float64)
            min_len = min(len(strat), len(bench))
            t_paired, p_paired = sp_stats.ttest_rel(
                strat[:min_len], bench[:min_len]
            )
            results["vs_benchmark"] = {
                "t_statistic": float(t_paired),
                "p_value": float(p_paired),
                "strategy_beats_benchmark": bool(
                    p_paired < 0.05 and np.mean(strat[:min_len]) > np.mean(bench[:min_len])
                ),
            }

        return results

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        metrics: PerformanceMetrics,
        mc_result: Optional[MonteCarloResult] = None,
    ) -> str:
        """Generate a formatted text performance report.

        Parameters
        ----------
        metrics : PerformanceMetrics
            Computed metrics.
        mc_result : MonteCarloResult, optional
            Monte Carlo simulation results.

        Returns
        -------
        str
            Multi-line formatted report.
        """
        lines = [
            "=" * 60,
            "  AI PERFORMANCE ANALYSIS REPORT",
            "=" * 60,
            "",
            "--- Returns ---",
            f"  Total Return:       {metrics.total_return_pct:>10.2f}%",
            f"  Annualized Return:  {metrics.annualized_return_pct:>10.2f}%",
            f"  CAGR:               {metrics.cagr_pct:>10.2f}%",
            "",
            "--- Risk-Adjusted Ratios ---",
            f"  Sharpe Ratio:       {metrics.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio:      {metrics.sortino_ratio:>10.3f}",
            f"  Calmar Ratio:       {metrics.calmar_ratio:>10.3f}",
            f"  Omega Ratio:        {metrics.omega_ratio:>10.3f}",
            "",
            "--- Drawdown ---",
            f"  Max Drawdown:       {metrics.max_drawdown_pct:>10.2f}%",
            f"  Max DD Duration:    {metrics.max_drawdown_duration_bars:>10d} bars",
            f"  Avg Drawdown:       {metrics.avg_drawdown_pct:>10.2f}%",
            "",
            "--- Tail Risk ---",
            f"  VaR (95%):          {metrics.var_95:>10.3f}%",
            f"  CVaR (95%):         {metrics.cvar_95:>10.3f}%",
            f"  VaR (99%):          {metrics.var_99:>10.3f}%",
            f"  CVaR (99%):         {metrics.cvar_99:>10.3f}%",
            "",
            "--- Trade Stats ---",
            f"  Total Trades:       {metrics.total_trades:>10d}",
            f"  Win Rate:           {metrics.win_rate:>10.1f}%",
            f"  Profit Factor:      {metrics.profit_factor:>10.3f}",
            f"  Expectancy:         ${metrics.expectancy:>9.2f}",
            f"  Payoff Ratio:       {metrics.payoff_ratio:>10.3f}",
            "",
            "--- Statistical Significance ---",
            f"  t-statistic:        {metrics.t_stat:>10.3f}",
            f"  p-value:            {metrics.p_value:>10.6f}",
            f"  Significant (5%):   {'YES' if metrics.is_significant else 'NO':>10s}",
        ]

        if mc_result and mc_result.n_simulations > 0:
            lines.extend([
                "",
                f"--- Monte Carlo ({mc_result.n_simulations:,} simulations) ---",
                f"  Median Final Equity:  ${mc_result.median_final_equity:>12,.2f}",
                f"  5th Percentile:       ${mc_result.p5_final_equity:>12,.2f}",
                f"  95th Percentile:      ${mc_result.p95_final_equity:>12,.2f}",
                f"  P95 Max Drawdown:     {mc_result.p95_max_drawdown:>10.2f}%",
                f"  P99 Max Drawdown:     {mc_result.p99_max_drawdown:>10.2f}%",
                f"  Prob of Profit:       {mc_result.probability_of_profit * 100:>10.1f}%",
                f"  Prob of Ruin:         {mc_result.probability_of_ruin * 100:>10.1f}%",
            ])

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sharpe(self, returns: np.ndarray) -> float:
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        std = np.std(excess, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(self.periods_per_year))

    def _sortino(self, returns: np.ndarray) -> float:
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        downside = excess[excess < 0]
        down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
        if down_std < 1e-10:
            return 0.0
        return float(np.mean(excess) / down_std * np.sqrt(self.periods_per_year))

    def _calmar(self, returns: np.ndarray, equity: np.ndarray) -> float:
        dd = self._drawdown_series(equity)
        max_dd = abs(np.min(dd)) if len(dd) > 0 else 1e-10
        ann_ret = np.mean(returns) * self.periods_per_year
        if max_dd < 1e-10:
            return 0.0
        return float(ann_ret / max_dd)

    def _omega(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        gains = np.sum(returns[returns > threshold] - threshold)
        losses = np.sum(threshold - returns[returns <= threshold])
        if losses < 1e-10:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    @staticmethod
    def _drawdown_series(equity: np.ndarray) -> np.ndarray:
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
        return dd

    @staticmethod
    def _max_dd_duration(equity: np.ndarray) -> int:
        peak = np.maximum.accumulate(equity)
        in_dd = equity < peak
        max_dur = 0
        current = 0
        for is_dd in in_dd:
            if is_dd:
                current += 1
                max_dur = max(max_dur, current)
            else:
                current = 0
        return max_dur

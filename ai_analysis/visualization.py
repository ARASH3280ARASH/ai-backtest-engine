"""
ML Visualization Module — Publication-quality charts for model evaluation.

Generates feature importance plots, model comparison charts, equity curves
with confidence bands, prediction-vs-actual overlays, and confusion matrices.

All plots are saved as PNG files and can optionally be displayed inline.

Typical usage:
    >>> viz = MLVisualizer(output_dir="reports/plots")
    >>> viz.plot_feature_importances(importances, top_n=20)
    >>> viz.plot_equity_curve(equity, trades)
    >>> viz.plot_model_comparison(results)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not installed — visualization disabled")

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

STYLE_DEFAULTS = {
    "figure.figsize": (14, 8),
    "figure.dpi": 150,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "font.size": 11,
}


class MLVisualizer:
    """Generate and save ML evaluation charts.

    Parameters
    ----------
    output_dir : str
        Directory to save PNG files (created if needed).
    style : str
        Matplotlib style name (e.g. ``"seaborn-v0_8-darkgrid"``).
    """

    def __init__(
        self,
        output_dir: str = "reports/plots",
        style: Optional[str] = None,
    ) -> None:
        if not _HAS_MPL:
            raise ImportError("matplotlib is required — pip install matplotlib")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(STYLE_DEFAULTS)
        if style:
            plt.style.use(style)
        if _HAS_SNS:
            sns.set_palette("deep")

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------

    def plot_feature_importances(
        self,
        importances: Dict[str, float],
        top_n: int = 20,
        title: str = "Top Feature Importances",
        filename: str = "feature_importances.png",
    ) -> Path:
        """Horizontal bar chart of feature importances.

        Parameters
        ----------
        importances : dict[str, float]
            Feature name → importance score.
        top_n : int
            Number of top features to display.
        title : str
            Chart title.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Saved file path.
        """
        sorted_imp = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        names = [k for k, _ in reversed(sorted_imp)]
        values = [v for _, v in reversed(sorted_imp)]

        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        ax.barh(names, values, color=colors)
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        filename: str = "model_comparison.png",
    ) -> Path:
        """Grouped bar chart comparing models across metrics.

        Parameters
        ----------
        results : dict
            ``{model_name: {metric_name: value, ...}, ...}``.
        metrics : list[str], optional
            Metrics to compare (defaults to accuracy, precision, recall, f1).
        filename : str
            Output filename.

        Returns
        -------
        Path
            Saved file path.
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]

        models = list(results.keys())
        n_metrics = len(metrics)
        x = np.arange(len(models))
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

        for i, metric in enumerate(metrics):
            vals = [results[m].get(metric, 0) for m in models]
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=metric.title(), color=colors[i])

            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def plot_equity_curve(
        self,
        equity: np.ndarray,
        trades: Optional[List[Dict[str, Any]]] = None,
        confidence_band: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        title: str = "Strategy Equity Curve",
        filename: str = "equity_curve.png",
    ) -> Path:
        """Equity curve with optional trade markers and confidence band.

        Parameters
        ----------
        equity : np.ndarray
            Equity values over time.
        trades : list[dict], optional
            Trade entries/exits with keys ``bar_index``, ``pnl``, ``direction``.
        confidence_band : tuple(lower, upper), optional
            Lower and upper equity bounds (e.g. from Monte Carlo).
        title : str
            Chart title.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Saved file path.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.25)

        # ── Panel 1: Equity ──
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(equity, color="#2196F3", linewidth=1.2, label="Equity")

        if confidence_band is not None:
            lower, upper = confidence_band
            ax1.fill_between(
                range(len(equity)), lower, upper,
                alpha=0.15, color="#2196F3", label="Confidence Band",
            )

        if trades:
            for t in trades:
                idx = t.get("bar_index", t.get("exit_bar_index", 0))
                pnl = t.get("pnl", t.get("net_pnl", 0))
                if 0 <= idx < len(equity):
                    color = "#4CAF50" if pnl > 0 else "#F44336"
                    marker = "^" if t.get("direction") == "BUY" else "v"
                    ax1.scatter(idx, equity[idx], c=color, marker=marker, s=20, alpha=0.7, zorder=3)

        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")

        # ── Panel 2: Drawdown ──
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        peak = np.maximum.accumulate(equity)
        dd_pct = (equity - peak) / np.where(peak > 0, peak, 1.0) * 100
        ax2.fill_between(range(len(dd_pct)), dd_pct, 0, color="#F44336", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")

        # ── Panel 3: Returns ──
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        returns = np.diff(equity) / equity[:-1] * 100
        colors = ["#4CAF50" if r > 0 else "#F44336" for r in returns]
        ax3.bar(range(len(returns)), returns, color=colors, alpha=0.6, width=1.0)
        ax3.set_ylabel("Return (%)")
        ax3.set_xlabel("Bar")

        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Prediction vs actual
    # ------------------------------------------------------------------

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: Optional[np.ndarray] = None,
        filename: str = "predictions_vs_actual.png",
    ) -> Path:
        """Overlay predicted signals on actual price/return movements.

        Parameters
        ----------
        y_true : np.ndarray
            True labels (-1, 0, 1).
        y_pred : np.ndarray
            Predicted labels.
        prices : np.ndarray, optional
            Price series for overlay.  Uses cumulative returns if omitted.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Saved file path.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])

        n = len(y_true)
        x = np.arange(n)

        # Panel 1: Price with signal markers
        ax = axes[0]
        if prices is not None:
            ax.plot(x, prices[:n], color="#607D8B", linewidth=0.8, alpha=0.8)
            y_vals = prices[:n]
        else:
            cum = np.cumsum(y_true)
            ax.plot(x, cum, color="#607D8B", linewidth=0.8, alpha=0.8)
            y_vals = cum

        buy_mask = y_pred == 1
        sell_mask = y_pred == -1
        ax.scatter(x[buy_mask], y_vals[buy_mask], c="#4CAF50", marker="^", s=30, label="Pred BUY", zorder=3)
        ax.scatter(x[sell_mask], y_vals[sell_mask], c="#F44336", marker="v", s=30, label="Pred SELL", zorder=3)
        ax.set_title("Predictions vs Actual", fontsize=14, fontweight="bold")
        ax.legend()
        ax.set_ylabel("Price" if prices is not None else "Cumulative Signal")

        # Panel 2: Agreement
        ax2 = axes[1]
        agreement = (y_true == y_pred).astype(int)
        rolling_acc = pd.Series(agreement).rolling(50, min_periods=1).mean()
        ax2.plot(rolling_acc, color="#FF9800", linewidth=1.2)
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50%")
        ax2.set_ylabel("Rolling Accuracy (50-bar)")
        ax2.set_xlabel("Bar")
        ax2.legend()
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        filename: str = "confusion_matrix.png",
    ) -> Path:
        """Plot a confusion matrix heatmap.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        labels : list[str], optional
            Class names (defaults to ["Sell", "Hold", "Buy"]).
        filename : str
            Output filename.

        Returns
        -------
        Path
            Saved file path.
        """
        from sklearn.metrics import confusion_matrix

        if labels is None:
            labels = ["Sell", "Hold", "Buy"]

        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Counts
        if _HAS_SNS:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                        yticklabels=labels, ax=ax1)
        else:
            im = ax1.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax1.text(j, i, str(cm[i, j]), ha="center", va="center")
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels)
            ax1.set_yticks(range(len(labels)))
            ax1.set_yticklabels(labels)
        ax1.set_title("Confusion Matrix (Counts)", fontweight="bold")
        ax1.set_ylabel("Actual")
        ax1.set_xlabel("Predicted")

        # Percentages
        if _HAS_SNS:
            sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Oranges",
                        xticklabels=labels, yticklabels=labels, ax=ax2)
        else:
            ax2.imshow(cm_pct, cmap="Oranges")
            for i in range(cm_pct.shape[0]):
                for j in range(cm_pct.shape[1]):
                    ax2.text(j, i, f"{cm_pct[i, j]:.1f}%", ha="center", va="center")
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels)
            ax2.set_yticks(range(len(labels)))
            ax2.set_yticklabels(labels)
        ax2.set_title("Confusion Matrix (Normalized %)", fontweight="bold")
        ax2.set_ylabel("Actual")
        ax2.set_xlabel("Predicted")

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        filename: str = "training_history.png",
    ) -> Path:
        """Plot training & validation loss/accuracy curves.

        Parameters
        ----------
        history : dict
            Keras-style history dict with keys like ``loss``, ``val_loss``,
            ``accuracy``, ``val_accuracy``.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Saved file path.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        epochs = range(1, len(history.get("loss", [])) + 1)

        # Loss
        if "loss" in history:
            ax1.plot(epochs, history["loss"], label="Train Loss", color="#2196F3")
        if "val_loss" in history:
            ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#F44336", linestyle="--")
        ax1.set_title("Loss", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Accuracy
        if "accuracy" in history:
            ax2.plot(epochs, history["accuracy"], label="Train Acc", color="#2196F3")
        if "val_accuracy" in history:
            ax2.plot(epochs, history["val_accuracy"], label="Val Acc", color="#F44336", linestyle="--")
        ax2.set_title("Accuracy", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Monte Carlo fan chart
    # ------------------------------------------------------------------

    def plot_monte_carlo(
        self,
        simulated_paths: np.ndarray,
        filename: str = "monte_carlo.png",
    ) -> Path:
        """Fan chart of Monte Carlo simulated equity paths.

        Parameters
        ----------
        simulated_paths : np.ndarray
            Shape ``(n_simulations, n_periods)``.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Saved file path.
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        n_sims = min(simulated_paths.shape[0], 200)
        x = np.arange(simulated_paths.shape[1])

        for i in range(n_sims):
            ax.plot(x, simulated_paths[i], color="#90CAF9", alpha=0.05, linewidth=0.5)

        p5 = np.percentile(simulated_paths, 5, axis=0)
        p50 = np.percentile(simulated_paths, 50, axis=0)
        p95 = np.percentile(simulated_paths, 95, axis=0)

        ax.fill_between(x, p5, p95, alpha=0.2, color="#2196F3", label="90% CI")
        ax.plot(x, p50, color="#1565C0", linewidth=2, label="Median")
        ax.plot(x, p5, color="#F44336", linewidth=1, linestyle="--", label="5th %ile")
        ax.plot(x, p95, color="#4CAF50", linewidth=1, linestyle="--", label="95th %ile")

        ax.set_title("Monte Carlo Simulation", fontsize=14, fontweight="bold")
        ax.set_xlabel("Period")
        ax.set_ylabel("Equity ($)")
        ax.legend()

        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

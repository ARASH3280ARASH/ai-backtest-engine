"""
Prediction Engine — Ensemble signal generation with confidence scoring.

Combines predictions from multiple trained ML models into a unified
buy / sell / hold signal with calibrated confidence.  Designed to
integrate directly with the existing BacktestStrategy interface.

Typical usage:
    >>> engine = PredictionEngine.from_directory("models/")
    >>> signal = engine.predict(features_row)
    >>> print(signal.direction, signal.confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    """Prediction signal directions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class PredictionSignal:
    """Output of the ensemble prediction engine."""

    direction: SignalDirection
    confidence: float  # 0.0 – 1.0
    model_votes: Dict[str, int]
    model_probabilities: Dict[str, List[float]]
    agreement_ratio: float  # fraction of models that agree


class EnsembleMethod(str, Enum):
    """Supported ensemble aggregation methods."""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    PROBABILITY_AVG = "probability_avg"
    UNANIMOUS = "unanimous"


class PredictionEngine:
    """Generate ensemble predictions from multiple ML models.

    The engine loads persisted models from disk, runs each one on the
    same feature vector, and aggregates their outputs into a single
    signal with an associated confidence score.

    Parameters
    ----------
    models : dict[str, Any]
        Name → fitted model mapping.
    scaler : object, optional
        Fitted ``StandardScaler`` for feature normalization.
    feature_names : list[str], optional
        Expected feature column order.
    ensemble_method : EnsembleMethod
        Aggregation strategy.
    model_weights : dict[str, float], optional
        Per-model weights for ``WEIGHTED_VOTE``.
    min_confidence : float
        Minimum confidence to emit a directional signal (else HOLD).
    """

    def __init__(
        self,
        models: Dict[str, Any],
        scaler: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        ensemble_method: EnsembleMethod = EnsembleMethod.PROBABILITY_AVG,
        model_weights: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.45,
    ) -> None:
        self._models = models
        self._scaler = scaler
        self._feature_names = feature_names or []
        self.ensemble_method = ensemble_method
        self.model_weights = model_weights or {k: 1.0 for k in models}
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        directory: str,
        ensemble_method: EnsembleMethod = EnsembleMethod.PROBABILITY_AVG,
        min_confidence: float = 0.45,
    ) -> PredictionEngine:
        """Load models and scaler from a directory of ``.joblib`` files.

        Parameters
        ----------
        directory : str
            Path to saved model artifacts.
        ensemble_method : EnsembleMethod
            Aggregation strategy.
        min_confidence : float
            Threshold below which the engine emits HOLD.
        """
        path = Path(directory)
        models: Dict[str, Any] = {}
        scaler = None
        feature_names: List[str] = []

        scaler_path = path / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        names_path = path / "feature_names.joblib"
        if names_path.exists():
            feature_names = joblib.load(names_path)

        for fp in sorted(path.glob("*.joblib")):
            if fp.name in ("scaler.joblib", "feature_names.joblib"):
                continue
            models[fp.stem] = joblib.load(fp)
            logger.info("Loaded model: %s", fp.stem)

        logger.info("PredictionEngine loaded %d models", len(models))
        return cls(
            models=models,
            scaler=scaler,
            feature_names=feature_names,
            ensemble_method=ensemble_method,
            min_confidence=min_confidence,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> PredictionSignal:
        """Generate an ensemble prediction for a single sample or batch.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (one or more rows).  Uses the last row when
            multiple rows are provided (latest bar).

        Returns
        -------
        PredictionSignal
            Aggregated signal with confidence and per-model detail.
        """
        row = features.iloc[[-1]] if len(features) > 1 else features

        if self._feature_names:
            missing = set(self._feature_names) - set(row.columns)
            if missing:
                logger.warning("Missing features: %s — filling with 0", missing)
                for col in missing:
                    row[col] = 0.0
            row = row[self._feature_names]

        X = row.values.astype(np.float64)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        votes: Dict[str, int] = {}
        probas: Dict[str, List[float]] = {}

        for name, model in self._models.items():
            pred = int(model.predict(X)[0])
            votes[name] = pred

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0].tolist()
            else:
                proba = self._hard_to_proba(pred)
            probas[name] = proba

        return self._aggregate(votes, probas)

    def predict_batch(self, features: pd.DataFrame) -> List[PredictionSignal]:
        """Run predictions row-by-row for a full DataFrame.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with N rows.

        Returns
        -------
        list[PredictionSignal]
            One signal per row.
        """
        signals = []
        for i in range(len(features)):
            signals.append(self.predict(features.iloc[[i]]))
        return signals

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        votes: Dict[str, int],
        probas: Dict[str, List[float]],
    ) -> PredictionSignal:
        """Aggregate individual model outputs into a single signal."""

        if self.ensemble_method == EnsembleMethod.MAJORITY_VOTE:
            direction, confidence, agreement = self._majority_vote(votes)
        elif self.ensemble_method == EnsembleMethod.WEIGHTED_VOTE:
            direction, confidence, agreement = self._weighted_vote(votes)
        elif self.ensemble_method == EnsembleMethod.PROBABILITY_AVG:
            direction, confidence, agreement = self._probability_average(probas)
        elif self.ensemble_method == EnsembleMethod.UNANIMOUS:
            direction, confidence, agreement = self._unanimous(votes)
        else:
            direction, confidence, agreement = self._majority_vote(votes)

        if confidence < self.min_confidence:
            direction = SignalDirection.HOLD

        return PredictionSignal(
            direction=direction,
            confidence=round(confidence, 4),
            model_votes=votes,
            model_probabilities=probas,
            agreement_ratio=round(agreement, 4),
        )

    def _majority_vote(
        self, votes: Dict[str, int]
    ) -> Tuple[SignalDirection, float, float]:
        """Simple majority vote among models."""
        vote_list = list(votes.values())
        counts = {v: vote_list.count(v) for v in set(vote_list)}
        winner = max(counts, key=counts.get)  # type: ignore[arg-type]
        agreement = counts[winner] / len(vote_list)
        return self._int_to_direction(winner), agreement, agreement

    def _weighted_vote(
        self, votes: Dict[str, int]
    ) -> Tuple[SignalDirection, float, float]:
        """Weighted vote using per-model weights."""
        weighted: Dict[int, float] = {}
        total_weight = 0.0

        for name, vote in votes.items():
            w = self.model_weights.get(name, 1.0)
            weighted[vote] = weighted.get(vote, 0.0) + w
            total_weight += w

        winner = max(weighted, key=weighted.get)  # type: ignore[arg-type]
        confidence = weighted[winner] / total_weight
        agreement = sum(1 for v in votes.values() if v == winner) / len(votes)
        return self._int_to_direction(winner), confidence, agreement

    def _probability_average(
        self, probas: Dict[str, List[float]]
    ) -> Tuple[SignalDirection, float, float]:
        """Average predicted probabilities across models."""
        all_proba = np.array(list(probas.values()))
        avg_proba = all_proba.mean(axis=0)

        winner_idx = int(np.argmax(avg_proba))
        confidence = float(avg_proba[winner_idx])

        # Map index back to class: 0 → -1 (sell), 1 → 0 (hold), 2 → 1 (buy)
        class_map = {0: -1, 1: 0, 2: 1}
        winner = class_map.get(winner_idx, 0)

        predictions = [int(np.argmax(p)) for p in all_proba]
        agreement = sum(1 for p in predictions if p == winner_idx) / len(predictions)

        return self._int_to_direction(winner), confidence, agreement

    def _unanimous(
        self, votes: Dict[str, int]
    ) -> Tuple[SignalDirection, float, float]:
        """Require all models to agree; else HOLD."""
        unique = set(votes.values())
        if len(unique) == 1:
            winner = unique.pop()
            return self._int_to_direction(winner), 1.0, 1.0
        return SignalDirection.HOLD, 0.0, 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _int_to_direction(value: int) -> SignalDirection:
        if value == 1:
            return SignalDirection.BUY
        elif value == -1:
            return SignalDirection.SELL
        return SignalDirection.HOLD

    @staticmethod
    def _hard_to_proba(pred: int) -> List[float]:
        """Convert a hard prediction to a pseudo-probability vector."""
        proba = [0.1, 0.1, 0.1]
        idx = {-1: 0, 0: 1, 1: 2}.get(pred, 1)
        proba[idx] = 0.8
        return proba

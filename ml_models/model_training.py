"""
Model Training Pipeline — Time-series aware ML training with hyperparameter tuning.

Trains multiple model families (Random Forest, XGBoost, LightGBM, Ridge) using
walk-forward cross-validation that respects temporal ordering of financial data.

Typical usage:
    >>> trainer = ModelTrainer(n_splits=5)
    >>> results = trainer.train_all(X_train, y_train)
    >>> trainer.save_models("models/")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("XGBoost not installed — skipping XGB models")

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("LightGBM not installed — skipping LGB models")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    logger.info("Optuna not installed — using default hyperparameters")


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Evaluation results for a single model."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_scores: List[float]
    feature_importances: Optional[Dict[str, float]]
    best_params: Dict[str, Any]
    train_time_sec: float
    classification_report: str


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    n_splits: int = 5
    test_size: Optional[int] = None
    optimize_hyperparams: bool = True
    optuna_n_trials: int = 50
    random_state: int = 42
    scale_features: bool = True
    class_labels: List[int] = field(default_factory=lambda: [-1, 0, 1])


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """Train and evaluate multiple ML models with time-series cross-validation.

    Parameters
    ----------
    config : TrainingConfig, optional
        Pipeline configuration.
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()
        self._models: Dict[str, Any] = {}
        self._results: Dict[str, ModelResult] = {}
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []

    @property
    def models(self) -> Dict[str, Any]:
        return dict(self._models)

    @property
    def results(self) -> Dict[str, ModelResult]:
        return dict(self._results)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, ModelResult]:
        """Train all available model families and return evaluation results.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (rows = samples, cols = features).
        y : pd.Series
            Target labels (-1, 0, 1).

        Returns
        -------
        dict[str, ModelResult]
            Results keyed by model name.
        """
        self._feature_names = X.columns.tolist()

        X_arr, y_arr = self._prepare_data(X, y)
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            test_size=self.config.test_size,
        )

        model_factories = self._get_model_factories()

        for name, factory_fn in model_factories.items():
            logger.info("Training %s …", name)
            t0 = time.perf_counter()

            try:
                result = self._train_single(name, factory_fn, X_arr, y_arr, tscv)
                elapsed = time.perf_counter() - t0
                result.train_time_sec = elapsed
                self._results[name] = result
                logger.info(
                    "%s — F1=%.4f  Acc=%.4f  (%.1fs)",
                    name, result.f1, result.accuracy, elapsed,
                )
            except Exception:
                logger.exception("Failed to train %s", name)

        return self._results

    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Generate predictions using a trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        model_name : str, optional
            Which model to use.  Defaults to the best by F1 score.
        """
        if model_name is None:
            model_name = self.best_model_name

        model = self._models[model_name]
        X_arr = self._scale(X.values) if self.config.scale_features else X.values
        return model.predict(X_arr)

    @property
    def best_model_name(self) -> str:
        """Return the model name with the highest F1 score."""
        if not self._results:
            raise RuntimeError("No models trained yet")
        return max(self._results, key=lambda k: self._results[k].f1)

    def get_feature_importances(
        self, model_name: Optional[str] = None, top_n: int = 20
    ) -> Dict[str, float]:
        """Return top-N feature importances for a model.

        Parameters
        ----------
        model_name : str, optional
            Model to inspect (defaults to best).
        top_n : int
            Number of features to return.
        """
        if model_name is None:
            model_name = self.best_model_name
        imp = self._results[model_name].feature_importances
        if imp is None:
            return {}
        sorted_imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
        return dict(list(sorted_imp.items())[:top_n])

    def save_models(self, directory: str) -> None:
        """Persist all trained models and scaler to disk.

        Parameters
        ----------
        directory : str
            Target directory (created if it doesn't exist).
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self._models.items():
            fp = path / f"{name}.joblib"
            joblib.dump(model, fp)
            logger.info("Saved %s → %s", name, fp)

        if self._scaler is not None:
            joblib.dump(self._scaler, path / "scaler.joblib")

        joblib.dump(self._feature_names, path / "feature_names.joblib")
        logger.info("All models saved to %s", path)

    def load_models(self, directory: str) -> None:
        """Load previously saved models from disk.

        Parameters
        ----------
        directory : str
            Directory containing ``.joblib`` model files.
        """
        path = Path(directory)

        scaler_path = path / "scaler.joblib"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)

        names_path = path / "feature_names.joblib"
        if names_path.exists():
            self._feature_names = joblib.load(names_path)

        for fp in sorted(path.glob("*.joblib")):
            if fp.name in ("scaler.joblib", "feature_names.joblib"):
                continue
            name = fp.stem
            self._models[name] = joblib.load(fp)
            logger.info("Loaded model: %s", name)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features and align arrays."""
        X_arr = X.values.astype(np.float64)
        y_arr = y.values.astype(np.int64)

        if self.config.scale_features:
            self._scaler = StandardScaler()
            X_arr = self._scaler.fit_transform(X_arr)

        return X_arr, y_arr

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            return self._scaler.transform(X)
        return X

    def _get_model_factories(self) -> Dict[str, Any]:
        """Return a dict of model-name → factory callable."""
        factories: Dict[str, Any] = {
            "random_forest": self._build_rf,
            "ridge": self._build_ridge,
        }
        if _HAS_XGB:
            factories["xgboost"] = self._build_xgb
        if _HAS_LGB:
            factories["lightgbm"] = self._build_lgb
        return factories

    def _train_single(
        self,
        name: str,
        factory_fn: Any,
        X: np.ndarray,
        y: np.ndarray,
        tscv: TimeSeriesSplit,
    ) -> ModelResult:
        """Train one model with optional hyperparameter tuning."""

        if self.config.optimize_hyperparams and _HAS_OPTUNA:
            best_params = self._optimize_params(name, factory_fn, X, y, tscv)
        else:
            best_params = {}

        model = factory_fn(best_params)

        cv_scores: List[float] = []
        y_pred_full = np.zeros_like(y)
        last_test_idx = np.array([], dtype=int)

        for train_idx, test_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            score = f1_score(y[test_idx], preds, average="weighted", zero_division=0)
            cv_scores.append(score)
            y_pred_full[test_idx] = preds
            last_test_idx = test_idx

        # Final fit on full training data
        model.fit(X, y)
        self._models[name] = model

        # Evaluate on last fold
        y_true_eval = y[last_test_idx]
        y_pred_eval = y_pred_full[last_test_idx]

        importances = self._extract_importances(model)

        return ModelResult(
            model_name=name,
            accuracy=accuracy_score(y_true_eval, y_pred_eval),
            precision=precision_score(
                y_true_eval, y_pred_eval, average="weighted", zero_division=0
            ),
            recall=recall_score(
                y_true_eval, y_pred_eval, average="weighted", zero_division=0
            ),
            f1=f1_score(
                y_true_eval, y_pred_eval, average="weighted", zero_division=0
            ),
            cv_scores=cv_scores,
            feature_importances=importances,
            best_params=best_params,
            train_time_sec=0.0,
            classification_report=classification_report(
                y_true_eval, y_pred_eval, zero_division=0
            ),
        )

    def _optimize_params(
        self,
        name: str,
        factory_fn: Any,
        X: np.ndarray,
        y: np.ndarray,
        tscv: TimeSeriesSplit,
    ) -> Dict[str, Any]:
        """Run Optuna hyperparameter search."""

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, name)
            model = factory_fn(params)
            scores = []
            for train_idx, test_idx in tscv.split(X):
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[test_idx])
                scores.append(
                    f1_score(y[test_idx], preds, average="weighted", zero_division=0)
                )
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.optuna_n_trials, show_progress_bar=False)

        logger.info("%s best Optuna F1=%.4f", name, study.best_value)
        return study.best_params

    @staticmethod
    def _suggest_params(trial: optuna.Trial, name: str) -> Dict[str, Any]:
        """Define search spaces for each model family."""
        if name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
            }
        elif name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        elif name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        elif name == "ridge":
            return {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
            }
        return {}

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _build_rf(self, params: Dict[str, Any]) -> RandomForestClassifier:
        defaults = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": self.config.random_state,
            "class_weight": "balanced",
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    def _build_xgb(self, params: Dict[str, Any]) -> xgb.XGBClassifier:
        defaults = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": self.config.random_state,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
        defaults.update(params)
        return xgb.XGBClassifier(**defaults)

    def _build_lgb(self, params: Dict[str, Any]) -> lgb.LGBMClassifier:
        defaults = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": self.config.random_state,
            "verbose": -1,
            "class_weight": "balanced",
        }
        defaults.update(params)
        return lgb.LGBMClassifier(**defaults)

    def _build_ridge(self, params: Dict[str, Any]) -> RidgeClassifier:
        defaults = {
            "alpha": 1.0,
            "class_weight": "balanced",
        }
        defaults.update(params)
        return RidgeClassifier(**defaults)

    def _extract_importances(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importances from a fitted model."""
        imp_arr: Optional[np.ndarray] = None

        if hasattr(model, "feature_importances_"):
            imp_arr = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp_arr = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)

        if imp_arr is None or len(self._feature_names) != len(imp_arr):
            return None

        return {name: float(val) for name, val in zip(self._feature_names, imp_arr)}

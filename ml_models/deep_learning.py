"""
Deep Learning Models — LSTM/GRU with Attention for time-series prediction.

Implements sequence-to-one architectures for financial price forecasting.
The attention mechanism lets the model learn which past time steps are
most informative for the current prediction.

Typical usage:
    >>> dl = DeepLearningModel(sequence_length=60, n_features=50)
    >>> dl.build_lstm_attention()
    >>> history = dl.train(X_seq, y_seq, epochs=100)
    >>> predictions = dl.predict(X_new)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers

    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    logger.warning("TensorFlow not installed — deep learning models unavailable")


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DLConfig:
    """Deep learning model configuration."""

    sequence_length: int = 60
    n_features: int = 50
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])
    gru_units: List[int] = field(default_factory=lambda: [128, 64])
    attention_heads: int = 4
    dropout_rate: float = 0.3
    recurrent_dropout: float = 0.2
    dense_units: List[int] = field(default_factory=lambda: [64, 32])
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    n_classes: int = 3  # buy / hold / sell
    use_attention: bool = True


# ──────────────────────────────────────────────────────────────────────
# Attention Layer
# ──────────────────────────────────────────────────────────────────────

if _HAS_TF:

    class BahdanauAttention(layers.Layer):
        """Additive (Bahdanau) attention for sequence models.

        Given a sequence of hidden states, learns an alignment score for
        each time step and returns a weighted context vector.

        Parameters
        ----------
        units : int
            Dimensionality of the attention hidden layer.
        """

        def __init__(self, units: int = 64, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.W = layers.Dense(units, use_bias=False)
            self.V = layers.Dense(1, use_bias=False)

        def call(self, hidden_states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """Compute attention-weighted context.

            Parameters
            ----------
            hidden_states : tf.Tensor
                Shape ``(batch, timesteps, features)``.

            Returns
            -------
            context : tf.Tensor
                Shape ``(batch, features)``.
            weights : tf.Tensor
                Shape ``(batch, timesteps, 1)``.
            """
            score = self.V(tf.nn.tanh(self.W(hidden_states)))
            weights = tf.nn.softmax(score, axis=1)
            context = tf.reduce_sum(weights * hidden_states, axis=1)
            return context, weights

        def get_config(self) -> Dict[str, Any]:
            config = super().get_config()
            config["units"] = self.W.units
            return config


# ──────────────────────────────────────────────────────────────────────
# Main Model Class
# ──────────────────────────────────────────────────────────────────────

class DeepLearningModel:
    """LSTM / GRU time-series classifier with optional attention.

    Parameters
    ----------
    config : DLConfig, optional
        Model architecture and training configuration.
    """

    def __init__(self, config: Optional[DLConfig] = None) -> None:
        if not _HAS_TF:
            raise ImportError(
                "TensorFlow is required for deep learning models. "
                "Install with: pip install tensorflow"
            )
        self.config = config or DLConfig()
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Architecture builders
    # ------------------------------------------------------------------

    def build_lstm_attention(self) -> keras.Model:
        """Build a stacked LSTM with Bahdanau attention.

        Architecture::

            Input → LSTM(128, return_seq) → Dropout
                  → LSTM(64, return_seq)  → Dropout
                  → Attention             → Dense(64) → Dense(32)
                  → Softmax(3)

        Returns
        -------
        keras.Model
            Compiled model.
        """
        cfg = self.config
        inp = layers.Input(shape=(cfg.sequence_length, cfg.n_features), name="input")

        x = inp
        for i, units in enumerate(cfg.lstm_units):
            x = layers.LSTM(
                units,
                return_sequences=True,
                dropout=cfg.dropout_rate,
                recurrent_dropout=cfg.recurrent_dropout,
                name=f"lstm_{i}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_lstm_{i}")(x)

        if cfg.use_attention:
            context, attn_weights = BahdanauAttention(
                units=cfg.lstm_units[-1], name="attention"
            )(x)
            x = context
        else:
            x = layers.GlobalAveragePooling1D(name="global_pool")(x)

        for i, units in enumerate(cfg.dense_units):
            x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
            x = layers.Dropout(cfg.dropout_rate, name=f"drop_dense_{i}")(x)

        output = layers.Dense(cfg.n_classes, activation="softmax", name="output")(x)

        self.model = keras.Model(inputs=inp, outputs=output, name="lstm_attention")
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info("LSTM-Attention model built: %s params", f"{self.model.count_params():,}")
        return self.model

    def build_gru(self) -> keras.Model:
        """Build a stacked GRU model (lighter alternative to LSTM).

        Returns
        -------
        keras.Model
            Compiled model.
        """
        cfg = self.config
        inp = layers.Input(shape=(cfg.sequence_length, cfg.n_features), name="input")

        x = inp
        for i, units in enumerate(cfg.gru_units):
            return_seq = i < len(cfg.gru_units) - 1 or cfg.use_attention
            x = layers.GRU(
                units,
                return_sequences=return_seq,
                dropout=cfg.dropout_rate,
                recurrent_dropout=cfg.recurrent_dropout,
                name=f"gru_{i}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_gru_{i}")(x)

        if cfg.use_attention and x.shape.rank == 3:
            context, _ = BahdanauAttention(
                units=cfg.gru_units[-1], name="attention"
            )(x)
            x = context

        for i, units in enumerate(cfg.dense_units):
            x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
            x = layers.Dropout(cfg.dropout_rate, name=f"drop_dense_{i}")(x)

        output = layers.Dense(cfg.n_classes, activation="softmax", name="output")(x)

        self.model = keras.Model(inputs=inp, outputs=output, name="gru_model")
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info("GRU model built: %s params", f"{self.model.count_params():,}")
        return self.model

    # ------------------------------------------------------------------
    # Sequence preparation
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert flat feature matrix into overlapping sequences.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Target array of shape ``(n_samples,)``.

        Returns
        -------
        X_seq : np.ndarray
            Shape ``(n_sequences, sequence_length, n_features)``.
        y_seq : np.ndarray
            Shape ``(n_sequences,)`` — label for the *last* bar in each window.
        """
        seq_len = self.config.sequence_length
        n_samples = len(X)

        if n_samples <= seq_len:
            raise ValueError(
                f"Need at least {seq_len + 1} samples, got {n_samples}"
            )

        X_seq = np.array([X[i : i + seq_len] for i in range(n_samples - seq_len)])
        y_seq = y[seq_len:]

        logger.info(
            "Created %d sequences of length %d × %d features",
            len(X_seq), seq_len, X.shape[1],
        )
        return X_seq, y_seq

    def normalize_features(
        self, X: np.ndarray, fit: bool = True
    ) -> np.ndarray:
        """Z-score normalize features (per-feature across samples).

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n, features)`` or ``(n, seq_len, features)``.
        fit : bool
            If True, compute and store mean/std.  If False, use stored.

        Returns
        -------
        np.ndarray
            Normalized array.
        """
        orig_shape = X.shape
        if X.ndim == 3:
            n, s, f = X.shape
            X = X.reshape(-1, f)

        if fit:
            self._scaler_mean = X.mean(axis=0)
            self._scaler_std = X.std(axis=0) + 1e-8
        elif self._scaler_mean is None:
            raise RuntimeError("Scaler not fitted — call with fit=True first")

        X_norm = (X - self._scaler_mean) / self._scaler_std

        if len(orig_shape) == 3:
            X_norm = X_norm.reshape(orig_shape)

        return X_norm

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        validation_split: float = 0.15,
    ) -> Dict[str, List[float]]:
        """Train the model with early stopping and learning rate scheduling.

        Parameters
        ----------
        X_seq : np.ndarray
            Sequence input, shape ``(n, seq_len, features)``.
        y_seq : np.ndarray
            Labels, shape ``(n,)``.
        validation_split : float
            Fraction of data reserved for validation (taken from the end
            to preserve temporal order).

        Returns
        -------
        dict
            Training history (loss, accuracy per epoch).
        """
        if self.model is None:
            raise RuntimeError("Model not built — call build_lstm_attention() first")

        # Remap labels: {-1, 0, 1} → {0, 1, 2}
        y_mapped = y_seq.copy()
        y_mapped[y_seq == -1] = 0
        y_mapped[y_seq == 0] = 1
        y_mapped[y_seq == 1] = 2

        cfg = self.config

        # Temporal split (no shuffle)
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_mapped[:split_idx], y_mapped[split_idx:]

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=cfg.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        logger.info(
            "Training started — %d train / %d val samples",
            len(X_train), len(X_val),
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=cb,
            verbose=1,
            shuffle=False,  # preserve temporal order within batches
        )

        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info("Training complete — val_loss=%.4f, val_acc=%.4f", val_loss, val_acc)

        return {k: [float(v) for v in vals] for k, vals in self.history.history.items()}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Generate class predictions for sequences.

        Parameters
        ----------
        X_seq : np.ndarray
            Shape ``(n, seq_len, features)``.

        Returns
        -------
        np.ndarray
            Predicted labels mapped back to {-1, 0, 1}.
        """
        if self.model is None:
            raise RuntimeError("No model available")

        proba = self.model.predict(X_seq, verbose=0)
        preds = np.argmax(proba, axis=1)

        # Reverse map: {0, 1, 2} → {-1, 0, 1}
        label_map = np.array([-1, 0, 1])
        return label_map[preds]

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Returns
        -------
        np.ndarray
            Shape ``(n, 3)`` — columns are [sell, hold, buy].
        """
        if self.model is None:
            raise RuntimeError("No model available")
        return self.model.predict(X_seq, verbose=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save model weights, architecture, and scaler state.

        Parameters
        ----------
        directory : str
            Target directory.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(path / "dl_model.keras")

        if self._scaler_mean is not None:
            np.savez(
                path / "dl_scaler.npz",
                mean=self._scaler_mean,
                std=self._scaler_std,
            )
        logger.info("Deep learning model saved to %s", path)

    def load(self, directory: str) -> None:
        """Load a previously saved model.

        Parameters
        ----------
        directory : str
            Directory containing saved artifacts.
        """
        path = Path(directory)

        model_path = path / "dl_model.keras"
        if model_path.exists():
            self.model = keras.models.load_model(
                model_path,
                custom_objects={"BahdanauAttention": BahdanauAttention},
            )
            logger.info("Loaded DL model from %s", model_path)

        scaler_path = path / "dl_scaler.npz"
        if scaler_path.exists():
            data = np.load(scaler_path)
            self._scaler_mean = data["mean"]
            self._scaler_std = data["std"]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a string summary of the model architecture."""
        if self.model is None:
            return "No model built"
        lines: List[str] = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)

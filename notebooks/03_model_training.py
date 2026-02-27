# %% [markdown]
# # 03 — Model Training & Evaluation
#
# Trains multiple ML models (Random Forest, XGBoost, LightGBM, Ridge)
# using time-series cross-validation, evaluates performance, and
# generates comparison visualizations.

# %% Imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.feature_engineering import FeatureEngineer
from ml_models.model_training import ModelTrainer, TrainingConfig
from ml_models.prediction_engine import PredictionEngine, EnsembleMethod
from ai_analysis.performance_analyzer import PerformanceAnalyzer
from ai_analysis.visualization import MLVisualizer

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# %% Load feature dataset
feature_path = PROJECT_ROOT / "data" / "processed" / "ml_features.parquet"

if feature_path.exists():
    dataset = pd.read_parquet(feature_path)
    print(f"Loaded features: {dataset.shape}")
else:
    print("Feature file not found — run 02_feature_engineering.py first.")
    print("Generating synthetic dataset for demonstration...")
    np.random.seed(42)
    n = 3000
    n_features = 50
    X_syn = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_syn = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    dataset = X_syn.copy()
    dataset["target"] = y_syn

X = dataset.drop(columns=["target"])
y = dataset["target"]
print(f"Features: {X.shape[1]}, Samples: {len(X)}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# %% Temporal train/test split (no shuffle — critical for financial data)
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"\nTrain: {len(X_train):,} samples | Test: {len(X_test):,} samples")

# %% Train all models
config = TrainingConfig(
    n_splits=5,
    optimize_hyperparams=False,  # Set True for full Optuna search
    random_state=42,
    scale_features=True,
)

trainer = ModelTrainer(config=config)
results = trainer.train_all(X_train, y_train)

# %% Print results
print("\n" + "=" * 70)
print("  MODEL EVALUATION RESULTS")
print("=" * 70)
for name, res in sorted(results.items(), key=lambda kv: kv[1].f1, reverse=True):
    print(f"\n--- {name.upper()} ---")
    print(f"  Accuracy:  {res.accuracy:.4f}")
    print(f"  Precision: {res.precision:.4f}")
    print(f"  Recall:    {res.recall:.4f}")
    print(f"  F1 Score:  {res.f1:.4f}")
    print(f"  CV Scores: {[f'{s:.4f}' for s in res.cv_scores]}")
    print(f"  Time:      {res.train_time_sec:.1f}s")
    if res.best_params:
        print(f"  Params:    {res.best_params}")

# %% Feature importance
print(f"\nBest model: {trainer.best_model_name}")
top_features = trainer.get_feature_importances(top_n=15)
if top_features:
    print("\nTop 15 Features:")
    for feat, imp in top_features.items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<30s} {imp:.4f}  {bar}")

# %% Visualizations
viz = MLVisualizer(output_dir=str(PROJECT_ROOT / "reports" / "plots"))

# Model comparison chart
comparison_data = {
    name: {
        "accuracy": res.accuracy,
        "precision": res.precision,
        "recall": res.recall,
        "f1": res.f1,
    }
    for name, res in results.items()
}
viz.plot_model_comparison(comparison_data)
print("Saved: model_comparison.png")

# Feature importance chart
if top_features:
    viz.plot_feature_importances(top_features, top_n=15)
    print("Saved: feature_importances.png")

# %% Evaluate on test set
print("\n" + "=" * 70)
print("  OUT-OF-SAMPLE TEST EVALUATION")
print("=" * 70)

from sklearn.metrics import classification_report

for name in results:
    y_pred = trainer.predict(X_test, model_name=name)
    print(f"\n--- {name.upper()} ---")
    print(classification_report(y_test.values, y_pred, zero_division=0))

# Confusion matrix for best model
y_pred_best = trainer.predict(X_test)
viz.plot_confusion_matrix(y_test.values, y_pred_best)
print("Saved: confusion_matrix.png")

# Predictions vs actual
viz.plot_predictions_vs_actual(y_test.values, y_pred_best)
print("Saved: predictions_vs_actual.png")

# %% Ensemble prediction engine
engine = PredictionEngine(
    models=trainer.models,
    scaler=trainer._scaler,
    feature_names=trainer._feature_names,
    ensemble_method=EnsembleMethod.PROBABILITY_AVG,
    min_confidence=0.40,
)

sample_signal = engine.predict(X_test.iloc[:5])
print(f"\nEnsemble Signal: {sample_signal.direction.value}")
print(f"Confidence:      {sample_signal.confidence:.4f}")
print(f"Agreement:       {sample_signal.agreement_ratio:.4f}")
print(f"Model Votes:     {sample_signal.model_votes}")

# %% Performance analysis of ML strategy
analyzer = PerformanceAnalyzer(trading_periods_per_year=6300)

# Simulate equity from predictions
all_signals = engine.predict_batch(X_test)
initial_equity = 10000.0
equity = [initial_equity]
for i, sig in enumerate(all_signals):
    if i + 1 >= len(y_test):
        break
    actual_return = y_test.iloc[i + 1] * 0.001  # scaled
    if sig.direction.value == "BUY":
        pnl = actual_return
    elif sig.direction.value == "SELL":
        pnl = -actual_return
    else:
        pnl = 0
    equity.append(equity[-1] * (1 + pnl))

equity_arr = np.array(equity)
metrics = analyzer.compute_metrics(equity_arr)
print("\n" + analyzer.generate_report(metrics))

# Equity curve
viz.plot_equity_curve(equity_arr)
print("Saved: equity_curve.png")

# %% Save models
model_dir = str(PROJECT_ROOT / "models" / "trained")
trainer.save_models(model_dir)
print(f"\nModels saved to: {model_dir}")

# %% Summary
print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"  Models trained:  {len(results)}")
print(f"  Best model:      {trainer.best_model_name}")
print(f"  Best F1:         {results[trainer.best_model_name].f1:.4f}")
print(f"  Features used:   {len(trainer._feature_names)}")
print(f"  Models saved:    {model_dir}")

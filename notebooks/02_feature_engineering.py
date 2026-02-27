# %% [markdown]
# # 02 — Feature Engineering
#
# Demonstrates the full feature engineering pipeline: computing 50+
# technical features, building lag/momentum features, and preparing
# the ML-ready dataset for model training.

# %% Imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import TRAIN_DIR
from indicators.compute import compute_all
from ml_models.feature_engineering import FeatureEngineer, FeatureConfig

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

# %% Load training data
train_files = sorted(Path(TRAIN_DIR).glob("*.csv"))
if train_files:
    df = pd.read_csv(train_files[0])
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    print(f"Loaded {len(df):,} bars from {train_files[0].name}")
else:
    print("No training data found — using synthetic data")
    np.random.seed(42)
    n = 5000
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    price = 25000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 30),
        "low": price - np.abs(np.random.randn(n) * 30),
        "close": price + np.random.randn(n) * 20,
        "tick_volume": np.random.randint(100, 10000, n),
    }, index=idx)

# %% Compute technical indicators
print("\nComputing technical indicators...")
indicators = compute_all(df, timeframe="H1")
print(f"Indicators computed: {len(indicators)} keys")
print(f"Sample keys: {list(indicators.keys())[:15]}")

# %% Build features
config = FeatureConfig(
    lookback=20,
    lag_periods=[1, 2, 3, 5, 10],
    rolling_windows=[5, 10, 20, 50],
    momentum_periods=[1, 3, 5, 10, 20],
    include_time_features=True,
    include_volume_features=True,
    include_volatility_regime=True,
)

fe = FeatureEngineer(config=config)
features, feature_names = fe.build_features(df, indicators)

print(f"\nFeature matrix shape: {features.shape}")
print(f"Total features: {len(feature_names)}")
print(f"\nFeature names:\n{feature_names}")

# %% Build target
target = fe.build_target(df, horizon=5, threshold_pct=0.3)
target_aligned = target.loc[features.index]

print(f"\nTarget distribution:")
print(target_aligned.value_counts().sort_index())
print(f"\nClass balance:")
for label, name in [(-1, "Sell"), (0, "Hold"), (1, "Buy")]:
    pct = (target_aligned == label).mean() * 100
    print(f"  {name:>4s}: {pct:.1f}%")

# %% Feature correlation heatmap
top_features = features.columns[:20]
corr = features[top_features].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=False,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix (Top 20)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "feature_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Feature distributions by target class
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sample_features = ["ind_rsi_14", "ind_macd_histogram", "bb_position",
                    "momentum_5", "ret_std_20", "atr_pct"]

for ax, feat in zip(axes.flat, sample_features):
    if feat in features.columns:
        for label, color, name in [(-1, "#F44336", "Sell"), (0, "#9E9E9E", "Hold"), (1, "#4CAF50", "Buy")]:
            mask = target_aligned == label
            vals = features.loc[mask.index[mask], feat].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=50, alpha=0.5, color=color, label=name, density=True)
        ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)

plt.suptitle("Feature Distributions by Signal Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Feature statistics
print("\n=== Feature Statistics ===")
print(features.describe().round(4).to_string())

# %% Check for multicollinearity
high_corr_pairs = []
corr_full = features.corr().abs()
for i in range(len(corr_full.columns)):
    for j in range(i + 1, len(corr_full.columns)):
        if corr_full.iloc[i, j] > 0.95:
            high_corr_pairs.append((
                corr_full.columns[i],
                corr_full.columns[j],
                corr_full.iloc[i, j],
            ))

print(f"\nHighly correlated pairs (|r| > 0.95): {len(high_corr_pairs)}")
for f1, f2, r in high_corr_pairs[:10]:
    print(f"  {f1} <-> {f2}: {r:.3f}")

# %% Save processed features
output_path = PROJECT_ROOT / "data" / "processed" / "ml_features.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)

dataset = features.copy()
dataset["target"] = target_aligned
dataset.to_parquet(output_path)
print(f"\nSaved feature dataset: {output_path} ({len(dataset):,} rows)")

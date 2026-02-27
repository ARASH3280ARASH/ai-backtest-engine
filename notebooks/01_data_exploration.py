# %% [markdown]
# # 01 — Data Exploration & EDA
#
# Exploratory analysis of BTCUSD historical data used by the backtesting engine.
# This notebook examines price distributions, volatility regimes, and temporal
# patterns that inform feature engineering decisions.

# %% Imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import TRAIN_DIR, VALIDATION_DIR, TEST_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 120

# %% Load data
print("Loading training data...")
train_files = sorted(Path(TRAIN_DIR).glob("*.csv"))
if train_files:
    df = pd.read_csv(train_files[0], parse_dates=["time"] if "time" in
                      pd.read_csv(train_files[0], nrows=1).columns else None)
    print(f"Loaded: {train_files[0].name} — {len(df):,} rows")
else:
    print("No CSV files found in training directory. Generating sample data.")
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    price = 25000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({
        "time": dates,
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 30),
        "low": price - np.abs(np.random.randn(n) * 30),
        "close": price + np.random.randn(n) * 20,
        "tick_volume": np.random.randint(100, 10000, n),
    })

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()

# %% Basic statistics
print("\n=== Descriptive Statistics ===")
df[["open", "high", "low", "close"]].describe().round(2)

# %% Price chart
fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

ax = axes[0]
ax.plot(df["close"].values, linewidth=0.8, color="#1565C0")
ax.set_title("BTCUSD Close Price — Training Data", fontsize=14, fontweight="bold")
ax.set_ylabel("Price ($)")

ax2 = axes[1]
vol = df.get("tick_volume", df.get("volume", pd.Series(0, index=df.index)))
ax2.bar(range(len(vol)), vol, width=1.0, color="#90CAF9", alpha=0.7)
ax2.set_ylabel("Volume")
ax2.set_xlabel("Bar Index")

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "price_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Returns distribution
returns = df["close"].pct_change().dropna()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
axes[0].hist(returns * 100, bins=100, color="#42A5F5", edgecolor="white", alpha=0.8)
axes[0].axvline(returns.mean() * 100, color="red", linestyle="--", label=f"Mean: {returns.mean()*100:.4f}%")
axes[0].set_title("Return Distribution", fontweight="bold")
axes[0].set_xlabel("Return (%)")
axes[0].legend()

# QQ plot
from scipy import stats
stats.probplot(returns, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot (Normal)", fontweight="bold")

# Rolling volatility
rolling_vol = returns.rolling(24).std() * np.sqrt(252 * 24) * 100
axes[2].plot(rolling_vol, color="#FF7043", linewidth=0.8)
axes[2].set_title("Rolling 24h Annualized Volatility", fontweight="bold")
axes[2].set_ylabel("Volatility (%)")

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "return_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Key statistics
print("\n=== Return Statistics ===")
print(f"Mean Return:     {returns.mean()*100:.4f}%")
print(f"Std Dev:         {returns.std()*100:.4f}%")
print(f"Skewness:        {returns.skew():.4f}")
print(f"Kurtosis:        {returns.kurtosis():.4f}")
print(f"Min Return:      {returns.min()*100:.4f}%")
print(f"Max Return:      {returns.max()*100:.4f}%")
print(f"VaR (5%):        {np.percentile(returns, 5)*100:.4f}%")

# %% Bar range analysis
df["bar_range"] = df["high"] - df["low"]
df["body"] = abs(df["close"] - df["open"])
df["body_ratio"] = df["body"] / df["bar_range"].replace(0, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["bar_range"].dropna(), bins=80, color="#66BB6A", alpha=0.8)
axes[0].set_title("Bar Range Distribution", fontweight="bold")
axes[0].set_xlabel("Range ($)")

axes[1].hist(df["body_ratio"].dropna(), bins=80, color="#AB47BC", alpha=0.8)
axes[1].set_title("Body / Range Ratio", fontweight="bold")
axes[1].set_xlabel("Ratio")

plt.tight_layout()
plt.show()

# %% Autocorrelation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pd.plotting.autocorrelation_plot(returns.iloc[:2000], ax=axes[0])
axes[0].set_title("Return Autocorrelation", fontweight="bold")
axes[0].set_xlim(0, 100)

abs_returns = returns.abs()
pd.plotting.autocorrelation_plot(abs_returns.iloc[:2000], ax=axes[1])
axes[1].set_title("|Return| Autocorrelation (Volatility Clustering)", fontweight="bold")
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.show()

# %% Summary
print("\n=== EDA Summary ===")
print(f"Total bars:           {len(df):,}")
print(f"Price range:          ${df['close'].min():,.0f} — ${df['close'].max():,.0f}")
print(f"Avg bar range:        ${df['bar_range'].mean():.2f}")
print(f"Return distribution:  {'Heavy tails' if returns.kurtosis() > 3 else 'Near-normal'}")
print(f"Volatility clusters:  {'Yes' if abs_returns.autocorr(lag=1) > 0.1 else 'Weak'}")

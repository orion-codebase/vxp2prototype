"""Train a RandomForestRegressor on NASA CMAPSS FD001 and save to models/rf_v1.pkl."""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# --- Load training data ---
cols = ["unit_id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
df = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None, names=cols)

# --- Create RUL target ---
max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
df = df.merge(max_cycles, on="unit_id")
df["RUL"] = df["max_cycle"] - df["cycle"]

# --- Train ---
features = [f"s{i}" for i in range(1, 22)]
X = df[features]
y = df["RUL"]
print(f"Training set: {X.shape[0]} rows, {X.shape[1]} features")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
print(f"Training R2: {rf.score(X, y):.4f}")

# --- Save ---
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_v1.pkl")
size_mb = os.path.getsize("models/rf_v1.pkl") / 1e6
print(f"Model saved to models/rf_v1.pkl ({size_mb:.1f} MB)")

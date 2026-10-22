import os
import joblib
import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DBN_DIR = os.path.join(BASE, "data", "processed", "dbn")
MODEL_DIR = os.path.join(BASE, "data", "processed", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# LOAD TRAIN STATES
# ==========================================================

train = pd.read_csv(
    os.path.join(DBN_DIR, "train_dbn_states.csv")
)

features = [c for c in train.columns if c.startswith("z")]

print("="*60)
print("GAUSSIAN OBSERVATION MODEL")
print("="*60)

means = {}
covs = {}

# ==========================================================
# ESTIMATE μ AND Σ FOR EACH STATE
# ==========================================================

for state in sorted(train["State"].unique()):

    data = train[train["State"] == state][features].values

    mu = data.mean(axis=0)

    cov = np.cov(data.T)

    # numerical stability
    cov += np.eye(cov.shape[0]) * 1e-3

    means[int(state)] = mu
    covs[int(state)] = cov

    print(
        f"State {state} | Samples = {len(data):,}"
    )

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    means,
    os.path.join(MODEL_DIR, "gaussian_means.pkl")
)

joblib.dump(
    covs,
    os.path.join(MODEL_DIR, "gaussian_covs.pkl")
)

print("\nSaved:")
print(os.path.join(MODEL_DIR, "gaussian_means.pkl"))
print(os.path.join(MODEL_DIR, "gaussian_covs.pkl"))
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DBN_DIR = os.path.join(BASE, "data", "processed", "dbn")
MODEL_DIR = os.path.join(BASE, "data", "processed", "models")
RESULT_DIR = os.path.join(BASE, "data", "processed", "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================

test = pd.read_csv(os.path.join(DBN_DIR, "test_dbn_states.csv"))

transition = pd.read_csv(
    os.path.join(DBN_DIR, "transition_matrix.csv"),
    index_col=0
).values

emission = pd.read_csv(
    os.path.join(DBN_DIR, "emission_probability.csv"),
    index_col=0
).values.flatten()

means = joblib.load(os.path.join(MODEL_DIR, "gaussian_means.pkl"))
covs = joblib.load(os.path.join(MODEL_DIR, "gaussian_covs.pkl"))

feature_cols = [c for c in test.columns if c.startswith("z")]

print("="*60)
print("GAUSSIAN PARTICLE FILTER")
print("="*60)

# ==========================================================
# BUILD GAUSSIAN DISTRIBUTIONS
# ==========================================================

gaussians = {}

for s in range(4):

    gaussians[s] = multivariate_normal(
        mean=means[s],
        cov=covs[s],
        allow_singular=True
    )

# ==========================================================
# PARTICLE FILTER
# ==========================================================

predictions = []

for pid, patient in test.groupby("PatientID"):

    patient = patient.sort_values("WindowID")

    belief = np.ones(4) / 4

    for _, row in patient.iterrows():

        obs = row[feature_cols].values.astype(float)

        # Prediction
        belief = belief @ transition

        # Observation likelihood
        likelihood = np.zeros(4)

        for s in range(4):
            likelihood[s] = gaussians[s].pdf(obs)

        likelihood += 1e-12

        # Bayesian Update
        belief = belief * likelihood
        belief = belief / belief.sum()

        # Final probability
        prob = np.sum(belief * emission)

        predictions.append([
            pid,
            row["WindowID"],
            row["SepsisLabel"],
            prob
        ])

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

pred = pd.DataFrame(
    predictions,
    columns=[
        "PatientID",
        "WindowID",
        "Actual",
        "Probability"
    ]
)

pred["Prediction"] = (pred["Probability"] >= 0.5).astype(int)

pred.to_csv(
    os.path.join(RESULT_DIR, "gaussian_particle_predictions.csv"),
    index=False
)

# ==========================================================
# METRICS
# ==========================================================

y = pred["Actual"]
p = pred["Probability"]
yp = pred["Prediction"]

tn, fp, fn, tp = confusion_matrix(y, yp).ravel()

metrics = {
    "ROC_AUC": roc_auc_score(y, p),
    "PR_AUC": average_precision_score(y, p),
    "F1": f1_score(y, yp),
    "Accuracy": accuracy_score(y, yp),
    "Sensitivity": recall_score(y, yp),
    "Specificity": tn / (tn + fp),
    "Precision": precision_score(y, yp)
}

print("\nRESULTS")
print("-"*60)

for k, v in metrics.items():
    print(f"{k:15}: {v:.4f}")

pd.DataFrame([metrics]).to_csv(
    os.path.join(RESULT_DIR, "gaussian_particle_metrics.csv"),
    index=False
)

print("\nSaved:")
print(os.path.join(RESULT_DIR, "gaussian_particle_metrics.csv"))
print(os.path.join(RESULT_DIR, "gaussian_particle_predictions.csv"))
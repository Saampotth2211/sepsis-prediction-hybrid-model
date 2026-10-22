import os
import numpy as np
import pandas as pd
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

DBN = os.path.join(BASE, "data", "processed", "dbn")
RESULT = os.path.join(BASE, "data", "processed", "results")

os.makedirs(RESULT, exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================

test = pd.read_csv(os.path.join(DBN, "test_dbn_states.csv"))

transition = pd.read_csv(
    os.path.join(DBN, "transition_matrix.csv"),
    index_col=0
).values

emission = pd.read_csv(
    os.path.join(DBN, "emission_probability.csv"),
    index_col=0
).values.flatten()

# ==========================================================
# PARTICLE FILTER
# ==========================================================

NUM_STATES = 4

predictions = []

for pid, patient in test.groupby("PatientID"):

    patient = patient.sort_values("WindowID")

    belief = np.ones(NUM_STATES) / NUM_STATES

    for _, row in patient.iterrows():

        current_state = int(row["State"])

        # Prediction
        belief = belief @ transition

        # Observation likelihood
        likelihood = np.ones(NUM_STATES) * 0.05
        likelihood[current_state] = 0.95

        # Update
        belief = belief * likelihood

        belief = belief / (belief.sum() + 1e-10)

        # Final sepsis probability
        prob = np.sum(belief * emission)

        predictions.append([
            pid,
            row["WindowID"],
            row["SepsisLabel"],
            prob
        ])

# ==========================================================
# RESULTS
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
    os.path.join(RESULT, "particle_filter_predictions.csv"),
    index=False
)

# ==========================================================
# METRICS
# ==========================================================

y = pred["Actual"]
p = pred["Probability"]
yp = pred["Prediction"]

roc = roc_auc_score(y, p)
pr = average_precision_score(y, p)
f1 = f1_score(y, yp)
acc = accuracy_score(y, yp)
prec = precision_score(y, yp)
rec = recall_score(y, yp)

tn, fp, fn, tp = confusion_matrix(y, yp).ravel()

spec = tn / (tn + fp)

metrics = pd.DataFrame({
    "Model": ["Sliding BiLSTM + DBN + PF"],
    "ROC_AUC": [roc],
    "PR_AUC": [pr],
    "F1": [f1],
    "Accuracy": [acc],
    "Sensitivity": [rec],
    "Specificity": [spec],
    "Precision": [prec]
})

metrics.to_csv(
    os.path.join(RESULT, "particle_filter_metrics.csv"),
    index=False
)

# ==========================================================
# PRINT
# ==========================================================

print("=" * 60)
print("PARTICLE FILTER RESULTS")
print("=" * 60)

for c in metrics.columns[1:]:
    print(f"{c:15}: {metrics[c][0]:.4f}")

print("\nSaved:")
print(os.path.join(RESULT, "particle_filter_metrics.csv"))
print(os.path.join(RESULT, "particle_filter_predictions.csv"))
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DBN_DIR = os.path.join(BASE, "data", "processed", "dbn")
MODEL_DIR = os.path.join(BASE, "data", "processed", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================

train = pd.read_csv(os.path.join(DBN_DIR, "train_sliding_temporal_embeddings.csv"))
valid = pd.read_csv(os.path.join(DBN_DIR, "validation_sliding_temporal_embeddings.csv"))
test = pd.read_csv(os.path.join(DBN_DIR, "test_sliding_temporal_embeddings.csv"))

features = [c for c in train.columns if c.startswith("z")]

# ==========================================================
# STANDARDIZE
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(train[features])
X_valid = scaler.transform(valid[features])
X_test = scaler.transform(test[features])

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "dbn_scaler.pkl")
)

# ==========================================================
# HIDDEN STATES (KMEANS)
# ==========================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

train_state = kmeans.fit_predict(X_train)
valid_state = kmeans.predict(X_valid)
test_state = kmeans.predict(X_test)

joblib.dump(
    kmeans,
    os.path.join(MODEL_DIR, "dbn_kmeans.pkl")
)

# ==========================================================
# ORDER STATES BY SEPSIS RISK
# ==========================================================

train["State"] = train_state

risk = train.groupby("State")["SepsisLabel"].mean().sort_values()

mapping = {old: new for new, old in enumerate(risk.index)}

train["State"] = train["State"].map(mapping)
valid["State"] = pd.Series(valid_state).map(mapping)
test["State"] = pd.Series(test_state).map(mapping)

print("="*60)
print("Hidden State Risk")
print("="*60)
print(train.groupby("State")["SepsisLabel"].mean())

# ==========================================================
# TRANSITION MATRIX
# ==========================================================

T = np.zeros((4,4))

for pid, patient in train.groupby("PatientID"):

    patient = patient.sort_values("WindowID")

    s = patient["State"].values

    for i in range(len(s)-1):
        T[s[i], s[i+1]] += 1

# normalize

T = T / (T.sum(axis=1, keepdims=True)+1e-8)

transition = pd.DataFrame(
    T,
    columns=[f"Next_{i}" for i in range(4)]
)

transition.index = [f"State_{i}" for i in range(4)]

transition.to_csv(
    os.path.join(DBN_DIR, "transition_matrix.csv")
)

print("\nTransition Matrix")
print(transition)

# ==========================================================
# EMISSION PROBABILITY
# ==========================================================

emission = train.groupby("State")["SepsisLabel"].mean()

emission.to_csv(
    os.path.join(DBN_DIR, "emission_probability.csv")
)

print("\nEmission Probability")
print(emission)

# ==========================================================
# SAVE STATES
# ==========================================================

train.to_csv(
    os.path.join(DBN_DIR,"train_dbn_states.csv"),
    index=False
)

valid.to_csv(
    os.path.join(DBN_DIR,"validation_dbn_states.csv"),
    index=False
)

test.to_csv(
    os.path.join(DBN_DIR,"test_dbn_states.csv"),
    index=False
)

print("\nDBN Training Completed")
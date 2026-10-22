import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

EMBED_DIR = os.path.join(BASE, "data", "processed", "embeddings")
OUT_DIR = os.path.join(BASE, "data", "processed", "dbn")

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# LOAD EMBEDDINGS
# ==========================================================

print("=" * 60)
print("DBN STATE PREPARATION")
print("=" * 60)

train = pd.read_csv(os.path.join(EMBED_DIR, "train_embeddings.csv"))
valid = pd.read_csv(os.path.join(EMBED_DIR, "validation_embeddings.csv"))
test = pd.read_csv(os.path.join(EMBED_DIR, "test_embeddings.csv"))

feature_cols = [c for c in train.columns if c.startswith("z")]

X_train = train[feature_cols].values
X_valid = valid[feature_cols].values
X_test = test[feature_cols].values

print("Train :", X_train.shape)
print("Valid :", X_valid.shape)
print("Test  :", X_test.shape)

# ==========================================================
# STANDARDIZE
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(OUT_DIR, "embedding_scaler.pkl")
)

# ==========================================================
# KMEANS -> 4 HIDDEN STATES
# ==========================================================

print("\nCreating Hidden Clinical States...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

train_states = kmeans.fit_predict(X_train_scaled)
valid_states = kmeans.predict(X_valid_scaled)
test_states = kmeans.predict(X_test_scaled)

joblib.dump(
    kmeans,
    os.path.join(OUT_DIR, "kmeans_states.pkl")
)

# ==========================================================
# ORDER STATES BY SEPSIS RISK
# ==========================================================

risk = []

for s in range(4):

    idx = train_states == s

    rate = train.loc[idx, "SepsisLabel"].mean()

    risk.append((s, rate))

risk = sorted(risk, key=lambda x: x[1])

mapping = {}

for new_state, (old_state, _) in enumerate(risk):
    mapping[old_state] = new_state

train_states = np.vectorize(mapping.get)(train_states)
valid_states = np.vectorize(mapping.get)(valid_states)
test_states = np.vectorize(mapping.get)(test_states)

print("\nState Mapping (Low Risk → High Risk)")
for old, new in mapping.items():
    print(f"Cluster {old} → State {new}")

# ==========================================================
# ADD STATES
# ==========================================================

train["HiddenState"] = train_states
valid["HiddenState"] = valid_states
test["HiddenState"] = test_states

# ==========================================================
# STATE STATISTICS
# ==========================================================

print("\nTraining State Distribution")

summary = (
    train.groupby("HiddenState")["SepsisLabel"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

summary.columns = [
    "State",
    "Patients",
    "SepsisCases",
    "SepsisRate"
]

print(summary)

summary.to_csv(
    os.path.join(OUT_DIR, "state_summary.csv"),
    index=False
)

# ==========================================================
# SAVE
# ==========================================================

train.to_csv(
    os.path.join(OUT_DIR, "train_states.csv"),
    index=False
)

valid.to_csv(
    os.path.join(OUT_DIR, "validation_states.csv"),
    index=False
)

test.to_csv(
    os.path.join(OUT_DIR, "test_states.csv"),
    index=False
)

print("\n" + "=" * 60)
print("DBN STATE PREPARATION COMPLETED")
print("=" * 60)

print("\nSaved Files:")
print(os.path.join(OUT_DIR, "train_states.csv"))
print(os.path.join(OUT_DIR, "validation_states.csv"))
print(os.path.join(OUT_DIR, "test_states.csv"))
print(os.path.join(OUT_DIR, "state_summary.csv"))
import os
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix
)

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"
ML = os.path.join(BASE, "data", "processed", "ml")

train = pd.read_csv(os.path.join(ML, "train_ml.csv"))
valid = pd.read_csv(os.path.join(ML, "validation_ml.csv"))
test  = pd.read_csv(os.path.join(ML, "test_ml.csv"))

# ==========================================================
# FEATURES
# ==========================================================

X_train = train.drop(columns=["ID", "SepsisLabel"])
y_train = train["SepsisLabel"]

X_valid = valid.drop(columns=["ID", "SepsisLabel"])
y_valid = valid["SepsisLabel"]

X_test = test.drop(columns=["ID", "SepsisLabel"])
y_test = test["SepsisLabel"]

# Remove columns that are completely empty
keep = X_train.columns[X_train.notna().any()]

X_train = X_train[keep]
X_valid = X_valid[keep]
X_test = X_test[keep]

print("Train :", X_train.shape)
print("Valid :", X_valid.shape)
print("Test  :", X_test.shape)

# ==========================================================
# MODEL
# ==========================================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate(name, X, y):

    prob = model.predict_proba(X)[:,1]
    pred = (prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    print("\n" + "="*55)
    print(name)
    print("="*55)

    print("ROC-AUC      :", round(roc_auc_score(y, prob), 4))
    print("PR-AUC       :", round(average_precision_score(y, prob), 4))
    print("F1-score     :", round(f1_score(y, pred), 4))
    print("Accuracy     :", round(accuracy_score(y, pred), 4))
    print("Sensitivity  :", round(recall_score(y, pred), 4))
    print("Specificity  :", round(tn/(tn+fp), 4))
    print("Precision    :", round(precision_score(y, pred), 4))

# ==========================================================
# RESULTS
# ==========================================================

evaluate("TRAIN", X_train, y_train)
evaluate("VALIDATION", X_valid, y_valid)
evaluate("TEST", X_test, y_test)
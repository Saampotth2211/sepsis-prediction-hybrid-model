import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score,
    recall_score, precision_score,
    confusion_matrix
)

# =====================================================
# PATHS
# =====================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

LSTM_DIR = os.path.join(BASE, "data", "processed", "lstm")
MODEL_DIR = os.path.join(BASE, "data", "processed", "models")
RESULT_DIR = os.path.join(BASE, "data", "processed", "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================================================
# LOAD DATA
# =====================================================

train = np.load(os.path.join(LSTM_DIR, "train_lstm.npz"))
valid = np.load(os.path.join(LSTM_DIR, "validation_lstm.npz"))
test = np.load(os.path.join(LSTM_DIR, "test_lstm.npz"))

X_train = np.concatenate([train["X"], train["M"]], axis=2)
X_valid = np.concatenate([valid["X"], valid["M"]], axis=2)
X_test = np.concatenate([test["X"], test["M"]], axis=2)

y_train = train["y"]
y_valid = valid["y"]
y_test = test["y"]

# =====================================================
# MODEL
# =====================================================

class ETSLayer(nn.Module):
    def __init__(self, alpha=0.35):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        smooth = torch.zeros_like(x)
        smooth[:,0,:] = x[:,0,:]
        for t in range(1,x.size(1)):
            smooth[:,t,:] = self.alpha*x[:,t,:] + (1-self.alpha)*smooth[:,t-1,:]
        return smooth


class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden*2,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        score = self.attention(x)
        w = torch.softmax(score, dim=1)
        context = torch.sum(w*x, dim=1)
        return context,w


class ETS_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.ets = ETSLayer()

        self.norm = nn.LayerNorm(80)

        self.bilstm = nn.LSTM(
            input_size=80,
            hidden_size=48,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        self.attention = TemporalAttention(48)

        self.fc = nn.Sequential(
            nn.Linear(96,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,1)
        )

    def forward(self,x):

        x = self.ets(x)
        x = self.norm(x)

        out,_ = self.bilstm(x)

        context,att = self.attention(out)

        context = self.dropout(context)

        out = self.fc(context)

        return out.squeeze(),att


model = ETS_BiLSTM().to(device)

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR,"ets_bilstm_v2.pt"),
        map_location=device
    )
)

model.eval()

print("Model Loaded")

# =====================================================
# PREDICTION
# =====================================================

def predict(X):

    probs=[]

    with torch.no_grad():

        for i in range(0,len(X),256):

            batch=torch.tensor(
                X[i:i+256],
                dtype=torch.float32
            ).to(device)

            out,_=model(batch)

            p=torch.sigmoid(out)

            probs.extend(p.cpu().numpy())

    return np.array(probs)

# =====================================================
# METRICS
# =====================================================

def evaluate(name,X,y):

    prob=predict(X)

    pred=(prob>=0.5).astype(int)

    tn,fp,fn,tp=confusion_matrix(y,pred).ravel()

    result={
        "Dataset":name,
        "ROC_AUC":roc_auc_score(y,prob),
        "PR_AUC":average_precision_score(y,prob),
        "F1":f1_score(y,pred),
        "Accuracy":accuracy_score(y,pred),
        "Sensitivity":recall_score(y,pred),
        "Specificity":tn/(tn+fp),
        "Precision":precision_score(y,pred)
    }

    print("\n"+"="*50)
    print(name.upper())
    print("="*50)

    for k,v in result.items():
        if k!="Dataset":
            print(f"{k:15}: {v:.4f}")

    return result,prob,pred


train_res,_,_=evaluate("Train",X_train,y_train)
valid_res,_,_=evaluate("Validation",X_valid,y_valid)
test_res,test_prob,test_pred=evaluate("Test",X_test,y_test)

# =====================================================
# SAVE
# =====================================================

metrics=pd.DataFrame([train_res,valid_res,test_res])

metrics.to_csv(
    os.path.join(
        RESULT_DIR,
        "ets_bilstm_metrics.csv"
    ),
    index=False
)

pred=pd.DataFrame({
    "Actual":y_test,
    "Probability":test_prob,
    "Prediction":test_pred
})

pred.to_csv(
    os.path.join(
        RESULT_DIR,
        "ets_bilstm_predictions.csv"
    ),
    index=False
)

print("\nEvaluation Completed")
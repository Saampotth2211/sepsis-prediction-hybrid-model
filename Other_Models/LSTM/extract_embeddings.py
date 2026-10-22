import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

LSTM_DIR = os.path.join(BASE, "data", "processed", "lstm")
MODEL_DIR = os.path.join(BASE, "data", "processed", "models")
OUT_DIR = os.path.join(BASE, "data", "processed", "embeddings")

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("LATENT EMBEDDING EXTRACTION")
print("="*60)
print("Device:", device)

# ==========================================================
# LOAD DATA
# ==========================================================

train = np.load(os.path.join(LSTM_DIR,"train_lstm.npz"))
valid = np.load(os.path.join(LSTM_DIR,"validation_lstm.npz"))
test = np.load(os.path.join(LSTM_DIR,"test_lstm.npz"))

# concatenate values + mask (40+40=80)

X_train = np.concatenate([train["X"],train["M"]],axis=2)
X_valid = np.concatenate([valid["X"],valid["M"]],axis=2)
X_test = np.concatenate([test["X"],test["M"]],axis=2)

y_train = train["y"]
y_valid = valid["y"]
y_test = test["y"]

print("Train :",X_train.shape)
print("Valid :",X_valid.shape)
print("Test  :",X_test.shape)

# ==========================================================
# TEMPORAL ATTENTION
# ==========================================================

class TemporalAttention(nn.Module):

    def __init__(self,hidden):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden*2,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self,x):

        w = torch.softmax(self.score(x),dim=1)

        context = torch.sum(w*x,dim=1)

        return context,w

# ==========================================================
# LSTM ENCODER
# ==========================================================

class LSTMEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm(80)

        self.lstm = nn.LSTM(
            input_size=80,
            hidden_size=32,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )

        self.attention = TemporalAttention(32)

        self.embedding = nn.Linear(64,64)

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(64,1)

    def forward(self,x):

        x = self.norm(x)

        out,_ = self.lstm(x)

        context,_ = self.attention(out)

        z = torch.relu(self.embedding(context))

        z = self.dropout(z)

        logit = self.classifier(z)

        return logit.squeeze(),z

# ==========================================================
# LOAD MODEL
# ==========================================================

model = LSTMEncoder().to(device)

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR,"lstm_encoder.pt"),
        map_location=device
    )
)

model.eval()

print("Encoder Loaded Successfully")

# ==========================================================
# EMBEDDING FUNCTION
# ==========================================================

def get_embeddings(X):

    emb=[]

    with torch.no_grad():

        for i in range(0,len(X),256):

            batch=torch.tensor(
                X[i:i+256],
                dtype=torch.float32
            ).to(device)

            _,z=model(batch)

            emb.append(z.cpu().numpy())

    return np.vstack(emb)

# ==========================================================
# EXTRACT
# ==========================================================

print("\nExtracting Train Embeddings...")
E_train=get_embeddings(X_train)

print("Extracting Validation Embeddings...")
E_valid=get_embeddings(X_valid)

print("Extracting Test Embeddings...")
E_test=get_embeddings(X_test)

# ==========================================================
# SAVE CSV
# ==========================================================

cols=[f"z{i+1}" for i in range(64)]

train_df=pd.DataFrame(E_train,columns=cols)
train_df["SepsisLabel"]=y_train

valid_df=pd.DataFrame(E_valid,columns=cols)
valid_df["SepsisLabel"]=y_valid

test_df=pd.DataFrame(E_test,columns=cols)
test_df["SepsisLabel"]=y_test

train_df.to_csv(
    os.path.join(OUT_DIR,"train_embeddings.csv"),
    index=False
)

valid_df.to_csv(
    os.path.join(OUT_DIR,"validation_embeddings.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUT_DIR,"test_embeddings.csv"),
    index=False
)

print("\n"+"="*60)
print("EMBEDDINGS GENERATED SUCCESSFULLY")
print("="*60)

print("Train :",train_df.shape)
print("Valid :",valid_df.shape)
print("Test  :",test_df.shape)

print("\nSaved To:")
print(OUT_DIR)
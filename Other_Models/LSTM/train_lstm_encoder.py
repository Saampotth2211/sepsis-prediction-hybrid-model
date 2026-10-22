import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

LSTM_DIR = os.path.join(BASE, "data", "processed", "lstm")
MODEL_DIR = os.path.join(BASE, "data", "processed", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "lstm_encoder.pt")

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("LSTM REPRESENTATION ENCODER")
print("=" * 60)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ==========================================================
# LOAD DATA
# ==========================================================

train = np.load(os.path.join(LSTM_DIR, "train_lstm.npz"))
valid = np.load(os.path.join(LSTM_DIR, "validation_lstm.npz"))

X_train = np.concatenate([train["X"], train["M"]], axis=2)
X_valid = np.concatenate([valid["X"], valid["M"]], axis=2)

y_train = train["y"]
y_valid = valid["y"]

print("Train:", X_train.shape)
print("Validation:", X_valid.shape)

# ==========================================================
# DATALOADER
# ==========================================================

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ),
    batch_size=64,
    shuffle=True
)

valid_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_valid, dtype=torch.float32),
        torch.tensor(y_valid, dtype=torch.float32)
    ),
    batch_size=128,
    shuffle=False
)

# ==========================================================
# TEMPORAL ATTENTION
# ==========================================================

class TemporalAttention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        # x = (batch, 24, 64)

        weights = self.score(x)

        weights = torch.softmax(weights, dim=1)

        context = torch.sum(weights * x, dim=1)

        return context, weights


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

        self.embedding = nn.Linear(64, 64)

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(64, 1)

    def forward(self, x):

        x = self.norm(x)

        lstm_out, _ = self.lstm(x)

        context, attn = self.attention(lstm_out)

        z = torch.relu(self.embedding(context))

        z = self.dropout(z)

        logits = self.classifier(z)

        return logits.squeeze(), z, attn


# ==========================================================
# MODEL
# ==========================================================

model = LSTMEncoder().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# ==========================================================
# TRAINING
# ==========================================================

EPOCHS = 40
PATIENCE = 8

best_loss = float("inf")
counter = 0

print("\nTraining Started\n")

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------

    model.train()

    train_loss = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _, _ = model(X)

        loss = criterion(logits, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------- VALID ----------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X, y in valid_loader:

            X = X.to(device)
            y = y.to(device)

            logits, _, _ = model(X)

            loss = criterion(logits, y)

            val_loss += loss.item()

    val_loss /= len(valid_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train {train_loss:.4f} | "
        f"Val {val_loss:.4f}"
    )

    # Save Best Model

    if val_loss < best_loss:

        best_loss = val_loss
        counter = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

    else:

        counter += 1

    if counter >= PATIENCE:

        print("\nEarly Stopping Triggered")
        break

# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("Best Validation Loss:", round(best_loss, 4))
print("Model Saved To:")
print(MODEL_PATH)
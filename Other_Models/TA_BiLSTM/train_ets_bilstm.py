import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

LSTM_DIR = os.path.join(
    BASE,
    "data",
    "processed",
    "lstm"
)

MODEL_DIR = os.path.join(
    BASE,
    "data",
    "processed",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ets_bilstm_v2.pt"
)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("="*60)
print("ETS ATTENTION BiLSTM V2")
print("="*60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================================
# LOAD DATA
# ==========================================================

train = np.load(
    os.path.join(
        LSTM_DIR,
        "train_lstm.npz"
    )
)

valid = np.load(
    os.path.join(
        LSTM_DIR,
        "validation_lstm.npz"
    )
)


X_train = np.concatenate(
    [
        train["X"],
        train["M"]
    ],
    axis=2
)


X_valid = np.concatenate(
    [
        valid["X"],
        valid["M"]
    ],
    axis=2
)


y_train = train["y"]
y_valid = valid["y"]


print("\nTrain:",X_train.shape)
print("Valid:",X_valid.shape)


# ==========================================================
# DATA LOADERS
# ==========================================================

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32
        )
    ),
    batch_size=64,
    shuffle=True
)


valid_loader = DataLoader(
    TensorDataset(
        torch.tensor(
            X_valid,
            dtype=torch.float32
        ),
        torch.tensor(
            y_valid,
            dtype=torch.float32
        )
    ),
    batch_size=128,
    shuffle=False
)



# ==========================================================
# ETS SMOOTHING
# ==========================================================

class ETSLayer(nn.Module):

    def __init__(self, alpha=0.35):

        super().__init__()

        self.alpha = alpha


    def forward(self,x):

        smooth = torch.zeros_like(x)

        smooth[:,0,:] = x[:,0,:]


        for t in range(1,x.size(1)):

            smooth[:,t,:] = (
                self.alpha*x[:,t,:]
                +
                (1-self.alpha)*smooth[:,t-1,:]
            )


        return smooth



# ==========================================================
# TEMPORAL ATTENTION
# ==========================================================

class TemporalAttention(nn.Module):

    def __init__(self,hidden):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                hidden*2,
                64
            ),

            nn.Tanh(),

            nn.Linear(
                64,
                1
            )

        )


    def forward(self,x):

        score = self.attention(x)

        weights = torch.softmax(
            score,
            dim=1
        )


        context = torch.sum(
            weights*x,
            dim=1
        )


        return context,weights




# ==========================================================
# ETS ATTENTION BiLSTM
# ==========================================================

class ETS_BiLSTM(nn.Module):


    def __init__(self):

        super().__init__()


        self.ets = ETSLayer(
            alpha=0.35
        )


        self.norm = nn.LayerNorm(
            80
        )


        self.bilstm = nn.LSTM(

            input_size=80,

            hidden_size=48,

            num_layers=1,

            batch_first=True,

            bidirectional=True
        )


        self.dropout = nn.Dropout(
            0.3
        )


        self.attention = TemporalAttention(
            48
        )


        self.fc = nn.Sequential(

            nn.Linear(
                96,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),


            nn.Linear(
                64,
                1
            )
        )



    def forward(self,x):


        x = self.ets(x)


        x = self.norm(x)


        output,_ = self.bilstm(x)


        context,weights = self.attention(
            output
        )


        context = self.dropout(
            context
        )


        result = self.fc(
            context
        )


        return result.squeeze(),weights




# ==========================================================
# CREATE MODEL
# ==========================================================


model = ETS_BiLSTM().to(device)


print("\nModel created")



# ==========================================================
# LOSS
# ==========================================================

criterion = nn.BCEWithLogitsLoss()



# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=0.0002,

    weight_decay=1e-4
)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    patience=3,

    factor=0.5
)



# ==========================================================
# TRAINING
# ==========================================================

EPOCHS = 50

PATIENCE = 10


best_loss = float("inf")

counter = 0



print("\nTRAINING STARTED")
print("="*60)



for epoch in range(EPOCHS):


    model.train()


    train_loss=0



    for X,y in train_loader:


        X=X.to(device)

        y=y.to(device)



        optimizer.zero_grad()


        output,_ = model(X)


        loss = criterion(
            output,
            y
        )


        loss.backward()



        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            1.0

        )


        optimizer.step()



        train_loss += loss.item()



    train_loss /= len(train_loader)




    # VALIDATION

    model.eval()


    val_loss=0


    with torch.no_grad():


        for X,y in valid_loader:


            X=X.to(device)

            y=y.to(device)


            output,_ = model(X)


            loss = criterion(
                output,
                y
            )


            val_loss += loss.item()



    val_loss /= len(valid_loader)



    scheduler.step(
        val_loss
    )



    print(
        f"Epoch {epoch+1:02d} | "
        f"Train {train_loss:.4f} | "
        f"Val {val_loss:.4f}"
    )



    if val_loss < best_loss:


        best_loss = val_loss

        counter=0


        torch.save(

            model.state_dict(),

            MODEL_PATH

        )


    else:

        counter+=1



    if counter >= PATIENCE:


        print("\nEarly stopping")

        break




print("\n"+"="*60)

print("TRAINING COMPLETED")

print("="*60)

print(
    "Best Validation Loss:",
    round(best_loss,4)
)


print(
    "Saved Model:"
)

print(
    MODEL_PATH
)
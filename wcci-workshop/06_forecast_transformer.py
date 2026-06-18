"""
06 – Temporal Transformer forecast for multiple bus stops.

Architecture:
  Positional Encoding
  → N × TransformerEncoderLayer (self-attention over time)
  → Global-average pooling across time
  → Linear head (forecast horizon × n_features)

The model treats the input sequence (T, N_stops) as a token stream
where each time step is one token of dimension N_stops.
"""

from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import load_timeseries, prepare_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR   = Path("outputs/transformer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ         = "1h"
TOP_N        = 20
INPUT_LEN    = 72
HORIZON      = 24
D_MODEL      = 64        # embedding dimension (must be divisible by N_HEADS)
N_HEADS      = 4
N_LAYERS     = 3
FFN_DIM      = 256
DROPOUT      = 0.1
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 5e-4
WARMUP_STEPS = 500
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Loading data ...")
ts = load_timeseries(start_date="2024-04-01", n_days=60, freq=FREQ, top_n=TOP_N)
splits   = prepare_sequences(ts, input_len=INPUT_LEN, horizon=HORIZON)
N        = splits["n_features"]

def to_tensor(a): return torch.tensor(a, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(to_tensor(splits["X_train"]), to_tensor(splits["y_train"])),
    batch_size=BATCH_SIZE, shuffle=True,
)
val_loader  = DataLoader(
    TensorDataset(to_tensor(splits["X_val"]),  to_tensor(splits["y_val"])),
    batch_size=BATCH_SIZE,
)
test_loader = DataLoader(
    TensorDataset(to_tensor(splits["X_test"]), to_tensor(splits["y_test"])),
    batch_size=BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_layers,
                 ffn_dim, horizon, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # Pre-LN (more stable training)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features * horizon),
        )
        self.n_features = n_features
        self.horizon    = horizon

    def forward(self, x):
        # x: (B, T, N)
        h = self.pos_enc(self.input_proj(x))   # (B, T, d_model)
        h = self.encoder(h)                     # (B, T, d_model)
        h = h.mean(dim=1)                       # (B, d_model)  global avg pooling
        out = self.head(h)                      # (B, N*horizon)
        return out.view(-1, self.horizon, self.n_features)


model = TransformerForecaster(
    N, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, HORIZON, DROPOUT
).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# Optimizer with linear warm-up + cosine decay
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98),
                               weight_decay=1e-4)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, EPOCHS * len(train_loader) - WARMUP_STEPS)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
criterion = nn.HuberLoss(delta=1.0)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
history = {"train_loss": [], "val_loss": []}
best_val   = float("inf")
best_state = None
global_step = 0

print(f"\nTraining on {DEVICE} for {EPOCHS} epochs ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        t_loss += loss.item() * len(xb)
        global_step += 1

    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            v_loss += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)

    t_loss /= len(splits["X_train"])
    v_loss /= len(splits["X_val"])
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)

    if v_loss < best_val:
        best_val   = v_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"train={t_loss:.5f}  val={v_loss:.5f}  lr={lr_now:.2e}")

model.load_state_dict(best_state)
torch.save(best_state, OUTPUT_DIR / "transformer_best.pt")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
        trues.append(yb.numpy())

preds     = np.concatenate(preds)
trues     = np.concatenate(trues)
scaler    = splits["scaler"]
pred_flat = np.clip(scaler.inverse_transform(preds.reshape(-1, N)), 0, None)
true_flat = np.clip(scaler.inverse_transform(trues.reshape(-1, N)), 0, None)

mae  = mean_absolute_error(true_flat, pred_flat)
rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
mask = true_flat != 0
mape = np.mean(np.abs((true_flat[mask] - pred_flat[mask]) / true_flat[mask])) * 100

print(f"\n{'='*40}")
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")
print(f"  MAPE : {mape:.2f}%")
print(f"{'='*40}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history["train_loss"], label="Train (Huber)")
ax.plot(history["val_loss"],   label="Val (Huber)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Huber loss")
ax.set_title("Transformer – training curves"); ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close(fig)

stop_idx  = 0
stop_name = splits["columns"][stop_idx]
series    = ts[stop_name]
train_tail = series.iloc[-(7 * 24 + HORIZON):-HORIZON]
actual     = series.iloc[-HORIZON:]
last_pred  = pred_flat[-HORIZON:, stop_idx]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_tail.index, train_tail.values, color="steelblue", lw=0.9, label="Train (tail)")
ax.plot(actual.index, actual.values, color="black", lw=1.2, label="Actual")
ax.plot(actual.index, last_pred, color="mediumorchid", lw=1.5, linestyle="--", label="Transformer forecast")
ax.set_title(f"Transformer | Stop {stop_name} | MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%",
             fontsize=11)
ax.set_ylabel("Boardings / hour"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "transformer_forecast.png", dpi=150)
plt.close(fig)

print(f"\nOutputs saved to {OUTPUT_DIR.resolve()}")

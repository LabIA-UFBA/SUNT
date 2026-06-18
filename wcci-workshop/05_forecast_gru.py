"""
05 – GRU forecast for multiple bus stops (multivariate).

Same pipeline as 04_forecast_lstm.py but using a Gated Recurrent Unit.
GRUs are computationally cheaper than LSTMs with comparable performance
on many time-series tasks.

Architecture: stacked Bidirectional-GRU → LayerNorm → Linear head
"""

from pathlib import Path
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
OUTPUT_DIR = Path("outputs/gru")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ        = "1h"
TOP_N       = 20
INPUT_LEN   = 72
HORIZON     = 24
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
BIDIREC     = True       # Bidirectional GRU
DROPOUT     = 0.2
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data  (identical to LSTM script)
# ---------------------------------------------------------------------------
print("Loading data ...")
ts = load_timeseries(start_date="2024-04-01", n_days=60, freq=FREQ, top_n=TOP_N)
splits   = prepare_sequences(ts, input_len=INPUT_LEN, horizon=HORIZON)
N        = splits["n_features"]

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

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

print(f"  Train: {len(splits['X_train'])}  "
      f"Val: {len(splits['X_val'])}  "
      f"Test: {len(splits['X_test'])}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GRUForecaster(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers, horizon,
                 dropout=0.2, bidirectional=True):
        super().__init__()
        self.bidirec  = bidirectional
        d_mult        = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.norm = nn.LayerNorm(hidden_dim * d_mult)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * d_mult, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features * horizon),
        )
        self.n_features = n_features
        self.horizon    = horizon

    def forward(self, x):
        # x: (B, T, N)
        out, _ = self.gru(x)                       # (B, T, H*d)
        last   = self.norm(out[:, -1, :])           # (B, H*d)
        pred   = self.head(last)                    # (B, N*horizon)
        return pred.view(-1, self.horizon, self.n_features)


model = GRUForecaster(N, HIDDEN_DIM, NUM_LAYERS, HORIZON,
                      DROPOUT, BIDIREC).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.HuberLoss(delta=1.0)  # robust to outliers

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
history = {"train_loss": [], "val_loss": []}
best_val = float("inf")
best_state = None

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
        t_loss += loss.item() * len(xb)

    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            v_loss += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)

    t_loss /= len(splits["X_train"])
    v_loss /= len(splits["X_val"])
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    scheduler.step()

    if v_loss < best_val:
        best_val   = v_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train={t_loss:.5f}  val={v_loss:.5f}")

model.load_state_dict(best_state)
torch.save(best_state, OUTPUT_DIR / "gru_best.pt")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
        trues.append(yb.numpy())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

scaler     = splits["scaler"]
pred_flat  = scaler.inverse_transform(preds.reshape(-1, N))
true_flat  = scaler.inverse_transform(trues.reshape(-1, N))
pred_flat  = np.clip(pred_flat, 0, None)
true_flat  = np.clip(true_flat, 0, None)

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
ax.set_title("GRU – training curves"); ax.legend()
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
ax.plot(actual.index, last_pred, color="darkorange", lw=1.5, linestyle="--", label="GRU forecast")
ax.set_title(f"Bidirectional GRU | Stop {stop_name} | "
             f"MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%", fontsize=11)
ax.set_ylabel("Boardings / hour"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "gru_forecast.png", dpi=150)
plt.close(fig)

print(f"\nOutputs saved to {OUTPUT_DIR.resolve()}")

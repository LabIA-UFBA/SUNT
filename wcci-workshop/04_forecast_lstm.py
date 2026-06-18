"""
04 – LSTM forecast for multiple bus stops (multivariate).

Architecture: stacked LSTM → Linear output head
Input  : past INPUT_LEN hourly readings for all N stops
Output : next HORIZON readings for all N stops

Evaluation: MAE, RMSE, MAPE on inverse-scaled values.
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
OUTPUT_DIR = Path("outputs/lstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ       = "1h"
TOP_N      = 20          # number of stops (= input features)
INPUT_LEN  = 72          # 3 days of hourly data
HORIZON    = 24          # forecast 1 day ahead
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT    = 0.2
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Loading data ...")
ts = load_timeseries(start_date="2024-04-01", n_days=60, freq=FREQ, top_n=TOP_N, source="hgface")
#ts = load_timeseries(start_date="2024-04-01", n_days=60, freq=FREQ, top_n=TOP_N, source="pip")
print(f"  Time series shape: {ts.shape}  (steps × stops)")

splits = prepare_sequences(ts, input_len=INPUT_LEN, horizon=HORIZON)
N = splits["n_features"]

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

train_ds = TensorDataset(to_tensor(splits["X_train"]), to_tensor(splits["y_train"]))
val_ds   = TensorDataset(to_tensor(splits["X_val"]),   to_tensor(splits["y_val"]))
test_ds  = TensorDataset(to_tensor(splits["X_test"]),  to_tensor(splits["y_test"]))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers, horizon, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, n_features * horizon)
        self.n_features = n_features
        self.horizon    = horizon

    def forward(self, x):
        # x: (B, T, N)
        out, _ = self.lstm(x)          # (B, T, H)
        last   = out[:, -1, :]         # (B, H)
        pred   = self.head(last)       # (B, N*horizon)
        return pred.view(-1, self.horizon, self.n_features)  # (B, S, N)


model = LSTMForecaster(N, HIDDEN_DIM, NUM_LAYERS, HORIZON, DROPOUT).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)
criterion = nn.MSELoss()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")
best_state    = None

print(f"\nTraining on {DEVICE} for {EPOCHS} epochs ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += criterion(model(xb), yb).item() * len(xb)

    train_loss /= len(train_ds)
    val_loss   /= len(val_ds)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.5f}  val={val_loss:.5f}")

model.load_state_dict(best_state)
torch.save(best_state, OUTPUT_DIR / "lstm_best.pt")
print(f"\nBest val loss: {best_val_loss:.5f} — weights saved.")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb.to(DEVICE)).cpu().numpy()
        all_pred.append(pred)
        all_true.append(yb.numpy())

all_pred = np.concatenate(all_pred, axis=0)  # (samples, horizon, N)
all_true = np.concatenate(all_true, axis=0)

# Inverse-scale: flatten (samples*horizon, N), inverse, reshape
scaler = splits["scaler"]
shape  = all_pred.shape
pred_flat = scaler.inverse_transform(all_pred.reshape(-1, N))
true_flat = scaler.inverse_transform(all_true.reshape(-1, N))

pred_flat = np.clip(pred_flat, 0, None)
true_flat = np.clip(true_flat, 0, None)

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
# A) Training curves
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history["train_loss"], label="Train loss")
ax.plot(history["val_loss"],   label="Val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.set_title("LSTM – training curves"); ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close(fig)

# B) Predicted vs actual — last test window on the original time axis
stop_idx  = 0
stop_name = splits["columns"][stop_idx]
series    = ts[stop_name]
train_tail = series.iloc[-(7 * 24 + HORIZON):-HORIZON]
actual     = series.iloc[-HORIZON:]
last_pred  = pred_flat[-HORIZON:, stop_idx]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_tail.index, train_tail.values, color="steelblue", lw=0.9, label="Train (tail)")
ax.plot(actual.index, actual.values, color="black", lw=1.2, label="Actual")
ax.plot(actual.index, last_pred, color="crimson", lw=1.5, linestyle="--", label="LSTM forecast")
ax.set_title(f"LSTM | Stop {stop_name} | MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%",
             fontsize=11)
ax.set_ylabel("Boardings / hour"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "lstm_forecast.png", dpi=150)
plt.close(fig)

print(f"\nOutputs saved to {OUTPUT_DIR.resolve()}")

"""
08 – Chronos zero-shot forecast for multiple bus stops.

Chronos is Amazon's foundation model for time series forecasting.
No training is required — pre-trained weights are loaded from HuggingFace
and applied zero-shot to the SUNT boarding data.

Architecture : T5-based encoder-decoder Transformer (200 M parameters)
Input        : past INPUT_LEN hourly readings, one stop at a time (univariate)
Output       : probabilistic forecast — median used as point estimate
Mode         : zero-shot (no fine-tuning)

Install: pip install chronos-forecasting
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import load_timeseries, prepare_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs/chronos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ        = "1h"
TOP_N       = 20
INPUT_LEN   = 128    # 128 steps ≈ 5 days; keeps memory manageable on CPU
HORIZON     = 24     # forecast 1 day ahead
NUM_SAMPLES = 10     # probabilistic samples — median used as point forecast
BATCH_SIZE  = 16     # windows per inference call (avoids OOM on CPU)
MODEL_ID    = "amazon/chronos-t5-small"  # 46 M params — good balance for CPU
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Loading data ...")
ts = load_timeseries(start_date="2024-04-01", n_days=60, freq=FREQ, top_n=TOP_N)
print(f"  Time series shape: {ts.shape}  (steps × stops)")

splits = prepare_sequences(ts, input_len=INPUT_LEN, horizon=HORIZON)
N      = splits["n_features"]
scaler = splits["scaler"]

print(f"  Train: {len(splits['X_train'])}  "
      f"Val: {len(splits['X_val'])}  "
      f"Test: {len(splits['X_test'])}")

# Inverse-scale test windows so Chronos receives actual boarding counts.
# Foundation models normalise internally; feeding MinMax-scaled data
# would distort its learned priors.
X_test = splits["X_test"]   # (S, INPUT_LEN, N) — scaled
y_test = splits["y_test"]   # (S, HORIZON,   N) — scaled

X_test_raw = np.stack([
    np.clip(scaler.inverse_transform(X_test[s]), 0, None)
    for s in range(len(X_test))
])  # (S, INPUT_LEN, N)

y_test_raw = np.stack([
    np.clip(scaler.inverse_transform(y_test[s]), 0, None)
    for s in range(len(y_test))
])  # (S, HORIZON, N)

# ---------------------------------------------------------------------------
# Model  (zero-shot — no training)
# ---------------------------------------------------------------------------
print(f"\nLoading Chronos checkpoint from HuggingFace ({MODEL_ID}) ...")
pipeline = ChronosPipeline.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
)
print("  Model ready (zero-shot, no fine-tuning).")

# ---------------------------------------------------------------------------
# Zero-shot inference — per stop, batched over test windows
# ---------------------------------------------------------------------------
S = len(X_test_raw)
all_pred = np.zeros((S, HORIZON, N), dtype=np.float32)

print(f"\nRunning zero-shot inference on {S} test windows × {N} stops ...")
for stop_i in range(N):
    preds_stop = []
    for batch_start in range(0, S, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, S)
        contexts = [
            torch.tensor(X_test_raw[s, :, stop_i], dtype=torch.float32)
            for s in range(batch_start, batch_end)
        ]
        # forecast: (num_samples, batch, HORIZON)
        forecast = pipeline.predict(contexts, prediction_length=HORIZON,
                                    num_samples=NUM_SAMPLES)
        # forecast: (batch, num_samples, HORIZON) — median over samples axis
        preds_stop.append(np.clip(forecast.median(dim=1).values.numpy(), 0, None))

    all_pred[:, :, stop_i] = np.concatenate(preds_stop, axis=0)

    if (stop_i + 1) % 5 == 0 or stop_i == N - 1:
        print(f"  Stop {stop_i + 1}/{N} done")

# ---------------------------------------------------------------------------
# Evaluation  (same metrics as LSTM / GRU)
# ---------------------------------------------------------------------------
pred_flat = all_pred.reshape(-1, N)    # already in original scale
true_flat = y_test_raw.reshape(-1, N)

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
stop_idx  = 0
stop_name = splits["columns"][stop_idx]
series    = ts[stop_name]
train_tail = series.iloc[-(7 * 24 + HORIZON):-HORIZON]
actual     = series.iloc[-HORIZON:]
last_pred  = pred_flat[-HORIZON:, stop_idx]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_tail.index, train_tail.values, color="steelblue", lw=0.9, label="Train (tail)")
ax.plot(actual.index, actual.values, color="black", lw=1.2, label="Actual")
ax.plot(actual.index, last_pred, color="royalblue", lw=1.5, linestyle="--",
        label="Chronos forecast (zero-shot)")
ax.set_title(
    f"Chronos (zero-shot) | Stop {stop_name} | "
    f"MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%",
    fontsize=11,
)
ax.set_ylabel("Boardings / hour")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "chronos_forecast.png", dpi=150)
plt.close(fig)

print(f"\nOutputs saved to {OUTPUT_DIR.resolve()}")

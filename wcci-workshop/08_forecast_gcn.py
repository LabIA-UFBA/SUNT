"""
09 – Pure GCN (Graph Convolutional Network) spatial forecast for the bus network.

Unlike T-GCN (08) which pairs GCN with a GRU for temporal recurrence, this
script uses GCN alone.  The INPUT_LEN past time steps are flattened into a
per-node feature vector and fed through stacked GCN layers, which aggregate
spatial information from neighbouring stops, before a linear head predicts
the next HORIZON hours.

This isolates the spatial contribution of graph convolution and makes the
role of the adjacency matrix easy to inspect.

Architecture
------------
  Input  : (B, N, INPUT_LEN)  —  B samples, N stops, INPUT_LEN past hours
  GCN ×k : aggregate from graph neighbours   → (B, N, GCN_HIDDEN)
  Linear : map hidden state to forecast       → (B, N, HORIZON)

Reference: Kipf & Welling, "Semi-Supervised Classification with Graph
           Convolutional Networks" (ICLR 2017).
"""

from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except ImportError:
    warnings.warn(
        "torch-geometric not installed — using custom adjacency-matrix GCN.\n"
        "Install with: pip install torch-geometric"
    )
    PYG_AVAILABLE = False

from utils import load_timeseries, load_od, build_graph_from_od

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs/gcn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ       = "1h"
TOP_N      = 30
INPUT_LEN  = 72    # 3 days of past boardings as node features
HORIZON    = 24    # predict next 24 hours
GCN_HIDDEN = 64
NUM_LAYERS = 3
DROPOUT    = 0.1
BATCH_SIZE = 32
EPOCHS     = 60
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data — same graph + sequence pipeline as 08_forecast_gnn.py
# ---------------------------------------------------------------------------
print("Loading data ...")
ts_all = load_timeseries(start_date="2024-04-01", n_days=60, freq=FREQ, top_n=TOP_N)
N      = ts_all.shape[1]

od     = load_od()
G_full = build_graph_from_od(od)

def _norm_id(s):
    s = str(s)
    return s[:-2] if s.endswith(".0") else s

G_full    = nx.relabel_nodes(G_full, {n: _norm_id(n) for n in G_full.nodes})
top_stops = [str(s) for s in ts_all.columns.tolist()]
ts_all.columns = top_stops

# Build adjacency matrix
adj = np.zeros((N, N), dtype=np.float32)
stop_to_idx = {s: i for i, s in enumerate(top_stops)}
for u, v, data in G_full.edges(data=True):
    if u in stop_to_idx and v in stop_to_idx:
        i, j = stop_to_idx[u], stop_to_idx[v]
        w = float(data.get("weight", 1))
        adj[i, j] = w
        adj[j, i] = w

# Symmetric degree normalisation:  D^{-1/2} A D^{-1/2} + I
deg          = adj.sum(axis=1)
deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
D            = np.diag(deg_inv_sqrt)
adj_norm     = D @ adj @ D + np.eye(N)
adj_tensor   = torch.tensor(adj_norm, dtype=torch.float32).to(DEVICE)

# Sparse edge representation for PyG
if PYG_AVAILABLE:
    nz          = np.nonzero(adj_norm)
    edge_index  = torch.tensor(np.stack(nz), dtype=torch.long).to(DEVICE)
    edge_weight = torch.tensor(adj_norm[nz], dtype=torch.float32).to(DEVICE)

print(f"  Graph: {N} nodes | non-zero edges: {int((adj > 0).sum())}")

# Scale and create sliding-window sequences
values = ts_all.values.astype(np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

X_list, y_list = [], []
for t in range(INPUT_LEN, len(scaled) - HORIZON + 1):
    X_list.append(scaled[t - INPUT_LEN : t].T)   # (N, INPUT_LEN)
    y_list.append(scaled[t : t + HORIZON].T)       # (N, HORIZON)

X_arr = np.array(X_list, dtype=np.float32)        # (S, N, INPUT_LEN)
y_arr = np.array(y_list, dtype=np.float32)        # (S, N, HORIZON)

n       = len(X_arr)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

from torch.utils.data import TensorDataset, DataLoader
Xt, yt = torch.tensor(X_arr), torch.tensor(y_arr)
X_train, y_train = Xt[:n_train],              yt[:n_train]
X_val,   y_val   = Xt[n_train:n_train+n_val], yt[n_train:n_train+n_val]
X_test,  y_test  = Xt[n_train+n_val:],        yt[n_train+n_val:]

train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  BATCH_SIZE)

print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GraphConvolution(nn.Module):
    """
    Spectral GCN layer using a pre-computed normalised adjacency matrix.

    Each node aggregates a weighted sum of its neighbours' features:
        H' = σ( Â · H · W )
    where Â = D^{-1/2} A D^{-1/2} + I  (self-loops included).

    Because Â is fixed, the layer reduces to two matrix multiplications
    and is therefore very efficient.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x  : (B, N, in_features)
        # adj: (N, N)  — shared across the batch, broadcasts automatically
        return torch.matmul(adj, x @ self.weight)   # (B, N, out_features)


class GCNForecaster(nn.Module):
    """
    Pure-spatial GCN forecaster (custom adjacency-matrix implementation).

    Stacked GCN layers propagate information across the graph; a linear
    head then maps each node's hidden state to its HORIZON-step forecast.
    No recurrence — all temporal context lives in the INPUT_LEN feature dim.
    """
    def __init__(self, input_len, gcn_hidden, horizon, num_layers, dropout):
        super().__init__()
        dims = [input_len] + [gcn_hidden] * num_layers
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(dims[i], dims[i + 1])
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(gcn_hidden, horizon)

    def forward(self, x, adj):
        # x: (B, N, INPUT_LEN)
        h = x
        for gcn in self.gcn_layers:
            h = torch.relu(gcn(h, adj))
            h = self.dropout(h)
        return self.head(h)   # (B, N, HORIZON)


if PYG_AVAILABLE:
    class GCNForecasterPyG(nn.Module):
        """
        Same architecture using torch_geometric GCNConv.

        GCNConv expects a flat (total_nodes, features) tensor, so the batch
        dimension is handled by stacking B identical graphs into one
        disconnected super-graph: each copy gets its own node-index offset so
        edges never cross batch boundaries.
        """
        def __init__(self, input_len, gcn_hidden, horizon, num_layers, dropout):
            super().__init__()
            dims = [input_len] + [gcn_hidden] * num_layers
            self.convs   = nn.ModuleList([
                GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
            ])
            self.dropout = nn.Dropout(dropout)
            self.head    = nn.Linear(gcn_hidden, horizon)

        @staticmethod
        def _batch_edge_index(edge_index, edge_weight, n_nodes, batch_size, device):
            """Replicate edge_index for B identical graphs with node offsets."""
            offsets = torch.arange(batch_size, device=device) * n_nodes   # (B,)
            # edge_index: (2, E) → repeat B times with offset
            ei = torch.cat([edge_index + off for off in offsets], dim=1)  # (2, B*E)
            ew = edge_weight.repeat(batch_size)                            # (B*E,)
            return ei, ew

        def forward(self, x, edge_index, edge_weight):
            # x: (B, N, INPUT_LEN)
            B, N, _ = x.shape
            ei, ew  = self._batch_edge_index(edge_index, edge_weight, N, B, x.device)
            h = x.reshape(B * N, -1)              # (B*N, INPUT_LEN)
            for conv in self.convs:
                h = torch.relu(conv(h, ei, ew))
                h = self.dropout(h)
            return self.head(h).reshape(B, N, -1) # (B, N, HORIZON)

    model = GCNForecasterPyG(INPUT_LEN, GCN_HIDDEN, HORIZON, NUM_LAYERS, DROPOUT).to(DEVICE)
else:
    model = GCNForecaster(INPUT_LEN, GCN_HIDDEN, HORIZON, NUM_LAYERS, DROPOUT).to(DEVICE)

print(f"\nModel: {'GCN (PyG)' if PYG_AVAILABLE else 'GCN (custom)'}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.MSELoss()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
history    = {"train": [], "val": []}
best_val   = float("inf")
best_state = None

print(f"\nTraining on {DEVICE} for {EPOCHS} epochs ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb, edge_index, edge_weight) if PYG_AVAILABLE else model(xb, adj_tensor)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item() * len(xb)

    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb, edge_index, edge_weight) if PYG_AVAILABLE else model(xb, adj_tensor)
            v_loss += criterion(pred, yb).item() * len(xb)

    t_loss /= len(X_train)
    v_loss /= len(X_val)
    history["train"].append(t_loss)
    history["val"].append(v_loss)
    scheduler.step()

    if v_loss < best_val:
        best_val   = v_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train={t_loss:.6f}  val={v_loss:.6f}")

model.load_state_dict(best_state)
torch.save(best_state, OUTPUT_DIR / "gcn_best.pt")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        pred = model(xb, edge_index, edge_weight) if PYG_AVAILABLE else model(xb, adj_tensor)
        preds.append(pred.cpu().numpy())
        trues.append(yb.numpy())

preds = np.concatenate(preds)   # (S, N, HORIZON)
trues = np.concatenate(trues)

pred_flat = scaler.inverse_transform(preds.transpose(0, 2, 1).reshape(-1, N))
true_flat = scaler.inverse_transform(trues.transpose(0, 2, 1).reshape(-1, N))
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
ax.plot(history["train"], label="Train loss")
ax.plot(history["val"],   label="Val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.set_title("GCN – training curves"); ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close(fig)

# B) Forecast on the real time axis
node_idx   = 0
stop_name  = top_stops[node_idx]
series     = ts_all[stop_name]
train_tail = series.iloc[-(7 * 24 + HORIZON):-HORIZON]
actual     = series.iloc[-HORIZON:]
last_pred  = pred_flat[-HORIZON:, node_idx]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_tail.index, train_tail.values, color="steelblue", lw=0.9, label="Train (tail)")
ax.plot(actual.index, actual.values, color="black", lw=1.2, label="Actual")
ax.plot(actual.index, last_pred, color="seagreen", lw=1.5, linestyle="--", label="GCN forecast")
ax.set_title(f"GCN | Stop {stop_name} | MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%",
             fontsize=11)
ax.set_ylabel("Boardings / hour"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "gcn_forecast.png", dpi=150)
plt.close(fig)

# C) Per-node MAE — shows which stops benefit most from spatial context
node_mae = np.abs(true_flat - pred_flat).mean(axis=0)
fig, ax  = plt.subplots(figsize=(12, 3))
ax.bar(range(N), node_mae, color="seagreen", alpha=0.7)
ax.set_xticks(range(N))
ax.set_xticklabels([str(s)[:8] for s in top_stops], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("MAE (boardings/h)")
ax.set_title("GCN – per-node MAE")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "gcn_per_node_mae.png", dpi=150)
plt.close(fig)

print(f"\nOutputs saved to {OUTPUT_DIR.resolve()}")

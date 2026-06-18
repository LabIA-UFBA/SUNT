"""
08 – Spatio-Temporal GNN forecast (T-GCN) for the bus network.

T-GCN = Graph Convolutional Network (spatial) + GRU (temporal).
Reference: Zhao et al., "T-GCN: A Temporal Graph Convolutional Network
for Traffic Forecasting" (2020).

Input  : (N_nodes, T_past)  — one value per node per time step
Output : (N_nodes, T_future)

Requires: torch-geometric  (pip install torch-geometric)
"""

from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import from_networkx, add_self_loops
    PYG_AVAILABLE = True
except ImportError:
    warnings.warn(
        "torch-geometric not installed.\n"
        "Install with: pip install torch-geometric\n"
        "Falling back to a simple adjacency-matrix GCN."
    )
    PYG_AVAILABLE = False

from utils import load_timeseries, load_od, build_graph_from_od

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR  = Path("outputs/gnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ        = "1h"
TOP_N       = 30          # nodes in the graph
INPUT_LEN   = 72          # 3 days of hourly data (same as LSTM/GRU/Transformer)
HORIZON     = 24          # predict next 24 hours (same as all other models)
GCN_HIDDEN  = 64
GRU_HIDDEN  = 64
NUM_LAYERS  = 2
DROPOUT     = 0.1
BATCH_SIZE  = 32
EPOCHS      = 60
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
print("Loading data ...")
ts_all = load_timeseries(start_date="2024-04-01", n_days=60, freq=FREQ, top_n=TOP_N)
N        = ts_all.shape[1]   # number of nodes

# Build graph restricted to these N stops
od = load_od()
G_full = build_graph_from_od(od)

# SUNTVisualizer converts float stop_ids to "44042532.0"; strip ".0" to match boarding
def _norm_id(s):
    s = str(s)
    return s[:-2] if s.endswith(".0") else s

G_full = nx.relabel_nodes(G_full, {n: _norm_id(n) for n in G_full.nodes})

top_stops = [str(s) for s in ts_all.columns.tolist()]
ts_all.columns = top_stops   # keep index consistent

# Subgraph adjacency
adj = np.zeros((N, N), dtype=np.float32)
stop_to_idx = {s: i for i, s in enumerate(top_stops)}

for u, v, data in G_full.edges(data=True):
    if u in stop_to_idx and v in stop_to_idx:
        i, j = stop_to_idx[u], stop_to_idx[v]
        w = float(data.get("weight", 1))
        adj[i, j] = w
        adj[j, i] = w  # symmetrize for undirected GCN

# Normalize adjacency (symmetric degree normalization)
deg = adj.sum(axis=1)
deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
D = np.diag(deg_inv_sqrt)
adj_norm = D @ adj @ D + np.eye(N)  # add self-loops
adj_tensor = torch.tensor(adj_norm, dtype=torch.float32).to(DEVICE)

# Build edge_index for PyG (from normalized adj)
if PYG_AVAILABLE:
    nz = np.nonzero(adj_norm)
    edge_index = torch.tensor(np.stack(nz), dtype=torch.long).to(DEVICE)
    edge_weight = torch.tensor(adj_norm[nz], dtype=torch.float32).to(DEVICE)

print(f"  Graph: {N} nodes | non-zero edges: {int((adj > 0).sum())}")

# Scale & create sequences
values = ts_all.values.astype(np.float32)   # (T, N)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)        # (T, N)

# Sliding windows → shape (samples, N, T)
X_list, y_list = [], []
for t in range(INPUT_LEN, len(scaled) - HORIZON + 1):
    X_list.append(scaled[t - INPUT_LEN : t].T)        # (N, INPUT_LEN)
    y_list.append(scaled[t : t + HORIZON].T)           # (N, HORIZON)

X_arr = np.array(X_list, dtype=np.float32)            # (S, N, T_in)
y_arr = np.array(y_list, dtype=np.float32)            # (S, N, T_out)

n        = len(X_arr)
n_train  = int(n * 0.7)
n_val    = int(n * 0.15)

Xt, yt = torch.tensor(X_arr), torch.tensor(y_arr)
X_train, y_train = Xt[:n_train],              yt[:n_train]
X_val,   y_val   = Xt[n_train:n_train+n_val], yt[n_train:n_train+n_val]
X_test,  y_test  = Xt[n_train+n_val:],        yt[n_train+n_val:]

from torch.utils.data import TensorDataset, DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  BATCH_SIZE)

print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ---------------------------------------------------------------------------
# T-GCN Model
# ---------------------------------------------------------------------------
class GraphConvolution(nn.Module):
    """Simple spectral GCN layer using pre-computed normalized adjacency."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x  : (B, N, in_features)
        # adj: (N, N)
        support = x @ self.weight               # (B, N, out)
        return torch.matmul(adj, support)       # (B, N, out)


class TGCN(nn.Module):
    """
    T-GCN: for each GRU step, the hidden state is updated via GCN
    so spatial dependencies are encoded at every time step.
    """
    def __init__(self, n_nodes, in_features, gcn_hidden, gru_hidden,
                 horizon, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru_hidden  = gru_hidden
        self.num_layers  = num_layers
        self.horizon     = horizon
        self.n_nodes     = n_nodes

        self.gcn_layers = nn.ModuleList([
            GraphConvolution(
                in_features if i == 0 else gcn_hidden,
                gcn_hidden
            )
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # GRU takes GCN output as input per time step
        self.gru  = nn.GRU(gcn_hidden, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, horizon)

    def forward(self, x, adj):
        # x  : (B, N, T_in)
        # adj: (N, N)
        B, N, T = x.shape

        # Process each time step through GCN
        gcn_out = []
        for t in range(T):
            h = x[:, :, t].unsqueeze(-1)          # (B, N, 1)
            for gcn in self.gcn_layers:
                h = torch.relu(gcn(h, adj))        # (B, N, gcn_hidden)
                h = self.dropout(h)
            gcn_out.append(h)                      # (B, N, gcn_hidden)

        gcn_seq = torch.stack(gcn_out, dim=2)      # (B, N, T, gcn_hidden)
        gcn_seq = gcn_seq.reshape(B * N, T, -1)    # (B*N, T, gcn_hidden)

        gru_out, _ = self.gru(gcn_seq)             # (B*N, T, gru_hidden)
        last       = gru_out[:, -1, :]             # (B*N, gru_hidden)
        pred       = self.head(last)               # (B*N, horizon)
        return pred.reshape(B, N, self.horizon)    # (B, N, horizon)


# Use PyG GCNConv if available, otherwise fallback to custom layer above
if PYG_AVAILABLE:
    class TGCNPyG(nn.Module):
        def __init__(self, n_nodes, gcn_hidden, gru_hidden, horizon, dropout=0.1):
            super().__init__()
            self.gcn1  = GCNConv(1, gcn_hidden)
            self.gcn2  = GCNConv(gcn_hidden, gcn_hidden)
            self.drop  = nn.Dropout(dropout)
            self.gru   = nn.GRU(gcn_hidden, gru_hidden, batch_first=True)
            self.head  = nn.Linear(gru_hidden, horizon)
            self.n_nodes = n_nodes
            self.horizon = horizon

        def forward(self, x, edge_index, edge_weight):
            B, N, T = x.shape
            gcn_out = []
            for t in range(T):
                xt = x[:, :, t].reshape(B * N, 1)          # (B*N, 1)
                # Batch edge_index
                ei = edge_index.clone()
                ew = edge_weight.clone()
                h  = torch.relu(self.gcn1(xt, ei, ew))
                h  = self.drop(h)
                h  = torch.relu(self.gcn2(h, ei, ew))
                gcn_out.append(h.reshape(B, N, -1))

            gcn_seq = torch.stack(gcn_out, dim=2).reshape(B * N, T, -1)
            out, _  = self.gru(gcn_seq)
            pred    = self.head(out[:, -1, :])
            return pred.reshape(B, N, self.horizon)

    model = TGCNPyG(N, GCN_HIDDEN, GRU_HIDDEN, HORIZON, DROPOUT).to(DEVICE)
else:
    model = TGCN(N, 1, GCN_HIDDEN, GRU_HIDDEN, HORIZON, NUM_LAYERS, DROPOUT).to(DEVICE)

print(f"\nModel: {'T-GCN (PyG)' if PYG_AVAILABLE else 'T-GCN (custom)'}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.MSELoss()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
history = {"train": [], "val": []}
best_val   = float("inf")
best_state = None

print(f"\nTraining on {DEVICE} for {EPOCHS} epochs ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        if PYG_AVAILABLE:
            pred = model(xb, edge_index, edge_weight)
        else:
            pred = model(xb, adj_tensor)
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
torch.save(best_state, OUTPUT_DIR / "tgcn_best.pt")

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

preds = np.concatenate(preds)    # (S, N, horizon)
trues = np.concatenate(trues)

# Inverse scale: (S*horizon, N) → inverse → reshape
S = preds.shape[0]
pred_flat = scaler.inverse_transform(
    preds.transpose(0, 2, 1).reshape(-1, N)
)
true_flat = scaler.inverse_transform(
    trues.transpose(0, 2, 1).reshape(-1, N)
)
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
ax.set_title("T-GCN – training curves"); ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close(fig)

# B) Predicted vs actual — last test window on the original time axis
node_idx  = 0
stop_name = top_stops[node_idx]
series    = ts_all[stop_name]
train_tail = series.iloc[-(7 * 24 + HORIZON):-HORIZON]
actual     = series.iloc[-HORIZON:]
last_pred  = pred_flat[-HORIZON:, node_idx]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_tail.index, train_tail.values, color="steelblue", lw=0.9, label="Train (tail)")
ax.plot(actual.index, actual.values, color="black", lw=1.2, label="Actual")
ax.plot(actual.index, last_pred, color="crimson", lw=1.5, linestyle="--", label="T-GCN forecast")
ax.set_title(f"T-GCN | Stop {stop_name} | MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%",
             fontsize=11)
ax.set_ylabel("Boardings / hour"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "tgcn_forecast.png", dpi=150)
plt.close(fig)

# Spatial error map: mean absolute error per node
node_mae = np.abs(true_flat - pred_flat).mean(axis=0)
fig, ax  = plt.subplots(figsize=(12, 3))
ax.bar(range(N), node_mae, color="teal", alpha=0.7)
ax.set_xticks(range(N))
ax.set_xticklabels([str(s)[:8] for s in top_stops], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("MAE (boardings/h)")
ax.set_title("T-GCN – per-node MAE")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "tgcn_per_node_mae.png", dpi=150)
plt.close(fig)

print(f"\nOutputs saved to {OUTPUT_DIR.resolve()}")

"""
10 – Spatio-Temporal GAT forecast (T-GAT) for the bus network.

T-GAT = Graph Attention Network (spatial) + GRU (temporal).

Key advantage over T-GCN (09): instead of fixed normalized adjacency weights,
GAT learns per-edge attention scores at each layer, so the model can
dynamically focus on the most informative neighbour stops.

Reference: Veličković et al., "Graph Attention Networks" (ICLR 2018).

Architecture
------------
  Input  : (B, N, INPUT_LEN)  —  B samples, N stops, INPUT_LEN past hours
  For each time step t:
    GAT ×2 : multi-head attention aggregation  → (B, N, GAT_HIDDEN)
  GRU      : temporal recurrence over the GAT sequence
  Linear   : forecast head  → (B, N, HORIZON)

Requires: torch-geometric  (pip install torch-geometric)
Falls back to a custom dense-attention GAT if PyG is absent.
"""

from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    warnings.warn(
        "torch-geometric not installed — using custom dense-attention GAT.\n"
        "Install with: pip install torch-geometric"
    )
    PYG_AVAILABLE = False

from utils import load_timeseries, load_od, build_graph_from_od

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs/gat")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ       = "1h"
TOP_N      = 30           # nodes in the graph
INPUT_LEN  = 72           # 3 days of hourly data
HORIZON    = 24           # predict next 24 hours
N_HEADS    = 4            # attention heads per GAT layer
GAT_HIDDEN = 64           # output features per GAT layer (after head aggregation)
GRU_HIDDEN = 64
NUM_LAYERS = 2            # stacked GAT layers per time step
DROPOUT    = 0.1
BATCH_SIZE = 32
EPOCHS     = 60
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data — same graph + sequence pipeline as 08/09
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

# Build adjacency matrix (used as attention mask in the custom fallback)
adj = np.zeros((N, N), dtype=np.float32)
stop_to_idx = {s: i for i, s in enumerate(top_stops)}
for u, v, data in G_full.edges(data=True):
    if u in stop_to_idx and v in stop_to_idx:
        i, j = stop_to_idx[u], stop_to_idx[v]
        w = float(data.get("weight", 1))
        adj[i, j] = w
        adj[j, i] = w  # symmetrize

# Add self-loops so every node can attend to itself
adj_with_loops = adj + np.eye(N, dtype=np.float32)
adj_tensor     = torch.tensor(adj_with_loops, dtype=torch.float32).to(DEVICE)

# Sparse edge representation for PyG (raw adjacency — GAT learns its own weights)
if PYG_AVAILABLE:
    nz         = np.nonzero(adj_with_loops)
    edge_index = torch.tensor(np.stack(nz), dtype=torch.long).to(DEVICE)

print(f"  Graph: {N} nodes | non-zero edges: {int((adj > 0).sum())}")

# Scale and build sliding-window sequences
values = ts_all.values.astype(np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

X_list, y_list = [], []
for t in range(INPUT_LEN, len(scaled) - HORIZON + 1):
    X_list.append(scaled[t - INPUT_LEN : t].T)    # (N, INPUT_LEN)
    y_list.append(scaled[t : t + HORIZON].T)       # (N, HORIZON)

X_arr = np.array(X_list, dtype=np.float32)         # (S, N, INPUT_LEN)
y_arr = np.array(y_list, dtype=np.float32)         # (S, N, HORIZON)

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
# Model — Custom dense GAT (fallback when torch-geometric is absent)
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """
    Single GAT layer using a dense N×N attention matrix.

    For each pair (i, j) where adj[i,j] > 0:
        e_ij = LeakyReLU( a^T [W·h_i || W·h_j] )
        α_ij = softmax_j( e_ij )          masked to neighbours only
        h'_i = σ( Σ_j α_ij · W·h_j )     aggregation

    Multi-head variant averages the per-head outputs (concat=False style).
    """
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads     = n_heads
        self.out_features = out_features

        # Shared linear per head
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        # Attention vector: a ∈ R^{2·out_features} per head
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x   : (B, N, in_features)
        # adj : (N, N)  — used as neighbourhood mask (values > 0 = valid edge)
        B, N, _ = x.shape
        H, F    = self.n_heads, self.out_features

        # Linear projection → (B, N, H, F)
        Wh = self.W(x).view(B, N, H, F)

        # Pair-wise concatenation for attention: (B, N, N, H, 2F)
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, H, F)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, H, F)
        cat  = torch.cat([Wh_i, Wh_j], dim=-1)          # (B, N, N, H, 2F)

        # Attention score per head: (B, N, N, H)
        e = self.leaky_relu((cat * self.a).sum(dim=-1))

        # Mask out non-edges (set to -inf before softmax)
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)    # (1, N, N, 1)
        e    = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=2)                     # (B, N, N, H)
        alpha = self.dropout(alpha)

        # Aggregate: Σ_j α_ij · Wh_j  →  average over heads  → (B, N, F)
        out = torch.einsum("bnjh,bjhf->bnhf", alpha, Wh)  # (B, N, H, F)
        return F.elu(out.mean(dim=2))                      # (B, N, F)


class TGAT(nn.Module):
    """
    T-GAT: at each GRU time step, node features are passed through stacked
    GAT layers for spatial aggregation, then the sequence is fed into a GRU
    for temporal modelling.
    """
    def __init__(self, n_nodes, gat_hidden, gru_hidden, horizon,
                 num_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.horizon   = horizon
        self.n_nodes   = n_nodes

        # First GAT layer: 1 input feature (one time step) → gat_hidden
        # Subsequent layers: gat_hidden → gat_hidden
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_f = 1 if i == 0 else gat_hidden
            self.gat_layers.append(
                GraphAttentionLayer(in_f, gat_hidden, n_heads, dropout)
            )

        self.dropout = nn.Dropout(dropout)
        self.gru     = nn.GRU(gat_hidden, gru_hidden, batch_first=True)
        self.head    = nn.Linear(gru_hidden, horizon)

    def forward(self, x, adj):
        # x  : (B, N, T_in)
        # adj: (N, N)
        B, N, T = x.shape

        gat_seq = []
        for t in range(T):
            h = x[:, :, t].unsqueeze(-1)       # (B, N, 1)
            for gat in self.gat_layers:
                h = gat(h, adj)                # (B, N, gat_hidden)
                h = self.dropout(h)
            gat_seq.append(h)

        gat_seq = torch.stack(gat_seq, dim=2)  # (B, N, T, gat_hidden)
        gat_seq = gat_seq.reshape(B * N, T, -1)# (B*N, T, gat_hidden)

        gru_out, _ = self.gru(gat_seq)         # (B*N, T, gru_hidden)
        pred = self.head(gru_out[:, -1, :])    # (B*N, horizon)
        return pred.reshape(B, N, self.horizon) # (B, N, horizon)


# ---------------------------------------------------------------------------
# Model — PyG variant (GATConv)
# ---------------------------------------------------------------------------
if PYG_AVAILABLE:
    class TGATPyG(nn.Module):
        """
        T-GAT using torch_geometric GATConv.

        GATConv(concat=False) averages attention heads so out_channels is
        exactly GAT_HIDDEN regardless of N_HEADS, keeping the GRU input
        dimension fixed.
        """
        def __init__(self, gat_hidden, gru_hidden, horizon, n_heads=4,
                     num_layers=2, dropout=0.1):
            super().__init__()
            self.horizon = horizon

            self.gat_layers = nn.ModuleList()
            for i in range(num_layers):
                in_c = 1 if i == 0 else gat_hidden
                self.gat_layers.append(
                    GATConv(in_c, gat_hidden, heads=n_heads,
                            concat=False, dropout=dropout)
                )

            self.dropout = nn.Dropout(dropout)
            self.gru     = nn.GRU(gat_hidden, gru_hidden, batch_first=True)
            self.head    = nn.Linear(gru_hidden, horizon)

        @staticmethod
        def _batch_edge_index(edge_index, n_nodes, batch_size, device):
            offsets = torch.arange(batch_size, device=device) * n_nodes
            return torch.cat([edge_index + off for off in offsets], dim=1)

        def forward(self, x, edge_index):
            B, N, T = x.shape
            ei = self._batch_edge_index(edge_index, N, B, x.device)

            gat_seq = []
            for t in range(T):
                h = x[:, :, t].reshape(B * N, 1)
                for gat in self.gat_layers:
                    h = F.elu(gat(h, ei))
                    h = self.dropout(h)
                gat_seq.append(h.reshape(B, N, -1))

            gat_seq = torch.stack(gat_seq, dim=2).reshape(B * N, T, -1)
            out, _  = self.gru(gat_seq)
            pred    = self.head(out[:, -1, :])
            return pred.reshape(B, N, self.horizon)

    model = TGATPyG(GAT_HIDDEN, GRU_HIDDEN, HORIZON, N_HEADS, NUM_LAYERS, DROPOUT).to(DEVICE)
else:
    model = TGAT(N, GAT_HIDDEN, GRU_HIDDEN, HORIZON, NUM_LAYERS, N_HEADS, DROPOUT).to(DEVICE)

print(f"\nModel: {'T-GAT (PyG)' if PYG_AVAILABLE else 'T-GAT (custom)'}")
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
        pred = model(xb, edge_index) if PYG_AVAILABLE else model(xb, adj_tensor)
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
            pred = model(xb, edge_index) if PYG_AVAILABLE else model(xb, adj_tensor)
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
torch.save(best_state, OUTPUT_DIR / "gat_best.pt")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        pred = model(xb, edge_index) if PYG_AVAILABLE else model(xb, adj_tensor)
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
ax.set_title("T-GAT – training curves"); ax.legend()
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
ax.plot(actual.index, last_pred, color="darkorange", lw=1.5, linestyle="--", label="T-GAT forecast")
ax.set_title(f"T-GAT | Stop {stop_name} | MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%",
             fontsize=11)
ax.set_ylabel("Boardings / hour"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "gat_forecast.png", dpi=150)
plt.close(fig)

# C) Per-node MAE
node_mae = np.abs(true_flat - pred_flat).mean(axis=0)
fig, ax  = plt.subplots(figsize=(12, 3))
ax.bar(range(N), node_mae, color="darkorange", alpha=0.7)
ax.set_xticks(range(N))
ax.set_xticklabels([str(s)[:8] for s in top_stops], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("MAE (boardings/h)")
ax.set_title("T-GAT – per-node MAE")
fig.tight_layout(); fig.savefig(OUTPUT_DIR / "gat_per_node_mae.png", dpi=150)
plt.close(fig)

# D) Attention heatmap — only available with the custom implementation
# Shows the average attention weight from each source stop (row) to each
# target stop (column) on the last test batch, averaged across layers and heads.
if not PYG_AVAILABLE:
    model.eval()
    sample_x = X_test[:1].to(DEVICE)  # (1, N, T)

    attn_maps = []
    with torch.no_grad():
        B, N_nodes, T = sample_x.shape
        for t in range(T):
            h = sample_x[:, :, t].unsqueeze(-1)
            for gat_layer in model.gat_layers:
                # Re-compute attention weights for this layer
                H, F  = gat_layer.n_heads, gat_layer.out_features
                Wh    = gat_layer.W(h).view(B, N_nodes, H, F)
                Wh_i  = Wh.unsqueeze(2).expand(B, N_nodes, N_nodes, H, F)
                Wh_j  = Wh.unsqueeze(1).expand(B, N_nodes, N_nodes, H, F)
                cat   = torch.cat([Wh_i, Wh_j], dim=-1)
                e     = gat_layer.leaky_relu((cat * gat_layer.a).sum(-1))
                mask_a = (adj_tensor == 0).unsqueeze(0).unsqueeze(-1)
                e     = e.masked_fill(mask_a, float("-inf"))
                alpha = F.softmax(e, dim=2)           # (1, N, N, H)
                attn_maps.append(alpha[0].mean(dim=-1).cpu().numpy())  # (N, N)
                h = gat_layer(h, adj_tensor)

    mean_attn = np.stack(attn_maps).mean(axis=0)     # (N, N)
    fig, ax   = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mean_attn, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(N)); ax.set_xticklabels([str(s)[:8] for s in top_stops],
                                                  rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(N)); ax.set_yticklabels([str(s)[:8] for s in top_stops], fontsize=6)
    ax.set_title("T-GAT – mean attention weight (source → target)")
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "gat_attention.png", dpi=150)
    plt.close(fig)
    print("  Attention heatmap saved.")

print(f"\nOutputs saved to {OUTPUT_DIR.resolve()}")

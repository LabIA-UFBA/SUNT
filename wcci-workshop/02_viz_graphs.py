"""
02 – Graph visualization of the bus network.

Produces:
  A) Simplified graph: top 20 stops by mean occupancy, geographic layout
  B) Same graph, but node size ∝ occupancy, edge width ∝ mean loading
  C) Interactive Folium map exported to HTML

Requires: networkx, matplotlib, folium
Optional: cartopy (for basemap background)
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    warnings.warn("folium not installed – skipping interactive map (pip install folium)")
    FOLIUM_AVAILABLE = False

from suntdataset import SUNTVisualizer
from utils import load_timeseries, load_od, build_graph_from_od

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs/viz_graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20          # number of top stops to display
FREQ  = "1h"        # aggregation frequency for occupancy computation

# ---------------------------------------------------------------------------
# Load data & compute mean occupancy per stop
# ---------------------------------------------------------------------------
print("Loading data ...")
ts = load_timeseries(freq=FREQ, top_n=200)  # wider pool, aggregated on the fly

mean_occ = ts.mean().sort_values(ascending=False)
top_stops = mean_occ.head(TOP_N).index.tolist()
print(f"Top {TOP_N} stops by mean hourly boardings:")
for s, v in mean_occ.head(TOP_N).items():
    print(f"  {s:>10} → {v:.1f} boardings/h")

# ---------------------------------------------------------------------------
# Build full graph and extract subgraph of top stops
# ---------------------------------------------------------------------------
print("\nBuilding graph from OD data ...")
od = load_od()

# SUNTVisualizer.build_od_graph() uses the original column name "n-boardings"
# build_graph_from_od() in data_loader handles the rename transparently
G_full = build_graph_from_od(od)
print(f"  Full graph: {G_full.number_of_nodes():,} nodes, {G_full.number_of_edges():,} edges")

# SUNTVisualizer converts stop_id via str(), so floats become "44042532.0".
# Normalize all node IDs: strip trailing ".0" to match boarding stop_ids.
def _norm_id(s):
    s = str(s)
    return s[:-2] if s.endswith(".0") else s

G_norm = nx.relabel_nodes(G_full, {n: _norm_id(n) for n in G_full.nodes})

# Enrich nodes with lat/lon from GTFS stops
try:
    from utils import load_gtfs
    stops_gtfs = load_gtfs("stops")
    for _, row in stops_gtfs.iterrows():
        sid = _norm_id(row.get("stop_id", ""))
        if G_norm.has_node(sid):
            G_norm.nodes[sid]["lat"]  = float(row.get("stop_lat", 0))
            G_norm.nodes[sid]["lon"]  = float(row.get("stop_lon", 0))
            G_norm.nodes[sid]["name"] = str(row.get("stop_name", sid))
except Exception as e:
    print(f"  (GTFS stop metadata not loaded: {e})")

# top_stops come from boarding time series — already without ".0"
# Subgraph induced by top 20 stops
G_top = G_norm.subgraph(top_stops).copy()
# Add any missing top nodes (stops isolated from the OD subset)
for s in top_stops:
    if s not in G_top:
        G_top.add_node(s)

# Annotate nodes with mean occupancy
for node in G_top.nodes:
    G_top.nodes[node]["mean_boardings"] = float(mean_occ.get(node, 0))

print(f"  Top-{TOP_N} subgraph: {G_top.number_of_nodes()} nodes, {G_top.number_of_edges()} edges")

# ---------------------------------------------------------------------------
# Layout: prefer geographic coordinates; fall back to spring layout
# ---------------------------------------------------------------------------
def get_pos(graph: nx.DiGraph) -> dict:
    geo_pos = {}
    for n, data in graph.nodes(data=True):
        lat = data.get("lat")
        lon = data.get("lon")
        if lat and lon and not np.isnan(lat) and not np.isnan(lon):
            geo_pos[n] = (lon, lat)
    # Use geographic layout only when every node is covered; mixing coordinate
    # systems (degrees vs. spring's [-1,1]) clusters the unmatched nodes.
    if len(geo_pos) == len(graph):
        return geo_pos
    return nx.spring_layout(graph, seed=42, k=2.0)


pos = get_pos(G_norm)

# ---------------------------------------------------------------------------
# Figure A – Full graph, top 20 stops highlighted
# ---------------------------------------------------------------------------
top_set = set(top_stops)

# Annotate all nodes with mean boardings (0 for non-top stops)
for node in G_norm.nodes:
    G_norm.nodes[node]["mean_boardings"] = float(mean_occ.get(node, 0))

# Separate node lists for layered drawing
other_nodes = [n for n in G_norm.nodes if n not in top_set]
top_nodes   = [n for n in G_norm.nodes if n in top_set]

# Top-20 node sizes scaled by boardings
top_boardings = [G_norm.nodes[n]["mean_boardings"] for n in top_nodes]
max_b = max(top_boardings) or 1
top_sizes = [400 + 2000 * (b / max_b) for b in top_boardings]

# Edge widths from full graph
edge_weights = [G_norm[u][v].get("weight", 1) for u, v in G_norm.edges]
max_w        = max(edge_weights, default=1) or 1
edge_widths  = [0.3 + 2.5 * (w / max_w) for w in edge_weights]

cmap = cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=max_b)
top_colors = [cmap(norm(b)) for b in top_boardings]

fig, ax = plt.subplots(figsize=(12, 9))

# Background: full network edges (thin, low opacity)
nx.draw_networkx_edges(G_norm, pos, ax=ax, width=edge_widths,
                       edge_color="steelblue", alpha=0.25,
                       arrows=False)

# Background: non-top nodes (small, grey)
nx.draw_networkx_nodes(G_norm, pos, ax=ax, nodelist=other_nodes,
                       node_size=30, node_color="lightgrey", alpha=0.5)

# Foreground: top-20 nodes (large, colored)
nx.draw_networkx_nodes(G_norm, pos, ax=ax, nodelist=top_nodes,
                       node_size=top_sizes, node_color=top_colors, alpha=0.95)

# Labels only for top-20
nx.draw_networkx_labels(G_norm, pos, ax=ax,
                        labels={n: str(n) for n in top_nodes},
                        font_size=6, font_color="black")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Mean boardings/h (top 20)")
ax.set_title(f"Bus network – top {TOP_N} stops by occupancy highlighted (Salvador, BA)", fontsize=13)
ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "A_top20_graph.png", dpi=150)
plt.close(fig)
print("\nSaved A_top20_graph.png")

# ---------------------------------------------------------------------------
# Figure B – Degree-centrality highlight
# ---------------------------------------------------------------------------
centrality = nx.degree_centrality(G_full)
top_centrality = sorted(centrality, key=centrality.get, reverse=True)[:TOP_N]
G_cent = G_full.subgraph(top_centrality).copy()
pos_c  = get_pos(G_cent)
missing_c = [n for n in G_cent.nodes if n not in pos_c]
if missing_c:
    pos_c.update(nx.spring_layout(G_cent.subgraph(missing_c), seed=0))

cent_values = [centrality[n] for n in G_cent.nodes]
fig, ax = plt.subplots(figsize=(10, 8))
nx.draw_networkx(
    G_cent, pos_c, ax=ax,
    node_size=[500 + 3000 * v for v in cent_values],
    node_color=cent_values, cmap="plasma",
    edge_color="grey", width=0.8, alpha=0.85,
    font_size=7, arrows=True, arrowsize=12,
    connectionstyle="arc3,rad=0.08",
)
ax.set_title(f"Top {TOP_N} stops by degree centrality", fontsize=13)
ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "B_centrality_graph.png", dpi=150)
plt.close(fig)
print("Saved B_centrality_graph.png")

# ---------------------------------------------------------------------------
# Figure C – Interactive Folium map
# ---------------------------------------------------------------------------
if FOLIUM_AVAILABLE:
    # Center the map on Salvador, BA
    lat_center = -12.9714
    lon_center = -38.5014

    fmap = folium.Map(location=[lat_center, lon_center], zoom_start=12,
                      tiles="CartoDB positron")

    max_b = mean_occ.head(TOP_N).max()
    for stop in top_stops:
        node_data = G_top.nodes.get(stop, {})
        lat = node_data.get("lat", lat_center)
        lon = node_data.get("lon", lon_center)
        if np.isnan(lat) or lat == 0.0:
            continue
        occ = mean_occ.get(stop, 0)
        radius = 5 + 20 * (occ / max_b)
        color_val = int(255 * occ / max_b)
        color = f"#{color_val:02x}{255 - color_val:02x}00"
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"Stop {stop}<br>Mean: {occ:.1f} board/h",
        ).add_to(fmap)

    for u, v, data in G_top.edges(data=True):
        u_d = G_top.nodes.get(u, {})
        v_d = G_top.nodes.get(v, {})
        ulat, ulon = u_d.get("lat", 0), u_d.get("lon", 0)
        vlat, vlon = v_d.get("lat", 0), v_d.get("lon", 0)
        if any(np.isnan([ulat, ulon, vlat, vlon])) or ulat == 0:
            continue
        w = data.get("weight", 1)
        folium.PolyLine(
            locations=[[ulat, ulon], [vlat, vlon]],
            weight=1 + 3 * (w / max_w),
            color="royalblue",
            opacity=0.5,
            tooltip=f"Loading: {w:.1f}",
        ).add_to(fmap)

    out_html = OUTPUT_DIR / "C_interactive_map.html"
    fmap.save(str(out_html))
    print(f"Saved C_interactive_map.html → open in browser")
else:
    print("Skipped interactive map (folium not available)")

print(f"\nAll outputs saved to {OUTPUT_DIR.resolve()}")

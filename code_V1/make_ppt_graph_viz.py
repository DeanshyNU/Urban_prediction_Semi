"""Generate a PPT-quality figure showing:
   - Train (50, blue) + Valid (8, red star) + Unlabeled (400, gray)
   - k-NN graph edges (k=10) on Chicago basemap

Usage:
    python code_V1/make_ppt_graph_viz.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.io as sio
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import contextily as cx

from data import fps_select_stations, build_knn_adj

DATA_DIR = '/home/hhz6461/Urban_prediction_Semi/data'
OUT_DIR  = '/home/hhz6461/Urban_prediction_Semi/data/figures'

# ---- Load 58 labeled stations + spatial split ----
d = sio.loadmat(f'{DATA_DIR}/Labeled_Finalized_new.mat')
loc_l = d['NodeLocation']  # (58, 2) = (lat, lon)
nL = loc_l.shape[0]
valid_idx = fps_select_stations(loc_l.T, n_select=8, seed=42)
train_idx = sorted(set(range(nL)) - set(valid_idx))

# ---- Load unlabeled, FPS select 400 ----
with h5py.File(f'{DATA_DIR}/Unlabeled_Finalized.mat', 'r') as f:
    loc_u_all = np.array(f['NodeLocation']).T  # (N_total, 2)
unl_sel = fps_select_stations(loc_u_all.T, n_select=400, seed=0)
loc_u = loc_u_all[unl_sel]  # (400, 2)
nU = loc_u.shape[0]

# ---- Build kNN graph (k=10) over all 458 nodes ----
locs_all = np.vstack([loc_l, loc_u]).T   # (2, 458)
n_total = locs_all.shape[1]
print(f"Building k-NN graph on {n_total} nodes (k=10)...")
Adj = build_knn_adj(locs_all, k=10)
src, dst = np.nonzero(Adj)
print(f"k-NN edges: {len(src)}")

# Web Mercator
def lonlat_to_mercator(lon, lat):
    R = 6378137.0
    return R * np.radians(lon), R * np.log(np.tan(np.pi/4 + np.radians(lat)/2))

all_lat = np.concatenate([loc_l[:, 0], loc_u[:, 0]])
all_lon = np.concatenate([loc_l[:, 1], loc_u[:, 1]])
all_x, all_y = lonlat_to_mercator(all_lon, all_lat)

# Slice arrays by node type
train_x, train_y = all_x[train_idx], all_y[train_idx]
valid_x, valid_y = all_x[valid_idx], all_y[valid_idx]
unl_x, unl_y     = all_x[nL:], all_y[nL:]

# Build edge segments only for unique edges (keep i<j)
mask = src < dst
src_u, dst_u = src[mask], dst[mask]
edge_segs = np.array([[(all_x[s], all_y[s]), (all_x[t], all_y[t])]
                       for s, t in zip(src_u, dst_u)])
print(f"Unique edges to draw: {len(edge_segs)}")

# Color scheme
TRAIN_COLOR = '#1f77b4'
VALID_COLOR = '#d62728'
UNL_COLOR   = '#7f7f7f'
EDGE_COLOR  = '#444444'

PAD = 1500
xmin, xmax = all_x.min() - PAD, all_x.max() + PAD
ymin, ymax = all_y.min() - PAD, all_y.max() + PAD

# =========================================================
# Figure
# =========================================================
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

# Edges first (background)
lc = LineCollection(edge_segs, colors=EDGE_COLOR, linewidths=0.3, alpha=0.35, zorder=1)
ax.add_collection(lc)

# Nodes on top, in stacking order so labeled stays visible
ax.scatter(unl_x, unl_y, c=UNL_COLOR, s=22, alpha=0.75, zorder=2,
           edgecolor='white', linewidth=0.3,
           label=f'Unlabeled auxiliary ({nU})')
ax.scatter(train_x, train_y, c=TRAIN_COLOR, s=90, zorder=3,
           edgecolor='white', linewidth=1.0,
           label=f'Train labeled ({len(train_idx)})')
ax.scatter(valid_x, valid_y, c=VALID_COLOR, s=240, marker='*', zorder=4,
           edgecolor='white', linewidth=1.2,
           label=f'Valid labeled ({len(valid_idx)})')

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
cx.add_basemap(ax, crs='EPSG:3857', source=cx.providers.CartoDB.Positron,
               attribution_size=7)
ax.set_axis_off()

# Legend with title
leg = ax.legend(loc='lower right', fontsize=12, framealpha=0.95,
                title=f'k-NN graph (k=10, {len(edge_segs)} edges)',
                title_fontsize=11)

ax.set_title('Chicago Station Graph: 50 Train + 8 Valid + 400 Auxiliary, k-NN (k=10)',
             fontsize=14, pad=10)

plt.tight_layout()
out_path = f'{OUT_DIR}/ppt_graph_structure.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")

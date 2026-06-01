"""Self-train pipeline visualized on the actual station graph.

Shows 6 small panels: R0 → R5, where each round adds K=40 pseudo stations.
Simulates the selection using the actual 3-metric greedy logic:
  confidence proxy = -dist to nearest train (closer = more confident)
  embeddings      = (lat, lon)
  valid_emb       = valid station coords

This is for visualization — the real selection uses model emb + neighbor error;
the visual pattern (spreading from train, diverse, biased toward valid) holds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.io as sio
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import contextily as cx

from data import fps_select_stations
from selftrain import greedy_select_3metric

DATA_DIR = '/home/hhz6461/Urban_prediction_Semi/data'
OUT = '/home/hhz6461/Urban_prediction_Semi/data/figures/ppt_selftrain_graph_rounds.png'

# ---- Load locations ----
d = sio.loadmat(f'{DATA_DIR}/Labeled_Finalized_new.mat')
loc_l = d['NodeLocation']  # (58, 2) = (lat, lon)
nL = loc_l.shape[0]
valid_idx = fps_select_stations(loc_l.T, n_select=8, seed=42)
train_idx = sorted(set(range(nL)) - set(valid_idx))

with h5py.File(f'{DATA_DIR}/Unlabeled_Finalized.mat', 'r') as f:
    loc_u_all = np.array(f['NodeLocation']).T
unl_sel = fps_select_stations(loc_u_all.T, n_select=400, seed=0)
loc_u = loc_u_all[unl_sel]  # (400, 2)

# Stack: nodes 0..nL-1 are labeled, nL..nL+400-1 are unlabeled
locs_all = np.vstack([loc_l, loc_u])   # (458, 2)
n_total = locs_all.shape[0]
n_labeled = nL
n_unl = loc_u.shape[0]

# Web mercator
def lonlat_to_mercator(lon, lat):
    R = 6378137.0
    return R * np.radians(lon), R * np.log(np.tan(np.pi/4 + np.radians(lat)/2))

all_x, all_y = lonlat_to_mercator(locs_all[:, 1], locs_all[:, 0])

# ---- Simulate the 3-metric greedy_select for 5 rounds ----
# Proxies:
train_coords = locs_all[train_idx]   # (50, 2)
valid_coords = locs_all[valid_idx]    # (8, 2)
emb_proxy = locs_all                  # (458, 2) — use coords as emb
valid_emb_proxy = valid_coords        # (8, 2)

# confidence: -mean distance to 5 nearest train (higher = closer to train = more confident)
confidence = np.full(n_total, -np.inf, dtype=np.float32)
for u in range(n_total):
    if u in train_idx or u in valid_idx:
        continue
    d2train = np.linalg.norm(locs_all[u] - train_coords, axis=1)
    confidence[u] = -np.sort(d2train)[:5].mean()

K_per_round = 40
N_ROUNDS = 5
cumulative = []
rounds_added = []  # list of K lists, one per round

for r in range(N_ROUNDS):
    new_sel = greedy_select_3metric(
        confidence=confidence,
        embeddings=emb_proxy,
        valid_emb=valid_emb_proxy,
        n_select=K_per_round,
        alpha_div=1.0, beta_rel=1.0, tau_quantile=0.5,
        already_selected=list(cumulative),
        train_idx=train_idx)
    rounds_added.append(new_sel)
    cumulative.extend(new_sel)
    print(f"R{r+1}: +{len(new_sel)}, cumulative={len(cumulative)}")

# ---- 6-panel figure ----
ROUND_COLORS = plt.cm.viridis(np.linspace(0.1, 0.85, N_ROUNDS))

fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
axes = axes.flatten()

# Global xy range with padding
PAD = 1200
xmin, xmax = all_x.min() - PAD, all_x.max() + PAD
ymin, ymax = all_y.min() - PAD, all_y.max() + PAD

TRAIN_COLOR = '#1f77b4'
VALID_COLOR = '#d62728'
UNL_COLOR   = '#cccccc'

panel_titles = ['R0 (initial)', 'After R1', 'After R2', 'After R3', 'After R4', 'After R5']
for p in range(6):
    ax = axes[p]
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    # Determine pseudo accumulated up to this panel
    if p == 0:
        accumulated = []
    else:
        accumulated = []
        for r in range(p):
            accumulated.extend(rounds_added[r])
    accumulated_set = set(accumulated)

    # 1. Unlabeled remaining (gray, faded)
    unl_remaining = [i for i in range(n_labeled, n_total) if i not in accumulated_set]
    ax.scatter(all_x[unl_remaining], all_y[unl_remaining],
               c=UNL_COLOR, s=10, alpha=0.65, zorder=1, edgecolor='none')

    # 2. Pseudo stations colored by round
    for r in range(p):
        idx_r = rounds_added[r]
        ax.scatter(all_x[idx_r], all_y[idx_r],
                   c=[ROUND_COLORS[r]], s=30, zorder=2,
                   edgecolor='white', linewidth=0.4,
                   label=f'+R{r+1}' if p == 5 else None)

    # 3. Train (fixed)
    ax.scatter(all_x[train_idx], all_y[train_idx],
               c=TRAIN_COLOR, s=55, zorder=3,
               edgecolor='white', linewidth=0.6)

    # 4. Valid (fixed, on top)
    ax.scatter(all_x[valid_idx], all_y[valid_idx],
               c=VALID_COLOR, s=130, marker='*', zorder=4,
               edgecolor='white', linewidth=0.7)

    cx.add_basemap(ax, crs='EPSG:3857', source=cx.providers.CartoDB.Positron,
                   attribution_size=5)
    ax.set_axis_off()

    n_pseudo_so_far = len(accumulated)
    ax.set_title(f'{panel_titles[p]}\npseudo cumulative = {n_pseudo_so_far}',
                 fontsize=11)

# Legend on bottom-right panel (last)
ax = axes[5]
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=TRAIN_COLOR, markersize=10, label='Train (50, fixed)'),
    plt.Line2D([0], [0], marker='*', color='w',
               markerfacecolor=VALID_COLOR, markersize=14, label='Valid (8, fixed)'),
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=UNL_COLOR, markersize=8, label='Unlabeled remaining'),
] + [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=ROUND_COLORS[r], markersize=10,
               label=f'Pseudo R{r+1} (+40)')
    for r in range(N_ROUNDS)
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.suptitle('Self-Train: 5 rounds × 40 pseudo stations selected by 3-metric greedy '
             '(confidence + diversity + valid-relevance)',
             fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(OUT, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUT}")

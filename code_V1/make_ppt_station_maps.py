"""Generate two PPT-quality station maps with Chicago street basemap.

Fig 1: 58 labeled stations on Chicago basemap
Fig 2: 58 labeled + 400 unlabeled stations on Chicago basemap

Usage:
    python code_V1/make_ppt_station_maps.py
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

DATA_DIR = '/home/hhz6461/Urban_prediction_Semi/data'
OUT_DIR  = '/home/hhz6461/Urban_prediction_Semi/data/figures'

# ---- Load 58 labeled stations ----
d = sio.loadmat(f'{DATA_DIR}/Labeled_Finalized_new.mat')
loc_l = d['NodeLocation']  # (58, 2) = (lat, lon)
nL = loc_l.shape[0]
print(f"Labeled: {nL} stations")

# ---- Load 2000 unlabeled, FPS select 400 (seed=0) ----
with h5py.File(f'{DATA_DIR}/Unlabeled_Finalized.mat', 'r') as f:
    loc_u_all = np.array(f['NodeLocation']).T  # (N_total, 2)
loc_u_fps = loc_u_all.T  # (2, N_total)
unl_sel = fps_select_stations(loc_u_fps, n_select=400, seed=0)
loc_u = loc_u_all[unl_sel]  # (400, 2)
print(f"Unlabeled selected: {len(unl_sel)}")

# Web Mercator conversion (EPSG:4326 -> EPSG:3857)
def lonlat_to_mercator(lon, lat):
    R = 6378137.0
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y

lab_x, lab_y = lonlat_to_mercator(loc_l[:, 1], loc_l[:, 0])
unl_x, unl_y = lonlat_to_mercator(loc_u[:, 1], loc_u[:, 0])

LABEL_COLOR = '#d62728'   # red
UNL_COLOR   = '#1f77b4'   # blue

# Padding around the data extent (in mercator meters)
PAD = 1500

def add_basemap(ax, source=cx.providers.CartoDB.Positron):
    cx.add_basemap(ax, crs='EPSG:3857', source=source, attribution_size=7)

# =========================================================
# Fig 1: 58 labeled only
# =========================================================
fig, ax = plt.subplots(figsize=(8, 8.5), dpi=150)
ax.scatter(lab_x, lab_y, c=LABEL_COLOR, s=80,
           edgecolor='white', linewidth=1.2, zorder=3,
           label=f'Labeled stations ({nL})')

xmin, xmax = lab_x.min() - PAD, lab_x.max() + PAD
ymin, ymax = lab_y.min() - PAD, lab_y.max() + PAD
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

add_basemap(ax)
ax.set_axis_off()
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
ax.set_title('Chicago Weather Stations (Labeled)', fontsize=14, pad=10)

plt.tight_layout()
fig1_path = f'{OUT_DIR}/ppt_labeled_stations_map.png'
plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")

# =========================================================
# Fig 2: 58 labeled + 400 unlabeled
# =========================================================
fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
ax.scatter(unl_x, unl_y, c=UNL_COLOR, s=14, alpha=0.55, zorder=2,
           edgecolor='none',
           label=f'Auxiliary points ({len(loc_u)})')
ax.scatter(lab_x, lab_y, c=LABEL_COLOR, s=80,
           edgecolor='white', linewidth=1.2, zorder=3,
           label=f'Labeled stations ({nL})')

xmin = min(lab_x.min(), unl_x.min()) - PAD
xmax = max(lab_x.max(), unl_x.max()) + PAD
ymin = min(lab_y.min(), unl_y.min()) - PAD
ymax = max(lab_y.max(), unl_y.max()) + PAD
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

add_basemap(ax)
ax.set_axis_off()
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
ax.set_title('Chicago Station Graph (Labeled + Auxiliary)', fontsize=14, pad=10)

plt.tight_layout()
fig2_path = f'{OUT_DIR}/ppt_labeled_plus_unlabeled_map.png'
plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

print("\nDone:")
print(f"  1. {fig1_path}")
print(f"  2. {fig2_path}")

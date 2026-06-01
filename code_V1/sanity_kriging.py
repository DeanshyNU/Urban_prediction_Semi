"""
Kriging accuracy sanity check.

Goal:
  Compute IDW (inverse-distance weighted) kriging at the 8 valid stations using
  the 50 train labeled stations as sources. Compare RMSE against model 13860
  baseline (0.0453).

Decision rule:
  - Kriging RMSE < 0.06 (≈2°C):    kriging-pseudo experiment B viable
  - Kriging RMSE 0.06–0.10:        only as soft anchor with low λ
  - Kriging RMSE > 0.10:            external pseudo not viable, skip experiment B

Run with:
  cd /home/hhz6461/Urban_prediction_Semi
  conda activate urban
  mkdir -p logs
  python -u code_V1/sanity_kriging.py 2>&1 | tee logs/kriging_sanity.log
"""

import os
import sys
import numpy as np
from pathlib import Path

# allow imports from code_V1
sys.path.insert(0, str(Path(__file__).parent))
import data as v1_data

# -------------------------------------------------
# Config (mirror 13860 baseline exactly)
# -------------------------------------------------
V1_DATA_DIR = os.environ.get('V1_DATA_DIR', '/home/hhz6461/Urban_prediction_Semi/data')
V1_N_VALID_STATIONS = 8
V1_SPATIAL_SEED = 42       # same as 13860 / data.py default
V1_KNN_K_KRIGING = 10      # how many nearest train stations to use for kriging at each valid

print("=" * 70)
print("Kriging accuracy sanity check (IDW from train labeled to valid 8)")
print("=" * 70)
print(f"data_dir         = {V1_DATA_DIR}")
print(f"n_valid_stations = {V1_N_VALID_STATIONS}")
print(f"spatial_seed     = {V1_SPATIAL_SEED}")
print(f"k_kriging        = {V1_KNN_K_KRIGING}")

# -------------------------------------------------
# Step 1: load V2 labeled
# -------------------------------------------------
print("\n[Step 1] Loading V2 labeled data...")
labeled = v1_data.load_v2_labeled(V1_DATA_DIR)
targets_norm = labeled["targets_norm"]    # (T, N)
locations    = labeled["locations"]        # (2, N) — note: (2, N), not (N, 2)
tgt_min      = labeled["tgt_min"]
tgt_scl      = labeled["tgt_scl"]
T, n_labeled = targets_norm.shape
print(f"  T={T}, n_labeled={n_labeled}, locations shape={locations.shape}")
print(f"  target_norm range=[{targets_norm.min():.4f}, {targets_norm.max():.4f}], "
      f"tgt_scl_C={tgt_scl:.4f}")

# -------------------------------------------------
# Step 2: spatial split (same as data.py: FPS-based)
# -------------------------------------------------
print(f"\n[Step 2] Spatial split via FPS (seed={V1_SPATIAL_SEED}, n_valid={V1_N_VALID_STATIONS})...")
valid_idx = v1_data.fps_select_stations(locations, V1_N_VALID_STATIONS, seed=V1_SPATIAL_SEED)
train_idx = sorted([i for i in range(n_labeled) if i not in valid_idx])
print(f"  valid_idx ({len(valid_idx)}): {valid_idx}")
print(f"  train_idx: {len(train_idx)} stations")

# -------------------------------------------------
# Step 3: IDW kriging at each valid station
# -------------------------------------------------
print(f"\n[Step 3] IDW kriging at {len(valid_idx)} valid using {V1_KNN_K_KRIGING} nearest train...")

# locations is (2, N), transpose to (N, 2) for indexing
loc_T = locations.T   # (N, 2)
train_loc  = loc_T[train_idx]    # (50, 2)
valid_loc  = loc_T[valid_idx]    # (8, 2)
train_targets = targets_norm[:, train_idx]  # (T, 50)
valid_truth   = targets_norm[:, valid_idx]  # (T, 8)

valid_pred = np.zeros((T, len(valid_idx)), dtype=np.float32)
neighbor_dist_stats = []   # for reporting how close train neighbors are

for i, vi in enumerate(valid_idx):
    # Distance from this valid station to all 50 train stations
    d = np.sqrt(((valid_loc[i] - train_loc) ** 2).sum(axis=1))    # (50,)
    # Take k nearest train
    nearest = np.argsort(d)[:V1_KNN_K_KRIGING]
    d_nearest = d[nearest]
    neighbor_dist_stats.append((d_nearest.min(), d_nearest.mean(), d_nearest.max()))
    # IDW weights
    w = 1.0 / (d_nearest + 1e-6)
    w = w / w.sum()
    # Weighted average
    valid_pred[:, i] = train_targets[:, nearest] @ w

# -------------------------------------------------
# Step 4: Compute RMSE
# -------------------------------------------------
print(f"\n[Step 4] Compute RMSE in normalized space + °C...")
per_station_rmse = np.sqrt(((valid_pred - valid_truth) ** 2).mean(axis=0))   # (8,)
overall_rmse     = np.sqrt(((valid_pred - valid_truth) ** 2).mean())
overall_rmse_C   = overall_rmse * tgt_scl

# Also per-station MBE / MAE
per_station_mbe = (valid_pred - valid_truth).mean(axis=0)
per_station_mae = np.abs(valid_pred - valid_truth).mean(axis=0)

print(f"\nPer-valid-station kriging error:")
print(f"  {'station':>8} {'rmse':>8} {'°C':>6} {'mbe':>8} {'mae':>8} {'k-min':>7} {'k-mean':>8} {'k-max':>7}")
for i, vi in enumerate(valid_idx):
    d_min, d_mean, d_max = neighbor_dist_stats[i]
    print(f"  {vi:>8} {per_station_rmse[i]:>8.4f} {per_station_rmse[i]*tgt_scl:>6.2f} "
          f"{per_station_mbe[i]:>+8.4f} {per_station_mae[i]:>8.4f} "
          f"{d_min:>7.4f} {d_mean:>8.4f} {d_max:>7.4f}")

print()
print(f"  ===== OVERALL kriging valid RMSE = {overall_rmse:.4f} (≈ {overall_rmse_C:.3f}°C) =====")

# -------------------------------------------------
# Step 5: Decision summary
# -------------------------------------------------
MODEL_BASELINE = 0.0453   # 13860 V2_semi_spatial

print()
print("=" * 70)
print("DECISION SUMMARY")
print("=" * 70)
print(f"Model 13860 baseline valid RMSE = {MODEL_BASELINE:.4f} (≈ 1.57°C)")
print(f"Kriging                valid RMSE = {overall_rmse:.4f} (≈ {overall_rmse_C:.3f}°C)")
print(f"Δ (kriging − model) = {overall_rmse - MODEL_BASELINE:+.4f}")
print()
if overall_rmse < 0.06:
    print("→ Kriging RMSE < 0.06: external-pseudo experiment B is VIABLE.")
    print("  Recommend: hybrid pseudo (0.7×model + 0.3×kriging) as soft anchor.")
elif overall_rmse < 0.10:
    print("→ Kriging RMSE in [0.06, 0.10]: marginal. Use ONLY as low-λ anchor.")
    print("  Recommend: pseudo = 0.9×model + 0.1×kriging, expect Δ ≈ 0 ~ −0.001.")
else:
    print("→ Kriging RMSE > 0.10: external pseudo NOT viable.")
    print("  Skip experiment B. Stick with self-distillation only (A1, A2).")
print("=" * 70)

# Outliers
worst_i = int(np.argmax(per_station_rmse))
best_i  = int(np.argmin(per_station_rmse))
print()
print(f"Worst valid station: idx={valid_idx[worst_i]} RMSE={per_station_rmse[worst_i]:.4f} "
      f"(°C={per_station_rmse[worst_i]*tgt_scl:.2f}) → likely far from train neighbors")
print(f"Best  valid station: idx={valid_idx[best_i]} RMSE={per_station_rmse[best_i]:.4f} "
      f"(°C={per_station_rmse[best_i]*tgt_scl:.2f}) → likely well-surrounded by train")
print(f"Per-station spread: std={per_station_rmse.std():.4f}, "
      f"range=[{per_station_rmse.min():.4f}, {per_station_rmse.max():.4f}]")

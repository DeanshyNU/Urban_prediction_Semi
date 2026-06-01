"""
Verify the Physics Decomposition hypothesis:
    Δ(i, t) = T_true(i, t) - WRF_T2(i, t) = Δ_global(t) + Δ_local(i, t)

Question: does Δ have a strong "global temporal component"
that is shared across all 58 labeled stations?

Method:
  1. Compute Δ matrix (T, N) = (3672, 58)
  2. PCA across stations (treat each station as a feature, time as samples)
  3. Check PC1 variance ratio
     - If PC1 explains >30% → Δ_global is a real, strong signal
     - If PC1 explains <10% → no shared temporal structure, decomposition won't help
  4. Visualize PC1 temporal pattern (should show diurnal/seasonal structure)
  5. Analyze Δ_local = Δ - PC1_reconstructed (residual, supposed to be spatial-only)

Output: figures + printed statistics
"""
import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_PATH = '/home/hhz6461/Urban_prediction_Semi/code/data/Labeled_Finalized_new.mat'
OUT_DIR = '/home/hhz6461/Urban_prediction_Semi/analysis/figures'
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    """Load T_true, WRF_T2, station locations from labeled mat file."""
    print(f"Loading {DATA_PATH}...")
    labeled = sio.loadmat(DATA_PATH)

    AoT = labeled['AoT_filled']  # (3672, 58) or (58, 3672)
    if AoT.shape[1] > AoT.shape[0]:
        AoT = AoT.T
    T_true = AoT  # (T, N) in °C

    WRFMat = labeled['WRFMat']  # could be (3672, 63, 58) or transposed
    if WRFMat.shape[0] != AoT.shape[0]:
        WRFMat = np.transpose(WRFMat, (2, 1, 0))
    # channel 0 is T2 in Kelvin → convert to °C
    WRF_T2 = WRFMat[:, 0, :] - 273.15  # (T, N)

    Map_ = np.squeeze(labeled['Map'])  # (58, 2) lat/lon

    print(f"  T_true: shape={T_true.shape}, range=[{T_true.min():.1f}, {T_true.max():.1f}]°C")
    print(f"  WRF_T2: shape={WRF_T2.shape}, range=[{WRF_T2.min():.1f}, {WRF_T2.max():.1f}]°C")
    print(f"  Map:    shape={Map_.shape}")
    return T_true, WRF_T2, Map_


def compute_delta(T_true, WRF_T2):
    """Δ(i, t) = T_true(i, t) - WRF_T2(i, t). Returns (T, N)."""
    Delta = T_true - WRF_T2
    print("\n--- Delta statistics (raw) ---")
    print(f"  shape: {Delta.shape}")
    print(f"  mean:  {Delta.mean():.3f} °C")
    print(f"  std:   {Delta.std():.3f} °C")
    print(f"  range: [{Delta.min():.2f}, {Delta.max():.2f}] °C")
    print(f"  Per-station mean across time (first 5): {Delta.mean(axis=0)[:5]}")
    print(f"  Per-timestep mean across stations (first 5): {Delta.mean(axis=1)[:5]}")
    return Delta


def pca_decomposition(Delta, n_components=10):
    """
    Run PCA on Delta (T, N).
      - Each row = one timestep, a 58-dim vector
      - PC1 direction v1 is the 'common mode' across stations
      - score on PC1 = 'global temporal signal' Δ_global(t)
    """
    # Center Delta per station (remove per-station mean bias)
    Delta_centered = Delta - Delta.mean(axis=0, keepdims=True)
    per_station_mean = Delta.mean(axis=0)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Delta_centered)  # (T, n_comp)
    components = pca.components_                # (n_comp, N)
    var_ratio = pca.explained_variance_ratio_   # (n_comp,)

    print("\n--- PCA on centered Delta ---")
    print(f"  Explained variance ratio (top {n_components}):")
    for k in range(n_components):
        cum = var_ratio[:k+1].sum()
        print(f"    PC{k+1:2d}: {var_ratio[k]*100:6.2f}%   (cumulative: {cum*100:6.2f}%)")

    # Verdict
    pc1 = var_ratio[0] * 100
    pc12 = var_ratio[:2].sum() * 100
    print("\n--- Verdict ---")
    if pc1 > 30:
        print(f"  PC1 explains {pc1:.1f}% of variance -> STRONG global component. ")
        print(f"  => Physics decomposition hypothesis is WELL-SUPPORTED.")
    elif pc1 > 15:
        print(f"  PC1 explains {pc1:.1f}% of variance -> MODERATE global component.")
        print(f"  => Physics decomposition MAY help.")
    else:
        print(f"  PC1 explains only {pc1:.1f}% of variance -> WEAK global component.")
        print(f"  => Physics decomposition unlikely to help much.")
    print(f"  PC1+PC2 together: {pc12:.1f}%")
    return pca, scores, components, var_ratio, per_station_mean, Delta_centered


def analyze_pc1_temporal(scores, components, var_ratio):
    """Look at PC1 score vs time. Does it show diurnal/seasonal pattern?"""
    pc1_score = scores[:, 0]
    pc1_dir = components[0]  # (N,) loading on each station
    pc1_std = pc1_score.std()
    pc1_dir_uniformity = pc1_dir.std() / (np.abs(pc1_dir.mean()) + 1e-8)

    print("\n--- PC1 temporal signal ---")
    print(f"  PC1 score std across time: {pc1_std:.3f}")
    print(f"  PC1 loading on stations:  mean={pc1_dir.mean():.3f}, std={pc1_dir.std():.3f}")
    print(f"  PC1 loading abs-mean={np.abs(pc1_dir).mean():.3f}")
    print(f"    (if loading is roughly uniform across stations,")
    print(f"     PC1 is a 'true global' signal, not regional)")

    # If loadings are all positive and similar magnitude, it's a "global mode"
    same_sign_ratio = (pc1_dir > 0).sum() / len(pc1_dir) if pc1_dir.mean() > 0 else (pc1_dir < 0).sum() / len(pc1_dir)
    print(f"  Fraction of stations with same sign as mean: {same_sign_ratio*100:.1f}%")
    if same_sign_ratio > 0.85:
        print(f"  -> PC1 is a 'whole-city-uniform' mode (ideal for Δ_global).")
    elif same_sign_ratio > 0.7:
        print(f"  -> PC1 is largely uniform but has some regional variation.")
    else:
        print(f"  -> PC1 splits stations into positive/negative groups -> NOT uniform global.")

    return pc1_score, pc1_dir


def analyze_residual(Delta_centered, scores, components, n_remove=1):
    """Compute residual after removing first k PCs. Check if residual is purely spatial (less temporal)."""
    # Reconstruct using first k components
    reconstructed = scores[:, :n_remove] @ components[:n_remove]
    residual = Delta_centered - reconstructed

    orig_var = Delta_centered.var()
    res_var = residual.var()

    # Temporal variance: for each station, variance across time
    temporal_var_orig = Delta_centered.var(axis=0).mean()
    temporal_var_res = residual.var(axis=0).mean()

    # Spatial variance: for each time step, variance across stations
    spatial_var_orig = Delta_centered.var(axis=1).mean()
    spatial_var_res = residual.var(axis=1).mean()

    print(f"\n--- After removing top {n_remove} PC(s) ---")
    print(f"  Total variance:   {orig_var:.4f} -> {res_var:.4f} ({(1-res_var/orig_var)*100:.1f}% removed)")
    print(f"  Temporal var (avg per station):  {temporal_var_orig:.4f} -> {temporal_var_res:.4f} ({(1-temporal_var_res/temporal_var_orig)*100:.1f}% removed)")
    print(f"  Spatial var  (avg per timestep): {spatial_var_orig:.4f} -> {spatial_var_res:.4f} ({(1-spatial_var_res/spatial_var_orig)*100:.1f}% removed)")
    print(f"  Expected: temporal var should drop a lot, spatial should stay.")
    print(f"    If temporal drops >> spatial -> decomposition is clean.")
    return residual


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------
def plot_variance_explained(var_ratio, out_path):
    n = len(var_ratio)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, n+1), var_ratio * 100)
    axes[0].axhline(30, color='r', ls='--', label='30% threshold')
    axes[0].set_xlabel('PC #'); axes[0].set_ylabel('% variance')
    axes[0].set_title('Per-PC variance explained')
    axes[0].legend()

    axes[1].plot(range(1, n+1), np.cumsum(var_ratio) * 100, 'o-')
    axes[1].axhline(80, color='g', ls='--', label='80%')
    axes[1].set_xlabel('PC #'); axes[1].set_ylabel('cumulative % variance')
    axes[1].set_title('Cumulative variance explained')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out_path}")


def plot_pc1_timeseries(pc1_score, out_path, window=168):
    T = len(pc1_score)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].plot(pc1_score, lw=0.4, alpha=0.7)
    axes[0].set_title(f'PC1 score across all timesteps (T={T})')
    axes[0].set_xlabel('timestep'); axes[0].set_ylabel('PC1 score')
    axes[0].grid(alpha=0.3)

    # First 2 weeks (if hourly, 168h = 1 week)
    first = pc1_score[:min(window*2, T)]
    axes[1].plot(first, '.-', lw=0.8, ms=2)
    axes[1].set_title(f'PC1 score - first {len(first)} timesteps (zoom, look for diurnal pattern)')
    axes[1].set_xlabel('timestep'); axes[1].set_ylabel('PC1 score')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out_path}")


def plot_pc1_spatial_loading(pc1_dir, Map_, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sc = ax.scatter(Map_[:, 1], Map_[:, 0], c=pc1_dir, cmap='RdBu_r', s=80,
                    vmin=-np.abs(pc1_dir).max(), vmax=np.abs(pc1_dir).max(),
                    edgecolors='k', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='PC1 loading')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('PC1 loading per station\n(uniform color -> global mode; split -> regional mode)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out_path}")


def plot_diurnal_pattern(pc1_score, out_path):
    """Assume hourly data. Fold by hour-of-day (mod 24) and show average pattern."""
    T = len(pc1_score)
    hours = np.arange(T) % 24
    avg_by_hour = np.array([pc1_score[hours == h].mean() for h in range(24)])
    std_by_hour = np.array([pc1_score[hours == h].std() for h in range(24)])

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.errorbar(range(24), avg_by_hour, yerr=std_by_hour, marker='o', capsize=3)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel('Hour of day (assumed hourly timesteps)')
    ax.set_ylabel('PC1 score (mean ± std)')
    ax.set_title('Diurnal pattern of PC1 score\n(clear wave -> Δ_global has diurnal bias)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out_path}")
    print(f"  Diurnal amplitude (max-min of hourly mean): {avg_by_hour.max()-avg_by_hour.min():.3f}")


def plot_delta_heatmap(Delta, residual, out_path, n_time=500):
    """Compare Delta vs residual (after removing PC1). Residual should look less 'striped'."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    vmax = np.abs(Delta[:n_time]).max()
    im0 = axes[0].imshow(Delta[:n_time].T, aspect='auto', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[0].set_ylabel('Station index')
    axes[0].set_title(f'Original Δ = T_true - WRF_T2 (first {n_time} timesteps)\nVertical stripes = shared temporal signal')
    plt.colorbar(im0, ax=axes[0], label='Δ (°C)')

    vmax_res = np.abs(residual[:n_time]).max()
    im1 = axes[1].imshow(residual[:n_time].T, aspect='auto', cmap='RdBu_r',
                         vmin=-vmax_res, vmax=vmax_res, interpolation='nearest')
    axes[1].set_xlabel('timestep')
    axes[1].set_ylabel('Station index')
    axes[1].set_title(f'Residual Δ_local = Δ - PC1 (first {n_time} timesteps)\nStripes gone -> clean local signal')
    plt.colorbar(im1, ax=axes[1], label='Δ_local (°C)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out_path}")


def plot_single_station(Delta_centered, reconstructed_pc1, residual, station_idx, out_path, n_time=500):
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(Delta_centered[:n_time, station_idx], lw=0.8, alpha=0.7, label='Δ (centered)')
    ax.plot(reconstructed_pc1[:n_time, station_idx], lw=1.0, label='Δ_global (PC1 reconstruction)')
    ax.plot(residual[:n_time, station_idx], lw=0.6, alpha=0.7, label='Δ_local (residual)')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('timestep'); ax.set_ylabel('°C')
    ax.set_title(f'Station {station_idx}: decomposition over time (first {n_time} timesteps)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    T_true, WRF_T2, Map_ = load_data()
    Delta = compute_delta(T_true, WRF_T2)

    # PCA
    pca, scores, components, var_ratio, per_station_mean, Delta_centered = pca_decomposition(Delta, n_components=10)

    # PC1 analysis
    pc1_score, pc1_dir = analyze_pc1_temporal(scores, components, var_ratio)

    # Residual
    residual = analyze_residual(Delta_centered, scores, components, n_remove=1)
    reconstructed_pc1 = scores[:, :1] @ components[:1]

    # Plots
    print("\n--- Generating figures ---")
    plot_variance_explained(var_ratio, os.path.join(OUT_DIR, '01_variance_explained.png'))
    plot_pc1_timeseries(pc1_score,       os.path.join(OUT_DIR, '02_pc1_timeseries.png'))
    plot_pc1_spatial_loading(pc1_dir, Map_, os.path.join(OUT_DIR, '03_pc1_spatial_loading.png'))
    plot_diurnal_pattern(pc1_score,      os.path.join(OUT_DIR, '04_pc1_diurnal.png'))
    plot_delta_heatmap(Delta_centered, residual, os.path.join(OUT_DIR, '05_delta_vs_residual_heatmap.png'))
    plot_single_station(Delta_centered, reconstructed_pc1, residual, station_idx=0,
                        out_path=os.path.join(OUT_DIR, '06_station0_decomposition.png'))
    plot_single_station(Delta_centered, reconstructed_pc1, residual, station_idx=25,
                        out_path=os.path.join(OUT_DIR, '07_station25_decomposition.png'))

    # Final verdict summary
    print("\n" + "="*70)
    print("FINAL VERDICT — Physics decomposition feasibility")
    print("="*70)
    print(f"  PC1 variance ratio:                   {var_ratio[0]*100:.1f}%")
    print(f"  PC1+PC2 cumulative:                    {var_ratio[:2].sum()*100:.1f}%")
    print(f"  PC1 loading uniformity (same-sign %): {(np.sign(pc1_dir)==np.sign(pc1_dir.mean())).mean()*100:.1f}%")
    print(f"  Temporal variance removed by PC1:     {(1 - (Delta_centered - reconstructed_pc1).var(axis=0).mean() / Delta_centered.var(axis=0).mean())*100:.1f}%")

    if var_ratio[0] > 0.30 and (np.sign(pc1_dir)==np.sign(pc1_dir.mean())).mean() > 0.85:
        print("\n  ✓ Hypothesis SUPPORTED: implement physics decomposition.")
    elif var_ratio[0] > 0.15:
        print("\n  ? Hypothesis PARTIAL: decomposition may help but not guaranteed.")
    else:
        print("\n  ✗ Hypothesis NOT SUPPORTED: global component too weak.")
    print("="*70)


if __name__ == '__main__':
    main()

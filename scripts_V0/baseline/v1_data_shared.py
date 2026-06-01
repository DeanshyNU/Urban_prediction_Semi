"""
Shared V1 data loading utilities — used by BOTH V1 sup and V1 semi to ensure
identical feature construction for labeled and unlabeled stations.

Feature schema (1298 dim per (station, timestep)):
  - WRF window      : 5 timesteps × 54 features = 270  (window=2, dynamic)
  - Raw geo @ t     : 20 (16 UF morph + 3 CLMS + 1 dist; CLMS dynamic, others static)
  - Geo embed       : 1008 (7 channels × 12×12 avg pool, static per station)

Drops:
  - Target column (col 0 in V1 raw)
  - 4 aux columns (cols 55-58: hour/month/year/station_id) — for fair sup vs semi comparison
  - V1's 8th FeaturePatch channel — to match V2's UrbanFeatureMat 7 channels
"""
import numpy as np
import torch
import h5py
import mat73


T_END = 2948   # V1's timestep coverage
WINDOW = 2     # past + future window
N_WRF = 54
RAW_GEO_DIM = 20
GEO_EMB_CHANNELS = 7
GEO_EMB_POOL = 12          # 12×12 spatial after avg pool
GEO_EMB_DIM = GEO_EMB_CHANNELS * GEO_EMB_POOL * GEO_EMB_POOL   # 1008


def avg_pool_7ch(raw_patches):
    """raw_patches: (N, C≥7, H, W) → (N, 7×12×12 = 1008) flat."""
    if raw_patches.shape[1] > GEO_EMB_CHANNELS:
        raw_patches = raw_patches[:, :GEO_EMB_CHANNELS, :, :]
    raw_t = torch.from_numpy(np.nan_to_num(raw_patches, nan=0.0)).float()
    pool = torch.nn.AdaptiveAvgPool2d((GEO_EMB_POOL, GEO_EMB_POOL))
    pooled = pool(raw_t).reshape(raw_t.shape[0], -1)             # (N, 1008)
    return pooled.numpy()


def normalize_by_stats(arr, ref_min, ref_max):
    """Normalize to [0,1] using given per-feature min/max."""
    rng = ref_max - ref_min
    rng = np.where(rng == 0, 1.0, rng)
    return ((arr - ref_min) / rng).astype(np.float32)


# =========================================================================
# Load V1 labeled (68 stations)
# =========================================================================
def load_v1_labeled(data_dir):
    """
    Returns:
        wrf       : (68, T, 54)  WRF dynamic features (already pre-normalized in StationMat)
        raw_geo   : (68, T, 20)  16 UF morph + 3 CLMS + 1 dist (mostly static, CLMS dynamic)
        targets   : (68, T)      temperature
        locations : (2, 68)
        geo_emb   : (68, 1008)   7-channel × 12×12 avg pool, RAW values (need normalization)
    """
    sm = mat73.loadmat(f'{data_dir}/GNN_N1_StationMat.mat')
    raw = sm['StationMat_se_fill']   # mat73 returns (T, F, N)
    if raw.shape == (T_END, 79, 68):
        raw = np.transpose(raw, (2, 0, 1))                       # (68, T, 79)
    elif raw.shape == (68, 79, T_END):
        raw = np.transpose(raw, (0, 2, 1))                       # (68, T, 79)
    else:
        raise RuntimeError(f"Unexpected StationMat shape {raw.shape}")

    targets = raw[:, :, 0]                                        # (68, T)
    wrf = raw[:, :, 1:55]                                         # (68, T, 54)
    raw_geo = raw[:, :, 59:79]                                    # (68, T, 20)

    locations = sm['Locations']
    if locations.shape[0] != 2:
        locations = locations.T

    # Geo embed from FeaturePatch_401 (8 channels → take first 7)
    fp = mat73.loadmat(f'{data_dir}/FeaturePatch_401.mat')['FeatureMat_zeros']
    if fp.shape == (68, 8, 401, 401):
        pass
    elif fp.shape == (8, 401, 401, 68):
        fp = np.transpose(fp, (3, 0, 1, 2))
    elif fp.shape == (401, 401, 8, 68):
        fp = np.transpose(fp, (3, 2, 0, 1))
    else:
        raise RuntimeError(f"Unexpected FeaturePatch_401 shape {fp.shape}")
    geo_emb = avg_pool_7ch(fp)                                    # (68, 1008)

    print(f"  [V1 labeled] WRF: {wrf.shape}, raw_geo: {raw_geo.shape}, "
          f"target: {targets.shape}, locs: {locations.shape}, geo_emb: {geo_emb.shape}")
    return (wrf.astype(np.float32),
            raw_geo.astype(np.float32),
            targets.astype(np.float32),
            np.asarray(locations, dtype=np.float32),
            geo_emb.astype(np.float32))


# =========================================================================
# Load V1 unlabeled (V2-source, V1-aligned)
# =========================================================================
def load_v1_unlabeled(data_dir, n_select, fps_seed=0):
    """
    Returns SAME schema as load_v1_labeled():
        wrf, raw_geo, locations, geo_emb (no targets — unlabeled)
    Internally:
        WRF V1 (54) = WRF V2 first 45 + sqrt(WindX² + WindY²)
        raw_geo (20) = 16 UF morph + 3 CLMS + 1 dist
        geo_emb (1008) = 7 channels of UrbanFeatureMat → 12×12 pool
    """
    f = h5py.File(f'{data_dir}/Unlabeled_Finalized.mat', 'r')
    wrf  = f['WRFMat'][:]                       # (2000, 63, 6624)
    clms = f['CLMSMat'][:]                      # (2000, 3, 6624)
    uf   = f['UrbanFeature'][:]                 # (17, 2000)
    locs = f['NodeLocation'][:]                 # (2, 2000)
    ufm  = f['UrbanFeatureMat'][:]              # (2000, 7, 401, 401)
    f.close()

    # Truncate timesteps
    wrf  = wrf[:, :, :T_END]
    clms = clms[:, :, :T_END]

    if locs.shape[0] != 2:
        locs = locs.T
    n_total = locs.shape[1]

    # FPS select n_select
    rng = np.random.default_rng(fps_seed)
    coords = locs.T
    first = int(rng.integers(0, n_total))
    selected = [first]
    dists = np.linalg.norm(coords - coords[first], axis=1)
    while len(selected) < n_select:
        i = int(np.argmax(dists))
        selected.append(i)
        d_new = np.linalg.norm(coords - coords[i], axis=1)
        dists = np.minimum(dists, d_new)
    selected = sorted(selected)
    print(f"  [V1 unlabeled] FPS selected {len(selected)} from {n_total}, seed={fps_seed}")

    wrf_s  = wrf[selected, :, :]                # (n_u, 63, T)
    clms_s = clms[selected, :, :]               # (n_u, 3,  T)
    uf_s   = uf[:, selected]                    # (17, n_u)
    locs_s = locs[:, selected]                  # (2,  n_u)
    ufm_s  = ufm[selected, :, :, :]             # (n_u, 7, 401, 401)

    # WRF V1-aligned: first 45 (Tair/Tskin/Tsoil/Humid/Irrad × 9) + |Wind| × 9 = 54
    wrf_no_wind = wrf_s[:, :45, :]                                # (n_u, 45, T)
    wind_x = wrf_s[:, 45:54, :]
    wind_y = wrf_s[:, 54:63, :]
    wind_mag = np.sqrt(wind_x ** 2 + wind_y ** 2)                 # (n_u, 9, T)
    wrf_v1 = np.concatenate([wrf_no_wind, wind_mag], axis=1)       # (n_u, 54, T)
    wrf_v1 = np.transpose(wrf_v1, (0, 2, 1))                       # (n_u, T, 54)

    # raw_geo (20): 16 UF morph + 3 CLMS dynamic + 1 dist
    uf_morph = uf_s[:16, :]                                        # (16, n_u)
    uf_dist  = uf_s[16:17, :]                                      # (1,  n_u)
    n_u = len(selected); T = wrf_v1.shape[1]
    uf_morph_t = np.broadcast_to(uf_morph.T[:, None, :], (n_u, T, 16)).copy()
    uf_dist_t  = np.broadcast_to(uf_dist.T[:, None, :],  (n_u, T, 1)).copy()
    clms_t     = np.transpose(clms_s, (0, 2, 1))                   # (n_u, T, 3)
    raw_geo = np.concatenate([uf_morph_t, clms_t, uf_dist_t], axis=2)   # (n_u, T, 20)

    # Geo embed: 7-channel UFM → 12×12 pool
    geo_emb = avg_pool_7ch(ufm_s)                                  # (n_u, 1008)

    print(f"  [V1 unlabeled] WRF: {wrf_v1.shape}, raw_geo: {raw_geo.shape}, "
          f"locs: {locs_s.shape}, geo_emb: {geo_emb.shape}")
    return (wrf_v1.astype(np.float32),
            raw_geo.astype(np.float32),
            np.asarray(locs_s, dtype=np.float32),
            geo_emb.astype(np.float32))


# =========================================================================
# Normalize unlabeled to V1 labeled scale
# =========================================================================
def normalize_unlabeled(wrf_l, raw_geo_l, geo_emb_l,
                        wrf_u, raw_geo_u, geo_emb_u):
    """
    V1 labeled is pre-normalized to [0,1] in StationMat — but geo_emb_l from
    FeaturePatch is RAW, needs normalization too.

    For unlabeled:
      - WRF: V2 raw Kelvin → normalize to [0,1] using V2's own (unlabeled) per-feature min/max
      - raw_geo: UF/CLMS/dist mostly already in [0,1] from V2; just clip
      - geo_emb: combined V1+V2 per-channel min/max for joint normalization

    Returns normalized (wrf_l, raw_geo_l, geo_emb_l_norm, wrf_u_norm, raw_geo_u_norm, geo_emb_u_norm)
    where labeled WRF and labeled raw_geo are unchanged (already normalized).
    """
    # ---- WRF: V1 labeled is already [0,1]; V2 raw → use V2's own min/max ----
    wrf_u_flat = wrf_u.reshape(-1, wrf_u.shape[-1])
    wrf_u_min = wrf_u_flat.min(axis=0)
    wrf_u_max = wrf_u_flat.max(axis=0)
    wrf_u_norm = normalize_by_stats(wrf_u, wrf_u_min, wrf_u_max)

    # ---- raw_geo: V1 labeled already [0,1]; V2 unlabeled → V2 own min/max ----
    raw_geo_u_flat = raw_geo_u.reshape(-1, raw_geo_u.shape[-1])
    rg_min = raw_geo_u_flat.min(axis=0)
    rg_max = raw_geo_u_flat.max(axis=0)
    raw_geo_u_norm = normalize_by_stats(raw_geo_u, rg_min, rg_max)

    # ---- geo_emb: jointly normalize V1+V2 per-feature [min, max] ----
    combined = np.vstack([geo_emb_l, geo_emb_u])
    g_min = combined.min(axis=0)
    g_max = combined.max(axis=0)
    geo_emb_l_norm = normalize_by_stats(geo_emb_l, g_min, g_max)
    geo_emb_u_norm = normalize_by_stats(geo_emb_u, g_min, g_max)

    print(f"  [Normalize] WRF unl: [{wrf_u_norm.min():.3f}, {wrf_u_norm.max():.3f}]")
    print(f"  [Normalize] raw_geo unl: [{raw_geo_u_norm.min():.3f}, {raw_geo_u_norm.max():.3f}]")
    print(f"  [Normalize] geo_emb lab/unl: "
          f"[{geo_emb_l_norm.min():.3f}, {geo_emb_l_norm.max():.3f}] / "
          f"[{geo_emb_u_norm.min():.3f}, {geo_emb_u_norm.max():.3f}]")
    return wrf_l, raw_geo_l, geo_emb_l_norm, wrf_u_norm, raw_geo_u_norm, geo_emb_u_norm


# =========================================================================
# Build per-(station, t) feature vector with WRF window + static raw_geo + geo_emb
# =========================================================================
def build_features_per_t(wrf, raw_geo, geo_emb, t, window=WINDOW):
    """
    Args:
        wrf      : (N, T, 54)   normalized
        raw_geo  : (N, T, 20)   normalized (CLMS dynamic, others static)
        geo_emb  : (N, 1008)    normalized, station-static
    Returns:
        x_t      : (N, 1298) — features for all stations at timestep t
                    Layout: [WRF window (270), raw_geo at t (20), geo_emb (1008)]
    """
    N = wrf.shape[0]
    wrf_window = wrf[:, t - window:t + window + 1, :]            # (N, 5, 54)
    wrf_flat = wrf_window.reshape(N, -1)                          # (N, 270)
    rg_t = raw_geo[:, t, :]                                       # (N, 20)
    return np.concatenate([wrf_flat, rg_t, geo_emb], axis=-1)     # (N, 1298)

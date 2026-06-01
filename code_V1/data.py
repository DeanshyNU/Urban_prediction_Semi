"""
V1 data loading — extended from original_code/data.py to support:
  - 3 validation modes: random / sequential / spatial   (env: V1_VAL_MODE)
  - n_unlabeled = 0 (supervised) or > 0 (semi-supervised, V2-source unlabeled aligned to V1 schema)

Feature schema (1302 dim, sup ≡ semi):
  WRF window (5×54=270) + station_aux (4: hour/month/year/station_id) + raw geo (20) + geoEmbed (1008)

Faithful to original_code which keeps all 4 station-aux cols (raw cols 55-58):
  - col 55 = hour / 23                  (shared across stations at same t)
  - col 56 = month / 12                 (shared)
  - col 57 = year (2018 raw value)      (shared)
  - col 58 = station_id (1..98 raw)     (per-station unique)

For semi-supervised V2 unlabeled:
  - unlabeled WRF: V2 raw 63ch → V1-aligned 54ch (45 thermal + 9 |Wind|)
  - unlabeled hour/month/year: copy V1's value at same n (they're shared across all stations)
  - unlabeled station_id: 100, 101, ..., 99 + n_unlabeled (avoids V1's 1..98 range)
  - unlabeled raw geo: 16 UF morph + 3 CLMS + 1 dist = 20
  - unlabeled geoEmbed: V2 UrbanFeatureMat (already 7-ch) × 12×12 avg pool
  - graph: k-NN k=10 on combined (lat, lon)

Time alignment between V1 labeled and V2 unlabeled:
  V1 t=0 = 2018-05-01 02:00 (hour=2, derived from V1 col 55 at t=0)
  V2 t=0 = 2018-05-01 00:00 (per dataset metadata)
  → V1 t=n corresponds to V2 t=(n+V1_OFFSET) where V1_OFFSET=2.
  V2 unlabeled WRF/CLMS are sliced [V1_OFFSET : V1_OFFSET + V1_T] to align physical times.
"""
import numpy as np
import os, mat73, h5py
import torch
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA


# =========================================================================
# V1 labeled geo features (FeaturePatch_401, 8-ch → drop 5th → 7-ch)
# =========================================================================
# 加载 V1 labeled 的 7-channel UFM,归一化后做 12×12 avg pool → 每站 1008 维 GeoEmbed
def genGeoFeatures(path, geoMethod='average', poolSize=15, nCompPCA=40):
    """加载 V1 labeled 的地理嵌入(GeoEmbed)。

    从 `FeaturePatch_401.mat` 读 8 通道 401×401 的城市形态学栅格,**丢掉第 5 通道**
    (water-related,V1 站全是陆基),per-feature 归一化到 [0,1],然后做
    `AdaptiveAvgPool2d((poolSize, poolSize))` → 每站 7 × poolSize² = 1008 维(默认 12×12)。

    Returns: (geoFeatures (nStations, 7*poolSize²) tensor, off, scl, nStations)
    """
    _raw = mat73.loadmat(f'{path}/FeaturePatch_401.mat')['FeatureMat_zeros']
    # Remove 5th dimension because all stations are land based
    _idx = np.arange(_raw.shape[2])
    _idx = np.delete(_idx, 4)
    _raw = _raw[:, :, _idx, :]
    _imageSize, _, _nFeatures, _nStations = _raw.shape

    if geoMethod == 'average':
        _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        _min, _max = np.min(_norm, axis=1), np.max(_norm, axis=1)
        _max[_max == 0] = 1e-5
        _off = _min
        _scl = _max - _min
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        _avgPool = torch.nn.AdaptiveAvgPool2d((poolSize, poolSize))
        _geoFeatures = _avgPool(_geoFeatures).reshape(_nStations, -1)

    elif geoMethod == 'pca':
        _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        _off, _scl = np.mean(_norm, axis=1), np.std(_norm, axis=1)
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        _geo2D = _geoFeatures.reshape(_nStations, -1)
        _pca = PCA(n_components=nCompPCA)
        _geoFeatures = _pca.fit_transform(_geo2D)
        _geoFeatures = (_geoFeatures - _geoFeatures.min()) / (_geoFeatures.max() - _geoFeatures.min())
        _geoFeatures = torch.FloatTensor(_geoFeatures)

    return _geoFeatures, _off, _scl, _nStations


# =========================================================================
# V2-source unlabeled data, FPS-selected and aligned to V1 schema
# =========================================================================
V1_TIMESTEP_OFFSET = 2   # V1 t=0 = V2 t=2 (V1 starts 02:00, V2 starts 00:00 on 2018-05-01)


# 加载 V2 unlabeled 数据,做时间偏移 + FPS 选站 + 转 V1 schema(54ch WRF + 20 raw_geo + 1008 geoEmbed)
def load_unlabeled_v1_aligned(path, n_select, T_end, geo_pool_size=12, fps_seed=0,
                              v1_offset=V1_TIMESTEP_OFFSET):
    """加载 V2 unlabeled 数据,转成 V1 schema。

    从 `Unlabeled_Finalized.mat` 读 2000 站的 WRF/CLMS/UF/UFM,做以下处理:
    1) **时间对齐**:V2 timestep 取 [v1_offset, v1_offset+T_end),让输出 index n 对应 V1 t=n
       的同一物理时刻(V1 起 02:00 / V2 起 00:00,offset=2)。
    2) **FPS 选站**:在 (lat, lon) 上做 farthest-point sampling,选 n_select 个站,seed 确定。
    3) **WRF 对齐 V1 schema**:V2 的 63 通道(45 thermal + WindX 9 + WindY 9)→ V1 的 54
       通道(45 thermal + |Wind| 9)。Per-feature min-max 归一化到 [0,1]。
    4) **raw_geo**:UF 16 形态学 + CLMS 3 动态 + dist 1 = 20 维,per-feature 归一化。
    5) **geoEmbed**:UrbanFeatureMat 7 通道经 12×12 avg pool → 1008 维,per-feature 归一化。

    Returns: (wrf (n,T,54), raw_geo (n,T,20), geo_emb (n,1008), locs (2,n))
    """
    f = h5py.File(f'{path}/Unlabeled_Finalized.mat', 'r')
    wrf  = f['WRFMat'][:]              # (2000, 63, 6624)
    clms = f['CLMSMat'][:]             # (2000, 3, 6624)
    uf   = f['UrbanFeature'][:]        # (17, 2000)
    locs = f['NodeLocation'][:]        # (2, 2000)
    ufm  = f['UrbanFeatureMat'][:]     # (2000, 7, 401, 401)
    f.close()

    # Truncate V2 to V1's T, with V1_OFFSET so that V2 t=(v1_offset+n) ↔ V1 t=n in physical time
    wrf  = wrf[:, :, v1_offset : v1_offset + T_end]
    clms = clms[:, :, v1_offset : v1_offset + T_end]
    print(f"  [V2 truncate] using V2 t={v1_offset}..{v1_offset + T_end - 1} "
          f"to align with V1 t=0..{T_end - 1} (offset={v1_offset})")

    if locs.shape[0] != 2:
        locs = locs.T
    n_total = locs.shape[1]

    # FPS select n_select stations on (lat, lon)
    rng = np.random.default_rng(fps_seed)
    coords = locs.T  # (2000, 2)
    first = int(rng.integers(0, n_total))
    selected = [first]
    dists = np.linalg.norm(coords - coords[first], axis=1)
    while len(selected) < n_select:
        i = int(np.argmax(dists))
        selected.append(i)
        d_new = np.linalg.norm(coords - coords[i], axis=1)
        dists = np.minimum(dists, d_new)
    selected = sorted(selected)
    print(f"  [Unlabeled FPS] selected {len(selected)} from {n_total}, seed={fps_seed}")

    wrf_s  = wrf[selected, :, :]         # (n, 63, T)
    clms_s = clms[selected, :, :]        # (n, 3,  T)
    uf_s   = uf[:, selected]             # (17, n)
    locs_s = locs[:, selected]           # (2,  n)
    ufm_s  = ufm[selected, :, :, :]      # (n, 7, 401, 401)

    # WRF V1-aligned: first 45 thermal + |Wind| × 9 = 54
    wrf_no_wind = wrf_s[:, :45, :]
    wind_x = wrf_s[:, 45:54, :]
    wind_y = wrf_s[:, 54:63, :]
    wind_mag = np.sqrt(wind_x ** 2 + wind_y ** 2)
    wrf_v1 = np.concatenate([wrf_no_wind, wind_mag], axis=1)   # (n, 54, T)
    wrf_v1 = np.transpose(wrf_v1, (0, 2, 1))                    # (n, T, 54)

    # Per-feature min-max normalization to [0,1] (V2 raw is in Kelvin etc.)
    flat = wrf_v1.reshape(-1, wrf_v1.shape[-1])
    wmin, wmax = flat.min(axis=0), flat.max(axis=0)
    wrng = np.where(wmax - wmin > 0, wmax - wmin, 1.0)
    wrf_v1 = ((wrf_v1 - wmin) / wrng).astype(np.float32)

    # raw_geo (20): 16 UF morph + 3 CLMS dynamic + 1 dist (matches V1 cols 59-78)
    n = len(selected)
    T = wrf_v1.shape[1]
    uf_morph = uf_s[:16, :]                                          # (16, n)
    uf_dist  = uf_s[16:17, :]                                        # (1,  n)
    uf_morph_t = np.broadcast_to(uf_morph.T[:, None, :], (n, T, 16)).copy()
    uf_dist_t  = np.broadcast_to(uf_dist.T[:, None, :],  (n, T, 1)).copy()
    clms_t = np.transpose(clms_s, (0, 2, 1))                         # (n, T, 3)
    raw_geo = np.concatenate([uf_morph_t, clms_t, uf_dist_t], axis=2)  # (n, T, 20)

    # Normalize raw_geo per-feature to [0,1]
    flat = raw_geo.reshape(-1, raw_geo.shape[-1])
    rmin, rmax = flat.min(axis=0), flat.max(axis=0)
    rrng = np.where(rmax - rmin > 0, rmax - rmin, 1.0)
    raw_geo = ((raw_geo - rmin) / rrng).astype(np.float32)

    # GeoEmbed: 7-ch × pool×pool avg, normalize per-channel
    ufm_t = torch.from_numpy(np.nan_to_num(ufm_s, nan=0.0)).float()    # (n, 7, 401, 401)
    pool = torch.nn.AdaptiveAvgPool2d((geo_pool_size, geo_pool_size))
    geo_emb = pool(ufm_t).reshape(n, -1).numpy()                       # (n, 7*pool²)

    # Normalize geo_emb per-feature
    g_min, g_max = geo_emb.min(axis=0), geo_emb.max(axis=0)
    g_rng = np.where(g_max - g_min > 0, g_max - g_min, 1.0)
    geo_emb = ((geo_emb - g_min) / g_rng).astype(np.float32)

    print(f"  [Unlabeled V1-aligned] WRF: {wrf_v1.shape}, raw_geo: {raw_geo.shape}, "
          f"geo_emb: {geo_emb.shape}, locs: {locs_s.shape}")
    return wrf_v1, raw_geo, geo_emb, np.asarray(locs_s, dtype=np.float32)


# =========================================================================
# Spatial validation: FPS select held-out stations
# =========================================================================
# FPS 在 (lat, lon) 上选 n_select 个最分散的站点(用于 spatial validation 的 hold-out 集)
def fps_select_stations(node_locations, n_select, seed=42):
    """Farthest-Point Sampling 在 (lat, lon) 上选 n_select 个最分散的站点。

    用于 spatial validation 的 hold-out 站点选择。从随机起点开始,每步选距已选集
    最远的点。同样的 seed 保证选出的站集合**确定**。
    输入 (2, N),输出 sorted list of n_select 个 station indices。
    """
    rng = np.random.default_rng(seed)
    coords = node_locations.T
    n_total = coords.shape[0]
    first = int(rng.integers(0, n_total))
    selected = [first]
    dists = np.linalg.norm(coords - coords[first], axis=1)
    for _ in range(n_select - 1):
        i = int(np.argmax(dists))
        selected.append(i)
        d_new = np.linalg.norm(coords - coords[i], axis=1)
        dists = np.minimum(dists, d_new)
    return sorted(selected)


# 把 spatial split 站点分布画成散点图(灰=unlabeled / 红=train / 蓝=valid),保存到 png
def visualize_spatial_split(node_locations, train_idx, valid_idx, save_path,
                            unlabeled_locations=None):
    """画出 spatial split 的站点分布图(灰=unlabeled,红=train labeled,蓝=valid labeled)。

    保存到 save_path。失败时静默跳过(不影响训练)。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    coords = node_locations.T if node_locations.shape[0] == 2 else node_locations
    fig, ax = plt.subplots(figsize=(8, 8))
    if unlabeled_locations is not None:
        u = unlabeled_locations.T if unlabeled_locations.shape[0] == 2 else unlabeled_locations
        ax.scatter(u[:, 1], u[:, 0], c='lightgray', s=8, alpha=0.4, label=f'Unlabeled ({len(u)})')
    ax.scatter(coords[train_idx, 1], coords[train_idx, 0], c='red', s=60, marker='^',
               zorder=5, label=f'Train labeled ({len(train_idx)})')
    ax.scatter(coords[valid_idx, 1], coords[valid_idx, 0], c='blue', s=120, marker='*',
               zorder=6, label=f'Valid labeled ({len(valid_idx)})')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# k-NN graph on combined (labeled + unlabeled) locations
# =========================================================================
# 用 (lat, lon) 距离建 k-NN 对称邻接图,边权 = 1 - normalized_dist(semi 模式专用)
def build_knn_adj(locations, k=10):
    """k-NN 图构建(基于地理距离)。

    给每个节点找最近 k 个邻居,边权 = 1 - normalized_dist;再 max(A, Aᵀ) 让矩阵对称。
    用于 semi 模式下 V1 labeled + V2 unlabeled 的合并图。

    输入 locations: (2, N),输出 Adj: (N, N) symmetric float32。
    """
    N = locations.shape[1]
    coords = locations.T
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    dmax = dist.max() + 1e-8
    dist_norm = dist / dmax
    Adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        idxs = np.argsort(dist[i])[1:k + 1]
        for j in idxs:
            Adj[i, j] = 1.0 - dist_norm[i, j]
    Adj = np.maximum(Adj, Adj.T)
    return Adj


def build_feature_sim_knn_adj(static_features, k=10):
    """Feature similarity k-NN graph(Experiment D,2026-05-31)。

    用 static features(UF + GeoEmb,每站时间不变)算 cosine similarity,
    取每个节点的 k 个最相似邻居 → 边权 = cosine_sim ∈ [0, 1]。
    对称化:max(A, A^T)。

    输入 static_features: (N, D) — 每站的静态特征向量(已 [0,1] 归一化)
    输出 Adj: (N, N) symmetric float32
    """
    N = static_features.shape[0]
    # Normalize for cosine similarity
    norms = np.linalg.norm(static_features, axis=1, keepdims=True) + 1e-8
    feat_normed = static_features / norms                              # (N, D)
    sim = feat_normed @ feat_normed.T                                  # (N, N) cosine
    sim = np.clip(sim, 0.0, 1.0).astype(np.float32)                     # 避免轻微负值

    Adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        # 排除自己:取 sim[i] 最大的 k+1 个(包含自己),去掉自己
        idxs = np.argsort(-sim[i])[1:k + 1]   # descending,跳过 idx 0 (self)
        for j in idxs:
            Adj[i, j] = sim[i, j]
    Adj = np.maximum(Adj, Adj.T)
    return Adj


# =========================================================================
# V2 数据加载 (V1_DATASET=V2 模式)
# =========================================================================
# 加载 V2 labeled (Labeled_Finalized_new.mat,58 stations,T=3672 = Period 1 only)
def load_v2_labeled(path, geo_pool_size=12):
    """加载 V2 labeled 全部数据并归一化。

    返回 dict 含:
      wrf (T, N, 63), clms (T, N, 3), uf (N, 17), geo_emb (N, 1008),
      targets_norm (T, N) 全局归一化, locations (2, N), similarity (N, 10),
      tgt_min, tgt_scl(用于反归一化回 °C)。
    """
    import scipy.io as sio
    d = sio.loadmat(f'{path}/Labeled_Finalized_new.mat')
    wrf = np.transpose(d['WRFMat'].astype(np.float32), (0, 2, 1))    # (T, N, 63)
    clms = np.transpose(d['CLMSMat'].astype(np.float32), (0, 2, 1))  # (T, N, 3)
    uf = d['UrbanFeature'].astype(np.float32).copy()                  # (N, 17)
    ufm = d['UrbanFeatureMat'].astype(np.float32)                     # (401, 401, 7, N)
    targets = d['AoT_filled'].astype(np.float32)                      # (T, N) raw °C
    locs = d['NodeLocation'].astype(np.float32)                       # (N, 2)
    sim = d['SimilarityMat'].astype(np.float32)                       # (N, 10)

    if locs.shape[0] != 2:
        locs = locs.T   # (2, N)

    # NaN handle: UF cols 0-15 → 0, col 16 → median
    n_nan_uf = int(np.isnan(uf).sum())
    if n_nan_uf > 0:
        uf[:, :16] = np.nan_to_num(uf[:, :16], nan=0.0)
        col16 = uf[:, 16]
        if np.any(np.isnan(col16)):
            uf[np.isnan(col16), 16] = np.nanmedian(col16)
    n_nan_ufm = int(np.isnan(ufm).sum())
    ufm = np.nan_to_num(ufm, nan=0.0)
    print(f"  [V2 labeled] NaN: UF={n_nan_uf} UFM={n_nan_ufm} (filled with 0/median)")

    # UFM (401, 401, 7, N) → (N, 7, 401, 401) → 12×12 avg pool → (N, 1008)
    ufm = np.transpose(ufm, (3, 2, 0, 1))
    pool = torch.nn.AdaptiveAvgPool2d((geo_pool_size, geo_pool_size))
    geo_emb = pool(torch.from_numpy(ufm).float()).reshape(ufm.shape[0], -1).numpy()

    # Target: global min-max normalize across all stations + timesteps
    tgt_min = float(targets.min()); tgt_max = float(targets.max())
    tgt_scl = max(tgt_max - tgt_min, 1e-8)
    targets_norm = ((targets - tgt_min) / tgt_scl).astype(np.float32)
    print(f"  [V2 labeled] target raw °C range=[{tgt_min:.2f}, {tgt_max:.2f}], "
          f"scl={tgt_scl:.2f}, normalized to [0, 1]")

    # ===== EAD: wrf_t2 必须在 _normf 之前抠出来,放到 target 归一化空间 =====
    # WRF channel 0 is Tair (Kelvin)
    wrf_t2_celsius = wrf[:, :, 0] - 273.15                            # (T, N) raw °C
    wrf_t2_norm = ((wrf_t2_celsius - tgt_min) / tgt_scl).astype(np.float32)  # (T, N) target 空间
    print(f"  [V2 labeled] wrf_t2 °C range=[{wrf_t2_celsius.min():.2f}, {wrf_t2_celsius.max():.2f}], "
          f"normalized range=[{wrf_t2_norm.min():.4f}, {wrf_t2_norm.max():.4f}]")

    # Per-feature min-max normalize (last axis) — features get this AFTER wrf_t2 extracted
    def _normf(x):
        flat = x.reshape(-1, x.shape[-1])
        mn, mx = flat.min(0), flat.max(0)
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        return ((x - mn) / rng).astype(np.float32)
    wrf = _normf(wrf); clms = _normf(clms); uf = _normf(uf); geo_emb = _normf(geo_emb)

    return dict(wrf=wrf, clms=clms, uf=uf, geo_emb=geo_emb,
                targets_norm=targets_norm, targets_raw=targets,
                wrf_t2_norm=wrf_t2_norm,                              # 新增:供 EAD 用
                locations=locs, similarity=sim,
                tgt_min=tgt_min, tgt_scl=tgt_scl)


# V2 模式时间特征:基于已知起点 2018-05-01 00:00 计算每个 t 的 (hour, month, year)
V2_BASE_DATE_STR = '2018-05-01 00:00'
def v2_compute_time_features(T):
    """对 V2 timestep [0, T) 计算 (hour/23, month/12, year) 标量。

    返回 (T, 3) 数组,与 V2 t=0 = 2018-05-01 00:00 锚定。
    用于给 V2 schema 加 station_aux 时间 3 列(broadcast 到所有 station)。
    """
    from datetime import datetime, timedelta
    base = datetime.strptime(V2_BASE_DATE_STR, '%Y-%m-%d %H:%M')
    aux = np.empty((T, 3), dtype=np.float32)
    for t in range(T):
        dt = base + timedelta(hours=int(t))
        aux[t, 0] = dt.hour / 23.0     # hour
        aux[t, 1] = dt.month / 12.0    # month
        aux[t, 2] = float(dt.year)     # year (raw)
    return aux


# 加载 V2 unlabeled,V2 schema (63ch WRF,无 |Wind| merge,无时间偏移)
def load_v2_unlabeled(path, n_select, T_end, geo_pool_size=12, fps_seed=0,
                      tgt_min=None, tgt_scl=None):
    """V2 mode 专用 unlabeled 加载 —— V2 raw schema,直接对齐 V2 labeled 的 t=0..T_end-1。

    Args:
        tgt_min, tgt_scl:V2 labeled 的全局 target 归一化参数,如提供则同时返回 wrf_t2_norm
                         (供 EAD 用,unlabeled WRF channel 0 在 target 空间的值)
    返回 dict 含:wrf (n, T, 63), clms (n, T, 3), uf (n, 17), geo_emb (n, 1008), locations (2, n),
                 wrf_t2_norm (n, T) 当 tgt_min/scl 提供时。
    """
    f = h5py.File(f'{path}/Unlabeled_Finalized.mat', 'r')
    wrf = f['WRFMat'][:]; clms = f['CLMSMat'][:]
    uf = f['UrbanFeature'][:]; locs = f['NodeLocation'][:]
    ufm = f['UrbanFeatureMat'][:]
    f.close()

    # No offset for V2 mode (V2 labeled 起 t=0 = V2 unlabeled 起 t=0 = 2018-05-01 00:00)
    wrf = wrf[:, :, :T_end]
    clms = clms[:, :, :T_end]
    print(f"  [V2 unlabeled] truncated to t=0..{T_end - 1} (no offset, same start as V2 labeled)")

    if locs.shape[0] != 2:
        locs = locs.T

    # FPS select (same algorithm as V1 path)
    rng = np.random.default_rng(fps_seed)
    coords = locs.T
    n_total = locs.shape[1]
    first = int(rng.integers(0, n_total))
    selected = [first]
    dists = np.linalg.norm(coords - coords[first], axis=1)
    while len(selected) < n_select:
        i = int(np.argmax(dists))
        selected.append(i)
        d_new = np.linalg.norm(coords - coords[i], axis=1)
        dists = np.minimum(dists, d_new)
    selected = sorted(selected)
    print(f"  [V2 unlabeled FPS] selected {len(selected)} from {n_total}, seed={fps_seed}")

    wrf_s  = wrf[selected]                           # (n, 63, T)
    clms_s = clms[selected]                          # (n, 3, T)
    uf_s   = uf[:, selected].T.copy()                # (n, 17)
    locs_s = locs[:, selected]                       # (2, n)
    ufm_s  = np.nan_to_num(ufm[selected], nan=0.0)   # (n, 7, 401, 401)

    # NaN handle UF (same as V2 labeled)
    if np.isnan(uf_s).any():
        uf_s[:, :16] = np.nan_to_num(uf_s[:, :16], nan=0.0)
        if np.any(np.isnan(uf_s[:, 16])):
            uf_s[np.isnan(uf_s[:, 16]), 16] = np.nanmedian(uf_s[:, 16])

    # Reshape to (T, n, F) → (n, T, F)
    wrf_v2 = np.transpose(wrf_s, (0, 2, 1)).astype(np.float32)    # (n, T, 63)
    clms_v2 = np.transpose(clms_s, (0, 2, 1)).astype(np.float32)  # (n, T, 3)

    # ===== EAD: 在 _normf 之前抠 wrf_t2 (Tair channel 0,Kelvin → °C → target 归一化) =====
    wrf_t2_norm = None
    if tgt_min is not None and tgt_scl is not None:
        wrf_t2_celsius = wrf_v2[:, :, 0] - 273.15            # (n, T) raw °C
        wrf_t2_norm = ((wrf_t2_celsius - tgt_min) / tgt_scl).astype(np.float32)
        print(f"  [V2 unlabeled] wrf_t2 °C range=[{wrf_t2_celsius.min():.2f}, {wrf_t2_celsius.max():.2f}], "
              f"normalized=[{wrf_t2_norm.min():.4f}, {wrf_t2_norm.max():.4f}]")

    # GeoEmbed
    pool = torch.nn.AdaptiveAvgPool2d((geo_pool_size, geo_pool_size))
    geo_emb = pool(torch.from_numpy(ufm_s).float()).reshape(ufm_s.shape[0], -1).numpy()

    # Per-feature normalize
    def _normf(x):
        flat = x.reshape(-1, x.shape[-1])
        mn, mx = flat.min(0), flat.max(0)
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        return ((x - mn) / rng).astype(np.float32)
    wrf_v2 = _normf(wrf_v2); clms_v2 = _normf(clms_v2)
    uf_n = _normf(uf_s); geo_emb = _normf(geo_emb)

    out = dict(wrf=wrf_v2, clms=clms_v2, uf=uf_n, geo_emb=geo_emb,
               locations=np.asarray(locs_s, dtype=np.float32))
    if wrf_t2_norm is not None:
        out['wrf_t2_norm'] = wrf_t2_norm
    return out


# EAD: 用图加权把 train 站的 β kriging 到 valid + unlabeled 站
def kriging_beta(adj, beta_train, train_idx, n_total):
    """β_hat[i] = (Σ_{j ∈ train} Adj[i,j] × β_train[j]) / (Σ_{j ∈ train} Adj[i,j])

    train 站直接保留原值。无 train 邻居的孤立节点 fallback 到 0。
    """
    W = adj[:, train_idx]                          # (n_total, n_train) 边权 to train
    num = W @ beta_train                           # (n_total,)
    denom = W.sum(axis=1)                          # (n_total,)
    isolated = denom < 1e-12
    n_iso = int(isolated.sum())
    denom = np.where(isolated, 1.0, denom)         # 防 div0
    beta_hat = (num / denom).astype(np.float32)
    beta_hat[isolated] = 0.0                       # 孤立节点回到 0
    beta_hat[train_idx] = beta_train               # train 站精确(覆盖 kriging 结果)
    if n_iso > 0:
        print(f"  [EAD/kriging] {n_iso} isolated nodes (no train neighbors), fallback β=0")
    return beta_hat


# EAD: 计算时间锚 α_t 和空间锚 β_i (后者通过 kriging 传到 valid+unlabeled)
def compute_ead_anchors(targets_norm, wrf_t2_norm_labeled, adj, train_mask_in_total,
                        nL, n_total, use_alpha, use_beta,
                        beta_mode='kriging', station_aux_full=None):
    """计算 EAD 锚点。**严格只用 train labeled 站的 target**(防 spatial leak)。

    Args:
        targets_norm:    (T, nL) labeled target,target 空间归一化
        wrf_t2_norm_labeled: (T, nL) labeled wrf_t2,target 空间
        adj:             (n_total, n_total) graph (用于 kriging mode)
        train_mask_in_total: (n_total,) bool,True at train station 位置
        nL:              labeled 站总数(掩码必须在 [0, nL))
        n_total:         labeled + unlabeled 总数
        use_alpha, use_beta: 开关
        beta_mode:       'kriging' (V0 法,图加权平均)| 'mlp' (V1 EAD-Plus,β = MLP(aux))
        station_aux_full: (n_total, aux_dim) 仅 mlp 模式必须;aux = UF + GeoEmb + locations

    Returns:
        alpha_T  (T,):       时间锚 (use_alpha=False 时为 0)
        beta_hat (n_total,): 空间锚 kriged 或 MLP-pred 到所有节点 (use_beta=False 时为 0)
    """
    T = targets_norm.shape[0]
    train_idx_in_total = np.where(train_mask_in_total)[0]    # train 站索引(都在 [0, nL))
    assert (train_idx_in_total < nL).all(), \
        f"[ERR/EAD] train mask 越界:max idx={train_idx_in_total.max()} but nL={nL}"
    assert len(train_idx_in_total) > 0, "[ERR/EAD] no train stations in mask"

    # Δ 在 labeled 上有定义(只有 labeled 有 target)
    delta_labeled = (targets_norm - wrf_t2_norm_labeled).astype(np.float32)   # (T, nL)
    delta_train = delta_labeled[:, train_idx_in_total]                         # (T, n_train)

    if use_alpha:
        alpha_T = delta_train.mean(axis=1).astype(np.float32)                  # (T,)
    else:
        alpha_T = np.zeros(T, dtype=np.float32)

    if use_beta:
        beta_train = (delta_train - alpha_T[:, None]).mean(axis=0).astype(np.float32)  # (n_train,)
        if beta_mode == 'kriging':
            beta_hat = kriging_beta(adj, beta_train, train_idx_in_total, n_total)  # (n_total,)
        elif beta_mode == 'mlp':
            assert station_aux_full is not None, "[ERR/EAD-Plus] beta_mode='mlp' requires station_aux_full"
            beta_hat = _fit_beta_mlp(station_aux_full, beta_train, train_idx_in_total, n_total)
        else:
            raise ValueError(f"unknown beta_mode={beta_mode}")
    else:
        beta_hat = np.zeros(n_total, dtype=np.float32)

    print(f"  [EAD] computed: use_alpha={bool(use_alpha)}, use_beta={bool(use_beta)}, "
          f"beta_mode={beta_mode if use_beta else '-'}, n_train_stations={len(train_idx_in_total)}")
    if use_alpha:
        print(f"  [EAD] α_t stats: mean={alpha_T.mean():.4f}, std={alpha_T.std():.4f}, "
              f"range=[{alpha_T.min():.4f}, {alpha_T.max():.4f}]")
    if use_beta:
        print(f"  [EAD] β_train stats: mean={beta_train.mean():.4f}, std={beta_train.std():.4f}, "
              f"range=[{beta_train.min():.4f}, {beta_train.max():.4f}]")
        print(f"  [EAD] β_hat (after {beta_mode} to all {n_total} nodes): mean={beta_hat.mean():.4f}, "
              f"std={beta_hat.std():.4f}, range=[{beta_hat.min():.4f}, {beta_hat.max():.4f}]")
        if beta_mode == 'kriging':
            assert np.allclose(beta_hat[train_idx_in_total], beta_train, atol=1e-5), \
                "[ERR/EAD] kriging 后 train 站 β 不等于 direct β"
        # mlp 模式 train 站 β 是 MLP fit 的近似,不一定 = beta_train,所以不 assert
    return alpha_T, beta_hat


def _fit_beta_mlp(station_aux_full, beta_train, train_idx_in_total, n_total,
                   hidden=64, epochs=500, lr=1e-3, weight_decay=1e-3):
    """训练小 MLP 拟合 β_train,然后预测全节点 β。

    Args:
        station_aux_full: (n_total, aux_dim) 每站静态特征(UF + GeoEmb + locations)
        beta_train:       (n_train,) 50 train 站的真 β
        train_idx_in_total: indices in [0, n_total)
        n_total:          总节点数
        hidden / epochs / lr / weight_decay: MLP 超参

    Returns:
        beta_hat: (n_total,) numpy
    """
    aux_dim = station_aux_full.shape[1]
    print(f"  [EAD-Plus β MLP] aux_dim={aux_dim}, n_train={len(beta_train)}, hidden={hidden}, epochs={epochs}")

    # 构 MLP
    mlp = torch.nn.Sequential(
        torch.nn.Linear(aux_dim, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1)
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    X_all = torch.from_numpy(station_aux_full).float()
    X_train = X_all[train_idx_in_total]                                  # (n_train, aux_dim)
    y_train = torch.from_numpy(beta_train).float().view(-1, 1)           # (n_train, 1)

    mlp.train()
    losses = []
    for ep in range(epochs):
        pred = mlp(X_train)
        loss = ((pred - y_train) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"  [EAD-Plus β MLP] loss progression: ep0={losses[0]:.6f}, "
          f"ep{epochs//4}={losses[epochs//4]:.6f}, "
          f"ep{epochs//2}={losses[epochs//2]:.6f}, ep{epochs-1}={losses[-1]:.6f}")

    # Predict β for all stations
    mlp.eval()
    with torch.no_grad():
        beta_hat = mlp(X_all).numpy().squeeze()    # (n_total,)
    # Sanity check on train stations
    train_pred_diff = np.abs(beta_hat[train_idx_in_total] - beta_train).mean()
    print(f"  [EAD-Plus β MLP] train 站预测 vs 真值 mean abs diff: {train_pred_diff:.4f} "
          f"(完美 fit 应 ≈ 0,有 weight_decay 所以非零)")

    return beta_hat.astype(np.float32)


# V2 sup graph:Similarity (corrcoef of SimilarityMat) × exp(-dist/max_dist),阈值过滤
def build_v2_adj(locations, similarity, thres=0.1):
    """V2 SimilarityMat-based 图。Faithful Yu et al. (2024) / V0 V2 实现。

    locations: (2, N),similarity: (N, 10) 每站的相似度向量。
    """
    n = locations.shape[1]
    coords = locations.T
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    sim_corr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            cov = np.corrcoef(similarity[i], similarity[j])
            r = cov[0, 1] if not np.isnan(cov[0, 1]) else 0.0
            sim_corr[i, j] = max(r, 0.0)

    max_dist = dist.max() + 1e-8
    Adj = sim_corr * np.exp(-dist / max_dist)
    Adj[Adj < thres] = 0.0
    np.fill_diagonal(Adj, 0.0)
    return Adj.astype(np.float32)


# =========================================================================
# Main dataGen — supports val_mode × (sup / semi),V1 / V2 dataset
# =========================================================================
# 主入口:按 env vars 构造 sup/semi × random/sequential/spatial 的 PyG DataLoader 和 metadata
def dataGen(dataParam, path, nTrn=0.75, predMode=False):
    """构建 PyG 数据集(总入口)。Dispatcher:V1_DATASET=V2(默认,与 run.py 一致)→ V2 路径,V1 → V1 路径。"""
    dataset_kind = os.environ.get('V1_DATASET', 'V2').upper()
    assert dataset_kind in ('V1', 'V2'), f"bad V1_DATASET={dataset_kind}"
    if dataset_kind == 'V2':
        return _dataGen_V2(dataParam, path, nTrn, predMode)
    return _dataGen_V1(dataParam, path, nTrn, predMode)


# V1 数据路径(原始实现,faithful original_code)
def _dataGen_V1(dataParam, path, nTrn=0.75, predMode=False):
    """V1 数据路径:GNN_N1_StationMat 68 站 + V2-source-V1-aligned 400 unlabeled。

    流程:
    1) 加载 V1 labeled 的 features + targets + 各种归一化的 GeoEmbed;
    2) 若 V1_N_UNLABELED > 0,加载 V2 unlabeled(已对齐时间);
    3) 构图(sup 用 V1 AJM sim×dist,semi 用 k-NN k=10);
    4) 构 label_mask(spatial 模式有 train_mask / valid_mask 两套,其它共用一套);
    5) 逐 timestep 生成 PyG `Data` 对象,组成 _dataset;
    6) 按 val_mode 切 train / valid,封装 DataLoader;
    7) 大量 [DEBUG/...] 打印 + assert,确保 shape / mask / 数值合理。
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']

    # ----- env knobs -----
    val_mode    = os.environ.get('V1_VAL_MODE', 'spatial').lower()
    n_unlabeled = int(os.environ.get('V1_N_UNLABELED', '0'))
    fps_seed    = int(os.environ.get('V1_FPS_SEED', '0'))
    n_valid_spatial = int(os.environ.get('V1_N_VALID_STATIONS', '10'))   # V1 default 10 of 68
    spatial_seed    = int(os.environ.get('V1_SPATIAL_SEED', '42'))
    temporal_frac   = float(os.environ.get('V1_TEMPORAL_FRAC', '0.8'))
    knn_k       = int(os.environ.get('V1_KNN_K', '10'))
    output_dir  = os.environ.get('V1_OUTPUT_DIR', '.')
    assert val_mode in ('random', 'sequential', 'spatial'), f"bad V1_VAL_MODE={val_mode}"

    print(f"[data.py] val_mode={val_mode}, n_unlabeled={n_unlabeled}, fps_seed={fps_seed}")

    # ----- V1 labeled geo features (existing original_code logic) -----
    _geoFeatures, _off, _scl, _nStations = genGeoFeatures(
        path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])

    # ----- V1 labeled features + targets -----
    _raw = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    _raw = np.transpose(_raw, (0, 2, 1))   # (T, N, F)
    nCFDFeats = 54
    nStationFeats = 4
    cfdIdx        = np.arange(nCFDFeats)
    stationFeatIdx = np.arange(nCFDFeats, nCFDFeats + nStationFeats)
    rawGeoFeatIdx  = np.arange(nCFDFeats + nStationFeats, _raw.shape[-1] - 1)
    features_l = _raw[:, :, 1:]                      # (T, N, F-1)
    targets_l  = _raw[:, :, 0]                       # (T, N)
    T = len(features_l)
    nL = features_l.shape[1]
    print(f"[data.py] V1 labeled: T={T}, n_labeled={nL}, raw_feature_dim={_raw.shape[-1]}")

    # ===== DEBUG: data sanity =====
    print(f"[DEBUG/data] target stats: min={targets_l.min():.4f}, max={targets_l.max():.4f}, "
          f"mean={targets_l.mean():.4f}, std={targets_l.std():.4f}, NaN={np.isnan(targets_l).sum()}")
    print(f"[DEBUG/data] WRF (cols 1-54) stats: range=[{features_l[:, :, cfdIdx].min():.3f}, "
          f"{features_l[:, :, cfdIdx].max():.3f}], NaN={np.isnan(features_l[:, :, cfdIdx]).sum()}")
    print(f"[DEBUG/data] raw_geo (cols 59-78, KEPT) stats: range=[{features_l[:, :, rawGeoFeatIdx].min():.3f}, "
          f"{features_l[:, :, rawGeoFeatIdx].max():.3f}]")
    print(f"[DEBUG/data] geoEmbed stats: range=[{_geoFeatures.min():.3f}, {_geoFeatures.max():.3f}], "
          f"shape={_geoFeatures.shape}")
    print(f"[DEBUG/data] KEPT cols: WRF=1-54, station_aux=55-58 (hour/month/year/station_id), "
          f"raw_geo=59-78. DROPPED cols: target=col 0, junk=col 79")
    # Inspect station-aux at t=0 / sample station to verify physical interpretation
    print(f"[DEBUG/data] station_aux@t=0,sta=0: hour={features_l[0,0,54]:.4f}, "
          f"month={features_l[0,0,55]:.4f}, year={features_l[0,0,56]:.0f}, "
          f"station_id={features_l[0,0,57]:.0f}")

    # ----- Optional unlabeled (V2-source, V1-aligned) -----
    if n_unlabeled > 0:
        wrf_u, raw_geo_u, geo_emb_u, locs_u = load_unlabeled_v1_aligned(
            path, n_unlabeled, T_end=T,
            geo_pool_size=dataParam['poolSize'], fps_seed=fps_seed)
        # Pre-cache the geoEmbed for unlabeled (already pooled)
        if not torch.is_tensor(geo_emb_u):
            geo_emb_u = torch.FloatTensor(geo_emb_u)
    else:
        wrf_u = raw_geo_u = geo_emb_u = locs_u = None

    n_total = nL + (n_unlabeled if n_unlabeled > 0 else 0)

    # ----- Adjacency -----
    # Pull labeled locations from AJM (cheaper than reloading StationMat)
    _ajm = mat73.loadmat(f'{path}/GNN_N1_AJM.mat')
    _dist, _loc_ajm, _simiW = _ajm['dist'], _ajm['location'], _ajm['similarity']
    locs_l_for_split = _loc_ajm if _loc_ajm.shape[0] == 2 else _loc_ajm.T
    if n_unlabeled == 0:
        # Sup mode: V1 native sim×dist graph (faithful to original_code)
        _distW = np.exp(-_dist)
        _ao, _as = np.min(_distW), np.max(_distW) - np.min(_distW)
        _distW = (_distW - _ao) / _as
        Adj = np.abs(_simiW * _distW)
        Adj[Adj < dataParam['thres']] = 0.0
        np.fill_diagonal(Adj, 0)
        assert np.allclose(Adj, Adj.T), "[ERR] AJM graph not symmetric"
        print(f"[data.py] sup graph (V1 AJM): {nL} nodes, {int((Adj > 0).sum())} edges")
        # ===== DEBUG: graph =====
        nz = Adj[Adj > 0]
        print(f"[DEBUG/graph] V1 AJM density={int((Adj > 0).sum()) / nL / (nL-1) * 100:.1f}%, "
              f"weight range=[{nz.min():.4f}, {nz.max():.4f}], mean={nz.mean():.4f}")
        deg = (Adj > 0).sum(axis=1)
        print(f"[DEBUG/graph] degree distribution: min={deg.min()}, max={deg.max()}, "
              f"mean={deg.mean():.1f}, isolated={int((deg == 0).sum())}")
    else:
        # Semi mode: k-NN graph on combined (labeled + unlabeled) locations
        locs_all = np.concatenate([locs_l_for_split, locs_u], axis=1)
        Adj = build_knn_adj(locs_all, k=knn_k)
        print(f"[data.py] semi k-NN graph (k={knn_k}): {n_total} nodes, "
              f"{int((Adj > 0).sum())} edges, density={(Adj > 0).sum() / n_total / (n_total-1) * 100:.1f}%")
        # ===== DEBUG: graph =====
        n_ll = int((Adj[:nL, :nL] > 0).sum())
        n_uu = int((Adj[nL:, nL:] > 0).sum())
        n_lu = int((Adj[:nL, nL:] > 0).sum())
        print(f"[DEBUG/graph] L-L={n_ll // 2} | U-U={n_uu // 2} | L-U={n_lu} edges")
        deg = (Adj > 0).sum(axis=1)
        print(f"[DEBUG/graph] degree (k-NN k={knn_k}): min={deg.min()}, max={deg.max()}, "
              f"mean={deg.mean():.1f}, isolated={int((deg == 0).sum())}")
        nz = Adj[Adj > 0]
        print(f"[DEBUG/graph] edge weight: min={nz.min():.4f}, max={nz.max():.4f}, mean={nz.mean():.4f}")

    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))

    # ----- label_mask logic -----
    # For spatial: train_mask covers (nL - n_valid_spatial) train labeled, valid_mask covers n_valid_spatial valid
    # For random/seq: train and valid both use first nL nodes as "labeled"
    if val_mode == 'spatial':
        valid_station_idx = fps_select_stations(locs_l_for_split, n_valid_spatial, seed=spatial_seed)
        train_station_idx = sorted(set(range(nL)) - set(valid_station_idx))
        train_label_mask = np.zeros(n_total, dtype=bool)
        train_label_mask[train_station_idx] = True
        valid_label_mask = np.zeros(n_total, dtype=bool)
        valid_label_mask[valid_station_idx] = True
        print(f"[data.py] spatial valid stations: {valid_station_idx}, train: {len(train_station_idx)}")
        # ===== DEBUG: mask =====
        assert int(train_label_mask.sum()) == nL - n_valid_spatial
        assert int(valid_label_mask.sum()) == n_valid_spatial
        assert not np.any(train_label_mask & valid_label_mask), "[ERR] train/valid mask overlap"
        if n_unlabeled > 0:
            assert not np.any(train_label_mask[nL:]), "[ERR] train mask covers unlabeled"
            assert not np.any(valid_label_mask[nL:]), "[ERR] valid mask covers unlabeled"
        print(f"[DEBUG/mask] spatial: train={train_label_mask.sum()} | valid={valid_label_mask.sum()} | "
              f"disjoint=YES | unlabeled excluded=YES")
        try:
            visualize_spatial_split(
                locs_l_for_split, train_station_idx, valid_station_idx,
                save_path=os.path.join(output_dir, 'spatial_split.png'),
                unlabeled_locations=locs_u if n_unlabeled > 0 else None)
        except Exception as e:
            print(f"[data.py] spatial split viz skipped: {e}")
    else:
        # All labeled (first nL) participate in both train and valid loss; split is on time
        full_label_mask = np.zeros(n_total, dtype=bool)
        full_label_mask[:nL] = True
        train_label_mask = full_label_mask
        valid_label_mask = full_label_mask
        # ===== DEBUG: mask =====
        assert int(full_label_mask.sum()) == nL
        if n_unlabeled > 0:
            assert not np.any(full_label_mask[nL:]), "[ERR] mask covers unlabeled"
        print(f"[DEBUG/mask] {val_mode}: shared mask of {full_label_mask.sum()} labeled nodes "
              f"(unlabeled excluded={'YES' if n_unlabeled > 0 else 'N/A'})")

    # ----- Pre-compute V2 unlabeled station_id (per-station unique, range 100..99+n_unl) -----
    # Avoid collision with V1's 1..98 range. station_id is constant per station.
    if n_unlabeled > 0:
        station_id_u = np.arange(100, 100 + n_unlabeled, dtype=np.float32)  # (n_u,)

    # ----- Build per-timestep dataset -----
    # Schema: WRF window (270) + station_aux (4) + raw_geo (20) + geoEmbed (1008) = 1302 dim
    # All 4 station-aux cols (55-58: hour/month/year/station_id) KEPT, faithful to original_code.
    _dataset = []
    for n in range(_window, T - _window):
        _f = features_l[n]                                    # (nL, F-1)
        _tdb = features_l[n - _window:n, :, cfdIdx]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(nL, -1)
        _tdf = features_l[n:n + _window, :, cfdIdx]
        _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(nL, -1)
        feat_l = np.hstack([
            _f[:, cfdIdx], _tdb, _tdf,
            _f[:, stationFeatIdx],                            # 4 station-aux KEPT
            _f[:, rawGeoFeatIdx],
            _geoFeatures.numpy() if torch.is_tensor(_geoFeatures) else _geoFeatures,
        ])  # (nL, 1302)

        if n_unlabeled > 0:
            wrf_un = wrf_u[:, n, :]                                       # (nU, 54)
            tdb_u = wrf_u[:, n - _window:n, :].reshape(n_unlabeled, -1)   # (nU, 2*54)
            tdf_u = wrf_u[:, n:n + _window, :].reshape(n_unlabeled, -1)   # (nU, 2*54)
            # Build unlabeled station_aux 4 cols:
            #   hour, month, year: V1's value at this t (shared across all stations) → broadcast
            #   station_id: per-unlabeled unique, 100..99+n_unlabeled
            hour_t  = features_l[n, 0, stationFeatIdx[0]]  # V1 col 55 hour at t=n
            month_t = features_l[n, 0, stationFeatIdx[1]]  # V1 col 56 month
            year_t  = features_l[n, 0, stationFeatIdx[2]]  # V1 col 57 year
            station_aux_u = np.empty((n_unlabeled, 4), dtype=np.float32)
            station_aux_u[:, 0] = hour_t
            station_aux_u[:, 1] = month_t
            station_aux_u[:, 2] = year_t
            station_aux_u[:, 3] = station_id_u
            raw_geo_un = raw_geo_u[:, n, :]                                # (nU, 20)
            geo_emb_un = geo_emb_u.numpy() if torch.is_tensor(geo_emb_u) else geo_emb_u  # (nU, 1008)
            feat_u = np.hstack([wrf_un, tdb_u, tdf_u, station_aux_u, raw_geo_un, geo_emb_un])  # (nU, 1302)
            features_t = np.vstack([feat_l, feat_u])
            target_t = np.concatenate([targets_l[n], np.zeros(n_unlabeled, dtype=np.float32)])
        else:
            features_t = feat_l
            target_t = targets_l[n]

        d = Data(
            x=torch.FloatTensor(features_t),
            y=torch.FloatTensor(target_t.reshape(-1, 1)),
            edge_index=edgeIdxV,
            edge_attr=edgeAttrV,
        )
        _dataset.append(d)

    # ----- Train / valid split -----
    nSamples = len(_dataset)
    if val_mode == 'spatial':
        # All timesteps used for train AND valid; only label_mask differs
        trainSet, validSet = [], []
        for d in _dataset:
            d_t = d.clone(); d_t.label_mask = torch.BoolTensor(train_label_mask); trainSet.append(d_t)
            d_v = d.clone(); d_v.label_mask = torch.BoolTensor(valid_label_mask); validSet.append(d_v)
        train_indices = list(range(nSamples))
        valid_indices = list(range(nSamples))
    elif val_mode == 'sequential':
        n_train = int(nSamples * temporal_frac)
        for d in _dataset:
            d.label_mask = torch.BoolTensor(train_label_mask)
        trainSet = _dataset[:n_train]
        validSet = _dataset[n_train:]
        train_indices = list(range(n_train))
        valid_indices = list(range(n_train, nSamples))
        print(f"[data.py] sequential split: train={n_train}, valid={nSamples - n_train}")
    else:  # random
        _generator = torch.Generator().manual_seed(19)
        _trainLength = int(nSamples * nTrn)
        _validLength = nSamples - _trainLength
        for d in _dataset:
            d.label_mask = torch.BoolTensor(train_label_mask)
        trainSet, validSet = torch.utils.data.random_split(
            _dataset, [_trainLength, _validLength], _generator)
        train_indices = trainSet.indices
        valid_indices = validSet.indices
        print(f"[data.py] random split: train={_trainLength}, valid={_validLength}")

    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    valid_bs = _batchSize if val_mode == 'spatial' else len(validSet)
    validLoader = DataLoader(validSet, batch_size=valid_bs, shuffle=False)

    # ===== DEBUG: dataset sample shapes =====
    s = trainSet[0]
    print(f"[DEBUG/dataset] sample 0: x={tuple(s.x.shape)}, y={tuple(s.y.shape)}, "
          f"edge_index={tuple(s.edge_index.shape)}, label_mask={int(s.label_mask.sum())} True / "
          f"{int((~s.label_mask).sum())} False, target range=[{s.y.min().item():.4f}, {s.y.max().item():.4f}]")
    if n_unlabeled > 0:
        # Unlabeled rows are at indices [nL : nL + n_unlabeled]
        unl_y = s.y[nL:nL + n_unlabeled]
        print(f"[DEBUG/dataset] unlabeled rows y placeholder: "
              f"max abs={unl_y.abs().max().item():.4f} (should be 0)")
        # Verify station_aux on first unlabeled row.
        # station_aux is at offset = WRF_window_dim (270).
        wrf_dim = nCFDFeats * (2 * _window + 1)
        unl_aux = s.x[nL, wrf_dim : wrf_dim + nStationFeats]   # 4 station-aux cols
        print(f"[DEBUG/dataset] unlabeled station_aux row 0: hour={unl_aux[0].item():.4f}, "
              f"month={unl_aux[1].item():.4f}, year={unl_aux[2].item():.0f}, "
              f"station_id={unl_aux[3].item():.0f}")

    iDim = features_t.shape[-1]
    oDim = 1
    expected_iDim = (nCFDFeats * (2 * _window + 1) + nStationFeats
                     + len(rawGeoFeatIdx) + _geoFeatures.shape[-1])
    assert iDim == expected_iDim, f"[ERR] iDim mismatch: actual={iDim}, expected={expected_iDim}"
    _featureLen = np.cumsum([
        nCFDFeats * (2 * _window + 1),
        nStationFeats,
        len(rawGeoFeatIdx),
        _geoFeatures.shape[-1] if torch.is_tensor(_geoFeatures) else _geoFeatures.shape[-1],
    ])
    _featureIdx = {
        'CFD':       np.arange(0, _featureLen[0]),
        'station':   np.arange(_featureLen[0], _featureLen[1]),
        'rawGeo':    np.arange(_featureLen[1], _featureLen[2]),
        'embedGeo':  np.arange(_featureLen[2], _featureLen[3]),
    }
    print(f"[DEBUG/dataset] iDim breakdown: WRF window={int(_featureLen[0])} "
          f"+ station_aux={int(_featureLen[1] - _featureLen[0])} "
          f"+ raw_geo={int(_featureLen[2] - _featureLen[1])} "
          f"+ geoEmbed={int(_featureLen[3] - _featureLen[2])} = {iDim}")

    metadata = {
        'nNodes':          n_total,
        'nNodes_labeled':  nL,
        'nNodes_unlabeled': n_unlabeled,
        'geoOff':    _off,
        'geoScl':    _scl,
        'iDim':      iDim,
        'oDim':      oDim,
        'featureIdx': _featureIdx,
        'geoMethod': dataParam['geoMethod'],
        'poolSize':  dataParam['poolSize'],
        'nCompPCA':  dataParam['nCompPCA'],
        'trainIdx':  train_indices,
        'validIdx':  valid_indices,
        'AdjMatrix': Adj,
        'val_mode':  val_mode,
    }
    print(f"[data.py] iDim={iDim}, nNodes={n_total} ({nL} labeled + {n_unlabeled} unlabeled), "
          f"val_mode={val_mode}")
    return trainLoader, validLoader, metadata, validSet


# V2 数据路径:Labeled_Finalized_new (58 站, T=3672) + 同源 V2 unlabeled
def _dataGen_V2(dataParam, path, nTrn=0.75, predMode=False):
    """V2 数据路径。schema = WRF window 5×63 + station_aux 4 + CLMS_t 3 + UF 17 + GeoEmbed 1008 = 1347 维。

    无 V1 OFFSET(V2 labeled 起 t=0 = V2 unlabeled 起 t=0 = 2018-05-01 00:00)。
    station_aux 4 列(hour/month/year/station_id):V2 .mat 没自带,但代码 broadcast 时间 + station_id 构造。
    target 全局归一化 [0,1],tgt_scl 用于反归一化回 °C。
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']

    val_mode    = os.environ.get('V1_VAL_MODE', 'spatial').lower()
    n_unlabeled = int(os.environ.get('V1_N_UNLABELED', '0'))
    fps_seed    = int(os.environ.get('V1_FPS_SEED', '0'))
    n_valid_spatial = int(os.environ.get('V1_N_VALID_STATIONS', '8'))   # V2 默认 8(58 站太少)
    spatial_seed    = int(os.environ.get('V1_SPATIAL_SEED', '42'))
    temporal_frac   = float(os.environ.get('V1_TEMPORAL_FRAC', '0.8'))
    knn_k       = int(os.environ.get('V1_KNN_K', '10'))
    output_dir  = os.environ.get('V1_OUTPUT_DIR', '.')
    # ---- EAD env (新增,默认 off,完全等同 baseline)----
    ead_alpha_on = int(os.environ.get('V1_EAD_ALPHA', '0'))
    ead_beta_on  = int(os.environ.get('V1_EAD_BETA',  '0'))
    ead_active   = bool(ead_alpha_on or ead_beta_on)
    assert val_mode in ('random', 'sequential', 'spatial')
    print(f"[data.py V2] val_mode={val_mode}, n_unlabeled={n_unlabeled}, "
          f"EAD α={ead_alpha_on} β={ead_beta_on}")

    # ----- Load V2 labeled -----
    L = load_v2_labeled(path, geo_pool_size=dataParam['poolSize'])
    wrf_l    = L['wrf']           # (T, N, 63)
    clms_l   = L['clms']          # (T, N, 3)
    uf_l     = L['uf']            # (N, 17)
    geo_emb_l = L['geo_emb']      # (N, 1008)
    targets_l = L['targets_norm'] # (T, N)
    locs_l    = L['locations']    # (2, N)
    sim_l     = L['similarity']   # (N, 10)
    T = wrf_l.shape[0]
    nL = wrf_l.shape[1]
    print(f"[data.py V2] V2 labeled: T={T}, n_labeled={nL}, target_scl_C={L['tgt_scl']:.2f}")

    # ===== DEBUG: V2 data sanity =====
    print(f"[DEBUG/V2_data] target_norm: range=[{targets_l.min():.4f}, {targets_l.max():.4f}], "
          f"mean={targets_l.mean():.4f}, std={targets_l.std():.4f}, NaN={int(np.isnan(targets_l).sum())}")
    print(f"[DEBUG/V2_data] target_raw °C: min={L['tgt_min']:.2f}, scl={L['tgt_scl']:.2f}, "
          f"max=tgt_min+scl={L['tgt_min']+L['tgt_scl']:.2f}")
    print(f"[DEBUG/V2_data] WRF (63ch) range=[{wrf_l.min():.3f}, {wrf_l.max():.3f}], "
          f"NaN={int(np.isnan(wrf_l).sum())}")
    print(f"[DEBUG/V2_data] CLMS range=[{clms_l.min():.3f}, {clms_l.max():.3f}], "
          f"UF range=[{uf_l.min():.3f}, {uf_l.max():.3f}], "
          f"GeoEmbed range=[{geo_emb_l.min():.3f}, {geo_emb_l.max():.3f}]")
    assert not np.isnan(targets_l).any(), "[ERR/V2] target has NaN after normalization"
    assert not np.isnan(wrf_l).any(), "[ERR/V2] WRF has NaN after normalization"

    # ----- V2 labeled wrf_t2 in target normalization space (for EAD) -----
    wrf_t2_norm_l = L['wrf_t2_norm']  # (T, nL),target 空间

    # ----- Optional V2 unlabeled -----
    if n_unlabeled > 0:
        # 当 EAD 启用,把 tgt_min/scl 传给 unlabeled,让它也算 wrf_t2_norm
        U = load_v2_unlabeled(path, n_unlabeled, T_end=T,
                              geo_pool_size=dataParam['poolSize'], fps_seed=fps_seed,
                              tgt_min=L['tgt_min'] if ead_active else None,
                              tgt_scl=L['tgt_scl'] if ead_active else None)
        wrf_u, clms_u, uf_u, geo_emb_u, locs_u = (U['wrf'], U['clms'], U['uf'],
                                                   U['geo_emb'], U['locations'])
        wrf_t2_norm_u = U.get('wrf_t2_norm')   # (n_unl, T) or None
    else:
        wrf_u = clms_u = uf_u = geo_emb_u = locs_u = None
        wrf_t2_norm_u = None
    n_total = nL + (n_unlabeled if n_unlabeled > 0 else 0)

    # ----- Adjacency -----
    if n_unlabeled == 0:
        # V2 sup graph: SimilarityMat-based,faithful Yu et al.
        Adj = build_v2_adj(locs_l, sim_l, thres=dataParam['thres'])
        print(f"[data.py V2] V2 sup graph: {nL} nodes, {int((Adj > 0).sum())} edges")
        # ===== DEBUG: V2 graph =====
        assert np.allclose(Adj, Adj.T), "[ERR/V2] sup graph not symmetric"
        nz = Adj[Adj > 0]
        print(f"[DEBUG/V2_graph] V2 sup density={int((Adj > 0).sum()) / nL / (nL-1) * 100:.1f}%, "
              f"weight range=[{nz.min():.4f}, {nz.max():.4f}], mean={nz.mean():.4f}")
        deg = (Adj > 0).sum(axis=1)
        print(f"[DEBUG/V2_graph] degree: min={deg.min()}, max={deg.max()}, "
              f"mean={deg.mean():.1f}, isolated={int((deg == 0).sum())}")
    else:
        # V2 semi graph: k-NN k=10 over combined locations
        locs_all = np.concatenate([locs_l, locs_u], axis=1)
        Adj = build_knn_adj(locs_all, k=knn_k)
        print(f"[data.py V2] V2 semi k-NN graph (k={knn_k}): {n_total} nodes, "
              f"{int((Adj > 0).sum())} edges")
        # ===== DEBUG: V2 semi graph block分布 =====
        n_ll = int((Adj[:nL, :nL] > 0).sum())
        n_uu = int((Adj[nL:, nL:] > 0).sum())
        n_lu = int((Adj[:nL, nL:] > 0).sum())
        deg = (Adj > 0).sum(axis=1)
        nz = Adj[Adj > 0]
        print(f"[DEBUG/V2_graph] L-L={n_ll // 2} | U-U={n_uu // 2} | L-U={n_lu} edges")
        print(f"[DEBUG/V2_graph] degree (k-NN k={knn_k}): min={deg.min()}, max={deg.max()}, "
              f"mean={deg.mean():.1f}, isolated={int((deg == 0).sum())}")
        print(f"[DEBUG/V2_graph] edge weight: min={nz.min():.4f}, max={nz.max():.4f}, mean={nz.mean():.4f}")
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))

    # ----- GeoEmbed dim reduction (env: V1_GEO_DIM_REDUCE) -----
    # 'pool' (default): 已经在 load_v2_labeled / load_v2_unlabeled 里完成池化,这里不动
    # 'pca':           对 geo_emb 用 PCA 进一步降维
    #                  fit 集合 = train labeled + unlabeled(共 ~450 样本)
    #                  排除 valid 8 站防止 feature 分布 leak;unlabeled 无 target 进入,无 leak
    geo_dim_reduce = os.environ.get('V1_GEO_DIM_REDUCE', 'pool').lower()
    if geo_dim_reduce == 'pca':
        from sklearn.decomposition import PCA
        geo_pca_dim = int(os.environ.get('V1_GEO_PCA_DIM', '256'))
        # spatial 模式排除 valid;其它模式全用 labeled
        if val_mode == 'spatial':
            tmp_valid = fps_select_stations(locs_l, n_valid_spatial, seed=spatial_seed)
            pca_train_lbl_idx = sorted(set(range(nL)) - set(tmp_valid))
        else:
            pca_train_lbl_idx = list(range(nL))
        # Combine train labeled + unlabeled for PCA fit
        fit_geo_l = geo_emb_l[pca_train_lbl_idx]   # (n_train, 1008)
        if n_unlabeled > 0 and geo_emb_u is not None:
            all_fit_geo = np.concatenate([fit_geo_l, geo_emb_u], axis=0)  # (n_train + n_unl, 1008)
            print(f"[data.py V2] PCA fit set: {len(pca_train_lbl_idx)} train labeled + {n_unlabeled} unlabeled = {all_fit_geo.shape[0]} samples")
        else:
            all_fit_geo = fit_geo_l
            print(f"[data.py V2] PCA fit set: {len(pca_train_lbl_idx)} train labeled (no unlabeled)")
        n_comp = min(geo_pca_dim, all_fit_geo.shape[0], all_fit_geo.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(all_fit_geo)
        geo_emb_l_new = pca.transform(geo_emb_l).astype(np.float32)   # transform all 58 labeled
        print(f"[data.py V2] PCA: {pca.n_features_in_} → {n_comp} dim, "
              f"explained variance ratio sum = {pca.explained_variance_ratio_.sum():.4f}")
        print(f"[DEBUG/V2_PCA] geo_emb labeled before: shape={geo_emb_l.shape}, "
              f"range=[{geo_emb_l.min():.3f}, {geo_emb_l.max():.3f}]")
        geo_emb_l = geo_emb_l_new
        print(f"[DEBUG/V2_PCA] geo_emb labeled after:  shape={geo_emb_l.shape}, "
              f"range=[{geo_emb_l.min():.3f}, {geo_emb_l.max():.3f}]")
        if n_unlabeled > 0 and geo_emb_u is not None:
            geo_emb_u_new = pca.transform(geo_emb_u).astype(np.float32)
            print(f"[DEBUG/V2_PCA] geo_emb unlabeled: shape={geo_emb_u_new.shape}, "
                  f"range=[{geo_emb_u_new.min():.3f}, {geo_emb_u_new.max():.3f}]")
            geo_emb_u = geo_emb_u_new
        # 防 PCA dim < 请求 dim 时 method_full 不一致(实际维度可能小于 256)
        if n_comp < geo_pca_dim:
            print(f"[WARN/V2_PCA] requested {geo_pca_dim} components but only {n_comp} available "
                  f"(限于 fit 集大小 {all_fit_geo.shape[0]});method_full 仍标 _pca{geo_pca_dim}")

    # ----- label_mask -----
    if val_mode == 'spatial':
        valid_station_idx = fps_select_stations(locs_l, n_valid_spatial, seed=spatial_seed)
        train_station_idx = sorted(set(range(nL)) - set(valid_station_idx))
        train_label_mask = np.zeros(n_total, dtype=bool); train_label_mask[train_station_idx] = True
        valid_label_mask = np.zeros(n_total, dtype=bool); valid_label_mask[valid_station_idx] = True
        print(f"[data.py V2] spatial valid stations: {valid_station_idx}, train: {len(train_station_idx)}")
        # ===== DEBUG: V2 mask =====
        assert int(train_label_mask.sum()) == nL - n_valid_spatial
        assert int(valid_label_mask.sum()) == n_valid_spatial
        assert not np.any(train_label_mask & valid_label_mask), "[ERR/V2] train/valid mask overlap"
        if n_unlabeled > 0:
            assert not np.any(train_label_mask[nL:]), "[ERR/V2] train mask covers unlabeled"
            assert not np.any(valid_label_mask[nL:]), "[ERR/V2] valid mask covers unlabeled"
        print(f"[DEBUG/V2_mask] spatial: train={train_label_mask.sum()} | valid={valid_label_mask.sum()} | "
              f"disjoint=YES | unlabeled excluded=YES")
        try:
            visualize_spatial_split(locs_l, train_station_idx, valid_station_idx,
                                    save_path=os.path.join(output_dir, 'spatial_split.png'),
                                    unlabeled_locations=locs_u if n_unlabeled > 0 else None)
        except Exception as e:
            print(f"[data.py V2] viz skipped: {e}")
    else:
        full_label_mask = np.zeros(n_total, dtype=bool); full_label_mask[:nL] = True
        train_label_mask = full_label_mask
        valid_label_mask = full_label_mask
        # ===== DEBUG: V2 mask =====
        assert int(full_label_mask.sum()) == nL
        if n_unlabeled > 0:
            assert not np.any(full_label_mask[nL:]), "[ERR/V2] mask covers unlabeled"
        print(f"[DEBUG/V2_mask] {val_mode}: shared mask of {full_label_mask.sum()} labeled nodes "
              f"(unlabeled excluded={'YES' if n_unlabeled > 0 else 'N/A'})")

    # ----- Build per-timestep dataset -----
    # V2 schema:WRF window (5×63=315) + station_aux (4) + CLMS_t (3) + UF (17) + GeoEmbed (1008) = 1347
    # station_aux 4 列:hour/month/year(从 t 算出,broadcast 到所有 station)+ station_id(per-station 唯一)
    print("[data.py V2] computing time features (hour/month/year) per timestep...")
    time_aux = v2_compute_time_features(T)   # (T, 3) — V2 t=0=2018-05-01 00:00 锚定
    station_id_l = np.arange(0, nL, dtype=np.float32)   # V2 labeled IDs: 0..57
    if n_unlabeled > 0:
        station_id_u = np.arange(100, 100 + n_unlabeled, dtype=np.float32)  # V2 unlabeled IDs: 100..99+n
    print(f"[data.py V2] V2 station_id ranges: labeled=[{station_id_l.min():.0f}, {station_id_l.max():.0f}], "
          f"unlabeled={'[100, ' + str(int(99 + n_unlabeled)) + ']' if n_unlabeled > 0 else 'N/A'}")
    print(f"[data.py V2] V2 time aux @ t=0: hour={time_aux[0,0]:.4f}, month={time_aux[0,1]:.4f}, "
          f"year={time_aux[0,2]:.0f}")

    # ===== EAD: 计算 α_t 和 β_hat(只在 V1_EAD_ALPHA / V1_EAD_BETA 启用时)=====
    ead_beta_mode = os.environ.get('V1_EAD_BETA_MODE', 'kriging').lower()   # 'kriging' (V0) | 'mlp' (V1 EAD-Plus)
    # EAD-Plus aux 子集模式(防 MLP 在 1027 维 + 50 train 上过拟合)
    ead_beta_aux = os.environ.get('V1_EAD_BETA_AUX', 'full').lower()  # 'full' (UF+GeoEmb+loc=1027) | 'uf' (only UF=17) | 'uf_loc' (UF+loc=19)
    if ead_active:
        # train_label_mask 是 (n_total,),前 nL 位是 labeled
        # spatial mode: train_label_mask 标 50 train 站(在 [0, nL) 内);valid_label_mask 标 8 valid 站
        # train mask 已经严格排除 valid 站,可以直接用作 EAD 的 train_mask
        # ===== EAD-Plus 准备 station_aux_full(只在 beta_mode='mlp' 时用)=====
        if ead_beta_on and ead_beta_mode == 'mlp':
            # 构造每站静态特征:UF (17) + GeoEmb (varies) + locations (2 = lat/lon)
            uf_full = np.zeros((n_total, uf_l.shape[-1]), dtype=np.float32)
            uf_full[:nL] = uf_l
            if n_unlabeled > 0:
                uf_full[nL:] = uf_u
            geo_full = np.zeros((n_total, geo_emb_l.shape[-1]), dtype=np.float32)
            geo_full[:nL] = geo_emb_l
            if n_unlabeled > 0:
                geo_full[nL:] = geo_emb_u
            loc_full = np.zeros((n_total, 2), dtype=np.float32)
            loc_full[:nL] = locs_l.T
            if n_unlabeled > 0:
                loc_full[nL:] = locs_u.T
            # 子集选择(防 MLP 在 1027 维上 overfit 50 train)
            if ead_beta_aux == 'full':
                station_aux_full = np.concatenate([uf_full, geo_full, loc_full], axis=1)
                aux_desc = f"UF={uf_full.shape[1]} + GeoEmb={geo_full.shape[1]} + loc=2"
            elif ead_beta_aux == 'uf':
                station_aux_full = uf_full.astype(np.float32)
                aux_desc = f"UF only={uf_full.shape[1]}"
            elif ead_beta_aux == 'uf_loc':
                station_aux_full = np.concatenate([uf_full, loc_full], axis=1)
                aux_desc = f"UF={uf_full.shape[1]} + loc=2"
            else:
                raise ValueError(f"unknown V1_EAD_BETA_AUX={ead_beta_aux}")
            print(f"  [EAD-Plus] station_aux_full shape: {station_aux_full.shape} ({aux_desc})")
        else:
            station_aux_full = None
        alpha_T, beta_hat_N = compute_ead_anchors(
            targets_l, wrf_t2_norm_l, Adj, train_label_mask,
            nL=nL, n_total=n_total,
            use_alpha=bool(ead_alpha_on), use_beta=bool(ead_beta_on),
            beta_mode=ead_beta_mode, station_aux_full=station_aux_full)
        # wrf_t2 全节点版本(labeled 直接用,unlabeled 来自 U['wrf_t2_norm'].T)
        wrf_t2_norm_full = np.zeros((T, n_total), dtype=np.float32)
        wrf_t2_norm_full[:, :nL] = wrf_t2_norm_l                     # (T, nL)
        if n_unlabeled > 0 and wrf_t2_norm_u is not None:
            wrf_t2_norm_full[:, nL:] = wrf_t2_norm_u.T               # (T, n_unl)  shape from (n_unl, T)
    else:
        alpha_T = np.zeros(T, dtype=np.float32)
        beta_hat_N = np.zeros(n_total, dtype=np.float32)
        wrf_t2_norm_full = np.zeros((T, n_total), dtype=np.float32)

    _dataset = []
    for n in range(_window, T - _window):
        # Labeled features
        wrf_t_l   = wrf_l[n]                              # (nL, 63)
        wrf_pre_l = wrf_l[n - _window:n].transpose(1, 0, 2).reshape(nL, -1)   # (nL, 2*63)
        wrf_post_l = wrf_l[n:n + _window].transpose(1, 0, 2).reshape(nL, -1)  # (nL, 2*63)
        clms_t_l = clms_l[n]                              # (nL, 3)
        # station_aux for labeled: time 3 列 broadcast + station_id
        sa_l = np.empty((nL, 4), dtype=np.float32)
        sa_l[:, 0:3] = time_aux[n]    # broadcast (3,) to (nL, 3)
        sa_l[:, 3] = station_id_l
        feat_l = np.hstack([wrf_t_l, wrf_pre_l, wrf_post_l, sa_l, clms_t_l, uf_l, geo_emb_l])  # (nL, 1347)

        if n_unlabeled > 0:
            wrf_t_u    = wrf_u[:, n, :]
            wrf_pre_u  = wrf_u[:, n - _window:n, :].reshape(n_unlabeled, -1)
            wrf_post_u = wrf_u[:, n:n + _window, :].reshape(n_unlabeled, -1)
            clms_t_u   = clms_u[:, n, :]
            # station_aux for unlabeled: 同 t 的 time 3 列 + 独立 station_id
            sa_u = np.empty((n_unlabeled, 4), dtype=np.float32)
            sa_u[:, 0:3] = time_aux[n]
            sa_u[:, 3] = station_id_u
            feat_u = np.hstack([wrf_t_u, wrf_pre_u, wrf_post_u, sa_u, clms_t_u, uf_u, geo_emb_u])  # (nU, 1347)
            features_t = np.vstack([feat_l, feat_u])
            target_t = np.concatenate([targets_l[n], np.zeros(n_unlabeled, dtype=np.float32)])
        else:
            features_t = feat_l
            target_t = targets_l[n]

        # === EAD 字段 attach 到 Data(每 timestep 一个 Data 对象)===
        kwargs_ead = {}
        if ead_active:
            kwargs_ead['alpha_t']  = torch.tensor(alpha_T[n], dtype=torch.float32)        # 标量
            kwargs_ead['beta_hat'] = torch.FloatTensor(beta_hat_N).reshape(-1, 1)         # (n_total, 1)
            kwargs_ead['wrf_t2']   = torch.FloatTensor(wrf_t2_norm_full[n]).reshape(-1, 1)  # (n_total, 1)

        d = Data(
            x=torch.FloatTensor(features_t),
            y=torch.FloatTensor(target_t.reshape(-1, 1)),
            edge_index=edgeIdxV,
            edge_attr=edgeAttrV,
            **kwargs_ead,
        )
        _dataset.append(d)

    if ead_active:
        # ===== DEBUG: EAD attached =====
        s_first = _dataset[0]
        print(f"[DEBUG/EAD] sample 0 (V2 t={_window}): alpha_t={s_first.alpha_t.item():.4f}, "
              f"beta_hat shape={tuple(s_first.beta_hat.shape)}, wrf_t2 shape={tuple(s_first.wrf_t2.shape)}")
        # 验证:对 train 站,y - wrf_t2 - alpha - beta ≈ ε(应该接近 0 在均值上)
        eps_check = (s_first.y.squeeze() - s_first.wrf_t2.squeeze()
                     - s_first.alpha_t - s_first.beta_hat.squeeze())
        train_mask_t = torch.BoolTensor(train_label_mask)
        eps_train = eps_check[train_mask_t]
        print(f"[DEBUG/EAD] @ t={_window} train ε = y - wrf_t2 - α - β: mean={eps_train.mean().item():.4f}, "
              f"std={eps_train.std().item():.4f} (应该 mean ≈ 0,说明 α/β 锚正确)")

    # Train/valid split (same logic as V1 path)
    nSamples = len(_dataset)
    if val_mode == 'spatial':
        trainSet, validSet = [], []
        for d in _dataset:
            d_t = d.clone(); d_t.label_mask = torch.BoolTensor(train_label_mask); trainSet.append(d_t)
            d_v = d.clone(); d_v.label_mask = torch.BoolTensor(valid_label_mask); validSet.append(d_v)
        train_indices = list(range(nSamples)); valid_indices = list(range(nSamples))
    elif val_mode == 'sequential':
        n_train = int(nSamples * temporal_frac)
        for d in _dataset: d.label_mask = torch.BoolTensor(train_label_mask)
        trainSet = _dataset[:n_train]; validSet = _dataset[n_train:]
        train_indices = list(range(n_train)); valid_indices = list(range(n_train, nSamples))
    else:  # random
        _generator = torch.Generator().manual_seed(19)
        _trainLength = int(nSamples * nTrn); _validLength = nSamples - _trainLength
        for d in _dataset: d.label_mask = torch.BoolTensor(train_label_mask)
        trainSet, validSet = torch.utils.data.random_split(
            _dataset, [_trainLength, _validLength], _generator)
        train_indices = trainSet.indices; valid_indices = validSet.indices

    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    valid_bs = _batchSize if val_mode == 'spatial' else len(validSet)
    validLoader = DataLoader(validSet, batch_size=valid_bs, shuffle=False)

    # ===== DEBUG: V2 dataset sample =====
    s = trainSet[0]
    print(f"[DEBUG/V2_dataset] sample 0: x={tuple(s.x.shape)}, y={tuple(s.y.shape)}, "
          f"edge_index={tuple(s.edge_index.shape)}, label_mask={int(s.label_mask.sum())} True / "
          f"{int((~s.label_mask).sum())} False, target range=[{s.y.min().item():.4f}, {s.y.max().item():.4f}]")
    # 检查 station_aux 在 V2 schema 中的位置:WRF window dim = 5×63=315
    wrf_dim = wrf_l.shape[-1] * (2 * _window + 1)
    lbl_aux = s.x[0, wrf_dim : wrf_dim + 4]   # labeled station 0 的 4 列 aux
    print(f"[DEBUG/V2_dataset] labeled station 0 aux: hour={lbl_aux[0].item():.4f}, "
          f"month={lbl_aux[1].item():.4f}, year={lbl_aux[2].item():.0f}, station_id={lbl_aux[3].item():.0f}")
    if n_unlabeled > 0:
        unl_y = s.y[nL:nL + n_unlabeled]
        print(f"[DEBUG/V2_dataset] unlabeled rows y placeholder: max abs={unl_y.abs().max().item():.4f} (should be 0)")
        unl_aux = s.x[nL, wrf_dim : wrf_dim + 4]
        print(f"[DEBUG/V2_dataset] unlabeled station 0 aux: hour={unl_aux[0].item():.4f}, "
              f"month={unl_aux[1].item():.4f}, year={unl_aux[2].item():.0f}, station_id={unl_aux[3].item():.0f}")

    iDim = features_t.shape[-1]
    expected = (wrf_l.shape[-1] * (2 * _window + 1)
                + 4   # station_aux: hour/month/year/station_id
                + clms_l.shape[-1] + uf_l.shape[-1] + geo_emb_l.shape[-1])
    assert iDim == expected, f"[ERR] V2 iDim mismatch: actual={iDim}, expected={expected}"
    print(f"[data.py V2] iDim breakdown: WRF window={wrf_l.shape[-1] * (2*_window+1)} "
          f"+ station_aux=4 + CLMS={clms_l.shape[-1]} + UF={uf_l.shape[-1]} "
          f"+ GeoEmbed={geo_emb_l.shape[-1]} = {iDim}")

    print(f"[data.py V2] iDim={iDim}, nNodes={n_total} ({nL}+{n_unlabeled})")
    # ===== Build full-graph locations + targets for self-train (kriging needs them) =====
    if n_unlabeled > 0:
        locations_full = np.concatenate([locs_l, locs_u], axis=1)   # (2, n_total)
    else:
        locations_full = locs_l
    targets_norm_full_arr = np.zeros((T, n_total), dtype=np.float32)
    targets_norm_full_arr[:, :nL] = targets_l   # train + valid 真值
    # unlabeled 位置保持 0(kriging 只用 train 真值,不用 unlabeled 0)

    # ===== Build train_station_idx / valid_station_idx for self-train =====
    if val_mode == 'spatial':
        st_train_idx = list(train_station_idx)
        st_valid_idx = list(valid_station_idx)
    else:
        # random / sequential 模式没有 spatial split
        st_train_idx = list(range(nL))
        st_valid_idx = []

    metadata = dict(
        nNodes=n_total, nNodes_labeled=nL, nNodes_unlabeled=n_unlabeled,
        iDim=iDim, oDim=1,
        trainIdx=train_indices, validIdx=valid_indices,
        AdjMatrix=Adj, val_mode=val_mode,
        tgt_min=L['tgt_min'], tgt_scl=L['tgt_scl'],   # for °C conversion
        dataset='V2',
        ead_active=ead_active, ead_alpha=bool(ead_alpha_on), ead_beta=bool(ead_beta_on),
        # ===== self-train extras =====
        train_station_idx=st_train_idx,
        valid_station_idx=st_valid_idx,
        locations=locations_full,
        targets_norm_full=targets_norm_full_arr,
    )
    return trainLoader, validLoader, metadata, validSet

"""
PyG数据集生成模块
包含有标签和无标签数据的PyG数据集生成功能

dataGen: 有标签数据（监督学习，68节点）— 移植自 downscale-gnn/data.py
dataGen_unlabeled: 无标签数据（独立图）— 特征维度与 dataGen 对齐
build_unified_graph / dataGen_semi: 统一图（保留，暂不使用）
"""
import os
import numpy as np
import torch
import mat73
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .geo_features import genGeoFeatures, genGeoFeatures_unlabeled


# ============================================================
# 辅助函数
# ============================================================

def apply_knn_sparsify(adj: np.ndarray, k: int = 8) -> np.ndarray:
    """
    kNN保底稀疏化：确保每个节点至少保留k个最强连接
    """
    n = adj.shape[0]
    keep = np.zeros_like(adj, dtype=bool)

    for i in range(n):
        row = adj[i].copy()
        row[i] = -np.inf
        kk = min(k, n - 1)
        idx = np.argpartition(-row, kth=kk - 1)[:kk]
        keep[i, idx] = True

    mask = np.logical_or(keep, keep.T)
    out = np.zeros_like(adj)
    out[mask] = adj[mask]
    np.fill_diagonal(out, 0.0)

    return out


# ============================================================
# dataGen: 有标签数据（监督学习）
# 移植自 downscale-gnn/data.py
# ============================================================

def dataGen(dataParam, path, nTrn=0.75, predMode=False):
    """
    生成有标签数据的PyG数据集（V2数据，58个节点）

    数据源: Labeled_Finalized_new.mat (V2, GPR filled)
    特征: WRF(63) + CLMS(3) + UrbanFeature(17) + GeoEmbed(poolSize依赖)
    图结构: SimilarityMat + Map → corrcoef * exp(-dist_normalized)
    """
    import scipy.io as sio

    _window = dataParam['window']
    _batchSize = dataParam['batchSize']

    # --------------------------加载V2数据--------------------------
    labeled = sio.loadmat(f'{path}/Labeled_Finalized_new.mat')

    # Target: GPR filled temperature
    AoT = labeled['AoT_filled']  # (3672, 58)
    if AoT.shape[0] < AoT.shape[1]:
        AoT = AoT.T

    # WRF features (63 dims, no wind merge)
    WRFMat = labeled['WRFMat']  # (3672, 63, 58)
    if WRFMat.shape[0] != AoT.shape[0]:
        WRFMat = np.transpose(WRFMat, (2, 1, 0))
    wrf_features = np.transpose(WRFMat, (0, 2, 1))  # (T, 58, 63)

    # CLMS features (3 dims)
    CLMSMat = labeled['CLMSMat']  # (3672, 3, 58)
    if CLMSMat.shape[0] != AoT.shape[0]:
        CLMSMat = np.transpose(CLMSMat, (2, 1, 0))
    clms_features = np.transpose(CLMSMat, (0, 2, 1))  # (T, 58, 3)

    _nStations = AoT.shape[1]
    T = AoT.shape[0]
    nWRF = wrf_features.shape[2]  # 63
    print(f"  [debug] Labeled data: {_nStations} stations, {T} timesteps, WRF={nWRF}d")
    print(f"  [debug] AoT: shape={AoT.shape}, NaN={np.isnan(AoT).sum()}, range=[{np.nanmin(AoT):.2f}, {np.nanmax(AoT):.2f}]")
    print(f"  [debug] WRF: shape={wrf_features.shape}, NaN={np.isnan(wrf_features).sum()}")
    print(f"  [debug] CLMS: shape={clms_features.shape}, NaN={np.isnan(clms_features).sum()}")

    # Normalize WRF (min-max per channel)
    wrf_flat = wrf_features.reshape(-1, nWRF)
    wrf_min = wrf_flat.min(axis=0)
    wrf_range = wrf_flat.max(axis=0) - wrf_min
    wrf_range[wrf_range == 0] = 1e-5
    wrf_features = (wrf_features - wrf_min) / wrf_range

    # Normalize CLMS
    clms_flat = clms_features.reshape(-1, 3)
    clms_min = clms_flat.min(axis=0)
    clms_range = clms_flat.max(axis=0) - clms_min
    clms_range[clms_range == 0] = 1e-5
    clms_features = (clms_features - clms_min) / clms_range

    # Normalize target
    target_min, target_max = np.nanmin(AoT), np.nanmax(AoT)
    targets = (AoT - target_min) / (target_max - target_min)  # (T, 58)

    # UrbanFeature (17 dims) — NaN handling
    UF = labeled.get('UrbanFeature', None)
    if UF is not None:
        if UF.shape[0] < UF.shape[1]:
            UF = UF.T  # -> (58, 17)
        # Height/fraction cols (0-15): NaN → 0; Distance col (16): NaN → median
        UF[:, :16] = np.nan_to_num(UF[:, :16], nan=0.0)
        col16 = UF[:, 16]
        if np.any(np.isnan(col16)):
            UF[np.isnan(col16), 16] = np.nanmedian(col16)
        # Min-max normalize
        uf_min = UF.min(axis=0)
        uf_range = UF.max(axis=0) - uf_min
        uf_range[uf_range == 0] = 1e-5
        UF_norm = (UF - uf_min) / uf_range
    else:
        UF_norm = np.zeros((_nStations, 0))

    # GeoEmbed from UrbanFeatureMat
    UFM = labeled.get('UrbanFeatureMat', None)
    if UFM is not None:
        UFM = np.nan_to_num(UFM, nan=0.0)
        print(f"  [debug] UrbanFeatureMat shape: {UFM.shape}, NaN after fill: {np.isnan(UFM).sum()}")
        _geoFeatures, _off, _scl, _ = genGeoFeatures_unlabeled(
            {'UrbanFeatureMat': UFM}, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
        )
        if torch.is_tensor(_geoFeatures):
            _geoFeatures = _geoFeatures.numpy()
    else:
        _geoFeatures = np.zeros((_nStations, 0))
        _off, _scl = 0, 1

    # --------------------------图构建（SimilarityMat + Map）--------------------------
    SM = labeled['SimilarityMat']  # (58, 10)
    Map = labeled['Map']           # (58, 2)
    if SM.shape[0] < SM.shape[1]:
        SM = SM.T
    if Map.shape[0] < Map.shape[1]:
        Map = Map.T

    # Distance from Map (normalized)
    Distance = np.zeros((_nStations, _nStations))
    for i in range(_nStations):
        for j in range(_nStations):
            Distance[i, j] = np.sqrt((Map[i, 0] - Map[j, 0])**2 + (Map[i, 1] - Map[j, 1])**2)
    max_dist = Distance.max()
    if max_dist > 0:
        Distance = Distance / max_dist

    # Similarity from SimilarityMat (corrcoef, max(r,0))
    Similarity = np.zeros((_nStations, _nStations))
    for i in range(_nStations):
        for j in range(_nStations):
            if i == j:
                Similarity[i, j] = 1.0
            else:
                cov = np.corrcoef(SM[i, :], SM[j, :])
                r = cov[0, 1] if not np.isnan(cov[0, 1]) else 0.0
                Similarity[i, j] = max(r, 0.0)

    Adj = Similarity * np.exp(-Distance)
    Adj[Adj < dataParam['thres']] = 0.
    np.fill_diagonal(Adj, 0.)
    Adj = (Adj + Adj.T) / 2  # ensure symmetry

    print(f"  Graph: {_nStations} nodes, {int(np.sum(Adj > 0))} edges, "
          f"density={np.sum(Adj > 0) / (_nStations * (_nStations - 1)) * 100:.1f}%")

    # --------------------------构建PyG数据集--------------------------
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
    wrf_idx = np.arange(nWRF)
    _dataset = []

    for n in range(_window, T - _window):
        _wrf_curr = wrf_features[n]  # (58, 63)
        _tdb = wrf_features[n - _window:n, :, :]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(_nStations, -1)
        _tdf = wrf_features[n:n + _window, :, :]
        _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(_nStations, -1)
        _clms = clms_features[n]  # (58, 3)

        _feat = np.hstack([_wrf_curr, _tdb, _tdf, _clms, UF_norm, _geoFeatures])

        _target = targets[n].reshape(-1, 1)
        _dataset.append(
            Data(
                x=torch.FloatTensor(_feat),
                y=torch.FloatTensor(_target),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV)
        )

    # 特征索引跟踪
    _wrfLen = nWRF * (2 * _window + 1)
    _clmsLen = 3
    _ufLen = UF_norm.shape[1]
    _geoLen = _geoFeatures.shape[1]
    _featureLen = np.cumsum([_wrfLen, _clmsLen, _ufLen, _geoLen])
    _featureIdx = {
        'WRF':          np.arange(0, _featureLen[0]),
        'CLMS':         np.arange(_featureLen[0], _featureLen[1]),
        'UrbanFeature': np.arange(_featureLen[1], _featureLen[2]),
        'GeoEmbed':     np.arange(_featureLen[2], _featureLen[3]),
    }

    iDim = _feat.shape[-1]
    print(f"  Total feature dim: {iDim} "
          f"(WRF={_wrfLen} + CLMS={_clmsLen} + UF={_ufLen} + Geo={_geoLen})")

    # Sanity check: first sample
    _sample = _dataset[0]
    print(f"  [debug] First sample: x={_sample.x.shape}, y={_sample.y.shape}, "
          f"x_NaN={torch.isnan(_sample.x).sum().item()}, y_NaN={torch.isnan(_sample.y).sum().item()}, "
          f"x_range=[{_sample.x.min():.4f}, {_sample.x.max():.4f}]")

    # 数据集分割
    _generator = torch.Generator().manual_seed(19)
    _trainLength = int(len(_dataset) * nTrn)
    _validLength = len(_dataset) - _trainLength
    trainSet, validSet = torch.utils.data.random_split(
        _dataset, [_trainLength, _validLength], _generator
    )
    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)

    metadata = {
        'nNodes':       _nStations,
        'geoOff':       _off,
        'geoScl':       _scl,
        'iDim':         _feat.shape[-1],
        'oDim':         _target.shape[-1],
        'featureIdx':   _featureIdx,
        'geoMethod':    dataParam['geoMethod'],
        'poolSize':     dataParam['poolSize'],
        'nCompPCA':     dataParam['nCompPCA'],
        'trainIdx':     trainSet.indices,
        'validIdx':     validSet.indices,
        'AdjMatrix':    Adj,
        'target_min':   target_min,
        'target_max':   target_max,
        'wrf_min':      wrf_min,
        'wrf_range':    wrf_range,
        'clms_min':     clms_min,
        'clms_range':   clms_range,
        'uf_min':       uf_min if UF is not None else None,
        'uf_range':     uf_range if UF is not None else None,
    }
    return trainLoader, validLoader, metadata, validSet


# ============================================================
# dataGen_unlabeled: 无标签数据（独立图）
# 特征维度与 dataGen 对齐（通过 zero-padding）
# ============================================================

def dataGen_unlabeled(dataParam, data, nTrn=0.75, seed=19, predMode=False,
                      labeled=False, path=None, labeled_metadata=None):
    """
    生成无标签数据的PyG数据集（独立图，V2数据）
    特征与 dataGen 对齐: WRF(63) + CLMS(3) + UrbanFeature(17) + GeoEmbed

    Args:
        dataParam: 数据参数字典
        data: 预处理后的无标签数据字典（含 WRFMat, CLMSMat, UrbanFeature 等）
        labeled_metadata: dataGen 返回的 metadata（用于归一化参数对齐）
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    T_2018 = 3672

    # GeoEmbed (use labeled norm params if available)
    if labeled_metadata is not None:
        _geoFeatures, _off, _scl, _nStations = genGeoFeatures_unlabeled(
            data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'],
            norm_off=labeled_metadata['geoOff'], norm_scl=labeled_metadata['geoScl']
        )
    else:
        _geoFeatures, _off, _scl, _nStations = genGeoFeatures_unlabeled(
            data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
        )
    if torch.is_tensor(_geoFeatures):
        _geoFeatures = _geoFeatures.numpy()

    # --------------------------图构建（Map + SimilarityMat）--------------------------
    Map = data['Map']
    SimilarityMat = data['SimilarityMat']
    nNodes = Map.shape[0]

    Distance = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            Distance[i, j] = np.sqrt((Map[i, 0] - Map[j, 0])**2 + (Map[i, 1] - Map[j, 1])**2)
    max_dist = Distance.max()
    if max_dist > 0:
        Distance = Distance / max_dist

    Similarity = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            if i == j:
                Similarity[i, j] = 1.0
            else:
                cov = np.corrcoef(SimilarityMat[i, :], SimilarityMat[j, :])
                r = cov[0, 1] if not np.isnan(cov[0, 1]) else 0.0
                Similarity[i, j] = max(r, 0.0)

    Adj = Similarity * np.exp(-Distance)
    Adj[Adj < dataParam['thres']] = 0.
    np.fill_diagonal(Adj, 0.)
    Adj = (Adj + Adj.T) / 2

    # --------------------------节点特征--------------------------
    WRFMat = data['WRFMat']
    CLMSMat = data['CLMSMat']
    cfd_features = np.transpose(WRFMat, (0, 2, 1))[:T_2018]    # (T, nNodes, 63)
    clms_features = np.transpose(CLMSMat, (0, 2, 1))[:T_2018]  # (T, nNodes, 3)
    nWRF = cfd_features.shape[2]

    # Normalize using labeled params if available
    if labeled_metadata is not None and 'wrf_min' in labeled_metadata:
        cfd_features = (cfd_features - labeled_metadata['wrf_min']) / labeled_metadata['wrf_range']
        cfd_features = np.clip(cfd_features, 0, 1)
        clms_features = (clms_features - labeled_metadata['clms_min']) / labeled_metadata['clms_range']
        clms_features = np.clip(clms_features, 0, 1)

    # UrbanFeature (NaN handling)
    UF = data.get('UrbanFeature', None)
    if UF is not None:
        UF[:, :16] = np.nan_to_num(UF[:, :16], nan=0.0)
        col16 = UF[:, 16]
        if np.any(np.isnan(col16)):
            UF[np.isnan(col16), 16] = np.nanmedian(col16)
        if labeled_metadata is not None and labeled_metadata.get('uf_min') is not None:
            UF_norm = (UF - labeled_metadata['uf_min']) / labeled_metadata['uf_range']
            UF_norm = np.clip(UF_norm, 0, 1)
        else:
            uf_min = UF.min(axis=0)
            uf_range = UF.max(axis=0) - uf_min
            uf_range[uf_range == 0] = 1e-5
            UF_norm = (UF - uf_min) / uf_range
    else:
        UF_norm = np.zeros((nNodes, 0))

    # --------------------------构建PyG数据集--------------------------
    T = len(cfd_features)
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
    _dataset = []

    for n in range(_window, T - _window):
        _wrf = cfd_features[n]
        _tdb = cfd_features[n - _window:n]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(nNodes, -1)
        _tdf = cfd_features[n:n + _window]
        _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(nNodes, -1)
        _clms = clms_features[n]

        _feat = np.hstack([_wrf, _tdb, _tdf, _clms, UF_norm, _geoFeatures])

        data_obj = Data(
            x=torch.FloatTensor(_feat),
            edge_index=edgeIdxV,
            edge_attr=edgeAttrV
        )
        _dataset.append(data_obj)

    # 数据集分割
    _generator = torch.Generator().manual_seed(seed)
    _trainLength = int(len(_dataset) * nTrn)
    _validLength = int(len(_dataset) * 0.2)
    _testLength = len(_dataset) - _validLength - _trainLength

    trainSet, validSet, testSet = torch.utils.data.random_split(
        _dataset, [_trainLength, _validLength, _testLength], _generator
    )
    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)
    testLoader = DataLoader(testSet, batch_size=len(testSet), shuffle=False)

    metadata = {
        'nNodes':    _nStations,
        'geoOff':    _off,
        'geoScl':    _scl,
        'iDim':      _feat.shape[-1],
        'oDim':      None,
        'geoMethod': dataParam['geoMethod'],
        'poolSize':  dataParam['poolSize'],
        'nCompPCA':  dataParam['nCompPCA'],
        'trainIdx':  trainSet.indices,
        'validIdx':  validSet.indices,
        'AdjMatrix': Adj,
    }

    return trainLoader, validLoader, testLoader, metadata, validSet


# ============================================================
# dataGen_unified: 统一图结构（labeled + unlabeled 合并为268节点大图）
# labeled 在前（0:n_labeled），unlabeled 在后（n_labeled:total_nodes）
# ============================================================

def dataGen_unified(dataParam, path, unlabeled_data, nTrn=0.75, seed=19, predMode=False,
                    output_dir=None, augmenter=None):
    """
    统一图结构：将58个有标签节点和200个无标签节点合并为258节点的大图
    V2数据源：Labeled_Finalized_new.mat + Unlabeled_Finalized.mat
    每个时间步生成一个 Data 对象，包含 labeled_mask 和 unlabeled_mask

    Args:
        dataParam: 数据参数字典
        path: 有标签数据路径
        unlabeled_data: 预处理（+增强）后的无标签数据字典
        nTrn: 训练集比例
        output_dir: 可选，若提供则将图结构可视化保存到该目录
        seed: 随机种子
        predMode: 预测模式
        augmenter: 可选，TransformFixMatch实例，用于对labeled数据施加同样的增强
    """
    import scipy.io as sio

    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    n_unlabeled = unlabeled_data['Map'].shape[0]
    T_2018 = 3672

    # 1. 加载V2有标签数据
    labeled = sio.loadmat(f'{path}/Labeled_Finalized_new.mat')
    AoT = labeled['AoT_filled']  # (3672, 58)
    if AoT.shape[0] < AoT.shape[1]:
        AoT = AoT.T
    WRFMat_l = labeled['WRFMat']
    CLMSMat_l = labeled['CLMSMat']
    if WRFMat_l.shape[0] != AoT.shape[0]:
        WRFMat_l = np.transpose(WRFMat_l, (2, 1, 0))
    if CLMSMat_l.shape[0] != AoT.shape[0]:
        CLMSMat_l = np.transpose(CLMSMat_l, (2, 1, 0))
    wrf_labeled = np.transpose(WRFMat_l, (0, 2, 1))  # (T, 58, 63)
    clms_labeled = np.transpose(CLMSMat_l, (0, 2, 1))  # (T, 58, 3)

    n_labeled = AoT.shape[1]
    nWRF = wrf_labeled.shape[2]  # 63
    print(f"  [debug] Labeled: {n_labeled} stations, {AoT.shape[0]} timesteps, WRF={nWRF}d")
    print(f"  [debug] AoT: NaN={np.isnan(AoT).sum()}, range=[{np.nanmin(AoT):.2f}, {np.nanmax(AoT):.2f}]")
    print(f"  [debug] WRF labeled: shape={wrf_labeled.shape}, NaN={np.isnan(wrf_labeled).sum()}")
    print(f"  [debug] CLMS labeled: shape={clms_labeled.shape}, NaN={np.isnan(clms_labeled).sum()}")

    # Normalize WRF (labeled params, shared with unlabeled)
    wrf_flat = wrf_labeled.reshape(-1, nWRF)
    wrf_min = wrf_flat.min(axis=0)
    wrf_range = wrf_flat.max(axis=0) - wrf_min
    wrf_range[wrf_range == 0] = 1e-5
    wrf_labeled = (wrf_labeled - wrf_min) / wrf_range

    # Normalize CLMS
    clms_flat = clms_labeled.reshape(-1, 3)
    clms_min = clms_flat.min(axis=0)
    clms_range = clms_flat.max(axis=0) - clms_min
    clms_range[clms_range == 0] = 1e-5
    clms_labeled = (clms_labeled - clms_min) / clms_range

    # Apply same augmentation to labeled data (per original Mean Teacher paper)
    # Both labeled and unlabeled should be treated identically except for supervised loss
    if augmenter is not None:
        # WRF: (T, nNodes, 63) → transpose to (T, 63, nNodes) for augmenter
        wrf_for_aug = np.transpose(wrf_labeled, (0, 2, 1))
        wrf_aug, _ = augmenter.augment_variable(wrf_for_aug)
        wrf_labeled = np.transpose(wrf_aug, (0, 2, 1))
        # CLMS: (T, nNodes, 3) → transpose to (T, 3, nNodes)
        clms_for_aug = np.transpose(clms_labeled, (0, 2, 1))
        clms_aug, _ = augmenter.augment_variable(clms_for_aug)
        clms_labeled = np.transpose(clms_aug, (0, 2, 1))
        print("  ✓ Labeled data augmented (same as unlabeled, per original MT paper)")

    # Normalize target
    target_min, target_max = np.nanmin(AoT), np.nanmax(AoT)
    targets = (AoT - target_min) / (target_max - target_min)

    # UrbanFeature labeled (17 dims)
    UF_l = labeled.get('UrbanFeature', None)
    if UF_l is not None:
        if UF_l.shape[0] < UF_l.shape[1]:
            UF_l = UF_l.T
        UF_l[:, :16] = np.nan_to_num(UF_l[:, :16], nan=0.0)
        col16 = UF_l[:, 16]
        if np.any(np.isnan(col16)):
            UF_l[np.isnan(col16), 16] = np.nanmedian(col16)
        uf_min = UF_l.min(axis=0)
        uf_range = UF_l.max(axis=0) - uf_min
        uf_range[uf_range == 0] = 1e-5
        UF_l_norm = (UF_l - uf_min) / uf_range
    else:
        UF_l_norm = np.zeros((n_labeled, 0))
        uf_min, uf_range = 0, 1

    # GeoEmbed labeled
    UFM_l = labeled.get('UrbanFeatureMat', None)
    if UFM_l is not None:
        UFM_l = np.nan_to_num(UFM_l, nan=0.0)
        # genGeoFeatures_unlabeled expects dict with 'UrbanFeatureMat' key
        _geoFeatures_labeled, _off, _scl, _ = genGeoFeatures_unlabeled(
            {'UrbanFeatureMat': UFM_l}, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
        )
        if torch.is_tensor(_geoFeatures_labeled):
            _geoFeatures_labeled = _geoFeatures_labeled.numpy()
    else:
        _geoFeatures_labeled = np.zeros((n_labeled, 0))
        _off, _scl = 0, 1

    # 2. 无标签节点特征
    # preprocess_unlabeled_data outputs: WRFMat (nStations, nFeats, T) or (T, nFeats, nStations)
    WRFMat_u = unlabeled_data['WRFMat']
    CLMSMat_u = unlabeled_data['CLMSMat']
    print(f"  [debug] Unlabeled WRF raw shape: {WRFMat_u.shape}, CLMS raw shape: {CLMSMat_u.shape}")

    # Detect shape and transpose to (T, nStations, nFeats)
    if WRFMat_u.ndim == 3:
        # Find which axis is time (largest dim, should be 6624 or 3672)
        shapes = WRFMat_u.shape
        time_axis = np.argmax(shapes)
        feat_axis = list(shapes).index(63) if 63 in shapes else np.argmin(shapes)
        station_axis = 3 - time_axis - feat_axis
        cfd_unlabeled = np.transpose(WRFMat_u, (time_axis, station_axis, feat_axis))[:T_2018]
    else:
        raise ValueError(f"Unexpected WRFMat shape: {WRFMat_u.shape}")

    if CLMSMat_u.ndim == 3:
        shapes_c = CLMSMat_u.shape
        time_axis_c = np.argmax(shapes_c)
        feat_axis_c = list(shapes_c).index(3) if 3 in shapes_c else np.argmin(shapes_c)
        station_axis_c = 3 - time_axis_c - feat_axis_c
        clms_unlabeled = np.transpose(CLMSMat_u, (time_axis_c, station_axis_c, feat_axis_c))[:T_2018]
    else:
        raise ValueError(f"Unexpected CLMSMat shape: {CLMSMat_u.shape}")

    print(f"  [debug] Unlabeled after transpose: WRF={cfd_unlabeled.shape}, CLMS={clms_unlabeled.shape}")

    # Normalize unlabeled using labeled params
    cfd_unlabeled = (cfd_unlabeled - wrf_min) / wrf_range
    cfd_unlabeled = np.clip(cfd_unlabeled, 0, 1)
    clms_unlabeled = (clms_unlabeled - clms_min) / clms_range
    clms_unlabeled = np.clip(clms_unlabeled, 0, 1)

    # UrbanFeature unlabeled
    UF_u = unlabeled_data.get('UrbanFeature', None)
    if UF_u is not None:
        UF_u[:, :16] = np.nan_to_num(UF_u[:, :16], nan=0.0)
        col16_u = UF_u[:, 16]
        if np.any(np.isnan(col16_u)):
            UF_u[np.isnan(col16_u), 16] = np.nanmedian(col16_u)
        UF_u_norm = (UF_u - uf_min) / uf_range
        UF_u_norm = np.clip(UF_u_norm, 0, 1)
    else:
        UF_u_norm = np.zeros((n_unlabeled, 0))

    # GeoEmbed unlabeled (use labeled norm params)
    _geoFeatures_unlabeled, _, _, _ = genGeoFeatures_unlabeled(
        unlabeled_data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'],
        norm_off=_off, norm_scl=_scl
    )
    if torch.is_tensor(_geoFeatures_unlabeled):
        _geoFeatures_unlabeled = _geoFeatures_unlabeled.numpy()

    T = len(wrf_labeled)
    assert len(cfd_unlabeled) == T, f"Time mismatch: labeled={T}, unlabeled={len(cfd_unlabeled)}"

    # 3. 构建统一图
    unified_adj, labeled_indices, unlabeled_indices, unified_locations = build_unified_graph(
        dataParam, path, unlabeled_data, n_unlabeled=n_unlabeled
    )
    total_nodes = n_labeled + n_unlabeled

    # 图结构统计
    n_ll = int(np.sum(unified_adj[:n_labeled, :n_labeled] > 0) // 2)
    n_uu = int(np.sum(unified_adj[n_labeled:, n_labeled:] > 0) // 2)
    n_lu = int(np.sum(unified_adj[:n_labeled, n_labeled:] > 0))
    degrees = np.sum(unified_adj > 0, axis=1)
    print(f"[统一图] 节点: {total_nodes} (labeled={n_labeled}, unlabeled={n_unlabeled})")
    print(f"[统一图] 边数: 总={n_ll+n_uu+n_lu} | labeled内部={n_ll} | unlabeled内部={n_uu} | 跨图={n_lu}")
    print(f"[统一图] 节点度数: min={degrees.min()} max={degrees.max()} mean={degrees.mean():.1f} | 孤立节点={np.sum(degrees==0)}")
    print(f"  Total feature dim: WRF={nWRF*(2*_window+1)} + CLMS=3 + UF={UF_l_norm.shape[1]} + Geo={_geoFeatures_labeled.shape[1]}")

    # 可视化图结构
    if output_dir is not None:
        try:
            from visualize_graph import visualize_unified_graph
            save_path = os.path.join(output_dir, 'unified_graph.png')
            visualize_unified_graph(unified_adj, unified_locations, n_labeled, save_path)
        except Exception as e:
            print(f"  [warning] Graph visualization failed: {e}")

    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(unified_adj))

    # 静态 mask
    labeled_mask = torch.zeros(total_nodes, dtype=torch.bool)
    labeled_mask[labeled_indices] = True
    unlabeled_mask = torch.zeros(total_nodes, dtype=torch.bool)
    unlabeled_mask[unlabeled_indices] = True

    # 4. 构建 PyG 数据集
    _dataset = []
    for n in range(_window, T - _window):
        # Labeled features
        _wrf_l = wrf_labeled[n]
        _tdb_l = wrf_labeled[n - _window:n]
        _tdb_l = np.transpose(_tdb_l, (1, 0, 2)).reshape(n_labeled, -1)
        _tdf_l = wrf_labeled[n:n + _window]
        _tdf_l = np.transpose(_tdf_l, (1, 0, 2)).reshape(n_labeled, -1)
        _clms_l = clms_labeled[n]
        _x_l = np.hstack([_wrf_l, _tdb_l, _tdf_l, _clms_l, UF_l_norm, _geoFeatures_labeled])

        # Unlabeled features (same structure)
        _wrf_u = cfd_unlabeled[n]
        _tdb_u = cfd_unlabeled[n - _window:n]
        _tdb_u = np.transpose(_tdb_u, (1, 0, 2)).reshape(n_unlabeled, -1)
        _tdf_u = cfd_unlabeled[n:n + _window]
        _tdf_u = np.transpose(_tdf_u, (1, 0, 2)).reshape(n_unlabeled, -1)
        _clms_u = clms_unlabeled[n]
        _x_u = np.hstack([_wrf_u, _tdb_u, _tdf_u, _clms_u, UF_u_norm, _geoFeatures_unlabeled])

        # Merge: labeled first, unlabeled after
        _x = np.vstack([_x_l, _x_u])
        _target = targets[n].reshape(-1, 1)

        _dataset.append(Data(
            x=torch.FloatTensor(_x),
            y=torch.FloatTensor(_target),
            edge_index=edgeIdxV,
            edge_attr=edgeAttrV,
            labeled_mask=labeled_mask,
            unlabeled_mask=unlabeled_mask,
        ))

    # Sanity check: first sample
    _sample = _dataset[0]
    iDim = _sample.x.shape[1]
    print(f"  [debug] First sample: x={_sample.x.shape}, y={_sample.y.shape}, edges={_sample.edge_index.shape}")
    print(f"  [debug] x NaN={torch.isnan(_sample.x).sum().item()}, "
          f"x range=[{_sample.x.min():.4f}, {_sample.x.max():.4f}], "
          f"y range=[{_sample.y.min():.4f}, {_sample.y.max():.4f}]")
    print(f"  [debug] labeled_mask: {_sample.labeled_mask.sum().item()} True, "
          f"unlabeled_mask: {_sample.unlabeled_mask.sum().item()} True")
    print(f"  [debug] Feature dim breakdown: WRF={nWRF*(2*_window+1)} + CLMS=3 + "
          f"UF={UF_l_norm.shape[1]} + Geo={_geoFeatures_labeled.shape[1]} = {iDim}")
    print(f"  [debug] Total samples: {len(_dataset)}")

    # 数据集分割
    _generator = torch.Generator().manual_seed(seed)
    _trainLength = int(len(_dataset) * nTrn)
    _validLength = len(_dataset) - _trainLength
    trainSet, validSet = torch.utils.data.random_split(
        _dataset, [_trainLength, _validLength], _generator
    )
    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)

    metadata = {
        'nNodes':         n_labeled,
        'n_labeled':      n_labeled,
        'n_unlabeled':    n_unlabeled,
        'total_nodes':    total_nodes,
        'geoOff':         _off,
        'geoScl':         _scl,
        'iDim':           _x_l.shape[-1],
        'oDim':           1,
        'geoMethod':      dataParam['geoMethod'],
        'poolSize':       dataParam['poolSize'],
        'nCompPCA':       dataParam['nCompPCA'],
        'trainIdx':       trainSet.indices,
        'validIdx':       validSet.indices,
        'AdjMatrix':      unified_adj,
        'target_min':     target_min,
        'target_max':     target_max,
        'wrf_min':        wrf_min,
        'wrf_range':      wrf_range,
    }
    return trainLoader, validLoader, metadata, validSet


# ============================================================
# 统一图相关函数（保留，暂不使用）
# ============================================================

def build_unified_graph(dataParam, path, unlabeled_data, n_unlabeled=200, seed=42):
    """
    构建统一的图结构，包含有标签站点(58个)和无标签站点(200个)

    V2数据源：
    - Labeled: Labeled_Finalized_new.mat (58 stations, SimilarityMat + Map)
    - Unlabeled: unlabeled_data dict (200 stations, SimilarityMat + Map)
    L-L/U-U/L-U 全部用同一公式: max(corrcoef, 0) * exp(-dist_normalized)
    """
    import scipy.io as sio

    # 1. Labeled部分：从V2 new mat加载（58站）
    labeled = sio.loadmat(f'{path}/Labeled_Finalized_new.mat')
    SM_labeled = labeled['SimilarityMat']  # (58, 10)
    Map_labeled = labeled['Map']           # (58, 2)
    if SM_labeled.shape[0] < SM_labeled.shape[1]:
        SM_labeled = SM_labeled.T
    if Map_labeled.shape[0] < Map_labeled.shape[1]:
        Map_labeled = Map_labeled.T
    n_labeled = SM_labeled.shape[0]

    # 2. Unlabeled部分：从预处理数据（200站）
    SM_unlabeled = unlabeled_data['SimilarityMat'][:n_unlabeled, :]  # (200, 10)
    Map_unlabeled = unlabeled_data['Map'][:n_unlabeled, :]           # (200, 2)

    # 3. 合并所有268个节点
    total_nodes = n_labeled + n_unlabeled
    Map_all = np.vstack([Map_labeled, Map_unlabeled])    # (268, 2)
    SM_all = np.vstack([SM_labeled, SM_unlabeled])       # (268, 10)

    print(f"  [统一图] 节点: {total_nodes} (labeled={n_labeled}, unlabeled={n_unlabeled})")
    print(f"  [统一图] Map range: [{Map_all.min():.2f}, {Map_all.max():.2f}]")
    print(f"  [统一图] SimilarityMat range: [{SM_all.min():.4f}, {SM_all.max():.4f}]")

    # 4. 按照 Yu et al. (2024) 统一计算距离矩阵（全部用 Map，同一数据源）
    print("  正在计算距离矩阵...")
    Distance = np.zeros((total_nodes, total_nodes))
    for i in range(total_nodes):
        for j in range(total_nodes):
            Distance[i, j] = np.sqrt(
                (Map_all[i, 0] - Map_all[j, 0])**2 +
                (Map_all[i, 1] - Map_all[j, 1])**2
            )

    # 5. 统一计算相似性矩阵（全部用 SimilarityMat，同一公式）
    print("  正在计算相似性矩阵...")
    Similarity = np.zeros((total_nodes, total_nodes))
    for i in range(total_nodes):
        for j in range(total_nodes):
            Cov = np.corrcoef(SM_all[i, :], SM_all[j, :])
            r = Cov[0, 1] if not np.isnan(Cov[0, 1]) else 0.0
            r = max(r, 0.0)  # 截断负相关为0（与data_semi.py一致）
            Similarity[i, j] = r

    # 6. 构建邻接矩阵：Adj = Similarity * exp(-Distance_normalized)
    #    截断负相关后 Similarity >= 0，不需要 abs（与data_semi.py一致）
    #    距离归一化到 [0, 1]，避免 exp(-大数) ≈ 0
    max_dist = Distance.max()
    Distance_normalized = Distance / max_dist if max_dist > 0 else Distance
    Adj = Similarity * np.exp(-Distance_normalized)

    print(f"  [debug] Distance range: [{Distance.min():.2f}, {Distance.max():.2f}], "
          f"normalized max={Distance_normalized.max():.4f}")
    print(f"  [debug] Similarity range: [{Similarity.min():.4f}, {Similarity.max():.4f}]")
    print(f"  [debug] Adj before threshold: nonzero={np.sum(Adj > 0)}, "
          f"range=[{Adj[Adj > 0].min():.4f}, {Adj.max():.4f}]")

    # 7. 阈值稀疏化（统一阈值，和 data_semi.py 一致）
    thres = dataParam['thres']
    Adj[Adj < thres] = 0.0
    np.fill_diagonal(Adj, 0.0)

    # 确保对称
    Adj = (Adj + Adj.T) / 2

    # 8. 统计信息
    labeled_indices = np.arange(n_labeled)
    unlabeled_indices = np.arange(n_labeled, total_nodes)

    ll_adj = Adj[:n_labeled, :n_labeled]
    uu_adj = Adj[n_labeled:, n_labeled:]
    lu_adj = Adj[:n_labeled, n_labeled:]
    n_ll = int(np.sum(ll_adj > 0) // 2)
    n_uu = int(np.sum(uu_adj > 0) // 2)
    n_lu = int(np.sum(lu_adj > 0))

    degrees = np.sum(Adj > 0, axis=1)
    n_isolated = int(np.sum(degrees == 0))

    print(f"  [统一图] 阈值: thres={thres}")
    print(f"  [统一图] 边数: L-L={n_ll}, U-U={n_uu}, L-U={n_lu}, 总={n_ll + n_uu + n_lu}")
    print(f"  [统一图] 节点度数: min={degrees.min()} max={degrees.max()} mean={degrees.mean():.1f}")
    print(f"  [统一图] 孤立节点: {n_isolated}")

    if n_ll > 0:
        print(f"  [统一图] L-L边权重: mean={ll_adj[ll_adj > 0].mean():.4f}")
    if n_uu > 0:
        print(f"  [统一图] U-U边权重: mean={uu_adj[uu_adj > 0].mean():.4f}")
    if n_lu > 0:
        print(f"  [统一图] L-U边权重: mean={lu_adj[lu_adj > 0].mean():.4f}")

    # 用 NodeLocation 作为可视化坐标（lat/lon），Map 是网格坐标不适合可视化
    labeled_loc_latlon = labeled['NodeLocation']
    if labeled_loc_latlon.shape[0] < labeled_loc_latlon.shape[1]:
        labeled_loc_latlon = labeled_loc_latlon.T
    unlabeled_loc_latlon = unlabeled_data['NodeLocation'][:n_unlabeled, :]
    unified_locations = np.vstack([labeled_loc_latlon, unlabeled_loc_latlon])

    return Adj, labeled_indices, unlabeled_indices, unified_locations


def diagnose_graph_structure(unified_adj, labeled_indices, unlabeled_indices,
                             unified_locations, dataParam, verbose=True):
    """诊断统一图结构是否合理"""
    try:
        import networkx as nx
    except ImportError:
        print("警告: NetworkX未安装，跳过图结构诊断")
        return None

    n_labeled = len(labeled_indices)
    n_unlabeled = len(unlabeled_indices)
    total_nodes = n_labeled + n_unlabeled

    G = nx.from_numpy_array(unified_adj)
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = 2 * n_edges / total_nodes if total_nodes > 0 else 0
    is_connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)
    largest_cc_size = len(max(nx.connected_components(G), key=len)) if n_components > 0 else 0

    if verbose:
        print("=" * 60)
        print("图结构诊断报告")
        print("=" * 60)
        print(f"  总节点数: {total_nodes} (有标签: {n_labeled}, 无标签: {n_unlabeled})")
        print(f"  总边数: {n_edges}")
        print(f"  图密度: {density:.4f}")
        print(f"  平均度数: {avg_degree:.2f}")
        print(f"  是否连通: {'是' if is_connected else '否'}")
        print(f"  连通分量数: {n_components}")
        print(f"  最大连通分量大小: {largest_cc_size}/{total_nodes}")
        print("=" * 60)

    return {
        'n_nodes': total_nodes,
        'n_edges': n_edges,
        'density': density,
        'is_connected': is_connected,
        'n_components': n_components,
    }

"""
PyG数据集生成模块
包含有标签和无标签数据的PyG数据集生成功能

dataGen: 有标签数据（监督学习，68节点）— 移植自 downscale-gnn/data.py
dataGen_unlabeled: 无标签数据（独立图）— 特征维度与 dataGen 对齐
build_unified_graph / dataGen_semi: 统一图（保留，暂不使用）
"""
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
    生成有标签数据的PyG数据集（68个节点）

    特征组合选项（由 dataParam['geoFeatures'] 控制）：
    - 'full': CFD + tdb + tdf + stationFeat + rawGeo + geoEmbed
    - 'raw':  CFD + tdb + tdf + stationFeat + rawGeo
    - 'embed':CFD + tdb + tdf + stationFeat + geoEmbed
    - 'no':   CFD + tdb + tdf + stationFeat
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    _geoFeatures, _off, _scl, _nStations = genGeoFeatures(
        path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
    )

    # --------------------------图构建--------------------------
    _raw = mat73.loadmat(f'{path}/GNN_N1_AJM.mat')
    _dist, _, _simiW = _raw['dist'], _raw['location'], _raw['similarity']
    _nNodes = _dist.shape[0]
    _distW = np.exp(-_dist)
    _off_dist, _scl_dist = np.min(_distW), np.max(_distW) - np.min(_distW)
    _distW = (_distW - _off_dist) / _scl_dist
    Adj = np.abs(_simiW * _distW)
    Adj[Adj < dataParam['thres']] = 0.
    assert np.allclose(Adj, Adj.T)

    # --------------------------节点特征--------------------------
    _raw = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    _raw = np.transpose(_raw, (0, 2, 1))  # (timestep, nNodes, nFeatures)
    nCFDFeats = 54
    nStationFeats = 4
    cfdIdx = np.arange(nCFDFeats)
    stationFeatIdx = np.arange(nCFDFeats, nCFDFeats + nStationFeats)
    rawGeoFeatIdx = np.arange(nCFDFeats + nStationFeats, _raw.shape[-1] - 1)
    features = _raw[:, :, 1:]
    targets = _raw[:, :, 0]

    # --------------------------构建PyG数据集--------------------------
    T = len(features)
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
    _dataset = []

    for n in range(_window, T - _window):
        _feature = features[n]
        _tdb = features[n - _window:n, :, cfdIdx]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(len(Adj), -1)
        _tdf = features[n:n + _window, :, cfdIdx]
        _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(len(Adj), -1)

        if dataParam['geoFeatures'] == 'full':
            _feat = np.hstack([_feature[:, cfdIdx],
                               _tdb, _tdf,
                               _feature[:, stationFeatIdx],
                               _feature[:, rawGeoFeatIdx],
                               _geoFeatures])
        elif dataParam['geoFeatures'] == 'raw':
            _feat = np.hstack([_feature[:, cfdIdx],
                               _tdb, _tdf,
                               _feature[:, stationFeatIdx],
                               _feature[:, rawGeoFeatIdx]])
        elif dataParam['geoFeatures'] == 'embed':
            _feat = np.hstack([_feature[:, cfdIdx],
                               _tdb, _tdf,
                               _feature[:, stationFeatIdx],
                               _geoFeatures])
        elif dataParam['geoFeatures'] == 'no':
            _feat = np.hstack([_feature[:, cfdIdx],
                               _tdb, _tdf,
                               _feature[:, stationFeatIdx]])
        else:
            raise RuntimeError('Geo feature option does not exist.')

        _target = targets[n].reshape(-1, 1)
        _dataset.append(
            Data(
                x=torch.FloatTensor(_feat),
                y=torch.FloatTensor(_target),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV)
        )

    # 特征索引跟踪
    _cfdFeatLen = len(cfdIdx) * (2 * _window + 1)
    _stationFeatLen = len(stationFeatIdx)
    _rawGeoFeatLen = len(rawGeoFeatIdx)
    _geoFeatLen = len(_geoFeatures.T)
    _featureLen = np.cumsum([_cfdFeatLen, _stationFeatLen, _rawGeoFeatLen, _geoFeatLen])
    _featureIdx = {
        'CFD':      np.arange(0, _featureLen[0]),
        'station':  np.arange(_featureLen[0], _featureLen[1]),
        'rawGeo':   np.arange(_featureLen[1], _featureLen[2]),
        'embedGeo': np.arange(_featureLen[2], _featureLen[3]),
    }

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
        # 供 dataGen_unlabeled 对齐维度使用
        'stationFeatLen':   _stationFeatLen,
        'rawGeoFeatLen':    _rawGeoFeatLen,
    }
    return trainLoader, validLoader, metadata, validSet


# ============================================================
# dataGen_unlabeled: 无标签数据（独立图）
# 特征维度与 dataGen 对齐（通过 zero-padding）
# ============================================================

def dataGen_unlabeled(dataParam, data, nTrn=0.75, seed=19, predMode=False,
                      labeled=False, path=None, labeled_metadata=None):
    """
    生成无标签数据的PyG数据集

    Args:
        dataParam: 数据参数字典
        data: 预处理后的无标签数据字典（含 WRFMat, CLMSMat, UrbanFeature 等）
        nTrn: 训练集比例
        seed: 随机种子
        predMode: 预测模式
        labeled: 是否有标签
        path: 有标签数据路径（用于统一 UrbanFeatureMat 归一化参数）
        labeled_metadata: dataGen 返回的 metadata（用于维度对齐）
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']

    # UrbanFeatureMat 统一归一化
    if path is not None:
        _, _off_labeled, _scl_labeled, _ = genGeoFeatures(
            path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
        )
        _geoFeatures, _off, _scl, _nStations = genGeoFeatures_unlabeled(
            data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'],
            norm_off=_off_labeled, norm_scl=_scl_labeled
        )
    else:
        _geoFeatures, _off, _scl, _nStations = genGeoFeatures_unlabeled(
            data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
        )

    # --------------------------图构建--------------------------
    Map = data['Map']
    SimilarityMat = data['SimilarityMat']
    nNodes = Map.shape[0]

    Distance = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            Distance[i, j] = np.sqrt(
                (Map[i, 0] - Map[j, 0])**2 + (Map[i, 1] - Map[j, 1])**2
            )

    Matrix = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            Cov = np.corrcoef(SimilarityMat[i, :], SimilarityMat[j, :])
            r = Cov[0, 1] if not np.isnan(Cov[0, 1]) else 0.0
            r = max(r, 0.0)
            Matrix[i, j] = r
    _simiW = Matrix

    _distW = np.exp(-Distance)
    _off_dist, _scl_dist = np.min(_distW), np.max(_distW) - np.min(_distW)
    if _scl_dist > 0:
        _distW = (_distW - _off_dist) / _scl_dist
    else:
        _distW = np.ones_like(_distW)
    Adj = np.abs(_simiW * _distW)
    Adj[Adj < dataParam['thres']] = 0.
    np.fill_diagonal(Adj, 0.)
    assert np.allclose(Adj, Adj.T)

    # --------------------------节点特征--------------------------
    WRFMat = data['WRFMat']           # (T, 54, nNodes)
    CLMSMat = data['CLMSMat']         # (T, 3, nNodes)
    UrbanFeature = data['UrbanFeature']  # (nNodes, 17)

    cfd_features = np.transpose(WRFMat, (0, 2, 1))    # (T, nNodes, 54)
    clms_features = np.transpose(CLMSMat, (0, 2, 1))  # (T, nNodes, 3)

    # 从 labeled_metadata 获取对齐维度
    if labeled_metadata is not None:
        stationFeatLen = labeled_metadata['stationFeatLen']
        rawGeoFeatLen = labeled_metadata['rawGeoFeatLen']
    else:
        # 如果没有 metadata，从有标签数据文件推算
        if path is not None:
            _raw_labeled = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
            _raw_labeled = np.transpose(_raw_labeled, (0, 2, 1))
            stationFeatLen = 4
            rawGeoFeatLen = _raw_labeled.shape[-1] - 1 - 54 - 4
        else:
            # 无法对齐，使用原始维度
            stationFeatLen = clms_features.shape[2]
            rawGeoFeatLen = UrbanFeature.shape[1]

    # 目标值（如果有标签）
    targets = data.get('AoT_obs', None) if labeled else None

    # --------------------------构建PyG数据集--------------------------
    T = len(cfd_features)
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
    _dataset = []

    for n in range(_window, T - _window):
        _cfd = cfd_features[n]
        _clms = clms_features[n]
        _tdb = cfd_features[n - _window:n]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(nNodes, -1)
        _tdf = cfd_features[n:n + _window]
        _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(nNodes, -1)

        # Pad station features: CLMSMat (3) → stationFeatLen (4)
        _station = np.zeros((nNodes, stationFeatLen))
        _station[:, :clms_features.shape[2]] = _clms

        # Pad raw geo features: UrbanFeature (17) → rawGeoFeatLen
        _rawGeo = np.zeros((nNodes, rawGeoFeatLen))
        _rawGeo[:, :UrbanFeature.shape[1]] = UrbanFeature

        if dataParam['geoFeatures'] == 'full':
            _feat = np.hstack([_cfd, _tdb, _tdf, _station, _rawGeo, _geoFeatures])
        elif dataParam['geoFeatures'] == 'raw':
            _feat = np.hstack([_cfd, _tdb, _tdf, _station, _rawGeo])
        elif dataParam['geoFeatures'] == 'embed':
            _feat = np.hstack([_cfd, _tdb, _tdf, _station, _geoFeatures])
        elif dataParam['geoFeatures'] == 'no':
            _feat = np.hstack([_cfd, _tdb, _tdf, _station])
        else:
            raise RuntimeError('Geo feature option does not exist.')

        if labeled and targets is not None:
            _target = targets[n].reshape(-1, 1)
            data_obj = Data(
                x=torch.FloatTensor(_feat),
                y=torch.FloatTensor(_target),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV
            )
        else:
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
        'oDim':      1 if labeled else None,
        'geoMethod': dataParam['geoMethod'],
        'poolSize':  dataParam['poolSize'],
        'nCompPCA':  dataParam['nCompPCA'],
        'trainIdx':  trainSet.indices,
        'validIdx':  validSet.indices,
        'AdjMatrix': Adj,
        'isLabeled': labeled,
        'UrbanFeature': UrbanFeature,
    }

    return trainLoader, validLoader, testLoader, metadata, validSet


# ============================================================
# dataGen_unified: 统一图结构（labeled + unlabeled 合并为268节点大图）
# labeled 在前（0:n_labeled），unlabeled 在后（n_labeled:total_nodes）
# ============================================================

def dataGen_unified(dataParam, path, unlabeled_data, nTrn=0.75, seed=19, predMode=False):
    """
    统一图结构：将68个有标签节点和200个无标签节点合并为268节点的大图
    每个时间步生成一个 Data 对象，包含 labeled_mask 和 unlabeled_mask

    Args:
        dataParam: 数据参数字典
        path: 有标签数据路径
        unlabeled_data: 预处理（+增强）后的无标签数据字典
        nTrn: 训练集比例
        seed: 随机种子
        predMode: 预测模式
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    n_unlabeled = unlabeled_data['Map'].shape[0]

    # 1. Geo features（统一用有标签的归一化参数）
    _geoFeatures_labeled, _off, _scl, _nStations = genGeoFeatures(
        path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
    )
    _geoFeatures_unlabeled, _, _, _ = genGeoFeatures_unlabeled(
        unlabeled_data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'],
        norm_off=_off, norm_scl=_scl
    )

    # 2. 构建统一图
    unified_adj, labeled_indices, unlabeled_indices, _ = build_unified_graph(
        dataParam, path, unlabeled_data, n_unlabeled=n_unlabeled
    )
    n_labeled = len(labeled_indices)
    total_nodes = n_labeled + n_unlabeled

    # 图结构统计
    n_ll = int(np.sum(unified_adj[:n_labeled, :n_labeled] > 0) // 2)
    n_uu = int(np.sum(unified_adj[n_labeled:, n_labeled:] > 0) // 2)
    n_lu = int(np.sum(unified_adj[:n_labeled, n_labeled:] > 0))
    degrees = np.sum(unified_adj > 0, axis=1)
    print(f"[统一图] 节点: {total_nodes} (labeled={n_labeled}, unlabeled={n_unlabeled})")
    print(f"[统一图] 边数: 总={n_ll+n_uu+n_lu} | labeled内部={n_ll} | unlabeled内部={n_uu} | 跨图={n_lu}")
    print(f"[统一图] 节点度数: min={degrees.min()} max={degrees.max()} mean={degrees.mean():.1f} | 孤立节点={np.sum(degrees==0)}")

    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(unified_adj))

    # 3. 有标签节点特征
    _raw = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    _raw = np.transpose(_raw, (0, 2, 1))      # (T, n_labeled, nFeatures)
    nCFDFeats = 54
    nStationFeats = 4
    cfdIdx = np.arange(nCFDFeats)
    stationFeatIdx = np.arange(nCFDFeats, nCFDFeats + nStationFeats)
    rawGeoFeatIdx = np.arange(nCFDFeats + nStationFeats, _raw.shape[-1] - 1)
    labeled_features = _raw[:, :, 1:]         # (T, n_labeled, nFeats)
    labeled_targets = _raw[:, :, 0]           # (T, n_labeled)
    stationFeatLen = nStationFeats
    rawGeoFeatLen = len(rawGeoFeatIdx)

    # 4. 无标签节点特征
    WRFMat = unlabeled_data['WRFMat']             # (T, 54, n_unlabeled)
    CLMSMat = unlabeled_data['CLMSMat']           # (T, 3, n_unlabeled)
    UrbanFeature = unlabeled_data['UrbanFeature']  # (n_unlabeled, 17)
    cfd_unlabeled = np.transpose(WRFMat, (0, 2, 1))    # (T, n_unlabeled, 54)
    clms_unlabeled = np.transpose(CLMSMat, (0, 2, 1))  # (T, n_unlabeled, 3)

    T = len(labeled_features)
    assert len(cfd_unlabeled) == T

    # 静态 mask（每个时间步复用）
    labeled_mask = torch.zeros(total_nodes, dtype=torch.bool)
    labeled_mask[labeled_indices] = True
    unlabeled_mask = torch.zeros(total_nodes, dtype=torch.bool)
    unlabeled_mask[unlabeled_indices] = True

    # 5. 构建 PyG 数据集
    _dataset = []
    for n in range(_window, T - _window):
        # 有标签节点特征
        _feat_l = labeled_features[n]
        _tdb_l = labeled_features[n - _window:n, :, cfdIdx]
        _tdb_l = np.transpose(_tdb_l, (1, 0, 2)).reshape(n_labeled, -1)
        _tdf_l = labeled_features[n:n + _window, :, cfdIdx]
        _tdf_l = np.transpose(_tdf_l, (1, 0, 2)).reshape(n_labeled, -1)

        if dataParam['geoFeatures'] == 'full':
            _x_l = np.hstack([_feat_l[:, cfdIdx], _tdb_l, _tdf_l,
                               _feat_l[:, stationFeatIdx], _feat_l[:, rawGeoFeatIdx],
                               _geoFeatures_labeled])
        elif dataParam['geoFeatures'] == 'raw':
            _x_l = np.hstack([_feat_l[:, cfdIdx], _tdb_l, _tdf_l,
                               _feat_l[:, stationFeatIdx], _feat_l[:, rawGeoFeatIdx]])
        elif dataParam['geoFeatures'] == 'embed':
            _x_l = np.hstack([_feat_l[:, cfdIdx], _tdb_l, _tdf_l,
                               _feat_l[:, stationFeatIdx], _geoFeatures_labeled])
        elif dataParam['geoFeatures'] == 'no':
            _x_l = np.hstack([_feat_l[:, cfdIdx], _tdb_l, _tdf_l,
                               _feat_l[:, stationFeatIdx]])
        else:
            raise RuntimeError('Geo feature option does not exist.')

        # 无标签节点特征（zero-padding 对齐 iDim）
        _cfd_u = cfd_unlabeled[n]
        _clms_u = clms_unlabeled[n]
        _tdb_u = cfd_unlabeled[n - _window:n]
        _tdb_u = np.transpose(_tdb_u, (1, 0, 2)).reshape(n_unlabeled, -1)
        _tdf_u = cfd_unlabeled[n:n + _window]
        _tdf_u = np.transpose(_tdf_u, (1, 0, 2)).reshape(n_unlabeled, -1)

        _station_u = np.zeros((n_unlabeled, stationFeatLen))
        _station_u[:, :clms_unlabeled.shape[2]] = _clms_u
        _rawGeo_u = np.zeros((n_unlabeled, rawGeoFeatLen))
        _rawGeo_u[:, :UrbanFeature.shape[1]] = UrbanFeature

        if dataParam['geoFeatures'] == 'full':
            _x_u = np.hstack([_cfd_u, _tdb_u, _tdf_u, _station_u, _rawGeo_u, _geoFeatures_unlabeled])
        elif dataParam['geoFeatures'] == 'raw':
            _x_u = np.hstack([_cfd_u, _tdb_u, _tdf_u, _station_u, _rawGeo_u])
        elif dataParam['geoFeatures'] == 'embed':
            _x_u = np.hstack([_cfd_u, _tdb_u, _tdf_u, _station_u, _geoFeatures_unlabeled])
        elif dataParam['geoFeatures'] == 'no':
            _x_u = np.hstack([_cfd_u, _tdb_u, _tdf_u, _station_u])

        # 合并节点：labeled 在前，unlabeled 在后
        _x = np.vstack([_x_l, _x_u])              # (total_nodes, iDim)
        _target = labeled_targets[n].reshape(-1, 1)  # (n_labeled, 1)

        _dataset.append(Data(
            x=torch.FloatTensor(_x),
            y=torch.FloatTensor(_target),
            edge_index=edgeIdxV,
            edge_attr=edgeAttrV,
            labeled_mask=labeled_mask,
            unlabeled_mask=unlabeled_mask,
        ))

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
        'nNodes':         _nStations,
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
        'stationFeatLen': stationFeatLen,
        'rawGeoFeatLen':  rawGeoFeatLen,
    }
    return trainLoader, validLoader, metadata, validSet


# ============================================================
# 统一图相关函数（保留，暂不使用）
# ============================================================

def build_unified_graph(dataParam, path, unlabeled_data, n_unlabeled=200, seed=42):
    """
    构建统一的图结构，包含有标签站点(68个)和无标签站点(200/500个)
    """
    # 1. 加载有标签站点的位置、距离和相似性信息
    labeled_raw = mat73.loadmat(f'{path}/GNN_N1_AJM.mat')
    labeled_locations = labeled_raw['location']
    labeled_similarity = labeled_raw['similarity']
    labeled_dist = labeled_raw['dist']
    n_labeled = labeled_locations.shape[0]

    # 2. 获取无标签站点的位置和相似性信息
    unlabeled_locations = unlabeled_data['Map']
    unlabeled_similarity_mat = unlabeled_data['SimilarityMat']

    assert unlabeled_locations.shape[0] >= n_unlabeled
    if unlabeled_locations.shape[0] > n_unlabeled:
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(
            unlabeled_locations.shape[0], size=n_unlabeled, replace=False
        )
        unlabeled_locations = unlabeled_locations[selected_indices]
        unlabeled_similarity_mat = unlabeled_similarity_mat[selected_indices]

    # 3. 合并位置信息
    unified_locations = np.vstack([labeled_locations, unlabeled_locations])
    total_nodes = n_labeled + n_unlabeled

    # 4. 计算统一的距离矩阵
    unified_distance = np.zeros((total_nodes, total_nodes))
    unified_distance[:n_labeled, :n_labeled] = labeled_dist
    for i in range(n_unlabeled):
        for j in range(n_unlabeled):
            idx_i = n_labeled + i
            idx_j = n_labeled + j
            if i != j:
                unified_distance[idx_i, idx_j] = np.sqrt(
                    (unlabeled_locations[i, 0] - unlabeled_locations[j, 0])**2 +
                    (unlabeled_locations[i, 1] - unlabeled_locations[j, 1])**2
                )
    for i in range(n_labeled):
        for j in range(n_unlabeled):
            idx_j = n_labeled + j
            dist = np.sqrt(
                (labeled_locations[i, 0] - unlabeled_locations[j, 0])**2 +
                (labeled_locations[i, 1] - unlabeled_locations[j, 1])**2
            )
            unified_distance[i, idx_j] = dist
            unified_distance[idx_j, i] = dist

    # 5. 计算统一的相似性矩阵
    unified_similarity = np.zeros((total_nodes, total_nodes))
    unified_similarity[:n_labeled, :n_labeled] = labeled_similarity
    np.fill_diagonal(unified_similarity[:n_labeled, :n_labeled], 1.0)
    for i in range(n_unlabeled):
        for j in range(n_unlabeled):
            idx_i = n_labeled + i
            idx_j = n_labeled + j
            if i != j:
                cov = np.corrcoef(unlabeled_similarity_mat[i, :], unlabeled_similarity_mat[j, :])
                r = cov[0, 1] if not np.isnan(cov[0, 1]) else 0.0
                r = max(r, 0.0)
                unified_similarity[idx_i, idx_j] = r
            else:
                unified_similarity[idx_i, idx_j] = 1.0
    max_dist = np.max(unified_distance)
    for i in range(n_labeled):
        for j in range(n_unlabeled):
            idx_j = n_labeled + j
            dist = unified_distance[i, idx_j]
            normalized_dist = dist / max_dist
            similarity = np.exp(-normalized_dist * 2.0)
            unified_similarity[i, idx_j] = similarity
            unified_similarity[idx_j, i] = similarity

    # 6. 构建邻接矩阵
    unified_dist_w = np.exp(-unified_distance)
    _off = np.min(unified_dist_w)
    _scl = np.max(unified_dist_w) - np.min(unified_dist_w)
    if _scl > 0:
        unified_dist_w = (unified_dist_w - _off) / _scl
    else:
        unified_dist_w = np.ones_like(unified_dist_w)
    unified_adj = np.abs(unified_similarity * unified_dist_w)
    unified_adj[unified_adj < dataParam['thres']] = 0.0
    # 跨图边（labeled-unlabeled）单独用更严格的阈值过滤（纯距离边，噪声较多）
    cross_thres = dataParam.get('cross_thres', 0.5)
    unified_adj[:n_labeled, n_labeled:][unified_adj[:n_labeled, n_labeled:] < cross_thres] = 0.0
    unified_adj[n_labeled:, :n_labeled][unified_adj[n_labeled:, :n_labeled] < cross_thres] = 0.0
    np.fill_diagonal(unified_adj, 0.0)

    # 7. kNN保底稀疏化
    unified_adj = apply_knn_sparsify(unified_adj, k=8)
    unified_adj = (unified_adj + unified_adj.T) / 2
    assert np.allclose(unified_adj, unified_adj.T)

    # 8. 定义索引
    labeled_indices = np.arange(n_labeled)
    unlabeled_indices = np.arange(n_labeled, total_nodes)

    return unified_adj, labeled_indices, unlabeled_indices, unified_locations


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

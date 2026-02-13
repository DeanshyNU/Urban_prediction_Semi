"""
PyG数据集生成模块
包含有标签和无标签数据的PyG数据集生成功能
"""
import numpy as np
import torch
import mat73
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from geo_features import genGeoFeatures, genGeoFeatures_unlabeled


def apply_knn_sparsify(adj: np.ndarray, k: int = 8) -> np.ndarray:
    """
    kNN保底稀疏化：确保每个节点至少保留k个最强连接
    
    Args:
        adj: 邻接矩阵
        k: 每个节点保留的最强连接数
    
    Returns:
        稀疏化后的邻接矩阵
    """
    n = adj.shape[0]
    keep = np.zeros_like(adj, dtype=bool)
    
    for i in range(n):
        row = adj[i].copy()
        row[i] = -np.inf  # 排除自环
        kk = min(k, n-1)  # 确保不超过节点数-1
        idx = np.argpartition(-row, kth=kk-1)[:kk]
        keep[i, idx] = True
    
    # 保证对称性
    mask = np.logical_or(keep, keep.T)
    out = np.zeros_like(adj)
    out[mask] = adj[mask]
    np.fill_diagonal(out, 0.0)  # 清零自环
    
    return out


def build_unified_graph(dataParam, path, unlabeled_data, n_unlabeled=200, seed=42):
    """
    构建统一的图结构，包含有标签站点(68个)和无标签站点(200/500个)
    
    Args:
        dataParam: 数据参数字典
        path: 有标签数据路径
        unlabeled_data: 无标签数据字典
        n_unlabeled: 无标签站点数量
        seed: 随机种子，用于选择无标签站点（确保可复现性）
    
    Returns:
        unified_adj: 统一的邻接矩阵
        labeled_indices: 有标签站点在统一图中的索引
        unlabeled_indices: 无标签站点在统一图中的索引
        unified_locations: 统一的位置信息
    """
    
    # 1. 加载有标签站点的位置、距离和相似性信息
    labeled_raw = mat73.loadmat(f'{path}/GNN_N1_AJM.mat')
    labeled_locations = labeled_raw['location']  # (68, 2)
    labeled_similarity = labeled_raw['similarity']  # (68, 68)
    labeled_dist = labeled_raw['dist']  # (68, 68) 预计算的距离矩阵
    n_labeled = labeled_locations.shape[0]
    
    # 2. 获取无标签站点的位置和相似性信息
    unlabeled_locations = unlabeled_data['Map']  # (n_unlabeled, 2)
    unlabeled_similarity_mat = unlabeled_data['SimilarityMat']  # (n_unlabeled, 10)
    
    # 确保无标签站点数量匹配
    assert unlabeled_locations.shape[0] >= n_unlabeled, f"无标签站点数量不足，需要{n_unlabeled}个"
    if unlabeled_locations.shape[0] > n_unlabeled:
        # 如果无标签站点数量超过需求，随机选择（使用固定seed确保可复现性）
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(
            unlabeled_locations.shape[0], 
            size=n_unlabeled, 
            replace=False
        )
        unlabeled_locations = unlabeled_locations[selected_indices]
        unlabeled_similarity_mat = unlabeled_similarity_mat[selected_indices]
    
    # 3. 合并位置信息
    unified_locations = np.vstack([labeled_locations, unlabeled_locations])
    total_nodes = n_labeled + n_unlabeled
    
    # 4. 计算统一的距离矩阵
    unified_distance = np.zeros((total_nodes, total_nodes))
    
    # 4.1 有标签站点之间：使用预计算的距离
    unified_distance[:n_labeled, :n_labeled] = labeled_dist
    
    # 4.2 无标签站点之间：计算距离
    for i in range(n_unlabeled):
        for j in range(n_unlabeled):
            idx_i = n_labeled + i
            idx_j = n_labeled + j
            if i != j:
                unified_distance[idx_i, idx_j] = np.sqrt(
                    (unlabeled_locations[i, 0] - unlabeled_locations[j, 0])**2 + 
                    (unlabeled_locations[i, 1] - unlabeled_locations[j, 1])**2
                )
            else:
                unified_distance[idx_i, idx_j] = 0.0
    
    # 4.3 有标签和无标签站点之间：计算距离
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
    
    # 5.1 有标签站点之间的相似性
    unified_similarity[:n_labeled, :n_labeled] = labeled_similarity
    np.fill_diagonal(unified_similarity[:n_labeled, :n_labeled], 1.0)
    
    # 5.2 无标签站点之间的相似性
    for i in range(n_unlabeled):
        for j in range(n_unlabeled):
            idx_i = n_labeled + i
            idx_j = n_labeled + j
            if i != j:
                cov = np.corrcoef(unlabeled_similarity_mat[i, :], unlabeled_similarity_mat[j, :])
                r = cov[0, 1] if not np.isnan(cov[0, 1]) else 0.0
                r = max(r, 0.0)  # 截断负值为0
                unified_similarity[idx_i, idx_j] = r
            else:
                unified_similarity[idx_i, idx_j] = 1.0
    
    # 5.3 有标签和无标签站点之间的相似性（基于距离）
    max_dist = np.max(unified_distance)
    for i in range(n_labeled):
        for j in range(n_unlabeled):
            idx_j = n_labeled + j
            dist = unified_distance[i, idx_j]
            normalized_dist = dist / max_dist
            similarity = np.exp(-normalized_dist * 2.0)
            unified_similarity[i, idx_j] = similarity
            unified_similarity[idx_j, i] = similarity
    
    # 6. 构建统一的邻接矩阵（使用旧逻辑：MinMax归一化）
    # 6.1 计算距离权重
    unified_dist_w = np.exp(-unified_distance)
    
    # 6.2 MinMax归一化（旧逻辑）
    _off = np.min(unified_dist_w)
    _scl = np.max(unified_dist_w) - np.min(unified_dist_w)
    if _scl > 0:
        unified_dist_w = (unified_dist_w - _off) / _scl
    else:
        unified_dist_w = np.ones_like(unified_dist_w)
    
    # 6.3 构建邻接矩阵
    unified_adj = np.abs(unified_similarity * unified_dist_w)
    unified_adj[unified_adj < dataParam['thres']] = 0.0
    np.fill_diagonal(unified_adj, 0.0)
    
    # 7. 应用kNN保底稀疏化
    unified_adj = apply_knn_sparsify(unified_adj, k=8)
    
    # 确保对称性
    unified_adj = (unified_adj + unified_adj.T) / 2
    assert np.allclose(unified_adj, unified_adj.T)
    
    # 8. 定义索引
    labeled_indices = np.arange(n_labeled)
    unlabeled_indices = np.arange(n_labeled, total_nodes)
    
    return unified_adj, labeled_indices, unlabeled_indices, unified_locations


def diagnose_graph_structure(unified_adj, labeled_indices, unlabeled_indices, 
                           unified_locations, dataParam, verbose=True):
    """
    诊断统一图结构是否合理
    """
    try:
        import networkx as nx
    except ImportError:
        print("警告: NetworkX未安装，跳过图结构诊断")
        return None
    
    n_labeled = len(labeled_indices)
    n_unlabeled = len(unlabeled_indices)
    total_nodes = n_labeled + n_unlabeled
    
    G = nx.from_numpy_array(unified_adj)
    
    # 基本统计
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = 2 * n_edges / total_nodes if total_nodes > 0 else 0
    
    # 连通性检查
    is_connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)
    largest_cc_size = len(max(nx.connected_components(G), key=len)) if n_components > 0 else 0
    
    # 子图统计
    labeled_adj = unified_adj[np.ix_(labeled_indices, labeled_indices)]
    unlabeled_adj = unified_adj[np.ix_(unlabeled_indices, unlabeled_indices)]
    G_labeled = nx.from_numpy_array(labeled_adj)
    G_unlabeled = nx.from_numpy_array(unlabeled_adj)
    labeled_density = nx.density(G_labeled)
    unlabeled_density = nx.density(G_unlabeled)
    
    # 权重分布
    edge_weights = [unified_adj[i,j] for i,j in G.edges()]
    weight_stats = {
        'min': np.min(edge_weights) if edge_weights else 0,
        'max': np.max(edge_weights) if edge_weights else 0,
        'mean': np.mean(edge_weights) if edge_weights else 0,
        'std': np.std(edge_weights) if edge_weights else 0,
        'median': np.median(edge_weights) if edge_weights else 0
    }
    
    # 度数分布
    degrees = [G.degree(n) for n in G.nodes()]
    degree_stats = {
        'min': np.min(degrees) if degrees else 0,
        'max': np.max(degrees) if degrees else 0,
        'mean': np.mean(degrees) if degrees else 0,
        'std': np.std(degrees) if degrees else 0
    }
    
    # 孤立节点检查
    isolated_nodes = list(nx.isolates(G))
    n_isolated = len(isolated_nodes)
    diagonal_sum = np.sum(np.diag(unified_adj))
    is_symmetric = np.allclose(unified_adj, unified_adj.T)
    
    # 健康度评估
    health_score = 0
    max_score = 8
    if is_connected: health_score += 1
    if n_isolated == 0: health_score += 1
    if diagonal_sum == 0: health_score += 1
    if is_symmetric: health_score += 1
    if 0.01 <= density <= 0.3: health_score += 1
    if degree_stats['min'] >= 1: health_score += 1
    if weight_stats['std'] > 0: health_score += 1
    if largest_cc_size >= total_nodes * 0.9: health_score += 1
    
    health_percentage = (health_score / max_score) * 100
    
    if verbose:
        print("=" * 60)
        print("图结构诊断报告")
        print("=" * 60)
        print(f"\n📊 基本统计:")
        print(f"  总节点数: {total_nodes} (有标签: {n_labeled}, 无标签: {n_unlabeled})")
        print(f"  总边数: {n_edges}")
        print(f"  图密度: {density:.4f}")
        print(f"  平均度数: {avg_degree:.2f}")
        print(f"\n🔗 连通性:")
        print(f"  是否连通: {'✅ 是' if is_connected else '❌ 否'}")
        print(f"  连通分量数: {n_components}")
        print(f"  最大连通分量大小: {largest_cc_size}/{total_nodes}")
        print(f"  孤立节点数: {n_isolated}")
        print(f"\n📈 子图密度:")
        print(f"  有标签子图密度: {labeled_density:.4f}")
        print(f"  无标签子图密度: {unlabeled_density:.4f}")
        print(f"\n⚖️ 权重分布:")
        print(f"  最小值: {weight_stats['min']:.6f}")
        print(f"  最大值: {weight_stats['max']:.6f}")
        print(f"  平均值: {weight_stats['mean']:.6f}")
        print(f"  中位数: {weight_stats['median']:.6f}")
        print(f"  标准差: {weight_stats['std']:.6f}")
        print(f"\n📏 度数分布:")
        print(f"  最小度数: {degree_stats['min']}")
        print(f"  最大度数: {degree_stats['max']}")
        print(f"  平均度数: {degree_stats['mean']:.2f}")
        print(f"  度数标准差: {degree_stats['std']:.2f}")
        print(f"\n🔍 结构检查:")
        print(f"  对角线元素和: {diagonal_sum:.6f} {'✅ 已清零' if diagonal_sum == 0 else '❌ 未清零'}")
        print(f"  矩阵对称性: {'✅ 对称' if is_symmetric else '❌ 不对称'}")
        print(f"\n🏥 健康度评估:")
        print(f"  健康度评分: {health_score}/{max_score} ({health_percentage:.1f}%)")
        if health_percentage >= 80:
            print("  🎉 图结构健康！")
        elif health_percentage >= 60:
            print("  ⚠️ 图结构基本正常，有改进空间")
        else:
            print("  🚨 图结构存在问题，建议调整参数")
    
    return {
        'n_nodes': total_nodes,
        'n_edges': n_edges,
        'density': density,
        'is_connected': is_connected,
        'n_components': n_components,
        'n_isolated': n_isolated,
        'weight_stats': weight_stats,
        'degree_stats': degree_stats,
        'health_score': health_score,
        'health_percentage': health_percentage
    }


def dataGen_ESTnet(dataParam, path, nTrn=0.75, predMode=False):
    """
    按照ESTNet风格生成数据
    
    静态特征：静态地理特征（不包含CLMS动态特征）
    动态特征：CFD/WRF特征（包括当前和时间窗口）+ CLMS动态特征
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    _geoFeatures, _off, _scl, _nStations = genGeoFeatures(path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])
    
    # --------------------------图构建--------------------------
    _raw = mat73.loadmat(f'{path}/GNN_N1_AJM.mat')
    _dist, _, _simiW = _raw['dist'], _raw['location'], _raw['similarity']
    _nNodes = _dist.shape[0]
    _distW = np.exp(-_dist)
    _off_dist, _scl_dist = np.min(_distW), np.max(_distW) - np.min(_distW)
    if _scl_dist > 0:
        _distW = (_distW - _off_dist) / _scl_dist
    else:
        _distW = np.ones_like(_distW)
    Adj = np.abs(_simiW * _distW)
    Adj[Adj < dataParam['thres']] = 0.0
    np.fill_diagonal(Adj, 0.0)
    assert np.allclose(Adj, Adj.T)
    
    # --------------------------节点特征--------------------------
    _raw = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    _raw = np.transpose(_raw, (0, 2, 1))  # Dim: timestep * nNodes * nFeatures
    nCFDFeats = 54  # WRF变量数量
    cfdIdx = np.arange(nCFDFeats)
    
    # 修改：正确区分静态和动态特征
    # 静态地理特征（不包括最后3个CLMS动态特征）
    rawGeoFeatIdx = np.arange(nCFDFeats + 4, _raw.shape[-1] - 3 - 1)
    # CLMS动态特征（最后3个）
    clmsIdx = np.arange(_raw.shape[-1] - 3 - 1, _raw.shape[-1] - 1)
    
    features = _raw[:, :, 1:]  # 提取除温度外的所有特征
    targets = _raw[:, :, 0]    # 提取温度作为目标
    
    # --------------------------构建PyG数据集--------------------------
    T = len(features)
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
    _dataset = []
    
    for n in range(_window, T - _window):
        _feature = features[n]
        # 处理历史时间窗口
        _tdb = features[n - _window:n, :, cfdIdx]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(len(Adj), -1)
        # 处理未来时间窗口（根据geoFeatures决定是否使用）
        if dataParam.get('geoFeatures', 'full') == 'full':
            # 'full' 模式：只使用历史窗口
            _tdf = None
        else:
            # 其他模式：使用历史和未来窗口
            _tdf = features[n:n + _window, :, cfdIdx]
            _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(len(Adj), -1)
        
        # 合并所有特征为一个统一的特征向量（参考 downscale-gnn/network.py）
        # 1. 动态特征：CFD特征（当前+时间窗口）+ CLMS动态特征
        if _tdf is not None:
            # 使用历史和未来窗口
            dynamic_features = np.hstack([
                _feature[:, cfdIdx],      # 当前CFD特征
                _tdb, _tdf,              # 历史+未来时间窗口CFD特征
                _feature[:, clmsIdx]      # CLMS动态特征
            ])
        else:
            # 只使用历史窗口
            dynamic_features = np.hstack([
                _feature[:, cfdIdx],      # 当前CFD特征
                _tdb,                    # 历史时间窗口CFD特征
                _feature[:, clmsIdx]      # CLMS动态特征
            ])
        
        # 2. 静态特征：静态地理特征
        static_features = np.hstack([
            _feature[:, rawGeoFeatIdx],  # 静态地理特征
            _geoFeatures                 # 处理后的空间特征
        ])
        
        # 3. 合并动态和静态特征为一个统一的特征向量
        combined_features = np.hstack([dynamic_features, static_features])
        
        _target = targets[n].reshape(-1, 1)
        _dataset.append(
            Data(
                x=torch.FloatTensor(combined_features),  # 使用统一的 x 而不是 x_dynamic 和 x_static
                y=torch.FloatTensor(_target),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV
            )
        )
    
    # 特征索引跟踪（统一特征）
    # 根据geoFeatures决定CFD特征长度
    if dataParam.get('geoFeatures', 'full') == 'full':
        _cfdFeatLen = len(cfdIdx) * (_window + 1)  # 当前+历史窗口（不使用未来窗口）
    else:
        _cfdFeatLen = len(cfdIdx) * (2 * _window + 1)  # 当前+历史窗口+未来窗口
    _clmsFeatLen = len(clmsIdx)  # CLMS特征长度
    _rawGeoFeatLen = len(rawGeoFeatIdx)
    # 使用.shape[1]更稳定（兼容tensor和array）
    _geoFeatLen = _geoFeatures.shape[1] if hasattr(_geoFeatures, 'shape') else len(_geoFeatures.T)
    
    # 统一特征维度
    _iDim = _cfdFeatLen + _clmsFeatLen + _rawGeoFeatLen + _geoFeatLen
    
    # 数据集分割
    _generator = torch.Generator().manual_seed(19)
    _trainLength = int(len(_dataset) * nTrn)
    _validLength = len(_dataset) - _trainLength
    trainSet, validSet = torch.utils.data.random_split(_dataset, [_trainLength, _validLength], _generator)
    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)
    
    # 记录元数据（使用统一的 iDim）
    metadata = {
        'nNodes': _nStations,
        'geoOff': _off,
        'geoScl': _scl,
        'iDim': _iDim,  # 统一特征维度
        'oDim': _target.shape[-1],
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': Adj,
        'UrbanFeature': features[0, :, rawGeoFeatIdx],  # (nNodes, 17) 静态特征取任一时间步
    }
    
    return trainLoader, validLoader, metadata, validSet


def dataGen_unlabeled_ESTnet(dataParam, data, nTrn=0.75, seed=19, predMode=False, labeled=False, path=None):
    """
    按照ESTNet风格生成无标签数据
    
    静态特征：静态地理特征（UrbanFeature，不包含CLMS）
    动态特征：CFD/WRF特征（包括当前和时间窗口）+ CLMS动态特征
    不包含辅助变量（站点ID和时间信息）
    
    归一化策略：UrbanFeatureMat（地理嵌入）必须统一归一化
    - 如果提供了 path，则从有标签数据获取归一化参数，保证特征尺度一致
    - 否则使用无标签数据自己的归一化参数（不推荐）
    
    Args:
        dataParam: 数据参数字典
        data: 无标签数据字典
        nTrn: 训练集比例
        seed: 随机种子
        predMode: 预测模式
        labeled: 是否有标签
        path: 可选。有标签数据目录路径。若提供，则使用有标签数据的归一化参数归一化 UrbanFeatureMat。
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    
    # UrbanFeatureMat 统一归一化：如果提供了 path，使用有标签数据的归一化参数
    if path is not None:
        print("  从有标签数据获取 UrbanFeatureMat 归一化参数...")
        _, _off_labeled, _scl_labeled, _ = genGeoFeatures(path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])
        _geoFeatures, _off, _scl, _nStations = genGeoFeatures_unlabeled(
            data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'],
            norm_off=_off_labeled, norm_scl=_scl_labeled  # 使用有标签数据的归一化参数
        )
    else:
        print("  ⚠️  使用无标签数据自己的归一化参数（建议传入 path 参数以统一归一化）")
        _geoFeatures, _off, _scl, _nStations = genGeoFeatures_unlabeled(data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])

    # --------------------------图构建--------------------------
    Map = data['Map']  # (nNodes, 2)
    SimilarityMat = data['SimilarityMat']  # (nNodes, 10)
    nNodes = Map.shape[0]

    # 计算距离矩阵
    Distance = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            Distance[i, j] = np.sqrt((Map[i, 0] - Map[j, 0])**2 + (Map[i, 1] - Map[j, 1])**2)

    # 计算相似性矩阵
    Matrix = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            Cov = np.corrcoef(SimilarityMat[i, :], SimilarityMat[j, :])
            r = Cov[0, 1] if not np.isnan(Cov[0, 1]) else 0.0
            r = max(r, 0.0)  # 截断负值为0
            Matrix[i, j] = r
    _simiW = Matrix

    # 构建加权邻接矩阵（与有标签数据一致：原始距离 → exp(-dist) → MinMax归一化）
    _distW = np.exp(-Distance)  # 直接对原始距离取exp，与有标签数据一致
    _off_dist, _scl_dist = np.min(_distW), np.max(_distW) - np.min(_distW)
    if _scl_dist > 0:
        _distW = (_distW - _off_dist) / _scl_dist
    else:
        _distW = np.ones_like(_distW)
    Adj = np.abs(_simiW * _distW)
    Adj[Adj < dataParam['thres']] = 0.0
    np.fill_diagonal(Adj, 0.0)
    assert np.allclose(Adj, Adj.T)

    # --------------------------节点特征--------------------------
    # 获取特征数据
    WRFMat = data['WRFMat']  # (6624, 54, nNodes)  # 注意：已经是54维
    CLMSMat = data['CLMSMat']  # (6624, 3, nNodes)
    UrbanFeature = data['UrbanFeature']  # (nNodes, 17)

    # 处理CFD特征（动态）
    cfd_features = np.transpose(WRFMat, (0, 2, 1))  # (6624, nNodes, 54)
    
    # 处理CLMS特征（动态）
    clms_features = np.transpose(CLMSMat, (0, 2, 1))  # (6624, nNodes, 3)
    
    # 处理UrbanFeature（静态）
    expanded_urban_feature = np.repeat(UrbanFeature[np.newaxis, :, :], cfd_features.shape[0], axis=0)
    
    # 准备特征索引
    nCFDFeats = cfd_features.shape[2]  # 54
    nCLMSFeats = clms_features.shape[2]  # 3
    
    # 如果有标签数据，获取目标值
    if labeled and 'AoT_obs' in data:
        targets = data['AoT_obs']
    else:
        targets = None
    
    # --------------------------构建PyG数据集--------------------------
    T = len(cfd_features)
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
    _dataset = []
    
    for n in range(_window, T - _window):
        # 获取当前CFD特征
        _cfd_feature = cfd_features[n]
        # 获取当前CLMS特征
        _clms_feature = clms_features[n]
        
        # 处理历史时间窗口
        _tdb = cfd_features[n - _window:n]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(len(Adj), -1)
        
        # 处理未来时间窗口（根据geoFeatures决定是否使用）
        if dataParam.get('geoFeatures', 'full') == 'full':
            # 'full' 模式：只使用历史窗口
            _tdf = None
        else:
            # 其他模式：使用历史和未来窗口
            _tdf = cfd_features[n:n + _window]
            _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(len(Adj), -1)
        
        # 获取当前静态特征
        _urban_feature = expanded_urban_feature[n]
        
        # 合并所有特征为一个统一的特征向量（参考 downscale-gnn/network.py）
        # 1. 动态特征：当前CFD + 历史和未来CFD窗口 + CLMS
        if _tdf is not None:
            # 使用历史和未来窗口
            dynamic_features = np.hstack([
                _cfd_feature,  # 当前CFD特征
                _tdb, _tdf,    # 历史+未来时间窗口CFD特征
                _clms_feature  # CLMS特征
            ])
        else:
            # 只使用历史窗口
            dynamic_features = np.hstack([
                _cfd_feature,  # 当前CFD特征
                _tdb,         # 历史时间窗口CFD特征
                _clms_feature  # CLMS特征
            ])
        
        # 2. 静态特征：UrbanFeature + 空间嵌入特征
        static_features = np.hstack([
            _urban_feature,  # UrbanFeature
            _geoFeatures     # 空间嵌入特征
        ])
        
        # 3. 合并动态和静态特征为一个统一的特征向量
        combined_features = np.hstack([dynamic_features, static_features])
        
        # 创建Data对象（使用统一的 x）
        if labeled and targets is not None:
            _target = targets[n].reshape(-1, 1)
            data_obj = Data(
                x=torch.FloatTensor(combined_features),  # 使用统一的 x
                y=torch.FloatTensor(_target),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV
            )
        else:
            data_obj = Data(
                x=torch.FloatTensor(combined_features),  # 使用统一的 x
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV
            )
        
        _dataset.append(data_obj)
    
    # 计算统一特征维度
    # 根据geoFeatures决定CFD特征长度
    if dataParam.get('geoFeatures', 'full') == 'full':
        _cfdFeatLen = nCFDFeats * (_window + 1)  # 当前+历史窗口（不使用未来窗口）
    else:
        _cfdFeatLen = nCFDFeats * (2 * _window + 1)  # 当前+历史窗口+未来窗口
    _clmsFeatLen = nCLMSFeats
    _urbanFeatLen = UrbanFeature.shape[1]
    _geoFeatLen = _geoFeatures.shape[1]
    
    # 统一特征维度
    _iDim = _cfdFeatLen + _clmsFeatLen + _urbanFeatLen + _geoFeatLen
    
    # 数据集分割
    _generator = torch.Generator().manual_seed(seed)
    _trainLength = int(len(_dataset) * nTrn)
    _validLength = int(len(_dataset) * 0.2)
    _testLength = len(_dataset) - _validLength - _trainLength
    
    trainSet, validSet, testSet = torch.utils.data.random_split(_dataset, [_trainLength, _validLength, _testLength], _generator)
    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)
    testLoader = DataLoader(testSet, batch_size=len(testSet), shuffle=False)
    
    # 记录元数据（使用统一的 iDim）
    metadata = {
        'nNodes': _nStations,
        'geoOff': _off,
        'geoScl': _scl,
        'iDim': _iDim,  # 统一特征维度
        'oDim': 1 if labeled else None,
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': Adj,
        'isLabeled': labeled,
        'UrbanFeature': UrbanFeature,  # (nNodes, 17) 已在第519行定义
    }
    
    return trainLoader, validLoader, testLoader, metadata, validSet


import numpy as np
import torch,pickle,os,mat73,utils
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA

def merge_wind_components(data):
    """
    合并X和Y方向的风速为单一的风速大小特征
    """
    WRFMat = data['WRFMat'].copy()  # 复制以免修改原始数据
    
    # 第6个变量是X方向风速(索引45-53)，第7个变量是Y方向风速(索引54-62)
    # 每个变量有9个网格点
    wind_x = WRFMat[:, 45:54, :]  # X方向风速
    wind_y = WRFMat[:, 54:63, :]  # Y方向风速
    
    # 计算风速大小(矢量模)
    wind_magnitude = np.sqrt(wind_x**2 + wind_y**2)
    
    # 用风速大小替换X方向风速
    WRFMat[:, 45:54, :] = wind_magnitude
    
    # 创建新的WRFMat，只保留前54个特征(删除Y方向风速)
    new_WRFMat = WRFMat[:, :54, :]
    
    # 更新数据
    data['WRFMat'] = new_WRFMat
    return data

def genGeoFeatures(path,geoMethod='average',poolSize=15,nCompPCA=40):
    # --------------------------Geo features--------------------------
    _raw = mat73.loadmat(f'{path}/FeaturePatch_401.mat')['FeatureMat_zeros']
    print(f"  有标签原始特征数: {_raw.shape[2]}")
    
    # Remove 5th dimension because all stations are land based
    _idx = np.arange(_raw.shape[2])
    _idx = np.delete(_idx,4)
    _raw = _raw[:,:,_idx,:]
    _imageSize,_,_nFeatures,_nStations = _raw.shape
    print(f"  删除第5维后特征数: {_nFeatures}")
    
    if geoMethod == 'average':     
        # Min-max normalization
        _norm = np.transpose(_raw,(2,0,1,3)).reshape(_nFeatures,-1)
        _min,_max = np.min(_norm,axis=1), np.max(_norm,axis=1)
        _max[_max==0] = 1e-5
        _off = _min
        _scl = _max-_min
        _scl[_scl == 0] = 1e-5  # 避免除以0产生NaN
        _norm = np.transpose(_raw,(0,1,3,2))
        _norm = (_norm-_off)/_scl
        _geoFeatures = np.transpose(_norm,(2,3,0,1))
        # Average pooling 
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        _avgPool = torch.nn.AdaptiveAvgPool2d((poolSize,poolSize))
        _geoFeatures = _avgPool(_geoFeatures).reshape(_nStations,-1)

    if geoMethod == 'pca':       
        # Mean-std normalization  
        _norm = np.transpose(_raw,(2,0,1,3)).reshape(_nFeatures,-1)
        _off,_scl = np.mean(_norm,axis=1), np.std(_norm,axis=1)
        _scl[_scl == 0] = 1e-5  # 避免除以0产生NaN
        _norm = np.transpose(_raw,(0,1,3,2))
        _norm = (_norm-_off)/_scl
        _geoFeatures = np.transpose(_norm,(2,3,0,1))
        # PCA
        _geo2D = _geoFeatures.reshape(_nStations,-1)
        _pca = PCA(n_components=nCompPCA)
        _geoFeatures = _pca.fit_transform(_geo2D)
        _geo_min, _geo_max = _geoFeatures.min(), _geoFeatures.max()
        _geo_range = _geo_max - _geo_min
        if _geo_range == 0:
            _geoFeatures = np.zeros_like(_geoFeatures)  # 如果所有值相同，设为0
        else:
            _geoFeatures = (_geoFeatures - _geo_min) / _geo_range
        _geoFeatures = torch.FloatTensor(_geoFeatures)
    
    return _geoFeatures,_off,_scl,_nStations

def genGeoFeatures_unlabeled(unlabeled_data, geoMethod='average', poolSize=15, nCompPCA=40, 
                              norm_off=None, norm_scl=None):
    """
    为无标签数据生成地理特征
    如果提供了 norm_off 和 norm_scl，则使用这些参数进行归一化（与有标签数据一致）
    """
    _raw = unlabeled_data['UrbanFeatureMat']  # (401, 401, 7, nStations)
    print(f"  无标签原始特征数: {_raw.shape[2]}")
    
    # 无标签数据不需要删除第5维，保持原始7个特征
    _imageSize, _, _nFeatures, _nStations = _raw.shape
    print(f"  使用特征数: {_nFeatures}")
    
    # 归一化参数：如果提供了则使用，否则自己计算
    use_provided_norm = (norm_off is not None and norm_scl is not None)
    
    if geoMethod == 'average':
        if use_provided_norm:
            _off = norm_off
            _scl = norm_scl
            print(f"  ✓ 使用有标签数据的归一化参数")
        else:
            _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
            _min, _max = np.min(_norm, axis=1), np.max(_norm, axis=1)
            _max[_max == 0] = 1e-5
            _off = _min
            _scl = _max - _min
            _scl[_scl == 0] = 1e-5  # 避免除以0产生NaN
            print(f"  ⚠️  使用无标签数据自己的归一化参数")
        
        # 归一化前统计
        _raw_flat = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        print(f"  归一化前统计: min=[{np.min(_raw_flat, axis=1)}], max=[{np.max(_raw_flat, axis=1)}]")
        
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        # ✅ 添加clip，限制到[0,1]范围
        _norm = np.clip(_norm, 0.0, 1.0)
        
        # 归一化后统计
        _norm_flat = np.transpose(_norm, (2, 3, 0, 1)).reshape(_nStations, -1)
        print(f"  归一化后统计（clip后）: min={np.min(_norm_flat):.6f}, max={np.max(_norm_flat):.6f}, mean={np.mean(_norm_flat):.6f}")
        
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        _avgPool = torch.nn.AdaptiveAvgPool2d((poolSize, poolSize))
        _geoFeatures = _avgPool(_geoFeatures).reshape(_nStations, -1)
    
    elif geoMethod == 'pca':
        if use_provided_norm:
            _off = norm_off
            _scl = norm_scl
            print(f"  ✓ 使用有标签数据的归一化参数")
        else:
            _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
            _off, _scl = np.mean(_norm, axis=1), np.std(_norm, axis=1)
            _scl[_scl == 0] = 1e-5  # 避免除以0产生NaN
            print(f"  ⚠️  使用无标签数据自己的归一化参数")
        
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        _geo2D = _geoFeatures.reshape(_nStations, -1)
        _pca = PCA(n_components=nCompPCA)
        _geoFeatures = _pca.fit_transform(_geo2D)
        _geo_min, _geo_max = _geoFeatures.min(), _geoFeatures.max()
        _geo_range = _geo_max - _geo_min
        if _geo_range == 0:
            _geoFeatures = np.zeros_like(_geoFeatures)  # 如果所有值相同，设为0
        else:
            _geoFeatures = (_geoFeatures - _geo_min) / _geo_range
        _geoFeatures = torch.FloatTensor(_geoFeatures)
    
    return _geoFeatures, _off, _scl, _nStations

def dataGen(dataParam,path,nTrn=0.75,predMode=False,n_unlabeled=200):
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    
    # --------------------------加载无标签数据--------------------------
    print("正在加载无标签数据...")
    unlabeled_data = mat73.loadmat(f'{path}/Unlabeled_Finalized.mat')
    
    # 首先获取有标签站点数量
    _raw_labeled_temp = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    # 原始形状是 (timestep, nFeatures, nNodes)，所以节点数在shape[2]
    _nNodes_labeled = _raw_labeled_temp.shape[2]  # 68
    
    # 数据源2的前68个站点对应有标签站点，从第68个开始选择n_unlabeled个无标签站点
    start_idx = _nNodes_labeled
    end_idx = start_idx + n_unlabeled
    
    # 检查并合并风速（如果WRFMat是63维，需要合并为54维）
    if unlabeled_data['WRFMat'].shape[1] == 63:
        print("检测到WRFMat为63维，正在合并风速分量...")
        # 创建临时数据字典用于合并风速
        temp_data = {'WRFMat': unlabeled_data['WRFMat'][:, :, start_idx:end_idx]}
        temp_data = merge_wind_components(temp_data)
        WRFMat_unlabeled = temp_data['WRFMat']  # (6624, 54, n_unlabeled)
    else:
        WRFMat_unlabeled = unlabeled_data['WRFMat'][:, :, start_idx:end_idx]  # (6624, 54, n_unlabeled)
    CLMSMat_unlabeled = unlabeled_data['CLMSMat'][:, :, start_idx:end_idx]  # (6624, 3, n_unlabeled)
    UrbanFeature_unlabeled = unlabeled_data['UrbanFeature'][start_idx:end_idx, :]  # (n_unlabeled, 17)
    UrbanFeatureMat_unlabeled = unlabeled_data['UrbanFeatureMat'][:, :, :, start_idx:end_idx]  # (401, 401, 7, n_unlabeled)
    
    # 在使用前将UrbanFeatureMat中的NaN替换为0
    UrbanFeatureMat_unlabeled = np.nan_to_num(UrbanFeatureMat_unlabeled, nan=0.0)
    
    # 归一化无标签数据的WRF和CLMS特征（与有标签数据范围一致）
    print("\n正在归一化无标签数据的WRF和CLMS特征...")
    print(f"  WRFMat归一化前: min={np.min(WRFMat_unlabeled):.6f}, max={np.max(WRFMat_unlabeled):.6f}, mean={np.mean(WRFMat_unlabeled):.6f}")
    print(f"  CLMSMat归一化前: min={np.min(CLMSMat_unlabeled):.6f}, max={np.max(CLMSMat_unlabeled):.6f}, mean={np.mean(CLMSMat_unlabeled):.6f}")
    WRFMat_unlabeled, _, _ = utils.MinMax(WRFMat_unlabeled)
    CLMSMat_unlabeled, _, _ = utils.MinMax(CLMSMat_unlabeled)
    print(f"  WRFMat归一化后: min={np.min(WRFMat_unlabeled):.6f}, max={np.max(WRFMat_unlabeled):.6f}, mean={np.mean(WRFMat_unlabeled):.6f}")
    print(f"  CLMSMat归一化后: min={np.min(CLMSMat_unlabeled):.6f}, max={np.max(CLMSMat_unlabeled):.6f}, mean={np.mean(CLMSMat_unlabeled):.6f}")
    
    # 为无标签数据生成地理特征
    # --------------------------加载有标签数据--------------------------
    print("正在加载有标签数据...")
    print(f"  使用参数: poolSize={dataParam['poolSize']}, geoMethod={dataParam['geoMethod']}")
    _geoFeatures_labeled, _off, _scl, _nStations_labeled = genGeoFeatures(
        path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
    )
    
    # 打印有标签数据归一化参数
    print(f"\n有标签数据归一化参数:")
    print(f"  _off: min={np.min(_off):.6f}, max={np.max(_off):.6f}, mean={np.mean(_off):.6f}")
    print(f"  _scl: min={np.min(_scl):.6f}, max={np.max(_scl):.6f}, mean={np.mean(_scl):.6f}")
    
    # 无标签数据使用有标签数据的归一化参数（关键修复）
    unlabeled_subset = {'UrbanFeatureMat': UrbanFeatureMat_unlabeled}
    print(f"\n正在生成无标签数据地理特征...")
    print(f"  使用参数: poolSize={dataParam['poolSize']}, geoMethod={dataParam['geoMethod']}")
    _geoFeatures_unlabeled, _off_unlabeled, _scl_unlabeled, _ = genGeoFeatures_unlabeled(
        unlabeled_subset, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'],
        norm_off=_off, norm_scl=_scl  # 使用有标签数据的归一化参数
    )
    
    # 验证归一化一致性
    print(f"\n归一化一致性检查:")
    print(f"  有标签 _off: shape={_off.shape}, mean={np.mean(_off):.6f}")
    print(f"  无标签 _off: shape={_off_unlabeled.shape}, mean={np.mean(_off_unlabeled):.6f}")
    print(f"  参数是否一致: {np.allclose(_off, _off_unlabeled)}")
    
    # 转换为numpy array以便后续处理
    if torch.is_tensor(_geoFeatures_labeled):
        _geoFeatures_labeled = _geoFeatures_labeled.numpy()
    if torch.is_tensor(_geoFeatures_unlabeled):
        _geoFeatures_unlabeled = _geoFeatures_unlabeled.numpy()
    
    print(f"地理嵌入特征维度:")
    print(f"  有标签: {_geoFeatures_labeled.shape}")
    print(f"  无标签: {_geoFeatures_unlabeled.shape}")
    
    # 验证维度是否一致
    if _geoFeatures_labeled.shape[1] != _geoFeatures_unlabeled.shape[1]:
        raise ValueError(
            f"地理嵌入特征维度不一致！有标签: {_geoFeatures_labeled.shape[1]}维, "
            f"无标签: {_geoFeatures_unlabeled.shape[1]}维"
        )
    
    # --------------------------构建统一图--------------------------
    print("正在构建统一图...")
    
    # 获取有标签站点数量
    _raw_labeled = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    # 原始形状是 (timestep, nFeatures, nNodes)，所以节点数在shape[2]
    _nNodes_labeled = _raw_labeled.shape[2]  # 68
    
    # 从数据源2中找到前68个站点作为有标签站点的对应位置
    # 这68个站点对应数据源1的有标签站点
    Map_labeled_from_unlabeled = unlabeled_data['Map'][:_nNodes_labeled, :]  # (68, 2)
    SimilarityMat_labeled_from_unlabeled = unlabeled_data['SimilarityMat'][:_nNodes_labeled, :]  # (68, 10)
    
    # 从第68个站点开始选择n_unlabeled个无标签站点
    Map_unlabeled_subset = unlabeled_data['Map'][_nNodes_labeled:_nNodes_labeled+n_unlabeled, :]  # (n_unlabeled, 2)
    SimilarityMat_unlabeled_subset = unlabeled_data['SimilarityMat'][_nNodes_labeled:_nNodes_labeled+n_unlabeled, :]  # (n_unlabeled, 10)
    
    # 合并所有268个节点的Map和SimilarityMat
    _nNodes_total = _nNodes_labeled + n_unlabeled
    Map_all = np.vstack([Map_labeled_from_unlabeled, Map_unlabeled_subset])  # (268, 2)
    SimilarityMat_all = np.vstack([SimilarityMat_labeled_from_unlabeled, SimilarityMat_unlabeled_subset])  # (268, 10)
    
    # 按照Yu et al (2024)的方法计算统一的邻接矩阵
    print("正在计算距离矩阵...")
    Distance = np.zeros((_nNodes_total, _nNodes_total))
    for i in range(_nNodes_total):
        for j in range(_nNodes_total):
            Distance[i, j] = np.sqrt(
                (Map_all[i, 0] - Map_all[j, 0])**2 + 
                (Map_all[i, 1] - Map_all[j, 1])**2
            )
    
    print("正在计算相似性矩阵...")
    Similarity = np.zeros((_nNodes_total, _nNodes_total))
    for i in range(_nNodes_total):
        for j in range(_nNodes_total):
            Cov = np.corrcoef(SimilarityMat_all[i, :], SimilarityMat_all[j, :])
            r = Cov[0, 1] if not np.isnan(Cov[0, 1]) else 0.0
            r = max(r, 0.0)  # 截断负值为0
            Similarity[i, j] = r
    
    # 构建邻接矩阵：Adj = Similarity * exp(-Distance)
    print("正在构建邻接矩阵...")
    # 归一化Distance矩阵到0-1范围
    max_dist = Distance.max()
    Distance_normalized = Distance / max_dist
    Adj = Similarity * np.exp(-Distance_normalized)
    
    # 阈值稀疏化
    Adj[Adj < dataParam['thres']] = 0.0
    np.fill_diagonal(Adj, 0.0)
    
    # 计算边密度和统计信息
    n_edges_after = np.sum(Adj > 0)
    edge_density = n_edges_after / (Adj.shape[0] * (Adj.shape[0] - 1)) * 100
    print(f"图边密度: {edge_density:.2f}% ({n_edges_after}条边)")
    
    # 边权重统计
    edge_weights = Adj[Adj > 0]
    if len(edge_weights) > 0:
        print(f"边权重统计: min={np.min(edge_weights):.6f}, max={np.max(edge_weights):.6f}, mean={np.mean(edge_weights):.6f}")
    
    # 检查有标签↔无标签之间的边
    labeled_indices = np.arange(_nNodes_labeled)
    unlabeled_indices = np.arange(_nNodes_labeled, _nNodes_total)
    cross_edges = np.sum(Adj[np.ix_(labeled_indices, unlabeled_indices)] > 0)
    print(f"有标签↔无标签边数: {cross_edges}")
    
    # 检查孤立节点
    node_degrees = np.sum(Adj > 0, axis=1)
    isolated_nodes = np.sum(node_degrees == 0)
    if isolated_nodes > 0:
        print(f"⚠️  警告: 发现{isolated_nodes}个孤立节点")
    
    # 确保对称性
    assert np.allclose(Adj, Adj.T)
    
    # --------------------------处理有标签数据的特征--------------------------
    _raw = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    _raw = np.transpose(_raw, (0, 2, 1))  # (timestep, nNodes, nFeatures)
    
    nCFDFeats = 54
    cfdIdx = np.arange(nCFDFeats)
    # 静态地理特征（不包括最后3个CLMS动态特征）
    rawGeoFeatIdx = np.arange(nCFDFeats + 4, _raw.shape[-1] - 3 - 1)
    # CLMS动态特征
    clmsIdx_labeled = np.arange(_raw.shape[-1] - 3 - 1, _raw.shape[-1] - 1)
    
    features_labeled = _raw[:, :, 1:]
    targets_labeled = _raw[:, :, 0]
    
    # --------------------------处理无标签数据的特征--------------------------
    # 转置为 (timestep, nNodes, nFeatures)
    cfd_features_unlabeled = np.transpose(WRFMat_unlabeled, (0, 2, 1))  # (6624, n_unlabeled, 54)
    clms_features_unlabeled = np.transpose(CLMSMat_unlabeled, (0, 2, 1))  # (6624, n_unlabeled, 3)
    
    # 对齐时间步
    T_labeled = len(features_labeled)
    T_unlabeled = len(cfd_features_unlabeled)
    T = min(T_labeled, T_unlabeled)
    
    features_labeled = features_labeled[:T]
    targets_labeled = targets_labeled[:T]
    cfd_features_unlabeled = cfd_features_unlabeled[:T]
    clms_features_unlabeled = clms_features_unlabeled[:T]
    
    # -----------------构建PyG数据集 ---------------------------
    # 确保对称性
    assert np.allclose(Adj, Adj.T)
    
    # 转换Adj矩阵为PyG格式
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
    
    _dataset = []
    
    # 创建标签掩码
    label_mask = np.zeros(_nNodes_total, dtype=bool)
    label_mask[:_nNodes_labeled] = True
    
    for n in range(_window, T - _window):
        # ========== 有标签节点的特征 ==========
        _feature_labeled = features_labeled[n]
        # 历史时间窗口
        _tdb_labeled = features_labeled[n - _window:n, :, cfdIdx]
        _tdb_labeled = np.transpose(_tdb_labeled, (1, 0, 2)).reshape(_nNodes_labeled, -1)
        # 未来时间窗口（根据geoFeatures决定）
        if dataParam.get('geoFeatures', 'full') == 'full':
            _tdf_labeled = None
        else:
            _tdf_labeled = features_labeled[n:n + _window, :, cfdIdx]
            _tdf_labeled = np.transpose(_tdf_labeled, (1, 0, 2)).reshape(_nNodes_labeled, -1)
        
        # ========== 无标签节点的特征 ==========
        _cfd_unlabeled = cfd_features_unlabeled[n]
        _clms_unlabeled = clms_features_unlabeled[n]
        # 历史时间窗口
        _tdb_unlabeled = cfd_features_unlabeled[n - _window:n]
        _tdb_unlabeled = np.transpose(_tdb_unlabeled, (1, 0, 2)).reshape(n_unlabeled, -1)
        # 未来时间窗口
        if dataParam.get('geoFeatures', 'full') == 'full':
            _tdf_unlabeled = None
        else:
            _tdf_unlabeled = cfd_features_unlabeled[n:n + _window]
            _tdf_unlabeled = np.transpose(_tdf_unlabeled, (1, 0, 2)).reshape(n_unlabeled, -1)
        
        # ========== 组合特征（参考dataGen_ESTnet的逻辑）==========
        # 有标签节点特征组合
        if _tdf_labeled is not None:
            feature_labeled_combined = np.hstack([
                _feature_labeled[:, cfdIdx],      # 当前CFD特征
                _tdb_labeled, _tdf_labeled,       # 历史+未来时间窗口
                _feature_labeled[:, clmsIdx_labeled],  # CLMS动态特征
                _feature_labeled[:, rawGeoFeatIdx],    # 静态地理特征
                _geoFeatures_labeled                   # 空间嵌入特征
            ])
        else:
            feature_labeled_combined = np.hstack([
                _feature_labeled[:, cfdIdx],      # 当前CFD特征
                _tdb_labeled,                     # 历史时间窗口
                _feature_labeled[:, clmsIdx_labeled],  # CLMS动态特征
                _feature_labeled[:, rawGeoFeatIdx],    # 静态地理特征
                _geoFeatures_labeled                   # 空间嵌入特征
            ])
        
        # 无标签节点特征组合
        # 扩展UrbanFeature为与有标签数据相同的特征空间
        urban_feature_expanded = np.zeros((n_unlabeled, len(rawGeoFeatIdx)))
        urban_feature_expanded[:, :UrbanFeature_unlabeled.shape[1]] = UrbanFeature_unlabeled
        
        if _tdf_unlabeled is not None:
            feature_unlabeled_combined = np.hstack([
                _cfd_unlabeled,                    # 当前CFD特征
                _tdb_unlabeled, _tdf_unlabeled,   # 历史+未来时间窗口
                _clms_unlabeled,                   # CLMS动态特征
                urban_feature_expanded,            # 静态地理特征（对齐维度）
                _geoFeatures_unlabeled             # 空间嵌入特征
            ])
        else:
            feature_unlabeled_combined = np.hstack([
                _cfd_unlabeled,                    # 当前CFD特征
                _tdb_unlabeled,                    # 历史时间窗口
                _clms_unlabeled,                   # CLMS动态特征
                urban_feature_expanded,            # 静态地理特征（对齐维度）
                _geoFeatures_unlabeled             # 空间嵌入特征
            ])
        
        # 合并所有节点的特征
        features_all = np.vstack([feature_labeled_combined, feature_unlabeled_combined])
        
        # 目标值（只有有标签节点有目标）
        targets_all = np.concatenate([
            targets_labeled[n],
            np.zeros(n_unlabeled)
        ]).reshape(-1, 1)
        
        _dataset.append(
            Data(
                x=torch.FloatTensor(features_all),
                y=torch.FloatTensor(targets_all),
                label_mask=torch.BoolTensor(label_mask),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV
            )
        )
    
    # 特征索引跟踪
    if dataParam.get('geoFeatures', 'full') == 'full':
        _cfdFeatLen = len(cfdIdx) * (_window + 1)  # 当前+历史窗口
    else:
        _cfdFeatLen = len(cfdIdx) * (2 * _window + 1)  # 当前+历史+未来窗口
    _clmsFeatLen = len(clmsIdx_labeled)
    _rawGeoFeatLen = len(rawGeoFeatIdx)
    _geoFeatLen = _geoFeatures_labeled.shape[1]
    _featureLen = np.cumsum([_cfdFeatLen, _clmsFeatLen, _rawGeoFeatLen, _geoFeatLen])
    _featureIdx = {
        'CFD':      np.arange(0, _featureLen[0]),
        'CLMS':     np.arange(_featureLen[0], _featureLen[1]),
        'rawGeo':   np.arange(_featureLen[1], _featureLen[2]),
        'embedGeo': np.arange(_featureLen[2], _featureLen[3]),
    }

    # 数据集分割
    _generator = torch.Generator().manual_seed(19)
    _trainLength = int(len(_dataset)*nTrn)
    _validLength = len(_dataset)-_trainLength  
    trainSet, validSet = torch.utils.data.random_split(_dataset,[_trainLength,_validLength],_generator)
    trainLoader = DataLoader(trainSet,batch_size=_batchSize,shuffle=not predMode) 
    # 验证集使用与训练集相同的batch_size，避免OOM
    valid_batch_size = min(_batchSize, len(validSet))  # 不超过验证集大小
    validLoader = DataLoader(validSet,batch_size=valid_batch_size,shuffle=False)

    metadata = {
        'nNodes':    _nNodes_total,  # 总节点数268
        'nNodes_labeled': _nNodes_labeled,  # 有标签节点数68
        'nNodes_unlabeled': n_unlabeled,  # 无标签节点数200
        'geoOff':    _off,
        'geoScl':    _scl,
        'iDim':      features_all.shape[-1],
        'oDim':      targets_all.shape[-1],
        'featureIdx':_featureIdx,
        'geoMethod': dataParam['geoMethod'],
        'poolSize':  dataParam['poolSize'],
        'nCompPCA':  dataParam['nCompPCA'],
        'trainIdx':  trainSet.indices,
        'validIdx':  validSet.indices,
        'AdjMatrix': Adj,
        'label_mask': label_mask,  # 添加标签掩码
    }
    return trainLoader, validLoader, metadata, validSet

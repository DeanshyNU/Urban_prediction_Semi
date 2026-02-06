import numpy as np
import matplotlib.pyplot as plt
import os, torch
# changes
import mat73
unlabeled_file = 'data/Unlabeled_Finalized.mat'
unlabeled_data = mat73.loadmat(unlabeled_file)

def merge_wind_components(data):
    """合并X和Y方向的风速为单一的风速大小特征"""
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

# 把风速X和Y方向的风速合并为单一的风速大小特征
unlabeled_data = merge_wind_components(unlabeled_data)

# 不使用全部的无标签数据
def reduce_stations(unlabeled_data, target_station_count=500):

    original_station_count = unlabeled_data['Map'].shape[0]

    # 确保目标站点数量小于或等于原始站点数量
    assert target_station_count <= original_station_count, "目标站点数量不能大于原始站点数量"

    # 随机选择要保留的站点索引
    selected_indices = np.random.choice(
        np.arange(original_station_count),
        size=target_station_count,
        replace=False
    )

    # 创建新的字典用于存储削减后的数据
    reduced_data = {}

    # 遍历字典中的每个变量，按索引提取数据
    for key, value in unlabeled_data.items():
        if key == 'CLMSMat':
            # CLMSMat: (6624, 3, 2000)
            reduced_data[key] = value[:, :, selected_indices]
        elif key == 'WRFMat':
            # WRFMat: (6624, 63, 2000)
            reduced_data[key] = value[:, :, selected_indices]
        elif key == 'auxiliary_variables':
            # auxiliary_variables: (6624, 2000, 4)
            reduced_data[key] = value[:, selected_indices, :]
        elif key in ['Map', 'NodeLocation', 'UrbanFeature', 'SimilarityMat']:
            # Map, NodeLocation: (2000, 2)
            # UrbanFeature: (2000, 17)
            # SimilarityMat: (2000, 10)
            reduced_data[key] = value[selected_indices]
        elif key == 'UrbanFeatureMat':
            # UrbanFeatureMat: (401, 401, 7, 2000)
            reduced_data[key] = value[:, :, :, selected_indices]
        else:
            # 未识别的变量直接保留
            reduced_data[key] = value

    return reduced_data

unlabeled_data = reduce_stations(unlabeled_data)

# 修补unlabeled data
if 'UrbanFeatureMat' in unlabeled_data:
    print("正在处理 'UrbanFeatureMat'...")
    # 获取 UrbanFeatureMat
    urban_feature_mat = unlabeled_data['UrbanFeatureMat']
    # 将 NaN 替换为 0
    unlabeled_data['UrbanFeatureMat'] = np.nan_to_num(urban_feature_mat, nan=0.0)
    print("'UrbanFeatureMat' 中的 NaN 值已全部替换为 0")
else:
    print("未找到 'UrbanFeatureMat' 变量！")

def add_auxiliary_variables(data, nStations, nTimesteps):
    # Generate auxiliary variables
    hours = np.tile(np.arange(24), int(nTimesteps / 24))
    months = np.concatenate([
        np.repeat(5, 31 * 24),  # May
        np.repeat(6, 30 * 24),  # June
        np.repeat(7, 31 * 24),  # July
        np.repeat(8, 31 * 24),  # August
        np.repeat(9, 30 * 24),  # September
        np.repeat(5, 31 * 24),  # May (next year)
        np.repeat(6, 30 * 24),  # June (next year)
        np.repeat(7, 31 * 24),  # July (next year)
        np.repeat(8, 31 * 24)   # August (next year)
    ])
    years = np.concatenate([
        np.repeat(2018, 3672),  # First year (May to September)
        np.repeat(2019, 2952)   # Second year (May to August)
    ])
    station_numbers = np.arange(1, nStations + 1)

    # Normalize auxiliary variables
    hours = hours / 23.0
    months = (months - 1) / 11.0

    # Repeat auxiliary variables for all stations
    hour_features = np.repeat(hours[:, np.newaxis], nStations, axis=1)
    month_features = np.repeat(months[:, np.newaxis], nStations, axis=1)
    year_features = np.repeat(years[:, np.newaxis], nStations, axis=1)
    station_features = np.tile(station_numbers, (nTimesteps, 1))

    # Combine all features
    auxiliary_variables = np.stack(
        [hour_features, month_features, year_features, station_features], axis=2
    )
    return auxiliary_variables

# 注意这里的station的数量应该匹配上面的station的数量
# Add auxiliary variables to unlabeled_data
unlabeled_data["auxiliary_variables"] = add_auxiliary_variables(
    unlabeled_data, nStations=500, nTimesteps=6624
)

def plotPrediction(data,modelName):
    _truth,_pred,_rmse = data
    plt.figure(figsize=(10,5))
    plt.plot(_pred,'-r',label='Prediction')
    plt.plot(_truth,'--k',label='Truth')
    plt.title(f'{modelName}, RMSE {_rmse:1.4f}')
    plt.xlim([0,3000])
    plt.ylim([0,1])
    plt.ylabel("Normalized Tobs", fontsize=20)
    plt.legend(loc='upper right')
    plt.savefig(f'./{modelName}_prediction.png',dpi=200)
    plt.close()

def plotHist(hist,modelName):
    _,_ax = plt.subplots(1,2,figsize=(20,10))
    _trainLoss,_validLoss,_trainRMSE,_validRMSE  = np.array(hist).T
    _ax[0].semilogy(_trainLoss,label='Training')
    _ax[0].semilogy(_validLoss,label='Validation')
    _ax[0].set_title("Loss")
    _ax[0].legend('upper right')
    _ax[1].semilogy(_trainRMSE,label="Training")
    _ax[1].semilogy(_validRMSE,label="Validation")
    _ax[1].set_title("Station-wised Mean RMSE")
    _ax[1].legend('upper right')
    plt.savefig(f'./{modelName}_hist.png',dpi=200)
    plt.close()

def RMSE(truth,pred,axis=0):
    # Truth and prediction should have (nSteps,nStations)
    # This function computes RMSE per station then return mean RMSE by station.
    _rmse = np.linalg.norm(pred-truth,axis=axis)/np.sqrt(pred.shape[0])
    return [np.mean(_rmse),np.std(_rmse),np.min(_rmse),np.max(_rmse)]

def MinMax(x):
    # Min-Max normalization based on the last dimension of the raw data
    _axis = tuple(np.arange(0,x.ndim-1))
    _min, _max = np.min(x,axis=_axis), np.max(x,axis=_axis)
    _idx = np.where(_min==_max)
    _min[_idx], _max[_idx] = 0., 1.
    _off = _min
    _scl = (_max-_min)
    return (x-_off)/_scl, _off, _scl

def MinMax_first_dim(x):
    # Min-Max normalization based on the first dimension
    _min = np.min(x, axis=0)  # 按第一维计算最小值
    _max = np.max(x, axis=0)  # 按第一维计算最大值
    _idx = np.where(_min == _max)
    _min[_idx], _max[_idx] = 0., 1.  # 避免除以 0 的情况
    _off = _min
    _scl = (_max - _min)
    return (x - _off) / _scl, _off, _scl

unlabeled_data['CLMSMat'],_,_ = MinMax(np.array(unlabeled_data['CLMSMat'], dtype=np.float64))
unlabeled_data['WRFMat'],_,_ = MinMax(np.array(unlabeled_data['WRFMat'], dtype=np.float64))
unlabeled_data['UrbanFeature'],_,_ = MinMax_first_dim(np.array(unlabeled_data['UrbanFeature'], dtype=np.float64))


import numpy as np
import torch,pickle,os,mat73
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from torch_geometric.nn import APPNP

def genGeoFeatures(path,geoMethod='average',poolSize=15,nCompPCA=40):
    # --------------------------Geo features--------------------------
    _raw = mat73.loadmat(f'{path}/FeaturePatch_401.mat')['FeatureMat_zeros']
    # Remove 5th dimension because all stations are land based
    _idx = np.arange(_raw.shape[2])
    _idx = np.delete(_idx,4)
    _raw = _raw[:,:,_idx,:]
    _imageSize,_,_nFeatures,_nStations = _raw.shape
    
    if geoMethod == 'average':     
        # Min-max normalization
        _norm = np.transpose(_raw,(2,0,1,3)).reshape(_nFeatures,-1)
        _min,_max = np.min(_norm,axis=1), np.max(_norm,axis=1)
        _max[_max==0] = 1e-5
        _off = _min
        _scl = _max-_min
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
        _norm = np.transpose(_raw,(0,1,3,2))
        _norm = (_norm-_off)/_scl
        _geoFeatures = np.transpose(_norm,(2,3,0,1))
        # PCA
        _geo2D = _geoFeatures.reshape(_nStations,-1)
        _pca = PCA(n_components=nCompPCA)
        _geoFeatures = _pca.fit_transform(_geo2D)
        _geoFeatures = (_geoFeatures-_geoFeatures.min())/(_geoFeatures.max()-_geoFeatures.min())
        _geoFeatures = torch.FloatTensor(_geoFeatures)
    
    return _geoFeatures,_off,_scl,_nStations

def genGeoFeatures_unlabeled(data, geoMethod='average', poolSize=15, nCompPCA=40):
    """
    生成地理特征，不需要路径而是直接使用输入数据。
    数据需要包含 'UrbanFeatureMat' 变量。
    """
    # --------------------------Geo features--------------------------
    # 获取 UrbanFeatureMat 数据
    if 'UrbanFeatureMat' not in data:
        raise ValueError("输入数据中未找到 'UrbanFeatureMat' 变量")
    
    _raw = data['UrbanFeatureMat']  # 提取 UrbanFeatureMat
    _imageSize, _, _nFeatures, _nStations = _raw.shape  # 获取维度信息
    
    if geoMethod == 'average':     
        # Min-max normalization
        _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        _min, _max = np.min(_norm, axis=1), np.max(_norm, axis=1)
        _max[_max == 0] = 1e-5  # 避免除以 0
        _off = _min
        _scl = _max - _min
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        # Average pooling 
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        _avgPool = torch.nn.AdaptiveAvgPool2d((poolSize, poolSize))
        _geoFeatures = _avgPool(_geoFeatures).reshape(_nStations, -1)

    elif geoMethod == 'pca':       
        # Mean-std normalization  
        _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        _off, _scl = np.mean(_norm, axis=1), np.std(_norm, axis=1)
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        # PCA
        _geo2D = _geoFeatures.reshape(_nStations, -1)
        _pca = PCA(n_components=nCompPCA)
        _geoFeatures = _pca.fit_transform(_geo2D)
        _geoFeatures = (_geoFeatures - _geoFeatures.min()) / (_geoFeatures.max() - _geoFeatures.min())
        _geoFeatures = torch.FloatTensor(_geoFeatures)
    
    else:
        raise ValueError(f"未知的 geoMethod: {geoMethod}")

    return _geoFeatures, _off, _scl, _nStations


# def dataGen_ESTnet(dataParam, path, nTrn=0.75, predMode=False):
#     """
#     按照ESTNet风格生成数据，将特征分为静态特征和动态特征
    
#     静态特征：静态地理特征（不包含CLMS动态特征）
#     动态特征：CFD/WRF特征（包括当前和时间窗口）+ CLMS动态特征
#     """
#     _window = dataParam['window']
#     _batchSize = dataParam['batchSize']
#     _geoFeatures, _off, _scl, _nStations = genGeoFeatures(path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])
    
#     # --------------------------图构建--------------------------
#     _raw = mat73.loadmat(f'{path}/GNN_N1_AJM.mat')
#     _dist, _, _simiW = _raw['dist'], _raw['location'], _raw['similarity']
#     _nNodes = _dist.shape[0]
#     _distW = np.exp(-_dist)
#     _off, _scl = np.min(_distW), np.max(_distW) - np.min(_distW)
#     _distW = (_distW - _off) / _scl
#     Adj = np.abs(_simiW * _distW)
#     Adj[Adj < dataParam['thres']] = 0.
#     assert np.allclose(Adj, Adj.T)
    
#     # --------------------------节点特征--------------------------
#     _raw = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
#     _raw = np.transpose(_raw, (0, 2, 1))  # Dim: timestep * nNodes * nFeatures
#     nCFDFeats = 54  # WRF变量数量
#     cfdIdx = np.arange(nCFDFeats)
    
#     # 修改：正确区分静态和动态特征
#     # 静态地理特征（不包括最后3个CLMS动态特征）
#     rawGeoFeatIdx = np.arange(nCFDFeats + 4, _raw.shape[-1] - 3 - 1)
#     # CLMS动态特征（最后3个）
#     clmsIdx = np.arange(_raw.shape[-1] - 3 - 1, _raw.shape[-1] - 1)
    
#     features = _raw[:, :, 1:]  # 提取除温度外的所有特征
#     targets = _raw[:, :, 0]    # 提取温度作为目标
    
#     # --------------------------构建PyG数据集--------------------------
#     T = len(features)
#     edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
#     _dataset = []
    
#     for n in range(_window, T - _window):
#         _feature = features[n]
#         # 处理历史时间窗口
#         _tdb = features[n - _window:n, :, cfdIdx]
#         _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(len(Adj), -1)
#         # 处理未来时间窗口
#         _tdf = features[n:n + _window, :, cfdIdx]
#         _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(len(Adj), -1)
        
#         # ESTNet风格：分离动态和静态特征
#         # 1. 动态特征：CFD特征（当前+时间窗口）+ CLMS动态特征
#         dynamic_features = np.hstack([
#             _feature[:, cfdIdx],      # 当前CFD特征
#             _tdb, _tdf,               # 时间窗口CFD特征
#             _feature[:, clmsIdx]      # CLMS动态特征
#         ])
        
#         # 2. 静态特征：静态地理特征
#         static_features = np.hstack([
#             _feature[:, rawGeoFeatIdx],  # 静态地理特征
#             _geoFeatures                 # 处理后的空间特征
#         ])
        
#         _target = targets[n].reshape(-1, 1)
#         _dataset.append(
#             Data(
#                 x_dynamic=torch.FloatTensor(dynamic_features),
#                 x_static=torch.FloatTensor(static_features),
#                 y=torch.FloatTensor(_target),
#                 edge_index=edgeIdxV,
#                 edge_attr=edgeAttrV
#             )
#         )
    
#     # 特征索引跟踪
#     # 1. 动态特征索引
#     _cfdFeatLen = len(cfdIdx) * (2 * _window + 1)  # 所有CFD特征长度（当前+时间窗口）
#     _clmsFeatLen = len(clmsIdx)  # CLMS特征长度
#     _dynamic_feat_lengths = np.cumsum([_cfdFeatLen, _clmsFeatLen])
#     _dynamic_feature_idx = {
#         'CFD': np.arange(0, _dynamic_feat_lengths[0]),
#         'CLMS': np.arange(_dynamic_feat_lengths[0], _dynamic_feat_lengths[1]),
#     }
    
#     # 2. 静态特征索引
#     _rawGeoFeatLen = len(rawGeoFeatIdx)
#     _geoFeatLen = len(_geoFeatures.T)
#     _static_feat_lengths = np.cumsum([_rawGeoFeatLen, _geoFeatLen])
#     _static_feature_idx = {
#         'rawGeo': np.arange(0, _static_feat_lengths[0]),
#         'embedGeo': np.arange(_static_feat_lengths[0], _static_feat_lengths[1]),
#     }
    
#     # 数据集分割
#     _generator = torch.Generator().manual_seed(19)
#     _trainLength = int(len(_dataset) * nTrn)
#     _validLength = len(_dataset) - _trainLength
#     trainSet, validSet = torch.utils.data.random_split(_dataset, [_trainLength, _validLength], _generator)
#     trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
#     validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)
    
#     # 记录元数据
#     metadata = {
#         'nNodes': _nStations,
#         'geoOff': _off,
#         'geoScl': _scl,
#         'dynamic_dim': dynamic_features.shape[1],
#         'static_dim': static_features.shape[1],
#         'oDim': _target.shape[-1],
#         'dynamic_feature_idx': _dynamic_feature_idx,
#         'static_feature_idx': _static_feature_idx,
#         'geoMethod': dataParam['geoMethod'],
#         'poolSize': dataParam['poolSize'],
#         'nCompPCA': dataParam['nCompPCA'],
#         'trainIdx': trainSet.indices,
#         'validIdx': validSet.indices,
#         'AdjMatrix': Adj,
#     }
    
#     return trainLoader, validLoader, metadata, validSet

# def dataGen_unlabeled_ESTnet(dataParam, data, nTrn=0.75, seed=19, predMode=False, labeled=False):
#     """
#     按照ESTNet风格生成无标签数据，将特征分为静态特征和动态特征
    
#     静态特征：静态地理特征（UrbanFeature，不包含CLMS）
#     动态特征：CFD/WRF特征（包括当前和时间窗口）+ CLMS动态特征
#     不包含辅助变量（站点ID和时间信息）
#     """
#     _window = dataParam['window']
#     _batchSize = dataParam['batchSize']
#     _geoFeatures, _off, _scl, _nStations = genGeoFeatures_unlabeled(data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])

#     # --------------------------图构建--------------------------
#     Map = data['Map']  # (nNodes, 2)
#     SimilarityMat = data['SimilarityMat']  # (nNodes, 10)
#     nNodes = Map.shape[0]

#     # 计算距离矩阵
#     Distance = np.zeros((nNodes, nNodes))
#     for i in range(nNodes):
#         for j in range(nNodes):
#             Distance[i, j] = np.sqrt((Map[i, 0] - Map[j, 0])**2 + (Map[i, 1] - Map[j, 1])**2)

#     # 归一化距离矩阵
#     Distance_min = np.min(Distance)
#     Distance_max = np.max(Distance)
#     Distance = (Distance - Distance_min) / (Distance_max - Distance_min)
#     _dist = Distance

#     # 计算相似性矩阵
#     Matrix = np.zeros((nNodes, nNodes))
#     for i in range(nNodes):
#         for j in range(nNodes):
#             Cov = np.corrcoef(SimilarityMat[i, :], SimilarityMat[j, :])
#             Matrix[i, j] = Cov[0, 1]
#     _simiW = Matrix

#     # 构建加权邻接矩阵
#     _distW = np.exp(-_dist)
#     _off, _scl = np.min(_distW), np.max(_distW) - np.min(_distW)
#     _distW = (_distW - _off) / _scl
#     Adj = np.abs(_simiW * _distW)
#     Adj[Adj < dataParam['thres']] = 0.0
#     assert np.allclose(Adj, Adj.T)

#     # --------------------------节点特征--------------------------
#     # 获取特征数据
#     WRFMat = data['WRFMat']  # (6624, 63, nNodes)
#     CLMSMat = data['CLMSMat']  # (6624, 3, nNodes)
#     UrbanFeature = data['UrbanFeature']  # (nNodes, 17)

#     # 处理CFD特征（动态）
#     cfd_features = np.transpose(WRFMat, (0, 2, 1))  # (6624, nNodes, 63)
    
#     # 处理CLMS特征（动态）
#     clms_features = np.transpose(CLMSMat, (0, 2, 1))  # (6624, nNodes, 3)
    
#     # 处理UrbanFeature（静态）
#     expanded_urban_feature = np.repeat(UrbanFeature[np.newaxis, :, :], cfd_features.shape[0], axis=0)
    
#     # 准备特征索引
#     nCFDFeats = cfd_features.shape[2]  # 54
#     nCLMSFeats = clms_features.shape[2]  # 3
    
#     # 如果有标签数据，获取目标值
#     if labeled and 'AoT_obs' in data:
#         targets = data['AoT_obs']
#     else:
#         targets = None
    
#     # --------------------------构建PyG数据集--------------------------
#     T = len(cfd_features)
#     edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
#     _dataset = []
    
#     for n in range(_window, T - _window):
#         # 获取当前CFD特征
#         _cfd_feature = cfd_features[n]
#         # 获取当前CLMS特征
#         _clms_feature = clms_features[n]
        
#         # 处理历史时间窗口
#         _tdb = cfd_features[n - _window:n]
#         _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(len(Adj), -1)
        
#         # 处理未来时间窗口
#         _tdf = cfd_features[n:n + _window]
#         _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(len(Adj), -1)
        
#         # 获取当前静态特征
#         _urban_feature = expanded_urban_feature[n]
        
#         # 按照ESTNet风格组合特征
#         # 1. 动态特征：当前CFD + 历史和未来CFD窗口 + CLMS
#         dynamic_features = np.hstack([
#             _cfd_feature,  # 当前CFD特征
#             _tdb, _tdf,    # 时间窗口CFD特征
#             _clms_feature  # CLMS特征
#         ])
        
#         # 2. 静态特征：UrbanFeature + 空间嵌入特征
#         static_features = np.hstack([
#             _urban_feature,  # UrbanFeature
#             _geoFeatures     # 空间嵌入特征
#         ])
        
#         # 创建Data对象
#         if labeled and targets is not None:
#             _target = targets[n].reshape(-1, 1)
#             data_obj = Data(
#                 x_dynamic=torch.FloatTensor(dynamic_features),
#                 x_static=torch.FloatTensor(static_features),
#                 y=torch.FloatTensor(_target),
#                 edge_index=edgeIdxV,
#                 edge_attr=edgeAttrV
#             )
#         else:
#             data_obj = Data(
#                 x_dynamic=torch.FloatTensor(dynamic_features),
#                 x_static=torch.FloatTensor(static_features),
#                 edge_index=edgeIdxV,
#                 edge_attr=edgeAttrV
#             )
        
#         _dataset.append(data_obj)
    
#     # 计算特征索引长度以便跟踪
#     _cfdFeatLen = nCFDFeats * (2 * _window + 1)  # 时间窗口中的CFD特征
#     _clmsFeatLen = nCLMSFeats
    
#     # 1. 动态特征索引
#     _dynamic_feat_lengths = np.cumsum([_cfdFeatLen, _clmsFeatLen])
#     _dynamic_feature_idx = {
#         'CFD': np.arange(0, _dynamic_feat_lengths[0]),
#         'CLMS': np.arange(_dynamic_feat_lengths[0], _dynamic_feat_lengths[1]),
#     }
    
#     # 2. 静态特征索引
#     _urbanFeatLen = UrbanFeature.shape[1]
#     _geoFeatLen = _geoFeatures.shape[1]
#     _static_feat_lengths = np.cumsum([_urbanFeatLen, _geoFeatLen])
#     _static_feature_idx = {
#         'Urban': np.arange(0, _static_feat_lengths[0]),
#         'embedGeo': np.arange(_static_feat_lengths[0], _static_feat_lengths[1]),
#     }
    
#     # 数据集分割
#     _generator = torch.Generator().manual_seed(seed)
#     _trainLength = int(len(_dataset) * nTrn)
#     _validLength = int(len(_dataset) * 0.2)
#     _testLength = len(_dataset) - _validLength - _trainLength
    
#     trainSet, validSet, testSet = torch.utils.data.random_split(_dataset, [_trainLength, _validLength, _testLength], _generator)
#     trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
#     validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)
#     testLoader = DataLoader(testSet, batch_size=len(testSet), shuffle=False)
    
#     # 记录元数据
#     metadata = {
#         'nNodes': _nStations,
#         'geoOff': _off,
#         'geoScl': _scl,
#         'dynamic_dim': dynamic_features.shape[1],
#         'static_dim': static_features.shape[1],
#         'oDim': 1 if labeled else None,
#         'dynamic_feature_idx': _dynamic_feature_idx,
#         'static_feature_idx': _static_feature_idx,
#         'geoMethod': dataParam['geoMethod'],
#         'poolSize': dataParam['poolSize'],
#         'nCompPCA': dataParam['nCompPCA'],
#         'trainIdx': trainSet.indices,
#         'validIdx': validSet.indices,
#         'AdjMatrix': Adj,
#         'isLabeled': labeled
#     }
    
#     return trainLoader, validLoader, testLoader, metadata, validSet

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

def build_unified_graph(dataParam, path, unlabeled_data, n_unlabeled=200):
    """
    构建统一的图结构，包含有标签站点(68个)和无标签站点(200个)
    
    Returns:
        unified_adj: 统一的邻接矩阵 (268, 268)
        labeled_indices: 有标签站点在统一图中的索引 [0, 1, ..., 67]
        unlabeled_indices: 无标签站点在统一图中的索引 [68, 69, ..., 267]
        unified_locations: 统一的位置信息 (268, 2)
    """
    
    # 1. 加载有标签站点的位置和相似性信息
    labeled_raw = mat73.loadmat(f'{path}/GNN_N1_AJM.mat')
    labeled_locations = labeled_raw['location']  # (68, 2)
    labeled_similarity = labeled_raw['similarity']  # (68, 68)
    n_labeled = labeled_locations.shape[0]
    
    # 2. 获取无标签站点的位置和相似性信息
    unlabeled_locations = unlabeled_data['Map']  # (200, 2)
    unlabeled_similarity_mat = unlabeled_data['SimilarityMat']  # (200, 10)
    
    # 3. 合并位置信息
    unified_locations = np.vstack([labeled_locations, unlabeled_locations])  # (268, 2)
    total_nodes = n_labeled + n_unlabeled
    
    # 4. 计算统一的距离矩阵
    unified_distance = np.zeros((total_nodes, total_nodes))
    for i in range(total_nodes):
        for j in range(total_nodes):
            unified_distance[i, j] = np.sqrt(
                (unified_locations[i, 0] - unified_locations[j, 0])**2 + 
                (unified_locations[i, 1] - unified_locations[j, 1])**2
            )
    
    # 5. 计算统一的相似性矩阵
    unified_similarity = np.zeros((total_nodes, total_nodes))
    
    # 5.1 有标签站点之间的相似性
    unified_similarity[:n_labeled, :n_labeled] = labeled_similarity
    # 确保对角线为1.0
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
    
    # 5.3 有标签和无标签站点之间的相似性
    max_dist = np.max(unified_distance)  # 获取最大距离用于归一化
    for i in range(n_labeled):
        for j in range(n_unlabeled):
            idx_j = n_labeled + j
            dist = unified_distance[i, idx_j]
            # 使用原始距离进行归一化，然后计算相似性
            normalized_dist = dist / max_dist
            similarity = np.exp(-normalized_dist * 2.0)
            unified_similarity[i, idx_j] = similarity
            unified_similarity[idx_j, i] = similarity
    
    # 6. 构建统一的邻接矩阵
    unified_dist_w = np.exp(-unified_distance / max_dist)  # 直接使用原始距离
    
    unified_adj = np.abs(unified_similarity * unified_dist_w)
    unified_adj[unified_adj < dataParam['thres']] = 0.0
    np.fill_diagonal(unified_adj, 0.0)  # 清零自环
    
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
    
    Args:
        unified_adj: 统一邻接矩阵 (268, 268)
        labeled_indices: 有标签站点索引
        unlabeled_indices: 无标签站点索引  
        unified_locations: 位置信息 (268, 2)
        dataParam: 数据参数
        verbose: 是否打印详细信息
    """
    import networkx as nx
    from scipy.sparse import csr_matrix
    
    n_labeled = len(labeled_indices)
    n_unlabeled = len(unlabeled_indices)
    total_nodes = n_labeled + n_unlabeled
    
    # 转换为NetworkX图
    G = nx.from_numpy_array(unified_adj)
    
    # 1. 基本统计
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = 2 * n_edges / total_nodes if total_nodes > 0 else 0
    
    # 2. 连通性检查
    is_connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)
    largest_cc_size = len(max(nx.connected_components(G), key=len)) if n_components > 0 else 0
    
    # 3. 子图统计
    labeled_adj = unified_adj[np.ix_(labeled_indices, labeled_indices)]
    unlabeled_adj = unified_adj[np.ix_(unlabeled_indices, unlabeled_indices)]
    
    G_labeled = nx.from_numpy_array(labeled_adj)
    G_unlabeled = nx.from_numpy_array(unlabeled_adj)
    
    labeled_density = nx.density(G_labeled)
    unlabeled_density = nx.density(G_unlabeled)
    
    # 4. 权重分布
    edge_weights = [unified_adj[i,j] for i,j in G.edges()]
    weight_stats = {
        'min': np.min(edge_weights) if edge_weights else 0,
        'max': np.max(edge_weights) if edge_weights else 0,
        'mean': np.mean(edge_weights) if edge_weights else 0,
        'std': np.std(edge_weights) if edge_weights else 0,
        'median': np.median(edge_weights) if edge_weights else 0
    }
    
    # 5. 阈值影响分析
    threshold = dataParam['thres']
    edges_above_thresh = np.sum(unified_adj > threshold)
    edges_below_thresh = np.sum((unified_adj > 0) & (unified_adj <= threshold))
    
    # 6. 度数分布
    degrees = [G.degree(n) for n in G.nodes()]
    degree_stats = {
        'min': np.min(degrees) if degrees else 0,
        'max': np.max(degrees) if degrees else 0,
        'mean': np.mean(degrees) if degrees else 0,
        'std': np.std(degrees) if degrees else 0
    }
    
    # 7. 孤立节点检查
    isolated_nodes = list(nx.isolates(G))
    n_isolated = len(isolated_nodes)
    
    # 8. 对角线检查
    diagonal_sum = np.sum(np.diag(unified_adj))
    
    # 9. 对称性检查
    is_symmetric = np.allclose(unified_adj, unified_adj.T)
    
    # 10. 距离统计（用于验证）
    distances = []
    for i in range(total_nodes):
        for j in range(i+1, total_nodes):
            if unified_adj[i,j] > 0:  # 只统计有边的节点对
                dist = np.sqrt((unified_locations[i,0] - unified_locations[j,0])**2 + 
                              (unified_locations[i,1] - unified_locations[j,1])**2)
                distances.append(dist)
    
    distance_stats = {
        'min': np.min(distances) if distances else 0,
        'max': np.max(distances) if distances else 0,
        'mean': np.mean(distances) if distances else 0,
        'median': np.median(distances) if distances else 0
    }
    
    # 打印诊断结果
    if verbose:
        print("=" * 60)
        print("图结构诊断报告")
        print("=" * 60)
        
        print(f"\n�� 基本统计:")
        print(f"  总节点数: {total_nodes} (有标签: {n_labeled}, 无标签: {n_unlabeled})")
        print(f"  总边数: {n_edges}")
        print(f"  图密度: {density:.4f}")
        print(f"  平均度数: {avg_degree:.2f}")
        
        print(f"\n�� 连通性:")
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
        
        print(f"\n🎯 阈值分析 (阈值={threshold}):")
        print(f"  超过阈值的边: {edges_above_thresh}")
        print(f"  被阈值过滤的边: {edges_below_thresh}")
        
        print(f"\n📏 度数分布:")
        print(f"  最小度数: {degree_stats['min']}")
        print(f"  最大度数: {degree_stats['max']}")
        print(f"  平均度数: {degree_stats['mean']:.2f}")
        print(f"  度数标准差: {degree_stats['std']:.2f}")
        
        print(f"\n�� 结构检查:")
        print(f"  对角线元素和: {diagonal_sum:.6f} {'✅ 已清零' if diagonal_sum == 0 else '❌ 未清零'}")
        print(f"  矩阵对称性: {'✅ 对称' if is_symmetric else '❌ 不对称'}")
        
        print(f"\n📍 距离统计 (有边的节点对):")
        print(f"  最小距离: {distance_stats['min']:.2f}")
        print(f"  最大距离: {distance_stats['max']:.2f}")
        print(f"  平均距离: {distance_stats['mean']:.2f}")
        print(f"  中位距离: {distance_stats['median']:.2f}")
        
        # 健康度评估
        print(f"\n🏥 健康度评估:")
        health_score = 0
        max_score = 8
        
        if is_connected: health_score += 1
        if n_isolated == 0: health_score += 1
        if diagonal_sum == 0: health_score += 1
        if is_symmetric: health_score += 1
        if 0.01 <= density <= 0.3: health_score += 1  # 合理的密度范围
        if degree_stats['min'] >= 1: health_score += 1  # 无孤立节点
        if weight_stats['std'] > 0: health_score += 1  # 权重有变化
        if largest_cc_size >= total_nodes * 0.9: health_score += 1  # 大部分节点连通
        
        health_percentage = (health_score / max_score) * 100
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

# 修改后的 dataGen_ESTnet 函数
def dataGen_ESTnet(dataParam, path, unlabeled_data, n_unlabeled=200, nTrn=0.75, predMode=False):
    """
    按照ESTNet风格生成数据，使用统一图结构
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    _geoFeatures, _off, _scl, _nStations = genGeoFeatures(path, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])
    
    # --------------------------统一图构建--------------------------
    unified_adj, labeled_indices, unlabeled_indices, unified_locations = build_unified_graph(
        dataParam, path, unlabeled_data, n_unlabeled
    )
    
    # 只提取有标签部分的邻接矩阵用于当前数据集
    labeled_adj = unified_adj[np.ix_(labeled_indices, labeled_indices)]
    
    # --------------------------节点特征--------------------------
    _raw = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    _raw = np.transpose(_raw, (0, 2, 1))  # Dim: timestep * nNodes * nFeatures
    nCFDFeats = 54  # WRF变量数量
    cfdIdx = np.arange(nCFDFeats)
    
    # 修改：正确区分静态和动态特征
    rawGeoFeatIdx = np.arange(nCFDFeats + 4, _raw.shape[-1] - 3 - 1)
    clmsIdx = np.arange(_raw.shape[-1] - 3 - 1, _raw.shape[-1] - 1)
    
    features = _raw[:, :, 1:]  # 提取除温度外的所有特征
    targets = _raw[:, :, 0]    # 提取温度作为目标
    
    # --------------------------构建PyG数据集--------------------------
    T = len(features)
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(labeled_adj))
    _dataset = []
    
    for n in range(_window, T - _window):
        _feature = features[n]
        # 处理历史时间窗口
        _tdb = features[n - _window:n, :, cfdIdx]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(len(labeled_adj), -1)
        # 处理未来时间窗口
        _tdf = features[n:n + _window, :, cfdIdx]
        _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(len(labeled_adj), -1)
        
        # ESTNet风格：分离动态和静态特征
        dynamic_features = np.hstack([
            _feature[:, cfdIdx],      # 当前CFD特征
            _tdb, _tdf,               # 时间窗口CFD特征
            _feature[:, clmsIdx]      # CLMS动态特征
        ])
        
        static_features = np.hstack([
            _feature[:, rawGeoFeatIdx],  # 静态地理特征
            _geoFeatures                 # 处理后的空间特征
        ])
        
        _target = targets[n].reshape(-1, 1)
        _dataset.append(
            Data(
                x_dynamic=torch.FloatTensor(dynamic_features),
                x_static=torch.FloatTensor(static_features),
                y=torch.FloatTensor(_target),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV
            )
        )
    
    # 特征索引跟踪
    _cfdFeatLen = len(cfdIdx) * (2 * _window + 1)
    _clmsFeatLen = len(clmsIdx)
    _dynamic_feat_lengths = np.cumsum([_cfdFeatLen, _clmsFeatLen])
    _dynamic_feature_idx = {
        'CFD': np.arange(0, _dynamic_feat_lengths[0]),
        'CLMS': np.arange(_dynamic_feat_lengths[0], _dynamic_feat_lengths[1]),
    }
    
    _rawGeoFeatLen = len(rawGeoFeatIdx)
    _geoFeatLen = len(_geoFeatures.T)
    _static_feat_lengths = np.cumsum([_rawGeoFeatLen, _geoFeatLen])
    _static_feature_idx = {
        'rawGeo': np.arange(0, _static_feat_lengths[0]),
        'embedGeo': np.arange(_static_feat_lengths[0], _static_feat_lengths[1]),
    }
    
    # 数据集分割
    _generator = torch.Generator().manual_seed(19)
    _trainLength = int(len(_dataset) * nTrn)
    _validLength = len(_dataset) - _trainLength
    trainSet, validSet = torch.utils.data.random_split(_dataset, [_trainLength, _validLength], _generator)
    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)
    
    # 记录元数据
    metadata = {
        'nNodes': _nStations,
        'geoOff': _off,
        'geoScl': _scl,
        'dynamic_dim': dynamic_features.shape[1],
        'static_dim': static_features.shape[1],
        'oDim': _target.shape[-1],
        'dynamic_feature_idx': _dynamic_feature_idx,
        'static_feature_idx': _static_feature_idx,
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': labeled_adj,
        'unified_adj': unified_adj,  # 保存统一图结构
        'labeled_indices': labeled_indices,
        'unlabeled_indices': unlabeled_indices,
        'unified_locations': unified_locations,
    }
    
    return trainLoader, validLoader, metadata, validSet


# 修改后的 dataGen_unlabeled_ESTnet 函数
def dataGen_unlabeled_ESTnet(dataParam, data, unified_graph_info, nTrn=0.75, seed=19, predMode=False, labeled=False):
    """
    按照ESTNet风格生成无标签数据，使用统一图结构
    
    Args:
        unified_graph_info: 包含统一图信息的字典，从dataGen_ESTnet的metadata获取
    """
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    _geoFeatures, _off, _scl, _nStations = genGeoFeatures_unlabeled(data, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])

    # --------------------------使用统一图结构--------------------------
    unified_adj = unified_graph_info['unified_adj']
    unlabeled_indices = unified_graph_info['unlabeled_indices']
    
    # 只提取无标签部分的邻接矩阵
    unlabeled_adj = unified_adj[np.ix_(unlabeled_indices, unlabeled_indices)]

    # --------------------------节点特征--------------------------
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
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(unlabeled_adj))
    _dataset = []
    
    for n in range(_window, T - _window):
        # 获取当前CFD特征
        _cfd_feature = cfd_features[n]
        # 获取当前CLMS特征
        _clms_feature = clms_features[n]
        
        # 处理历史时间窗口
        _tdb = cfd_features[n - _window:n]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(len(unlabeled_adj), -1)
        
        # 处理未来时间窗口
        _tdf = cfd_features[n:n + _window]
        _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(len(unlabeled_adj), -1)
        
        # 获取当前静态特征
        _urban_feature = expanded_urban_feature[n]
        
        # 按照ESTNet风格组合特征
        dynamic_features = np.hstack([
            _cfd_feature,  # 当前CFD特征
            _tdb, _tdf,    # 时间窗口CFD特征
            _clms_feature  # CLMS特征
        ])
        
        static_features = np.hstack([
            _urban_feature,  # UrbanFeature
            _geoFeatures     # 空间嵌入特征
        ])
        
        # 创建Data对象
        if labeled and targets is not None:
            _target = targets[n].reshape(-1, 1)
            data_obj = Data(
                x_dynamic=torch.FloatTensor(dynamic_features),
                x_static=torch.FloatTensor(static_features),
                y=torch.FloatTensor(_target),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV
            )
        else:
            data_obj = Data(
                x_dynamic=torch.FloatTensor(dynamic_features),
                x_static=torch.FloatTensor(static_features),
                edge_index=edgeIdxV,
                edge_attr=edgeAttrV
            )
        
        _dataset.append(data_obj)
    
    # 计算特征索引长度以便跟踪
    _cfdFeatLen = nCFDFeats * (2 * _window + 1)
    _clmsFeatLen = nCLMSFeats
    
    _dynamic_feat_lengths = np.cumsum([_cfdFeatLen, _clmsFeatLen])
    _dynamic_feature_idx = {
        'CFD': np.arange(0, _dynamic_feat_lengths[0]),
        'CLMS': np.arange(_dynamic_feat_lengths[0], _dynamic_feat_lengths[1]),
    }
    
    _urbanFeatLen = UrbanFeature.shape[1]
    _geoFeatLen = _geoFeatures.shape[1]
    _static_feat_lengths = np.cumsum([_urbanFeatLen, _geoFeatLen])
    _static_feature_idx = {
        'Urban': np.arange(0, _static_feat_lengths[0]),
        'embedGeo': np.arange(_static_feat_lengths[0], _static_feat_lengths[1]),
    }
    
    # 数据集分割
    _generator = torch.Generator().manual_seed(seed)
    _trainLength = int(len(_dataset) * nTrn)
    _validLength = int(len(_dataset) * 0.2)
    _testLength = len(_dataset) - _validLength - _trainLength
    
    trainSet, validSet, testSet = torch.utils.data.random_split(_dataset, [_trainLength, _validLength, _testLength], _generator)
    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
    validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)
    testLoader = DataLoader(testSet, batch_size=len(testSet), shuffle=False)
    
    # 记录元数据
    metadata = {
        'nNodes': _nStations,
        'geoOff': _off,
        'geoScl': _scl,
        'dynamic_dim': dynamic_features.shape[1],
        'static_dim': static_features.shape[1],
        'oDim': 1 if labeled else None,
        'dynamic_feature_idx': _dynamic_feature_idx,
        'static_feature_idx': _static_feature_idx,
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': unlabeled_adj,
        'isLabeled': labeled
    }
    
    return trainLoader, validLoader, testLoader, metadata, validSet



# 数据增强
# 之前的版本
import numpy as np
import random
from typing import Dict, Tuple, Union

PARAMETER_MAX = 5  # 最大增强强度

def _float_parameter(v: float, max_v: float) -> float:
    return float(v) * max_v / PARAMETER_MAX

# 特征维度增强
def FeatureNoise(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征维度上添加噪声"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:  # WRFMat, CLMSMat: (time, features, nodes)
        noise = np.random.normal(0, v, size=(1, data.shape[1], 1))
    else:  # UrbanFeature: (nodes, features)
        noise = np.random.normal(0, v, size=(1, data.shape[1]))
    return data + noise

def FeatureScale(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征缩放"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        scale = np.random.uniform(1 - v, 1 + v, size=(1, data.shape[1], 1))
    else:
        scale = np.random.uniform(1 - v, 1 + v, size=(1, data.shape[1]))
    return data * scale

def FeatureShift(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征偏移"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        shift = np.random.uniform(-v, v, size=(1, data.shape[1], 1))
    else:
        shift = np.random.uniform(-v, v, size=(1, data.shape[1]))
    return data + shift

def FeatureClip(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征值剪裁"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        min_val = np.min(data, axis=(0,2), keepdims=True)
        max_val = np.max(data, axis=(0,2), keepdims=True)
    else:
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
    clip_range = (min_val + v, max_val - v)
    return np.clip(data, clip_range[0], clip_range[1])

def FeatureZero(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """特征随机置零"""
    v = _float_parameter(v, max_v) + bias
    if len(data.shape) == 3:
        mask = np.random.random((1, data.shape[1], 1)) > v
    else:
        mask = np.random.random((1, data.shape[1])) > v
    return data * mask

# 时间维度增强 (只用于 WRFMat 和 CLMSMat)
def TimeNoise(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """时间维度上添加噪声"""
    if len(data.shape) != 3:  # 跳过 UrbanFeature
        return data
    v = _float_parameter(v, max_v) + bias
    noise = np.random.normal(0, v, size=(data.shape[0], 1, 1))
    return data + noise

def TimeShift(data: np.ndarray, v: float, max_v: float, bias: float = 0) -> np.ndarray:
    """时间序列整体平移"""
    if len(data.shape) != 3:  # 跳过 UrbanFeature
        return data
    v = _float_parameter(v, max_v) + bias
    shift = int(v * data.shape[0])
    return np.roll(data, shift, axis=0)

def my_augment_pool():
    augs = [
        # 时间维度增强 (会自动跳过 UrbanFeature)
        (TimeNoise, 0.2, 0.01),    # 时间噪声
        (TimeShift, 0.2, 0.01),    # 时间平移
        
        # 特征维度增强 (适用于所有数据)
        (FeatureNoise, 0.2, 0.01), # 特征噪声
        (FeatureScale, 0.3, 0.02), # 特征缩放
        (FeatureShift, 0.2, 0.01), # 特征偏移
        (FeatureClip, 0.2, 0.01),  # 特征剪裁
        (FeatureZero, 0.2, 0)      # 特征置零
    ]
    return augs

class RandAugment:
    def __init__(self, n: int, m: float):
        """
        初始化随机增强器
        Args:
            n: 每次应用的增强操作数量
            m: 增强强度
        """
        assert n >= 1, "增强操作数量必须大于等于1"
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        对数据应用随机增强
        Args:
            data: 输入数据
        Returns:
            增强后的数据
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("输入数据必须是numpy数组")
        
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            if random.random() < 0.8:  # 80%的概率应用操作
                data = op(data, v=self.m, max_v=max_v, bias=bias)
        return data

class TransformFixMatch:
    # 定义数据配置
    DATA_CONFIG = {
        'WRFMat': {'dims': 3, 'time_aug': True},      # 时空数据
        'CLMSMat': {'dims': 3, 'time_aug': True},     # 时空数据
        'UrbanFeature': {'dims': 2, 'time_aug': False} # 静态特征
    }

    def __init__(self, weak_n: int = 2, weak_m: float = 5,
                 strong_n: int = 3, strong_m: float = 8):
        """
        初始化FixMatch增强器
        Args:
            weak_n: 弱增强操作数量
            weak_m: 弱增强强度
            strong_n: 强增强操作数量
            strong_m: 强增强强度
        """
        self.augmenter_weak = RandAugment(n=weak_n, m=weak_m)
        self.augmenter_strong = RandAugment(n=strong_n, m=strong_m)

    def _validate_data(self, data: np.ndarray, key: str):
        """验证数据格式"""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{key} 必须是numpy数组")
        expected_dims = self.DATA_CONFIG[key]['dims']
        if data.ndim != expected_dims:
            raise ValueError(f"{key} 维度必须是{expected_dims}")

    def augment_variable(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对单个变量执行增强操作"""
        weak_augmented = self.augmenter_weak(data)
        strong_augmented = self.augmenter_strong(data)
        return weak_augmented, strong_augmented

    def __call__(self, data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        对输入数据执行增强
        Args:
            data: 包含所有变量的字典
        Returns:
            弱增强和强增强后的数据字典
        """
        weak_data = data.copy()
        strong_data = data.copy()

        for key in self.DATA_CONFIG:
            if key in data:
                self._validate_data(data[key], key)
                weak_aug, strong_aug = self.augment_variable(data[key])
                weak_data[key] = weak_aug
                strong_data[key] = strong_aug

        return weak_data, strong_data


import torch,os
import numpy as np
from torch_geometric.nn import GraphConv
from torch_geometric.nn import (
    GATConv,
    SAGEConv, 
    TransformerConv
)


class GNN_ESTNet(torch.nn.Module):
    def __init__(self, modelPara):
        super(GNN_ESTNet, self).__init__()
        self.nGNNLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        
        # 只在关键位置添加dropout
        # self.dropout = torch.nn.Dropout(0.1)  # 10%的dropout率
        
        # 1. 动态特征编码器
        self.dynamic_encoder = torch.nn.GRU(
            input_size=modelPara['dynamic_dim'],
            hidden_size=_HLD,
            batch_first=True
        )
        
        # 2. 静态特征编码器
        self.static_encoder = torch.nn.Linear(modelPara['static_dim'], _HLD)
        self.static_encoder_act = torch.nn.PReLU()
        
        # 3. 静态特征多尺度GCN
        self.static_gcns = torch.nn.ModuleList()
        for _ in range(self.nGNNLayers):
            self.static_gcns.append(SAGEConv(_HLD, _HLD, aggr="mean"))
            self.static_gcns.append(torch.nn.PReLU())
        
        # 4. 特征融合层
        self.fusion = torch.nn.Linear(_HLD * 2, _HLD)
        self.fusion_act = torch.nn.PReLU()
        
        # 5. 图处理层
        self.graph_processors = torch.nn.ModuleList()
        for _ in range(self.nGNNLayers):
            self.graph_processors.append(SAGEConv(_HLD, _HLD, aggr="mean"))
            self.graph_processors.append(torch.nn.PReLU())
        
        # 6. 输出解码器
        self.decoder = torch.nn.ModuleList()
        for _n in range(self.nMLPLayers):
            _inputChannel = _HLD
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers-1 else _HLD
            self.decoder.append(torch.nn.Linear(_inputChannel, _outputChannel))
            if _n < self.nMLPLayers-1:  # 最后一层不需要激活函数
                self.decoder.append(torch.nn.PReLU())
    
    def forward(self, x_dynamic, x_static, edge_index, edge_attr=None):
        # 动态特征处理
        # 增加批次维度以适应GRU
        batch_size = x_dynamic.size(0)
        x_dynamic = x_dynamic.unsqueeze(1)  # [N, 1, F_dyn]
        
        # 1. 处理动态特征：通过GRU编码时序信息
        dynamic_out, _ = self.dynamic_encoder(x_dynamic)
        dynamic_out = dynamic_out.squeeze(1)  # [N, H]
        
        # 2. 处理静态特征：初始编码
        static_out = self.static_encoder(x_static)
        static_out = self.static_encoder_act(static_out)
        
        # 3. 通过多尺度GCN处理静态特征
        static_original = static_out.clone()  # 保存初始特征用于残差连接
        for i, layer in enumerate(self.static_gcns):
            if i % 2 == 0:  # GCN层
                static_out = layer(static_out, edge_index)
            else:  # 激活函数
                static_out = layer(static_out)
        
        # 残差连接增强表示能力
        static_out = static_out + static_original
        
        # 4. 融合特征：将静态和动态特征连接并融合
        fused_features = torch.cat([dynamic_out, static_out], dim=1)
        fused_features = self.fusion(fused_features)
        fused_features = self.fusion_act(fused_features)
        
        # 5. 图处理：通过图卷积处理融合特征
        fused_original = fused_features.clone()  # 用于残差连接
        for i, layer in enumerate(self.graph_processors):
            if i % 2 == 0:  # GCN层
                fused_features = layer(fused_features, edge_index)
            else:  # 激活函数
                fused_features = layer(fused_features)
        
        # 残差连接
        fused_features = fused_features + fused_original
        
        # 只在最终输出前添加一个dropout - 这是最关键的uncertainty来源
        # fused_features = self.dropout(fused_features)
        
        # 6. 解码：最终输出层
        for layer in self.decoder:
            fused_features = layer(fused_features)
        
        return fused_features


def loadCheckPoint(modelName,model,opt,device,load=False,resetLr=False,lr=5e-5,predMode=False):
    chkptPath = f'./{modelName}.pt'
    if os.path.exists(chkptPath) and load:
        chkpt = torch.load(chkptPath,map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        opt.load_state_dict(chkpt['opt_state_dict'])
        EPOCH = chkpt['epoch']
        bestLoss = chkpt['bestLoss']
        hist = chkpt['hist']
        with open(f'./{modelName}_log','a') as f: print("Checkpoint loaded.")
        if opt.param_groups[0]['lr'] < 1e-6 and resetLr:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            with open(f'./{modelName}_log','a') as f: print(f"Resetting LR from {opt.param_groups[0]['lr']} to {lr}",file=f)
    elif predMode:
        if not os.path.exists(chkptPath): chkptPath = f'./trainedModels/{modelName}.pt'
        chkpt = torch.load(chkptPath,map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        print("Checkpoint loaded.")
        return -1
    else:
        EPOCH = 0
        bestLoss = np.inf
        hist = []
        with open(f'./{modelName}_log','w') as f: print("No checkpoint found, starting new model.",file=f)
    return EPOCH,bestLoss,chkptPath,hist

import numpy as np
import torch, pickle

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = './data'

def train_fixmatch(loader, weak_loader, strong_loader, model, lossFn, loss_unlabaledFn, opt, scheduler, device, nNodes, lambda_U=10.0):
    model.train()
    total_labeled_loss = 0
    total_unlabeled_loss = 0
    total_batches = 0
    pred, truth = [], []
    
    # 修改这些参数 - 稳定UQ
    epsilon = 1e-5      # 更精确的数值计算
    temperature = 0.5   # 更敏感的权重分布
    n_augments = 5      # 更稳定的伪标签

    for batch_idx, (batch, weak_batch, strong_batch) in enumerate(zip(loader, weak_loader, strong_loader)):
        batch = batch.to(device)
        weak_batch = weak_batch.to(device)
        strong_batch = strong_batch.to(device)
        
        opt.zero_grad(set_to_none=True)
        
        # 有标签损失计算 - 使用分离的静态和动态特征
        logits = model(batch.x_dynamic, batch.x_static, batch.edge_index, batch.edge_attr)
        labeled_loss = lossFn(logits, batch.y)
        
        # 无标签损失计算 - 使用分离的静态和动态特征
        with torch.no_grad():
            weak_predictions = []
            for _ in range(n_augments):
                pred_weak = model(weak_batch.x_dynamic, weak_batch.x_static, weak_batch.edge_index, weak_batch.edge_attr)
                weak_predictions.append(pred_weak)
            
            weak_predictions = torch.stack(weak_predictions)
            pseudo_labels = torch.mean(weak_predictions, dim=0)
            pred_std = torch.std(weak_predictions, dim=0)
            
            # 添加方差检测 - 应该在这里
            variance_stats = {
                'mean_std': pred_std.mean().item(),
                'max_std': pred_std.max().item(),
                'min_std': pred_std.min().item(),
                'std_std': pred_std.std().item()
            }
            
            weights = torch.exp(-pred_std / temperature)
            weights = (weights - weights.min()) / (weights.max() - weights.min() + epsilon)


        print(f"方差统计: 均值={variance_stats['mean_std']:.6f}, "
              f"最大={variance_stats['max_std']:.6f}, "
              f"最小={variance_stats['min_std']:.6f}, "
              f"标准差={variance_stats['std_std']:.6f}")

        logits_strong = model(strong_batch.x_dynamic, strong_batch.x_static, strong_batch.edge_index, strong_batch.edge_attr)
        point_wise_loss = loss_unlabaledFn(logits_strong, pseudo_labels)
        unlabeled_loss = (weights * point_wise_loss).mean()
        
        # 总损失
        total_loss = labeled_loss + lambda_U * unlabeled_loss
        total_loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 修改：从0.5改为0.1
        
        opt.step()
        
        # reshape操作
        _pred = logits.reshape(-1, nNodes)
        _truth = batch.y.reshape(-1, nNodes)
        
        total_labeled_loss += labeled_loss
        total_unlabeled_loss += unlabeled_loss
        
        pred.append(_pred.cpu().detach().numpy())
        truth.append(_truth.cpu().detach().numpy())
        total_batches += 1

        print(f"批次 {batch_idx + 1}: "
              f"标签数据损失 = {labeled_loss.item():.4f}, "
              f"无标签数据损失 = {unlabeled_loss.item():.4f}, "
              f"总损失 = {total_loss.item():.4f}")

    scheduler.step()

    # 计算平均损失
    avg_labeled_loss = (total_labeled_loss / total_batches).item()
    avg_unlabeled_loss = (total_unlabeled_loss / total_batches).item()

    # 计算RMSE
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    # 打印总的统计信息
    print(f"轮次摘要: "
          f"平均标签数据损失 = {avg_labeled_loss:.4f}, "
          f"平均无标签数据损失 = {avg_unlabeled_loss:.4f}, "
          f"RMSE = {rmse[0]:.4f}")

    return avg_labeled_loss + lambda_U * avg_unlabeled_loss, rmse, truth, pred

def test_fixmatch(loader, model, lossFn, device, nNodes):
    """评估模型性能（适配ESTNet架构）"""
    model.eval()
    total_labeled_loss = 0
    pred, truth = [], []

    if len(loader) == 0:
        raise ValueError("数据加载器为空，无法进行测试。")

    with torch.no_grad():
        for batch in loader:
            if batch.x_dynamic is None or batch.y is None:
                continue

            batch = batch.to(device)
            # 前向传播 - 使用分离的静态和动态特征
            logits = model(batch.x_dynamic, batch.x_static, batch.edge_index, batch.edge_attr)

            # 确保输出和标签形状兼容
            logits = logits.reshape(-1, nNodes)
            batch_y = batch.y.reshape(-1, nNodes)

            # 计算损失
            loss = lossFn(logits, batch_y)
            total_labeled_loss += loss.item()

            # 存储预测和真实值用于RMSE计算
            pred.append(logits.cpu().numpy())
            truth.append(batch_y.cpu().numpy())

    # 归一化标签损失
    avg_labeled_loss = total_labeled_loss / len(loader)

    # 计算标签数据的RMSE
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss, rmse, truth, pred

def main():
    # 创建输出目录（如果不存在）
    output_dir = './Fixmatch_ESTnet'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ##----------------------创建数据集----------------------
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': 12,
        'batchSize': 128,
        'thres': 0.4,  # 修改：从0.6改为0.4
        'geoFeatures': 'full',
    }
    
    print("步骤1: 生成有标签和无标签数据集")
    # 修改：传入unlabeled_data以构建统一图结构
    trainLoader, validLoader, metadata, _ = dataGen_ESTnet(dataParam, './data', unlabeled_data, n_unlabeled=500)

    # 添加图结构诊断
    print("\n步骤1.5: 诊断统一图结构")
    unified_adj = metadata['unified_adj']
    labeled_indices = metadata['labeled_indices']
    unlabeled_indices = metadata['unlabeled_indices']
    unified_locations = metadata.get('unified_locations', None)
    
    # 如果unified_locations不在metadata中，需要重新获取
    if unified_locations is None:
        # 重新构建一次以获取位置信息
        unified_adj, labeled_indices, unlabeled_indices, unified_locations = build_unified_graph(
            dataParam, './data', unlabeled_data, n_unlabeled=500
        )
    
    # 执行图结构诊断
    diagnose_results = diagnose_graph_structure(
        unified_adj, labeled_indices, unlabeled_indices, 
        unified_locations, dataParam, verbose=True
    )
    
    # 如果图结构不健康，给出建议
    if diagnose_results['health_percentage'] < 60:
        print("\n⚠️ 图结构存在问题，建议调整以下参数：")
        if diagnose_results['n_isolated'] > 0:
            print(f"  - 降低阈值 thres (当前: {dataParam['thres']}) 以减少孤立节点")
        if diagnose_results['density'] < 0.01:
            print(f"  - 降低阈值 thres 以增加图密度")
        if diagnose_results['density'] > 0.3:
            print(f"  - 提高阈值 thres 以减少图密度")
        if not diagnose_results['is_connected']:
            print(f"  - 考虑添加 kNN 保底连接以确保连通性")
    
    print("\n" + "="*60)

    ##----------------------数据增强----------------------
    print("步骤2: 对无标签数据应用数据增强")
    augmenter = TransformFixMatch(weak_n=2, weak_m=3, strong_n=3, strong_m=6)
    weak_augmented_data, strong_augmented_data = augmenter(unlabeled_data)
    
    # 修改：传入统一图信息
    unified_graph_info = {
        'unified_adj': metadata['unified_adj'],
        'labeled_indices': metadata['labeled_indices'], 
        'unlabeled_indices': metadata['unlabeled_indices']
    }
    
    weak_loader, _, _, _, _ = dataGen_unlabeled_ESTnet(dataParam, weak_augmented_data, unified_graph_info, labeled=False)
    strong_loader, _, _, _, _ = dataGen_unlabeled_ESTnet(dataParam, strong_augmented_data, unified_graph_info, labeled=False)
    
    print(f"弱增强样本数量: {len(weak_loader.dataset)}")
    print(f"强增强样本数量: {len(strong_loader.dataset)}")
    print(f"trainLoader长度: {len(trainLoader)}")
    print(f"validLoader长度: {len(validLoader)}")
    print(f"weak_loader长度: {len(weak_loader)}")     
    print(f"strong_loader长度: {len(strong_loader)}")

    ##----------------------生成模型----------------------
    nEpoch = 2000
    # 更新模型参数，使用分离特征维度
    modelParam = {
        'HLD': 128,
        'nMLP': 2,
        'nGNN': 3,  # 这个参数APPNP版本不再使用，但保留以免报错
        'nGAT': 1,
        'nHeads': 1,
        'K': 1,
        'dynamic_dim': metadata['dynamic_dim'],
        'static_dim': metadata['static_dim'],
        'oDim': metadata['oDim'],
        'BN': False,
        'Dropout': True,  # 启用dropout
    }

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_fixmatch_ESTnet_{dataParam["geoFeatures"]}Geo'  # 修改名称以区分
    model_path = os.path.join(output_dir, modelName)
    print("步骤3: 初始化模型、优化器和损失函数")
    model = GNN_ESTNet(modelParam).to(device)  # 使用ESTNet版本的GNN
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # 从 weight_decay=1e-5 改为无weight_decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)  
    lossFn = torch.nn.HuberLoss().to(device)
    loss_unlabel_fn = torch.nn.HuberLoss(reduction='none').to(device)  

    # 加载检查点或初始化训练
    EPOCH, bestLoss, chkptPath, hist = loadCheckPoint(model_path, model, opt, device, load=False)
    # 保存元数据
    with open(f'./{model_path}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)
    if torch.cuda.is_available():
        with open(f'./{model_path}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'./{model_path}_log', 'a') as f:
        print(model, file=f)


    # 在模型初始化后记录配置
    with open(f'./{model_path}_log', 'a') as f:
        print("模型配置:", file=f)
        print("  数据增强: weak_n=2, weak_m=3, strong_n=3, strong_m=6", file=f)
        print("  UQ参数: epsilon=1e-3, temperature=0.9, n_augments=3", file=f)
        print("  权重处理: 标准归一化，无权重限幅", file=f)
        print("  优化: lr=5e-4, weight_decay=1e-5, 无梯度裁剪", file=f)
        print("  无标签损失: lambda_U采用ramp-up, 前30个epoch从0线性升至10", file=f)
        print("  Dropout: 在融合后解码前设置dropout(p=0.1)用于MC UQ", file=f)
        print("  训练/验证: train启用dropout, eval禁用dropout(确定性推理)", file=f)
        print("  其他: 200个无标签站点, ESTNet静/动特征分离", file=f)
        print("  新添加的改动：threshold=0.4; kNN保底稀疏化(k=8); lambda_U=10*ramp; ramp_ep=30; 无梯度裁剪; gamma=0.9992", file=f)
        print("----------------------------------------", file=f)

    ##----------------------训练----------------------
    print("步骤4: 开始训练")
    print(f"当前轮次: {EPOCH}, 总轮次: {nEpoch}")
    def rampup_factor(epoch, ramp_ep=30):  # 修改：从20改为30
        return min(1.0, epoch / ramp_ep)

    for epoch in range(EPOCH, nEpoch):
        ramp = rampup_factor(epoch, 30)  # 修改：从20改为30
        # 使用FixMatch训练
        trainLoss, trainRMSE, _, _ = train_fixmatch(
            trainLoader, weak_loader, strong_loader, model, lossFn, loss_unlabel_fn, opt, scheduler, device, metadata['nNodes'], lambda_U=10  # 固定值
        )
        validLoss, validRMSE, _, _ = test_fixmatch(
            validLoader, model, lossFn, device, metadata['nNodes']
        )

        # 记录结果
        with open(f'./{model_path}_log', 'a') as f:
            print("", file=f)
            print(f"轮次 {epoch}: 损失 {trainLoss:1.4e}/{validLoss:1.4e}; RMSE {trainRMSE[0]:1.3f}/{validRMSE[0]:1.3f}; 学习率 {scheduler.get_last_lr()[0]:1.6f}; ramp={ramp:1.3f} ", file=f)
            print(f"RMSE 标准差: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; 最小值: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; 最大值: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};", file=f)

        # 保存最佳模型
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            with open(f'./{model_path}_log', 'a') as f:
                print("模型已保存。", file=f)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'bestLoss': bestLoss,
                'hist': hist,  # 添加hist以便加载后继续训练
            }, chkptPath)

        # 绘制训练历史
        hist.append([trainLoss, validLoss, trainRMSE[0], validRMSE[0]])
        plotHist(hist, model_path)

    print("训练完成")

if __name__ == "__main__":
    main()


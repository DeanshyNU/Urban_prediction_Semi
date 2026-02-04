"""
地理特征生成模块
包含从文件或数据中生成地理特征的功能
"""
import numpy as np
import torch
import mat73
from sklearn.decomposition import PCA


def genGeoFeatures(path, geoMethod='average', poolSize=15, nCompPCA=40):
    """
    从文件路径生成地理特征（用于有标签数据）
    """
    # --------------------------Geo features--------------------------
    _raw = mat73.loadmat(f'{path}/FeaturePatch_401.mat')['FeatureMat_zeros']
    # Remove 5th dimension because all stations are land based
    _idx = np.arange(_raw.shape[2])
    _idx = np.delete(_idx, 4)
    _raw = _raw[:, :, _idx, :]
    _imageSize, _, _nFeatures, _nStations = _raw.shape
    
    if geoMethod == 'average':     
        # Min-max normalization
        _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        _min, _max = np.min(_norm, axis=1), np.max(_norm, axis=1)
        _max[_max == 0] = 1e-5
        _off = _min
        _scl = _max - _min
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        # Average pooling 
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        _avgPool = torch.nn.AdaptiveAvgPool2d((poolSize, poolSize))
        _geoFeatures = _avgPool(_geoFeatures).reshape(_nStations, -1)

    if geoMethod == 'pca':       
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
    
    return _geoFeatures, _off, _scl, _nStations


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


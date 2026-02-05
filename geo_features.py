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


def genGeoFeatures_unlabeled(data, geoMethod='average', poolSize=15, nCompPCA=40, norm_off=None, norm_scl=None):
    """
    生成地理特征，不需要路径而是直接使用输入数据。
    数据需要包含 'UrbanFeatureMat' 变量。
    
    归一化策略：UrbanFeatureMat（地理嵌入）必须统一归一化
    - 如果提供了 norm_off 和 norm_scl，则使用这些参数进行归一化（与有标签数据一致）
    - 否则使用无标签数据自己的归一化参数（不推荐，可能导致特征尺度不一致）
    
    Args:
        data: 数据字典，必须包含 'UrbanFeatureMat' 变量
        geoMethod: 地理特征生成方法 ('average' 或 'pca')
        poolSize: 平均池化大小
        nCompPCA: PCA 组件数量
        norm_off: 可选。归一化偏移量（来自有标签数据）
        norm_scl: 可选。归一化缩放因子（来自有标签数据）
    
    Returns:
        _geoFeatures: 地理特征
        _off: 使用的归一化偏移量
        _scl: 使用的归一化缩放因子
        _nStations: 站点数量
    """
    # --------------------------Geo features--------------------------
    # 获取 UrbanFeatureMat 数据
    if 'UrbanFeatureMat' not in data:
        raise ValueError("输入数据中未找到 'UrbanFeatureMat' 变量")
    
    _raw = data['UrbanFeatureMat']  # 提取 UrbanFeatureMat
    _imageSize, _, _nFeatures, _nStations = _raw.shape  # 获取维度信息
    
    # 归一化参数：如果提供了则使用，否则自己计算
    use_provided_norm = (norm_off is not None and norm_scl is not None)
    
    if geoMethod == 'average':     
        if use_provided_norm:
            _off = norm_off
            _scl = norm_scl
            print(f"  ✓ 使用有标签数据的归一化参数（UrbanFeatureMat统一归一化）")
        else:
            # Min-max normalization（使用无标签数据自己的参数）
            _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
            _min, _max = np.min(_norm, axis=1), np.max(_norm, axis=1)
            _max[_max == 0] = 1e-5  # 避免除以 0
            _off = _min
            _scl = _max - _min
            _scl[_scl == 0] = 1e-5  # 避免除以0产生NaN
            print(f"  ⚠️  使用无标签数据自己的归一化参数（建议传入 norm_off 和 norm_scl）")
        
        # 归一化前统计
        _raw_flat = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        print(f"  归一化前统计: min=[{np.min(_raw_flat, axis=1)}], max=[{np.max(_raw_flat, axis=1)}]")
        
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / (_scl + 1e-12)
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
            print(f"  ✓ 使用有标签数据的归一化参数（UrbanFeatureMat统一归一化）")
        else:
            # Mean-std normalization（使用无标签数据自己的参数）
            _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
            _off, _scl = np.mean(_norm, axis=1), np.std(_norm, axis=1)
            _scl[_scl == 0] = 1e-5  # 避免除以0产生NaN
            print(f"  ⚠️  使用无标签数据自己的归一化参数（建议传入 norm_off 和 norm_scl）")
        
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / (_scl + 1e-12)
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


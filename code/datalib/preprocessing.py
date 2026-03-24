"""
数据预处理模块
包含数据加载、清洗、转换等预处理功能
"""
import numpy as np
import mat73
from utils import MinMax, MinMax_first_dim, add_auxiliary_variables


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


def reorder_wrf_to_labeled_order(data):
    """
    将无标签WRF变量顺序对齐到有标签数据

    无标签(合并后): [Tair, Tskin, Ttopsoil, Humidity, Irradiance, Wind]
    有标签:        [Tair, Humidity, Irradiance, Wind, Tskin, Ttopsoil]
    """
    WRFMat = data['WRFMat']  # (6624, 54, nNodes)
    reorder_indices = np.concatenate([
        np.arange(0, 9),     # Tair → pos 0 (不变)
        np.arange(27, 36),   # Humidity → pos 1
        np.arange(36, 45),   # Irradiance → pos 2
        np.arange(45, 54),   # Wind → pos 3
        np.arange(9, 18),    # Tskin → pos 4
        np.arange(18, 27),   # Ttopsoil → pos 5
    ])
    data['WRFMat'] = WRFMat[:, reorder_indices, :]
    return data


def reduce_stations(unlabeled_data, target_station_count=500, seed=42):
    """
    削减站点数量，随机选择指定数量的站点
    
    Args:
        unlabeled_data: 无标签数据字典
        target_station_count: 目标站点数量
        seed: 随机种子，确保可复现性
    """
    original_station_count = unlabeled_data['Map'].shape[0]

    # 确保目标站点数量小于或等于原始站点数量
    assert target_station_count <= original_station_count, "目标站点数量不能大于原始站点数量"

    # 设置随机种子，确保可复现性
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
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


def preprocess_unlabeled_data(unlabeled_file, target_station_count=500, nTimesteps=6624, seed=42):
    """
    完整的无标签数据预处理流程
    
    归一化策略（参考 downscale-gnn/data_semi.py）：
    ✅ WRF/CLMS（时序特征）：分开归一化
       - 时间段不同（有标签：2018.5-8，无标签：2018.5-9 + 2019.5-8）
       - 数据源不同 → 分布本来就不同
       - 分开归一化 + clip到[0,1] → 保证范围一致即可
    ✅ UrbanFeatureMat（地理嵌入）：统一归一化
       - 静态空间特征，同一地理区域
       - 在 data_generation.py 的 genGeoFeatures_unlabeled 中统一处理
       - 本函数不处理 UrbanFeatureMat 的归一化
    
    Args:
        unlabeled_file: 无标签数据文件路径
        target_station_count: 目标站点数量
        nTimesteps: 时间步数
        seed: 随机种子，用于站点选择的可复现性
    
    Returns:
        处理后的数据字典
    """
    # 1. 加载数据
    print(f"正在加载数据: {unlabeled_file}")
    unlabeled_data = mat73.loadmat(unlabeled_file)
    
    # 2. V2: keep all 63 WRF channels (no wind merge, no reorder)
    # Wind components (X/Y) kept separate to match V2 labeled data format
    print("V2模式: 保留全部63个WRF通道（不合并风速分量）")
    # # Old V1 processing (merge 63→54, reorder):
    # print("正在合并风速分量...")
    # unlabeled_data = merge_wind_components(unlabeled_data)
    # print("正在重排WRF变量顺序（对齐到有标签数据）...")
    # unlabeled_data = reorder_wrf_to_labeled_order(unlabeled_data)
    
    # 3. 统一数据类型为float64（提前转换，确保后续操作在统一类型上进行）
    print("正在统一数据类型...")
    if 'CLMSMat' in unlabeled_data:
        unlabeled_data['CLMSMat'] = np.array(unlabeled_data['CLMSMat'], dtype=np.float64)
    if 'WRFMat' in unlabeled_data:
        unlabeled_data['WRFMat'] = np.array(unlabeled_data['WRFMat'], dtype=np.float64)
    if 'UrbanFeature' in unlabeled_data:
        unlabeled_data['UrbanFeature'] = np.array(unlabeled_data['UrbanFeature'], dtype=np.float64)
    
    # 4. 削减站点
    print(f"正在削减站点数量至 {target_station_count}...")
    unlabeled_data = reduce_stations(unlabeled_data, target_station_count=target_station_count, seed=seed)
    
    # 5. 处理NaN值
    if 'UrbanFeatureMat' in unlabeled_data:
        print("正在处理 'UrbanFeatureMat'...")
        urban_feature_mat = unlabeled_data['UrbanFeatureMat']
        unlabeled_data['UrbanFeatureMat'] = np.nan_to_num(urban_feature_mat, nan=0.0)
        print("'UrbanFeatureMat' 中的 NaN 值已全部替换为 0")
    else:
        print("未找到 'UrbanFeatureMat' 变量！")
    
    # 6. 添加辅助变量
    print("正在添加辅助变量...")
    unlabeled_data["auxiliary_variables"] = add_auxiliary_variables(
        unlabeled_data, nStations=target_station_count, nTimesteps=nTimesteps
    )
    
    # 7. 归一化（WRF/CLMS：分开归一化；UrbanFeatureMat：在 data_generation 中统一处理）
    print("\n正在归一化数据...")
    print("  归一化策略：")
    print("    - WRF/CLMS（时序特征）：分开归一化（时间段不同、数据源不同）")
    print("    - UrbanFeatureMat（地理嵌入）：在 data_generation.py 中统一归一化")
    
    # 7.1 归一化 CLMSMat（独立归一化 + clip到[0,1]）
    print("\n  归一化 CLMSMat（独立归一化）...")
    print(f"    归一化前: min={np.min(unlabeled_data['CLMSMat']):.6f}, max={np.max(unlabeled_data['CLMSMat']):.6f}, mean={np.mean(unlabeled_data['CLMSMat']):.6f}")
    unlabeled_data['CLMSMat'], _, _ = MinMax(unlabeled_data['CLMSMat'])
    unlabeled_data['CLMSMat'] = np.clip(unlabeled_data['CLMSMat'], 0.0, 1.0)  # clip到[0,1]
    print(f"    归一化后（clip后）: min={np.min(unlabeled_data['CLMSMat']):.6f}, max={np.max(unlabeled_data['CLMSMat']):.6f}, mean={np.mean(unlabeled_data['CLMSMat']):.6f}")
    
    # 7.2 归一化 WRFMat（独立归一化 + clip到[0,1]）
    print("\n  归一化 WRFMat（独立归一化）...")
    print(f"    归一化前: min={np.min(unlabeled_data['WRFMat']):.6f}, max={np.max(unlabeled_data['WRFMat']):.6f}, mean={np.mean(unlabeled_data['WRFMat']):.6f}")
    unlabeled_data['WRFMat'], _, _ = MinMax(unlabeled_data['WRFMat'])
    unlabeled_data['WRFMat'] = np.clip(unlabeled_data['WRFMat'], 0.0, 1.0)  # clip到[0,1]
    print(f"    归一化后（clip后）: min={np.min(unlabeled_data['WRFMat']):.6f}, max={np.max(unlabeled_data['WRFMat']):.6f}, mean={np.mean(unlabeled_data['WRFMat']):.6f}")
    
    # 注意：UrbanFeature 和 UrbanFeatureMat 的归一化在 data_generation.py 中统一处理
    # 这里不处理 UrbanFeature 的归一化
    
    print("\n数据预处理完成！")
    return unlabeled_data


"""
全面的数据文件检查脚本
检查所有数据文件的维度、取值范围、统计信息等基础信息
"""
import os
import numpy as np
import mat73
from data_preprocessing import (
    merge_wind_components, 
    reduce_stations, 
    preprocess_unlabeled_data
)
from data_generation import build_unified_graph
from utils import MinMax, MinMax_first_dim, add_auxiliary_variables
from geo_features import genGeoFeatures


def print_separator(title="", char="=", length=80):
    """打印分隔线"""
    if title:
        print(f"\n{char * length}")
        print(f"{title:^{length}}")
        print(f"{char * length}\n")
    else:
        print(f"{char * length}\n")


def check_array_info(name, arr, show_stats=True, show_sample=False):
    """
    检查数组的详细信息
    
    Args:
        name: 数组名称
        arr: numpy数组
        show_stats: 是否显示统计信息
        show_sample: 是否显示样本值
    """
    print(f"\n【{name}】")
    print(f"  形状 (Shape): {arr.shape}")
    print(f"  数据类型 (Dtype): {arr.dtype}")
    print(f"  内存大小 (Size): {arr.nbytes / 1024 / 1024:.2f} MB")
    
    if show_stats:
        if arr.size > 0:
            print(f"  最小值 (Min): {np.nanmin(arr):.6f}")
            print(f"  最大值 (Max): {np.nanmax(arr):.6f}")
            print(f"  平均值 (Mean): {np.nanmean(arr):.6f}")
            print(f"  标准差 (Std): {np.nanstd(arr):.6f}")
            
            # 检查NaN和Inf
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0:
                print(f"  ⚠️  NaN数量: {nan_count} ({nan_count/arr.size*100:.2f}%)")
            if inf_count > 0:
                print(f"  ⚠️  Inf数量: {inf_count} ({inf_count/arr.size*100:.2f}%)")
            
            # 检查零值
            zero_count = (arr == 0).sum()
            if zero_count > 0:
                print(f"  零值数量: {zero_count} ({zero_count/arr.size*100:.2f}%)")
    
    if show_sample and arr.size > 0:
        print(f"  样本值 (前5个): {arr.flat[:5]}")


def check_each_feature_variable(features, targets, feature_names=None):
    """
    详细检查每个特征变量的信息
    
    Args:
        features: 特征矩阵 (timestep, nNodes, nFeatures)
        targets: 目标变量 (timestep, nNodes)
        feature_names: 可选的特征名称列表
    """
    print_separator("详细特征变量分析", "-")
    
    nTimesteps, nNodes, nFeatures = features.shape
    
    print(f"数据维度: {features.shape}")
    print(f"  时间步数: {nTimesteps}")
    print(f"  节点数: {nNodes}")
    print(f"  特征数: {nFeatures}")
    
    # 为每个特征创建详细报告
    feature_info = []
    
    for feat_idx in range(nFeatures):
        feat_data = features[:, :, feat_idx]  # (timestep, nNodes)
        
        # 计算统计信息
        min_val = np.nanmin(feat_data)
        max_val = np.nanmax(feat_data)
        mean_val = np.nanmean(feat_data)
        std_val = np.nanstd(feat_data)
        
        # 检查是否为静态特征（所有时间步的值都相同）
        # 方法1: 检查每个节点在不同时间步的值是否变化
        node_std = np.std(feat_data, axis=0)  # 每个节点在不同时间步的标准差
        is_static_by_node = (node_std < 1e-6).sum()  # 标准差接近0的节点数
        static_ratio = is_static_by_node / nNodes
        
        # 方法2: 检查所有值是否相同
        all_values_same = np.allclose(feat_data, feat_data[0, 0])
        
        # 方法3: 计算时间维度上的变化程度
        time_variance = np.var(feat_data, axis=0).mean()  # 平均时间方差
        
        # 判断特征类型
        if all_values_same or static_ratio > 0.95:
            feature_type = "静态 (Static)"
        elif time_variance < 1e-6:
            feature_type = "准静态 (Quasi-Static)"
        else:
            feature_type = "动态 (Dynamic)"
        
        feature_info.append({
            'idx': feat_idx,
            'name': feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"Feature_{feat_idx}",
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'static_ratio': static_ratio,
            'time_variance': time_variance,
            'type': feature_type,
            'all_same': all_values_same
        })
    
    # 打印详细表格
    print(f"\n{'索引':<6} {'特征名':<20} {'类型':<15} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'标准差':<12} {'静态比例':<10}")
    print("-" * 120)
    
    for info in feature_info:
        print(f"{info['idx']:<6} {info['name']:<20} {info['type']:<15} "
              f"{info['min']:<12.6f} {info['max']:<12.6f} {info['mean']:<12.6f} "
              f"{info['std']:<12.6f} {info['static_ratio']*100:<9.1f}%")
    
    # 按类型分组统计
    print_separator("特征类型统计", "-")
    static_features = [f for f in feature_info if '静态' in f['type']]
    dynamic_features = [f for f in feature_info if '动态' in f['type']]
    
    print(f"静态特征数量: {len(static_features)}")
    if static_features:
        print(f"  索引范围: {min(f['idx'] for f in static_features)} - {max(f['idx'] for f in static_features)}")
        print(f"  索引列表: {[f['idx'] for f in static_features]}")
    
    print(f"\n动态特征数量: {len(dynamic_features)}")
    if dynamic_features:
        print(f"  索引范围: {min(f['idx'] for f in dynamic_features)} - {max(f['idx'] for f in dynamic_features)}")
    
    return feature_info


def check_labeled_data(data_path):
    """检查有标签数据文件"""
    print_separator("1. 有标签数据文件检查", "=")
    
    # 1.1 检查 StationMat 文件
    station_file = os.path.join(data_path, 'GNN_N1_StationMat.mat')
    if os.path.exists(station_file):
        print(f"✓ 找到文件: {station_file}")
        try:
            station_data = mat73.loadmat(station_file)
            print(f"  文件中的变量: {list(station_data.keys())}")
            
            # 检查 StationMat_se (原始数据，可能有缺失值)
            if 'StationMat_se' in station_data:
                print_separator("1.0.1 StationMat_se (原始数据) 缺失值检查", "-")
                raw_se = station_data['StationMat_se']
                print(f"  StationMat_se 原始维度: {raw_se.shape}")
                
                # 转置后检查
                raw_se_transposed = np.transpose(raw_se, (0, 2, 1))
                print(f"  转置后维度 (timestep * nNodes * nFeatures): {raw_se_transposed.shape}")
                
                # 检查整体缺失情况
                total_elements = raw_se.size
                nan_count = np.isnan(raw_se).sum()
                nan_percentage = (nan_count / total_elements) * 100
                
                print(f"\n  整体缺失情况:")
                print(f"    总元素数: {total_elements:,}")
                print(f"    NaN数量: {nan_count:,} ({nan_percentage:.2f}%)")
                
                # 检查每个特征维度的缺失情况
                features_se = raw_se_transposed[:, :, 1:]  # 去掉温度目标
                targets_se = raw_se_transposed[:, :, 0]    # 温度目标
                
                print(f"\n  目标变量 (温度) 缺失情况:")
                target_nan = np.isnan(targets_se).sum()
                target_nan_pct = (target_nan / targets_se.size) * 100
                print(f"    NaN数量: {target_nan:,} ({target_nan_pct:.2f}%)")
                
                print(f"\n  特征矩阵缺失情况:")
                feature_nan = np.isnan(features_se).sum()
                feature_nan_pct = (feature_nan / features_se.size) * 100
                print(f"    NaN数量: {feature_nan:,} ({feature_nan_pct:.2f}%)")
                
                # 检查每个特征维度的缺失情况
                print(f"\n  各特征维度的缺失情况:")
                print(f"    {'特征索引':<10} {'NaN数量':<15} {'NaN比例':<15} {'缺失站点数':<15}")
                print("    " + "-" * 55)
                
                for feat_idx in range(features_se.shape[-1]):
                    feat_data = features_se[:, :, feat_idx]
                    feat_nan = np.isnan(feat_data).sum()
                    feat_nan_pct = (feat_nan / feat_data.size) * 100
                    # 计算有多少个站点在这个特征上有缺失
                    stations_with_nan = (np.isnan(feat_data).any(axis=0)).sum()
                    print(f"    {feat_idx:<10} {feat_nan:<15,} {feat_nan_pct:<14.2f}% {stations_with_nan:<15}")
                
                # 检查每个站点的缺失情况
                print(f"\n  各站点的缺失情况:")
                print(f"    {'站点索引':<10} {'NaN数量':<15} {'NaN比例':<15} {'缺失特征数':<15}")
                print("    " + "-" * 55)
                
                for node_idx in range(features_se.shape[1]):
                    node_data = features_se[:, node_idx, :]
                    node_nan = np.isnan(node_data).sum()
                    node_nan_pct = (node_nan / node_data.size) * 100
                    # 计算有多少个特征在这个站点上有缺失
                    features_with_nan = (np.isnan(node_data).any(axis=0)).sum()
                    print(f"    {node_idx:<10} {node_nan:<15,} {node_nan_pct:<14.2f}% {features_with_nan:<15}")
                
                # 检查时间步的缺失情况
                print(f"\n  各时间步的缺失情况:")
                print(f"    {'时间步索引':<12} {'NaN数量':<15} {'NaN比例':<15}")
                print("    " + "-" * 42)
                
                # 只显示前10个和后10个时间步，以及缺失最多的几个
                time_nan_counts = []
                for t_idx in range(features_se.shape[0]):
                    time_data = features_se[t_idx, :, :]
                    time_nan = np.isnan(time_data).sum()
                    time_nan_pct = (time_nan / time_data.size) * 100
                    time_nan_counts.append((t_idx, time_nan, time_nan_pct))
                
                # 显示前5个和后5个
                print("    前5个时间步:")
                for t_idx, t_nan, t_pct in time_nan_counts[:5]:
                    print(f"    {t_idx:<12} {t_nan:<15,} {t_pct:<14.2f}%")
                
                print("    后5个时间步:")
                for t_idx, t_nan, t_pct in time_nan_counts[-5:]:
                    print(f"    {t_idx:<12} {t_nan:<15,} {t_pct:<14.2f}%")
                
                # 显示缺失最多的5个时间步
                time_nan_counts_sorted = sorted(time_nan_counts, key=lambda x: x[1], reverse=True)
                print("    缺失最多的5个时间步:")
                for t_idx, t_nan, t_pct in time_nan_counts_sorted[:5]:
                    print(f"    {t_idx:<12} {t_nan:<15,} {t_pct:<14.2f}%")
            
            # 检查 StationMat_se_fill (填充后的数据)
            if 'StationMat_se_fill' in station_data:
                print_separator("1.0.2 StationMat_se_fill (填充后数据) 对比", "-")
                raw = station_data['StationMat_se_fill']
                print(f"  StationMat_se_fill 原始维度: {raw.shape}")
                
                # 检查填充后是否还有缺失值
                fill_nan_count = np.isnan(raw).sum()
                if fill_nan_count > 0:
                    print(f"  ⚠️  填充后仍有 NaN: {fill_nan_count:,} ({fill_nan_count/raw.size*100:.2f}%)")
                else:
                    print(f"  ✓ 填充后无 NaN 值")
                
                # 如果两个都存在，进行对比
                if 'StationMat_se' in station_data:
                    raw_se = station_data['StationMat_se']
                    raw_se_transposed = np.transpose(raw_se, (0, 2, 1))
                    raw_transposed = np.transpose(raw, (0, 2, 1))
                    
                    print(f"\n  填充前后对比:")
                    se_nan = np.isnan(raw_se_transposed).sum()
                    fill_nan = np.isnan(raw_transposed).sum()
                    print(f"    StationMat_se NaN: {se_nan:,} ({se_nan/raw_se_transposed.size*100:.2f}%)")
                    print(f"    StationMat_se_fill NaN: {fill_nan:,} ({fill_nan/raw_transposed.size*100:.2f}%)")
                    print(f"    填充的NaN数量: {se_nan - fill_nan:,}")
                    
                    # 检查填充方法（通过对比非NaN位置的值是否相同）
                    non_nan_mask = ~np.isnan(raw_se_transposed)
                    if non_nan_mask.sum() > 0:
                        values_match = np.allclose(
                            raw_se_transposed[non_nan_mask], 
                            raw_transposed[non_nan_mask], 
                            equal_nan=True
                        )
                        if values_match:
                            print(f"    ✓ 非NaN位置的值保持一致（填充未改变原始值）")
                        else:
                            print(f"    ⚠️  非NaN位置的值有差异（可能进行了平滑处理）")
                
                print_separator("1.1 StationMat_se_fill 详细分析", "=")
                
                # 转置后检查
                raw_transposed = np.transpose(raw, (0, 2, 1))
                print(f"  转置后维度 (timestep * nNodes * nFeatures): {raw_transposed.shape}")
                
                # 提取特征和目标
                features = raw_transposed[:, :, 1:]  # 去掉温度目标
                targets = raw_transposed[:, :, 0]     # 温度目标
                
                print(f"\n  注意: 索引0是温度目标，特征从索引1开始")
                print(f"  实际特征索引范围: 1-{raw_transposed.shape[-1]-1} (共{features.shape[-1]}个特征)")
                
                check_array_info("特征矩阵 (Features)", features, show_stats=True)
                check_array_info("目标变量 (Targets - 温度)", targets, show_stats=True)
                
                # 详细检查每个特征变量
                print_separator("1.1.1 每个特征变量的详细分析", "=")
                
                # 创建特征名称（基于已知的划分）
                nCFDFeats = 54
                feature_names = []
                for i in range(features.shape[-1]):
                    if i < nCFDFeats:
                        feature_names.append(f"CFD_{i}")
                    elif i < nCFDFeats + 4:
                        feature_names.append(f"Unknown_{i}")
                    elif i < features.shape[-1] - 3:
                        feature_names.append(f"Geo_{i}")
                    else:
                        feature_names.append(f"CLMS_{i}")
                
                feature_info = check_each_feature_variable(features, targets, feature_names)
                
                # 检查特征维度划分
                print_separator("1.1.2 特征索引划分验证", "=")
                nCFDFeats = 54
                cfdIdx = np.arange(nCFDFeats)
                rawGeoFeatIdx = np.arange(nCFDFeats + 4, features.shape[-1] - 3 - 1)
                clmsIdx = np.arange(features.shape[-1] - 3 - 1, features.shape[-1] - 1)
                unknownIdx = np.arange(nCFDFeats, nCFDFeats + 4)  # 被跳过的4个特征
                
                print(f"\n当前代码的划分:")
                print(f"  CFD特征索引: {cfdIdx[0]}-{cfdIdx[-1]} (共{len(cfdIdx)}个)")
                print(f"  未知特征索引: {unknownIdx[0]}-{unknownIdx[-1]} (共{len(unknownIdx)}个) - 被跳过")
                print(f"  静态地理特征索引: {rawGeoFeatIdx[0]}-{rawGeoFeatIdx[-1]} (共{len(rawGeoFeatIdx)}个)")
                print(f"  CLMS动态特征索引: {clmsIdx[0]}-{clmsIdx[-1]} (共{len(clmsIdx)}个)")
                
                # 验证每个分组的特征类型
                print(f"\n验证各分组的特征类型:")
                
                # CFD特征
                cfd_types = [feature_info[i]['type'] for i in cfdIdx]
                cfd_static_count = sum(1 for t in cfd_types if '静态' in t)
                print(f"  CFD特征 (0-{cfdIdx[-1]}): {cfd_static_count}/{len(cfdIdx)} 个静态, {len(cfdIdx)-cfd_static_count} 个动态")
                
                # 未知特征
                if len(unknownIdx) > 0:
                    unknown_types = [feature_info[i]['type'] for i in unknownIdx]
                    unknown_static_count = sum(1 for t in unknown_types if '静态' in t)
                    print(f"  未知特征 ({unknownIdx[0]}-{unknownIdx[-1]}): {unknown_static_count}/{len(unknownIdx)} 个静态, {len(unknownIdx)-unknown_static_count} 个动态")
                    print(f"    详细: {[feature_info[i]['name'] + ':' + feature_info[i]['type'] for i in unknownIdx]}")
                
                # 静态地理特征
                geo_types = [feature_info[i]['type'] for i in rawGeoFeatIdx]
                geo_static_count = sum(1 for t in geo_types if '静态' in t)
                print(f"  静态地理特征 ({rawGeoFeatIdx[0]}-{rawGeoFeatIdx[-1]}): {geo_static_count}/{len(rawGeoFeatIdx)} 个静态, {len(rawGeoFeatIdx)-geo_static_count} 个动态")
                
                # CLMS特征
                clms_types = [feature_info[i]['type'] for i in clmsIdx]
                clms_static_count = sum(1 for t in clms_types if '静态' in t)
                print(f"  CLMS特征 ({clmsIdx[0]}-{clmsIdx[-1]}): {clms_static_count}/{len(clmsIdx)} 个静态, {len(clmsIdx)-clms_static_count} 个动态")
                
                # 显示每个分组的取值范围
                print(f"\n各分组的取值范围:")
                print(f"  CFD特征:")
                cfd_mins = [feature_info[i]['min'] for i in cfdIdx]
                cfd_maxs = [feature_info[i]['max'] for i in cfdIdx]
                print(f"    最小值范围: [{min(cfd_mins):.6f}, {max(cfd_mins):.6f}]")
                print(f"    最大值范围: [{min(cfd_maxs):.6f}, {max(cfd_maxs):.6f}]")
                
                if len(unknownIdx) > 0:
                    print(f"  未知特征:")
                    unknown_mins = [feature_info[i]['min'] for i in unknownIdx]
                    unknown_maxs = [feature_info[i]['max'] for i in unknownIdx]
                    print(f"    最小值范围: [{min(unknown_mins):.6f}, {max(unknown_mins):.6f}]")
                    print(f"    最大值范围: [{min(unknown_maxs):.6f}, {max(unknown_maxs):.6f}]")
                
                print(f"  静态地理特征:")
                geo_mins = [feature_info[i]['min'] for i in rawGeoFeatIdx]
                geo_maxs = [feature_info[i]['max'] for i in rawGeoFeatIdx]
                print(f"    最小值范围: [{min(geo_mins):.6f}, {max(geo_mins):.6f}]")
                print(f"    最大值范围: [{min(geo_maxs):.6f}, {max(geo_maxs):.6f}]")
                
                print(f"  CLMS特征:")
                clms_mins = [feature_info[i]['min'] for i in clmsIdx]
                clms_maxs = [feature_info[i]['max'] for i in clmsIdx]
                print(f"    最小值范围: [{min(clms_mins):.6f}, {max(clms_mins):.6f}]")
                print(f"    最大值范围: [{min(clms_maxs):.6f}, {max(clms_maxs):.6f}]")
                
            else:
                print(f"  ⚠️  未找到 'StationMat_se_fill' 变量")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ 文件不存在: {station_file}")
    
    # 1.2 检查 AJM 文件（位置和相似性）
    ajm_file = os.path.join(data_path, 'GNN_N1_AJM.mat')
    if os.path.exists(ajm_file):
        print(f"\n✓ 找到文件: {ajm_file}")
        try:
            ajm_data = mat73.loadmat(ajm_file)
            print(f"  文件中的变量: {list(ajm_data.keys())}")
            
            if 'location' in ajm_data:
                location = ajm_data['location']
                check_array_info("站点位置 (Location)", location, show_stats=True)
                print(f"  站点数量: {location.shape[0]}")
            
            if 'similarity' in ajm_data:
                similarity = ajm_data['similarity']
                check_array_info("相似性矩阵 (Similarity)", similarity, show_stats=True)
                print(f"  矩阵维度: {similarity.shape}")
                print(f"  是否对称: {np.allclose(similarity, similarity.T)}")
                
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print(f"❌ 文件不存在: {ajm_file}")
    
    # 1.3 检查 Labeled_Finalized.mat 文件
    labeled_finalized_file = os.path.join(data_path, 'Labeled_Finalized.mat')
    if os.path.exists(labeled_finalized_file):
        print(f"\n✓ 找到文件: {labeled_finalized_file}")
        try:
            labeled_finalized_data = mat73.loadmat(labeled_finalized_file)
            print(f"  文件中的变量: {list(labeled_finalized_data.keys())}")
            
            for key, value in labeled_finalized_data.items():
                if not key.startswith('__'):  # 跳过MATLAB元数据
                    if isinstance(value, np.ndarray):
                        check_array_info(f"Labeled_Finalized.{key}", value, show_stats=True)
                    else:
                        print(f"\n  {key}: {type(value)} (非数组)")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print(f"\n⚠️  文件不存在: {labeled_finalized_file}")


def check_unlabeled_data(data_path, target_station_count=200):
    """检查无标签数据文件"""
    print_separator("2. 无标签数据文件检查", "=")
    
    unlabeled_file = os.path.join(data_path, 'Unlabeled_Finalized.mat')
    if not os.path.exists(unlabeled_file):
        print(f"❌ 文件不存在: {unlabeled_file}")
        return None
    
    print(f"✓ 找到文件: {unlabeled_file}")
    
    # 2.1 检查原始数据
    print_separator("2.1 原始无标签数据", "-")
    try:
        unlabeled_data = mat73.loadmat(unlabeled_file)
        print(f"文件中的变量: {list(unlabeled_data.keys())}")
        
        # 检查各个变量
        for key in ['CLMSMat', 'WRFMat', 'Map', 'NodeLocation', 'UrbanFeature', 
                   'SimilarityMat', 'UrbanFeatureMat']:
            if key in unlabeled_data:
                arr = unlabeled_data[key]
                check_array_info(f"原始 {key}", arr, show_stats=True)
            else:
                print(f"\n⚠️  未找到变量: {key}")
        
        # 2.2 检查合并风速后的数据
        print_separator("2.2 合并风速分量后", "-")
        unlabeled_data_merged = merge_wind_components(unlabeled_data.copy())
        check_array_info("合并后 WRFMat", unlabeled_data_merged['WRFMat'], show_stats=True)
        print(f"  原始WRFMat维度: {unlabeled_data['WRFMat'].shape}")
        print(f"  合并后WRFMat维度: {unlabeled_data_merged['WRFMat'].shape}")
        
        # 2.3 检查削减站点后的数据
        print_separator("2.3 削减站点后", "-")
        original_count = unlabeled_data['Map'].shape[0]
        print(f"原始站点数量: {original_count}")
        print(f"目标站点数量: {target_station_count}")
        
        # 设置随机种子以确保可复现
        np.random.seed(42)
        unlabeled_data_reduced = reduce_stations(
            unlabeled_data_merged.copy(), 
            target_station_count=target_station_count
        )
        
        for key in ['CLMSMat', 'WRFMat', 'Map', 'UrbanFeature', 'SimilarityMat']:
            if key in unlabeled_data_reduced:
                arr = unlabeled_data_reduced[key]
                check_array_info(f"削减后 {key}", arr, show_stats=True)
        
        # 2.4 检查归一化前后的数据
        print_separator("2.4 归一化前后对比", "-")
        
        # 归一化前
        clms_before = np.array(unlabeled_data_reduced['CLMSMat'], dtype=np.float64)
        wrf_before = np.array(unlabeled_data_reduced['WRFMat'], dtype=np.float64)
        urban_before = np.array(unlabeled_data_reduced['UrbanFeature'], dtype=np.float64)
        
        print("\n【归一化前】")
        check_array_info("CLMSMat (归一化前)", clms_before, show_stats=True)
        check_array_info("WRFMat (归一化前)", wrf_before, show_stats=True)
        check_array_info("UrbanFeature (归一化前)", urban_before, show_stats=True)
        
        # 归一化后
        clms_after, clms_min, clms_max = MinMax(clms_before)
        wrf_after, wrf_min, wrf_max = MinMax(wrf_before)
        urban_after, urban_min, urban_max = MinMax_first_dim(urban_before)
        
        print("\n【归一化后】")
        check_array_info("CLMSMat (归一化后)", clms_after, show_stats=True)
        print(f"  归一化参数 - Min: {clms_min.shape}, Max: {clms_max.shape}")
        check_array_info("WRFMat (归一化后)", wrf_after, show_stats=True)
        print(f"  归一化参数 - Min: {wrf_min.shape}, Max: {wrf_max.shape}")
        check_array_info("UrbanFeature (归一化后)", urban_after, show_stats=True)
        print(f"  归一化参数 - Min: {urban_min.shape}, Max: {urban_max.shape}")
        
        # 2.5 检查辅助变量
        print_separator("2.5 辅助变量", "-")
        nTimesteps = unlabeled_data_reduced['CLMSMat'].shape[0]
        auxiliary_vars = add_auxiliary_variables(
            unlabeled_data_reduced, 
            nStations=target_station_count, 
            nTimesteps=nTimesteps
        )
        check_array_info("辅助变量 (Auxiliary Variables)", auxiliary_vars, show_stats=True)
        print(f"  包含: 小时、月份、年份、站点编号")
        
        return unlabeled_data_reduced
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_graph_structure(data_path, unlabeled_data, n_unlabeled=200):
    """检查图结构"""
    print_separator("3. 图结构检查", "=")
    
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': 12,
        'batchSize': 128,
        'thres': 0.4,
        'geoFeatures': 'full',
    }
    
    try:
        # 构建统一图
        unified_adj, labeled_indices, unlabeled_indices, unified_locations = build_unified_graph(
            dataParam, data_path, unlabeled_data, n_unlabeled=n_unlabeled
        )
        
        print(f"统一图节点总数: {unified_adj.shape[0]}")
        print(f"有标签节点数: {len(labeled_indices)}")
        print(f"无标签节点数: {len(unlabeled_indices)}")
        
        check_array_info("统一邻接矩阵 (Unified Adjacency)", unified_adj, show_stats=True)
        
        # 图统计信息
        n_edges = (unified_adj > 0).sum() / 2  # 无向图，除以2
        density = n_edges / (unified_adj.shape[0] * (unified_adj.shape[0] - 1) / 2)
        
        print(f"\n图统计信息:")
        print(f"  边数: {int(n_edges)}")
        print(f"  图密度: {density:.6f}")
        print(f"  平均度数: {n_edges * 2 / unified_adj.shape[0]:.2f}")
        
        # 检查有标签子图
        labeled_adj = unified_adj[np.ix_(labeled_indices, labeled_indices)]
        n_labeled_edges = (labeled_adj > 0).sum() / 2
        labeled_density = n_labeled_edges / (len(labeled_indices) * (len(labeled_indices) - 1) / 2)
        print(f"\n有标签子图:")
        print(f"  边数: {int(n_labeled_edges)}")
        print(f"  图密度: {labeled_density:.6f}")
        
        # 检查无标签子图
        unlabeled_adj = unified_adj[np.ix_(unlabeled_indices, unlabeled_indices)]
        n_unlabeled_edges = (unlabeled_adj > 0).sum() / 2
        unlabeled_density = n_unlabeled_edges / (len(unlabeled_indices) * (len(unlabeled_indices) - 1) / 2)
        print(f"\n无标签子图:")
        print(f"  边数: {int(n_unlabeled_edges)}")
        print(f"  图密度: {unlabeled_density:.6f}")
        
        check_array_info("统一位置信息 (Unified Locations)", unified_locations, show_stats=True)
        
    except Exception as e:
        print(f"❌ 图结构检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_geo_features(data_path):
    """检查地理特征"""
    print_separator("4. 地理特征检查", "=")
    
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'poolSize': 12,
    }
    
    try:
        geoFeatures, off, scl, nStations = genGeoFeatures(
            data_path, 
            dataParam['geoMethod'], 
            dataParam['poolSize'], 
            dataParam['nCompPCA']
        )
        
        check_array_info("地理特征 (GeoFeatures)", geoFeatures, show_stats=True)
        print(f"  站点数量: {nStations}")
        print(f"  特征维度: {geoFeatures.shape}")
        
    except Exception as e:
        print(f"❌ 地理特征检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_feature_patches(data_path):
    """检查特征补丁文件"""
    print_separator("5. 特征补丁文件检查", "=")
    
    # 检查 FeaturePatch_201.mat
    patch_201_file = os.path.join(data_path, 'FeaturePatch_201.mat')
    if os.path.exists(patch_201_file):
        print(f"✓ 找到文件: {patch_201_file}")
        try:
            patch_201_data = mat73.loadmat(patch_201_file)
            print(f"  文件中的变量: {list(patch_201_data.keys())}")
            
            for key, value in patch_201_data.items():
                if not key.startswith('__'):  # 跳过MATLAB元数据
                    if isinstance(value, np.ndarray):
                        check_array_info(f"FeaturePatch_201.{key}", value, show_stats=True)
                    else:
                        print(f"\n  {key}: {type(value)} (非数组)")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print(f"⚠️  文件不存在: {patch_201_file}")
    
    # 检查 FeaturePatch_401.mat
    patch_401_file = os.path.join(data_path, 'FeaturePatch_401.mat')
    if os.path.exists(patch_401_file):
        print(f"\n✓ 找到文件: {patch_401_file}")
        try:
            patch_401_data = mat73.loadmat(patch_401_file)
            print(f"  文件中的变量: {list(patch_401_data.keys())}")
            
            # 详细检查 FeatureMat_zeros（这是代码中实际使用的变量）
            if 'FeatureMat_zeros' in patch_401_data:
                print_separator("5.1 FeaturePatch_401.mat 详细维度分析", "-")
                feature_mat = patch_401_data['FeatureMat_zeros']
                
                print(f"\n  FeatureMat_zeros 原始维度: {feature_mat.shape}")
                print(f"    维度含义: (imageSize, imageSize, nFeatures, nStations)")
                
                if len(feature_mat.shape) == 4:
                    imageSize, _, nFeatures, nStations = feature_mat.shape
                    print(f"    图像大小: {imageSize} × {imageSize}")
                    print(f"    特征通道数: {nFeatures}")
                    print(f"    站点数量: {nStations}")
                    
                    # 检查代码中的处理逻辑
                    print(f"\n  代码处理逻辑:")
                    print(f"    1. 删除第4个特征维度（索引4，因为所有站点都是陆地）")
                    if nFeatures > 4:
                        _idx = np.arange(nFeatures)
                        _idx = np.delete(_idx, 4)
                        print(f"    2. 保留的特征维度索引: {_idx.tolist()} (共{len(_idx)}个)")
                        print(f"    3. 处理后维度: ({imageSize}, {imageSize}, {len(_idx)}, {nStations})")
                    
                    # 检查每个特征通道的统计信息
                    print(f"\n  各特征通道的统计信息:")
                    print(f"    {'通道索引':<10} {'最小值':<15} {'最大值':<15} {'平均值':<15} {'标准差':<15} {'NaN数量':<15}")
                    print("    " + "-" * 85)
                    
                    for feat_idx in range(nFeatures):
                        feat_channel = feature_mat[:, :, feat_idx, :]
                        min_val = np.nanmin(feat_channel)
                        max_val = np.nanmax(feat_channel)
                        mean_val = np.nanmean(feat_channel)
                        std_val = np.nanstd(feat_channel)
                        nan_count = np.isnan(feat_channel).sum()
                        nan_pct = (nan_count / feat_channel.size) * 100
                        print(f"    {feat_idx:<10} {min_val:<15.6f} {max_val:<15.6f} {mean_val:<15.6f} {std_val:<15.6f} {nan_count:<15,} ({nan_pct:.2f}%)")
                    
                    # 检查每个站点的统计信息（采样）
                    print(f"\n  各站点的统计信息（前10个站点）:")
                    print(f"    {'站点索引':<10} {'最小值':<15} {'最大值':<15} {'平均值':<15} {'NaN数量':<15}")
                    print("    " + "-" * 70)
                    
                    for station_idx in range(min(10, nStations)):
                        station_data = feature_mat[:, :, :, station_idx]
                        min_val = np.nanmin(station_data)
                        max_val = np.nanmax(station_data)
                        mean_val = np.nanmean(station_data)
                        nan_count = np.isnan(station_data).sum()
                        nan_pct = (nan_count / station_data.size) * 100
                        print(f"    {station_idx:<10} {min_val:<15.6f} {max_val:<15.6f} {mean_val:<15.6f} {nan_count:<15,} ({nan_pct:.2f}%)")
                    
                    if nStations > 10:
                        print(f"    ... (还有 {nStations - 10} 个站点)")
                
                check_array_info("FeaturePatch_401.FeatureMat_zeros", feature_mat, show_stats=True)
            
            # 检查其他变量
            for key, value in patch_401_data.items():
                if not key.startswith('__') and key != 'FeatureMat_zeros':  # 跳过MATLAB元数据和已检查的变量
                    if isinstance(value, np.ndarray):
                        check_array_info(f"FeaturePatch_401.{key}", value, show_stats=True)
                    else:
                        print(f"\n  {key}: {type(value)} (非数组)")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  文件不存在: {patch_401_file}")


def check_data_consistency(data_path, unlabeled_data, n_unlabeled=200):
    """检查数据一致性"""
    print_separator("6. 数据一致性检查", "=")
    
    # 检查维度一致性
    print("\n【维度一致性检查】")
    
    # 检查时间步数
    if 'CLMSMat' in unlabeled_data and 'WRFMat' in unlabeled_data:
        clms_timesteps = unlabeled_data['CLMSMat'].shape[0]
        wrf_timesteps = unlabeled_data['WRFMat'].shape[0]
        print(f"CLMSMat 时间步数: {clms_timesteps}")
        print(f"WRFMat 时间步数: {wrf_timesteps}")
        if clms_timesteps != wrf_timesteps:
            print(f"  ⚠️  时间步数不一致！")
        else:
            print(f"  ✓ 时间步数一致")
    
    # 检查站点数一致性
    station_keys = ['Map', 'NodeLocation', 'UrbanFeature', 'SimilarityMat']
    station_counts = {}
    for key in station_keys:
        if key in unlabeled_data:
            station_counts[key] = unlabeled_data[key].shape[0]
            print(f"{key} 站点数: {station_counts[key]}")
    
    if len(set(station_counts.values())) == 1:
        print(f"  ✓ 所有站点相关变量的站点数一致: {list(station_counts.values())[0]}")
    else:
        print(f"  ⚠️  站点数不一致: {station_counts}")
    
    # 检查有标签和无标签数据的特征维度
    print("\n【特征维度检查】")
    try:
        station_data = mat73.loadmat(os.path.join(data_path, 'GNN_N1_StationMat.mat'))
        if 'StationMat_se_fill' in station_data:
            labeled_raw = station_data['StationMat_se_fill']
            labeled_raw = np.transpose(labeled_raw, (0, 2, 1))
            labeled_features = labeled_raw[:, :, 1:]
            print(f"有标签数据特征维度: {labeled_features.shape}")
            print(f"  特征数: {labeled_features.shape[-1]}")
            
            if 'WRFMat' in unlabeled_data:
                print(f"无标签数据 WRFMat 维度: {unlabeled_data['WRFMat'].shape}")
                print(f"  特征数: {unlabeled_data['WRFMat'].shape[1]}")
                
                # 检查CFD特征数（应该是54）
                if unlabeled_data['WRFMat'].shape[1] == 54:
                    print(f"  ✓ WRFMat特征数正确 (54，已合并风速)")
                else:
                    print(f"  ⚠️  WRFMat特征数异常，期望54，实际{unlabeled_data['WRFMat'].shape[1]}")
    except Exception as e:
        print(f"  ⚠️  无法检查特征维度: {e}")


def list_all_data_files(data_path):
    """列出所有数据文件"""
    print_separator("0. 数据文件夹内容清单", "=")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return
    
    print(f"数据路径: {data_path}\n")
    
    all_files = os.listdir(data_path)
    mat_files = [f for f in all_files if f.endswith('.mat')]
    other_files = [f for f in all_files if not f.endswith('.mat')]
    
    print(f"✓ 找到 {len(mat_files)} 个 .mat 文件:")
    for i, f in enumerate(sorted(mat_files), 1):
        file_path = os.path.join(data_path, f)
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
        print(f"  {i}. {f} ({file_size:.2f} MB)")
    
    if other_files:
        print(f"\n其他文件 ({len(other_files)} 个):")
        for f in sorted(other_files):
            file_path = os.path.join(data_path, f)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"  - {f} ({file_size:.2f} MB)")
    
    print(f"\n总计: {len(all_files)} 个文件")


def compare_feature_mat_distributions(data_path):
    """比较 FeaturePatch_401.mat 的 FeatureMat_zeros、Labeled_Finalized.mat 的 UrbanFeatureMat 和 Unlabeled_Finalized.mat 的 UrbanFeatureMat 的分布"""
    print_separator("7. FeatureMat 分布比较（三组数据）", "=")
    
    try:
        # 1. 加载有标签数据的 FeatureMat_zeros (来自 FeaturePatch_401.mat)
        patch_401_file = os.path.join(data_path, 'FeaturePatch_401.mat')
        if not os.path.exists(patch_401_file):
            print(f"❌ 文件不存在: {patch_401_file}")
            return
        
        patch_401_data = mat73.loadmat(patch_401_file)
        if 'FeatureMat_zeros' not in patch_401_data:
            print(f"❌ FeaturePatch_401.mat 中未找到 'FeatureMat_zeros'")
            return
        
        labeled_feat = patch_401_data['FeatureMat_zeros']  # (imageSize, imageSize, nFeatures, nStations)
        
        # 删除第4个特征维度（与代码逻辑一致）
        _idx = np.arange(labeled_feat.shape[2])
        _idx = np.delete(_idx, 4)
        labeled_feat = labeled_feat[:, :, _idx, :]
        
        print(f"\n【数据源1】FeaturePatch_401.mat -> FeatureMat_zeros (删除第4个特征后)")
        print(f"  维度: {labeled_feat.shape}")
        print(f"  特征通道数: {labeled_feat.shape[2]}")
        print(f"  站点数: {labeled_feat.shape[3]}")
        
        # 2. 加载有标签数据的 UrbanFeatureMat (来自 Labeled_Finalized.mat)
        labeled_finalized_file = os.path.join(data_path, 'Labeled_Finalized.mat')
        labeled_finalized_feat = None
        if os.path.exists(labeled_finalized_file):
            labeled_finalized_data = mat73.loadmat(labeled_finalized_file)
            if 'UrbanFeatureMat' in labeled_finalized_data:
                labeled_finalized_feat = labeled_finalized_data['UrbanFeatureMat']  # (imageSize, imageSize, nFeatures, nStations)
                # 不删除第4个特征维度，保持原样
                
                print(f"\n【数据源2】Labeled_Finalized.mat -> UrbanFeatureMat (原始数据，未删除特征)")
                print(f"  维度: {labeled_finalized_feat.shape}")
                print(f"  特征通道数: {labeled_finalized_feat.shape[2]}")
                print(f"  站点数: {labeled_finalized_feat.shape[3]}")
            else:
                print(f"\n⚠️  Labeled_Finalized.mat 中未找到 'UrbanFeatureMat'，跳过此数据源")
        else:
            print(f"\n⚠️  文件不存在: {labeled_finalized_file}，跳过此数据源")
        
        # 3. 加载无标签数据的 UrbanFeatureMat (来自 Unlabeled_Finalized.mat)
        unlabeled_file = os.path.join(data_path, 'Unlabeled_Finalized.mat')
        if not os.path.exists(unlabeled_file):
            print(f"❌ 文件不存在: {unlabeled_file}")
            return
        
        unlabeled_data = mat73.loadmat(unlabeled_file)
        if 'UrbanFeatureMat' not in unlabeled_data:
            print(f"❌ Unlabeled_Finalized.mat 中未找到 'UrbanFeatureMat'")
            return
        
        unlabeled_feat = unlabeled_data['UrbanFeatureMat']  # (imageSize, imageSize, nFeatures, nStations)
        # 不删除第4个特征维度，保持原样
        
        print(f"\n【数据源3】Unlabeled_Finalized.mat -> UrbanFeatureMat (原始数据，未删除特征)")
        print(f"  维度: {unlabeled_feat.shape}")
        print(f"  特征通道数: {unlabeled_feat.shape[2]}")
        print(f"  站点数: {unlabeled_feat.shape[3]}")
        
        # 检查维度是否匹配
        all_shapes = [labeled_feat.shape[:3]]
        all_names = ["FeaturePatch_401"]
        if labeled_finalized_feat is not None:
            all_shapes.append(labeled_finalized_feat.shape[:3])
            all_names.append("Labeled_Finalized")
        all_shapes.append(unlabeled_feat.shape[:3])
        all_names.append("Unlabeled_Finalized")
        
        if len(set(all_shapes)) > 1:
            print(f"\n⚠️  维度不匹配！")
            for name, shape in zip(all_names, all_shapes):
                print(f"  {name}: {shape}")
        else:
            print(f"\n✓ 所有数据源的图像大小和特征通道数匹配")
        
        # 确定要比较的特征通道数
        # 第一个数据源删除了第4个特征，所以特征数可能比其他两个少1
        nFeatures_labeled = labeled_feat.shape[2]  # 第一个数据源的特征数（已删除第4个，应该是7）
        nFeatures_others = unlabeled_feat.shape[2]  # 其他数据源的特征数（全部保留，应该是7）
        
        # 创建索引映射：第一个数据源的索引 -> 其他数据源的索引
        # 第一个数据源删除了索引4，所以：
        # 第一个数据源的索引0-3 -> 其他数据源的索引0-3
        # 映射关系：由于都是删除了索引4后的结果，所以映射是直接的
        # 第一个数据源的新索引0-6 -> 其他数据源的索引0-6（一一对应）
        def map_index_to_others(idx_in_labeled):
            """将第一个数据源的索引映射到其他数据源的索引
            数据源1：新索引0-6对应原始索引0-3, 5-7
            数据源2/3：索引0-6对应原始索引0-3, 5-7（也删除了索引4）
            所以映射关系是：新索引0-3->0-3, 新索引4-6->4-6
            """
            # 数据源1的新索引直接对应数据源2/3的索引（都是删除了索引4后的结果）
            return idx_in_labeled
        
        # 比较逻辑：只比较数据源1中存在的特征
        # 数据源2和3使用所有7个特征（索引0-6），全部保留
        nFeatures = nFeatures_labeled  # 只比较数据源1中存在的特征
        
        # 比较每个特征通道的分布
        print_separator("7.1 各特征通道分布比较", "-")
        print(f"{'通道':<8} {'数据源':<25} {'最小值':<15} {'最大值':<15} {'平均值':<15} {'标准差':<15} {'NaN数量':<12}")
        print("-" * 120)
        print(f"注意: FeaturePatch_401原始8个特征，删除索引4后变成7个；其他数据源原始7个特征，全部保留")
        print()
        
        for feat_idx in range(nFeatures):
            # 数据源1: FeaturePatch_401 (已删除第4个特征)
            labeled_channel = labeled_feat[:, :, feat_idx, :]
            labeled_min = np.nanmin(labeled_channel)
            labeled_max = np.nanmax(labeled_channel)
            labeled_mean = np.nanmean(labeled_channel)
            labeled_std = np.nanstd(labeled_channel)
            labeled_nan = np.isnan(labeled_channel).sum()
            
            print(f"{feat_idx:<8} {'FeaturePatch_401':<25} {labeled_min:<15.6f} {labeled_max:<15.6f} "
                  f"{labeled_mean:<15.6f} {labeled_std:<15.6f} {labeled_nan:<12,}")
            
            # 数据源2和3: 使用映射后的索引
            mapped_idx = map_index_to_others(feat_idx)
            
            # 数据源2: Labeled_Finalized (如果存在)
            if labeled_finalized_feat is not None:
                if mapped_idx < labeled_finalized_feat.shape[2]:
                    labeled_final_channel = labeled_finalized_feat[:, :, mapped_idx, :]
                    labeled_final_min = np.nanmin(labeled_final_channel)
                    labeled_final_max = np.nanmax(labeled_final_channel)
                    labeled_final_mean = np.nanmean(labeled_final_channel)
                    labeled_final_std = np.nanstd(labeled_final_channel)
                    labeled_final_nan = np.isnan(labeled_final_channel).sum()
                    
                    print(f"{mapped_idx:<8} {'Labeled_Finalized':<25} {labeled_final_min:<15.6f} {labeled_final_max:<15.6f} "
                          f"{labeled_final_mean:<15.6f} {labeled_final_std:<15.6f} {labeled_final_nan:<12,}")
                else:
                    print(f"{'':<8} {'Labeled_Finalized':<25} {'N/A (索引超出)':<15}")
            
            # 数据源3: Unlabeled_Finalized
            if mapped_idx < unlabeled_feat.shape[2]:
                unlabeled_channel = unlabeled_feat[:, :, mapped_idx, :]
                unlabeled_min = np.nanmin(unlabeled_channel)
                unlabeled_max = np.nanmax(unlabeled_channel)
                unlabeled_mean = np.nanmean(unlabeled_channel)
                unlabeled_std = np.nanstd(unlabeled_channel)
                unlabeled_nan = np.isnan(unlabeled_channel).sum()
                
                print(f"{mapped_idx:<8} {'Unlabeled_Finalized':<25} {unlabeled_min:<15.6f} {unlabeled_max:<15.6f} "
                      f"{unlabeled_mean:<15.6f} {unlabeled_std:<15.6f} {unlabeled_nan:<12,}")
            else:
                print(f"{'':<8} {'Unlabeled_Finalized':<25} {'N/A (索引超出)':<15}")
            
            # 计算分布差异（比较所有可用的数据源）
            all_mins = [labeled_min]
            all_maxs = [labeled_max]
            all_means = [labeled_mean]
            if labeled_finalized_feat is not None and mapped_idx < labeled_finalized_feat.shape[2]:
                all_mins.append(labeled_final_min)
                all_maxs.append(labeled_final_max)
                all_means.append(labeled_final_mean)
            if mapped_idx < unlabeled_feat.shape[2]:
                all_mins.append(unlabeled_min)
                all_maxs.append(unlabeled_max)
                all_means.append(unlabeled_mean)
            
            overall_min = min(all_mins)
            overall_max = max(all_maxs)
            overall_range = overall_max - overall_min
            
            # 计算范围重叠
            range_overlap = min(all_maxs) - max(all_mins)
            overlap_ratio = range_overlap / overall_range if overall_range > 0 else 0
            
            # 计算均值差异
            mean_diff = max(all_means) - min(all_means)
            mean_diff_relative = mean_diff / max(abs(max(all_means)), abs(min(all_means)), 1)
            
            if overlap_ratio > 0.8 and mean_diff_relative < 0.1:
                print(f"{'':<8} {'→ 分布相似':<25} 范围重叠: {overlap_ratio*100:.1f}%, 均值差异: {mean_diff:.6f}")
            else:
                print(f"{'':<8} {'→ ⚠️ 分布差异':<25} 范围重叠: {overlap_ratio*100:.1f}%, 均值差异: {mean_diff:.6f}")
            
            print()
        
        # 整体分布比较
        print_separator("7.2 整体分布统计", "-")
        
        # 展平所有数据进行比较
        labeled_flat = labeled_feat.flatten()
        unlabeled_flat = unlabeled_feat.flatten()
        
        print(f"【整体统计】")
        print(f"  FeaturePatch_401 - 最小值: {np.nanmin(labeled_flat):.6f}, 最大值: {np.nanmax(labeled_flat):.6f}, "
              f"平均值: {np.nanmean(labeled_flat):.6f}, 标准差: {np.nanstd(labeled_flat):.6f}")
        
        if labeled_finalized_feat is not None:
            labeled_final_flat = labeled_finalized_feat.flatten()
            print(f"  Labeled_Finalized - 最小值: {np.nanmin(labeled_final_flat):.6f}, 最大值: {np.nanmax(labeled_final_flat):.6f}, "
                  f"平均值: {np.nanmean(labeled_final_flat):.6f}, 标准差: {np.nanstd(labeled_final_flat):.6f}")
        
        print(f"  Unlabeled_Finalized - 最小值: {np.nanmin(unlabeled_flat):.6f}, 最大值: {np.nanmax(unlabeled_flat):.6f}, "
              f"平均值: {np.nanmean(unlabeled_flat):.6f}, 标准差: {np.nanstd(unlabeled_flat):.6f}")
        
        # 计算分位数
        percentiles = [5, 25, 50, 75, 95]
        print(f"\n【分位数比较】")
        if labeled_finalized_feat is not None:
            print(f"{'分位数':<10} {'FeaturePatch_401':<20} {'Labeled_Finalized':<20} {'Unlabeled_Finalized':<20}")
            print("-" * 70)
            labeled_final_flat = labeled_finalized_feat.flatten()
            for p in percentiles:
                labeled_p = np.nanpercentile(labeled_flat, p)
                labeled_final_p = np.nanpercentile(labeled_final_flat, p)
                unlabeled_p = np.nanpercentile(unlabeled_flat, p)
                print(f"{p}%{'':<6} {labeled_p:<20.6f} {labeled_final_p:<20.6f} {unlabeled_p:<20.6f}")
        else:
            print(f"{'分位数':<10} {'FeaturePatch_401':<20} {'Unlabeled_Finalized':<20} {'差异':<15}")
            print("-" * 65)
            for p in percentiles:
                labeled_p = np.nanpercentile(labeled_flat, p)
                unlabeled_p = np.nanpercentile(unlabeled_flat, p)
                diff = abs(labeled_p - unlabeled_p)
                print(f"{p}%{'':<6} {labeled_p:<20.6f} {unlabeled_p:<20.6f} {diff:<15.6f}")
        
        # 结论
        print_separator("7.3 分布一致性结论", "-")
        
        # 计算所有数据源的整体范围
        all_flats = [labeled_flat]
        if labeled_finalized_feat is not None:
            all_flats.append(labeled_finalized_feat.flatten())
        all_flats.append(unlabeled_flat)
        
        all_mins_flat = [np.nanmin(f) for f in all_flats]
        all_maxs_flat = [np.nanmax(f) for f in all_flats]
        all_means_flat = [np.nanmean(f) for f in all_flats]
        all_stds_flat = [np.nanstd(f) for f in all_flats]
        
        overall_min_flat = min(all_mins_flat)
        overall_max_flat = max(all_maxs_flat)
        overall_range_flat = overall_max_flat - overall_min_flat
        
        # 计算范围重叠
        overall_overlap = min(all_maxs_flat) - max(all_mins_flat)
        overall_overlap_ratio = overall_overlap / overall_range_flat if overall_range_flat > 0 else 0
        
        # 计算均值差异
        mean_diff_overall = max(all_means_flat) - min(all_means_flat)
        mean_diff_relative = mean_diff_overall / max(abs(max(all_means_flat)), abs(min(all_means_flat)), 1)
        
        # 计算标准差差异
        std_diff_overall = max(all_stds_flat) - min(all_stds_flat)
        
        print(f"范围重叠比例: {overall_overlap_ratio*100:.2f}%")
        print(f"均值差异: {mean_diff_overall:.6f} (相对差异: {mean_diff_relative*100:.2f}%)")
        print(f"标准差差异: {std_diff_overall:.6f}")
        
        if overall_overlap_ratio > 0.7 and mean_diff_relative < 0.1:
            print(f"\n✓ 结论: 所有数据源分布**相似**，各自归一化到[0,1]应该可行")
        else:
            print(f"\n⚠️  结论: 数据源分布存在**差异**，建议统一归一化参数")
        
    except Exception as e:
        print(f"❌ 比较失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print_separator("=" * 80)
    print("全面的数据文件检查脚本".center(80))
    print_separator("=" * 80)
    
    # 数据路径
    DATA_PATH = '/projects/p32685/Fixmatch/data'
    n_unlabeled = 200  # 默认200个无标签站点
    
    print(f"数据路径: {DATA_PATH}")
    print(f"目标无标签站点数: {n_unlabeled}")
    
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ 数据路径不存在: {DATA_PATH}")
        return
    
    # 0. 列出所有文件
    list_all_data_files(DATA_PATH)
    
    # 1. 检查有标签数据
    check_labeled_data(DATA_PATH)
    
    # 2. 检查无标签数据
    unlabeled_data = check_unlabeled_data(DATA_PATH, target_station_count=n_unlabeled)
    
    if unlabeled_data is None:
        print("\n❌ 无标签数据检查失败，跳过后续检查")
        return
    
    # 3. 检查图结构
    check_graph_structure(DATA_PATH, unlabeled_data, n_unlabeled=n_unlabeled)
    
    # 4. 检查地理特征
    check_geo_features(DATA_PATH)
    
    # 5. 检查特征补丁文件
    check_feature_patches(DATA_PATH)
    
    # 6. 检查数据一致性
    check_data_consistency(DATA_PATH, unlabeled_data, n_unlabeled=n_unlabeled)
    
    # 7. 比较 FeatureMat 分布
    compare_feature_mat_distributions(DATA_PATH)
    
    print_separator("=" * 80)
    print("检查完成！".center(80))
    print_separator("=" * 80)


if __name__ == "__main__":
    main()



"""
全面的数据文件检查脚本
检查所有数据文件的维度、取值范围、统计信息等基础信息
"""
import os
import numpy as np
import mat73
from data_preprocessing import (
    merge_wind_components, 
    reduce_stations, 
    preprocess_unlabeled_data
)
from data_generation import build_unified_graph
from utils import MinMax, MinMax_first_dim, add_auxiliary_variables
from geo_features import genGeoFeatures


def print_separator(title="", char="=", length=80):
    """打印分隔线"""
    if title:
        print(f"\n{char * length}")
        print(f"{title:^{length}}")
        print(f"{char * length}\n")
    else:
        print(f"{char * length}\n")


def check_array_info(name, arr, show_stats=True, show_sample=False):
    """
    检查数组的详细信息
    
    Args:
        name: 数组名称
        arr: numpy数组
        show_stats: 是否显示统计信息
        show_sample: 是否显示样本值
    """
    print(f"\n【{name}】")
    print(f"  形状 (Shape): {arr.shape}")
    print(f"  数据类型 (Dtype): {arr.dtype}")
    print(f"  内存大小 (Size): {arr.nbytes / 1024 / 1024:.2f} MB")
    
    if show_stats:
        if arr.size > 0:
            print(f"  最小值 (Min): {np.nanmin(arr):.6f}")
            print(f"  最大值 (Max): {np.nanmax(arr):.6f}")
            print(f"  平均值 (Mean): {np.nanmean(arr):.6f}")
            print(f"  标准差 (Std): {np.nanstd(arr):.6f}")
            
            # 检查NaN和Inf
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0:
                print(f"  ⚠️  NaN数量: {nan_count} ({nan_count/arr.size*100:.2f}%)")
            if inf_count > 0:
                print(f"  ⚠️  Inf数量: {inf_count} ({inf_count/arr.size*100:.2f}%)")
            
            # 检查零值
            zero_count = (arr == 0).sum()
            if zero_count > 0:
                print(f"  零值数量: {zero_count} ({zero_count/arr.size*100:.2f}%)")
    
    if show_sample and arr.size > 0:
        print(f"  样本值 (前5个): {arr.flat[:5]}")


def check_each_feature_variable(features, targets, feature_names=None):
    """
    详细检查每个特征变量的信息
    
    Args:
        features: 特征矩阵 (timestep, nNodes, nFeatures)
        targets: 目标变量 (timestep, nNodes)
        feature_names: 可选的特征名称列表
    """
    print_separator("详细特征变量分析", "-")
    
    nTimesteps, nNodes, nFeatures = features.shape
    
    print(f"数据维度: {features.shape}")
    print(f"  时间步数: {nTimesteps}")
    print(f"  节点数: {nNodes}")
    print(f"  特征数: {nFeatures}")
    
    # 为每个特征创建详细报告
    feature_info = []
    
    for feat_idx in range(nFeatures):
        feat_data = features[:, :, feat_idx]  # (timestep, nNodes)
        
        # 计算统计信息
        min_val = np.nanmin(feat_data)
        max_val = np.nanmax(feat_data)
        mean_val = np.nanmean(feat_data)
        std_val = np.nanstd(feat_data)
        
        # 检查是否为静态特征（所有时间步的值都相同）
        # 方法1: 检查每个节点在不同时间步的值是否变化
        node_std = np.std(feat_data, axis=0)  # 每个节点在不同时间步的标准差
        is_static_by_node = (node_std < 1e-6).sum()  # 标准差接近0的节点数
        static_ratio = is_static_by_node / nNodes
        
        # 方法2: 检查所有值是否相同
        all_values_same = np.allclose(feat_data, feat_data[0, 0])
        
        # 方法3: 计算时间维度上的变化程度
        time_variance = np.var(feat_data, axis=0).mean()  # 平均时间方差
        
        # 判断特征类型
        if all_values_same or static_ratio > 0.95:
            feature_type = "静态 (Static)"
        elif time_variance < 1e-6:
            feature_type = "准静态 (Quasi-Static)"
        else:
            feature_type = "动态 (Dynamic)"
        
        feature_info.append({
            'idx': feat_idx,
            'name': feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"Feature_{feat_idx}",
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'static_ratio': static_ratio,
            'time_variance': time_variance,
            'type': feature_type,
            'all_same': all_values_same
        })
    
    # 打印详细表格
    print(f"\n{'索引':<6} {'特征名':<20} {'类型':<15} {'最小值':<12} {'最大值':<12} {'平均值':<12} {'标准差':<12} {'静态比例':<10}")
    print("-" * 120)
    
    for info in feature_info:
        print(f"{info['idx']:<6} {info['name']:<20} {info['type']:<15} "
              f"{info['min']:<12.6f} {info['max']:<12.6f} {info['mean']:<12.6f} "
              f"{info['std']:<12.6f} {info['static_ratio']*100:<9.1f}%")
    
    # 按类型分组统计
    print_separator("特征类型统计", "-")
    static_features = [f for f in feature_info if '静态' in f['type']]
    dynamic_features = [f for f in feature_info if '动态' in f['type']]
    
    print(f"静态特征数量: {len(static_features)}")
    if static_features:
        print(f"  索引范围: {min(f['idx'] for f in static_features)} - {max(f['idx'] for f in static_features)}")
        print(f"  索引列表: {[f['idx'] for f in static_features]}")
    
    print(f"\n动态特征数量: {len(dynamic_features)}")
    if dynamic_features:
        print(f"  索引范围: {min(f['idx'] for f in dynamic_features)} - {max(f['idx'] for f in dynamic_features)}")
    
    return feature_info


def check_labeled_data(data_path):
    """检查有标签数据文件"""
    print_separator("1. 有标签数据文件检查", "=")
    
    # 1.1 检查 StationMat 文件
    station_file = os.path.join(data_path, 'GNN_N1_StationMat.mat')
    if os.path.exists(station_file):
        print(f"✓ 找到文件: {station_file}")
        try:
            station_data = mat73.loadmat(station_file)
            print(f"  文件中的变量: {list(station_data.keys())}")
            
            # 检查 StationMat_se (原始数据，可能有缺失值)
            if 'StationMat_se' in station_data:
                print_separator("1.0.1 StationMat_se (原始数据) 缺失值检查", "-")
                raw_se = station_data['StationMat_se']
                print(f"  StationMat_se 原始维度: {raw_se.shape}")
                
                # 转置后检查
                raw_se_transposed = np.transpose(raw_se, (0, 2, 1))
                print(f"  转置后维度 (timestep * nNodes * nFeatures): {raw_se_transposed.shape}")
                
                # 检查整体缺失情况
                total_elements = raw_se.size
                nan_count = np.isnan(raw_se).sum()
                nan_percentage = (nan_count / total_elements) * 100
                
                print(f"\n  整体缺失情况:")
                print(f"    总元素数: {total_elements:,}")
                print(f"    NaN数量: {nan_count:,} ({nan_percentage:.2f}%)")
                
                # 检查每个特征维度的缺失情况
                features_se = raw_se_transposed[:, :, 1:]  # 去掉温度目标
                targets_se = raw_se_transposed[:, :, 0]    # 温度目标
                
                print(f"\n  目标变量 (温度) 缺失情况:")
                target_nan = np.isnan(targets_se).sum()
                target_nan_pct = (target_nan / targets_se.size) * 100
                print(f"    NaN数量: {target_nan:,} ({target_nan_pct:.2f}%)")
                
                print(f"\n  特征矩阵缺失情况:")
                feature_nan = np.isnan(features_se).sum()
                feature_nan_pct = (feature_nan / features_se.size) * 100
                print(f"    NaN数量: {feature_nan:,} ({feature_nan_pct:.2f}%)")
                
                # 检查每个特征维度的缺失情况
                print(f"\n  各特征维度的缺失情况:")
                print(f"    {'特征索引':<10} {'NaN数量':<15} {'NaN比例':<15} {'缺失站点数':<15}")
                print("    " + "-" * 55)
                
                for feat_idx in range(features_se.shape[-1]):
                    feat_data = features_se[:, :, feat_idx]
                    feat_nan = np.isnan(feat_data).sum()
                    feat_nan_pct = (feat_nan / feat_data.size) * 100
                    # 计算有多少个站点在这个特征上有缺失
                    stations_with_nan = (np.isnan(feat_data).any(axis=0)).sum()
                    print(f"    {feat_idx:<10} {feat_nan:<15,} {feat_nan_pct:<14.2f}% {stations_with_nan:<15}")
                
                # 检查每个站点的缺失情况
                print(f"\n  各站点的缺失情况:")
                print(f"    {'站点索引':<10} {'NaN数量':<15} {'NaN比例':<15} {'缺失特征数':<15}")
                print("    " + "-" * 55)
                
                for node_idx in range(features_se.shape[1]):
                    node_data = features_se[:, node_idx, :]
                    node_nan = np.isnan(node_data).sum()
                    node_nan_pct = (node_nan / node_data.size) * 100
                    # 计算有多少个特征在这个站点上有缺失
                    features_with_nan = (np.isnan(node_data).any(axis=0)).sum()
                    print(f"    {node_idx:<10} {node_nan:<15,} {node_nan_pct:<14.2f}% {features_with_nan:<15}")
                
                # 检查时间步的缺失情况
                print(f"\n  各时间步的缺失情况:")
                print(f"    {'时间步索引':<12} {'NaN数量':<15} {'NaN比例':<15}")
                print("    " + "-" * 42)
                
                # 只显示前10个和后10个时间步，以及缺失最多的几个
                time_nan_counts = []
                for t_idx in range(features_se.shape[0]):
                    time_data = features_se[t_idx, :, :]
                    time_nan = np.isnan(time_data).sum()
                    time_nan_pct = (time_nan / time_data.size) * 100
                    time_nan_counts.append((t_idx, time_nan, time_nan_pct))
                
                # 显示前5个和后5个
                print("    前5个时间步:")
                for t_idx, t_nan, t_pct in time_nan_counts[:5]:
                    print(f"    {t_idx:<12} {t_nan:<15,} {t_pct:<14.2f}%")
                
                print("    后5个时间步:")
                for t_idx, t_nan, t_pct in time_nan_counts[-5:]:
                    print(f"    {t_idx:<12} {t_nan:<15,} {t_pct:<14.2f}%")
                
                # 显示缺失最多的5个时间步
                time_nan_counts_sorted = sorted(time_nan_counts, key=lambda x: x[1], reverse=True)
                print("    缺失最多的5个时间步:")
                for t_idx, t_nan, t_pct in time_nan_counts_sorted[:5]:
                    print(f"    {t_idx:<12} {t_nan:<15,} {t_pct:<14.2f}%")
            
            # 检查 StationMat_se_fill (填充后的数据)
            if 'StationMat_se_fill' in station_data:
                print_separator("1.0.2 StationMat_se_fill (填充后数据) 对比", "-")
                raw = station_data['StationMat_se_fill']
                print(f"  StationMat_se_fill 原始维度: {raw.shape}")
                
                # 检查填充后是否还有缺失值
                fill_nan_count = np.isnan(raw).sum()
                if fill_nan_count > 0:
                    print(f"  ⚠️  填充后仍有 NaN: {fill_nan_count:,} ({fill_nan_count/raw.size*100:.2f}%)")
                else:
                    print(f"  ✓ 填充后无 NaN 值")
                
                # 如果两个都存在，进行对比
                if 'StationMat_se' in station_data:
                    raw_se = station_data['StationMat_se']
                    raw_se_transposed = np.transpose(raw_se, (0, 2, 1))
                    raw_transposed = np.transpose(raw, (0, 2, 1))
                    
                    print(f"\n  填充前后对比:")
                    se_nan = np.isnan(raw_se_transposed).sum()
                    fill_nan = np.isnan(raw_transposed).sum()
                    print(f"    StationMat_se NaN: {se_nan:,} ({se_nan/raw_se_transposed.size*100:.2f}%)")
                    print(f"    StationMat_se_fill NaN: {fill_nan:,} ({fill_nan/raw_transposed.size*100:.2f}%)")
                    print(f"    填充的NaN数量: {se_nan - fill_nan:,}")
                    
                    # 检查填充方法（通过对比非NaN位置的值是否相同）
                    non_nan_mask = ~np.isnan(raw_se_transposed)
                    if non_nan_mask.sum() > 0:
                        values_match = np.allclose(
                            raw_se_transposed[non_nan_mask], 
                            raw_transposed[non_nan_mask], 
                            equal_nan=True
                        )
                        if values_match:
                            print(f"    ✓ 非NaN位置的值保持一致（填充未改变原始值）")
                        else:
                            print(f"    ⚠️  非NaN位置的值有差异（可能进行了平滑处理）")
                
                print_separator("1.1 StationMat_se_fill 详细分析", "=")
                
                # 转置后检查
                raw_transposed = np.transpose(raw, (0, 2, 1))
                print(f"  转置后维度 (timestep * nNodes * nFeatures): {raw_transposed.shape}")
                
                # 提取特征和目标
                features = raw_transposed[:, :, 1:]  # 去掉温度目标
                targets = raw_transposed[:, :, 0]     # 温度目标
                
                print(f"\n  注意: 索引0是温度目标，特征从索引1开始")
                print(f"  实际特征索引范围: 1-{raw_transposed.shape[-1]-1} (共{features.shape[-1]}个特征)")
                
                check_array_info("特征矩阵 (Features)", features, show_stats=True)
                check_array_info("目标变量 (Targets - 温度)", targets, show_stats=True)
                
                # 详细检查每个特征变量
                print_separator("1.1.1 每个特征变量的详细分析", "=")
                
                # 创建特征名称（基于已知的划分）
                nCFDFeats = 54
                feature_names = []
                for i in range(features.shape[-1]):
                    if i < nCFDFeats:
                        feature_names.append(f"CFD_{i}")
                    elif i < nCFDFeats + 4:
                        feature_names.append(f"Unknown_{i}")
                    elif i < features.shape[-1] - 3:
                        feature_names.append(f"Geo_{i}")
                    else:
                        feature_names.append(f"CLMS_{i}")
                
                feature_info = check_each_feature_variable(features, targets, feature_names)
                
                # 检查特征维度划分
                print_separator("1.1.2 特征索引划分验证", "=")
                nCFDFeats = 54
                cfdIdx = np.arange(nCFDFeats)
                rawGeoFeatIdx = np.arange(nCFDFeats + 4, features.shape[-1] - 3 - 1)
                clmsIdx = np.arange(features.shape[-1] - 3 - 1, features.shape[-1] - 1)
                unknownIdx = np.arange(nCFDFeats, nCFDFeats + 4)  # 被跳过的4个特征
                
                print(f"\n当前代码的划分:")
                print(f"  CFD特征索引: {cfdIdx[0]}-{cfdIdx[-1]} (共{len(cfdIdx)}个)")
                print(f"  未知特征索引: {unknownIdx[0]}-{unknownIdx[-1]} (共{len(unknownIdx)}个) - 被跳过")
                print(f"  静态地理特征索引: {rawGeoFeatIdx[0]}-{rawGeoFeatIdx[-1]} (共{len(rawGeoFeatIdx)}个)")
                print(f"  CLMS动态特征索引: {clmsIdx[0]}-{clmsIdx[-1]} (共{len(clmsIdx)}个)")
                
                # 验证每个分组的特征类型
                print(f"\n验证各分组的特征类型:")
                
                # CFD特征
                cfd_types = [feature_info[i]['type'] for i in cfdIdx]
                cfd_static_count = sum(1 for t in cfd_types if '静态' in t)
                print(f"  CFD特征 (0-{cfdIdx[-1]}): {cfd_static_count}/{len(cfdIdx)} 个静态, {len(cfdIdx)-cfd_static_count} 个动态")
                
                # 未知特征
                if len(unknownIdx) > 0:
                    unknown_types = [feature_info[i]['type'] for i in unknownIdx]
                    unknown_static_count = sum(1 for t in unknown_types if '静态' in t)
                    print(f"  未知特征 ({unknownIdx[0]}-{unknownIdx[-1]}): {unknown_static_count}/{len(unknownIdx)} 个静态, {len(unknownIdx)-unknown_static_count} 个动态")
                    print(f"    详细: {[feature_info[i]['name'] + ':' + feature_info[i]['type'] for i in unknownIdx]}")
                
                # 静态地理特征
                geo_types = [feature_info[i]['type'] for i in rawGeoFeatIdx]
                geo_static_count = sum(1 for t in geo_types if '静态' in t)
                print(f"  静态地理特征 ({rawGeoFeatIdx[0]}-{rawGeoFeatIdx[-1]}): {geo_static_count}/{len(rawGeoFeatIdx)} 个静态, {len(rawGeoFeatIdx)-geo_static_count} 个动态")
                
                # CLMS特征
                clms_types = [feature_info[i]['type'] for i in clmsIdx]
                clms_static_count = sum(1 for t in clms_types if '静态' in t)
                print(f"  CLMS特征 ({clmsIdx[0]}-{clmsIdx[-1]}): {clms_static_count}/{len(clmsIdx)} 个静态, {len(clmsIdx)-clms_static_count} 个动态")
                
                # 显示每个分组的取值范围
                print(f"\n各分组的取值范围:")
                print(f"  CFD特征:")
                cfd_mins = [feature_info[i]['min'] for i in cfdIdx]
                cfd_maxs = [feature_info[i]['max'] for i in cfdIdx]
                print(f"    最小值范围: [{min(cfd_mins):.6f}, {max(cfd_mins):.6f}]")
                print(f"    最大值范围: [{min(cfd_maxs):.6f}, {max(cfd_maxs):.6f}]")
                
                if len(unknownIdx) > 0:
                    print(f"  未知特征:")
                    unknown_mins = [feature_info[i]['min'] for i in unknownIdx]
                    unknown_maxs = [feature_info[i]['max'] for i in unknownIdx]
                    print(f"    最小值范围: [{min(unknown_mins):.6f}, {max(unknown_mins):.6f}]")
                    print(f"    最大值范围: [{min(unknown_maxs):.6f}, {max(unknown_maxs):.6f}]")
                
                print(f"  静态地理特征:")
                geo_mins = [feature_info[i]['min'] for i in rawGeoFeatIdx]
                geo_maxs = [feature_info[i]['max'] for i in rawGeoFeatIdx]
                print(f"    最小值范围: [{min(geo_mins):.6f}, {max(geo_mins):.6f}]")
                print(f"    最大值范围: [{min(geo_maxs):.6f}, {max(geo_maxs):.6f}]")
                
                print(f"  CLMS特征:")
                clms_mins = [feature_info[i]['min'] for i in clmsIdx]
                clms_maxs = [feature_info[i]['max'] for i in clmsIdx]
                print(f"    最小值范围: [{min(clms_mins):.6f}, {max(clms_mins):.6f}]")
                print(f"    最大值范围: [{min(clms_maxs):.6f}, {max(clms_maxs):.6f}]")
                
            else:
                print(f"  ⚠️  未找到 'StationMat_se_fill' 变量")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ 文件不存在: {station_file}")
    
    # 1.2 检查 AJM 文件（位置和相似性）
    ajm_file = os.path.join(data_path, 'GNN_N1_AJM.mat')
    if os.path.exists(ajm_file):
        print(f"\n✓ 找到文件: {ajm_file}")
        try:
            ajm_data = mat73.loadmat(ajm_file)
            print(f"  文件中的变量: {list(ajm_data.keys())}")
            
            if 'location' in ajm_data:
                location = ajm_data['location']
                check_array_info("站点位置 (Location)", location, show_stats=True)
                print(f"  站点数量: {location.shape[0]}")
            
            if 'similarity' in ajm_data:
                similarity = ajm_data['similarity']
                check_array_info("相似性矩阵 (Similarity)", similarity, show_stats=True)
                print(f"  矩阵维度: {similarity.shape}")
                print(f"  是否对称: {np.allclose(similarity, similarity.T)}")
                
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print(f"❌ 文件不存在: {ajm_file}")
    
    # 1.3 检查 Labeled_Finalized.mat 文件
    labeled_finalized_file = os.path.join(data_path, 'Labeled_Finalized.mat')
    if os.path.exists(labeled_finalized_file):
        print(f"\n✓ 找到文件: {labeled_finalized_file}")
        try:
            labeled_finalized_data = mat73.loadmat(labeled_finalized_file)
            print(f"  文件中的变量: {list(labeled_finalized_data.keys())}")
            
            for key, value in labeled_finalized_data.items():
                if not key.startswith('__'):  # 跳过MATLAB元数据
                    if isinstance(value, np.ndarray):
                        check_array_info(f"Labeled_Finalized.{key}", value, show_stats=True)
                    else:
                        print(f"\n  {key}: {type(value)} (非数组)")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print(f"\n⚠️  文件不存在: {labeled_finalized_file}")


def check_unlabeled_data(data_path, target_station_count=200):
    """检查无标签数据文件"""
    print_separator("2. 无标签数据文件检查", "=")
    
    unlabeled_file = os.path.join(data_path, 'Unlabeled_Finalized.mat')
    if not os.path.exists(unlabeled_file):
        print(f"❌ 文件不存在: {unlabeled_file}")
        return None
    
    print(f"✓ 找到文件: {unlabeled_file}")
    
    # 2.1 检查原始数据
    print_separator("2.1 原始无标签数据", "-")
    try:
        unlabeled_data = mat73.loadmat(unlabeled_file)
        print(f"文件中的变量: {list(unlabeled_data.keys())}")
        
        # 检查各个变量
        for key in ['CLMSMat', 'WRFMat', 'Map', 'NodeLocation', 'UrbanFeature', 
                   'SimilarityMat', 'UrbanFeatureMat']:
            if key in unlabeled_data:
                arr = unlabeled_data[key]
                check_array_info(f"原始 {key}", arr, show_stats=True)
            else:
                print(f"\n⚠️  未找到变量: {key}")
        
        # 2.2 检查合并风速后的数据
        print_separator("2.2 合并风速分量后", "-")
        unlabeled_data_merged = merge_wind_components(unlabeled_data.copy())
        check_array_info("合并后 WRFMat", unlabeled_data_merged['WRFMat'], show_stats=True)
        print(f"  原始WRFMat维度: {unlabeled_data['WRFMat'].shape}")
        print(f"  合并后WRFMat维度: {unlabeled_data_merged['WRFMat'].shape}")
        
        # 2.3 检查削减站点后的数据
        print_separator("2.3 削减站点后", "-")
        original_count = unlabeled_data['Map'].shape[0]
        print(f"原始站点数量: {original_count}")
        print(f"目标站点数量: {target_station_count}")
        
        # 设置随机种子以确保可复现
        np.random.seed(42)
        unlabeled_data_reduced = reduce_stations(
            unlabeled_data_merged.copy(), 
            target_station_count=target_station_count
        )
        
        for key in ['CLMSMat', 'WRFMat', 'Map', 'UrbanFeature', 'SimilarityMat']:
            if key in unlabeled_data_reduced:
                arr = unlabeled_data_reduced[key]
                check_array_info(f"削减后 {key}", arr, show_stats=True)
        
        # 2.4 检查归一化前后的数据
        print_separator("2.4 归一化前后对比", "-")
        
        # 归一化前
        clms_before = np.array(unlabeled_data_reduced['CLMSMat'], dtype=np.float64)
        wrf_before = np.array(unlabeled_data_reduced['WRFMat'], dtype=np.float64)
        urban_before = np.array(unlabeled_data_reduced['UrbanFeature'], dtype=np.float64)
        
        print("\n【归一化前】")
        check_array_info("CLMSMat (归一化前)", clms_before, show_stats=True)
        check_array_info("WRFMat (归一化前)", wrf_before, show_stats=True)
        check_array_info("UrbanFeature (归一化前)", urban_before, show_stats=True)
        
        # 归一化后
        clms_after, clms_min, clms_max = MinMax(clms_before)
        wrf_after, wrf_min, wrf_max = MinMax(wrf_before)
        urban_after, urban_min, urban_max = MinMax_first_dim(urban_before)
        
        print("\n【归一化后】")
        check_array_info("CLMSMat (归一化后)", clms_after, show_stats=True)
        print(f"  归一化参数 - Min: {clms_min.shape}, Max: {clms_max.shape}")
        check_array_info("WRFMat (归一化后)", wrf_after, show_stats=True)
        print(f"  归一化参数 - Min: {wrf_min.shape}, Max: {wrf_max.shape}")
        check_array_info("UrbanFeature (归一化后)", urban_after, show_stats=True)
        print(f"  归一化参数 - Min: {urban_min.shape}, Max: {urban_max.shape}")
        
        # 2.5 检查辅助变量
        print_separator("2.5 辅助变量", "-")
        nTimesteps = unlabeled_data_reduced['CLMSMat'].shape[0]
        auxiliary_vars = add_auxiliary_variables(
            unlabeled_data_reduced, 
            nStations=target_station_count, 
            nTimesteps=nTimesteps
        )
        check_array_info("辅助变量 (Auxiliary Variables)", auxiliary_vars, show_stats=True)
        print(f"  包含: 小时、月份、年份、站点编号")
        
        return unlabeled_data_reduced
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_graph_structure(data_path, unlabeled_data, n_unlabeled=200):
    """检查图结构"""
    print_separator("3. 图结构检查", "=")
    
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': 12,
        'batchSize': 128,
        'thres': 0.4,
        'geoFeatures': 'full',
    }
    
    try:
        # 构建统一图
        unified_adj, labeled_indices, unlabeled_indices, unified_locations = build_unified_graph(
            dataParam, data_path, unlabeled_data, n_unlabeled=n_unlabeled
        )
        
        print(f"统一图节点总数: {unified_adj.shape[0]}")
        print(f"有标签节点数: {len(labeled_indices)}")
        print(f"无标签节点数: {len(unlabeled_indices)}")
        
        check_array_info("统一邻接矩阵 (Unified Adjacency)", unified_adj, show_stats=True)
        
        # 图统计信息
        n_edges = (unified_adj > 0).sum() / 2  # 无向图，除以2
        density = n_edges / (unified_adj.shape[0] * (unified_adj.shape[0] - 1) / 2)
        
        print(f"\n图统计信息:")
        print(f"  边数: {int(n_edges)}")
        print(f"  图密度: {density:.6f}")
        print(f"  平均度数: {n_edges * 2 / unified_adj.shape[0]:.2f}")
        
        # 检查有标签子图
        labeled_adj = unified_adj[np.ix_(labeled_indices, labeled_indices)]
        n_labeled_edges = (labeled_adj > 0).sum() / 2
        labeled_density = n_labeled_edges / (len(labeled_indices) * (len(labeled_indices) - 1) / 2)
        print(f"\n有标签子图:")
        print(f"  边数: {int(n_labeled_edges)}")
        print(f"  图密度: {labeled_density:.6f}")
        
        # 检查无标签子图
        unlabeled_adj = unified_adj[np.ix_(unlabeled_indices, unlabeled_indices)]
        n_unlabeled_edges = (unlabeled_adj > 0).sum() / 2
        unlabeled_density = n_unlabeled_edges / (len(unlabeled_indices) * (len(unlabeled_indices) - 1) / 2)
        print(f"\n无标签子图:")
        print(f"  边数: {int(n_unlabeled_edges)}")
        print(f"  图密度: {unlabeled_density:.6f}")
        
        check_array_info("统一位置信息 (Unified Locations)", unified_locations, show_stats=True)
        
    except Exception as e:
        print(f"❌ 图结构检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_geo_features(data_path):
    """检查地理特征"""
    print_separator("4. 地理特征检查", "=")
    
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'poolSize': 12,
    }
    
    try:
        geoFeatures, off, scl, nStations = genGeoFeatures(
            data_path, 
            dataParam['geoMethod'], 
            dataParam['poolSize'], 
            dataParam['nCompPCA']
        )
        
        check_array_info("地理特征 (GeoFeatures)", geoFeatures, show_stats=True)
        print(f"  站点数量: {nStations}")
        print(f"  特征维度: {geoFeatures.shape}")
        
    except Exception as e:
        print(f"❌ 地理特征检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_feature_patches(data_path):
    """检查特征补丁文件"""
    print_separator("5. 特征补丁文件检查", "=")
    
    # 检查 FeaturePatch_201.mat
    patch_201_file = os.path.join(data_path, 'FeaturePatch_201.mat')
    if os.path.exists(patch_201_file):
        print(f"✓ 找到文件: {patch_201_file}")
        try:
            patch_201_data = mat73.loadmat(patch_201_file)
            print(f"  文件中的变量: {list(patch_201_data.keys())}")
            
            for key, value in patch_201_data.items():
                if not key.startswith('__'):  # 跳过MATLAB元数据
                    if isinstance(value, np.ndarray):
                        check_array_info(f"FeaturePatch_201.{key}", value, show_stats=True)
                    else:
                        print(f"\n  {key}: {type(value)} (非数组)")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print(f"⚠️  文件不存在: {patch_201_file}")
    
    # 检查 FeaturePatch_401.mat
    patch_401_file = os.path.join(data_path, 'FeaturePatch_401.mat')
    if os.path.exists(patch_401_file):
        print(f"\n✓ 找到文件: {patch_401_file}")
        try:
            patch_401_data = mat73.loadmat(patch_401_file)
            print(f"  文件中的变量: {list(patch_401_data.keys())}")
            
            # 详细检查 FeatureMat_zeros（这是代码中实际使用的变量）
            if 'FeatureMat_zeros' in patch_401_data:
                print_separator("5.1 FeaturePatch_401.mat 详细维度分析", "-")
                feature_mat = patch_401_data['FeatureMat_zeros']
                
                print(f"\n  FeatureMat_zeros 原始维度: {feature_mat.shape}")
                print(f"    维度含义: (imageSize, imageSize, nFeatures, nStations)")
                
                if len(feature_mat.shape) == 4:
                    imageSize, _, nFeatures, nStations = feature_mat.shape
                    print(f"    图像大小: {imageSize} × {imageSize}")
                    print(f"    特征通道数: {nFeatures}")
                    print(f"    站点数量: {nStations}")
                    
                    # 检查代码中的处理逻辑
                    print(f"\n  代码处理逻辑:")
                    print(f"    1. 删除第4个特征维度（索引4，因为所有站点都是陆地）")
                    if nFeatures > 4:
                        _idx = np.arange(nFeatures)
                        _idx = np.delete(_idx, 4)
                        print(f"    2. 保留的特征维度索引: {_idx.tolist()} (共{len(_idx)}个)")
                        print(f"    3. 处理后维度: ({imageSize}, {imageSize}, {len(_idx)}, {nStations})")
                    
                    # 检查每个特征通道的统计信息
                    print(f"\n  各特征通道的统计信息:")
                    print(f"    {'通道索引':<10} {'最小值':<15} {'最大值':<15} {'平均值':<15} {'标准差':<15} {'NaN数量':<15}")
                    print("    " + "-" * 85)
                    
                    for feat_idx in range(nFeatures):
                        feat_channel = feature_mat[:, :, feat_idx, :]
                        min_val = np.nanmin(feat_channel)
                        max_val = np.nanmax(feat_channel)
                        mean_val = np.nanmean(feat_channel)
                        std_val = np.nanstd(feat_channel)
                        nan_count = np.isnan(feat_channel).sum()
                        nan_pct = (nan_count / feat_channel.size) * 100
                        print(f"    {feat_idx:<10} {min_val:<15.6f} {max_val:<15.6f} {mean_val:<15.6f} {std_val:<15.6f} {nan_count:<15,} ({nan_pct:.2f}%)")
                    
                    # 检查每个站点的统计信息（采样）
                    print(f"\n  各站点的统计信息（前10个站点）:")
                    print(f"    {'站点索引':<10} {'最小值':<15} {'最大值':<15} {'平均值':<15} {'NaN数量':<15}")
                    print("    " + "-" * 70)
                    
                    for station_idx in range(min(10, nStations)):
                        station_data = feature_mat[:, :, :, station_idx]
                        min_val = np.nanmin(station_data)
                        max_val = np.nanmax(station_data)
                        mean_val = np.nanmean(station_data)
                        nan_count = np.isnan(station_data).sum()
                        nan_pct = (nan_count / station_data.size) * 100
                        print(f"    {station_idx:<10} {min_val:<15.6f} {max_val:<15.6f} {mean_val:<15.6f} {nan_count:<15,} ({nan_pct:.2f}%)")
                    
                    if nStations > 10:
                        print(f"    ... (还有 {nStations - 10} 个站点)")
                
                check_array_info("FeaturePatch_401.FeatureMat_zeros", feature_mat, show_stats=True)
            
            # 检查其他变量
            for key, value in patch_401_data.items():
                if not key.startswith('__') and key != 'FeatureMat_zeros':  # 跳过MATLAB元数据和已检查的变量
                    if isinstance(value, np.ndarray):
                        check_array_info(f"FeaturePatch_401.{key}", value, show_stats=True)
                    else:
                        print(f"\n  {key}: {type(value)} (非数组)")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  文件不存在: {patch_401_file}")


def check_data_consistency(data_path, unlabeled_data, n_unlabeled=200):
    """检查数据一致性"""
    print_separator("6. 数据一致性检查", "=")
    
    # 检查维度一致性
    print("\n【维度一致性检查】")
    
    # 检查时间步数
    if 'CLMSMat' in unlabeled_data and 'WRFMat' in unlabeled_data:
        clms_timesteps = unlabeled_data['CLMSMat'].shape[0]
        wrf_timesteps = unlabeled_data['WRFMat'].shape[0]
        print(f"CLMSMat 时间步数: {clms_timesteps}")
        print(f"WRFMat 时间步数: {wrf_timesteps}")
        if clms_timesteps != wrf_timesteps:
            print(f"  ⚠️  时间步数不一致！")
        else:
            print(f"  ✓ 时间步数一致")
    
    # 检查站点数一致性
    station_keys = ['Map', 'NodeLocation', 'UrbanFeature', 'SimilarityMat']
    station_counts = {}
    for key in station_keys:
        if key in unlabeled_data:
            station_counts[key] = unlabeled_data[key].shape[0]
            print(f"{key} 站点数: {station_counts[key]}")
    
    if len(set(station_counts.values())) == 1:
        print(f"  ✓ 所有站点相关变量的站点数一致: {list(station_counts.values())[0]}")
    else:
        print(f"  ⚠️  站点数不一致: {station_counts}")
    
    # 检查有标签和无标签数据的特征维度
    print("\n【特征维度检查】")
    try:
        station_data = mat73.loadmat(os.path.join(data_path, 'GNN_N1_StationMat.mat'))
        if 'StationMat_se_fill' in station_data:
            labeled_raw = station_data['StationMat_se_fill']
            labeled_raw = np.transpose(labeled_raw, (0, 2, 1))
            labeled_features = labeled_raw[:, :, 1:]
            print(f"有标签数据特征维度: {labeled_features.shape}")
            print(f"  特征数: {labeled_features.shape[-1]}")
            
            if 'WRFMat' in unlabeled_data:
                print(f"无标签数据 WRFMat 维度: {unlabeled_data['WRFMat'].shape}")
                print(f"  特征数: {unlabeled_data['WRFMat'].shape[1]}")
                
                # 检查CFD特征数（应该是54）
                if unlabeled_data['WRFMat'].shape[1] == 54:
                    print(f"  ✓ WRFMat特征数正确 (54，已合并风速)")
                else:
                    print(f"  ⚠️  WRFMat特征数异常，期望54，实际{unlabeled_data['WRFMat'].shape[1]}")
    except Exception as e:
        print(f"  ⚠️  无法检查特征维度: {e}")


def list_all_data_files(data_path):
    """列出所有数据文件"""
    print_separator("0. 数据文件夹内容清单", "=")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return
    
    print(f"数据路径: {data_path}\n")
    
    all_files = os.listdir(data_path)
    mat_files = [f for f in all_files if f.endswith('.mat')]
    other_files = [f for f in all_files if not f.endswith('.mat')]
    
    print(f"✓ 找到 {len(mat_files)} 个 .mat 文件:")
    for i, f in enumerate(sorted(mat_files), 1):
        file_path = os.path.join(data_path, f)
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
        print(f"  {i}. {f} ({file_size:.2f} MB)")
    
    if other_files:
        print(f"\n其他文件 ({len(other_files)} 个):")
        for f in sorted(other_files):
            file_path = os.path.join(data_path, f)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"  - {f} ({file_size:.2f} MB)")
    
    print(f"\n总计: {len(all_files)} 个文件")


def compare_feature_mat_distributions(data_path):
    """比较 FeaturePatch_401.mat 的 FeatureMat_zeros、Labeled_Finalized.mat 的 UrbanFeatureMat 和 Unlabeled_Finalized.mat 的 UrbanFeatureMat 的分布"""
    print_separator("7. FeatureMat 分布比较（三组数据）", "=")
    
    try:
        # 1. 加载有标签数据的 FeatureMat_zeros (来自 FeaturePatch_401.mat)
        patch_401_file = os.path.join(data_path, 'FeaturePatch_401.mat')
        if not os.path.exists(patch_401_file):
            print(f"❌ 文件不存在: {patch_401_file}")
            return
        
        patch_401_data = mat73.loadmat(patch_401_file)
        if 'FeatureMat_zeros' not in patch_401_data:
            print(f"❌ FeaturePatch_401.mat 中未找到 'FeatureMat_zeros'")
            return
        
        labeled_feat = patch_401_data['FeatureMat_zeros']  # (imageSize, imageSize, nFeatures, nStations)
        
        # 删除第4个特征维度（与代码逻辑一致）
        _idx = np.arange(labeled_feat.shape[2])
        _idx = np.delete(_idx, 4)
        labeled_feat = labeled_feat[:, :, _idx, :]
        
        print(f"\n【数据源1】FeaturePatch_401.mat -> FeatureMat_zeros (删除第4个特征后)")
        print(f"  维度: {labeled_feat.shape}")
        print(f"  特征通道数: {labeled_feat.shape[2]}")
        print(f"  站点数: {labeled_feat.shape[3]}")
        
        # 2. 加载有标签数据的 UrbanFeatureMat (来自 Labeled_Finalized.mat)
        labeled_finalized_file = os.path.join(data_path, 'Labeled_Finalized.mat')
        labeled_finalized_feat = None
        if os.path.exists(labeled_finalized_file):
            labeled_finalized_data = mat73.loadmat(labeled_finalized_file)
            if 'UrbanFeatureMat' in labeled_finalized_data:
                labeled_finalized_feat = labeled_finalized_data['UrbanFeatureMat']  # (imageSize, imageSize, nFeatures, nStations)
                # 不删除第4个特征维度，保持原样
                
                print(f"\n【数据源2】Labeled_Finalized.mat -> UrbanFeatureMat (原始数据，未删除特征)")
                print(f"  维度: {labeled_finalized_feat.shape}")
                print(f"  特征通道数: {labeled_finalized_feat.shape[2]}")
                print(f"  站点数: {labeled_finalized_feat.shape[3]}")
            else:
                print(f"\n⚠️  Labeled_Finalized.mat 中未找到 'UrbanFeatureMat'，跳过此数据源")
        else:
            print(f"\n⚠️  文件不存在: {labeled_finalized_file}，跳过此数据源")
        
        # 3. 加载无标签数据的 UrbanFeatureMat (来自 Unlabeled_Finalized.mat)
        unlabeled_file = os.path.join(data_path, 'Unlabeled_Finalized.mat')
        if not os.path.exists(unlabeled_file):
            print(f"❌ 文件不存在: {unlabeled_file}")
            return
        
        unlabeled_data = mat73.loadmat(unlabeled_file)
        if 'UrbanFeatureMat' not in unlabeled_data:
            print(f"❌ Unlabeled_Finalized.mat 中未找到 'UrbanFeatureMat'")
            return
        
        unlabeled_feat = unlabeled_data['UrbanFeatureMat']  # (imageSize, imageSize, nFeatures, nStations)
        # 不删除第4个特征维度，保持原样
        
        print(f"\n【数据源3】Unlabeled_Finalized.mat -> UrbanFeatureMat (原始数据，未删除特征)")
        print(f"  维度: {unlabeled_feat.shape}")
        print(f"  特征通道数: {unlabeled_feat.shape[2]}")
        print(f"  站点数: {unlabeled_feat.shape[3]}")
        
        # 检查维度是否匹配
        all_shapes = [labeled_feat.shape[:3]]
        all_names = ["FeaturePatch_401"]
        if labeled_finalized_feat is not None:
            all_shapes.append(labeled_finalized_feat.shape[:3])
            all_names.append("Labeled_Finalized")
        all_shapes.append(unlabeled_feat.shape[:3])
        all_names.append("Unlabeled_Finalized")
        
        if len(set(all_shapes)) > 1:
            print(f"\n⚠️  维度不匹配！")
            for name, shape in zip(all_names, all_shapes):
                print(f"  {name}: {shape}")
        else:
            print(f"\n✓ 所有数据源的图像大小和特征通道数匹配")
        
        # 确定要比较的特征通道数
        # 第一个数据源删除了第4个特征，所以特征数可能比其他两个少1
        nFeatures_labeled = labeled_feat.shape[2]  # 第一个数据源的特征数（已删除第4个，应该是7）
        nFeatures_others = unlabeled_feat.shape[2]  # 其他数据源的特征数（全部保留，应该是7）
        
        # 创建索引映射：第一个数据源的索引 -> 其他数据源的索引
        # 第一个数据源删除了索引4，所以：
        # 第一个数据源的索引0-3 -> 其他数据源的索引0-3
        # 映射关系：由于都是删除了索引4后的结果，所以映射是直接的
        # 第一个数据源的新索引0-6 -> 其他数据源的索引0-6（一一对应）
        def map_index_to_others(idx_in_labeled):
            """将第一个数据源的索引映射到其他数据源的索引
            数据源1：新索引0-6对应原始索引0-3, 5-7
            数据源2/3：索引0-6对应原始索引0-3, 5-7（也删除了索引4）
            所以映射关系是：新索引0-3->0-3, 新索引4-6->4-6
            """
            # 数据源1的新索引直接对应数据源2/3的索引（都是删除了索引4后的结果）
            return idx_in_labeled
        
        # 比较逻辑：只比较数据源1中存在的特征
        # 数据源2和3使用所有7个特征（索引0-6），全部保留
        nFeatures = nFeatures_labeled  # 只比较数据源1中存在的特征
        
        # 比较每个特征通道的分布
        print_separator("7.1 各特征通道分布比较", "-")
        print(f"{'通道':<8} {'数据源':<25} {'最小值':<15} {'最大值':<15} {'平均值':<15} {'标准差':<15} {'NaN数量':<12}")
        print("-" * 120)
        print(f"注意: FeaturePatch_401原始8个特征，删除索引4后变成7个；其他数据源原始7个特征，全部保留")
        print()
        
        for feat_idx in range(nFeatures):
            # 数据源1: FeaturePatch_401 (已删除第4个特征)
            labeled_channel = labeled_feat[:, :, feat_idx, :]
            labeled_min = np.nanmin(labeled_channel)
            labeled_max = np.nanmax(labeled_channel)
            labeled_mean = np.nanmean(labeled_channel)
            labeled_std = np.nanstd(labeled_channel)
            labeled_nan = np.isnan(labeled_channel).sum()
            
            print(f"{feat_idx:<8} {'FeaturePatch_401':<25} {labeled_min:<15.6f} {labeled_max:<15.6f} "
                  f"{labeled_mean:<15.6f} {labeled_std:<15.6f} {labeled_nan:<12,}")
            
            # 数据源2和3: 使用映射后的索引
            mapped_idx = map_index_to_others(feat_idx)
            
            # 数据源2: Labeled_Finalized (如果存在)
            if labeled_finalized_feat is not None:
                if mapped_idx < labeled_finalized_feat.shape[2]:
                    labeled_final_channel = labeled_finalized_feat[:, :, mapped_idx, :]
                    labeled_final_min = np.nanmin(labeled_final_channel)
                    labeled_final_max = np.nanmax(labeled_final_channel)
                    labeled_final_mean = np.nanmean(labeled_final_channel)
                    labeled_final_std = np.nanstd(labeled_final_channel)
                    labeled_final_nan = np.isnan(labeled_final_channel).sum()
                    
                    print(f"{mapped_idx:<8} {'Labeled_Finalized':<25} {labeled_final_min:<15.6f} {labeled_final_max:<15.6f} "
                          f"{labeled_final_mean:<15.6f} {labeled_final_std:<15.6f} {labeled_final_nan:<12,}")
                else:
                    print(f"{'':<8} {'Labeled_Finalized':<25} {'N/A (索引超出)':<15}")
            
            # 数据源3: Unlabeled_Finalized
            if mapped_idx < unlabeled_feat.shape[2]:
                unlabeled_channel = unlabeled_feat[:, :, mapped_idx, :]
                unlabeled_min = np.nanmin(unlabeled_channel)
                unlabeled_max = np.nanmax(unlabeled_channel)
                unlabeled_mean = np.nanmean(unlabeled_channel)
                unlabeled_std = np.nanstd(unlabeled_channel)
                unlabeled_nan = np.isnan(unlabeled_channel).sum()
                
                print(f"{mapped_idx:<8} {'Unlabeled_Finalized':<25} {unlabeled_min:<15.6f} {unlabeled_max:<15.6f} "
                      f"{unlabeled_mean:<15.6f} {unlabeled_std:<15.6f} {unlabeled_nan:<12,}")
            else:
                print(f"{'':<8} {'Unlabeled_Finalized':<25} {'N/A (索引超出)':<15}")
            
            # 计算分布差异（比较所有可用的数据源）
            all_mins = [labeled_min]
            all_maxs = [labeled_max]
            all_means = [labeled_mean]
            if labeled_finalized_feat is not None and mapped_idx < labeled_finalized_feat.shape[2]:
                all_mins.append(labeled_final_min)
                all_maxs.append(labeled_final_max)
                all_means.append(labeled_final_mean)
            if mapped_idx < unlabeled_feat.shape[2]:
                all_mins.append(unlabeled_min)
                all_maxs.append(unlabeled_max)
                all_means.append(unlabeled_mean)
            
            overall_min = min(all_mins)
            overall_max = max(all_maxs)
            overall_range = overall_max - overall_min
            
            # 计算范围重叠
            range_overlap = min(all_maxs) - max(all_mins)
            overlap_ratio = range_overlap / overall_range if overall_range > 0 else 0
            
            # 计算均值差异
            mean_diff = max(all_means) - min(all_means)
            mean_diff_relative = mean_diff / max(abs(max(all_means)), abs(min(all_means)), 1)
            
            if overlap_ratio > 0.8 and mean_diff_relative < 0.1:
                print(f"{'':<8} {'→ 分布相似':<25} 范围重叠: {overlap_ratio*100:.1f}%, 均值差异: {mean_diff:.6f}")
            else:
                print(f"{'':<8} {'→ ⚠️ 分布差异':<25} 范围重叠: {overlap_ratio*100:.1f}%, 均值差异: {mean_diff:.6f}")
            
            print()
        
        # 整体分布比较
        print_separator("7.2 整体分布统计", "-")
        
        # 展平所有数据进行比较
        labeled_flat = labeled_feat.flatten()
        unlabeled_flat = unlabeled_feat.flatten()
        
        print(f"【整体统计】")
        print(f"  FeaturePatch_401 - 最小值: {np.nanmin(labeled_flat):.6f}, 最大值: {np.nanmax(labeled_flat):.6f}, "
              f"平均值: {np.nanmean(labeled_flat):.6f}, 标准差: {np.nanstd(labeled_flat):.6f}")
        
        if labeled_finalized_feat is not None:
            labeled_final_flat = labeled_finalized_feat.flatten()
            print(f"  Labeled_Finalized - 最小值: {np.nanmin(labeled_final_flat):.6f}, 最大值: {np.nanmax(labeled_final_flat):.6f}, "
                  f"平均值: {np.nanmean(labeled_final_flat):.6f}, 标准差: {np.nanstd(labeled_final_flat):.6f}")
        
        print(f"  Unlabeled_Finalized - 最小值: {np.nanmin(unlabeled_flat):.6f}, 最大值: {np.nanmax(unlabeled_flat):.6f}, "
              f"平均值: {np.nanmean(unlabeled_flat):.6f}, 标准差: {np.nanstd(unlabeled_flat):.6f}")
        
        # 计算分位数
        percentiles = [5, 25, 50, 75, 95]
        print(f"\n【分位数比较】")
        if labeled_finalized_feat is not None:
            print(f"{'分位数':<10} {'FeaturePatch_401':<20} {'Labeled_Finalized':<20} {'Unlabeled_Finalized':<20}")
            print("-" * 70)
            labeled_final_flat = labeled_finalized_feat.flatten()
            for p in percentiles:
                labeled_p = np.nanpercentile(labeled_flat, p)
                labeled_final_p = np.nanpercentile(labeled_final_flat, p)
                unlabeled_p = np.nanpercentile(unlabeled_flat, p)
                print(f"{p}%{'':<6} {labeled_p:<20.6f} {labeled_final_p:<20.6f} {unlabeled_p:<20.6f}")
        else:
            print(f"{'分位数':<10} {'FeaturePatch_401':<20} {'Unlabeled_Finalized':<20} {'差异':<15}")
            print("-" * 65)
            for p in percentiles:
                labeled_p = np.nanpercentile(labeled_flat, p)
                unlabeled_p = np.nanpercentile(unlabeled_flat, p)
                diff = abs(labeled_p - unlabeled_p)
                print(f"{p}%{'':<6} {labeled_p:<20.6f} {unlabeled_p:<20.6f} {diff:<15.6f}")
        
        # 结论
        print_separator("7.3 分布一致性结论", "-")
        
        # 计算所有数据源的整体范围
        all_flats = [labeled_flat]
        if labeled_finalized_feat is not None:
            all_flats.append(labeled_finalized_feat.flatten())
        all_flats.append(unlabeled_flat)
        
        all_mins_flat = [np.nanmin(f) for f in all_flats]
        all_maxs_flat = [np.nanmax(f) for f in all_flats]
        all_means_flat = [np.nanmean(f) for f in all_flats]
        all_stds_flat = [np.nanstd(f) for f in all_flats]
        
        overall_min_flat = min(all_mins_flat)
        overall_max_flat = max(all_maxs_flat)
        overall_range_flat = overall_max_flat - overall_min_flat
        
        # 计算范围重叠
        overall_overlap = min(all_maxs_flat) - max(all_mins_flat)
        overall_overlap_ratio = overall_overlap / overall_range_flat if overall_range_flat > 0 else 0
        
        # 计算均值差异
        mean_diff_overall = max(all_means_flat) - min(all_means_flat)
        mean_diff_relative = mean_diff_overall / max(abs(max(all_means_flat)), abs(min(all_means_flat)), 1)
        
        # 计算标准差差异
        std_diff_overall = max(all_stds_flat) - min(all_stds_flat)
        
        print(f"范围重叠比例: {overall_overlap_ratio*100:.2f}%")
        print(f"均值差异: {mean_diff_overall:.6f} (相对差异: {mean_diff_relative*100:.2f}%)")
        print(f"标准差差异: {std_diff_overall:.6f}")
        
        if overall_overlap_ratio > 0.7 and mean_diff_relative < 0.1:
            print(f"\n✓ 结论: 所有数据源分布**相似**，各自归一化到[0,1]应该可行")
        else:
            print(f"\n⚠️  结论: 数据源分布存在**差异**，建议统一归一化参数")
        
    except Exception as e:
        print(f"❌ 比较失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print_separator("=" * 80)
    print("全面的数据文件检查脚本".center(80))
    print_separator("=" * 80)
    
    # 数据路径
    DATA_PATH = '/projects/p32685/Fixmatch/data'
    n_unlabeled = 200  # 默认200个无标签站点
    
    print(f"数据路径: {DATA_PATH}")
    print(f"目标无标签站点数: {n_unlabeled}")
    
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ 数据路径不存在: {DATA_PATH}")
        return
    
    # 0. 列出所有文件
    list_all_data_files(DATA_PATH)
    
    # 1. 检查有标签数据
    check_labeled_data(DATA_PATH)
    
    # 2. 检查无标签数据
    unlabeled_data = check_unlabeled_data(DATA_PATH, target_station_count=n_unlabeled)
    
    if unlabeled_data is None:
        print("\n❌ 无标签数据检查失败，跳过后续检查")
        return
    
    # 3. 检查图结构
    check_graph_structure(DATA_PATH, unlabeled_data, n_unlabeled=n_unlabeled)
    
    # 4. 检查地理特征
    check_geo_features(DATA_PATH)
    
    # 5. 检查特征补丁文件
    check_feature_patches(DATA_PATH)
    
    # 6. 检查数据一致性
    check_data_consistency(DATA_PATH, unlabeled_data, n_unlabeled=n_unlabeled)
    
    # 7. 比较 FeatureMat 分布
    compare_feature_mat_distributions(DATA_PATH)
    
    print_separator("=" * 80)
    print("检查完成！".center(80))
    print_separator("=" * 80)


if __name__ == "__main__":
    main()


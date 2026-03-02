import os
import mat73
import numpy as np

def check_urban_feature_nan(file_path):
    print("=" * 70)
    print(f"检查 {file_path} 中的 UrbanFeatureMat 的 NaN 情况")
    print("=" * 70)
    
    try:
        data = mat73.loadmat(file_path)
        
        if 'UrbanFeatureMat' in data:
            urban_feat = data['UrbanFeatureMat']
            print(f"\nUrbanFeatureMat 形状: {urban_feat.shape}")
            print(f"数据类型: {urban_feat.dtype}")
            
            # 检查NaN总数
            nan_count = np.isnan(urban_feat).sum()
            total_elements = urban_feat.size
            nan_percentage = (nan_count / total_elements) * 100
            
            print(f"\nNaN 统计:")
            print(f"  总元素数: {total_elements:,}")
            print(f"  NaN 数量: {nan_count:,}")
            print(f"  NaN 百分比: {nan_percentage:.4f}%")
            
            # 检查每个特征维度的NaN情况
            if urban_feat.ndim == 4:  # (401, 401, nFeatures, nStations)
                n_features = urban_feat.shape[2]
                n_stations = urban_feat.shape[3]
                print(f"\n按特征维度检查 (共 {n_features} 个特征):")
                for feat_idx in range(n_features):
                    feat_slice = urban_feat[:, :, feat_idx, :]
                    nan_count_feat = np.isnan(feat_slice).sum()
                    if nan_count_feat > 0:
                        nan_pct_feat = (nan_count_feat / feat_slice.size) * 100
                        print(f"  特征 {feat_idx}: {nan_count_feat:,} 个NaN ({nan_pct_feat:.4f}%)")
                
                # 检查每个站点的NaN情况
                print(f"\n按站点检查 (共 {n_stations} 个站点):")
                stations_with_nan = []
                for station_idx in range(min(10, n_stations)):  # 只检查前10个站点
                    station_slice = urban_feat[:, :, :, station_idx]
                    nan_count_station = np.isnan(station_slice).sum()
                    if nan_count_station > 0:
                        nan_pct_station = (nan_count_station / station_slice.size) * 100
                        stations_with_nan.append(station_idx)
                        print(f"  站点 {station_idx}: {nan_count_station:,} 个NaN ({nan_pct_station:.4f}%)")
                
                if len(stations_with_nan) == 0 and n_stations > 10:
                    print("  前10个站点均无NaN")
                
                # 检查NaN的分布模式
                print(f"\nNaN 分布模式检查:")
                # 检查是否整行/整列为NaN
                for feat_idx in range(min(3, n_features)):  # 检查前3个特征
                    for station_idx in range(min(3, n_stations)):  # 检查前3个站点
                        feat_2d = urban_feat[:, :, feat_idx, station_idx]
                        nan_rows = np.isnan(feat_2d).all(axis=1).sum()
                        nan_cols = np.isnan(feat_2d).all(axis=0).sum()
                        if nan_rows > 0 or nan_cols > 0:
                            print(f"  特征 {feat_idx}, 站点 {station_idx}: {nan_rows} 整行为NaN, {nan_cols} 整列为NaN")
            
            # 检查数值范围（排除NaN）
            valid_data = urban_feat[~np.isnan(urban_feat)]
            if len(valid_data) > 0:
                print(f"\n有效数据统计 (排除NaN):")
                print(f"  最小值: {valid_data.min():.6f}")
                print(f"  最大值: {valid_data.max():.6f}")
                print(f"  平均值: {valid_data.mean():.6f}")
                print(f"  标准差: {valid_data.std():.6f}")
                print(f"  中位数: {np.median(valid_data):.6f}")
            else:
                print(f"\n⚠️  警告：所有数据都是NaN！")
            
            # 检查Inf值
            inf_count = np.isinf(urban_feat).sum()
            if inf_count > 0:
                print(f"\n⚠️  警告：发现 {inf_count} 个 Inf 值")
            
        else:
            print(f"\n❌ 文件 {file_path} 中未找到 'UrbanFeatureMat' 变量")
            print("\n文件中的变量列表:")
            for key in data.keys():
                if not key.startswith('__'):
                    print(f"  - {key}: {type(data[key])}")
                    if isinstance(data[key], np.ndarray):
                        print(f"    形状: {data[key].shape}")
        
    except Exception as e:
        print(f"\n❌ 加载或检查文件时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Unlabeled_Finalized.mat')
    check_urban_feature_nan(data_path)


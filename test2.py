"""
站点位置重合检查脚本
检查三个数据源中的站点是否有重合部分
并提取匹配站点的 Map 和 SimilarityMat 供后续使用
"""
import numpy as np
import mat73
import pickle
import os
from scipy.spatial.distance import cdist


def print_separator(title="", char="=", length=80):
    """打印分隔线"""
    if title:
        print(f"\n{char * length}")
        print(f"{title:^{length}}")
        print(f"{char * length}\n")
    else:
        print(f"{char * length}\n")


def load_locations():
    """加载三个数据源的位置信息"""
    data_path = '/projects/p32685/Fixmatch/data'
    
    # 1. GNN_N1_StationMat.mat - 原始有标签数据
    print("\n正在加载: GNN_N1_StationMat.mat")
    try:
        data1 = mat73.loadmat(f"{data_path}/GNN_N1_StationMat.mat")
        if 'Locations' in data1:
            loc1 = np.array(data1['Locations'])  # (68, 2)
            print(f"  ✓ 找到 Locations: {loc1.shape}")
        else:
            print("  ✗ 未找到 Locations 变量")
            print(f"  可用变量: {list(data1.keys())}")
            return None, None, None
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None, None, None
    
    # 2. Labeled_Finalized.mat - 新的有标签数据
    print("\n正在加载: Labeled_Finalized.mat")
    try:
        data2 = mat73.loadmat(f"{data_path}/Labeled_Finalized.mat")
        # 优先使用 NodeLocation，如果没有则使用 Map
        if 'NodeLocation' in data2:
            loc2 = np.array(data2['NodeLocation'])  # (106, 2)
            print(f"  ✓ 找到 NodeLocation: {loc2.shape}")
        elif 'Map' in data2:
            loc2 = np.array(data2['Map'])  # (106, 2)
            print(f"  ✓ 找到 Map: {loc2.shape}")
        else:
            print("  ✗ 未找到位置变量")
            print(f"  可用变量: {list(data2.keys())}")
            return None, None, None
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None, None, None
    
    # 3. Unlabeled_Finalized.mat - 无标签数据
    print("\n正在加载: Unlabeled_Finalized.mat")
    try:
        data3 = mat73.loadmat(f"{data_path}/Unlabeled_Finalized.mat")
        # 优先使用 NodeLocation，如果没有则使用 Map
        if 'NodeLocation' in data3:
            loc3 = np.array(data3['NodeLocation'])  # (2000, 2)
            print(f"  ✓ 找到 NodeLocation: {loc3.shape}")
        elif 'Map' in data3:
            loc3 = np.array(data3['Map'])  # (2000, 2)
            print(f"  ✓ 找到 Map: {loc3.shape}")
        else:
            print("  ✗ 未找到位置变量")
            print(f"  可用变量: {list(data3.keys())}")
            return None, None, None
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None, None, None
    
    return loc1, loc2, loc3


def calculate_distance_matrix(loc1, loc2):
    """
    计算两个位置矩阵之间的距离矩阵
    
    Args:
        loc1: (n1, 2) 位置矩阵
        loc2: (n2, 2) 位置矩阵
    
    Returns:
        dist_matrix: (n1, n2) 距离矩阵（欧氏距离）
    """
    return cdist(loc1, loc2, metric='euclidean')


def find_matching_stations(loc1, loc2, threshold=0.001, name1="数据源1", name2="数据源2"):
    """
    找出两个数据源中匹配的站点
    
    Args:
        loc1: (n1, 2) 第一个数据源的位置
        loc2: (n2, 2) 第二个数据源的位置
        threshold: 距离阈值，小于此值认为匹配
        name1: 第一个数据源的名称
        name2: 第二个数据源的名称
    
    Returns:
        matches: list of tuples [(idx1, idx2, distance), ...]
    """
    dist_matrix = calculate_distance_matrix(loc1, loc2)
    
    # 找出每个站点在另一个数据源中的最近邻
    matches = []
    for i in range(len(loc1)):
        min_dist_idx = np.argmin(dist_matrix[i, :])
        min_dist = dist_matrix[i, min_dist_idx]
        if min_dist < threshold:
            matches.append((i, min_dist_idx, min_dist))
    
    return matches, dist_matrix


def verify_matching_logic(loc1, loc2, matches, name1="数据源1", name2="数据源2"):
    """
    验证匹配逻辑的合理性
    
    Args:
        loc1: 数据源1的位置矩阵 (n1, 2)
        loc2: 数据源2的位置矩阵 (n2, 2)
        matches: 匹配结果列表 [(idx1, idx2, distance), ...]
        name1: 数据源1的名称
        name2: 数据源2的名称
    """
    print_separator("匹配逻辑验证", "=", 80)
    
    if len(matches) == 0:
        print("  ⚠️ 没有匹配结果，无法验证")
        return
    
    # 1. 检查数据源1中匹配到同一数据源2站点的多个站点
    print(f"\n1. 检查 {name1} 中匹配到同一 {name2} 站点的多个站点:")
    idx2_to_idx1s = {}  # {idx2: [idx1_list]}
    for idx1, idx2, dist in matches:
        if idx2 not in idx2_to_idx1s:
            idx2_to_idx1s[idx2] = []
        idx2_to_idx1s[idx2].append((idx1, dist))
    
    duplicate_matches = {idx2: idx1s for idx2, idx1s in idx2_to_idx1s.items() if len(idx1s) > 1}
    
    if len(duplicate_matches) > 0:
        print(f"  ⚠️ 发现 {len(duplicate_matches)} 个 {name2} 站点被多个 {name1} 站点匹配:")
        for idx2, idx1s in sorted(duplicate_matches.items()):
            print(f"\n  {name2} 索引 {idx2} 被以下 {name1} 站点匹配:")
            print(f"    {'索引':<10} {'距离':<15} {'位置坐标':<30} {'是否相同位置'}")
            print(f"    {'-'*10} {'-'*15} {'-'*30} {'-'*15}")
            
            positions = []
            for idx1, dist in idx1s:
                pos = loc1[idx1]
                positions.append(pos)
                is_duplicate = "是" if any(np.allclose(pos, loc1[other_idx1]) for other_idx1, _ in idx1s if other_idx1 != idx1) else "否"
                print(f"    {idx1:<10} {dist:<15.8f} ({pos[0]:.6f}, {pos[1]:.6f}) {'相同' if is_duplicate == '是' else '不同'}")
            
            # 检查这些位置是否相同
            positions = np.array(positions)
            if len(positions) > 1:
                # 计算所有位置对之间的距离
                pairwise_dists = cdist(positions, positions)
                # 排除对角线（自己到自己的距离为0）
                pairwise_dists = pairwise_dists[np.triu_indices(len(positions), k=1)]
                max_dist = np.max(pairwise_dists) if len(pairwise_dists) > 0 else 0
                
                if max_dist < 1e-6:  # 如果最大距离很小，认为位置相同
                    print(f"    ✓ 这些站点位置相同（最大间距: {max_dist:.10f}，可能是重复数据）")
                else:
                    print(f"    ⚠️ 这些站点位置不同，最大间距: {max_dist:.8f}")
                    print(f"    💡 说明：这些 {name1} 站点都距离 {name2} 索引 {idx2} 最近，所以都匹配到了它")
    else:
        print(f"  ✓ 没有发现重复匹配（每个 {name2} 站点最多被一个 {name1} 站点匹配）")
    
    # 2. 检查数据源2中是否有多个站点匹配到数据源1的同一站点（反向检查）
    print(f"\n2. 检查 {name2} 中是否有多个站点匹配到同一 {name1} 站点:")
    idx1_to_idx2s = {}  # {idx1: [idx2_list]}
    for idx1, idx2, dist in matches:
        if idx1 not in idx1_to_idx2s:
            idx1_to_idx2s[idx1] = []
        idx1_to_idx2s[idx1].append((idx2, dist))
    
    reverse_duplicate = {idx1: idx2s for idx1, idx2s in idx1_to_idx2s.items() if len(idx2s) > 1}
    
    if len(reverse_duplicate) > 0:
        print(f"  ⚠️ 发现 {len(reverse_duplicate)} 个 {name1} 站点被多个 {name2} 站点匹配:")
        for idx1, idx2s in sorted(reverse_duplicate.items()):
            print(f"\n  {name1} 索引 {idx1} 被以下 {name2} 站点匹配:")
            print(f"    {'索引':<10} {'距离':<15} {'位置坐标'}")
            print(f"    {'-'*10} {'-'*15} {'-'*30}")
            for idx2, dist in idx2s:
                pos = loc2[idx2]
                print(f"    {idx2:<10} {dist:<15.8f} ({pos[0]:.6f}, {pos[1]:.6f})")
            print(f"    💡 说明：这些 {name2} 站点都距离 {name1} 索引 {idx1} 最近")
    else:
        print(f"  ✓ 没有发现反向重复匹配（每个 {name1} 站点最多匹配一个 {name2} 站点）")
    
    # 3. 分析匹配逻辑
    print(f"\n3. 匹配逻辑分析:")
    print(f"  - 匹配方式：最近邻匹配（对每个 {name1} 站点，找到 {name2} 中距离最近的站点）")
    print(f"  - 匹配结果：{len(matches)} 个匹配对")
    print(f"  - {name1} 站点数：{len(loc1)}")
    print(f"  - {name2} 站点数：{len(loc2)}")
    print(f"  - 匹配率：{len(matches)/len(loc1)*100:.1f}% ({name1} 站点中成功匹配的比例)")
    
    # 4. 统计信息
    print(f"\n4. 匹配统计:")
    unique_idx1 = len(set(m[0] for m in matches))
    unique_idx2 = len(set(m[1] for m in matches))
    print(f"  - 唯一 {name1} 站点数：{unique_idx1} / {len(loc1)}")
    print(f"  - 唯一 {name2} 站点数：{unique_idx2} / {len(loc2)}")
    print(f"  - 平均每个 {name2} 站点被匹配次数：{len(matches)/unique_idx2:.2f}" if unique_idx2 > 0 else "")
    print(f"  - 平均每个 {name1} 站点匹配次数：{len(matches)/unique_idx1:.2f}" if unique_idx1 > 0 else "")


def extract_matched_data(matches, data_path, output_path=None):
    """
    从 Labeled_Finalized.mat 中提取匹配站点的 Map 和 SimilarityMat
    
    Args:
        matches: 匹配站点列表 [(idx1, idx2, distance), ...]
        data_path: 数据文件路径
        output_path: 输出保存路径（可选）
    
    Returns:
        matched_indices: 数据源2中的匹配索引数组 (68,)
        matched_map: 匹配站点的 Map (68, 2)
        matched_similarity: 匹配站点的 SimilarityMat (68, 10)
    """
    print_separator("提取匹配站点的 Map 和 SimilarityMat", "=", 80)
    
    if len(matches) == 0:
        print("  ✗ 没有匹配的站点，无法提取数据")
        return None, None, None
    
    # 加载 Labeled_Finalized.mat
    print(f"\n正在加载: {data_path}/Labeled_Finalized.mat")
    try:
        data2 = mat73.loadmat(f"{data_path}/Labeled_Finalized.mat")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None, None, None
    
    # 提取匹配索引（按数据源1的顺序排序）
    matches_sorted = sorted(matches, key=lambda x: x[0])  # 按数据源1的索引排序
    matched_indices = np.array([m[1] for m in matches_sorted])  # 数据源2中的索引
    
    print(f"  ✓ 找到 {len(matched_indices)} 个匹配站点")
    print(f"  匹配索引范围: [{matched_indices.min()}, {matched_indices.max()}]")
    
    # 打印索引映射关系
    print(f"\n索引映射关系 (数据源1索引 -> 数据源2索引):")
    print(f"{'数据源1索引':<12} {'数据源2索引':<12} {'距离':<15}")
    print("-" * 40)
    for idx1, idx2, dist in matches_sorted:
        print(f"{idx1:<12} {idx2:<12} {dist:<15.8f}")
    
    # 打印可以直接使用的索引数组
    print(f"\n数据源2中的匹配索引数组 (可直接用于索引):")
    print(f"matched_indices = np.array({matched_indices.tolist()})")
    print(f"\n或者使用切片/布尔索引:")
    print(f"# 方法1: 使用索引数组")
    print(f"matched_map = data2['Map'][matched_indices]")
    print(f"matched_similarity = data2['SimilarityMat'][matched_indices]")
    print(f"\n# 方法2: 使用布尔掩码")
    print(f"mask = np.zeros(data2['Map'].shape[0], dtype=bool)")
    print(f"mask[{matched_indices.tolist()}] = True")
    print(f"matched_map = data2['Map'][mask]")
    print(f"matched_similarity = data2['SimilarityMat'][mask]")
    
    # 提取 Map
    if 'Map' in data2:
        full_map = np.array(data2['Map'])  # (106, 2)
        matched_map = full_map[matched_indices]  # (68, 2)
        print(f"  ✓ 提取 Map: {matched_map.shape}")
    elif 'NodeLocation' in data2:
        full_map = np.array(data2['NodeLocation'])  # (106, 2)
        matched_map = full_map[matched_indices]  # (68, 2)
        print(f"  ✓ 提取 NodeLocation (作为 Map): {matched_map.shape}")
    else:
        print("  ✗ 未找到 Map 或 NodeLocation")
        return None, None, None
    
    # 提取 SimilarityMat
    if 'SimilarityMat' in data2:
        full_similarity = np.array(data2['SimilarityMat'])  # (106, 10)
        matched_similarity = full_similarity[matched_indices]  # (68, 10)
        print(f"  ✓ 提取 SimilarityMat: {matched_similarity.shape}")
    else:
        print("  ✗ 未找到 SimilarityMat")
        print(f"  可用变量: {list(data2.keys())}")
        return None, None, None
    
    # 验证数据
    print(f"\n数据验证:")
    print(f"  Map 形状: {matched_map.shape}, 范围: [{matched_map.min():.6f}, {matched_map.max():.6f}]")
    print(f"  SimilarityMat 形状: {matched_similarity.shape}, 范围: [{matched_similarity.min():.6f}, {matched_similarity.max():.6f}]")
    
    # 保存数据（可选）
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        save_data = {
            'matched_indices': matched_indices,  # 数据源2中的索引
            'matched_map': matched_map,  # (68, 2)
            'matched_similarity': matched_similarity,  # (68, 10)
            'n_matches': len(matched_indices),
            'matches_detail': matches_sorted  # 完整的匹配信息
        }
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\n  ✓ 数据已保存到: {output_path} (可选)")
    
    return matched_indices, matched_map, matched_similarity


def analyze_location_overlap():
    """分析三个数据源的站点位置重合情况"""
    print_separator("站点位置重合检查", "=", 80)
    
    # 加载位置数据
    loc1, loc2, loc3 = load_locations()
    
    if loc1 is None or loc2 is None or loc3 is None:
        print("\n✗ 数据加载失败，无法继续分析")
        return
    
    print_separator("位置数据基本信息")
    print(f"数据源1 (GNN_N1_StationMat): {loc1.shape[0]} 个站点")
    print(f"数据源2 (Labeled_Finalized): {loc2.shape[0]} 个站点")
    print(f"数据源3 (Unlabeled_Finalized): {loc3.shape[0]} 个站点")
    
    # 显示位置范围
    print(f"\n数据源1位置范围:")
    print(f"  经度: [{np.min(loc1[:, 0]):.6f}, {np.max(loc1[:, 0]):.6f}]")
    print(f"  纬度: [{np.min(loc1[:, 1]):.6f}, {np.max(loc1[:, 1]):.6f}]")
    
    print(f"\n数据源2位置范围:")
    print(f"  经度: [{np.min(loc2[:, 0]):.6f}, {np.max(loc2[:, 0]):.6f}]")
    print(f"  纬度: [{np.min(loc2[:, 1]):.6f}, {np.max(loc2[:, 1]):.6f}]")
    
    print(f"\n数据源3位置范围:")
    print(f"  经度: [{np.min(loc3[:, 0]):.6f}, {np.max(loc3[:, 0]):.6f}]")
    print(f"  纬度: [{np.min(loc3[:, 1]):.6f}, {np.max(loc3[:, 1]):.6f}]")
    
    # 检查位置坐标系是否一致
    print_separator("位置坐标系检查")
    coord_system_1 = "经纬度" if (np.abs(loc1[:, 0]).max() < 180 and np.abs(loc1[:, 1]).max() < 90) else "其他坐标系"
    coord_system_2 = "经纬度" if (np.abs(loc2[:, 0]).max() < 180 and np.abs(loc2[:, 1]).max() < 90) else "其他坐标系"
    coord_system_3 = "经纬度" if (np.abs(loc3[:, 0]).max() < 180 and np.abs(loc3[:, 1]).max() < 90) else "其他坐标系"
    
    print(f"数据源1坐标系: {coord_system_1}")
    print(f"数据源2坐标系: {coord_system_2}")
    print(f"数据源3坐标系: {coord_system_3}")
    
    # 如果坐标系不一致，需要转换或使用不同的阈值
    # 对于经纬度，使用较小的阈值（约0.001度 ≈ 111米）
    # 对于其他坐标系，可能需要更大的阈值
    
    # 根据坐标系设置阈值
    if coord_system_1 == "经纬度" and coord_system_2 == "经纬度":
        threshold_12 = 0.001  # 约111米
    else:
        threshold_12 = 1.0  # 对于其他坐标系使用更大的阈值
    
    if coord_system_1 == "经纬度" and coord_system_3 == "经纬度":
        threshold_13 = 0.001
    else:
        threshold_13 = 1.0
    
    if coord_system_2 == "经纬度" and coord_system_3 == "经纬度":
        threshold_23 = 0.001
    else:
        threshold_23 = 1.0
    
    print(f"\n使用距离阈值:")
    print(f"  数据源1 vs 数据源2: {threshold_12}")
    print(f"  数据源1 vs 数据源3: {threshold_13}")
    print(f"  数据源2 vs 数据源3: {threshold_23}")
    
    # 比较数据源1和数据源2
    print_separator("数据源1 vs 数据源2 (原始有标签 vs 新有标签)")
    matches_12, dist_matrix_12 = find_matching_stations(
        loc1, loc2, threshold_12, 
        "GNN_N1_StationMat", "Labeled_Finalized"
    )
    print(f"找到 {len(matches_12)} 个匹配站点 (阈值: {threshold_12})")
    if len(matches_12) > 0:
        print(f"\n匹配详情 (前10个):")
        print(f"{'数据源1索引':<12} {'数据源2索引':<12} {'距离':<15}")
        print("-" * 40)
        for i, (idx1, idx2, dist) in enumerate(matches_12[:10]):
            print(f"{idx1:<12} {idx2:<12} {dist:<15.8f}")
        if len(matches_12) > 10:
            print(f"... (还有 {len(matches_12) - 10} 个匹配)")
        
        # 统计信息
        distances = [m[2] for m in matches_12]
        print(f"\n匹配距离统计:")
        print(f"  最小距离: {np.min(distances):.8f}")
        print(f"  最大距离: {np.max(distances):.8f}")
        print(f"  平均距离: {np.mean(distances):.8f}")
        print(f"  中位数距离: {np.median(distances):.8f}")
        
        # 验证匹配逻辑
        verify_matching_logic(loc1, loc2, matches_12, "数据源1", "数据源2")
        
        # 提取匹配站点的 Map 和 SimilarityMat
        data_path = '/projects/p32685/Fixmatch/data'
        output_path = '/projects/p32685/Fixmatch/data/matched_stations_data.pkl'
        matched_indices, matched_map, matched_similarity = extract_matched_data(
            matches_12, data_path, output_path
        )
        
        if matched_indices is not None:
            print_separator("提取结果总结", "=", 80)
            print(f"✓ 成功提取 {len(matched_indices)} 个匹配站点的数据")
            print(f"  - 匹配索引数组形状: {matched_indices.shape}")
            print(f"  - Map 形状: {matched_map.shape}")
            print(f"  - SimilarityMat 形状: {matched_similarity.shape}")
            print(f"\n💡 提示: 上面的索引信息可以直接复制使用，从数据源2中提取对应的 Map 和 SimilarityMat")
    else:
        print("  ⚠️ 未找到匹配站点，显示最近距离统计:")
        min_dists = np.min(dist_matrix_12, axis=1)
        print(f"  最小距离范围: [{np.min(min_dists):.8f}, {np.max(min_dists):.8f}]")
        print(f"  平均最小距离: {np.mean(min_dists):.8f}")
    
    # 比较数据源1和数据源3
    print_separator("数据源1 vs 数据源3 (原始有标签 vs 无标签)")
    matches_13, dist_matrix_13 = find_matching_stations(
        loc1, loc3, threshold_13,
        "GNN_N1_StationMat", "Unlabeled_Finalized"
    )
    print(f"找到 {len(matches_13)} 个匹配站点 (阈值: {threshold_13})")
    if len(matches_13) > 0:
        print(f"\n匹配详情 (前10个):")
        print(f"{'数据源1索引':<12} {'数据源3索引':<12} {'距离':<15}")
        print("-" * 40)
        for i, (idx1, idx3, dist) in enumerate(matches_13[:10]):
            print(f"{idx1:<12} {idx3:<12} {dist:<15.8f}")
        if len(matches_13) > 10:
            print(f"... (还有 {len(matches_13) - 10} 个匹配)")
        
        distances = [m[2] for m in matches_13]
        print(f"\n匹配距离统计:")
        print(f"  最小距离: {np.min(distances):.8f}")
        print(f"  最大距离: {np.max(distances):.8f}")
        print(f"  平均距离: {np.mean(distances):.8f}")
        print(f"  中位数距离: {np.median(distances):.8f}")
    else:
        print("  ⚠️ 未找到匹配站点，显示最近距离统计:")
        min_dists = np.min(dist_matrix_13, axis=1)
        print(f"  最小距离范围: [{np.min(min_dists):.8f}, {np.max(min_dists):.8f}]")
        print(f"  平均最小距离: {np.mean(min_dists):.8f}")
    
    # 比较数据源2和数据源3
    print_separator("数据源2 vs 数据源3 (新有标签 vs 无标签)")
    matches_23, dist_matrix_23 = find_matching_stations(
        loc2, loc3, threshold_23,
        "Labeled_Finalized", "Unlabeled_Finalized"
    )
    print(f"找到 {len(matches_23)} 个匹配站点 (阈值: {threshold_23})")
    if len(matches_23) > 0:
        print(f"\n匹配详情 (前10个):")
        print(f"{'数据源2索引':<12} {'数据源3索引':<12} {'距离':<15}")
        print("-" * 40)
        for i, (idx2, idx3, dist) in enumerate(matches_23[:10]):
            print(f"{idx2:<12} {idx3:<12} {dist:<15.8f}")
        if len(matches_23) > 10:
            print(f"... (还有 {len(matches_23) - 10} 个匹配)")
        
        distances = [m[2] for m in matches_23]
        print(f"\n匹配距离统计:")
        print(f"  最小距离: {np.min(distances):.8f}")
        print(f"  最大距离: {np.max(distances):.8f}")
        print(f"  平均距离: {np.mean(distances):.8f}")
        print(f"  中位数距离: {np.median(distances):.8f}")
    else:
        print("  ⚠️ 未找到匹配站点，显示最近距离统计:")
        min_dists = np.min(dist_matrix_23, axis=1)
        print(f"  最小距离范围: [{np.min(min_dists):.8f}, {np.max(min_dists):.8f}]")
        print(f"  平均最小距离: {np.mean(min_dists):.8f}")
    
    # 总结
    print_separator("重合情况总结")
    print(f"数据源1 (GNN_N1_StationMat) - {loc1.shape[0]} 个站点:")
    print(f"  与数据源2重合: {len(matches_12)} 个 ({len(matches_12)/loc1.shape[0]*100:.1f}%)")
    print(f"  与数据源3重合: {len(matches_13)} 个 ({len(matches_13)/loc1.shape[0]*100:.1f}%)")
    
    print(f"\n数据源2 (Labeled_Finalized) - {loc2.shape[0]} 个站点:")
    print(f"  与数据源1重合: {len(matches_12)} 个 ({len(matches_12)/loc2.shape[0]*100:.1f}%)")
    print(f"  与数据源3重合: {len(matches_23)} 个 ({len(matches_23)/loc2.shape[0]*100:.1f}%)")
    
    print(f"\n数据源3 (Unlabeled_Finalized) - {loc3.shape[0]} 个站点:")
    print(f"  与数据源1重合: {len(matches_13)} 个 ({len(matches_13)/loc3.shape[0]*100:.1f}%)")
    print(f"  与数据源2重合: {len(matches_23)} 个 ({len(matches_23)/loc3.shape[0]*100:.1f}%)")
    
    # 检查是否有三向重合
    if len(matches_12) > 0 and len(matches_13) > 0:
        idx1_in_12 = {m[0] for m in matches_12}
        idx1_in_13 = {m[0] for m in matches_13}
        triple_overlap = idx1_in_12 & idx1_in_13
        if len(triple_overlap) > 0:
            print(f"\n三向重合站点数: {len(triple_overlap)} 个")
            print(f"  这些站点同时出现在三个数据源中")
    
    print_separator("检查完成", "=", 80)


def verify_duplicate_stations_features(loc1, matches, data_path):
    """
    验证重复站点的所有特征是否相同
    
    Args:
        loc1: 数据源1的位置矩阵
        matches: 匹配结果列表
        data_path: 数据文件路径
    """
    print_separator("验证重复站点的所有特征", "=", 80)
    
    # 找出重复站点对
    idx2_to_idx1s = {}
    for idx1, idx2, dist in matches:
        if idx2 not in idx2_to_idx1s:
            idx2_to_idx1s[idx2] = []
        idx2_to_idx1s[idx2].append(idx1)
    
    duplicate_pairs = [(idx2, idx1s) for idx2, idx1s in idx2_to_idx1s.items() if len(idx1s) > 1]
    
    if len(duplicate_pairs) == 0:
        print("  ✓ 没有发现重复站点")
        return
    
    # 加载数据源1的完整数据
    print(f"\n正在加载: {data_path}/GNN_N1_StationMat.mat")
    try:
        data1 = mat73.loadmat(f"{data_path}/GNN_N1_StationMat.mat")
        print(f"  ✓ 数据加载成功")
        print(f"  可用变量: {[k for k in data1.keys() if not k.startswith('__')]}")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    # 检查每个重复站点对的所有特征
    print(f"\n检查 {len(duplicate_pairs)} 对重复站点的所有特征:")
    
    all_same = True
    for idx2, idx1s in duplicate_pairs[:5]:  # 只检查前5对，避免输出过多
        print(f"\n  数据源2索引 {idx2} 对应的重复站点对:")
        print(f"    数据源1索引: {idx1s}")
        
        # 检查位置
        positions = [loc1[idx] for idx in idx1s]
        if np.allclose(positions, positions[0], atol=1e-6):
            print(f"    ✓ 位置相同")
        else:
            print(f"    ✗ 位置不同")
            all_same = False
        
        # 检查 StationMat_se_fill 的第一个feature（温度）
        if 'StationMat_se_fill' in data1:
            try:
                station_mat = np.array(data1['StationMat_se_fill'])  # (时间步, 特征数, 站点数)
                print(f"    检查 StationMat_se_fill: 形状 {station_mat.shape}")
                
                # 检查第三维是否是站点数（形状应该是 (时间, 特征, 站点)）
                if len(station_mat.shape) == 3 and station_mat.shape[2] == len(loc1):
                    # 提取每个重复站点的第一个feature（温度）：station_mat[:, 0, idx]
                    temp_values = [station_mat[:, 0, idx] for idx in idx1s]  # 第一个feature是温度
                    
                    # 检查温度数据是否相同
                    all_equal = all(np.allclose(v, temp_values[0], atol=1e-6) for v in temp_values[1:])
                    
                    if all_equal:
                        print(f"    ✓ StationMat_se_fill 第一个feature（温度）: 相同")
                    else:
                        print(f"    ✗ StationMat_se_fill 第一个feature（温度）: 不同")
                        all_same = False
                        # 显示差异（前5个时间步）
                        print(f"      前5个时间步的值:")
                        for i, v in enumerate(temp_values):
                            print(f"        站点{idx1s[i]}: {v[:5]}")
                    
                    # 检查第56-59个feature（auxiliary variables: hour, month, year, station number）
                    # 索引是55-58（从0开始）
                    aux_features = [
                        (55, 'hour'),
                        (56, 'month'),
                        (57, 'year'),
                        (58, 'station number')
                    ]
                    
                    print(f"\n    检查 auxiliary variables (feature 56-59):")
                    for feat_idx, feat_name in aux_features:
                        if feat_idx < station_mat.shape[1]:  # 确保索引有效
                            aux_values = [station_mat[:, feat_idx, idx] for idx in idx1s]
                            aux_equal = all(np.allclose(v, aux_values[0], atol=1e-6) for v in aux_values[1:])
                            
                            if aux_equal:
                                print(f"      ✓ feature {feat_idx+1} ({feat_name}): 相同")
                            else:
                                print(f"      ✗ feature {feat_idx+1} ({feat_name}): 不同")
                                all_same = False
                                # 显示差异（前5个时间步）
                                print(f"        前5个时间步的值:")
                                for i, v in enumerate(aux_values):
                                    print(f"          站点{idx1s[i]}: {v[:5]}")
                        else:
                            print(f"      ⚠️ feature {feat_idx+1} ({feat_name}): 索引超出范围（总特征数: {station_mat.shape[1]}）")
                else:
                    print(f"    ⚠️ StationMat_se_fill 形状不匹配（期望第三维={len(loc1)}，实际形状={station_mat.shape}）")
            except Exception as e:
                print(f"    ✗ StationMat_se_fill 检查失败 - {e}")
                all_same = False
        else:
            print(f"    ⚠️ 未找到 StationMat_se_fill 变量")
    
    if len(duplicate_pairs) > 5:
        print(f"\n  ... (还有 {len(duplicate_pairs) - 5} 对未显示)")
    
    print(f"\n总结:")
    if all_same:
        print(f"  ✓ 所有检查的重复站点对，除了位置外，其他特征也完全相同")
        print(f"  💡 结论：这些重复站点是完全重复的数据，可以安全去重")
    else:
        print(f"  ⚠️ 部分重复站点对的特征不完全相同")
        print(f"  💡 结论：这些站点虽然位置相同，但特征不同，可能是不同时间或不同来源的数据")


def check_locations_consistency():
    """检查 GNN_N1_StationMat.mat 和 GNN_N1_AJM.mat 的 Locations 是否相同"""
    print_separator("检查 Locations 一致性", "=", 80)
    
    data_path = '/projects/p32685/Fixmatch/data'
    
    # 加载 GNN_N1_StationMat.mat
    print(f"\n正在加载: {data_path}/GNN_N1_StationMat.mat")
    try:
        data1 = mat73.loadmat(f"{data_path}/GNN_N1_StationMat.mat")
        if 'Locations' in data1:
            loc1 = np.array(data1['Locations'])
            print(f"  ✓ 找到 Locations: {loc1.shape}")
        else:
            print(f"  ✗ 未找到 Locations")
            print(f"  可用变量: {[k for k in data1.keys() if not k.startswith('__')]}")
            return
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    # 加载 GNN_N1_AJM.mat
    print(f"\n正在加载: {data_path}/GNN_N1_AJM.mat")
    try:
        data2 = mat73.loadmat(f"{data_path}/GNN_N1_AJM.mat")
        # 检查可能的变量名
        loc2 = None
        loc2_key = None
        for key in ['Locations', 'Location', 'location', 'locations']:
            if key in data2:
                loc2 = np.array(data2[key])
                loc2_key = key
                print(f"  ✓ 找到 {key}: {loc2.shape}")
                break
        
        if loc2 is None:
            print(f"  ✗ 未找到位置变量")
            print(f"  可用变量: {[k for k in data2.keys() if not k.startswith('__')]}")
            return
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    # 比较两个 Locations
    print(f"\n比较 Locations:")
    print(f"  GNN_N1_StationMat.mat Locations: {loc1.shape}")
    print(f"  GNN_N1_AJM.mat {loc2_key}: {loc2.shape}")
    
    if loc1.shape != loc2.shape:
        print(f"  ✗ 形状不同！")
        return
    
    # 检查是否完全相同
    if np.allclose(loc1, loc2, atol=1e-6):
        print(f"  ✓ Locations 完全相同")
        print(f"  💡 结论：两个文件使用相同的 Locations 变量")
    else:
        print(f"  ✗ Locations 不同")
        # 计算差异
        diff = np.abs(loc1 - loc2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"    最大差异: {max_diff:.8f}")
        print(f"    平均差异: {mean_diff:.8f}")
        print(f"    不同位置数: {np.sum(np.any(diff > 1e-6, axis=1))} / {len(loc1)}")
        print(f"  💡 结论：两个文件的 Locations 不同，可能索引顺序不一致")


def check_dist_similarity_consistency():
    """检查 GNN_N1_AJM.mat 中位置相同的站点，dist 和 similarity 是否相同"""
    print_separator("检查 dist 和 similarity 一致性", "=", 80)
    
    data_path = '/projects/p32685/Fixmatch/data'
    
    # 加载 GNN_N1_AJM.mat
    print(f"\n正在加载: {data_path}/GNN_N1_AJM.mat")
    try:
        data = mat73.loadmat(f"{data_path}/GNN_N1_AJM.mat")
        print(f"  ✓ 数据加载成功")
        print(f"  可用变量: {[k for k in data.keys() if not k.startswith('__')]}")
        
        # 查找 location 变量
        location = None
        location_key = None
        for key in ['Locations', 'Location', 'location', 'locations']:
            if key in data:
                location = np.array(data[key])
                location_key = key
                print(f"  ✓ 找到 {key}: {location.shape}")
                break
        
        if location is None:
            print(f"  ✗ 未找到位置变量")
            return
        
        # 查找 dist 和 similarity 变量
        dist = None
        similarity = None
        dist_key = None
        sim_key = None
        
        for key in ['dist', 'Dist', 'DIST', 'distance', 'Distance']:
            if key in data:
                dist = np.array(data[key])
                dist_key = key
                print(f"  ✓ 找到 {key}: {dist.shape}")
                break
        
        for key in ['similarity', 'Similarity', 'SIMILARITY', 'SimilarityMat', 'similarityMat']:
            if key in data:
                similarity = np.array(data[key])
                sim_key = key
                print(f"  ✓ 找到 {key}: {similarity.shape}")
                break
        
        if dist is None and similarity is None:
            print(f"  ✗ 未找到 dist 或 similarity 变量")
            return
        
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    # 找出位置相同的站点对
    print(f"\n查找位置相同的站点对:")
    duplicate_pairs = []
    for i in range(len(location)):
        for j in range(i + 1, len(location)):
            if np.allclose(location[i], location[j], atol=1e-6):
                duplicate_pairs.append((i, j))
    
    if len(duplicate_pairs) == 0:
        print(f"  ✓ 没有发现位置相同的站点对")
        return
    
    print(f"  找到 {len(duplicate_pairs)} 对位置相同的站点")
    
    # 检查 dist 和 similarity
    all_same = True
    print(f"\n检查前 {min(10, len(duplicate_pairs))} 对位置相同站点的 dist 和 similarity:")
    
    for idx1, idx2 in duplicate_pairs[:10]:
        print(f"\n  站点 {idx1} 和站点 {idx2} (位置相同):")
        
        # 检查 dist
        if dist is not None:
            if len(dist.shape) == 2 and dist.shape[0] == len(location):
                dist1 = dist[idx1, :]
                dist2 = dist[idx2, :]
                if np.allclose(dist1, dist2, atol=1e-6):
                    print(f"    ✓ {dist_key}: 相同")
                else:
                    print(f"    ✗ {dist_key}: 不同")
                    max_diff = np.max(np.abs(dist1 - dist2))
                    print(f"      最大差异: {max_diff:.8f}")
                    all_same = False
            else:
                print(f"    ⚠️ {dist_key}: 形状不匹配 ({dist.shape} vs 期望站点数 {len(location)})")
        
        # 检查 similarity
        if similarity is not None:
            if len(similarity.shape) == 2 and similarity.shape[0] == len(location):
                sim1 = similarity[idx1, :]
                sim2 = similarity[idx2, :]
                if np.allclose(sim1, sim2, atol=1e-6):
                    print(f"    ✓ {sim_key}: 相同")
                else:
                    print(f"    ✗ {sim_key}: 不同")
                    max_diff = np.max(np.abs(sim1 - sim2))
                    print(f"      最大差异: {max_diff:.8f}")
                    all_same = False
            else:
                print(f"    ⚠️ {sim_key}: 形状不匹配 ({similarity.shape} vs 期望站点数 {len(location)})")
    
    if len(duplicate_pairs) > 10:
        print(f"\n  ... (还有 {len(duplicate_pairs) - 10} 对未显示)")
    
    print(f"\n总结:")
    if all_same:
        print(f"  ✓ 所有位置相同的站点对，dist 和 similarity 也相同")
        print(f"  💡 结论：索引顺序一致，数据对应关系正确")
    else:
        print(f"  ✗ 部分位置相同的站点对，dist 或 similarity 不同")
        print(f"  💡 结论：可能存在索引顺序不一致的问题")


if __name__ == "__main__":
    analyze_location_overlap()
    
    # 额外验证：检查重复站点的所有特征
    print("\n" + "="*80)
    data_path = '/projects/p32685/Fixmatch/data'
    loc1, loc2, loc3 = load_locations()
    if loc1 is not None:
        # 重新计算匹配以获取matches
        threshold_12 = 0.001
        matches_12, _ = find_matching_stations(loc1, loc2, threshold_12)
        verify_duplicate_stations_features(loc1, matches_12, data_path)
    
    # 检查 Locations 一致性
    print("\n" + "="*80)
    check_locations_consistency()
    
    # 检查 dist 和 similarity 一致性
    print("\n" + "="*80)
    check_dist_similarity_consistency()


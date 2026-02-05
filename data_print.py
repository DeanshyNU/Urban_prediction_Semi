"""
打印 /projects/p32685/Fixmatch/data 目录下所有数据文件的信息
"""
import os
import numpy as np
import mat73
from pathlib import Path

data_path = '/projects/p32685/Fixmatch/data'

def print_mat_info(filepath):
    """打印 .mat 文件的信息"""
    print(f"\n{'='*80}")
    print(f"文件: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    try:
        data = mat73.loadmat(filepath)
        print(f"包含的变量:")
        for key in data.keys():
            if not key.startswith('__'):  # 跳过MATLAB内部变量
                value = data[key]
                if isinstance(value, np.ndarray):
                    print(f"  {key}:")
                    print(f"    形状: {value.shape}")
                    print(f"    数据类型: {value.dtype}")
                    print(f"    最小值: {np.nanmin(value):.6f}" if value.size > 0 else "    最小值: N/A")
                    print(f"    最大值: {np.nanmax(value):.6f}" if value.size > 0 else "    最大值: N/A")
                    print(f"    均值: {np.nanmean(value):.6f}" if value.size > 0 else "    均值: N/A")
                    nan_count = np.isnan(value).sum()
                    if nan_count > 0:
                        print(f"    NaN数量: {nan_count} ({nan_count/value.size*100:.2f}%)")
                    else:
                        print(f"    NaN数量: 0")
                elif isinstance(value, (int, float, str)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value)}")
    except Exception as e:
        print(f"  错误: {str(e)}")

def print_pkl_info(filepath):
    """打印 .pkl 文件的信息"""
    import pickle
    print(f"\n{'='*80}")
    print(f"文件: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"类型: {type(data)}")
        if isinstance(data, dict):
            print(f"包含的键: {list(data.keys())}")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        elif isinstance(data, np.ndarray):
            print(f"形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
        else:
            print(f"内容: {data}")
    except Exception as e:
        print(f"  错误: {str(e)}")

def main():
    print("="*80)
    print("数据目录信息打印")
    print(f"路径: {data_path}")
    print("="*80)
    
    if not os.path.exists(data_path):
        print(f"错误: 路径不存在 {data_path}")
        return
    
    # 获取所有文件
    files = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
    
    print(f"\n找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {f}")
    
    # 处理 .mat 文件
    mat_files = [f for f in files if f.endswith('.mat')]
    print(f"\n\n处理 {len(mat_files)} 个 .mat 文件...")
    for mat_file in mat_files:
        filepath = os.path.join(data_path, mat_file)
        print_mat_info(filepath)
    
    # 处理 .pkl 文件
    pkl_files = [f for f in files if f.endswith('.pkl')]
    if pkl_files:
        print(f"\n\n处理 {len(pkl_files)} 个 .pkl 文件...")
        for pkl_file in pkl_files:
            filepath = os.path.join(data_path, pkl_file)
            print_pkl_info(filepath)

if __name__ == "__main__":
    main()

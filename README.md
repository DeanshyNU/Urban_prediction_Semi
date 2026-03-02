# FixMatch ESTNet 模块化代码

## 目录结构

```
Urban_prediction_Semi/
├── __init__.py              # 包初始化文件
├── main.py                  # 主程序入口
├── utils.py                 # 工具函数模块
├── data_preprocessing.py    # 数据预处理模块
├── geo_features.py          # 地理特征生成模块
├── data_generation.py       # PyG数据集生成模块
├── data_augmentation.py     # 数据增强模块
├── models.py                # 模型定义模块
└── training.py              # 训练相关模块
```

## 模块说明

### 1. `utils.py`
包含通用工具函数：
- `RMSE()` - 计算RMSE指标
- `MinMax()` - Min-Max归一化
- `MinMax_first_dim()` - 基于第一维的归一化
- `add_auxiliary_variables()` - 添加辅助变量
- `plotPrediction()` - 绘制预测结果
- `plotHist()` - 绘制训练历史

### 2. `data_preprocessing.py`
数据预处理功能：
- `merge_wind_components()` - 合并风速分量
- `reduce_stations()` - 削减站点数量
- `preprocess_unlabeled_data()` - 完整的无标签数据预处理流程

### 3. `geo_features.py`
地理特征生成：
- `genGeoFeatures()` - 从文件生成地理特征（有标签数据）
- `genGeoFeatures_unlabeled()` - 从数据生成地理特征（无标签数据）

### 4. `data_generation.py`
PyG数据集生成：
- `dataGen_ESTnet()` - 生成有标签数据的PyG数据集
- `dataGen_unlabeled_ESTnet()` - 生成无标签数据的PyG数据集

### 5. `data_augmentation.py`
FixMatch数据增强策略：
- `RandAugment` - 随机增强类
- `TransformFixMatch` - FixMatch增强类
- 各种增强操作函数（FeatureNoise, FeatureScale, TimeNoise等）

### 6. `models.py`
模型定义：
- `GNN_ESTNet` - ESTNet架构的图神经网络模型

### 7. `training.py`
训练相关功能：
- `train_fixmatch()` - FixMatch训练函数
- `test_fixmatch()` - 测试函数
- `loadCheckPoint()` - 检查点加载函数

### 8. `main.py`
主程序入口，整合所有模块，执行完整的训练流程。

## 数据路径配置

数据文件位于项目目录下的 `data/` 文件夹。

各主程序通过 `DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')` 自动定位。

## 使用方法

直接运行主程序：

```bash
cd Urban_prediction_Semi
python Fixmatch_no_UQ.py
```

## 主要改进

1. **模块化设计**：将原来1146行的单文件代码拆分为8个功能模块
2. **数据预处理整合**：将所有全局执行的数据预处理操作移到 `main()` 函数中
3. **路径配置**：统一管理数据路径，便于修改
4. **代码可维护性**：每个模块职责清晰，便于维护和扩展


# 物理一致性损失 (Physics Consistency Loss) 详细修改计划

## 概述

为 Pi-Model 添加基于城市地表特征相似度的残差平滑正则化损失。

**核心思想**: 如果两个站点的城市地表特征（不透水面、植被、建筑等）相似且在图中相连，那么它们对 WRF 的温度修正（残差 = pred - WRF_Tair）也应该相似。

**涉及修改的文件**:
1. `data_preprocessing.py` — 修复 WRF 变量顺序对齐（Critical Bug Fix）
2. `physics_loss.py` — **新建文件**，包含相似度计算和物理损失函数
3. `data_generation.py` — 在 metadata 中添加 UrbanFeature
4. `pimodel.py` — 预计算相似度权重，传递给训练函数
5. `pimodel_training.py` — 在训练循环中集成物理损失

---

## 修改 0: 修复 WRF 变量顺序对齐（Critical Bug Fix）

### 背景

有标签数据和无标签数据的 WRF 变量在 54 维特征中的排列顺序不同：

| 索引(每变量×9grid) | 有标签数据 | 无标签数据(合并后) |
|---|---|---|
| 0-8   | Tair (空气温度)      | Tair (空气温度)      |
| 9-17  | **Humidity (湿度)**  | **Tskin (地表温度)**  |
| 18-26 | **Irradiance (辐射)** | **Ttopsoil (土壤温度)** |
| 27-35 | **Wind speed (风速)** | **Humidity (湿度)**   |
| 36-44 | **Tskin (地表温度)**  | **Irradiance (辐射)** |
| 45-53 | **Ttopsoil (土壤温度)** | **Wind magnitude (风速)** |

除 Tair 外全部错位！模型在有标签数据上学到 "index 9-17 = 湿度"，但无标签数据中 index 9-17 实际是地表温度。

### 文件: `data_preprocessing.py`

### 修改位置: 在 `merge_wind_components` 函数之后（第 32 行之后），添加新函数

```python
def reorder_wrf_to_labeled_order(data):
    """
    将无标签数据的WRF变量顺序对齐到有标签数据的顺序

    无标签数据合并风速后的顺序(6变量×9grid=54):
        [0:9]   Air temperature
        [9:18]  Skin temperature
        [18:27] Topsoil temperature
        [27:36] Air humidity
        [36:45] Solar radiation
        [45:54] Wind magnitude

    有标签数据的顺序(6变量×9grid=54):
        [0:9]   Tair (Air temperature)
        [9:18]  Humidity
        [18:27] Irradiance (Solar radiation)
        [27:36] Wind speed
        [36:45] Tskin (Skin temperature)
        [45:54] Ttopsoil (Topsoil temperature)

    重排映射: 无标签 → 有标签
        Tair(0-8)       → pos 0-8    (不变)
        Humidity(27-35)  → pos 9-17
        Irradiance(36-44)→ pos 18-26
        Wind(45-53)      → pos 27-35
        Tskin(9-17)      → pos 36-44
        Ttopsoil(18-26)  → pos 45-53
    """
    WRFMat = data['WRFMat']  # shape: (6624, 54, nNodes)

    reorder_indices = np.concatenate([
        np.arange(0, 9),     # Tair → pos 0 (不变)
        np.arange(27, 36),   # Humidity → pos 1
        np.arange(36, 45),   # Solar radiation/Irradiance → pos 2
        np.arange(45, 54),   # Wind magnitude → pos 3
        np.arange(9, 18),    # Skin temperature → pos 4
        np.arange(18, 27),   # Topsoil temperature → pos 5
    ])

    data['WRFMat'] = WRFMat[:, reorder_indices, :]
    return data
```

### 修改位置: `preprocess_unlabeled_data` 函数中，第 115 行之后（`merge_wind_components` 调用之后）

原代码:
```python
    # 2. 合并风速分量
    print("正在合并风速分量...")
    unlabeled_data = merge_wind_components(unlabeled_data)

    # 3. 统一数据类型为float64
```

改为:
```python
    # 2. 合并风速分量
    print("正在合并风速分量...")
    unlabeled_data = merge_wind_components(unlabeled_data)

    # 2.5 重排WRF变量顺序，对齐到有标签数据
    print("正在重排WRF变量顺序（对齐到有标签数据）...")
    unlabeled_data = reorder_wrf_to_labeled_order(unlabeled_data)

    # 3. 统一数据类型为float64
```

### 验证方法

在 reorder 后添加一个 print 检查（调试完成后可删除）:
```python
print(f"  重排后 WRFMat shape: {unlabeled_data['WRFMat'].shape}")
print(f"  重排后顺序: [Tair, Humidity, Irradiance, Wind, Tskin, Ttopsoil]")
```

---

## 修改 1: 新建 `physics_loss.py`

### 文件: `physics_loss.py`（在项目根目录 `/home/user/Urban_prediction_Semi/` 下新建）

### 完整代码

```python
"""
物理一致性损失模块
基于城市地表特征相似度的残差平滑正则化

核心思想:
    residual_i = pred_i - WRF_Tair_i (模型对WRF的温度修正)
    如果站点 i 和 j 的城市地表特征相似，且它们在图中相连，
    则 residual_i ≈ residual_j（修正量应该接近）

    L_phys = mean_{(i,j)∈E} [ sim(i,j) * (residual_i - residual_j)² ]
"""
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse


def compute_urban_similarity(urban_features, sigma=0.2):
    """
    基于全部17维 UrbanFeature 计算站点间地表相似度矩阵

    Args:
        urban_features: np.ndarray, shape (nNodes, 17)
            17维城市地表特征（已归一化），包括:
            - 0-3: 树木特征 (mean height, max height, std, south direction)
            - 4-7: 建筑特征 (mean height, max height, std, south direction)
            - 8-11: 土地利用密度 (high, medium, low, open)
            - 12: Fi (不透水面比例)
            - 13: Fv (低矮植被比例)
            - 14: Fw (水体比例)
            - 15: Ft (树木覆盖比例)
            - 16: 到密歇根湖距离
        sigma: float, 高斯核带宽参数
            控制"多相似才算相似"，值越小要求越严格

    Returns:
        similarity_matrix: np.ndarray, shape (nNodes, nNodes)
            sim(i,j) = exp(-||feat_i - feat_j||² / (2 * sigma²))
            值域 [0, 1]，1=完全相同，0=完全不同
    """
    nNodes = urban_features.shape[0]

    # 计算两两欧式距离的平方: ||feat_i - feat_j||²
    # 使用广播: (nNodes, 1, 17) - (1, nNodes, 17) → (nNodes, nNodes, 17)
    diff = urban_features[:, np.newaxis, :] - urban_features[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)  # (nNodes, nNodes)

    # 高斯核
    similarity_matrix = np.exp(-dist_sq / (2 * sigma ** 2))

    return similarity_matrix


def compute_similarity_edge_weights(urban_features, adj_matrix, sigma=0.2):
    """
    预计算每条边的相似度权重，与 edge_index 对齐

    Args:
        urban_features: np.ndarray, shape (nNodes, 17)
        adj_matrix: np.ndarray, shape (nNodes, nNodes), 邻接矩阵
        sigma: float, 高斯核带宽

    Returns:
        sim_edge_weights: torch.FloatTensor, shape (num_edges,)
            每条边对应的城市地表特征相似度权重
            注意: 顺序与 dense_to_sparse(adj_matrix) 返回的 edge_index 完全对齐
    """
    # 1. 计算相似度矩阵
    sim_matrix = compute_urban_similarity(urban_features, sigma=sigma)

    # 2. 获取 edge_index（与 data_generation.py 中完全一致的方式）
    edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_matrix))

    # 3. 提取每条边的相似度
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    sim_edge_weights = sim_matrix[src, dst]

    return torch.FloatTensor(sim_edge_weights)


def physical_consistency_loss(pred, wrf_tair, edge_index, sim_edge_weights, nNodes):
    """
    计算物理一致性损失（残差平滑正则化）

    对于图中的每条边 (i, j):
        residual_i = pred_i - wrf_tair_i  (模型对WRF的修正)
        residual_j = pred_j - wrf_tair_j
        loss_edge = sim(i,j) * (residual_i - residual_j)²

    最终 loss = mean(所有边的 loss_edge)

    Args:
        pred: torch.Tensor, shape (batch_nodes,) 或 (batch_nodes, 1)
            模型预测值（所有 batch 中的节点拼在一起）
        wrf_tair: torch.Tensor, shape (batch_nodes,)
            从 batch.x 中提取的 WRF 空气温度中心格点值
            即 batch.x[:, 4]（Tair 在 3x3 grid 中的中心点）
        edge_index: torch.LongTensor, shape (2, batch_edges)
            PyG 自动 batching 后的 edge_index
            （已经自动处理了节点索引偏移）
        sim_edge_weights: torch.FloatTensor, shape (num_edges_per_graph,)
            单个图（一个时间步）的相似度边权重
            会在函数内部自动 repeat 匹配 batch 大小
        nNodes: int
            单个图中的节点数（用于计算 batch 中有多少个图）

    Returns:
        loss: torch.Tensor, scalar
            物理一致性损失值
    """
    # 处理 pred 维度
    pred = pred.squeeze()  # (batch_nodes,)

    # 计算残差: 模型对WRF的修正量
    residual = pred - wrf_tair  # (batch_nodes,)

    # 将单图的 sim_edge_weights repeat 到整个 batch
    num_graphs = pred.shape[0] // nNodes
    sim_batch = sim_edge_weights.repeat(num_graphs).to(pred.device)  # (batch_edges,)

    # 提取每条边两端节点的残差
    src, dst = edge_index  # src, dst shape: (batch_edges,)
    residual_diff = residual[src] - residual[dst]  # (batch_edges,)

    # 加权残差差异的平方
    loss = (sim_batch * residual_diff ** 2).mean()

    return loss
```

---

## 修改 2: `data_generation.py` — 在 metadata 中添加 UrbanFeature

### 目的
让 `pimodel.py` 能获取到有标签数据的 UrbanFeature 用于预计算相似度。

### 修改位置: `dataGen_ESTnet` 函数，第 429-441 行（metadata 字典）

原代码（第 429-441 行）:
```python
    metadata = {
        'nNodes': _nStations,
        'geoOff': _off,
        'geoScl': _scl,
        'iDim': _iDim,
        'oDim': _target.shape[-1],
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': Adj,
    }
```

改为（添加 `UrbanFeature` 字段）:
```python
    metadata = {
        'nNodes': _nStations,
        'geoOff': _off,
        'geoScl': _scl,
        'iDim': _iDim,
        'oDim': _target.shape[-1],
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': Adj,
        # 提取 UrbanFeature (静态, 取第一个时间步即可)
        # rawGeoFeatIdx 对应的特征就是 UrbanFeature (17维)
        'UrbanFeature': features[0, :, rawGeoFeatIdx],  # shape: (nNodes, 17)
    }
```

**注意**: `features[0, :, rawGeoFeatIdx]` 取的是第一个时间步的 rawGeo 特征，因为这些是静态特征（不随时间变化），所以取哪个时间步都一样。`rawGeoFeatIdx = np.arange(58, 75)` 对应的 17 维就是 UrbanFeature。

### 修改位置: `dataGen_unlabeled_ESTnet` 函数，第 636-649 行（metadata 字典）

原代码（第 636-649 行）:
```python
    metadata = {
        'nNodes': _nStations,
        'geoOff': _off,
        'geoScl': _scl,
        'iDim': _iDim,
        'oDim': 1 if labeled else None,
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': Adj,
        'isLabeled': labeled
    }
```

改为（添加 `UrbanFeature` 字段）:
```python
    metadata = {
        'nNodes': _nStations,
        'geoOff': _off,
        'geoScl': _scl,
        'iDim': _iDim,
        'oDim': 1 if labeled else None,
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': Adj,
        'isLabeled': labeled,
        'UrbanFeature': UrbanFeature,  # shape: (nNodes, 17), 已在上方定义
    }
```

**注意**: 在 `dataGen_unlabeled_ESTnet` 中，`UrbanFeature` 已经在第 519 行定义:
```python
UrbanFeature = data['UrbanFeature']  # (nNodes, 17)
```
所以直接引用即可。

---

## 修改 3: `pimodel.py` — 预计算相似度，传递给训练函数

### 修改位置: 文件顶部 import（第 1-18 行附近）

原代码:
```python
from data_preprocessing import preprocess_unlabeled_data
from data_generation import dataGen_ESTnet, dataGen_unlabeled_ESTnet
from data_augmentation import TransformFixMatch
from models import GNN
from pimodel_training import train_pimodel, test_pimodel, loadCheckPoint
from utils import plotHist
```

改为（添加 physics_loss 导入）:
```python
from data_preprocessing import preprocess_unlabeled_data
from data_generation import dataGen_ESTnet, dataGen_unlabeled_ESTnet
from data_augmentation import TransformFixMatch
from models import GNN
from pimodel_training import train_pimodel, test_pimodel, loadCheckPoint
from physics_loss import compute_similarity_edge_weights
from utils import plotHist
```

### 修改位置: 数据增强之后、模型创建之前（约第 78-82 行之间），添加相似度预计算

在第 81 行 (`print(f"有标签验证样本数量: ..."`) 之后，添加:

```python
    ##----------------------预计算物理一致性损失的相似度权重----------------------
    print("步骤2.5: 预计算城市地表特征相似度权重")

    # 有标签数据的相似度权重
    sim_edge_weights_labeled = compute_similarity_edge_weights(
        urban_features=metadata['UrbanFeature'],  # (68, 17)
        adj_matrix=metadata['AdjMatrix'],          # (68, 68)
        sigma=0.2
    )
    print(f"  有标签数据: {sim_edge_weights_labeled.shape[0]} 条边的相似度权重已计算")

    # 无标签数据的相似度权重
    # 注意: 需要从无标签数据中获取 UrbanFeature 和 AdjMatrix
    # aug1 和 aug2 共享相同的图结构和 UrbanFeature（UrbanFeature 不被增强）
    # 因此只需要用任一增强数据的 metadata
    _, _, _, aug1_metadata, _ = dataGen_unlabeled_ESTnet(dataParam, aug1_data, labeled=False, path=DATA_PATH)
    sim_edge_weights_unlabeled = compute_similarity_edge_weights(
        urban_features=aug1_metadata['UrbanFeature'],  # (200, 17)
        adj_matrix=aug1_metadata['AdjMatrix'],          # (200, 200)
        sigma=0.2
    )
    nNodes_unlabeled = aug1_metadata['nNodes']
    print(f"  无标签数据: {sim_edge_weights_unlabeled.shape[0]} 条边的相似度权重已计算")
```

**重要**: 这里需要调整原来的 aug1_loader 和 aug2_loader 生成逻辑。原代码（第 76-77 行）:

```python
    aug1_loader, _, _, _, _ = dataGen_unlabeled_ESTnet(dataParam, aug1_data, labeled=False, path=DATA_PATH)
    aug2_loader, _, _, _, _ = dataGen_unlabeled_ESTnet(dataParam, aug2_data, labeled=False, path=DATA_PATH)
```

改为（保留 aug1 的 metadata 用于获取无标签图信息）:

```python
    aug1_loader, _, _, aug1_metadata, _ = dataGen_unlabeled_ESTnet(dataParam, aug1_data, labeled=False, path=DATA_PATH)
    aug2_loader, _, _, _, _ = dataGen_unlabeled_ESTnet(dataParam, aug2_data, labeled=False, path=DATA_PATH)
```

然后在上面的 "步骤2.5" 中就不需要再调用一次 `dataGen_unlabeled_ESTnet`，直接用 `aug1_metadata`：

```python
    ##----------------------预计算物理一致性损失的相似度权重----------------------
    print("步骤2.5: 预计算城市地表特征相似度权重")

    # 有标签数据的相似度权重
    sim_edge_weights_labeled = compute_similarity_edge_weights(
        urban_features=metadata['UrbanFeature'],
        adj_matrix=metadata['AdjMatrix'],
        sigma=0.2
    )
    print(f"  有标签数据: {sim_edge_weights_labeled.shape[0]} 条边的相似度权重已计算")

    # 无标签数据的相似度权重（aug1 和 aug2 共享相同的图结构）
    sim_edge_weights_unlabeled = compute_similarity_edge_weights(
        urban_features=aug1_metadata['UrbanFeature'],
        adj_matrix=aug1_metadata['AdjMatrix'],
        sigma=0.2
    )
    nNodes_unlabeled = aug1_metadata['nNodes']
    print(f"  无标签数据: {sim_edge_weights_unlabeled.shape[0]} 条边的相似度权重已计算")
```

### 修改位置: wandb.init 的 config（约第 122-134 行）

在 config 字典中添加新超参数:
```python
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=run_name,
        config={
            **dataParam,
            **modelParam,
            'method': 'PiModel_PhysLoss',  # 修改方法名以区分
            'n_unlabeled': n_unlabeled,
            'nNodes': metadata['nNodes'],
            'nEpoch': nEpoch,
            'lr': 1e-3,
            'scheduler_gamma': 0.9992,
            'lambda_U': 10.0,
            'lambda_phys': 0.1,       # 新增: 物理损失权重
            'phys_sigma': 0.2,         # 新增: 相似度高斯核带宽
            'ramp_epochs': 30,
            'output_dir': output_dir,
        }
    )
```

### 修改位置: 训练循环（第 162-168 行）

原代码:
```python
    for epoch in range(EPOCH, nEpoch):
        ramp = rampup_factor(epoch, 30)
        # Π-Model训练
        trainLoss, trainRMSE, _, _ = train_pimodel(
            trainLoader, aug1_loader, aug2_loader, model, lossFn, consistency_loss_fn,
            opt, scheduler, device, metadata['nNodes'], lambda_U=10.0 * ramp
        )
```

改为:
```python
    for epoch in range(EPOCH, nEpoch):
        ramp = rampup_factor(epoch, 30)
        # Π-Model训练（含物理一致性损失）
        trainLoss, trainRMSE, _, _ = train_pimodel(
            trainLoader, aug1_loader, aug2_loader, model, lossFn, consistency_loss_fn,
            opt, scheduler, device, metadata['nNodes'],
            lambda_U=10.0 * ramp,
            lambda_phys=0.1 * ramp,
            sim_edge_weights_labeled=sim_edge_weights_labeled,
            sim_edge_weights_unlabeled=sim_edge_weights_unlabeled,
            nNodes_unlabeled=nNodes_unlabeled
        )
```

### 修改位置: wandb.log（约第 184-199 行）

在 wandb.log 中添加 lambda_phys:
```python
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/rmse_std': trainRMSE[1],
            'train/rmse_min': trainRMSE[2],
            'train/rmse_max': trainRMSE[3],
            'valid/loss': validLoss,
            'valid/rmse': validRMSE[0],
            'valid/rmse_std': validRMSE[1],
            'valid/rmse_min': validRMSE[2],
            'valid/rmse_max': validRMSE[3],
            'learning_rate': scheduler.get_last_lr()[0],
            'lambda_U': 10.0 * ramp,
            'lambda_phys': 0.1 * ramp,  # 新增
            'ramp_factor': ramp,
        })
```

---

## 修改 4: `pimodel_training.py` — 集成物理损失

### 修改位置: 文件顶部 import（第 1-8 行）

原代码:
```python
import numpy as np
import torch
import os
from utils import RMSE
```

改为:
```python
import numpy as np
import torch
import os
from utils import RMSE
from physics_loss import physical_consistency_loss
```

### 修改位置: `train_pimodel` 函数签名（第 46 行）

原代码:
```python
def train_pimodel(loader, aug1_loader, aug2_loader, model, lossFn, consistency_loss_fn, opt, scheduler, device, nNodes, lambda_U=1.0):
```

改为:
```python
def train_pimodel(loader, aug1_loader, aug2_loader, model, lossFn, consistency_loss_fn, opt, scheduler, device, nNodes, lambda_U=1.0, lambda_phys=0.0, sim_edge_weights_labeled=None, sim_edge_weights_unlabeled=None, nNodes_unlabeled=None):
```

### 修改位置: `train_pimodel` 函数的 docstring（第 47-62 行）

在参数说明中添加:
```python
    """
    Π-Model训练函数

    参数:
        loader: 有标签数据加载器
        aug1_loader: 无标签数据的第一次增强
        aug2_loader: 无标签数据的第二次增强
        model: GNN模型
        lossFn: 有标签损失函数
        consistency_loss_fn: 一致性损失函数（MSE）
        opt: 优化器
        scheduler: 学习率调度器
        device: 设备
        nNodes: 有标签数据节点数
        lambda_U: 无标签损失权重
        lambda_phys: 物理一致性损失权重 (新增)
        sim_edge_weights_labeled: 有标签数据的相似度边权重 (新增)
        sim_edge_weights_unlabeled: 无标签数据的相似度边权重 (新增)
        nNodes_unlabeled: 无标签数据节点数 (新增)
    """
```

### 修改位置: loss 追踪变量（第 64-66 行）

原代码:
```python
    total_labeled_loss = 0
    total_consistency_loss = 0
    total_batches = 0
```

改为:
```python
    total_labeled_loss = 0
    total_consistency_loss = 0
    total_phys_loss = 0  # 新增
    total_batches = 0
```

### 修改位置: 训练循环内部，替换第 76-86 行

原代码（第 76-86 行）:
```python
        # 有标签损失计算
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labeled_loss = lossFn(logits, batch.y)

        # 一致性损失计算：对同一无标签数据的两次不同增强预测应该一致
        pred1 = model(aug1_batch.x, aug1_batch.edge_index, aug1_batch.edge_attr)
        pred2 = model(aug2_batch.x, aug2_batch.edge_index, aug2_batch.edge_attr)
        consistency_loss = consistency_loss_fn(pred1, pred2)

        # 总损失
        total_loss = labeled_loss + lambda_U * consistency_loss
```

改为:
```python
        # 有标签损失计算
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labeled_loss = lossFn(logits, batch.y)

        # 一致性损失计算：对同一无标签数据的两次不同增强预测应该一致
        pred1 = model(aug1_batch.x, aug1_batch.edge_index, aug1_batch.edge_attr)
        pred2 = model(aug2_batch.x, aug2_batch.edge_index, aug2_batch.edge_attr)
        consistency_loss = consistency_loss_fn(pred1, pred2)

        # 物理一致性损失计算
        phys_loss = torch.tensor(0.0, device=device)
        if lambda_phys > 0 and sim_edge_weights_labeled is not None:
            # 从 batch.x 提取 WRF 空气温度中心格点
            # WRF Tair 在 CFD 特征的前9维 (3×3 grid)，中心格点 = index 4
            wrf_tair_labeled = batch.x[:, 4]
            phys_loss_labeled = physical_consistency_loss(
                pred=logits,
                wrf_tair=wrf_tair_labeled,
                edge_index=batch.edge_index,
                sim_edge_weights=sim_edge_weights_labeled,
                nNodes=nNodes
            )

            # 对无标签数据也计算物理损失（使用 pred1 的预测）
            wrf_tair_unlabeled = aug1_batch.x[:, 4]
            phys_loss_unlabeled = physical_consistency_loss(
                pred=pred1,
                wrf_tair=wrf_tair_unlabeled,
                edge_index=aug1_batch.edge_index,
                sim_edge_weights=sim_edge_weights_unlabeled,
                nNodes=nNodes_unlabeled
            )

            # 物理损失 = 有标签 + 无标签 的平均
            phys_loss = (phys_loss_labeled + phys_loss_unlabeled) / 2.0

        # 总损失
        total_loss = labeled_loss + lambda_U * consistency_loss + lambda_phys * phys_loss
```

### 修改位置: loss 累积（第 95-96 行附近）

原代码:
```python
        total_labeled_loss += labeled_loss.item()
        total_consistency_loss += consistency_loss.item()
```

改为:
```python
        total_labeled_loss += labeled_loss.item()
        total_consistency_loss += consistency_loss.item()
        total_phys_loss += phys_loss.item()  # 新增
```

### 修改位置: 批次打印（第 102-106 行）

原代码:
```python
        if batch_idx % 10 == 0:
            print(f"批次 {batch_idx + 1}: "
                  f"标签损失 = {labeled_loss.item():.4f}, "
                  f"一致性损失 = {consistency_loss.item():.4f}, "
                  f"总损失 = {total_loss.item():.4f}")
```

改为:
```python
        if batch_idx % 10 == 0:
            print(f"批次 {batch_idx + 1}: "
                  f"标签损失 = {labeled_loss.item():.4f}, "
                  f"一致性损失 = {consistency_loss.item():.4f}, "
                  f"物理损失 = {phys_loss.item():.4f}, "
                  f"总损失 = {total_loss.item():.4f}")
```

### 修改位置: 平均损失计算（第 110-112 行）

原代码:
```python
    avg_labeled_loss = total_labeled_loss / total_batches
    avg_consistency_loss = total_consistency_loss / total_batches
```

改为:
```python
    avg_labeled_loss = total_labeled_loss / total_batches
    avg_consistency_loss = total_consistency_loss / total_batches
    avg_phys_loss = total_phys_loss / total_batches  # 新增
```

### 修改位置: 轮次摘要打印（第 120-123 行）

原代码:
```python
    print(f"轮次摘要: "
          f"平均标签损失 = {avg_labeled_loss:.4f}, "
          f"平均一致性损失 = {avg_consistency_loss:.4f}, "
          f"RMSE = {rmse[0]:.4f}")
```

改为:
```python
    print(f"轮次摘要: "
          f"平均标签损失 = {avg_labeled_loss:.4f}, "
          f"平均一致性损失 = {avg_consistency_loss:.4f}, "
          f"平均物理损失 = {avg_phys_loss:.4f}, "
          f"RMSE = {rmse[0]:.4f}")
```

### 修改位置: 返回值（第 125 行）

原代码:
```python
    return avg_labeled_loss + lambda_U * avg_consistency_loss, rmse, truth, pred
```

改为:
```python
    return avg_labeled_loss + lambda_U * avg_consistency_loss + lambda_phys * avg_phys_loss, rmse, truth, pred
```

---

## 超参数总结

| 参数 | 位置 | 初始值 | 说明 |
|---|---|---|---|
| `lambda_phys` | `pimodel.py` 训练循环 | `0.1` | 物理损失权重，带 ramp-up |
| `sigma` | `pimodel.py` 预计算相似度 | `0.2` | 高斯核带宽，控制相似度的严格程度 |
| ramp-up | `pimodel.py` 训练循环 | 与 `lambda_U` 相同 (30 epochs) | `lambda_phys * ramp` |

### 调参建议

1. **`lambda_phys`**: 从 0.1 开始。如果 RMSE 改善则保持，如果恶化则减小到 0.01。注意物理损失不应该主导总损失——观察 wandb 中三个损失的比例。
2. **`sigma`**:
   - 太小 (< 0.1): 只有几乎完全相同的站点才有权重，约束太弱
   - 太大 (> 0.5): 所有站点都被当作相似，变成无条件平滑
   - 建议尝试 [0.1, 0.2, 0.3]
3. **是否只在有标签数据上用**: 如果要简化，可以先只在有标签数据上加物理损失（去掉 `phys_loss_unlabeled` 部分），验证效果后再扩展到无标签数据。

---

## 数据流图示

```
pimodel.py main():
│
├── preprocess_unlabeled_data()     ← 修改0: 添加 reorder_wrf_to_labeled_order
│
├── dataGen_ESTnet()                ← 修改2: metadata 添加 UrbanFeature
│   └── metadata['UrbanFeature']    → (68, 17)
│
├── dataGen_unlabeled_ESTnet()      ← 修改2: metadata 添加 UrbanFeature
│   └── aug1_metadata['UrbanFeature'] → (200, 17)
│
├── compute_similarity_edge_weights()  ← 修改1: 新文件 physics_loss.py
│   ├── sim_edge_weights_labeled      → (num_edges_labeled,)
│   └── sim_edge_weights_unlabeled    → (num_edges_unlabeled,)
│
└── 训练循环:
    └── train_pimodel()               ← 修改4: pimodel_training.py
        ├── logits = model(batch.x, ...)
        ├── labeled_loss = lossFn(logits, batch.y)
        ├── consistency_loss = MSE(pred1, pred2)
        ├── wrf_tair = batch.x[:, 4]                    ← 提取 WRF Tair 中心格点
        ├── phys_loss = physical_consistency_loss(...)    ← 新增
        └── total = labeled + λ_U·consistency + λ_phys·phys
```

---

## batch.x 特征索引参考（geoFeatures='full', window=2）

```
batch.x 布局 (每个节点):
├── [0:54]     当前时间步 CFD 特征 (54 = 6变量 × 9grid)
│   ├── [0:9]   Tair (空气温度, 3×3 grid)
│   │   └── [4]  ★ Tair 中心格点 ← 物理损失从这里提取
│   ├── [9:18]  Humidity (湿度)
│   ├── [18:27] Irradiance (辐射)
│   ├── [27:36] Wind speed (风速)
│   ├── [36:45] Tskin (地表温度)
│   └── [45:54] Ttopsoil (土壤温度)
├── [54:162]   历史窗口 CFD 特征 (54 × 2 = 108)
├── [162:165]  CLMS 动态特征 (3)
├── [165:182]  UrbanFeature / rawGeoFeats (17)  ← 相似度从这里的原始数据计算
└── [182:...]  geoFeatures (PCA空间嵌入, 维度取决于配置)
```

---

## 修改顺序建议

1. **先做修改0**（WRF重排序），这是 bug fix，不影响现有逻辑
2. **再做修改1**（创建 physics_loss.py），独立文件，不影响现有代码
3. **然后做修改2**（metadata 添加 UrbanFeature），小改动
4. **最后做修改3和4**（pimodel.py 和 pimodel_training.py），这两个要一起改
5. 运行测试验证维度匹配

---

## 验证检查清单

- [ ] WRF 重排后，无标签数据 index 0-8 是 Tair（可 print 统计值对比有标签数据）
- [ ] `sim_edge_weights_labeled.shape[0]` == `metadata['AdjMatrix']` 中非零元素数量
- [ ] `sim_edge_weights_unlabeled.shape[0]` == `aug1_metadata['AdjMatrix']` 中非零元素数量
- [ ] `phys_loss` 值不是 NaN 或 Inf
- [ ] 训练初期 `phys_loss` 应该比 `labeled_loss` 小 1-2 个数量级（如果太大说明 lambda_phys 需要减小）
- [ ] 总损失收敛，RMSE 不应恶化（如果恶化则减小 lambda_phys）

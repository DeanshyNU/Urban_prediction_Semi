"""
物理一致性损失: 城市地表特征相似的相邻站点，对WRF的温度修正应该接近
L_phys = mean_{(i,j)∈E} [ sim(i,j) * (residual_i - residual_j)² ]
"""
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse


def compute_urban_similarity(urban_features, sigma=0.2):
    """基于全部17维UrbanFeature计算高斯相似度矩阵"""
    diff = urban_features[:, np.newaxis, :] - urban_features[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    return np.exp(-dist_sq / (2 * sigma ** 2))


def compute_similarity_edge_weights(urban_features, adj_matrix, sigma=0.2):
    """预计算每条边的相似度权重，与 dense_to_sparse 的 edge_index 对齐"""
    nNodes_urban = urban_features.shape[0]
    nNodes_adj = adj_matrix.shape[0]
    
    if nNodes_urban != nNodes_adj:
        print(f"警告: UrbanFeature节点数 ({nNodes_urban}) 与 AdjMatrix节点数 ({nNodes_adj}) 不匹配")
        print(f"  使用 AdjMatrix 的节点数 ({nNodes_adj})")
        # 如果 UrbanFeature 节点数更多，截断；如果更少，报错
        if nNodes_urban > nNodes_adj:
            urban_features = urban_features[:nNodes_adj, :]
        else:
            raise ValueError(f"UrbanFeature节点数 ({nNodes_urban}) 小于 AdjMatrix节点数 ({nNodes_adj})")
    
    nNodes = nNodes_adj
    sim_matrix = compute_urban_similarity(urban_features, sigma=sigma)
    edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_matrix))
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    
    # 确保索引在有效范围内
    if src.max() >= nNodes or dst.max() >= nNodes:
        raise ValueError(f"边索引超出范围: src最大={src.max()}, dst最大={dst.max()}, nNodes={nNodes}, "
                        f"sim_matrix形状={sim_matrix.shape}, adj_matrix形状={adj_matrix.shape}")
    
    return torch.FloatTensor(sim_matrix[src, dst])


def physical_consistency_loss(pred, wrf_tair, edge_index, sim_edge_weights, nNodes):
    """
    残差平滑损失
    
    Args:
        pred: (batch_nodes,) 或 (batch_nodes, 1)，模型预测
        wrf_tair: (batch_nodes,)，batch.x[:, 4] 提取的WRF Tair中心格点
        edge_index: (2, batch_edges)，PyG batched后的边
        sim_edge_weights: (num_edges_per_graph,)，单图的相似度权重
        nNodes: 单图节点数
    """
    pred = pred.squeeze()
    residual = pred - wrf_tair
    
    num_graphs = pred.shape[0] // nNodes
    sim_batch = sim_edge_weights.repeat(num_graphs).to(pred.device)
    
    src, dst = edge_index
    residual_diff = residual[src] - residual[dst]
    return (sim_batch * residual_diff ** 2).mean()

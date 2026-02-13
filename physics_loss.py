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
    sim_matrix = compute_urban_similarity(urban_features, sigma=sigma)
    edge_index, _ = dense_to_sparse(torch.FloatTensor(adj_matrix))
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
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

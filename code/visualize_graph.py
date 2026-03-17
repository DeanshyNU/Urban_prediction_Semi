"""
统一图结构可视化
- 可作为独立脚本运行：python code/visualize_graph.py
- 也可被 dataGen_unified 调用：from visualize_graph import visualize_unified_graph
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_unified_graph(unified_adj, unified_locations, n_labeled, save_path):
    """
    可视化统一图结构，保存为 PNG。

    Args:
        unified_adj:       邻接矩阵 (total_nodes, total_nodes)
        unified_locations: 节点坐标 (total_nodes, 2)，[lat, lon]
        n_labeled:         有标签节点数量（前 n_labeled 行/列）
        save_path:         输出图片路径
    """
    total_nodes = unified_adj.shape[0]
    n_unlabeled = total_nodes - n_labeled

    labeled_locs   = unified_locations[:n_labeled]   # (n_labeled, 2)
    unlabeled_locs = unified_locations[n_labeled:]   # (n_unlabeled, 2)

    # 收集三类边（只取上三角避免重复）
    adj = unified_adj.copy()
    np.fill_diagonal(adj, 0)
    ll_edges, uu_edges, lu_edges = [], [], []
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            w = adj[i, j]
            if w <= 0:
                continue
            if i < n_labeled and j < n_labeled:
                ll_edges.append((i, j, w))
            elif i >= n_labeled and j >= n_labeled:
                uu_edges.append((i, j, w))
            else:
                lu_edges.append((i, j, w))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # ---------- 辅助函数：在指定 ax 上画边和节点 ----------
    def draw_edges_and_nodes(ax, show_ll=True, show_uu=True, show_lu=True, alpha=0.3):
        if show_uu:
            for i, j, w in uu_edges:
                ax.plot([unified_locations[i, 1], unified_locations[j, 1]],
                        [unified_locations[i, 0], unified_locations[j, 0]],
                        color='orange', alpha=alpha * 0.6, linewidth=0.4, zorder=1)
        if show_ll:
            for i, j, w in ll_edges:
                ax.plot([unified_locations[i, 1], unified_locations[j, 1]],
                        [unified_locations[i, 0], unified_locations[j, 0]],
                        color='steelblue', alpha=alpha, linewidth=0.8 * w + 0.3, zorder=2)
        if show_lu:
            for i, j, w in lu_edges:
                ax.plot([unified_locations[i, 1], unified_locations[j, 1]],
                        [unified_locations[i, 0], unified_locations[j, 0]],
                        color='green', alpha=alpha, linewidth=0.8 * w + 0.3, zorder=2)
        # 节点：unlabeled 在下，labeled 在上
        ax.scatter(unlabeled_locs[:, 1], unlabeled_locs[:, 0],
                   c='orange', s=18, zorder=3, alpha=0.75, label=f'Unlabeled ({n_unlabeled})')
        ax.scatter(labeled_locs[:, 1], labeled_locs[:, 0],
                   c='red', s=60, zorder=4, marker='^', label=f'Labeled ({n_labeled})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    # ---------- 子图1：全图总览 ----------
    ax = axes[0]
    draw_edges_and_nodes(ax, show_ll=True, show_uu=True, show_lu=True, alpha=0.25)
    ax.set_title(f'全图总览\nL-L={len(ll_edges)}  U-U={len(uu_edges)}  L-U={len(lu_edges)}')
    ax.legend(fontsize=8)

    # ---------- 子图2：Labeled节点视角（去掉U-U噪声） ----------
    ax = axes[1]
    draw_edges_and_nodes(ax, show_ll=True, show_uu=False, show_lu=True, alpha=0.45)
    ax.set_title(f'Labeled节点视角（L-L + L-U）\n'
                 f'每labeled节点平均 {np.sum(adj[:n_labeled, n_labeled:]>0, axis=1).mean():.1f} 个unlabeled邻居')
    patches = [
        mpatches.Patch(color='steelblue', label=f'L-L ({len(ll_edges)})'),
        mpatches.Patch(color='green',     label=f'L-U ({len(lu_edges)})'),
    ]
    ax.legend(handles=patches, fontsize=8)

    # ---------- 子图3：边权重分布直方图 ----------
    ax = axes[2]
    if ll_edges:
        ax.hist([e[2] for e in ll_edges], bins=30, alpha=0.6,
                label=f'L-L ({len(ll_edges)})', color='steelblue')
    if uu_edges:
        ax.hist([e[2] for e in uu_edges], bins=30, alpha=0.5,
                label=f'U-U ({len(uu_edges)})', color='orange')
    if lu_edges:
        ax.hist([e[2] for e in lu_edges], bins=30, alpha=0.6,
                label=f'L-U ({len(lu_edges)})', color='green')
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Count')
    ax.set_title('边权重分布\n（L-U现在基于时序相关性）')
    ax.legend(fontsize=8)

    # ---------- 打印节点度数信息 ----------
    degrees = np.sum(adj > 0, axis=1)
    lu_per_labeled = np.sum(adj[:n_labeled, n_labeled:] > 0, axis=1)
    print(f"  [可视化] Labeled度数: min={degrees[:n_labeled].min()} "
          f"max={degrees[:n_labeled].max()} mean={degrees[:n_labeled].mean():.1f}")
    print(f"  [可视化] Unlabeled度数: min={degrees[n_labeled:].min()} "
          f"max={degrees[n_labeled:].max()} mean={degrees[n_labeled:].mean():.1f}")
    print(f"  [可视化] 每labeled节点的unlabeled邻居数: "
          f"min={lu_per_labeled.min()} max={lu_per_labeled.max()} mean={lu_per_labeled.mean():.1f} "
          f"| 无unlabeled邻居的labeled节点={np.sum(lu_per_labeled==0)}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [可视化] 图结构已保存: {save_path}")


# ============================================================
# 独立运行入口（python code/visualize_graph.py）
# ============================================================
if __name__ == '__main__':
    import sys
    import mat73
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    from datalib import preprocess_unlabeled_data
    from datalib.generation import build_unified_graph

    print("加载无标签数据...")
    unlabeled_data = preprocess_unlabeled_data(
        unlabeled_file=os.path.join(DATA_PATH, 'Unlabeled_Finalized.mat'),
        target_station_count=200,
        nTimesteps=6624
    )
    dataParam = {'thres': 0.1, 'geoMethod': 'average', 'nCompPCA': 40, 'poolSize': 12}

    print("构建统一图...")
    unified_adj, labeled_indices, unlabeled_indices, unified_locations = build_unified_graph(
        dataParam, DATA_PATH, unlabeled_data, n_unlabeled=200
    )
    n_labeled = len(labeled_indices)

    save_path = os.path.join(OUTPUT_PATH, 'unified_graph.png')
    visualize_unified_graph(unified_adj, unified_locations, n_labeled, save_path)

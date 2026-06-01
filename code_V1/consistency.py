"""
Multi-view consistency regularization for V1 GNN baseline (GRAND-style for regression).

设计:
  - 每 batch 跑 K_views 个不同图增强后的 forward
  - 增强 = DropEdge + DropNode + Edge weight noise(三种正交扰动组合)
  - 三种 view 不同强度(curriculum-like:弱 / 中 / 强),增大 view diversity
  - Sup loss 在 view 0(最弱增强)上算
  - Consistency loss:所有 K view 预测对齐到 mean(detached)
  - λ_cons 控制 consistency 强度

不动 EAD / Lap / Self-Train,完全独立 dispatch(method_full 加 _cons 后缀)。

==============================================================
🔑 LEAK 规则
==============================================================
  - DropNode 不丢 train_mask=True 节点(它们要算 sup loss,信息不能丢)
  - 增强只动图结构 / node features 置 0,**不变 target T**
  - Consistency loss 在所有节点上算(包括 valid),valid 受惠的是"模型在不同视角下预测一致"这种鲁棒性,**不接触 valid target**
"""

import torch
import numpy as np
import utils

_CONS_DEBUG_PRINTED = {'train': False}


def graph_augment(x, edge_index, edge_attr, view_id, train_mask, generator=None):
    """对 (x, edge_index, edge_attr) 做三种正交增强:
      1. DropEdge(按 p_edge 按概率丢边)
      2. DropNode(随机置 0 部分 unlabeled / valid 节点 features,**不丢 train**)
      3. Edge weight Gaussian noise(σ 加性噪声)

    三种 view 不同强度:[弱, 中, 强]。

    Args:
        x:           (N_total, iDim) features (batched)
        edge_index:  (2, E) edges (batched)
        edge_attr:   (E,) edge weights (batched)
        view_id:     用第 view_id % 3 套强度参数
        train_mask:  (N_total,) bool,train_mask=True 的节点不被 DropNode

    Returns:
        x_aug, edge_aug, attr_aug
    """
    # 增强强度档位(env V1_CONS_AUG_SCALE 可整体缩放)
    import os as _os
    aug_scale = float(_os.environ.get('V1_CONS_AUG_SCALE', '1.0'))   # 1.0=默认,>1 更激进
    p_edge_list  = [0.20 * aug_scale, 0.35 * aug_scale, 0.50 * aug_scale]
    p_node_list  = [0.05 * aug_scale, 0.10 * aug_scale, 0.15 * aug_scale]
    edge_noise_list = [0.05 * aug_scale, 0.10 * aug_scale, 0.15 * aug_scale]
    # Clamp p ≤ 0.9 防止全部丢光
    p_edge_list = [min(p, 0.9) for p in p_edge_list]
    p_node_list = [min(p, 0.9) for p in p_node_list]
    cfg = view_id % 3
    p_edge = p_edge_list[cfg]
    p_node = p_node_list[cfg]
    edge_noise_std = edge_noise_list[cfg]

    device = x.device
    n_total = x.shape[0]

    # 1. DropNode (不丢 train)
    rand_node = torch.rand(n_total, device=device)
    drop_node_mask = (rand_node < p_node) & ~train_mask
    x_aug = x.clone()
    x_aug[drop_node_mask] = 0.0

    # 2. DropEdge
    rand_edge = torch.rand(edge_index.shape[1], device=device)
    keep_edge = rand_edge >= p_edge
    edge_aug = edge_index[:, keep_edge]
    attr_aug = edge_attr[keep_edge] if edge_attr is not None else None

    # 3. Edge weight noise (clamp ≥ 0)
    if attr_aug is not None and edge_noise_std > 0:
        attr_aug = attr_aug + edge_noise_std * torch.randn_like(attr_aug)
        attr_aug = attr_aug.clamp(min=0.0)

    return x_aug, edge_aug, attr_aug, dict(p_edge=p_edge, p_node=p_node,
                                             edge_noise_std=edge_noise_std,
                                             n_dropped_nodes=int(drop_node_mask.sum().item()),
                                             n_dropped_edges=int((~keep_edge).sum().item()))


def train_consistency(loader, model, lossFn, opt, scheduler, device, n_total,
                      K_views=3, lambda_cons=0.1, sup_view_id=0):
    """One epoch:K-view consistency training.

    Loss = sup_loss(view sup_view_id 上算)+ lambda_cons × consistency_loss(K views align)

    Returns: (avg_total_loss, RMSE_arr, sup_loss_avg, cons_loss_avg)
    """
    global _CONS_DEBUG_PRINTED
    model.train()
    _LOSS = 0; _SUP = 0; _CONS = 0
    pred, truth = [], []

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        # 用 train_mask 防 DropNode 丢监督节点
        if hasattr(_batch, 'train_mask'):
            tm = _batch.train_mask
        elif hasattr(_batch, 'label_mask'):
            tm = _batch.label_mask
        else:
            tm = torch.zeros(_batch.x.shape[0], dtype=torch.bool, device=device)

        # K views forward
        yhat_views = []
        aug_stats = []
        for k in range(K_views):
            x_aug, edge_aug, attr_aug, stats = graph_augment(
                _batch.x, _batch.edge_index, _batch.edge_attr, k, tm)
            yhat_k = model(x_aug, edge_aug, attr_aug)
            yhat_views.append(yhat_k)
            aug_stats.append(stats)

        yhat_stack = torch.stack(yhat_views, dim=0)   # (K, N*bs, oDim)

        # Sup loss: 用 sup_view_id 那一 view 在 label_mask 节点上算
        yhat_sup = yhat_views[sup_view_id]
        if hasattr(_batch, 'label_mask') and _batch.label_mask is not None:
            mask = _batch.label_mask
            sup_loss = lossFn(yhat_sup[mask], _batch.y[mask])
            n_lbl_per_g = mask.sum().item() // max(_batch.x.shape[0] // n_total, 1)
            pt = yhat_sup[mask].reshape(-1, n_lbl_per_g).cpu().detach().numpy()
            tt = _batch.y[mask].reshape(-1, n_lbl_per_g).cpu().detach().numpy()
        else:
            sup_loss = lossFn(yhat_sup, _batch.y)
            bs = _batch.x.shape[0] // n_total
            pt = yhat_sup.reshape(bs, n_total).cpu().detach().numpy()
            tt = _batch.y.reshape(bs, n_total).cpu().detach().numpy()

        # Consistency loss: 所有 view 与 mean 对齐(mean detached → grad 不流过 mean)
        yhat_mean = yhat_stack.mean(dim=0).detach()
        cons_loss = ((yhat_stack - yhat_mean.unsqueeze(0)) ** 2).mean()

        total = sup_loss + lambda_cons * cons_loss
        total.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total
        _SUP  += sup_loss
        _CONS += cons_loss
        pred += list(pt)
        truth += list(tt)

        # Debug: first batch, first epoch
        if _n == 0 and not _CONS_DEBUG_PRINTED['train']:
            print(f"[DEBUG/train_cons] K_views={K_views}, lambda_cons={lambda_cons}")
            for k, st in enumerate(aug_stats):
                print(f"[DEBUG/train_cons] view {k}: p_edge={st['p_edge']:.2f}, "
                      f"p_node={st['p_node']:.2f}, edge_noise_std={st['edge_noise_std']:.2f}, "
                      f"dropped {st['n_dropped_nodes']} nodes / {st['n_dropped_edges']} edges")
            print(f"[DEBUG/train_cons] sup_loss={sup_loss.item():.4e}, "
                  f"cons_loss={cons_loss.item():.4e}, λ={lambda_cons}, total={total.item():.4e}")
            print(f"[DEBUG/train_cons] yhat_views std (across K): "
                  f"min={yhat_stack.std(0).min().item():.4e}, "
                  f"mean={yhat_stack.std(0).mean().item():.4e}, "
                  f"max={yhat_stack.std(0).max().item():.4e}")
            assert torch.isfinite(yhat_stack).all(), "[ERR/cons] yhat NaN/Inf"
            assert torch.isfinite(total), f"[ERR/cons] total loss NaN/Inf: {total}"
            _CONS_DEBUG_PRINTED['train'] = True

    scheduler.step()
    truth = np.array(truth); pred = np.array(pred)
    RMSE = utils.RMSE(truth, pred)
    n = _n + 1
    # 用 module attr 存 sup / cons 分量(同 train_lap 风格,run.py 主循环不需读)
    train_consistency.last_sup = (_SUP / n).item()
    train_consistency.last_cons = (_CONS / n).item()
    train_consistency.last_lambda = lambda_cons
    return (_LOSS / n).item(), RMSE, truth, pred

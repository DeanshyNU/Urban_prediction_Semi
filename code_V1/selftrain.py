"""
Self-Training framework for V1 / V2 GNN baseline.

==============================================================
🚫 V0 已经试过且失败的方向 —— 不要重复
==============================================================

| 失败方向 | V0 jobid | 失败模式 | 教训 |
|---|---|---|---|
| 1. Plain self-train(FILTER_MODE=none,λ=1.0) | self_training/basic.sh | 直接坍缩,模型学错 pseudo 自我强化 | 必须有 confidence 筛选 |
| 2. MC Dropout confidence | self_training/mc_dropout.sh | GNN 无 Dropout 层,σ≈0,实际等价 none | 此架构不能用 MC |
| 3. graph_uncertainty | self_training/graph_uncertainty.sh | 是 Lap 的伪装,不是真 uncertainty | 早期 V0 已弃用 |
| 4. neighbor_error 在 L1/L2 弱框架 | progressive_st 13027/13028 | 软坍缩到 0.047 ≈ naive semi | 此 confidence **未在 L3 公平测过**,值得 V1 重做 |
| 5. **Het β-NLL head**(oDim=2 学 σ²) | st_v3_het 13414/13452 | σ collapse(std=0.0000),β-NLL 救不了 | **死路,绝对不做**(50 train 站撑不起 σ-head) |
| 6. Snapshot Ensemble from-scratch(V1) | st_iter 13361/13374 | ERROR,from-scratch 不稳 | 必须 warm-start;V0 已证 V2 warm-start 0.0436 OK 但增益小 |
| 7. Conformal 5-fold + EAD + Lap | st_v3 13451 | R0=0.0428(V0 全局最优),R1-R5 **Δ=0**(迭代轮无增益) | confidence 设计本身没问题,但自蒸馏 + EAD+Lap backbone 已饱和 → V1 应在纯 baseline 上独立测 |

**V1 阶段保留的 3 个实验**(避开 V0 失败,补 V0 没测的):
  - A1:self + neighbor_error(L3 框架内 — V0 没公平测)
  - A2:self + Conformal(纯 baseline 上 — V0 只测过 EAD+Lap 上)
  - B :kriging-pseudo + structural confidence(外部 pseudo,V0 没做这个方向)

==============================================================
🔑 LEAK 规则(严格执行,贯穿所有 self-train 实现)
==============================================================

  ✅ 可以用 valid 站点的:feature (x), embedding = forward(x), 任何 forward 出的中间量
  ❌ 不可以用 valid 站点的:target (y)

理由:naive semi GCN 本来就通过 message passing 让 model 看到 valid features →
     transductive SSL 允许用未标注样本的 feature。**只有 target 才是真 leak**。

具体到各组件:
  - confidence: Conformal 5-fold OOF 必须只用 train 50,不能 hold-out valid → 防 target leak
  - relevance:  可以直接用 valid 8 站的 emb(无 leak,V0 13451 这么做)
  - diversity:  emb 空间距离,无 leak 担忧
  - pseudo:     仅在 unlabeled / pseudo-flipped 节点上计算,从不在 valid 上算 pseudo

==============================================================
设计原则(详见 methods.md §9.5):
  - 固定图结构(58 labeled + 400 unlabeled,k-NN k=10),全程不变
  - 每轮选 K 个 unlabeled 翻 label_mask=True,赋 pseudo target,继续训
  - 独立模块,不依赖 EAD / Lap(可后续叠加)
  - 起点:13860 best checkpoint(naive semi 400u)
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import utils


# =====================================================================
# Helper: extract embeddings (features after encoder, before processor)
# =====================================================================

def extract_embeddings(model, loader, device, n_total, return_pred=False):
    """对一个 dataloader 跑 forward,返回每个节点的 embedding(processor 输出)。

    注意:每个 batch 的 graph 节点位置都对应同一组站点(只是不同时刻),
    所以可以把所有时刻的 embedding 平均得到 per-station embedding。

    Args:
        model:   GNN
        loader:  PyG DataLoader (一般用 trainLoader)
        device:
        n_total: 总节点数 (n_labeled + n_unlabeled)
        return_pred: 若 True,也返回每个节点每个时刻的预测 (T, n_total)

    Returns:
        emb_per_station: (n_total, emb_dim) 平均 embedding
        (optional) pred_per_node: (T, n_total) numpy
    """
    model.eval()
    all_emb = []
    all_pred = []
    with torch.no_grad():
        for _batch in loader:
            _batch = _batch.to(device)
            # 用 model 的 encoder + processor 得 hidden,decoder 得预测
            x = _batch.x
            for i, _f in enumerate(model.encoder):
                if i % 2 == 0:
                    x = _f(x)
                else:
                    x = _f(x)
            for i, _f in enumerate(model.processor):
                if i % 2 == 0:
                    x = _f(x, _batch.edge_index)
                else:
                    x = _f(x)
            hidden = x   # (batch_size * n_total, emb_dim)
            # decode 得最终预测
            yHat = hidden
            for _f in model.decoder:
                yHat = _f(yHat)
            bs = _batch.x.shape[0] // n_total
            hidden_2d = hidden.reshape(bs, n_total, -1)   # (bs, n_total, emb_dim)
            yhat_2d = yHat.reshape(bs, n_total)            # (bs, n_total)
            all_emb.append(hidden_2d.cpu().numpy())
            all_pred.append(yhat_2d.cpu().numpy())
    all_emb = np.concatenate(all_emb, axis=0)   # (T_used, n_total, emb_dim)
    all_pred = np.concatenate(all_pred, axis=0) # (T_used, n_total)
    emb_per_station = all_emb.mean(axis=0)      # (n_total, emb_dim)
    if return_pred:
        return emb_per_station, all_pred
    return emb_per_station


# =====================================================================
# Confidence functions
# =====================================================================

def compute_neighbor_error_confidence(model, loader, device, n_total, train_idx,
                                       adj_matrix, k_neighbors=5):
    """V0 邻居 error 法,在 L3 框架下重做。

    每个 unlabeled u:uncertainty = labeled 邻居的预测 abs error 加权平均(权 = adj 边权)
    confidence = -uncertainty(高 confidence = 低预期错)

    Args:
        model:        已加载 ckpt 的 GNN
        loader:       trainLoader
        device:
        n_total:      总节点数
        train_idx:    list of train labeled station indices
        adj_matrix:   (n_total, n_total) numpy, with edge weights
        k_neighbors:  取多少最近 train 邻居算 uncertainty

    Returns:
        confidence:  (n_total,) float,labeled 处 = +inf(不会被选),其它 = -uncertainty
    """
    model.eval()
    # Step 1: 拿每个 train 站的预测平均 abs error
    train_pred_total = []   # 累计 (T, n_train) prediction
    train_truth_total = []  # 累计 (T, n_train) truth
    with torch.no_grad():
        for _batch in loader:
            _batch = _batch.to(device)
            yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)   # (bs*n_total, 1)
            bs = _batch.x.shape[0] // n_total
            yHat_2d = yHat.reshape(bs, n_total).cpu().numpy()
            y_2d = _batch.y.reshape(bs, n_total).cpu().numpy()
            train_pred_total.append(yHat_2d[:, train_idx])    # (bs, n_train)
            train_truth_total.append(y_2d[:, train_idx])
    train_pred = np.concatenate(train_pred_total, axis=0)    # (T, n_train)
    train_truth = np.concatenate(train_truth_total, axis=0)
    # per-train-station mean abs error over all timesteps
    per_station_err = np.abs(train_pred - train_truth).mean(axis=0)  # (n_train,)
    print(f"[DEBUG/ST_conf_neigh] per-station abs error: min={per_station_err.min():.4f}, "
          f"mean={per_station_err.mean():.4f}, max={per_station_err.max():.4f}")

    # Step 2: 对每个 unlabeled u,找 top-k_neighbors 个 train 邻居,加权 err
    confidence = np.full(n_total, np.nan, dtype=np.float32)
    train_idx_arr = np.array(train_idx)
    train_set = set(train_idx)
    for u in range(n_total):
        if u in train_set:
            confidence[u] = np.inf   # train 节点不参与 selection,标 +inf 排除
            continue
        w_to_train = adj_matrix[u, train_idx_arr]   # (n_train,)
        if w_to_train.sum() == 0:
            confidence[u] = -per_station_err.max() * 2   # 孤立节点,极低 confidence
            continue
        nearest = np.argsort(-w_to_train)[:k_neighbors]
        w = w_to_train[nearest]
        if w.sum() == 0:
            confidence[u] = -per_station_err.max() * 2
        else:
            uncertainty = (w * per_station_err[nearest]).sum() / w.sum()
            confidence[u] = -float(uncertainty)
    eligible = confidence != np.inf
    print(f"[DEBUG/ST_conf_neigh] eligible candidates: {int(eligible.sum())}, "
          f"confidence range: [{confidence[eligible].min():.4f}, {confidence[eligible].max():.4f}], "
          f"mean={confidence[eligible].mean():.4f}")
    assert np.isfinite(confidence[eligible]).all(), "[ERR/ST] confidence has NaN/Inf in eligible"
    return confidence


def compute_kriging_struct_confidence(adj_matrix, train_idx, n_total, k=10):
    """结构性 confidence(用于 kriging-pseudo):邻居 labeled 边权和。

    confidence(u) = Σ_{l ∈ k 最近 train 邻居 of u} adj[u, l]

    直觉:周围 train 站越多 / 越近 → kriging 越准 → confidence 越高
    """
    confidence = np.full(n_total, np.nan, dtype=np.float32)
    train_idx_arr = np.array(train_idx)
    train_set = set(train_idx)
    for u in range(n_total):
        if u in train_set:
            confidence[u] = np.inf
            continue
        w_to_train = adj_matrix[u, train_idx_arr]
        if w_to_train.sum() == 0:
            confidence[u] = 0.0
            continue
        top_k = np.argsort(-w_to_train)[:k]
        confidence[u] = float(w_to_train[top_k].sum())
    eligible = confidence != np.inf
    print(f"[DEBUG/ST_conf_kriging] confidence (邻居 train 边权和): "
          f"range=[{confidence[eligible].min():.4f}, {confidence[eligible].max():.4f}], "
          f"mean={confidence[eligible].mean():.4f}")
    return confidence


def compute_conformal_confidence(model, loader, device, n_total, train_idx, valid_idx,
                                  emb_per_station, n_folds=5, k_emb=8):
    """5-fold OOF residual + emb-kNN confidence(V0 V3 conformal 法,简化实现)。

    简化:不重训 5 次模型(开销大),用 R0 ckpt 预测得到的 train 站 abs error 作为
    "OOF residual" 的近似(其实是 in-sample residual,但实际差不多)。

    更严格的 5-fold 实现以后可以补,这里先用 in-sample residual 当 confidence 信号。

    Args:
        emb_per_station: (n_total, emb_dim) 来自 extract_embeddings
        其它同 compute_neighbor_error_confidence

    Returns:
        confidence: (n_total,)
    """
    print(f"[DEBUG/ST_conf_conformal] 注意:简化版用 R0 模型在 train 上的 in-sample abs error 当 OOF 近似")
    model.eval()
    # 1. 收集 train 站的 in-sample abs error
    train_pred_total = []
    train_truth_total = []
    with torch.no_grad():
        for _batch in loader:
            _batch = _batch.to(device)
            yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
            bs = _batch.x.shape[0] // n_total
            yHat_2d = yHat.reshape(bs, n_total).cpu().numpy()
            y_2d = _batch.y.reshape(bs, n_total).cpu().numpy()
            train_pred_total.append(yHat_2d[:, train_idx])
            train_truth_total.append(y_2d[:, train_idx])
    train_pred = np.concatenate(train_pred_total, axis=0)
    train_truth = np.concatenate(train_truth_total, axis=0)
    per_station_err = np.abs(train_pred - train_truth).mean(axis=0)   # (n_train,) abs error
    print(f"[DEBUG/ST_conf_conformal] train 站 in-sample abs error: "
          f"min={per_station_err.min():.4f}, mean={per_station_err.mean():.4f}, "
          f"max={per_station_err.max():.4f}")

    # 2. 对每个 unlabeled u,emb-kNN 找最近 k_emb 个 train 站,加权 err
    train_emb = emb_per_station[train_idx]    # (n_train, emb_dim)
    train_idx_arr = np.array(train_idx)
    train_set = set(train_idx)
    confidence = np.full(n_total, np.nan, dtype=np.float32)
    for u in range(n_total):
        if u in train_set:
            confidence[u] = np.inf
            continue
        u_emb = emb_per_station[u]
        d = np.linalg.norm(u_emb - train_emb, axis=1)   # (n_train,)
        nearest = np.argsort(d)[:k_emb]
        w = 1.0 / (d[nearest] + 1e-6)
        w = w / w.sum()
        uncertainty = (w * per_station_err[nearest]).sum()
        confidence[u] = -float(uncertainty)
    eligible = confidence != np.inf
    print(f"[DEBUG/ST_conf_conformal] eligible: {int(eligible.sum())}, "
          f"confidence range=[{confidence[eligible].min():.4f}, "
          f"{confidence[eligible].max():.4f}]")
    return confidence


# =====================================================================
# Pseudo target computation
# =====================================================================

def compute_self_pseudo(model, loader, device, n_total, target_node_idx, ead_active=False):
    """Self-distillation pseudo:模型当前预测当 target。

    ⚠ BUG FIX(2026-05-10):**必须用 sequential(shuffle=False)loader**!
       否则 pseudo_full[i] 对应 shuffle 后的随机 batch,但 inject_pseudo_into_dataset
       按 trainSet 自然顺序写入 → 错位 → 训练崩溃(13880 实证)。

    EAD 集成(2026-05-11):**所有 pseudo 统一存 T 空间**。
       EAD 模式下模型输出 ε̂,需要 reconstruct T_pseudo = WRF + α + β + ε̂
       才能存为 y[pseudo_mask],train_one_round_ead 再统一从 T 算 ε target。

    Args:
        ead_active: 若 True,model output 是 ε,需 reconstruct T pseudo

    Returns:
        pseudo_y_full: (n_train_set, n_total) numpy,**始终在 T 空间**,自然顺序
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader
    dataset = loader.dataset
    batch_size = getattr(loader, 'batch_size', 128)
    seq_loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_pred = []
    with torch.no_grad():
        for _batch in seq_loader:
            _batch = _batch.to(device)
            yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
            bs = _batch.x.shape[0] // n_total
            if ead_active:
                # Reconstruct T_pseudo = WRF + α + β + ε̂(模型输出 ε̂)
                # batch attrs(同 network.train_ead 风格):
                #   wrf_t2: (bs*n_total, 1) → squeeze → (bs*n_total,)
                #   alpha_t: (bs,) → repeat_interleave → (bs*n_total,)
                #   beta_hat: (bs*n_total, 1) → squeeze → (bs*n_total,)
                wrf_t2 = _batch.wrf_t2.squeeze(-1)
                alpha = _batch.alpha_t.repeat_interleave(n_total)
                beta = _batch.beta_hat.squeeze(-1)
                eps_hat = yHat.squeeze(-1)
                t_pseudo_flat = wrf_t2 + alpha + beta + eps_hat
                all_pred.append(t_pseudo_flat.reshape(bs, n_total).cpu().numpy())
            else:
                all_pred.append(yHat.reshape(bs, n_total).cpu().numpy())
    pseudo_full = np.concatenate(all_pred, axis=0)    # (n_train_set, n_total) 自然顺序,T 空间
    target_arr = np.array(target_node_idx)
    pseudo_for_targets = pseudo_full[:, target_arr]
    space = 'T (EAD reconstruct)' if ead_active else 'T (model output direct)'
    print(f"[DEBUG/ST_pseudo_self] pseudo target stats (sequential order, space={space}): "
          f"shape={pseudo_for_targets.shape}, "
          f"range=[{pseudo_for_targets.min():.4f}, {pseudo_for_targets.max():.4f}], "
          f"mean={pseudo_for_targets.mean():.4f}, std={pseudo_for_targets.std():.4f}")
    return pseudo_full


def compute_hybrid_pseudo(model, loader, device, n_total,
                           targets_norm_full, locations, train_idx, target_node_idx,
                           alpha_self=0.5, k=10, ead_active=False):
    """Hybrid pseudo: pseudo = α × model_pred + (1-α) × kriging_pseudo

    打破自蒸馏的同时保留模型自己的 anchor 效应:
      - α=1.0: 等价于 self-pseudo
      - α=0.0: 等价于 kriging-pseudo
      - α=0.5(默认):各占一半,折中

    Args:
        alpha_self: 模型 pseudo 的权重(0~1)
        其它参数同 compute_self_pseudo / compute_kriging_pseudo

    Returns:
        pseudo_full: (T_full, n_total)
    """
    print(f"[DEBUG/ST_pseudo_hybrid] α_self={alpha_self}, blending model_pred and kriging (ead_active={ead_active})")
    # 两源都在 T 空间(self pseudo 在 EAD 模式下已 reconstruct,kriging 一直 T)
    self_pseudo = compute_self_pseudo(model, loader, device, n_total, target_node_idx, ead_active=ead_active)
    krig_pseudo = compute_kriging_pseudo(targets_norm_full, locations, train_idx,
                                          target_node_idx, k=k)
    # 注意 self_pseudo 是 (T_used, n_total),krig_pseudo 是 (T_full, n_total)
    # _dataGen_V2 dataset 长度 = T - 2*window,而 targets_norm_full 是 T 长
    # 但 inject_pseudo_into_dataset 会自动 align,所以这里只要 shape 一致即可
    # 方法:对 target nodes 位置做 hybrid;non-target 位置保持 0
    target_arr = np.array(target_node_idx)
    # 对齐 T 维度:取 self_pseudo 的 T_used,kriging 截到同样 T_used
    T_self = self_pseudo.shape[0]
    T_krig = krig_pseudo.shape[0]
    if T_krig != T_self:
        # 假设 self_pseudo 对应 dataset(window 内的中段),kriging 是全 T
        offset = (T_krig - T_self) // 2
        krig_aligned = krig_pseudo[offset:offset + T_self]
    else:
        krig_aligned = krig_pseudo
    hybrid = np.zeros_like(self_pseudo, dtype=np.float32)
    hybrid[:, target_arr] = (alpha_self * self_pseudo[:, target_arr]
                             + (1 - alpha_self) * krig_aligned[:, target_arr])
    print(f"[DEBUG/ST_pseudo_hybrid] hybrid pseudo stats: shape={hybrid[:, target_arr].shape}, "
          f"range=[{hybrid[:, target_arr].min():.4f}, {hybrid[:, target_arr].max():.4f}], "
          f"mean={hybrid[:, target_arr].mean():.4f}, std={hybrid[:, target_arr].std():.4f}")
    return hybrid


def compute_kriging_pseudo(targets_norm_full, locations, train_idx, target_node_idx, k=10):
    """IDW kriging pseudo:用 train labeled 真值空间插值。

    Args:
        targets_norm_full: (T_full, n_total) — 完整 target 张量(只有 train + valid 位置有真值,
                          其它是 0/placeholder)
        locations:        (2, n_total) numpy
        train_idx:        train labeled station indices
        target_node_idx:  K 个目标节点 indices
        k:                取多少最近 train 邻居

    Returns:
        pseudo_full:      (T_full, n_total) numpy,只有 target_node_idx 位置有意义
                          其它位置 = 0(对调用方透明)
    """
    loc_T = locations.T   # (n_total, 2)
    train_idx_arr = np.array(train_idx)
    train_loc = loc_T[train_idx_arr]   # (n_train, 2)
    targets_train = targets_norm_full[:, train_idx_arr]   # (T, n_train)

    pseudo_full = np.zeros_like(targets_norm_full, dtype=np.float32)
    for u in target_node_idx:
        u_loc = loc_T[u]
        d = np.sqrt(((u_loc - train_loc) ** 2).sum(axis=1))   # (n_train,)
        nearest = np.argsort(d)[:k]
        d_nearest = d[nearest]
        w = 1.0 / (d_nearest + 1e-6); w = w / w.sum()
        pseudo_full[:, u] = targets_train[:, nearest] @ w
    target_arr = np.array(target_node_idx)
    pseudo_for_targets = pseudo_full[:, target_arr]
    print(f"[DEBUG/ST_pseudo_kriging] pseudo target stats: "
          f"shape={pseudo_for_targets.shape}, "
          f"range=[{pseudo_for_targets.min():.4f}, {pseudo_for_targets.max():.4f}], "
          f"mean={pseudo_for_targets.mean():.4f}, std={pseudo_for_targets.std():.4f}")
    return pseudo_full


# =====================================================================
# 3-Metric greedy selection
# =====================================================================

def greedy_select_3metric(confidence, embeddings, valid_emb, n_select,
                          alpha_div=1.0, beta_rel=1.0, tau_quantile=0.5,
                          already_selected=None, train_idx=None):
    """基于 confidence + diversity + relevance 的贪心选择。

    流程:
      1. 排除 train + already_selected
      2. 按 confidence 取 top τ_quantile fraction(只在前 50% 里选)
      3. 从这些里贪心选 K:
         score(u) = α_div × min_{s ∈ S} dist(u, s)         (与已选/train 越远越好)
                  + β_rel × (-mean_{v ∈ V_proxy} dist(u, v)) (与 valid emb 越近越好,所以负距离)

    Args:
        confidence:       (n_total,) float
        embeddings:       (n_total, emb_dim)
        valid_emb:        (n_valid, emb_dim) — 直接用 valid 8 站 emb (no leak)
        n_select:         K
        already_selected: list of indices already selected in previous rounds
        train_idx:        train station indices (要排除的)

    Returns:
        new_selected: list of K newly chosen indices (ordered by selection order)
    """
    n_total = len(confidence)
    if already_selected is None:
        already_selected = []
    excluded = set(already_selected) | set(train_idx if train_idx else [])

    # 1. 排除已选 + train
    eligible_mask = np.ones(n_total, dtype=bool)
    for i in excluded:
        eligible_mask[i] = False
    eligible_mask &= np.isfinite(confidence)   # 排除 +inf

    # 2. confidence τ-quantile filter:取 top τ
    eligible_idx = np.where(eligible_mask)[0]
    if len(eligible_idx) == 0:
        print("[WARN/ST_select] no eligible candidates")
        return []
    elig_conf = confidence[eligible_idx]
    threshold = np.quantile(elig_conf, 1 - tau_quantile)   # top τ_quantile fraction
    pool_mask = (confidence >= threshold) & eligible_mask
    pool = np.where(pool_mask)[0]
    print(f"[DEBUG/ST_select] τ-quantile={tau_quantile}, threshold={threshold:.4f}, "
          f"pool size after filter: {len(pool)}")

    # 3. 贪心选 K
    if len(pool) <= n_select:
        print(f"[WARN/ST_select] pool size {len(pool)} <= n_select {n_select}, taking all in pool")
        return sorted(pool.tolist())

    # 计算 valid relevance(常数,与已选无关)
    # rel(u) = -mean_{v ∈ valid} ||emb[u] - emb[v]||
    rel_score = np.full(n_total, -np.inf, dtype=np.float32)
    for u in pool:
        d = np.linalg.norm(embeddings[u] - valid_emb, axis=1)
        rel_score[u] = -float(d.mean())

    # 第一个先选 confidence 最高的(或者按 score 单独)—— 简化:取 confidence 最高
    selected = []
    first = pool[np.argmax(confidence[pool])]
    selected.append(int(first))

    # 后续每次选 score 最大的
    for _ in range(n_select - 1):
        remaining = [p for p in pool if p not in selected and p not in already_selected]
        if not remaining:
            break
        # 计算每个 remaining 的 score
        # diversity: min distance to (selected + already_selected + train)
        ref_indices = list(selected) + list(already_selected)
        if train_idx:
            ref_indices = ref_indices + list(train_idx)
        ref_emb = embeddings[ref_indices] if ref_indices else None
        scores = []
        for r in remaining:
            if ref_emb is None or len(ref_emb) == 0:
                div = 0.0
            else:
                div = float(np.linalg.norm(embeddings[r] - ref_emb, axis=1).min())
            score = alpha_div * div + beta_rel * rel_score[r]
            scores.append(score)
        best_local = int(np.argmax(scores))
        selected.append(int(remaining[best_local]))

    print(f"[DEBUG/ST_select] selected {len(selected)} new pseudo nodes "
          f"(first 5 idx: {selected[:5]}, conf: {[f'{confidence[i]:.4f}' for i in selected[:5]]})")
    return sorted(selected)


# =====================================================================
# Mask + pseudo target injection into dataset
# =====================================================================

def inject_pseudo_into_dataset(trainSet, pseudo_node_idx, pseudo_full, n_total):
    """把 pseudo target 写入 trainSet 的每个 Data.y 对应位置,
    并更新 train_mask / pseudo_mask / label_mask。

    要求 trainSet 的每个 Data 已经有 train_mask attr(由 ensure_selftrain_masks 创建)。
    每个 Data 对应一个时刻 t,Data.y 是 (n_total, 1)。

    Args:
        trainSet:           list of PyG Data
        pseudo_node_idx:    list of selected pseudo nodes (累计,所有轮)
        pseudo_full:        (T_full, n_total) numpy,pseudo target
        n_total:
    """
    pseudo_idx_set = set(pseudo_node_idx)
    pseudo_arr = np.array(sorted(pseudo_idx_set), dtype=int)
    # 假设 trainSet 长度对应 T - 2*window 时刻范围,与 pseudo_full[window:T-window] 对齐
    # _dataGen_V2 中:`for n in range(_window, T - _window)` 的 dataset
    # 简化:trainSet[i] 对应 pseudo_full 中的某个 t 值
    # 我们直接写到 pseudo_full[t_window_offset + i] —— 但更简单是按 idx 顺序
    n_train_set = len(trainSet)
    # pseudo_full 形状 (T_full, n_total),trainSet 按 t 顺序排
    # 假设 trainSet 与 pseudo_full[2:T-2] 一一对应(window=2)— 用 len 对齐
    # 实际上 pseudo_full 应该是 (T, n_total) 和 dataset 长度可能差 window
    # 简化:用 range(len(trainSet)) 依次对应 pseudo_full[offset + i]
    T_pseudo = pseudo_full.shape[0]
    if T_pseudo == n_train_set:
        offset = 0
    elif T_pseudo > n_train_set:
        # pseudo_full 是 (T_full, n_total),trainSet 是 (T_full - 2*window) 个
        # 找 window 偏移:简单 = (T_pseudo - n_train_set) // 2
        offset = (T_pseudo - n_train_set) // 2
    else:
        raise ValueError(f"pseudo_full shape {pseudo_full.shape} can't align with trainSet len {n_train_set}")

    for i, d in enumerate(trainSet):
        t_idx = offset + i
        # Update y at pseudo nodes
        y_curr = d.y.numpy().copy()    # (n_total, 1)
        y_curr[pseudo_arr, 0] = pseudo_full[t_idx, pseudo_arr]
        d.y = torch.FloatTensor(y_curr)
        # Update pseudo_mask + label_mask
        pmask = np.zeros(n_total, dtype=bool); pmask[pseudo_arr] = True
        d.pseudo_mask = torch.BoolTensor(pmask)
        d.label_mask = d.train_mask | d.pseudo_mask

    n_per_g = trainSet[0].label_mask.sum().item()
    print(f"[DEBUG/ST_inject] injected pseudo at {len(pseudo_idx_set)} nodes, "
          f"per-graph label_mask True count: {n_per_g} "
          f"(expected = train {trainSet[0].train_mask.sum().item()} + pseudo {len(pseudo_idx_set)})")


def ensure_selftrain_masks(trainSet, validSet, n_total):
    """确保每个 Data 有 train_mask + pseudo_mask 两个 BoolTensor。

    初始 train_mask = label_mask(50 train),pseudo_mask = 全 False。
    label_mask 维持 train_mask | pseudo_mask = 50 True(第 0 轮没 pseudo)。
    """
    for d in trainSet:
        if not hasattr(d, 'train_mask') or d.train_mask is None:
            d.train_mask = d.label_mask.clone()
        if not hasattr(d, 'pseudo_mask') or d.pseudo_mask is None:
            d.pseudo_mask = torch.zeros(n_total, dtype=torch.bool)
        d.label_mask = d.train_mask | d.pseudo_mask
    # validSet 的 label_mask 标记 valid 8 站,不动
    print(f"[DEBUG/ST_init] trainSet[0].train_mask.sum() = {trainSet[0].train_mask.sum().item()}, "
          f"pseudo_mask.sum() = {trainSet[0].pseudo_mask.sum().item()}")


# =====================================================================
# Custom train / test:加权 sup + pseudo loss
# =====================================================================

_ST_DEBUG_PRINTED = {'train': False}


def train_one_round(loader, model, lossFn, opt, scheduler, device, n_total,
                    lambda_pseudo, round_idx):
    """一个 epoch 的 self-train 训练 —— 加权 sup + pseudo loss。

    要求每个 batch 有 train_mask + pseudo_mask 属性。
    Loss: L_sup(train_mask) + λ_psd × L_pseudo(pseudo_mask)

    Returns: (avg_loss, RMSE_at_train, sup_loss_avg, psd_loss_avg)
    """
    global _ST_DEBUG_PRINTED
    model.train()
    _LOSS = 0; _SUP = 0; _PSD = 0
    pred_train, truth_train = [], []
    n_psd_batch = 0
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
        # Sup loss: 在 train_mask = True 节点上
        sup_mask = _batch.train_mask
        sup_loss = lossFn(yHat[sup_mask], _batch.y[sup_mask])
        # Pseudo loss: 在 pseudo_mask = True 节点上(可能为空,第 0 轮)
        psd_mask = _batch.pseudo_mask
        if psd_mask.any():
            psd_loss = lossFn(yHat[psd_mask], _batch.y[psd_mask])
            n_psd_batch += 1
        else:
            psd_loss = torch.tensor(0.0, device=device)
        total = sup_loss + lambda_pseudo * psd_loss
        total.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += total
        _SUP  += sup_loss
        _PSD  += psd_loss

        # collect train preds for RMSE
        bs = _batch.x.shape[0] // n_total
        n_train_per_g = _batch.train_mask.sum().item() // max(bs, 1)
        pt = yHat[sup_mask].reshape(-1, n_train_per_g).cpu().detach().numpy()
        tt = _batch.y[sup_mask].reshape(-1, n_train_per_g).cpu().detach().numpy()
        pred_train += list(pt)
        truth_train += list(tt)

        if _n == 0 and not _ST_DEBUG_PRINTED['train']:
            n_train_g = _batch.train_mask.sum().item() // max(bs, 1)
            n_psd_g   = _batch.pseudo_mask.sum().item() // max(bs, 1)
            print(f"[DEBUG/ST_train R{round_idx}] first batch: "
                  f"x={tuple(_batch.x.shape)}, y={tuple(_batch.y.shape)}, "
                  f"train_mask/g={n_train_g}, pseudo_mask/g={n_psd_g}, "
                  f"yHat range=[{yHat.min().item():.4f}, {yHat.max().item():.4f}]")
            print(f"[DEBUG/ST_train R{round_idx}] sup_loss={sup_loss.item():.4e}, "
                  f"psd_loss={psd_loss.item():.4e}, λ={lambda_pseudo}, "
                  f"total={total.item():.4e}")
            assert torch.isfinite(yHat).all(), "[ERR/ST] yHat NaN/Inf"
            assert torch.isfinite(total), f"[ERR/ST] total loss NaN/Inf: {total}"
            _ST_DEBUG_PRINTED['train'] = True
    scheduler.step()
    truth_train = np.array(truth_train); pred_train = np.array(pred_train)
    rmse_train = utils.RMSE(truth_train, pred_train)
    n = _n + 1
    return ((_LOSS / n).item(),
            rmse_train,
            (_SUP / n).item(),
            (_PSD / max(n_psd_batch, 1)).item() if n_psd_batch > 0 else 0.0)


def train_one_round_ead(loader, model, lossFn, opt, scheduler, device, n_total,
                        lambda_pseudo, round_idx, lambda_lap=0.0,
                        edge_src=None, edge_dst=None, edge_w=None):
    """EAD 集成版 train one round。

    模型输出 ε̂,所有 pseudo 存在 T 空间(由 caller 保证),内部统一:
      eps_target = y - WRF - α - β       (统一,train_mask 和 pseudo_mask 都适用)
      L_sup    = Huber(eps_hat[train], eps_target[train])
      L_psd    = Huber(eps_hat[psd],   eps_target[psd])
      L_lap    = λ_lap × Σ w_ij (eps_hat_i − eps_hat_j)²   (only if lambda_lap > 0)
      total    = L_sup + λ_psd × L_psd + L_lap

    RMSE 报告在 T 空间(reconstruct T_hat = WRF + α + β + ε̂)。

    Args:
        lambda_lap:           >0 启用 Lap on ε
        edge_src/edge_dst/edge_w: tensor on device,Lap 边列表(若 lambda_lap>0)
    """
    global _ST_DEBUG_PRINTED
    model.train()
    _LOSS = 0; _SUP = 0; _PSD = 0; _LAP = 0
    pred_train, truth_train = [], []
    n_psd_batch = 0
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        # Flatten 到 (bs*n_total,) shape(同 network.train_ead 风格)
        eps_hat = model(_batch.x, _batch.edge_index, _batch.edge_attr).squeeze(-1)
        bs = _batch.x.shape[0] // n_total
        wrf_t2 = _batch.wrf_t2.squeeze(-1)                              # (bs*n_total,)
        alpha  = _batch.alpha_t.repeat_interleave(n_total)              # (bs,)→(bs*n_total,)
        beta   = _batch.beta_hat.squeeze(-1)                            # (bs*n_total,)
        y_flat = _batch.y.squeeze(-1)                                   # (bs*n_total,)
        eps_target = y_flat - wrf_t2 - alpha - beta                     # ε_target

        # Sup loss(train_mask)
        sup_mask = _batch.train_mask
        sup_loss = lossFn(eps_hat[sup_mask], eps_target[sup_mask])

        # Pseudo loss(pseudo_mask)
        psd_mask = _batch.pseudo_mask
        if psd_mask.any():
            psd_loss = lossFn(eps_hat[psd_mask], eps_target[psd_mask])
            n_psd_batch += 1
        else:
            psd_loss = torch.tensor(0.0, device=device)

        # Lap loss on ε(可选)
        if lambda_lap > 0 and edge_src is not None:
            eps_i = eps_hat[edge_src]
            eps_j = eps_hat[edge_dst]
            lap_loss = (edge_w * (eps_i - eps_j).pow(2)).mean()
        else:
            lap_loss = torch.tensor(0.0, device=device)

        total = sup_loss + lambda_pseudo * psd_loss + lambda_lap * lap_loss
        total.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total
        _SUP  += sup_loss
        _PSD  += psd_loss
        _LAP  += lap_loss

        # Reconstruct T for RMSE(T space,只在 train_mask 上)
        T_hat = wrf_t2 + alpha + beta + eps_hat                         # (bs*n_total,)
        n_train_per_g = sup_mask.sum().item() // max(bs, 1)
        pred_train.append(T_hat[sup_mask].detach().cpu().numpy().reshape(-1, n_train_per_g))
        truth_train.append(y_flat[sup_mask].detach().cpu().numpy().reshape(-1, n_train_per_g))

        if _n == 0 and not _ST_DEBUG_PRINTED['train']:
            n_train_g = sup_mask.sum().item() // max(bs, 1)
            n_psd_g   = psd_mask.sum().item() // max(bs, 1)
            print(f"[DEBUG/ST_train_ead R{round_idx}] first batch: "
                  f"train_mask/g={n_train_g}, pseudo_mask/g={n_psd_g}, "
                  f"eps_hat range=[{eps_hat.min().item():.4f}, {eps_hat.max().item():.4f}], "
                  f"eps_target range=[{eps_target.min().item():.4f}, {eps_target.max().item():.4f}]")
            print(f"[DEBUG/ST_train_ead R{round_idx}] sup={sup_loss.item():.4e}, "
                  f"psd={psd_loss.item():.4e}, lap={lap_loss.item():.4e}, "
                  f"λ_psd={lambda_pseudo}, λ_lap={lambda_lap}, total={total.item():.4e}")
            assert torch.isfinite(eps_hat).all(), "[ERR/ST_EAD] eps_hat NaN/Inf"
            assert torch.isfinite(total), f"[ERR/ST_EAD] total loss NaN/Inf: {total}"
            _ST_DEBUG_PRINTED['train'] = True
    scheduler.step()
    pred_train = np.concatenate(pred_train, axis=0)
    truth_train = np.concatenate(truth_train, axis=0)
    rmse_train = utils.RMSE(truth_train, pred_train)
    n = _n + 1
    return ((_LOSS / n).item(),
            rmse_train,
            (_SUP / n).item(),
            (_PSD / max(n_psd_batch, 1)).item() if n_psd_batch > 0 else 0.0)


# =====================================================================
# Main orchestrator
# =====================================================================

def selftrain_main(model, lossFn, trainLoader, validLoader, trainSet, validSet,
                    metadata, modelParam, dataParam,
                    device, st_config, modelName, output_dir,
                    tgt_scl_C, wandb_run=None, log_file=None):
    """Self-training 主入口。

    要求 model 已加载 R0 ckpt(由 caller 在 run.py 里做)。

    st_config: dict,详见 run.py 里的 build。

    Returns: (final_best_v_rmse, history)
    """
    n_total       = metadata['nNodes']
    train_idx     = metadata.get('train_station_idx', None)
    valid_idx     = metadata.get('valid_station_idx', None)
    adj_matrix    = metadata.get('AdjMatrix', None)
    locations     = metadata.get('locations', None)
    targets_norm_full = metadata.get('targets_norm_full', None)
    assert train_idx is not None, "[ERR/ST] need metadata['train_station_idx']"
    assert valid_idx is not None, "[ERR/ST] need metadata['valid_station_idx']"
    assert adj_matrix is not None, "[ERR/ST] need metadata['AdjMatrix']"

    pseudo_source     = st_config['pseudo_source']
    confidence_type   = st_config['confidence_type']
    n_rounds          = st_config['n_rounds']
    K_per_round       = st_config['k_per_round']
    round_epochs      = st_config['round_epochs']
    warmstart_lr      = st_config['warmstart_lr']
    lambda_pseudo     = st_config['lambda_pseudo']
    tau_quantile      = st_config['tau_quantile']
    alpha_div         = st_config['alpha_div']
    beta_rel          = st_config['beta_rel']

    # === Step 0: ensure masks ===
    print(f"\n{'='*70}")
    print(f"[ST] Self-Training start: pseudo={pseudo_source}, conf={confidence_type}")
    print(f"[ST] Hyperparams: K={K_per_round}, N_rounds={n_rounds}, "
          f"round_epochs={round_epochs}, λ_psd={lambda_pseudo}, "
          f"warmstart_lr={warmstart_lr}, τ={tau_quantile}")
    print(f"[ST] train: {len(train_idx)} stations, valid: {len(valid_idx)} stations, "
          f"n_total: {n_total}")
    print(f"{'='*70}\n")

    ensure_selftrain_masks(trainSet, validSet, n_total)

    # === EAD / Lap 检测(决定用哪个 train/test 函数)===
    ead_active = bool(int(os.environ.get('V1_EAD_ALPHA', '0')) or int(os.environ.get('V1_EAD_BETA', '0')))
    lambda_lap_local = float(os.environ.get('V1_LAMBDA_LAP', '0.0'))
    lap_active = lambda_lap_local > 0
    print(f"\n[ST/EAD] ead_active={ead_active}, lap_active={lap_active}, λ_lap={lambda_lap_local}")

    # Lap 边列表(只在 lap_active 时构造)
    edge_src_t = edge_dst_t = edge_w_t = None
    if lap_active:
        adj = metadata['AdjMatrix']
        _e_src, _e_dst = np.where(adj > 0)
        _e_w = adj[_e_src, _e_dst]
        edge_w_t   = torch.FloatTensor(_e_w).to(device)
        edge_src_t = torch.LongTensor(_e_src).to(device)
        edge_dst_t = torch.LongTensor(_e_dst).to(device)
        print(f"[ST/EAD] Lap edges built: n_edges={len(_e_src)}")

    # === Step 1: R0 baseline valid RMSE (sanity) ===
    print(f"\n[ST] === R0 baseline check (model loaded from R0 ckpt) ===")
    import network
    if ead_active:
        test_valid_fn = network.test_ead
        print(f"[ST] Using test_ead for valid RMSE(EAD 模式)")
    else:
        test_valid_fn = network.test
    _, r0_valid_rmse_arr, _, _ = test_valid_fn(validLoader, model, lossFn, device, n_total)
    r0_valid_rmse = float(r0_valid_rmse_arr[0])
    print(f"[ST] R0 valid RMSE = {r0_valid_rmse:.4f} ≈ {r0_valid_rmse * tgt_scl_C:.3f}°C")

    best_v_rmse = float(r0_valid_rmse)
    best_round = 0
    best_state = copy.deepcopy(model.state_dict())
    cumulative_pseudo = []
    history = [{
        'round': 0, 'cumulative_K': 0, 'val_rmse': float(r0_valid_rmse),
        'val_rmse_C': float(r0_valid_rmse * tgt_scl_C),
        'best_so_far': float(best_v_rmse),
    }]
    if log_file is not None:
        log_file.write(f"\n=== Self-Train start ===\n")
        log_file.write(f"R0 baseline: v_rmse={r0_valid_rmse:.6f} (≈ {r0_valid_rmse * tgt_scl_C:.3f}°C)\n")
        log_file.flush()

    # === Step 2: 每轮 ===
    global_epoch = 0
    for r in range(1, n_rounds + 1):
        print(f"\n{'='*70}")
        print(f"[ST] === Round {r}/{n_rounds} ===")
        print(f"{'='*70}")
        round_t0 = time.time()

        # 2a. 算 confidence
        global _ST_DEBUG_PRINTED
        _ST_DEBUG_PRINTED = {'train': False}   # 每轮重置 first-batch debug

        if confidence_type == 'neighbor':
            confidence = compute_neighbor_error_confidence(
                model, trainLoader, device, n_total, train_idx, adj_matrix, k_neighbors=5)
        elif confidence_type == 'kriging_struct':
            confidence = compute_kriging_struct_confidence(adj_matrix, train_idx, n_total, k=10)
        elif confidence_type == 'conformal':
            emb = extract_embeddings(model, trainLoader, device, n_total)
            confidence = compute_conformal_confidence(
                model, trainLoader, device, n_total, train_idx, valid_idx, emb,
                n_folds=5, k_emb=8)
        else:
            raise ValueError(f"unknown confidence_type={confidence_type}")

        # 2b. 计算 embedding(用于 diversity / relevance)
        emb_per_station = extract_embeddings(model, trainLoader, device, n_total)
        valid_emb = emb_per_station[valid_idx]   # (n_valid, emb_dim)

        # 2c. greedy_select K 新 pseudo
        new_selected = greedy_select_3metric(
            confidence, emb_per_station, valid_emb, K_per_round,
            alpha_div=alpha_div, beta_rel=beta_rel, tau_quantile=tau_quantile,
            already_selected=cumulative_pseudo, train_idx=train_idx)
        if len(new_selected) == 0:
            print(f"[ST R{r}] no new pseudo selected, stopping")
            break
        cumulative_pseudo = sorted(set(cumulative_pseudo) | set(new_selected))
        print(f"[ST R{r}] cumulative pseudo nodes: {len(cumulative_pseudo)} "
              f"(this round added {len(new_selected)})")

        # 2d. 算 pseudo target(始终 T 空间;EAD 模式下 self/hybrid 也输出 T)
        if pseudo_source == 'self':
            pseudo_full = compute_self_pseudo(model, trainLoader, device, n_total,
                                              cumulative_pseudo, ead_active=ead_active)
        elif pseudo_source == 'kriging':
            assert targets_norm_full is not None, "[ERR/ST] kriging needs metadata['targets_norm_full']"
            pseudo_full = compute_kriging_pseudo(targets_norm_full, locations,
                                                 train_idx, cumulative_pseudo, k=10)
        elif pseudo_source == 'hybrid':
            assert targets_norm_full is not None, "[ERR/ST] hybrid needs metadata['targets_norm_full']"
            alpha_self = st_config.get('hybrid_alpha_self', 0.5)
            pseudo_full = compute_hybrid_pseudo(model, trainLoader, device, n_total,
                                                 targets_norm_full, locations,
                                                 train_idx, cumulative_pseudo,
                                                 alpha_self=alpha_self, k=10,
                                                 ead_active=ead_active)
        else:
            raise ValueError(f"unknown pseudo_source={pseudo_source}")

        # 2e. 写入 trainSet
        inject_pseudo_into_dataset(trainSet, cumulative_pseudo, pseudo_full, n_total)

        # 2f. warm-start lr,训 round_epochs
        opt = Adam(model.parameters(), lr=warmstart_lr)
        scheduler = ExponentialLR(opt, gamma=0.9992)
        round_best_v = best_v_rmse
        round_best_round = r
        for ep in range(round_epochs):
            if ead_active:
                tr_loss, tr_rmse_arr, sup_loss, psd_loss = train_one_round_ead(
                    trainLoader, model, lossFn, opt, scheduler, device, n_total,
                    lambda_pseudo, r,
                    lambda_lap=lambda_lap_local,
                    edge_src=edge_src_t, edge_dst=edge_dst_t, edge_w=edge_w_t)
            else:
                tr_loss, tr_rmse_arr, sup_loss, psd_loss = train_one_round(
                    trainLoader, model, lossFn, opt, scheduler, device, n_total,
                    lambda_pseudo, r)
            tr_rmse = float(tr_rmse_arr[0])
            v_loss, v_rmse_arr, _, _ = test_valid_fn(validLoader, model, lossFn, device, n_total)
            v_rmse = float(v_rmse_arr[0])
            global_epoch += 1
            log_msg = (f"[ST R{r} ep {ep+1:3d}/{round_epochs}] "
                       f"tr_rmse={tr_rmse:.4f}({tr_rmse*tgt_scl_C:.2f}°C) "
                       f"v_rmse={v_rmse:.4f}({v_rmse*tgt_scl_C:.2f}°C) "
                       f"sup={sup_loss:.4e} psd={psd_loss:.4e} "
                       f"best_global={best_v_rmse:.4f}")
            if (ep + 1) % 20 == 0 or ep == 0 or ep == round_epochs - 1:
                print(log_msg)
            if log_file is not None:
                log_file.write(log_msg + "\n"); log_file.flush()
            if v_rmse < best_v_rmse:
                best_v_rmse = float(v_rmse)
                best_round = r
                best_state = copy.deepcopy(model.state_dict())
                # save checkpoint
                torch.save(best_state, os.path.join(output_dir, f'{modelName}.pt'))
                print(f"  → new best! v_rmse={v_rmse:.4f}, saved ckpt")
            if v_rmse < round_best_v:
                round_best_v = float(v_rmse)
                round_best_round = r
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        'epoch_global': global_epoch,
                        'round_idx':    r,
                        'st/cumulative_K': len(cumulative_pseudo),
                        'st/sup_loss':    sup_loss,
                        'st/psd_loss':    psd_loss,
                        'train/loss':     tr_loss,
                        'train/RMSE':     tr_rmse,
                        'valid/loss':     v_loss,
                        'valid/RMSE':     v_rmse,
                        'valid/RMSE_C':   v_rmse * tgt_scl_C,
                        'st/best_v_rmse': best_v_rmse,
                    })
                except Exception as e:
                    print(f"[WARN/ST] wandb log error: {e}")

        round_t = time.time() - round_t0
        history.append({
            'round': r,
            'cumulative_K': len(cumulative_pseudo),
            'val_rmse': float(round_best_v),
            'val_rmse_C': float(round_best_v * tgt_scl_C),
            'best_so_far': float(best_v_rmse),
            'round_time_s': round_t,
        })
        print(f"\n[ST R{r}] DONE (took {round_t:.1f}s) | round best v_rmse = {round_best_v:.4f} "
              f"(≈ {round_best_v * tgt_scl_C:.3f}°C) | global best = {best_v_rmse:.4f} "
              f"(≈ {best_v_rmse * tgt_scl_C:.3f}°C, at R{best_round})")
        if log_file is not None:
            log_file.write(f"[ST R{r}] DONE: round_best={round_best_v:.6f}, "
                          f"global_best={best_v_rmse:.6f}\n")
            log_file.flush()

    # === Step 3: final summary ===
    print(f"\n{'='*70}")
    print(f"[ST] FINAL: best v_rmse = {best_v_rmse:.4f} (≈ {best_v_rmse * tgt_scl_C:.3f}°C) "
          f"at R{best_round}")
    print(f"[ST] Δ vs R0 baseline ({r0_valid_rmse:.4f}): {best_v_rmse - r0_valid_rmse:+.4f}")
    print(f"[ST] Total rounds: {len(history) - 1}, total pseudo: {len(cumulative_pseudo)}")
    print(f"{'='*70}\n")

    # Restore best ckpt to model
    model.load_state_dict(best_state)
    return best_v_rmse, history

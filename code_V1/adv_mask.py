"""
Adversarial near-valid mask augmentation for V1 GNN baseline.

设计:
  - 训练前一次性算好"离 valid 8 站最近的 K 个 unlabeled 节点"(emb 空间 kNN)
  - 训练时,每个 batch 随机 mask 这 K 个候选的一部分(p_mask_per_batch 概率)
  - mask = 把对应节点的 features 置 0(node-level mask)
  - 主任务 forward 后照常算 sup loss(只在 50 train),无 reconstruction loss
  - 模型被强迫"在 valid-similar 区域信息缺失时,仍能从邻居 MP 凑出 train 站好预测"
  - 实质是 transductive learning + input augmentation

==============================================================
🔑 LEAK 规则
==============================================================
  ✓ Mask 用 valid 8 站的 emb / location(feature 类信息)选择候选 → no leak
  ✗ 不接触 valid 8 站的 target
  ✓ Mask 的对象是 unlabeled(非 train,非 valid)→ 不破坏监督
"""

import torch
import numpy as np
import utils

_ADVMASK_DEBUG_PRINTED = {'train': False}


def compute_near_valid_mask_idx(model, train_loader, device, n_total, valid_idx, K=20,
                                 n_labeled=None):
    """选 K 个离 valid 8 站最近(emb 空间)的 **unlabeled** 节点。

    Args:
        model:        当前 GNN(用来提 emb)
        train_loader: 用来跑 forward 拿 embeddings(只用其图拓扑,不需 shuffle 一致)
        device:
        n_total:      总节点数 = n_labeled + n_unlabeled
        valid_idx:    valid 8 站 indices
        K:            选多少 unlabeled 候选
        n_labeled:    labeled 站总数(必须传,否则用 valid_idx 推但不准)

    Returns:
        mask_idx: list of K node indices,**严格在 [n_labeled, n_total) 范围内**
                  → 都是 unlabeled,不会 mask labeled / valid 自身
    """
    # 跑一次 forward 拿全节点 emb
    model.eval()
    all_emb = []
    with torch.no_grad():
        for _batch in train_loader:
            _batch = _batch.to(device)
            x = _batch.x
            # 走 encoder + processor 拿 hidden(decoder 之前)
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
            bs = _batch.x.shape[0] // n_total
            all_emb.append(x.reshape(bs, n_total, -1).cpu().numpy())
    all_emb = np.concatenate(all_emb, axis=0)              # (T_used, n_total, emb_dim)
    emb_per_station = all_emb.mean(axis=0)                  # (n_total, emb_dim)

    valid_emb = emb_per_station[valid_idx]                  # (n_valid, emb_dim)
    valid_proto = valid_emb.mean(axis=0, keepdims=True)     # (1, emb_dim) 平均代表 valid

    # 严格只考虑 **unlabeled 节点**:[n_labeled, n_total)
    # n_labeled 必须传入,否则保守用 max(valid_idx)+1 推断(不准)
    assert n_labeled is not None, "[ERR/AdvMask] need n_labeled to exclude all labeled stations"

    distances = np.full(n_total, np.inf, dtype=np.float32)
    for u in range(n_labeled, n_total):    # 只在 unlabeled 范围算距离
        distances[u] = float(np.linalg.norm(emb_per_station[u] - valid_proto[0]))

    nearest_K = np.argsort(distances)[:K].tolist()
    # Sanity check: 都应该 ≥ n_labeled
    assert all(u >= n_labeled for u in nearest_K), \
        f"[ERR/AdvMask] selected mask 包含 labeled 节点!{nearest_K}"

    print(f"[DEBUG/AdvMask] 选了 K={K} 个 unlabeled 站,离 valid 平均 emb 最近")
    print(f"[DEBUG/AdvMask] valid_idx={list(valid_idx)},n_labeled={n_labeled},"
          f"unlabeled 候选范围 [{n_labeled}, {n_total})")
    print(f"[DEBUG/AdvMask] mask 候选 idx 前 10:{nearest_K[:10]} (全都应 ≥ {n_labeled} ✓)")
    print(f"[DEBUG/AdvMask] 这些站到 valid_proto 距离 range: "
          f"[{distances[nearest_K].min():.4f}, {distances[nearest_K].max():.4f}]")
    return nearest_K


def train_advmask(loader, model, lossFn, opt, scheduler, device, n_total,
                   mask_idx, p_mask_per_batch=0.5):
    """One epoch:adversarial near-valid mask training.

    每个 batch:从 mask_idx 里随机选 p_mask_per_batch 比例的节点 → mask 它们的 features
    然后正常 forward + sup loss。
    """
    global _ADVMASK_DEBUG_PRINTED
    model.train()
    _LOSS = 0
    pred, truth = [], []
    mask_idx_arr = np.array(sorted(mask_idx), dtype=np.int64)

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)

        # 随机选 mask_idx 中的 p_mask 比例(每 batch 不同子集 → 增加 diversity)
        K = len(mask_idx_arr)
        n_mask = int(np.ceil(K * p_mask_per_batch))
        rand_perm = np.random.permutation(K)
        this_mask = mask_idx_arr[rand_perm[:n_mask]]

        # 在 batched x 上对每个 graph 重复 mask 这些 station
        # batched x shape: (bs * n_total, iDim);第 g 个 graph 的 station u 在 row = g * n_total + u
        bs = _batch.x.shape[0] // n_total
        x_masked = _batch.x.clone()
        for g in range(bs):
            offsets = g * n_total + this_mask
            x_masked[offsets] = 0.0

        yHat = model(x_masked, _batch.edge_index, _batch.edge_attr)

        # 用现有 _split_for_loss 的等效:label_mask 控制 loss 节点
        if hasattr(_batch, 'label_mask') and _batch.label_mask is not None:
            mask_lbl = _batch.label_mask
            yHat_l = yHat[mask_lbl]
            y_l = _batch.y[mask_lbl]
            n_lbl = mask_lbl.sum().item() // max(bs, 1)
            pt = yHat_l.reshape(-1, n_lbl).cpu().detach().numpy()
            tt = y_l.reshape(-1, n_lbl).cpu().detach().numpy()
        else:
            yHat_l = yHat; y_l = _batch.y
            pt = yHat.reshape(bs, n_total).cpu().detach().numpy()
            tt = _batch.y.reshape(bs, n_total).cpu().detach().numpy()
        loss = lossFn(yHat_l, y_l)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += loss
        pred += list(pt); truth += list(tt)

        if _n == 0 and not _ADVMASK_DEBUG_PRINTED['train']:
            print(f"[DEBUG/train_advmask] mask K={K},this batch masked {n_mask} stations")
            print(f"[DEBUG/train_advmask] this_mask first 5: {this_mask[:5].tolist()}")
            print(f"[DEBUG/train_advmask] yHat range=[{yHat.min().item():.4f}, {yHat.max().item():.4f}], "
                  f"loss={loss.item():.4e}")
            assert torch.isfinite(yHat).all(), "[ERR/AdvMask] yHat NaN/Inf"
            assert torch.isfinite(loss), f"[ERR/AdvMask] loss NaN/Inf: {loss}"
            _ADVMASK_DEBUG_PRINTED['train'] = True

    scheduler.step()
    truth = np.array(truth); pred = np.array(pred)
    RMSE = utils.RMSE(truth, pred)
    n = _n + 1
    train_advmask.last_n_masked = n_mask
    return (_LOSS / n).item(), RMSE, truth, pred

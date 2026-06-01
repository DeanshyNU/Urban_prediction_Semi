"""
Distribution Alignment(MMD-based,"半 CycleGAN" 思路)。

设计:
  - 让 unlabeled 节点的 embedding 分布 ≈ labeled(train+valid)节点的 embedding 分布
  - 用 Maximum Mean Discrepancy(MMD)度量两个 emb 集分布的距离
  - 加到 sup loss 上当 aux 正则:L = L_sup + λ × MMD(emb_labeled, emb_unlabeled)
  - 无需 discriminator,无对抗训练 → 稳定

为什么有意义:
  - 在 spatial OOD 任务里,labeled(58 站)和 unlabeled(400 站)在 emb 空间可能错位
  - MMD 强制两者对齐 → 模型对 unlabeled 的预测更可信
  - **不接触 unlabeled 的 target**(没有,本来就没)
  - **不接触 valid 的 target**(只用 features → embedding,不算 leak)

==============================================================
🔑 LEAK 规则
==============================================================
  ✓ 用 train + valid + unlabeled 的 **features → embedding**
  ✗ 不用 valid target(loss 仍只在 train 50 站算)
"""

import torch
import numpy as np
import utils
from torch_geometric.nn import GraphConv

_MMD_DEBUG_PRINTED = {'train': False}


def _gaussian_kernel(x, y, sigmas):
    """Multi-scale RBF kernel K(x_i, y_j) averaged over sigmas.

    Args:
        x: (n_x, d)
        y: (n_y, d)
        sigmas: list of bandwidth scalars

    Returns:
        K: (n_x, n_y) kernel matrix (averaged over sigma)
    """
    xx = (x * x).sum(dim=-1, keepdim=True)        # (n_x, 1)
    yy = (y * y).sum(dim=-1, keepdim=True)        # (n_y, 1)
    xy = x @ y.t()                                 # (n_x, n_y)
    d2 = xx + yy.t() - 2 * xy
    d2 = d2.clamp(min=0)                           # numerical: 不能为负

    K_sum = torch.zeros_like(d2)
    for sigma in sigmas:
        K_sum = K_sum + torch.exp(-d2 / (2 * sigma * sigma + 1e-9))
    return K_sum / len(sigmas)


def mmd_loss(x_a, x_b, sigmas=None):
    """MMD^2 between two distributions x_a (n_a, d) and x_b (n_b, d)."""
    if sigmas is None:
        # Median heuristic:用 pairwise distance 的 median 当 bandwidth
        with torch.no_grad():
            d = torch.cdist(x_a, x_b).flatten()
            d_pos = d[d > 0]
            if d_pos.numel() == 0:
                median = torch.tensor(1.0, device=x_a.device)
            else:
                median = d_pos.median().clamp(min=1e-3)
        # Multi-scale: 在 median 周围用 3 个 sigma
        sigmas = [median * 0.5, median * 1.0, median * 2.0]

    K_aa = _gaussian_kernel(x_a, x_a, sigmas)
    K_bb = _gaussian_kernel(x_b, x_b, sigmas)
    K_ab = _gaussian_kernel(x_a, x_b, sigmas)
    # Biased MMD^2 estimator(简单 + 可微)
    return K_aa.mean() + K_bb.mean() - 2 * K_ab.mean()


def _forward_with_hidden(model, x, edge_index, edge_attr):
    """模仿 GNN.forward 但额外返回 processor 后的 hidden(decoder 前)。

    Returns:
        yhat:   (N, oDim)
        hidden: (N, HLD) — processor 输出
    """
    h = x
    for f in model.encoder:
        h = f(h)
    for n_idx, f in enumerate(model.processor):
        if n_idx % 2 == 0:   # conv 层
            if isinstance(f, GraphConv):
                h = f(h, edge_index, edge_attr)
            else:
                h = f(h, edge_index)
        else:                # PReLU
            h = f(h)
    hidden = h
    yhat = hidden
    for f in model.decoder:
        yhat = f(yhat)
    return yhat, hidden


def train_dist_align(loader, model, lossFn, opt, scheduler, device, n_total,
                     n_labeled, lambda_mmd):
    """One epoch: sup loss + λ × MMD(emb_labeled, emb_unlabeled).

    每 batch:
      1. forward 模型 → 拿 hidden (bs × n_total, HLD)
      2. per-station aggregate(对 batch 内 timesteps 求平均)→ (n_total, HLD)
      3. emb_labeled = emb[:n_labeled],emb_unlabeled = emb[n_labeled:]
      4. MMD = MMD(emb_labeled, emb_unlabeled)
      5. loss = sup + λ × MMD,backward + step

    Returns: (loss, RMSE, truth, pred) 同 train() 签名
    """
    global _MMD_DEBUG_PRINTED
    model.train()
    _LOSS = 0; _SUP = 0; _MMD = 0
    pred, truth = [], []

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        yhat, hidden = _forward_with_hidden(model, _batch.x, _batch.edge_index, _batch.edge_attr)

        # Sup loss
        if hasattr(_batch, 'label_mask') and _batch.label_mask is not None:
            mask = _batch.label_mask
            sup_loss = lossFn(yhat[mask], _batch.y[mask])
            bs = _batch.x.shape[0] // n_total
            n_lbl_per_g = mask.sum().item() // max(bs, 1)
            pt = yhat[mask].reshape(-1, n_lbl_per_g).cpu().detach().numpy()
            tt = _batch.y[mask].reshape(-1, n_lbl_per_g).cpu().detach().numpy()
        else:
            sup_loss = lossFn(yhat, _batch.y)
            bs = _batch.x.shape[0] // n_total
            pt = yhat.reshape(bs, n_total).cpu().detach().numpy()
            tt = _batch.y.reshape(bs, n_total).cpu().detach().numpy()

        # MMD: per-station aggregate hidden(对 batch 内 128 个 timestep 求平均)
        bs = _batch.x.shape[0] // n_total
        hidden_2d = hidden.reshape(bs, n_total, -1)        # (bs, n_total, HLD)
        emb_per_station = hidden_2d.mean(dim=0)            # (n_total, HLD)

        emb_lbl = emb_per_station[:n_labeled]              # (n_labeled, HLD) — train + valid 一起
        emb_unl = emb_per_station[n_labeled:]              # (n_unlabeled, HLD)

        mmd = mmd_loss(emb_lbl, emb_unl)

        total = sup_loss + lambda_mmd * mmd
        total.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total
        _SUP  += sup_loss
        _MMD  += mmd
        pred += list(pt); truth += list(tt)

        if _n == 0 and not _MMD_DEBUG_PRINTED['train']:
            print(f"[DEBUG/train_mmd] sup_loss={sup_loss.item():.4e}, "
                  f"mmd={mmd.item():.4e}, λ={lambda_mmd}, total={total.item():.4e}")
            print(f"[DEBUG/train_mmd] emb_lbl shape={tuple(emb_lbl.shape)} "
                  f"(train+valid={n_labeled}), emb_unl shape={tuple(emb_unl.shape)}")
            print(f"[DEBUG/train_mmd] hidden range=[{hidden.min().item():.4f}, "
                  f"{hidden.max().item():.4f}], "
                  f"emb_per_station: lbl mean={emb_lbl.mean().item():.4f}, "
                  f"unl mean={emb_unl.mean().item():.4f}")
            assert torch.isfinite(mmd), f"[ERR/MMD] mmd NaN/Inf: {mmd}"
            assert torch.isfinite(total), f"[ERR/MMD] total NaN/Inf: {total}"
            _MMD_DEBUG_PRINTED['train'] = True

    scheduler.step()
    truth = np.array(truth); pred = np.array(pred)
    RMSE = utils.RMSE(truth, pred)
    n = _n + 1
    train_dist_align.last_sup = (_SUP / n).item()
    train_dist_align.last_mmd = (_MMD / n).item()
    train_dist_align.last_lambda = lambda_mmd
    return (_LOSS / n).item(), RMSE, truth, pred

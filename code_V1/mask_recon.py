"""
Mask Reconstruct multi-task aux head(给 unlabeled 节点 active supervision)。

设计:
  - 每 batch 随机 mask K 个 unlabeled 节点的 features(置 0)
  - 主任务:正常 forward + sup loss(在 train_mask 上)
  - Aux 任务:加一个 ReconHead,从 hidden 重建被 mask 的 features
  - 总 loss = L_main + λ_recon × L_recon

理论上突破 Pseudo Quality Ceiling:
  - features 是 ground truth(完美 quality)
  - 不像 kriging-pseudo 受 spatial smoothing 噪声限制

注意:
  - Mask 仅作用在 unlabeled(索引 ≥ n_labeled),不破坏 labeled supervision
  - Recon target 选 UF(17 维,静态站点特征),不选 WRF/CLMS(时序,噪声大)或 GeoEmbed(1008 维,信号稀释)
  - λ_recon 默认 0.1 防止 aux 主导

==============================================================
🔑 LEAK 规则
==============================================================
  ✓ Mask 仅在 unlabeled(idx ≥ n_labeled),保护 labeled / valid features
  ✓ Recon target 是 features 自己(no leak,因为 features 训练时本来就有)
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GraphConv
import utils

_MR_DEBUG_PRINTED = {'train': False}


class ReconHead(nn.Module):
    """从 hidden representation 重建 features。

    Architecture: hidden(HLD)→ Linear(64)→ PReLU → Linear(recon_dim)
    """
    def __init__(self, hidden_dim=128, recon_dim=17):
        super().__init__()
        self.h1 = nn.Linear(hidden_dim, 64)
        self.act = nn.PReLU(64)
        self.h2 = nn.Linear(64, recon_dim)

    def forward(self, h):
        return self.h2(self.act(self.h1(h)))


def _forward_with_hidden(model, x, edge_index, edge_attr):
    """Forward model 返回 (yhat, hidden):同 dist_align.py。"""
    h = x
    for f in model.encoder:
        h = f(h)
    for n_idx, f in enumerate(model.processor):
        if n_idx % 2 == 0:
            if isinstance(f, GraphConv):
                h = f(h, edge_index, edge_attr)
            else:
                h = f(h, edge_index)
        else:
            h = f(h)
    hidden = h
    yhat = hidden
    for f in model.decoder:
        yhat = f(yhat)
    return yhat, hidden


def get_recon_feat_slice(target='uf', geo_dim=1008):
    """根据 V2 schema 返回要重建的 feature slice (start, end)。

    schema:WRF(315) + aux(4) + CLMS(3) + UF(17) + GeoEmb(geo_dim) = iDim
    indices:
      WRF:    [0, 315)
      aux:    [315, 319)
      CLMS:   [319, 322)
      UF:     [322, 339)
      GeoEmb: [339, 339+geo_dim)
    """
    if target == 'uf':
        return (322, 339)        # 17 维,静态,最稳
    elif target == 'aux':
        return (315, 319)        # 4 维,时间特征
    elif target == 'clms':
        return (319, 322)        # 3 维
    elif target == 'uf_geo':
        return (322, 339 + geo_dim)  # UF + GeoEmb
    else:
        raise ValueError(f"unknown recon target={target}")


def train_mask_recon(loader, model, recon_head, lossFn, opt, scheduler, device,
                    n_total, n_labeled, lambda_recon,
                    mask_K=30, recon_feat_slice=(322, 339)):
    """One epoch:masked feature reconstruction multi-task。

    Args:
        recon_head:        ReconHead module(参数已加进 opt)
        lambda_recon:      recon loss 权重
        mask_K:            每 batch 随机 mask 的 unlabeled 节点数
        recon_feat_slice:  (start, end) 在 iDim 上,要重建的 feature 列范围

    Returns: (loss, RMSE_arr, truth, pred) 同 train() 签名
    """
    global _MR_DEBUG_PRINTED
    model.train()
    recon_head.train()
    _LOSS = 0; _SUP = 0; _REC = 0
    pred, truth = [], []

    f_start, f_end = recon_feat_slice
    recon_dim = f_end - f_start
    n_unlabeled = n_total - n_labeled

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        bs = _batch.x.shape[0] // n_total

        # 随机选 K 个 unlabeled 节点(每 batch 不同)
        mask_K_actual = min(mask_K, n_unlabeled)
        rand_perm = torch.randperm(n_unlabeled, device=device)
        mask_local = rand_perm[:mask_K_actual] + n_labeled   # 索引 in [n_labeled, n_total)

        # 扩展到所有 graphs in batch
        graph_offsets = torch.arange(bs, device=device) * n_total   # (bs,)
        mask_batched = (graph_offsets[:, None] + mask_local[None, :]).flatten()  # (bs * mask_K,)

        # 保留原 features 当 recon target
        recon_target = _batch.x[mask_batched, f_start:f_end].clone()  # (bs * mask_K, recon_dim)

        # Mask features 置 0
        x_masked = _batch.x.clone()
        x_masked[mask_batched] = 0

        # Forward with masked input
        yhat, hidden = _forward_with_hidden(model, x_masked,
                                             _batch.edge_index, _batch.edge_attr)

        # Sup loss(主任务,只在 label_mask 上算)
        if hasattr(_batch, 'label_mask') and _batch.label_mask is not None:
            mask_lbl = _batch.label_mask
            sup_loss = lossFn(yhat[mask_lbl], _batch.y[mask_lbl])
            n_lbl_per_g = mask_lbl.sum().item() // max(bs, 1)
            pt = yhat[mask_lbl].reshape(-1, n_lbl_per_g).cpu().detach().numpy()
            tt = _batch.y[mask_lbl].reshape(-1, n_lbl_per_g).cpu().detach().numpy()
        else:
            sup_loss = lossFn(yhat, _batch.y)
            pt = yhat.reshape(bs, n_total).cpu().detach().numpy()
            tt = _batch.y.reshape(bs, n_total).cpu().detach().numpy()

        # Recon loss:从 hidden 重建被 mask 的 features
        hidden_masked = hidden[mask_batched]                          # (bs * mask_K, HLD)
        recon_pred = recon_head(hidden_masked)                        # (bs * mask_K, recon_dim)
        recon_loss = ((recon_pred - recon_target) ** 2).mean()

        total = sup_loss + lambda_recon * recon_loss
        total.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total
        _SUP  += sup_loss
        _REC  += recon_loss
        pred += list(pt); truth += list(tt)

        if _n == 0 and not _MR_DEBUG_PRINTED['train']:
            print(f"[DEBUG/train_mask_recon] first batch: bs={bs}, n_unlabeled={n_unlabeled}, "
                  f"mask_K={mask_K_actual}, mask_local first 5: {mask_local[:5].tolist()}")
            print(f"[DEBUG/train_mask_recon] recon slice [{f_start},{f_end}) = {recon_dim} dim "
                  f"(对应 V2 schema 的 UF/aux/etc.)")
            print(f"[DEBUG/train_mask_recon] sup_loss={sup_loss.item():.4e}, "
                  f"recon_loss={recon_loss.item():.4e}, λ={lambda_recon}, total={total.item():.4e}")
            print(f"[DEBUG/train_mask_recon] yhat range=[{yhat.min().item():.4f}, "
                  f"{yhat.max().item():.4f}], recon_pred range=[{recon_pred.min().item():.4f}, "
                  f"{recon_pred.max().item():.4f}], recon_target range=[{recon_target.min().item():.4f}, "
                  f"{recon_target.max().item():.4f}]")
            assert torch.isfinite(yhat).all(), "[ERR/MR] yhat NaN/Inf"
            assert torch.isfinite(recon_loss), f"[ERR/MR] recon_loss NaN/Inf: {recon_loss}"
            assert torch.isfinite(total), f"[ERR/MR] total NaN/Inf: {total}"
            _MR_DEBUG_PRINTED['train'] = True

    scheduler.step()
    truth = np.array(truth); pred = np.array(pred)
    RMSE = utils.RMSE(truth, pred)
    n = _n + 1
    train_mask_recon.last_sup = (_SUP / n).item()
    train_mask_recon.last_recon = (_REC / n).item()
    train_mask_recon.last_lambda = lambda_recon
    return (_LOSS / n).item(), RMSE, truth, pred

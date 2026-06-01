"""Cross-Modality Prediction SSL (Experiment B,2026-05-31)

核心:naive semi baseline 上加 aux head — 在 mask 掉 CLMS feature 的节点上 重建 CLMS
       使用所有 458 nodes 的 features 自身当 target(无 T leak)

跟 13916 Mask Reconstruct 的关键区别:
  - 13916 target: UF static 17 dim  → trivial 重建
  - 本次 target:  CLMS dynamic 3 dim → 通过 vegetation phenology 跟 T 间接关联(蒸腾散热)
  - 13916 mask 仅 unlabeled 节点    → 但用 multi-task,sup 主导
  - 本次 mask 所有 458 节点的 25%   → aux 在所有 节点提供 supervision

数学:
  x_input ← x.clone(); x_input[mask_idx, CLMS_cols=319:322] = 0
  h = encoder + processor(x_input)
  y_T_pred = head_T(h)        # 主任务,通过 model.decoder
  y_CLMS_pred = head_CMP(h)   # aux head (新增)

  L = HuberLoss(y_T_pred[train_mask], y_train[train_mask])     # main
    + λ × MSE(y_CLMS_pred[mask_idx], CLMS[mask_idx])           # aux

==============================================================
🔑 LEAK 规则
==============================================================
  ✓ Aux target 是 CLMS features 自己(数据本身的一部分,no leak)
  ✓ Mask 后 model 看不到 mask 节点的 CLMS → reconstruction 非平凡
  ✓ 主任务 sup loss 不受影响(只在 train_mask 上算)
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GraphConv
import utils


_CMP_DEBUG = {'train': False}

# CLMS 在 1347 维 V2 schema 中的索引(详 [data.md](../code_log_V1/data.md) §7.2)
CLMS_START = 319
CLMS_END = 322   # exclusive
CLMS_DIM = CLMS_END - CLMS_START  # 3


class CMPHead(nn.Module):
    """Cross-Modality Prediction head: hidden → CLMS reconstruction.

    架构:HLD → 32 → CLMS_DIM (3)
    """
    def __init__(self, hidden_dim=128, recon_dim=CLMS_DIM):
        super().__init__()
        self.h1 = nn.Linear(hidden_dim, 32)
        self.act = nn.PReLU(32)
        self.h2 = nn.Linear(32, recon_dim)

    def forward(self, h):
        return self.h2(self.act(self.h1(h)))


def _forward_encode(model, x, edge_index, edge_attr):
    """Run model's encoder + processor (不含 decoder),返回 hidden state."""
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
    return h


def _forward_decode(model, hidden):
    """Run model's decoder on hidden,返回 T predictions."""
    yhat = hidden
    for f in model.decoder:
        yhat = f(yhat)
    return yhat


def train_cmp_ssl(loader, model, cmp_head, lossFn, opt, scheduler, device, n_total,
                  lambda_cmp=0.1, mask_ratio=0.25):
    """One epoch of CMP-SSL on naive semi baseline.

    Args:
        loader:       DataLoader (per-timestep batches)
        model:        GNN model (encoder + processor + decoder)
        cmp_head:     CMPHead instance (CLMS reconstruction)
        lossFn:       HuberLoss for main task
        opt:          optimizer over model + cmp_head
        scheduler:    LR scheduler
        device:       cuda
        n_total:      total nodes per graph (458)
        lambda_cmp:   weight on CMP aux loss (default 0.1)
        mask_ratio:   fraction of nodes to mask CLMS at (default 0.25)

    Returns: (loss, RMSE, truth, pred) — 标准 signature
    """
    model.train()
    cmp_head.train()
    _LOSS_TOTAL = 0
    _LOSS_SUP = 0
    _LOSS_CMP = 0
    pred, truth = [], []

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        x_orig = _batch.x                                 # (bs * n_total, 1347)
        N = x_orig.shape[0]

        # ---- Mask CLMS at random nodes ----
        # Randomly select mask_ratio fraction of (bs * n_total) nodes
        n_mask = max(1, int(N * mask_ratio))
        mask_idx = torch.randperm(N, device=device)[:n_mask]
        mask_bool = torch.zeros(N, dtype=torch.bool, device=device)
        mask_bool[mask_idx] = True

        # Store original CLMS values for these masked nodes as target
        clms_target = x_orig[mask_bool, CLMS_START:CLMS_END].clone()   # (n_mask, 3)

        # Build masked input
        x_masked = x_orig.clone()
        x_masked[mask_bool, CLMS_START:CLMS_END] = 0.0

        # ---- Forward: encode + decode (main task) + CMP head (aux task) ----
        hidden = _forward_encode(model, x_masked, _batch.edge_index, _batch.edge_attr)
        yhat_T = _forward_decode(model, hidden)            # (bs * n_total, 1)
        yhat_CLMS = cmp_head(hidden)                       # (bs * n_total, 3)

        # ---- Main loss: T prediction on train_mask ----
        bs = N // n_total
        if hasattr(_batch, 'label_mask') and _batch.label_mask is not None:
            train_mask = _batch.label_mask
            sup_loss = lossFn(yhat_T[train_mask], _batch.y[train_mask])
            n_lbl_per_g = train_mask.sum().item() // max(bs, 1)
            pt = yhat_T[train_mask].reshape(-1, n_lbl_per_g).cpu().detach().numpy()
            tt = _batch.y[train_mask].reshape(-1, n_lbl_per_g).cpu().detach().numpy()
        else:
            sup_loss = lossFn(yhat_T, _batch.y)
            pt = yhat_T.reshape(bs, n_total).cpu().detach().numpy()
            tt = _batch.y.reshape(bs, n_total).cpu().detach().numpy()

        # ---- CMP loss: CLMS reconstruction on masked nodes ----
        clms_pred_masked = yhat_CLMS[mask_bool]            # (n_mask, 3)
        cmp_loss = ((clms_pred_masked - clms_target) ** 2).mean()

        total_loss = sup_loss + lambda_cmp * cmp_loss
        total_loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS_TOTAL += total_loss
        _LOSS_SUP += sup_loss.item()
        _LOSS_CMP += cmp_loss.item()
        pred += list(pt)
        truth += list(tt)

        # ===== DEBUG: first batch =====
        if _n == 0 and not _CMP_DEBUG['train']:
            assert torch.isfinite(yhat_T).all(), "[ERR/CMP] yhat_T NaN/Inf"
            assert torch.isfinite(yhat_CLMS).all(), "[ERR/CMP] yhat_CLMS NaN/Inf"
            assert torch.isfinite(total_loss), f"[ERR/CMP] total NaN/Inf: {total_loss}"
            print(f"[DEBUG/cmp_ssl] first batch: N={N}, bs={bs}, n_mask={n_mask} "
                  f"({n_mask/N:.3f} of nodes), mask_ratio={mask_ratio}")
            print(f"[DEBUG/cmp_ssl] CLMS cols=[{CLMS_START},{CLMS_END}), "
                  f"target shape={tuple(clms_target.shape)}")
            print(f"[DEBUG/cmp_ssl] sup_T={sup_loss.item():.4e}, "
                  f"cmp_CLMS={cmp_loss.item():.4e} (λ={lambda_cmp}), "
                  f"total={total_loss.item():.4e}")
            print(f"[DEBUG/cmp_ssl] yhat_T range=[{yhat_T.min().item():.4f}, "
                  f"{yhat_T.max().item():.4f}], "
                  f"yhat_CLMS range=[{yhat_CLMS.min().item():.4f}, "
                  f"{yhat_CLMS.max().item():.4f}]")
            print(f"[DEBUG/cmp_ssl] CLMS target range=[{clms_target.min().item():.4f}, "
                  f"{clms_target.max().item():.4f}], "
                  f"|recon_error| mean={(clms_pred_masked - clms_target).abs().mean().item():.4f}")
            _CMP_DEBUG['train'] = True

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    train_cmp_ssl.last_sup = _LOSS_SUP / (_n + 1)
    train_cmp_ssl.last_cmp = _LOSS_CMP / (_n + 1)
    train_cmp_ssl.last_lambda = lambda_cmp
    return (_LOSS_TOTAL/(_n+1)).item(), _RMSE, truth, pred

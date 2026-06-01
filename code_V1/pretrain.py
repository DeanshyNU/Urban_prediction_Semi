"""Modality-aware masked feature reconstruction pretraining (V2 schema).

Two stages:
  Stage 1: pretrain encoder+processor on mask reconstruction (no T labels)
  Stage 2: finetune full model (encoder+processor+decoder) with EAD+Lap on T labels

Mask strategy:
  - WRF Tair (per window: indices [0,9), [63,72), [126,135), [189,198), [252,261))
    ALWAYS visible (baseline anchor for EAD)
  - WRF non-Tair (per window: indices [9,63), [72,126), [135,189), [198,252), [261,315))
    MASKABLE (25% randomly mask)
  - station_aux [315, 319):  ALWAYS visible (time index, needed for α(t))
  - CLMS [319, 322):          MASKABLE (25% randomly mask)
  - UF [322, 339):            ALWAYS visible (static context)
  - GeoEmbed [339, 1347):     ALWAYS visible (static context)

Loss: MSE only on masked positions (per-modality decoders, summed).

==============================================================
🔑 防 13916 失败的关键差别
==============================================================
  1. 两阶段(pretrain 单独 / finetune 单独),不是 multi-task
  2. Mask 的是 dynamic features(WRF/CLMS),不是 trivial-to-reconstruct static UF
  3. WRF Tair anchor + UF + GeoEmb 一直当 context,降低 reconstruction trivialness
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GraphConv

_PT_DEBUG = {'printed': False}


def get_maskable_ranges(window=2):
    """Return list of (start, end) ranges of maskable feature indices.

    WRF non-Tair per window: indices k*63 + [9, 63)
    CLMS:                    [319, 322)
    """
    n_windows = 2 * window + 1   # =5 for window=2
    ranges = []
    for k in range(n_windows):
        ranges.append((k * 63 + 9, k * 63 + 63))   # WRF non-Tair
    ranges.append((319, 322))                       # CLMS
    return ranges


def build_maskable_cols(window=2, device='cpu'):
    """Flatten ranges into a (n_maskable,) tensor of column indices."""
    cols = []
    for s, e in get_maskable_ranges(window):
        cols.extend(range(s, e))
    return torch.tensor(cols, dtype=torch.long, device=device)


class ReconHead(nn.Module):
    """Per-modality reconstruction head: hidden → recon_dim."""
    def __init__(self, hidden_dim=128, recon_dim=270):
        super().__init__()
        self.h1 = nn.Linear(hidden_dim, 64)
        self.act = nn.PReLU(64)
        self.h2 = nn.Linear(64, recon_dim)

    def forward(self, h):
        return self.h2(self.act(self.h1(h)))


def forward_encoder(model, x, edge_index, edge_attr):
    """Run model's encoder + processor (decoder excluded). Returns hidden state."""
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


def apply_mask(x, maskable_cols, mask_ratio=0.25):
    """Randomly mask `mask_ratio` of maskable_cols per node, set to 0.

    Args:
        x:             (N_nodes, F=1347) input
        maskable_cols: (n_maskable,) LongTensor of column indices to mask
        mask_ratio:    fraction of maskable positions to mask per node

    Returns:
        x_masked: (N, F) with masked positions zeroed
        mask:     (N, F) bool, True at masked positions
    """
    N, F = x.shape
    n_maskable = maskable_cols.shape[0]
    n_mask_per_node = max(1, int(n_maskable * mask_ratio))

    # Random per-node ranking → top-k indices into maskable_cols
    rand = torch.rand(N, n_maskable, device=x.device)
    _, sel = torch.topk(rand, n_mask_per_node, dim=1)        # (N, K)
    actual_cols = maskable_cols[sel]                          # (N, K)

    mask = torch.zeros(N, F, dtype=torch.bool, device=x.device)
    row = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, n_mask_per_node)
    mask[row, actual_cols] = True

    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def train_pretrain(loader, model, recon_heads, opt, scheduler, device,
                   n_total, mask_ratio=0.25, window=2):
    """One epoch of mask reconstruction pretraining.

    Args:
        loader:       DataLoader yielding PyG Data batches (per timestep)
        model:        GNN model (use only encoder + processor)
        recon_heads:  dict {'wrf_dyn': ReconHead(270), 'clms': ReconHead(3)}
        opt:          optimizer over model + recon_heads
        scheduler:    LR scheduler
        device:       cuda device
        n_total:      nodes per graph (for compat)
        mask_ratio:   fraction of maskable positions to mask
        window:       WRF window param (default 2)

    Returns: (loss, dummy_rmse_tuple, dummy_truth, dummy_pred) — signature compat
    """
    model.train()
    for k in recon_heads:
        recon_heads[k].train()

    maskable_cols = build_maskable_cols(window=window, device=device)
    ranges = get_maskable_ranges(window=window)
    wrf_dyn_cols = torch.tensor(
        [c for s, e in ranges[:-1] for c in range(s, e)],
        dtype=torch.long, device=device)
    clms_cols = torch.tensor(list(range(*ranges[-1])), dtype=torch.long, device=device)

    total_loss = 0.0
    n_batches = 0

    for _batch in loader:
        _batch = _batch.to(device)
        x_orig = _batch.x   # (bs * n_total, 1347)

        x_masked, mask = apply_mask(x_orig, maskable_cols, mask_ratio=mask_ratio)

        # Encode (use masked input)
        hidden = forward_encoder(model, x_masked, _batch.edge_index, _batch.edge_attr)
        # hidden: (bs * n_total, HLD)

        # Decode per modality (output FULL maskable size; only masked positions contribute to loss)
        recon_wrf = recon_heads['wrf_dyn'](hidden)   # (N, 270)
        recon_clms = recon_heads['clms'](hidden)      # (N, 3)

        target_wrf = x_orig[:, wrf_dyn_cols]          # (N, 270)
        target_clms = x_orig[:, clms_cols]            # (N, 3)
        mask_wrf = mask[:, wrf_dyn_cols].float()      # (N, 270)
        mask_clms = mask[:, clms_cols].float()        # (N, 3)

        # MSE on masked positions only
        denom_wrf = mask_wrf.sum() + 1e-8
        denom_clms = mask_clms.sum() + 1e-8
        loss_wrf = ((recon_wrf - target_wrf) ** 2 * mask_wrf).sum() / denom_wrf
        loss_clms = ((recon_clms - target_clms) ** 2 * mask_clms).sum() / denom_clms

        loss = loss_wrf + loss_clms

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n_batches += 1

        if not _PT_DEBUG['printed']:
            print(f"[DEBUG/pretrain] first batch: x_orig=(N={x_orig.shape[0]}, F={x_orig.shape[1]}), "
                  f"mask total coverage={mask.float().mean().item():.4f} (expect ~ 273*0.25/1347 ≈ 0.0507)")
            print(f"[DEBUG/pretrain] WRF dyn mask: {mask_wrf.mean().item():.4f} (expect ~ 0.25), "
                  f"CLMS mask: {mask_clms.mean().item():.4f} (expect ~ 0.25)")
            print(f"[DEBUG/pretrain] loss_wrf={loss_wrf.item():.4e}, "
                  f"loss_clms={loss_clms.item():.4e}, total={loss.item():.4e}")
            assert torch.isfinite(loss), f"[ERR/pretrain] loss NaN/Inf: {loss}"
            _PT_DEBUG['printed'] = True

    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)
    # Dummy returns for compat with main loop signature
    dummy_rmse = (0.0, 0.0, 0.0, 0.0)
    dummy = np.zeros((1, n_total))
    return avg_loss, dummy_rmse, dummy, dummy

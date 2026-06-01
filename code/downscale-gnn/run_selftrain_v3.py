"""
V3 self-training — fixes V2's failure modes by:

  1. **Fixed graph from R0** (1857 nodes always; no cross-graph transfer issue)
     What changes per round: ONLY the `pseudo_select_mask` (which positions are
     supervised by pseudo-targets). The graph itself never grows/shrinks.
     → eliminates the "warm-start init RMSE degrades each round" problem.

  2. **Confidence via two configurable methods** (env: ITER_CONFIDENCE):
     - `het`        : Heteroscedastic head — model outputs (μ_ε, log σ²_ε).
                      σ² comes directly from model, no snapshot ensemble.
                      Trained with Gaussian NLL.
     - `conformal`  : 5-fold conformal calibration once at start. For each
                      candidate u: nearest-neighbor weighted residuals from
                      valid 8 stations' features → per-candidate σ.

  3. **Slower pseudo-label addition** (K=30 per round, was 50/100)
     → reduces noise injection per round, gives model time to digest.

  4. **Progressive mask, warm-start each round** (works because graph fixed).

Extensive debug at every key step:
  - Round 0 base RMSE vs B1 reference
  - Confidence distributions per group (train/valid/remaining)
  - Pseudo-label quality on V every round
  - Over-smoothing ratio every round
  - Selected vs unselected diagnostic
  - Per-round Δ (warm-start verification)
  - Confidence-method-specific diagnostics:
    · het: σ² range, well-calibrated check (residual on V vs predicted σ²)
    · conformal: per-fold residual stats, kNN bandwidth check
"""
import os
import copy
import pickle
from datetime import datetime

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import wandb

import data_semi, network_semi, utils
from run_selftrain_ead import (
    precompute_alpha_beta,
    attach_to_dataset,
    eval_ead,
    predict_eps_all,
    per_station_embedding,
    greedy_select,
)


# =========================================================================
# Setup
# =========================================================================
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')

EAD_ALPHA = int(os.environ.get('EAD_ALPHA', '1'))
EAD_BETA  = int(os.environ.get('EAD_BETA',  '0'))
EAD_LAP   = int(os.environ.get('EAD_LAP',   '1'))
LAMBDA_LAP = float(os.environ.get('LAMBDA_LAP', '0.1')) if EAD_LAP else 0.0
USE_GNN = int(os.environ.get('USE_GNN', '1'))
N_GNN   = int(os.environ.get('N_GNN', '3'))

# V3-specific
ITER_N_UNLABELED   = int(os.environ.get('ITER_N_UNLABELED', '1799'))
ITER_N_ROUNDS_MAX  = int(os.environ.get('ITER_N_ROUNDS_MAX', '15'))
ITER_K_PER_ROUND   = int(os.environ.get('ITER_K_PER_ROUND', '30'))         # smaller
ITER_BASE_EPOCHS   = int(os.environ.get('ITER_BASE_EPOCHS', '500'))
ITER_ROUND_EPOCHS  = int(os.environ.get('ITER_ROUND_EPOCHS', '200'))
ITER_WARMSTART_LR  = float(os.environ.get('ITER_WARMSTART_LR', '1e-4'))
ITER_TAU_QUANTILE  = float(os.environ.get('ITER_TAU_QUANTILE', '0.5'))
ITER_ALPHA_DIV     = float(os.environ.get('ITER_ALPHA_DIV', '1.0'))
ITER_BETA_REL      = float(os.environ.get('ITER_BETA_REL', '1.0'))
ITER_K_NN_REL      = int(os.environ.get('ITER_K_NN_REL', '5'))
ITER_LAMBDA_PSEUDO = float(os.environ.get('ITER_LAMBDA_PSEUDO', '0.3'))
ITER_CONFIDENCE    = os.environ.get('ITER_CONFIDENCE', 'het').lower()
assert ITER_CONFIDENCE in ('het', 'conformal'), \
    f"ITER_CONFIDENCE must be het|conformal, got {ITER_CONFIDENCE}"
ITER_CONFORMAL_K   = int(os.environ.get('ITER_CONFORMAL_K', '8'))    # k-NN to valid for conformal
N_FOLDS            = int(os.environ.get('ITER_N_FOLDS', '5'))

iter_tag = f"st_v3_{ITER_CONFIDENCE}_K{ITER_K_PER_ROUND}_R{ITER_N_ROUNDS_MAX}"
output_dir = os.path.join(project_root, 'log',
    f'job{job_id}_{iter_tag}_{timestamp}' if job_id else f'{iter_tag}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
os.environ['OUTPUT_DIR'] = output_dir


# =========================================================================
# Heteroscedastic GNN — outputs (μ, log_σ²) per node
# =========================================================================
class HeteroGNN(torch.nn.Module):
    """Wraps a standard GNN but its decoder outputs 2 dims: μ and log_σ²."""
    def __init__(self, modelPara):
        super().__init__()
        # build like GNN but oDim=2
        modelPara_2 = {**modelPara, 'oDim': 2}
        self.base = network_semi.GNN(modelPara_2)

    def forward(self, x, edge_index, edge_attr, return_hidden=False):
        out = self.base(x, edge_index, edge_attr, return_hidden=return_hidden)
        if return_hidden:
            y, h = out
            mu, log_var = y[..., 0:1], y[..., 1:2]
            return torch.cat([mu, log_var], dim=-1), h
        mu, log_var = out[..., 0:1], out[..., 1:2]
        return torch.cat([mu, log_var], dim=-1)


def gaussian_nll_loss(mu, log_var, y, mask=None, beta=0.5):
    """
    β-NLL loss (Seitzer et al. 2022, "On the Pitfalls of Heteroscedastic
    Uncertainty Estimation with Probabilistic Neural Networks", arXiv:2203.09168).

    Standard heteroscedastic NLL has a known failure mode: σ collapses to
    the floor because high-error samples are "absorbed" by large σ (low loss),
    but easy samples dominate, pushing σ smaller globally.

    β-NLL fixes this by re-weighting each sample's NLL by σ^β (with stop-gradient
    on σ), so the σ estimation gradient does not get distorted by the variable
    weighting.

    Args:
        mu, log_var, y : (N,) — model μ, log σ², target
        beta           : 0=plain NLL, 1=MSE-equivalent for μ; paper recommends 0.5
        mask           : optional (N,) bool to select labeled positions only
    """
    log_var = log_var.clamp(min=-7.0, max=4.0)         # standard numerical stability
    var = torch.exp(log_var)                            # σ²
    se = (y - mu) ** 2
    nll = 0.5 * se / var + 0.5 * log_var               # standard Gaussian NLL
    # β-NLL: weight by σ^β (detached: weight does NOT affect σ gradient)
    weight = (var.detach() ** (beta / 2))               # σ^β
    weighted_nll = weight * nll
    if mask is not None:
        weighted_nll = weighted_nll[mask]
    return weighted_nll.mean()


# =========================================================================
# Train epoch — supports both standard and heteroscedastic, fixed graph
# =========================================================================
def train_v3_epoch(loader, model, opt, scheduler, device,
                   nNodes, edge_src, edge_dst, edge_w,
                   use_pseudo=False, lambda_pseudo=0.0,
                   confidence_method='het'):
    """One epoch on the FIXED graph. Pseudo loss only on selected positions."""
    model.train()
    stats = dict(n=0, sup=0.0, lap=0.0, pseudo=0.0, total=0.0)
    pred_delta, truth_delta = [], []

    for _n, batch in enumerate(loader):
        batch = batch.to(device)
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        label_mask = batch.label_mask
        bs = x.shape[0] // nNodes

        out = model(x, edge_index, edge_attr)
        if confidence_method == 'het':
            eps_hat = out[..., 0:1].squeeze(-1)
            log_var = out[..., 1:2].squeeze(-1)
        else:
            eps_hat = out.squeeze(-1)
            log_var = None

        alpha_t = (batch.alpha_t.repeat_interleave(nNodes) if EAD_ALPHA
                   else torch.zeros(bs * nNodes, device=device))
        y_delta = batch.y_residual.squeeze(-1)
        beta_hat_b = (batch.beta_hat.squeeze(-1) if EAD_BETA
                      else torch.zeros_like(eps_hat))
        target_eps = y_delta - alpha_t - beta_hat_b

        # Sup loss
        if confidence_method == 'het':
            sup_loss = gaussian_nll_loss(eps_hat, log_var, target_eps, mask=label_mask)
        else:
            sup_loss = torch.nn.functional.huber_loss(eps_hat[label_mask], target_eps[label_mask])

        # Lap on ε̂ (use μ for het)
        eps_hat_g = eps_hat.reshape(bs, nNodes)
        lap_loss = torch.tensor(0.0, device=device)
        for g in range(bs):
            diff = eps_hat_g[g][edge_src] - eps_hat_g[g][edge_dst]
            lap_loss = lap_loss + torch.mean(edge_w * diff ** 2)
        lap_loss = lap_loss / bs

        # Pseudo loss on selected
        pseudo_loss = torch.tensor(0.0, device=device)
        if use_pseudo and lambda_pseudo > 0:
            pseudo_eps_b = batch.pseudo_eps.squeeze(-1)
            pseudo_sigma_b = batch.pseudo_sigma.squeeze(-1)
            sel_mask = batch.pseudo_select_mask.squeeze(-1).bool()
            if sel_mask.any():
                inv_var = 1.0 / (pseudo_sigma_b ** 2 + 1e-3)
                w_sel = inv_var[sel_mask]; w_sel = w_sel / (w_sel.mean() + 1e-8)
                diff_sq = (eps_hat[sel_mask] - pseudo_eps_b[sel_mask]) ** 2
                pseudo_loss = (w_sel * diff_sq).mean()

        total = sup_loss + LAMBDA_LAP * lap_loss + lambda_pseudo * pseudo_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

        stats['n'] += 1
        stats['sup'] += float(sup_loss.item())
        stats['lap'] += float(lap_loss.item())
        stats['pseudo'] += float(pseudo_loss.item())
        stats['total'] += float(total.item())

        delta_pred = (alpha_t + beta_hat_b + eps_hat).detach().reshape(bs, nNodes)
        delta_true = y_delta.reshape(bs, nNodes)
        lm_per = label_mask[:nNodes]
        for g in range(bs):
            pred_delta.append(delta_pred[g][lm_per].cpu().numpy())
            truth_delta.append(delta_true[g][lm_per].cpu().numpy())

    if scheduler is not None:
        scheduler.step()
    for k in ('sup', 'lap', 'pseudo', 'total'):
        stats[k] /= max(stats['n'], 1)
    rmse = utils.RMSE(np.array(truth_delta), np.array(pred_delta))
    return stats, rmse


@torch.no_grad()
def eval_v3(loader, model, device, nNodes, nNodes_labeled, tgt_scl, confidence_method):
    """Evaluation — for het, drops log_var from output."""
    model.eval()
    total_pred, total_truth = [], []
    eps_pred_list, eps_truth_list = [], []
    for batch in loader:
        batch = batch.to(device)
        bs = batch.x.shape[0] // nNodes
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        eps_hat = out[..., 0:1].squeeze(-1) if confidence_method == 'het' else out.squeeze(-1)
        alpha_t = (batch.alpha_t.repeat_interleave(nNodes) if EAD_ALPHA
                   else torch.zeros(bs * nNodes, device=device))
        beta_hat_b = (batch.beta_hat.squeeze(-1) if EAD_BETA else torch.zeros_like(eps_hat))
        y_delta = batch.y_residual.squeeze(-1)

        eps_hat_g = eps_hat.reshape(bs, nNodes)
        beta_g = beta_hat_b.reshape(bs, nNodes)
        alpha_g = alpha_t.reshape(bs, nNodes)
        y_g = y_delta.reshape(bs, nNodes)
        lm_per = batch.label_mask[:nNodes]

        delta_pred = alpha_g + beta_g + eps_hat_g
        for g in range(bs):
            total_pred.append(delta_pred[g][lm_per].cpu().numpy())
            total_truth.append(y_g[g][lm_per].cpu().numpy())
            eps_target = y_g[g][lm_per] - alpha_g[g][lm_per] - beta_g[g][lm_per]
            eps_pred_list.append(eps_hat_g[g][lm_per].cpu().numpy())
            eps_truth_list.append(eps_target.cpu().numpy())

    total_pred = np.array(total_pred); total_truth = np.array(total_truth)
    rmse_total = utils.RMSE(total_truth, total_pred)
    rmse_eps = utils.RMSE(np.array(eps_truth_list), np.array(eps_pred_list))
    metrics = utils.compute_all_metrics(total_truth, total_pred, scl=tgt_scl)
    return dict(rmse_total=rmse_total, rmse_eps=rmse_eps, **metrics)


# =========================================================================
# Predict ε̂ (and σ² for het) over full dataset on the FIXED graph
# =========================================================================
@torch.no_grad()
def predict_v3(model, dataset, device, nNodes, confidence_method, batch_size=512):
    """Returns ε̂_mean (T, N), σ² (T, N) — σ² from het output or zeros for non-het."""
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    eps_list, var_list = [], []
    for batch in loader:
        batch = batch.to(device)
        bs = batch.x.shape[0] // nNodes
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        if confidence_method == 'het':
            eps = out[..., 0:1].squeeze(-1).reshape(bs, nNodes).cpu().numpy()
            log_var = out[..., 1:2].squeeze(-1).reshape(bs, nNodes).cpu().numpy()
            log_var = np.clip(log_var, -7.0, 4.0)
            var = np.exp(log_var)
        else:
            eps = out.squeeze(-1).reshape(bs, nNodes).cpu().numpy()
            var = np.zeros_like(eps)
        eps_list.append(eps); var_list.append(var)
    return np.concatenate(eps_list, axis=0), np.concatenate(var_list, axis=0)


# =========================================================================
# 5-fold conformal calibration on TRAIN stations (NO valid leak)
# =========================================================================
def conformal_5fold_calibrate(make_model_fn, full_train_ds, train_loader, device,
                                nNodes, train_idx, true_eps_full,
                                edge_src, edge_dst, edge_w,
                                fold_epochs=200, n_folds=5, seed=42):
    """
    5-fold CV on TRAIN labeled stations only (valid 8 excluded entirely).

    For each fold: train fresh model on (50 - 10) train stations, compute residuals
    on the held-out 10 → out-of-fold residuals.

    Returns:
        train_residuals : (T, n_train) — each train station's OOF residual
                           (rows: timesteps; cols: train_idx ordering)
    """
    print(f"\n  [Conformal 5-fold] starting calibration ({n_folds} folds × {fold_epochs} ep)")
    train_arr = np.array(train_idx)
    n_train = len(train_arr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    fold_size = n_train // n_folds

    T = len(full_train_ds)
    train_residuals = np.full((T, n_train), np.nan, dtype=np.float32)

    # Save original masks once
    orig_masks = [d.label_mask.clone() for d in full_train_ds]

    for fold_k in range(n_folds):
        held_local = perm[fold_k * fold_size : (fold_k + 1) * fold_size]
        held_stations = train_arr[held_local]                 # original node IDs
        fold_train_stations = np.setdiff1d(train_arr, held_stations)

        # Override label_mask: only fold_train stations contribute sup loss
        fold_mask_per = torch.zeros(nNodes, dtype=torch.bool)
        fold_mask_per[torch.LongTensor(fold_train_stations)] = True
        for d in full_train_ds:
            d.label_mask = fold_mask_per.clone()

        # Fresh model + optimizer for this fold
        m = make_model_fn()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)

        # Train fold (uses confidence_method='conformal' = standard GNN, Huber loss)
        for ep in range(fold_epochs):
            train_v3_epoch(train_loader, m, opt, sched, device,
                           nNodes, edge_src, edge_dst, edge_w,
                           use_pseudo=False, lambda_pseudo=0.0,
                           confidence_method='conformal')

        # Predict ε̂ on full graph (on held stations, this is OUT-OF-FOLD)
        eps_pred, _ = predict_v3(m, full_train_ds, device, nNodes,
                                  confidence_method='conformal')

        # Residuals on held stations (true_eps_full has true ε)
        for i_local, s_orig in zip(held_local, held_stations):
            residual = np.abs(eps_pred[:, s_orig] - true_eps_full[:, s_orig])  # (T,)
            train_residuals[:, i_local] = residual

        held_mean = float(np.nanmean(train_residuals[:, held_local]))
        print(f"  [Fold {fold_k+1}/{n_folds}] held {len(held_stations)} stations, "
              f"OOF residual mean={held_mean:.4f}")

    # Restore original label_masks
    for d, om in zip(full_train_ds, orig_masks):
        d.label_mask = om

    print(f"  [Conformal 5-fold done] residuals shape={train_residuals.shape}, "
          f"global mean={float(np.nanmean(train_residuals)):.4f}")
    return train_residuals


# =========================================================================
# kNN-weighted conformal: σ from train OOF residuals (no valid leak)
# =========================================================================
def conformal_score(emb_full, anchor_idx, residuals_v, k=8, bandwidth=None):
    """
    Real kNN-weighted conformal calibration.

    For each station u (whole graph): σ_u(t) = weighted mean of anchor stations'
    residuals at time t, weights = softmax(−distance(emb_u, emb_anchor) / bandwidth).

    Args:
        emb_full     : (N, HLD)   — embeddings for ALL stations
        anchor_idx   : (n_anchor,) — indices of calibration anchor stations
                                     (e.g., train 50 from 5-fold OOF)
        residuals_v  : (T, n_anchor) — |ε̂_pred - ε_true| at anchors (OOF for train)
        k            : int         — top-k anchors per candidate
        bandwidth    : float       — RBF kernel; None = adaptive (median dist)
    Returns:
        sigma_per_station   : (N,)    — per-station mean σ over time
        sigma_per_t_station : (T, N)  — per-(t, station) σ
    """
    N = emb_full.shape[0]
    val_emb  = emb_full[anchor_idx]                                         # (n_anchor, HLD)
    dists = np.linalg.norm(emb_full[:, None, :] - val_emb[None, :, :], axis=-1)  # (N, n_anchor)

    # Optionally: only use top-k closest valid for each candidate (others get 0 weight)
    if k < dists.shape[1]:
        # zero-out non-top-k
        sort_idx = np.argsort(dists, axis=1)
        keep = sort_idx[:, :k]
        mask = np.zeros_like(dists, dtype=bool)
        rows = np.arange(N)[:, None]
        mask[rows, keep] = True
        dists_masked = np.where(mask, dists, np.inf)
    else:
        dists_masked = dists

    if bandwidth is None or bandwidth <= 0:
        # adaptive: median of finite top-k distances
        finite = dists_masked[np.isfinite(dists_masked)]
        bandwidth = float(np.median(finite)) if len(finite) > 0 else 1.0
        bandwidth = max(bandwidth, 1e-6)

    weights = np.exp(-dists_masked / bandwidth)                              # (N, n_v); inf → 0
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

    # σ_u(t) = Σ_v weights[u, v] · residuals_v[t, v]
    # residuals_v: (T, n_v); weights.T: (n_v, N) → result: (T, N)
    sigma_full = residuals_v @ weights.T                                     # (T, N)
    sigma_per_station = sigma_full.mean(axis=0)                              # (N,)

    print(f"  [Conformal] bandwidth={bandwidth:.4f}, k={k}")
    print(f"  [Conformal] residuals_v: mean={residuals_v.mean():.4f}, "
          f"std={residuals_v.std():.4f}, max={residuals_v.max():.4f}")
    print(f"  [Conformal] σ per-station: min={sigma_per_station.min():.4f}, "
          f"mean={sigma_per_station.mean():.4f}, max={sigma_per_station.max():.4f}")
    return sigma_per_station, sigma_full


# =========================================================================
# Main pipeline
# =========================================================================
def main():
    log_path = os.path.join(output_dir, f'selftrain_v3_{ITER_CONFIDENCE}_log')
    with open(log_path, 'w') as f:
        f.write(f"=== V3 self-training (confidence={ITER_CONFIDENCE}) ===\n")
        f.write(f"K_PER_ROUND={ITER_K_PER_ROUND}, MAX_ROUNDS={ITER_N_ROUNDS_MAX}\n")
        f.write(f"BASE_EP={ITER_BASE_EPOCHS}, ROUND_EP={ITER_ROUND_EPOCHS} (warm-start)\n")
        f.write(f"τ={ITER_TAU_QUANTILE}, λ_pseudo={ITER_LAMBDA_PSEUDO}, "
                f"warm_lr={ITER_WARMSTART_LR}\n")

    # Load data ONCE (full graph: 1857 nodes throughout)
    print(f"\n# Loading FULL dataset (N_UNLABELED={ITER_N_UNLABELED}, fixed graph)")
    dataParam = {
        'geoMethod': 'average', 'nCompPCA': 40, 'window': 2,
        'poolSize': int(os.environ.get('POOL_SIZE', '12')),
        'batchSize': 512, 'thres': 0.1,
        'geoFeatures': 'full', 'n_unlabeled': ITER_N_UNLABELED,
    }
    os.environ.setdefault('EVAL_MODE', 'spatial')
    os.environ.setdefault('USE_FPS', '2')
    os.environ.setdefault('EDGE_MODE', 'no_uu')
    os.environ.setdefault('NORM_MODE', 'global')

    trainL, validL, meta, _ = data_semi.dataGen(dataParam, path)
    full_train_ds = trainL.dataset
    full_valid_ds = validL.dataset
    nNodes = meta['nNodes']
    nNodes_labeled = meta['nNodes_labeled']
    train_idx = np.array(meta['train_station_idx'])
    valid_idx = np.array(meta['valid_station_idx'])
    tgt_scl = float(meta.get('tgt_global_scl', 1.0))
    print(f"  nNodes={nNodes}, train={len(train_idx)}, valid={len(valid_idx)}, "
          f"unlabeled={nNodes - nNodes_labeled}")

    alpha_per_sample, beta_hat, _ = precompute_alpha_beta(trainL, meta)
    attach_to_dataset(full_train_ds, alpha_per_sample, beta_hat)
    attach_to_dataset(full_valid_ds, alpha_per_sample, beta_hat)

    Adj = meta['AdjMatrix']
    src, dst = np.nonzero(Adj)
    edge_src = torch.LongTensor(src).to(device)
    edge_dst = torch.LongTensor(dst).to(device)
    edge_w   = torch.FloatTensor(Adj[src, dst]).to(device)

    all_unlabeled = set(range(nNodes_labeled, nNodes))
    selected_so_far = set()

    modelParam = {
        'HLD': 128, 'nMLP': 2, 'nGNN': N_GNN, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'iDim': meta['iDim'], 'oDim': 1,
        'BN': False, 'Dropout': True, 'conv_type': conv_type,
        'use_gnn': bool(USE_GNN),
    }

    def make_model():
        if ITER_CONFIDENCE == 'het':
            return HeteroGNN(modelParam).to(device)
        return network_semi.GNN(modelParam).to(device)

    modelName = f"selftrain_v3_{ITER_CONFIDENCE}"
    wandb.init(
        entity="urban_prediction", project="Semi-supervised GNN",
        name=f"{modelName}_job{job_id}" if job_id else modelName,
        config={**modelParam, 'CONFIDENCE': ITER_CONFIDENCE,
                'K_PER_ROUND': ITER_K_PER_ROUND, 'N_ROUNDS_MAX': ITER_N_ROUNDS_MAX,
                'LAMBDA_PSEUDO': ITER_LAMBDA_PSEUDO,
                'EAD_ALPHA': EAD_ALPHA, 'LAMBDA_LAP': LAMBDA_LAP,
                'variant': 'v3'},
    )

    # Pre-extract y_residual once
    print("  Pre-extracting y_residual for diagnostics...")
    all_y_residual = np.zeros((len(full_train_ds), nNodes), dtype=np.float32)
    for t, d in enumerate(full_train_ds):
        all_y_residual[t] = d.y_residual.squeeze(-1).cpu().numpy()
    true_eps_full = all_y_residual - alpha_per_sample[:, None] - beta_hat[None, :]

    train_loader = PyGDataLoader(full_train_ds, batch_size=512, shuffle=True)
    valid_loader = PyGDataLoader(full_valid_ds, batch_size=512, shuffle=False)

    # =====================================================================
    # ROUND 0 — train base on FIXED full graph (1857 nodes)
    # =====================================================================
    print("\n" + "#"*72)
    print(f"# ROUND 0 — base training on FIXED graph ({nNodes} nodes, no pseudo)")
    print("#"*72)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)

    best_r0_rmse = float('inf')
    best_r0_state = None
    for epoch in range(ITER_BASE_EPOCHS):
        stats, train_rmse = train_v3_epoch(
            train_loader, model, opt, sched, device,
            nNodes, edge_src, edge_dst, edge_w, use_pseudo=False,
            confidence_method=ITER_CONFIDENCE)
        em = eval_v3(valid_loader, model, device, nNodes, nNodes_labeled, tgt_scl,
                     ITER_CONFIDENCE)
        v_rmse = em['rmse_norm']
        if v_rmse < best_r0_rmse:
            best_r0_rmse = v_rmse
            best_r0_state = copy.deepcopy(model.state_dict())
        if epoch % 25 == 0 or epoch < 3:
            line = (f"R0 ep{epoch:4d}: train={train_rmse[0]:.4f}  "
                    f"valid={v_rmse:.4f} ({em['rmse_C']:.3f}°C)  "
                    f"sup={stats['sup']:.4e}  lap={stats['lap']:.4e}")
            print(line)
            with open(log_path, 'a') as f: f.write(line + '\n')
        wandb.log({'round': 0, 'epoch': epoch,
                   'r0/train_rmse': train_rmse[0], 'r0/valid_rmse': v_rmse,
                   'r0/best_rmse': best_r0_rmse})

    print(f"\n[Round 0 done] best_valid_RMSE = {best_r0_rmse:.4f}")
    with open(log_path, 'a') as f: f.write(f"[R0 done] best={best_r0_rmse:.4f}\n")
    model.load_state_dict(best_r0_state)
    overall_best = {'round': 0, 'rmse': best_r0_rmse, 'state': best_r0_state}

    # =====================================================================
    # 5-fold conformal calibration (only for ITER_CONFIDENCE=conformal)
    # =====================================================================
    train_residuals_oof = None
    if ITER_CONFIDENCE == 'conformal':
        # Use SHORTER fold trainings (200 ep) — calibration only, not need full convergence
        train_residuals_oof = conformal_5fold_calibrate(
            make_model_fn=make_model,
            full_train_ds=full_train_ds,
            train_loader=train_loader,
            device=device,
            nNodes=nNodes, train_idx=train_idx,
            true_eps_full=true_eps_full,
            edge_src=edge_src, edge_dst=edge_dst, edge_w=edge_w,
            fold_epochs=200, n_folds=N_FOLDS, seed=42,
        )
        # Save for analysis
        np.save(os.path.join(output_dir, 'train_residuals_oof.npy'), train_residuals_oof)
        print(f"  [Conformal] saved train OOF residuals to {output_dir}/train_residuals_oof.npy")
        # Restore best_r0_state on the main model (folds messed up label_mask but model state is separate)
        model.load_state_dict(best_r0_state)

    # Pre-compute embeddings (used by selection)
    # For het, use base_gnn's hidden; for std, normal GNN
    print("  Pre-computing per-station embeddings ...")
    if ITER_CONFIDENCE == 'het':
        emb_per_st = per_station_embedding(model.base, full_train_ds, device, nNodes)
    else:
        emb_per_st = per_station_embedding(model, full_train_ds, device, nNodes)

    round_results = []
    cand_idx_global = np.arange(nNodes_labeled, nNodes)

    # =====================================================================
    # ROUND 1..N — selection + warm-start retrain
    # =====================================================================
    for round_k in range(1, ITER_N_ROUNDS_MAX + 1):
        print("\n" + "#"*72)
        print(f"# ROUND {round_k} (V3 fixed-graph, confidence={ITER_CONFIDENCE})")
        print("#"*72)

        remaining = sorted(all_unlabeled - selected_so_far)
        if len(remaining) < ITER_K_PER_ROUND:
            print(f"  [STOP] only {len(remaining)} remaining."); break

        # Generate ε̂ + σ² over full graph (using best previous model)
        print(f"  Predicting on full graph with best_R{round_k-1} state...")
        if round_k > 1:
            model.load_state_dict(overall_best['state'])
        ensemble_mean, ensemble_var = predict_v3(model, full_train_ds, device, nNodes, ITER_CONFIDENCE)

        # ---- Confidence per station (per ITER_CONFIDENCE) ----
        if ITER_CONFIDENCE == 'het':
            # σ from heteroscedastic head (per-(t,u))
            ensemble_std = np.sqrt(ensemble_var)             # (T, N)
            conf_per_station = -ensemble_std.mean(axis=0)     # (N,)
            print(f"  [HET σ] mean={ensemble_std.mean():.4f}, std={ensemble_std.std():.4f}, "
                  f"max={ensemble_std.max():.4f}")
            print(f"  [HET σ on V (het well-calibrated check)] "
                  f"mean σ on valid stations = {ensemble_std[:, valid_idx].mean():.4f}, "
                  f"actual residual std on V = {(true_eps_full - ensemble_mean)[:, valid_idx].std():.4f}")

        else:    # conformal
            # Real 5-fold conformal: use train 50 OOF residuals (NO valid leak)
            print(f"  [CONFORMAL] kNN σ from train OOF residuals (k={ITER_CONFORMAL_K}, "
                  f"5-fold calibration)")
            assert train_residuals_oof is not None, "5-fold calibration not run"

            # train_residuals_oof: (T, n_train) — each train station's OOF residual from R0 5-fold
            print(f"  [CONFORMAL] OOF residual stats: mean={np.nanmean(train_residuals_oof):.4f}, "
                  f"std={np.nanstd(train_residuals_oof):.4f}")

            # kNN-weighted σ using train 50 as anchors
            sigma_per_station, sigma_full = conformal_score(
                emb_full=emb_per_st,
                anchor_idx=train_idx,
                residuals_v=train_residuals_oof,
                k=ITER_CONFORMAL_K,
            )
            ensemble_std = sigma_full.copy()
            conf_per_station = -sigma_per_station

            # === Diagnostic B: conformal calibration cross-check ===
            # Predict σ on valid 8 (held-out, NOT used in calibration) → see if σ ≈ actual residual
            actual_resid_v = np.abs(ensemble_mean[:, valid_idx] - true_eps_full[:, valid_idx])
            predicted_sigma_v = sigma_full[:, valid_idx]
            print(f"  [Conformal calibration check on valid 8]")
            print(f"    predicted σ on valid: mean={predicted_sigma_v.mean():.4f}, "
                  f"std={predicted_sigma_v.std():.4f}")
            print(f"    actual residual on valid: mean={actual_resid_v.mean():.4f}, "
                  f"std={actual_resid_v.std():.4f}")
            ratio = predicted_sigma_v.mean() / (actual_resid_v.mean() + 1e-8)
            print(f"    ratio (predicted/actual) = {ratio:.3f}  (≈ 1 → well-calibrated)")

        # ---- Common diagnostics ----
        print(f"  [Confidence breakdown] train mean={conf_per_station[train_idx].mean():+.4f}, "
              f"valid mean={conf_per_station[valid_idx].mean():+.4f}, "
              f"remaining mean={conf_per_station[np.array(remaining)].mean():+.4f}")

        # Pseudo-label quality on V
        pseudo_rmse_v = float(np.sqrt(
            ((ensemble_mean[:, valid_idx] - true_eps_full[:, valid_idx]) ** 2).mean(axis=0)
        ).mean())
        print(f"  [Pseudo QUALITY on V] = {pseudo_rmse_v:.4f}  (R0 best = {best_r0_rmse:.4f})")
        warns = []
        if pseudo_rmse_v > 1.5 * best_r0_rmse:
            warns.append(f"Pseudo on V > 1.5× R0 best ({pseudo_rmse_v:.4f})")

        # Over-smoothing
        rem_arr = np.array(remaining)
        ps_std_lab = ensemble_mean[:, np.concatenate([train_idx, valid_idx])].std(axis=1).mean()
        ps_std_rem = ensemble_mean[:, rem_arr].std(axis=1).mean()
        smooth_ratio = ps_std_rem / (ps_std_lab + 1e-8)
        print(f"  [Over-smoothing] ratio rem/lab = {smooth_ratio:.3f}")
        if smooth_ratio < 0.5:
            warns.append(f"Smooth ratio = {smooth_ratio:.2f} < 0.5")

        # === Diagnostic A: filter pass rate ===
        tau_thresh = np.quantile(conf_per_station[rem_arr], ITER_TAU_QUANTILE)
        n_pass = int((conf_per_station[rem_arr] > tau_thresh).sum())
        print(f"  [Filter] τ_conf = {tau_thresh:+.4f}, "
              f"{n_pass}/{len(rem_arr)} candidates pass τ filter "
              f"({100*n_pass/max(len(rem_arr),1):.1f}%)")

        # ---- Selection ----
        new_picks = greedy_select(
            emb_all=emb_per_st,
            candidate_idx=rem_arr,
            valid_idx=valid_idx,
            confidence_per_station=conf_per_station,
            K=ITER_K_PER_ROUND,
            alpha=ITER_ALPHA_DIV, beta=ITER_BETA_REL,
            k_nn_rel=ITER_K_NN_REL,
            tau_quantile=ITER_TAU_QUANTILE,
            ablation='full',
        )
        if len(new_picks) < ITER_K_PER_ROUND:
            print(f"  [AUTO-STOP] picked only {len(new_picks)}/{ITER_K_PER_ROUND}")
            if new_picks: selected_so_far.update(new_picks)
            break
        selected_so_far.update(new_picks)
        print(f"  [Selection] picked {len(new_picks)}, |S_total|={len(selected_so_far)}")

        # === Diagnostic C: σ on selected vs unselected ===
        sel_arr = np.array(new_picks)
        unsel_arr = np.array([u for u in rem_arr if u not in set(new_picks)])
        if ITER_CONFIDENCE == 'het':
            sel_sigma   = ensemble_std[:, sel_arr].mean()
            unsel_sigma = ensemble_std[:, unsel_arr].mean() if len(unsel_arr) > 0 else 0
        else:
            sel_sigma   = sigma_per_station[sel_arr].mean()
            unsel_sigma = sigma_per_station[unsel_arr].mean() if len(unsel_arr) > 0 else 0
        print(f"  [σ check] selected mean σ={sel_sigma:.4f}, unselected mean σ={unsel_sigma:.4f}, "
              f"Δ={sel_sigma - unsel_sigma:+.4f} (want < 0: selected = lower σ = more confident)")

        # ---- Update pseudo_select_mask (graph DOES NOT change) ----
        select_mask_full = np.zeros(nNodes, dtype=bool)
        for s in selected_so_far:
            select_mask_full[s] = True

        attach_to_dataset(full_train_ds, alpha_per_sample, beta_hat,
                          pseudo_eps_all=ensemble_mean,
                          pseudo_sigma_all=ensemble_std,
                          pseudo_select_mask=select_mask_full)
        # Valid loader: no pseudo (pure eval) — re-attach without pseudo
        attach_to_dataset(full_valid_ds, alpha_per_sample, beta_hat)

        # ---- Warm-start training ----
        print(f"\n  Warm-starting from R{round_k-1} best (LR={ITER_WARMSTART_LR:.0e})")
        model.load_state_dict(overall_best['state'])
        em_init = eval_v3(valid_loader, model, device, nNodes, nNodes_labeled, tgt_scl,
                          ITER_CONFIDENCE)
        init_rmse = em_init['rmse_norm']
        print(f"  [V3 SANITY] init valid_RMSE on FIXED graph = {init_rmse:.4f}  "
              f"(should == R{round_k-1} best = {overall_best['rmse']:.4f})")

        opt = torch.optim.Adam(model.parameters(), lr=ITER_WARMSTART_LR)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)

        best_r_rmse = init_rmse
        best_r_state = copy.deepcopy(model.state_dict())
        for epoch in range(ITER_ROUND_EPOCHS):
            stats, train_rmse = train_v3_epoch(
                train_loader, model, opt, sched, device,
                nNodes, edge_src, edge_dst, edge_w,
                use_pseudo=True, lambda_pseudo=ITER_LAMBDA_PSEUDO,
                confidence_method=ITER_CONFIDENCE)
            em = eval_v3(valid_loader, model, device, nNodes, nNodes_labeled, tgt_scl,
                         ITER_CONFIDENCE)
            v_rmse = em['rmse_norm']
            if v_rmse < best_r_rmse:
                best_r_rmse = v_rmse
                best_r_state = copy.deepcopy(model.state_dict())
            if epoch % 25 == 0 or epoch < 3:
                line = (f"R{round_k} ep{epoch:4d}: train={train_rmse[0]:.4f} "
                        f"valid={v_rmse:.4f}  sup={stats['sup']:.4e} "
                        f"pseudo={stats['pseudo']:.4e} lap={stats['lap']:.4e}")
                print(line)
                with open(log_path, 'a') as f: f.write(line + '\n')
            wandb.log({'round': round_k, 'epoch': epoch,
                       f'r{round_k}/train_rmse': train_rmse[0],
                       f'r{round_k}/valid_rmse': v_rmse,
                       f'r{round_k}/best_rmse': best_r_rmse,
                       f'r{round_k}/sup': stats['sup'],
                       f'r{round_k}/pseudo': stats['pseudo'],
                       'meta/n_selected': len(selected_so_far)})

        delta_init = best_r_rmse - init_rmse
        print(f"\n[Round {round_k} done] best={best_r_rmse:.4f}, "
              f"init={init_rmse:.4f}, Δ={delta_init:+.4f}")
        if best_r_rmse < overall_best['rmse']:
            overall_best = {'round': round_k, 'rmse': best_r_rmse, 'state': best_r_state}

        round_results.append({
            'round': round_k, 'best_rmse': best_r_rmse, 'init_rmse': init_rmse,
            'n_selected': len(selected_so_far), 'pseudo_rmse_v': pseudo_rmse_v,
            'smooth_ratio': float(smooth_ratio), 'warns': warns,
        })

        # Update embeddings for next round (use updated model)
        if ITER_CONFIDENCE == 'het':
            emb_per_st = per_station_embedding(model.base, full_train_ds, device, nNodes)
        else:
            emb_per_st = per_station_embedding(model, full_train_ds, device, nNodes)

        # Save checkpoint
        torch.save({'overall_best_round': overall_best['round'],
                    'overall_best_rmse': overall_best['rmse'],
                    'gnn_state_dict': overall_best['state'],
                    'selected_so_far': sorted(selected_so_far),
                    'round_results': round_results},
                   os.path.join(output_dir, f'{modelName}.pt'))

    # ---- FINAL ----
    print("\n" + "="*72)
    print(f"  V3 FINAL ({ITER_CONFIDENCE})")
    print("="*72)
    print(f"  Pseudo-labels: {len(selected_so_far)}; Rounds: {len(round_results)}")
    print(f"  R0 best: {best_r0_rmse:.4f}")
    print(f"  Overall: R{overall_best['round']} = {overall_best['rmse']:.4f}")
    print(f"  Reference B1: 0.0428")
    print(f"  {'Round':>6} {'#PL':>5} {'init':>8} {'best':>8} {'Δ':>7} {'pseudo_v':>9}")
    print(f"  {'R0':>6} {'0':>5} {'-':>8} {best_r0_rmse:>8.4f} {'-':>7} {'-':>9}")
    for r in round_results:
        d = r['best_rmse'] - r['init_rmse']
        w = ' ⚠' if r['warns'] else ''
        print(f"  {'R'+str(r['round']):>6} {r['n_selected']:>5} "
              f"{r['init_rmse']:>8.4f} {r['best_rmse']:>8.4f} {d:>+7.4f} {r['pseudo_rmse_v']:>9.4f}{w}")
    print("="*72)
    with open(log_path, 'a') as f:
        f.write(f"\nFINAL: R0={best_r0_rmse:.4f}, overall=R{overall_best['round']}={overall_best['rmse']:.4f}\n")


if __name__ == '__main__':
    main()

"""
Empirically-Anchored Decomposition (EAD) training.

Decomposition:    Δ(i, t) = α_t  +  β_i  +  ε(i, t)
                              ↑       ↑       ↑
                         cross-station time-mean GNN learns
                         station mean  of Δ-α_t  residual

Environment variables (layer on top of NORM_MODE=global, SEMI_MODE=laplacian,
EDGE_MODE=no_uu, USE_FPS=2, N_UNLABELED=400, CONV_TYPE=graphconv, LAMBDA_LAP=0.1):

    EAD_ALPHA      = 1   -> subtract α_t from target (B1: anchor)
    EAD_BETA       = 1   -> use Kriging-interpolated β̂ in decomposition (B2)
                            (no learned MLP — β̂ is precomputed once from train labeled)
    EAD_ZERO_MEAN  = 1   -> add λ·(mean_i ε̂)² loss                (B3)
    LAMBDA_ZM      = 1.0 -> weight of zero-mean loss               (B3)
    USE_GNN        = 0/1 -> disable GNN processor (B4)
    LAMBDA_LAP     = 0.1 -> Laplacian on ε̂ (always on, like baseline)

Baseline reference:  13020 (Lap + Residual + Global norm + no_uu) = 0.0441
"""
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import wandb

import data_semi, network_semi, utils

# =========================================================================
# Setup
# =========================================================================
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')

EAD_ALPHA     = int(os.environ.get('EAD_ALPHA', '1'))
EAD_BETA      = int(os.environ.get('EAD_BETA', '0'))
EAD_ZERO_MEAN = int(os.environ.get('EAD_ZERO_MEAN', '0'))
EAD_LAP       = int(os.environ.get('EAD_LAP', '1'))      # 1 = Laplacian on ε̂, 0 = off
EAD_PSEUDO    = int(os.environ.get('EAD_PSEUDO', '0'))   # 1 = Kriging pseudo-label loss on unlabeled ε̂
EAD_MASK_RECON = int(os.environ.get('EAD_MASK_RECON', '0'))  # 1 = mask-reconstruct training (IGNNK-style)
MASK_K        = int(os.environ.get('MASK_K', '10'))        # how many train labeled to mask per batch
LAMBDA_LAP    = float(os.environ.get('LAMBDA_LAP', '0.1'))
if EAD_LAP == 0:
    LAMBDA_LAP = 0.0
LAMBDA_ZM     = float(os.environ.get('LAMBDA_ZM', '1.0'))
LAMBDA_PSD    = float(os.environ.get('LAMBDA_PSD', '0.01'))   # pseudo-label loss weight
LAMBDA_RECON  = float(os.environ.get('LAMBDA_RECON', '1.0'))   # mask-reconstruct loss weight
USE_GNN       = int(os.environ.get('USE_GNN', '1'))
N_GNN         = int(os.environ.get('N_GNN', '3'))
n_unlabeled   = int(os.environ.get('N_UNLABELED', '400'))

ead_tag = f"ead_a{EAD_ALPHA}_b{EAD_BETA}_z{EAD_ZERO_MEAN}_p{EAD_PSEUDO}"
output_dir = os.path.join(project_root, 'log', f'job{job_id}_{ead_tag}_{timestamp}' if job_id else f'{ead_tag}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
os.environ['OUTPUT_DIR'] = output_dir


# =========================================================================
# Kriging-style interpolation of β_i from train labeled stations
#   β̂_i = Σ_{j ∈ train} w_ij · β_train_j  /  Σ w_ij
#   weights w_ij come from the adjacency matrix (distance + feature similarity)
# =========================================================================
def kriging_beta(Adj, beta_train_values, train_idx, nNodes):
    """
    Args:
        Adj: (nNodes, nNodes) adjacency matrix (already built with spatial + similarity)
        beta_train_values: (n_train,) true β for train labeled stations
        train_idx: indices of train stations within the full node list
        nNodes: total node count (458)

    Returns:
        beta_hat: (nNodes,) with:
          - β_true at train stations
          - spatially-interpolated β at valid + unlabeled stations
    """
    W = Adj[:, train_idx]                          # (nNodes, n_train)
    num = W @ beta_train_values                    # (nNodes,)
    denom = W.sum(axis=1) + 1e-8
    beta_hat = num / denom
    # For train stations themselves, use the true β (don't re-interpolate)
    beta_hat[train_idx] = beta_train_values
    return beta_hat.astype(np.float32)


def kriging_pseudo_epsilon(Adj, y_residual_all, alpha_per_sample, beta_hat,
                           train_idx, nNodes, nNodes_labeled):
    """
    Generate Kriging pseudo-labels for ε at every timestep, for every node.
      ε_true(t, i) = Δ(t, i) − α_t − β_hat_i   (on train labeled, known)
      ε_pseudo(t, u) = Σ_{j ∈ train} w_uj · ε_true(t, j)  /  Σ w_uj   (Kriging interp)

    Uncertainty per node: σ²_i ∝ 1 / Σ w_ij   (farther/fewer neighbors → higher σ²)
      Normalized so mean(σ²) = 1 to avoid scale issues with LAMBDA_PSD.

    Returns:
        pseudo_eps: (nSamples, nNodes) pseudo-label for ε at every time/node
        sigma2:     (nNodes,) static uncertainty per node
    """
    nSamples = y_residual_all.shape[0]

    # Adjacency slice: weights from each node to train labeled nodes
    W = Adj[:, train_idx].astype(np.float64)              # (nNodes, n_train)
    W_sum = W.sum(axis=1) + 1e-8                          # (nNodes,)
    W_norm = W / W_sum[:, None]                           # row-normalized (nNodes, n_train)

    # ε_true at train labeled, for every timestep
    eps_train = y_residual_all[:, train_idx] \
                - alpha_per_sample[:, None] \
                - beta_hat[train_idx][None, :]            # (nSamples, n_train)

    # Kriging interpolation at every node
    pseudo_eps = eps_train @ W_norm.T                     # (nSamples, nNodes)

    # Uncertainty: inverse of total connection weight
    sigma2 = 1.0 / W_sum
    sigma2 = sigma2 / sigma2.mean()                       # normalize mean → 1

    # Diagnostic printout
    # Partition by node type
    unlabeled_idx = np.arange(nNodes_labeled, nNodes)
    eps_true_std = eps_train.std()
    eps_pseudo_unlabeled_std = pseudo_eps[:, unlabeled_idx].std()
    print("\n" + "="*72)
    print("  Kriging ε pseudo-labels")
    print("="*72)
    print(f"  ε_true (train labeled) std over all t: {eps_true_std:.4f}")
    print(f"  ε_pseudo (unlabeled) std over all t:    {eps_pseudo_unlabeled_std:.4f}")
    print(f"  ratio pseudo/true:                      {eps_pseudo_unlabeled_std/(eps_true_std+1e-8):.3f}")
    print(f"  (ratio ≈ 1 → preserved diversity; << 1 → over-smoothed like GSR 13006)")
    print(f"  σ² (unlabeled): mean={sigma2[unlabeled_idx].mean():.3f}, "
          f"std={sigma2[unlabeled_idx].std():.3f}, "
          f"range=[{sigma2[unlabeled_idx].min():.3f}, {sigma2[unlabeled_idx].max():.3f}]")
    print(f"  Per-node inverse weight: high σ² → far from any train labeled")
    print("="*72 + "\n")

    return pseudo_eps.astype(np.float32), sigma2.astype(np.float32)


# =========================================================================
# α_t and β_i precomputation — uses ONLY train labeled stations (no leak)
# =========================================================================
def precompute_alpha_beta(trainLoader, validLoader, metadata):
    """
    α_t(t)  = mean_{i ∈ train labeled} [y_residual(i, t)]           → (nSamples,)
    β_true  = mean_t [y_residual(i, t) - α_t]   for train labeled   → (nNodes,)
               (other nodes start at 0)
    β_hat   = Kriging interpolation from β_true[train] to all nodes → (nNodes,)
               (train nodes keep β_true; valid+unlabeled get spatial interp)

    Returns (alpha_per_sample, beta_hat, beta_true_valid_diag).
    """
    nNodes = metadata['nNodes']
    train_station_idx = metadata.get('train_station_idx')
    valid_station_idx = metadata.get('valid_station_idx')
    assert train_station_idx is not None, "EAD requires EVAL_MODE=spatial (train_station_idx populated)."

    # Build label_mask for train-only labeled stations (50 of 58)
    train_only_mask = np.zeros(nNodes, dtype=bool)
    train_only_mask[train_station_idx] = True

    # Collect all y_residual samples from training set (ordered by sample index)
    dataset = trainLoader.dataset
    nSamples = len(dataset)
    all_delta = np.zeros((nSamples, nNodes), dtype=np.float32)
    for t, d in enumerate(dataset):
        all_delta[t] = d.y_residual.squeeze(-1).cpu().numpy()

    # α_t: mean over 50 train labeled stations at each timestep
    alpha_per_sample = all_delta[:, train_only_mask].mean(axis=1)        # (nSamples,)

    # Δ - α_t at train labeled stations → average across time → β_i for each train station
    delta_minus_alpha = all_delta - alpha_per_sample[:, None]            # (nSamples, nNodes)
    beta_true = np.zeros(nNodes, dtype=np.float32)
    beta_true[train_station_idx] = delta_minus_alpha[:, train_only_mask].mean(axis=0)
    # Center so that mean over train stations = 0 (identifiability constraint for the model)
    train_beta_mean = beta_true[train_station_idx].mean()
    beta_true[train_station_idx] -= train_beta_mean

    # [Diagnostic only — NOT used for training] β_i for valid 8 stations (for Kriging accuracy check)
    beta_true_valid_diag = np.zeros(nNodes, dtype=np.float32)
    beta_true_valid_diag[valid_station_idx] = delta_minus_alpha[:, valid_station_idx].mean(axis=0) - train_beta_mean

    # ---- Kriging: spatially interpolate β̂ for valid + unlabeled from train labeled ----
    Adj = metadata['AdjMatrix']
    beta_train_vals = beta_true[train_station_idx]
    beta_hat = kriging_beta(Adj, beta_train_vals, train_station_idx, nNodes)

    # Sanity: check Kriging accuracy on valid stations (we secretly know β_true there)
    kriging_valid_rmse = np.sqrt(np.mean(
        (beta_hat[valid_station_idx] - beta_true_valid_diag[valid_station_idx]) ** 2
    ))
    true_valid_std = float(beta_true_valid_diag[valid_station_idx].std()) + 1e-8
    print(f"  β̂ Kriging:  valid_RMSE={kriging_valid_rmse:.4f} "
          f"(β_true_valid std={true_valid_std:.4f}, RMSE/std={kriging_valid_rmse/true_valid_std:.2f})")
    print(f"  β̂ Kriging:  β̂_unlabeled mean={beta_hat[58:].mean():+.4f}, std={beta_hat[58:].std():.4f}, "
          f"range=[{beta_hat[58:].min():+.4f}, {beta_hat[58:].max():+.4f}]")
    if kriging_valid_rmse / true_valid_std > 1.0:
        print(f"  ⚠ Kriging β̂ on valid is worse than predicting zero "
              f"(ratio={kriging_valid_rmse/true_valid_std:.2f}). Consider alternatives.")
    else:
        print(f"  ✓ Kriging β̂ on valid is better than zero baseline.")
    print(f"  β̂ (valid 8 stations, diag): "
          f"mean={beta_true_valid_diag[valid_station_idx].mean():+.4f}, "
          f"std={beta_true_valid_diag[valid_station_idx].std():.4f}, "
          f"range=[{beta_true_valid_diag[valid_station_idx].min():+.4f}, "
          f"{beta_true_valid_diag[valid_station_idx].max():+.4f}]")

    # ---- Diagnostic printout ----
    print("\n" + "="*72)
    print("  EAD precomputation (α_t and β_i from TRAIN labeled stations only)")
    print("="*72)
    print(f"  Train labeled stations:  {len(train_station_idx)}  (indices: {sorted(train_station_idx.tolist())[:5]}...)")
    print(f"  Valid labeled stations:  {len(valid_station_idx)}  (held out — NOT used for α_t/β_i)")
    print(f"  Total timesteps:         {nSamples}")
    print(f"  α_t:  mean={alpha_per_sample.mean():+.4f}, std={alpha_per_sample.std():.4f}, "
          f"range=[{alpha_per_sample.min():+.4f}, {alpha_per_sample.max():+.4f}]")
    beta_tr = beta_true[train_station_idx]
    print(f"  β_i (train stations):  mean={beta_tr.mean():+.4f} (≈0 after centering), "
          f"std={beta_tr.std():.4f}, range=[{beta_tr.min():+.4f}, {beta_tr.max():+.4f}]")

    # Variance breakdown (should match Stage A4)
    total_var = (all_delta[:, train_only_mask]).var()
    alpha_mat = np.broadcast_to(alpha_per_sample[:, None], (nSamples, train_only_mask.sum()))
    beta_mat  = np.broadcast_to(beta_true[train_only_mask][None, :], (nSamples, train_only_mask.sum()))
    eps_mat   = all_delta[:, train_only_mask] - alpha_mat - beta_mat
    alpha_share = alpha_mat.var() / total_var * 100
    beta_share  = beta_mat.var()  / total_var * 100
    eps_share   = eps_mat.var()   / total_var * 100
    print(f"  Variance decomposition of Δ (train labeled):")
    print(f"    α_t share: {alpha_share:6.2f}%   (Stage A expected ≈ 68%)")
    print(f"    β_i share: {beta_share:6.2f}%   (Stage A expected ≈ 9%)")
    print(f"    ε   share: {eps_share:6.2f}%   (Stage A expected ≈ 23%, GNN learns this)")
    # Sanity check: warn if numbers deviate materially from Stage A
    if abs(alpha_share - 68) > 10 or abs(beta_share - 9) > 5 or abs(eps_share - 23) > 10:
        print(f"  ⚠ WARNING: variance shares deviate from Stage A — "
              f"check train/valid station mask leakage or EVAL_MODE=spatial.")
    else:
        print(f"  ✓ Variance shares match Stage A → α_t/β_i computed correctly.")
    print("="*72 + "\n")

    # ---- Kriging pseudo-ε (only if enabled) ----
    pseudo_eps_all, sigma2_all = None, None
    if EAD_PSEUDO:
        pseudo_eps_all, sigma2_all = kriging_pseudo_epsilon(
            Adj, all_delta, alpha_per_sample, beta_hat,
            train_station_idx, nNodes, nNodes_labeled=58
        )
    return alpha_per_sample, beta_hat, beta_true_valid_diag, pseudo_eps_all, sigma2_all


# =========================================================================
# Attach α_t and β_i_true to each Data object (so batching is natural)
# =========================================================================
def attach_alpha_beta_to_dataset(dataset, alpha_per_sample, beta_hat, metadata,
                                 pseudo_eps_all=None, sigma2_all=None):
    """
    Attach α_t (per-graph scalar), β̂ (static per-node), and optional pseudo-ε (per-graph per-node)
    to Data objects. Default values are the same for every timestep.
    """
    nNodes = metadata['nNodes']
    beta_hat_t = torch.FloatTensor(beta_hat).unsqueeze(-1)                # (nNodes, 1)
    sigma2_t   = torch.FloatTensor(sigma2_all).unsqueeze(-1) if sigma2_all is not None else None
    for t, d in enumerate(dataset):
        d.alpha_t  = torch.FloatTensor([alpha_per_sample[t]])             # scalar per graph
        d.beta_hat = beta_hat_t.clone()                                   # (nNodes, 1)
        if pseudo_eps_all is not None:
            d.pseudo_eps = torch.FloatTensor(pseudo_eps_all[t]).unsqueeze(-1)   # (nNodes, 1) — per-t
            d.sigma2     = sigma2_t.clone()                                     # (nNodes, 1) — static


# =========================================================================
# EAD training step (one epoch)
# =========================================================================
def train_ead(loader, model, opt, scheduler, device,
              nNodes, edge_src, edge_dst, edge_w,
              nNodes_labeled=58):
    model.train()

    stats = dict(
        n_batches=0, sup_eps=0.0, lap=0.0, zm=0.0, pseudo=0.0, recon=0.0, total=0.0,
        eps_mean_abs=0.0,
        # Mask-Reconstruct diagnostics
        recon_n_total=0, recon_sq_err=0.0,   # for computing recon-only RMSE on masked positions
        recon_mask_count=0,                  # how many positions masked across batches
    )
    pred_delta, truth_delta = [], []

    # Static pseudo-label mask: True for the 400 true-unlabeled nodes (positions 58-457)
    # NOT applied to 8 valid labeled (keep them fully held out)
    pseudo_mask_per = torch.zeros(nNodes, dtype=torch.bool, device=device)
    pseudo_mask_per[nNodes_labeled:] = True

    for _n, batch in enumerate(loader):
        batch = batch.to(device)
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        label_mask = batch.label_mask                                   # (batch_size*nNodes,)
        bs = x.shape[0] // nNodes

        # --- Forward: GNN predicts ε̂; β̂ is precomputed (Kriging), not learned ---
        eps_hat = model(x, edge_index, edge_attr).squeeze(-1)            # (bs*nNodes,)

        # α_t broadcast: batch.alpha_t is (bs,), need (bs*nNodes,)
        alpha_t = batch.alpha_t.repeat_interleave(nNodes) if EAD_ALPHA else torch.zeros(bs*nNodes, device=device)

        # y_residual is Δ for labeled nodes (NOT batch.y — that's absolute T)
        y_delta = batch.y_residual.squeeze(-1)                           # (bs*nNodes,)
        beta_hat_batch = batch.beta_hat.squeeze(-1) if EAD_BETA else torch.zeros_like(eps_hat)

        # Reshape per graph for Laplacian + zero-mean
        eps_hat_g = eps_hat.reshape(bs, nNodes)

        # --- Mask-Reconstruct: randomly mask K labeled stations, reconstruct with their true labels ---
        # Pick K positions out of the labeled set; same K stations across all graphs in this batch
        # (each batch picks new random K → over many batches, all stations get masked)
        if EAD_MASK_RECON:
            # label_mask[:nNodes] = per-graph mask; same for all graphs in batch
            lm_per = label_mask[:nNodes]
            labeled_positions = torch.where(lm_per)[0]                  # indices of labeled within nNodes
            n_labeled = len(labeled_positions)
            k = min(MASK_K, max(n_labeled - 5, 0))                      # leave at least 5 for sup
            if k > 0:
                perm = torch.randperm(n_labeled, device=device)[:k]
                mask_idx_local = labeled_positions[perm]                # (k,) within nNodes
                # Build per-batch reconstruction mask
                recon_mask = torch.zeros(bs * nNodes, dtype=torch.bool, device=device)
                for g in range(bs):
                    recon_mask[g * nNodes + mask_idx_local] = True
                # Sup mask = label_mask minus recon
                sup_mask = label_mask & ~recon_mask
            else:
                recon_mask = torch.zeros(bs * nNodes, dtype=torch.bool, device=device)
                sup_mask = label_mask
        else:
            recon_mask = torch.zeros(bs * nNodes, dtype=torch.bool, device=device)
            sup_mask = label_mask

        # --- Supervised loss on ε (non-masked labeled only) ---
        target_eps = y_delta - alpha_t - beta_hat_batch
        sup_eps = torch.nn.functional.huber_loss(eps_hat[sup_mask], target_eps[sup_mask])

        # --- Reconstruction loss on masked labeled (using their TRUE labels) ---
        if EAD_MASK_RECON and recon_mask.any():
            recon_loss = torch.nn.functional.huber_loss(eps_hat[recon_mask], target_eps[recon_mask])
            # Diagnostic: track squared error for recon-only RMSE
            with torch.no_grad():
                _err = (eps_hat[recon_mask] - target_eps[recon_mask]).detach()
                stats['recon_sq_err'] += float((_err ** 2).sum().item())
                stats['recon_n_total'] += int(recon_mask.sum().item())
                stats['recon_mask_count'] += int(recon_mask.sum().item())
        else:
            recon_loss = torch.tensor(0.0, device=device)

        # --- Laplacian on ε̂ (same as baseline's Lap on model output) ---
        lap_loss = torch.tensor(0.0, device=device)
        for g in range(bs):
            diff = eps_hat_g[g][edge_src] - eps_hat_g[g][edge_dst]
            lap_loss = lap_loss + torch.mean(edge_w * diff ** 2)
        lap_loss = lap_loss / bs

        # --- Zero-mean constraint on ε̂ (B3) ---
        if EAD_ZERO_MEAN:
            # mean across nodes per graph → (bs,). Penalize squared mean.
            node_mean = eps_hat_g.mean(dim=1)
            zm_loss = (node_mean ** 2).mean()
        else:
            zm_loss = torch.tensor(0.0, device=device)

        # --- Kriging pseudo-label loss on unlabeled ε̂ ---
        if EAD_PSEUDO:
            pseudo_eps_batch = batch.pseudo_eps.squeeze(-1)              # (bs*nNodes,)
            sigma2_batch     = batch.sigma2.squeeze(-1)                  # (bs*nNodes,)
            inv_sigma2       = 1.0 / (sigma2_batch + 1e-3)
            # apply mask on all 400 unlabeled positions across graphs
            pseudo_mask_batched = pseudo_mask_per.repeat(bs)
            diff_sq = (eps_hat - pseudo_eps_batch) ** 2
            pseudo_loss = (inv_sigma2[pseudo_mask_batched] * diff_sq[pseudo_mask_batched]).mean()
        else:
            pseudo_loss = torch.tensor(0.0, device=device)

        total = (sup_eps + LAMBDA_LAP * lap_loss + LAMBDA_ZM * zm_loss
                 + LAMBDA_PSD * pseudo_loss + LAMBDA_RECON * recon_loss)

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        # --- Diagnostics ---
        stats['n_batches'] += 1
        stats['sup_eps']   += float(sup_eps.item())
        stats['lap']       += float(lap_loss.item())
        stats['zm']        += float(zm_loss.item())
        stats['pseudo']    += float(pseudo_loss.item())
        stats['recon']     += float(recon_loss.item())
        stats['total']     += float(total.item())
        stats['eps_mean_abs'] += float(eps_hat_g.mean(dim=1).abs().mean().item())

        # Final prediction in Δ space for RMSE
        delta_pred = (alpha_t + beta_hat_batch + eps_hat).detach().reshape(bs, nNodes)
        delta_true = y_delta.reshape(bs, nNodes)
        lm_per = label_mask[:nNodes]
        for g in range(bs):
            pred_delta.append(delta_pred[g][lm_per].cpu().numpy())
            truth_delta.append(delta_true[g][lm_per].cpu().numpy())

    scheduler.step()
    for k in ('sup_eps', 'lap', 'zm', 'pseudo', 'recon', 'total', 'eps_mean_abs'):
        stats[k] /= max(stats['n_batches'], 1)
    # Compute recon-only RMSE (sqrt of mean squared error on masked positions)
    if stats['recon_n_total'] > 0:
        stats['recon_rmse'] = float(np.sqrt(stats['recon_sq_err'] / stats['recon_n_total']))
    else:
        stats['recon_rmse'] = 0.0

    pred_delta, truth_delta = np.array(pred_delta), np.array(truth_delta)
    rmse_stats = utils.RMSE(truth_delta, pred_delta)
    return stats, rmse_stats


# =========================================================================
# EAD evaluation with full diagnostics
# =========================================================================
@torch.no_grad()
def eval_ead(loader, model, device, nNodes, nNodes_labeled=58, tgt_scl=1.0):
    model.eval()

    total_pred, total_truth = [], []       # Δ̂ vs Δ on VALID labeled
    eps_pred, eps_truth = [], []           # ε̂ vs ε_true on VALID labeled (diagnostic)
    disc_ratios = []                       # std(ε̂ unlabeled) / std(ε̂ labeled)
    eps_mean_abs_list = []

    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        label_mask = batch.label_mask                  # valid_label_mask (8 stations True)
        bs = x.shape[0] // nNodes

        eps_hat = model(x, edge_index, edge_attr).squeeze(-1)
        beta_hat_batch = batch.beta_hat.squeeze(-1) if EAD_BETA else torch.zeros_like(eps_hat)
        alpha_t = batch.alpha_t.repeat_interleave(nNodes) if EAD_ALPHA else torch.zeros(bs*nNodes, device=device)

        y_delta = batch.y_residual.squeeze(-1)        # Δ, NOT batch.y (which is absolute T)

        eps_hat_g  = eps_hat.reshape(bs, nNodes)
        beta_hat_g = beta_hat_batch.reshape(bs, nNodes)
        alpha_g    = alpha_t.reshape(bs, nNodes)
        y_delta_g  = y_delta.reshape(bs, nNodes)
        lm_per     = label_mask[:nNodes]

        # Δ̂ = α + β̂ + ε̂
        delta_pred_g = alpha_g + beta_hat_g + eps_hat_g

        # For labeled valid stations
        for g in range(bs):
            total_pred.append(delta_pred_g[g][lm_per].cpu().numpy())
            total_truth.append(y_delta_g[g][lm_per].cpu().numpy())
            # ε target = Δ - α - β̂ (Kriging β̂ applies to valid too)
            eps_target_g = y_delta_g[g][lm_per] - alpha_g[g][lm_per] - beta_hat_g[g][lm_per]
            eps_pred.append(eps_hat_g[g][lm_per].cpu().numpy())
            eps_truth.append(eps_target_g.cpu().numpy())

            # Discrimination ratio: std over unlabeled vs all labeled nodes (positions based on nNodes_labeled)
            eps_all = eps_hat_g[g].cpu().numpy()
            labeled_idx_full = np.arange(nNodes_labeled)
            unlabeled_idx = np.arange(nNodes_labeled, nNodes)
            std_l = eps_all[labeled_idx_full].std()
            std_u = eps_all[unlabeled_idx].std()
            disc_ratios.append(std_u / (std_l + 1e-8))

            eps_mean_abs_list.append(abs(eps_hat_g[g].mean().item()))

    total_pred  = np.array(total_pred)
    total_truth = np.array(total_truth)
    eps_pred    = np.array(eps_pred)
    eps_truth   = np.array(eps_truth)

    rmse_total = utils.RMSE(total_truth, total_pred)
    rmse_eps   = utils.RMSE(eps_truth, eps_pred)

    # 6-metric suite (RMSE, MBE, MAE) × (normalized, Celsius) on absolute prediction
    metrics_6 = utils.compute_all_metrics(total_truth, total_pred, scl=tgt_scl)

    return dict(
        rmse_total = rmse_total,
        rmse_eps   = rmse_eps,
        disc_ratio = float(np.mean(disc_ratios)),
        eps_mean_abs_eval = float(np.mean(eps_mean_abs_list)),
        **metrics_6,
    )


# =========================================================================
# Main
# =========================================================================
def main():
    # --- Data ---
    dataParam = {
        'geoMethod': 'average', 'nCompPCA': 40, 'window': 2,
        'poolSize': int(os.environ.get('POOL_SIZE', '12')),
        'batchSize': 512, 'thres': 0.1,
        'geoFeatures': 'full', 'n_unlabeled': n_unlabeled,
    }
    os.environ.setdefault('EVAL_MODE', 'spatial')   # EAD requires spatial mode
    os.environ.setdefault('USE_FPS', '2')
    os.environ.setdefault('EDGE_MODE', 'no_uu')
    os.environ.setdefault('NORM_MODE', 'global')

    trainLoader, validLoader, metadata, validSet = data_semi.dataGen(dataParam, path)
    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']

    # --- Sanity check: graph structure matches baseline 13020 ---
    Adj_np = metadata['AdjMatrix']
    full_label_mask = metadata['label_mask']                    # (nNodes,) — all 58 labeled
    labeled_idx = np.where(full_label_mask)[0]
    unlabeled_idx = np.where(~full_label_mask)[0]
    ll_edges = int(((Adj_np[np.ix_(labeled_idx, labeled_idx)] > 0).sum() - (Adj_np.diagonal()[labeled_idx] > 0).sum()) // 2)
    uu_edges = int(((Adj_np[np.ix_(unlabeled_idx, unlabeled_idx)] > 0).sum() - (Adj_np.diagonal()[unlabeled_idx] > 0).sum()) // 2)
    lu_edges = int((Adj_np[np.ix_(labeled_idx, unlabeled_idx)] > 0).sum())
    total_edges = ll_edges + uu_edges + lu_edges
    print("\n" + "="*72)
    print("  Graph structure sanity check (should match baseline 13020)")
    print("="*72)
    print(f"  Total nodes:         {nNodes}  ({nNodes_labeled} labeled + {nNodes - nNodes_labeled} unlabeled)")
    print(f"  L-L edges:           {ll_edges}")
    print(f"  L-U edges:           {lu_edges}")
    print(f"  U-U edges:           {uu_edges}   (should be 0 with EDGE_MODE=no_uu)")
    print(f"  Total edges:         {total_edges}")
    print(f"  Expected (job 13020): L-L=951, L-U=15845, U-U=0, total=16796")
    if uu_edges != 0:
        print(f"  ⚠ WARNING: U-U edges not 0 — check EDGE_MODE env var")
    else:
        print(f"  ✓ U-U edges removed as expected.")
    print("="*72)

    # --- Precompute α_t, β̂ (Kriging), optional pseudo-ε (Kriging) ---
    alpha_per_sample, beta_hat_all, beta_true_valid_diag, pseudo_eps_all, sigma2_all = \
        precompute_alpha_beta(trainLoader, validLoader, metadata)
    attach_alpha_beta_to_dataset(trainLoader.dataset, alpha_per_sample, beta_hat_all, metadata,
                                 pseudo_eps_all, sigma2_all)
    attach_alpha_beta_to_dataset(validLoader.dataset, alpha_per_sample, beta_hat_all, metadata,
                                 pseudo_eps_all, sigma2_all)

    # --- Model: GNN only (no MLP_β anymore — β̂ is precomputed Kriging) ---
    USE_CNN_ENCODER  = int(os.environ.get('USE_CNN_ENCODER', '0'))
    USE_CNN_RAW_UFM  = int(os.environ.get('USE_CNN_RAW_UFM', '0'))
    modelParam = {
        'HLD': 128, 'nMLP': 2, 'nGNN': N_GNN, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'iDim': metadata['iDim'], 'oDim': metadata['oDim'],
        'BN': False, 'Dropout': True, 'conv_type': conv_type,
        'use_gnn': bool(USE_GNN),
        'ufm_channels': int(os.environ.get('UFM_CHANNELS', '7')),
        'ufm_spatial':  int(os.environ.get('UFM_SPATIAL', os.environ.get('POOL_SIZE', '12'))),
        'cnn_out_dim':  int(os.environ.get('CNN_OUT_DIM', '64')),
    }
    if USE_CNN_RAW_UFM:
        # Raw UFM: data_semi must have set POOL_TYPE=raw, metadata['raw_ufm'] available
        assert 'raw_ufm' in metadata, \
            "USE_CNN_RAW_UFM=1 requires POOL_TYPE=raw (metadata['raw_ufm'] missing)"
        all_ufm_raw = torch.from_numpy(metadata['raw_ufm']).float()   # (N, C, H, W)
        gnn = network_semi.GNN_CNN_Raw(modelParam, all_ufm_raw).to(device)
    elif USE_CNN_ENCODER:
        gnn = network_semi.GNN_CNN(modelParam).to(device)
    else:
        gnn = network_semi.GNN(modelParam).to(device)

    # --- Precompute edge tensors for Laplacian ---
    Adj = metadata['AdjMatrix']
    edge_src_np, edge_dst_np = np.nonzero(Adj)
    edge_src = torch.LongTensor(edge_src_np).to(device)
    edge_dst = torch.LongTensor(edge_dst_np).to(device)
    edge_w   = torch.FloatTensor(Adj[edge_src_np, edge_dst_np]).to(device)

    # --- Optimizer ---
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)

    # --- wandb ---
    modelName = f'ead_a{EAD_ALPHA}_b{EAD_BETA}_z{EAD_ZERO_MEAN}'
    wandb.init(
        entity="urban_prediction", project="Semi-supervised GNN",
        name=f'{modelName}_job{job_id}' if job_id else modelName,
        config={**dataParam, **modelParam,
                'EAD_ALPHA': EAD_ALPHA, 'EAD_BETA': EAD_BETA,
                'EAD_ZERO_MEAN': EAD_ZERO_MEAN,
                'LAMBDA_LAP': LAMBDA_LAP, 'LAMBDA_ZM': LAMBDA_ZM,
                'USE_GNN': USE_GNN, 'N_GNN': N_GNN},
    )

    # ==================== Training loop ====================
    nEpoch = 5000
    best_rmse = float('inf')
    log_path = os.path.join(output_dir, f'{modelName}_log')

    with open(log_path, 'a') as f:
        f.write(f"\nEAD config: alpha={EAD_ALPHA} beta={EAD_BETA} zero_mean={EAD_ZERO_MEAN} lap={EAD_LAP} pseudo={EAD_PSEUDO}\n")
        f.write(f"  (β̂ mode: Kriging spatial interpolation, no learned MLP)\n")
        f.write(f"  LAMBDA_LAP={LAMBDA_LAP}, LAMBDA_ZM={LAMBDA_ZM}, LAMBDA_PSD={LAMBDA_PSD}\n")
        f.write(f"  USE_GNN={USE_GNN}, N_GNN={N_GNN}\n")
    print(f"\n  EAD config: alpha={EAD_ALPHA} beta={EAD_BETA} zero_mean={EAD_ZERO_MEAN} lap={EAD_LAP} pseudo={EAD_PSEUDO}")
    print(f"  β̂ mode: Kriging spatial interpolation (no learned MLP)")
    print(f"  LAMBDA_LAP={LAMBDA_LAP}, LAMBDA_ZM={LAMBDA_ZM}, LAMBDA_PSD={LAMBDA_PSD}")
    print(f"  USE_GNN={USE_GNN}, N_GNN={N_GNN}")
    print(f"  Model: GNN={sum(p.numel() for p in gnn.parameters())} params")

    for epoch in range(nEpoch):
        stats, train_rmse = train_ead(
            trainLoader, gnn, opt, scheduler, device,
            nNodes, edge_src, edge_dst, edge_w, nNodes_labeled
        )
        # Use global scaling for Celsius conversion (one number for all stations)
        tgt_scl_celsius = float(metadata.get('tgt_global_scl', 1.0))
        eval_metrics = eval_ead(validLoader, gnn, device, nNodes, nNodes_labeled, tgt_scl=tgt_scl_celsius)

        # ---- Log ----
        valid_rmse  = eval_metrics['rmse_norm']
        valid_rmseC = eval_metrics['rmse_C']
        line = (
            f"Epoch {epoch:4d}: "
            f"train_loss={stats['total']:.4e}  "
            f"train_RMSE={train_rmse[0]:.4f}  "
            f"valid_RMSE={valid_rmse:.4f} ({valid_rmseC:.3f}°C)  "
            f"eps_RMSE(val)={eval_metrics['rmse_eps'][0]:.4f}  "
            f"disc_ratio={eval_metrics['disc_ratio']:.3f}  "
            f"|eps_mean|={eval_metrics['eps_mean_abs_eval']:.4f}  "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )
        if epoch % 10 == 0 or epoch < 3:
            print(line)
        with open(log_path, 'a') as f:
            f.write(line + '\n')
            f.write(f"  [components] sup_eps={stats['sup_eps']:.4e}  "
                    f"lap={stats['lap']:.4e}  zm={stats['zm']:.4e}  "
                    f"pseudo={stats['pseudo']:.4e}  recon={stats['recon']:.4e}\n")
            if EAD_MASK_RECON:
                f.write(f"  [mask-recon] masked_positions={stats['recon_mask_count']}  "
                        f"recon_only_RMSE={stats['recon_rmse']:.4f}  (vs train_RMSE on visible labeled)\n")

        wandb.log({
            'epoch': epoch,
            'train/total_loss': stats['total'],
            'train/sup_eps':    stats['sup_eps'],
            'train/lap':        stats['lap'],
            'train/zm':         stats['zm'],
            'train/pseudo':     stats['pseudo'],
            'train/recon':      stats['recon'],
            'train/recon_only_rmse': stats['recon_rmse'],
            'train/rmse':       train_rmse[0],
            'train/eps_mean_abs_batch': stats['eps_mean_abs'],
            'valid/rmse':       valid_rmse,                       # normalized RMSE (kept for backwards compat)
            'valid/eps_rmse':   eval_metrics['rmse_eps'][0],
            'valid/disc_ratio': eval_metrics['disc_ratio'],
            'valid/eps_mean_abs': eval_metrics['eps_mean_abs_eval'],
            # ---- 6-metric suite ----
            'metrics/rmse_norm': eval_metrics['rmse_norm'],
            'metrics/mbe_norm':  eval_metrics['mbe_norm'],
            'metrics/mae_norm':  eval_metrics['mae_norm'],
            'metrics/rmse_C':    eval_metrics['rmse_C'],
            'metrics/mbe_C':     eval_metrics['mbe_C'],
            'metrics/mae_C':     eval_metrics['mae_C'],
            'lr': scheduler.get_last_lr()[0],
            'best_valid_rmse': best_rmse,
        })

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            with open(log_path, 'a') as f:
                f.write(f"  Model saved. best_valid_RMSE={best_rmse:.4f}\n")
            torch.save({
                'epoch': epoch,
                'gnn_state_dict':      gnn.state_dict(),
                'opt_state_dict':      opt.state_dict(),
                'bestLoss':            best_rmse,
                'alpha_per_sample':    alpha_per_sample,
                'beta_hat':            beta_hat_all,
            }, os.path.join(output_dir, f'{modelName}.pt'))

    print(f"\nFinal best valid RMSE: {best_rmse:.4f}")


if __name__ == '__main__':
    main()

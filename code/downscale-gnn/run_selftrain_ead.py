"""
Self-Training pipeline on top of EAD (B1 baseline).

Pipeline:
  Phase 1   : train base EAD model (= B1 setup)
  Phase 1.5 : continue training with CosineAnnealingWarmRestarts to
              collect M=5 snapshots (Snapshot Ensemble)
  Phase 2   : aggregate snapshots → confidence (std), pseudo-label (mean),
              embeddings (last snapshot)
  Phase 3   : pseudo-label selection from 400 unlabeled candidates
                · hard threshold by confidence (top τ%)
                · greedy: score(u) = α·diversity(u, S) + β·relevance(u, V)
  Phase 4   : retrain EAD from scratch with extra pseudo-label loss on the K selected nodes

Standard B1 reference  : 13204, valid_RMSE = 0.0428.
We reuse run_ead's hyper-params (NORM_MODE=global, USE_FPS=2, EDGE_MODE=no_uu,
LAMBDA_LAP=0.1, EAD_ALPHA=1, etc.) so this is a proper apples-to-apples extension.

Environment variables (in addition to run_ead's):
  ST_PHASE1_EPOCHS    base training epochs                       (default 4000)
  ST_PHASE4_EPOCHS    retraining epochs                          (default 4000)
  ST_M_SNAPSHOTS      number of snapshots                        (default 5)
  ST_T_CYCLE          epochs per cyclic-LR cycle                 (default 40)
  ST_LR_MAX           peak LR for cyclic phase                   (default 1e-3)
  ST_LR_MIN           min LR for cyclic phase                    (default 1e-5)
  ST_TAU_QUANTILE     keep candidates with conf > this quantile  (default 0.5)
  ST_K                number of pseudo-labels to select          (default 100)
  ST_ALPHA_DIV        diversity weight in greedy score           (default 1.0)
  ST_BETA_REL         relevance weight in greedy score           (default 1.0)
  ST_K_NN_REL         k-NN over V for relevance                  (default 5)
  ST_LAMBDA_PSEUDO    pseudo-label loss weight in Phase 4        (default 0.5)
  ST_ABLATION         full | random | conf_only | conf_div | conf_rel
                                                                 (default 'full')
"""
import os, copy, pickle
from datetime import datetime

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
import wandb

import data_semi, network_semi, utils


# =========================================================================
# Setup
# =========================================================================
device   = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id    = os.environ.get('SLURM_JOB_ID', '')

# === EAD config (mirror run_ead.py) ===
EAD_ALPHA     = int(os.environ.get('EAD_ALPHA', '1'))
EAD_BETA      = int(os.environ.get('EAD_BETA', '0'))
EAD_ZERO_MEAN = int(os.environ.get('EAD_ZERO_MEAN', '0'))
EAD_LAP       = int(os.environ.get('EAD_LAP', '1'))
LAMBDA_LAP    = float(os.environ.get('LAMBDA_LAP', '0.1'))
if EAD_LAP == 0:
    LAMBDA_LAP = 0.0
LAMBDA_ZM     = float(os.environ.get('LAMBDA_ZM', '1.0'))
USE_GNN       = int(os.environ.get('USE_GNN', '1'))
N_GNN         = int(os.environ.get('N_GNN', '3'))
n_unlabeled   = int(os.environ.get('N_UNLABELED', '400'))

# === Self-training config ===
ST_PHASE1_EPOCHS = int(os.environ.get('ST_PHASE1_EPOCHS', '4000'))
ST_PHASE4_EPOCHS = int(os.environ.get('ST_PHASE4_EPOCHS', '4000'))
ST_M_SNAPSHOTS   = int(os.environ.get('ST_M_SNAPSHOTS', '5'))
ST_T_CYCLE       = int(os.environ.get('ST_T_CYCLE', '40'))
ST_LR_MAX        = float(os.environ.get('ST_LR_MAX', '1e-3'))
ST_LR_MIN        = float(os.environ.get('ST_LR_MIN', '1e-5'))
ST_TAU_QUANTILE  = float(os.environ.get('ST_TAU_QUANTILE', '0.5'))
ST_K             = int(os.environ.get('ST_K', '100'))
ST_ALPHA_DIV     = float(os.environ.get('ST_ALPHA_DIV', '1.0'))
ST_BETA_REL      = float(os.environ.get('ST_BETA_REL', '1.0'))
ST_K_NN_REL      = int(os.environ.get('ST_K_NN_REL', '5'))
ST_LAMBDA_PSEUDO = float(os.environ.get('ST_LAMBDA_PSEUDO', '0.5'))
ST_ABLATION      = os.environ.get('ST_ABLATION', 'full').lower()

assert ST_ABLATION in ('full', 'random', 'conf_only', 'conf_div', 'conf_rel'), \
    f"unknown ST_ABLATION={ST_ABLATION}"

st_tag = f"st_{ST_ABLATION}_K{ST_K}_M{ST_M_SNAPSHOTS}"
output_dir = os.path.join(project_root, 'log',
                          f'job{job_id}_{st_tag}_{timestamp}' if job_id else f'{st_tag}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
os.environ['OUTPUT_DIR'] = output_dir


# =========================================================================
# Kriging β̂ (used for EAD_BETA=1; reused from run_ead)
# =========================================================================
def kriging_beta(Adj, beta_train_values, train_idx, nNodes):
    W = Adj[:, train_idx]
    num = W @ beta_train_values
    denom = W.sum(axis=1) + 1e-8
    beta_hat = num / denom
    beta_hat[train_idx] = beta_train_values
    return beta_hat.astype(np.float32)


def precompute_alpha_beta(trainLoader, metadata):
    """Compute α_t (per-timestep) and β̂ (per-station, Kriging) from train labeled."""
    nNodes = metadata['nNodes']
    train_station_idx = metadata['train_station_idx']
    valid_station_idx = metadata['valid_station_idx']

    train_only_mask = np.zeros(nNodes, dtype=bool)
    train_only_mask[train_station_idx] = True

    dataset  = trainLoader.dataset
    nSamples = len(dataset)
    all_delta = np.zeros((nSamples, nNodes), dtype=np.float32)
    for t, d in enumerate(dataset):
        all_delta[t] = d.y_residual.squeeze(-1).cpu().numpy()

    alpha_per_sample = all_delta[:, train_only_mask].mean(axis=1)            # (nSamples,)

    delta_minus_alpha = all_delta - alpha_per_sample[:, None]
    beta_true = np.zeros(nNodes, dtype=np.float32)
    beta_true[train_station_idx] = delta_minus_alpha[:, train_only_mask].mean(axis=0)
    beta_true[train_station_idx] -= beta_true[train_station_idx].mean()      # center

    Adj = metadata['AdjMatrix']
    beta_hat = kriging_beta(Adj, beta_true[train_station_idx], train_station_idx, nNodes)

    print(f"\n  [α/β precompute] α_t std={alpha_per_sample.std():.4f}, "
          f"β̂_train std={beta_hat[train_station_idx].std():.4f}, "
          f"β̂_unlabeled std={beta_hat[58:].std():.4f}")
    return alpha_per_sample, beta_hat, all_delta


# =========================================================================
# Attach EAD per-graph fields + (optional) pseudo-labels and selection mask
# =========================================================================
def attach_to_dataset(dataset, alpha_per_sample, beta_hat,
                      pseudo_eps_all=None, pseudo_sigma_all=None,
                      pseudo_select_mask=None):
    """
    For every sample t in `dataset`, attach:
      d.alpha_t                 scalar       α_t
      d.beta_hat                (nNodes, 1)  β̂
      d.pseudo_eps              (nNodes, 1)  ensemble-mean ε̂  (if given)
      d.pseudo_sigma            (nNodes, 1)  ensemble-std  σ   (if given)
      d.pseudo_select_mask      (nNodes, 1)  bool         selection mask (if given)
    """
    nNodes = beta_hat.shape[0]
    beta_hat_t = torch.FloatTensor(beta_hat).unsqueeze(-1)
    select_t = (torch.from_numpy(pseudo_select_mask).bool().unsqueeze(-1)
                if pseudo_select_mask is not None else None)
    for t, d in enumerate(dataset):
        d.alpha_t  = torch.FloatTensor([alpha_per_sample[t]])
        d.beta_hat = beta_hat_t.clone()
        if pseudo_eps_all is not None:
            d.pseudo_eps   = torch.FloatTensor(pseudo_eps_all[t]).unsqueeze(-1)
            d.pseudo_sigma = torch.FloatTensor(pseudo_sigma_all[t]).unsqueeze(-1)
            d.pseudo_select_mask = select_t.clone()


# =========================================================================
# EAD train epoch  (supports optional pseudo-label loss on selected nodes)
# =========================================================================
def train_ead_epoch(loader, model, opt, scheduler, device,
                    nNodes, edge_src, edge_dst, edge_w,
                    use_pseudo=False, lambda_pseudo=0.0):
    model.train()
    stats = dict(n_batches=0, sup_eps=0.0, lap=0.0, zm=0.0, pseudo=0.0,
                 total=0.0, n_pseudo_eff=0.0)
    pred_delta, truth_delta = [], []

    for _n, batch in enumerate(loader):
        batch = batch.to(device)
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        label_mask = batch.label_mask
        bs = x.shape[0] // nNodes

        eps_hat = model(x, edge_index, edge_attr).squeeze(-1)
        alpha_t = (batch.alpha_t.repeat_interleave(nNodes) if EAD_ALPHA
                   else torch.zeros(bs * nNodes, device=device))
        y_delta = batch.y_residual.squeeze(-1)
        beta_hat_batch = (batch.beta_hat.squeeze(-1) if EAD_BETA
                         else torch.zeros_like(eps_hat))

        target_eps = y_delta - alpha_t - beta_hat_batch

        # --- Supervised loss on labeled stations (Huber, same as run_ead) ---
        sup_eps = torch.nn.functional.huber_loss(eps_hat[label_mask], target_eps[label_mask])

        # --- Laplacian on ε̂ ---
        eps_hat_g = eps_hat.reshape(bs, nNodes)
        lap_loss = torch.tensor(0.0, device=device)
        for g in range(bs):
            diff = eps_hat_g[g][edge_src] - eps_hat_g[g][edge_dst]
            lap_loss = lap_loss + torch.mean(edge_w * diff ** 2)
        lap_loss = lap_loss / bs

        # --- Zero-mean ---
        if EAD_ZERO_MEAN:
            zm_loss = (eps_hat_g.mean(dim=1) ** 2).mean()
        else:
            zm_loss = torch.tensor(0.0, device=device)

        # --- Pseudo-label loss on K selected unlabeled ---
        if use_pseudo and lambda_pseudo > 0:
            pseudo_eps_batch  = batch.pseudo_eps.squeeze(-1)
            pseudo_sigma_batch = batch.pseudo_sigma.squeeze(-1)
            select_mask_batch = batch.pseudo_select_mask.squeeze(-1).bool()
            # Confidence weight: low σ → high weight. Normalize so mean weight = 1.
            inv_var = 1.0 / (pseudo_sigma_batch ** 2 + 1e-3)
            if select_mask_batch.any():
                w_sel = inv_var[select_mask_batch]
                w_sel = w_sel / (w_sel.mean() + 1e-8)
                diff  = (eps_hat[select_mask_batch] - pseudo_eps_batch[select_mask_batch])
                pseudo_loss = (w_sel * diff ** 2).mean()
            else:
                pseudo_loss = torch.tensor(0.0, device=device)
            stats['n_pseudo_eff'] += float(select_mask_batch.sum().item())
        else:
            pseudo_loss = torch.tensor(0.0, device=device)

        total = sup_eps + LAMBDA_LAP * lap_loss + LAMBDA_ZM * zm_loss + lambda_pseudo * pseudo_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        stats['n_batches'] += 1
        stats['sup_eps']   += float(sup_eps.item())
        stats['lap']       += float(lap_loss.item())
        stats['zm']        += float(zm_loss.item())
        stats['pseudo']    += float(pseudo_loss.item())
        stats['total']     += float(total.item())

        # Δ space prediction for train RMSE
        delta_pred = (alpha_t + beta_hat_batch + eps_hat).detach().reshape(bs, nNodes)
        delta_true = y_delta.reshape(bs, nNodes)
        lm_per = label_mask[:nNodes]
        for g in range(bs):
            pred_delta.append(delta_pred[g][lm_per].cpu().numpy())
            truth_delta.append(delta_true[g][lm_per].cpu().numpy())

    if scheduler is not None:
        scheduler.step()
    for k in ('sup_eps', 'lap', 'zm', 'pseudo', 'total'):
        stats[k] /= max(stats['n_batches'], 1)
    pred_delta, truth_delta = np.array(pred_delta), np.array(truth_delta)
    rmse_stats = utils.RMSE(truth_delta, pred_delta)
    return stats, rmse_stats


# =========================================================================
# Eval — same metrics as run_ead.eval_ead
# =========================================================================
@torch.no_grad()
def eval_ead(loader, model, device, nNodes, nNodes_labeled=58, tgt_scl=1.0):
    model.eval()
    total_pred, total_truth = [], []
    eps_pred, eps_truth = [], []

    for batch in loader:
        batch = batch.to(device)
        bs = batch.x.shape[0] // nNodes
        eps_hat = model(batch.x, batch.edge_index, batch.edge_attr).squeeze(-1)
        alpha_t = (batch.alpha_t.repeat_interleave(nNodes) if EAD_ALPHA
                   else torch.zeros(bs * nNodes, device=device))
        beta_hat_batch = (batch.beta_hat.squeeze(-1) if EAD_BETA
                          else torch.zeros_like(eps_hat))
        y_delta = batch.y_residual.squeeze(-1)
        label_mask = batch.label_mask

        eps_hat_g  = eps_hat.reshape(bs, nNodes)
        beta_hat_g = beta_hat_batch.reshape(bs, nNodes)
        alpha_g    = alpha_t.reshape(bs, nNodes)
        y_delta_g  = y_delta.reshape(bs, nNodes)
        lm_per     = label_mask[:nNodes]

        delta_pred_g = alpha_g + beta_hat_g + eps_hat_g
        for g in range(bs):
            total_pred.append(delta_pred_g[g][lm_per].cpu().numpy())
            total_truth.append(y_delta_g[g][lm_per].cpu().numpy())
            eps_target_g = y_delta_g[g][lm_per] - alpha_g[g][lm_per] - beta_hat_g[g][lm_per]
            eps_pred.append(eps_hat_g[g][lm_per].cpu().numpy())
            eps_truth.append(eps_target_g.cpu().numpy())

    total_pred  = np.array(total_pred)
    total_truth = np.array(total_truth)
    eps_pred    = np.array(eps_pred)
    eps_truth   = np.array(eps_truth)

    rmse_total = utils.RMSE(total_truth, total_pred)
    rmse_eps   = utils.RMSE(eps_truth, eps_pred)
    metrics_6  = utils.compute_all_metrics(total_truth, total_pred, scl=tgt_scl)
    return dict(rmse_total=rmse_total, rmse_eps=rmse_eps, **metrics_6)


# =========================================================================
# Predict ε̂ over the entire dataset (ordered, no shuffle)
# =========================================================================
@torch.no_grad()
def predict_eps_all(model, dataset, device, nNodes, batch_size=512):
    """Returns: eps_pred (nSamples, nNodes) — ε̂ for every (t, station)."""
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    for batch in loader:
        batch = batch.to(device)
        bs = batch.x.shape[0] // nNodes
        eps_hat = model(batch.x, batch.edge_index, batch.edge_attr).squeeze(-1)
        out.append(eps_hat.reshape(bs, nNodes).cpu().numpy())
    return np.concatenate(out, axis=0)


# =========================================================================
# Per-station embedding: average GNN penultimate hidden state across timesteps
# =========================================================================
@torch.no_grad()
def per_station_embedding(model, dataset, device, nNodes, batch_size=512):
    """Returns: emb (nNodes, HLD)."""
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    sum_emb = None
    count   = 0
    for batch in loader:
        batch = batch.to(device)
        bs = batch.x.shape[0] // nNodes
        _, h = model(batch.x, batch.edge_index, batch.edge_attr, return_hidden=True)
        h = h.reshape(bs, nNodes, -1)             # (bs, nNodes, HLD)
        if sum_emb is None:
            sum_emb = h.sum(dim=0).cpu().numpy()  # (nNodes, HLD)
        else:
            sum_emb += h.sum(dim=0).cpu().numpy()
        count += bs
    return sum_emb / max(count, 1)


# =========================================================================
# Greedy selection: confidence filter + diversity (max-min) + relevance (kNN to V)
# =========================================================================
def greedy_select(emb_all, candidate_idx, valid_idx,
                  confidence_per_station, K, alpha, beta, k_nn_rel,
                  tau_quantile, ablation):
    """
    Args:
        emb_all                : (nNodes, HLD)
        candidate_idx          : np.array of indices into nNodes (the 400 unlabeled)
        valid_idx              : np.array of indices into nNodes (the 8 valid stations = V)
        confidence_per_station : (nNodes,) — higher = more confident (e.g. -mean σ)
        K                      : how many to pick
        alpha, beta            : weights for diversity / relevance
        k_nn_rel               : k for kNN relevance to V
        tau_quantile           : keep top (1-τ) fraction by confidence over candidates
        ablation               : 'full' | 'random' | 'conf_only' | 'conf_div' | 'conf_rel'
    Returns:
        selected_global_idx (list of station indices, length up to K)
    """
    cand_emb = emb_all[candidate_idx]                                # (nCand, HLD)
    val_emb  = emb_all[valid_idx]                                    # (nVal,  HLD)
    cand_conf = confidence_per_station[candidate_idx]                # (nCand,)
    nCand = len(candidate_idx)

    # Confidence diagnostic
    print(f"\n  [Selection] confidence over candidates: "
          f"min={cand_conf.min():+.4f}, mean={cand_conf.mean():+.4f}, max={cand_conf.max():+.4f}")

    # ===== Step A: Confidence filter (hard gate) =====
    if ablation == 'random':
        # baseline: ignore confidence entirely, sample K random candidates
        rng = np.random.default_rng(0)
        chosen = rng.choice(nCand, size=min(K, nCand), replace=False)
        sel_global = candidate_idx[chosen].tolist()
        print(f"  [Selection][random] picked {len(sel_global)} random candidates.")
        return sel_global

    tau = np.quantile(cand_conf, tau_quantile)
    pass_mask = cand_conf > tau
    n_pass = int(pass_mask.sum())
    print(f"  [Selection] τ = quantile({tau_quantile:.2f}) = {tau:+.4f}; "
          f"{n_pass}/{nCand} candidates passed confidence filter.")
    if n_pass < K:
        print(f"  [Selection] WARNING: only {n_pass} pass filter; reducing K from {K} to {n_pass}.")
        K = n_pass

    # restrict to passing candidates from now on
    pool_local_idx = np.where(pass_mask)[0]                          # local indices into cand_*
    pool_emb = cand_emb[pool_local_idx]                              # (nPool, HLD)
    pool_global = candidate_idx[pool_local_idx]                      # (nPool,) global node idx
    pool_conf = cand_conf[pool_local_idx]
    nPool = len(pool_local_idx)

    # ===== Step B: relevance to V (precomputed, fixed) =====
    # dists_pv[i,j] = ||emb(pool_i) - emb(val_j)||
    dists_pv = np.linalg.norm(pool_emb[:, None, :] - val_emb[None, :, :], axis=-1)  # (nPool, nVal)
    k = min(k_nn_rel, val_emb.shape[0])
    knn = np.sort(dists_pv, axis=1)[:, :k]                           # (nPool, k)
    relevance_raw = -knn.mean(axis=1)                                # higher = closer to V
    rel_norm = (relevance_raw - relevance_raw.min()) / (relevance_raw.max() - relevance_raw.min() + 1e-8)

    # ===== Step C: greedy max-min diversity + relevance =====
    if ablation == 'conf_only':
        # rank purely by confidence among the passing pool, take top K
        order = np.argsort(-pool_conf)[:K]
        sel_global = pool_global[order].tolist()
        print(f"  [Selection][conf_only] picked top-{K} by confidence; "
              f"conf range=[{pool_conf[order].min():.4f}, {pool_conf[order].max():.4f}]")
        return sel_global

    use_div = (ablation in ('full', 'conf_div')) and alpha > 0
    use_rel = (ablation in ('full', 'conf_rel')) and beta  > 0

    selected_local = []
    min_dist_to_S = np.full(nPool, np.inf)

    for step in range(K):
        if step == 0:
            # First pick: most relevant if available, else random
            if use_rel:
                base = rel_norm.copy()
            else:
                base = pool_conf.copy()
            score = base
        else:
            div_norm = (min_dist_to_S - min_dist_to_S.min()) / \
                       (min_dist_to_S.max() - min_dist_to_S.min() + 1e-8)
            score = (alpha * div_norm if use_div else 0) + \
                    (beta  * rel_norm if use_rel else 0)

        # forbid already-picked
        for s in selected_local:
            score[s] = -np.inf
        best = int(np.argmax(score))
        selected_local.append(best)

        # update min distance from each pool point to current S
        d = np.linalg.norm(pool_emb - pool_emb[best], axis=-1)
        min_dist_to_S = np.minimum(min_dist_to_S, d)

    sel_global = pool_global[selected_local].tolist()

    # ===== Diagnostics =====
    print(f"\n  [Selection] greedy picked {len(sel_global)} pseudo-label nodes "
          f"(use_div={use_div}, use_rel={use_rel})")
    if use_rel and use_div:
        # show top-5 with score breakdown
        top5 = selected_local[:5]
        for r, s in enumerate(top5):
            d_s = min_dist_to_S[s]   # not exactly the diversity at the time it was picked, but indicative
            print(f"    rank {r+1}: node={pool_global[s]:4d}, "
                  f"conf={pool_conf[s]:+.4f}, rel_norm={rel_norm[s]:.3f}")
    # spatial spread sanity
    coords_msg = ""
    sel_global_arr = np.array(sel_global)
    print(f"  [Selection] selected node IDs (first 10): {sel_global[:10]}")
    print(f"  [Selection] mean pairwise dist within S "
          f"(emb space) = "
          f"{np.linalg.norm(pool_emb[selected_local][:, None] - pool_emb[selected_local][None, :], axis=-1).sum() / max(len(selected_local)*(len(selected_local)-1), 1):.4f}")
    return sel_global


# =========================================================================
# Confidence aggregation across snapshots
# =========================================================================
def aggregate_snapshots(snapshot_preds_list):
    """
    Args:
        snapshot_preds_list : list of (nSamples, nNodes) ε̂ predictions, length M
    Returns:
        ensemble_mean       : (nSamples, nNodes)  — pseudo-label ε̂
        ensemble_std        : (nSamples, nNodes)  — uncertainty σ
        confidence_per_station: (nNodes,)         — −mean over t of σ(t,·)
    """
    stack = np.stack(snapshot_preds_list, axis=0)   # (M, nSamples, nNodes)
    ensemble_mean = stack.mean(axis=0)
    ensemble_std  = stack.std(axis=0)               # std across snapshots
    confidence_per_station = -ensemble_std.mean(axis=0)   # higher = more confident
    return ensemble_mean.astype(np.float32), ensemble_std.astype(np.float32), confidence_per_station


# =========================================================================
# Main pipeline
# =========================================================================
def main():
    # --- Data ---
    dataParam = {
        'geoMethod': 'average', 'nCompPCA': 40, 'window': 2,
        'poolSize': int(os.environ.get('POOL_SIZE', '12')),
        'batchSize': 512, 'thres': 0.1,
        'geoFeatures': 'full', 'n_unlabeled': n_unlabeled,
    }
    os.environ.setdefault('EVAL_MODE', 'spatial')
    os.environ.setdefault('USE_FPS', '2')
    os.environ.setdefault('EDGE_MODE', 'no_uu')
    os.environ.setdefault('NORM_MODE', 'global')

    trainLoader, validLoader, metadata, validSet = data_semi.dataGen(dataParam, path)
    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    train_station_idx = metadata['train_station_idx']
    valid_station_idx = metadata['valid_station_idx']
    tgt_scl_celsius = float(metadata.get('tgt_global_scl', 1.0))

    # --- Precompute α, β ---
    alpha_per_sample, beta_hat, all_delta = precompute_alpha_beta(trainLoader, metadata)
    attach_to_dataset(trainLoader.dataset, alpha_per_sample, beta_hat)
    attach_to_dataset(validLoader.dataset, alpha_per_sample, beta_hat)

    # --- Edge tensors for Laplacian ---
    Adj = metadata['AdjMatrix']
    edge_src_np, edge_dst_np = np.nonzero(Adj)
    edge_src = torch.LongTensor(edge_src_np).to(device)
    edge_dst = torch.LongTensor(edge_dst_np).to(device)
    edge_w   = torch.FloatTensor(Adj[edge_src_np, edge_dst_np]).to(device)

    # --- Model factory ---
    modelParam = {
        'HLD': 128, 'nMLP': 2, 'nGNN': N_GNN, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'iDim': metadata['iDim'], 'oDim': metadata['oDim'],
        'BN': False, 'Dropout': True, 'conv_type': conv_type,
        'use_gnn': bool(USE_GNN),
    }

    def fresh_model():
        m = network_semi.GNN(modelParam).to(device)
        return m

    # --- wandb ---
    modelName = f"selftrain_{ST_ABLATION}"
    wandb.init(
        entity="urban_prediction", project="Semi-supervised GNN",
        name=f"{modelName}_job{job_id}" if job_id else modelName,
        config={**dataParam, **modelParam,
                'ST_ABLATION': ST_ABLATION,
                'ST_PHASE1_EPOCHS': ST_PHASE1_EPOCHS,
                'ST_PHASE4_EPOCHS': ST_PHASE4_EPOCHS,
                'ST_M_SNAPSHOTS': ST_M_SNAPSHOTS, 'ST_T_CYCLE': ST_T_CYCLE,
                'ST_TAU_QUANTILE': ST_TAU_QUANTILE, 'ST_K': ST_K,
                'ST_ALPHA_DIV': ST_ALPHA_DIV, 'ST_BETA_REL': ST_BETA_REL,
                'ST_LAMBDA_PSEUDO': ST_LAMBDA_PSEUDO,
                'EAD_ALPHA': EAD_ALPHA, 'EAD_BETA': EAD_BETA,
                'LAMBDA_LAP': LAMBDA_LAP},
    )
    log_path = os.path.join(output_dir, f'{modelName}_log')
    with open(log_path, 'w') as f:
        f.write(f"=== Self-training pipeline ===\n")
        f.write(f"ablation={ST_ABLATION}, K={ST_K}, M={ST_M_SNAPSHOTS}, "
                f"τ={ST_TAU_QUANTILE}, α={ST_ALPHA_DIV}, β={ST_BETA_REL}, "
                f"λ_pseudo={ST_LAMBDA_PSEUDO}\n")

    # ============================================================================
    # PHASE 1 : Train base EAD model
    # ============================================================================
    print("\n" + "#"*72)
    print("# PHASE 1 — train base EAD model")
    print("#"*72)
    gnn = fresh_model()
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)

    best_phase1_rmse = float('inf')
    best_phase1_state = None
    for epoch in range(ST_PHASE1_EPOCHS):
        stats, train_rmse = train_ead_epoch(
            trainLoader, gnn, opt, scheduler, device,
            nNodes, edge_src, edge_dst, edge_w, use_pseudo=False)
        eval_metrics = eval_ead(validLoader, gnn, device, nNodes, nNodes_labeled, tgt_scl=tgt_scl_celsius)
        valid_rmse = eval_metrics['rmse_norm']
        if valid_rmse < best_phase1_rmse:
            best_phase1_rmse = valid_rmse
            best_phase1_state = copy.deepcopy(gnn.state_dict())
        if epoch % 50 == 0 or epoch < 3:
            line = (f"P1 ep{epoch:4d}: train_RMSE={train_rmse[0]:.4f}  "
                    f"valid_RMSE={valid_rmse:.4f} ({eval_metrics['rmse_C']:.3f}°C)  "
                    f"LR={scheduler.get_last_lr()[0]:.2e}")
            print(line)
            with open(log_path, 'a') as f: f.write(line + '\n')
        wandb.log({'phase': 1, 'epoch_p1': epoch,
                   'p1/train_rmse': train_rmse[0], 'p1/valid_rmse': valid_rmse,
                   'p1/best_valid_rmse': best_phase1_rmse,
                   'p1/lr': scheduler.get_last_lr()[0]})

    print(f"\n  [Phase 1] best valid_RMSE = {best_phase1_rmse:.4f}")
    with open(log_path, 'a') as f:
        f.write(f"[Phase 1 done] best_valid_RMSE={best_phase1_rmse:.4f}\n")
    # Restore best phase-1 state for snapshot collection start
    gnn.load_state_dict(best_phase1_state)

    # ============================================================================
    # PHASE 1.5 : Snapshot collection with cyclic LR
    # ============================================================================
    print("\n" + "#"*72)
    print(f"# PHASE 1.5 — Snapshot Ensemble  (M={ST_M_SNAPSHOTS} cycles × T={ST_T_CYCLE} epochs)")
    print("#"*72)
    # Reset LR to ST_LR_MAX, switch to cosine warm restarts
    for pg in opt.param_groups: pg['lr'] = ST_LR_MAX
    snap_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=ST_T_CYCLE, T_mult=1, eta_min=ST_LR_MIN)

    snapshot_states = []
    snapshot_valid_rmses = []
    total_snap_epochs = ST_M_SNAPSHOTS * ST_T_CYCLE
    for ep in range(total_snap_epochs):
        stats, train_rmse = train_ead_epoch(
            trainLoader, gnn, opt, snap_scheduler, device,
            nNodes, edge_src, edge_dst, edge_w, use_pseudo=False)
        cyc_idx = (ep + 1) // ST_T_CYCLE
        if (ep + 1) % ST_T_CYCLE == 0:
            # End of a cycle — save snapshot
            snap_state = copy.deepcopy(gnn.state_dict())
            snapshot_states.append(snap_state)
            ev = eval_ead(validLoader, gnn, device, nNodes, nNodes_labeled, tgt_scl=tgt_scl_celsius)
            snapshot_valid_rmses.append(ev['rmse_norm'])
            line = (f"  [Snapshot {len(snapshot_states)}/{ST_M_SNAPSHOTS}] "
                    f"ep={ep+1:4d}, valid_RMSE={ev['rmse_norm']:.4f} "
                    f"({ev['rmse_C']:.3f}°C), LR_at_save={snap_scheduler.get_last_lr()[0]:.2e}")
            print(line)
            with open(log_path, 'a') as f: f.write(line + '\n')
        wandb.log({'phase': 1.5, 'epoch_p15': ep,
                   'p15/lr': snap_scheduler.get_last_lr()[0],
                   'p15/train_rmse': train_rmse[0]})

    # Diagnostic: snapshot valid-RMSE diversity
    sn = np.array(snapshot_valid_rmses)
    print(f"\n  [Snapshot diag] valid_RMSE per snapshot: "
          f"{[f'{v:.4f}' for v in snapshot_valid_rmses]}")
    print(f"  [Snapshot diag] mean={sn.mean():.4f}, std={sn.std():.4f} "
          f"(want std > 0 — confirms snapshots are diverse, not collapsed)")

    # ============================================================================
    # PHASE 2 : Aggregate snapshots
    # ============================================================================
    print("\n" + "#"*72)
    print("# PHASE 2 — aggregate snapshots → confidence + pseudo-labels + embeddings")
    print("#"*72)
    snap_preds = []
    for i, state in enumerate(snapshot_states):
        gnn.load_state_dict(state)
        eps_pred = predict_eps_all(gnn, trainLoader.dataset, device, nNodes)
        snap_preds.append(eps_pred)
        print(f"  [snapshot {i+1}] ε̂ stats: mean={eps_pred.mean():+.4f}, "
              f"std={eps_pred.std():.4f}, "
              f"per-node std (over t): mean={eps_pred.std(axis=0).mean():.4f}")

    ensemble_mean, ensemble_std, conf_per_station = aggregate_snapshots(snap_preds)

    print(f"\n  [Ensemble] ε̂ mean: shape={ensemble_mean.shape}, "
          f"global std={ensemble_mean.std():.4f}")
    print(f"  [Ensemble] σ (across snapshots): "
          f"mean={ensemble_std.mean():.4f}, "
          f"std={ensemble_std.std():.4f}, "
          f"max={ensemble_std.max():.4f}")

    # === CRITICAL SANITY CHECK ===
    # Pseudo-label quality on V (V is held-out — ensemble never saw labels there)
    # Comparing ensemble ε̂ on V to TRUE ε on V tells us how good our pseudo-labels actually are.
    # If this RMSE >> Phase 1 valid_RMSE, pseudo-labels are unreliable and Phase 4 will likely degrade.
    true_eps_all = all_delta - alpha_per_sample[:, None] - beta_hat[None, :]   # (nSamples, nNodes)
    pseudo_rmse_v_per_station = np.sqrt(
        ((ensemble_mean[:, valid_station_idx] - true_eps_all[:, valid_station_idx]) ** 2).mean(axis=0)
    )                                                                          # (nVal,)
    pseudo_rmse_train = np.sqrt(
        ((ensemble_mean[:, train_station_idx] - true_eps_all[:, train_station_idx]) ** 2).mean(axis=0)
    )
    print(f"\n  [Pseudo-label QUALITY check]")
    print(f"    ε̂ RMSE on TRAIN labeled  (seen)    : "
          f"mean={pseudo_rmse_train.mean():.4f}  (should be small — model memorized)")
    print(f"    ε̂ RMSE on VALID labeled (held-out): "
          f"mean={pseudo_rmse_v_per_station.mean():.4f}  std={pseudo_rmse_v_per_station.std():.4f}")
    print(f"    Reference Phase 1 best valid_RMSE   : {best_phase1_rmse:.4f} (Δ space, for context)")
    quality_warnings = []
    if pseudo_rmse_v_per_station.mean() > 1.5 * best_phase1_rmse:
        msg = (f"⚠ Pseudo-label ε̂ RMSE on V = {pseudo_rmse_v_per_station.mean():.4f} "
               f"(> 1.5× Phase 1 best {best_phase1_rmse:.4f}) — pseudo-labels may be unreliable.")
        print(f"    {msg}")
        quality_warnings.append(msg)
    else:
        print(f"    ✓ Ensemble quality on V is comparable to Phase 1 → pseudo-labels usable.")

    # === Over-smoothing diagnostic ===
    # Compare per-timestep spatial-std of pseudo-labels vs true ε.
    # If pseudo std << true std at unlabeled positions, the ensemble has collapsed
    # to a smoothed mean (the GSR-13006 failure mode).
    cand_idx = np.arange(nNodes_labeled, nNodes)   # ← define here so over-smoothing check works
    true_eps_spatial_std   = true_eps_all[:, cand_idx].std(axis=1).mean()
    pseudo_eps_spatial_std = ensemble_mean[:, cand_idx].std(axis=1).mean()
    smooth_ratio = pseudo_eps_spatial_std / (true_eps_spatial_std + 1e-8)
    # Reference: spatial std on TRAIN labeled (where model has seen labels)
    pseudo_eps_train_spatial_std = ensemble_mean[:, train_station_idx].std(axis=1).mean()
    print(f"\n  [Over-smoothing check] mean per-timestep spatial std of ε:")
    print(f"    true ε (cand pool, ground truth)    : {true_eps_spatial_std:.4f}")
    print(f"    pseudo ε (ensemble mean, cand pool) : {pseudo_eps_spatial_std:.4f}")
    print(f"    pseudo ε (ensemble mean, train)     : {pseudo_eps_train_spatial_std:.4f}  (sanity)")
    print(f"    ratio pseudo/true on cand           : {smooth_ratio:.3f}")
    if smooth_ratio < 0.7:
        msg = (f"⚠ Over-smoothing risk: pseudo/true spatial-std ratio = {smooth_ratio:.3f} (< 0.7). "
               f"Ensemble has flattened spatial pattern — like GSR 13006.")
        print(f"    {msg}")
        quality_warnings.append(msg)
    elif smooth_ratio > 1.3:
        msg = (f"⚠ pseudo/true spatial-std ratio = {smooth_ratio:.3f} (> 1.3) — "
               f"ensemble is more variable than truth (overfitting noise?).")
        print(f"    {msg}")
        quality_warnings.append(msg)
    else:
        print(f"    ✓ ratio in [0.7, 1.3] — ensemble preserves spatial diversity.")

    print(f"  [Confidence] per-station: "
          f"min={conf_per_station.min():+.4f}, "
          f"max={conf_per_station.max():+.4f}, "
          f"std={conf_per_station.std():.4f}")
    print(f"    — train labeled : mean conf = {conf_per_station[train_station_idx].mean():+.4f}")
    print(f"    — valid labeled : mean conf = {conf_per_station[valid_station_idx].mean():+.4f}")
    print(f"    — unlabeled cand: mean conf = {conf_per_station[cand_idx].mean():+.4f}")

    # Embeddings from LAST snapshot (most converged)
    gnn.load_state_dict(snapshot_states[-1])
    emb_per_st = per_station_embedding(gnn, trainLoader.dataset, device, nNodes)
    print(f"\n  [Embedding] shape={emb_per_st.shape}, "
          f"per-station norm: mean={np.linalg.norm(emb_per_st, axis=1).mean():.3f}, "
          f"std={np.linalg.norm(emb_per_st, axis=1).std():.3f}")

    # ============================================================================
    # PHASE 3 : Pseudo-label selection
    # ============================================================================
    print("\n" + "#"*72)
    print(f"# PHASE 3 — selection  (ablation = {ST_ABLATION})")
    print("#"*72)
    selected_global = greedy_select(
        emb_all=emb_per_st,
        candidate_idx=cand_idx,
        valid_idx=valid_station_idx,
        confidence_per_station=conf_per_station,
        K=ST_K,
        alpha=ST_ALPHA_DIV,
        beta=ST_BETA_REL,
        k_nn_rel=ST_K_NN_REL,
        tau_quantile=ST_TAU_QUANTILE,
        ablation=ST_ABLATION,
    )

    select_mask_per_station = np.zeros(nNodes, dtype=bool)
    select_mask_per_station[selected_global] = True
    print(f"  [Selection] |S|={select_mask_per_station.sum()} "
          f"(out of {len(cand_idx)} candidates)")

    # === Selection sanity check: selected vs unselected ===
    sel_idx = np.array(selected_global)
    unsel_idx = np.array([i for i in cand_idx if i not in set(sel_idx)])
    sel_conf = conf_per_station[sel_idx]
    unsel_conf = conf_per_station[unsel_idx] if len(unsel_idx) > 0 else np.array([0.0])
    # Mean kNN distance to V — proxy for relevance
    val_emb = emb_per_st[valid_station_idx]
    def mean_knn_to_v(emb):
        d = np.linalg.norm(emb[:, None, :] - val_emb[None, :, :], axis=-1)
        k = min(ST_K_NN_REL, val_emb.shape[0])
        return np.sort(d, axis=1)[:, :k].mean(axis=1)
    sel_relV   = mean_knn_to_v(emb_per_st[sel_idx])
    unsel_relV = mean_knn_to_v(emb_per_st[unsel_idx]) if len(unsel_idx) > 0 else np.array([0.0])

    # In-set diversity (mean min pairwise distance)
    if len(sel_idx) > 1:
        sel_emb = emb_per_st[sel_idx]
        d = np.linalg.norm(sel_emb[:, None, :] - sel_emb[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        sel_minpair = d.min(axis=1).mean()
    else:
        sel_minpair = 0.0

    print(f"\n  [Selection sanity] selected vs unselected (within candidates):")
    print(f"    confidence       : selected mean={sel_conf.mean():+.4f}  | "
          f"unselected mean={unsel_conf.mean():+.4f}  "
          f"(Δ={sel_conf.mean()-unsel_conf.mean():+.4f}, want > 0)")
    print(f"    mean kNN dist→V  : selected mean={sel_relV.mean():.3f}   | "
          f"unselected mean={unsel_relV.mean():.3f}   "
          f"(Δ={sel_relV.mean()-unsel_relV.mean():+.3f}, want < 0)")
    print(f"    in-set min-pair  : {sel_minpair:.3f}  (large = diverse selections)")


    # save selection artifacts for later inspection
    sel_pkl = os.path.join(output_dir, 'selection.pkl')
    with open(sel_pkl, 'wb') as f:
        pickle.dump({
            'selected_global': selected_global,
            'select_mask_per_station': select_mask_per_station,
            'confidence_per_station': conf_per_station,
            'embeddings': emb_per_st,
            'ablation': ST_ABLATION,
            'snapshot_valid_rmses': snapshot_valid_rmses,
        }, f)
    print(f"  [Selection] saved to {sel_pkl}")

    # ============================================================================
    # PHASE 4 : Retrain from scratch with pseudo-labels on K selected
    # ============================================================================
    print("\n" + "#"*72)
    print(f"# PHASE 4 — retrain (λ_pseudo={ST_LAMBDA_PSEUDO}, K={int(select_mask_per_station.sum())})")
    print("#"*72)

    # Re-attach datasets with pseudo-labels and select-mask
    attach_to_dataset(trainLoader.dataset, alpha_per_sample, beta_hat,
                      pseudo_eps_all=ensemble_mean,
                      pseudo_sigma_all=ensemble_std,
                      pseudo_select_mask=select_mask_per_station)
    # Validation set never uses pseudo-labels
    attach_to_dataset(validLoader.dataset, alpha_per_sample, beta_hat)

    gnn2 = fresh_model()
    opt2 = torch.optim.Adam(gnn2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(opt2, gamma=0.9992)

    best_p4_rmse = float('inf')
    best_p4_metrics = None
    for epoch in range(ST_PHASE4_EPOCHS):
        stats, train_rmse = train_ead_epoch(
            trainLoader, gnn2, opt2, scheduler2, device,
            nNodes, edge_src, edge_dst, edge_w,
            use_pseudo=True, lambda_pseudo=ST_LAMBDA_PSEUDO)
        eval_metrics = eval_ead(validLoader, gnn2, device, nNodes, nNodes_labeled, tgt_scl=tgt_scl_celsius)
        valid_rmse = eval_metrics['rmse_norm']
        if valid_rmse < best_p4_rmse:
            best_p4_rmse = valid_rmse
            best_p4_metrics = eval_metrics
            torch.save({
                'epoch': epoch,
                'gnn_state_dict': gnn2.state_dict(),
                'bestLoss': best_p4_rmse,
                'selected_global': selected_global,
            }, os.path.join(output_dir, f'{modelName}.pt'))

        if epoch % 25 == 0 or epoch < 3:
            line = (f"P4 ep{epoch:4d}: "
                    f"train_RMSE={train_rmse[0]:.4f}  "
                    f"valid_RMSE={valid_rmse:.4f} ({eval_metrics['rmse_C']:.3f}°C)  "
                    f"sup={stats['sup_eps']:.4e}  pseudo={stats['pseudo']:.4e}  "
                    f"lap={stats['lap']:.4e}  LR={scheduler2.get_last_lr()[0]:.2e}")
            print(line)
            with open(log_path, 'a') as f: f.write(line + '\n')

        # ratio of weighted pseudo loss vs sup loss — useful to see if pseudo dominates
        pseudo_to_sup = (ST_LAMBDA_PSEUDO * stats['pseudo']) / (stats['sup_eps'] + 1e-8)
        wandb.log({
            'phase': 4, 'epoch_p4': epoch,
            'p4/train_rmse': train_rmse[0],
            'p4/valid_rmse': valid_rmse,
            'p4/valid_rmseC': eval_metrics['rmse_C'],
            'p4/sup_loss':    stats['sup_eps'],
            'p4/pseudo_loss': stats['pseudo'],
            'p4/lap_loss':    stats['lap'],
            'p4/pseudo_to_sup_ratio': pseudo_to_sup,
            'p4/best_valid_rmse': best_p4_rmse,
            'metrics/rmse_norm': eval_metrics['rmse_norm'],
            'metrics/mbe_norm':  eval_metrics['mbe_norm'],
            'metrics/mae_norm':  eval_metrics['mae_norm'],
            'metrics/rmse_C':    eval_metrics['rmse_C'],
            'metrics/mbe_C':     eval_metrics['mbe_C'],
            'metrics/mae_C':     eval_metrics['mae_C'],
            'lr': scheduler2.get_last_lr()[0],
        })

    # ============================================================================
    # Final report
    # ============================================================================
    print("\n" + "="*72)
    print("  FINAL RESULTS")
    print("="*72)
    print(f"  Phase 1 best valid_RMSE             : {best_phase1_rmse:.4f} "
          f"({best_phase1_rmse * tgt_scl_celsius:.3f}°C)")
    print(f"  Phase 4 best valid_RMSE (with PL)   : {best_p4_rmse:.4f} "
          f"({best_p4_metrics['rmse_C']:.3f}°C)")
    delta = best_phase1_rmse - best_p4_rmse
    print(f"  Δ (Phase1 − Phase4)                 : {delta:+.4f} "
          f"({'better with pseudo-labels' if delta > 0 else 'worse with pseudo-labels'})")
    print(f"  Ablation                            : {ST_ABLATION}")
    print(f"  Reference baseline B1 = 0.0428")
    if quality_warnings:
        print(f"\n  ⚠ Phase 2 quality warnings (replayed from earlier):")
        for w in quality_warnings:
            print(f"    · {w}")
        print(f"  → If Phase 4 underperformed, these are the likely cause.")
    else:
        print(f"\n  ✓ No pseudo-label quality warnings raised in Phase 2.")
    print("="*72)
    with open(log_path, 'a') as f:
        f.write(f"\n=== FINAL ===\nphase1_best={best_phase1_rmse:.4f}, "
                f"phase4_best={best_p4_rmse:.4f}, delta={delta:+.4f}, "
                f"ablation={ST_ABLATION}\n")
        if quality_warnings:
            f.write(f"Phase 2 warnings:\n")
            for w in quality_warnings:
                f.write(f"  · {w}\n")


if __name__ == '__main__':
    main()

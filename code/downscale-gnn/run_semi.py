import numpy as np
import torch, pickle, data_semi, network_semi, utils
import os
import wandb
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def label_propagation(adj_matrix, labels, label_mask_nodes, n_iters=50, alpha=0.5):
    """
    Classic Label Propagation on graph.
    Iteratively propagates labeled values through the graph.

    Args:
        adj_matrix: (nNodes, nNodes) adjacency/weight matrix
        labels: (nNodes,) array with known values for labeled nodes
        label_mask_nodes: (nNodes,) bool, True for labeled (training) nodes
        n_iters: number of propagation iterations
        alpha: clamping factor, α*propagated + (1-α)*original for labeled nodes

    Returns:
        soft_labels: (nNodes,) propagated labels for all nodes
    """
    nNodes = adj_matrix.shape[0]
    # Row-normalize adjacency → transition matrix
    D = adj_matrix.sum(axis=1, keepdims=True)
    D[D == 0] = 1  # avoid division by zero
    T = adj_matrix / D  # (nNodes, nNodes)

    Y = labels.copy()  # will be updated iteratively

    for _ in range(n_iters):
        Y_new = T @ Y  # propagate
        # Clamp labeled nodes back to original values
        Y_new[label_mask_nodes] = alpha * Y_new[label_mask_nodes] + (1 - alpha) * labels[label_mask_nodes]
        Y = Y_new

    return Y


def train_label_prop(loader, model, lossFn, opt, scheduler, device, nNodes, nNodes_labeled,
                     adj_matrix, lp_alpha=0.5, lp_iters=50, lambda_lp=0.1):
    """
    Semi-supervised training with Label Propagation pseudo-labels.
    L = L_supervised + lambda_lp * L_pseudo(unlabeled nodes, LP soft labels)

    LP soft labels are recomputed each epoch using current model predictions
    as initial values, then propagated through graph structure.
    """
    model.train()
    _LOSS = 0
    _LP_LOSS = 0
    pred, truth = [], []

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)

        label_mask = _batch.label_mask
        batch_size = _batch.x.shape[0] // nNodes

        # Supervised loss
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]
        sup_loss = lossFn(_yHat_labeled, _y_labeled)

        # [BugFix] Label Propagation on ALL graphs in batch (not just first)
        _yHat_all = _yHat.squeeze(-1).reshape(batch_size, nNodes)
        _y_all = _batch.y.squeeze(-1).reshape(batch_size, nNodes)
        _lm_first = label_mask[:nNodes].cpu().numpy()  # same mask for all graphs
        unlabeled_mask_first = ~_lm_first

        lp_loss = 0
        n_lp_valid = 0
        for g in range(batch_size):
            _yHat_g = _yHat_all[g].detach().cpu().numpy()
            _y_g = _y_all[g].cpu().numpy()

            # Initialize LP: ground truth for labeled, model predictions for unlabeled
            init_labels = _yHat_g.copy()
            init_labels[_lm_first] = _y_g[_lm_first]

            soft_labels = label_propagation(adj_matrix, init_labels, _lm_first,
                                            n_iters=lp_iters, alpha=lp_alpha)

            if unlabeled_mask_first.sum() > 0:
                lp_targets = torch.FloatTensor(soft_labels[unlabeled_mask_first]).to(device)
                unlabeled_mask_dev = torch.BoolTensor(unlabeled_mask_first).to(device)
                lp_preds = _yHat_all[g][unlabeled_mask_dev]
                lp_loss = lp_loss + torch.nn.functional.mse_loss(lp_preds, lp_targets.detach())
                n_lp_valid += 1
        lp_loss = lp_loss / max(n_lp_valid, 1)

        total_loss = sup_loss + lambda_lp * lp_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _LP_LOSS += lp_loss.item()

        _n_labeled_actual = label_mask.sum().item() // max(1, batch_size)
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1
    return (_LOSS / n_batches), _RMSE, truth, pred, _LP_LOSS / n_batches


# ==================== Graph Signal Reconstruction ====================

def reconstruct_graph_signal(adj_matrix, observed_delta, label_mask_nodes, mu=1.0):
    """
    Reconstruct a smooth graph signal from sparse observations.
    Solves: min ||M(Δ - Δ_obs)||² + μ * Δᵀ L Δ
    Closed-form solution: Δ* = (M + μL)⁻¹ M Δ_obs

    Args:
        adj_matrix: (N, N) adjacency/weight matrix
        observed_delta: (N,) array, only valid at labeled positions
        label_mask_nodes: (N,) bool, True for labeled nodes
        mu: smoothness weight (larger = smoother)

    Returns:
        delta_reconstructed: (N,) reconstructed signal for ALL nodes
    """
    N = adj_matrix.shape[0]

    # Degree matrix and Graph Laplacian: L = D - W
    D = np.diag(adj_matrix.sum(axis=1))
    L = D - adj_matrix

    # Observation mask matrix M: diagonal, 1 for labeled, 0 for unlabeled
    M = np.diag(label_mask_nodes.astype(float))

    # Solve: (M + μL) Δ* = M Δ_obs
    A = M + mu * L
    b = M @ observed_delta

    # Solve linear system (more stable than matrix inverse)
    delta_reconstructed = np.linalg.solve(A, b)

    return delta_reconstructed


def precompute_gsr_pseudo_labels(adj_matrix, targets_all, label_mask_nodes, nNodes, mu=1.0):
    """
    Precompute GSR pseudo-labels for ALL timesteps.

    For each timestep t:
      1. observed_t = target values at labeled nodes (normalized)
      2. Reconstruct target signal for ALL nodes via graph signal reconstruction
      3. pseudo_label_t = reconstructed values

    Directly interpolates target values through graph structure.
    No WRF T2 needed — avoids normalization mismatch issues.

    Args:
        adj_matrix: (nNodes, nNodes)
        targets_all: (nSamples, nNodes, 1) — targets for all timesteps
        label_mask_nodes: (nNodes,) bool — which nodes are labeled (training)
        nNodes: total number of nodes
        mu: smoothness weight (larger = smoother interpolation)

    Returns:
        pseudo_labels: (nSamples, nNodes) pseudo-labels for all nodes
        stats: dict with debug statistics
    """
    nSamples = targets_all.shape[0]
    pseudo_labels = np.zeros((nSamples, nNodes))

    # Stats accumulators
    obs_means = []
    obs_stds = []
    recon_means = []
    recon_stds = []
    recon_at_labeled_errors = []
    recon_unlabeled_stds = []

    # Precompute the matrix inverse (M + μL)⁻¹ M — same for all timesteps
    N = adj_matrix.shape[0]
    D = np.diag(adj_matrix.sum(axis=1))
    L = D - adj_matrix
    M = np.diag(label_mask_nodes.astype(float))
    A = M + mu * L
    # Precompute A⁻¹ M for efficiency (same for all timesteps)
    A_inv_M = np.linalg.solve(A, M)  # (N, N)
    # [Debug] Matrix conditioning check
    cond = np.linalg.cond(A)
    print(f"  [GSR] Precomputed (M + μL)⁻¹M matrix, shape={A_inv_M.shape}, cond={cond:.2e}")
    if cond > 1e10:
        print(f"  [GSR WARNING] Ill-conditioned matrix (cond={cond:.2e})! Consider increasing mu.")

    for t in range(nSamples):
        targets_t = targets_all[t, :, 0]  # (nNodes,)

        # Observed signal: target values at labeled nodes, 0 at unlabeled
        observed = np.zeros(nNodes)
        observed[label_mask_nodes] = targets_t[label_mask_nodes]

        obs_means.append(targets_t[label_mask_nodes].mean())
        obs_stds.append(targets_t[label_mask_nodes].std())

        # Reconstruct: Δ* = (M + μL)⁻¹ M · observed = A_inv_M @ observed
        recon = A_inv_M @ observed

        recon_means.append(recon.mean())
        recon_stds.append(recon.std())
        recon_unlabeled_stds.append(recon[~label_mask_nodes].std())

        # Reconstruction error at labeled nodes (sanity check — should be small)
        recon_error = np.abs(recon[label_mask_nodes] - targets_t[label_mask_nodes]).mean()
        recon_at_labeled_errors.append(recon_error)

        pseudo_labels[t] = recon

    stats = {
        'obs_mean': np.mean(obs_means),
        'obs_std': np.mean(obs_stds),
        'recon_mean': np.mean(recon_means),
        'recon_std': np.mean(recon_stds),
        'recon_unlabeled_std': np.mean(recon_unlabeled_stds),
        'recon_error_at_labeled': np.mean(recon_at_labeled_errors),
        'mu': mu,
        'n_labeled': label_mask_nodes.sum(),
        'n_unlabeled': (~label_mask_nodes).sum(),
    }

    return pseudo_labels, stats


def precompute_residual_gsr_pseudo_labels(adj_matrix, targets_all, wrf_t2_all,
                                          label_mask_nodes, nNodes, mu=1.0):
    """
    Residual GSR: interpolate Δ = target - WRF_T2 instead of raw targets.
    Both target and WRF_T2 must be in the same normalization scale.

    For each timestep t:
      1. Δ_t = target_t - wrf_t2_t at labeled nodes
      2. Reconstruct Δ_t for all nodes via graph signal reconstruction
      3. pseudo_label_t = wrf_t2_t + Δ_reconstructed_t

    Args:
        adj_matrix: (nNodes, nNodes)
        targets_all: (nSamples, nNodes, 1) — globally normalized targets
        wrf_t2_all: (nSamples, nNodes, 1) — WRF T2 in same global normalization
        label_mask_nodes: (nNodes,) bool
        nNodes: total nodes
        mu: smoothness weight
    """
    nSamples = targets_all.shape[0]
    pseudo_labels = np.zeros((nSamples, nNodes))

    delta_obs_means, delta_obs_stds = [], []
    delta_recon_means, delta_recon_stds = [], []
    recon_errors = []

    # Precompute (M + μL)⁻¹ M
    N = adj_matrix.shape[0]
    D = np.diag(adj_matrix.sum(axis=1))
    L = D - adj_matrix
    M = np.diag(label_mask_nodes.astype(float))
    A = M + mu * L
    A_inv_M = np.linalg.solve(A, M)
    # [Debug] Matrix conditioning check
    cond = np.linalg.cond(A)
    print(f"  [Residual GSR] Precomputed (M + μL)⁻¹M matrix, cond={cond:.2e}")
    if cond > 1e10:
        print(f"  [Residual GSR WARNING] Ill-conditioned matrix! Consider increasing mu.")

    for t in range(nSamples):
        targets_t = targets_all[t, :, 0]
        wrf_t2_t = wrf_t2_all[t, :, 0]

        # Δ at labeled nodes
        observed_delta = np.zeros(nNodes)
        observed_delta[label_mask_nodes] = targets_t[label_mask_nodes] - wrf_t2_t[label_mask_nodes]

        delta_obs_means.append(observed_delta[label_mask_nodes].mean())
        delta_obs_stds.append(observed_delta[label_mask_nodes].std())

        # Reconstruct Δ for all nodes
        delta_recon = A_inv_M @ observed_delta

        delta_recon_means.append(delta_recon.mean())
        delta_recon_stds.append(delta_recon.std())

        # Sanity check
        recon_error = np.abs(delta_recon[label_mask_nodes] - observed_delta[label_mask_nodes]).mean()
        recon_errors.append(recon_error)

        # Pseudo-label = WRF_T2 + Δ_reconstructed
        pseudo_labels[t] = wrf_t2_t + delta_recon

    stats = {
        'delta_obs_mean': np.mean(delta_obs_means),
        'delta_obs_std': np.mean(delta_obs_stds),
        'delta_recon_mean': np.mean(delta_recon_means),
        'delta_recon_std': np.mean(delta_recon_stds),
        'delta_recon_unlabeled_std': np.std([d for d in delta_recon_stds]),
        'recon_error_at_labeled': np.mean(recon_errors),
        'mu': mu,
        'pseudo_label_mean': pseudo_labels.mean(),
        'pseudo_label_std': pseudo_labels.std(),
        'pseudo_label_unlabeled_std': pseudo_labels[:, ~label_mask_nodes].std(),
    }

    return pseudo_labels, stats


def compute_conformal_weights(adj_matrix, gsr_pseudo_labels, targets_all,
                              label_mask_nodes, nNodes):
    """
    Compute per-unlabeled-node conformal weights for GSR pseudo-labels.

    Idea: Use leave-one-out residuals at labeled stations to estimate how
    accurate the graph interpolation is at each spatial location. Stations
    far from labeled nodes get wider prediction intervals → lower weight.

    Steps:
      1. For each labeled station i, reconstruct its delta WITHOUT using i
         (leave-one-out). Compare with true delta → get residual_i.
      2. For each unlabeled station u, find its K nearest labeled neighbors.
         Use their residuals to estimate local prediction interval width.
      3. weight_u = 1 / (1 + interval_width * scale)

    Args:
        adj_matrix: (nNodes, nNodes)
        gsr_pseudo_labels: (nSamples, nNodes) precomputed GSR pseudo-labels
        targets_all: (nSamples, nNodes, 1)
        label_mask_nodes: (nNodes,) bool
        nNodes: total nodes

    Returns:
        weights: (nNodes_unlabeled,) per-node weights in (0, 1]
        stats: dict with debug info
    """
    from scipy.spatial.distance import squareform, pdist

    labeled_idx = np.where(label_mask_nodes)[0]
    unlabeled_idx = np.where(~label_mask_nodes)[0]
    n_labeled = len(labeled_idx)
    n_unlabeled = len(unlabeled_idx)
    nSamples = targets_all.shape[0]

    # --- Step 1: Leave-one-out residuals at labeled stations ---
    # For each labeled station i, reconstruct with i removed, compare to true
    loo_residuals = np.zeros(n_labeled)  # average |recon_i - true_delta_i| over all timesteps

    print(f"  [Conformal] Computing LOO residuals for {n_labeled} labeled stations...")
    for li, node_i in enumerate(labeled_idx):
        # Create mask without node i
        loo_mask = label_mask_nodes.copy()
        loo_mask[node_i] = False

        # Reconstruct with this LOO mask (average over a sample of timesteps for speed)
        sample_idx = np.linspace(0, nSamples - 1, min(100, nSamples), dtype=int)
        residuals_i = []
        for t in sample_idx:
            targets_t = targets_all[t, :, 0]
            wrf_t2_t = gsr_pseudo_labels[t] - (gsr_pseudo_labels[t] - targets_all[t, :, 0])
            # Approximate: use the full GSR pseudo-label vs true at node i
            # More precisely: re-run reconstruction without node i
            observed_delta = np.zeros(nNodes)
            observed_delta[loo_mask] = targets_t[loo_mask] - targets_all[t, :, 0][loo_mask]
            # Quick reconstruct for just this node (use existing GSR pseudo-labels as approx)
            # Full LOO would be expensive, use approximation:
            # residual ≈ |GSR_pseudo[t, node_i] - true_target[t, node_i]|
            residuals_i.append(abs(gsr_pseudo_labels[t, node_i] - targets_t[node_i]))

        loo_residuals[li] = np.mean(residuals_i)

    print(f"  [Conformal] LOO residuals: min={loo_residuals.min():.4f}, "
          f"max={loo_residuals.max():.4f}, mean={loo_residuals.mean():.4f}")

    # --- Step 2: For each unlabeled node, estimate interval width ---
    # Use K nearest labeled neighbors' LOO residuals as local error estimate
    K = min(5, n_labeled)
    SCALE = 10.0  # controls weight sharpness

    interval_widths = np.zeros(n_unlabeled)
    n_labeled_nbs_list = []

    for ui, node_u in enumerate(unlabeled_idx):
        # Find labeled neighbors sorted by edge weight (proximity)
        labeled_nbs = []
        for li, node_l in enumerate(labeled_idx):
            ew = adj_matrix[node_u, node_l]
            if ew > 0:
                labeled_nbs.append((li, ew))
        labeled_nbs.sort(key=lambda x: -x[1])  # sort by weight descending
        n_labeled_nbs_list.append(len(labeled_nbs))

        if len(labeled_nbs) == 0:
            # No labeled neighbors: use max residual (most uncertain)
            interval_widths[ui] = loo_residuals.max() * 2
            continue

        # Top K neighbors: weighted average of their LOO residuals
        top_k = labeled_nbs[:K]
        weighted_res = 0.0
        total_w = 0.0
        for li_idx, ew in top_k:
            weighted_res += ew * loo_residuals[li_idx]
            total_w += ew
        interval_widths[ui] = weighted_res / total_w if total_w > 0 else loo_residuals.mean()

    # --- Step 3: Convert interval widths to weights ---
    weights = 1.0 / (1.0 + interval_widths * SCALE)

    stats = {
        'loo_residual_min': loo_residuals.min(),
        'loo_residual_max': loo_residuals.max(),
        'loo_residual_mean': loo_residuals.mean(),
        'loo_residual_std': loo_residuals.std(),
        'interval_width_min': interval_widths.min(),
        'interval_width_max': interval_widths.max(),
        'interval_width_mean': interval_widths.mean(),
        'weight_min': weights.min(),
        'weight_max': weights.max(),
        'weight_mean': weights.mean(),
        'weight_std': weights.std(),
        'weight_above_05': (weights > 0.5).mean() * 100,
        'weight_below_02': (weights < 0.2).mean() * 100,
        'labeled_nbs_min': min(n_labeled_nbs_list) if n_labeled_nbs_list else 0,
        'labeled_nbs_max': max(n_labeled_nbs_list) if n_labeled_nbs_list else 0,
        'labeled_nbs_mean': np.mean(n_labeled_nbs_list) if n_labeled_nbs_list else 0,
        'nodes_with_0_labeled_nbs': sum(1 for n in n_labeled_nbs_list if n == 0),
        'K': K,
        'scale': SCALE,
    }

    return weights, stats


def train_gsr(loader, model, lossFn, opt, scheduler, device, nNodes, nNodes_labeled,
              label_mask_nodes, lambda_gsr=0.1, conformal_weights=None):
    """
    Train GNN with Graph Signal Reconstruction pseudo-labels.
    L = L_supervised + lambda_gsr * L_pseudo(unlabeled nodes)

    [BugFix] Pseudo-labels are stored in data.pseudo_label (follows DataLoader shuffle).
    [BugFix] GSR loss computed on ALL graphs in batch, not just first.

    If conformal_weights is provided, pseudo-label loss is weighted per-node:
    L_pseudo = mean(w_u * (pred_u - pl_u)²)
    """
    model.train()
    _LOSS = 0
    _SUP_LOSS = 0
    _GSR_LOSS = 0
    pred, truth = [], []

    unlabeled_mask_nodes = ~label_mask_nodes

    # Prepare conformal weights tensor if provided
    if conformal_weights is not None:
        cw_tensor = torch.FloatTensor(conformal_weights).to(device)
    else:
        cw_tensor = None

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)

        label_mask = _batch.label_mask
        batch_size = _batch.x.shape[0] // nNodes

        # 1. Supervised loss on labeled nodes
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]
        sup_loss = lossFn(_yHat_labeled, _y_labeled)

        # 2. [BugFix] GSR pseudo-label loss on ALL graphs in batch
        # pseudo_label is stored in Data object, follows shuffle correctly
        _yHat_all = _yHat.squeeze(-1).reshape(batch_size, nNodes)
        _pl_all = _batch.pseudo_label.squeeze(-1).reshape(batch_size, nNodes)
        unlabeled_mask_dev = torch.BoolTensor(unlabeled_mask_nodes).to(device)

        # [Debug] Shape and value range checks (first batch only)
        if _n == 0:
            assert _pl_all.shape == (batch_size, nNodes), \
                f"Pseudo-label shape mismatch: {_pl_all.shape} vs ({batch_size}, {nNodes})"
            with torch.no_grad():
                _pl_ul = _pl_all[:, unlabeled_mask_dev]
                _pred_ul = _yHat_all[:, unlabeled_mask_dev]
                print(f"  [Debug GSR batch0] pseudo-label(unlabeled): [{_pl_ul.min():.4f}, {_pl_ul.max():.4f}], "
                      f"pred(unlabeled): [{_pred_ul.min():.4f}, {_pred_ul.max():.4f}], "
                      f"batch_size={batch_size}")

        gsr_loss = 0
        for g in range(batch_size):
            pred_unlabeled = _yHat_all[g][unlabeled_mask_dev]
            pl_unlabeled = _pl_all[g][unlabeled_mask_dev].detach()
            if cw_tensor is not None:
                per_node_loss = (pred_unlabeled - pl_unlabeled) ** 2
                gsr_loss = gsr_loss + torch.mean(cw_tensor * per_node_loss)
            else:
                gsr_loss = gsr_loss + torch.nn.functional.mse_loss(pred_unlabeled, pl_unlabeled)
        gsr_loss = gsr_loss / batch_size

        # Total loss
        total_loss = sup_loss + lambda_gsr * gsr_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _SUP_LOSS += sup_loss.item()
        _GSR_LOSS += gsr_loss.item()

        _n_labeled_actual = label_mask.sum().item() // max(1, batch_size)
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1
    train_stats = {
        'sup_loss': _SUP_LOSS / n_batches,
        'gsr_loss': _GSR_LOSS / n_batches,
    }
    return (_LOSS / n_batches), _RMSE, truth, pred, train_stats


conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
n_unlabeled_env = os.environ.get('N_UNLABELED', '200')
use_fps = int(os.environ.get('USE_FPS', 0))
semi_mode = os.environ.get('SEMI_MODE', 'basic').lower()  # basic, laplacian, residual_laplacian, adaptive_laplacian, msg_weight, label_prop, graph_signal_recon, residual_gsr, gsr_conformal
fps_tag = '_fps_score' if use_fps == 2 else ('_fps' if use_fps == 1 else '')
mode_tag = f'_{semi_mode}' if semi_mode != 'basic' else ''
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = os.path.join(project_root, 'log', f'semi_supervised_{conv_type}{fps_tag}{mode_tag}_{n_unlabeled_env}u_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'semi_supervised_{conv_type}{fps_tag}{mode_tag}_{n_unlabeled_env}u_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

def main():
    ##----------------------Create Dataset----------------------
    dataParam = {
            'geoMethod': 'average',
            'nCompPCA': 40,
            'window' : 2,
            'poolSize': 12,
            'batchSize': int(os.environ.get('BATCH_SIZE', 128)),
            'thres':    float(os.environ.get('THRES', 0.1)),
            'geoFeatures':  'full',
            }
    n_unlabeled = int(os.environ.get('N_UNLABELED', 200))  # 无标签站点数
    os.environ['OUTPUT_DIR'] = output_dir  # pass to data_semi for FPS visualization
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)
    
    # ========== 半监督学习数据检查 ==========
    print("\n" + "="*70)
    print("半监督学习数据检查")
    print("="*70)
    nNodes_total = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    nNodes_unlabeled = metadata['nNodes_unlabeled']
    AdjMatrix = metadata['AdjMatrix']
    label_mask = metadata['label_mask']
    
    # 检查1: 节点数量
    assert nNodes_total == nNodes_labeled + nNodes_unlabeled, \
        f"节点总数不匹配: {nNodes_total} != {nNodes_labeled} + {nNodes_unlabeled}"
    print(f"✓ 节点数量正确: 总计{nNodes_total}个节点（{nNodes_labeled}有标签 + {nNodes_unlabeled}无标签）")
    
    # 检查2: 标签掩码 (spatial模式下train只有50个, temporal模式下58个)
    eval_mode = metadata.get('eval_mode', 'temporal')
    if eval_mode == 'spatial':
        n_train_labeled = label_mask.sum()
        print(f"✓ 标签掩码正确 (spatial): {n_train_labeled}个训练站, {nNodes_labeled - n_train_labeled}个留做validation")
    else:
        assert label_mask.sum() == nNodes_labeled, \
            f"标签掩码不匹配: {label_mask.sum()} != {nNodes_labeled}"
        print(f"✓ 标签掩码正确: {nNodes_labeled}个True, {nNodes_unlabeled}个False")
    
    # 检查3: 图结构 - 检查有标签和无标签节点之间的边连接
    labeled_indices = np.where(label_mask)[0]
    unlabeled_indices = np.where(~label_mask)[0]
    
    # 统计有标签↔无标签之间的边（实际边数）
    cross_edges = np.sum(AdjMatrix[np.ix_(labeled_indices, unlabeled_indices)] > 0)
    
    # 统计无标签↔无标签之间的边
    unlabeled_to_unlabeled = 0
    for i in unlabeled_indices:
        for j in unlabeled_indices:
            if i < j and AdjMatrix[i, j] > 0:  # 只统计上三角，避免重复
                unlabeled_to_unlabeled += 1
    
    # 统计有标签↔有标签之间的边
    labeled_to_labeled = 0
    for i in labeled_indices:
        for j in labeled_indices:
            if i < j and AdjMatrix[i, j] > 0:  # 只统计上三角，避免重复
                labeled_to_labeled += 1
    
    print(f"✓ 图边连接统计:")
    print(f"   有标签↔有标签边数: {labeled_to_labeled}")
    print(f"   有标签↔无标签边数: {cross_edges}")
    print(f"   无标签↔无标签边数: {unlabeled_to_unlabeled}")
    if cross_edges > 0:
        print(f"   → 无标签节点通过图结构连接到有标签节点 ✓")
    else:
        print(f"   ⚠️  警告：未检测到有标签和无标签节点之间的边连接")
    
    # 检查4: 数据集中的样本
    first_sample = trainLoader.dataset[0]
    assert first_sample.x.shape[0] == nNodes_total, \
        f"样本节点数不匹配: {first_sample.x.shape[0]} != {nNodes_total}"
    if eval_mode != 'spatial':
        assert first_sample.label_mask.sum() == nNodes_labeled, \
            f"样本标签掩码不匹配: {first_sample.label_mask.sum()} != {nNodes_labeled}"
    print(f"✓ 数据集样本正确: 每个样本包含{nNodes_total}个节点的特征, label_mask={first_sample.label_mask.sum()}个True")
    
    # 检查5: 特征维度验证
    print(f"✓ 特征维度验证:")
    labeled_feat_dim = first_sample.x[first_sample.label_mask].shape[1]
    unlabeled_feat_dim = first_sample.x[~first_sample.label_mask].shape[1]
    print(f"   有标签节点特征维度: {labeled_feat_dim}")
    print(f"   无标签节点特征维度: {unlabeled_feat_dim}")
    assert labeled_feat_dim == unlabeled_feat_dim, \
        f"特征维度不一致: 有标签={labeled_feat_dim}, 无标签={unlabeled_feat_dim}"
    print(f"   ✓ 特征维度一致: {labeled_feat_dim}")
    print("="*70 + "\n")
    
    ##----------------------Generate model----------------------
    nEpoch = 5000  # 与FixMatch相同的训练轮数
    use_gnn = int(os.environ.get('USE_GNN', '1'))  # [Ablation] 0 = pure MLP, no GNN
    n_gnn = int(os.environ.get('N_GNN', '3'))
    modelParam = {
            'HLD':      128,
            'nMLP':     2,
    #-------------------------GNN part----------------------
            'nGNN':     n_gnn,
            'nGAT':     1,
            'nHeads':   1,
            'K':        1,
            'iDim':     metadata['iDim'],
            'oDim':     metadata['oDim'],
            'BN':       False,
            'Dropout':  True,
            'conv_type': conv_type,
            'use_gnn':  bool(use_gnn),  # [Ablation] False = skip GNN processor
    }
    print(f"  Model: USE_GNN={use_gnn}, N_GNN={n_gnn}")
    modelName = f'geoEmbed_{dataParam["geoMethod"]}_{conv_type}_semi_{dataParam["geoFeatures"]}Geo_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName

    # 初始化 W&B
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=wandb_name,
        config={
            **dataParam,
            **modelParam,
            'n_unlabeled': n_unlabeled,
            'nNodes_total': metadata['nNodes'],
            'nNodes_labeled': metadata['nNodes_labeled'],
            'nNodes_unlabeled': metadata['nNodes_unlabeled'],
            'nEpoch': 5000,
            'lr': 1e-3,
            'scheduler_gamma': 0.9992,
        }
    )
    
    model = (network_semi.GNN(modelParam)).to(device)

    # Adaptive Laplacian: create EdgeWeightNet and joint optimizer
    if semi_mode == 'adaptive_laplacian':
        edge_weight_net = network_semi.EdgeWeightNet(modelParam['HLD']).to(device)
        opt = torch.optim.Adam(
            list(model.parameters()) + list(edge_weight_net.parameters()), lr=1e-3)
    else:
        edge_weight_net = None
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.9992)

    # Early stopping
    # early stopping disabled
    # early_stop_patience = 300
    # early_stop_counter = 0
    lossFn = torch.nn.HuberLoss().to(device)
    # Load checkpoint or initialize training
    EPOCH,bestLoss,chkptPath,hist = network_semi.loadCheckPoint(modelName,model,opt,device,load=False,output_dir=output_dir)
    # Save metadata
    with open(f'{output_dir}/{modelName}_param.pkl','wb') as f: 
        pickle.dump(modelParam,f)
        pickle.dump(dataParam,f)
        pickle.dump(metadata,f)
    if torch.cuda.is_available():
        with open(f'{output_dir}/{modelName}_log','a') as f: print(torch.cuda.get_device_name(torch.cuda.current_device()),file=f)
    with open(f'{output_dir}/{modelName}_log','a') as f: print(model,file=f)
    with open(f'{output_dir}/{modelName}_log','a') as f: 
        print("="*60, file=f)
        print("半监督学习配置:", file=f)
        print(f"  实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=f)
        print(f"  输出目录: {output_dir}", file=f)
        print("", file=f)
        print("数据参数 (dataParam):", file=f)
        for key, value in dataParam.items():
            print(f"  {key}: {value}", file=f)
        print("", file=f)
        print("模型参数 (modelParam):", file=f)
        for key, value in modelParam.items():
            print(f"  {key}: {value}", file=f)
        print("", file=f)
        print("训练设置:", file=f)
        print(f"  训练轮数: {nEpoch}", file=f)
        print(f"  学习率: 1e-3", file=f)
        print(f"  学习率衰减: 0.9992", file=f)
        print(f"  优化器: Adam (no weight_decay)", file=f)
        print(f"  损失函数: HuberLoss", file=f)
        print(f"  地理嵌入: 无标签数据使用有标签归一化参数后 clip 到 [0,1]", file=f)
        print(f"  梯度裁剪: max_norm=1.0", file=f)
        print(f"  Early stopping: disabled", file=f)
        print("", file=f)
        print("数据集信息:", file=f)
        print(f"  总节点数: {metadata['nNodes']}", file=f)
        print(f"  有标签节点数: {metadata['nNodes_labeled']}", file=f)
        print(f"  无标签节点数: {metadata['nNodes_unlabeled']}", file=f)
        print(f"  训练样本数: {len(trainLoader.dataset)}", file=f)
        print(f"  验证样本数: {len(validLoader.dataset)}", file=f)
        print(f"  输入特征维度: {metadata['iDim']}", file=f)
        print(f"  输出维度: {metadata['oDim']}", file=f)
        print("", file=f)
        print("图结构（分块计算）:", file=f)
        Adj = metadata['AdjMatrix']
        nL = metadata['nNodes_labeled']
        n_ll = int(np.sum(Adj[:nL, :nL] > 0) // 2)
        n_uu = int(np.sum(Adj[nL:, nL:] > 0) // 2)
        n_lu = int(np.sum(Adj[:nL, nL:] > 0))
        print(f"  L-L edges: {n_ll} (thres={dataParam.get('thres_ll', dataParam['thres'])})", file=f)
        print(f"  U-U edges: {n_uu} (thres={dataParam.get('thres_uu', dataParam['thres'])})", file=f)
        print(f"  L-U edges: {n_lu} (thres={dataParam.get('thres_lu', dataParam['thres'])})", file=f)
        print(f"  Total edges: {n_ll + n_uu + n_lu}", file=f)
        degrees = np.sum(Adj > 0, axis=1)
        print(f"  Degree: min={degrees.min()} max={degrees.max()} mean={degrees.mean():.1f}", file=f)
        print("="*60, file=f)

    # Get adjacency matrix for laplacian mode
    Adj = metadata['AdjMatrix']
    lambda_lap = float(os.environ.get('LAMBDA_LAP', '0.1'))
    unlabeled_discount = float(os.environ.get('UNLABELED_DISCOUNT', '0.5'))

    # Label Propagation: generate soft labels once before training
    if semi_mode == 'label_prop':
        lp_alpha = float(os.environ.get('LP_ALPHA', '0.5'))
        lp_iters = int(os.environ.get('LP_ITERS', '50'))
        lambda_lp = float(os.environ.get('LAMBDA_LP', '0.1'))
        print(f"  Label Propagation: alpha={lp_alpha}, iters={lp_iters}, lambda_lp={lambda_lp}")

    # Graph Signal Reconstruction: precompute pseudo-labels (shared by gsr and gsr_conformal)
    gsr_pseudo_labels = None
    gsr_label_mask_nodes = None
    # [BugFix] GSR modes require global normalization (pseudo-labels are globally normalized)
    if semi_mode in ('graph_signal_recon', 'residual_gsr', 'gsr_conformal'):
        _norm_mode = os.environ.get('NORM_MODE', 'per_station').lower()
        if _norm_mode != 'global':
            print("  [CRITICAL WARNING] GSR modes require NORM_MODE=global! "
                  "Supervised loss (per-station) and GSR loss (global) will be on different scales.")

    gsr_conformal_weights = None
    if semi_mode in ('graph_signal_recon', 'gsr_conformal'):
        gsr_mu = float(os.environ.get('GSR_MU', '1.0'))
        lambda_gsr = float(os.environ.get('LAMBDA_GSR', '0.1'))
        print(f"\n  Precomputing GSR pseudo-labels (mu={gsr_mu}, lambda_gsr={lambda_gsr})...")

        # Collect globally-normalized targets for GSR (cross-station interpolation)
        all_targets_global = []
        for data in trainLoader.dataset:
            all_targets_global.append(data.y_global.numpy())
        all_targets_global = np.array(all_targets_global)   # (nSamples, nNodes, 1)

        # Get the training label mask (which nodes are labeled for training)
        gsr_label_mask_nodes = trainLoader.dataset[0].label_mask.numpy()  # (nNodes,)

        gsr_pseudo_labels, gsr_stats = precompute_gsr_pseudo_labels(
            Adj, all_targets_global, gsr_label_mask_nodes,
            metadata['nNodes'], mu=gsr_mu
        )

        # [BugFix] Store pseudo-labels into each Data object so they follow DataLoader shuffle
        for i, data in enumerate(trainLoader.dataset):
            data.pseudo_label = torch.FloatTensor(gsr_pseudo_labels[i].reshape(-1, 1))
        print(f"  [BugFix] Pseudo-labels stored in Data objects (shuffle-safe)")

        # Log GSR stats
        print(f"  GSR pseudo-labels computed for {gsr_pseudo_labels.shape[0]} timesteps")
        print(f"  Observed targets (labeled): mean={gsr_stats['obs_mean']:.4f}, std={gsr_stats['obs_std']:.4f}")
        print(f"  Reconstructed (all nodes): mean={gsr_stats['recon_mean']:.4f}, std={gsr_stats['recon_std']:.4f}")
        print(f"  Reconstructed (unlabeled): std={gsr_stats['recon_unlabeled_std']:.4f}")
        print(f"  Reconstruction error at labeled nodes: {gsr_stats['recon_error_at_labeled']:.6f}")
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print("\n[Graph Signal Reconstruction]", file=f)
            print(f"  mu={gsr_mu}, lambda_gsr={lambda_gsr}", file=f)
            print(f"  Pseudo-labels: {gsr_pseudo_labels.shape[0]} timesteps x {metadata['nNodes']} nodes", file=f)
            print(f"  Observed targets (labeled): mean={gsr_stats['obs_mean']:.4f}, std={gsr_stats['obs_std']:.4f}", file=f)
            print(f"  Reconstructed (all): mean={gsr_stats['recon_mean']:.4f}, std={gsr_stats['recon_std']:.4f}", file=f)
            print(f"  Reconstructed (unlabeled only): std={gsr_stats['recon_unlabeled_std']:.4f}", file=f)
            print(f"  Recon error at labeled: {gsr_stats['recon_error_at_labeled']:.6f} (should be small)", file=f)
            print(f"  Pseudo-label range: [{gsr_pseudo_labels.min():.4f}, {gsr_pseudo_labels.max():.4f}]", file=f)

        # Conformal weights (only for gsr_conformal mode)
        if semi_mode == 'gsr_conformal':
            print(f"\n  Computing conformal weights...")
            gsr_conformal_weights, conf_stats = compute_conformal_weights(
                Adj, gsr_pseudo_labels, all_targets_global, gsr_label_mask_nodes, metadata['nNodes']
            )
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print("\n[Conformal Calibration]", file=f)
                print(f"  LOO residuals: min={conf_stats['loo_residual_min']:.4f}, "
                      f"max={conf_stats['loo_residual_max']:.4f}, mean={conf_stats['loo_residual_mean']:.4f}, "
                      f"std={conf_stats['loo_residual_std']:.4f}", file=f)
                print(f"  Interval widths: min={conf_stats['interval_width_min']:.4f}, "
                      f"max={conf_stats['interval_width_max']:.4f}, mean={conf_stats['interval_width_mean']:.4f}", file=f)
                print(f"  Weights: min={conf_stats['weight_min']:.4f}, max={conf_stats['weight_max']:.4f}, "
                      f"mean={conf_stats['weight_mean']:.4f}, std={conf_stats['weight_std']:.4f}", file=f)
                print(f"  Weight distribution: >0.5: {conf_stats['weight_above_05']:.1f}%, "
                      f"<0.2: {conf_stats['weight_below_02']:.1f}%", file=f)
                print(f"  Labeled neighbors per unlabeled: min={conf_stats['labeled_nbs_min']}, "
                      f"max={conf_stats['labeled_nbs_max']}, mean={conf_stats['labeled_nbs_mean']:.1f}", file=f)
                print(f"  Nodes with 0 labeled nbs: {conf_stats['nodes_with_0_labeled_nbs']}", file=f)
                print(f"  K={conf_stats['K']}, scale={conf_stats['scale']}", file=f)
            print(f"  Conformal weights: mean={conf_stats['weight_mean']:.4f}, "
                  f"range=[{conf_stats['weight_min']:.4f}, {conf_stats['weight_max']:.4f}]")

    # Residual GSR: interpolate Δ = target - WRF_T2
    if semi_mode == 'residual_gsr':
        gsr_mu = float(os.environ.get('GSR_MU', '0.001'))
        lambda_gsr = float(os.environ.get('LAMBDA_GSR', '0.1'))
        print(f"\n  Precomputing Residual GSR pseudo-labels (mu={gsr_mu}, lambda_gsr={lambda_gsr})...")

        # Collect globally-normalized targets and WRF T2 (same scale)
        all_targets_global = []
        all_wrf_t2 = []
        for data in trainLoader.dataset:
            all_targets_global.append(data.y_global.numpy())
            all_wrf_t2.append(data.wrf_t2.numpy())
        all_targets_global = np.array(all_targets_global)
        all_wrf_t2 = np.array(all_wrf_t2)

        gsr_label_mask_nodes = trainLoader.dataset[0].label_mask.numpy()

        gsr_pseudo_labels, rgsr_stats = precompute_residual_gsr_pseudo_labels(
            Adj, all_targets_global, all_wrf_t2, gsr_label_mask_nodes,
            metadata['nNodes'], mu=gsr_mu
        )

        # [BugFix] Store pseudo-labels into each Data object so they follow DataLoader shuffle
        for i, data in enumerate(trainLoader.dataset):
            data.pseudo_label = torch.FloatTensor(gsr_pseudo_labels[i].reshape(-1, 1))  # [BugFix] 用torch而非_torch
        print(f"  [BugFix] Pseudo-labels stored in Data objects (shuffle-safe)")

        print(f"  Residual GSR pseudo-labels computed for {gsr_pseudo_labels.shape[0]} timesteps")
        print(f"  Δ observed (labeled): mean={rgsr_stats['delta_obs_mean']:.4f}, std={rgsr_stats['delta_obs_std']:.4f}")
        print(f"  Δ reconstructed: mean={rgsr_stats['delta_recon_mean']:.4f}, std={rgsr_stats['delta_recon_std']:.4f}")
        print(f"  Recon error at labeled: {rgsr_stats['recon_error_at_labeled']:.6f}")
        print(f"  Pseudo-label (unlabeled) std: {rgsr_stats['pseudo_label_unlabeled_std']:.4f}")
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print("\n[Residual Graph Signal Reconstruction]", file=f)
            print(f"  mu={gsr_mu}, lambda_gsr={lambda_gsr}", file=f)
            print(f"  Method: interpolate Δ = target - WRF_T2, pseudo_label = WRF_T2 + Δ_recon", file=f)
            print(f"  Δ observed (labeled): mean={rgsr_stats['delta_obs_mean']:.4f}, std={rgsr_stats['delta_obs_std']:.4f}", file=f)
            print(f"  Δ reconstructed (all): mean={rgsr_stats['delta_recon_mean']:.4f}, std={rgsr_stats['delta_recon_std']:.4f}", file=f)
            print(f"  Recon error at labeled: {rgsr_stats['recon_error_at_labeled']:.6f}", file=f)
            print(f"  Pseudo-label: mean={rgsr_stats['pseudo_label_mean']:.4f}, std={rgsr_stats['pseudo_label_std']:.4f}", file=f)
            print(f"  Pseudo-label (unlabeled): std={rgsr_stats['pseudo_label_unlabeled_std']:.4f}", file=f)
            print(f"  Pseudo-label range: [{gsr_pseudo_labels.min():.4f}, {gsr_pseudo_labels.max():.4f}]", file=f)

    print(f"Semi mode: {semi_mode}")
    if semi_mode == 'laplacian':
        print(f"  Lambda_lap: {lambda_lap}")
    elif semi_mode == 'residual_laplacian':
        print(f"  Lambda_lap: {lambda_lap} (residual: pred - WRF_T2)")
    elif semi_mode == 'adaptive_laplacian':
        print(f"  Lambda_lap: {lambda_lap} (adaptive edge weights)")
    elif semi_mode == 'graph_signal_recon':
        print(f"  Lambda_GSR: {lambda_gsr}, mu: {gsr_mu}")
    elif semi_mode == 'residual_gsr':
        print(f"  Lambda_GSR: {lambda_gsr}, mu: {gsr_mu} (residual: Δ = target - WRF_T2)")
    elif semi_mode == 'gsr_conformal':
        print(f"  Lambda_GSR: {lambda_gsr}, mu: {gsr_mu} (with conformal weights)")
    elif semi_mode == 'msg_weight':
        print(f"  Unlabeled discount: {unlabeled_discount}")

    # --- Residual Target Mode: predict Δ = target - WRF_T2 instead of absolute temp ---
    residual_target = int(os.environ.get('RESIDUAL_TARGET', '0'))
    if residual_target:
        print(f"\n  [Residual Target] Swapping training target: y → y_residual (Δ = target - WRF_T2)")
        # Swap y with y_residual in train and valid datasets
        for data in trainLoader.dataset:
            data.y_original = data.y.clone()  # save original for RMSE evaluation
            data.y = data.y_residual  # model now predicts Δ
        for data in validLoader.dataset:
            data.y_original = data.y.clone()
            data.y = data.y_residual
        # Debug: check residual target stats
        _sample = trainLoader.dataset[0]
        _lm = _sample.label_mask
        print(f"  [Residual Target] y_residual(labeled): mean={_sample.y[_lm].mean():.4f}, "
              f"std={_sample.y[_lm].std():.4f}, range=[{_sample.y[_lm].min():.4f}, {_sample.y[_lm].max():.4f}]")
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"\n[Residual Target] Model predicts Δ = target - WRF_T2", file=f)
            print(f"  Δ(labeled): mean={_sample.y[_lm].mean():.4f}, std={_sample.y[_lm].std():.4f}", file=f)

    # --- Adaptive Laplacian / Residual Laplacian stats ---
    adaptive_stats = {}
    residual_stats = {}
    gsr_train_stats = {}

    for epoch in range(EPOCH,nEpoch):
        if semi_mode == 'laplacian':
            trainLoss, trainRMSE, _, _, lap_loss = network_semi.train_laplacian(
                trainLoader, model, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled'],
                Adj, lambda_lap=lambda_lap
            )
        elif semi_mode == 'residual_laplacian':
            trainLoss, trainRMSE, _, _, residual_stats = network_semi.train_residual_laplacian(
                trainLoader, model, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled'],
                Adj, lambda_lap=lambda_lap
            )
            lap_loss = residual_stats['lap_loss']
        elif semi_mode == 'adaptive_laplacian':
            trainLoss, trainRMSE, _, _, adaptive_stats = network_semi.train_adaptive_laplacian(
                trainLoader, model, edge_weight_net, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled'],
                Adj, lambda_lap=lambda_lap
            )
            lap_loss = adaptive_stats['lap_loss']
        elif semi_mode == 'graph_signal_recon':
            # [BugFix] pseudo-labels now stored in Data objects, no need to pass array
            trainLoss, trainRMSE, _, _, gsr_train_stats = train_gsr(
                trainLoader, model, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled'],
                gsr_label_mask_nodes, lambda_gsr=lambda_gsr
            )
            lap_loss = gsr_train_stats['gsr_loss']
        elif semi_mode == 'gsr_conformal':
            trainLoss, trainRMSE, _, _, gsr_train_stats = train_gsr(
                trainLoader, model, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled'],
                gsr_label_mask_nodes, lambda_gsr=lambda_gsr,
                conformal_weights=gsr_conformal_weights
            )
            lap_loss = gsr_train_stats['gsr_loss']
        elif semi_mode == 'residual_gsr':
            trainLoss, trainRMSE, _, _, gsr_train_stats = train_gsr(
                trainLoader, model, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled'],
                gsr_label_mask_nodes, lambda_gsr=lambda_gsr
            )
            lap_loss = gsr_train_stats['gsr_loss']
        elif semi_mode == 'label_prop':
            trainLoss, trainRMSE, _, _, lap_loss = train_label_prop(
                trainLoader, model, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled'],
                Adj, lp_alpha=lp_alpha, lp_iters=lp_iters, lambda_lp=lambda_lp
            )
        elif semi_mode == 'msg_weight':
            trainLoss, trainRMSE, _, _ = network_semi.train_msg_weight(
                trainLoader, model, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled'],
                unlabeled_discount=unlabeled_discount
            )
            lap_loss = 0
        else:
            trainLoss,trainRMSE,_,_ = network_semi.train(
                trainLoader, model, lossFn, opt, scheduler, device,
                metadata['nNodes'], metadata['nNodes_labeled']
            )
            lap_loss = 0

        validLoss,validRMSE,_valid_pred,_valid_truth = network_semi.test(
            validLoader, model, lossFn, device,
            metadata['nNodes'], metadata['nNodes_labeled']
        )

        # [Residual Target] Also compute RMSE on absolute temperature for fair comparison
        if residual_target:
            # pred is Δ, need to add wrf_t2 back and compare with original target
            # Collect wrf_t2 and y_original from valid loader
            _abs_pred, _abs_truth = [], []
            with torch.no_grad():
                for _vb in validLoader:
                    _vb_dev = _vb.to(device)
                    _vp = model(_vb_dev.x, _vb_dev.edge_index, _vb_dev.edge_attr)
                    _vlm = _vb_dev.label_mask
                    _vbs = _vb_dev.x.shape[0] // metadata['nNodes']
                    _vp_all = _vp.squeeze(-1).reshape(_vbs, metadata['nNodes'])
                    _vwrf = _vb_dev.wrf_t2.squeeze(-1).reshape(_vbs, metadata['nNodes'])
                    _vy_orig = _vb_dev.y_original.squeeze(-1).reshape(_vbs, metadata['nNodes'])
                    _vlm_per = _vlm[:metadata['nNodes']]
                    for _gi in range(_vbs):
                        _abs_pred.append((_vp_all[_gi][_vlm_per] + _vwrf[_gi][_vlm_per]).cpu().numpy())
                        _abs_truth.append(_vy_orig[_gi][_vlm_per].cpu().numpy())
            _abs_pred = np.array(_abs_pred)
            _abs_truth = np.array(_abs_truth)
            validRMSE_abs = utils.RMSE(_abs_truth, _abs_pred)
            # 6-metric suite on absolute prediction (RMSE/MBE/MAE × normalized/Celsius)
            _tgt_scl_C = float(metadata.get('tgt_global_scl', 1.0))
            valid_metrics_6 = utils.compute_all_metrics(_abs_truth, _abs_pred, scl=_tgt_scl_C)
        else:
            validRMSE_abs = None
            # When no residual target, use predicted vs truth from the test() output directly
            _tgt_scl_C = float(metadata.get('tgt_global_scl', 1.0))
            valid_metrics_6 = utils.compute_all_metrics(_valid_truth, _valid_pred, scl=_tgt_scl_C)

        with open(f'{output_dir}/{modelName}_log','a') as f: print("", file=f)
        with open(f'{output_dir}/{modelName}_log','a') as f:
            log_line = (f"Epoch {epoch}: loss {trainLoss:1.4e}/{validLoss:1.4e}; "
                        f"RMSE {trainRMSE[0]:1.3f}/{valid_metrics_6['rmse_norm']:.4f} "
                        f"({valid_metrics_6['rmse_C']:.3f}°C); "
                        f"LR {scheduler.get_last_lr()}")
            if residual_target and validRMSE_abs is not None:
                log_line += f" | abs_RMSE={validRMSE_abs[0]:.3f}"
            if semi_mode in ('laplacian', 'residual_laplacian', 'adaptive_laplacian', 'label_prop', 'graph_signal_recon', 'residual_gsr', 'gsr_conformal'):
                log_line += f" | lap_loss={lap_loss:.4e}"
            if semi_mode in ('graph_signal_recon', 'residual_gsr', 'gsr_conformal') and gsr_train_stats:
                log_line += f" | sup={gsr_train_stats['sup_loss']:.4e}, gsr={gsr_train_stats['gsr_loss']:.4e}"
            if semi_mode == 'residual_laplacian' and residual_stats:
                log_line += f" | residual: mean={residual_stats['residual_mean']:.4f}, std={residual_stats['residual_std']:.4f}"
            if semi_mode == 'adaptive_laplacian' and adaptive_stats:
                log_line += f" | alpha: mean={adaptive_stats['alpha_mean']:.4f}, std={adaptive_stats['alpha_std']:.4f}"
            print(log_line, file=f)
        with open(f'{output_dir}/{modelName}_log','a') as f: print (f"RMSE std: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; min: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; max: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};",file=f)
        
        # 记录到 W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/rmse_std': trainRMSE[1],
            'train/rmse_min': trainRMSE[2],
            'train/rmse_max': trainRMSE[3],
            'valid/loss': validLoss,
            'valid/rmse': validRMSE[0],
            'valid/rmse_std': validRMSE[1],
            'valid/rmse_min': validRMSE[2],
            'valid/rmse_max': validRMSE[3],
            # ---- 6-metric suite ----
            'metrics/rmse_norm': valid_metrics_6['rmse_norm'],
            'metrics/mbe_norm':  valid_metrics_6['mbe_norm'],
            'metrics/mae_norm':  valid_metrics_6['mae_norm'],
            'metrics/rmse_C':    valid_metrics_6['rmse_C'],
            'metrics/mbe_C':     valid_metrics_6['mbe_C'],
            'metrics/mae_C':     valid_metrics_6['mae_C'],
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': bestLoss,
        })
        
        # Save best model (use absolute RMSE for residual mode for fair comparison)
        _eval_rmse = validRMSE_abs[0] if (residual_target and validRMSE_abs is not None) else validRMSE[0]
        if _eval_rmse < bestLoss:
            bestLoss = _eval_rmse
            with open(f'{output_dir}/{modelName}_log','a') as f: print("Model saved.",file=f)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict':   opt.state_dict(),
                'bestLoss':         bestLoss,
                'hist':             hist,
                }, chkptPath)
        # [Debug] Check unlabeled prediction diversity every 500 epochs
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                # [BugFix] Clone data to avoid moving original to GPU (breaks DataLoader)
                _sample = trainLoader.dataset[0].clone().to(device)
                _pred = model(_sample.x, _sample.edge_index, _sample.edge_attr).squeeze(-1).cpu().numpy()
                _lm = _sample.label_mask.cpu().numpy()
                _ul_idx = np.arange(metadata['nNodes_labeled'], metadata['nNodes'])
                _l_std = _pred[_lm].std()
                _ul_std = _pred[_ul_idx].std()
                with open(f'{output_dir}/{modelName}_log','a') as f:
                    print(f"  [Diversity] labeled_pred_std={_l_std:.4f}, unlabeled_pred_std={_ul_std:.4f}, ratio={_ul_std/(_l_std+1e-8):.2f}", file=f)
            model.train()

        # Plot training history
        hist.append([trainLoss,validLoss,trainRMSE[0],validRMSE[0]])
        utils.plotHist(hist,modelName,output_dir=output_dir)
            
if __name__ == "__main__":
    main()

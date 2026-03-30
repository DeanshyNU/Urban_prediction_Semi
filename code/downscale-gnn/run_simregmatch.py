"""
SimRegMatch adapted for GNN urban temperature prediction.
Based on: "Deep Semi-Supervised Regression via Pseudo-Label Filtering and Calibration"
(Jo et al., Applied Soft Computing, 2024)

Key adaptations for GNN:
- MC Dropout filtering: same as original (multiple forward passes, variance-based)
- Similarity calibration: use graph neighbors instead of feature cosine similarity
- Augmentation: feature-level noise instead of image augmentation
- Single model, no EMA teacher

Usage:
  CONV_TYPE=graphconv python -u code/downscale-gnn/run_simregmatch.py
"""
import numpy as np
import torch, pickle, data_semi, network_semi, utils
import os
import wandb
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# Config
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')

# SimRegMatch hyperparameters
MC_ITERS = int(os.environ.get('MC_ITERS', '5'))          # MC forward passes
PERCENTILE = float(os.environ.get('PERCENTILE', '95'))    # uncertainty filter percentile
BETA = float(os.environ.get('BETA', '0.6'))               # calibration blend (model vs graph-neighbor)
LAMBDA_U = float(os.environ.get('LAMBDA_U', '0.01'))      # unlabeled loss weight
WEAK_NOISE = float(os.environ.get('WEAK_NOISE', '0.05'))  # weak augmentation noise std
STRONG_NOISE = float(os.environ.get('STRONG_NOISE', '0.2'))  # strong augmentation noise std
RAMP_EPOCHS = int(os.environ.get('RAMP_EPOCHS', '100'))   # ramp-up period for lambda_u

if job_id:
    output_dir = os.path.join(project_root, 'log', f'simregmatch_{conv_type}_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'simregmatch_{conv_type}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


def precompute_graph_calibration_info(adj_matrix, nNodes_labeled, nNodes_unlabeled):
    """
    Pre-compute labeled neighbor info for graph-based calibration.
    For each unlabeled node, find its labeled neighbors and edge weights.

    Returns:
        neighbor_info: dict[u] = {'labeled': [(local_idx, edge_weight), ...]}
    """
    neighbor_info = {}
    for u in range(nNodes_unlabeled):
        u_global = nNodes_labeled + u
        labeled_nbs = []
        for nb in range(nNodes_labeled):
            ew = adj_matrix[u_global, nb]
            if ew > 0:
                labeled_nbs.append((nb, ew))
        neighbor_info[u] = labeled_nbs
    return neighbor_info


def graph_calibrate_pseudo_labels(pseudo_labels, labeled_targets, neighbor_info,
                                  nNodes_unlabeled, beta):
    """
    Calibrate pseudo-labels using graph-neighbor labeled targets.
    SimRegMatch original: cosine similarity in feature space → softmax → weighted labeled targets
    Our adaptation: edge weights in graph → weighted labeled targets (more natural for spatial data)

    calibrated = beta * model_pred + (1-beta) * graph_weighted_labeled_mean

    Args:
        pseudo_labels: (batch_size, nNodes_unlabeled) model predictions
        labeled_targets: (batch_size, nNodes_labeled) true targets
        neighbor_info: pre-computed labeled neighbors per unlabeled node
        nNodes_unlabeled: number of unlabeled nodes
        beta: blend factor (0.6 = 60% model, 40% graph reference)

    Returns:
        calibrated: (batch_size, nNodes_unlabeled)
        calib_stats: dict with calibration statistics
    """
    batch_size = pseudo_labels.shape[0]
    calibrated = pseudo_labels.copy()
    calib_diffs = []
    n_calibrated = 0
    n_uncalibrated = 0

    for u in range(nNodes_unlabeled):
        nbs = neighbor_info[u]
        if len(nbs) == 0:
            # No labeled neighbors → keep original pseudo-label
            n_uncalibrated += 1
            continue

        n_calibrated += 1
        # Edge-weight-weighted average of labeled targets
        for t in range(batch_size):
            weighted_sum = 0.0
            weight_sum = 0.0
            for nb_idx, ew in nbs:
                weighted_sum += ew * labeled_targets[t, nb_idx]
                weight_sum += ew
            graph_ref = weighted_sum / weight_sum

            # Calibration: blend model prediction with graph reference
            original = pseudo_labels[t, u]
            calibrated[t, u] = beta * original + (1 - beta) * graph_ref
            calib_diffs.append(abs(calibrated[t, u] - original))

    calib_diffs = np.array(calib_diffs) if calib_diffs else np.array([0.0])
    calib_stats = {
        'n_calibrated_nodes': n_calibrated,
        'n_uncalibrated_nodes': n_uncalibrated,
        'calib_diff_mean': calib_diffs.mean(),
        'calib_diff_max': calib_diffs.max(),
        'calib_diff_median': np.median(calib_diffs),
    }
    return calibrated, calib_stats


def apply_feature_noise(x, noise_std):
    """DEPRECATED: kept for reference. Use structured augmentation below."""
    if noise_std > 0:
        return x + torch.randn_like(x) * noise_std
    return x


# ==================== Structured Data Augmentation ====================
# Feature layout (1343 dims total, window=2):
#   WRF block: 315 dims = 5 time steps x 63 dims/step
#     Time step order: [current(63), hist[-2](63), hist[-1](63), fut[+1](63), fut[+2](63)]
#     Within each 63-dim step: 7 variables x 9 grid points
#       Tair(0-8), Tskin(9-17), Tsoil(18-26), Humid(27-35), Irrad(36-44), WindX(45-53), WindY(54-62)
#   CLMS: 3 dims (dynamic)
#   UrbanFeature: 17 dims (static)
#   GeoEmbed: remaining dims (static)
import random as _random

WRF_VAR_GROUPS = {
    'Tair': (0, 9), 'Tskin': (9, 18), 'Tsoil': (18, 27),
    'Humid': (27, 36), 'Irrad': (36, 45), 'WindX': (45, 54), 'WindY': (54, 63),
}
WRF_STEP_DIM = 63
WRF_WINDOW = 2
WRF_TOTAL_STEPS = 2 * WRF_WINDOW + 1  # 5
WRF_TOTAL_DIM = WRF_STEP_DIM * WRF_TOTAL_STEPS  # 315
CLMS_DIM = 3
URBAN_DIM = 17


def _get_feature_layout(total_dim):
    geo_dim = total_dim - WRF_TOTAL_DIM - CLMS_DIM - URBAN_DIM
    return {
        'wrf': (0, WRF_TOTAL_DIM),
        'clms': (WRF_TOTAL_DIM, WRF_TOTAL_DIM + CLMS_DIM),
        'urban': (WRF_TOTAL_DIM + CLMS_DIM, WRF_TOTAL_DIM + CLMS_DIM + URBAN_DIM),
        'geo': (WRF_TOTAL_DIM + CLMS_DIM + URBAN_DIM, total_dim),
        'geo_dim': geo_dim,
    }


def _build_wrf_var_indices(var_names):
    offsets = [i * WRF_STEP_DIM for i in range(WRF_TOTAL_STEPS)]
    indices = []
    for step_offset in offsets:
        for var in var_names:
            start, end = WRF_VAR_GROUPS[var]
            indices.extend(range(step_offset + start, step_offset + end))
    return indices


def _build_wrf_timestep_indices(step_idx):
    offset = step_idx * WRF_STEP_DIM
    return list(range(offset, offset + WRF_STEP_DIM))


def _drop_edges(edge_index, edge_attr, drop_rate):
    if drop_rate <= 0 or edge_index.size(1) == 0:
        return edge_index, edge_attr
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
    if keep_mask.sum() < num_edges * 0.5:
        keep_mask = torch.rand(num_edges, device=edge_index.device) > 0.5
    return edge_index[:, keep_mask], (edge_attr[keep_mask] if edge_attr is not None else None)


def structured_augment_weak(x, edge_index, edge_attr):
    """Weak augmentation: keep Tair+Tskin, mask 1 low group, mask 10% GeoEmbed."""
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    low_choices = ['WindX', 'WindY', 'CLMS']
    chosen = _random.choice(low_choices)
    if chosen == 'CLMS':
        s, e = layout['clms']
        x_aug[:, s:e] = 0.0
    else:
        x_aug[:, _build_wrf_var_indices([chosen])] = 0.0

    geo_start, geo_end = layout['geo']
    geo_mask = torch.rand(layout['geo_dim'], device=x.device) > 0.10
    x_aug[:, geo_start:geo_end] *= geo_mask.float().unsqueeze(0)

    return x_aug, edge_index, edge_attr


def structured_augment_strong(x, edge_index, edge_attr):
    """Strong augmentation: keep Tair only, mask 2-3 med/low groups, mask 1 time step,
    mask 30% GeoEmbed, drop 25% edges."""
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    candidates = ['Humid', 'Irrad', 'WindX', 'WindY', 'CLMS']
    n_mask = _random.choice([2, 3])
    chosen_groups = _random.sample(candidates, min(n_mask, len(candidates)))
    wrf_vars = [v for v in chosen_groups if v != 'CLMS']
    if wrf_vars:
        x_aug[:, _build_wrf_var_indices(wrf_vars)] = 0.0
    if 'CLMS' in chosen_groups:
        s, e = layout['clms']
        x_aug[:, s:e] = 0.0

    step_to_mask = _random.choice([1, 2, 3, 4])
    x_aug[:, _build_wrf_timestep_indices(step_to_mask)] = 0.0

    geo_start, geo_end = layout['geo']
    geo_mask = torch.rand(layout['geo_dim'], device=x.device) > 0.30
    x_aug[:, geo_start:geo_end] *= geo_mask.float().unsqueeze(0)

    edge_index_aug, edge_attr_aug = _drop_edges(edge_index, edge_attr, drop_rate=0.25)
    return x_aug, edge_index_aug, edge_attr_aug


def train_simregmatch(loader, model, lossFn, opt, scheduler, device,
                      nNodes, nNodes_labeled, adj_matrix, neighbor_info,
                      mc_iters, percentile, beta, lambda_u, current_lambda,
                      weak_noise, strong_noise):
    """
    SimRegMatch training step:
    1. Weak augmentation → MC Dropout → pseudo-labels + uncertainty
    2. Filter by uncertainty (percentile-based threshold)
    3. Calibrate pseudo-labels using graph neighbors
    4. Strong augmentation → model prediction → loss vs calibrated pseudo-labels
    """
    model.train()  # dropout active for MC Dropout
    nNodes_unlabeled = nNodes - nNodes_labeled

    _LOSS = 0
    _LABELED_LOSS = 0
    _UNLABELED_LOSS = 0
    pred, truth = [], []

    # Per-epoch statistics
    total_filtered = 0
    total_unlabeled_samples = 0
    all_uncertainties = []
    all_pred_diffs = []
    threshold_val = None

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        label_mask = _batch.label_mask
        batch_size = _batch.x.shape[0] // nNodes

        # ==================== Step 1: MC Dropout on weak augmentation ====================
        mc_preds = []
        for mc_i in range(mc_iters):
            x_weak, ei_weak, ea_weak = structured_augment_weak(
                _batch.x, _batch.edge_index, _batch.edge_attr)
            with torch.no_grad():
                _yHat_mc = model(x_weak, ei_weak, ea_weak)
            unlabeled_pred = _yHat_mc[~label_mask].squeeze(-1).reshape(batch_size, nNodes_unlabeled)
            mc_preds.append(unlabeled_pred)

        mc_preds = torch.stack(mc_preds, dim=0)  # (mc_iters, batch_size, nNodes_unlabeled)
        pseudo_labels = mc_preds.mean(dim=0)  # (batch_size, nNodes_unlabeled)
        mc_variance = mc_preds.var(dim=0)  # (batch_size, nNodes_unlabeled)

        # Per-node uncertainty (sum across output dim, here dim=1 so just the variance itself)
        uncertainty = mc_variance  # (batch_size, nNodes_unlabeled)

        # ==================== Step 2: Dynamic threshold filtering ====================
        uncertainty_np = uncertainty.detach().cpu().numpy().flatten()
        all_uncertainties.extend(uncertainty_np.tolist())
        threshold_val = np.percentile(uncertainty_np, q=percentile)
        mask = (uncertainty < threshold_val).float()  # (batch_size, nNodes_unlabeled)

        n_passed = mask.sum().item()
        n_total = mask.numel()
        total_filtered += (n_total - n_passed)
        total_unlabeled_samples += n_total

        # ==================== Step 3: Graph-based calibration ====================
        # Get labeled targets for this batch
        labeled_targets = _batch.y[label_mask].squeeze(-1).reshape(batch_size, nNodes_labeled)
        labeled_targets_np = labeled_targets.detach().cpu().numpy()
        pseudo_labels_np = pseudo_labels.detach().cpu().numpy()

        calibrated_np, _ = graph_calibrate_pseudo_labels(
            pseudo_labels_np, labeled_targets_np, neighbor_info,
            nNodes_unlabeled, beta
        )
        calibrated = torch.FloatTensor(calibrated_np).to(device)

        # ==================== Step 4: Strong augmentation → loss ====================
        # Labeled loss (with weak augmentation, standard supervised)
        x_weak_labeled, ei_weak_l, ea_weak_l = structured_augment_weak(
            _batch.x, _batch.edge_index, _batch.edge_attr)
        _yHat_labeled_full = model(x_weak_labeled, ei_weak_l, ea_weak_l)
        _yHat_labeled = _yHat_labeled_full[label_mask]
        _y_labeled = _batch.y[label_mask]
        labeled_loss = lossFn(_yHat_labeled, _y_labeled)

        # Unlabeled loss (with strong augmentation)
        x_strong, ei_strong, ea_strong = structured_augment_strong(
            _batch.x, _batch.edge_index, _batch.edge_attr)
        _yHat_strong_full = model(x_strong, ei_strong, ea_strong)
        _yHat_unlabeled = _yHat_strong_full[~label_mask].squeeze(-1).reshape(batch_size, nNodes_unlabeled)

        # Masked MSE loss (only on filtered pseudo-labels)
        pseudo_diff = (_yHat_unlabeled - calibrated.detach()) ** 2
        masked_loss = (pseudo_diff * mask)
        if mask.sum() > 0:
            unlabeled_loss = masked_loss.sum() / mask.sum()
        else:
            unlabeled_loss = torch.tensor(0.0).to(device)

        # Total loss
        total_loss = labeled_loss + current_lambda * unlabeled_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _LABELED_LOSS += labeled_loss.item()
        _UNLABELED_LOSS += unlabeled_loss.item()

        # Track strong-weak prediction difference (key metric for augmentation effectiveness)
        with torch.no_grad():
            pred_diff_sw = (_yHat_unlabeled - pseudo_labels).abs().mean().item()
            all_pred_diffs.append(pred_diff_sw)

        # Record labeled predictions for RMSE
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        _truth = _y_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1

    filter_rate = total_filtered / total_unlabeled_samples * 100 if total_unlabeled_samples > 0 else 0
    all_uncertainties = np.array(all_uncertainties)
    all_pred_diffs = np.array(all_pred_diffs) if all_pred_diffs else np.array([0.0])

    train_stats = {
        'labeled_loss': _LABELED_LOSS / n_batches,
        'unlabeled_loss': _UNLABELED_LOSS / n_batches,
        'filter_rate': filter_rate,
        'threshold': threshold_val if threshold_val is not None else 0,
        'uncertainty_mean': all_uncertainties.mean(),
        'uncertainty_median': np.median(all_uncertainties),
        'uncertainty_max': all_uncertainties.max(),
        'pseudo_labels_used_pct': 100 - filter_rate,
        'pred_diff_strong_weak': all_pred_diffs.mean(),
    }

    return (_LOSS / n_batches), _RMSE, truth, pred, train_stats


def main():
    # ==================== Data ====================
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': int(os.environ.get('POOL_SIZE', '12')),
        'batchSize': int(os.environ.get('BATCH_SIZE', 128)),
        'thres': 0.1,
        'geoFeatures': 'full',
    }
    n_unlabeled = int(os.environ.get('N_UNLABELED', 200))
    os.environ['OUTPUT_DIR'] = output_dir
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)

    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    nNodes_unlabeled = metadata['nNodes_unlabeled']
    adj_matrix = metadata['AdjMatrix']

    # Pre-compute graph calibration info
    neighbor_info = precompute_graph_calibration_info(adj_matrix, nNodes_labeled, nNodes_unlabeled)
    n_with_labeled_nbs = sum(1 for u in range(nNodes_unlabeled) if len(neighbor_info[u]) > 0)

    print(f"Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})")
    print(f"Unlabeled nodes with labeled neighbors: {n_with_labeled_nbs}/{nNodes_unlabeled}")
    print(f"SimRegMatch config: MC_ITERS={MC_ITERS}, PERCENTILE={PERCENTILE}, "
          f"BETA={BETA}, LAMBDA_U={LAMBDA_U}")
    print(f"Augmentation: structured (weak=Tair+Tskin kept, mask 1 low; strong=Tair kept, mask 2-3 groups+timestep+edges)")

    # ==================== Model ====================
    nEpoch = 5000
    modelParam = {
        'HLD': 128,
        'nMLP': 2,
        'nGNN': 3,
        'nGAT': 1,
        'nHeads': 1,
        'K': 1,
        'iDim': metadata['iDim'],
        'oDim': metadata['oDim'],
        'BN': False,
        'Dropout': True,
        'conv_type': conv_type,
    }

    model = network_semi.GNN(modelParam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)

    bestLoss = np.inf
    hist = []

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_{conv_type}_simregmatch_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName
    chkptPath = f'{output_dir}/{modelName}.pt'

    # ==================== W&B ====================
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=wandb_name,
        config={
            **dataParam,
            **modelParam,
            'method': 'simregmatch',
            'mc_iters': MC_ITERS,
            'percentile': PERCENTILE,
            'beta': BETA,
            'lambda_u': LAMBDA_U,
            'weak_noise': WEAK_NOISE,
            'strong_noise': STRONG_NOISE,
            'ramp_epochs': RAMP_EPOCHS,
            'n_unlabeled': n_unlabeled,
            'nEpoch': nEpoch,
            'lr': 1e-3,
        }
    )

    # ==================== Log config ====================
    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("SimRegMatch Configuration (adapted for GNN):", file=f)
        print(f"  Method: SimRegMatch (FixMatch for regression + graph calibration)", file=f)
        print(f"  MC Dropout iterations: {MC_ITERS}", file=f)
        print(f"  Uncertainty filter percentile: {PERCENTILE} (top {100-PERCENTILE:.0f}% filtered)", file=f)
        print(f"  Calibration beta: {BETA} ({BETA*100:.0f}% model + {(1-BETA)*100:.0f}% graph reference)", file=f)
        print(f"  Lambda_U: {LAMBDA_U} (unlabeled loss weight)", file=f)
        print(f"  Augmentation: structured (weak=keep Tair+Tskin/mask 1 low/10%geo; "
              f"strong=keep Tair/mask 2-3 groups/1 timestep/30%geo/25%edges)", file=f)
        print(f"  Ramp-up: {RAMP_EPOCHS} epochs", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})", file=f)
        print(f"  Unlabeled with labeled neighbors: {n_with_labeled_nbs}/{nNodes_unlabeled}", file=f)
        print(f"  Training samples: {len(trainLoader.dataset)}", file=f)
        print(f"  Epochs: {nEpoch}", file=f)
        print(f"  Graph threshold: {dataParam['thres']}", file=f)
        print("", file=f)

        # Graph stats
        nL = nNodes_labeled
        n_ll = int(np.sum(adj_matrix[:nL, :nL] > 0) // 2)
        n_uu = int(np.sum(adj_matrix[nL:, nL:] > 0) // 2)
        n_lu = int(np.sum(adj_matrix[:nL, nL:] > 0))
        print(f"  Graph: L-L={n_ll}, U-U={n_uu}, L-U={n_lu}, total={n_ll+n_uu+n_lu}", file=f)

        # Labeled neighbor stats
        nbs_counts = [len(neighbor_info[u]) for u in range(nNodes_unlabeled)]
        print(f"  Labeled neighbors per unlabeled: min={min(nbs_counts)}, "
              f"max={max(nbs_counts)}, mean={np.mean(nbs_counts):.1f}", file=f)
        print("=" * 60, file=f)

    if torch.cuda.is_available():
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(model, file=f)

    # Save metadata
    with open(f'{output_dir}/{modelName}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)

    # ==================== Training loop ====================
    print(f"\nStarting SimRegMatch training for {nEpoch} epochs...")

    for epoch in range(nEpoch):
        # Ramp up lambda_u
        if epoch < RAMP_EPOCHS:
            current_lambda = LAMBDA_U * (epoch / RAMP_EPOCHS)
        else:
            current_lambda = LAMBDA_U

        # Train
        trainLoss, trainRMSE, _, _, train_stats = train_simregmatch(
            trainLoader, model, lossFn, opt, scheduler, device,
            nNodes, nNodes_labeled, adj_matrix, neighbor_info,
            MC_ITERS, PERCENTILE, BETA, LAMBDA_U, current_lambda,
            WEAK_NOISE, STRONG_NOISE
        )

        # Validate
        validLoss, validRMSE, _, _ = network_semi.test(
            validLoader, model, lossFn, device, nNodes, nNodes_labeled
        )

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {trainLoss:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE[0]:.3f}/{validRMSE[0]:.3f}; "
                  f"LR {scheduler.get_last_lr()[0]:.6f}; lambda={current_lambda:.4f}", file=f)
            print(f"  labeled_loss={train_stats['labeled_loss']:.4e} | "
                  f"unlabeled_loss={train_stats['unlabeled_loss']:.4e} | "
                  f"filter_rate={train_stats['filter_rate']:.1f}% | "
                  f"threshold={train_stats['threshold']:.6f}", file=f)
            print(f"  uncertainty: mean={train_stats['uncertainty_mean']:.6f}, "
                  f"median={train_stats['uncertainty_median']:.6f}, "
                  f"max={train_stats['uncertainty_max']:.6f}", file=f)
            print(f"  [AUG] pred_diff(strong-weak): {train_stats['pred_diff_strong_weak']:.6f}", file=f)
            print(f"  RMSE std: {trainRMSE[1]:.3f}/{validRMSE[1]:.3f}; "
                  f"min: {trainRMSE[2]:.3f}/{validRMSE[2]:.3f}; "
                  f"max: {trainRMSE[3]:.3f}/{validRMSE[3]:.3f};", file=f)

        # W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/labeled_loss': train_stats['labeled_loss'],
            'train/unlabeled_loss': train_stats['unlabeled_loss'],
            'train/filter_rate': train_stats['filter_rate'],
            'train/threshold': train_stats['threshold'],
            'train/uncertainty_mean': train_stats['uncertainty_mean'],
            'train/lambda_u': current_lambda,
            'train/pseudo_labels_used_pct': train_stats['pseudo_labels_used_pct'],
            'valid/loss': validLoss,
            'valid/rmse': validRMSE[0],
            'valid/rmse_min': validRMSE[2],
            'valid/rmse_max': validRMSE[3],
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': bestLoss,
        })

        # Save best model
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print("  Model saved.", file=f)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'bestLoss': bestLoss,
                'hist': hist,
            }, chkptPath)

        hist.append([trainLoss, validLoss, trainRMSE[0], validRMSE[0]])
        utils.plotHist(hist, modelName, output_dir=output_dir)

    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    wandb.finish()


if __name__ == "__main__":
    main()

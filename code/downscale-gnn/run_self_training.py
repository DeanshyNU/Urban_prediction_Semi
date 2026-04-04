"""
Self-training for semi-supervised GNN regression.
Step 1: Load pre-trained semi-supervised GNN
Step 2: Generate pseudo-labels for unlabeled nodes
Step 3: Fine-tune with labeled + pseudo-labeled data
Step 4: Periodically update pseudo-labels

Usage:
  CONV_TYPE=graphconv PRETRAINED_PATH=path/to/model.pt python -u code/downscale-gnn/run_self_training.py

Environment variables:
  CONV_TYPE: graphconv (default), sageconv, appnp
  PRETRAINED_PATH: path to pre-trained model checkpoint
  FILTER_MODE: none (default), mc_dropout, graph_uncertainty, conformal
  LAMBDA_PSEUDO: pseudo-label loss weight (default 1.0)
  UPDATE_INTERVAL: epochs between pseudo-label updates (default 30)
"""
import numpy as np
import torch, pickle, data_semi, network_semi, utils
import os
import wandb
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_conformal_valid_loader = None  # set by main() for conformal calibration

# Config from environment
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
pretrained_path = os.environ.get('PRETRAINED_PATH', '')
filter_mode = os.environ.get('FILTER_MODE', 'none').lower()  # none, mc_dropout, graph_uncertainty
lambda_pseudo = float(os.environ.get('LAMBDA_PSEUDO', '1.0'))
update_interval = int(os.environ.get('UPDATE_INTERVAL', '30'))
mc_samples = int(os.environ.get('MC_SAMPLES', '10'))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
filter_tag = filter_mode if filter_mode != 'none' else 'basic'
if job_id:
    output_dir = os.path.join(project_root, 'log', f'self_training_{conv_type}_{filter_tag}_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'self_training_{conv_type}_{filter_tag}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


def generate_pseudo_labels(model, loader, device, nNodes, nNodes_labeled,
                           filter_mode='none', mc_samples=10, adj_matrix=None):
    """
    Generate pseudo-labels for unlabeled nodes with optional uncertainty filtering.

    Args:
        model: trained GNN model
        loader: data loader
        device: cuda/cpu
        nNodes: total nodes per graph
        nNodes_labeled: number of labeled nodes
        filter_mode: 'none', 'mc_dropout', 'graph_uncertainty'
        mc_samples: number of MC Dropout forward passes
        adj_matrix: adjacency matrix for graph uncertainty

    Returns:
        pseudo_labels: (nSamples, nNodes_unlabeled) array
        weights: (nSamples, nNodes_unlabeled) confidence weights [0, 1]
        stats: dict with statistics for logging
    """
    nNodes_unlabeled = nNodes - nNodes_labeled

    # ==================== Collect predictions ====================
    if filter_mode == 'mc_dropout':
        # MC Dropout: multiple forward passes with dropout ON
        model.train()  # keep dropout active
        mc_preds = []
        mc_labeled_preds = []  # also collect labeled predictions for debug
        for mc_i in range(mc_samples):
            sample_preds = []
            sample_labeled = []
            with torch.no_grad():
                for _batch in loader:
                    _batch = _batch.to(device)
                    _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
                    label_mask = _batch.label_mask
                    batch_size = _batch.x.shape[0] // nNodes
                    # Unlabeled
                    unlabeled_pred = _yHat[~label_mask].squeeze(-1).reshape(batch_size, nNodes_unlabeled)
                    sample_preds.append(unlabeled_pred.cpu().numpy())
                    # Labeled (for debug comparison)
                    labeled_pred = _yHat[label_mask].squeeze(-1).reshape(batch_size, nNodes_labeled)
                    sample_labeled.append(labeled_pred.cpu().numpy())
            mc_preds.append(np.concatenate(sample_preds, axis=0))
            mc_labeled_preds.append(np.concatenate(sample_labeled, axis=0))

        mc_preds = np.stack(mc_preds, axis=0)  # (mc_samples, nSamples, nNodes_unlabeled)
        mc_labeled = np.stack(mc_labeled_preds, axis=0)  # (mc_samples, nSamples, nNodes_labeled)

        pseudo_labels = mc_preds.mean(axis=0)  # (nSamples, nNodes_unlabeled)
        mc_std = mc_preds.std(axis=0)  # (nSamples, nNodes_unlabeled)
        mc_labeled_std = mc_labeled.std(axis=0)  # for debug

        # Weight: inverse of std, normalized
        mc_weights = 1.0 / (1.0 + mc_std * 10)
        final_weights = mc_weights

        model.eval()

        # Debug: per-node MC Dropout statistics
        mean_std_per_node = mc_std.mean(axis=0)  # (nNodes_unlabeled,)
        debug_mc = {
            'unlabeled_std_per_node_min': mean_std_per_node.min(),
            'unlabeled_std_per_node_max': mean_std_per_node.max(),
            'unlabeled_std_per_node_mean': mean_std_per_node.mean(),
            'labeled_std_mean': mc_labeled_std.mean(),
            'labeled_std_max': mc_labeled_std.max(),
            'high_confidence_nodes': int((mean_std_per_node < mean_std_per_node.mean()).sum()),
            'low_confidence_nodes': int((mean_std_per_node >= mean_std_per_node.mean()).sum()),
        }

    elif filter_mode == 'graph_uncertainty':
        # Single forward pass + graph-based spatial consistency (improved version)
        # Three improvements over basic version:
        #   1. Edge-weight-weighted neighbor reference (closer neighbors matter more)
        #   2. Labeled neighbors trusted fully, unlabeled discounted by factor alpha=0.3
        #   3. Labeled neighbor ratio → base confidence (more labeled = more reliable reference)
        UNLABELED_DISCOUNT = 0.3  # discount factor for unlabeled neighbor contributions
        SPATIAL_SCALE = 5.0       # controls penalty strength for spatial inconsistency

        model.eval()
        sample_preds_u = []
        sample_preds_l = []
        sample_targets_l = []
        with torch.no_grad():
            for _batch in loader:
                _batch = _batch.to(device)
                _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
                label_mask = _batch.label_mask
                batch_size = _batch.x.shape[0] // nNodes
                unlabeled_pred = _yHat[~label_mask].squeeze(-1).reshape(batch_size, nNodes_unlabeled)
                sample_preds_u.append(unlabeled_pred.cpu().numpy())
                labeled_pred = _yHat[label_mask].squeeze(-1).reshape(batch_size, nNodes_labeled)
                sample_preds_l.append(labeled_pred.cpu().numpy())
                labeled_target = _batch.y[label_mask].squeeze(-1).reshape(batch_size, nNodes_labeled)
                sample_targets_l.append(labeled_target.cpu().numpy())

        pseudo_labels = np.concatenate(sample_preds_u, axis=0)
        labeled_preds = np.concatenate(sample_preds_l, axis=0)
        labeled_targets = np.concatenate(sample_targets_l, axis=0)
        mc_std = np.zeros_like(pseudo_labels)

        # Pre-compute neighbor info with edge weights (once)
        neighbor_info = {}
        for u in range(nNodes_unlabeled):
            u_global = nNodes_labeled + u
            labeled_nbs = []  # list of (local_idx, edge_weight)
            unlabeled_nbs = []  # list of (local_idx, edge_weight)
            for nb in range(nNodes):
                ew = adj_matrix[u_global, nb]
                if ew > 0 and nb != u_global:
                    if nb < nNodes_labeled:
                        labeled_nbs.append((nb, ew))
                    else:
                        unlabeled_nbs.append((nb - nNodes_labeled, ew))
            neighbor_info[u] = {'labeled': labeled_nbs, 'unlabeled': unlabeled_nbs}

        # Compute weights per sample per unlabeled node
        graph_weights = np.ones_like(pseudo_labels)
        spatial_diffs_all = []
        base_confidences_all = []
        labeled_ratios_all = []

        for t in range(pseudo_labels.shape[0]):
            for u in range(nNodes_unlabeled):
                nbs = neighbor_info[u]
                n_labeled_nbs = len(nbs['labeled'])
                n_unlabeled_nbs = len(nbs['unlabeled'])
                n_total_nbs = n_labeled_nbs + n_unlabeled_nbs

                if n_total_nbs == 0:
                    graph_weights[t, u] = 0.3  # isolated node, low confidence
                    continue

                # --- Improvement 1: Edge-weight-weighted reference ---
                # --- Improvement 2: Labeled trusted fully, unlabeled discounted ---
                weighted_sum = 0.0
                weight_sum = 0.0

                # Labeled neighbors: full trust, weighted by edge weight
                for nb_l, ew in nbs['labeled']:
                    weighted_sum += ew * labeled_targets[t, nb_l]
                    weight_sum += ew

                # Unlabeled neighbors: discounted trust, weighted by edge weight
                for nb_u, ew in nbs['unlabeled']:
                    weighted_sum += UNLABELED_DISCOUNT * ew * pseudo_labels[t, nb_u]
                    weight_sum += UNLABELED_DISCOUNT * ew

                ref_value = weighted_sum / weight_sum if weight_sum > 0 else pseudo_labels[t, u]

                # Spatial consistency
                spatial_diff = abs(pseudo_labels[t, u] - ref_value)
                spatial_diffs_all.append(spatial_diff)

                # --- Improvement 3: Labeled ratio → base confidence ---
                labeled_ratio = n_labeled_nbs / n_total_nbs
                base_confidence = 0.5 + 0.5 * labeled_ratio
                # all labeled neighbors → base=1.0
                # all unlabeled neighbors → base=0.5
                labeled_ratios_all.append(labeled_ratio)
                base_confidences_all.append(base_confidence)

                # Final weight = base_confidence × spatial consistency
                graph_weights[t, u] = base_confidence / (1.0 + spatial_diff * SPATIAL_SCALE)

        final_weights = graph_weights
        spatial_diffs_all = np.array(spatial_diffs_all)
        base_confidences_all = np.array(base_confidences_all)
        labeled_ratios_all = np.array(labeled_ratios_all)

        # Debug statistics
        mean_weight_per_node = graph_weights.mean(axis=0)
        n_labeled_nbs_per_node = [len(neighbor_info[u]['labeled']) for u in range(nNodes_unlabeled)]
        n_unlabeled_nbs_per_node = [len(neighbor_info[u]['unlabeled']) for u in range(nNodes_unlabeled)]
        n_total_nbs_per_node = [n_labeled_nbs_per_node[u] + n_unlabeled_nbs_per_node[u] for u in range(nNodes_unlabeled)]

        debug_mc = {
            'spatial_diff_mean': spatial_diffs_all.mean() if len(spatial_diffs_all) > 0 else 0,
            'spatial_diff_median': np.median(spatial_diffs_all) if len(spatial_diffs_all) > 0 else 0,
            'spatial_diff_p90': np.percentile(spatial_diffs_all, 90) if len(spatial_diffs_all) > 0 else 0,
            'spatial_diff_max': spatial_diffs_all.max() if len(spatial_diffs_all) > 0 else 0,
            'base_confidence_mean': base_confidences_all.mean() if len(base_confidences_all) > 0 else 0,
            'base_confidence_min': base_confidences_all.min() if len(base_confidences_all) > 0 else 0,
            'labeled_ratio_mean': labeled_ratios_all.mean() if len(labeled_ratios_all) > 0 else 0,
            'weight_per_node_min': mean_weight_per_node.min(),
            'weight_per_node_max': mean_weight_per_node.max(),
            'weight_per_node_mean': mean_weight_per_node.mean(),
            'labeled_nbs_per_node_min': min(n_labeled_nbs_per_node),
            'labeled_nbs_per_node_max': max(n_labeled_nbs_per_node),
            'labeled_nbs_per_node_mean': np.mean(n_labeled_nbs_per_node),
            'unlabeled_nbs_per_node_mean': np.mean(n_unlabeled_nbs_per_node),
            'total_nbs_per_node_mean': np.mean(n_total_nbs_per_node),
            'nodes_with_0_labeled_nbs': sum(1 for n in n_labeled_nbs_per_node if n == 0),
            'nodes_with_0_total_nbs': sum(1 for n in n_total_nbs_per_node if n == 0),
            'labeled_pred_vs_target_rmse': np.sqrt(np.mean((labeled_preds - labeled_targets) ** 2)),
            'unlabeled_discount': UNLABELED_DISCOUNT,
            'spatial_scale': SPATIAL_SCALE,
        }

    elif filter_mode == 'conformal':
        # Conformal Prediction: use calibration residuals from valid set to weight pseudo-labels
        # Weight based on model's actual prediction accuracy near each unlabeled node
        K_NEIGHBORS = 5  # number of nearest labeled neighbors for local calibration
        WEIGHT_SCALE = 10.0  # controls how sharply interval width affects weight

        model.eval()

        # Step A: Collect predictions on TRAIN set (for pseudo-labels)
        sample_preds_u = []
        with torch.no_grad():
            for _batch in loader:
                _batch = _batch.to(device)
                _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
                label_mask = _batch.label_mask
                batch_size = _batch.x.shape[0] // nNodes
                unlabeled_pred = _yHat[~label_mask].squeeze(-1).reshape(batch_size, nNodes_unlabeled)
                sample_preds_u.append(unlabeled_pred.cpu().numpy())
        pseudo_labels = np.concatenate(sample_preds_u, axis=0)

        # Step B: Calibrate on VALID set (model's actual residuals on unseen data)
        # valid_loader is passed via adj_matrix argument (repurposed for conformal)
        # We need the valid loader - it's stored in the global scope by main()
        valid_loader_ref = _conformal_valid_loader  # set by main() before calling
        val_preds_l = []
        val_targets_l = []
        with torch.no_grad():
            for _batch in valid_loader_ref:
                _batch = _batch.to(device)
                _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
                label_mask = _batch.label_mask
                batch_size = _batch.x.shape[0] // nNodes
                labeled_pred = _yHat[label_mask].squeeze(-1).reshape(batch_size, nNodes_labeled)
                labeled_target = _batch.y[label_mask].squeeze(-1).reshape(batch_size, nNodes_labeled)
                val_preds_l.append(labeled_pred.cpu().numpy())
                val_targets_l.append(labeled_target.cpu().numpy())

        val_preds_l = np.concatenate(val_preds_l, axis=0)  # (nValSamples, nNodes_labeled)
        val_targets_l = np.concatenate(val_targets_l, axis=0)

        # Per-labeled-node calibration residual (mean absolute error on valid set)
        per_node_residual = np.mean(np.abs(val_preds_l - val_targets_l), axis=0)  # (nNodes_labeled,)
        print(f"  [Conformal] Per-node valid residual: min={per_node_residual.min():.4f}, "
              f"max={per_node_residual.max():.4f}, mean={per_node_residual.mean():.4f}")

        # Step C: For each unlabeled node, compute interval width from K nearest labeled neighbors
        # Use adjacency matrix to find nearest labeled neighbors (by edge weight)
        interval_widths = np.zeros(nNodes_unlabeled)
        n_labeled_nbs_list = []

        for u in range(nNodes_unlabeled):
            u_global = nNodes_labeled + u
            # Find labeled neighbors sorted by edge weight (descending)
            labeled_nbs = []
            for j in range(nNodes_labeled):
                ew = adj_matrix[u_global, j]
                if ew > 0:
                    labeled_nbs.append((j, ew))
            labeled_nbs.sort(key=lambda x: -x[1])  # sort by weight descending

            n_labeled_nbs_list.append(len(labeled_nbs))

            if len(labeled_nbs) == 0:
                # No labeled neighbors: maximum uncertainty
                interval_widths[u] = per_node_residual.max() * 2
                continue

            # Take top K neighbors
            top_k = labeled_nbs[:K_NEIGHBORS]
            # Weighted average of their calibration residuals
            weighted_residual = 0.0
            total_weight = 0.0
            for nb_idx, ew in top_k:
                weighted_residual += ew * per_node_residual[nb_idx]
                total_weight += ew
            interval_widths[u] = weighted_residual / total_weight if total_weight > 0 else per_node_residual.mean()

        # Step D: Convert interval widths to weights
        # Narrow interval → high weight, wide interval → low weight
        conformal_weights = 1.0 / (1.0 + interval_widths * WEIGHT_SCALE)

        # Broadcast weights to all time samples (weights are per-node, static)
        final_weights = np.tile(conformal_weights, (pseudo_labels.shape[0], 1))
        mc_std = np.zeros_like(pseudo_labels)

        # Debug statistics
        debug_mc = {
            'interval_width_min': interval_widths.min(),
            'interval_width_max': interval_widths.max(),
            'interval_width_mean': interval_widths.mean(),
            'interval_width_median': np.median(interval_widths),
            'conformal_weight_min': conformal_weights.min(),
            'conformal_weight_max': conformal_weights.max(),
            'conformal_weight_mean': conformal_weights.mean(),
            'conformal_weight_std': conformal_weights.std(),
            'conformal_weight_above_05': (conformal_weights > 0.5).mean() * 100,
            'conformal_weight_below_02': (conformal_weights < 0.2).mean() * 100,
            'per_node_residual_min': per_node_residual.min(),
            'per_node_residual_max': per_node_residual.max(),
            'per_node_residual_mean': per_node_residual.mean(),
            'labeled_nbs_min': min(n_labeled_nbs_list),
            'labeled_nbs_max': max(n_labeled_nbs_list),
            'labeled_nbs_mean': np.mean(n_labeled_nbs_list),
            'nodes_with_0_labeled_nbs': sum(1 for n in n_labeled_nbs_list if n == 0),
            'K_neighbors': K_NEIGHBORS,
            'weight_scale': WEIGHT_SCALE,
        }
        print(f"  [Conformal] Interval widths: [{interval_widths.min():.4f}, {interval_widths.max():.4f}], "
              f"mean={interval_widths.mean():.4f}")
        print(f"  [Conformal] Weights: [{conformal_weights.min():.4f}, {conformal_weights.max():.4f}], "
              f"mean={conformal_weights.mean():.4f}, std={conformal_weights.std():.4f}")
        print(f"  [Conformal] Nodes with 0 labeled nbs: {sum(1 for n in n_labeled_nbs_list if n == 0)}")

    else:
        # No filtering: single forward pass, equal weights
        model.eval()
        sample_preds = []
        with torch.no_grad():
            for _batch in loader:
                _batch = _batch.to(device)
                _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
                label_mask = _batch.label_mask
                batch_size = _batch.x.shape[0] // nNodes
                unlabeled_pred = _yHat[~label_mask].squeeze(-1).reshape(batch_size, nNodes_unlabeled)
                sample_preds.append(unlabeled_pred.cpu().numpy())

        pseudo_labels = np.concatenate(sample_preds, axis=0)
        mc_std = np.zeros_like(pseudo_labels)
        final_weights = np.ones_like(pseudo_labels)
        debug_mc = {}

    # ==================== Statistics ====================
    stats = {
        'pseudo_mean': pseudo_labels.mean(),
        'pseudo_std': pseudo_labels.std(),
        'pseudo_min': pseudo_labels.min(),
        'pseudo_max': pseudo_labels.max(),
        'mc_std_mean': mc_std.mean(),
        'mc_std_max': mc_std.max(),
        'weight_mean': final_weights.mean(),
        'weight_min': final_weights.min(),
        'weight_max': final_weights.max(),
        'weight_above_05': (final_weights > 0.5).mean() * 100,
        'weight_above_08': (final_weights > 0.8).mean() * 100,
        'weight_below_02': (final_weights < 0.2).mean() * 100,
        **debug_mc,
    }

    return pseudo_labels, final_weights, stats


def train_self_training(loader, model, lossFn, opt, scheduler, device,
                        nNodes, nNodes_labeled, pseudo_labels, pseudo_weights,
                        lambda_pseudo, sample_offset=0):
    """
    Self-training: supervised loss on labeled + weighted pseudo-label loss on unlabeled.
    """
    model.train()
    _LOSS = 0
    _LABELED_LOSS = 0
    _PSEUDO_LOSS = 0
    pred, truth = [], []
    nNodes_unlabeled = nNodes - nNodes_labeled

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)

        label_mask = _batch.label_mask
        batch_size = _batch.x.shape[0] // nNodes

        # 1. Supervised loss on labeled nodes
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]
        labeled_loss = lossFn(_yHat_labeled, _y_labeled)

        # 2. Pseudo-label loss on unlabeled nodes
        _yHat_unlabeled = _yHat[~label_mask].squeeze(-1)  # (batch_size * nNodes_unlabeled,)
        _yHat_unlabeled = _yHat_unlabeled.reshape(batch_size, nNodes_unlabeled)

        # Get corresponding pseudo-labels and weights for this batch
        batch_start = sample_offset + _n * batch_size
        batch_end = batch_start + batch_size

        if batch_end <= pseudo_labels.shape[0]:
            pl = torch.FloatTensor(pseudo_labels[batch_start:batch_end]).to(device)
            pw = torch.FloatTensor(pseudo_weights[batch_start:batch_end]).to(device)
        else:
            # Handle edge case: wrap around or skip
            pl = torch.FloatTensor(pseudo_labels[:batch_size]).to(device)
            pw = torch.FloatTensor(pseudo_weights[:batch_size]).to(device)

        # Weighted MSE loss on pseudo-labels
        pseudo_diff = (_yHat_unlabeled - pl) ** 2
        pseudo_loss = (pw * pseudo_diff).mean()

        # Total loss
        total_loss = labeled_loss + lambda_pseudo * pseudo_loss

        total_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss
        _LABELED_LOSS += labeled_loss.item()
        _PSEUDO_LOSS += pseudo_loss.item()

        # Record labeled predictions for RMSE
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        _truth = _y_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1
    return ((_LOSS / n_batches).item(), _RMSE, truth, pred,
            _LABELED_LOSS / n_batches, _PSEUDO_LOSS / n_batches)


def main():
    # ==================== Data ====================
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': int(os.environ.get('POOL_SIZE', '12')),
        'batchSize': 128,
        'thres': 0.1,
        'geoFeatures': 'full',
    }
    n_unlabeled = int(os.environ.get('N_UNLABELED', 200))
    dataParam['batchSize'] = int(os.environ.get('BATCH_SIZE', 128))
    os.environ['OUTPUT_DIR'] = output_dir
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)

    # Make valid_loader accessible for conformal calibration
    global _conformal_valid_loader
    _conformal_valid_loader = validLoader

    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    nNodes_unlabeled = metadata['nNodes_unlabeled']
    adj_matrix = metadata['AdjMatrix']

    print(f"Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})")
    print(f"Training samples: {len(trainLoader.dataset)}")
    print(f"Filter mode: {filter_mode}")
    print(f"Lambda pseudo: {lambda_pseudo}")
    print(f"Update interval: {update_interval}")

    # ==================== Model ====================
    nEpoch = 3000
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

    # ==================== Load pretrained model ====================
    if pretrained_path and os.path.exists(pretrained_path):
        chkpt = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        print(f"Loaded pretrained model from: {pretrained_path}")
        print(f"  Pretrained best valid RMSE: {chkpt.get('bestLoss', 'N/A')}")
    else:
        print(f"WARNING: No pretrained model found at '{pretrained_path}', starting from scratch")

    bestLoss = np.inf
    hist = []

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_{conv_type}_selftraining_{filter_tag}_{n_unlabeled}unlabeled'
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
            'method': 'self-training',
            'filter_mode': filter_mode,
            'lambda_pseudo': lambda_pseudo,
            'update_interval': update_interval,
            'mc_samples': mc_samples,
            'pretrained_path': pretrained_path,
            'n_unlabeled': n_unlabeled,
            'nEpoch': nEpoch,
            'lr': 1e-3,
            'weight_decay': 0,
        }
    )

    # ==================== Log config ====================
    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("Self-training configuration:", file=f)
        print(f"  Pretrained model: {pretrained_path}", file=f)
        print(f"  Filter mode: {filter_mode}", file=f)
        print(f"  Lambda pseudo: {lambda_pseudo}", file=f)
        print(f"  Update interval: {update_interval} epochs", file=f)
        print(f"  MC samples: {mc_samples}", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})", file=f)
        print(f"  Training samples: {len(trainLoader.dataset)}", file=f)
        print(f"  Weight decay: none", file=f)
        print(f"  Early stopping: none", file=f)
        print(f"  Epochs: {nEpoch}", file=f)
        print(f"  Graph threshold: {dataParam['thres']}", file=f)
        print("", file=f)

        # Graph stats
        nL = nNodes_labeled
        n_ll = int(np.sum(adj_matrix[:nL, :nL] > 0) // 2)
        n_uu = int(np.sum(adj_matrix[nL:, nL:] > 0) // 2)
        n_lu = int(np.sum(adj_matrix[:nL, nL:] > 0))
        print(f"  Graph: L-L={n_ll}, U-U={n_uu}, L-U={n_lu}, total={n_ll+n_uu+n_lu}", file=f)
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

    # ==================== Initial pseudo-label generation ====================
    print("\nGenerating initial pseudo-labels...")
    pseudo_labels, pseudo_weights, pl_stats = generate_pseudo_labels(
        model, trainLoader, device, nNodes, nNodes_labeled,
        filter_mode=filter_mode, mc_samples=mc_samples, adj_matrix=adj_matrix
    )

    def log_pseudo_stats(pl_stats, epoch_label, f=None):
        """Log pseudo-label and weight statistics."""
        lines = [
            f"\n[Pseudo-labels {epoch_label}]",
            f"  Pseudo-labels: mean={pl_stats['pseudo_mean']:.4f}, std={pl_stats['pseudo_std']:.4f}, "
            f"range=[{pl_stats['pseudo_min']:.4f}, {pl_stats['pseudo_max']:.4f}]",
            f"  MC std: mean={pl_stats['mc_std_mean']:.4f}, max={pl_stats['mc_std_max']:.4f}",
            f"  Weights: mean={pl_stats['weight_mean']:.4f}, min={pl_stats['weight_min']:.4f}, max={pl_stats['weight_max']:.4f}",
            f"  Weight distribution: >0.8: {pl_stats['weight_above_08']:.1f}%, "
            f">0.5: {pl_stats['weight_above_05']:.1f}%, <0.2: {pl_stats['weight_below_02']:.1f}%",
        ]
        # Filter-specific debug info
        if 'unlabeled_std_per_node_mean' in pl_stats:
            lines.append(f"  [MC Dropout] Per-node std: mean={pl_stats['unlabeled_std_per_node_mean']:.4f}, "
                        f"min={pl_stats['unlabeled_std_per_node_min']:.4f}, max={pl_stats['unlabeled_std_per_node_max']:.4f}")
            lines.append(f"  [MC Dropout] Labeled std: mean={pl_stats['labeled_std_mean']:.4f}, max={pl_stats['labeled_std_max']:.4f}")
            lines.append(f"  [MC Dropout] High/low confidence nodes: {pl_stats['high_confidence_nodes']}/{pl_stats['low_confidence_nodes']}")
        if 'spatial_diff_mean' in pl_stats:
            lines.append(f"  [Graph Uncertainty] Config: unlabeled_discount={pl_stats.get('unlabeled_discount', 'N/A')}, "
                        f"spatial_scale={pl_stats.get('spatial_scale', 'N/A')}")
            lines.append(f"  [Graph Uncertainty] Spatial diff: mean={pl_stats['spatial_diff_mean']:.4f}, "
                        f"median={pl_stats['spatial_diff_median']:.4f}, "
                        f"p90={pl_stats.get('spatial_diff_p90', 0):.4f}, max={pl_stats['spatial_diff_max']:.4f}")
            lines.append(f"  [Graph Uncertainty] Base confidence: mean={pl_stats.get('base_confidence_mean', 0):.4f}, "
                        f"min={pl_stats.get('base_confidence_min', 0):.4f}")
            lines.append(f"  [Graph Uncertainty] Labeled ratio in neighbors: mean={pl_stats.get('labeled_ratio_mean', 0):.4f}")
            lines.append(f"  [Graph Uncertainty] Per-node weight: min={pl_stats['weight_per_node_min']:.4f}, "
                        f"mean={pl_stats.get('weight_per_node_mean', 0):.4f}, "
                        f"max={pl_stats['weight_per_node_max']:.4f}")
            lines.append(f"  [Graph Uncertainty] Neighbors per unlabeled node: "
                        f"labeled: min={pl_stats.get('labeled_nbs_per_node_min', 0)}, "
                        f"max={pl_stats.get('labeled_nbs_per_node_max', 0)}, "
                        f"mean={pl_stats.get('labeled_nbs_per_node_mean', 0):.1f} | "
                        f"unlabeled: mean={pl_stats.get('unlabeled_nbs_per_node_mean', 0):.1f} | "
                        f"total: mean={pl_stats.get('total_nbs_per_node_mean', 0):.1f}")
            lines.append(f"  [Graph Uncertainty] Isolated nodes: "
                        f"no_labeled_nbs={pl_stats.get('nodes_with_0_labeled_nbs', 0)}, "
                        f"no_nbs_at_all={pl_stats.get('nodes_with_0_total_nbs', 0)}")
            lines.append(f"  [Graph Uncertainty] Model quality: labeled pred vs target RMSE={pl_stats['labeled_pred_vs_target_rmse']:.4f}")
        for line in lines:
            print(line)
            if f:
                print(line, file=f)

    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        log_pseudo_stats(pl_stats, "epoch 0", f)
    print(f"\nPseudo-labels generated. Filter mode: {filter_mode}")

    # ==================== Training loop ====================
    print(f"\nStarting self-training for {nEpoch} epochs...")

    for epoch in range(nEpoch):
        # Update pseudo-labels periodically
        if epoch > 0 and epoch % update_interval == 0:
            pseudo_labels, pseudo_weights, pl_stats = generate_pseudo_labels(
                model, trainLoader, device, nNodes, nNodes_labeled,
                filter_mode=filter_mode, mc_samples=mc_samples, adj_matrix=adj_matrix
            )
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                log_pseudo_stats(pl_stats, f"updated epoch {epoch}", f)
            # W&B: log pseudo-label quality over time
            wandb.log({
                'pseudo/weight_mean': pl_stats['weight_mean'],
                'pseudo/weight_above_05_pct': pl_stats['weight_above_05'],
                'pseudo/pseudo_std': pl_stats['pseudo_std'],
                'pseudo/mc_std_mean': pl_stats['mc_std_mean'],
            })

        # Ramp up lambda_pseudo
        if epoch < 100:
            current_lambda = lambda_pseudo * (epoch / 100.0)
        else:
            current_lambda = lambda_pseudo

        # Train
        trainLoss, trainRMSE, _, _, labeled_loss, pseudo_loss = train_self_training(
            trainLoader, model, lossFn, opt, scheduler, device,
            nNodes, nNodes_labeled, pseudo_labels, pseudo_weights,
            current_lambda
        )

        # Validate (only on labeled nodes, same as semi-supervised)
        validLoss, validRMSE, _, _ = network_semi.test(
            validLoader, model, lossFn, device, nNodes, nNodes_labeled
        )

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {trainLoss:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE[0]:.3f}/{validRMSE[0]:.3f}; "
                  f"LR {scheduler.get_last_lr()[0]:.6f}; lambda={current_lambda:.3f}", file=f)
            print(f"  labeled_loss={labeled_loss:.4e} | pseudo_loss={pseudo_loss:.4e}", file=f)
            print(f"  RMSE std: {trainRMSE[1]:.3f}/{validRMSE[1]:.3f}; "
                  f"min: {trainRMSE[2]:.3f}/{validRMSE[2]:.3f}; "
                  f"max: {trainRMSE[3]:.3f}/{validRMSE[3]:.3f};", file=f)

        # W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/rmse_std': trainRMSE[1],
            'train/rmse_min': trainRMSE[2],
            'train/rmse_max': trainRMSE[3],
            'train/labeled_loss': labeled_loss,
            'train/pseudo_loss': pseudo_loss,
            'train/lambda_pseudo': current_lambda,
            'valid/loss': validLoss,
            'valid/rmse': validRMSE[0],
            'valid/rmse_std': validRMSE[1],
            'valid/rmse_min': validRMSE[2],
            'valid/rmse_max': validRMSE[3],
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': bestLoss,
        })

        # Save best model
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            early_stop_counter = 0
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

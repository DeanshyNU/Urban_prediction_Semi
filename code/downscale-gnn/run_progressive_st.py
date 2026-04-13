"""
Progressive Self-Training with Uncertainty-guided Expansion.

Key innovations over standard self-training:
  1. Pseudo-labels from GSR (graph interpolation), not model predictions → no collapse
  2. Progressive expansion: add pseudo-labeled nodes gradually (near → far)
  3. Uncertainty-based selection: prioritize medium-uncertainty nodes
  4. Neighbor error uncertainty: use labeled neighbors' actual errors as proxy
  5. Combined with Laplacian for persistent spatial regularization

Two modes:
  - From scratch: warm-up with Laplacian, then progressive expansion
  - From checkpoint: load pretrained model, start expansion immediately

Usage:
  NORM_MODE=global N_UNLABELED=400 USE_FPS=2 CONV_TYPE=graphconv \
  EDGE_MODE=no_uu LAMBDA_LAP=0.1 LAMBDA_PSEUDO=0.05 \
  python -u code/downscale-gnn/run_progressive_st.py
"""
import numpy as np
import torch, pickle, data_semi, network_semi, utils
import os, wandb
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
n_unlabeled_env = os.environ.get('N_UNLABELED', '200')
use_fps = int(os.environ.get('USE_FPS', 0))
fps_tag = '_fps_score' if use_fps == 2 else ('_fps' if use_fps == 1 else '')
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
pretrained_path = os.environ.get('PRETRAINED_PATH', '')

if job_id:
    output_dir = os.path.join(project_root, 'log', f'progressive_st_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'progressive_st_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


def compute_neighbor_uncertainty(model, loader, device, nNodes, adj_matrix, label_mask_nodes, K=5,
                                  target_unlabeled_idx=None):
    """
    Compute uncertainty for each unlabeled node based on labeled neighbors' prediction errors.

    Args:
        target_unlabeled_idx: if provided, only compute uncertainty for these node indices
                              (to exclude validation stations from selection)

    Returns:
        uncertainties: (n_target,) uncertainty values
        per_node_errors: (n_labeled,) prediction errors at labeled nodes
    """
    model.eval()
    labeled_idx = np.where(label_mask_nodes)[0]
    if target_unlabeled_idx is not None:
        unlabeled_idx = target_unlabeled_idx  # only truly unlabeled nodes
    else:
        unlabeled_idx = np.where(~label_mask_nodes)[0]
    n_labeled = len(labeled_idx)
    n_unlabeled = len(unlabeled_idx)

    # Collect labeled node predictions and targets
    all_pred_labeled = []
    all_truth_labeled = []
    with torch.no_grad():
        for _batch in loader:
            _batch = _batch.to(device)
            _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
            label_mask = _batch.label_mask
            batch_size = _batch.x.shape[0] // nNodes
            _yHat_all = _yHat.squeeze(-1).reshape(batch_size, nNodes)
            _y_all = _batch.y.squeeze(-1).reshape(batch_size, nNodes)
            for g in range(batch_size):
                all_pred_labeled.append(_yHat_all[g][labeled_idx].cpu().numpy())
                all_truth_labeled.append(_y_all[g][labeled_idx].cpu().numpy())

    all_pred = np.array(all_pred_labeled)   # (nSamples, n_labeled)
    all_truth = np.array(all_truth_labeled)  # (nSamples, n_labeled)

    # Per-labeled-node average absolute error
    per_node_errors = np.mean(np.abs(all_pred - all_truth), axis=0)  # (n_labeled,)

    # For each unlabeled node, estimate uncertainty from K nearest labeled neighbors
    uncertainties = np.zeros(n_unlabeled)
    for ui, node_u in enumerate(unlabeled_idx):
        # Find labeled neighbors sorted by edge weight
        labeled_nbs = []
        for li, node_l in enumerate(labeled_idx):
            ew = adj_matrix[node_u, node_l]
            if ew > 0:
                labeled_nbs.append((li, ew))
        labeled_nbs.sort(key=lambda x: -x[1])

        if len(labeled_nbs) == 0:
            uncertainties[ui] = per_node_errors.max() * 2
            continue

        top_k = labeled_nbs[:K]
        weighted_err = sum(ew * per_node_errors[li] for li, ew in top_k)
        total_w = sum(ew for _, ew in top_k)
        uncertainties[ui] = weighted_err / total_w if total_w > 0 else per_node_errors.mean()

    return uncertainties, per_node_errors


def select_nodes_medium_uncertainty(uncertainties, already_selected, n_to_add):
    """
    Select n_to_add nodes with medium uncertainty from remaining candidates.

    Strategy:
      - Remove already selected nodes
      - Sort remaining by uncertainty
      - Skip top 20% (too certain, little info) and bottom 20% (too uncertain, noise)
      - Select from middle 60%
      - If not enough in middle, expand to include certain/uncertain ones

    Returns:
        new_indices: indices into unlabeled array of newly selected nodes
    """
    remaining = [i for i in range(len(uncertainties)) if i not in already_selected]
    if len(remaining) == 0:
        return []

    # Sort remaining by uncertainty
    remaining_sorted = sorted(remaining, key=lambda i: uncertainties[i])
    n_remaining = len(remaining_sorted)

    if n_remaining <= n_to_add:
        return remaining_sorted  # take all remaining

    # Middle 60%: skip top 20% and bottom 20%
    skip_top = max(1, int(n_remaining * 0.2))
    skip_bottom = max(1, int(n_remaining * 0.2))
    middle_start = skip_top
    middle_end = n_remaining - skip_bottom

    middle_candidates = remaining_sorted[middle_start:middle_end]

    if len(middle_candidates) >= n_to_add:
        # Sample from middle (pick evenly spaced)
        step = max(1, len(middle_candidates) // n_to_add)
        selected = middle_candidates[::step][:n_to_add]
    else:
        # Not enough in middle, expand outward
        selected = list(middle_candidates)
        # Add from certain side first
        for idx in remaining_sorted[:middle_start]:
            if len(selected) >= n_to_add:
                break
            selected.append(idx)
        # Then from uncertain side
        for idx in remaining_sorted[middle_end:]:
            if len(selected) >= n_to_add:
                break
            selected.append(idx)

    return selected[:n_to_add]


def main():
    # ==================== Data ====================
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': int(os.environ.get('POOL_SIZE', '12')),
        'batchSize': int(os.environ.get('BATCH_SIZE', 128)),
        'thres': float(os.environ.get('THRES', 0.1)),
        'geoFeatures': 'full',
    }
    n_unlabeled = int(os.environ.get('N_UNLABELED', 200))
    os.environ['OUTPUT_DIR'] = output_dir
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)

    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    Adj = metadata['AdjMatrix']

    # ==================== Hyperparameters ====================
    nEpoch = 5000
    lambda_lap = float(os.environ.get('LAMBDA_LAP', '0.1'))
    lambda_pseudo = float(os.environ.get('LAMBDA_PSEUDO', '0.05'))
    warmup_epochs = int(os.environ.get('WARMUP_EPOCHS', '200'))
    expand_interval = int(os.environ.get('EXPAND_INTERVAL', '200'))
    expand_schedule = [30, 50, 80, 100, 140]  # nodes to add each round
    K_neighbors = int(os.environ.get('K_NEIGHBORS', '5'))

    # ==================== Pseudo-label initialization ====================
    # Pseudo-labels come from MODEL predictions (not GSR)
    # Initialized to zeros; generated after warm-up (from scratch) or immediately (from checkpoint)
    gsr_label_mask = trainLoader.dataset[0].label_mask.numpy()
    for data in trainLoader.dataset:
        data.pseudo_label = torch.zeros_like(data.y)  # placeholder, filled at expansion time
    print(f"  Pseudo-label source: MODEL predictions (updated at each expansion round)")

    # ==================== Model ====================
    modelParam = {
        'HLD': 128, 'nMLP': 2, 'nGNN': 3, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'iDim': metadata['iDim'], 'oDim': metadata['oDim'],
        'BN': False, 'Dropout': True, 'conv_type': conv_type,
    }
    model = network_semi.GNN(modelParam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)

    # Load pretrained model (optional)
    if pretrained_path and os.path.exists(pretrained_path):
        chkpt = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        warmup_epochs = 0  # skip warm-up if pretrained
        print(f"  Loaded pretrained model: {pretrained_path}")
        print(f"  Skipping warm-up (pretrained), starting expansion immediately")
    else:
        print(f"  No pretrained model, warm-up for {warmup_epochs} epochs with Laplacian only")

    bestLoss = np.inf
    hist = []
    modelName = f'geoEmbed_average_{conv_type}_progressive_st_{n_unlabeled}unlabeled'
    chkptPath = f'{output_dir}/{modelName}.pt'

    # Laplacian edges
    _adj = Adj.copy()
    np.fill_diagonal(_adj, 0)
    _edge_src, _edge_dst = np.nonzero(_adj)
    _edge_w = torch.FloatTensor(_adj[_edge_src, _edge_dst]).to(device)
    _edge_src_t = torch.LongTensor(_edge_src).to(device)
    _edge_dst_t = torch.LongTensor(_edge_dst).to(device)

    # Unlabeled node tracking
    # [BugFix] Only select truly unlabeled nodes (positions >= nNodes_labeled), exclude validation stations
    unlabeled_mask = np.zeros(nNodes, dtype=bool)
    unlabeled_mask[nNodes_labeled:] = True  # only positions 58-457 are truly unlabeled
    unlabeled_idx = np.where(unlabeled_mask)[0]
    n_unlabeled_total = len(unlabeled_idx)
    selected_pseudo_nodes = set()  # indices into unlabeled_idx array
    current_pseudo_mask = np.zeros(nNodes, dtype=bool)  # which nodes have active pseudo-labels

    # ==================== W&B ====================
    wandb.init(
        entity="urban_prediction", project="Semi-supervised GNN",
        name=f'{modelName}_job{job_id}' if job_id else modelName,
        config={
            'method': 'Progressive Self-Training',
            'lambda_lap': lambda_lap, 'lambda_pseudo': lambda_pseudo,
            'warmup_epochs': warmup_epochs, 'expand_interval': expand_interval,
            'expand_schedule': expand_schedule,
            'K_neighbors': K_neighbors, 'pretrained': bool(pretrained_path),
            **dataParam, **modelParam,
        }
    )

    # ==================== Logging ====================
    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
        print(model, file=f)
        print("=" * 60, file=f)
        print("Progressive Self-Training Configuration:", file=f)
        print(f"  Pretrained: {pretrained_path if pretrained_path else 'None (from scratch)'}", file=f)
        print(f"  Warm-up epochs: {warmup_epochs}", file=f)
        print(f"  Expand interval: {expand_interval} epochs", file=f)
        print(f"  Expand schedule: {expand_schedule}", file=f)
        print(f"  Lambda_lap: {lambda_lap}, Lambda_pseudo: {lambda_pseudo}", file=f)
        print(f"  K_neighbors: {K_neighbors}", file=f)
        print(f"  Uncertainty: neighbor error + medium uncertainty selection", file=f)
        print(f"  Pseudo-label source: model predictions (regenerated at each expansion)", file=f)
        print(f"  NORM_MODE={os.environ.get('NORM_MODE', 'per_station')}", file=f)
        print(f"  EDGE_MODE={os.environ.get('EDGE_MODE', 'all')}", file=f)
        print("=" * 60, file=f)

    # ==================== Training Loop ====================
    print(f"\nStarting Progressive Self-Training for {nEpoch} epochs...")
    expansion_round = 0

    for epoch in range(nEpoch):
        # === Expansion check ===
        should_expand = False
        if pretrained_path:
            # From checkpoint: expand at epoch 0, then every expand_interval
            if epoch == 0 or (epoch > 0 and epoch % expand_interval == 0):
                should_expand = True
        else:
            # From scratch: expand after warm-up, then every expand_interval
            if epoch == warmup_epochs or (epoch > warmup_epochs and (epoch - warmup_epochs) % expand_interval == 0):
                should_expand = True

        if should_expand and expansion_round < len(expand_schedule):
            n_to_add = expand_schedule[expansion_round]

            # Generate pseudo-labels from current model predictions
            model.eval()
            with torch.no_grad():
                for data in trainLoader.dataset:
                    # Single-sample forward pass to get pseudo-labels
                    _x = data.x.unsqueeze(0).to(device) if data.x.dim() == 2 else data.x.to(device)
                    _ei = data.edge_index.to(device)
                    _ea = data.edge_attr.to(device)
                    _pred = model(_x.squeeze(0), _ei, _ea)
                    data.pseudo_label = _pred.cpu().detach()  # (nNodes, 1)
            model.train()

            # [Debug] Pseudo-label stats
            _sample_pl = trainLoader.dataset[0].pseudo_label.squeeze(-1)
            _ul_pl = _sample_pl[unlabeled_mask]
            print(f"  [Pseudo-labels] Generated from model: unlabeled mean={_ul_pl.mean():.4f}, "
                  f"std={_ul_pl.std():.4f}, range=[{_ul_pl.min():.4f}, {_ul_pl.max():.4f}]")

            # Compute uncertainty (only for truly unlabeled nodes, not validation stations)
            uncertainties, per_node_errors = compute_neighbor_uncertainty(
                model, trainLoader, device, nNodes, Adj, gsr_label_mask, K=K_neighbors,
                target_unlabeled_idx=unlabeled_idx
            )

            # Select nodes with medium uncertainty
            new_nodes = select_nodes_medium_uncertainty(
                uncertainties, selected_pseudo_nodes, n_to_add
            )

            # Update selected set and pseudo mask
            for idx in new_nodes:
                selected_pseudo_nodes.add(idx)
                current_pseudo_mask[unlabeled_idx[idx]] = True

            n_total_selected = len(selected_pseudo_nodes)
            n_remaining = n_unlabeled_total - n_total_selected

            # Debug logging
            if len(new_nodes) > 0:
                new_uncertainties = [uncertainties[i] for i in new_nodes]
                log_msg = (f"\n[Expansion Round {expansion_round}] Epoch {epoch}: "
                           f"added {len(new_nodes)} nodes (total: {n_total_selected}/{n_unlabeled_total})")
                log_detail = (f"  New nodes uncertainty: min={min(new_uncertainties):.4f}, "
                              f"max={max(new_uncertainties):.4f}, mean={np.mean(new_uncertainties):.4f}")
                log_error = (f"  Labeled node errors: min={per_node_errors.min():.4f}, "
                             f"max={per_node_errors.max():.4f}, mean={per_node_errors.mean():.4f}")
                log_pseudo = (f"  Pseudo-label (unlabeled): std={_ul_pl.std():.4f}")
                print(log_msg)
                print(log_detail)
                with open(f'{output_dir}/{modelName}_log', 'a') as f:
                    print(log_msg, file=f)
                    print(log_detail, file=f)
                    print(log_error, file=f)
                    print(log_pseudo, file=f)
                    print(f"  Remaining: {n_remaining} nodes", file=f)

            expansion_round += 1

        # === Training step ===
        model.train()
        _LOSS = 0
        _SUP_LOSS = 0
        _LAP_LOSS = 0
        _PSEUDO_LOSS = 0
        pred, truth = [], []

        for _n, _batch in enumerate(trainLoader):
            _batch = _batch.to(device)
            _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
            label_mask = _batch.label_mask
            batch_size = _batch.x.shape[0] // nNodes

            # 1. Supervised loss
            _yHat_labeled = _yHat[label_mask]
            _y_labeled = _batch.y[label_mask]
            sup_loss = lossFn(_yHat_labeled, _y_labeled)

            # 2. Laplacian loss (all graphs)
            _yHat_all = _yHat.squeeze(-1).reshape(batch_size, nNodes)
            lap_loss = 0
            for g in range(batch_size):
                diff = _yHat_all[g][_edge_src_t] - _yHat_all[g][_edge_dst_t]
                lap_loss = lap_loss + torch.mean(_edge_w * diff ** 2)
            lap_loss = lap_loss / batch_size

            # 3. Pseudo-label loss (only on currently selected nodes, all graphs)
            pseudo_loss = torch.tensor(0.0, device=device)
            if current_pseudo_mask.sum() > 0:
                _pl_all = _batch.pseudo_label.squeeze(-1).reshape(batch_size, nNodes)
                pseudo_mask_dev = torch.BoolTensor(current_pseudo_mask).to(device)

                for g in range(batch_size):
                    pred_pseudo = _yHat_all[g][pseudo_mask_dev]
                    pl_pseudo = _pl_all[g][pseudo_mask_dev].detach()
                    pseudo_loss = pseudo_loss + torch.nn.functional.mse_loss(pred_pseudo, pl_pseudo)
                pseudo_loss = pseudo_loss / batch_size

            # Total loss
            total_loss = sup_loss + lambda_lap * lap_loss + lambda_pseudo * pseudo_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            _LOSS += total_loss.item()
            _SUP_LOSS += sup_loss.item()
            _LAP_LOSS += lap_loss.item()
            _PSEUDO_LOSS += pseudo_loss.item()

            _n_labeled_actual = label_mask.sum().item() // max(1, batch_size)
            _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
            _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
            pred += list(_pred.cpu().detach().numpy())
            truth += list(_truth.cpu().detach().numpy())

        scheduler.step()
        truth, pred = np.array(truth), np.array(pred)
        trainRMSE = utils.RMSE(truth, pred)
        n_batches = _n + 1

        # Validation
        validLoss, validRMSE, _, _ = network_semi.test(
            validLoader, model, lossFn, device, nNodes, nNodes_labeled
        )

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            log_line = (f"Epoch {epoch}: RMSE {trainRMSE[0]:.3f}/{validRMSE[0]:.3f}; "
                        f"sup={_SUP_LOSS/n_batches:.4e} | lap={_LAP_LOSS/n_batches:.4e} | "
                        f"pseudo={_PSEUDO_LOSS/n_batches:.4e} ({len(selected_pseudo_nodes)} nodes); "
                        f"LR {scheduler.get_last_lr()[0]:.6f}")
            print(log_line, file=f)

        wandb.log({
            'epoch': epoch,
            'train/rmse': trainRMSE[0], 'valid/rmse': validRMSE[0],
            'train/sup_loss': _SUP_LOSS / n_batches,
            'train/lap_loss': _LAP_LOSS / n_batches,
            'train/pseudo_loss': _PSEUDO_LOSS / n_batches,
            'pseudo/n_selected': len(selected_pseudo_nodes),
            'pseudo/expansion_round': expansion_round,
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': bestLoss,
        })

        # Save best
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print(f"  Model saved. Best valid RMSE: {bestLoss:.6f}", file=f)
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(), 'bestLoss': bestLoss,
            }, chkptPath)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{nEpoch}: train={trainRMSE[0]:.4f}, valid={validRMSE[0]:.4f}, "
                  f"best={bestLoss:.4f}, pseudo_nodes={len(selected_pseudo_nodes)}")

        hist.append([_LOSS/n_batches, validLoss, trainRMSE[0], validRMSE[0]])
        utils.plotHist(hist, modelName, output_dir=output_dir)

    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}", file=f)
    wandb.finish()


if __name__ == "__main__":
    main()

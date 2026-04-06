"""
Graph-Weighted Pseudo-Labels for Semi-Supervised GNN Regression
Based on HPL framework but replaces MLP uncertainty estimator with
graph-structure-based weighting using labeled neighbors' real errors.

Key idea: Instead of learning uncertainty via bi-level optimization (HPL),
dynamically compute per-unlabeled-node weights each epoch by propagating
labeled nodes' prediction errors through the graph structure.

- Weak augmentation → pseudo-labels (same as HPL)
- Strong augmentation → model prediction → consistency loss
- Weight = f(labeled neighbors' real errors), updated every epoch
"""
import numpy as np
import torch, pickle, data_semi, network_semi, utils
import os, wandb, random as _random
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
n_unlabeled_env = os.environ.get('N_UNLABELED', '200')
use_fps = int(os.environ.get('USE_FPS', 1))
fps_tag = '_fps_score' if use_fps == 2 else ('_fps' if use_fps == 1 else '')
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = os.path.join(project_root, 'log', f'graph_weight_lap_pl_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'graph_weight_lap_pl_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


# ==================== Feature layout (same as HPL/SimRegMatch/MT) ====================
def _get_feature_layout(total_dim):
    wrf_dim = 63
    n_timesteps = 5
    wrf_total = wrf_dim * n_timesteps  # 315
    clms_dim = 3
    urban_dim = 17
    geo_dim = total_dim - wrf_total - clms_dim - urban_dim
    return {
        'wrf_dim': wrf_dim, 'n_timesteps': n_timesteps, 'wrf_total': wrf_total,
        'clms': (wrf_total, wrf_total + clms_dim), 'clms_dim': clms_dim,
        'urban': (wrf_total + clms_dim, wrf_total + clms_dim + urban_dim), 'urban_dim': urban_dim,
        'geo': (wrf_total + clms_dim + urban_dim, total_dim), 'geo_dim': geo_dim,
    }

def _build_wrf_var_indices(var_names):
    var_map = {'Tair': 0, 'Tskin': 1, 'Tsoil': 2, 'Humid': 3, 'Irrad': slice(4, 8),
               'WindX': slice(8, 11), 'WindY': slice(11, 14), 'Pressure': 14}
    indices = []
    for v in var_names:
        idx = var_map.get(v)
        if idx is None:
            continue
        if isinstance(idx, slice):
            base = list(range(idx.start, idx.stop))
        else:
            base = [idx]
        for t in range(5):
            for b in base:
                indices.append(t * 63 + b)
    return indices

def _build_wrf_timestep_indices(timestep):
    return list(range(timestep * 63, (timestep + 1) * 63))

def _drop_edges(edge_index, edge_attr, drop_rate):
    if drop_rate <= 0 or edge_index.size(1) == 0:
        return edge_index, edge_attr
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
    if keep_mask.sum() < num_edges * 0.5:
        keep_mask = torch.rand(num_edges, device=edge_index.device) > 0.5
    return edge_index[:, keep_mask], (edge_attr[keep_mask] if edge_attr is not None else None)


# ==================== Augmentation (same tuned version as HPL/SimRegMatch) ====================
def structured_augment_weak(x, edge_index, edge_attr):
    """Weak augmentation: keep Tair+Tskin, mask 1 low group, mask 10% GeoEmbed."""
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    low_groups = ['Humid', 'Irrad', 'WindX', 'WindY']
    chosen = _random.choice(low_groups)
    x_aug[:, _build_wrf_var_indices([chosen])] = 0.0

    geo_start, geo_end = layout['geo']
    geo_mask = torch.rand(layout['geo_dim'], device=x.device) > 0.10
    x_aug[:, geo_start:geo_end] *= geo_mask.float().unsqueeze(0)

    return x_aug, edge_index, edge_attr


def structured_augment_strong(x, edge_index, edge_attr):
    """Strong augmentation (tuned): keep Tair+Tskin, mask 1 med/low group,
    mask 1 time step, mask 10% GeoEmbed, drop 10% edges."""
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    candidates = ['Humid', 'Irrad', 'WindX', 'WindY', 'CLMS']
    chosen_groups = _random.sample(candidates, 1)
    wrf_vars = [v for v in chosen_groups if v != 'CLMS']
    if wrf_vars:
        x_aug[:, _build_wrf_var_indices(wrf_vars)] = 0.0
    if 'CLMS' in chosen_groups:
        s, e = layout['clms']
        x_aug[:, s:e] = 0.0

    step_to_mask = _random.choice([1, 2, 3, 4])
    x_aug[:, _build_wrf_timestep_indices(step_to_mask)] = 0.0

    geo_start, geo_end = layout['geo']
    geo_mask = torch.rand(layout['geo_dim'], device=x.device) > 0.10
    x_aug[:, geo_start:geo_end] *= geo_mask.float().unsqueeze(0)

    edge_index_aug, edge_attr_aug = _drop_edges(edge_index, edge_attr, drop_rate=0.10)
    return x_aug, edge_index_aug, edge_attr_aug


# ==================== Graph-based weight computation ====================
def compute_graph_weights(labeled_errors, adj_matrix, nNodes_labeled, nNodes_unlabeled, scale=10.0):
    """
    Compute per-unlabeled-node weights based on labeled neighbors' real errors.

    Args:
        labeled_errors: (nNodes_labeled,) mean absolute error per labeled node
        adj_matrix: (nNodes, nNodes) adjacency matrix
        nNodes_labeled: number of labeled nodes
        nNodes_unlabeled: number of unlabeled nodes
        scale: controls how sharply error maps to weight

    Returns:
        weights: (nNodes_unlabeled,) confidence weights in [0, 1]
        stats: dict with debug statistics
    """
    nNodes = nNodes_labeled + nNodes_unlabeled
    weights = np.ones(nNodes_unlabeled) * 0.5  # default for isolated nodes

    n_labeled_nbs_list = []
    estimated_errors = []

    for u in range(nNodes_unlabeled):
        u_global = nNodes_labeled + u
        # Find labeled neighbors and their edge weights
        labeled_nbs_weights = []
        for j in range(nNodes_labeled):
            ew = adj_matrix[u_global, j]
            if ew > 0:
                labeled_nbs_weights.append((j, ew))

        n_labeled_nbs_list.append(len(labeled_nbs_weights))

        if len(labeled_nbs_weights) == 0:
            weights[u] = 0.3  # isolated from labeled nodes, low confidence
            estimated_errors.append(labeled_errors.max() if len(labeled_errors) > 0 else 1.0)
            continue

        # Weighted average of labeled neighbors' errors
        weighted_error_sum = 0.0
        weight_sum = 0.0
        for nb_idx, ew in labeled_nbs_weights:
            weighted_error_sum += ew * labeled_errors[nb_idx]
            weight_sum += ew
        estimated_error = weighted_error_sum / weight_sum
        estimated_errors.append(estimated_error)

        # Weight: low error → high weight
        weights[u] = 1.0 / (1.0 + estimated_error * scale)

    estimated_errors = np.array(estimated_errors)
    n_labeled_nbs = np.array(n_labeled_nbs_list)

    stats = {
        'weight_mean': weights.mean(),
        'weight_min': weights.min(),
        'weight_max': weights.max(),
        'weight_std': weights.std(),
        'estimated_error_mean': estimated_errors.mean(),
        'estimated_error_min': estimated_errors.min(),
        'estimated_error_max': estimated_errors.max(),
        'labeled_nbs_mean': n_labeled_nbs.mean(),
        'labeled_nbs_min': n_labeled_nbs.min(),
        'labeled_nbs_max': n_labeled_nbs.max(),
        'nodes_with_0_labeled_nbs': int((n_labeled_nbs == 0).sum()),
        'nodes_with_weight_above_05': int((weights > 0.5).sum()),
        'nodes_with_weight_below_02': int((weights < 0.2).sum()),
    }
    return weights, stats


# ==================== Main ====================
def main():
    # ========== Data ==========
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': 12,
        'batchSize': int(os.environ.get('BATCH_SIZE', 128)),
        'thres': float(os.environ.get('THRES', 0.1)),
        'geoFeatures': 'full',
    }
    n_unlabeled = int(os.environ.get('N_UNLABELED', 200))
    w_ulb = float(os.environ.get('W_ULB', 1.0))
    lambda_lap = float(os.environ.get('LAMBDA_LAP', '0.1'))
    weight_scale = float(os.environ.get('WEIGHT_SCALE', '10.0'))
    weight_update_interval = int(os.environ.get('WEIGHT_UPDATE_INTERVAL', '1'))  # update weights every N epochs

    os.environ['OUTPUT_DIR'] = output_dir
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)

    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    nNodes_unlabeled = metadata['nNodes_unlabeled']
    adj_matrix = metadata['AdjMatrix']
    iDim = metadata['iDim']

    # Pre-compute Laplacian edges (for spatial smoothness loss)
    _adj_lap = adj_matrix.copy()
    np.fill_diagonal(_adj_lap, 0)
    _lap_src, _lap_dst = np.nonzero(_adj_lap)
    lap_edge_w = torch.FloatTensor(_adj_lap[_lap_src, _lap_dst]).to(device)
    lap_edge_src = torch.LongTensor(_lap_src).to(device)
    lap_edge_dst = torch.LongTensor(_lap_dst).to(device)
    print(f"Laplacian edges: {len(_lap_src)} (from adj_matrix)")

    # ========== Model ==========
    nEpoch = 5000
    modelParam = {
        'iDim': iDim, 'oDim': 1, 'HLD': 128,
        'nGNN': 3, 'nMLP': 2, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'BN': False, 'Dropout': True, 'conv_type': conv_type,
    }
    model = network_semi.GNN(modelParam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)

    # ========== Logging ==========
    modelName = f'geoEmbed_{dataParam["geoMethod"]}_{conv_type}_graph_weight_pl_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName

    wandb.init(
        project="Semi-supervised GNN",
        entity="urban_prediction",
        name=wandb_name,
        config={
            **dataParam, **modelParam,
            'method': 'Graph-Weighted Pseudo-Labels + Laplacian',
            'w_ulb': w_ulb,
            'lambda_lap': lambda_lap,
            'weight_scale': weight_scale,
            'weight_update_interval': weight_update_interval,
            'n_unlabeled': n_unlabeled,
            'use_fps': use_fps,
            'nEpoch': nEpoch,
            'lr': 1e-3,
        }
    )

    with open(f'{output_dir}/{modelName}_param.pkl', 'wb') as f:
        pickle.dump({'modelParam': modelParam, 'dataParam': dataParam}, f)

    # Log config
    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("No checkpoint found, starting new model.", file=f)
    if torch.cuda.is_available():
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(model, file=f)
        print("=" * 60, file=f)
        print("Graph-Weighted Pseudo-Labels + Laplacian Configuration:", file=f)
        print(f"  Method: Consistency training with graph-based neighbor error weighting + Laplacian", file=f)
        print(f"  Output dir: {output_dir}", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} ({nNodes_labeled} labeled + {nNodes_unlabeled} unlabeled)", file=f)
        print(f"  w_ulb: {w_ulb}", file=f)
        print(f"  lambda_lap: {lambda_lap}", file=f)
        print(f"  Weight scale: {weight_scale}", file=f)
        print(f"  Weight update interval: every {weight_update_interval} epoch(s)", file=f)
        print(f"  Augmentation: structured (weak=mask 1 low group/10%geo; "
              f"strong=mask 1 group/1 timestep/10%geo/10%edges)", file=f)
        print(f"  LR: 1e-3, scheduler: ExponentialLR(0.9992)", file=f)
        print(f"  Feature dim: {iDim}", file=f)
        print(f"  Use FPS: {use_fps}", file=f)

        # Graph neighbor stats
        nL = nNodes_labeled
        n_ll = int(np.sum(adj_matrix[:nL, :nL] > 0) // 2)
        n_uu = int(np.sum(adj_matrix[nL:, nL:] > 0) // 2)
        n_lu = int(np.sum(adj_matrix[:nL, nL:] > 0))
        print(f"  Graph: L-L={n_ll}, U-U={n_uu}, L-U={n_lu}, total={n_ll+n_uu+n_lu}", file=f)
        print("=" * 60, file=f)

    # ========== Training ==========
    bestLoss = np.inf
    hist = []
    # Initialize weights uniformly (first epoch has no error info yet)
    node_weights = np.ones(nNodes_unlabeled) * 0.5
    node_weights_tensor = torch.FloatTensor(node_weights).to(device)

    for epoch in range(nEpoch):
        model.train()
        epoch_labeled_loss = 0
        epoch_unlabeled_loss = 0
        epoch_lap_loss = 0
        epoch_total_loss = 0
        epoch_diff_mean = 0
        n_batches = 0
        pred_all, truth_all = [], []
        # Collect per-labeled-node errors for weight update
        labeled_preds_epoch = []
        labeled_truths_epoch = []

        for _n, _batch in enumerate(trainLoader):
            _batch = _batch.to(device)
            label_mask = _batch.label_mask
            unlabel_mask = ~label_mask
            batch_size = _batch.x.shape[0] // nNodes

            # --- Weak augmentation → pseudo-labels ---
            x_weak, ei_weak, ea_weak = structured_augment_weak(
                _batch.x, _batch.edge_index, _batch.edge_attr)
            with torch.no_grad():
                pred_weak_all = model(x_weak, ei_weak, ea_weak)
            pred_weak_ulb = pred_weak_all[unlabel_mask].detach()

            # --- Strong augmentation → model prediction ---
            x_strong, ei_strong, ea_strong = structured_augment_strong(
                _batch.x, _batch.edge_index, _batch.edge_attr)
            pred_strong_all = model(x_strong, ei_strong, ea_strong)
            pred_strong_labeled = pred_strong_all[label_mask]
            pred_strong_ulb = pred_strong_all[unlabel_mask]

            # --- Labeled loss ---
            y_labeled = _batch.y[label_mask]
            loss_labeled = lossFn(pred_strong_labeled, y_labeled)

            # --- Collect labeled predictions for weight update ---
            labeled_preds_epoch.append(pred_strong_labeled.detach().cpu())
            labeled_truths_epoch.append(y_labeled.detach().cpu())

            # --- Weighted unlabeled consistency loss ---
            # Expand node_weights to match batch: (batch_size * nNodes_unlabeled,)
            weight_expanded = node_weights_tensor.repeat(batch_size).unsqueeze(-1)  # (B*nU, 1)

            pred_diff = pred_strong_ulb - pred_weak_ulb.detach()
            unlabel_mse = pred_diff ** 2
            loss_unlabeled = torch.mean(weight_expanded * unlabel_mse)

            # --- Laplacian loss (spatial smoothness on unaugmented forward pass) ---
            pred_for_lap = model(_batch.x, _batch.edge_index, _batch.edge_attr)
            pred_first = pred_for_lap[:nNodes].squeeze(-1)  # first graph in batch
            lap_diff = pred_first[lap_edge_src] - pred_first[lap_edge_dst]
            loss_lap = torch.mean(lap_edge_w * lap_diff ** 2)

            # --- Total loss ---
            total_loss = loss_labeled + w_ulb * loss_unlabeled + lambda_lap * loss_lap

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # --- Stats ---
            epoch_labeled_loss += loss_labeled.item()
            epoch_unlabeled_loss += loss_unlabeled.item()
            epoch_lap_loss += loss_lap.item()
            epoch_total_loss += total_loss.item()
            epoch_diff_mean += pred_diff.abs().mean().item()
            n_batches += 1

            pred_all.append(pred_strong_labeled.detach().cpu())
            truth_all.append(y_labeled.detach().cpu())

        scheduler.step()

        # ========== Update graph-based weights ==========
        if epoch % weight_update_interval == 0:
            # Compute per-labeled-node error from this epoch
            all_preds = torch.cat(labeled_preds_epoch).numpy().reshape(-1, nNodes_labeled)
            all_truths = torch.cat(labeled_truths_epoch).numpy().reshape(-1, nNodes_labeled)
            per_node_mae = np.mean(np.abs(all_preds - all_truths), axis=0)  # (nNodes_labeled,)

            node_weights, weight_stats = compute_graph_weights(
                per_node_mae, adj_matrix, nNodes_labeled, nNodes_unlabeled, scale=weight_scale
            )
            node_weights_tensor = torch.FloatTensor(node_weights).to(device)

        # ========== Epoch stats ==========
        avg_labeled_loss = epoch_labeled_loss / max(n_batches, 1)
        avg_unlabeled_loss = epoch_unlabeled_loss / max(n_batches, 1)
        avg_lap_loss = epoch_lap_loss / max(n_batches, 1)
        avg_total_loss = epoch_total_loss / max(n_batches, 1)
        avg_diff_mean = epoch_diff_mean / max(n_batches, 1)

        pred_cat = torch.cat(pred_all).numpy()
        truth_cat = torch.cat(truth_all).numpy()
        trainRMSE = np.sqrt(np.mean((pred_cat - truth_cat) ** 2))

        # ========== Validation ==========
        model.eval()
        val_pred, val_truth = [], []
        with torch.no_grad():
            for _batch in validLoader:
                _batch = _batch.to(device)
                _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
                val_pred.append(_yHat[_batch.label_mask].cpu())
                val_truth.append(_batch.y[_batch.label_mask].cpu())
        val_pred = torch.cat(val_pred).numpy()
        val_truth = torch.cat(val_truth).numpy()
        validRMSE = np.sqrt(np.mean((val_pred - val_truth) ** 2))
        validLoss = np.mean((val_pred - val_truth) ** 2)

        # ========== Logging ==========
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {avg_total_loss:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE:.3f}/{validRMSE:.3f}; LR {scheduler.get_last_lr()[0]:.6f}", file=f)
            print(f"  labeled_loss={avg_labeled_loss:.4e} | "
                  f"unlabeled_loss={avg_unlabeled_loss:.4e} (w_ulb={w_ulb}) | "
                  f"lap_loss={avg_lap_loss:.4e} (λ={lambda_lap})", file=f)
            print(f"  [GW] weight: mean={node_weights.mean():.4f}, min={node_weights.min():.4f}, "
                  f"max={node_weights.max():.4f} | pred_diff: {avg_diff_mean:.4f}", file=f)

            # Detailed debug every 100 epochs + first 5
            if epoch % 100 == 0 or epoch < 5:
                print(f"  [GW debug] per-node labeled MAE: min={per_node_mae.min():.4f}, "
                      f"max={per_node_mae.max():.4f}, mean={per_node_mae.mean():.4f}", file=f)
                print(f"  [GW debug] estimated unlabeled error: min={weight_stats['estimated_error_min']:.4f}, "
                      f"max={weight_stats['estimated_error_max']:.4f}, mean={weight_stats['estimated_error_mean']:.4f}", file=f)
                print(f"  [GW debug] labeled nbs per unlabeled: min={weight_stats['labeled_nbs_min']}, "
                      f"max={weight_stats['labeled_nbs_max']}, mean={weight_stats['labeled_nbs_mean']:.1f}", file=f)
                print(f"  [GW debug] weight>0.5: {weight_stats['nodes_with_weight_above_05']}/{nNodes_unlabeled}, "
                      f"weight<0.2: {weight_stats['nodes_with_weight_below_02']}/{nNodes_unlabeled}, "
                      f"isolated: {weight_stats['nodes_with_0_labeled_nbs']}", file=f)

        # W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': avg_total_loss,
            'train/rmse': trainRMSE,
            'train/labeled_loss': avg_labeled_loss,
            'train/unlabeled_loss': avg_unlabeled_loss,
            'train/lap_loss': avg_lap_loss,
            'train/pred_diff': avg_diff_mean,
            'train/weight_mean': node_weights.mean(),
            'train/weight_min': node_weights.min(),
            'train/weight_max': node_weights.max(),
            'valid/loss': validLoss,
            'valid/rmse': validRMSE,
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': min(bestLoss, validRMSE),
        })

        # ========== Save best ==========
        if validRMSE < bestLoss:
            bestLoss = validRMSE
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'bestLoss': bestLoss,
                'node_weights': node_weights,
            }, f'{output_dir}/{modelName}.pt')
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print(f"  Model saved. Best valid RMSE: {bestLoss:.6f}", file=f)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{nEpoch}: train={trainRMSE:.4f}, valid={validRMSE:.4f}, "
                  f"best={bestLoss:.4f}, weight_mean={node_weights.mean():.4f}, diff={avg_diff_mean:.4f}")

        hist.append([avg_total_loss, validLoss, trainRMSE, validRMSE])
        utils.plotHist(hist, modelName, output_dir=output_dir)

    # ========== Final ==========
    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}", file=f)
    wandb.finish()


if __name__ == '__main__':
    main()

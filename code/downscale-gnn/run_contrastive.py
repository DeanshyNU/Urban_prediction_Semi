"""
Contrastive Learning for Semi-Supervised GNN Regression
Inspired by CLSS (NeurIPS 2023): uses ordinal relationships between samples.

Key idea: Temperature-similar nodes should have similar GNN representations.
- Labeled pairs: use actual temperature difference to define similarity
- Unlabeled pairs: use graph structure (connected nodes → likely similar temperature)
- No pseudo-labels, no augmentation difference needed

Loss = L_supervised + λ_sc × L_supervised_contrastive + λ_gc × L_graph_contrastive
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pickle, data_semi, network_semi, utils
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
if job_id:
    output_dir = os.path.join(project_root, 'log', f'contrastive_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'contrastive_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


# ==================== Contrastive Loss for Regression ====================

def supervised_contrastive_loss(representations, targets, temperature=0.1, margin=0.1):
    """
    Supervised contrastive loss for regression.
    For each anchor, pull representations of samples with similar targets closer,
    push representations of samples with different targets farther.

    Args:
        representations: (N, D) normalized feature vectors from GNN encoder
        targets: (N,) continuous target values
        temperature: controls sharpness of similarity
        margin: target difference threshold for positive/negative
    Returns:
        loss: scalar
    """
    N = representations.size(0)
    if N < 2:
        return torch.tensor(0.0, device=representations.device)

    # Normalize representations
    reps = F.normalize(representations, dim=1)

    # Pairwise cosine similarity
    sim_matrix = torch.mm(reps, reps.t()) / temperature  # (N, N)

    # Pairwise target distance
    target_diff = (targets.unsqueeze(0) - targets.unsqueeze(1)).abs()  # (N, N)

    # Soft weights: closer targets → higher weight (positive pair)
    # w_ij = exp(-target_diff / margin)
    pos_weights = torch.exp(-target_diff / margin)

    # Mask diagonal
    mask_diag = ~torch.eye(N, dtype=torch.bool, device=representations.device)
    sim_matrix = sim_matrix * mask_diag.float()
    pos_weights = pos_weights * mask_diag.float()

    # For each anchor i, contrastive loss:
    # L_i = -sum_j(w_ij * sim_ij) / sum_j(w_ij) + log(sum_k(exp(sim_ik)))
    # Simplified: weighted InfoNCE
    exp_sim = torch.exp(sim_matrix) * mask_diag.float()
    log_sum_exp = torch.log(exp_sim.sum(dim=1) + 1e-8)

    # Weighted positive similarity
    weighted_pos = (pos_weights * sim_matrix).sum(dim=1) / (pos_weights.sum(dim=1) + 1e-8)

    loss = (-weighted_pos + log_sum_exp).mean()
    return loss


def graph_contrastive_loss(representations, edge_index, edge_attr, nNodes_labeled,
                           temperature=0.1):
    """
    Graph-structure-based contrastive loss.
    Connected nodes should have similar representations (weighted by edge strength).
    Works for ALL nodes including unlabeled.

    Args:
        representations: (N, D) normalized feature vectors
        edge_index: (2, E) edge indices
        edge_attr: (E,) edge weights
        nNodes_labeled: number of labeled nodes
        temperature: controls sharpness
    Returns:
        loss: scalar
        stats: dict with debug info
    """
    N = representations.size(0)
    reps = F.normalize(representations, dim=1)

    # Pairwise cosine similarity for all nodes
    sim_matrix = torch.mm(reps, reps.t()) / temperature

    # Build adjacency-based positive weight matrix from edges
    pos_weight_matrix = torch.zeros(N, N, device=representations.device)
    src, dst = edge_index[0], edge_index[1]
    if edge_attr is not None:
        pos_weight_matrix[src, dst] = edge_attr
    else:
        pos_weight_matrix[src, dst] = 1.0

    # Mask diagonal
    mask_diag = ~torch.eye(N, dtype=torch.bool, device=representations.device)

    # For each node, connected neighbors are positives (weighted), rest are negatives
    exp_sim = torch.exp(sim_matrix) * mask_diag.float()
    log_sum_exp = torch.log(exp_sim.sum(dim=1) + 1e-8)

    # Weighted positive similarity from graph neighbors
    weighted_pos = (pos_weight_matrix * sim_matrix).sum(dim=1) / (pos_weight_matrix.sum(dim=1) + 1e-8)

    # Only compute for nodes that have neighbors
    has_neighbors = pos_weight_matrix.sum(dim=1) > 0
    if has_neighbors.sum() == 0:
        return torch.tensor(0.0, device=representations.device), {}

    loss = (-weighted_pos[has_neighbors] + log_sum_exp[has_neighbors]).mean()

    # Debug stats
    with torch.no_grad():
        # Avg similarity between connected vs unconnected pairs
        connected_mask = pos_weight_matrix > 0
        if connected_mask.sum() > 0:
            avg_sim_connected = (sim_matrix * temperature)[connected_mask].mean().item()
        else:
            avg_sim_connected = 0
        unconnected_mask = (pos_weight_matrix == 0) & mask_diag
        if unconnected_mask.sum() > 0:
            avg_sim_unconnected = (sim_matrix * temperature)[unconnected_mask].mean().item()
        else:
            avg_sim_unconnected = 0

        # Labeled vs unlabeled representation similarity
        labeled_reps = reps[:nNodes_labeled]
        unlabeled_reps = reps[nNodes_labeled:]
        ll_sim = F.cosine_similarity(labeled_reps.unsqueeze(1), labeled_reps.unsqueeze(0), dim=2).mean().item() if nNodes_labeled > 1 else 0

    stats = {
        'avg_sim_connected': avg_sim_connected,
        'avg_sim_unconnected': avg_sim_unconnected,
        'sim_gap': avg_sim_connected - avg_sim_unconnected,
        'll_avg_sim': ll_sim,
    }

    return loss, stats


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
    os.environ['OUTPUT_DIR'] = output_dir
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)

    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    iDim = metadata['iDim']

    # ========== Hyperparameters ==========
    nEpoch = 5000
    lambda_sc = float(os.environ.get('LAMBDA_SC', 0.1))    # supervised contrastive weight
    lambda_gc = float(os.environ.get('LAMBDA_GC', 0.05))   # graph contrastive weight
    temperature = 0.1
    margin = 0.1  # for target distance in supervised contrastive

    # ========== Model ==========
    modelParam = {
        'iDim': iDim, 'oDim': 1, 'HLD': 128,
        'nGNN': 3, 'nMLP': 2, 'conv_type': conv_type,
    }
    model = network_semi.GNN(modelParam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)

    # ========== Logging ==========
    modelName = f'geoEmbed_average_{conv_type}_contrastive_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName
    chkptPath = f'{output_dir}/{modelName}.pt'

    wandb.init(
        project="Semi-supervised GNN",
        entity="urban_prediction",
        name=wandb_name,
        config={
            'method': 'Contrastive Learning for Regression',
            'conv_type': conv_type, 'n_unlabeled': n_unlabeled,
            'lambda_sc': lambda_sc, 'lambda_gc': lambda_gc,
            'temperature': temperature, 'margin': margin,
            'use_fps': use_fps,
        }
    )

    with open(f'{output_dir}/{modelName}_param.pkl', 'wb') as f:
        pickle.dump({'modelParam': modelParam, 'dataParam': dataParam}, f)

    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("No checkpoint found, starting new model.", file=f)
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
            print(model, file=f)
            print("=" * 60, file=f)
            print("Contrastive Learning Configuration:", file=f)
            print(f"  Conv type: {conv_type}", file=f)
            print(f"  Nodes: {nNodes} ({nNodes_labeled} labeled + {n_unlabeled} unlabeled)", file=f)
            print(f"  Lambda_SC: {lambda_sc} (supervised contrastive)", file=f)
            print(f"  Lambda_GC: {lambda_gc} (graph contrastive)", file=f)
            print(f"  Temperature: {temperature}", file=f)
            print(f"  Margin: {margin} (target distance for soft positive)", file=f)
            print(f"  Feature dim: {iDim}", file=f)
            print(f"  No pseudo-labels, no augmentation needed", file=f)
            print("=" * 60, file=f)

    # ========== Training ==========
    bestLoss = np.inf
    hist = []

    for epoch in range(nEpoch):
        model.train()
        epoch_sup_loss = 0
        epoch_sc_loss = 0
        epoch_gc_loss = 0
        epoch_sim_gap = 0
        n_batches = 0
        pred_all, truth_all = [], []

        for _n, _batch in enumerate(trainLoader):
            _batch = _batch.to(device)
            label_mask = _batch.label_mask
            opt.zero_grad()

            # Forward: get both predictions and intermediate representations
            # Extract representation from after GNN processor (before decoder)
            x = _batch.x
            for _f in model.encoder:
                x = _f(x)
            for _f in model.processor:
                if hasattr(_f, 'propagate'):  # GNN layer
                    x = _f(x, _batch.edge_index, _batch.edge_attr)
                else:
                    x = _f(x)
            representations = x  # (batch_nodes, 128) — GNN output before decoder

            # Decoder for predictions
            pred = representations
            for _f in model.decoder:
                pred = _f(pred)

            # --- Supervised loss (labeled only) ---
            labeled_pred = pred[label_mask]
            y_labeled = _batch.y[label_mask]
            sup_loss = lossFn(labeled_pred, y_labeled)

            # --- Supervised contrastive loss (labeled pairs only) ---
            # Use labeled representations and targets
            labeled_reps = representations[label_mask]
            labeled_targets = y_labeled.squeeze(-1)
            batch_size = _batch.x.shape[0] // nNodes
            # Reshape to per-graph for contrastive (use first graph if batched)
            if batch_size > 1:
                # Sample a subset for efficiency
                n_sample = min(nNodes_labeled * 2, labeled_reps.shape[0])
                idx = torch.randperm(labeled_reps.shape[0])[:n_sample]
                sc_loss = supervised_contrastive_loss(
                    labeled_reps[idx], labeled_targets[idx],
                    temperature=temperature, margin=margin)
            else:
                sc_loss = supervised_contrastive_loss(
                    labeled_reps, labeled_targets,
                    temperature=temperature, margin=margin)

            # --- Graph contrastive loss (all nodes, uses graph structure) ---
            # For efficiency, use first graph in batch
            first_graph_mask = torch.arange(_batch.x.shape[0], device=device) < nNodes
            first_graph_reps = representations[:nNodes]
            # Get edges for first graph only
            edge_mask = (_batch.edge_index[0] < nNodes) & (_batch.edge_index[1] < nNodes)
            first_ei = _batch.edge_index[:, edge_mask]
            first_ea = _batch.edge_attr[edge_mask] if _batch.edge_attr is not None else None

            gc_loss, gc_stats = graph_contrastive_loss(
                first_graph_reps, first_ei, first_ea,
                nNodes_labeled, temperature=temperature)

            # --- Total loss ---
            total_loss = sup_loss + lambda_sc * sc_loss + lambda_gc * gc_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # Stats
            epoch_sup_loss += sup_loss.item()
            epoch_sc_loss += sc_loss.item()
            epoch_gc_loss += gc_loss.item()
            epoch_sim_gap += gc_stats.get('sim_gap', 0)
            n_batches += 1

            pred_all.append(labeled_pred.detach().cpu())
            truth_all.append(y_labeled.detach().cpu())

        scheduler.step()

        # Epoch averages
        avg_sup = epoch_sup_loss / n_batches
        avg_sc = epoch_sc_loss / n_batches
        avg_gc = epoch_gc_loss / n_batches
        avg_sim_gap = epoch_sim_gap / n_batches

        pred_cat = torch.cat(pred_all).numpy()
        truth_cat = torch.cat(truth_all).numpy()
        trainRMSE = np.sqrt(np.mean((pred_cat - truth_cat) ** 2))

        # Validation
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

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {avg_sup + lambda_sc*avg_sc + lambda_gc*avg_gc:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE:.3f}/{validRMSE:.3f}; LR {scheduler.get_last_lr()[0]:.6f}", file=f)
            print(f"  sup_loss={avg_sup:.4e} | sc_loss={avg_sc:.4e} | gc_loss={avg_gc:.4e} | "
                  f"sim_gap={avg_sim_gap:.4f}", file=f)

        wandb.log({
            'epoch': epoch, 'train_rmse': trainRMSE, 'valid_rmse': validRMSE,
            'sup_loss': avg_sup, 'sc_loss': avg_sc, 'gc_loss': avg_gc,
            'sim_gap': avg_sim_gap, 'best_valid_rmse': min(bestLoss, validRMSE),
            'lr': scheduler.get_last_lr()[0],
        })

        hist.append([avg_sup + lambda_sc*avg_sc + lambda_gc*avg_gc, validLoss])

        if validRMSE < bestLoss:
            bestLoss = validRMSE
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'bestLoss': bestLoss}, chkptPath)
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print(f"  Model saved. Best valid RMSE: {bestLoss:.6f}", file=f)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{nEpoch}: train={trainRMSE:.4f}, valid={validRMSE:.4f}, "
                  f"best={bestLoss:.4f}, sim_gap={avg_sim_gap:.4f}")

    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}", file=f)
    utils.plotHist(hist, modelName, output_dir=output_dir)
    wandb.finish()


if __name__ == '__main__':
    main()

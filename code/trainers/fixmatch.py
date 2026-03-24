"""
FixMatch + Graph Uncertainty for semi-supervised regression on unified graph.

Single model (no EMA teacher):
  - Weak augmentation → pseudo labels (with graph uncertainty filtering)
  - Strong augmentation → student prediction → learn from filtered pseudo labels

Graph Uncertainty:
  - u_spatial: |pred_i - mean(pred_neighbors)| — spatial smoothness prior
  - u_dist: shortest hop distance to nearest labeled node — supervision proximity
"""
import numpy as np
import torch
from collections import defaultdict


def compute_hop_distances(edge_index, n_labeled, total_nodes, device='cpu'):
    """
    Precompute shortest hop distance from each node to the nearest labeled node.
    Uses BFS from all labeled nodes simultaneously.

    Returns:
        hop_dist: (total_nodes,) tensor, hop distance to nearest labeled node
                  labeled nodes have distance 0
    """
    # Build adjacency list from edge_index
    adj = defaultdict(set)
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    for s, d in zip(src, dst):
        adj[int(s)].add(int(d))
        adj[int(d)].add(int(s))

    # BFS from all labeled nodes (multi-source BFS)
    hop_dist = np.full(total_nodes, fill_value=999, dtype=np.int32)
    queue = []
    for i in range(n_labeled):
        hop_dist[i] = 0
        queue.append(i)

    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for neighbor in adj[node]:
            if hop_dist[neighbor] > hop_dist[node] + 1:
                hop_dist[neighbor] = hop_dist[node] + 1
                queue.append(neighbor)

    return torch.tensor(hop_dist, dtype=torch.long, device=device)


def compute_spatial_uncertainty(predictions, edge_index, node_mask):
    """
    Compute spatial uncertainty for each node:
    u_spatial_i = |pred_i - mean(pred_neighbors)|

    Args:
        predictions: (total_nodes, 1) all node predictions
        edge_index: (2, E) graph edges
        node_mask: (total_nodes,) bool mask for nodes to compute uncertainty

    Returns:
        u_spatial: (n_masked,) spatial uncertainty for masked nodes
    """
    src, dst = edge_index  # (E,), (E,)
    pred_flat = predictions.squeeze(-1)  # (total_nodes,)

    # For each node, compute mean of neighbor predictions
    # Use scatter_mean approach
    n_nodes = predictions.shape[0]
    neighbor_sum = torch.zeros(n_nodes, device=predictions.device)
    neighbor_count = torch.zeros(n_nodes, device=predictions.device)

    neighbor_sum.scatter_add_(0, dst, pred_flat[src])
    neighbor_count.scatter_add_(0, dst, torch.ones_like(src, dtype=torch.float))

    # Avoid division by zero for isolated nodes
    neighbor_count = neighbor_count.clamp(min=1.0)
    neighbor_mean = neighbor_sum / neighbor_count

    # Spatial uncertainty = |pred - mean(neighbor_preds)|
    u_spatial_all = (pred_flat - neighbor_mean).abs()

    return u_spatial_all[node_mask]


def update_ema(model, ema_model, alpha, step):
    """Update EMA teacher model parameters."""
    alpha = min(1 - 1 / (step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def train_fixmatch_unified(loader, model, ema_model, lossFn, pseudo_loss_fn,
                           opt, scheduler, device, n_labeled,
                           hop_dist, lambda_U=3.0, tau_dist=3,
                           alpha=0.999, global_step=0):
    """
    FixMatch + Graph Uncertainty + EMA Teacher training (unified graph).

    EMA teacher generates stable pseudo labels from weak augmentation.
    Student learns from strong augmentation with graph uncertainty filtering.

    Args:
        loader: dict with 'weak' and 'strong' DataLoaders (same order, same split)
        model: student GNN model (trained with gradients)
        ema_model: EMA teacher model (generates pseudo labels, no gradients)
        lossFn: labeled loss (HuberLoss)
        pseudo_loss_fn: pseudo label loss (MSELoss)
        opt: optimizer
        scheduler: learning rate scheduler
        device: cuda/cpu
        n_labeled: number of labeled nodes
        hop_dist: (total_nodes,) precomputed hop distances to nearest labeled node
        lambda_U: unlabeled loss weight (after ramp-up)
        tau_dist: max hop distance threshold for graph uncertainty
        alpha: EMA decay coefficient
        global_step: global training step counter
    """
    model.train()
    ema_model.eval()

    total_labeled_loss = 0
    total_pseudo_loss = 0
    total_batches = 0
    pred_list, truth_list = [], []
    debug_stats = defaultdict(list)

    for batch_idx, (batch_weak, batch_strong) in enumerate(zip(loader['weak'], loader['strong'])):
        batch_weak = batch_weak.to(device)
        batch_strong = batch_strong.to(device)
        opt.zero_grad(set_to_none=True)

        # --- Step 1: EMA teacher generates pseudo labels from weak augmentation ---
        with torch.no_grad():
            all_pred_weak = ema_model(batch_weak.x, batch_weak.edge_index, batch_weak.edge_attr)
            pseudo_labels = all_pred_weak[batch_weak.unlabeled_mask].detach()  # (n_unlabeled, 1)

        # --- Step 2: Graph Uncertainty filtering ---
        # (a) u_spatial: spatial smoothness
        u_spatial = compute_spatial_uncertainty(
            all_pred_weak, batch_weak.edge_index, batch_weak.unlabeled_mask
        )  # (n_unlabeled,)

        # (b) u_dist: hop distance to nearest labeled node (precomputed)
        # Handle batched graphs: hop_dist is per single graph, tile for batch
        batch_size = batch_weak.y.shape[0] // n_labeled if n_labeled > 0 else 1
        if batch_size > 1:
            hop_dist_batch = hop_dist.repeat(batch_size)
        else:
            hop_dist_batch = hop_dist
        u_dist_unlabeled = hop_dist_batch[batch_weak.unlabeled_mask]  # (n_unlabeled,)

        # (c) Filter: dynamic τ_spatial (median) AND fixed τ_dist
        tau_spatial = u_spatial.median()
        mask_spatial = u_spatial <= tau_spatial
        mask_dist = u_dist_unlabeled <= tau_dist
        mask = mask_spatial & mask_dist

        n_unlabeled_total = int(batch_weak.unlabeled_mask.sum().item())
        n_pass_dist = int(mask_dist.sum().item())
        n_pass_both = int(mask.sum().item())

        # --- Step 3: Student learns from strong augmentation ---
        all_pred_strong = model(batch_strong.x, batch_strong.edge_index, batch_strong.edge_attr)

        # Labeled loss (from strong augmentation)
        labeled_pred = all_pred_strong[batch_strong.labeled_mask]
        labeled_loss = lossFn(labeled_pred, batch_strong.y)

        # Pseudo label loss (only filtered nodes)
        if n_pass_both > 0:
            student_unlabeled = all_pred_strong[batch_strong.unlabeled_mask]
            pseudo_loss = pseudo_loss_fn(student_unlabeled[mask], pseudo_labels[mask])
        else:
            pseudo_loss = torch.tensor(0.0, device=device)

        total_loss = labeled_loss + lambda_U * pseudo_loss
        total_loss.backward()

        # Gradient clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_after = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5

        opt.step()

        # --- EMA teacher update ---
        update_ema(model, ema_model, alpha, global_step + batch_idx)

        # --- Collect stats ---
        _pred = labeled_pred.reshape(-1, n_labeled)
        _truth = batch_strong.y.reshape(-1, n_labeled)

        total_labeled_loss += labeled_loss.item()
        total_pseudo_loss += pseudo_loss.item()
        pred_list.append(_pred.cpu().detach().numpy())
        truth_list.append(_truth.cpu().detach().numpy())
        total_batches += 1

        # Debug stats
        debug_stats['labeled_loss'].append(labeled_loss.item())
        debug_stats['pseudo_loss'].append(pseudo_loss.item())
        debug_stats['total_loss'].append(total_loss.item())
        debug_stats['n_unlabeled_total'].append(n_unlabeled_total)
        debug_stats['n_pass_dist'].append(n_pass_dist)
        debug_stats['n_pass_both'].append(n_pass_both)
        debug_stats['filter_rate'].append(n_pass_both / max(n_unlabeled_total, 1))
        debug_stats['tau_spatial'].append(tau_spatial.item())
        debug_stats['u_spatial_mean'].append(u_spatial.mean().item())
        debug_stats['u_spatial_max'].append(u_spatial.max().item())
        debug_stats['pseudo_label_mean'].append(pseudo_labels.mean().item())
        debug_stats['pseudo_label_std'].append(pseudo_labels.std().item())
        debug_stats['grad_norm_before_clip'].append(grad_norm_before.item() if torch.is_tensor(grad_norm_before) else grad_norm_before)
        debug_stats['grad_norm_after_clip'].append(grad_norm_after)

    scheduler.step()

    avg_labeled_loss = total_labeled_loss / max(total_batches, 1)
    avg_pseudo_loss = total_pseudo_loss / max(total_batches, 1)

    # Aggregate debug stats
    epoch_debug = {}
    for key, values in debug_stats.items():
        epoch_debug[f'debug/{key}_mean'] = np.mean(values)
        epoch_debug[f'debug/{key}_max'] = np.max(values)

    truth = np.concatenate(truth_list)
    pred = np.concatenate(pred_list)

    from .meanteacher import RMSE
    rmse = RMSE(truth, pred)

    return avg_labeled_loss + lambda_U * avg_pseudo_loss, rmse, truth, pred, epoch_debug


def test_fixmatch_unified(loader, model, lossFn, device, n_labeled):
    """
    Validation: only evaluate on labeled nodes (use weak loader or original).
    """
    model.eval()
    pred_list, truth_list = [], []
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_pred = model(batch.x, batch.edge_index, batch.edge_attr)
            labeled_pred = all_pred[batch.labeled_mask]
            loss = lossFn(labeled_pred, batch.y)

            _pred = labeled_pred.reshape(-1, n_labeled)
            _truth = batch.y.reshape(-1, n_labeled)

            total_loss += loss.item()
            pred_list.append(_pred.cpu().numpy())
            truth_list.append(_truth.cpu().numpy())
            total_batches += 1

    truth = np.concatenate(truth_list)
    pred = np.concatenate(pred_list)

    from .meanteacher import RMSE
    rmse = RMSE(truth, pred)

    return total_loss / max(total_batches, 1), rmse, truth, pred

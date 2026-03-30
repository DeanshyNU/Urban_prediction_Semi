"""
Mean Teacher training module
"""
import numpy as np
import torch
import os
import random as _random
from collections import defaultdict
from utils import RMSE
from copy import deepcopy


# ==================== Structured Data Augmentation ====================
# Feature layout (1343 dims total, window=2):
#   WRF block: 315 dims = 5 time steps x 63 dims/step
#     Time step order in feature vector: [current(63), hist[-2](63), hist[-1](63), fut[+1](63), fut[+2](63)]
#     Within each 63-dim step: 7 variables x 9 grid points
#       Tair(0-8), Tskin(9-17), Tsoil(18-26), Humid(27-35), Irrad(36-44), WindX(45-53), WindY(54-62)
#   CLMS: 3 dims (dynamic)
#   UrbanFeature: 17 dims (static)
#   GeoEmbed: 1008 dims (static)
#
# Variable importance for target temperature:
#   High:   Tair (corr=0.897), Tskin, Tsoil
#   Medium: Irrad, Humid
#   Low:    WindX, WindY, CLMS

# --- WRF variable group definitions (offsets within a single 63-dim time step) ---
WRF_VAR_GROUPS = {
    'Tair':  (0, 9),    # indices 0-8
    'Tskin': (9, 18),   # indices 9-17
    'Tsoil': (18, 27),  # indices 18-26
    'Humid': (27, 36),  # indices 27-35
    'Irrad': (36, 45),  # indices 36-44
    'WindX': (45, 54),  # indices 45-53
    'WindY': (54, 63),  # indices 54-62
}

# Variable importance tiers
HIGH_IMPORTANCE_VARS = ['Tair', 'Tskin', 'Tsoil']
MEDIUM_IMPORTANCE_VARS = ['Irrad', 'Humid']
LOW_IMPORTANCE_VARS = ['WindX', 'WindY']

# Feature dimension constants (window=2)
WRF_STEP_DIM = 63
WRF_WINDOW = 2
WRF_TOTAL_STEPS = 2 * WRF_WINDOW + 1  # 5 time steps
WRF_TOTAL_DIM = WRF_STEP_DIM * WRF_TOTAL_STEPS  # 315
CLMS_DIM = 3
URBAN_DIM = 17
# GeoEmbed dim is computed dynamically as total_dim - WRF_TOTAL_DIM - CLMS_DIM - URBAN_DIM


def _get_feature_layout(total_dim):
    """Compute feature block boundaries from total feature dimension."""
    geo_dim = total_dim - WRF_TOTAL_DIM - CLMS_DIM - URBAN_DIM
    return {
        'wrf':   (0, WRF_TOTAL_DIM),                                              # 0 : 315
        'clms':  (WRF_TOTAL_DIM, WRF_TOTAL_DIM + CLMS_DIM),                      # 315 : 318
        'urban': (WRF_TOTAL_DIM + CLMS_DIM, WRF_TOTAL_DIM + CLMS_DIM + URBAN_DIM),  # 318 : 335
        'geo':   (WRF_TOTAL_DIM + CLMS_DIM + URBAN_DIM, total_dim),              # 335 : 1343
        'geo_dim': geo_dim,
    }


def _get_wrf_time_step_offsets():
    """Return the starting offset of each WRF time step in the feature vector.
    Layout: [current(63), hist[-2](63), hist[-1](63), fut[+1](63), fut[+2](63)]
    Time step indices: 0=current, 1=hist[-2], 2=hist[-1], 3=fut[+1], 4=fut[+2]
    """
    return [i * WRF_STEP_DIM for i in range(WRF_TOTAL_STEPS)]


def _build_wrf_var_indices(var_names):
    """Build a list of absolute feature indices for given WRF variable names across ALL time steps."""
    offsets = _get_wrf_time_step_offsets()
    indices = []
    for step_offset in offsets:
        for var in var_names:
            start, end = WRF_VAR_GROUPS[var]
            indices.extend(range(step_offset + start, step_offset + end))
    return indices


def _build_wrf_timestep_indices(step_idx):
    """Build absolute feature indices for a single WRF time step (all variables)."""
    offset = step_idx * WRF_STEP_DIM
    return list(range(offset, offset + WRF_STEP_DIM))


def _drop_edges(edge_index, edge_attr, drop_rate):
    """Drop edges randomly. Returns augmented edge_index and edge_attr."""
    if drop_rate <= 0 or edge_index.size(1) == 0:
        return edge_index, edge_attr
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
    # Safety: keep at least 50% of edges
    if keep_mask.sum() < num_edges * 0.5:
        keep_mask = torch.rand(num_edges, device=edge_index.device) > 0.5
    edge_index_aug = edge_index[:, keep_mask]
    edge_attr_aug = edge_attr[keep_mask] if edge_attr is not None else None
    return edge_index_aug, edge_attr_aug


def structured_augment_weak(x, edge_index, edge_attr):
    """
    Weak augmentation for SimRegMatch / HPL pseudo-label generation.
    Design:
      - Keep Tair + Tskin (core temperature variables, always preserved)
      - Mask 1 low-importance variable group (random from WindX/WindY/CLMS)
      - Mask 10% of GeoEmbed dims
      - No edge dropout, no time step masking
    """
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    # 1. Mask 1 random low-importance group (from WindX, WindY, CLMS)
    low_choices = ['WindX', 'WindY', 'CLMS']
    chosen = _random.choice(low_choices)
    if chosen == 'CLMS':
        clms_start, clms_end = layout['clms']
        x_aug[:, clms_start:clms_end] = 0.0
    else:
        indices = _build_wrf_var_indices([chosen])
        x_aug[:, indices] = 0.0

    # 2. Mask 10% of GeoEmbed dims
    geo_start, geo_end = layout['geo']
    geo_dim = layout['geo_dim']
    geo_mask = torch.rand(geo_dim, device=x.device) > 0.10
    x_aug[:, geo_start:geo_end] = x_aug[:, geo_start:geo_end] * geo_mask.float().unsqueeze(0)

    return x_aug, edge_index, edge_attr


def structured_augment_strong(x, edge_index, edge_attr):
    """
    Strong augmentation for SimRegMatch / HPL student predictions.
    Design:
      - Keep only Tair (most critical, corr=0.897)
      - Mask 2-3 medium/low importance groups (random from Humid/Irrad/WindX/WindY/CLMS)
      - Mask 1 WRF time step (random, not current step 0)
      - Mask 30% of GeoEmbed dims
      - Drop 25% edges
    """
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    # 1. Mask 2-3 medium/low importance variable groups
    candidates = ['Humid', 'Irrad', 'WindX', 'WindY', 'CLMS']
    n_mask = _random.choice([2, 3])
    chosen_groups = _random.sample(candidates, min(n_mask, len(candidates)))
    wrf_vars_to_mask = [v for v in chosen_groups if v != 'CLMS']
    if wrf_vars_to_mask:
        indices = _build_wrf_var_indices(wrf_vars_to_mask)
        x_aug[:, indices] = 0.0
    if 'CLMS' in chosen_groups:
        clms_start, clms_end = layout['clms']
        x_aug[:, clms_start:clms_end] = 0.0

    # 2. Mask 1 random WRF time step (not current=step 0, to preserve some current info)
    #    Steps: 0=current, 1=hist[-2], 2=hist[-1], 3=fut[+1], 4=fut[+2]
    step_to_mask = _random.choice([1, 2, 3, 4])
    step_indices = _build_wrf_timestep_indices(step_to_mask)
    x_aug[:, step_indices] = 0.0

    # 3. Mask 30% of GeoEmbed dims
    geo_start, geo_end = layout['geo']
    geo_dim = layout['geo_dim']
    geo_mask = torch.rand(geo_dim, device=x.device) > 0.30
    x_aug[:, geo_start:geo_end] = x_aug[:, geo_start:geo_end] * geo_mask.float().unsqueeze(0)

    # 4. Drop 25% of edges
    edge_index_aug, edge_attr_aug = _drop_edges(edge_index, edge_attr, drop_rate=0.25)

    return x_aug, edge_index_aug, edge_attr_aug


def structured_augment_mt(x, edge_index, edge_attr):
    """
    Mean Teacher augmentation: equal strength, different random masks for eta and eta'.
    Per original MT paper: both views should have the same augmentation strength
    but different random realizations.
    Design (each call produces one view):
      - Always keep Tair (most critical)
      - Mask 2 random medium/low groups (from Humid/Irrad/WindX/WindY/CLMS)
      - Mask 1 random WRF time step (not current)
      - Mask 20% of GeoEmbed dims (random)
      - Drop 20% of edges (random)
    """
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    # 1. Mask 2 medium/low importance variable groups
    candidates = ['Humid', 'Irrad', 'WindX', 'WindY', 'CLMS']
    chosen_groups = _random.sample(candidates, 2)
    wrf_vars_to_mask = [v for v in chosen_groups if v != 'CLMS']
    if wrf_vars_to_mask:
        indices = _build_wrf_var_indices(wrf_vars_to_mask)
        x_aug[:, indices] = 0.0
    if 'CLMS' in chosen_groups:
        clms_start, clms_end = layout['clms']
        x_aug[:, clms_start:clms_end] = 0.0

    # 2. Mask 1 random WRF time step (not current)
    step_to_mask = _random.choice([1, 2, 3, 4])
    step_indices = _build_wrf_timestep_indices(step_to_mask)
    x_aug[:, step_indices] = 0.0

    # 3. Mask 20% of GeoEmbed dims
    geo_start, geo_end = layout['geo']
    geo_dim = layout['geo_dim']
    geo_mask = torch.rand(geo_dim, device=x.device) > 0.20
    x_aug[:, geo_start:geo_end] = x_aug[:, geo_start:geo_end] * geo_mask.float().unsqueeze(0)

    # 4. Drop 20% edges
    edge_index_aug, edge_attr_aug = _drop_edges(edge_index, edge_attr, drop_rate=0.20)

    return x_aug, edge_index_aug, edge_attr_aug


# Legacy wrapper for backward compatibility
def gnn_augment(x, edge_index, edge_attr, noise_std=0.15, edge_drop_rate=0.15,
                feat_mask_rate=0.15):
    """
    DEPRECATED: Use structured_augment_mt() instead.
    Kept for backward compatibility. Now delegates to structured_augment_mt.
    """
    return structured_augment_mt(x, edge_index, edge_attr)


def loadCheckPoint(modelName, model, opt, device, load=False, resetLr=False, lr=5e-5, predMode=False):
    """Load or initialize checkpoint"""
    chkptPath = f'{modelName}.pt'
    if os.path.exists(chkptPath) and load:
        chkpt = torch.load(chkptPath, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        opt.load_state_dict(chkpt['opt_state_dict'])
        EPOCH = chkpt['epoch']
        bestLoss = chkpt['bestLoss']
        hist = chkpt['hist']
        with open(f'{modelName}_log', 'a') as f:
            print("Checkpoint loaded.", file=f)
        if opt.param_groups[0]['lr'] < 1e-6 and resetLr:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            with open(f'{modelName}_log', 'a') as f:
                print(f"Resetting LR from {opt.param_groups[0]['lr']} to {lr}", file=f)
    elif predMode:
        if not os.path.exists(chkptPath):
            chkptPath = f'./trainedModels/{modelName}.pt'
        chkpt = torch.load(chkptPath, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        print("Checkpoint loaded.")
        return -1
    else:
        EPOCH = 0
        bestLoss = np.inf
        hist = []
        with open(f'{modelName}_log', 'w') as f:
            print("No checkpoint found, starting new model.", file=f)
    return EPOCH, bestLoss, chkptPath, hist


def update_ema_variables(model, ema_model, alpha, global_step):
    """Update teacher model (EMA) parameters"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def _set_model_nNodes(model, nNodes):
    """Temporarily set model's nNodes attribute"""
    if hasattr(model, 'nNodes'):
        model.nNodes = nNodes


def train_meanteacher(loader, unlabeled_loader, student_model, teacher_model, lossFn, consistency_loss_fn,
                      opt, scheduler, device, nNodes, lambda_U=1.0, alpha=0.999, global_step=0,
                      unlabeled_nNodes=None, noise_std=0.15, edge_drop_rate=0.15,
                      feat_mask_rate=0.15):
    """
    Standard Mean Teacher training (independent graph version)
    with structured data augmentation.

    Augmentation: structured_augment_mt (equal-strength, different-random for eta and eta')
    - Always keeps Tair (most important variable)
    - Masks 2 medium/low importance variable groups (different random per view)
    - Masks 1 WRF time step (different random per view)
    - Masks 20% GeoEmbed dims (different random per view)
    - Drops 20% edges (different random per view)
    """
    student_model.train()
    teacher_model.train()  # train mode for dropout noise

    if unlabeled_nNodes is None:
        unlabeled_nNodes = nNodes

    total_labeled_loss = 0
    total_consistency_loss = 0
    total_batches = 0
    pred, truth = [], []
    debug_stats = defaultdict(list)

    for batch_idx, (batch, unlabeled_batch) in enumerate(zip(loader, unlabeled_loader)):
        batch = batch.to(device)
        unlabeled_batch = unlabeled_batch.to(device)

        opt.zero_grad(set_to_none=True)

        # Labeled: student sees structured-augmented view, loss against ground truth
        _set_model_nNodes(student_model, nNodes)
        student_x_l, student_ei_l, student_ea_l = structured_augment_mt(
            batch.x, batch.edge_index, batch.edge_attr)
        student_logits = student_model(student_x_l, student_ei_l, student_ea_l)
        labeled_loss = lossFn(student_logits, batch.y)

        # Unlabeled: two independent structured augmentations (eta and eta') for consistency
        _set_model_nNodes(student_model, unlabeled_nNodes)
        _set_model_nNodes(teacher_model, unlabeled_nNodes)

        student_x_u, student_ei_u, student_ea_u = structured_augment_mt(
            unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.edge_attr)
        teacher_x_u, teacher_ei_u, teacher_ea_u = structured_augment_mt(
            unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.edge_attr)

        with torch.no_grad():
            teacher_predictions = teacher_model(teacher_x_u, teacher_ei_u, teacher_ea_u)

        student_predictions = student_model(student_x_u, student_ei_u, student_ea_u)

        _set_model_nNodes(student_model, nNodes)
        _set_model_nNodes(teacher_model, nNodes)

        consistency_loss = consistency_loss_fn(student_predictions, teacher_predictions)

        total_loss = labeled_loss + lambda_U * consistency_loss
        total_loss.backward()

        grad_norm_before = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=float('inf'))
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        grad_norm_after = sum(p.grad.norm().item()**2 for p in student_model.parameters() if p.grad is not None)**0.5

        opt.step()

        update_ema_variables(student_model, teacher_model, alpha, global_step + batch_idx)

        _pred = student_logits.reshape(-1, nNodes)
        _truth = batch.y.reshape(-1, nNodes)

        total_labeled_loss += labeled_loss.item()
        total_consistency_loss += consistency_loss.item()

        pred.append(_pred.cpu().detach().numpy())
        truth.append(_truth.cpu().detach().numpy())
        total_batches += 1

        debug_stats['labeled_loss'].append(labeled_loss.item())
        debug_stats['consistency_loss'].append(consistency_loss.item())
        debug_stats['total_loss'].append(total_loss.item())
        debug_stats['teacher_pred_mean'].append(teacher_predictions.mean().item())
        debug_stats['teacher_pred_std'].append(teacher_predictions.std().item())
        debug_stats['student_pred_mean'].append(student_predictions.mean().item())
        debug_stats['student_pred_std'].append(student_predictions.std().item())
        debug_stats['student_logits_mean'].append(student_logits.mean().item())
        debug_stats['student_logits_std'].append(student_logits.std().item())
        debug_stats['pred_diff_mean'].append((student_predictions - teacher_predictions).abs().mean().item())
        debug_stats['pred_diff_max'].append((student_predictions - teacher_predictions).abs().max().item())
        debug_stats['grad_norm_before_clip'].append(grad_norm_before.item() if torch.is_tensor(grad_norm_before) else grad_norm_before)
        debug_stats['grad_norm_after_clip'].append(grad_norm_after)

    scheduler.step()

    avg_labeled_loss = total_labeled_loss / total_batches
    avg_consistency_loss = total_consistency_loss / total_batches

    epoch_debug = {}
    for key, values in debug_stats.items():
        epoch_debug[f'debug/{key}_mean'] = np.mean(values)
        epoch_debug[f'debug/{key}_max'] = np.max(values)

    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss + lambda_U * avg_consistency_loss, rmse, truth, pred, epoch_debug


def train_meanteacher_unified(loader, student_model, teacher_model, lossFn, consistency_loss_fn,
                              opt, scheduler, device, n_labeled, lambda_U=1.0, alpha=0.999,
                              global_step=0, noise_std=0.15, edge_drop_rate=0.15,
                              feat_mask_rate=0.15):
    """
    Standard Mean Teacher training (unified graph version).
    Structured augmentation: two equal-strength, different-random views (eta and eta').
    - Always keeps Tair (most important variable)
    - Masks 2 medium/low importance variable groups (different random per view)
    - Masks 1 WRF time step (different random per view)
    - Masks 20% GeoEmbed dims (different random per view)
    - Drops 20% edges (different random per view)
    - Supervised loss: only on labeled nodes
    - Consistency loss: on ALL nodes (labeled + unlabeled)
    - Teacher updated via EMA, no gradient
    """
    student_model.train()
    teacher_model.train()  # train mode for dropout noise η'

    total_labeled_loss = 0
    total_consistency_loss = 0
    total_batches = 0
    pred, truth = [], []
    debug_stats = defaultdict(list)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)

        # Create two independent structured-augmented views (eta and eta')
        # Both have equal strength but different random masks/drops
        student_x, student_ei, student_ea = structured_augment_mt(
            batch.x, batch.edge_index, batch.edge_attr)
        teacher_x, teacher_ei, teacher_ea = structured_augment_mt(
            batch.x, batch.edge_index, batch.edge_attr)

        # Forward: both models process different augmented views of the unified graph
        all_student = student_model(student_x, student_ei, student_ea)
        with torch.no_grad():
            all_teacher = teacher_model(teacher_x, teacher_ei, teacher_ea)

        # Supervised loss: only on labeled nodes
        labeled_pred = all_student[batch.labeled_mask]
        labeled_loss = lossFn(labeled_pred, batch.y)

        # Consistency loss: on ALL nodes (per official MT implementation)
        consistency_loss = consistency_loss_fn(all_student, all_teacher)
        total_loss = labeled_loss + lambda_U * consistency_loss
        total_loss.backward()

        grad_norm_before = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=float('inf'))
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        grad_norm_after = sum(p.grad.norm().item()**2 for p in student_model.parameters() if p.grad is not None)**0.5

        opt.step()
        update_ema_variables(student_model, teacher_model, alpha, global_step + batch_idx)

        _pred = labeled_pred.reshape(-1, n_labeled)
        _truth = batch.y.reshape(-1, n_labeled)

        total_labeled_loss += labeled_loss.item()
        total_consistency_loss += consistency_loss.item()
        pred.append(_pred.cpu().detach().numpy())
        truth.append(_truth.cpu().detach().numpy())
        total_batches += 1

        debug_stats['labeled_loss'].append(labeled_loss.item())
        debug_stats['consistency_loss'].append(consistency_loss.item())
        debug_stats['total_loss'].append(total_loss.item())
        debug_stats['teacher_pred_mean'].append(all_teacher.mean().item())
        debug_stats['teacher_pred_std'].append(all_teacher.std().item())
        debug_stats['student_pred_mean'].append(all_student.mean().item())
        debug_stats['student_pred_std'].append(all_student.std().item())
        debug_stats['student_logits_mean'].append(labeled_pred.mean().item())
        debug_stats['student_logits_std'].append(labeled_pred.std().item())
        debug_stats['pred_diff_mean'].append((all_student - all_teacher).abs().mean().item())
        debug_stats['pred_diff_max'].append((all_student - all_teacher).abs().max().item())
        debug_stats['grad_norm_before_clip'].append(grad_norm_before.item() if torch.is_tensor(grad_norm_before) else grad_norm_before)
        debug_stats['grad_norm_after_clip'].append(grad_norm_after)

    scheduler.step()

    avg_labeled_loss = total_labeled_loss / total_batches
    avg_consistency_loss = total_consistency_loss / total_batches

    epoch_debug = {}
    for key, values in debug_stats.items():
        epoch_debug[f'debug/{key}_mean'] = np.mean(values)
        epoch_debug[f'debug/{key}_max'] = np.max(values)

    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss + lambda_U * avg_consistency_loss, rmse, truth, pred, epoch_debug


def test_meanteacher_unified_ablation(loader, model, lossFn, device, n_labeled):
    """
    消融测试：对比完整统一图 vs 仅labeled子图的RMSE
    用于诊断unlabeled节点是否对labeled预测有帮助
    """
    model.eval()
    rmse_full_list, rmse_labeled_only_list = [], []
    total_nodes = n_labeled + 200  # labeled + unlabeled per graph

    with torch.no_grad():
        for batch in loader:
            if batch.x is None or batch.y is None:
                continue
            batch = batch.to(device)
            batch_y = batch.y.reshape(-1, n_labeled)

            # --- 完整统一图（含unlabeled节点）---
            all_logits = model(batch.x, batch.edge_index, batch.edge_attr)
            labeled_logits_full = all_logits[batch.labeled_mask].reshape(-1, n_labeled)

            # --- 仅labeled子图（去掉unlabeled节点）---
            labeled_mask = batch.labeled_mask
            n_labeled_total = int(labeled_mask.sum().item())  # batch中所有labeled节点数
            labeled_x = batch.x[labeled_mask]

            # 过滤只含labeled节点的边
            src, dst = batch.edge_index
            edge_mask = labeled_mask[src] & labeled_mask[dst]
            labeled_edge_index_raw = batch.edge_index[:, edge_mask]
            labeled_edge_attr = batch.edge_attr[edge_mask] if batch.edge_attr is not None else None

            # 重新映射节点编号：labeled节点在batch中的全局索引 → 连续的0~n_labeled_total-1
            node_map = torch.full((batch.x.shape[0],), -1, dtype=torch.long, device=batch.x.device)
            node_map[labeled_mask] = torch.arange(n_labeled_total, device=batch.x.device)
            labeled_edge_index = node_map[labeled_edge_index_raw]

            logits_labeled_only = model(labeled_x, labeled_edge_index, labeled_edge_attr)
            logits_labeled_only = logits_labeled_only.reshape(-1, n_labeled)

            rmse_full_list.append(RMSE(batch_y.cpu().numpy(), labeled_logits_full.cpu().numpy())[0])
            rmse_labeled_only_list.append(RMSE(batch_y.cpu().numpy(), logits_labeled_only.cpu().numpy())[0])

    mean_full = np.mean(rmse_full_list)
    mean_labeled_only = np.mean(rmse_labeled_only_list)
    print(f"[消融] 完整统一图 RMSE={mean_full:.4f} | 仅labeled子图 RMSE={mean_labeled_only:.4f} | "
          f"差值={mean_full - mean_labeled_only:+.4f} ({'统一图更差' if mean_full > mean_labeled_only else '统一图更好'})")
    return mean_full, mean_labeled_only


def test_meanteacher_unified(loader, model, lossFn, device, n_labeled):
    """统一图版本的 Mean Teacher 测试函数"""
    model.eval()
    total_labeled_loss = 0
    pred, truth = [], []

    if len(loader) == 0:
        raise ValueError("Data loader is empty.")

    with torch.no_grad():
        for batch in loader:
            if batch.x is None or batch.y is None:
                continue
            batch = batch.to(device)
            all_logits = model(batch.x, batch.edge_index, batch.edge_attr)
            labeled_logits = all_logits[batch.labeled_mask].reshape(-1, n_labeled)
            batch_y = batch.y.reshape(-1, n_labeled)

            loss = lossFn(labeled_logits, batch_y)
            total_labeled_loss += loss.item()
            pred.append(labeled_logits.cpu().numpy())
            truth.append(batch_y.cpu().numpy())

    avg_labeled_loss = total_labeled_loss / len(loader)
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)
    return avg_labeled_loss, rmse, truth, pred


def test_meanteacher(loader, model, lossFn, device, nNodes):
    """Mean Teacher test function (uses teacher model)"""
    model.eval()
    total_labeled_loss = 0
    pred, truth = [], []

    if len(loader) == 0:
        raise ValueError("Data loader is empty.")

    with torch.no_grad():
        for batch in loader:
            if batch.x is None or batch.y is None:
                continue

            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)

            logits = logits.reshape(-1, nNodes)
            batch_y = batch.y.reshape(-1, nNodes)

            loss = lossFn(logits, batch_y)
            total_labeled_loss += loss.item()

            pred.append(logits.cpu().numpy())
            truth.append(batch_y.cpu().numpy())

    avg_labeled_loss = total_labeled_loss / len(loader)

    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss, rmse, truth, pred

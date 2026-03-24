"""
Mean Teacher training module
"""
import numpy as np
import torch
import os
from collections import defaultdict
from utils import RMSE
from copy import deepcopy


def gnn_augment(x, edge_index, edge_attr, noise_std=0.15, edge_drop_rate=0.15,
                feat_mask_rate=0.15):
    """
    GNN-aware stochastic augmentation for creating different views.
    Designed for urban spatial prediction with weather/climate features.

    Three types of perturbation (applied independently per call):
    1. Feature noise: Gaussian noise on node features (simulates WRF/CLMS measurement uncertainty)
    2. Edge dropout: randomly removes edges (forces model to learn robust spatial relationships)
    3. Feature masking: randomly zeros out entire feature dimensions (prevents over-reliance on
       specific WRF channels, e.g. model must predict without wind speed or without humidity)

    Default rates (0.15) are based on GNN augmentation literature:
    - DropEdge (ICLR 2020): 10-20% edge drop
    - Data Augmentation for GNNs (AAAI 2021): 10-20% feature mask
    - Higher than previous 0.1 to create meaningful prediction differences for MT consistency loss

    Args:
        x: node features (N, D)
        edge_index: edge indices (2, E)
        edge_attr: edge weights (E,)
        noise_std: std of Gaussian noise added to features
        edge_drop_rate: fraction of edges to randomly drop
        feat_mask_rate: fraction of feature dimensions to randomly mask
    Returns:
        x_aug, edge_index_aug, edge_attr_aug
    """
    # 1. Feature noise (simulates WRF/CLMS measurement uncertainty)
    x_aug = x + torch.randn_like(x) * noise_std

    # 2. Edge dropout (forces robust spatial relationship learning)
    if edge_drop_rate > 0 and edge_index.size(1) > 0:
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) > edge_drop_rate
        # Ensure at least 50% of edges are kept
        if keep_mask.sum() < num_edges * 0.5:
            keep_mask = torch.rand(num_edges, device=edge_index.device) > 0.5
        edge_index_aug = edge_index[:, keep_mask]
        edge_attr_aug = edge_attr[keep_mask] if edge_attr is not None else None
    else:
        edge_index_aug = edge_index
        edge_attr_aug = edge_attr

    # 3. Feature masking (prevents over-reliance on specific WRF channels)
    # Masks entire feature dimensions (columns), so all nodes lose the same features
    if feat_mask_rate > 0:
        feat_dim = x_aug.size(1)
        mask = torch.rand(feat_dim, device=x_aug.device) > feat_mask_rate
        # Ensure at least 50% of features are kept
        if mask.sum() < feat_dim * 0.5:
            mask = torch.rand(feat_dim, device=x_aug.device) > 0.5
        x_aug = x_aug * mask.float().unsqueeze(0)

    return x_aug, edge_index_aug, edge_attr_aug


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
    with GNN-aware augmentation on both labeled and unlabeled data.

    Key changes from basic MT:
    - Both student and teacher in train mode (dropout as implicit noise)
    - GNN augmentation (feature noise + edge dropout + feature masking) creates
      meaningful prediction differences for consistency loss
    - Consistency loss on unlabeled data with augmented views
    - Labeled data also augmented (per standard MT: same treatment for all data)
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

        # Labeled: student sees augmented view, loss against ground truth
        _set_model_nNodes(student_model, nNodes)
        student_x_l, student_ei_l, student_ea_l = gnn_augment(
            batch.x, batch.edge_index, batch.edge_attr,
            noise_std=noise_std, edge_drop_rate=edge_drop_rate,
            feat_mask_rate=feat_mask_rate)
        student_logits = student_model(student_x_l, student_ei_l, student_ea_l)
        labeled_loss = lossFn(student_logits, batch.y)

        # Unlabeled: two independent augmented views for consistency
        _set_model_nNodes(student_model, unlabeled_nNodes)
        _set_model_nNodes(teacher_model, unlabeled_nNodes)

        student_x_u, student_ei_u, student_ea_u = gnn_augment(
            unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.edge_attr,
            noise_std=noise_std, edge_drop_rate=edge_drop_rate,
            feat_mask_rate=feat_mask_rate)
        teacher_x_u, teacher_ei_u, teacher_ea_u = gnn_augment(
            unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.edge_attr,
            noise_std=noise_std, edge_drop_rate=edge_drop_rate,
            feat_mask_rate=feat_mask_rate)

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
    Standard Mean Teacher training (per CuriousAI official implementation)
    with GNN-aware augmentation:
    - Student and Teacher both in train mode (both have dropout noise)
    - Two independent GNN augmentations: feature noise + edge dropout + feature masking
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

        # Create two independent GNN-augmented views (per official MT + GNN augmentation)
        # View 1 (student): feature noise η + edge dropout + feature masking
        # View 2 (teacher): feature noise η' + different edge dropout + different feature masking
        student_x, student_ei, student_ea = gnn_augment(
            batch.x, batch.edge_index, batch.edge_attr,
            noise_std=noise_std, edge_drop_rate=edge_drop_rate,
            feat_mask_rate=feat_mask_rate)
        teacher_x, teacher_ei, teacher_ea = gnn_augment(
            batch.x, batch.edge_index, batch.edge_attr,
            noise_std=noise_std, edge_drop_rate=edge_drop_rate,
            feat_mask_rate=feat_mask_rate)

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

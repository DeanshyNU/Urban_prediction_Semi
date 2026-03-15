"""
Mean Teacher training module
"""
import numpy as np
import torch
import os
from collections import defaultdict
from utils import RMSE
from copy import deepcopy


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
                      unlabeled_nNodes=None):
    """
    Mean Teacher training function

    Args:
        loader: labeled data loader
        unlabeled_loader: unlabeled data loader
        student_model: student model
        teacher_model: teacher model (EMA)
        lossFn: labeled loss function
        consistency_loss_fn: consistency loss function (MSE)
        opt: optimizer
        scheduler: learning rate scheduler
        device: device
        nNodes: labeled data node count
        lambda_U: unlabeled loss weight
        alpha: EMA coefficient
        global_step: current global step
        unlabeled_nNodes: unlabeled data node count (optional)

    Returns:
        total_loss, rmse, truth, pred, epoch_debug
    """
    student_model.train()
    teacher_model.eval()

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

        # Labeled loss
        _set_model_nNodes(student_model, nNodes)
        student_logits = student_model(batch.x, batch.edge_index, batch.edge_attr)
        labeled_loss = lossFn(student_logits, batch.y)

        # Consistency loss
        _set_model_nNodes(student_model, unlabeled_nNodes)
        _set_model_nNodes(teacher_model, unlabeled_nNodes)

        with torch.no_grad():
            teacher_predictions = teacher_model(
                unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.edge_attr
            )

        student_predictions = student_model(
            unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.edge_attr
        )

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

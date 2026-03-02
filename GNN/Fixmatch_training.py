"""
训练相关模块
包含训练、测试和检查点加载功能
"""
import numpy as np
import torch
import os
from utils import RMSE


def loadCheckPoint(modelName, model, opt, device, load=False, resetLr=False, lr=5e-5, predMode=False):
    """
    加载或初始化检查点
    """
    chkptPath = f'./{modelName}.pt'
    if os.path.exists(chkptPath) and load:
        chkpt = torch.load(chkptPath, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        opt.load_state_dict(chkpt['opt_state_dict'])
        EPOCH = chkpt['epoch']
        bestLoss = chkpt['bestLoss']
        hist = chkpt['hist']
        with open(f'./{modelName}_log', 'a') as f: 
            print("Checkpoint loaded.", file=f)
        if opt.param_groups[0]['lr'] < 1e-6 and resetLr:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            with open(f'./{modelName}_log', 'a') as f: 
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
        with open(f'./{modelName}_log', 'w') as f: 
            print("No checkpoint found, starting new model.", file=f)
    return EPOCH, bestLoss, chkptPath, hist


def train_fixmatch_no_uq(trainLoader, weak_loader, strong_loader, model, loss_labeled_fn, loss_unlabeled_fn, opt, scheduler, device, nNodes, lambda_U=10.0):
    """
    标准FixMatch训练函数：1次弱增强生成伪标签（无UQ）
    
    参数:
        trainLoader: 有标签数据加载器
        weak_loader: 弱增强无标签数据加载器
        strong_loader: 强增强无标签数据加载器
        model: GNN模型
        loss_labeled_fn: 有标签损失函数
        loss_unlabeled_fn: 无标签损失函数（reduction='none'）
        opt: 优化器
        scheduler: 学习率调度器
        device: 设备
        nNodes: 节点数
        lambda_U: 无标签损失权重
    """
    model.train()
    total_labeled_loss = 0
    total_unlabeled_loss = 0
    total_batches = 0
    pred, truth = [], []

    for batch_idx, (batch, weak_batch, strong_batch) in enumerate(zip(trainLoader, weak_loader, strong_loader)):
        batch = batch.to(device)
        weak_batch = weak_batch.to(device)
        strong_batch = strong_batch.to(device)
        
        opt.zero_grad(set_to_none=True)
        
        # 有标签损失计算
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labeled_loss = loss_labeled_fn(logits, batch.y)
        
        # 无标签损失计算（标准版本：1次弱增强生成伪标签）
        with torch.no_grad():
            pseudo_labels = model(weak_batch.x, weak_batch.edge_index, weak_batch.edge_attr)
        
        logits_strong = model(strong_batch.x, strong_batch.edge_index, strong_batch.edge_attr)
        unlabeled_loss = loss_unlabeled_fn(logits_strong, pseudo_labels).mean()  # 直接mean，无权重
        
        # 总损失
        total_loss = labeled_loss + lambda_U * unlabeled_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止爆炸
        opt.step()

        # reshape操作
        _pred = logits.reshape(-1, nNodes)
        _truth = batch.y.reshape(-1, nNodes)

        total_labeled_loss += labeled_loss
        total_unlabeled_loss += unlabeled_loss

        pred.append(_pred.cpu().detach().numpy())
        truth.append(_truth.cpu().detach().numpy())
        total_batches += 1

    scheduler.step()

    # 计算平均损失
    avg_labeled_loss = (total_labeled_loss / total_batches).item()
    avg_unlabeled_loss = (total_unlabeled_loss / total_batches).item()

    # 计算RMSE
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss + lambda_U * avg_unlabeled_loss, rmse, truth, pred


def train_fixmatch_multiple_weak(trainLoader, weak_loaders, strong_loader, model, loss_labeled_fn, loss_unlabeled_fn, 
                                  opt, scheduler, device, nNodes, lambda_U=10.0):
    """
    FixMatch训练函数（UQ版本）：多次弱增强生成伪标签，方差加权
    
    参数:
        trainLoader: 有标签数据加载器
        weak_loaders: 多个弱增强无标签数据加载器列表（n_augments个不同的弱增强）
        strong_loader: 强增强无标签数据加载器
        model: GNN模型
        loss_labeled_fn: 有标签损失函数
        loss_unlabeled_fn: 无标签损失函数（reduction='none'）
        opt: 优化器
        scheduler: 学习率调度器
        device: 设备
        nNodes: 节点数
        lambda_U: 无标签损失权重
    """
    model.train()
    total_labeled_loss = 0
    total_unlabeled_loss = 0
    total_batches = 0
    pred, truth = [], []

    # UQ参数
    epsilon = 1e-2  # 从1e-5改为1e-2，防止std≈0时权重爆炸

    # 调试统计
    debug_stats = {
        'labeled_loss': [], 'unlabeled_loss': [], 'total_loss': [],
        'pseudo_label_mean': [], 'pseudo_label_std': [],
        'weight_min': [], 'weight_max': [], 'weight_mean': [],
        'pred_std_min': [], 'pred_std_max': [], 'pred_std_mean': [],
        'logits_strong_mean': [], 'logits_strong_std': [],
        'grad_norm_before_clip': [], 'grad_norm_after_clip': [],
        'point_wise_loss_max': [],
    }

    # 确保weak_loaders是列表
    if not isinstance(weak_loaders, list):
        weak_loaders = [weak_loaders]

    # 同时迭代所有loader（使用zip确保同步）
    for batch_idx, (batch, *weak_batches_list, strong_batch) in enumerate(zip(trainLoader, *weak_loaders, strong_loader)):
        batch = batch.to(device)
        weak_batches = [wb.to(device) for wb in weak_batches_list]
        strong_batch = strong_batch.to(device)

        opt.zero_grad(set_to_none=True)

        # 有标签损失计算
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labeled_loss = loss_labeled_fn(logits, batch.y)

        # 无标签损失计算 - 多次弱增强生成伪标签（UQ版本）
        with torch.no_grad():
            weak_predictions = []
            for weak_batch in weak_batches:
                pred_weak = model(weak_batch.x, weak_batch.edge_index, weak_batch.edge_attr)
                weak_predictions.append(pred_weak)

            # 计算伪标签和方差
            weak_predictions = torch.stack(weak_predictions)
            pseudo_labels = torch.mean(weak_predictions, dim=0)
            pred_std = torch.std(weak_predictions, dim=0)

            # 方差越大权重越小
            weights = 1.0 / (pred_std + epsilon)
            weights = torch.clamp(weights, max=10.0)  # 限制最大权重，防止爆炸
            weights = weights / weights.mean()  # 归一化

        logits_strong = model(strong_batch.x, strong_batch.edge_index, strong_batch.edge_attr)
        # UQ版本：方差加权
        point_wise_loss = loss_unlabeled_fn(logits_strong, pseudo_labels)
        unlabeled_loss = (weights * point_wise_loss).mean()

        # 总损失
        total_loss = labeled_loss + lambda_U * unlabeled_loss
        total_loss.backward()

        # 调试：记录裁剪前梯度范数
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        # 实际裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_after = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5

        opt.step()

        # 收集调试统计
        debug_stats['labeled_loss'].append(labeled_loss.item())
        debug_stats['unlabeled_loss'].append(unlabeled_loss.item())
        debug_stats['total_loss'].append(total_loss.item())
        debug_stats['pseudo_label_mean'].append(pseudo_labels.mean().item())
        debug_stats['pseudo_label_std'].append(pseudo_labels.std().item())
        debug_stats['weight_min'].append(weights.min().item())
        debug_stats['weight_max'].append(weights.max().item())
        debug_stats['weight_mean'].append(weights.mean().item())
        debug_stats['pred_std_min'].append(pred_std.min().item())
        debug_stats['pred_std_max'].append(pred_std.max().item())
        debug_stats['pred_std_mean'].append(pred_std.mean().item())
        debug_stats['logits_strong_mean'].append(logits_strong.mean().item())
        debug_stats['logits_strong_std'].append(logits_strong.std().item())
        debug_stats['grad_norm_before_clip'].append(grad_norm_before.item() if torch.is_tensor(grad_norm_before) else grad_norm_before)
        debug_stats['grad_norm_after_clip'].append(grad_norm_after)
        debug_stats['point_wise_loss_max'].append(point_wise_loss.max().item())

        # reshape操作
        _pred = logits.reshape(-1, nNodes)
        _truth = batch.y.reshape(-1, nNodes)

        total_labeled_loss += labeled_loss
        total_unlabeled_loss += unlabeled_loss

        pred.append(_pred.cpu().detach().numpy())
        truth.append(_truth.cpu().detach().numpy())
        total_batches += 1

    scheduler.step()

    # 计算平均损失
    avg_labeled_loss = (total_labeled_loss / total_batches).item()
    avg_unlabeled_loss = (total_unlabeled_loss / total_batches).item()

    # 汇总调试统计（取每个epoch的平均值）
    epoch_debug = {}
    for key, values in debug_stats.items():
        epoch_debug[f'debug/{key}_mean'] = np.mean(values)
        epoch_debug[f'debug/{key}_max'] = np.max(values)

    # 计算RMSE
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss + lambda_U * avg_unlabeled_loss, rmse, truth, pred, epoch_debug


def test_fixmatch(loader, model, lossFn, device, nNodes):
    """
    评估模型性能（适配ESTNet架构）
    """
    model.eval()
    total_labeled_loss = 0
    pred, truth = [], []

    if len(loader) == 0:
        raise ValueError("数据加载器为空，无法进行测试。")

    with torch.no_grad():
        for batch in loader:
            if batch.x is None or batch.y is None:
                continue

            batch = batch.to(device)
            # 前向传播 - 使用统一的特征
            logits = model(batch.x, batch.edge_index, batch.edge_attr)

            # 确保输出和标签形状兼容
            logits = logits.reshape(-1, nNodes)
            batch_y = batch.y.reshape(-1, nNodes)

            # 计算损失
            loss = lossFn(logits, batch_y)
            total_labeled_loss += loss.item()

            # 存储预测和真实值用于RMSE计算
            pred.append(logits.cpu().numpy())
            truth.append(batch_y.cpu().numpy())

    # 归一化标签损失
    avg_labeled_loss = total_labeled_loss / len(loader)

    # 计算标签数据的RMSE
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss, rmse, truth, pred


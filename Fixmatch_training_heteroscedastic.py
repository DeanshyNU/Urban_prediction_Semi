"""
训练相关模块（Heteroscedastic版本）
包含训练、测试和检查点加载功能
支持不确定性量化：使用 mu 和 log_var 进行 NLL 损失计算
"""
import numpy as np
import torch
import os
from utils import RMSE


def gaussian_nll_loss(mu, log_var, target):
    """
    高斯负对数似然损失（Gaussian Negative Log-Likelihood Loss）
    
    Args:
        mu: 预测均值，形状 (batch_size * nNodes, 1)
        log_var: 预测对数方差，形状 (batch_size * nNodes, 1)
        target: 真实值，形状 (batch_size * nNodes, 1)
    
    Returns:
        loss: NLL损失，形状 (batch_size * nNodes, 1)
    """
    var = torch.exp(log_var)  # 确保方差为正
    precision = 1.0 / (var + 1e-8)  # 精度 = 1/方差
    nll = 0.5 * (log_var + precision * (target - mu) ** 2)
    return nll


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


def train_fixmatch_heteroscedastic(trainLoader, weak_loader, strong_loader, model, opt, scheduler, device, nNodes, lambda_U=10.0, confidence_threshold=0.1):
    """
    FixMatch训练函数（Heteroscedastic版本）：使用方差加权和动态阈值过滤
    
    参数:
        trainLoader: 有标签数据加载器
        weak_loader: 弱增强无标签数据加载器
        strong_loader: 强增强无标签数据加载器
        model: Heteroscedastic GNN模型（返回 mu, log_var）
        opt: 优化器
        scheduler: 学习率调度器
        device: 设备
        nNodes: 节点数
        lambda_U: 无标签损失权重
        confidence_threshold: 置信度阈值（方差阈值），只使用高置信度样本
    """
    model.train()
    total_labeled_loss = 0
    total_unlabeled_loss = 0
    total_batches = 0
    pred, truth = [], []
    epsilon = 1e-8

    for batch_idx, (batch, weak_batch, strong_batch) in enumerate(zip(trainLoader, weak_loader, strong_loader)):
        batch = batch.to(device)
        weak_batch = weak_batch.to(device)
        strong_batch = strong_batch.to(device)
        
        opt.zero_grad(set_to_none=True)
        
        # ========== 有标签损失计算（使用 NLL）==========
        mu_labeled, log_var_labeled = model(batch.x, batch.edge_index, batch.edge_attr)
        # mu_labeled, log_var_labeled: (batch_size * nNodes, 1)
        labeled_loss = gaussian_nll_loss(mu_labeled, log_var_labeled, batch.y).mean()
        
        # ========== 无标签损失计算（使用方差加权和动态阈值）==========
        with torch.no_grad():
            # 弱增强：生成伪标签和不确定性
            mu_weak, log_var_weak = model(weak_batch.x, weak_batch.edge_index, weak_batch.edge_attr)
            var_weak = torch.exp(log_var_weak)  # 方差
            std_weak = torch.sqrt(var_weak + epsilon)  # 标准差
            
            # 动态阈值过滤：只使用高置信度（小方差）的样本
            confidence_mask = (var_weak < confidence_threshold).float()
            
            # 伪标签：使用 mu_weak
            pseudo_labels = mu_weak
        
        # 强增强：预测 mu 和 log_var
        mu_strong, log_var_strong = model(strong_batch.x, strong_batch.edge_index, strong_batch.edge_attr)
        
        # 方差加权：weight = 1 / var_weak（方差越小，权重越大）
        weights = 1.0 / (var_weak + epsilon)
        weights = weights * confidence_mask  # 应用置信度掩码
        weights = weights / (weights.sum() + epsilon) * weights.numel()  # 归一化（保持总权重不变）
        
        # 无标签损失：只关心 mu 的一致性（不涉及 var_strong），var_weak 仅作加权
        consistency_loss = (mu_strong - pseudo_labels) ** 2
        unlabeled_loss = (weights * consistency_loss).mean()
        
        # 总损失
        total_loss = labeled_loss + lambda_U * unlabeled_loss
        total_loss.backward()
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # reshape操作（用于RMSE计算，只用mu）
        _pred = mu_labeled.reshape(-1, nNodes)
        _truth = batch.y.reshape(-1, nNodes)
        
        total_labeled_loss += labeled_loss.item()
        total_unlabeled_loss += unlabeled_loss.item()
        
        pred.append(_pred.cpu().detach().numpy())
        truth.append(_truth.cpu().detach().numpy())
        total_batches += 1

    scheduler.step()

    # 计算平均损失
    avg_labeled_loss = total_labeled_loss / total_batches
    avg_unlabeled_loss = total_unlabeled_loss / total_batches

    # 计算RMSE（只用mu）
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
    epsilon = 1e-5

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
            weights = weights / weights.mean()  # 归一化

        logits_strong = model(strong_batch.x, strong_batch.edge_index, strong_batch.edge_attr)
        # UQ版本：方差加权
        point_wise_loss = loss_unlabeled_fn(logits_strong, pseudo_labels)
        unlabeled_loss = (weights * point_wise_loss).mean()
        
        # 总损失
        total_loss = labeled_loss + lambda_U * unlabeled_loss
        total_loss.backward()
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


def test_fixmatch_heteroscedastic(loader, model, device, nNodes):
    """
    评估模型性能（Heteroscedastic版本）
    只用 mu 做预测，log_var 用于不确定性分析
    """
    model.eval()
    total_labeled_loss = 0
    pred, truth = [], []
    uncertainty = []  # 存储不确定性（方差）

    if len(loader) == 0:
        raise ValueError("数据加载器为空，无法进行测试。")

    with torch.no_grad():
        for batch in loader:
            if batch.x is None or batch.y is None:
                continue

            batch = batch.to(device)
            # 前向传播 - 返回 mu 和 log_var
            mu, log_var = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # 计算方差（用于不确定性分析）
            var = torch.exp(log_var)

            # 确保输出和标签形状兼容
            mu = mu.reshape(-1, nNodes)
            var = var.reshape(-1, nNodes)
            batch_y = batch.y.reshape(-1, nNodes)

            # 计算损失（使用 NLL）
            nll = gaussian_nll_loss(mu.reshape(-1, 1), log_var.reshape(-1, 1), batch_y.reshape(-1, 1))
            total_labeled_loss += nll.mean().item()

            # 存储预测（只用mu）和真实值用于RMSE计算
            pred.append(mu.cpu().numpy())
            truth.append(batch_y.cpu().numpy())
            uncertainty.append(var.cpu().numpy())

    # 归一化标签损失
    avg_labeled_loss = total_labeled_loss / len(loader)

    # 计算标签数据的RMSE（只用mu）
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    uncertainty = np.concatenate(uncertainty)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss, rmse, truth, pred, uncertainty


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


def train_fixmatch(loader, weak_loader, strong_loader, model, lossFn, loss_unlabeled_fn, opt, scheduler, device, nNodes, lambda_U=10.0):
    """
    FixMatch训练函数
    """
    model.train()
    total_labeled_loss = 0
    total_unlabeled_loss = 0
    total_batches = 0
    pred, truth = [], []
    
    # 添加权重计算相关参数
    epsilon = 1e-5        
    temperature = 0.5     
    n_augments = 5        

    for batch_idx, (batch, weak_batch, strong_batch) in enumerate(zip(loader, weak_loader, strong_loader)):
        batch = batch.to(device)
        weak_batch = weak_batch.to(device)
        strong_batch = strong_batch.to(device)
        
        opt.zero_grad(set_to_none=True)
        
        # 有标签损失计算 - 使用统一的特征
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labeled_loss = lossFn(logits, batch.y)
        
        # 无标签损失计算 - 使用分离的静态和动态特征
        with torch.no_grad():
            weak_predictions = []
            for _ in range(n_augments):
                pred_weak = model(weak_batch.x, weak_batch.edge_index, weak_batch.edge_attr)
                weak_predictions.append(pred_weak)
            
            weak_predictions = torch.stack(weak_predictions)
            pseudo_labels = torch.mean(weak_predictions, dim=0)
            pred_std = torch.std(weak_predictions, dim=0)
            
            # V2: 添加方差统计
            variance_stats = {
                'mean_std': pred_std.mean().item(),
                'max_std': pred_std.max().item(),
                'min_std': pred_std.min().item(),
                'std_std': pred_std.std().item()
            }
            
            weights = torch.exp(-pred_std / temperature)
            # 改进的权重计算
            weights = (weights - weights.min()) / (weights.max() - weights.min() + epsilon)

        # V2: 输出方差统计信息
        print(f"方差统计: 均值={variance_stats['mean_std']:.6f}, "
              f"最大={variance_stats['max_std']:.6f}, "
              f"最小={variance_stats['min_std']:.6f}, "
              f"标准差={variance_stats['std_std']:.6f}")

        logits_strong = model(strong_batch.x, strong_batch.edge_index, strong_batch.edge_attr)
        # 直接使用模型输出形状，无需 reshape（参考 Fixmatch_ESTnet_V2.py）
        point_wise_loss = loss_unlabeled_fn(logits_strong, pseudo_labels)  # reduction='none'，返回逐点损失
        unlabeled_loss = (weights * point_wise_loss).mean()  # 加权平均
        
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

        print(f"批次 {batch_idx + 1}: "
              f"标签数据损失 = {labeled_loss.item():.4f}, "
              f"无标签数据损失 = {unlabeled_loss.item():.4f}, "
              f"总损失 = {total_loss.item():.4f}")

    scheduler.step()

    # 计算平均损失
    avg_labeled_loss = (total_labeled_loss / total_batches).item()
    avg_unlabeled_loss = (total_unlabeled_loss / total_batches).item()

    # 计算RMSE
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    # 打印总的统计信息
    print(f"轮次摘要: "
          f"平均标签数据损失 = {avg_labeled_loss:.4f}, "
          f"平均无标签数据损失 = {avg_unlabeled_loss:.4f}, "
          f"RMSE = {rmse[0]:.4f}")

    return avg_labeled_loss + lambda_U * avg_unlabeled_loss, rmse, truth, pred


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


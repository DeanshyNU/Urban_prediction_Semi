"""
Mean Teacher训练模块
实现Mean Teacher半监督学习算法
"""
import numpy as np
import torch
import os
from utils import RMSE
from copy import deepcopy
 
 
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
 
 
def update_ema_variables(model, ema_model, alpha, global_step):
    """
    更新教师模型（EMA）的参数

    参数:
        model: 学生模型
        ema_model: 教师模型（EMA）
        alpha: EMA系数（固定值，如0.999）
        global_step: 全局步数（当前未使用，保留接口）
    """
    # 使用固定alpha（推荐方案A）
    # 原因：动态调整alpha在global_step=0时会导致alpha=0，训练不稳定
    # 固定alpha=0.999可以保持教师模型的稳定性
    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
 
 
def train_meanteacher(loader, unlabeled_loader, student_model, teacher_model, lossFn, consistency_loss_fn,
                      opt, scheduler, device, nNodes, lambda_U=1.0, alpha=0.999, global_step=0):
    """
    Mean Teacher训练函数
 
    参数:
        loader: 有标签数据加载器
        unlabeled_loader: 无标签数据加载器
        student_model: 学生模型
        teacher_model: 教师模型（EMA）
        lossFn: 有标签损失函数
        consistency_loss_fn: 一致性损失函数（MSE）
        opt: 优化器
        scheduler: 学习率调度器
        device: 设备
        nNodes: 节点数
        lambda_U: 无标签损失权重
        alpha: EMA系数
        global_step: 当前全局步数
    """
    student_model.train()
    teacher_model.eval()  # 教师模型只用于推理，生成稳定的伪标签
 
    total_labeled_loss = 0
    total_consistency_loss = 0
    total_batches = 0
    pred, truth = [], []
 
    for batch_idx, (batch, unlabeled_batch) in enumerate(zip(loader, unlabeled_loader)):
        batch = batch.to(device)
        unlabeled_batch = unlabeled_batch.to(device)
 
        opt.zero_grad(set_to_none=True)
 
        # 有标签损失计算（学生模型）
        student_logits = student_model(batch.x, batch.edge_index, batch.edge_attr)
        labeled_loss = lossFn(student_logits, batch.y)

        # 一致性损失计算：学生模型学习教师模型的预测
        with torch.no_grad():
            # 教师模型为无标签数据生成伪标签
            teacher_predictions = teacher_model(
                unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.edge_attr
            )

        # 学生模型对无标签数据的预测
        student_predictions = student_model(
            unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.edge_attr
        )
 
        # 计算一致性损失
        consistency_loss = consistency_loss_fn(student_predictions, teacher_predictions)
 
        # 总损失
        total_loss = labeled_loss + lambda_U * consistency_loss
        total_loss.backward()
 
        opt.step()
 
        # 更新教师模型（EMA）
        update_ema_variables(student_model, teacher_model, alpha, global_step + batch_idx)
 
        # reshape操作
        _pred = student_logits.reshape(-1, nNodes)
        _truth = batch.y.reshape(-1, nNodes)
 
        total_labeled_loss += labeled_loss.item()
        total_consistency_loss += consistency_loss.item()
 
        pred.append(_pred.cpu().detach().numpy())
        truth.append(_truth.cpu().detach().numpy())
        total_batches += 1
 
        if batch_idx % 10 == 0:
            print(f"批次 {batch_idx + 1}: "
                  f"标签损失 = {labeled_loss.item():.4f}, "
                  f"一致性损失 = {consistency_loss.item():.4f}, "
                  f"总损失 = {total_loss.item():.4f}")
 
    scheduler.step()
 
    # 计算平均损失
    avg_labeled_loss = total_labeled_loss / total_batches
    avg_consistency_loss = total_consistency_loss / total_batches
 
    # 计算RMSE
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)
 
    # 打印总的统计信息
    print(f"轮次摘要: "
          f"平均标签损失 = {avg_labeled_loss:.4f}, "
          f"平均一致性损失 = {avg_consistency_loss:.4f}, "
          f"RMSE = {rmse[0]:.4f}")
 
    return avg_labeled_loss + lambda_U * avg_consistency_loss, rmse, truth, pred
 
 
def test_meanteacher(loader, model, lossFn, device, nNodes):
    """
    Mean Teacher测试函数（使用教师模型）
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
 
    # 计算RMSE
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)
 
    return avg_labeled_loss, rmse, truth, pred
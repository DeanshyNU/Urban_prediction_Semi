"""
Π-Model训练模块
实现Π-Model半监督学习算法
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
 
 
def train_pimodel(loader, aug1_loader, aug2_loader, model, lossFn, consistency_loss_fn, opt, scheduler, device, nNodes, lambda_U=1.0):
    """
    Π-Model训练函数
 
    参数:
        loader: 有标签数据加载器
        aug1_loader: 无标签数据的第一次增强
        aug2_loader: 无标签数据的第二次增强
        model: GNN模型
        lossFn: 有标签损失函数
        consistency_loss_fn: 一致性损失函数（MSE）
        opt: 优化器
        scheduler: 学习率调度器
        device: 设备
        nNodes: 节点数
        lambda_U: 无标签损失权重
    """
    model.train()
    total_labeled_loss = 0
    total_consistency_loss = 0
    total_batches = 0
    pred, truth = [], []
 
    for batch_idx, (batch, aug1_batch, aug2_batch) in enumerate(zip(loader, aug1_loader, aug2_loader)):
        batch = batch.to(device)
        aug1_batch = aug1_batch.to(device)
        aug2_batch = aug2_batch.to(device)
 
        opt.zero_grad(set_to_none=True)
 
        # 有标签损失计算
        logits = model(batch.x_dynamic, batch.x_static, batch.edge_index, batch.edge_attr)
        labeled_loss = lossFn(logits, batch.y)
 
        # 一致性损失计算：对同一无标签数据的两次不同增强预测应该一致
        pred1 = model(aug1_batch.x_dynamic, aug1_batch.x_static, aug1_batch.edge_index, aug1_batch.edge_attr)
        pred2 = model(aug2_batch.x_dynamic, aug2_batch.x_static, aug2_batch.edge_index, aug2_batch.edge_attr)
        consistency_loss = consistency_loss_fn(pred1, pred2)
 
        # 总损失
        total_loss = labeled_loss + lambda_U * consistency_loss
        total_loss.backward()
 
        opt.step()
 
        # reshape操作
        _pred = logits.reshape(-1, nNodes)
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
 
 
def test_pimodel(loader, model, lossFn, device, nNodes):
    """
    Π-Model测试函数
    """
    model.eval()
    total_labeled_loss = 0
    pred, truth = [], []
 
    if len(loader) == 0:
        raise ValueError("数据加载器为空，无法进行测试。")
 
    with torch.no_grad():
        for batch in loader:
            if batch.x_dynamic is None or batch.y is None:
                continue
 
            batch = batch.to(device)
            # 前向传播
            logits = model(batch.x_dynamic, batch.x_static, batch.edge_index, batch.edge_attr)
 
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
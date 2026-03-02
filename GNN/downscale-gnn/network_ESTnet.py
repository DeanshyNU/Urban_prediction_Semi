"""
监督学习网络模块（ESTnet 版本）
使用 Fixmatch_GNN/models_ESTnet.py 中的 GNN_ESTNet 架构
数据通过 batch.x 传入，在 train/test 中按 dynamic_dim/static_dim 拆分为 x_dynamic, x_static
"""
import sys
import os
import torch
import numpy as np
import utils

# 允许从上级目录导入 models_ESTnet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_ESTnet import GNN_ESTNet


def train(loader, model, lossFn, opt, scheduler, device, nNodes, dynamic_dim, static_dim):
    model.train()
    _LOSS = 0
    pred, truth = [], []
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        x_dynamic = _batch.x[:, :dynamic_dim]
        x_static = _batch.x[:, dynamic_dim:]
        _yHat = model(x_dynamic, x_static, _batch.edge_index, _batch.edge_attr)
        _loss = lossFn(_yHat, _batch.y)
        _loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += _loss
        _pred = _yHat.reshape(-1, nNodes)
        _truth = _batch.y.reshape(-1, nNodes)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    return (_LOSS / (_n + 1)).item(), _RMSE, truth, pred


def test(loader, model, lossFn, device, nNodes, dynamic_dim, static_dim):
    model.eval()
    _LOSS = 0
    pred, truth = [], []
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        x_dynamic = _batch.x[:, :dynamic_dim]
        x_static = _batch.x[:, dynamic_dim:]
        _yHat = model(x_dynamic, x_static, _batch.edge_index, _batch.edge_attr)
        _loss = lossFn(_yHat, _batch.y)
        _LOSS += _loss
        _pred = _yHat.reshape(-1, nNodes)
        _truth = _batch.y.reshape(-1, nNodes)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    return (_LOSS / (_n + 1)).item(), _RMSE, truth, pred


def loadCheckPoint(modelName, model, opt, device, load=False, resetLr=False, lr=5e-5, predMode=False, output_dir='./'):
    chkptPath = f'{output_dir}/{modelName}.pt'
    if os.path.exists(chkptPath) and load:
        chkpt = torch.load(chkptPath, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        opt.load_state_dict(chkpt['opt_state_dict'])
        EPOCH = chkpt['epoch']
        bestLoss = chkpt['bestLoss']
        hist = chkpt.get('hist', [])
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print("Checkpoint loaded.", file=f)
        if opt.param_groups[0]['lr'] < 1e-6 and resetLr:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
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
        with open(f'{output_dir}/{modelName}_log', 'w') as f:
            print("No checkpoint found, starting new model.", file=f)
    return EPOCH, bestLoss, chkptPath, hist

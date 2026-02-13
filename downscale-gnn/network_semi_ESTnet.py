"""
半监督学习网络模块（ESTnet 版本）
使用 Fixmatch_GNN/models_ESTnet.py 中的 GNN_ESTNet 架构
数据通过 batch.x 传入，在 train/test 中按 dynamic_dim/static_dim 拆分为 x_dynamic, x_static
"""
import sys
import os
import torch
import numpy as np
import utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_ESTnet import GNN_ESTNet

# 全局标志，确保半监督学习验证只打印一次
_verification_printed = False


def train(loader, model, lossFn, opt, scheduler, device, nNodes, nNodes_labeled, dynamic_dim, static_dim):
    """
    半监督训练：只在有标签节点上计算监督损失
    """
    global _verification_printed
    model.train()
    _LOSS = 0
    pred, truth = [], []
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        x_dynamic = _batch.x[:, :dynamic_dim]
        x_static = _batch.x[:, dynamic_dim:]
        _yHat = model(x_dynamic, x_static, _batch.edge_index, _batch.edge_attr)

        # 只在有标签节点上计算损失
        label_mask = _batch.label_mask  # (batch_size * nNodes,)
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]

        # ========== 半监督学习验证检查（仅第一个batch的第一个epoch）==========
        if not _verification_printed and _n == 0:
            nNodes_unlabeled = nNodes - nNodes_labeled
            nNodes_total_batch = _batch.x.shape[0]
            nNodes_labeled_batch = label_mask.sum().item()
            nNodes_unlabeled_batch = nNodes_total_batch - nNodes_labeled_batch

            batch_size = nNodes_total_batch // nNodes
            nNodes_per_graph = nNodes_total_batch // batch_size if batch_size > 0 else nNodes_total_batch
            nNodes_labeled_per_graph = nNodes_labeled_batch // batch_size if batch_size > 0 else nNodes_labeled_batch

            assert nNodes_total_batch % nNodes == 0, \
                f"节点数不匹配: batch中有{nNodes_total_batch}个节点，不是{nNodes}的整数倍"
            assert nNodes_per_graph == nNodes, \
                f"每个图的节点数不匹配: {nNodes_per_graph} vs {nNodes}"
            assert nNodes_labeled_per_graph == nNodes_labeled, \
                f"每个图的有标签节点数不匹配: {nNodes_labeled_per_graph} vs {nNodes_labeled}"
            assert _yHat.shape[0] == nNodes_total_batch, \
                f"预测输出节点数不匹配: {_yHat.shape[0]} vs {nNodes_total_batch}"
            assert _yHat_labeled.shape[0] == nNodes_labeled_batch, \
                f"损失计算节点数不匹配: {_yHat_labeled.shape[0]} vs {nNodes_labeled_batch}"

            edge_index = _batch.edge_index.cpu().numpy()
            labeled_nodes = set(range(nNodes_labeled))
            unlabeled_nodes = set(range(nNodes_labeled, nNodes))
            edges_labeled_to_unlabeled = 0
            edges_unlabeled_to_labeled = 0
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src < nNodes and dst < nNodes:
                    if src in labeled_nodes and dst in unlabeled_nodes:
                        edges_labeled_to_unlabeled += 1
                    elif src in unlabeled_nodes and dst in labeled_nodes:
                        edges_unlabeled_to_labeled += 1
            total_cross_edges = edges_labeled_to_unlabeled + edges_unlabeled_to_labeled

            print("\n" + "="*70)
            print("半监督学习验证检查（训练时）- ESTnet")
            print("="*70)
            print(f"✓ Batch信息: batch_size={batch_size}, 每个图{nNodes}个节点")
            print(f"✓ 所有节点参与前向传播: batch中总计{nNodes_total_batch}个节点（每个图: {nNodes_labeled}有标签 + {nNodes_unlabeled}无标签）")
            print(f"✓ 所有节点都有预测输出: {_yHat.shape[0]}个预测值")
            print(f"✓ 损失只计算在有标签节点: batch中总计{nNodes_labeled_batch}个节点用于损失计算（每个图{nNodes_labeled}个）")
            print(f"✓ 图边连接情况（第一个图）: 有标签↔无标签交叉边数: {total_cross_edges}")
            _yHat_unlabeled = _yHat[~label_mask]
            print(f"✓ 无标签节点预测值统计: 数量={_yHat_unlabeled.shape[0]}, 均值={_yHat_unlabeled.mean().item():.4f}")
            print("="*70 + "\n")
            _verification_printed = True

        _loss = lossFn(_yHat_labeled, _y_labeled)
        _loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += _loss

        _pred = _yHat_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        _truth = _y_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    return (_LOSS/(_n+1)).item(), _RMSE, truth, pred


def test(loader, model, lossFn, device, nNodes, nNodes_labeled, dynamic_dim, static_dim):
    """
    测试：只在有标签节点上计算损失和RMSE
    """
    model.eval()
    _LOSS = 0
    pred, truth = [], []
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        x_dynamic = _batch.x[:, :dynamic_dim]
        x_static = _batch.x[:, dynamic_dim:]
        _yHat = model(x_dynamic, x_static, _batch.edge_index, _batch.edge_attr)

        label_mask = _batch.label_mask
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]

        _loss = lossFn(_yHat_labeled, _y_labeled)
        _LOSS += _loss

        _pred = _yHat_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        _truth = _y_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    return (_LOSS/(_n+1)).item(), _RMSE, truth, pred


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
                print(f"Resetting LR to {lr}", file=f)
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

import torch,os,utils
import numpy as np
from torch_geometric.nn import GraphConv, SAGEConv, APPNP
# -----------------Construct GNN model---------------------------
class GNN(torch.nn.Module):

    def __init__(self,modelPara):
        super(GNN, self).__init__()
        self.nGNNLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        self.conv_type = modelPara.get('conv_type', 'graphconv').lower()
        _HLD = modelPara['HLD']
        _encoder, _processor, _decoder = [],[],[]

        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _outputChannel = _HLD
            _encoder.append(torch.nn.Linear(_inputChannel,_outputChannel))
            _encoder.append(torch.nn.PReLU(_outputChannel))

        for _n in range(self.nGNNLayers):
            _inputChannel = _HLD
            _outputChannel = _HLD
            if self.conv_type == 'sageconv':
                _processor.append(SAGEConv(_inputChannel, _outputChannel, aggr='mean'))
            elif self.conv_type == 'appnp':
                if _n == 0:
                    _processor.append(torch.nn.Linear(_inputChannel, _outputChannel))
                    _processor.append(torch.nn.PReLU(_outputChannel))
                    _processor.append(APPNP(K=10, alpha=0.1))
                continue
            else:  # graphconv (default)
                _processor.append(GraphConv(_inputChannel, _outputChannel, aggr='mean'))
            _processor.append(torch.nn.PReLU(_outputChannel))

        for _n in range(self.nMLPLayers):
            _inputChannel = _HLD
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers-1 else _HLD
            _decoder.append(torch.nn.Linear(_inputChannel,_outputChannel))
            _decoder.append(torch.nn.PReLU(_outputChannel))

        self.encoder = torch.nn.ModuleList(_encoder)
        self.processor = torch.nn.ModuleList(_processor)
        self.decoder = torch.nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx, edgeAttr):
        for _f in self.encoder:
            x = _f(x)
        for _n, _f in enumerate(self.processor):
            if isinstance(_f, (GraphConv,)):
                x = _f(x, edgeIdx, edgeAttr)
            elif isinstance(_f, (SAGEConv,)):
                x = _f(x, edgeIdx)
            elif isinstance(_f, (APPNP,)):
                x = _f(x, edgeIdx, edgeAttr)
            else:
                x = _f(x)
        for _f in self.decoder:
            x = _f(x)
        return x
    
# 全局标志，确保半监督学习验证只打印一次
_verification_printed = False

def train(loader,model,lossFn,opt,scheduler,device,nNodes,nNodes_labeled):
    """
    半监督训练：只在有标签节点上计算监督损失
    """
    global _verification_printed
    model.train()
    _LOSS = 0
    pred,truth = [],[]
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        
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
            
            # 计算batch_size（当batch_size > 1时，多个图被打包）
            batch_size = nNodes_total_batch // nNodes
            nNodes_per_graph = nNodes_total_batch // batch_size if batch_size > 0 else nNodes_total_batch
            nNodes_labeled_per_graph = nNodes_labeled_batch // batch_size if batch_size > 0 else nNodes_labeled_batch
            
            # 检查1: 确认batch中的节点数是nNodes的整数倍（说明包含多个图）
            assert nNodes_total_batch % nNodes == 0, \
                f"节点数不匹配: batch中有{nNodes_total_batch}个节点，不是{nNodes}的整数倍"
            
            # 检查2: 确认每个图的节点数量正确
            assert nNodes_per_graph == nNodes, \
                f"每个图的节点数不匹配: {nNodes_per_graph} vs {nNodes}"
            # Note: spatial模式下nNodes_labeled_per_graph可能是50(train)或8(valid)，不一定等于nNodes_labeled(58)
            # assert nNodes_labeled_per_graph == nNodes_labeled
            
            # 检查3: 确认所有节点都有预测输出
            assert _yHat.shape[0] == nNodes_total_batch, \
                f"预测输出节点数不匹配: {_yHat.shape[0]} vs {nNodes_total_batch}"
            
            # 检查4: 确认损失只计算在有标签节点上
            assert _yHat_labeled.shape[0] == nNodes_labeled_batch, \
                f"损失计算节点数不匹配: {_yHat_labeled.shape[0]} vs {nNodes_labeled_batch}"
            
            # 检查5: 检查有标签节点和无标签节点之间是否有边连接（关键：消息传递）
            edge_index = _batch.edge_index.cpu().numpy()
            # 在PyG中，batch中的节点索引是连续的，每个图的节点索引会偏移
            # 对于batch中的第一个图（batch_size可能>1），节点索引范围是[0, nNodes-1]
            # 检查是否有边连接有标签和无标签节点
            labeled_nodes = set(range(nNodes_labeled))
            unlabeled_nodes = set(range(nNodes_labeled, nNodes))
            edges_labeled_to_unlabeled = 0
            edges_unlabeled_to_labeled = 0
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                # 只检查第一个图内的边（节点索引 < nNodes）
                if src < nNodes and dst < nNodes:
                    if src in labeled_nodes and dst in unlabeled_nodes:
                        edges_labeled_to_unlabeled += 1
                    elif src in unlabeled_nodes and dst in labeled_nodes:
                        edges_unlabeled_to_labeled += 1
            total_cross_edges = edges_labeled_to_unlabeled + edges_unlabeled_to_labeled
            
            print("\n" + "="*70)
            print("半监督学习验证检查（训练时）")
            print("="*70)
            print(f"✓ Batch信息: batch_size={batch_size}, 每个图{nNodes}个节点")
            print(f"✓ 所有节点参与前向传播: batch中总计{nNodes_total_batch}个节点（每个图: {nNodes_labeled}有标签 + {nNodes_unlabeled}无标签）")
            print(f"✓ 所有节点都有预测输出: {_yHat.shape[0]}个预测值")
            print(f"✓ 损失只计算在有标签节点: batch中总计{nNodes_labeled_batch}个节点用于损失计算（每个图{nNodes_labeled}个）")
            print(f"✓ 图边连接情况（第一个图）:")
            print(f"   总边数: {edge_index.shape[1]}")
            print(f"   有标签↔无标签交叉边数: {total_cross_edges}")
            if total_cross_edges > 0:
                print(f"   → 无标签节点通过图结构参与消息传递 ✓")
            else:
                print(f"   ⚠️  警告：未检测到有标签和无标签节点之间的边连接")
            print(f"✓ 无标签节点预测值统计:")
            _yHat_unlabeled = _yHat[~label_mask]
            print(f"   数量: {_yHat_unlabeled.shape[0]}")
            print(f"   均值: {_yHat_unlabeled.mean().item():.4f}")
            print(f"   标准差: {_yHat_unlabeled.std().item():.4f}")
            print(f"   范围: [{_yHat_unlabeled.min().item():.4f}, {_yHat_unlabeled.max().item():.4f}]")
            
            # 边权重统计
            edge_attr = _batch.edge_attr.cpu().numpy()
            if len(edge_attr) > 0:
                print(f"✓ 边权重统计:")
                print(f"   数量: {len(edge_attr)}")
                print(f"   最小值: {np.min(edge_attr):.6f}")
                print(f"   最大值: {np.max(edge_attr):.6f}")
                print(f"   均值: {np.mean(edge_attr):.6f}")
                print(f"   标准差: {np.std(edge_attr):.6f}")
            print("="*70 + "\n")
            _verification_printed = True
        
        _loss = lossFn(_yHat_labeled, _y_labeled)
        _loss.backward(retain_graph=False)
        # ✅ 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += _loss
        
        # 只记录有标签节点的预测
        # spatial模式下每个图的labeled数可能不等于nNodes_labeled(58)
        _n_labeled_actual = label_mask.sum().item() // max(1, _batch.x.shape[0] // nNodes)
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth,pred)
    return (_LOSS/(_n+1)).item(), _RMSE, truth, pred

def test(loader,model,lossFn,device,nNodes,nNodes_labeled):
    """
    测试：只在有标签节点上计算损失和RMSE
    """
    model.eval()
    _LOSS = 0
    pred,truth = [],[]
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        
        # 只在有标签节点上计算损失
        label_mask = _batch.label_mask
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]
        
        _loss = lossFn(_yHat_labeled, _y_labeled)
        _LOSS += _loss
        
        # 只记录有标签节点的预测
        # spatial模式下每个图的labeled数可能不等于nNodes_labeled(58)
        _n_labeled_actual = label_mask.sum().item() // max(1, _batch.x.shape[0] // nNodes)
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth,pred)
    return (_LOSS/(_n+1)).item(), _RMSE, truth, pred

def loadCheckPoint(modelName,model,opt,device,load=False,resetLr=False,lr=5e-5,predMode=False,output_dir='./'):
    chkptPath = f'{output_dir}/{modelName}.pt'
    if os.path.exists(chkptPath) and load:
        chkpt = torch.load(chkptPath,map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        opt.load_state_dict(chkpt['opt_state_dict'])
        EPOCH = chkpt['epoch']
        bestLoss = chkpt['bestLoss']
        hist = chkpt['hist']
        with open(f'{output_dir}/{modelName}_log','a') as f: print("Checkpoint loaded.")
        if opt.param_groups[0]['lr'] < 1e-6 and resetLr:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            with open(f'{output_dir}/{modelName}_log','a') as f: print(f"Resetting LR from {opt.param_groups[0]['lr']} to {lr}",file=f)
    elif predMode:
        if not os.path.exists(chkptPath): chkptPath = f'./trainedModels/{modelName}.pt'
        chkpt = torch.load(chkptPath,map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        print("Checkpoint loaded.")
        return -1
    else:
        EPOCH = 0
        bestLoss = np.inf
        hist = []
        with open(f'{output_dir}/{modelName}_log','w') as f: print("No checkpoint found, starting new model.",file=f)
    return EPOCH,bestLoss,chkptPath,hist

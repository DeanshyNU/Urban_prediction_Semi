import torch,os,utils
import numpy as np
from torch_geometric.nn import GraphConv, SAGEConv, APPNP
# -----------------Edge Weight Network for Adaptive Laplacian---------------------------
class EdgeWeightNet(torch.nn.Module):
    """
    Learns per-edge smoothness weights for Adaptive Laplacian.
    Input: concatenation of node pair hidden representations [h_i, h_j]
    Output: scalar weight α_ij ∈ (0, 1) via sigmoid
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.PReLU(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, h_i, h_j):
        """h_i, h_j: (num_edges, hidden_dim) → (num_edges, 1)"""
        return self.net(torch.cat([h_i, h_j], dim=-1))

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
            if _n < self.nMLPLayers - 1:  # [BugFix] 最后一层不加激活函数（回归输出应无约束）
                _decoder.append(torch.nn.PReLU(_outputChannel))

        self.encoder = torch.nn.ModuleList(_encoder)
        self.processor = torch.nn.ModuleList(_processor)
        self.decoder = torch.nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx, edgeAttr, return_hidden=False):
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
        h = x  # hidden representation after GNN, before decoder
        for _f in self.decoder:
            x = _f(x)
        if return_hidden:
            return x, h
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

        # [Debug] 首个batch检查预测和目标范围
        if _n == 0 and not _verification_printed:
            with torch.no_grad():
                _p_min, _p_max = _yHat.min().item(), _yHat.max().item()
                _t_min, _t_max = _batch.y[label_mask].min().item(), _batch.y[label_mask].max().item()
                print(f"  [Debug] Epoch start: pred range=[{_p_min:.4f}, {_p_max:.4f}], "
                      f"target range=[{_t_min:.4f}, {_t_max:.4f}], loss={_loss.item():.4e}")

        _loss.backward(retain_graph=False)
        # ✅ 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += _loss.item()  # [BugFix] 用.item()提取标量，避免GPU内存泄漏

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
    return (_LOSS/(_n+1)), _RMSE, truth, pred

def test(loader,model,lossFn,device,nNodes,nNodes_labeled):
    """
    测试：只在有标签节点上计算损失和RMSE
    """
    model.eval()
    _LOSS = 0
    pred,truth = [],[]
    with torch.no_grad():  # [BugFix] 添加no_grad，避免验证时计算梯度浪费GPU内存
        for _n, _batch in enumerate(loader):
            _batch = _batch.to(device)
            _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)

            # 只在有标签节点上计算损失
            label_mask = _batch.label_mask
            _yHat_labeled = _yHat[label_mask]
            _y_labeled = _batch.y[label_mask]

            _loss = lossFn(_yHat_labeled, _y_labeled)
            _LOSS += _loss.item()  # [BugFix] 用.item()提取标量

            # 只记录有标签节点的预测
            _n_labeled_actual = label_mask.sum().item() // max(1, _batch.x.shape[0] // nNodes)
            _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
            _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
            pred += list(_pred.cpu().numpy())
            truth += list(_truth.cpu().numpy())
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth,pred)
    return (_LOSS/(_n+1)), _RMSE, truth, pred


# ==================== Enhanced Semi-supervised Training Modes ====================

def train_laplacian(loader, model, lossFn, opt, scheduler, device, nNodes, nNodes_labeled,
                    adj_matrix, lambda_lap=0.1):
    """
    Semi-supervised + Graph Laplacian regularization.
    L = L_supervised + lambda_lap * Σ w_ij * (pred_i - pred_j)²
    Gives unlabeled nodes a direct training signal (spatial smoothness).
    """
    model.train()
    _LOSS = 0
    _LAP_LOSS = 0
    pred, truth = [], []

    # Pre-compute edge list from adjacency (once per epoch)
    _adj = adj_matrix.copy()
    np.fill_diagonal(_adj, 0)
    _edge_src, _edge_dst = np.nonzero(_adj)
    _edge_w = torch.FloatTensor(_adj[_edge_src, _edge_dst]).to(device)
    _edge_src = torch.LongTensor(_edge_src).to(device)
    _edge_dst = torch.LongTensor(_edge_dst).to(device)

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)

        label_mask = _batch.label_mask
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]

        # Supervised loss
        sup_loss = lossFn(_yHat_labeled, _y_labeled)

        # [BugFix] Graph Laplacian loss on ALL graphs in batch (not just first)
        batch_size = _batch.x.shape[0] // nNodes
        _yHat_all = _yHat.squeeze(-1).reshape(batch_size, nNodes)  # (batch_size, nNodes)
        lap_loss = 0
        for g in range(batch_size):
            diff = _yHat_all[g][_edge_src] - _yHat_all[g][_edge_dst]
            lap_loss = lap_loss + torch.mean(_edge_w * diff ** 2)
        lap_loss = lap_loss / batch_size

        total_loss = sup_loss + lambda_lap * lap_loss

        # [Debug] Shape and batch size verification (first batch only)
        if _n == 0:
            assert _batch.x.shape[0] % nNodes == 0, \
                f"Batch nodes {_batch.x.shape[0]} not divisible by nNodes {nNodes}"
            assert _yHat_all.shape == (batch_size, nNodes), \
                f"yHat reshape failed: {_yHat_all.shape} vs expected ({batch_size}, {nNodes})"

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _LAP_LOSS += lap_loss.item()

        _n_labeled_actual = label_mask.sum().item() // max(1, batch_size)
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1
    return (_LOSS / n_batches), _RMSE, truth, pred, _LAP_LOSS / n_batches


def train_adaptive_laplacian(loader, model, edge_weight_net, lossFn, opt, scheduler, device,
                             nNodes, nNodes_labeled, adj_matrix, lambda_lap=0.1):
    """
    Adaptive Laplacian: learns per-edge smoothness weights.
    L = L_supervised + lambda_lap * Σ α_ij * w_ij * (pred_i - pred_j)²
    where α_ij = EdgeWeightNet(h_i, h_j) ∈ (0,1) decides which edges should be smooth.

    This allows the model to suppress smoothness at urban/rural boundaries
    while enforcing it in homogeneous areas.
    """
    model.train()
    edge_weight_net.train()
    _LOSS = 0
    _LAP_LOSS = 0
    _ALPHA_MEAN = 0
    _ALPHA_STD = 0
    pred, truth = [], []

    # Pre-compute edge list from adjacency (once per epoch)
    _adj = adj_matrix.copy()
    np.fill_diagonal(_adj, 0)
    _edge_src, _edge_dst = np.nonzero(_adj)
    _edge_w = torch.FloatTensor(_adj[_edge_src, _edge_dst]).to(device)
    _edge_src = torch.LongTensor(_edge_src).to(device)
    _edge_dst = torch.LongTensor(_edge_dst).to(device)

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat, _hidden = model(_batch.x, _batch.edge_index, _batch.edge_attr, return_hidden=True)

        label_mask = _batch.label_mask
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]

        # Supervised loss
        sup_loss = lossFn(_yHat_labeled, _y_labeled)

        # [BugFix] Adaptive Laplacian loss on ALL graphs in batch
        batch_size = _batch.x.shape[0] // nNodes
        _yHat_all = _yHat.squeeze(-1).reshape(batch_size, nNodes)
        _h_all = _hidden.reshape(batch_size, nNodes, -1)

        lap_loss = 0
        alpha_sum = None
        for g in range(batch_size):
            h_src = _h_all[g][_edge_src]
            h_dst = _h_all[g][_edge_dst]
            alpha_g = edge_weight_net(h_src, h_dst).squeeze(-1)
            diff = _yHat_all[g][_edge_src] - _yHat_all[g][_edge_dst]
            lap_loss = lap_loss + torch.mean(alpha_g * _edge_w * diff ** 2)
            alpha_sum = alpha_g if alpha_sum is None else alpha_sum + alpha_g
        lap_loss = lap_loss / batch_size
        alpha = alpha_sum / batch_size  # average alpha for logging

        # Entropy regularization: encourage α to be decisive (not all ~0.5)
        entropy_reg = -torch.mean(alpha * torch.log(alpha + 1e-8) +
                                  (1 - alpha) * torch.log(1 - alpha + 1e-8))

        total_loss = sup_loss + lambda_lap * lap_loss + 0.01 * entropy_reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(edge_weight_net.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _LAP_LOSS += lap_loss.item()
        _ALPHA_MEAN += alpha.mean().item()
        _ALPHA_STD += alpha.std().item()

        _n_labeled_actual = label_mask.sum().item() // max(1, batch_size)
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1
    stats = {
        'lap_loss': _LAP_LOSS / n_batches,
        'alpha_mean': _ALPHA_MEAN / n_batches,
        'alpha_std': _ALPHA_STD / n_batches,
    }
    return (_LOSS / n_batches), _RMSE, truth, pred, stats


def train_residual_laplacian(loader, model, lossFn, opt, scheduler, device, nNodes, nNodes_labeled,
                             adj_matrix, lambda_lap=0.1):
    """
    Residual Laplacian regularization.
    L = L_supervised + lambda_lap * Σ w_ij * (residual_i - residual_j)²
    where residual = pred - WRF_T2 (the model's correction to WRF).

    Physical motivation: the urban heat island effect (correction to WRF background)
    should vary smoothly in space. Unlike standard Laplacian which forces absolute
    temperature predictions to be similar, this only smooths the UHI correction,
    allowing different WRF backgrounds for neighboring nodes.
    """
    model.train()
    _LOSS = 0
    _LAP_LOSS = 0
    _RESIDUAL_STATS = {'mean': 0, 'std': 0}
    pred, truth = [], []

    # Pre-compute edge list from adjacency (once per epoch)
    _adj = adj_matrix.copy()
    np.fill_diagonal(_adj, 0)
    _edge_src, _edge_dst = np.nonzero(_adj)
    _edge_w = torch.FloatTensor(_adj[_edge_src, _edge_dst]).to(device)
    _edge_src = torch.LongTensor(_edge_src).to(device)
    _edge_dst = torch.LongTensor(_edge_dst).to(device)

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)

        label_mask = _batch.label_mask
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]

        # Supervised loss
        sup_loss = lossFn(_yHat_labeled, _y_labeled)

        # [BugFix] Residual Laplacian loss on ALL graphs in batch
        batch_size = _batch.x.shape[0] // nNodes
        _yHat_all = _yHat.squeeze(-1).reshape(batch_size, nNodes)
        _wrf_t2_all = _batch.wrf_t2.squeeze(-1).reshape(batch_size, nNodes)

        lap_loss = 0
        _res_mean_sum = 0
        _res_std_sum = 0
        for g in range(batch_size):
            residual_g = _yHat_all[g] - _wrf_t2_all[g]
            diff = residual_g[_edge_src] - residual_g[_edge_dst]
            lap_loss = lap_loss + torch.mean(_edge_w * diff ** 2)
            with torch.no_grad():
                _res_mean_sum += residual_g.mean().item()
                _res_std_sum += residual_g.std().item()
        lap_loss = lap_loss / batch_size

        total_loss = sup_loss + lambda_lap * lap_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _LAP_LOSS += lap_loss.item()
        _RESIDUAL_STATS['mean'] += _res_mean_sum / batch_size
        _RESIDUAL_STATS['std'] += _res_std_sum / batch_size

        _n_labeled_actual = label_mask.sum().item() // max(1, batch_size)
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1
    stats = {
        'lap_loss': _LAP_LOSS / n_batches,
        'residual_mean': _RESIDUAL_STATS['mean'] / n_batches,
        'residual_std': _RESIDUAL_STATS['std'] / n_batches,
    }
    return (_LOSS / n_batches), _RMSE, truth, pred, stats


def train_msg_weight(loader, model, lossFn, opt, scheduler, device, nNodes, nNodes_labeled,
                     unlabeled_discount=0.5):
    """
    Semi-supervised with message weighting: labeled neighbors full weight,
    unlabeled neighbors discounted.
    Modifies edge_attr before forward pass.
    """
    model.train()
    _LOSS = 0
    pred, truth = [], []

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        label_mask = _batch.label_mask

        # Modify edge weights: discount edges FROM unlabeled nodes (vectorized)
        edge_attr_mod = _batch.edge_attr.clone()
        src_nodes = _batch.edge_index[0]
        src_is_unlabeled = ~label_mask[src_nodes]  # bool mask: True if source is unlabeled
        edge_attr_mod[src_is_unlabeled] *= unlabeled_discount

        _yHat = model(_batch.x, _batch.edge_index, edge_attr_mod)

        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]
        _loss = lossFn(_yHat_labeled, _y_labeled)

        _loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += _loss.item()

        batch_size = _batch.x.shape[0] // nNodes
        _n_labeled_actual = label_mask.sum().item() // max(1, batch_size)
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        _truth = _y_labeled.squeeze(-1).reshape(-1, _n_labeled_actual)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    return (_LOSS / (_n + 1)), _RMSE, truth, pred

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

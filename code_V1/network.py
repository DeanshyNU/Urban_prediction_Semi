import torch,os,utils
import numpy as np
from torch_geometric.nn import GraphConv, SAGEConv, GATConv


# -----------------Per-modality encoder----------------------------
# V2 schema feature 拆 5 个模态分别编码 → fusion(避免 GeoEmbed 1008 维 dominate)
class ModalEncoder(torch.nn.Module):
    """V2 schema 的 per-modality encoder。

    Feature 顺序(_dataGen_V2 的 hstack):
      WRF window (315) | station_aux (4) | CLMS (3) | UF (17) | GeoEmbed (var)
    每个模态独立 MLP → concat → fusion 到 HLD 维。
    """
    def __init__(self, iDim, hid_per_mod=32, hld=128, geo_dim=None):
        super().__init__()
        # 根据 iDim 推算 geo_dim(其它都固定):
        # iDim = 315 + 4 + 3 + 17 + geo_dim = 339 + geo_dim
        if geo_dim is None:
            geo_dim = iDim - 339
            assert geo_dim > 0, f"unexpected iDim={iDim} (expected ≥ 340)"
        self.dims = [315, 4, 3, 17, geo_dim]
        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(d, hid_per_mod),
                                torch.nn.PReLU(hid_per_mod))
            for d in self.dims
        ])
        fused_in = len(self.dims) * hid_per_mod
        self.fusion1 = torch.nn.Linear(fused_in, hld)
        self.fusion1_act = torch.nn.PReLU(hld)
        # 第二层 Linear+PReLU 让架构和 flat 版深度对齐(都是 2 层)
        self.fusion2 = torch.nn.Linear(hld, hld)
        self.fusion2_act = torch.nn.PReLU(hld)

    def forward(self, x):
        # x: (N, iDim)
        offsets = [0]
        for d in self.dims:
            offsets.append(offsets[-1] + d)
        modal_emb = []
        for i, enc in enumerate(self.encoders):
            modal_emb.append(enc(x[:, offsets[i]:offsets[i+1]]))
        fused = torch.cat(modal_emb, dim=-1)        # (N, 5×hid_per_mod = 160)
        out = self.fusion1_act(self.fusion1(fused)) # (N, HLD)
        out = self.fusion2_act(self.fusion2(out))   # (N, HLD)
        return out


# -----------------Modality-Aware CNN Encoder (Experiment A, 2026-05-31)----
# 跟 ModalEncoder 的关键区别:GeoEmb 1008 维 reshape (7, 12, 12) → 2D CNN(空间结构 ≠ flat)
# 其它模态保持简单 Linear+PReLU
class ModalityAwareCNN(torch.nn.Module):
    """V2 schema 模态感知 encoder + 2D CNN on GeoEmb.

    Feature schema (V2):
      [   0,  315) WRF window (5 timesteps × 63 channels)
      [ 315,  319) station_aux (hour/month/year/station_id)
      [ 319,  322) CLMS (3 vegetation indices)
      [ 322,  339) UF (17 static urban form)
      [ 339, 1347) GeoEmbed (7 channels × 12 × 12 = 1008 spatial pool)

    架构:
      WRF (315)         → Linear → PReLU → hid_wrf
      station_aux (4)   → Linear → PReLU → hid_aux
      CLMS (3)          → Linear → PReLU → hid_clms
      UF (17)           → Linear → PReLU → hid_uf
      GeoEmb (1008)     → reshape(7,12,12) → Conv2D(7→16,3x3) → PReLU
                                          → Conv2D(16→32,3x3) → PReLU
                                          → AdaptiveAvgPool2d(4)
                                          → Flatten → Linear → PReLU → hid_geo
      Concat all (hid_wrf+hid_aux+hid_clms+hid_uf+hid_geo) → Linear → PReLU → HLD
                                                          → Linear → PReLU → HLD
    """
    def __init__(self, iDim, hld=128,
                 hid_wrf=48, hid_aux=8, hid_clms=8, hid_uf=16, hid_geo=32,
                 geo_channels=7, geo_spatial=12):
        super().__init__()
        assert iDim >= 1347, f"V2 ModalityAwareCNN expects iDim >= 1347, got {iDim}"
        self.geo_channels = geo_channels
        self.geo_spatial = geo_spatial
        self.geo_dim = geo_channels * geo_spatial * geo_spatial
        assert self.geo_dim == 1008, f"expected GeoEmb 1008, got {self.geo_dim}"

        # Modality boundaries (V2 schema)
        self.b_wrf = 315
        self.b_aux = 319
        self.b_clms = 322
        self.b_uf = 339
        self.b_geo = self.b_uf + self.geo_dim  # 1347

        # Per-modality encoders (small MLP)
        self.enc_wrf = torch.nn.Sequential(
            torch.nn.Linear(self.b_wrf, hid_wrf),
            torch.nn.PReLU(hid_wrf))
        self.enc_aux = torch.nn.Sequential(
            torch.nn.Linear(self.b_aux - self.b_wrf, hid_aux),
            torch.nn.PReLU(hid_aux))
        self.enc_clms = torch.nn.Sequential(
            torch.nn.Linear(self.b_clms - self.b_aux, hid_clms),
            torch.nn.PReLU(hid_clms))
        self.enc_uf = torch.nn.Sequential(
            torch.nn.Linear(self.b_uf - self.b_clms, hid_uf),
            torch.nn.PReLU(hid_uf))

        # 2D CNN on GeoEmb (KEY INNOVATION)
        # input shape per node: (7, 12, 12)
        self.enc_geo = torch.nn.Sequential(
            torch.nn.Conv2d(geo_channels, 16, kernel_size=3, padding=1),    # (16, 12, 12)
            torch.nn.PReLU(16),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),               # (32, 12, 12)
            torch.nn.PReLU(32),
            torch.nn.AdaptiveAvgPool2d(4),                                    # (32, 4, 4)
            torch.nn.Flatten(),                                               # (32 × 16 = 512)
            torch.nn.Linear(32 * 16, hid_geo),
            torch.nn.PReLU(hid_geo))

        # Fusion: concat → 2-layer Linear+PReLU (match flat encoder depth)
        fused_dim = hid_wrf + hid_aux + hid_clms + hid_uf + hid_geo
        self.fusion1 = torch.nn.Linear(fused_dim, hld)
        self.fusion1_act = torch.nn.PReLU(hld)
        self.fusion2 = torch.nn.Linear(hld, hld)
        self.fusion2_act = torch.nn.PReLU(hld)

        # Store for debug
        self._fused_dim = fused_dim
        self._hid_dims = (hid_wrf, hid_aux, hid_clms, hid_uf, hid_geo)

    def forward(self, x):
        # x: (N, 1347)
        N = x.shape[0]
        x_wrf  = x[:, 0:self.b_wrf]                              # (N, 315)
        x_aux  = x[:, self.b_wrf:self.b_aux]                     # (N, 4)
        x_clms = x[:, self.b_aux:self.b_clms]                    # (N, 3)
        x_uf   = x[:, self.b_clms:self.b_uf]                     # (N, 17)
        x_geo_flat = x[:, self.b_uf:self.b_geo]                  # (N, 1008)

        # Reshape GeoEmb to (N, 7, 12, 12) for 2D CNN
        x_geo = x_geo_flat.reshape(N, self.geo_channels,
                                    self.geo_spatial, self.geo_spatial)

        # Per-modality encoding
        h_wrf  = self.enc_wrf(x_wrf)
        h_aux  = self.enc_aux(x_aux)
        h_clms = self.enc_clms(x_clms)
        h_uf   = self.enc_uf(x_uf)
        h_geo  = self.enc_geo(x_geo)

        # Fusion
        fused = torch.cat([h_wrf, h_aux, h_clms, h_uf, h_geo], dim=-1)
        out = self.fusion1_act(self.fusion1(fused))
        out = self.fusion2_act(self.fusion2(out))
        return out


# -----------------Construct GNN model---------------------------
# 主 GNN 模型:Linear×2 → GraphConv×3 → Linear×2(faithful original_code)
class GNN(torch.nn.Module):
    """主 GNN 模型(faithful original_code)。

    结构:Linear×2(MLP encoder) → GraphConv(aggr=mean)×3(GNN processor)
        → Linear×2(MLP decoder,末尾带 PReLU)。所有隐藏维 HLD=128。
    输入特征 iDim 维向量,输出 oDim 维(回归目标 oDim=1)。

    新增 encoder_type='per_modality':每个模态独立编码后 fusion(降参数 + 解耦)。
    """

    # 按 modelPara 中的 nMLP / nGNN / HLD 堆叠 encoder / processor / decoder
    def __init__(self,modelPara):
        """构造模型,各层模块按 nMLP / nGNN 数量循环堆叠。

        conv_type 支持 'graphconv'(默认,faithful original_code)/ 'sageconv' / 'gat'。
        encoder_type 支持 'flat'(默认)/ 'per_modality'(分模态独立编码)。
        """
        super(GNN, self).__init__()
        self.nGNNLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        self.conv_type = modelPara.get('conv_type', 'graphconv').lower()
        self.encoder_type = modelPara.get('encoder_type', 'flat').lower()
        _HLD = modelPara['HLD']
        _encoder, _processor, _decoder = [],[],[]

        if self.encoder_type == 'per_modality':
            # 用单个 ModalEncoder 替代 Linear×2(从 iDim 自动推 geo_dim)
            _encoder.append(ModalEncoder(iDim=modelPara['iDim'],
                                          hid_per_mod=modelPara.get('modal_hid', 32),
                                          hld=_HLD))
            # 这一个 module 已经做了 2 层(fusion1 + fusion2),保持深度对齐
        elif self.encoder_type == 'modaware':
            # 模态感知 encoder + 2D CNN on GeoEmb (Experiment A, 2026-05-31)
            _encoder.append(ModalityAwareCNN(iDim=modelPara['iDim'],
                                              hld=_HLD))
        else:  # flat (default,原 Linear×2)
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
            elif self.conv_type == 'gat':
                # 4 heads,concat 后输出 _outputChannel = heads × per_head
                n_heads = 4
                _processor.append(GATConv(_inputChannel, _outputChannel // n_heads,
                                          heads=n_heads, concat=True))
            else:  # graphconv (默认,faithful original_code)
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
    
    # 前向:encoder → GNN 层消息传递(全图节点都过)→ decoder
    def forward(self, x, edgeIdx, edgeAttr):
        """前向:encoder(Linear+PReLU 或 ModalEncoder)→ processor → decoder。

        不同 conv_type 的 forward 接口不同:
          - GraphConv:用 edgeIdx + edgeAttr(weighted aggregation)
          - SAGEConv / GATConv:只用 edgeIdx(SAGE 不带边权,GAT 自己学注意力)

        **全图所有节点都过一遍**(包括 unlabeled / valid),
        loss/RMSE 在 train/test 函数里再用 label_mask 过滤。
        """
        for _f in self.encoder:
            # ModalEncoder 是单一可调用 module;Linear+PReLU 是逐层调
            x = _f(x)
        for _n, _f in enumerate(self.processor):
            if _n % 2 == 0:  # 偶数索引是 conv 层
                if isinstance(_f, GraphConv):
                    x = _f(x, edgeIdx, edgeAttr)
                else:  # SAGEConv / GATConv 不用 edgeAttr
                    x = _f(x, edgeIdx)
            else:  # 奇数索引是 PReLU
                x = _f(x)
        for _f in self.decoder:
            x = _f(x)
        return x
    
# 用 label_mask 把 yHat / y 切成"参与 loss"和"per-station 形状"两份,无 mask 时退化全节点
def _split_for_loss(yHat, batch, nNodes):
    """**核心 helper**:用 label_mask 过滤出真正参与 loss/RMSE 的预测和 target。

    - 若 batch 上有 `label_mask`:只取 mask=True 的位置(spatial 排除 valid 站,
      semi 排除 unlabeled 站)。
    - 若没 label_mask:退化到全节点(等价 original_code 行为)。
    - 同时 reshape 成 (batch_size, n_lbl_per_graph) 以便算 per-station RMSE。

    返回 (yHat_l, y_l, pred_2d, truth_2d)。
    """
    bs = batch.x.shape[0] // nNodes
    if hasattr(batch, 'label_mask') and batch.label_mask is not None:
        mask = batch.label_mask
        yHat_l = yHat[mask]
        y_l = batch.y[mask]
        n_lbl = mask.sum().item() // max(bs, 1)
        pred_2d = yHat_l.reshape(-1, n_lbl)
        truth_2d = y_l.reshape(-1, n_lbl)
        return yHat_l, y_l, pred_2d, truth_2d
    return yHat, batch.y, yHat.reshape(-1, nNodes), batch.y.reshape(-1, nNodes)


_DEBUG_PRINTED = {'train': False, 'test': False}


# 一个 epoch 的训练循环(forward → mask 后算 loss → backward → optimizer.step + scheduler.step)
def train(loader,model,lossFn,opt,scheduler,device,nNodes):
    """一个 epoch 训练:遍历 trainLoader 每个 batch → forward → loss
    (mask 过滤后)→ backward → optimizer.step()。

    末尾 scheduler.step()(epoch-level decay)。返回 (avg_loss, RMSE, truth, pred)。
    第 0 个 batch 会打印 [DEBUG/train] 一次,做 NaN/Inf 断言。
    """
    model.train()
    _LOSS = 0
    pred,truth = [],[]
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        _yHat_l, _y_l, _pred, _truth = _split_for_loss(_yHat, _batch, nNodes)
        _loss = lossFn(_yHat_l, _y_l)
        _loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += _loss
        # ===== DEBUG: first batch first epoch only =====
        if _n == 0 and not _DEBUG_PRINTED['train']:
            bs = _batch.x.shape[0] // nNodes
            mask_per_g = _batch.label_mask.sum().item() // max(bs, 1) if hasattr(_batch, 'label_mask') else nNodes
            n_edges = _batch.edge_index.shape[1]
            print(f"[DEBUG/train] first batch: x={tuple(_batch.x.shape)}, y={tuple(_batch.y.shape)}, "
                  f"edges={n_edges}, batch_size={bs}, masked_per_graph={mask_per_g}")
            assert torch.isfinite(_yHat).all(), "[ERR] yHat has NaN/Inf in first batch"
            assert torch.isfinite(_loss), f"[ERR] train loss is NaN/Inf: {_loss}"
            assert torch.isfinite(_batch.x).all(), "[ERR] input x has NaN/Inf"
            print(f"[DEBUG/train] yHat range=[{_yHat.min().item():.4f}, {_yHat.max().item():.4f}], "
                  f"y range=[{_batch.y.min().item():.4f}, {_batch.y.max().item():.4f}], "
                  f"loss={_loss.item():.4e}")
            print(f"[DEBUG/train] x feature stats: NaN={int(torch.isnan(_batch.x).sum())}, "
                  f"Inf={int(torch.isinf(_batch.x).sum())}, "
                  f"range=[{_batch.x.min().item():.3f}, {_batch.x.max().item():.3f}]")
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            if grad_norms:
                print(f"[DEBUG/train] grad norm: mean={sum(grad_norms)/len(grad_norms):.4e}, "
                      f"max={max(grad_norms):.4e}, n_params={len(grad_norms)}")
            _DEBUG_PRINTED['train'] = True
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth,pred)
    return (_LOSS/(_n+1)).item(), _RMSE, truth, pred

# 一个 epoch 的验证循环(torch.no_grad,只算 loss/RMSE 不更新参数)
def test(loader,model,lossFn,device,nNodes):
    """一个 epoch 验证:torch.no_grad() 下遍历 validLoader,各 batch forward + 算 loss/RMSE
    (用 label_mask 过滤)。

    返回 (avg_loss, RMSE, truth, pred)。第 0 个 batch 会打印 [DEBUG/test] 一次。
    """
    model.eval()
    _LOSS = 0
    pred,truth = [],[]
    with torch.no_grad():
        for _n, _batch in enumerate(loader):
            _batch = _batch.to(device)
            _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
            _yHat_l, _y_l, _pred, _truth = _split_for_loss(_yHat, _batch, nNodes)
            _loss = lossFn(_yHat_l, _y_l)
            _LOSS += _loss
            # ===== DEBUG: first eval batch only =====
            if _n == 0 and not _DEBUG_PRINTED['test']:
                bs = _batch.x.shape[0] // nNodes
                mask_per_g = _batch.label_mask.sum().item() // max(bs, 1) if hasattr(_batch, 'label_mask') else nNodes
                assert torch.isfinite(_yHat).all(), "[ERR] valid yHat has NaN/Inf"
                assert torch.isfinite(_loss), f"[ERR] valid loss is NaN/Inf: {_loss}"
                print(f"[DEBUG/test] first batch: x={tuple(_batch.x.shape)}, "
                      f"masked_per_graph={mask_per_g}, "
                      f"yHat range=[{_yHat.min().item():.4f}, {_yHat.max().item():.4f}], "
                      f"y range=[{_batch.y.min().item():.4f}, {_batch.y.max().item():.4f}], "
                      f"loss={_loss.item():.4e}")
                _DEBUG_PRINTED['test'] = True
            pred += list(_pred.cpu().numpy())
            truth += list(_truth.cpu().numpy())
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth,pred)
    return (_LOSS/(_n+1)).item(), _RMSE, truth, pred


# =====================================================================
# EAD train/test:模型输出 ε̂(残差),target = y - wrf_t2 - α - β,
# loss 在 ε 空间算,RMSE 在 T 空间报(reconstruct T_hat 与 baseline 同尺度可比)
# =====================================================================

# 一个 epoch 的 EAD 训练循环
def train_ead(loader, model, lossFn, opt, scheduler, device, nNodes):
    """模型预测 ε̂,target_ε = y - wrf_t2 - α - β,RMSE 在 T 空间报(可与 baseline 比)。"""
    model.train()
    _LOSS = 0
    pred_T, truth_T = [], []
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        eps_hat = model(_batch.x, _batch.edge_index, _batch.edge_attr).squeeze(-1)
        bs = _batch.x.shape[0] // nNodes
        # 重组 EAD 三块到 (bs * nNodes,)
        wrf_t2 = _batch.wrf_t2.squeeze(-1)                               # (bs*nNodes,)
        alpha  = _batch.alpha_t.repeat_interleave(nNodes)                # (bs*nNodes,) from (bs,)
        beta   = _batch.beta_hat.squeeze(-1)                             # (bs*nNodes,)
        y_flat = _batch.y.squeeze(-1)                                    # (bs*nNodes,)
        eps_target = y_flat - wrf_t2 - alpha - beta                       # ε_target

        # 用 label_mask 过滤参与 loss 的节点
        mask = _batch.label_mask
        loss = lossFn(eps_hat[mask], eps_target[mask])

        loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += loss

        # T 空间 reconstruct,RMSE 与 baseline 同尺度
        T_hat = wrf_t2 + alpha + beta + eps_hat                          # (bs*nNodes,)
        n_lbl = mask.sum().item() // max(bs, 1)
        pred_T.append(T_hat[mask].detach().cpu().numpy().reshape(-1, n_lbl))
        truth_T.append(y_flat[mask].detach().cpu().numpy().reshape(-1, n_lbl))

        # ===== DEBUG: first batch =====
        if _n == 0 and not _DEBUG_PRINTED.get('train_ead', False):
            assert torch.isfinite(eps_hat).all(), "[ERR/EAD] eps_hat NaN/Inf"
            assert torch.isfinite(eps_target).all(), "[ERR/EAD] eps_target NaN/Inf"
            print(f"[DEBUG/train_ead] eps_hat range=[{eps_hat.min().item():.4f}, {eps_hat.max().item():.4f}]")
            print(f"[DEBUG/train_ead] eps_target range=[{eps_target.min().item():.4f}, {eps_target.max().item():.4f}], "
                  f"mean={eps_target[mask].mean().item():.4f}(应该接近 0)")
            print(f"[DEBUG/train_ead] T_hat range=[{T_hat.min().item():.4f}, {T_hat.max().item():.4f}], "
                  f"y range=[{y_flat.min().item():.4f}, {y_flat.max().item():.4f}]")
            print(f"[DEBUG/train_ead] α first 3 graphs={_batch.alpha_t[:3].tolist()}, "
                  f"β graph0[:3]={beta.reshape(bs, nNodes)[0, :3].tolist()}, "
                  f"wrf_t2 graph0[:3]={wrf_t2.reshape(bs, nNodes)[0, :3].tolist()}")
            print(f"[DEBUG/train_ead] loss (ε空间)={loss.item():.4e}")
            _DEBUG_PRINTED['train_ead'] = True

    scheduler.step()
    pred_T = np.concatenate(pred_T, axis=0)
    truth_T = np.concatenate(truth_T, axis=0)
    _RMSE = utils.RMSE(truth_T, pred_T)                                  # T 空间 RMSE
    return (_LOSS/(_n+1)).item(), _RMSE, truth_T, pred_T


# 一个 epoch 的 EAD 验证循环
def test_ead(loader, model, lossFn, device, nNodes):
    """同 train_ead 的 forward 逻辑,no_grad,RMSE 在 T 空间报。"""
    model.eval()
    _LOSS = 0
    pred_T, truth_T = [], []
    with torch.no_grad():
        for _n, _batch in enumerate(loader):
            _batch = _batch.to(device)
            eps_hat = model(_batch.x, _batch.edge_index, _batch.edge_attr).squeeze(-1)
            bs = _batch.x.shape[0] // nNodes
            wrf_t2 = _batch.wrf_t2.squeeze(-1)
            alpha  = _batch.alpha_t.repeat_interleave(nNodes)
            beta   = _batch.beta_hat.squeeze(-1)
            y_flat = _batch.y.squeeze(-1)
            eps_target = y_flat - wrf_t2 - alpha - beta
            mask = _batch.label_mask
            loss = lossFn(eps_hat[mask], eps_target[mask])
            _LOSS += loss

            T_hat = wrf_t2 + alpha + beta + eps_hat
            n_lbl = mask.sum().item() // max(bs, 1)
            pred_T.append(T_hat[mask].cpu().numpy().reshape(-1, n_lbl))
            truth_T.append(y_flat[mask].cpu().numpy().reshape(-1, n_lbl))

            if _n == 0 and not _DEBUG_PRINTED.get('test_ead', False):
                assert torch.isfinite(eps_hat).all()
                print(f"[DEBUG/test_ead] eps_hat range=[{eps_hat.min().item():.4f}, {eps_hat.max().item():.4f}]")
                print(f"[DEBUG/test_ead] T_hat range=[{T_hat.min().item():.4f}, {T_hat.max().item():.4f}], "
                      f"y range=[{y_flat.min().item():.4f}, {y_flat.max().item():.4f}], loss={loss.item():.4e}")
                _DEBUG_PRINTED['test_ead'] = True

    pred_T = np.concatenate(pred_T, axis=0)
    truth_T = np.concatenate(truth_T, axis=0)
    _RMSE = utils.RMSE(truth_T, pred_T)
    return (_LOSS/(_n+1)).item(), _RMSE, truth_T, pred_T


# =====================================================================
# Laplacian 正则:在 baseline 之上加 λ × Σ w_ij (ŷ_i − ŷ_j)²
# 让相邻节点预测相近(独立 plug-in,可与 EAD 叠加)
# =====================================================================

# 一个 epoch 的 Laplacian-augmented 训练循环
def train_lap(loader, model, lossFn, opt, scheduler, device, nNodes,
              edge_src, edge_dst, edge_w, lambda_lap):
    """sup_loss + λ_lap × mean_edges w_ij (ŷ_i − ŷ_j)² (per-graph 平均后再跨 batch 平均)。

    edge_src, edge_dst:LongTensor 一阶邻接索引(单图,会广播到 batch 里每个图)
    edge_w:FloatTensor 边权
    lambda_lap:float,Lap loss 权重
    """
    model.train()
    _LOSS_TOTAL = 0
    _LOSS_SUP = 0
    _LOSS_LAP = 0
    pred, truth = [], []
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
        yHat_l, y_l, _pred, _truth = _split_for_loss(yHat, _batch, nNodes)
        sup_loss = lossFn(yHat_l, y_l)

        # ---- Laplacian loss(向量化:跨 batch 同图)----
        bs = _batch.x.shape[0] // nNodes
        yHat_g = yHat.squeeze(-1).reshape(bs, nNodes)                  # (bs, nNodes)
        diff = yHat_g[:, edge_src] - yHat_g[:, edge_dst]                # (bs, n_edges)
        lap_loss = (edge_w[None, :] * diff ** 2).mean()                  # scalar
        total_loss = sup_loss + lambda_lap * lap_loss

        total_loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS_TOTAL += total_loss
        _LOSS_SUP += sup_loss.item()
        _LOSS_LAP += lap_loss.item()
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

        # ===== DEBUG: first batch =====
        if _n == 0 and not _DEBUG_PRINTED.get('train_lap', False):
            assert torch.isfinite(yHat).all(), "[ERR/LAP] yHat NaN/Inf"
            assert torch.isfinite(lap_loss), "[ERR/LAP] lap_loss NaN/Inf"
            print(f"[DEBUG/train_lap] sup_loss={sup_loss.item():.4e}, "
                  f"lap_loss={lap_loss.item():.4e}, λ_lap={lambda_lap}, "
                  f"total={total_loss.item():.4e}")
            print(f"[DEBUG/train_lap] yHat range=[{yHat.min().item():.4f}, {yHat.max().item():.4f}], "
                  f"diff |ŷ_i−ŷ_j| max={diff.abs().max().item():.4f}, n_edges={edge_src.shape[0]}")
            print(f"[DEBUG/train_lap] edge_w range=[{edge_w.min().item():.4f}, {edge_w.max().item():.4f}], "
                  f"mean={edge_w.mean().item():.4f}")
            _DEBUG_PRINTED['train_lap'] = True

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    # 报告 total loss(主指标);sup/lap 分量通过模块属性透出供 run.py 单独 log
    train_lap.last_sup = _LOSS_SUP / (_n + 1)
    train_lap.last_lap = _LOSS_LAP / (_n + 1)
    return (_LOSS_TOTAL/(_n+1)).item(), _RMSE, truth, pred


# =====================================================================
# Experiment E:Temporal Laplacian on top of naive semi baseline
# 时序平滑约束:相邻 timestep 的 prediction 应接近
# REQUIRES shuffle=False in trainLoader 才能保证 batch 内样本时序相邻
# =====================================================================
def train_templap(loader, model, lossFn, opt, scheduler, device, nNodes,
                  lambda_temp):
    """sup_loss + λ_temp × mean_(i, u) (ŷ_{t_i, u} − ŷ_{t_{i-1}, u})²

    Within each batch (size bs),samples are at consecutive timesteps if shuffle=False.
    Temporal diff between sample i and i+1 → 对所有 458 station 应用 smoothness 约束。

    Args:
        lambda_temp: weight on temporal smoothness term

    Returns: (loss, RMSE, truth, pred) — 同 train()
    """
    model.train()
    _LOSS_TOTAL = 0
    _LOSS_SUP = 0
    _LOSS_TEMP = 0
    pred, truth = [], []

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
        yHat_l, y_l, _pred, _truth = _split_for_loss(yHat, _batch, nNodes)
        sup_loss = lossFn(yHat_l, y_l)

        # ---- Temporal Lap:相邻 batch 样本 (假设时序连续 = shuffle=False) ----
        bs = _batch.x.shape[0] // nNodes
        if bs >= 2:
            yHat_g = yHat.squeeze(-1).reshape(bs, nNodes)        # (bs, nNodes)
            temp_diff = yHat_g[1:] - yHat_g[:-1]                   # (bs-1, nNodes)
            temp_loss = (temp_diff ** 2).mean()
        else:
            temp_loss = torch.tensor(0.0, device=device)

        total_loss = sup_loss + lambda_temp * temp_loss
        total_loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS_TOTAL += total_loss
        _LOSS_SUP += sup_loss.item()
        _LOSS_TEMP += temp_loss.item()
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

        # ===== DEBUG: first batch =====
        if _n == 0 and not _DEBUG_PRINTED.get('train_templap', False):
            assert torch.isfinite(yHat).all(), "[ERR/TEMPLAP] yHat NaN/Inf"
            assert torch.isfinite(temp_loss), f"[ERR/TEMPLAP] temp_loss NaN/Inf: {temp_loss}"
            assert torch.isfinite(total_loss), f"[ERR/TEMPLAP] total NaN/Inf: {total_loss}"
            print(f"[DEBUG/train_templap] bs={bs}, sup={sup_loss.item():.4e}, "
                  f"temp={temp_loss.item():.4e}, λ_temp={lambda_temp}, "
                  f"total={total_loss.item():.4e}")
            if bs >= 2:
                print(f"[DEBUG/train_templap] yHat_g shape={tuple(yHat_g.shape)}, "
                      f"temp_diff shape={tuple(temp_diff.shape)}")
                print(f"[DEBUG/train_templap] |ŷ_t − ŷ_(t-1)|: mean={temp_diff.abs().mean().item():.4f}, "
                      f"max={temp_diff.abs().max().item():.4f}, "
                      f"std={temp_diff.abs().std().item():.4f}")
                # Sanity: if shuffle=False, diff 应该比较小;shuffle=True 会大很多
                if temp_diff.abs().mean().item() > 0.5:
                    print(f"[WARN/TEMPLAP] |ŷ_t − ŷ_(t-1)| 平均 > 0.5 — 可能 shuffle=True "
                          f"导致相邻 samples 不是连续时间步,temp_loss 含义混乱")
            _DEBUG_PRINTED['train_templap'] = True

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    train_templap.last_sup = _LOSS_SUP / (_n + 1)
    train_templap.last_temp = _LOSS_TEMP / (_n + 1)
    return (_LOSS_TOTAL/(_n+1)).item(), _RMSE, truth, pred


# =====================================================================
# Experiment D:Dual Laplacian — spatial + feature similarity
# 同时使用空间 kNN 图 和 feature 相似度 kNN 图
# =====================================================================
def train_duallap(loader, model, lossFn, opt, scheduler, device, nNodes,
                  edge_src_s, edge_dst_s, edge_w_s,
                  edge_src_f, edge_dst_f, edge_w_f,
                  lambda_s, lambda_f):
    """sup_loss + λ_s × spatial_Lap + λ_f × feature_Lap

    两个 Laplacian 项:
      - spatial: 基于 kNN spatial graph 的边权(传统空间平滑)
      - feature: 基于 1347-D feature cosine similarity kNN 的边权(新增)

    Args:
        edge_src_s/dst_s/w_s: 空间图边
        edge_src_f/dst_f/w_f: feature 相似度图边
        lambda_s, lambda_f: 两个 Lap 的权重

    Returns: (loss, RMSE, truth, pred)
    """
    model.train()
    _LOSS_TOTAL = 0
    _LOSS_SUP = 0
    _LOSS_S = 0
    _LOSS_F = 0
    pred, truth = [], []

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
        yHat_l, y_l, _pred, _truth = _split_for_loss(yHat, _batch, nNodes)
        sup_loss = lossFn(yHat_l, y_l)

        bs = _batch.x.shape[0] // nNodes
        yHat_g = yHat.squeeze(-1).reshape(bs, nNodes)         # (bs, nNodes)

        # Spatial Lap
        if lambda_s > 0:
            diff_s = yHat_g[:, edge_src_s] - yHat_g[:, edge_dst_s]
            lap_s = (edge_w_s[None, :] * diff_s ** 2).mean()
        else:
            lap_s = torch.tensor(0.0, device=device)

        # Feature Lap
        if lambda_f > 0:
            diff_f = yHat_g[:, edge_src_f] - yHat_g[:, edge_dst_f]
            lap_f = (edge_w_f[None, :] * diff_f ** 2).mean()
        else:
            lap_f = torch.tensor(0.0, device=device)

        total_loss = sup_loss + lambda_s * lap_s + lambda_f * lap_f
        total_loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS_TOTAL += total_loss
        _LOSS_SUP += sup_loss.item()
        _LOSS_S += lap_s.item()
        _LOSS_F += lap_f.item()
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

        if _n == 0 and not _DEBUG_PRINTED.get('train_duallap', False):
            assert torch.isfinite(yHat).all() and torch.isfinite(total_loss)
            print(f"[DEBUG/train_duallap] sup={sup_loss.item():.4e}, "
                  f"lap_s={lap_s.item():.4e} (λ_s={lambda_s}), "
                  f"lap_f={lap_f.item():.4e} (λ_f={lambda_f}), "
                  f"total={total_loss.item():.4e}")
            print(f"[DEBUG/train_duallap] n_edges_spatial={edge_src_s.shape[0]}, "
                  f"n_edges_feature={edge_src_f.shape[0]}")
            print(f"[DEBUG/train_duallap] edge_w_s range=[{edge_w_s.min().item():.4f}, "
                  f"{edge_w_s.max().item():.4f}], "
                  f"edge_w_f range=[{edge_w_f.min().item():.4f}, "
                  f"{edge_w_f.max().item():.4f}]")
            _DEBUG_PRINTED['train_duallap'] = True

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    train_duallap.last_sup = _LOSS_SUP / (_n + 1)
    train_duallap.last_lap_s = _LOSS_S / (_n + 1)
    train_duallap.last_lap_f = _LOSS_F / (_n + 1)
    return (_LOSS_TOTAL/(_n+1)).item(), _RMSE, truth, pred


# 一个 epoch 的 EAD + Lap 联合训练(模型预测 ε,Lap 平滑 ε,对应 V0 "residual Laplacian")
def train_ead_lap(loader, model, lossFn, opt, scheduler, device, nNodes,
                  edge_src, edge_dst, edge_w, lambda_lap):
    """sup_loss(ε) + λ_lap × mean_edges w_ij (ε̂_i − ε̂_j)²;RMSE 在 T 空间报。"""
    model.train()
    _LOSS_TOTAL = 0
    _LOSS_SUP = 0
    _LOSS_LAP = 0
    pred_T, truth_T = [], []
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        eps_hat = model(_batch.x, _batch.edge_index, _batch.edge_attr).squeeze(-1)
        bs = _batch.x.shape[0] // nNodes
        wrf_t2 = _batch.wrf_t2.squeeze(-1)
        alpha  = _batch.alpha_t.repeat_interleave(nNodes)
        beta   = _batch.beta_hat.squeeze(-1)
        y_flat = _batch.y.squeeze(-1)
        eps_target = y_flat - wrf_t2 - alpha - beta
        mask = _batch.label_mask
        sup_loss = lossFn(eps_hat[mask], eps_target[mask])

        # Lap on ε(平滑残差,而非 T)
        eps_g = eps_hat.reshape(bs, nNodes)
        diff = eps_g[:, edge_src] - eps_g[:, edge_dst]                   # (bs, n_edges)
        lap_loss = (edge_w[None, :] * diff ** 2).mean()
        total_loss = sup_loss + lambda_lap * lap_loss

        total_loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS_TOTAL += total_loss
        _LOSS_SUP += sup_loss.item()
        _LOSS_LAP += lap_loss.item()

        # T-space RMSE
        T_hat = wrf_t2 + alpha + beta + eps_hat
        n_lbl = mask.sum().item() // max(bs, 1)
        pred_T.append(T_hat[mask].detach().cpu().numpy().reshape(-1, n_lbl))
        truth_T.append(y_flat[mask].detach().cpu().numpy().reshape(-1, n_lbl))

        if _n == 0 and not _DEBUG_PRINTED.get('train_ead_lap', False):
            assert torch.isfinite(eps_hat).all() and torch.isfinite(lap_loss)
            print(f"[DEBUG/train_ead_lap] sup(ε)={sup_loss.item():.4e}, "
                  f"lap(ε)={lap_loss.item():.4e}, λ={lambda_lap}, total={total_loss.item():.4e}")
            print(f"[DEBUG/train_ead_lap] eps_hat range=[{eps_hat.min().item():.4f}, {eps_hat.max().item():.4f}], "
                  f"|ε_i−ε_j| max={diff.abs().max().item():.4f}, T_hat range=[{T_hat.min().item():.4f}, "
                  f"{T_hat.max().item():.4f}], y range=[{y_flat.min().item():.4f}, {y_flat.max().item():.4f}]")
            _DEBUG_PRINTED['train_ead_lap'] = True

    scheduler.step()
    pred_T = np.concatenate(pred_T, axis=0)
    truth_T = np.concatenate(truth_T, axis=0)
    _RMSE = utils.RMSE(truth_T, pred_T)
    train_ead_lap.last_sup = _LOSS_SUP / (_n + 1)
    train_ead_lap.last_lap = _LOSS_LAP / (_n + 1)
    return (_LOSS_TOTAL/(_n+1)).item(), _RMSE, truth_T, pred_T


# 模型 checkpoint 的加载或全新初始化(返回 EPOCH/bestLoss/chkptPath/hist)
def loadCheckPoint(modelName,model,opt,device,load=False,resetLr=False,lr=5e-5,predMode=False):
    """模型 checkpoint 加载/初始化。

    `load=True` 时尝试从 `./{modelName}.pt` 恢复(state_dict + optimizer + epoch + bestLoss);
    `predMode=True` 时只加载 model state(用于 predict.py 推理);
    其它情况下从头开始(EPOCH=0, bestLoss=inf, hist=[])。

    返回 (EPOCH, bestLoss, chkptPath, hist)。
    """
    chkptPath = f'./{modelName}.pt'
    if os.path.exists(chkptPath) and load:
        chkpt = torch.load(chkptPath,map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        opt.load_state_dict(chkpt['opt_state_dict'])
        EPOCH = chkpt['epoch']
        bestLoss = chkpt['bestLoss']
        hist = chkpt.get('hist', [])
        print("Checkpoint loaded.")
        if opt.param_groups[0]['lr'] < 1e-6 and resetLr:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            print(f"Resetting LR to {lr}")
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
        print("No checkpoint found, starting new model.")
    return EPOCH,bestLoss,chkptPath,hist

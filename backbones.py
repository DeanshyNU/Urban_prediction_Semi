"""
模型模块：GNN的替代模型
包括同级别替代模型和高级专用架构（如 iTransformer）
所有模型保持相同的 forward(x, edgeIdx, edgeAttr) 接口，兼容现有训练代码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. Pure MLP (Baseline - 无空间建模)
# =============================================================================
class MLPModel(nn.Module):
    """
    纯MLP模型：每个节点独立处理，无空间交互
    Processor 用 Linear + PReLU 替代 SAGEConv
    作为最简单的 baseline，验证半监督框架本身的效果
    """
    def __init__(self, modelPara):
        super(MLPModel, self).__init__()
        self.nProcessorLayers = modelPara['nGNN']  # 保持相同层数
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        _encoder, _processor, _decoder = [], [], []

        # Encoder: 和 GNN 完全相同
        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _outputChannel = _HLD
            _encoder.append(nn.Linear(_inputChannel, _outputChannel))
            _encoder.append(nn.PReLU(_outputChannel))

        # Processor: Linear 替代 SAGEConv（无空间聚合）
        for _n in range(self.nProcessorLayers):
            _processor.append(nn.Linear(_HLD, _HLD))
            _processor.append(nn.PReLU(_HLD))

        # Decoder: 和 GNN 完全相同
        for _n in range(self.nMLPLayers):
            _inputChannel = _HLD
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(_inputChannel, _outputChannel))
            _decoder.append(nn.PReLU(_outputChannel))

        self.encoder = nn.ModuleList(_encoder)
        self.processor = nn.ModuleList(_processor)
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        # Encoder
        for _f in self.encoder:
            x = _f(x)
        # Processor (纯MLP，忽略图结构)
        for _f in self.processor:
            x = _f(x)
        # Decoder
        for _f in self.decoder:
            x = _f(x)
        return x


# =============================================================================
# 2. Self-Attention Model (Transformer Processor)
# =============================================================================
class SelfAttentionProcessor(nn.Module):
    """
    单层 Self-Attention：节点间通过注意力交互（替代图卷积的空间聚合）
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super(SelfAttentionProcessor, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-Attention + Residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN + Residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerModel(nn.Module):
    """
    Transformer模型：Self-Attention 替代 SAGEConv
    节点间通过注意力机制学习空间关系（隐式），不需要显式图结构

    架构: Encoder(MLP) → Processor(Self-Attention × N) → Decoder(MLP)
    """
    def __init__(self, modelPara):
        super(TransformerModel, self).__init__()
        self.nProcessorLayers = modelPara['nGNN']  # 保持相同层数
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        self.nNodes = modelPara.get('nNodes', None)  # 需要知道节点数来 reshape

        # Encoder: 和 GNN 完全相同
        _encoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _outputChannel = _HLD
            _encoder.append(nn.Linear(_inputChannel, _outputChannel))
            _encoder.append(nn.PReLU(_outputChannel))
        self.encoder = nn.ModuleList(_encoder)

        # Processor: Self-Attention layers
        nhead = modelPara.get('nhead', 4)
        dropout = modelPara.get('dropout', 0.1)
        self.processor = nn.ModuleList([
            SelfAttentionProcessor(_HLD, nhead=nhead, dropout=dropout)
            for _ in range(self.nProcessorLayers)
        ])

        # Decoder: 和 GNN 完全相同
        _decoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = _HLD
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(_inputChannel, _outputChannel))
            _decoder.append(nn.PReLU(_outputChannel))
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        nNodes = self.nNodes

        # Encoder: (B*N, F) → (B*N, H)
        for _f in self.encoder:
            x = _f(x)

        # Reshape for attention: (B*N, H) → (B, N, H)
        B = x.shape[0] // nNodes
        x = x.view(B, nNodes, -1)

        # Processor: Self-Attention between nodes
        for layer in self.processor:
            x = layer(x)

        # Reshape back: (B, N, H) → (B*N, H)
        x = x.view(B * nNodes, -1)

        # Decoder
        for _f in self.decoder:
            x = _f(x)
        return x


# =============================================================================
# 3. 1D CNN Model (局部空间模式)
# =============================================================================
class CNNModel(nn.Module):
    """
    1D CNN模型：在节点维度上做卷积，捕捉局部空间模式

    架构: Encoder(MLP) → Processor(Conv1d × N) → Decoder(MLP)
    """
    def __init__(self, modelPara):
        super(CNNModel, self).__init__()
        self.nProcessorLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        self.nNodes = modelPara.get('nNodes', None)

        # Encoder
        _encoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _outputChannel = _HLD
            _encoder.append(nn.Linear(_inputChannel, _outputChannel))
            _encoder.append(nn.PReLU(_outputChannel))
        self.encoder = nn.ModuleList(_encoder)

        # Processor: 1D Conv on node dimension (with padding to keep size)
        _processor = []
        for _n in range(self.nProcessorLayers):
            _processor.append(nn.Conv1d(_HLD, _HLD, kernel_size=3, padding=1))
            _processor.append(nn.PReLU(_HLD))
        self.processor = nn.ModuleList(_processor)

        # Decoder
        _decoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = _HLD
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(_inputChannel, _outputChannel))
            _decoder.append(nn.PReLU(_outputChannel))
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        nNodes = self.nNodes

        # Encoder: (B*N, F) → (B*N, H)
        for _f in self.encoder:
            x = _f(x)

        # Reshape: (B*N, H) → (B, H, N) for Conv1d
        B = x.shape[0] // nNodes
        x = x.view(B, nNodes, -1).permute(0, 2, 1)  # (B, H, N)

        # Processor: Conv1d on node dimension
        for _n, _f in enumerate(self.processor):
            if _n % 2 == 0:  # Conv layer
                x = _f(x)
            else:  # PReLU (needs channel-first format, which we already have)
                x = _f(x)

        # Reshape back: (B, H, N) → (B*N, H)
        x = x.permute(0, 2, 1).contiguous().view(B * nNodes, -1)

        # Decoder
        for _f in self.decoder:
            x = _f(x)
        return x


# =============================================================================
# 4. LSTM Model (时序建模 - 把节点序列当时序)
# =============================================================================
class LSTMModel(nn.Module):
    """
    LSTM模型：把节点序列看作时序，用LSTM捕捉节点间的依赖
    注意：这里是把 N 个节点当作序列长度（空间维度当时序处理）

    架构: Encoder(MLP) → Processor(BiLSTM × N) → Decoder(MLP)
    """
    def __init__(self, modelPara):
        super(LSTMModel, self).__init__()
        self.nProcessorLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        self.nNodes = modelPara.get('nNodes', None)

        # Encoder
        _encoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _outputChannel = _HLD
            _encoder.append(nn.Linear(_inputChannel, _outputChannel))
            _encoder.append(nn.PReLU(_outputChannel))
        self.encoder = nn.ModuleList(_encoder)

        # Processor: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=_HLD,
            hidden_size=_HLD // 2,  # 双向，所以 /2 保持输出维度一致
            num_layers=self.nProcessorLayers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if self.nProcessorLayers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(_HLD)

        # Decoder
        _decoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = _HLD
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(_inputChannel, _outputChannel))
            _decoder.append(nn.PReLU(_outputChannel))
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        nNodes = self.nNodes

        # Encoder: (B*N, F) → (B*N, H)
        for _f in self.encoder:
            x = _f(x)

        # Reshape: (B*N, H) → (B, N, H)
        B = x.shape[0] // nNodes
        x_reshaped = x.view(B, nNodes, -1)

        # Processor: BiLSTM + Residual
        lstm_out, _ = self.lstm(x_reshaped)
        x = self.lstm_norm(x_reshaped + lstm_out)  # residual connection

        # Reshape back: (B, N, H) → (B*N, H)
        x = x.contiguous().view(B * nNodes, -1)

        # Decoder
        for _f in self.decoder:
            x = _f(x)
        return x


# =============================================================================
# 5. iTransformer Model (Inverted Transformer - 站点级注意力)
# =============================================================================
class iTransformerModel(nn.Module):
    """
    iTransformer: 反转Transformer架构 (ICLR 2024)

    核心思想：把每个气象站当作一个 token，特征维度当作"序列"
    - 注意力在站点间计算 → 学习空间关系（N×N attention matrix）
    - FFN 处理每个站点的特征表示 → 学习特征模式

    与原始 iTransformer 的适配：
    - 原始: variates=传感器通道, seq_len=时间步
    - 本项目: variates=气象站(nNodes), seq_len=特征维度(iDim)
    - 无位置编码（站点无固有顺序，与原始iTransformer设计一致）
    - 无RevIN（数据已预处理，输出空间≠输入空间）

    接口: forward(x, edgeIdx=None, edgeAttr=None) 兼容现有训练代码
    """
    def __init__(self, modelPara):
        super(iTransformerModel, self).__init__()
        self.nNodes = modelPara['nNodes']  # 站点数 = token数
        iDim = modelPara['iDim']           # 输入特征维度
        oDim = modelPara['oDim']           # 输出维度（=1, 温度预测）
        d_model = modelPara['HLD']         # 隐藏维度（d_model）
        n_heads = modelPara.get('nhead', 8)
        e_layers = modelPara.get('nGNN', 3)  # encoder层数
        d_ff = modelPara.get('d_ff', d_model * 4)  # FFN维度
        dropout = modelPara.get('dropout', 0.1)

        # --- Inverted Embedding ---
        # 每个站点的 iDim 维特征 → d_model 维表示
        # 等价于 iTransformer 中的 DataEmbedding_inverted: Linear(seq_len, d_model)
        self.input_norm = nn.LayerNorm(iDim)
        self.embedding = nn.Linear(iDim, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # --- Transformer Encoder ---
        # 标准 Transformer Encoder，注意力在 N 个站点 token 之间计算
        # norm_first=True: Pre-LayerNorm (更稳定的训练)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers,
            norm=nn.LayerNorm(d_model),  # 最终 LayerNorm
        )

        # --- Output Projection ---
        # d_model → oDim (通常 oDim=1)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, oDim),
        )

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        """
        参数:
            x: (B*N, iDim) — PyG flat format
            edgeIdx: 不使用（注意力隐式学习空间关系）
            edgeAttr: 不使用

        返回:
            (B*N, oDim)
        """
        N = self.nNodes
        B = x.shape[0] // N

        # Reshape: (B*N, iDim) → (B, N, iDim)
        x = x.view(B, N, -1)

        # Inverted Embedding: (B, N, iDim) → (B, N, d_model)
        x = self.input_norm(x)
        x = self.embedding(x)
        x = self.embed_dropout(x)

        # Transformer Encoder: attention across N station tokens
        # (B, N, d_model) → (B, N, d_model)
        x = self.encoder(x)

        # Output Projection: (B, N, d_model) → (B, N, oDim)
        x = self.projector(x)

        # Reshape back: (B, N, oDim) → (B*N, oDim)
        x = x.reshape(B * N, -1)

        return x


# =============================================================================
# 6. TimeMixer Model (Multi-Scale Mixing MLP - ICLR 2024)
# =============================================================================
class MultiScaleMixingBlock(nn.Module):
    """
    多尺度混合块（受 TimeMixer PDM 启发）

    核心操作：
    - Bottom-up mixing: 细粒度 → 粗粒度（类似季节性分量传递）
    - Top-down mixing: 粗粒度 → 细粒度（类似趋势分量传递）
    - Per-scale FFN: 每个尺度独立的特征变换
    """
    def __init__(self, d_model, d_ff, n_scales, dropout):
        super(MultiScaleMixingBlock, self).__init__()
        self.n_scales = n_scales

        # Bottom-up mixing (fine → coarse)
        self.bottom_up = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_scales - 1)
        ])

        # Top-down mixing (coarse → fine)
        self.top_down = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_scales - 1)
        ])

        # Per-scale FFN
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_scales)
        ])

        # Layer norms: bottom-up, top-down, and FFN
        self.norms_bu = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])
        self.norms_td = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])
        self.norms_ffn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])

    def forward(self, x_list):
        """
        参数:
            x_list: list of (B, N, d_model)，每个尺度一个

        返回:
            list of (B, N, d_model)，混合后的多尺度表示
        """
        # Bottom-up: 细粒度信息传递到粗粒度
        for i in range(self.n_scales - 1):
            x_list[i + 1] = self.norms_bu[i + 1](
                x_list[i + 1] + self.bottom_up[i](x_list[i])
            )

        # Top-down: 粗粒度信息传递到细粒度
        for i in range(self.n_scales - 2, -1, -1):
            x_list[i] = self.norms_td[i](
                x_list[i] + self.top_down[i](x_list[i + 1])
            )

        # Per-scale FFN + residual + norm
        out_list = []
        for i in range(self.n_scales):
            out = x_list[i] + self.ffns[i](x_list[i])
            out = self.norms_ffn[i](out)
            out_list.append(out)

        return out_list


class TimeMixerModel(nn.Module):
    """
    TimeMixer: 多尺度混合MLP架构 (ICLR 2024)

    核心思想：
    1. 多尺度特征提取: AvgPool1d 创建多分辨率特征视图
    2. 反转嵌入: 每个尺度 Linear(L_i, d_model)，将特征维度映射到隐藏空间
    3. 跨尺度混合: Bottom-up（细→粗）+ Top-down（粗→细）MLP混合
    4. 多尺度集成预测 (FMM): 每个尺度独立预测，求和平均
    5. 站点上下文: 全局平均池化提供空间感知

    与原始 TimeMixer 的适配：
    - 原始: seq_len=时间步, variates=传感器通道
    - 本项目: seq_len=特征维度(iDim), variates=气象站(nNodes)
    - 多尺度分解作用于特征维度: [1190, 595, 297, 148]
    - 反转嵌入: Linear(L_i, d_model) 替代 Conv1d embedding
    - 全MLP架构: 无注意力机制，计算更轻量

    接口: forward(x, edgeIdx=None, edgeAttr=None) 兼容现有训练代码
    """
    def __init__(self, modelPara):
        super(TimeMixerModel, self).__init__()
        self.nNodes = modelPara['nNodes']
        iDim = modelPara['iDim']
        oDim = modelPara['oDim']
        d_model = modelPara['HLD']

        # TimeMixer 特有参数
        down_sampling_layers = modelPara.get('down_sampling_layers', 3)
        down_sampling_window = modelPara.get('down_sampling_window', 2)
        e_layers = modelPara.get('nGNN', 2)  # PDM block 数量
        d_ff = modelPara.get('d_ff', d_model * 2)
        dropout = modelPara.get('dropout', 0.1)

        self.n_scales = down_sampling_layers + 1

        # 计算每个尺度的特征长度
        # e.g., iDim=1190: [1190, 595, 297, 148]
        seq_lengths = [iDim]
        for i in range(down_sampling_layers):
            seq_lengths.append(seq_lengths[-1] // down_sampling_window)
        self.seq_lengths = seq_lengths

        # 下采样层 (AvgPool1d)
        self.downsampling = nn.ModuleList([
            nn.AvgPool1d(kernel_size=down_sampling_window,
                         stride=down_sampling_window)
            for _ in range(down_sampling_layers)
        ])

        # 每个尺度的输入归一化
        self.input_norms = nn.ModuleList([
            nn.LayerNorm(L) for L in seq_lengths
        ])

        # 反转嵌入: Linear(L_i, d_model) per scale
        self.embeddings = nn.ModuleList([
            nn.Linear(L, d_model) for L in seq_lengths
        ])
        self.embed_dropout = nn.Dropout(dropout)

        # 多尺度混合块 (PDM-inspired)
        self.mixing_blocks = nn.ModuleList([
            MultiScaleMixingBlock(d_model, d_ff, self.n_scales, dropout)
            for _ in range(e_layers)
        ])

        # 站点上下文模块: 全局平均 → 投影 → 加回每个站点
        self.station_context = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.station_norm = nn.LayerNorm(d_model)

        # FMM: 每个尺度独立的输出投影
        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, oDim),
            )
            for _ in range(self.n_scales)
        ])

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        """
        参数:
            x: (B*N, iDim) — PyG flat format
            edgeIdx: 不使用
            edgeAttr: 不使用

        返回:
            (B*N, oDim)
        """
        N = self.nNodes
        B = x.shape[0] // N

        # Reshape: (B*N, iDim) → (B, N, iDim)
        x = x.view(B, N, -1)

        # --- 多尺度下采样 ---
        x_scales = [x]  # scale 0: (B, N, L_0)
        for pool in self.downsampling:
            x_prev = x_scales[-1]
            # (B, N, L_i) → (B*N, 1, L_i) → AvgPool → (B*N, 1, L_{i+1}) → (B, N, L_{i+1})
            x_down = pool(x_prev.reshape(B * N, 1, -1)).reshape(B, N, -1)
            x_scales.append(x_down)

        # --- 每个尺度的反转嵌入 ---
        enc_list = []
        for i in range(self.n_scales):
            x_s = self.input_norms[i](x_scales[i])
            x_s = self.embed_dropout(self.embeddings[i](x_s))
            enc_list.append(x_s)

        # --- 多尺度混合 (PDM) ---
        for block in self.mixing_blocks:
            enc_list = block(enc_list)

        # --- 站点上下文 (空间感知) ---
        for i in range(self.n_scales):
            # 全局平均 → 投影 → 加回每个站点
            ctx = self.station_context(enc_list[i].mean(dim=1, keepdim=True))
            enc_list[i] = self.station_norm(enc_list[i] + ctx)

        # --- FMM: 多尺度集成预测 ---
        dec_out = self.scale_projectors[0](enc_list[0])
        for i in range(1, self.n_scales):
            dec_out = dec_out + self.scale_projectors[i](enc_list[i])
        dec_out = dec_out / self.n_scales

        # Reshape back: (B, N, oDim) → (B*N, oDim)
        return dec_out.reshape(B * N, -1)


# =============================================================================
# 7. ModernTCN Model (Large Kernel Conv - ICLR 2024)
# =============================================================================
class ModernTCNBlock(nn.Module):
    """
    ModernTCN 基本块

    核心操作（受 ModernTCN ICLR 2024 启发）：
    - 大核深度可分离卷积 (DWConv): 沿站点维度捕捉空间模式
    - 小核重参数化: 训练时大核+小核并行，改善优化
    - ConvFFN: 逐点卷积（等价于 per-position Linear）处理特征

    输入/输出: (B, d_model, N) 其中 N=站点数
    """
    def __init__(self, d_model, d_ff, large_kernel, small_kernel, dropout):
        super(ModernTCNBlock, self).__init__()

        # 大核 DWConv: 沿站点维度卷积，每个 d_model 通道独立
        self.dw_large = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=large_kernel,
                      stride=1, padding=large_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
        )

        # 小核 DWConv (重参数化分支)
        self.dw_small = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=small_kernel,
                      stride=1, padding=small_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
        )

        # DWConv 后的 dropout（防止空间卷积过拟合）
        self.dw_dropout = nn.Dropout(dropout)

        # 归一化
        self.norm = nn.BatchNorm1d(d_model)

        # ConvFFN: 逐点卷积 (kernel=1)，等价于 per-position Linear
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, d_model, N)"""
        residual = x

        # DWConv: 大核 + 小核重参数化
        x = self.dw_large(x) + self.dw_small(x)
        x = self.dw_dropout(x)  # 正则化空间卷积输出
        x = self.norm(x)

        # ConvFFN
        x = self.ffn(x)

        return residual + x


class ModernTCNModel(nn.Module):
    """
    ModernTCN: 大核卷积时序模型 (ICLR 2024)

    核心思想：
    1. 反转嵌入: Linear(iDim, d_model)，每个站点的特征映射到隐藏空间
    2. 大核 DWConv: 沿站点维度做深度可分离卷积，捕捉空间模式
    3. 重参数化: 大核 + 小核 并行，改善训练
    4. ConvFFN: 逐点卷积处理每个位置的特征
    5. 多层堆叠: 扩大有效感受野

    与原始 ModernTCN 的适配：
    - 原始: DWConv沿时间维度, FFN2做跨变量交互
    - 本项目: DWConv沿站点维度(空间建模), FFN做特征处理
    - 反转嵌入替代 patch embedding
    - 默认 large_kernel=13, 适中感受野避免过拟合
    - DWConv 后加 Dropout 防止空间卷积记忆训练集
    - 输出前加 LayerNorm 稳定预测分布

    接口: forward(x, edgeIdx=None, edgeAttr=None) 兼容现有训练代码
    """
    def __init__(self, modelPara):
        super(ModernTCNModel, self).__init__()
        self.nNodes = modelPara['nNodes']
        iDim = modelPara['iDim']
        oDim = modelPara['oDim']
        d_model = modelPara['HLD']

        # ModernTCN 特有参数（默认值已调整为正则化版本）
        large_kernel = modelPara.get('large_kernel', 13)
        small_kernel = modelPara.get('small_kernel', 5)
        e_layers = modelPara.get('nGNN', 2)
        d_ff = modelPara.get('d_ff', d_model * 2)
        dropout = modelPara.get('dropout', 0.3)

        # --- 反转嵌入 ---
        self.input_norm = nn.LayerNorm(iDim)
        self.embedding = nn.Linear(iDim, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # --- ModernTCN Blocks ---
        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, d_ff, large_kernel, small_kernel, dropout)
            for _ in range(e_layers)
        ])

        # --- 输出前归一化（稳定预测分布） ---
        self.output_norm = nn.LayerNorm(d_model)

        # --- 输出投影 ---
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, oDim),
        )

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        """
        参数:
            x: (B*N, iDim) — PyG flat format
            edgeIdx: 不使用
            edgeAttr: 不使用

        返回:
            (B*N, oDim)
        """
        N = self.nNodes
        B = x.shape[0] // N

        # Reshape: (B*N, iDim) → (B, N, iDim)
        x = x.view(B, N, -1)

        # 反转嵌入: (B, N, iDim) → (B, N, d_model)
        x = self.input_norm(x)
        x = self.embed_dropout(self.embedding(x))

        # 转置为 Conv1d 格式: (B, N, d_model) → (B, d_model, N)
        x = x.permute(0, 2, 1)

        # ModernTCN blocks: 大核 DWConv + ConvFFN
        for block in self.blocks:
            x = block(x)

        # 转置回来: (B, d_model, N) → (B, N, d_model)
        x = x.permute(0, 2, 1)

        # 输出前归一化
        x = self.output_norm(x)

        # 输出投影: (B, N, d_model) → (B, N, oDim)
        x = self.projector(x)

        # Reshape back: (B, N, oDim) → (B*N, oDim)
        return x.reshape(B * N, -1)


# =============================================================================
# 8. S-Mamba Model (Bidirectional Mamba - Neurocomputing 2024)
# =============================================================================
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[WARNING] mamba_ssm 未安装，SMambaModel 不可用。请运行: pip install mamba-ssm")


class SMambaEncoderLayer(nn.Module):
    """
    S-Mamba 编码器层: 双向Mamba + FFN

    核心操作：
    - 前向 Mamba: 沿站点序列正向扫描
    - 反向 Mamba: 沿站点序列反向扫描（flip → Mamba → flip）
    - 双向融合: forward + backward
    - FFN: Conv1d(d_model, d_ff) → GELU → Conv1d(d_ff, d_model)

    输入/输出: (B, N, d_model)
    """
    def __init__(self, d_model, d_state, d_ff, dropout, activation='gelu'):
        super(SMambaEncoderLayer, self).__init__()
        assert MAMBA_AVAILABLE, "mamba_ssm 未安装，无法使用 SMambaEncoderLayer"

        # 双向 Mamba
        self.mamba_forward = Mamba(
            d_model=d_model, d_state=d_state, d_conv=2, expand=1
        )
        self.mamba_backward = Mamba(
            d_model=d_model, d_state=d_state, d_conv=2, expand=1
        )

        # FFN (与原始 S-Mamba 一致: Conv1d pointwise)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x):
        """x: (B, N, d_model)"""
        # 双向 Mamba: 前向 + 反向
        new_x = self.mamba_forward(x) + self.mamba_backward(x.flip(dims=[1])).flip(dims=[1])
        x = x + new_x
        y = x = self.norm1(x)

        # FFN
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class SMambaModel(nn.Module):
    """
    S-Mamba: 双向Mamba时序模型 (Neurocomputing 2024)

    核心思想：
    1. 反转嵌入: Linear(iDim, d_model)，每个站点的特征映射到隐藏空间
    2. 双向 Mamba: 前向+反向选择性状态空间扫描，捕捉站点间双向依赖
    3. FFN: Conv1d 逐点卷积处理特征
    4. 多层堆叠: 逐步提取更高层次的空间-特征模式

    与原始 S-Mamba 的适配：
    - 原始: variates作为tokens, seq_len作为embedding维度
    - 本项目: 站点(nNodes)作为tokens, 特征(iDim)作为embedding维度
    - 双向 Mamba 沿站点维度扫描: 捕捉空间依赖（不依赖站点顺序固定）
    - Mamba 对任意序列长度都支持: 天然兼容 68↔200 站点切换

    接口: forward(x, edgeIdx=None, edgeAttr=None) 兼容现有训练代码
    """
    def __init__(self, modelPara):
        super(SMambaModel, self).__init__()
        assert MAMBA_AVAILABLE, "mamba_ssm 未安装，无法使用 SMambaModel"

        self.nNodes = modelPara['nNodes']
        iDim = modelPara['iDim']
        oDim = modelPara['oDim']
        d_model = modelPara['HLD']

        # S-Mamba 特有参数
        d_state = modelPara.get('d_state', 16)
        d_ff = modelPara.get('d_ff', d_model * 4)
        e_layers = modelPara.get('nGNN', 3)
        dropout = modelPara.get('dropout', 0.1)

        # --- 反转嵌入 ---
        self.input_norm = nn.LayerNorm(iDim)
        self.embedding = nn.Linear(iDim, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # --- S-Mamba 编码器层 ---
        self.layers = nn.ModuleList([
            SMambaEncoderLayer(d_model, d_state, d_ff, dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # --- 输出投影 ---
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, oDim),
        )

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        """
        参数:
            x: (B*N, iDim) — PyG flat format
            edgeIdx: 不使用
            edgeAttr: 不使用

        返回:
            (B*N, oDim)
        """
        N = self.nNodes
        B = x.shape[0] // N

        # Reshape: (B*N, iDim) → (B, N, iDim)
        x = x.view(B, N, -1)

        # 反转嵌入: (B, N, iDim) → (B, N, d_model)
        x = self.input_norm(x)
        x = self.embed_dropout(self.embedding(x))

        # S-Mamba 编码器: 双向 Mamba + FFN
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # 输出投影: (B, N, d_model) → (B, N, oDim)
        x = self.projector(x)

        # Reshape back: (B, N, oDim) → (B*N, oDim)
        return x.reshape(B * N, -1)

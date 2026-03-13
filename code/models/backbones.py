"""
Backbone models for urban temperature prediction.
All models share the same forward(x, edgeIdx, edgeAttr) interface for compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. Pure MLP (Baseline - no spatial modeling)
# =============================================================================
class MLPModel(nn.Module):
    """Pure MLP: each node processed independently, no spatial interaction."""
    def __init__(self, modelPara):
        super(MLPModel, self).__init__()
        self.nProcessorLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        _encoder, _processor, _decoder = [], [], []

        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _encoder.append(nn.Linear(_inputChannel, _HLD))
            _encoder.append(nn.PReLU(_HLD))

        for _n in range(self.nProcessorLayers):
            _processor.append(nn.Linear(_HLD, _HLD))
            _processor.append(nn.PReLU(_HLD))

        for _n in range(self.nMLPLayers):
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(_HLD, _outputChannel))
            _decoder.append(nn.PReLU(_outputChannel))

        self.encoder = nn.ModuleList(_encoder)
        self.processor = nn.ModuleList(_processor)
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        for _f in self.encoder:
            x = _f(x)
        for _f in self.processor:
            x = _f(x)
        for _f in self.decoder:
            x = _f(x)
        return x


# =============================================================================
# 2. Self-Attention Model (Transformer Processor)
# =============================================================================
class SelfAttentionProcessor(nn.Module):
    """Single Self-Attention layer: nodes interact via attention."""
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super(SelfAttentionProcessor, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerModel(nn.Module):
    """Transformer: Self-Attention replaces SAGEConv for spatial modeling."""
    def __init__(self, modelPara):
        super(TransformerModel, self).__init__()
        self.nProcessorLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        self.nNodes = modelPara.get('nNodes', None)

        _encoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _encoder.append(nn.Linear(_inputChannel, _HLD))
            _encoder.append(nn.PReLU(_HLD))
        self.encoder = nn.ModuleList(_encoder)

        nhead = modelPara.get('nhead', 4)
        dropout = modelPara.get('dropout', 0.1)
        self.processor = nn.ModuleList([
            SelfAttentionProcessor(_HLD, nhead=nhead, dropout=dropout)
            for _ in range(self.nProcessorLayers)
        ])

        _decoder = []
        for _n in range(self.nMLPLayers):
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(_HLD, _outputChannel))
            _decoder.append(nn.PReLU(_outputChannel))
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        nNodes = self.nNodes
        for _f in self.encoder:
            x = _f(x)
        B = x.shape[0] // nNodes
        x = x.view(B, nNodes, -1)
        for layer in self.processor:
            x = layer(x)
        x = x.view(B * nNodes, -1)
        for _f in self.decoder:
            x = _f(x)
        return x


# =============================================================================
# 3. 1D CNN Model (local spatial patterns)
# =============================================================================
class CNNModel(nn.Module):
    """1D CNN: convolution along node dimension for local spatial patterns."""
    def __init__(self, modelPara):
        super(CNNModel, self).__init__()
        self.nProcessorLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        self.nNodes = modelPara.get('nNodes', None)

        _encoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _encoder.append(nn.Linear(_inputChannel, _HLD))
            _encoder.append(nn.PReLU(_HLD))
        self.encoder = nn.ModuleList(_encoder)

        _processor = []
        for _n in range(self.nProcessorLayers):
            _processor.append(nn.Conv1d(_HLD, _HLD, kernel_size=3, padding=1))
            _processor.append(nn.PReLU(_HLD))
        self.processor = nn.ModuleList(_processor)

        _decoder = []
        for _n in range(self.nMLPLayers):
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(_HLD, _outputChannel))
            _decoder.append(nn.PReLU(_outputChannel))
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        nNodes = self.nNodes
        for _f in self.encoder:
            x = _f(x)
        B = x.shape[0] // nNodes
        x = x.view(B, nNodes, -1).permute(0, 2, 1)
        for _n, _f in enumerate(self.processor):
            x = _f(x)
        x = x.permute(0, 2, 1).contiguous().view(B * nNodes, -1)
        for _f in self.decoder:
            x = _f(x)
        return x


# =============================================================================
# 4. LSTM Model (BiLSTM along node sequence)
# =============================================================================
class LSTMModel(nn.Module):
    """BiLSTM: treats node sequence as temporal sequence for spatial modeling."""
    def __init__(self, modelPara):
        super(LSTMModel, self).__init__()
        self.nProcessorLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        self.nNodes = modelPara.get('nNodes', None)

        _encoder = []
        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _encoder.append(nn.Linear(_inputChannel, _HLD))
            _encoder.append(nn.PReLU(_HLD))
        self.encoder = nn.ModuleList(_encoder)

        self.lstm = nn.LSTM(
            input_size=_HLD, hidden_size=_HLD // 2,
            num_layers=self.nProcessorLayers, batch_first=True,
            bidirectional=True, dropout=0.1 if self.nProcessorLayers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(_HLD)

        _decoder = []
        for _n in range(self.nMLPLayers):
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(_HLD, _outputChannel))
            _decoder.append(nn.PReLU(_outputChannel))
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        nNodes = self.nNodes
        for _f in self.encoder:
            x = _f(x)
        B = x.shape[0] // nNodes
        x_reshaped = x.view(B, nNodes, -1)
        lstm_out, _ = self.lstm(x_reshaped)
        x = self.lstm_norm(x_reshaped + lstm_out)
        x = x.contiguous().view(B * nNodes, -1)
        for _f in self.decoder:
            x = _f(x)
        return x


# =============================================================================
# 5. iTransformer Model (Inverted Transformer - ICLR 2024)
# =============================================================================
class iTransformerModel(nn.Module):
    """
    iTransformer: Inverted Transformer (ICLR 2024)

    Each weather station = one token, feature dimension = "sequence".
    Attention computed across stations → learns spatial relationships.
    """
    def __init__(self, modelPara):
        super(iTransformerModel, self).__init__()
        self.nNodes = modelPara['nNodes']
        iDim = modelPara['iDim']
        oDim = modelPara['oDim']
        d_model = modelPara['HLD']
        n_heads = modelPara.get('nhead', 8)
        e_layers = modelPara.get('nGNN', 3)
        d_ff = modelPara.get('d_ff', d_model * 4)
        dropout = modelPara.get('dropout', 0.1)

        # Inverted Embedding
        self.input_norm = nn.LayerNorm(iDim)
        self.embedding = nn.Linear(iDim, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer Encoder
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
            norm=nn.LayerNorm(d_model),
        )

        # Output Projection
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, oDim),
        )

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        N = self.nNodes
        B = x.shape[0] // N
        x = x.view(B, N, -1)
        x = self.input_norm(x)
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = self.encoder(x)
        x = self.projector(x)
        return x.reshape(B * N, -1)


# =============================================================================
# 2. TimeMixer Model (Multi-Scale Mixing MLP - ICLR 2024)
# =============================================================================
class MultiScaleMixingBlock(nn.Module):
    """
    Multi-scale mixing block (inspired by TimeMixer PDM).
    Bottom-up mixing: fine → coarse
    Top-down mixing: coarse → fine
    Per-scale FFN
    """
    def __init__(self, d_model, d_ff, n_scales, dropout):
        super(MultiScaleMixingBlock, self).__init__()
        self.n_scales = n_scales

        self.bottom_up = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            ) for _ in range(n_scales - 1)
        ])

        self.top_down = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            ) for _ in range(n_scales - 1)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout),
            ) for _ in range(n_scales)
        ])

        self.norms_bu = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])
        self.norms_td = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])
        self.norms_ffn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])

    def forward(self, x_list):
        for i in range(self.n_scales - 1):
            x_list[i + 1] = self.norms_bu[i + 1](
                x_list[i + 1] + self.bottom_up[i](x_list[i])
            )
        for i in range(self.n_scales - 2, -1, -1):
            x_list[i] = self.norms_td[i](
                x_list[i] + self.top_down[i](x_list[i + 1])
            )
        out_list = []
        for i in range(self.n_scales):
            out = x_list[i] + self.ffns[i](x_list[i])
            out = self.norms_ffn[i](out)
            out_list.append(out)
        return out_list


class TimeMixerModel(nn.Module):
    """
    TimeMixer: Multi-Scale Mixing MLP (ICLR 2024)

    Multi-scale feature extraction via AvgPool1d, inverted embedding per scale,
    cross-scale mixing (bottom-up + top-down), and multi-scale ensemble prediction.
    """
    def __init__(self, modelPara):
        super(TimeMixerModel, self).__init__()
        self.nNodes = modelPara['nNodes']
        iDim = modelPara['iDim']
        oDim = modelPara['oDim']
        d_model = modelPara['HLD']

        down_sampling_layers = modelPara.get('down_sampling_layers', 3)
        down_sampling_window = modelPara.get('down_sampling_window', 2)
        e_layers = modelPara.get('nGNN', 2)
        d_ff = modelPara.get('d_ff', d_model * 2)
        dropout = modelPara.get('dropout', 0.1)

        self.n_scales = down_sampling_layers + 1

        seq_lengths = [iDim]
        for i in range(down_sampling_layers):
            seq_lengths.append(seq_lengths[-1] // down_sampling_window)
        self.seq_lengths = seq_lengths

        self.downsampling = nn.ModuleList([
            nn.AvgPool1d(kernel_size=down_sampling_window, stride=down_sampling_window)
            for _ in range(down_sampling_layers)
        ])

        self.input_norms = nn.ModuleList([nn.LayerNorm(L) for L in seq_lengths])
        self.embeddings = nn.ModuleList([nn.Linear(L, d_model) for L in seq_lengths])
        self.embed_dropout = nn.Dropout(dropout)

        self.mixing_blocks = nn.ModuleList([
            MultiScaleMixingBlock(d_model, d_ff, self.n_scales, dropout)
            for _ in range(e_layers)
        ])

        self.station_context = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.station_norm = nn.LayerNorm(d_model)

        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, oDim),
            ) for _ in range(self.n_scales)
        ])

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        N = self.nNodes
        B = x.shape[0] // N
        x = x.view(B, N, -1)

        x_scales = [x]
        for pool in self.downsampling:
            x_down = pool(x_scales[-1].reshape(B * N, 1, -1)).reshape(B, N, -1)
            x_scales.append(x_down)

        enc_list = []
        for i in range(self.n_scales):
            x_s = self.input_norms[i](x_scales[i])
            x_s = self.embed_dropout(self.embeddings[i](x_s))
            enc_list.append(x_s)

        for block in self.mixing_blocks:
            enc_list = block(enc_list)

        for i in range(self.n_scales):
            ctx = self.station_context(enc_list[i].mean(dim=1, keepdim=True))
            enc_list[i] = self.station_norm(enc_list[i] + ctx)

        dec_out = self.scale_projectors[0](enc_list[0])
        for i in range(1, self.n_scales):
            dec_out = dec_out + self.scale_projectors[i](enc_list[i])
        dec_out = dec_out / self.n_scales

        return dec_out.reshape(B * N, -1)


# =============================================================================
# 3. ModernTCN Model (Large Kernel Conv - ICLR 2024)
# =============================================================================
class ModernTCNBlock(nn.Module):
    """
    ModernTCN block: Large-kernel DWConv + small-kernel re-parameterization + ConvFFN.
    """
    def __init__(self, d_model, d_ff, large_kernel, small_kernel, dropout):
        super(ModernTCNBlock, self).__init__()

        self.dw_large = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=large_kernel,
                      stride=1, padding=large_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
        )

        self.dw_small = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=small_kernel,
                      stride=1, padding=small_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
        )

        self.dw_dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)

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
        x = self.dw_large(x) + self.dw_small(x)
        x = self.dw_dropout(x)
        x = self.norm(x)
        x = self.ffn(x)
        return residual + x


class ModernTCNModel(nn.Module):
    """
    ModernTCN: Large Kernel Conv (ICLR 2024)

    Inverted embedding, large-kernel DWConv along station dimension,
    re-parameterization (large + small kernel), and ConvFFN.
    """
    def __init__(self, modelPara):
        super(ModernTCNModel, self).__init__()
        self.nNodes = modelPara['nNodes']
        iDim = modelPara['iDim']
        oDim = modelPara['oDim']
        d_model = modelPara['HLD']

        large_kernel = modelPara.get('large_kernel', 13)
        small_kernel = modelPara.get('small_kernel', 5)
        e_layers = modelPara.get('nGNN', 2)
        d_ff = modelPara.get('d_ff', d_model * 2)
        dropout = modelPara.get('dropout', 0.3)

        self.input_norm = nn.LayerNorm(iDim)
        self.embedding = nn.Linear(iDim, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, d_ff, large_kernel, small_kernel, dropout)
            for _ in range(e_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, oDim),
        )

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        N = self.nNodes
        B = x.shape[0] // N
        x = x.view(B, N, -1)
        x = self.input_norm(x)
        x = self.embed_dropout(self.embedding(x))
        x = x.permute(0, 2, 1)  # (B, d_model, N) for Conv1d
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # back to (B, N, d_model)
        x = self.output_norm(x)
        x = self.projector(x)
        return x.reshape(B * N, -1)


# =============================================================================
# 8. S-Mamba Model (Bidirectional Mamba - Neurocomputing 2024)
# =============================================================================
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class SMambaEncoderLayer(nn.Module):
    """S-Mamba encoder layer: Bidirectional Mamba + FFN."""
    def __init__(self, d_model, d_state, d_ff, dropout, activation='gelu'):
        super(SMambaEncoderLayer, self).__init__()
        assert MAMBA_AVAILABLE, "mamba_ssm not installed"
        self.mamba_forward = Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1)
        self.mamba_backward = Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x):
        new_x = self.mamba_forward(x) + self.mamba_backward(x.flip(dims=[1])).flip(dims=[1])
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class SMambaModel(nn.Module):
    """S-Mamba: Bidirectional Mamba (Neurocomputing 2024)."""
    def __init__(self, modelPara):
        super(SMambaModel, self).__init__()
        assert MAMBA_AVAILABLE, "mamba_ssm not installed"
        self.nNodes = modelPara['nNodes']
        iDim = modelPara['iDim']
        oDim = modelPara['oDim']
        d_model = modelPara['HLD']
        d_state = modelPara.get('d_state', 16)
        d_ff = modelPara.get('d_ff', d_model * 4)
        e_layers = modelPara.get('nGNN', 3)
        dropout = modelPara.get('dropout', 0.1)

        self.input_norm = nn.LayerNorm(iDim)
        self.embedding = nn.Linear(iDim, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            SMambaEncoderLayer(d_model, d_state, d_ff, dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, oDim),
        )

    def forward(self, x, edgeIdx=None, edgeAttr=None):
        N = self.nNodes
        B = x.shape[0] // N
        x = x.view(B, N, -1)
        x = self.input_norm(x)
        x = self.embed_dropout(self.embedding(x))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.projector(x)
        return x.reshape(B * N, -1)

"""
GNN Model (SAGEConv or GraphConv encoder-processor-decoder)
- GNN: base model, unified MLP encoder (all features concatenated)
- FeatureGNN: ESTNet-style dynamic/static separation (GRU + MLP branches)
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GraphConv


class GNN(torch.nn.Module):

    def __init__(self, modelPara):
        super(GNN, self).__init__()
        self.nGNNLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        self.conv_type = modelPara.get('conv_type', 'sage')
        _encoder, _processor, _decoder = [], [], []

        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _outputChannel = _HLD
            _encoder.append(torch.nn.Linear(_inputChannel, _outputChannel))
            _encoder.append(torch.nn.PReLU(_outputChannel))

        for _n in range(self.nGNNLayers):
            _inputChannel = _HLD
            _outputChannel = _HLD
            if self.conv_type == 'graphconv':
                _processor.append(GraphConv(_inputChannel, _outputChannel, aggr="mean"))
            else:
                _processor.append(SAGEConv(_inputChannel, _outputChannel, aggr="mean"))
            _processor.append(torch.nn.PReLU(_outputChannel))

        for _n in range(self.nMLPLayers):
            _inputChannel = _HLD
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(torch.nn.Linear(_inputChannel, _outputChannel))
            _decoder.append(torch.nn.PReLU(_outputChannel))

        self.encoder = torch.nn.ModuleList(_encoder)
        self.processor = torch.nn.ModuleList(_processor)
        self.decoder = torch.nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx, edgeAttr):
        for _f in self.encoder:
            x = _f(x)
        for _n, _f in enumerate(self.processor):
            if not _n % 2:
                x = _f(x, edgeIdx, edgeAttr) if self.conv_type == 'graphconv' else _f(x, edgeIdx)
            else:
                x = _f(x)
        for _f in self.decoder:
            x = _f(x)
        return x


class FeatureGNN(torch.nn.Module):
    """
    Feature-aware GNN (ESTNet-style dynamic/static separation):
    - Dynamic branch: WRF(315) + CLMS(3) → GRU encoder (captures temporal evolution)
    - Static branch: UrbanFeature(17) + GeoEmbed → MLP encoder (spatial/urban context)
    - Fusion → shared GNN processor → decoder
    """

    def __init__(self, modelPara):
        super(FeatureGNN, self).__init__()
        _HLD = modelPara['HLD']
        self.conv_type = modelPara.get('conv_type', 'graphconv')
        self.nGNNLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']

        # Feature split dimensions (V2: WRF=63 no merge, CLMS=3, no station feats)
        self.n_wrf_channels = modelPara.get('n_wrf_channels', 63)
        window = modelPara.get('window', 2)
        self.n_timesteps = 2 * window + 1  # 5 (history + current + future)
        self.wrf_dim = self.n_wrf_channels * self.n_timesteps  # 315
        self.clms_dim = modelPara.get('clms_dim', 3)
        # Dynamic = WRF + CLMS
        self.dynamic_dim = self.wrf_dim + self.clms_dim  # 318
        # Static = everything else (RawGeo + GeoEmbed)
        self.static_dim = modelPara['iDim'] - self.dynamic_dim

        # ---------- Dynamic branch: GRU ----------
        # Reshape WRF into (n_timesteps, n_wrf_channels), concat CLMS at each step
        # Input per step: 63 + 3 = 66 (CLMS repeated at each timestep)
        gru_input_dim = self.n_wrf_channels + self.clms_dim  # 66
        gru_hidden = 64
        self.dynamic_gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dynamic_out_dim = gru_hidden

        # ---------- Static branch: MLP ----------
        static_hidden = 64
        self.static_encoder = nn.Sequential(
            nn.Linear(self.static_dim, static_hidden),
            nn.PReLU(static_hidden),
            nn.Linear(static_hidden, static_hidden),
            nn.PReLU(static_hidden),
        )
        self.static_out_dim = static_hidden

        # ---------- GNN processor ----------
        fusion_dim = self.dynamic_out_dim + self.static_out_dim  # 64+64=128
        _processor = []
        for _n in range(self.nGNNLayers):
            in_dim = fusion_dim if _n == 0 else _HLD
            if self.conv_type == 'graphconv':
                _processor.append(GraphConv(in_dim, _HLD, aggr="mean"))
            else:
                _processor.append(SAGEConv(in_dim, _HLD, aggr="mean"))
            _processor.append(nn.PReLU(_HLD))
        self.processor = nn.ModuleList(_processor)

        # ---------- Decoder ----------
        _decoder = []
        for _n in range(self.nMLPLayers):
            in_dim = _HLD
            out_dim = modelPara['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            _decoder.append(nn.Linear(in_dim, out_dim))
            _decoder.append(nn.PReLU(out_dim))
        self.decoder = nn.ModuleList(_decoder)

    def forward(self, x, edgeIdx, edgeAttr):
        # Split into dynamic and static
        x_dynamic = x[:, :self.dynamic_dim]   # (N, 318) = WRF(315) + CLMS(3)
        x_static = x[:, self.dynamic_dim:]    # (N, static_dim) = UrbanFeature + GeoEmbed

        # --- Dynamic branch ---
        # Split WRF and CLMS
        x_wrf = x_dynamic[:, :self.wrf_dim]           # (N, 315)
        x_clms = x_dynamic[:, self.wrf_dim:]           # (N, 3)

        # Reshape WRF to (N, 5, 63) — 5 timesteps × 63 channels
        x_wrf = x_wrf.view(-1, self.n_timesteps, self.n_wrf_channels)  # (N, 5, 63)

        # Repeat CLMS at each timestep and concat
        x_clms_expanded = x_clms.unsqueeze(1).expand(-1, self.n_timesteps, -1)  # (N, 5, 3)
        x_seq = torch.cat([x_wrf, x_clms_expanded], dim=-1)  # (N, 5, 66)

        # GRU: take last hidden state as dynamic representation
        _, h_dynamic = self.dynamic_gru(x_seq)  # h_dynamic: (1, N, gru_hidden)
        h_dynamic = h_dynamic.squeeze(0)          # (N, gru_hidden)

        # --- Static branch ---
        h_static = self.static_encoder(x_static)  # (N, static_hidden)

        # --- Fusion → GNN → Decoder ---
        h = torch.cat([h_dynamic, h_static], dim=-1)  # (N, 128)

        for _n, _f in enumerate(self.processor):
            if not _n % 2:
                h = _f(h, edgeIdx, edgeAttr) if self.conv_type == 'graphconv' else _f(h, edgeIdx)
            else:
                h = _f(h)

        for _f in self.decoder:
            h = _f(h)

        return h

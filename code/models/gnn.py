"""
GNN Model (SAGEConv-based encoder-processor-decoder)
"""
import torch
from torch_geometric.nn import SAGEConv


class GNN(torch.nn.Module):

    def __init__(self, modelPara):
        super(GNN, self).__init__()
        self.nGNNLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        _encoder, _processor, _decoder = [], [], []

        for _n in range(self.nMLPLayers):
            _inputChannel = modelPara['iDim'] if _n == 0 else _HLD
            _outputChannel = _HLD
            _encoder.append(torch.nn.Linear(_inputChannel, _outputChannel))
            _encoder.append(torch.nn.PReLU(_outputChannel))

        for _n in range(self.nGNNLayers):
            _inputChannel = _HLD
            _outputChannel = _HLD
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
            x = _f(x, edgeIdx) if not _n % 2 else _f(x)
        for _f in self.decoder:
            x = _f(x)
        return x

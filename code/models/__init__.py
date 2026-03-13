from .gnn import GNN
from .backbones import (
    MLPModel, TransformerModel, CNNModel, LSTMModel,
    iTransformerModel, TimeMixerModel, ModernTCNModel,
)

# SMamba requires mamba_ssm; import conditionally
try:
    from .backbones import SMambaModel
except (ImportError, AssertionError):
    pass

"""
模型定义模块
包含GNN_ESTNet模型架构
"""
import torch
from torch_geometric.nn import SAGEConv


class GNN_ESTNet(torch.nn.Module):
    def __init__(self, modelPara):
        super(GNN_ESTNet, self).__init__()
        self.nGNNLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
        _HLD = modelPara['HLD']
        
        # 1. 动态特征编码器
        self.dynamic_encoder = torch.nn.GRU(
            input_size=modelPara['dynamic_dim'],
            hidden_size=_HLD,
            batch_first=True
        )
        
        # 2. 静态特征编码器
        self.static_encoder = torch.nn.Linear(modelPara['static_dim'], _HLD)
        self.static_encoder_act = torch.nn.PReLU()
        
        # 3. 静态特征多尺度GCN
        self.static_gcns = torch.nn.ModuleList()
        for _ in range(self.nGNNLayers):
            self.static_gcns.append(SAGEConv(_HLD, _HLD, aggr="mean"))
            self.static_gcns.append(torch.nn.PReLU())
        
        # 4. 特征融合层
        self.fusion = torch.nn.Linear(_HLD * 2, _HLD)
        self.fusion_act = torch.nn.PReLU()
        
        # 5. 图处理层
        self.graph_processors = torch.nn.ModuleList()
        for _ in range(self.nGNNLayers):
            self.graph_processors.append(SAGEConv(_HLD, _HLD, aggr="mean"))
            self.graph_processors.append(torch.nn.PReLU())
        
        # 6. 输出解码器
        self.decoder = torch.nn.ModuleList()
        for _n in range(self.nMLPLayers):
            _inputChannel = _HLD
            _outputChannel = modelPara['oDim'] if _n == self.nMLPLayers-1 else _HLD
            self.decoder.append(torch.nn.Linear(_inputChannel, _outputChannel))
            if _n < self.nMLPLayers-1:  # 最后一层不需要激活函数
                self.decoder.append(torch.nn.PReLU())
    
    def forward(self, x_dynamic, x_static, edge_index, edge_attr=None):
        # 动态特征处理
        # 增加批次维度以适应GRU
        batch_size = x_dynamic.size(0)
        x_dynamic = x_dynamic.unsqueeze(1)  # [N, 1, F_dyn]
        
        # 1. 处理动态特征：通过GRU编码时序信息
        dynamic_out, _ = self.dynamic_encoder(x_dynamic)
        dynamic_out = dynamic_out.squeeze(1)  # [N, H]
        
        # 2. 处理静态特征：初始编码
        static_out = self.static_encoder(x_static)
        static_out = self.static_encoder_act(static_out)
        
        # 3. 通过多尺度GCN处理静态特征
        static_original = static_out.clone()  # 保存初始特征用于残差连接
        for i, layer in enumerate(self.static_gcns):
            if i % 2 == 0:  # GCN层
                static_out = layer(static_out, edge_index)
            else:  # 激活函数
                static_out = layer(static_out)
        
        # 残差连接增强表示能力
        static_out = static_out + static_original
        
        # 4. 融合特征：将静态和动态特征连接并融合
        fused_features = torch.cat([dynamic_out, static_out], dim=1)
        fused_features = self.fusion(fused_features)
        fused_features = self.fusion_act(fused_features)
        
        # 5. 图处理：通过图卷积处理融合特征
        fused_original = fused_features.clone()  # 用于残差连接
        for i, layer in enumerate(self.graph_processors):
            if i % 2 == 0:  # GCN层
                fused_features = layer(fused_features, edge_index)
            else:  # 激活函数
                fused_features = layer(fused_features)
        
        # 残差连接
        fused_features = fused_features + fused_original
        
        # 6. 解码：最终输出层
        for layer in self.decoder:
            fused_features = layer(fused_features)
        
        return fused_features


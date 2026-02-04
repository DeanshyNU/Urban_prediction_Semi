import torch,os,utils
import numpy as np
from torch_geometric.nn import GraphConv
from torch_geometric.nn import (
    GATConv,
    SAGEConv, 
    TransformerConv
)
# -----------------Construct GNN model---------------------------  
class GNN(torch.nn.Module):

    def __init__(self,modelPara):
        super(GNN, self).__init__()
        self.nGNNLayers = modelPara['nGNN']
        self.nMLPLayers = modelPara['nMLP']
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
            # _processor.append(GraphConv(_inputChannel,_outputChannel,aggr='mean'))
            # Option 1: GATConv
            # _processor.append(GATConv(_inputChannel, _outputChannel, 
            #                         heads=4, edge_dim=1, concat=False))
            
            # Option 2: SAGEConv
            _processor.append(SAGEConv(_inputChannel, _outputChannel,aggr = "mean"))
            
            # Option 3: TransformerConv
            # _processor.append(TransformerConv(_inputChannel, _outputChannel, 
            #                                  heads=1, edge_dim=None, concat = False, beta = True))
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
        # x_copy = x.clone()
        for _n,_f in enumerate(self.processor):
                x = _f(x, edgeIdx) if not _n%2 else _f(x) 
                # x = _f(x, edgeIdx, edgeAttr) if not _n%2 else _f(x)
        # x = x + x_copy # skip connection
        for _f in self.decoder:
            x = _f(x)
        return x
    
def train(loader,model,lossFn,opt,scheduler,device,nNodes):
    model.train()
    _LOSS = 0
    pred,truth = [],[]
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        _loss = lossFn(_yHat,_batch.y)
        _loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += _loss
        _pred = _yHat.reshape(-1,nNodes)
        _truth = _batch.y.reshape(-1,nNodes)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth,pred)
    return (_LOSS/(_n+1)).item(), _RMSE, truth, pred

def test(loader,model,lossFn,device,nNodes):
    model.eval()
    _LOSS = 0
    pred,truth = [],[]
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        _loss = lossFn(_yHat,_batch.y)
        _LOSS += _loss
        _pred = _yHat.reshape(-1,nNodes)
        _truth = _batch.y.reshape(-1,nNodes)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth,pred)
    return (_LOSS/(_n+1)).item(), _RMSE, truth, pred

def loadCheckPoint(modelName,model,opt,device,load=False,resetLr=False,lr=5e-5,predMode=False):
    chkptPath = f'./{modelName}.pt'
    if os.path.exists(chkptPath) and load:
        chkpt = torch.load(chkptPath,map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        opt.load_state_dict(chkpt['opt_state_dict'])
        EPOCH = chkpt['epoch']
        bestLoss = chkpt['bestLoss']
        hist = chkpt['hist']
        with open(f'./{modelName}_log','a') as f: print("Checkpoint loaded.")
        if opt.param_groups[0]['lr'] < 1e-6 and resetLr:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            with open(f'./{modelName}_log','a') as f: print(f"Resetting LR from {opt.param_groups[0]['lr']} to {lr}",file=f)
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
        with open(f'./{modelName}_log','w') as f: print("No checkpoint found, starting new model.",file=f)
    return EPOCH,bestLoss,chkptPath,hist

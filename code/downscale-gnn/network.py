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
                # APPNP only needs one linear layer, uses K-step propagation
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
    
def train(loader,model,lossFn,opt,scheduler,device,nNodes):
    model.train()
    _LOSS = 0
    pred,truth = [],[]
    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x,_batch.edge_index,_batch.edge_attr)
        # Use label_mask if available (spatial split mode)
        if hasattr(_batch, 'label_mask'):
            mask = _batch.label_mask
            _loss = lossFn(_yHat[mask], _batch.y[mask])
            _pred = _yHat[mask].reshape(-1, mask.reshape(-1, nNodes)[0].sum().item())
            _truth = _batch.y[mask].reshape(-1, mask.reshape(-1, nNodes)[0].sum().item())
        else:
            _loss = lossFn(_yHat,_batch.y)
            _pred = _yHat.reshape(-1,nNodes)
            _truth = _batch.y.reshape(-1,nNodes)
        _loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        _LOSS += _loss
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
        # Use label_mask if available (spatial split mode)
        if hasattr(_batch, 'label_mask'):
            mask = _batch.label_mask
            _loss = lossFn(_yHat[mask], _batch.y[mask])
            _pred = _yHat[mask].reshape(-1, mask.reshape(-1, nNodes)[0].sum().item())
            _truth = _batch.y[mask].reshape(-1, mask.reshape(-1, nNodes)[0].sum().item())
        else:
            _loss = lossFn(_yHat,_batch.y)
            _pred = _yHat.reshape(-1,nNodes)
            _truth = _batch.y.reshape(-1,nNodes)
        _LOSS += _loss
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

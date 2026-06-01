import numpy as np
import torch, pickle, data, network, utils

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = './data'

def main():
    ##----------------------Create Dataset----------------------
    dataParam = {
            'geoMethod': 'average',
            'nCompPCA': 40,
            'window' : 2,
            'poolSize': 12,
            'batchSize': 512,
            'thres':    0.1,
            'geoFeatures':  'full',
            }
    trainLoader, validLoader, metadata, _ = data.dataGen(dataParam,path)
    ##----------------------Generate model----------------------
    nEpoch = 5000
    modelParam = {
            'HLD':      128,
            'nMLP':     2,
    #-------------------------GNN part----------------------
            'nGNN':     3,
            'nGAT':     1,
            'nHeads':   1,
            'K':        1,
            'iDim':     metadata['iDim'],
            'oDim':     metadata['oDim'],
            'BN':       False,
            'Dropout':  False,
    }
    modelName = f'geoEmbed_{dataParam["geoMethod"]}_gconv_full_{dataParam["geoFeatures"]}Geo'
    model = (network.GNN(modelParam)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)
    # Load checkpoint or initialize training
    EPOCH,bestLoss,chkptPath,hist = network.loadCheckPoint(modelName,model,opt,device,load=False)
    # Save metadata
    with open(f'./{modelName}_param.pkl','wb') as f: 
        pickle.dump(modelParam,f)
        pickle.dump(dataParam,f)
        pickle.dump(metadata,f)
    if torch.cuda.is_available():
        with open(f'./{modelName}_log','a') as f: print(torch.cuda.get_device_name(torch.cuda.current_device()),file=f)
    with open(f'./{modelName}_log','a') as f: print(model,file=f)

    for epoch in range(EPOCH,nEpoch):
        trainLoss,trainRMSE,_,_ = network.train(trainLoader,model,lossFn,opt,scheduler,device,metadata['nNodes'])
        validLoss,validRMSE,_,_ = network.test(validLoader,model,lossFn,device,metadata['nNodes'])
        with open(f'./{modelName}_log','a') as f: print("", file=f)
        with open(f'./{modelName}_log','a') as f: print (f"Epoch {epoch}: loss {trainLoss:1.4e}/{validLoss:1.4e}; RMSE {trainRMSE[0]:1.3f}/{validRMSE[0]:1.3f}; LR {scheduler.get_last_lr()} ",file=f)
        with open(f'./{modelName}_log','a') as f: print (f"RMSE std: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; min: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; max: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};",file=f)
        # Save best model
        if validRMSE[0]<bestLoss:
            bestLoss = validRMSE[0]
            with open(f'./{modelName}_log','a') as f: print("Model saved.",file=f)
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict':   opt.state_dict(),
                'bestLoss':         bestLoss,
                }, chkptPath)
        # Plot training history
        hist.append([trainLoss,validLoss,trainRMSE[0],validRMSE[0]])
        utils.plotHist(hist,modelName)
            
if __name__ == "__main__":
    main()

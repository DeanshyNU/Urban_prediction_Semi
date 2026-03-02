import numpy as np
import torch, pickle, data, utils
import network_ESTnet as network
import os
import wandb
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
# 添加完整时间戳和SLURM作业ID到输出文件夹名称
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = f'./supervised_ESTnet_{timestamp}_job{job_id}'
else:
    output_dir = f'./supervised_ESTnet_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

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
    # ESTnet: 动态特征 = CFD + station，静态特征 = rawGeo + embedGeo
    dynamic_dim = int(metadata['featureIdx']['rawGeo'][0])
    static_dim = metadata['iDim'] - dynamic_dim
    ##----------------------Generate model----------------------
    nEpoch = 5000
    modelParam = {
            'HLD':      128,
            'nMLP':     2,
            'nGNN':     3,
            'dynamic_dim': dynamic_dim,
            'static_dim':  static_dim,
            'oDim':     metadata['oDim'],
    }
    modelName = f'geoEmbed_{dataParam["geoMethod"]}_gconv_full_{dataParam["geoFeatures"]}Geo'
    
    # 初始化 W&B
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=f'supervised_ESTnet_{modelName}',
        config={
            **dataParam,
            **modelParam,
            'nNodes': metadata['nNodes'],
            'nEpoch': 5000,
            'lr': 1e-3,
            'scheduler_gamma': 0.9992,
            'model_type': 'supervised_ESTnet',
        }
    )
    
    model = (network.GNN_ESTNet(modelParam)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)
    # Load checkpoint or initialize training
    EPOCH,bestLoss,chkptPath,hist = network.loadCheckPoint(modelName,model,opt,device,load=False,output_dir=output_dir)
    # Save metadata
    with open(f'{output_dir}/{modelName}_param.pkl','wb') as f: 
        pickle.dump(modelParam,f)
        pickle.dump(dataParam,f)
        pickle.dump(metadata,f)
    if torch.cuda.is_available():
        with open(f'{output_dir}/{modelName}_log','a') as f: print(torch.cuda.get_device_name(torch.cuda.current_device()),file=f)
    with open(f'{output_dir}/{modelName}_log','a') as f: print(model,file=f)

    for epoch in range(EPOCH,nEpoch):
        trainLoss,trainRMSE,_,_ = network.train(trainLoader,model,lossFn,opt,scheduler,device,metadata['nNodes'],dynamic_dim,static_dim)
        validLoss,validRMSE,_,_ = network.test(validLoader,model,lossFn,device,metadata['nNodes'],dynamic_dim,static_dim)
        with open(f'{output_dir}/{modelName}_log','a') as f: print("", file=f)
        with open(f'{output_dir}/{modelName}_log','a') as f: print (f"Epoch {epoch}: loss {trainLoss:1.4e}/{validLoss:1.4e}; RMSE {trainRMSE[0]:1.3f}/{validRMSE[0]:1.3f}; LR {scheduler.get_last_lr()} ",file=f)
        with open(f'{output_dir}/{modelName}_log','a') as f: print (f"RMSE std: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; min: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; max: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};",file=f)
        
        # 记录到 W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/rmse_std': trainRMSE[1],
            'train/rmse_min': trainRMSE[2],
            'train/rmse_max': trainRMSE[3],
            'valid/loss': validLoss,
            'valid/rmse': validRMSE[0],
            'valid/rmse_std': validRMSE[1],
            'valid/rmse_min': validRMSE[2],
            'valid/rmse_max': validRMSE[3],
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': bestLoss,
        })
        
        # Save best model
        if validRMSE[0]<bestLoss:
            bestLoss = validRMSE[0]
            with open(f'{output_dir}/{modelName}_log','a') as f: print("Model saved.",file=f)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict':   opt.state_dict(),
                'bestLoss':         bestLoss,
                'hist':             hist,
            }, chkptPath)
        # Plot training history
        hist.append([trainLoss,validLoss,trainRMSE[0],validRMSE[0]])
        utils.plotHist(hist,modelName,output_dir=output_dir)
            
if __name__ == "__main__":
    main()

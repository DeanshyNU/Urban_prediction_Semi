import numpy as np
import torch, pickle, data_semi, network_semi, utils
import os
import wandb
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = '/projects/p32685/Fixmatch/data'
# 添加完整时间戳和SLURM作业ID到输出文件夹名称
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = f'./semi_supervised_{timestamp}_job{job_id}'
else:
    output_dir = f'./semi_supervised_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

def main():
    ##----------------------Create Dataset----------------------
    dataParam = {
            'geoMethod': 'average',
            'nCompPCA': 40,
            'window' : 2,
            'poolSize': 12,
            'batchSize': 128,  # 降低batch_size避免OOM
            'thres':    0.1,  # 使用与监督学习相同的阈值
            'geoFeatures':  'full',
            }
    n_unlabeled = 200  # 无标签站点数
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)
    
    # ========== 半监督学习数据检查 ==========
    print("\n" + "="*70)
    print("半监督学习数据检查")
    print("="*70)
    nNodes_total = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    nNodes_unlabeled = metadata['nNodes_unlabeled']
    AdjMatrix = metadata['AdjMatrix']
    label_mask = metadata['label_mask']
    
    # 检查1: 节点数量
    assert nNodes_total == nNodes_labeled + nNodes_unlabeled, \
        f"节点总数不匹配: {nNodes_total} != {nNodes_labeled} + {nNodes_unlabeled}"
    print(f"✓ 节点数量正确: 总计{nNodes_total}个节点（{nNodes_labeled}有标签 + {nNodes_unlabeled}无标签）")
    
    # 检查2: 标签掩码
    assert label_mask.sum() == nNodes_labeled, \
        f"标签掩码不匹配: {label_mask.sum()} != {nNodes_labeled}"
    print(f"✓ 标签掩码正确: {nNodes_labeled}个True, {nNodes_unlabeled}个False")
    
    # 检查3: 图结构 - 检查有标签和无标签节点之间的边连接
    labeled_indices = np.where(label_mask)[0]
    unlabeled_indices = np.where(~label_mask)[0]
    
    # 统计有标签↔无标签之间的边
    cross_edges = 0
    for i in labeled_indices:
        for j in unlabeled_indices:
            if AdjMatrix[i, j] > 0 or AdjMatrix[j, i] > 0:
                cross_edges += 1
                break  # 每个有标签节点只统计一次
    
    # 统计无标签↔无标签之间的边
    unlabeled_to_unlabeled = 0
    for i in unlabeled_indices:
        for j in unlabeled_indices:
            if i < j and AdjMatrix[i, j] > 0:  # 只统计上三角，避免重复
                unlabeled_to_unlabeled += 1
    
    # 统计有标签↔有标签之间的边
    labeled_to_labeled = 0
    for i in labeled_indices:
        for j in labeled_indices:
            if i < j and AdjMatrix[i, j] > 0:  # 只统计上三角，避免重复
                labeled_to_labeled += 1
    
    print(f"✓ 图边连接统计:")
    print(f"   有标签↔有标签边数: {labeled_to_labeled}")
    print(f"   有标签↔无标签边数: {cross_edges}")
    print(f"   无标签↔无标签边数: {unlabeled_to_unlabeled}")
    if cross_edges > 0:
        print(f"   → 无标签节点通过图结构连接到有标签节点 ✓")
    else:
        print(f"   ⚠️  警告：未检测到有标签和无标签节点之间的边连接")
    
    # 检查4: 数据集中的样本
    first_sample = trainLoader.dataset[0]
    assert first_sample.x.shape[0] == nNodes_total, \
        f"样本节点数不匹配: {first_sample.x.shape[0]} != {nNodes_total}"
    assert first_sample.label_mask.sum() == nNodes_labeled, \
        f"样本标签掩码不匹配: {first_sample.label_mask.sum()} != {nNodes_labeled}"
    print(f"✓ 数据集样本正确: 每个样本包含{nNodes_total}个节点的特征")
    
    # 检查5: 特征维度验证
    print(f"✓ 特征维度验证:")
    labeled_feat_dim = first_sample.x[first_sample.label_mask].shape[1]
    unlabeled_feat_dim = first_sample.x[~first_sample.label_mask].shape[1]
    print(f"   有标签节点特征维度: {labeled_feat_dim}")
    print(f"   无标签节点特征维度: {unlabeled_feat_dim}")
    assert labeled_feat_dim == unlabeled_feat_dim, \
        f"特征维度不一致: 有标签={labeled_feat_dim}, 无标签={unlabeled_feat_dim}"
    print(f"   ✓ 特征维度一致: {labeled_feat_dim}")
    print("="*70 + "\n")
    
    ##----------------------Generate model----------------------
    nEpoch = 5000  # 与FixMatch相同的训练轮数
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
            'Dropout':  True,  # 启用dropout
    }
    modelName = f'geoEmbed_{dataParam["geoMethod"]}_gconv_semi_{dataParam["geoFeatures"]}Geo_{n_unlabeled}unlabeled'
    
    # 初始化 W&B
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=modelName,
        config={
            **dataParam,
            **modelParam,
            'n_unlabeled': n_unlabeled,
            'nNodes_total': metadata['nNodes'],
            'nNodes_labeled': metadata['nNodes_labeled'],
            'nNodes_unlabeled': metadata['nNodes_unlabeled'],
            'nEpoch': 5000,
            'lr': 1e-3,
            'scheduler_gamma': 0.9992,
        }
    )
    
    model = (network_semi.GNN(modelParam)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)
    # Load checkpoint or initialize training
    EPOCH,bestLoss,chkptPath,hist = network_semi.loadCheckPoint(modelName,model,opt,device,load=False,output_dir=output_dir)
    # Save metadata
    with open(f'{output_dir}/{modelName}_param.pkl','wb') as f: 
        pickle.dump(modelParam,f)
        pickle.dump(dataParam,f)
        pickle.dump(metadata,f)
    if torch.cuda.is_available():
        with open(f'{output_dir}/{modelName}_log','a') as f: print(torch.cuda.get_device_name(torch.cuda.current_device()),file=f)
    with open(f'{output_dir}/{modelName}_log','a') as f: print(model,file=f)
    with open(f'{output_dir}/{modelName}_log','a') as f: 
        print("="*60, file=f)
        print("半监督学习配置:", file=f)
        print(f"  实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=f)
        print(f"  输出目录: {output_dir}", file=f)
        print("", file=f)
        print("数据参数 (dataParam):", file=f)
        for key, value in dataParam.items():
            print(f"  {key}: {value}", file=f)
        print("", file=f)
        print("模型参数 (modelParam):", file=f)
        for key, value in modelParam.items():
            print(f"  {key}: {value}", file=f)
        print("", file=f)
        print("训练设置:", file=f)
        print(f"  训练轮数: {nEpoch}", file=f)
        print(f"  学习率: 1e-3", file=f)
        print(f"  学习率衰减: 0.9992", file=f)
        print(f"  优化器: Adam", file=f)
        print(f"  损失函数: HuberLoss", file=f)
        print("", file=f)
        print("数据集信息:", file=f)
        print(f"  总节点数: {metadata['nNodes']}", file=f)
        print(f"  有标签节点数: {metadata['nNodes_labeled']}", file=f)
        print(f"  无标签节点数: {metadata['nNodes_unlabeled']}", file=f)
        print(f"  训练样本数: {len(trainLoader.dataset)}", file=f)
        print(f"  验证样本数: {len(validLoader.dataset)}", file=f)
        print(f"  输入特征维度: {metadata['iDim']}", file=f)
        print(f"  输出维度: {metadata['oDim']}", file=f)
        print("="*60, file=f)

    for epoch in range(EPOCH,nEpoch):
        trainLoss,trainRMSE,_,_ = network_semi.train(
            trainLoader, model, lossFn, opt, scheduler, device, 
            metadata['nNodes'], metadata['nNodes_labeled']
        )
        validLoss,validRMSE,_,_ = network_semi.test(
            validLoader, model, lossFn, device, 
            metadata['nNodes'], metadata['nNodes_labeled']
        )
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
                'hist':             hist,  # 修复：添加hist字段
                }, chkptPath)
        # Plot training history
        hist.append([trainLoss,validLoss,trainRMSE[0],validRMSE[0]])
        utils.plotHist(hist,modelName,output_dir=output_dir)
            
if __name__ == "__main__":
    main()

"""
主程序入口
整合所有模块，执行完整的训练流程
"""
import os
import torch
import pickle
import numpy as np
from datetime import datetime
import wandb

# 导入自定义模块
from data_preprocessing import preprocess_unlabeled_data
from data_generation import dataGen_ESTnet, dataGen_unlabeled_ESTnet
from data_augmentation import TransformFixMatch
from models_heteroscedastic import GNN
from Fixmatch_training_heteroscedastic import train_fixmatch_heteroscedastic, test_fixmatch_heteroscedastic, loadCheckPoint
from utils import plotHist

# 设备配置
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# 数据路径配置
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def main():
    # 获取当前时间和 job id
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # 创建输出目录：方法名_时间_jobid
    method_name = 'fixmatch_heteroscedastic'
    output_dir = f'./{method_name}_{current_time}_job{job_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"输出目录: {output_dir}")
    
    ##----------------------数据预处理----------------------
    print("步骤0: 预处理无标签数据")
    unlabeled_file = os.path.join(DATA_PATH, 'Unlabeled_Finalized.mat')
    n_unlabeled = 200  # V2默认200个无标签站点（可配置为500）
    unlabeled_data = preprocess_unlabeled_data(
        unlabeled_file=unlabeled_file,
        target_station_count=n_unlabeled,
        nTimesteps=6624
    )
    
    ##----------------------创建数据集----------------------
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,  # V2: 从4改为2
        'poolSize': 12,
        'batchSize': 128,
        'thres': 0.4,  # V2: 从0.1改为0.4
        'geoFeatures': 'full',
    }
    print("步骤1: 生成有标签和无标签数据集")
    trainLoader, validLoader, metadata, _ = dataGen_ESTnet(dataParam, DATA_PATH)
    
    print("\n" + "="*60)

    ##----------------------数据增强----------------------
    print("步骤2: 对无标签数据应用数据增强")
    # 增强强度：weak_m=1.5（温和），strong_m=4.5（明显更强，保持约1:3比例）
    augmenter = TransformFixMatch(weak_n=2, weak_m=1.5, strong_n=3, strong_m=4.5, seed=42)
    weak_augmented_data, strong_augmented_data = augmenter(unlabeled_data)
    
    weak_loader, _, _, _, _ = dataGen_unlabeled_ESTnet(dataParam, weak_augmented_data, labeled=False, path=DATA_PATH)
    strong_loader, _, _, _, _ = dataGen_unlabeled_ESTnet(dataParam, strong_augmented_data, labeled=False, path=DATA_PATH)
    print(f"弱增强样本数量: {len(weak_loader.dataset)}")
    print(f"强增强样本数量: {len(strong_loader.dataset)}")
    print(f"trainLoader长度: {len(trainLoader)}")
    print(f"validLoader长度: {len(validLoader)}")
    print(f"weak_loader长度: {len(weak_loader)}")     
    print(f"strong_loader长度: {len(strong_loader)}")

    ##----------------------生成模型----------------------
    nEpoch = 2000
    # 更新模型参数，使用分离特征维度
    modelParam = {
        'HLD': 128,
        'nMLP': 2,
        'nGNN': 3,
        'iDim': metadata['iDim'],  # 使用统一特征维度
        'oDim': metadata['oDim'],
    }

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_fixmatch_{dataParam["geoFeatures"]}Geo'
    model_path = os.path.join(output_dir, modelName)
    print("步骤3: 初始化模型、优化器和损失函数")
    model = GNN(modelParam).to(device)  # 使用 Heteroscedastic GNN 模型（返回 mu, log_var）
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)  # 降低学习率到1e-4以稳定训练
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)
    # Heteroscedastic 模型使用 NLL 损失，不需要额外的损失函数  

    # 加载检查点或初始化训练
    EPOCH, bestLoss, chkptPath, hist = loadCheckPoint(model_path, model, opt, device, load=False)
    # 保存元数据
    with open(f'./{model_path}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)
    if torch.cuda.is_available():
        with open(f'./{model_path}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'./{model_path}_log', 'a') as f:
        print(model, file=f)

    # 初始化 W&B
    run_name = f'{method_name}_{current_time}_job{job_id}'
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=run_name,
        config={
            **dataParam,
            **modelParam,
            'method': 'FixMatch_Heteroscedastic',
            'n_unlabeled': n_unlabeled,
            'nNodes': metadata['nNodes'],
            'nEpoch': nEpoch,
            'lr': 1e-4,
            'scheduler_gamma': 0.9995,
            'lambda_U': 3.0,
            'ramp_epochs': 100,
            'confidence_threshold': 0.1,
            'output_dir': output_dir,
        }
    )
    
    # 在模型初始化后记录配置
    with open(f'./{model_path}_log', 'a') as f:
        print("="*60, file=f)
        print("FixMatch配置（Heteroscedastic版本）:", file=f)
        print(f"  实验时间: {current_time}", file=f)
        print(f"  Job ID: {job_id}", file=f)
        print(f"  输出目录: {output_dir}", file=f)
        print("  方法: FixMatch + Heteroscedastic Head（不确定性量化，方案A）", file=f)
        print("  数据增强: weak_m=1.5, strong_m=4.5（UrbanFeature不增强）", file=f)
        print("  伪标签: 1次弱增强推理，方差加权", file=f)
        print("  有标签损失: MSE（只优化mu）", file=f)
        print("  无标签损失: MSE一致性 + 方差加权（weight = 1/var_weak）", file=f)
        print("  模型: GNN Heteroscedastic (SAGEConv, 输出 mu + log_var)", file=f)
        print("  特征: 统一特征向量（动态+静态合并）", file=f)
        print("  优化: lr=1e-4, 学习率衰减=0.9995", file=f)
        print("  无标签损失权重: lambda_U=3（带ramp-up，100 epochs）", file=f)
        print("  梯度裁剪: max_norm=1.0（防止梯度爆炸）", file=f)
        print(f"  {n_unlabeled}个无标签站点", file=f)
        print("="*60, file=f)

    ##----------------------训练----------------------
    print("步骤4: 开始训练")
    print(f"当前轮次: {EPOCH}, 总轮次: {nEpoch}")
    
    def rampup_factor(epoch, ramp_ep=100):
        """Ramp-up函数，用于逐渐增加无标签损失的权重"""
        return min(1.0, epoch / ramp_ep)
    
    for epoch in range(EPOCH, nEpoch):
        ramp = rampup_factor(epoch, 100)
        # 使用FixMatch训练（Heteroscedastic版本）
        trainLoss, trainRMSE, _, _ = train_fixmatch_heteroscedastic(
            trainLoader, weak_loader, strong_loader, model, opt, scheduler, device, 
            metadata['nNodes'], lambda_U=3 * ramp
        )
        validLoss, validRMSE, _, _, valid_uncertainty = test_fixmatch_heteroscedastic(
            validLoader, model, device, metadata['nNodes']
        )

        # 记录结果
        with open(f'./{model_path}_log', 'a') as f:
            print("", file=f)
            print(f"轮次 {epoch}: 损失 {trainLoss:1.4e}/{validLoss:1.4e}; RMSE {trainRMSE[0]:1.3f}/{validRMSE[0]:1.3f}; 学习率 {scheduler.get_last_lr()[0]:1.6f}; ramp={ramp:1.3f} ", file=f)
            print(f"RMSE 标准差: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; 最小值: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; 最大值: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};", file=f)
            print(f"验证集平均不确定性（方差）: {np.mean(valid_uncertainty):.6f}", file=f)
        
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
            'lambda_U': 3.0 * ramp,
            'ramp_factor': ramp,
            'valid/uncertainty_mean': np.mean(valid_uncertainty),
            'valid/uncertainty_std': np.std(valid_uncertainty),
        })

        # 保存最佳模型
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            with open(f'./{model_path}_log', 'a') as f:
                print("模型已保存。", file=f)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'bestLoss': bestLoss,
                'hist': hist,  # 添加hist以便加载后继续训练
            }, chkptPath)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})

        # 绘制训练历史
        hist.append([trainLoss, validLoss, trainRMSE[0], validRMSE[0]])
        plotHist(hist, model_path)

    print("训练完成")


if __name__ == "__main__":
    main()


"""
Mean Teacher主程序
实现Mean Teacher半监督学习完整训练流程
"""
import os
import torch
import pickle
import numpy as np
from copy import deepcopy
from datetime import datetime
import wandb

# 导入自定义模块
from data_preprocessing import preprocess_unlabeled_data
from data_generation import dataGen_ESTnet, dataGen_unlabeled_ESTnet
from data_augmentation import TransformFixMatch
from models import GNN
from meanteacher_training import train_meanteacher, test_meanteacher, loadCheckPoint
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
    method_name = 'meanteacher'
    output_dir = f'./{method_name}_{current_time}_job{job_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"输出目录: {output_dir}")

    ##----------------------数据预处理----------------------
    print("步骤0: 预处理无标签数据")
    unlabeled_file = os.path.join(DATA_PATH, 'Unlabeled_Finalized.mat')
    n_unlabeled = 200
    unlabeled_data = preprocess_unlabeled_data(
        unlabeled_file=unlabeled_file,
        target_station_count=n_unlabeled,
        nTimesteps=6624
    )

    ##----------------------创建数据集----------------------
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': 12,
        'batchSize': 128,
        'thres': 0.4,
        'geoFeatures': 'full',
    }
    print("步骤1: 生成有标签和无标签数据集")
    trainLoader, validLoader, metadata, _ = dataGen_ESTnet(dataParam, DATA_PATH)

    print("\n" + "="*60)

    ##----------------------数据增强----------------------
    print("步骤2: 对无标签数据应用数据增强（Mean Teacher）")
    # Mean Teacher只需要一次增强
    # 增强强度：weak_m=1.5（温和），strong_m=4.5（明显更强，保持约1:3比例）
    augmenter = TransformFixMatch(weak_n=2, weak_m=1.5, strong_n=3, strong_m=4.5, seed=42)
    augmented_data, _ = augmenter(unlabeled_data)

    unlabeled_loader, _, _, _, _ = dataGen_unlabeled_ESTnet(dataParam, augmented_data, labeled=False, path=DATA_PATH)
    print(f"增强后无标签样本数量: {len(unlabeled_loader.dataset)}")
    print(f"有标签训练样本数量: {len(trainLoader.dataset)}")
    print(f"有标签验证样本数量: {len(validLoader.dataset)}")

    ##----------------------生成模型----------------------
    nEpoch = 5000
    modelParam = {
        'HLD': 128,
        'nMLP': 2,
        'nGNN': 3,
        'iDim': metadata['iDim'],  # 使用统一特征维度
        'oDim': metadata['oDim'],
    }

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_meanteacher_{dataParam["geoFeatures"]}Geo'
    model_path = os.path.join(output_dir, modelName)
    print("步骤3: 初始化学生模型和教师模型")

    # 初始化学生模型
    student_model = GNN(modelParam).to(device)  # 使用统一的GNN模型
    # 初始化教师模型（作为学生模型的副本）
    teacher_model = GNN(modelParam).to(device)  # 使用统一的GNN模型
    # 复制学生模型的初始参数到教师模型
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(student_param.data)
    # 教师模型不需要梯度
    for param in teacher_model.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)
    consistency_loss_fn = torch.nn.MSELoss().to(device)  # Mean Teacher使用MSE作为一致性损失

    # 加载检查点或初始化训练
    EPOCH, bestLoss, chkptPath, hist = loadCheckPoint(model_path, student_model, opt, device, load=False)

    # 保存元数据
    with open(f'./{model_path}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)
    if torch.cuda.is_available():
        with open(f'./{model_path}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'./{model_path}_log', 'a') as f:
        print("学生模型架构:", file=f)
        print(student_model, file=f)

    # 初始化 W&B
    run_name = f'{method_name}_{current_time}_job{job_id}'
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=run_name,
        config={
            **dataParam,
            **modelParam,
            'method': 'MeanTeacher',
            'n_unlabeled': n_unlabeled,
            'nNodes': metadata['nNodes'],
            'nEpoch': nEpoch,
            'lr': 1e-3,
            'scheduler_gamma': 0.9992,
            'lambda_U': 10.0,
            'ramp_epochs': 30,
            'ema_alpha': 0.999,
            'output_dir': output_dir,
        }
    )
    
    # 记录配置
    with open(f'./{model_path}_log', 'a') as f:
        print("="*60, file=f)
        print("Mean Teacher配置:", file=f)
        print(f"  实验时间: {current_time}", file=f)
        print(f"  Job ID: {job_id}", file=f)
        print(f"  输出目录: {output_dir}", file=f)
        print("  方法: Mean Teacher（教师-学生架构）", file=f)
        print("  数据增强: weak_n=2, weak_m=1.5, strong_n=3, strong_m=4.5（UrbanFeature不增强）", file=f)
        print("  一致性损失: MSELoss", file=f)
        print("  优化: lr=1e-3, 学习率衰减=0.9992", file=f)
        print(f"  无标签损失权重: lambda_U（带ramp-up）", file=f)
        print("  EMA系数: alpha=0.999", file=f)
        print("  模型: GNN (SAGEConv)", file=f)
        print(f"  {n_unlabeled}个无标签站点", file=f)
        print("  特征: 统一特征向量（动态+静态合并）", file=f)
        print("="*60, file=f)

    ##----------------------训练----------------------
    print("步骤4: 开始Mean Teacher训练")
    print(f"当前轮次: {EPOCH}, 总轮次: {nEpoch}")

    def rampup_factor(epoch, ramp_ep=30):
        """Ramp-up函数，用于逐渐增加无标签损失的权重"""
        return min(1.0, epoch / ramp_ep)

    global_step = 0
    for epoch in range(EPOCH, nEpoch):
        ramp = rampup_factor(epoch, 30)
        # Mean Teacher训练
        trainLoss, trainRMSE, _, _ = train_meanteacher(
            trainLoader, unlabeled_loader, student_model, teacher_model, lossFn, consistency_loss_fn,
            opt, scheduler, device, metadata['nNodes'], lambda_U=10.0 * ramp, alpha=0.999,
            global_step=global_step
        )
        # 使用教师模型进行验证
        validLoss, validRMSE, _, _ = test_meanteacher(
            validLoader, teacher_model, lossFn, device, metadata['nNodes']
        )

        global_step += len(trainLoader)

        # 记录结果
        with open(f'./{model_path}_log', 'a') as f:
            print("", file=f)
            print(f"轮次 {epoch}: 损失 {trainLoss:1.4e}/{validLoss:1.4e}; "
                  f"RMSE {trainRMSE[0]:1.3f}/{validRMSE[0]:1.3f}; "
                  f"学习率 {scheduler.get_last_lr()[0]:1.6f}; ramp={ramp:1.3f}", file=f)
            print(f"RMSE 标准差: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; "
                  f"最小值: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; "
                  f"最大值: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};", file=f)
        
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
            'lambda_U': 10.0 * ramp,
            'ramp_factor': ramp,
            'global_step': global_step,
        })

        # 保存最佳模型（保存教师模型）
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            with open(f'./{model_path}_log', 'a') as f:
                print("模型已保存。", file=f)
            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher_model.state_dict(),  # 保存教师模型
                'student_state_dict': student_model.state_dict(),  # 也保存学生模型
                'opt_state_dict': opt.state_dict(),
                'bestLoss': bestLoss,
                'hist': hist,
            }, chkptPath)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})

        # 绘制训练历史
        hist.append([trainLoss, validLoss, trainRMSE[0], validRMSE[0]])
        plotHist(hist, model_path)

    print("训练完成")


if __name__ == "__main__":
    main()
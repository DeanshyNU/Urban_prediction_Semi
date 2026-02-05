"""
主程序入口
整合所有模块，执行完整的训练流程
"""
import os
import torch
import pickle
import numpy as np

# 导入自定义模块
from data_preprocessing import preprocess_unlabeled_data
from data_generation import dataGen_ESTnet, dataGen_unlabeled_ESTnet
from data_augmentation import TransformFixMatch
from models import GNN_ESTNet
from Fixmatch_training import train_fixmatch, test_fixmatch, loadCheckPoint
from utils import plotHist

# 设备配置
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# 数据路径配置
DATA_PATH = '/projects/p32685/Fixmatch/data'


def main():
    # 创建输出目录（如果不存在）
    output_dir = './Fixmatch_ESTnet'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    # V2: 增强参数从 weak_n=3, weak_m=8, strong_n=6, strong_m=15 改为 weak_n=2, weak_m=3, strong_n=3, strong_m=6
    augmenter = TransformFixMatch(weak_n=2, weak_m=3, strong_n=3, strong_m=6)
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
    nEpoch = 2000  # V2: 从5000改为2000
    # 更新模型参数，使用分离特征维度
    modelParam = {
        'HLD': 128,
        'nMLP': 2,
        'nGNN': 3,
        'nGAT': 1,
        'nHeads': 1,
        'K': 1,
        'dynamic_dim': metadata['dynamic_dim'],  # 使用动态特征维度
        'static_dim': metadata['static_dim'],    # 使用静态特征维度
        'oDim': metadata['oDim'],
        'BN': False,
        'Dropout': True,  # V2: 启用dropout
    }

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_fixmatch_ESTnet_{dataParam["geoFeatures"]}Geo'  # 修改名称以区分
    model_path = os.path.join(output_dir, modelName)
    print("步骤3: 初始化模型、优化器和损失函数")
    model = GNN_ESTNet(modelParam).to(device)  # 使用ESTNet版本的GNN
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)
    loss_unlabel_fn = torch.nn.HuberLoss(reduction='none').to(device)  # V2: 使用reduction='none'以正确应用权重  

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

    # 在模型初始化后记录配置
    with open(f'./{model_path}_log', 'a') as f:
        print("模型配置:", file=f)
        print("  数据增强: weak_n=2, weak_m=3, strong_n=3, strong_m=6", file=f)
        print("  UQ参数: epsilon=1e-5, temperature=0.5, n_augments=5", file=f)
        print("  权重处理: 标准归一化，无权重限幅", file=f)
        print("  优化: lr=1e-3, 无weight_decay, 无梯度裁剪", file=f)
        print("  无标签损失: lambda_U=10", file=f)
        print("  Dropout: 启用", file=f)
        print(f"  {n_unlabeled}个无标签站点", file=f)
        print("  ESTNet架构（静态和动态特征分离）", file=f)
        print("  新添加的改动：threshold=0.4; kNN保底稀疏化(k=8); 统一图结构; 未来时间窗口", file=f)
        print("----------------------------------------", file=f)

    ##----------------------训练----------------------
    print("步骤4: 开始训练")
    print(f"当前轮次: {EPOCH}, 总轮次: {nEpoch}")
    
    def rampup_factor(epoch, ramp_ep=30):
        """Ramp-up函数，用于逐渐增加无标签损失的权重"""
        return min(1.0, epoch / ramp_ep)
    
    for epoch in range(EPOCH, nEpoch):
        ramp = rampup_factor(epoch, 30)
        # 使用FixMatch训练
        trainLoss, trainRMSE, _, _ = train_fixmatch(
            trainLoader, weak_loader, strong_loader, model, lossFn, loss_unlabel_fn, opt, scheduler, device, metadata['nNodes'], lambda_U=10 * ramp  # V2: 使用ramp-up
        )
        validLoss, validRMSE, _, _ = test_fixmatch(
            validLoader, model, lossFn, device, metadata['nNodes']
        )

        # 记录结果
        with open(f'./{model_path}_log', 'a') as f:
            print("", file=f)
            print(f"轮次 {epoch}: 损失 {trainLoss:1.4e}/{validLoss:1.4e}; RMSE {trainRMSE[0]:1.3f}/{validRMSE[0]:1.3f}; 学习率 {scheduler.get_last_lr()[0]:1.6f}; ramp={ramp:1.3f} ", file=f)
            print(f"RMSE 标准差: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; 最小值: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; 最大值: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};", file=f)

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

        # 绘制训练历史
        hist.append([trainLoss, validLoss, trainRMSE[0], validRMSE[0]])
        plotHist(hist, model_path)

    print("训练完成")


if __name__ == "__main__":
    main()


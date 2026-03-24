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
from datalib import preprocess_unlabeled_data, dataGen_unified, TransformFixMatch
from datalib import dataGen, dataGen_unlabeled  # 独立图
from models import GNN, FeatureGNN
from trainers import train_meanteacher_unified, test_meanteacher_unified, loadCheckPoint, test_meanteacher_unified_ablation
from trainers import train_meanteacher, test_meanteacher  # 独立图
from utils import plotHist

# 设备配置
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# 数据路径配置
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def main():
    # 获取当前时间和 job id
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    job_id = os.environ.get('SLURM_JOB_ID', 'local')

    # 读取卷积类型（默认 graphconv，可通过环境变量 CONV_TYPE=sage 切换）
    conv_type = os.environ.get('CONV_TYPE', 'graphconv')
    # 图类型：unified（统一图）或 independent（独立图）
    graph_type = os.environ.get('GRAPH_TYPE', 'unified')
    # 模型类型：base（原始MLP编码器）或 feature（分支编码器+残差学习）
    model_type = os.environ.get('MODEL_TYPE', 'base')

    # 创建输出目录：方法名_时间_jobid（相对项目根目录 Fixmatch_GNN/log/）
    method_name = f'meanteacher_{conv_type}_{graph_type}{"_feature" if model_type == "feature" else ""}'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'log', f'{method_name}_{current_time}_job{job_id}')
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

    ##----------------------数据参数----------------------
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': 4 if model_type == 'feature' else 12,  # feature模式下压缩GeoEmbed（112维 vs 1008维）
        'batchSize': 128,
        'thres': 0.4,             # 统一阈值（与 data_semi.py 一致）
        'geoFeatures': 'full',
    }

    ##----------------------数据增强----------------------
    print("步骤1: 对数据应用增强（labeled + unlabeled 统一增强，per original MT paper）")
    augmenter = TransformFixMatch(weak_n=2, weak_m=1.5, strong_n=3, strong_m=4.5, seed=42)
    augmented_data, _ = augmenter(unlabeled_data)

    ##----------------------构建数据集----------------------
    if graph_type == 'unified':
        print("步骤2: 构建统一图数据集（labeled + unlabeled，268节点）")
        # Pass augmenter so labeled data gets same augmentation as unlabeled
        trainLoader, validLoader, metadata, _ = dataGen_unified(dataParam, DATA_PATH, augmented_data,
                                                                 output_dir=output_dir,
                                                                 augmenter=augmenter)
        unlabeled_loader = None  # 统一图不需要单独的unlabeled loader
        print(f"统一图训练样本数量: {len(trainLoader.dataset)}")
        print(f"统一图验证样本数量: {len(validLoader.dataset)}")
        print(f"有标签节点: {metadata['n_labeled']}, 无标签节点: {metadata['n_unlabeled']}")
    else:
        print("步骤2: 构建独立图数据集（labeled和unlabeled分开）")
        trainLoader, validLoader, metadata, _ = dataGen(dataParam, DATA_PATH)
        unlabeled_loader, _, _, _, _ = dataGen_unlabeled(dataParam, augmented_data, labeled=False,
                                                          path=DATA_PATH, labeled_metadata=metadata)
        print(f"Labeled训练样本数量: {len(trainLoader.dataset)}")
        print(f"Labeled验证样本数量: {len(validLoader.dataset)}")
        print(f"Unlabeled样本数量: {len(unlabeled_loader.dataset)}")
        print(f"有标签节点: {metadata['nNodes']}")

    print("\n" + "="*60)

    ##----------------------生成模型----------------------
    nEpoch = 5000
    modelParam = {
        'HLD': 128,
        'nMLP': 2,
        'nGNN': 3,
        'iDim': metadata['iDim'],
        'oDim': metadata['oDim'],
        'conv_type': conv_type,
        'model_type': model_type,
        'window': dataParam['window'],          # for FeatureGNN (dynamic/static split)
        'station_dim': 4,                       # for FeatureGNN (CLMS/station features)
    }

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_meanteacher_{dataParam["geoFeatures"]}Geo'
    model_path = os.path.join(output_dir, modelName)
    print(f"步骤3: 初始化学生模型和教师模型 (model_type={model_type})")

    # 选择模型类型
    ModelClass = FeatureGNN if model_type == 'feature' else GNN
    student_model = ModelClass(modelParam).to(device)
    teacher_model = ModelClass(modelParam).to(device)
    # 复制学生模型的初始参数到教师模型
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(student_param.data)
    # 教师模型不需要梯度
    for param in teacher_model.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)  # 与FixMatch一致
    lossFn = torch.nn.HuberLoss().to(device)
    consistency_loss_fn = torch.nn.MSELoss().to(device)  # Mean Teacher使用MSE作为一致性损失

    # 加载检查点或初始化训练
    EPOCH, bestLoss, chkptPath, hist = loadCheckPoint(model_path, student_model, opt, device, load=False)

    # 保存元数据
    with open(f'{model_path}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)
    if torch.cuda.is_available():
        with open(f'{model_path}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'{model_path}_log', 'a') as f:
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
            'scheduler_gamma': 0.9995,
            'lambda_U': 1.0,
            'ramp_epochs': 100,
            'ema_alpha': 0.999,
            'grad_clip_norm': 1.0,
            'output_dir': output_dir,
        }
    )

    # 记录配置
    with open(f'{model_path}_log', 'a') as f:
        print("="*60, file=f)
        print("Mean Teacher配置:", file=f)
        print(f"  实验时间: {current_time}", file=f)
        print(f"  Job ID: {job_id}", file=f)
        print(f"  输出目录: {output_dir}", file=f)
        print("  方法: Mean Teacher（教师-学生架构）", file=f)
        print("  数据增强: weak_n=2, weak_m=1.5（UrbanFeature不增强）", file=f)
        print("  GNN增强: noise_std=0.15, edge_drop=15%, feat_mask=15%", file=f)
        print("  一致性损失: MSELoss (on ALL nodes)", file=f)
        print("  优化: lr=1e-3, 学习率衰减=0.9995", file=f)
        print("  梯度裁剪: max_norm=1.0", file=f)
        print(f"  无标签损失权重: lambda_U=1.0（sigmoid ramp-up, ramp_ep=100）", file=f)
        print(f"  EMA decay: 0.99 (epoch<100) → 0.999 (epoch>=100)", file=f)
        print("  EMA系数: alpha=0.999", file=f)
        if model_type == 'feature':
            print(f"  模型: FeatureGNN ({conv_type}) — ESTNet-style 动态/静态分离", file=f)
            print(f"    Dynamic branch: GRU (WRF 54ch×5steps + Station 4dim → 64dim)", file=f)
            print(f"    Static branch: MLP (RawGeo+GeoEmbed → 64dim)", file=f)
            print(f"    poolSize={dataParam['poolSize']} (GeoEmbed压缩)", file=f)
        else:
            print(f"  模型: GNN ({conv_type})", file=f)
        if graph_type == 'unified':
            print(f"  图结构: 统一图（labeled={metadata['n_labeled']}节点 + unlabeled={metadata['n_unlabeled']}节点 = {metadata['total_nodes']}节点）", file=f)
        else:
            print(f"  图结构: 独立图（labeled={metadata['nNodes']}节点，unlabeled单独处理）", file=f)
        print(f"  {n_unlabeled}个无标签站点", file=f)
        print("  特征: 统一特征向量（动态+静态合并）", file=f)
        print(f"  图稀疏化阈值: thres={dataParam['thres']}（统一阈值，参考data_semi.py）", file=f)
        print("  时间对齐: V2偏移26小时，只用V1对应时间段（2948步）", file=f)
        print("="*60, file=f)

    ##----------------------训练----------------------
    print("步骤4: 开始Mean Teacher训练")
    print(f"当前轮次: {EPOCH}, 总轮次: {nEpoch}")

    def sigmoid_rampup(current, ramp_length):
        """Sigmoid ramp-up (per official MT implementation).
        Starts near 0, smooth transition, reaches 1.0 at ramp_length.
        Formula: exp(-5.0 * (1 - current/ramp_length)^2)
        """
        if ramp_length == 0:
            return 1.0
        current = np.clip(current, 0.0, ramp_length)
        phase = 1.0 - current / ramp_length
        return float(np.exp(-5.0 * phase * phase))

    def get_ema_decay(epoch, ramp_ep=100):
        """EMA decay ramp-up (per official MT): 0.99 early → 0.999 later.
        Early training: teacher follows student closely (faster update).
        Later training: teacher is more stable (slower update).
        """
        if epoch < ramp_ep:
            return 0.99
        return 0.999

    global_step = 0
    for epoch in range(EPOCH, nEpoch):
        ramp = sigmoid_rampup(epoch, 100)
        ema_alpha = get_ema_decay(epoch, 100)
        if graph_type == 'unified':
            # Mean Teacher训练（统一图）
            trainLoss, trainRMSE, _, _, epoch_debug = train_meanteacher_unified(
                trainLoader, student_model, teacher_model, lossFn, consistency_loss_fn,
                opt, scheduler, device, metadata['n_labeled'], lambda_U=1.0 * ramp, alpha=ema_alpha,
                global_step=global_step
            )
            validLoss, validRMSE, _, _ = test_meanteacher_unified(
                validLoader, teacher_model, lossFn, device, metadata['n_labeled']
            )
        else:
            # Mean Teacher训练（独立图）
            trainLoss, trainRMSE, _, _, epoch_debug = train_meanteacher(
                trainLoader, unlabeled_loader, student_model, teacher_model, lossFn, consistency_loss_fn,
                opt, scheduler, device, metadata['nNodes'], lambda_U=1.0 * ramp, alpha=ema_alpha,
                global_step=global_step
            )
            validLoss, validRMSE, _, _ = test_meanteacher(
                validLoader, teacher_model, lossFn, device, metadata['nNodes']
            )

        global_step += len(trainLoader)

        # 记录结果 + 调试信息
        with open(f'{model_path}_log', 'a') as f:
            print("", file=f)
            print(f"轮次 {epoch}: 损失 {trainLoss:1.4e}/{validLoss:1.4e}; "
                  f"RMSE {trainRMSE[0]:1.3f}/{validRMSE[0]:1.3f}; "
                  f"学习率 {scheduler.get_last_lr()[0]:1.6f}; ramp={ramp:1.3f}", file=f)
            print(f"RMSE 标准差: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; "
                  f"最小值: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; "
                  f"最大值: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};", file=f)
            # 调试信息
            print(f"  [DEBUG] labeled_loss={epoch_debug.get('debug/labeled_loss_mean',0):1.4e} | "
                  f"consistency_loss={epoch_debug.get('debug/consistency_loss_mean',0):1.4e}", file=f)
            print(f"  [DEBUG] teacher_pred: mean={epoch_debug.get('debug/teacher_pred_mean_mean',0):1.4f}, "
                  f"std={epoch_debug.get('debug/teacher_pred_std_mean',0):1.4f}", file=f)
            print(f"  [DEBUG] student_pred: mean={epoch_debug.get('debug/student_pred_mean_mean',0):1.4f}, "
                  f"std={epoch_debug.get('debug/student_pred_std_mean',0):1.4f}", file=f)
            print(f"  [DEBUG] pred_diff: mean={epoch_debug.get('debug/pred_diff_mean_mean',0):1.6f}, "
                  f"max={epoch_debug.get('debug/pred_diff_max_max',0):1.6f}", file=f)
            print(f"  [DEBUG] grad_norm: before_clip_max={epoch_debug.get('debug/grad_norm_before_clip_max',0):1.4f}, "
                  f"after_clip_max={epoch_debug.get('debug/grad_norm_after_clip_max',0):1.4f}", file=f)

        # 记录到 W&B
        log_dict = {
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
            'lambda_U': 1.0 * ramp,
            'ramp_factor': ramp,
            'global_step': global_step,
        }
        log_dict.update(epoch_debug)  # 把调试信息也记录到 W&B
        wandb.log(log_dict)

        # 保存最佳模型（保存教师模型）
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            with open(f'{model_path}_log', 'a') as f:
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

        # 每100个epoch做一次消融测试：统一图 vs 仅labeled子图（仅统一图模式）
        if graph_type == 'unified' and epoch % 100 == 0:
            test_meanteacher_unified_ablation(validLoader, teacher_model, lossFn, device, metadata['n_labeled'])

        # 绘制训练历史
        hist.append([trainLoss, validLoss, trainRMSE[0], validRMSE[0]])
        plotHist(hist, model_path)

    print("训练完成")


if __name__ == "__main__":
    main()

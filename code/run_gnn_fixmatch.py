"""
FixMatch + Graph Uncertainty 主程序
单模型（无EMA），强弱增强 + graph-structured uncertainty 过滤 pseudo label
"""
import os
import torch
import pickle
import numpy as np
from datetime import datetime
import wandb

# 导入自定义模块
from datalib import preprocess_unlabeled_data, dataGen_unified, TransformFixMatch
from models import GNN, FeatureGNN
from trainers import train_fixmatch_unified, test_fixmatch_unified, loadCheckPoint, compute_hop_distances
from utils import plotHist

# 设备配置
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# 数据路径配置
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def main():
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    job_id = os.environ.get('SLURM_JOB_ID', 'local')

    conv_type = os.environ.get('CONV_TYPE', 'graphconv')
    model_type = os.environ.get('MODEL_TYPE', 'base')

    # 输出目录
    method_name = f'fixmatch_{conv_type}_unified{"_feature" if model_type == "feature" else ""}'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'log', f'{method_name}_{current_time}_job{job_id}')
    os.makedirs(output_dir, exist_ok=True)
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
        'poolSize': 4 if model_type == 'feature' else 12,
        'batchSize': 128,
        'thres': 0.4,
        'geoFeatures': 'full',
    }

    ##----------------------数据增强（弱+强）----------------------
    print("步骤1: 对无标签数据应用强弱增强")
    augmenter = TransformFixMatch(weak_n=2, weak_m=1.5, strong_n=3, strong_m=4.5, seed=42)
    weak_data, strong_data = augmenter(unlabeled_data)

    ##----------------------构建两套数据集（共享图结构和split）----------------------
    print("步骤2a: 构建弱增强统一图数据集")
    trainLoader_weak, validLoader_weak, metadata, _ = dataGen_unified(
        dataParam, DATA_PATH, weak_data, output_dir=output_dir
    )

    print("步骤2b: 构建强增强统一图数据集（共享图结构）")
    # 强增强数据集使用相同的train/valid split
    trainLoader_strong, validLoader_strong, metadata_strong, _ = dataGen_unified(
        dataParam, DATA_PATH, strong_data
    )

    n_labeled = metadata['n_labeled']
    total_nodes = metadata['total_nodes']

    print(f"有标签节点: {n_labeled}, 无标签节点: {metadata['n_unlabeled']}")
    print(f"训练样本: {len(trainLoader_weak.dataset)}, 验证样本: {len(validLoader_weak.dataset)}")

    ##----------------------预计算 hop distance----------------------
    print("步骤2c: 预计算 hop distance（graph uncertainty）")
    # 从第一个batch获取edge_index
    sample_batch = next(iter(trainLoader_weak))
    hop_dist = compute_hop_distances(
        sample_batch.edge_index, n_labeled, total_nodes, device=device
    )

    # Hop distance统计
    hop_unlabeled = hop_dist[n_labeled:]
    print(f"  Unlabeled节点hop distance: min={hop_unlabeled.min().item()}, "
          f"max={hop_unlabeled.max().item()}, mean={hop_unlabeled.float().mean():.1f}")
    for h in range(1, 6):
        n_at_hop = int((hop_unlabeled == h).sum().item())
        print(f"    hop={h}: {n_at_hop} nodes")
    n_beyond = int((hop_unlabeled > 3).sum().item())
    print(f"    hop>3 (will be filtered): {n_beyond} nodes")

    print("\n" + "="*60)

    ##----------------------模型初始化----------------------
    nEpoch = 5000
    modelParam = {
        'HLD': 128,
        'nMLP': 2,
        'nGNN': 3,
        'iDim': metadata['iDim'],
        'oDim': metadata['oDim'],
        'conv_type': conv_type,
        'model_type': model_type,
        'window': dataParam['window'],
        'station_dim': 4,
    }

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_fixmatch_{dataParam["geoFeatures"]}Geo'
    model_path = os.path.join(output_dir, modelName)
    print(f"步骤3: 初始化模型 (model_type={model_type})")

    ModelClass = FeatureGNN if model_type == 'feature' else GNN
    model = ModelClass(modelParam).to(device)

    # EMA teacher（参数从student复制，不需要梯度）
    from copy import deepcopy
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)
    lossFn = torch.nn.HuberLoss().to(device)
    pseudo_loss_fn = torch.nn.MSELoss().to(device)

    # FixMatch超参数
    LAMBDA_U = 3.0
    RAMP_EPOCHS = 100
    TAU_DIST = 3  # hop distance阈值
    EMA_ALPHA = 0.999

    # 加载检查点或初始化
    EPOCH, bestLoss, chkptPath, hist = loadCheckPoint(model_path, model, opt, device, load=False)

    # 保存元数据
    with open(f'{model_path}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)
    if torch.cuda.is_available():
        with open(f'{model_path}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'{model_path}_log', 'a') as f:
        print("模型架构:", file=f)
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
            'method': 'FixMatch + Graph Uncertainty + EMA',
            'ema_alpha': EMA_ALPHA,
            'n_unlabeled': n_unlabeled,
            'nNodes': metadata['nNodes'],
            'nEpoch': nEpoch,
            'lr': 1e-3,
            'scheduler_gamma': 0.9995,
            'lambda_U': LAMBDA_U,
            'ramp_epochs': RAMP_EPOCHS,
            'tau_dist': TAU_DIST,
            'tau_spatial': 'dynamic (median)',
            'weak_aug': 'n=2, m=1.5',
            'strong_aug': 'n=3, m=4.5',
            'grad_clip_norm': 1.0,
            'output_dir': output_dir,
        }
    )

    # 记录配置
    with open(f'{model_path}_log', 'a') as f:
        print("="*60, file=f)
        print("FixMatch + Graph Uncertainty 配置:", file=f)
        print(f"  实验时间: {current_time}", file=f)
        print(f"  Job ID: {job_id}", file=f)
        print(f"  输出目录: {output_dir}", file=f)
        print(f"  方法: FixMatch + Graph Uncertainty + EMA Teacher", file=f)
        print(f"  EMA系数: alpha={EMA_ALPHA}", file=f)
        print(f"  数据增强: weak(n=2, m=1.5), strong(n=3, m=4.5)", file=f)
        print(f"    增强对象: WRFMat, CLMSMat（动态）; UrbanFeature不增强（静态）", file=f)
        print(f"  Pseudo label损失: MSELoss", file=f)
        print(f"  监督损失: HuberLoss", file=f)
        print(f"  优化: lr=1e-3, 学习率衰减=0.9995", file=f)
        print(f"  梯度裁剪: max_norm=1.0", file=f)
        print(f"  lambda_U={LAMBDA_U}（带ramp-up, ramp_ep={RAMP_EPOCHS}）", file=f)
        print(f"  Graph Uncertainty:", file=f)
        print(f"    u_spatial: |pred_i - mean(pred_neighbors)| < median（动态阈值）", file=f)
        print(f"    u_dist: hop_distance <= {TAU_DIST}（到最近labeled节点的hop数）", file=f)
        print(f"    过滤逻辑: AND（两个条件都满足才保留）", file=f)
        print(f"  Hop distance统计:", file=f)
        print(f"    unlabeled min={hop_unlabeled.min().item()}, max={hop_unlabeled.max().item()}, "
              f"mean={hop_unlabeled.float().mean():.1f}", file=f)
        print(f"    hop>3 被过滤: {n_beyond} nodes", file=f)
        if model_type == 'feature':
            print(f"  模型: FeatureGNN ({conv_type}) — ESTNet-style 动态/静态分离", file=f)
            print(f"    poolSize={dataParam['poolSize']}", file=f)
        else:
            print(f"  模型: GNN ({conv_type})", file=f)
        print(f"  图结构: 统一图（labeled={n_labeled} + unlabeled={metadata['n_unlabeled']} = {total_nodes}）", file=f)
        print(f"  图稀疏化阈值: thres={dataParam['thres']}（统一阈值，参考data_semi.py）", file=f)
        print(f"  时间对齐: V2偏移26小时，只用V1对应时间段（2948步）", file=f)
        print("="*60, file=f)

    ##----------------------训练----------------------
    print("步骤4: 开始FixMatch训练")
    print(f"当前轮次: {EPOCH}, 总轮次: {nEpoch}")

    def rampup_factor(epoch, ramp_ep=100):
        return min(1.0, epoch / ramp_ep)

    loader_dict = {'weak': trainLoader_weak, 'strong': trainLoader_strong}

    global_step = 0
    for epoch in range(EPOCH, nEpoch):
        ramp = rampup_factor(epoch, RAMP_EPOCHS)

        trainLoss, trainRMSE, _, _, epoch_debug = train_fixmatch_unified(
            loader_dict, model, ema_model, lossFn, pseudo_loss_fn,
            opt, scheduler, device, n_labeled,
            hop_dist=hop_dist, lambda_U=LAMBDA_U * ramp, tau_dist=TAU_DIST,
            alpha=EMA_ALPHA, global_step=global_step
        )

        # 验证用EMA teacher（更稳定）
        validLoss, validRMSE, _, _ = test_fixmatch_unified(
            validLoader_weak, ema_model, lossFn, device, n_labeled
        )

        global_step += len(trainLoader_weak)

        # 记录结果
        with open(f'{model_path}_log', 'a') as f:
            print("", file=f)
            print(f"轮次 {epoch}: 损失 {trainLoss:1.4e}/{validLoss:1.4e}; "
                  f"RMSE {trainRMSE[0]:1.3f}/{validRMSE[0]:1.3f}; "
                  f"学习率 {scheduler.get_last_lr()[0]:1.6f}; ramp={ramp:1.3f}", file=f)
            print(f"RMSE 标准差: {trainRMSE[1]:1.3f}/{validRMSE[1]:1.3f}; "
                  f"最小值: {trainRMSE[2]:1.3f}/{validRMSE[2]:1.3f}; "
                  f"最大值: {trainRMSE[3]:1.3f}/{validRMSE[3]:1.3f};", file=f)
            # FixMatch调试信息
            print(f"  [FixMatch] labeled_loss={epoch_debug.get('debug/labeled_loss_mean',0):1.4e} | "
                  f"pseudo_loss={epoch_debug.get('debug/pseudo_loss_mean',0):1.4e}", file=f)
            print(f"  [FixMatch] pseudo labels: total={epoch_debug.get('debug/n_unlabeled_total_mean',0):.0f}, "
                  f"pass_dist={epoch_debug.get('debug/n_pass_dist_mean',0):.0f}, "
                  f"pass_both={epoch_debug.get('debug/n_pass_both_mean',0):.0f} "
                  f"({epoch_debug.get('debug/filter_rate_mean',0)*100:.1f}%)", file=f)
            print(f"  [FixMatch] u_spatial: mean={epoch_debug.get('debug/u_spatial_mean_mean',0):.4f}, "
                  f"tau={epoch_debug.get('debug/tau_spatial_mean',0):.4f}", file=f)
            print(f"  [FixMatch] pseudo_label: mean={epoch_debug.get('debug/pseudo_label_mean_mean',0):.4f}, "
                  f"std={epoch_debug.get('debug/pseudo_label_std_mean',0):.4f}", file=f)
            print(f"  [FixMatch] grad_norm: before={epoch_debug.get('debug/grad_norm_before_clip_max',0):1.4f}, "
                  f"after={epoch_debug.get('debug/grad_norm_after_clip_max',0):1.4f}", file=f)

        # W&B
        log_dict = {
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/rmse_std': trainRMSE[1],
            'valid/loss': validLoss,
            'valid/rmse': validRMSE[0],
            'valid/rmse_std': validRMSE[1],
            'learning_rate': scheduler.get_last_lr()[0],
            'lambda_U': LAMBDA_U * ramp,
            'ramp_factor': ramp,
        }
        log_dict.update(epoch_debug)
        wandb.log(log_dict)

        # 保存最佳模型
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            with open(f'{model_path}_log', 'a') as f:
                print("模型已保存。", file=f)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'bestLoss': bestLoss,
                'hist': hist,
            }, chkptPath)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})

        hist.append([trainLoss, validLoss, trainRMSE[0], validRMSE[0]])
        plotHist(hist, model_path)

    print("训练完成")


if __name__ == "__main__":
    main()

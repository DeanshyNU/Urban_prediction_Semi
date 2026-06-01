"""
Test-Time Augmentation(TTA)on 13860 best ckpt(or any trained model).

不需要重训,只在推理时:
  - 对每个 valid batch,跑 K 个不同 DropEdge augmented forward
  - 取 K 个 prediction 的平均当最终 prediction
  - 计算 v_rmse

期望:TTA 让 model 在不同图视角下的预测平均,减少 noise,可能略涨点

Run:
  cd /home/hhz6461/Urban_prediction_Semi
  conda activate urban
  python -u code_V1/tta_eval.py --ckpt log_V1/13860_V2_semi_spatial/V2_semi_spatial.pt \
                                 --K 10 --dropedge_p 0.2
"""

import os, sys, argparse, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import data as v1_data
import network
import utils


def tta_inference(model, validLoader, device, n_total, K=10, dropedge_p=0.2):
    """对 validLoader 做 K-view TTA inference,返回 (TTA RMSE, single-view RMSE)。"""
    model.eval()
    # 收集每个 K 的预测
    all_K_preds = [[] for _ in range(K)]
    all_truths = []
    all_label_masks = []
    with torch.no_grad():
        for _batch in validLoader:
            _batch = _batch.to(device)
            for k in range(K):
                # 随机 DropEdge
                rand = torch.rand(_batch.edge_index.shape[1], device=device)
                keep = rand >= dropedge_p
                edge_aug = _batch.edge_index[:, keep]
                attr_aug = _batch.edge_attr[keep] if _batch.edge_attr is not None else None
                yhat = model(_batch.x, edge_aug, attr_aug)
                # 收集 valid 节点处的预测
                if hasattr(_batch, 'label_mask'):
                    pred_valid = yhat[_batch.label_mask].cpu().numpy().flatten()
                else:
                    pred_valid = yhat.cpu().numpy().flatten()
                all_K_preds[k].append(pred_valid)
            if hasattr(_batch, 'label_mask'):
                truth = _batch.y[_batch.label_mask].cpu().numpy().flatten()
            else:
                truth = _batch.y.cpu().numpy().flatten()
            all_truths.append(truth)
    # 合并
    all_K_preds = [np.concatenate(preds) for preds in all_K_preds]   # K × (T × n_valid)
    all_K_preds = np.stack(all_K_preds, axis=0)                       # (K, T × n_valid)
    all_truths = np.concatenate(all_truths)                           # (T × n_valid,)

    # TTA prediction = mean across K views
    tta_pred = all_K_preds.mean(axis=0)
    tta_rmse = float(np.sqrt(((tta_pred - all_truths) ** 2).mean()))

    # Single view RMSE(view 0)
    single_pred = all_K_preds[0]
    single_rmse = float(np.sqrt(((single_pred - all_truths) ** 2).mean()))

    # 不带 dropedge 的 baseline (重新跑一次 forward)
    model.eval()
    no_drop_preds = []
    with torch.no_grad():
        for _batch in validLoader:
            _batch = _batch.to(device)
            yhat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
            pred_valid = yhat[_batch.label_mask].cpu().numpy().flatten() if hasattr(_batch, 'label_mask') else yhat.cpu().numpy().flatten()
            no_drop_preds.append(pred_valid)
    no_drop_pred = np.concatenate(no_drop_preds)
    baseline_rmse = float(np.sqrt(((no_drop_pred - all_truths) ** 2).mean()))

    # 每 view std(across K)看 view 之间的差异性
    view_std = float(all_K_preds.std(axis=0).mean())

    return baseline_rmse, single_rmse, tta_rmse, view_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='/home/hhz6461/Urban_prediction_Semi/log_V1/13860_V2_semi_spatial/V2_semi_spatial.pt')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--dropedge_p', type=float, default=0.2)
    parser.add_argument('--scan_p', action='store_true', help='扫不同 dropedge_p')
    parser.add_argument('--scan_K', action='store_true', help='扫不同 K')
    args = parser.parse_args()

    # 设 env 与 13860 baseline 一致
    os.environ['V1_DATASET'] = 'V2'
    os.environ['V1_VAL_MODE'] = 'spatial'
    os.environ['V1_N_UNLABELED'] = '400'
    os.environ['V1_BATCH'] = '128'
    os.environ['V1_N_VALID_STATIONS'] = '8'
    os.environ['V1_SPATIAL_SEED'] = '42'
    os.environ['V1_KNN_K'] = '10'
    os.environ['V1_FPS_SEED'] = '0'

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[TTA] device = {device}")

    # 重建 V2 data + loader
    dataParam = {
        'geoMethod': 'average', 'nCompPCA': 40, 'window': 2,
        'poolSize': 12, 'batchSize': 128, 'thres': 0.1, 'geoFeatures': 'full',
    }
    _, validLoader, metadata, _ = v1_data.dataGen(dataParam, '/home/hhz6461/Urban_prediction_Semi/data')
    n_total = metadata['nNodes']
    tgt_scl_C = metadata.get('tgt_scl', 30.0)
    print(f"[TTA] n_total={n_total}, tgt_scl_C={tgt_scl_C:.4f}")

    # 重建 model + 加载 ckpt
    modelParam = {
        'HLD': 128, 'nMLP': 2, 'nGNN': 3, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'iDim': metadata['iDim'], 'oDim': metadata['oDim'],
        'BN': False, 'Dropout': False, 'conv_type': 'graphconv',
        'encoder_type': 'flat', 'modal_hid': 32,
    }
    model = network.GNN(modelParam).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[TTA] loaded ckpt (epoch={ckpt.get('epoch', '?')}, bestLoss={ckpt.get('bestLoss', '?')})")
    else:
        model.load_state_dict(ckpt)

    print(f"\n{'='*70}")
    print(f"TTA on {args.ckpt}")
    print(f"{'='*70}\n")

    if args.scan_K:
        print("Scanning K (fixed p=0.2):")
        for K in [1, 3, 5, 10, 20, 50]:
            t0 = time.time()
            baseline, single, tta, view_std = tta_inference(model, validLoader, device, n_total, K=K, dropedge_p=0.2)
            t = time.time() - t0
            delta = tta - baseline
            print(f"  K={K:>3}: baseline={baseline:.4f} (no aug), TTA={tta:.4f}, Δ={delta:+.4f} "
                  f"(view_std={view_std:.4f}, took {t:.1f}s)")
    elif args.scan_p:
        print("Scanning dropedge_p (fixed K=10):")
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            t0 = time.time()
            baseline, single, tta, view_std = tta_inference(model, validLoader, device, n_total, K=10, dropedge_p=p)
            t = time.time() - t0
            delta = tta - baseline
            print(f"  p={p:.2f}: baseline={baseline:.4f}, TTA={tta:.4f}, Δ={delta:+.4f} "
                  f"(view_std={view_std:.4f}, took {t:.1f}s)")
    else:
        t0 = time.time()
        baseline, single, tta, view_std = tta_inference(model, validLoader, device, n_total, K=args.K, dropedge_p=args.dropedge_p)
        t = time.time() - t0
        print(f"Baseline v_rmse (no TTA, single forward, no dropedge): {baseline:.6f}  ≈ {baseline*tgt_scl_C:.3f}°C")
        print(f"Single-view v_rmse (1 random dropedge forward):         {single:.6f}  ≈ {single*tgt_scl_C:.3f}°C")
        print(f"TTA v_rmse (K={args.K} dropedge views averaged):         {tta:.6f}  ≈ {tta*tgt_scl_C:.3f}°C")
        print(f"View std across K (diversity indicator):                {view_std:.4f}")
        print(f"Δ vs baseline (no aug):                                 {tta - baseline:+.6f}")
        print(f"Time: {t:.1f}s")


if __name__ == '__main__':
    main()

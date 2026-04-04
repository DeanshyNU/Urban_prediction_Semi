"""
Mean Teacher for Semi-Supervised GNN Regression
Uses data_semi.py for data loading (same as SimRegMatch/HPL for fair comparison).
Structured augmentation: two equal-strength, different-random views (η and η').
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pickle, data_semi, network_semi, utils
import os, wandb
from copy import deepcopy
from datetime import datetime
from torch_geometric.nn import GraphConv, SAGEConv, APPNP
import random as _random

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
n_unlabeled_env = os.environ.get('N_UNLABELED', '200')
use_fps = int(os.environ.get('USE_FPS', 0))
fps_tag = '_fps_score' if use_fps == 2 else ('_fps' if use_fps == 1 else '')
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = os.path.join(project_root, 'log', f'meanteacher_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'meanteacher_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


# ==================== Structured Augmentation (MT: equal strength, different random) ====================
WRF_VAR_GROUPS = {
    'Tair': (0, 9), 'Tskin': (9, 18), 'Tsoil': (18, 27),
    'Humid': (27, 36), 'Irrad': (36, 45), 'WindX': (45, 54), 'WindY': (54, 63),
}
WRF_STEP_DIM = 63
WRF_TOTAL_STEPS = 5
WRF_TOTAL_DIM = WRF_STEP_DIM * WRF_TOTAL_STEPS
CLMS_DIM = 3
URBAN_DIM = 17


def _get_feature_layout(total_dim):
    geo_dim = total_dim - WRF_TOTAL_DIM - CLMS_DIM - URBAN_DIM
    return {
        'wrf': (0, WRF_TOTAL_DIM),
        'clms': (WRF_TOTAL_DIM, WRF_TOTAL_DIM + CLMS_DIM),
        'urban': (WRF_TOTAL_DIM + CLMS_DIM, WRF_TOTAL_DIM + CLMS_DIM + URBAN_DIM),
        'geo': (WRF_TOTAL_DIM + CLMS_DIM + URBAN_DIM, total_dim),
        'geo_dim': geo_dim,
    }


def _build_wrf_var_indices(var_names):
    offsets = [i * WRF_STEP_DIM for i in range(WRF_TOTAL_STEPS)]
    indices = []
    for step_offset in offsets:
        for var in var_names:
            start, end = WRF_VAR_GROUPS[var]
            indices.extend(range(step_offset + start, step_offset + end))
    return indices


def _build_wrf_timestep_indices(step_idx):
    offset = step_idx * WRF_STEP_DIM
    return list(range(offset, offset + WRF_STEP_DIM))


def _drop_edges(edge_index, edge_attr, drop_rate):
    if drop_rate <= 0 or edge_index.size(1) == 0:
        return edge_index, edge_attr
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
    if keep_mask.sum() < num_edges * 0.5:
        keep_mask = torch.rand(num_edges, device=edge_index.device) > 0.5
    return edge_index[:, keep_mask], (edge_attr[keep_mask] if edge_attr is not None else None)


def structured_augment_mt(x, edge_index, edge_attr):
    """MT equal-strength augmentation: keep Tair, mask 2 med/low groups,
    mask 1 timestep, mask 20% GeoEmbed, drop 20% edges."""
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    # Mask 2 medium/low variable groups (keep Tair always)
    candidates = ['Tskin', 'Tsoil', 'Humid', 'Irrad', 'WindX', 'WindY', 'CLMS']
    chosen = _random.sample(candidates, 2)
    wrf_vars = [v for v in chosen if v != 'CLMS']
    if wrf_vars:
        x_aug[:, _build_wrf_var_indices(wrf_vars)] = 0.0
    if 'CLMS' in chosen:
        s, e = layout['clms']
        x_aug[:, s:e] = 0.0

    # Mask 1 non-current timestep
    step_to_mask = _random.choice([1, 2, 3, 4])  # 0=current, skip it
    x_aug[:, _build_wrf_timestep_indices(step_to_mask)] = 0.0

    # Mask 20% GeoEmbed
    geo_start, geo_end = layout['geo']
    geo_mask = torch.rand(layout['geo_dim'], device=x.device) > 0.20
    x_aug[:, geo_start:geo_end] *= geo_mask.float().unsqueeze(0)

    # Drop 20% edges
    edge_index_aug, edge_attr_aug = _drop_edges(edge_index, edge_attr, drop_rate=0.20)
    return x_aug, edge_index_aug, edge_attr_aug


# ==================== EMA update ====================
def update_ema(student, teacher, alpha):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)


# ==================== Main ====================
def main():
    # ========== Data (same as SimRegMatch/HPL) ==========
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': 12,
        'batchSize': int(os.environ.get('BATCH_SIZE', 128)),
        'thres': float(os.environ.get('THRES', 0.1)),
        'geoFeatures': 'full',
    }
    n_unlabeled = int(os.environ.get('N_UNLABELED', 200))
    os.environ['OUTPUT_DIR'] = output_dir
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)

    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    iDim = metadata['iDim']

    # ========== Hyperparameters ==========
    nEpoch = 5000
    lambda_U = float(os.environ.get('LAMBDA_U', 1.0))
    alpha_ema = 0.999
    lr = 1e-3

    # ========== Model ==========
    modelParam = {
        'iDim': iDim, 'oDim': 1, 'HLD': 128,
        'nGNN': 3, 'nMLP': 2, 'conv_type': conv_type,
    }
    student = network_semi.GNN(modelParam).to(device)
    teacher = deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)
    consistency_fn = torch.nn.MSELoss().to(device)

    # ========== Logging ==========
    modelName = f'geoEmbed_average_{conv_type}_meanteacher_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName
    chkptPath = f'{output_dir}/{modelName}.pt'

    wandb.init(
        project="Semi-supervised GNN",
        entity="urban_prediction",
        name=wandb_name,
        config={
            'method': 'Mean Teacher (structured augmentation)',
            'conv_type': conv_type, 'n_unlabeled': n_unlabeled,
            'lambda_U': lambda_U, 'alpha_ema': alpha_ema,
            'use_fps': use_fps,
            'augmentation': 'structured_mt (mask 2 var groups + 1 timestep + 20% geo + 20% edges)',
        }
    )

    with open(f'{output_dir}/{modelName}_param.pkl', 'wb') as f:
        pickle.dump({'modelParam': modelParam, 'dataParam': dataParam}, f)

    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("No checkpoint found, starting new model.", file=f)
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
        print(student, file=f)
        print("=" * 60, file=f)
        print("Mean Teacher Configuration (new framework, data_semi.py):", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} ({nNodes_labeled} labeled + {n_unlabeled} unlabeled)", file=f)
        print(f"  Lambda_U: {lambda_U}", file=f)
        print(f"  EMA alpha: {alpha_ema}", file=f)
        print(f"  Augmentation: structured MT (keep Tair, mask 2 groups + 1 timestep + 20% geo + 20% edges)", file=f)
        print(f"  Consistency loss: MSE on ALL nodes", file=f)
        print(f"  Feature dim: {iDim}", file=f)
        print(f"  Use FPS: {use_fps}", file=f)
        print("=" * 60, file=f)

    # ========== Training ==========
    bestLoss = np.inf
    hist = []

    for epoch in range(nEpoch):
        student.train()
        teacher.train()  # train mode for dropout noise
        epoch_labeled = 0
        epoch_consistency = 0
        epoch_pred_diff = 0
        n_batches = 0
        pred_all, truth_all = [], []

        for _n, _batch in enumerate(trainLoader):
            _batch = _batch.to(device)
            label_mask = _batch.label_mask
            opt.zero_grad()

            # Two equal-strength, different-random structured augmentations
            x_eta, ei_eta, ea_eta = structured_augment_mt(_batch.x, _batch.edge_index, _batch.edge_attr)
            x_eta2, ei_eta2, ea_eta2 = structured_augment_mt(_batch.x, _batch.edge_index, _batch.edge_attr)

            # Student forward (η)
            all_student = student(x_eta, ei_eta, ea_eta)
            # Teacher forward (η', no grad)
            with torch.no_grad():
                all_teacher = teacher(x_eta2, ei_eta2, ea_eta2)

            # Supervised loss (labeled only)
            labeled_pred = all_student[label_mask]
            labeled_loss = lossFn(labeled_pred, _batch.y[label_mask])

            # Consistency loss (ALL nodes)
            consistency_loss = consistency_fn(all_student, all_teacher)

            total_loss = labeled_loss + lambda_U * consistency_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            opt.step()

            # EMA update
            update_ema(student, teacher, alpha_ema)

            # Stats
            epoch_labeled += labeled_loss.item()
            epoch_consistency += consistency_loss.item()
            with torch.no_grad():
                epoch_pred_diff += (all_student - all_teacher).abs().mean().item()
            n_batches += 1

            pred_all.append(labeled_pred.detach().cpu())
            truth_all.append(_batch.y[label_mask].detach().cpu())

        scheduler.step()

        # Epoch averages
        avg_labeled = epoch_labeled / n_batches
        avg_consistency = epoch_consistency / n_batches
        avg_pred_diff = epoch_pred_diff / n_batches

        pred_cat = torch.cat(pred_all).numpy()
        truth_cat = torch.cat(truth_all).numpy()
        trainRMSE = np.sqrt(np.mean((pred_cat - truth_cat) ** 2))

        # Validation
        student.eval()
        val_pred, val_truth = [], []
        with torch.no_grad():
            for _batch in validLoader:
                _batch = _batch.to(device)
                _yHat = student(_batch.x, _batch.edge_index, _batch.edge_attr)
                val_pred.append(_yHat[_batch.label_mask].cpu())
                val_truth.append(_batch.y[_batch.label_mask].cpu())
        val_pred = torch.cat(val_pred).numpy()
        val_truth = torch.cat(val_truth).numpy()
        validRMSE = np.sqrt(np.mean((val_pred - val_truth) ** 2))
        validLoss = np.mean((val_pred - val_truth) ** 2)

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {avg_labeled + lambda_U * avg_consistency:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE:.3f}/{validRMSE:.3f}; LR {scheduler.get_last_lr()[0]:.6f}", file=f)
            print(f"  labeled_loss={avg_labeled:.4e} | consistency_loss={avg_consistency:.4e} | "
                  f"pred_diff={avg_pred_diff:.6f}", file=f)

        trainLoss_total = avg_labeled + lambda_U * avg_consistency
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss_total,
            'train/rmse': trainRMSE,
            'train/labeled_loss': avg_labeled,
            'train/consistency_loss': avg_consistency,
            'train/pred_diff': avg_pred_diff,
            'valid/loss': validLoss,
            'valid/rmse': validRMSE,
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': min(bestLoss, validRMSE),
        })

        if validRMSE < bestLoss:
            bestLoss = validRMSE
            torch.save({'model': student.state_dict(), 'teacher': teacher.state_dict(),
                         'epoch': epoch, 'bestLoss': bestLoss}, chkptPath)
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print(f"  Model saved. Best valid RMSE: {bestLoss:.6f}", file=f)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{nEpoch}: train={trainRMSE:.4f}, valid={validRMSE:.4f}, "
                  f"best={bestLoss:.4f}, pred_diff={avg_pred_diff:.6f}, consistency={avg_consistency:.6f}")

        hist.append([trainLoss_total, validLoss, trainRMSE, validRMSE])
        utils.plotHist(hist, modelName, output_dir=output_dir)

    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}", file=f)
    wandb.finish()


if __name__ == '__main__':
    main()

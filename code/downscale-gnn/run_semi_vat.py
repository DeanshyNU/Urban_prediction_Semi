"""
Semi-supervised GNN with Virtual Adversarial Training (VAT).
Based on: "Virtual Adversarial Training" (Miyato et al., 2018)

Key idea: Find the worst-case perturbation to input features that maximally
changes the prediction, then regularize the model to be robust to it.
Unlike MT/consistency where random perturbations barely change regression output,
VAT GUARANTEES a non-zero loss by targeting the model's weakest direction.

Usage:
  CONV_TYPE=graphconv python -u code/downscale-gnn/run_semi_vat.py
"""
import numpy as np
import torch, pickle, data_semi, network_semi, utils
import os
import wandb
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
LAMBDA_VAT = float(os.environ.get('LAMBDA_VAT', '1.0'))  # original paper uses 1.0
EPSILON = float(os.environ.get('EPSILON', '0.1'))       # adversarial perturbation magnitude
XI = float(os.environ.get('XI', '1e-6'))                 # small constant for initial perturbation
VAT_IP = int(os.environ.get('VAT_IP', '1'))              # power iteration steps for finding r_adv
RAMP_EPOCHS = int(os.environ.get('RAMP_EPOCHS', '100'))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = os.path.join(project_root, 'log', f'semi_vat_{conv_type}_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'semi_vat_{conv_type}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


def compute_vat_loss(model, x, edge_index, edge_attr, pred_original,
                     epsilon, xi, vat_ip):
    """
    Compute VAT loss:
    1. Find adversarial direction r_adv that maximally changes prediction
    2. Return MSE(pred_original, pred_adversarial)

    Args:
        model: GNN model (in train mode)
        x: input features (num_nodes, iDim)
        edge_index: graph edges
        edge_attr: edge weights
        pred_original: original predictions (num_nodes, 1), detached
        epsilon: perturbation magnitude
        xi: small constant for initial random direction
        vat_ip: number of power iteration steps

    Returns:
        vat_loss: scalar
        debug_stats: dict with detailed debug info
    """
    # Step 1: Initialize random perturbation
    d = torch.randn_like(x)
    # Normalize per-node (each node's perturbation has unit norm)
    d_flat = d.reshape(x.shape[0], -1)
    d_norm = d_flat / (torch.norm(d_flat, dim=1, keepdim=True) + 1e-12)
    d = d_norm.reshape(x.shape) * xi

    # Step 2: Power iteration to find adversarial direction
    for _ in range(vat_ip):
        d.requires_grad_(True)
        pred_perturbed = model(x + d, edge_index, edge_attr)
        adv_distance = torch.nn.functional.mse_loss(pred_perturbed, pred_original.detach())
        adv_distance.backward()
        d_grad = d.grad.detach()
        # IMPORTANT: clear model gradients accumulated from power iteration
        # (we only want d's gradient, not to update model params here)
        model.zero_grad()
        # Normalize gradient → adversarial direction
        d_flat = d_grad.reshape(x.shape[0], -1)
        d_norm = d_flat / (torch.norm(d_flat, dim=1, keepdim=True) + 1e-12)
        d = d_norm.reshape(x.shape).detach()

    # Step 3: Scale to epsilon magnitude
    r_adv = epsilon * d

    # Step 4: Compute VAT loss with adversarial perturbation
    pred_adversarial = model(x + r_adv, edge_index, edge_attr)
    vat_loss = torch.nn.functional.mse_loss(pred_adversarial, pred_original.detach())

    # ==================== Debug stats ====================
    with torch.no_grad():
        # Per-node VAT loss
        per_node_loss = (pred_adversarial - pred_original.detach()).pow(2).squeeze(-1)

        # Which feature dimensions are most perturbed (adversarial direction analysis)
        r_adv_abs = r_adv.abs()
        # Average across nodes, get per-feature adversarial strength
        per_feature_adv = r_adv_abs.mean(dim=0)  # (iDim,)

        # Prediction change magnitude
        pred_change = (pred_adversarial - pred_original.detach()).abs().squeeze(-1)

        debug_stats = {
            'vat_loss': vat_loss.item(),
            # Per-node analysis
            'per_node_loss_mean': per_node_loss.mean().item(),
            'per_node_loss_max': per_node_loss.max().item(),
            'per_node_loss_std': per_node_loss.std().item(),
            # Prediction change
            'pred_change_mean': pred_change.mean().item(),
            'pred_change_max': pred_change.max().item(),
            # Adversarial direction: top-5 most perturbed feature dimensions
            'top5_adv_dims': per_feature_adv.topk(min(5, per_feature_adv.shape[0])).indices.cpu().numpy().tolist(),
            'top5_adv_vals': per_feature_adv.topk(min(5, per_feature_adv.shape[0])).values.cpu().numpy().tolist(),
            # Perturbation magnitude
            'r_adv_norm_mean': r_adv.reshape(x.shape[0], -1).norm(dim=1).mean().item(),
        }

    return vat_loss, debug_stats


def train_vat(loader, model, lossFn, opt, scheduler, device,
              nNodes, nNodes_labeled, lambda_vat, epsilon, xi, vat_ip, epoch):
    """
    Training with labeled loss + VAT regularization on ALL nodes.
    """
    model.train()
    nNodes_unlabeled = nNodes - nNodes_labeled
    _LOSS = 0
    _LABELED_LOSS = 0
    _VAT_LOSS = 0
    pred, truth = [], []

    # Accumulate debug stats
    all_per_node_loss_labeled = []
    all_per_node_loss_unlabeled = []
    all_pred_change = []
    all_top_dims = []

    # Ramp up
    if epoch < RAMP_EPOCHS:
        current_lambda = lambda_vat * (epoch / RAMP_EPOCHS)
    else:
        current_lambda = lambda_vat

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        label_mask = _batch.label_mask

        # Forward pass (original, no perturbation)
        # Compute original predictions with no_grad for VAT target (following reference impl)
        with torch.no_grad():
            pred_original_detached = model(_batch.x, _batch.edge_index, _batch.edge_attr)
        # Also compute with grad for labeled loss
        pred_original = model(_batch.x, _batch.edge_index, _batch.edge_attr)

        # Labeled loss
        _yHat_labeled = pred_original[label_mask]
        _y_labeled = _batch.y[label_mask]
        labeled_loss = lossFn(_yHat_labeled, _y_labeled)

        # VAT loss on ALL nodes (labeled + unlabeled)
        vat_loss, debug = compute_vat_loss(
            model, _batch.x, _batch.edge_index, _batch.edge_attr,
            pred_original_detached, epsilon, xi, vat_ip
        )

        # Total loss
        total_loss = labeled_loss + current_lambda * vat_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _LABELED_LOSS += labeled_loss.item()
        _VAT_LOSS += vat_loss.item()

        # Collect debug info (first batch only for detailed per-node analysis)
        if _n == 0:
            with torch.no_grad():
                pred_change_all = debug['pred_change_mean']
                all_top_dims = debug['top5_adv_dims']

        # Record labeled predictions
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        _truth = _y_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1

    train_stats = {
        'labeled_loss': _LABELED_LOSS / n_batches,
        'vat_loss': _VAT_LOSS / n_batches,
        'current_lambda': current_lambda,
        'pred_change_mean': debug['pred_change_mean'],
        'pred_change_max': debug['pred_change_max'],
        'per_node_loss_mean': debug['per_node_loss_mean'],
        'per_node_loss_max': debug['per_node_loss_max'],
        'r_adv_norm': debug['r_adv_norm_mean'],
        'top5_adv_dims': all_top_dims,
        'top5_adv_vals': debug['top5_adv_vals'],
    }

    return (_LOSS / n_batches), _RMSE, truth, pred, train_stats


def main():
    # ==================== Data ====================
    dataParam = {
        'geoMethod': 'average',
        'nCompPCA': 40,
        'window': 2,
        'poolSize': int(os.environ.get('POOL_SIZE', '12')),
        'batchSize': 128,
        'thres': 0.1,
        'geoFeatures': 'full',
    }
    n_unlabeled = 200
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)

    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    nNodes_unlabeled = metadata['nNodes_unlabeled']
    adj_matrix = metadata['AdjMatrix']

    print(f"Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})")
    print(f"VAT config: epsilon={EPSILON}, lambda={LAMBDA_VAT}, xi={XI}, ip={VAT_IP}")

    # ==================== Model ====================
    nEpoch = 5000
    modelParam = {
        'HLD': 128,
        'nMLP': 2,
        'nGNN': 3,
        'nGAT': 1,
        'nHeads': 1,
        'K': 1,
        'iDim': metadata['iDim'],
        'oDim': metadata['oDim'],
        'BN': False,
        'Dropout': True,
        'conv_type': conv_type,
    }

    model = network_semi.GNN(modelParam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)

    bestLoss = np.inf
    hist = []

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_{conv_type}_vat_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName
    chkptPath = f'{output_dir}/{modelName}.pt'

    # ==================== W&B ====================
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=wandb_name,
        config={
            **dataParam,
            **modelParam,
            'method': 'vat',
            'lambda_vat': LAMBDA_VAT,
            'epsilon': EPSILON,
            'xi': XI,
            'vat_ip': VAT_IP,
            'ramp_epochs': RAMP_EPOCHS,
            'n_unlabeled': n_unlabeled,
            'nEpoch': nEpoch,
            'lr': 1e-3,
        }
    )

    # ==================== Log config ====================
    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("Virtual Adversarial Training (VAT) Configuration:", file=f)
        print(f"  Method: VAT (find worst-case perturbation, regularize for robustness)", file=f)
        print(f"  Epsilon: {EPSILON} (adversarial perturbation magnitude)", file=f)
        print(f"  Lambda_VAT: {LAMBDA_VAT} (VAT loss weight)", file=f)
        print(f"  Xi: {XI} (initial perturbation scale)", file=f)
        print(f"  Power iterations: {VAT_IP}", file=f)
        print(f"  Ramp-up: {RAMP_EPOCHS} epochs", file=f)
        print(f"  Key insight: adversarial perturbation guarantees non-zero loss", file=f)
        print(f"  → Solves MT's problem where random perturbations → loss ≈ 0 for regression", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})", file=f)
        print(f"  VAT applies to ALL {nNodes} nodes (labeled + unlabeled)", file=f)
        print(f"  Training samples: {len(trainLoader.dataset)}", file=f)
        print(f"  Epochs: {nEpoch}", file=f)
        print("", file=f)

        # Feature dimension info for interpreting adversarial directions
        print(f"  Feature breakdown (for interpreting top adversarial dims):", file=f)
        print(f"    WRF current: dims 0-62 (63 features)", file=f)
        print(f"    WRF history: dims 63-188 (126 features, 2 windows)", file=f)
        print(f"    WRF future:  dims 189-314 (126 features, 2 windows)", file=f)
        print(f"    CLMS:        dims 315-317 (3 features)", file=f)
        print(f"    UrbanFeat:   dims 318-334 (17 features)", file=f)
        print(f"    GeoEmbed:    dims 335+ (poolSize-dependent)", file=f)
        print("", file=f)

        nL = nNodes_labeled
        n_ll = int(np.sum(adj_matrix[:nL, :nL] > 0) // 2)
        n_uu = int(np.sum(adj_matrix[nL:, nL:] > 0) // 2)
        n_lu = int(np.sum(adj_matrix[:nL, nL:] > 0))
        print(f"  Graph: L-L={n_ll}, U-U={n_uu}, L-U={n_lu}, total={n_ll+n_uu+n_lu}", file=f)
        print("=" * 60, file=f)

    if torch.cuda.is_available():
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(model, file=f)

    with open(f'{output_dir}/{modelName}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)

    # ==================== Training loop ====================
    print(f"\nStarting VAT training for {nEpoch} epochs...")

    for epoch in range(nEpoch):
        trainLoss, trainRMSE, _, _, train_stats = train_vat(
            trainLoader, model, lossFn, opt, scheduler, device,
            nNodes, nNodes_labeled, LAMBDA_VAT, EPSILON, XI, VAT_IP, epoch
        )

        validLoss, validRMSE, _, _ = network_semi.test(
            validLoader, model, lossFn, device, nNodes, nNodes_labeled
        )

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {trainLoss:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE[0]:.3f}/{validRMSE[0]:.3f}; "
                  f"LR {scheduler.get_last_lr()[0]:.6f}; lambda={train_stats['current_lambda']:.4f}", file=f)
            print(f"  labeled_loss={train_stats['labeled_loss']:.4e} | "
                  f"vat_loss={train_stats['vat_loss']:.4e} | "
                  f"lambda*vat={train_stats['current_lambda'] * train_stats['vat_loss']:.4e}", file=f)
            print(f"  [VAT debug] pred_change: mean={train_stats['pred_change_mean']:.6f}, "
                  f"max={train_stats['pred_change_max']:.6f}", file=f)
            print(f"  [VAT debug] per_node_loss: mean={train_stats['per_node_loss_mean']:.6f}, "
                  f"max={train_stats['per_node_loss_max']:.6f}", file=f)
            print(f"  [VAT debug] r_adv_norm: {train_stats['r_adv_norm']:.6f}", file=f)
            # Interpret top adversarial dimensions
            top_dims = train_stats['top5_adv_dims']
            top_vals = train_stats['top5_adv_vals']
            dim_names = []
            for d in top_dims:
                if d < 63:
                    dim_names.append(f"WRF_curr[{d}]")
                elif d < 189:
                    dim_names.append(f"WRF_hist[{d-63}]")
                elif d < 315:
                    dim_names.append(f"WRF_fut[{d-189}]")
                elif d < 318:
                    dim_names.append(f"CLMS[{d-315}]")
                elif d < 335:
                    dim_names.append(f"Urban[{d-318}]")
                else:
                    dim_names.append(f"Geo[{d-335}]")
            print(f"  [VAT debug] top5 adversarial dims: {list(zip(dim_names, [f'{v:.4f}' for v in top_vals]))}", file=f)
            print(f"  RMSE std: {trainRMSE[1]:.3f}/{validRMSE[1]:.3f}; "
                  f"min: {trainRMSE[2]:.3f}/{validRMSE[2]:.3f}; "
                  f"max: {trainRMSE[3]:.3f}/{validRMSE[3]:.3f};", file=f)

        # W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/rmse_std': trainRMSE[1],
            'train/rmse_min': trainRMSE[2],
            'train/rmse_max': trainRMSE[3],
            'train/labeled_loss': train_stats['labeled_loss'],
            'train/vat_loss': train_stats['vat_loss'],
            'train/lambda_vat': train_stats['current_lambda'],
            'train/pred_change_mean': train_stats['pred_change_mean'],
            'train/pred_change_max': train_stats['pred_change_max'],
            'train/r_adv_norm': train_stats['r_adv_norm'],
            'valid/loss': validLoss,
            'valid/rmse': validRMSE[0],
            'valid/rmse_std': validRMSE[1],
            'valid/rmse_min': validRMSE[2],
            'valid/rmse_max': validRMSE[3],
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': bestLoss,
        })

        # Save best model
        if validRMSE[0] < bestLoss:
            bestLoss = validRMSE[0]
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print("  Model saved.", file=f)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'bestLoss': bestLoss,
                'hist': hist,
            }, chkptPath)

        hist.append([trainLoss, validLoss, trainRMSE[0], validRMSE[0]])
        utils.plotHist(hist, modelName, output_dir=output_dir)

    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    wandb.finish()


if __name__ == "__main__":
    main()

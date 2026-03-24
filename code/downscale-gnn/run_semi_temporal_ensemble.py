"""
Semi-supervised GNN with Temporal Ensembling.
Based on: "Temporal Ensembling for Semi-Supervised Learning" (Laine & Aila, 2017)

Key idea: Maintain a running EMA of model predictions across epochs as pseudo-labels.
Unlike MT (which averages model WEIGHTS), TE averages model OUTPUTS over time.
This ensures the target is fundamentally different from current prediction → loss != 0.

Usage:
  CONV_TYPE=graphconv python -u code/downscale-gnn/run_semi_temporal_ensemble.py
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
LAMBDA_TE = float(os.environ.get('LAMBDA_TE', '30.0'))  # original paper: 30 (SVHN) - 100 (CIFAR)
ALPHA_TE = float(os.environ.get('ALPHA_TE', '0.6'))    # EMA decay, matches original paper
RAMP_EPOCHS = int(os.environ.get('RAMP_EPOCHS', '80'))  # original paper: 80 epochs Gaussian ramp

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = os.path.join(project_root, 'log', f'temporal_ensemble_{conv_type}_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'temporal_ensemble_{conv_type}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


def gaussian_rampup(epoch, max_epochs):
    """Gaussian ramp-up from 0 to 1 over max_epochs (matches original TE paper)."""
    if epoch >= max_epochs:
        return 1.0
    return np.exp(-5.0 * (1.0 - epoch / max_epochs) ** 2)


def train_temporal_ensemble(loader, model, lossFn, opt, scheduler, device,
                            nNodes, nNodes_labeled, ensemble_preds, alpha_te,
                            lambda_te, epoch):
    """
    Temporal Ensembling training:
    1. Forward pass → get predictions for all nodes
    2. Labeled loss on labeled nodes
    3. TE loss: MSE(current_pred, accumulated_pred) on ALL nodes (original paper)
    4. Update accumulated predictions with EMA

    Note: DataLoader must NOT shuffle (predictions stored by position)
    """
    model.train()
    nNodes_unlabeled = nNodes - nNodes_labeled
    _LOSS = 0
    _LABELED_LOSS = 0
    _TE_LOSS = 0
    pred, truth = [], []

    # Collect current epoch's ALL node predictions for EMA update
    current_preds_all = []

    # Gaussian ramp-up (matches original paper, 80 epochs)
    rampup_weight = gaussian_rampup(epoch, RAMP_EPOCHS)
    current_lambda = lambda_te * rampup_weight

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)

        label_mask = _batch.label_mask
        batch_size = _batch.x.shape[0] // nNodes

        # 1. Labeled loss
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]
        labeled_loss = lossFn(_yHat_labeled, _y_labeled)

        # 2. TE loss on ALL nodes (matches original paper)
        # Reshape all predictions to (batch_size, nNodes)
        _yHat_all = _yHat.squeeze(-1).reshape(batch_size, nNodes)

        batch_start = _n * batch_size
        batch_end = batch_start + batch_size

        if ensemble_preds is not None and batch_end <= ensemble_preds.shape[0]:
            # Bias-corrected ensemble prediction
            correction = 1.0 / (1.0 - alpha_te ** (epoch + 1)) if epoch > 0 else 1.0
            te_target = torch.FloatTensor(ensemble_preds[batch_start:batch_end] * correction).to(device)

            te_loss = torch.nn.functional.mse_loss(_yHat_all, te_target.detach())
        else:
            te_loss = torch.tensor(0.0).to(device)

        total_loss = labeled_loss + current_lambda * te_loss

        total_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _LABELED_LOSS += labeled_loss.item()
        _TE_LOSS += te_loss.item()

        # Store ALL node predictions for EMA update
        current_preds_all.append(_yHat_all.detach().cpu().numpy())

        # Record labeled predictions for RMSE
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        _truth = _y_labeled.squeeze(-1).reshape(-1, nNodes_labeled)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1

    # Concatenate all current predictions (all nodes)
    current_preds = np.concatenate(current_preds_all, axis=0)  # (nSamples, nNodes)

    return (_LOSS / n_batches), _RMSE, truth, pred, _LABELED_LOSS / n_batches, _TE_LOSS / n_batches, current_preds, current_lambda


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
        'shuffle_train': False,  # TE requires fixed order (predictions stored by position)
    }
    n_unlabeled = 200
    trainLoader, validLoader, metadata, _ = data_semi.dataGen(dataParam, path, n_unlabeled=n_unlabeled)

    nNodes = metadata['nNodes']
    nNodes_labeled = metadata['nNodes_labeled']
    nNodes_unlabeled = metadata['nNodes_unlabeled']
    adj_matrix = metadata['AdjMatrix']
    nSamples_train = len(trainLoader.dataset)

    print(f"Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})")
    print(f"Training samples: {nSamples_train}")
    print(f"TE config: alpha={ALPHA_TE}, lambda={LAMBDA_TE}, ramp={RAMP_EPOCHS}")

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

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_{conv_type}_temporal_ensemble_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName
    chkptPath = f'{output_dir}/{modelName}.pt'

    # Initialize ensemble predictions for ALL nodes (zeros, updated via EMA)
    # Original paper stores predictions for all samples, not just unlabeled
    ensemble_preds = np.zeros((nSamples_train, nNodes))

    # ==================== W&B ====================
    wandb.init(
        entity="urban_prediction",
        project="Semi-supervised GNN",
        name=wandb_name,
        config={
            **dataParam,
            **modelParam,
            'method': 'temporal_ensembling',
            'alpha_te': ALPHA_TE,
            'lambda_te': LAMBDA_TE,
            'ramp_epochs': RAMP_EPOCHS,
            'n_unlabeled': n_unlabeled,
            'nEpoch': nEpoch,
            'lr': 1e-3,
        }
    )

    # ==================== Log config ====================
    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("Temporal Ensembling Configuration:", file=f)
        print(f"  Method: Temporal Ensembling (EMA of predictions across epochs)", file=f)
        print(f"  Alpha (EMA decay): {ALPHA_TE}", file=f)
        print(f"  Lambda_TE: {LAMBDA_TE}", file=f)
        print(f"  Ramp-up: {RAMP_EPOCHS} epochs", file=f)
        print(f"  Key insight: target = historical avg of predictions (not current model)", file=f)
        print(f"  → Unlike MT, loss does NOT converge to 0 for regression", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})", file=f)
        print(f"  Training samples: {nSamples_train}", file=f)
        print(f"  Ensemble array shape: ({nSamples_train}, {nNodes_unlabeled})", file=f)
        print(f"  Epochs: {nEpoch}", file=f)
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
    print(f"\nStarting Temporal Ensembling training for {nEpoch} epochs...")

    for epoch in range(nEpoch):
        trainLoss, trainRMSE, _, _, labeled_loss, te_loss, current_preds, current_lambda = train_temporal_ensemble(
            trainLoader, model, lossFn, opt, scheduler, device,
            nNodes, nNodes_labeled, ensemble_preds, ALPHA_TE,
            LAMBDA_TE, epoch
        )

        # Update ensemble predictions with EMA
        if current_preds.shape[0] == ensemble_preds.shape[0]:
            ensemble_preds = ALPHA_TE * ensemble_preds + (1 - ALPHA_TE) * current_preds
        else:
            # Handle size mismatch (last batch might be smaller)
            n = min(current_preds.shape[0], ensemble_preds.shape[0])
            ensemble_preds[:n] = ALPHA_TE * ensemble_preds[:n] + (1 - ALPHA_TE) * current_preds[:n]

        # Validate
        validLoss, validRMSE, _, _ = network_semi.test(
            validLoader, model, lossFn, device, nNodes, nNodes_labeled
        )

        # Debug: how different is ensemble from current predictions
        correction = 1.0 / (1.0 - ALPHA_TE ** (epoch + 1)) if epoch > 0 else 1.0
        corrected_ensemble = ensemble_preds * correction
        if current_preds.shape[0] == corrected_ensemble.shape[0]:
            pred_diff = np.abs(current_preds - corrected_ensemble)
            diff_mean = pred_diff.mean()
            diff_max = pred_diff.max()
            # Separate labeled vs unlabeled diff
            diff_labeled = pred_diff[:, :nNodes_labeled].mean()
            diff_unlabeled = pred_diff[:, nNodes_labeled:].mean()
        else:
            diff_mean = 0
            diff_max = 0
            diff_labeled = 0
            diff_unlabeled = 0

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {trainLoss:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE[0]:.3f}/{validRMSE[0]:.3f}; "
                  f"LR {scheduler.get_last_lr()[0]:.6f}; lambda={current_lambda:.4f}", file=f)
            print(f"  labeled_loss={labeled_loss:.4e} | te_loss={te_loss:.4e}", file=f)
            print(f"  [TE debug] current_vs_ensemble: diff_mean={diff_mean:.6f}, diff_max={diff_max:.6f}", file=f)
            print(f"  [TE debug] diff_labeled={diff_labeled:.6f}, diff_unlabeled={diff_unlabeled:.6f}", file=f)
            print(f"  [TE debug] ensemble_pred: mean={corrected_ensemble.mean():.4f}, "
                  f"std={corrected_ensemble.std():.4f}, "
                  f"range=[{corrected_ensemble.min():.4f}, {corrected_ensemble.max():.4f}]", file=f)
            print(f"  [TE debug] rampup_weight={gaussian_rampup(epoch, RAMP_EPOCHS):.4f}", file=f)
            print(f"  RMSE std: {trainRMSE[1]:.3f}/{validRMSE[1]:.3f}; "
                  f"min: {trainRMSE[2]:.3f}/{validRMSE[2]:.3f}; "
                  f"max: {trainRMSE[3]:.3f}/{validRMSE[3]:.3f};", file=f)

        # W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/labeled_loss': labeled_loss,
            'train/te_loss': te_loss,
            'train/lambda_te': current_lambda,
            'train/ensemble_diff_mean': diff_mean,
            'train/ensemble_diff_max': diff_max,
            'valid/loss': validLoss,
            'valid/rmse': validRMSE[0],
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

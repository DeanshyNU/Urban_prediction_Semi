"""
Semi-supervised GNN with Graph Laplacian Regularization.
Adds a smoothness constraint: connected nodes should have similar predictions.

L_total = L_labeled + lambda_lap * L_laplacian
L_laplacian = Σ w_ij * (pred_i - pred_j)^2  for all edges (i,j)

This enforces spatial smoothness on ALL nodes (labeled + unlabeled),
directly leveraging the physical prior that temperature is spatially smooth.

Usage:
  CONV_TYPE=graphconv LAMBDA_LAP=0.1 python -u code/downscale-gnn/run_semi_laplacian.py
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
LAMBDA_LAP = float(os.environ.get('LAMBDA_LAP', '0.1'))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = os.path.join(project_root, 'log', f'semi_laplacian_{conv_type}_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'semi_laplacian_{conv_type}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


def compute_laplacian_loss(predictions, edge_index, edge_attr):
    """
    Graph Laplacian regularization loss:
    L_lap = Σ w_ij * (pred_i - pred_j)^2 / num_edges

    Encourages connected nodes to have similar predictions.
    Applies to ALL nodes (labeled + unlabeled).

    Args:
        predictions: (num_nodes, 1) model predictions for all nodes
        edge_index: (2, num_edges) edge indices
        edge_attr: (num_edges,) or (num_edges, 1) edge weights

    Returns:
        lap_loss: scalar
        stats: dict with debug info
    """
    src = edge_index[0]
    dst = edge_index[1]

    pred_src = predictions[src].squeeze(-1)  # (num_edges,)
    pred_dst = predictions[dst].squeeze(-1)  # (num_edges,)

    # Edge weights
    if edge_attr.dim() > 1:
        weights = edge_attr.squeeze(-1)
    else:
        weights = edge_attr

    # Weighted squared difference
    diff_sq = (pred_src - pred_dst) ** 2
    weighted_diff = weights * diff_sq

    lap_loss = weighted_diff.mean()

    # Debug stats
    with torch.no_grad():
        stats = {
            'lap_loss': lap_loss.item(),
            'diff_mean': diff_sq.mean().item(),
            'diff_max': diff_sq.max().item(),
            'weighted_diff_mean': weighted_diff.mean().item(),
            'n_edges': len(src),
        }

    return lap_loss, stats


def train_laplacian(loader, model, lossFn, opt, scheduler, device,
                    nNodes, nNodes_labeled, lambda_lap):
    """
    Training with labeled loss + graph Laplacian regularization.
    """
    global _verification_printed
    model.train()
    _LOSS = 0
    _LABELED_LOSS = 0
    _LAP_LOSS = 0
    pred, truth = [], []

    for _n, _batch in enumerate(loader):
        _batch = _batch.to(device)
        _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)

        # Labeled loss
        label_mask = _batch.label_mask
        _yHat_labeled = _yHat[label_mask]
        _y_labeled = _batch.y[label_mask]
        labeled_loss = lossFn(_yHat_labeled, _y_labeled)

        # Laplacian loss on ALL nodes (labeled + unlabeled)
        lap_loss, lap_stats = compute_laplacian_loss(
            _yHat, _batch.edge_index, _batch.edge_attr
        )

        # Total loss
        total_loss = labeled_loss + lambda_lap * lap_loss

        total_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _LOSS += total_loss.item()
        _LABELED_LOSS += labeled_loss.item()
        _LAP_LOSS += lap_loss.item()

        # Record labeled predictions
        nNodes_labeled_count = nNodes_labeled
        _pred = _yHat_labeled.squeeze(-1).reshape(-1, nNodes_labeled_count)
        _truth = _y_labeled.squeeze(-1).reshape(-1, nNodes_labeled_count)
        pred += list(_pred.cpu().detach().numpy())
        truth += list(_truth.cpu().detach().numpy())

    scheduler.step()
    truth, pred = np.array(truth), np.array(pred)
    _RMSE = utils.RMSE(truth, pred)
    n_batches = _n + 1
    return (_LOSS / n_batches), _RMSE, truth, pred, _LABELED_LOSS / n_batches, _LAP_LOSS / n_batches


# Track first-epoch verification
_verification_printed = False


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
    print(f"Lambda_lap: {LAMBDA_LAP}")

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

    modelName = f'geoEmbed_{dataParam["geoMethod"]}_{conv_type}_laplacian_{n_unlabeled}unlabeled'
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
            'method': 'graph_laplacian',
            'lambda_lap': LAMBDA_LAP,
            'n_unlabeled': n_unlabeled,
            'nEpoch': nEpoch,
            'lr': 1e-3,
        }
    )

    # ==================== Log config ====================
    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("Graph Laplacian Regularization Configuration:", file=f)
        print(f"  Method: Semi-supervised GNN + Laplacian smoothness constraint", file=f)
        print(f"  Lambda_lap: {LAMBDA_LAP}", file=f)
        print(f"  L_total = L_labeled + {LAMBDA_LAP} * Σ w_ij * (pred_i - pred_j)^2", file=f)
        print(f"  Physical prior: temperature is spatially smooth", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} (labeled={nNodes_labeled}, unlabeled={nNodes_unlabeled})", file=f)
        print(f"  Training samples: {len(trainLoader.dataset)}", file=f)
        print(f"  Epochs: {nEpoch}", file=f)
        print(f"  Graph threshold: {dataParam['thres']}", file=f)
        print("", file=f)

        nL = nNodes_labeled
        n_ll = int(np.sum(adj_matrix[:nL, :nL] > 0) // 2)
        n_uu = int(np.sum(adj_matrix[nL:, nL:] > 0) // 2)
        n_lu = int(np.sum(adj_matrix[:nL, nL:] > 0))
        print(f"  Graph: L-L={n_ll}, U-U={n_uu}, L-U={n_lu}, total={n_ll+n_uu+n_lu}", file=f)
        print(f"  Laplacian acts on ALL {n_ll+n_uu+n_lu} edges (labeled+unlabeled)", file=f)
        print("=" * 60, file=f)

    if torch.cuda.is_available():
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(model, file=f)

    # Save metadata
    with open(f'{output_dir}/{modelName}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)

    # ==================== Training loop ====================
    print(f"\nStarting training with Graph Laplacian (lambda={LAMBDA_LAP})...")

    for epoch in range(nEpoch):
        trainLoss, trainRMSE, _, _, labeled_loss, lap_loss = train_laplacian(
            trainLoader, model, lossFn, opt, scheduler, device,
            nNodes, nNodes_labeled, LAMBDA_LAP
        )

        validLoss, validRMSE, _, _ = network_semi.test(
            validLoader, model, lossFn, device, nNodes, nNodes_labeled
        )

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {trainLoss:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE[0]:.3f}/{validRMSE[0]:.3f}; "
                  f"LR {scheduler.get_last_lr()[0]:.6f}", file=f)
            print(f"  labeled_loss={labeled_loss:.4e} | lap_loss={lap_loss:.4e} | "
                  f"lambda*lap={LAMBDA_LAP * lap_loss:.4e}", file=f)
            print(f"  RMSE std: {trainRMSE[1]:.3f}/{validRMSE[1]:.3f}; "
                  f"min: {trainRMSE[2]:.3f}/{validRMSE[2]:.3f}; "
                  f"max: {trainRMSE[3]:.3f}/{validRMSE[3]:.3f};", file=f)

        # W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': trainLoss,
            'train/rmse': trainRMSE[0],
            'train/labeled_loss': labeled_loss,
            'train/lap_loss': lap_loss,
            'train/lambda_lap_loss': LAMBDA_LAP * lap_loss,
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

"""
Pi-Model training module
"""
import numpy as np
import torch
import os
from utils import RMSE


def loadCheckPoint(modelName, model, opt, device, load=False, resetLr=False, lr=5e-5, predMode=False):
    """Load or initialize checkpoint"""
    chkptPath = f'./{modelName}.pt'
    if os.path.exists(chkptPath) and load:
        chkpt = torch.load(chkptPath, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        opt.load_state_dict(chkpt['opt_state_dict'])
        EPOCH = chkpt['epoch']
        bestLoss = chkpt['bestLoss']
        hist = chkpt['hist']
        with open(f'./{modelName}_log', 'a') as f:
            print("Checkpoint loaded.", file=f)
        if opt.param_groups[0]['lr'] < 1e-6 and resetLr:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            with open(f'./{modelName}_log', 'a') as f:
                print(f"Resetting LR from {opt.param_groups[0]['lr']} to {lr}", file=f)
    elif predMode:
        if not os.path.exists(chkptPath):
            chkptPath = f'./trainedModels/{modelName}.pt'
        chkpt = torch.load(chkptPath, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'])
        print("Checkpoint loaded.")
        return -1
    else:
        EPOCH = 0
        bestLoss = np.inf
        hist = []
        with open(f'./{modelName}_log', 'w') as f:
            print("No checkpoint found, starting new model.", file=f)
    return EPOCH, bestLoss, chkptPath, hist


def train_pimodel(loader, aug1_loader, aug2_loader, model, lossFn, consistency_loss_fn, opt, scheduler, device, nNodes, lambda_U=1.0):
    """
    Pi-Model training function

    Args:
        loader: labeled data loader
        aug1_loader: first augmentation of unlabeled data
        aug2_loader: second augmentation of unlabeled data
        model: GNN model
        lossFn: labeled loss function
        consistency_loss_fn: consistency loss function (MSE)
        opt: optimizer
        scheduler: learning rate scheduler
        device: device
        nNodes: node count
        lambda_U: unlabeled loss weight
    """
    model.train()
    total_labeled_loss = 0
    total_consistency_loss = 0
    total_batches = 0
    pred, truth = [], []

    for batch_idx, (batch, aug1_batch, aug2_batch) in enumerate(zip(loader, aug1_loader, aug2_loader)):
        batch = batch.to(device)
        aug1_batch = aug1_batch.to(device)
        aug2_batch = aug2_batch.to(device)

        opt.zero_grad(set_to_none=True)

        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labeled_loss = lossFn(logits, batch.y)

        pred1 = model(aug1_batch.x, aug1_batch.edge_index, aug1_batch.edge_attr)
        pred2 = model(aug2_batch.x, aug2_batch.edge_index, aug2_batch.edge_attr)
        consistency_loss = consistency_loss_fn(pred1, pred2)

        total_loss = labeled_loss + lambda_U * consistency_loss
        total_loss.backward()

        opt.step()

        _pred = logits.reshape(-1, nNodes)
        _truth = batch.y.reshape(-1, nNodes)

        total_labeled_loss += labeled_loss.item()
        total_consistency_loss += consistency_loss.item()

        pred.append(_pred.cpu().detach().numpy())
        truth.append(_truth.cpu().detach().numpy())
        total_batches += 1

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx + 1}: "
                  f"labeled_loss={labeled_loss.item():.4f}, "
                  f"consistency_loss={consistency_loss.item():.4f}, "
                  f"total_loss={total_loss.item():.4f}")

    scheduler.step()

    avg_labeled_loss = total_labeled_loss / total_batches
    avg_consistency_loss = total_consistency_loss / total_batches

    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    print(f"Epoch summary: "
          f"avg_labeled_loss={avg_labeled_loss:.4f}, "
          f"avg_consistency_loss={avg_consistency_loss:.4f}, "
          f"RMSE={rmse[0]:.4f}")

    return avg_labeled_loss + lambda_U * avg_consistency_loss, rmse, truth, pred


def test_pimodel(loader, model, lossFn, device, nNodes):
    """Pi-Model test function"""
    model.eval()
    total_labeled_loss = 0
    pred, truth = [], []

    if len(loader) == 0:
        raise ValueError("Data loader is empty.")

    with torch.no_grad():
        for batch in loader:
            if batch.x is None or batch.y is None:
                continue

            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)

            logits = logits.reshape(-1, nNodes)
            batch_y = batch.y.reshape(-1, nNodes)

            loss = lossFn(logits, batch_y)
            total_labeled_loss += loss.item()

            pred.append(logits.cpu().numpy())
            truth.append(batch_y.cpu().numpy())

    avg_labeled_loss = total_labeled_loss / len(loader)

    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    rmse = RMSE(truth, pred)

    return avg_labeled_loss, rmse, truth, pred

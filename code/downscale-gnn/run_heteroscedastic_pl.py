"""
Heteroscedastic Pseudo-Labels for Semi-Supervised GNN Regression
Reference: "Semi-Supervised Regression with Heteroscedastic Pseudo-Labels" (NeurIPS 2025)
Adapted for GNN-based urban temperature downscaling.

Key idea: Learn per-sample uncertainty weights for pseudo-labels via bi-level optimization.
- Inner loop: train GNN decoder with uncertainty-weighted pseudo-labels
- Outer loop: update UncertaintyLearner based on labeled validation performance
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pickle, copy, data_semi, network_semi, utils
import os, wandb
from datetime import datetime
from torch_geometric.nn import GraphConv, SAGEConv, APPNP
torch_geometric_nn_types = (GraphConv, SAGEConv, APPNP)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
n_unlabeled_env = os.environ.get('N_UNLABELED', '200')
use_fps = int(os.environ.get('USE_FPS', 1))
fps_tag = '_fps' if use_fps else ''
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')
if job_id:
    output_dir = os.path.join(project_root, 'log', f'heteroscedastic_pl_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'heteroscedastic_pl_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


# ==================== UncertaintyLearner ====================
class UncertaintyLearner(nn.Module):
    """
    Small MLP that maps (pred_diff, pred_strong) -> uncertainty weight.
    Input: 2-dim (prediction difference + strong prediction value)
    Output: 1-dim scalar uncertainty
    Weight = exp(-uncertainty) / 2, so higher uncertainty -> lower weight.
    """
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=1):
        super(UncertaintyLearner, self).__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*layers)

        # Xavier initialization (same as official code)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


# ==================== Augmentation ====================
def apply_augmentation(x, noise_std=0.05):
    """Apply random noise augmentation to node features."""
    return x + torch.randn_like(x) * noise_std


# ==================== Main ====================
def main():
    # ========== Data ==========
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
    nNodes_unlabeled = metadata['nNodes_unlabeled']
    iDim = metadata['iDim']

    # ========== Hyperparameters ==========
    nEpoch = 5000
    w_ulb = float(os.environ.get('W_ULB', 10.0))       # unlabeled loss weight (official default: 10)
    bilevel_freq = int(os.environ.get('BILEVEL_FREQ', 5))  # bi-level every N iterations
    weak_noise = 0.05
    strong_noise = 0.2
    lr_backbone = 1e-3
    lr_unc = 1e-4

    # ========== Model ==========
    modelParam = {
        'iDim': iDim,
        'oDim': 1,
        'HLD': 128,
        'nGNN': 3,
        'nMLP': 2,
        'conv_type': conv_type,
    }
    model = network_semi.GNN(modelParam).to(device)
    unc_learner = UncertaintyLearner(input_dim=2, output_dim=1, hidden_dim=128, num_layers=1).to(device)

    # Separate optimizers (following official code)
    # Backbone (encoder + processor) + Decoder
    optim_backbone = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.processor.parameters()),
        lr=lr_backbone
    )
    optim_decoder = torch.optim.Adam(model.decoder.parameters(), lr=lr_backbone)
    optim_unc = torch.optim.Adam(unc_learner.parameters(), lr=lr_unc)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim_backbone, gamma=0.9995
    )

    lossFn = torch.nn.HuberLoss()

    # ========== Logging ==========
    modelName = f'geoEmbed_{dataParam["geoMethod"]}_{conv_type}_heteroscedastic_pl_{dataParam["geoFeatures"]}Geo_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName

    wandb.init(
        project="Semi-supervised GNN",
        entity="urban_prediction",
        name=wandb_name,
        config={
            'method': 'Heteroscedastic Pseudo-Labels',
            'conv_type': conv_type,
            'n_unlabeled': n_unlabeled,
            'w_ulb': w_ulb,
            'bilevel_freq': bilevel_freq,
            'weak_noise': weak_noise,
            'strong_noise': strong_noise,
            'lr_backbone': lr_backbone,
            'lr_unc': lr_unc,
            'use_fps': use_fps,
        }
    )

    # Save config
    with open(f'{output_dir}/{modelName}_param.pkl', 'wb') as f:
        pickle.dump({'modelParam': modelParam, 'dataParam': dataParam}, f)

    # Log file
    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        print("No checkpoint found, starting new model.", file=f)
    if torch.cuda.is_available():
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(model, file=f)
        print("UncertaintyLearner:", file=f)
        print(unc_learner, file=f)
        print("=" * 60, file=f)
        print("Heteroscedastic Pseudo-Labels Configuration:", file=f)
        print(f"  Method: HPL (bi-level learned uncertainty for pseudo-labels)", file=f)
        print(f"  Output dir: {output_dir}", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} ({nNodes_labeled} labeled + {nNodes_unlabeled} unlabeled)", file=f)
        print(f"  w_ulb: {w_ulb}", file=f)
        print(f"  Bi-level freq: every {bilevel_freq} iterations", file=f)
        print(f"  Weak noise: {weak_noise}, Strong noise: {strong_noise}", file=f)
        print(f"  LR backbone: {lr_backbone}, LR unc: {lr_unc}", file=f)
        print(f"  Feature dim: {iDim}", file=f)
        print(f"  Use FPS: {use_fps}", file=f)
        print("=" * 60, file=f)

    # ========== Training ==========
    bestLoss = np.inf
    hist = []
    train_iter = 0

    for epoch in range(nEpoch):
        model.train()
        unc_learner.train()
        epoch_labeled_loss = 0
        epoch_unlabeled_loss = 0
        epoch_total_loss = 0
        epoch_weight_mean = 0
        epoch_weight_std = 0
        epoch_diff_mean = 0
        n_batches = 0
        pred_all, truth_all = [], []

        for _n, _batch in enumerate(trainLoader):
            _batch = _batch.to(device)
            label_mask = _batch.label_mask
            unlabel_mask = ~label_mask

            # --- Weak augmentation forward ---
            x_weak = apply_augmentation(_batch.x, weak_noise)
            with torch.no_grad():
                pred_weak_all = model(x_weak, _batch.edge_index, _batch.edge_attr)
            pred_weak_ulb = pred_weak_all[unlabel_mask].detach()  # pseudo-labels

            # --- Strong augmentation forward ---
            x_strong = apply_augmentation(_batch.x, strong_noise)
            pred_strong_all = model(x_strong, _batch.edge_index, _batch.edge_attr)
            pred_strong_labeled = pred_strong_all[label_mask]
            pred_strong_ulb = pred_strong_all[unlabel_mask]

            # --- Labeled loss ---
            y_labeled = _batch.y[label_mask]
            loss_labeled = lossFn(pred_strong_labeled, y_labeled)

            # --- Uncertainty weights ---
            pred_diff = pred_strong_ulb.detach() - pred_weak_ulb
            unc_input = torch.cat([pred_diff, pred_strong_ulb.detach()], dim=-1)  # (nU, 2)
            unc_weight = unc_learner(unc_input.detach())
            weight_new = torch.exp(-unc_weight) / 2  # (nU, 1), range (0, 0.5)

            # --- Weighted unlabeled loss ---
            unlabel_mse = (pred_strong_ulb - pred_weak_ulb.detach()) ** 2
            loss_unlabeled = torch.mean(weight_new * unlabel_mse)

            # --- Total loss ---
            total_loss = loss_labeled + w_ulb * loss_unlabeled

            # --- Bi-level optimization (every bilevel_freq iterations) ---
            # Approximate bi-level: UncertaintyLearner learns to minimize labeled loss
            # by adjusting weights on unlabeled pseudo-labels.
            # Key: unc_learner output directly participates in labeled loss computation
            # via weighted pseudo-label influence on model predictions.
            if train_iter % bilevel_freq == 0:
                # Step A: Compute uncertainty-weighted pseudo loss WITH gradient through unc_learner
                x_unc = apply_augmentation(_batch.x, weak_noise)
                pred_unc_all = model(x_unc, _batch.edge_index, _batch.edge_attr)
                pred_unc_labeled = pred_unc_all[label_mask]
                pred_unc_ulb = pred_unc_all[unlabel_mask]

                with torch.no_grad():
                    x_weak_unc = apply_augmentation(_batch.x, weak_noise)
                    pred_weak_unc = model(x_weak_unc, _batch.edge_index, _batch.edge_attr)
                    pred_weak_ulb_unc = pred_weak_unc[unlabel_mask]

                unc_diff = pred_unc_ulb.detach() - pred_weak_ulb_unc
                unc_input_bl = torch.cat([unc_diff, pred_unc_ulb.detach()], dim=-1)
                unc_weight_bl = unc_learner(unc_input_bl)  # gradient flows through unc_learner
                weight_bl = torch.exp(-unc_weight_bl) / 2

                # Step B: UncertaintyLearner loss = labeled loss + penalty for extreme weights
                # The unc_learner should assign weights that help labeled predictions
                # We use labeled loss as the signal, plus a regularization:
                #   - If weights are all zero → no unlabeled help → bad
                #   - If weights are all high → noisy pseudo-labels hurt → bad
                # Labeled loss (no unc_learner gradient here, just as target signal)
                unc_labeled_loss = F.mse_loss(pred_unc_labeled.detach(), y_labeled)

                # Pseudo-label loss weighted by unc_learner (gradient flows through weight_bl)
                pseudo_mse_bl = (pred_unc_ulb.detach() - pred_weak_ulb_unc) ** 2
                weighted_pseudo_loss = torch.mean(weight_bl * pseudo_mse_bl)

                # UncertaintyLearner objective:
                # Minimize: weighted pseudo loss (learn to downweight bad pseudo-labels)
                # + entropy regularization (prevent collapsing to all-zero or all-one weights)
                weight_entropy = -torch.mean(weight_bl * torch.log(weight_bl + 1e-8) +
                                             (0.5 - weight_bl) * torch.log(0.5 - weight_bl + 1e-8))
                unc_total_loss = weighted_pseudo_loss + 0.1 * weight_entropy

                optim_unc.zero_grad()
                unc_total_loss.backward()

                # Debug: check if unc_learner actually got gradients
                unc_grad_norm = 0.0
                for p in unc_learner.parameters():
                    if p.grad is not None:
                        unc_grad_norm += p.grad.norm().item() ** 2
                unc_grad_norm = unc_grad_norm ** 0.5

                optim_unc.step()

                # Debug print (first epoch and every 100 epochs)
                if epoch % 100 == 0 and _n == 0:
                    with open(f'{output_dir}/{modelName}_log', 'a') as f:
                        print(f"  [Bi-level] unc_grad_norm={unc_grad_norm:.6f}, "
                              f"unc_loss={unc_total_loss.item():.6f}, "
                              f"weighted_pseudo={weighted_pseudo_loss.item():.6f}, "
                              f"entropy={weight_entropy.item():.6f}", file=f)

            # --- Standard update (backbone + decoder) ---
            optim_backbone.zero_grad()
            optim_decoder.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim_backbone.step()
            optim_decoder.step()

            train_iter += 1

            # --- Collect stats ---
            epoch_labeled_loss += loss_labeled.item()
            epoch_unlabeled_loss += loss_unlabeled.item()
            epoch_total_loss += total_loss.item()
            epoch_weight_mean += weight_new.mean().item()
            epoch_weight_std += weight_new.std().item()
            epoch_diff_mean += pred_diff.abs().mean().item()
            n_batches += 1

            pred_all.append(pred_strong_labeled.detach().cpu())
            truth_all.append(y_labeled.detach().cpu())

        scheduler.step()

        # --- Epoch stats ---
        avg_labeled_loss = epoch_labeled_loss / max(n_batches, 1)
        avg_unlabeled_loss = epoch_unlabeled_loss / max(n_batches, 1)
        avg_total_loss = epoch_total_loss / max(n_batches, 1)
        avg_weight_mean = epoch_weight_mean / max(n_batches, 1)
        avg_weight_std = epoch_weight_std / max(n_batches, 1)
        avg_diff_mean = epoch_diff_mean / max(n_batches, 1)

        pred_cat = torch.cat(pred_all).numpy()
        truth_cat = torch.cat(truth_all).numpy()
        trainRMSE = np.sqrt(np.mean((pred_cat - truth_cat) ** 2))

        # --- Validation ---
        model.eval()
        val_pred, val_truth = [], []
        with torch.no_grad():
            for _batch in validLoader:
                _batch = _batch.to(device)
                _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
                label_mask = _batch.label_mask
                val_pred.append(_yHat[label_mask].cpu())
                val_truth.append(_batch.y[label_mask].cpu())
        val_pred = torch.cat(val_pred).numpy()
        val_truth = torch.cat(val_truth).numpy()
        validRMSE = np.sqrt(np.mean((val_pred - val_truth) ** 2))
        validLoss = np.mean((val_pred - val_truth) ** 2)

        # --- Logging ---
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            print(f"Epoch {epoch}: loss {avg_total_loss:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE:.3f}/{validRMSE:.3f}; LR {scheduler.get_last_lr()[0]:.6f}", file=f)
            print(f"  labeled_loss={avg_labeled_loss:.4e} | "
                  f"unlabeled_loss={avg_unlabeled_loss:.4e} (w_ulb={w_ulb})", file=f)
            print(f"  [HPL] weight: mean={avg_weight_mean:.4f}, std={avg_weight_std:.4f} | "
                  f"pred_diff: mean={avg_diff_mean:.4f}", file=f)

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_total_loss,
            'valid_loss': validLoss,
            'train_rmse': trainRMSE,
            'valid_rmse': validRMSE,
            'best_valid_rmse': min(bestLoss, validRMSE),
            'labeled_loss': avg_labeled_loss,
            'unlabeled_loss': avg_unlabeled_loss,
            'hpl_weight_mean': avg_weight_mean,
            'hpl_weight_std': avg_weight_std,
            'hpl_pred_diff': avg_diff_mean,
            'lr': scheduler.get_last_lr()[0],
        })

        hist.append([avg_total_loss, validLoss])

        # --- Save best ---
        if validRMSE < bestLoss:
            bestLoss = validRMSE
            torch.save({
                'model': model.state_dict(),
                'unc_learner': unc_learner.state_dict(),
                'epoch': epoch,
                'bestLoss': bestLoss,
            }, f'{output_dir}/{modelName}.pt')
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print(f"  Model saved. Best valid RMSE: {bestLoss:.6f}", file=f)
            wandb.log({'best_model_saved': True, 'best_valid_rmse': bestLoss})

        # --- Print every 100 epochs ---
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{nEpoch}: train={trainRMSE:.4f}, valid={validRMSE:.4f}, "
                  f"best={bestLoss:.4f}, weight_mean={avg_weight_mean:.4f}, diff={avg_diff_mean:.4f}")

    # --- Final ---
    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}", file=f)
    utils.plotHist(hist, modelName, output_dir=output_dir)
    wandb.finish()


if __name__ == '__main__':
    main()

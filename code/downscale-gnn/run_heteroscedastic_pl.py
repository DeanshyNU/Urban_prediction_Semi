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
import os, higher, wandb
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
    """DEPRECATED: kept for reference. Use structured augmentation below."""
    return x + torch.randn_like(x) * noise_std


# ==================== Structured Data Augmentation ====================
# Feature layout (1343 dims total, window=2):
#   WRF block: 315 dims = 5 time steps x 63 dims/step
#     Time step order: [current(63), hist[-2](63), hist[-1](63), fut[+1](63), fut[+2](63)]
#     Within each 63-dim step: 7 variables x 9 grid points
#       Tair(0-8), Tskin(9-17), Tsoil(18-26), Humid(27-35), Irrad(36-44), WindX(45-53), WindY(54-62)
#   CLMS: 3 dims (dynamic)
#   UrbanFeature: 17 dims (static)
#   GeoEmbed: remaining dims (static)
import random as _random

WRF_VAR_GROUPS = {
    'Tair': (0, 9), 'Tskin': (9, 18), 'Tsoil': (18, 27),
    'Humid': (27, 36), 'Irrad': (36, 45), 'WindX': (45, 54), 'WindY': (54, 63),
}
WRF_STEP_DIM = 63
WRF_WINDOW = 2
WRF_TOTAL_STEPS = 2 * WRF_WINDOW + 1  # 5
WRF_TOTAL_DIM = WRF_STEP_DIM * WRF_TOTAL_STEPS  # 315
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


def structured_augment_weak(x, edge_index, edge_attr):
    """Weak augmentation: keep Tair+Tskin, mask 1 low group, mask 10% GeoEmbed."""
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    low_choices = ['WindX', 'WindY', 'CLMS']
    chosen = _random.choice(low_choices)
    if chosen == 'CLMS':
        s, e = layout['clms']
        x_aug[:, s:e] = 0.0
    else:
        x_aug[:, _build_wrf_var_indices([chosen])] = 0.0

    geo_start, geo_end = layout['geo']
    geo_mask = torch.rand(layout['geo_dim'], device=x.device) > 0.10
    x_aug[:, geo_start:geo_end] *= geo_mask.float().unsqueeze(0)

    return x_aug, edge_index, edge_attr


def structured_augment_strong(x, edge_index, edge_attr):
    """Strong augmentation: keep Tair only, mask 2-3 med/low groups, mask 1 time step,
    mask 30% GeoEmbed, drop 25% edges."""
    total_dim = x.size(1)
    layout = _get_feature_layout(total_dim)
    x_aug = x.clone()

    candidates = ['Humid', 'Irrad', 'WindX', 'WindY', 'CLMS']
    n_mask = _random.choice([2, 3])
    chosen_groups = _random.sample(candidates, min(n_mask, len(candidates)))
    wrf_vars = [v for v in chosen_groups if v != 'CLMS']
    if wrf_vars:
        x_aug[:, _build_wrf_var_indices(wrf_vars)] = 0.0
    if 'CLMS' in chosen_groups:
        s, e = layout['clms']
        x_aug[:, s:e] = 0.0

    step_to_mask = _random.choice([1, 2, 3, 4])
    x_aug[:, _build_wrf_timestep_indices(step_to_mask)] = 0.0

    geo_start, geo_end = layout['geo']
    geo_mask = torch.rand(layout['geo_dim'], device=x.device) > 0.30
    x_aug[:, geo_start:geo_end] *= geo_mask.float().unsqueeze(0)

    edge_index_aug, edge_attr_aug = _drop_edges(edge_index, edge_attr, drop_rate=0.25)
    return x_aug, edge_index_aug, edge_attr_aug


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

    # Wrap decoder as nn.Sequential for higher library compatibility
    # (higher has issues with ModuleList, works with Sequential)
    # Replace model.decoder with Sequential version so model.forward() also uses it
    model.decoder = nn.Sequential(*list(model.decoder))
    model.decoder.to(device)

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
        print(f"  Augmentation: structured (weak=keep Tair+Tskin/mask 1 low/10%geo; "
              f"strong=keep Tair/mask 2-3 groups/1 timestep/30%geo/25%edges)", file=f)
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

            # --- Weak augmentation forward (structured: keep Tair+Tskin, mask 1 low group, 10% geo) ---
            x_weak, ei_weak, ea_weak = structured_augment_weak(
                _batch.x, _batch.edge_index, _batch.edge_attr)
            with torch.no_grad():
                pred_weak_all = model(x_weak, ei_weak, ea_weak)
            pred_weak_ulb = pred_weak_all[unlabel_mask].detach()  # pseudo-labels

            # --- Strong augmentation forward (structured: keep Tair, mask 2-3 groups, 1 timestep, 30% geo, 25% edges) ---
            x_strong, ei_strong, ea_strong = structured_augment_strong(
                _batch.x, _batch.edge_index, _batch.edge_attr)
            pred_strong_all = model(x_strong, ei_strong, ea_strong)
            pred_strong_labeled = pred_strong_all[label_mask]
            pred_strong_ulb = pred_strong_all[unlabel_mask]

            # --- Labeled loss ---
            y_labeled = _batch.y[label_mask]
            loss_labeled = lossFn(pred_strong_labeled, y_labeled)

            # --- Uncertainty weights (no grad, following official code) ---
            with torch.no_grad():
                pred_diff = pred_strong_ulb.detach() - pred_weak_ulb
                unc_input = torch.cat([pred_diff, pred_strong_ulb.detach()], dim=-1)  # (nU, 2)
                unc_weight = unc_learner(unc_input)
            weight_new = torch.exp(-unc_weight) / 2  # (nU, 1), range (0, 0.5)

            # --- Weighted unlabeled loss ---
            unlabel_mse = (pred_strong_ulb - pred_weak_ulb.detach()) ** 2
            loss_unlabeled = torch.mean(weight_new * unlabel_mse)

            # --- Total loss ---
            total_loss = loss_labeled + w_ulb * loss_unlabeled

            # --- Bi-level optimization FIRST (every bilevel_freq iterations) ---
            # Following official code: bi-level runs inside higher context,
            # then standard update runs outside (no conflict).
            if train_iter % bilevel_freq == 0:
                with higher.innerloop_ctx(model.decoder, optim_decoder) as (fmodel_dec, diff_opt):
                    # Inner forward: encoder + processor (not in higher), functional decoder
                    feat_inner = _batch.x.clone()
                    for _f in model.encoder:
                        feat_inner = _f(feat_inner)
                    for _f in model.processor:
                        if isinstance(_f, torch_geometric_nn_types):
                            feat_inner = _f(feat_inner, _batch.edge_index, _batch.edge_attr)
                        else:
                            feat_inner = _f(feat_inner)

                    # Labeled prediction via functional decoder (Sequential, call directly)
                    pred_inner = fmodel_dec(feat_inner)
                    inner_labeled_loss = F.mse_loss(pred_inner[label_mask], y_labeled)

                    # Weak augmentation pseudo-labels (no grad, structured)
                    with torch.no_grad():
                        x_weak_bl, ei_weak_bl, ea_weak_bl = structured_augment_weak(
                            _batch.x, _batch.edge_index, _batch.edge_attr)
                        feat_weak = x_weak_bl
                        for _f in model.encoder:
                            feat_weak = _f(feat_weak)
                        for _f in model.processor:
                            if isinstance(_f, torch_geometric_nn_types):
                                feat_weak = _f(feat_weak, ei_weak_bl, ea_weak_bl)
                            else:
                                feat_weak = _f(feat_weak)
                        pred_weak_bl = fmodel_dec(feat_weak)

                    # Strong augmentation predictions (structured)
                    x_strong_bl, ei_strong_bl, ea_strong_bl = structured_augment_strong(
                        _batch.x, _batch.edge_index, _batch.edge_attr)
                    feat_strong = x_strong_bl
                    for _f in model.encoder:
                        feat_strong = _f(feat_strong)
                    for _f in model.processor:
                        if isinstance(_f, torch_geometric_nn_types):
                            feat_strong = _f(feat_strong, ei_strong_bl, ea_strong_bl)
                        else:
                            feat_strong = _f(feat_strong)
                    pred_strong_bl = fmodel_dec(feat_strong)

                    pred_strong_ulb = pred_strong_bl[unlabel_mask]
                    pred_weak_ulb = pred_weak_bl[unlabel_mask]

                    # Uncertainty weights (gradient through unc_learner)
                    unc_input_bl = torch.cat([pred_strong_ulb.detach() - pred_weak_ulb.detach(),
                                             pred_strong_ulb.detach()], dim=-1)
                    unc_weight_bl = unc_learner(unc_input_bl)
                    weight_bl = torch.exp(-unc_weight_bl) / 2

                    # Inner unlabeled loss
                    unlabel_mse_bl = (pred_strong_ulb - pred_weak_ulb.detach()) ** 2
                    inner_ulb_loss = torch.mean(weight_bl * unlabel_mse_bl)

                    inner_loss = inner_labeled_loss + w_ulb * inner_ulb_loss
                    diff_opt.step(inner_loss)  # differentiable decoder update

                    # Outer: evaluate updated decoder on labeled (meta-update unc_learner)
                    pred_outer = fmodel_dec(feat_inner.detach())
                    outer_loss = F.mse_loss(pred_outer[label_mask], y_labeled)

                    optim_unc.zero_grad()
                    outer_loss.backward()

                    # Debug: detailed gradient flow check
                    unc_grad_norm = 0.0
                    unc_has_grad = []
                    for name, p in unc_learner.named_parameters():
                        if p.grad is not None:
                            unc_grad_norm += p.grad.norm().item() ** 2
                            unc_has_grad.append(f"{name}:{p.grad.norm().item():.6f}")
                        else:
                            unc_has_grad.append(f"{name}:None")
                    unc_grad_norm = unc_grad_norm ** 0.5

                    optim_unc.step()

                    # Print debug every 100 epochs + first 5 epochs
                    if (epoch % 100 == 0 or epoch < 5) and _n == 0:
                        with open(f'{output_dir}/{modelName}_log', 'a') as f:
                            print(f"  [Bi-level] unc_grad_norm={unc_grad_norm:.6f}, "
                                  f"outer_loss={outer_loss.item():.6f}, "
                                  f"inner_loss={inner_loss.item():.6f}, "
                                  f"weight_mean={weight_bl.mean().item():.4f}", file=f)
                            print(f"  [Bi-level] grad_detail: {unc_has_grad}", file=f)
                            print(f"  [Bi-level] weight_bl requires_grad={weight_bl.requires_grad}, "
                                  f"inner_loss requires_grad={inner_loss.requires_grad}, "
                                  f"outer_loss requires_grad={outer_loss.requires_grad}", file=f)

            # --- Standard update (backbone + decoder) ---
            # unc_learner weights computed with no_grad (following official code)
            model.zero_grad()
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
            'train/loss': avg_total_loss,
            'train/rmse': trainRMSE,
            'train/labeled_loss': avg_labeled_loss,
            'train/unlabeled_loss': avg_unlabeled_loss,
            'train/hpl_weight_mean': avg_weight_mean,
            'train/hpl_weight_std': avg_weight_std,
            'train/hpl_pred_diff': avg_diff_mean,
            'valid/loss': validLoss,
            'valid/rmse': validRMSE,
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': min(bestLoss, validRMSE),
        })

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

        hist.append([avg_total_loss, validLoss, trainRMSE, validRMSE])
        utils.plotHist(hist, modelName, output_dir=output_dir)

    # --- Final ---
    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}", file=f)
    wandb.finish()


if __name__ == '__main__':
    main()

"""
GraphMix for Semi-Supervised GNN Regression (Urban Temperature Downscaling).
Adapted from: "GraphMix: Improved Training of GNNs for Semi-Supervised Learning" (AAAI 2021)
Original: https://github.com/vikasverma1077/GraphMix

Key idea for regression:
  - GNN processes the graph normally (supervised loss on labeled nodes)
  - FCN shares encoder+decoder parameters with GNN (skips GNN processor)
  - FCN is trained with Manifold Mixup: interpolate hidden representations
    of random node pairs and their targets → virtual training samples
  - For unlabeled nodes: GNN predictions serve as soft targets for FCN mixup
  - Parameter sharing transfers FCN's regularization effect back to GNN

Differences from original (classification → regression):
  - Loss: HuberLoss instead of CrossEntropy
  - Mixup on continuous targets: y_mix = λ*y_i + (1-λ)*y_j (natural for regression)
  - No sharpening (not needed for continuous outputs)

Usage:
  NORM_MODE=global N_UNLABELED=400 USE_FPS=2 CONV_TYPE=graphconv \
  EDGE_MODE=no_uu LAMBDA_MIX=1.0 \
  python -u code/downscale-gnn/run_graphmix.py
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pickle, data_semi, network_semi, utils
import os, wandb
from datetime import datetime

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
    output_dir = os.path.join(project_root, 'log', f'graphmix_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}_job{job_id}')
else:
    output_dir = os.path.join(project_root, 'log', f'graphmix_{conv_type}{fps_tag}_{n_unlabeled_env}u_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


class GNNWithFCN(nn.Module):
    """
    GNN + FCN with shared encoder/decoder parameters.
    GNN path: encoder → GNN processor → decoder (uses graph structure)
    FCN path: encoder → decoder (skips GNN, no graph structure)
    """
    def __init__(self, modelParam):
        super().__init__()
        self.nGNNLayers = modelParam['nGNN']
        self.nMLPLayers = modelParam['nMLP']
        self.conv_type = modelParam.get('conv_type', 'graphconv').lower()
        _HLD = modelParam['HLD']

        # Shared encoder
        encoder_layers = []
        for _n in range(self.nMLPLayers):
            _in = modelParam['iDim'] if _n == 0 else _HLD
            encoder_layers.append(nn.Linear(_in, _HLD))
            encoder_layers.append(nn.PReLU(_HLD))
        self.encoder = nn.ModuleList(encoder_layers)

        # GNN processor (NOT shared with FCN)
        from torch_geometric.nn import GraphConv, SAGEConv  # noqa
        self._SAGEConv = SAGEConv  # [BugFix] store reference for _process_gnn
        processor_layers = []
        for _n in range(self.nGNNLayers):
            if self.conv_type == 'sageconv':
                processor_layers.append(SAGEConv(_HLD, _HLD, aggr='mean'))
            else:
                processor_layers.append(GraphConv(_HLD, _HLD, aggr='mean'))
            processor_layers.append(nn.PReLU(_HLD))
        self.processor = nn.ModuleList(processor_layers)

        # Shared decoder
        decoder_layers = []
        for _n in range(self.nMLPLayers):
            _out = modelParam['oDim'] if _n == self.nMLPLayers - 1 else _HLD
            decoder_layers.append(nn.Linear(_HLD, _out))
            if _n < self.nMLPLayers - 1:  # no activation on last layer (regression)
                decoder_layers.append(nn.PReLU(_out))
        self.decoder = nn.ModuleList(decoder_layers)

    def _encode(self, x):
        for f in self.encoder:
            x = f(x)
        return x

    def _process_gnn(self, h, edge_index, edge_attr):
        for f in self.processor:
            if hasattr(f, 'propagate'):  # GNN layer
                if isinstance(f, self._SAGEConv):
                    h = f(h, edge_index)
                else:
                    h = f(h, edge_index, edge_attr)
            else:
                h = f(h)
        return h

    def _decode(self, h):
        for f in self.decoder:
            h = f(h)
        return h

    def forward(self, x, edge_index, edge_attr):
        """Standard GNN forward: encoder → GNN → decoder"""
        h = self._encode(x)
        h = self._process_gnn(h, edge_index, edge_attr)
        return self._decode(h)

    def forward_fcn(self, x):
        """FCN forward: encoder → decoder (no GNN, shared params)"""
        h = self._encode(x)
        return self._decode(h), h  # return both prediction and hidden state

    def forward_fcn_mixup(self, x, targets, label_mask, mix_layer=1, lam=None):
        """
        FCN forward with Manifold Mixup.
        Randomly pairs nodes and interpolates their hidden representations and targets.

        Args:
            x: (N, iDim) input features
            targets: (N, 1) targets (real for labeled, GNN predictions for unlabeled)
            label_mask: (N,) bool, which nodes have real labels
            mix_layer: which encoder layer to apply mixup (0 or 1)
            lam: mixup ratio, if None sample from Beta(1,1)

        Returns:
            pred_mixed: predictions from mixed hidden states
            targets_mixed: interpolated targets
            lam: the mixup ratio used
        """
        if lam is None:
            lam = np.random.beta(1.0, 1.0)

        # Forward through encoder, applying mixup at specified layer
        h = x
        for layer_idx, f in enumerate(self.encoder):
            h = f(h)
            # Apply mixup after the specified layer's activation
            if layer_idx == mix_layer * 2 + 1:  # after PReLU of specified layer
                # Random permutation for pairing
                N = h.shape[0]
                perm = torch.randperm(N, device=h.device)
                h_mixed = lam * h + (1 - lam) * h[perm]
                targets_mixed = lam * targets + (1 - lam) * targets[perm]
                h = h_mixed

        # Decode
        pred_mixed = self._decode(h)
        return pred_mixed, targets_mixed, lam


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
    iDim = metadata['iDim']

    # ========== Hyperparameters ==========
    nEpoch = 5000
    lambda_mix = float(os.environ.get('LAMBDA_MIX', '1.0'))  # weight for FCN mixup loss
    lambda_lap = float(os.environ.get('LAMBDA_LAP', '0.1'))   # optional Laplacian
    use_laplacian = int(os.environ.get('USE_LAPLACIAN', '1'))  # combine with Laplacian

    # ========== Model ==========
    modelParam = {
        'iDim': iDim, 'oDim': 1, 'HLD': 128,
        'nGNN': 3, 'nMLP': 2, 'conv_type': conv_type,
    }
    model = GNNWithFCN(modelParam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = nn.HuberLoss().to(device)

    bestLoss = np.inf
    hist = []

    modelName = f'geoEmbed_average_{conv_type}_graphmix_{n_unlabeled}unlabeled'
    wandb_name = f'{modelName}_job{job_id}' if job_id else modelName
    chkptPath = f'{output_dir}/{modelName}.pt'

    # Adjacency for optional Laplacian
    Adj = metadata['AdjMatrix']
    _adj = Adj.copy()
    np.fill_diagonal(_adj, 0)
    _edge_src, _edge_dst = np.nonzero(_adj)
    _edge_w = torch.FloatTensor(_adj[_edge_src, _edge_dst]).to(device)
    _edge_src_t = torch.LongTensor(_edge_src).to(device)
    _edge_dst_t = torch.LongTensor(_edge_dst).to(device)

    # ========== Logging ==========
    wandb.init(
        project="Semi-supervised GNN", entity="urban_prediction",
        name=wandb_name,
        config={
            'method': 'GraphMix (regression)', 'conv_type': conv_type,
            'n_unlabeled': n_unlabeled, 'lambda_mix': lambda_mix,
            'lambda_lap': lambda_lap, 'use_laplacian': use_laplacian,
        }
    )

    with open(f'{output_dir}/{modelName}_param.pkl', 'wb') as f:
        pickle.dump({'modelParam': modelParam, 'dataParam': dataParam}, f)

    with open(f'{output_dir}/{modelName}_log', 'w') as f:
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
        print(model, file=f)
        print("=" * 60, file=f)
        print("GraphMix for Regression Configuration:", file=f)
        print(f"  Conv type: {conv_type}", file=f)
        print(f"  Nodes: {nNodes} ({nNodes_labeled} labeled + {n_unlabeled} unlabeled)", file=f)
        print(f"  Lambda_mix: {lambda_mix} (FCN mixup loss weight)", file=f)
        print(f"  Lambda_lap: {lambda_lap} (Laplacian, enabled={use_laplacian})", file=f)
        print(f"  Feature dim: {iDim}", file=f)
        print(f"  Key: GNN+FCN share encoder/decoder; FCN trains with Manifold Mixup", file=f)
        print(f"  Unlabeled nodes: GNN predictions → soft targets for FCN mixup", file=f)
        print("=" * 60, file=f)

    # ========== Training ==========
    norm_mode = os.environ.get('NORM_MODE', 'per_station')
    edge_mode = os.environ.get('EDGE_MODE', 'all')
    print(f"\nStarting GraphMix training for {nEpoch} epochs...")
    print(f"  lambda_mix={lambda_mix}, lambda_lap={lambda_lap}, use_laplacian={use_laplacian}")
    print(f"  NORM_MODE={norm_mode}, EDGE_MODE={edge_mode}")
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(f"  NORM_MODE={norm_mode}, EDGE_MODE={edge_mode}", file=f)

    for epoch in range(nEpoch):
        model.train()
        epoch_sup_loss = 0
        epoch_mix_loss = 0
        epoch_lap_loss = 0
        n_batches = 0
        pred_all, truth_all = [], []

        for _n, _batch in enumerate(trainLoader):
            _batch = _batch.to(device)
            label_mask = _batch.label_mask
            batch_size = _batch.x.shape[0] // nNodes
            opt.zero_grad()

            # [Debug] First batch: verify shapes and ranges
            if _n == 0 and epoch == 0:
                print(f"  [Debug GraphMix] batch_size={batch_size}, nNodes={nNodes}, "
                      f"x.shape={_batch.x.shape}, labeled_count={label_mask.sum().item()}")

            # === 1. GNN forward: standard supervised loss ===
            gnn_pred = model(_batch.x, _batch.edge_index, _batch.edge_attr)
            gnn_pred_labeled = gnn_pred[label_mask]
            y_labeled = _batch.y[label_mask]
            sup_loss = lossFn(gnn_pred_labeled, y_labeled)

            # [Debug] First batch prediction range check
            if _n == 0 and epoch == 0:
                with torch.no_grad():
                    print(f"  [Debug GraphMix] pred range=[{gnn_pred.min():.4f}, {gnn_pred.max():.4f}], "
                          f"target range=[{y_labeled.min():.4f}, {y_labeled.max():.4f}]")

            # === 2. FCN Mixup loss ===
            # Create soft targets: real labels for labeled, GNN predictions for unlabeled
            soft_targets = gnn_pred.detach().clone()  # (N, 1), GNN predictions for all nodes
            soft_targets[label_mask] = _batch.y[label_mask]  # overwrite with real labels for labeled

            # FCN forward with Manifold Mixup
            mix_layer = np.random.randint(0, model.nMLPLayers)  # random layer for mixup
            pred_mixed, targets_mixed, lam = model.forward_fcn_mixup(
                _batch.x, soft_targets, label_mask, mix_layer=mix_layer
            )
            mix_loss = lossFn(pred_mixed, targets_mixed.detach())

            # === 3. Laplacian loss (optional, on GNN predictions) ===
            lap_loss_val = 0
            if use_laplacian:
                gnn_pred_all = gnn_pred.squeeze(-1).reshape(batch_size, nNodes)
                lap_loss = 0
                for g in range(batch_size):
                    diff = gnn_pred_all[g][_edge_src_t] - gnn_pred_all[g][_edge_dst_t]
                    lap_loss = lap_loss + torch.mean(_edge_w * diff ** 2)
                lap_loss = lap_loss / batch_size
                lap_loss_val = lap_loss.item()
            else:
                lap_loss = 0

            # === Total loss ===
            total_loss = sup_loss + lambda_mix * mix_loss + lambda_lap * lap_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # Stats
            epoch_sup_loss += sup_loss.item()
            epoch_mix_loss += mix_loss.item()
            epoch_lap_loss += lap_loss_val
            n_batches += 1

            pred_all.append(gnn_pred_labeled.detach().cpu())
            truth_all.append(y_labeled.detach().cpu())

            # [Debug] First batch of first few epochs: check mixup quality
            if _n == 0 and epoch % 200 == 0:
                with torch.no_grad():
                    fcn_pred, _ = model.forward_fcn(_batch.x[:nNodes])
                    gnn_pred_first = gnn_pred[:nNodes]
                    pred_diff = (fcn_pred - gnn_pred_first).abs().mean().item()
                    print(f"  [Debug GraphMix] Epoch {epoch}: FCN-GNN pred diff={pred_diff:.4f}, "
                          f"lam={lam:.3f}, mix_layer={mix_layer}")

        scheduler.step()

        # Epoch stats
        avg_sup = epoch_sup_loss / n_batches
        avg_mix = epoch_mix_loss / n_batches
        avg_lap = epoch_lap_loss / n_batches

        pred_cat = torch.cat(pred_all).numpy()
        truth_cat = torch.cat(truth_all).numpy()
        trainRMSE = np.sqrt(np.mean((pred_cat - truth_cat) ** 2))

        # Validation (GNN path only)
        model.eval()
        val_pred, val_truth = [], []
        with torch.no_grad():
            for _batch in validLoader:
                _batch = _batch.to(device)
                _yHat = model(_batch.x, _batch.edge_index, _batch.edge_attr)
                val_pred.append(_yHat[_batch.label_mask].cpu())
                val_truth.append(_batch.y[_batch.label_mask].cpu())
        val_pred = torch.cat(val_pred).numpy()
        val_truth = torch.cat(val_truth).numpy()
        validRMSE = np.sqrt(np.mean((val_pred - val_truth) ** 2))
        validLoss = np.mean((val_pred - val_truth) ** 2)

        # Log
        with open(f'{output_dir}/{modelName}_log', 'a') as f:
            log_line = (f"Epoch {epoch}: RMSE {trainRMSE:.3f}/{validRMSE:.3f}; "
                        f"sup={avg_sup:.4e} | mix={avg_mix:.4e} | lap={avg_lap:.4e}; "
                        f"LR {scheduler.get_last_lr()[0]:.6f}")
            print(log_line, file=f)

        wandb.log({
            'epoch': epoch,
            'train/rmse': trainRMSE, 'valid/rmse': validRMSE,
            'train/sup_loss': avg_sup, 'train/mix_loss': avg_mix,
            'train/lap_loss': avg_lap,
            'learning_rate': scheduler.get_last_lr()[0],
            'best_valid_rmse': min(bestLoss, validRMSE),
        })

        if validRMSE < bestLoss:
            bestLoss = validRMSE
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'bestLoss': bestLoss}, chkptPath)
            with open(f'{output_dir}/{modelName}_log', 'a') as f:
                print(f"  Model saved. Best valid RMSE: {bestLoss:.6f}", file=f)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{nEpoch}: train={trainRMSE:.4f}, valid={validRMSE:.4f}, "
                  f"best={bestLoss:.4f}, mix={avg_mix:.4e}")

        hist.append([avg_sup + lambda_mix * avg_mix, validLoss, trainRMSE, validRMSE])
        utils.plotHist(hist, modelName, output_dir=output_dir)

    print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}")
    with open(f'{output_dir}/{modelName}_log', 'a') as f:
        print(f"\nTraining complete. Best valid RMSE: {bestLoss:.6f}", file=f)
    wandb.finish()


if __name__ == '__main__':
    main()

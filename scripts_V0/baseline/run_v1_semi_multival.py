"""
V1 semi-supervised baseline (no Lap, no pseudo) with 3 validation modes.

Identical feature schema for labeled and unlabeled stations:
  - WRF window (5 × 54 = 270)
  - raw geo at t (20)
  - geo embed (7-channel × 12×12 = 1008, static)
  Total: 1298 dim per (station, t)

V1 sup vs V1 semi differ ONLY in: presence of 400 unlabeled aux nodes in graph.
"""
import os
import sys
from datetime import datetime
import numpy as np
import torch
import mat73
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ORIGINAL_DIR = '/home/hhz6461/Urban_prediction_Semi/original_code'
DATA_DIR     = '/home/hhz6461/Urban_prediction_Semi/code/data'
SCRIPT_DIR   = '/home/hhz6461/Urban_prediction_Semi/scripts/baseline'
sys.path.insert(0, ORIGINAL_DIR)
sys.path.insert(0, SCRIPT_DIR)

import network as v1_network
import utils as v1_utils
from v1_data_shared import (
    load_v1_labeled, load_v1_unlabeled, normalize_unlabeled, build_features_per_t,
    T_END, WINDOW, GEO_EMB_DIM, N_WRF, RAW_GEO_DIM,
)


# ---------- Config ----------
job_id = os.environ.get('SLURM_JOB_ID', 'local')
val_mode = os.environ.get('V1_VAL_MODE', 'random').lower()
assert val_mode in ('random', 'sequential', 'spatial')
n_unlabeled     = int(os.environ.get('V1_N_UNLABELED', '400'))
n_valid_spatial = int(os.environ.get('V1_N_VALID_STATIONS', '10'))
spatial_seed    = int(os.environ.get('V1_SPATIAL_SEED', '42'))
fps_seed        = int(os.environ.get('V1_FPS_SEED', '0'))
temporal_frac   = float(os.environ.get('V1_TEMPORAL_FRAC', '0.8'))
nEpoch          = int(os.environ.get('V1_EPOCHS', '5000'))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f'/home/hhz6461/Urban_prediction_Semi/log/v1_semi_{val_mode}_job{job_id}_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
print(f"[V1 semi] output_dir={output_dir}, val_mode={val_mode}, n_unlabeled={n_unlabeled}")


# ---------- FPS for spatial valid hold-out ----------
def fps_select_stations(node_locations, n_select, seed=42):
    rng = np.random.default_rng(seed)
    coords = node_locations.T
    n_total = coords.shape[0]
    first = int(rng.integers(0, n_total))
    selected = [first]
    dists = np.linalg.norm(coords - coords[first], axis=1)
    for _ in range(n_select - 1):
        i = int(np.argmax(dists))
        selected.append(i)
        d_new = np.linalg.norm(coords - coords[i], axis=1)
        dists = np.minimum(dists, d_new)
    return sorted(selected)


# ---------- Build adjacency for combined (V1 labeled + 400 unlabeled) graph ----------
def build_combined_adj(locations, k=10):
    """k-NN graph on combined locations. Returns dense Adj (N, N)."""
    N = locations.shape[1]
    coords = locations.T
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    dmax = dist.max() + 1e-8
    dist_norm = dist / dmax
    Adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        idxs = np.argsort(dist[i])[1:k+1]
        for j in idxs:
            Adj[i, j] = 1.0 - dist_norm[i, j]
    Adj = np.maximum(Adj, Adj.T)
    return Adj


# ---------- Build dataset ----------
def build_dataset(wrf_l, raw_geo_l, geo_emb_l, targets_l,
                  wrf_u, raw_geo_u, geo_emb_u,
                  locations_all, val_mode):
    """
    Concat labeled + unlabeled at each timestep into a single graph (468 nodes).
    Each node has 1298-dim features (WRF window + raw_geo at t + geo_emb).
    """
    n_l = wrf_l.shape[0]
    n_u = wrf_u.shape[0]
    n_total = n_l + n_u
    T = wrf_l.shape[1]
    feat_dim = N_WRF * (2 * WINDOW + 1) + RAW_GEO_DIM + GEO_EMB_DIM
    print(f"  [Build] n_labeled={n_l}, n_unlabeled={n_u}, n_total={n_total}, T={T}")
    print(f"  [Build] feature dim per (station, t) = {feat_dim} "
          f"({N_WRF*(2*WINDOW+1)} WRF window + {RAW_GEO_DIM} raw_geo + {GEO_EMB_DIM} geo_emb)")

    # Adjacency (k-NN over combined locations)
    Adj = build_combined_adj(locations_all, k=10)
    edge_src, edge_dst = np.nonzero(Adj)
    edge_idx_t = torch.LongTensor(np.stack([edge_src, edge_dst]))
    edge_attr_t = torch.FloatTensor(Adj[edge_src, edge_dst])
    print(f"  [Adj] {n_total} nodes, {len(edge_src)} edges, mean_degree={(Adj > 0).sum()/n_total:.1f}")

    # Spatial split: hold out 10 V1 labeled stations (from 68)
    if val_mode == 'spatial':
        v_locs = locations_all[:, :n_l]
        valid_station_idx = fps_select_stations(v_locs, n_valid_spatial, seed=spatial_seed)
        train_station_idx = sorted(set(range(n_l)) - set(valid_station_idx))
        train_mask = torch.zeros(n_total, dtype=torch.bool); train_mask[train_station_idx] = True
        valid_mask = torch.zeros(n_total, dtype=torch.bool); valid_mask[valid_station_idx] = True
        print(f"  [Spatial] valid stations: {valid_station_idx}, train: {len(train_station_idx)}")
    else:
        train_mask = torch.zeros(n_total, dtype=torch.bool); train_mask[:n_l] = True
        valid_mask = train_mask.clone()

    # Build samples (per-timestep): combine labeled + unlabeled features
    samples = []
    for t in range(WINDOW, T - WINDOW):
        x_l = build_features_per_t(wrf_l, raw_geo_l, geo_emb_l, t)   # (n_l, 1298)
        x_u = build_features_per_t(wrf_u, raw_geo_u, geo_emb_u, t)   # (n_u, 1298)
        x_all = np.concatenate([x_l, x_u], axis=0)                    # (n_total, 1298)
        target_t = np.zeros((n_total,), dtype=np.float32)
        target_t[:n_l] = targets_l[:, t]
        d = Data(
            x=torch.FloatTensor(x_all),
            y=torch.FloatTensor(target_t).unsqueeze(-1),
            edge_index=edge_idx_t,
            edge_attr=edge_attr_t,
        )
        samples.append(d)
    nSamples = len(samples)

    # Split
    if val_mode == 'spatial':
        train_dataset, valid_dataset = [], []
        for s in samples:
            d_t = s.clone(); d_t.label_mask = train_mask.clone(); train_dataset.append(d_t)
            d_v = s.clone(); d_v.label_mask = valid_mask.clone(); valid_dataset.append(d_v)
    elif val_mode == 'sequential':
        n_train = int(nSamples * temporal_frac)
        train_dataset, valid_dataset = [], []
        for i, s in enumerate(samples):
            s.label_mask = train_mask.clone()
            (train_dataset if i < n_train else valid_dataset).append(s)
        print(f"  [Sequential] train timesteps={n_train}, valid={nSamples - n_train}")
    else:
        n_train = int(nSamples * 0.75)
        gen = torch.Generator().manual_seed(19)
        idx = torch.randperm(nSamples, generator=gen).tolist()
        for s in samples:
            s.label_mask = train_mask.clone()
        train_dataset = [samples[i] for i in idx[:n_train]]
        valid_dataset = [samples[i] for i in idx[n_train:]]
        print(f"  [Random] train timesteps={n_train}, valid={nSamples - n_train}")

    return train_dataset, valid_dataset, n_total


# ---------- Train / test (mask-aware) ----------
def train_step(loader, model, lossFn, opt, scheduler, device, n_total):
    model.train()
    losses, preds, truths = [], [], []
    for n, batch in enumerate(loader):
        batch = batch.to(device)
        yhat = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = batch.label_mask
        yhat_l, y_l = yhat[mask], batch.y[mask]
        loss = lossFn(yhat_l, y_l)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        bs = batch.x.shape[0] // n_total
        n_lbl_per = int(mask.sum().item() / max(bs, 1))
        preds += list(yhat_l.reshape(-1, n_lbl_per).detach().cpu().numpy())
        truths += list(y_l.reshape(-1, n_lbl_per).detach().cpu().numpy())
    scheduler.step()
    rmse = v1_utils.RMSE(np.array(truths), np.array(preds))
    return np.mean(losses), rmse


@torch.no_grad()
def test_step(loader, model, lossFn, device, n_total):
    model.eval()
    losses, preds, truths = [], [], []
    for batch in loader:
        batch = batch.to(device)
        yhat = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = batch.label_mask
        yhat_l, y_l = yhat[mask], batch.y[mask]
        loss = lossFn(yhat_l, y_l)
        losses.append(loss.item())
        bs = batch.x.shape[0] // n_total
        n_lbl_per = int(mask.sum().item() / max(bs, 1))
        preds += list(yhat_l.reshape(-1, n_lbl_per).cpu().numpy())
        truths += list(y_l.reshape(-1, n_lbl_per).cpu().numpy())
    rmse = v1_utils.RMSE(np.array(truths), np.array(preds))
    return np.mean(losses), rmse


# ---------- Main ----------
def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

    print("\n=== Loading V1 labeled ===")
    wrf_l, raw_geo_l, targets_l, locs_l, geo_emb_l_raw = load_v1_labeled(DATA_DIR)

    print("\n=== Loading V1 unlabeled (V2-source, V1-aligned) ===")
    wrf_u_raw, raw_geo_u_raw, locs_u, geo_emb_u_raw = load_v1_unlabeled(
        DATA_DIR, n_select=n_unlabeled, fps_seed=fps_seed)

    print("\n=== Normalizing unlabeled to V1-compatible scale ===")
    wrf_l, raw_geo_l, geo_emb_l, wrf_u, raw_geo_u, geo_emb_u = normalize_unlabeled(
        wrf_l, raw_geo_l, geo_emb_l_raw,
        wrf_u_raw, raw_geo_u_raw, geo_emb_u_raw)

    locations_all = np.concatenate([locs_l, locs_u], axis=1)

    print(f"\n=== Building dataset (val_mode={val_mode}) ===")
    train_ds, valid_ds, n_total = build_dataset(
        wrf_l, raw_geo_l, geo_emb_l, targets_l,
        wrf_u, raw_geo_u, geo_emb_u,
        locations_all, val_mode)

    bs = 64
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False)

    iDim = train_ds[0].x.shape[-1]
    modelParam = {
        'HLD': 128, 'nMLP': 2,
        'nGNN': 3, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'iDim': iDim, 'oDim': 1,
        'BN': False, 'Dropout': False,
    }
    model = v1_network.GNN(modelParam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)

    modelName = f'v1_semi_{val_mode}'
    log_path = f'./{modelName}_log'
    with open(log_path, 'a') as f:
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0), file=f)
        print(f"V1_VAL_MODE={val_mode}, n_unlabeled={n_unlabeled}, n_total={n_total}, "
              f"iDim={iDim}, n_valid_spatial={n_valid_spatial}, "
              f"nEpoch={nEpoch}", file=f)
        print(model, file=f)

    bestLoss = float('inf')
    print(f"\n=== Training {modelName} ({nEpoch} epochs, iDim={iDim}) ===")
    for epoch in range(nEpoch):
        tr_loss, tr_rmse = train_step(train_loader, model, lossFn, opt, sched, device, n_total)
        v_loss,  v_rmse  = test_step (valid_loader, model, lossFn, device, n_total)
        with open(log_path, 'a') as f:
            print(f"Epoch {epoch}: loss {tr_loss:.4e}/{v_loss:.4e}; "
                  f"RMSE {tr_rmse[0]:.4f}/{v_rmse[0]:.4f}; LR {sched.get_last_lr()}", file=f)
        if v_rmse[0] < bestLoss:
            bestLoss = v_rmse[0]
            with open(log_path, 'a') as f:
                print(f"Model saved. best_valid_RMSE={bestLoss:.4f}", file=f)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'opt_state_dict': opt.state_dict(), 'bestLoss': bestLoss},
                       f'./{modelName}.pt')

    print(f"\n[V1 semi, val_mode={val_mode}] best_valid_RMSE = {bestLoss:.4f}")
    with open(log_path, 'a') as f:
        print(f"\n=== FINAL ===\nbest_valid_RMSE={bestLoss:.4f}, val_mode={val_mode}, "
              f"n_unlabeled={n_unlabeled}", file=f)


if __name__ == '__main__':
    main()

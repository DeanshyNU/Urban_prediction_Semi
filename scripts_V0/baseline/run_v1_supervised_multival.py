"""
V1 supervised baseline with 3 validation modes (random/sequential/spatial).

Feature schema (matches v1_data_shared.py):
  - WRF window (5 × 54 = 270)
  - raw geo at t (20)
  - geo embed (7-channel × 12×12 = 1008)
  Total: 1298 dim per (station, t)

Drops 4 aux columns (hour/month/year/station_id) for fair comparison with V1 semi.
"""
import os
import sys
import copy
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
    load_v1_labeled, normalize_unlabeled, build_features_per_t,
    T_END, WINDOW, GEO_EMB_DIM, N_WRF, RAW_GEO_DIM,
)


# ---------- Config ----------
job_id   = os.environ.get('SLURM_JOB_ID', 'local')
val_mode = os.environ.get('V1_VAL_MODE', 'random').lower()
assert val_mode in ('random', 'sequential', 'spatial')
n_valid_spatial = int(os.environ.get('V1_N_VALID_STATIONS', '10'))
spatial_seed    = int(os.environ.get('V1_SPATIAL_SEED', '42'))
temporal_frac   = float(os.environ.get('V1_TEMPORAL_FRAC', '0.8'))
nEpoch          = int(os.environ.get('V1_EPOCHS', '5000'))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f'/home/hhz6461/Urban_prediction_Semi/log/v1_supervised_{val_mode}_job{job_id}_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
print(f"[V1 sup] output_dir={output_dir}, val_mode={val_mode}")


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


# ---------- Adjacency (same as original_code/data.py V1) ----------
def build_v1_adj(thres=0.1):
    _ajm = mat73.loadmat(f'{DATA_DIR}/GNN_N1_AJM.mat')
    _dist  = _ajm['dist']
    _simiW = _ajm['similarity']
    _distW = np.exp(-_dist)
    _off, _scl = np.min(_distW), np.max(_distW) - np.min(_distW)
    _distW = (_distW - _off) / max(_scl, 1e-8)
    Adj = np.abs(_simiW * _distW)
    Adj[Adj < thres] = 0.0
    np.fill_diagonal(Adj, 0)
    return Adj


# ---------- Build dataset ----------
def build_dataset(wrf, raw_geo, geo_emb, targets, locations, val_mode):
    """
    Args:
        wrf:       (68, T, 54)  normalized
        raw_geo:   (68, T, 20)  normalized
        geo_emb:   (68, 1008)   normalized, static
        targets:   (68, T)
        locations: (2, 68)
    Returns: train_dataset, valid_dataset, n_total_nodes
    """
    n_stations = wrf.shape[0]
    T = wrf.shape[1]
    print(f"  [Build] n_stations={n_stations}, T={T}")
    print(f"  [Build] feature dim per (station, t) = {N_WRF*(2*WINDOW+1) + RAW_GEO_DIM + GEO_EMB_DIM} "
          f"({N_WRF*(2*WINDOW+1)} WRF window + {RAW_GEO_DIM} raw_geo + {GEO_EMB_DIM} geo_emb)")

    # Adjacency (same as original_code, 68 nodes)
    Adj = build_v1_adj(thres=0.1)
    edge_src, edge_dst = np.nonzero(Adj)
    edge_idx_t = torch.LongTensor(np.stack([edge_src, edge_dst]))
    edge_attr_t = torch.FloatTensor(Adj[edge_src, edge_dst])
    print(f"  [Adj] {n_stations} nodes, {len(edge_src)} edges, "
          f"weight range=[{Adj[Adj>0].min():.3f}, {Adj[Adj>0].max():.3f}]")

    # Spatial split: pick 10 valid stations (FPS, seed)
    if val_mode == 'spatial':
        valid_station_idx = fps_select_stations(locations, n_valid_spatial, seed=spatial_seed)
        train_station_idx = sorted(set(range(n_stations)) - set(valid_station_idx))
        print(f"  [Spatial] valid stations: {valid_station_idx}, train: {len(train_station_idx)}")
        train_mask = torch.zeros(n_stations, dtype=torch.bool); train_mask[train_station_idx] = True
        valid_mask = torch.zeros(n_stations, dtype=torch.bool); valid_mask[valid_station_idx] = True
    else:
        train_mask = torch.ones(n_stations, dtype=torch.bool)
        valid_mask = train_mask.clone()

    # Build samples (per-timestep)
    samples = []
    for t in range(WINDOW, T - WINDOW):
        x_t = build_features_per_t(wrf, raw_geo, geo_emb, t)        # (68, 1298)
        y_t = targets[:, t:t+1]                                      # (68, 1)
        d = Data(
            x=torch.FloatTensor(x_t),
            y=torch.FloatTensor(y_t),
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

    return train_dataset, valid_dataset, n_stations


# ---------- Train / test with mask ----------
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
    wrf_l, raw_geo_l, targets, locations, geo_emb_l_raw = load_v1_labeled(DATA_DIR)

    # Normalize geo_emb labeled by its own min/max (no unlabeled to compare)
    g_min, g_max = geo_emb_l_raw.min(axis=0), geo_emb_l_raw.max(axis=0)
    g_rng = np.where(g_max - g_min > 0, g_max - g_min, 1.0)
    geo_emb = ((geo_emb_l_raw - g_min) / g_rng).astype(np.float32)
    print(f"  [Norm] geo_emb after [0,1]: min={geo_emb.min():.3f}, max={geo_emb.max():.3f}")

    print(f"\n=== Building dataset (val_mode={val_mode}) ===")
    train_ds, valid_ds, n_total = build_dataset(wrf_l, raw_geo_l, geo_emb, targets, locations, val_mode)
    bs = 128
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

    modelName = f'v1_supervised_{val_mode}'
    log_path = f'./{modelName}_log'
    with open(log_path, 'a') as f:
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0), file=f)
        print(f"V1_VAL_MODE={val_mode}, iDim={iDim}, "
              f"n_valid_spatial={n_valid_spatial}, spatial_seed={spatial_seed}, "
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

    print(f"\n[V1 sup, val_mode={val_mode}] best_valid_RMSE = {bestLoss:.4f}")
    with open(log_path, 'a') as f:
        print(f"\n=== FINAL ===\nbest_valid_RMSE={bestLoss:.4f}, val_mode={val_mode}", file=f)


if __name__ == '__main__':
    main()

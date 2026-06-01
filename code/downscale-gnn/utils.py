import numpy as np
import matplotlib.pyplot as plt
import os, torch

def plotPrediction(data,modelName):
    _truth,_pred,_rmse = data
    plt.figure(figsize=(10,5))
    plt.plot(_pred,'-r',label='Prediction')
    plt.plot(_truth,'--k',label='Truth')
    plt.title(f'{modelName}, RMSE {_rmse:1.4f}')
    plt.xlim([0,3000])
    plt.ylim([0,1])
    plt.ylabel("Normalized Tobs", fontsize=20)
    plt.legend(loc='upper right')
    plt.savefig(f'./{modelName}_prediction.png',dpi=200)
    plt.close()

def plotHist(hist,modelName,output_dir='./'):
    _,_ax = plt.subplots(1,2,figsize=(20,10))
    _trainLoss,_validLoss,_trainRMSE,_validRMSE  = np.array(hist).T
    _ax[0].semilogy(_trainLoss,label='Training')
    _ax[0].semilogy(_validLoss,label='Validation')
    _ax[0].set_title("Loss")
    _ax[0].legend('upper right')
    _ax[1].semilogy(_trainRMSE,label="Training")
    _ax[1].semilogy(_validRMSE,label="Validation")
    _ax[1].set_title("Station-wised Mean RMSE")
    _ax[1].legend('upper right')
    plt.savefig(f'{output_dir}/{modelName}_hist.png',dpi=200)
    plt.close()

def RMSE(truth,pred,axis=0):
    # Truth and prediction should have (nSteps,nStations)
    # This function computes RMSE per station then return mean RMSE by station.
    _rmse = np.linalg.norm(pred-truth,axis=axis)/np.sqrt(pred.shape[0])
    return [np.mean(_rmse),np.std(_rmse),np.min(_rmse),np.max(_rmse)]

def MBE(truth, pred, axis=0):
    """Mean Bias Error per station. Returns [mean, std, min, max] across stations."""
    _mbe = np.mean(pred - truth, axis=axis)
    return [np.mean(_mbe), np.std(_mbe), np.min(_mbe), np.max(_mbe)]

def MAE(truth, pred, axis=0):
    """Mean Absolute Error per station. Returns [mean, std, min, max] across stations."""
    _mae = np.mean(np.abs(pred - truth), axis=axis)
    return [np.mean(_mae), np.std(_mae), np.min(_mae), np.max(_mae)]

def compute_all_metrics(truth, pred, scl=1.0):
    """
    Compute 6 metrics: {RMSE, MBE, MAE} × {normalized, Celsius}.
    All operate on per-station vectors then average.
    Args:
        truth, pred: (nSteps, nStations) in normalized space
        scl: scalar multiplier to convert normalized → Celsius
    Returns:
        dict with rmse_norm, mbe_norm, mae_norm, rmse_C, mbe_C, mae_C
    """
    rmse = RMSE(truth, pred)[0]   # mean across stations
    mbe  = MBE(truth, pred)[0]
    mae  = MAE(truth, pred)[0]
    return {
        'rmse_norm': rmse,
        'mbe_norm':  mbe,
        'mae_norm':  mae,
        'rmse_C':    rmse * scl,
        'mbe_C':     mbe * scl,
        'mae_C':     mae * scl,
    }

def directional_pool(patches, n_dirs=4):
    """
    Direction-aware pooling for spatial patches.

    Splits each station's HxW patch into n_dirs angular sectors centered at the station,
    averages each sector independently. Captures e.g. "south-side trees" vs "north-side buildings".

    Args:
        patches: torch.FloatTensor of shape (N, C, H, W) — N stations, C channels, HxW patch
        n_dirs: number of angular sectors (default 4 = N/E/S/W)

    Returns:
        out: torch.FloatTensor of shape (N, C * n_dirs)
    """
    N, C, H, W = patches.shape
    cy, cx = H / 2.0, W / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) - cy,
        torch.arange(W, dtype=torch.float32) - cx,
        indexing='ij',
    )
    # angle from center, in [-π, π]; 0 = east, π/2 = north
    angle = torch.atan2(-yy, xx)  # invert y so "up" = north
    # bin into n_dirs equal sectors starting at 0
    sector = ((angle + np.pi) / (2 * np.pi) * n_dirs).long().clamp(0, n_dirs - 1)
    masks = [(sector == d).float() for d in range(n_dirs)]   # each (H, W)
    out = []
    dir_names = ['East', 'North', 'West', 'South'] if n_dirs == 4 else [f'D{i}' for i in range(n_dirs)]
    for d in range(n_dirs):
        m = masks[d].to(patches.device)
        weighted = (patches * m).sum(dim=(2, 3))             # (N, C)
        denom = m.sum() + 1e-8
        sector_avg = weighted / denom                         # (N, C)
        out.append(sector_avg)
        # Diagnostic: print per-direction per-channel stats (mean over stations)
        per_ch_mean = sector_avg.mean(dim=0).cpu().numpy()    # (C,)
        per_ch_std = sector_avg.std(dim=0).cpu().numpy()
        print(f"  [DirPool] {dir_names[d] if d < len(dir_names) else f'D{d}'}: "
              f"per-channel mean={per_ch_mean.round(3).tolist()}, "
              f"std={per_ch_std.round(3).tolist()}")
    return torch.cat(out, dim=1)                              # (N, C*n_dirs)


def MinMax(x):
    # Min-Max normalization based on the last dimension of the raw data
    _axis = tuple(np.arange(0,x.ndim-1))
    _min, _max = np.min(x,axis=_axis), np.max(x,axis=_axis)
    _idx = np.where(_min==_max)
    _min[_idx], _max[_idx] = 0., 1.
    _off = _min
    _scl = (_max-_min)
    return (x-_off)/_scl, _off, _scl

import numpy as np
import matplotlib.pyplot as plt
import os, torch

# 画单 station 的预测 vs 真值时序图(legacy,主流程没用,留给 predict.py)
def plotPrediction(data,modelName):
    """画一条 station 的 truth vs prediction 时间序列图(留 predict.py 用)。"""
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

# 每 epoch 调一次,画 train/valid 的 loss + RMSE 双子图,保存到 {modelName}_hist.png
def plotHist(hist,modelName):
    """每 epoch 调用一次,画 train/valid 的 loss 和 RMSE 曲线 → `{modelName}_hist.png`。

    输入 hist 是 [trainLoss, validLoss, trainRMSE, validRMSE] 的逐 epoch 列表。
    """
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
    plt.savefig(f'./{modelName}_hist.png',dpi=200)
    plt.close()

# RMSE per station 后跨 station 聚合,返回 [mean, std, min, max]
def RMSE(truth,pred,axis=0):
    """RMSE per station,然后跨 station 取 mean / std / min / max。

    输入 (nSteps, nStations) 数组,返回 [mean, std, min, max] 4 元组。
    与 original_code 完全一致。
    """
    # Truth and prediction should have (nSteps,nStations)
    # This function computes RMSE per station then return mean RMSE by station.
    _rmse = np.linalg.norm(pred-truth,axis=axis)/np.sqrt(pred.shape[0])
    return [np.mean(_rmse),np.std(_rmse),np.min(_rmse),np.max(_rmse)]

# Mean Bias Error per station 后聚合,正值=系统性高估、负值=系统性低估
def MBE(truth, pred, axis=0):
    """Mean Bias Error(平均偏差):pred - truth 的均值,衡量系统性高估/低估。

    Per station 算后跨 station 聚合。返回 [mean, std, min, max]。
    """
    _mbe = np.mean(pred - truth, axis=axis)
    return [np.mean(_mbe), np.std(_mbe), np.min(_mbe), np.max(_mbe)]

# Mean Absolute Error per station 后聚合,对异常值不敏感(比 RMSE 鲁棒)
def MAE(truth, pred, axis=0):
    """Mean Absolute Error(平均绝对误差):|pred - truth| 的均值。

    Per station 算后跨 station 聚合。返回 [mean, std, min, max]。
    """
    _mae = np.mean(np.abs(pred - truth), axis=axis)
    return [np.mean(_mae), np.std(_mae), np.min(_mae), np.max(_mae)]

# 6 指标 = {RMSE, MBE, MAE} × {normalized, Celsius}, 给 wandb log 用
def compute_6_metrics(truth, pred, scl=1.0):
    """**6 指标套件**:RMSE / MBE / MAE × normalized / Celsius。

    truth, pred 在归一化空间 (nSteps, nStations)。`scl` 是反归一化倍数(V1 因为
    target 已预归一化到 [0,1] 又没保留原始 (min, max),只能用粗略估计 ~30K)。
    返回 dict 含 6 个 float metric,直接 wandb.log()。
    """
    rmse = RMSE(truth, pred)[0]
    mbe  = MBE(truth, pred)[0]
    mae  = MAE(truth, pred)[0]
    return {
        'rmse_norm': float(rmse), 'mbe_norm': float(mbe), 'mae_norm': float(mae),
        'rmse_C':    float(rmse * scl),
        'mbe_C':     float(mbe  * scl),
        'mae_C':     float(mae  * scl),
    }

# 通用 per-feature min-max 归一化工具(faithful original_code,本轮主流程没用)
def MinMax(x):
    """Per-feature min-max 归一化(对最后一维分别求 min/max)。

    x.shape = (..., F),返回 (x_norm, off, scl),让 (x - off) / scl ∈ [0,1] per feature。
    Constant 列(min==max)用 (0, 1) 防止除 0。在 dataGen 里没用到此函数(V1 数据
    本身已预归一化),保留是为了和 original_code 一致。
    """
    # Min-Max normalization based on the last dimension of the raw data
    _axis = tuple(np.arange(0,x.ndim-1))
    _min, _max = np.min(x,axis=_axis), np.max(x,axis=_axis)
    _idx = np.where(_min==_max)
    _min[_idx], _max[_idx] = 0., 1.
    _off = _min
    _scl = (_max-_min)
    return (x-_off)/_scl, _off, _scl

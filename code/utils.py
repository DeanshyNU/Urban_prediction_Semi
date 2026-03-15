"""
工具函数模块
包含评估指标、归一化、绘图等工具函数
"""
import numpy as np
import matplotlib.pyplot as plt


def RMSE(truth, pred, axis=0):
    """
    计算RMSE指标
    Truth and prediction should have (nSteps,nStations)
    This function computes RMSE per station then return mean RMSE by station.
    """
    _rmse = np.linalg.norm(pred-truth, axis=axis)/np.sqrt(pred.shape[0])
    return [np.mean(_rmse), np.std(_rmse), np.min(_rmse), np.max(_rmse)]


def MinMax(x):
    """
    Min-Max normalization based on the last dimension of the raw data
    """
    _axis = tuple(np.arange(0, x.ndim-1))
    _min, _max = np.min(x, axis=_axis), np.max(x, axis=_axis)
    _idx = np.where(_min == _max)
    _min[_idx], _max[_idx] = 0., 1.
    _off = _min
    _scl = (_max - _min)
    return (x - _off) / _scl, _off, _scl


def MinMax_first_dim(x):
    """
    Min-Max normalization based on the first dimension
    """
    _min = np.min(x, axis=0)  # 按第一维计算最小值
    _max = np.max(x, axis=0)  # 按第一维计算最大值
    _idx = np.where(_min == _max)
    _min[_idx], _max[_idx] = 0., 1.  # 避免除以 0 的情况
    _off = _min
    _scl = (_max - _min)
    return (x - _off) / _scl, _off, _scl


def add_auxiliary_variables(data, nStations, nTimesteps):
    """
    生成辅助变量（时间、站点信息等）
    """
    # Generate auxiliary variables
    hours = np.tile(np.arange(24), int(nTimesteps / 24))
    months = np.concatenate([
        np.repeat(5, 31 * 24),  # May
        np.repeat(6, 30 * 24),  # June
        np.repeat(7, 31 * 24),  # July
        np.repeat(8, 31 * 24),  # August
        np.repeat(9, 30 * 24),  # September
        np.repeat(5, 31 * 24),  # May (next year)
        np.repeat(6, 30 * 24),  # June (next year)
        np.repeat(7, 31 * 24),  # July (next year)
        np.repeat(8, 31 * 24)   # August (next year)
    ])
    years = np.concatenate([
        np.repeat(2018, 3672),  # First year (May to September)
        np.repeat(2019, 2952)   # Second year (May to August)
    ])
    station_numbers = np.arange(1, nStations + 1)

    # Normalize auxiliary variables
    hours = hours / 23.0
    months = (months - 1) / 11.0

    # Repeat auxiliary variables for all stations
    hour_features = np.repeat(hours[:, np.newaxis], nStations, axis=1)
    month_features = np.repeat(months[:, np.newaxis], nStations, axis=1)
    year_features = np.repeat(years[:, np.newaxis], nStations, axis=1)
    station_features = np.tile(station_numbers, (nTimesteps, 1))

    # Combine all features
    auxiliary_variables = np.stack(
        [hour_features, month_features, year_features, station_features], axis=2
    )
    return auxiliary_variables


def plotPrediction(data, modelName):
    """
    绘制预测结果图
    """
    _truth, _pred, _rmse = data
    plt.figure(figsize=(10, 5))
    plt.plot(_pred, '-r', label='Prediction')
    plt.plot(_truth, '--k', label='Truth')
    plt.title(f'{modelName}, RMSE {_rmse:1.4f}')
    plt.xlim([0, 3000])
    plt.ylim([0, 1])
    plt.ylabel("Normalized Tobs", fontsize=20)
    plt.legend(loc='upper right')
    plt.savefig(f'./{modelName}_prediction.png', dpi=200)
    plt.close()


def plotHist(hist, modelName):
    """
    绘制训练历史图
    """
    _, _ax = plt.subplots(1, 2, figsize=(20, 10))
    _trainLoss, _validLoss, _trainRMSE, _validRMSE = np.array(hist).T
    _ax[0].semilogy(_trainLoss, label='Training')
    _ax[0].semilogy(_validLoss, label='Validation')
    _ax[0].set_title("Loss")
    _ax[0].legend(loc='upper right')
    _ax[1].semilogy(_trainRMSE, label="Training")
    _ax[1].semilogy(_validRMSE, label="Validation")
    _ax[1].set_title("Station-wised Mean RMSE")
    _ax[1].legend(loc='upper right')
    plt.savefig(f'{modelName}_hist.png', dpi=200)
    plt.close()


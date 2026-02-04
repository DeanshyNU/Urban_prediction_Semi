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

def MinMax(x):
    # Min-Max normalization based on the last dimension of the raw data
    _axis = tuple(np.arange(0,x.ndim-1))
    _min, _max = np.min(x,axis=_axis), np.max(x,axis=_axis)
    _idx = np.where(_min==_max)
    _min[_idx], _max[_idx] = 0., 1.
    _off = _min
    _scl = (_max-_min)
    return (x-_off)/_scl, _off, _scl

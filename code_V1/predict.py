import numpy as np
import torch, pickle, data, network, explainer, time
from scipy.io import savemat
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

runExplainer = 1
expAlgo = 'GNN-E'

RMSEList, truthList, predList = [],[],[]
geoScoreEmbedList, geoScoreEmbedRankList = [],[]
geoScoreRawList, geoScoreRawRankList = [],[]
modelName = f'geoEmbed_average_sage_full_fullGeo'
# Load metadata
metaPath = f'./{modelName}_param.pkl'
with open(metaPath,'rb') as f:
    modelPara = pickle.load(f)
    dataParam = pickle.load(f)
    metadata = pickle.load(f)
print(f'Metadata loaded.')
trainLoader, validLoader, _, validSet = data.dataGen(dataParam,'/home/hhz6461/Urban_prediction_Semi/data',predMode=True,geoFeatures=True)
nNodes = validSet[0].x.shape[0]
model = (network.GNN(modelPara)).to(device)
network.loadCheckPoint(modelName,model,-1,device,predMode=True)
_, trainRMSE, trainTruth, trainPred = network.test(trainLoader,model,torch.nn.MSELoss(),device,nNodes)
_, testRMSE, testTruth, testPred = network.test(validLoader,model,torch.nn.MSELoss(),device,nNodes)
fullPred = np.zeros((len(trainPred)+len(testPred),nNodes))
fullTruth = np.zeros_like(fullPred)
for _n,_idx in enumerate(metadata['trainIdx']): 
    fullPred[_idx] = trainPred[_n]
    fullTruth[_idx] = trainTruth[_n]
for _n,_idx in enumerate(metadata['validIdx']): 
    fullPred[_idx] = testPred[_n]
    fullTruth[_idx] = testTruth[_n]

if runExplainer:
    print('Run explainer.')
    for idxV in range(68):
        time1 = time.time()
        geoScoreEmbed,geoScoreEmbedRank,geoScoreRaw,geoScoreRawRank = explainer.runExplainer(validSet, model, metadata, idxV, device,algo=expAlgo)
        print(f'Explainer for Station {idxV+1} took {(time.time()-time1):1.2f}s.')
        print('Embedded geo feature sensitivity rank.')
        print(geoScoreEmbedRank)
        print('Raw geo feature sensitivity rank.')
        print(geoScoreRawRank)
        geoScoreEmbedList.append(geoScoreEmbed)
        geoScoreEmbedRankList.append(geoScoreEmbedRank)
        geoScoreRawList.append(geoScoreRaw)
        geoScoreRawRankList.append(geoScoreRawRank)

        savePrediction = {
            'RMSE':             np.vstack([trainRMSE,testRMSE]),
            'Truth':            fullTruth,
            'Prediction':       fullPred,
            'geoScoreEmbed':    np.array(geoScoreEmbedList),
            'geoScoreEmbedRank':np.array(geoScoreEmbedRankList),
            'geoScoreRaw':      np.array(geoScoreRawList),
            'geoScoreRawRank':  np.array(geoScoreRawRankList),
            'trainIdx':         np.sort(metadata['trainIdx']),
            'testIdx':          np.sort(metadata['validIdx'])}
else:
    savePrediction = {
        'RMSE':             np.vstack([trainRMSE,testRMSE]),
        'Truth':            fullTruth,
        'Prediction':       fullPred,
        'trainIdx':         np.sort(metadata['trainIdx']),
        'testIdx':          np.sort(metadata['validIdx'])}
print(trainRMSE, testRMSE) 
savemat(f'full_gnn_pred_sens_{expAlgo}.mat',savePrediction)
print("Prediction saved.")

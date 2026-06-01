import numpy as np
import torch,pickle,os,mat73,utils
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA

def genGeoFeatures(path,geoMethod='average',poolSize=15,nCompPCA=40):
    # --------------------------Geo features--------------------------
    _raw = mat73.loadmat(f'{path}/FeaturePatch_401.mat')['FeatureMat_zeros']
    # Remove 5th dimension because all stations are land based
    _idx = np.arange(_raw.shape[2])
    _idx = np.delete(_idx,4)
    _raw = _raw[:,:,_idx,:]
    _imageSize,_,_nFeatures,_nStations = _raw.shape
    
    if geoMethod == 'average':     
        # Min-max normalization
        _norm = np.transpose(_raw,(2,0,1,3)).reshape(_nFeatures,-1)
        _min,_max = np.min(_norm,axis=1), np.max(_norm,axis=1)
        _max[_max==0] = 1e-5
        _off = _min
        _scl = _max-_min
        _norm = np.transpose(_raw,(0,1,3,2))
        _norm = (_norm-_off)/_scl
        _geoFeatures = np.transpose(_norm,(2,3,0,1))
        # Average pooling 
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        _avgPool = torch.nn.AdaptiveAvgPool2d((poolSize,poolSize))
        _geoFeatures = _avgPool(_geoFeatures).reshape(_nStations,-1)

    if geoMethod == 'pca':       
        # Mean-std normalization  
        _norm = np.transpose(_raw,(2,0,1,3)).reshape(_nFeatures,-1)
        _off,_scl = np.mean(_norm,axis=1), np.std(_norm,axis=1)
        _norm = np.transpose(_raw,(0,1,3,2))
        _norm = (_norm-_off)/_scl
        _geoFeatures = np.transpose(_norm,(2,3,0,1))
        # PCA
        _geo2D = _geoFeatures.reshape(_nStations,-1)
        _pca = PCA(n_components=nCompPCA)
        _geoFeatures = _pca.fit_transform(_geo2D)
        _geoFeatures = (_geoFeatures-_geoFeatures.min())/(_geoFeatures.max()-_geoFeatures.min())
        _geoFeatures = torch.FloatTensor(_geoFeatures)
    
    return _geoFeatures,_off,_scl,_nStations

def dataGen(dataParam,path,nTrn=0.75,predMode=False):
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    _geoFeatures,_off,_scl,_nStations = genGeoFeatures(path,dataParam['geoMethod'],dataParam['poolSize'],dataParam['nCompPCA'])
    # --------------------------Graph construction--------------------------
    _raw = mat73.loadmat(f'{path}/GNN_N1_AJM.mat')
    _dist, _, _simiW = _raw['dist'], _raw['location'], _raw['similarity']
    _nNodes = _dist.shape[0]
    _distW = np.exp(-_dist)
    _off,_scl = np.min(_distW), np.max(_distW)-np.min(_distW)
    _distW = (_distW-_off)/_scl
    Adj = np.abs(_simiW*_distW)
    Adj[Adj<dataParam['thres']] = 0.
    assert np.allclose(Adj,Adj.T)
    # Node Features
    _raw = mat73.loadmat(f'{path}/GNN_N1_StationMat.mat')['StationMat_se_fill']
    _raw = np.transpose(_raw,(0,2,1)) # Dim: timestep * nNodes * nFeatures
    nCFDFeats = 54
    nStationFeats = 4
    cfdIdx = np.arange(nCFDFeats)
    stationFeatIdx = np.arange(nCFDFeats, nCFDFeats+nStationFeats)
    rawGeoFeatIdx = np.arange(nCFDFeats+nStationFeats,_raw.shape[-1]-1)
    features = _raw[:,:,1:]
    targets = _raw[:,:,0]
    # -----------------Construct PyG dataset ---------------------------
    # full dataset
    T = len(features)
    edgeIdxV,edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))
    _dataset = []
    for n in range(_window,T-_window):
        _feature = features[n]
        _tdb = features[n-_window:n,:,cfdIdx]
        _tdb = np.transpose(_tdb,(1,0,2)).reshape(len(Adj),-1)
        _tdf = features[n:n+_window,:,cfdIdx]
        _tdf = np.transpose(_tdf,(1,0,2)).reshape(len(Adj),-1)
        if dataParam['geoFeatures']=='full':
            _feature = np.hstack([_feature[:,cfdIdx],
                                _tdb,_tdf,
                                _feature[:,stationFeatIdx],
                                _feature[:,rawGeoFeatIdx],
                                _geoFeatures
                                ])
        elif dataParam['geoFeatures']=='raw':
            _feature = np.hstack([_feature[:,cfdIdx],
                                _tdb,_tdf,
                                _feature[:,stationFeatIdx],
                                _feature[:,rawGeoFeatIdx],
                                #_geoFeatures
                                ])
        elif dataParam['geoFeatures']=='embed':
            _feature = np.hstack([_feature[:,cfdIdx],
                                _tdb,_tdf,
                                _feature[:,stationFeatIdx],
                                # _feature[:,rawGeoFeatIdx],
                                _geoFeatures
                                ])
        elif dataParam['geoFeatures']=='no':
            _feature = np.hstack([_feature[:,cfdIdx],
                                _tdb,_tdf,
                                _feature[:,stationFeatIdx],
                                # _feature[:,rawGeoFeatIdx],
                                # _geoFeatures
                                ])
        else:
            raise RuntimeError('Geo feature option does not exist.')

        _target = targets[n].reshape(-1,1)
        _dataset.append(
            Data(
            x=torch.FloatTensor(_feature),
            y=torch.FloatTensor(_target),
            edge_index = edgeIdxV,
            edge_attr = edgeAttrV)
        )
    _cfdFeatLen = len(cfdIdx)*(2*_window+1)
    _stationFeatLen = len(stationFeatIdx)
    _rawGeoFeatLen = len(rawGeoFeatIdx)
    _geoFeatLen = len(_geoFeatures.T)
    _featureLen = np.cumsum([_cfdFeatLen,_stationFeatLen,_rawGeoFeatLen,_geoFeatLen])
    _featureIdx = {
        'CFD':      np.arange(0,_featureLen[0]),
        'station':  np.arange(_featureLen[0],_featureLen[1]),
        'rawGeo':       np.arange(_featureLen[1],_featureLen[2]),
        'embedGeo':     np.arange(_featureLen[2],_featureLen[3]),
    }

    # Randomly pick 70% in the time sequence
    _generator = torch.Generator().manual_seed(19)
    _trainLength = int(len(_dataset)*nTrn)
    _validLength = len(_dataset)-_trainLength  
    trainSet, validSet = torch.utils.data.random_split(_dataset,[_trainLength,_validLength],_generator)
    trainLoader = DataLoader(trainSet,batch_size=_batchSize,shuffle=not predMode) 
    validLoader = DataLoader(validSet,batch_size=len(validSet),shuffle=False)

    metadata = {
        'nNodes':    _nStations,
        'geoOff':    _off,
        'geoScl':    _scl,
        'iDim':      _feature.shape[-1],
        'oDim':      _target.shape[-1],
        'featureIdx':_featureIdx,
        'geoMethod': dataParam['geoMethod'],
        'poolSize':  dataParam['poolSize'],
        'nCompPCA':  dataParam['nCompPCA'],
        'trainIdx':  trainSet.indices,
        'validIdx':  validSet.indices,
        'AdjMatrix': Adj,
    }
    return trainLoader, validLoader, metadata, validSet

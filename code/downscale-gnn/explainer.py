import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer

geoFeatures = ['Tree height','Building height','Tree patch','Veg patch','Imper surf patch','NLCD','CMAP']
geoFeaturesRaw = ['treeMean','treeMax','treeStd','southT','buildingMean','buildingMax','buildingStd','southB','densH','densM','densL','densO','Fi','Fv','Fw','Ft','Fcover','LAI','NDVI','distMILake']

def runExplainer(validSet, model, metadata, idxV, device, algo='GNN-E'):
    if algo == 'GNN-E':
        algorithm = GNNExplainer(epochs=200)
        _sortByAbs = False
    elif algo == 'IG':
        algorithm = CaptumExplainer(attribution_method='IntegratedGradients')
        _sortByAbs = True
    else:
        raise RuntimeError()
    explainer =  Explainer(
    model=model,
    algorithm=algorithm,
    explanation_type='phenomenon',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node',
        return_type='raw',
        ),
    )
    # # Randomly pick 100 validset data
    # if len(validSet) > 100:
    #     _validIdx = np.random.randint(0,len(validSet)-1,100)
    #     validSet = [validSet[_idx] for _idx in _validIdx]
    node_index = idxV
    # poolSize = metadata['poolSize']
    rawGeoIdx = metadata['featureIdx']['rawGeo']
    embedGeoIdx = metadata['featureIdx']['embedGeo']
    geoScoreRaw = np.empty((len(validSet),len(rawGeoIdx)))
    geoScoreEmbed = np.empty((len(validSet),len(embedGeoIdx)))
    # geoScore = np.empty((len(validSet),len(rawGeoIdx)+len(embedGeoIdx)))
    for n, data in enumerate(validSet):
        data = data.to(device)
        explanation = explainer(data.x, 
                                data.edge_index,  
                                target=data.y,
                                index=node_index, 
                                edgeAttr=data.edge_attr)
        geoScoreRaw[n] = explanation.node_mask[idxV,metadata['featureIdx']['rawGeo']].cpu().detach().numpy()
        geoScoreEmbed[n] = explanation.node_mask[idxV,metadata['featureIdx']['embedGeo']].cpu().detach().numpy()
    # geoScore = np.hstack([geoScoreRaw,geoScoreEmbed])
    # geoScoreMean,geoScoreVar = np.mean(geoScore,axis=0),np.var(geoScore,axis=0)

    # Compute average geoScore spatially and temporally
    geoScoreEmbed = geoScoreEmbed.reshape(len(validSet),7,-1) # reshape to (nSteps, channel, poolSize**2)
    geoScoreEmbed = geoScoreEmbed.mean(axis=0) # Time averaged
    geoScoreRaw = geoScoreRaw.mean(axis=0) # Time averaged
    if _sortByAbs:
        geoScoreEmbedRank = np.argsort(np.abs(geoScoreEmbed.mean(axis=-1)))[::-1]
        geoScoreRawRank = np.argsort(np.abs(geoScoreRaw))[::-1]
    else:
        geoScoreEmbedRank = np.argsort(geoScoreEmbed.mean(axis=-1))[::-1]
        geoScoreRawRank = np.argsort(geoScoreRaw)[::-1]

    return geoScoreEmbed,geoScoreEmbedRank,geoScoreRaw,geoScoreRawRank
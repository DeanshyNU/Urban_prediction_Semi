"""
Semi-supervised GNN data loading - V2 data format
Labeled: Labeled_Finalized_new.mat (58 stations, 3672 timesteps)
Unlabeled: Unlabeled_Finalized.mat (2000 stations, sliced to 3672)
Unified graph: 58 labeled + 200 unlabeled = 258 nodes
"""
import numpy as np
import torch, pickle, os, mat73, utils
import scipy.io as sio
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from data import genGeoFeatures_v2, build_adj_matrix


def genGeoFeatures_unlabeled(UrbanFeatureMat, geoMethod='average', poolSize=15,
                              nCompPCA=40, norm_off=None, norm_scl=None):
    """Generate geo features for unlabeled data, using labeled normalization params."""
    _raw = UrbanFeatureMat
    _raw = np.nan_to_num(_raw, nan=0.0)
    _imageSize, _, _nFeatures, _nStations = _raw.shape

    use_provided = (norm_off is not None and norm_scl is not None)

    if geoMethod == 'average':
        if use_provided:
            _off, _scl = norm_off, norm_scl
        else:
            _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
            _min, _max = np.min(_norm, axis=1), np.max(_norm, axis=1)
            _max[_max == 0] = 1e-5
            _off = _min
            _scl = _max - _min
            _scl[_scl == 0] = 1e-5

        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _norm = np.clip(_norm, 0.0, 1.0)
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        _avgPool = torch.nn.AdaptiveAvgPool2d((poolSize, poolSize))
        _geoFeatures = _avgPool(_geoFeatures).reshape(_nStations, -1)

    elif geoMethod == 'pca':
        if use_provided:
            _off, _scl = norm_off, norm_scl
        else:
            _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
            _off, _scl = np.mean(_norm, axis=1), np.std(_norm, axis=1)
            _scl[_scl == 0] = 1e-5

        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        _geo2D = _geoFeatures.reshape(_nStations, -1)
        _pca = PCA(n_components=nCompPCA)
        _geoFeatures = _pca.fit_transform(_geo2D)
        _geo_range = _geoFeatures.max() - _geoFeatures.min()
        if _geo_range == 0:
            _geoFeatures = np.zeros_like(_geoFeatures)
        else:
            _geoFeatures = (_geoFeatures - _geoFeatures.min()) / _geo_range
        _geoFeatures = torch.FloatTensor(_geoFeatures)

    return _geoFeatures, _off, _scl, _nStations


def dataGen(dataParam, path, nTrn=0.75, predMode=False, n_unlabeled=200):
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']
    T_2018 = 3672  # Only use 2018 data

    # ======================== Load labeled data ========================
    print("Loading Labeled_Finalized_new.mat...")
    labeled = sio.loadmat(f'{path}/Labeled_Finalized_new.mat')

    # Target
    AoT = labeled['AoT_filled']  # (3672, 58)
    if AoT.shape[1] > AoT.shape[0]:
        AoT = AoT.T
    targets_labeled = AoT  # (T, nNodes_labeled)

    # WRF (63 dims, no merge)
    WRFMat_l = labeled['WRFMat']
    if WRFMat_l.shape[0] != T_2018:
        WRFMat_l = np.transpose(WRFMat_l, (2, 1, 0))
    cfd_labeled = np.transpose(WRFMat_l, (0, 2, 1))  # (T, nL, 63)

    # CLMS (3 dims)
    CLMSMat_l = labeled['CLMSMat']
    if CLMSMat_l.shape[0] != T_2018:
        CLMSMat_l = np.transpose(CLMSMat_l, (2, 1, 0))
    clms_labeled = np.transpose(CLMSMat_l, (0, 2, 1))  # (T, nL, 3)

    # UrbanFeature (17 dims)
    # Cols 0-15: height/fraction (NaN → 0), Col 16: distance to lake (NaN → median)
    UF_labeled = labeled.get('UrbanFeature', None)
    if UF_labeled is not None:
        n_nan = np.isnan(UF_labeled).sum()
        if n_nan > 0:
            UF_labeled[:, :16] = np.nan_to_num(UF_labeled[:, :16], nan=0.0)
            col16 = UF_labeled[:, 16]
            if np.any(np.isnan(col16)):
                UF_labeled[np.isnan(col16), 16] = np.nanmedian(col16)
            print(f"  [Debug] Labeled UrbanFeature NaN: {n_nan}, filled (0 for height/frac, median for distance)")
        uf_min_l = UF_labeled.min(axis=0)
        uf_range_l = UF_labeled.max(axis=0) - uf_min_l
        uf_range_l[uf_range_l == 0] = 1e-5
        UF_labeled_norm = (UF_labeled - uf_min_l) / uf_range_l

    # GeoEmbed from UrbanFeatureMat
    UFM_l = labeled.get('UrbanFeatureMat', None)
    if UFM_l is not None:
        _geoFeatures_labeled, _off, _scl, nL = genGeoFeatures_v2(
            UFM_l, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'])
        if torch.is_tensor(_geoFeatures_labeled):
            _geoFeatures_labeled = _geoFeatures_labeled.numpy()
    else:
        nL = targets_labeled.shape[1]
        _geoFeatures_labeled = np.zeros((nL, 1))
        _off, _scl = np.array([0]), np.array([1])

    nNodes_labeled = targets_labeled.shape[1]
    nCFDFeats = cfd_labeled.shape[2]  # 63
    print(f"  Labeled: {nNodes_labeled} stations, {T_2018} timesteps, WRF={nCFDFeats}d")

    # ======================== Load unlabeled data ========================
    print("Loading Unlabeled_Finalized.mat...")
    unlabeled = mat73.loadmat(f'{path}/Unlabeled_Finalized.mat')

    # Select unlabeled stations (skip first nNodes_labeled which are labeled)
    # V2: first 58 usable stations are labeled, rest are unlabeled
    usable_indices = labeled.get('usable_station_indices', None)
    if usable_indices is not None:
        # Unlabeled starts after all 106 V2 labeled stations
        start_idx = 106
    else:
        start_idx = nNodes_labeled
    end_idx = start_idx + n_unlabeled

    # WRF unlabeled (63 dims, no merge)
    WRFMat_u = unlabeled['WRFMat']
    if WRFMat_u.shape[0] != 6624:
        WRFMat_u = np.transpose(WRFMat_u, (2, 1, 0))
    WRFMat_u = WRFMat_u[:T_2018, :, start_idx:end_idx]  # slice to 2018
    cfd_unlabeled = np.transpose(WRFMat_u, (0, 2, 1))  # (T, nU, 63)

    # CLMS unlabeled
    CLMSMat_u = unlabeled['CLMSMat']
    if CLMSMat_u.shape[0] != 6624:
        CLMSMat_u = np.transpose(CLMSMat_u, (2, 1, 0))
    CLMSMat_u = CLMSMat_u[:T_2018, :, start_idx:end_idx]
    clms_unlabeled = np.transpose(CLMSMat_u, (0, 2, 1))  # (T, nU, 3)

    # UrbanFeature unlabeled
    # Cols 0-15: height/fraction (NaN → 0), Col 16: distance to lake (NaN → median)
    UF_unlabeled = unlabeled.get('UrbanFeature', None)
    if UF_unlabeled is not None:
        UF_unlabeled = UF_unlabeled[start_idx:end_idx, :]
        n_nan_u = np.isnan(UF_unlabeled).sum()
        if n_nan_u > 0:
            UF_unlabeled[:, :16] = np.nan_to_num(UF_unlabeled[:, :16], nan=0.0)
            col16_u = UF_unlabeled[:, 16]
            if np.any(np.isnan(col16_u)):
                UF_unlabeled[np.isnan(col16_u), 16] = np.nanmedian(col16_u)
            print(f"  [Debug] Unlabeled UrbanFeature NaN: {n_nan_u}, filled (0 for height/frac, median for distance)")
        # Normalize using labeled params
        if UF_labeled is not None:
            UF_unlabeled_norm = (UF_unlabeled - uf_min_l) / uf_range_l
            UF_unlabeled_norm = np.clip(UF_unlabeled_norm, 0.0, 1.0)
        else:
            UF_unlabeled_norm = UF_unlabeled
    else:
        UF_unlabeled_norm = None

    # GeoEmbed unlabeled (use labeled norm params)
    UFM_u = unlabeled.get('UrbanFeatureMat', None)
    if UFM_u is not None:
        UFM_u = np.nan_to_num(UFM_u[:, :, :, start_idx:end_idx], nan=0.0)
        _geoFeatures_unlabeled, _, _, _ = genGeoFeatures_unlabeled(
            UFM_u, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA'],
            norm_off=_off, norm_scl=_scl)
        if torch.is_tensor(_geoFeatures_unlabeled):
            _geoFeatures_unlabeled = _geoFeatures_unlabeled.numpy()
    else:
        _geoFeatures_unlabeled = np.zeros((n_unlabeled, _geoFeatures_labeled.shape[1]))

    print(f"  Unlabeled: {n_unlabeled} stations, WRF={cfd_unlabeled.shape[2]}d")
    print(f"  GeoEmbed: labeled={_geoFeatures_labeled.shape[1]}d, unlabeled={_geoFeatures_unlabeled.shape[1]}d")

    # ======================== Normalize WRF & CLMS ========================
    print("Normalizing features...")
    cfd_labeled_norm, cfd_off, cfd_scl = utils.MinMax(cfd_labeled.copy())
    clms_labeled_norm, clms_off, clms_scl = utils.MinMax(clms_labeled.copy())
    targets_norm, tgt_off, tgt_scl = utils.MinMax(targets_labeled.copy())

    # Normalize unlabeled independently
    cfd_unlabeled_norm, _, _ = utils.MinMax(cfd_unlabeled.copy())
    clms_unlabeled_norm, _, _ = utils.MinMax(clms_unlabeled.copy())

    # ======================== Build unified graph ========================
    print("Building unified graph (labeled + unlabeled)...")

    # Get Map and SimilarityMat for all nodes from V2 source
    if 'usable_station_indices' in labeled:
        usable_idx = labeled['usable_station_indices'].squeeze().astype(int)
        Map_labeled = unlabeled['Map'][:106][usable_idx]
        SM_labeled = unlabeled['SimilarityMat'][:106][usable_idx]
    else:
        Map_labeled = np.squeeze(labeled['Map'])
        SM_labeled = np.squeeze(labeled['SimilarityMat'])

    Map_unlabeled = unlabeled['Map'][start_idx:end_idx]
    SM_unlabeled = unlabeled['SimilarityMat'][start_idx:end_idx]

    nNodes_total = nNodes_labeled + n_unlabeled

    # --- Block-wise graph construction ---
    # L-L, U-U, L-U computed separately with independent distance normalization
    # This ensures L-L connectivity matches supervised learning exactly
    thres_ll = dataParam.get('thres_ll', 0.1)   # same as supervised
    thres_uu = dataParam.get('thres_uu', 0.4)
    thres_lu = dataParam.get('thres_lu', 0.4)

    # L-L block (same as supervised graph)
    Adj_ll = build_adj_matrix(Map_labeled, SM_labeled, thres_ll)

    # U-U block
    Adj_uu = build_adj_matrix(Map_unlabeled, SM_unlabeled, thres_uu)

    # L-U block (cross edges, independent normalization)
    Map_cross = np.vstack([Map_labeled, Map_unlabeled])
    SM_cross = np.vstack([SM_labeled, SM_unlabeled])
    Adj_cross_full = build_adj_matrix(Map_cross, SM_cross, thres_lu)
    Adj_lu = Adj_cross_full[:nNodes_labeled, nNodes_labeled:]  # extract L-U block

    # Assemble unified adjacency matrix
    Adj = np.zeros((nNodes_total, nNodes_total))
    Adj[:nNodes_labeled, :nNodes_labeled] = Adj_ll
    Adj[nNodes_labeled:, nNodes_labeled:] = Adj_uu
    Adj[:nNodes_labeled, nNodes_labeled:] = Adj_lu
    Adj[nNodes_labeled:, :nNodes_labeled] = Adj_lu.T
    np.fill_diagonal(Adj, 0)
    assert np.allclose(Adj, Adj.T)

    # # --- Old: unified computation (all nodes same normalization) ---
    # Map_all = np.vstack([Map_labeled, Map_unlabeled])
    # SM_all = np.vstack([SM_labeled, SM_unlabeled])
    # Adj = build_adj_matrix(Map_all, SM_all, dataParam['thres'])
    # assert np.allclose(Adj, Adj.T)

    # Edge statistics
    n_ll = int(np.sum(Adj[:nNodes_labeled, :nNodes_labeled] > 0) // 2)
    n_uu = int(np.sum(Adj[nNodes_labeled:, nNodes_labeled:] > 0) // 2)
    n_lu = int(np.sum(Adj[:nNodes_labeled, nNodes_labeled:] > 0))
    total_edges = int(np.sum(Adj > 0) // 2)
    density = total_edges / (nNodes_total * (nNodes_total - 1) / 2) * 100
    print(f"  Graph: {nNodes_total} nodes, {total_edges*2} edges, density={density:.1f}%")
    print(f"  L-L={n_ll} (thres={thres_ll}), U-U={n_uu} (thres={thres_uu}), L-U={n_lu} (thres={thres_lu})")

    # ======================== Build PyG dataset ========================
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))

    label_mask = np.zeros(nNodes_total, dtype=bool)
    label_mask[:nNodes_labeled] = True

    # Align UrbanFeature dimensions
    if UF_labeled_norm is not None and UF_unlabeled_norm is not None:
        # Both have UrbanFeature, ensure same dim
        uf_dim = UF_labeled_norm.shape[1]
        if UF_unlabeled_norm.shape[1] != uf_dim:
            UF_unlabeled_padded = np.zeros((n_unlabeled, uf_dim))
            UF_unlabeled_padded[:, :UF_unlabeled_norm.shape[1]] = UF_unlabeled_norm
            UF_unlabeled_norm = UF_unlabeled_padded
    elif UF_labeled_norm is not None:
        UF_unlabeled_norm = np.zeros((n_unlabeled, UF_labeled_norm.shape[1]))

    T = T_2018
    _dataset = []
    for n in range(_window, T - _window):
        # ===== Labeled node features =====
        _cfd_l = cfd_labeled_norm[n]          # (nL, 63)
        _clms_l = clms_labeled_norm[n]        # (nL, 3)
        _tdb_l = cfd_labeled_norm[n - _window:n]
        _tdb_l = np.transpose(_tdb_l, (1, 0, 2)).reshape(nNodes_labeled, -1)
        _tdf_l = cfd_labeled_norm[n:n + _window]
        _tdf_l = np.transpose(_tdf_l, (1, 0, 2)).reshape(nNodes_labeled, -1)

        feat_l = [_cfd_l, _tdb_l, _tdf_l, _clms_l]
        if UF_labeled_norm is not None:
            feat_l.append(UF_labeled_norm)
        if dataParam.get('geoFeatures', 'full') != 'no':
            feat_l.append(_geoFeatures_labeled)
        feature_labeled = np.hstack(feat_l)

        # ===== Unlabeled node features =====
        _cfd_u = cfd_unlabeled_norm[n]        # (nU, 63)
        _clms_u = clms_unlabeled_norm[n]      # (nU, 3)
        _tdb_u = cfd_unlabeled_norm[n - _window:n]
        _tdb_u = np.transpose(_tdb_u, (1, 0, 2)).reshape(n_unlabeled, -1)
        _tdf_u = cfd_unlabeled_norm[n:n + _window]
        _tdf_u = np.transpose(_tdf_u, (1, 0, 2)).reshape(n_unlabeled, -1)

        feat_u = [_cfd_u, _tdb_u, _tdf_u, _clms_u]
        if UF_unlabeled_norm is not None:
            feat_u.append(UF_unlabeled_norm)
        if dataParam.get('geoFeatures', 'full') != 'no':
            feat_u.append(_geoFeatures_unlabeled)
        feature_unlabeled = np.hstack(feat_u)

        # Combine all nodes
        features_all = np.vstack([feature_labeled, feature_unlabeled])

        # Targets (only labeled have real targets)
        targets_all = np.concatenate([
            targets_norm[n],
            np.zeros(n_unlabeled)
        ]).reshape(-1, 1)

        _dataset.append(Data(
            x=torch.FloatTensor(features_all),
            y=torch.FloatTensor(targets_all),
            label_mask=torch.BoolTensor(label_mask),
            edge_index=edgeIdxV,
            edge_attr=edgeAttrV
        ))

    # Feature index tracking
    _cfdFeatLen = nCFDFeats * (2 * _window + 1)
    _clmsFeatLen = clms_labeled.shape[2]
    _urbanFeatLen = UF_labeled_norm.shape[1] if UF_labeled_norm is not None else 0
    _geoFeatLen = _geoFeatures_labeled.shape[1] if dataParam.get('geoFeatures', 'full') != 'no' else 0
    _featureLen = np.cumsum([_cfdFeatLen, _clmsFeatLen, _urbanFeatLen, _geoFeatLen])
    _featureIdx = {
        'CFD': np.arange(0, _featureLen[0]),
        'CLMS': np.arange(_featureLen[0], _featureLen[1]),
        'UrbanFeature': np.arange(_featureLen[1], _featureLen[2]),
        'embedGeo': np.arange(_featureLen[2], _featureLen[3]),
    }

    print(f"  Total feature dim: {features_all.shape[-1]} "
          f"(WRF={_cfdFeatLen} + CLMS={_clmsFeatLen} + UF={_urbanFeatLen} + Geo={_geoFeatLen})")

    # Train/valid split
    _generator = torch.Generator().manual_seed(19)
    _trainLength = int(len(_dataset) * nTrn)
    _validLength = len(_dataset) - _trainLength
    trainSet, validSet = torch.utils.data.random_split(
        _dataset, [_trainLength, _validLength], _generator)
    _shuffle_train = dataParam.get('shuffle_train', not predMode)
    trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=_shuffle_train)
    valid_batch_size = min(_batchSize, len(validSet))
    validLoader = DataLoader(validSet, batch_size=valid_batch_size, shuffle=False)

    metadata = {
        'nNodes': nNodes_total,
        'nNodes_labeled': nNodes_labeled,
        'nNodes_unlabeled': n_unlabeled,
        'geoOff': _off,
        'geoScl': _scl,
        'iDim': features_all.shape[-1],
        'oDim': 1,
        'featureIdx': _featureIdx,
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': trainSet.indices,
        'validIdx': validSet.indices,
        'AdjMatrix': Adj,
        'label_mask': label_mask,
        'tgt_off': tgt_off,
        'tgt_scl': tgt_scl,
    }
    return trainLoader, validLoader, metadata, validSet

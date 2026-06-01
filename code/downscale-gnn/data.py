"""
Supervised GNN data loading - V2 data format
Loads from Labeled_Finalized_new.mat (58 stations, 3672 timesteps)
Eval modes: spatial (default, leave-8-stations-out) or temporal (75/25 time split)
"""
import numpy as np
import torch, pickle, os, mat73, utils
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def select_validation_stations_fps(node_locations, n_valid=8):
    """Select n_valid most spatially spread stations using FPS. Deterministic."""
    n_total = node_locations.shape[0]
    if n_valid >= n_total:
        return np.arange(n_total), np.array([])
    centroid = node_locations.mean(axis=0)
    dists_to_centroid = np.sqrt(np.sum((node_locations - centroid) ** 2, axis=1))
    first = np.argmin(dists_to_centroid)
    selected = [first]
    min_dists = np.sqrt(np.sum((node_locations - node_locations[first:first+1]) ** 2, axis=1))
    for _ in range(n_valid - 1):
        for i in selected:
            min_dists[i] = -1
        best = np.argmax(min_dists)
        selected.append(best)
        new_dists = np.sqrt(np.sum((node_locations - node_locations[best:best+1]) ** 2, axis=1))
        min_dists = np.minimum(min_dists, new_dists)
    valid_indices = np.array(selected)
    train_indices = np.array([i for i in range(n_total) if i not in selected])
    return valid_indices, train_indices


def visualize_spatial_split(node_locations, train_indices, valid_indices,
                            unlabeled_locations=None, save_path=None):
    """Visualize train/valid/unlabeled station split."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if unlabeled_locations is not None:
        ax.scatter(unlabeled_locations[:, 1], unlabeled_locations[:, 0],
                   c='lightgray', s=8, alpha=0.4, label=f'Unlabeled ({len(unlabeled_locations)})')
    ax.scatter(node_locations[train_indices, 1], node_locations[train_indices, 0],
               c='red', s=60, marker='^', zorder=5, label=f'Train labeled ({len(train_indices)})')
    ax.scatter(node_locations[valid_indices, 1], node_locations[valid_indices, 0],
               c='blue', s=100, marker='*', zorder=6, label=f'Valid labeled ({len(valid_indices)})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Spatial Split: {len(train_indices)} train + {len(valid_indices)} validation stations')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Spatial Split] Visualization saved: {save_path}")
    plt.close()
from sklearn.decomposition import PCA


def genGeoFeatures_v2(UrbanFeatureMat, geoMethod='average', poolSize=15, nCompPCA=40):
    """Generate geo embedding features from UrbanFeatureMat (401x401xFxN)."""
    _raw = UrbanFeatureMat
    # Handle NaN
    _raw = np.nan_to_num(_raw, nan=0.0)
    _imageSize, _, _nFeatures, _nStations = _raw.shape

    if geoMethod == 'average':
        # Min-max normalization
        _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        _min, _max = np.min(_norm, axis=1), np.max(_norm, axis=1)
        _max[_max == 0] = 1e-5
        _off = _min
        _scl = _max - _min
        _scl[_scl == 0] = 1e-5
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        # POOL_TYPE: avg (default, current behavior) or directional (new)
        pool_type = os.environ.get('POOL_TYPE', 'avg').lower()
        if pool_type == 'directional':
            from utils import directional_pool
            n_dirs = int(os.environ.get('POOL_DIRS', '4'))
            _geoFeatures = directional_pool(_geoFeatures, n_dirs=n_dirs)
            print(f"  [POOL_TYPE=directional] n_dirs={n_dirs}, output dim={_geoFeatures.shape[1]} ({_nFeatures} ch × {n_dirs} dirs)")
        else:
            _avgPool = torch.nn.AdaptiveAvgPool2d((poolSize, poolSize))
            _geoFeatures = _avgPool(_geoFeatures).reshape(_nStations, -1)

    elif geoMethod == 'pca':
        # Mean-std normalization
        _norm = np.transpose(_raw, (2, 0, 1, 3)).reshape(_nFeatures, -1)
        _off, _scl = np.mean(_norm, axis=1), np.std(_norm, axis=1)
        _scl[_scl == 0] = 1e-5
        _norm = np.transpose(_raw, (0, 1, 3, 2))
        _norm = (_norm - _off) / _scl
        _geoFeatures = np.transpose(_norm, (2, 3, 0, 1))
        _geo2D = _geoFeatures.reshape(_nStations, -1)
        _pca = PCA(n_components=nCompPCA)
        _geoFeatures = _pca.fit_transform(_geo2D)
        _geo_min, _geo_max = _geoFeatures.min(), _geoFeatures.max()
        _geo_range = _geo_max - _geo_min
        if _geo_range == 0:
            _geoFeatures = np.zeros_like(_geoFeatures)
        else:
            _geoFeatures = (_geoFeatures - _geo_min) / _geo_range
        _geoFeatures = torch.FloatTensor(_geoFeatures)
        _off = _off
        _scl = _scl

    return _geoFeatures, _off, _scl, _nStations


def build_adj_matrix(Map, SimilarityMat, thres):
    """Build adjacency matrix from Map and SimilarityMat following Yu et al. (2024)."""
    nNodes = Map.shape[0]

    # Distance matrix from Map coordinates
    Distance = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            Distance[i, j] = np.sqrt(
                (Map[i, 0] - Map[j, 0])**2 + (Map[i, 1] - Map[j, 1])**2
            )

    # Similarity matrix: corrcoef of 10-dim SimilarityMat vectors
    Similarity = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            cov = np.corrcoef(SimilarityMat[i, :], SimilarityMat[j, :])
            r = cov[0, 1] if not np.isnan(cov[0, 1]) else 0.0
            r = max(r, 0.0)  # clip negative correlations
            Similarity[i, j] = r

    # Adjacency: Similarity * exp(-normalized_distance)
    max_dist = Distance.max()
    Adj = Similarity * np.exp(-Distance / max_dist)
    Adj[Adj < thres] = 0.0
    np.fill_diagonal(Adj, 0.0)

    print(f"  Graph: {nNodes} nodes, {int(np.sum(Adj > 0))} edges, "
          f"density={np.sum(Adj > 0) / (nNodes * (nNodes - 1)) * 100:.1f}%")

    return Adj


def dataGen(dataParam, path, nTrn=0.75, predMode=False):
    _window = dataParam['window']
    _batchSize = dataParam['batchSize']

    # ======================== Load V2 labeled data ========================
    print("Loading Labeled_Finalized_new.mat...")
    labeled = sio.loadmat(f'{path}/Labeled_Finalized_new.mat')

    # Target: GPR-filled temperature
    AoT = labeled['AoT_filled']  # (3672, 58)
    if AoT.shape[1] > AoT.shape[0]:
        AoT = AoT.T
    targets = AoT  # (T, nNodes)

    # WRF features (63 dims, no wind merge)
    WRFMat = labeled['WRFMat']  # (3672, 63, 58)
    if WRFMat.shape[0] != AoT.shape[0]:
        WRFMat = np.transpose(WRFMat, (2, 1, 0))
    cfd_features = np.transpose(WRFMat, (0, 2, 1))  # (T, nNodes, 63)

    # CLMS features (3 dims)
    CLMSMat = labeled['CLMSMat']  # (3672, 3, 58)
    if CLMSMat.shape[0] != AoT.shape[0]:
        CLMSMat = np.transpose(CLMSMat, (2, 1, 0))
    clms_features = np.transpose(CLMSMat, (0, 2, 1))  # (T, nNodes, 3)

    # UrbanFeature: station-level static features (17 dims)
    # Cols 0-15: height/fraction features (NaN = no structure/coverage → fill 0)
    # Col 16: distance to lake Michigan (NaN → fill column median)
    UrbanFeature = labeled.get('UrbanFeature', None)  # (58, 17)
    if UrbanFeature is not None:
        n_nan = np.isnan(UrbanFeature).sum()
        if n_nan > 0:
            # Height and fraction cols (0-15): fill with 0
            UrbanFeature[:, :16] = np.nan_to_num(UrbanFeature[:, :16], nan=0.0)
            # Distance to lake col (16): fill with median
            col16 = UrbanFeature[:, 16]
            if np.any(np.isnan(col16)):
                UrbanFeature[np.isnan(col16), 16] = np.nanmedian(col16)
            print(f"  [Debug] UrbanFeature NaN: {n_nan}, filled (0 for height/frac, median for distance)")
        # Min-max normalize
        uf_min = UrbanFeature.min(axis=0)
        uf_range = UrbanFeature.max(axis=0) - uf_min
        uf_range[uf_range == 0] = 1e-5
        UrbanFeature_norm = (UrbanFeature - uf_min) / uf_range
    else:
        UrbanFeature_norm = None

    # GeoEmbed from UrbanFeatureMat
    UFM = labeled.get('UrbanFeatureMat', None)
    if UFM is not None:
        print(f"  [Debug] UrbanFeatureMat shape: {UFM.shape}, dtype: {UFM.dtype}")
        print(f"  [Debug] UrbanFeatureMat NaN: {np.isnan(UFM).sum()}, range: [{np.nanmin(UFM):.4f}, {np.nanmax(UFM):.4f}]")
        _geoFeatures, _off, _scl, _nStations = genGeoFeatures_v2(
            UFM, dataParam['geoMethod'], dataParam['poolSize'], dataParam['nCompPCA']
        )
        if torch.is_tensor(_geoFeatures):
            _geoFeatures = _geoFeatures.numpy()
    else:
        _geoFeatures = np.zeros((targets.shape[1], 1))
        _off, _scl = np.array([0]), np.array([1])
        _nStations = targets.shape[1]

    nNodes = targets.shape[1]
    T = targets.shape[0]
    nCFDFeats = cfd_features.shape[2]  # 63

    print(f"  Stations: {nNodes}, Timesteps: {T}")
    print(f"  WRF: {nCFDFeats} dims, CLMS: {clms_features.shape[2]} dims")
    if UrbanFeature_norm is not None:
        print(f"  UrbanFeature: {UrbanFeature_norm.shape[1]} dims")
    print(f"  GeoEmbed: {_geoFeatures.shape[1]} dims")

    # ======================== Normalize WRF & CLMS ========================
    # Reshape for normalization: (T*nNodes, nFeats)
    cfd_flat = cfd_features.reshape(-1, nCFDFeats)
    cfd_features_norm, cfd_off, cfd_scl = utils.MinMax(cfd_features.copy())
    clms_features_norm, clms_off, clms_scl = utils.MinMax(clms_features.copy())
    # Target normalization
    norm_mode = os.environ.get('NORM_MODE', 'per_station').lower()
    _tgt_global_min = targets.min()
    _tgt_global_max = targets.max()
    _tgt_global_scl = _tgt_global_max - _tgt_global_min
    if _tgt_global_scl == 0:
        _tgt_global_scl = 1.0
    if norm_mode == 'global':
        targets_norm = (targets - _tgt_global_min) / _tgt_global_scl
        tgt_off = np.full(targets.shape[1], _tgt_global_min)
        tgt_scl = np.full(targets.shape[1], _tgt_global_scl)
        print(f"  Target normalization: GLOBAL (min={_tgt_global_min:.2f}, max={_tgt_global_max:.2f})")
    else:
        targets_norm, tgt_off, tgt_scl = utils.MinMax(targets.copy())
        print(f"  Target normalization: PER-STATION")

    print(f"  WRF normalized: [{cfd_features_norm.min():.3f}, {cfd_features_norm.max():.3f}]")
    print(f"  CLMS normalized: [{clms_features_norm.min():.3f}, {clms_features_norm.max():.3f}]")
    print(f"  Target normalized: [{targets_norm.min():.3f}, {targets_norm.max():.3f}]")

    # ======================== Build graph ========================
    Map = np.squeeze(labeled['Map'])  # (58, 2)
    SM = np.squeeze(labeled['SimilarityMat'])  # (58, 10)
    Adj = build_adj_matrix(Map, SM, dataParam['thres'])
    assert np.allclose(Adj, Adj.T)

    # ======================== Build PyG dataset ========================
    edgeIdxV, edgeAttrV = pyg_utils.dense_to_sparse(torch.FloatTensor(Adj))

    _dataset = []
    for n in range(_window, T - _window):
        _cfd = cfd_features_norm[n]          # (nNodes, 63)
        _clms = clms_features_norm[n]        # (nNodes, 3)

        # History WRF window
        _tdb = cfd_features_norm[n - _window:n]
        _tdb = np.transpose(_tdb, (1, 0, 2)).reshape(nNodes, -1)
        # Future WRF window
        _tdf = cfd_features_norm[n:n + _window]
        _tdf = np.transpose(_tdf, (1, 0, 2)).reshape(nNodes, -1)

        # Combine features
        feat_list = [_cfd, _tdb, _tdf, _clms]
        if UrbanFeature_norm is not None:
            feat_list.append(UrbanFeature_norm)
        if dataParam.get('geoFeatures', 'full') != 'no':
            feat_list.append(_geoFeatures)

        _feature = np.hstack(feat_list)
        _target = targets_norm[n].reshape(-1, 1)

        _dataset.append(Data(
            x=torch.FloatTensor(_feature),
            y=torch.FloatTensor(_target),
            edge_index=edgeIdxV,
            edge_attr=edgeAttrV
        ))

    # Feature index tracking
    _cfdFeatLen = nCFDFeats * (2 * _window + 1)
    _clmsFeatLen = clms_features.shape[2]
    _urbanFeatLen = UrbanFeature_norm.shape[1] if UrbanFeature_norm is not None else 0
    _geoFeatLen = _geoFeatures.shape[1] if dataParam.get('geoFeatures', 'full') != 'no' else 0
    _featureLen = np.cumsum([_cfdFeatLen, _clmsFeatLen, _urbanFeatLen, _geoFeatLen])
    _featureIdx = {
        'CFD': np.arange(0, _featureLen[0]),
        'CLMS': np.arange(_featureLen[0], _featureLen[1]),
        'UrbanFeature': np.arange(_featureLen[1], _featureLen[2]),
        'embedGeo': np.arange(_featureLen[2], _featureLen[3]),
    }

    print(f"  Total feature dim: {_feature.shape[-1]} "
          f"(WRF={_cfdFeatLen} + CLMS={_clmsFeatLen} + UF={_urbanFeatLen} + Geo={_geoFeatLen})")

    # Debug: check for NaN/Inf in features and targets
    print(f"\n  [Debug] NaN check:")
    print(f"    WRF norm: NaN={np.isnan(cfd_features_norm).sum()}, Inf={np.isinf(cfd_features_norm).sum()}")
    print(f"    CLMS norm: NaN={np.isnan(clms_features_norm).sum()}, Inf={np.isinf(clms_features_norm).sum()}")
    print(f"    Target norm: NaN={np.isnan(targets_norm).sum()}, Inf={np.isinf(targets_norm).sum()}")
    if UrbanFeature_norm is not None:
        print(f"    UrbanFeature: NaN={np.isnan(UrbanFeature_norm).sum()}, Inf={np.isinf(UrbanFeature_norm).sum()}")
    print(f"    GeoEmbed: NaN={np.isnan(_geoFeatures).sum() if not torch.is_tensor(_geoFeatures) else torch.isnan(_geoFeatures).sum()}, "
          f"Inf={np.isinf(_geoFeatures).sum() if not torch.is_tensor(_geoFeatures) else torch.isinf(_geoFeatures).sum()}")
    print(f"    Adj: NaN={np.isnan(Adj).sum()}, range=[{Adj.min():.4f}, {Adj.max():.4f}]")
    print(f"    Sample feature[0]: NaN={np.isnan(_feature[0]).sum()}, range=[{_feature[0].min():.4f}, {_feature[0].max():.4f}]")

    # ======================== Train/Valid Split ========================
    eval_mode = os.environ.get('EVAL_MODE', 'spatial').lower()

    if eval_mode == 'spatial':
        # --- Spatial split: leave 8 stations out for validation ---
        labeled_locs = labeled['NodeLocation']
        valid_station_idx, train_station_idx = select_validation_stations_fps(labeled_locs, n_valid=8)

        print(f"  [Spatial Split] Train stations: {len(train_station_idx)}, Valid stations: {len(valid_station_idx)}")
        print(f"  [Spatial Split] Valid station indices: {valid_station_idx.tolist()}")

        # Visualize
        try:
            _output_dir = os.environ.get('OUTPUT_DIR', '')
            if _output_dir:
                visualize_spatial_split(labeled_locs, train_station_idx, valid_station_idx,
                                       save_path=os.path.join(_output_dir, 'spatial_split.png'))
        except Exception as e:
            print(f"  [Spatial Split] Visualization skipped: {e}")

        # For supervised: train dataset uses only train station targets,
        # valid dataset uses only valid station targets
        # All nodes stay in graph for message passing
        train_mask = np.zeros(nNodes, dtype=bool)
        train_mask[train_station_idx] = True
        valid_mask = np.zeros(nNodes, dtype=bool)
        valid_mask[valid_station_idx] = True

        trainSet = []
        validSet = []
        for data in _dataset:
            trainSet.append(Data(
                x=data.x, y=data.y,
                label_mask=torch.BoolTensor(train_mask),
                edge_index=data.edge_index, edge_attr=data.edge_attr))
            validSet.append(Data(
                x=data.x, y=data.y,
                label_mask=torch.BoolTensor(valid_mask),
                edge_index=data.edge_index, edge_attr=data.edge_attr))

        trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
        valid_batch_size = min(_batchSize, len(validSet))
        validLoader = DataLoader(validSet, batch_size=valid_batch_size, shuffle=False)

        train_indices = list(range(len(trainSet)))
        valid_indices = list(range(len(validSet)))
    else:
        # --- Temporal split (original) ---
        print(f"  [Temporal Split] Eval mode: TEMPORAL (75/25 time split)")
        _generator = torch.Generator().manual_seed(19)
        _trainLength = int(len(_dataset) * nTrn)
        _validLength = len(_dataset) - _trainLength
        trainSet, validSet = torch.utils.data.random_split(
            _dataset, [_trainLength, _validLength], _generator)
        trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=not predMode)
        validLoader = DataLoader(validSet, batch_size=len(validSet), shuffle=False)

        train_indices = trainSet.indices
        valid_indices = validSet.indices
        valid_station_idx = None
        train_station_idx = None

    metadata = {
        'nNodes': nNodes,
        'geoOff': _off,
        'geoScl': _scl,
        'iDim': _feature.shape[-1],
        'oDim': 1,
        'featureIdx': _featureIdx,
        'geoMethod': dataParam['geoMethod'],
        'poolSize': dataParam['poolSize'],
        'nCompPCA': dataParam['nCompPCA'],
        'trainIdx': train_indices,
        'validIdx': valid_indices,
        'AdjMatrix': Adj,
        'tgt_off': tgt_off,
        'tgt_scl': tgt_scl,
        'tgt_global_scl': float(_tgt_global_scl),    # for denormalization to Celsius
        'tgt_global_min': float(_tgt_global_min),
        'eval_mode': eval_mode,
        'valid_station_idx': valid_station_idx,
        'train_station_idx': train_station_idx,
    }
    return trainLoader, validLoader, metadata, validSet

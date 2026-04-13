"""
Semi-supervised GNN data loading - V2 data format
Labeled: Labeled_Finalized_new.mat (58 stations, 3672 timesteps)
Unlabeled: Unlabeled_Finalized.mat (2000 stations, sliced to 3672)
Unified graph: 58 labeled + N unlabeled nodes
Station selection: FPS (Farthest Point Sampling) based on graph adjacency weights
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
from sklearn.decomposition import PCA
from data import genGeoFeatures_v2, build_adj_matrix


def select_validation_stations_fps(node_locations, n_valid=8):
    """
    Select n_valid most spatially spread stations from labeled stations using FPS.
    Deterministic (no randomness).

    Args:
        node_locations: (n_labeled, 2) lat/lon coordinates
        n_valid: number of validation stations to select
    Returns:
        valid_indices: (n_valid,) indices of validation stations
        train_indices: (n_train,) indices of training stations
    """
    n_total = node_locations.shape[0]
    if n_valid >= n_total:
        return np.arange(n_total), np.array([])

    # Start from the station closest to the centroid (deterministic)
    centroid = node_locations.mean(axis=0)
    dists_to_centroid = np.sqrt(np.sum((node_locations - centroid) ** 2, axis=1))
    first = np.argmin(dists_to_centroid)

    selected = [first]
    min_dists = np.sqrt(np.sum((node_locations - node_locations[first:first+1]) ** 2, axis=1))

    for _ in range(n_valid - 1):
        # Select station farthest from all selected
        for i in selected:
            min_dists[i] = -1  # exclude already selected
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


def select_unlabeled_fps(Map_labeled, SM_labeled, Map_all_unlabeled, SM_all_unlabeled, n_select,
                         UF_labeled=None, UF_all_unlabeled=None,
                         Loc_labeled=None, Loc_all_unlabeled=None,
                         use_score=False):
    """
    Three-step unlabeled station selection:
      Step 1: Filter - remove unreachable (adj=0) and feature-outlier (top 5%) candidates
      Step 2: Pure spatial FPS - select for spatial diversity only (no score weighting)
      Step 3: (Done in graph construction) Adaptive threshold to control density

    Returns:
        selected_indices: (n_select,) indices into the candidate array
    """
    n_candidates = Map_all_unlabeled.shape[0]
    n_labeled = Map_labeled.shape[0]
    if n_select >= n_candidates:
        return np.arange(n_candidates)

    print(f"  [Station Selection] Candidates: {n_candidates}, Target: {n_select}")

    # ========== Step 1: Filter candidates ==========
    valid_mask = np.ones(n_candidates, dtype=bool)

    # 1a. Remove candidates with no graph connection to any labeled station
    max_adj_to_labeled = np.zeros(n_candidates)
    # Compute normalized distances (all candidates to all labeled)
    all_map_dists = np.zeros((n_candidates, n_labeled))
    for j in range(n_labeled):
        all_map_dists[:, j] = np.sqrt(np.sum((Map_all_unlabeled - Map_labeled[j:j+1])**2, axis=1))
    global_max_dist = all_map_dists.max()
    if global_max_dist == 0:
        global_max_dist = 1.0
    all_map_dists_norm = all_map_dists / global_max_dist

    for i in range(n_candidates):
        best_adj = 0.0
        for j in range(n_labeled):
            cov = np.corrcoef(SM_all_unlabeled[i], SM_labeled[j])
            r = cov[0, 1] if not np.isnan(cov[0, 1]) else 0.0
            r = max(r, 0.0)
            adj_w = r * np.exp(-all_map_dists_norm[i, j])
            if adj_w > best_adj:
                best_adj = adj_w
        max_adj_to_labeled[i] = best_adj

    n_unreachable = int(np.sum(max_adj_to_labeled == 0))
    valid_mask[max_adj_to_labeled == 0] = False
    print(f"  [Step1] Unreachable (adj=0 to all labeled): {n_unreachable}/{n_candidates}")

    # 1b. Remove UrbanFeature outliers (top 5% by distance to nearest labeled)
    if UF_labeled is not None and UF_all_unlabeled is not None:
        uf_mean = np.nanmean(UF_labeled, axis=0)
        uf_std = np.nanstd(UF_labeled, axis=0)
        uf_std[uf_std == 0] = 1.0
        UF_l_std = (np.nan_to_num(UF_labeled, nan=0.0) - uf_mean) / uf_std
        UF_u_std = (np.nan_to_num(UF_all_unlabeled, nan=0.0) - uf_mean) / uf_std

        uf_dists = np.zeros(n_candidates)
        for i in range(n_candidates):
            d = np.sqrt(np.sum((UF_u_std[i:i+1] - UF_l_std)**2, axis=1))
            uf_dists[i] = d.min()

        threshold_95 = np.percentile(uf_dists[valid_mask], 95)
        n_outlier = int(np.sum((uf_dists > threshold_95) & valid_mask))
        valid_mask[uf_dists > threshold_95] = False
        print(f"  [Step1] UrbanFeature outliers (top 5%, dist>{threshold_95:.2f}): {n_outlier}")
        print(f"  [Step1] UF dist stats (valid): min={uf_dists[valid_mask].min():.2f}, "
              f"max={uf_dists[valid_mask].max():.2f}, mean={uf_dists[valid_mask].mean():.2f}")
    else:
        print(f"  [Step1] UrbanFeature not provided, skipping outlier filter")

    n_valid = int(np.sum(valid_mask))
    print(f"  [Step1] Valid candidates after filtering: {n_valid}/{n_candidates}")

    if n_select > n_valid:
        print(f"  [Step1] WARNING: requested {n_select} > valid {n_valid}, using all valid")
        return np.where(valid_mask)[0]

    # ========== Step 2: FPS (spatial or score-weighted) ==========
    # Use NodeLocation (lat/lon) if available for true spatial distance
    if Loc_labeled is not None and Loc_all_unlabeled is not None:
        pos_labeled = Loc_labeled
        pos_candidates = Loc_all_unlabeled
        print(f"  [Step2] Using NodeLocation (lat/lon) for spatial distance")
    else:
        pos_labeled = Map_labeled
        pos_candidates = Map_all_unlabeled
        print(f"  [Step2] Using Map coords for spatial distance")

    if use_score:
        # Normalize scores for valid candidates to [0, 1]
        valid_scores = max_adj_to_labeled.copy()
        valid_scores[~valid_mask] = 0
        if valid_scores.max() > valid_scores.min():
            scores_norm = (valid_scores - valid_scores[valid_mask].min()) / (valid_scores[valid_mask].max() - valid_scores[valid_mask].min())
        else:
            scores_norm = np.ones(n_candidates)
        scores_norm[~valid_mask] = 0
        print(f"  [Step2] Mode: SCORE-WEIGHTED FPS (score × diversity)")
        print(f"  [Step2] Score stats (valid): min={valid_scores[valid_mask].min():.4f}, "
              f"max={valid_scores[valid_mask].max():.4f}, mean={valid_scores[valid_mask].mean():.4f}")
    else:
        print(f"  [Step2] Mode: PURE SPATIAL FPS (diversity only)")

    valid_indices = np.where(valid_mask)[0]

    # Initialize min_dists: distance from each valid candidate to nearest labeled
    min_dists = np.full(n_candidates, -np.inf)
    for i in valid_indices:
        d = np.sqrt(np.sum((pos_candidates[i:i+1] - pos_labeled)**2, axis=1))
        min_dists[i] = d.min()

    selected = []
    selection_dists = []

    for step in range(n_select):
        # Normalize current distances for valid unselected candidates
        unselected_valid = [i for i in valid_indices if i not in set(selected)]
        if len(unselected_valid) == 0:
            print(f"  [Step2] WARNING: ran out of candidates at step {step}")
            break

        dists_arr = np.array([min_dists[i] for i in unselected_valid])
        if dists_arr.max() > 0:
            dists_norm = dists_arr / dists_arr.max()
        else:
            dists_norm = np.ones(len(unselected_valid))

        if use_score:
            # Combined: score × diversity
            scores_arr = np.array([scores_norm[i] for i in unselected_valid])
            combined = scores_arr * dists_norm
        else:
            # Pure spatial: only diversity
            combined = dists_norm

        best_local = np.argmax(combined)
        best = unselected_valid[best_local]
        best_dist = min_dists[best]

        selected.append(best)
        selection_dists.append(best_dist)

        # Update min distances: include newly selected point
        new_dists = np.sqrt(np.sum((pos_candidates - pos_candidates[best:best+1])**2, axis=1))
        for i in valid_indices:
            if i not in set(selected):
                min_dists[i] = min(min_dists[i], new_dists[i])

        # Progress print
        if (step + 1) % 200 == 0 or step == 0:
            print(f"  [Step2] Selected {step+1}/{n_select}, last pick dist={best_dist:.6f}")

    selected = np.array(selected)
    selection_dists = np.array(selection_dists)

    # ========== Debug stats ==========
    print(f"  [Step2] FPS complete: {len(selected)}/{n_select} stations selected")
    print(f"  [Step2] Pick-time distance: min={selection_dists.min():.6f}, "
          f"max={selection_dists.max():.6f}, mean={selection_dists.mean():.6f}")

    # Distance from each selected to nearest labeled
    dist_to_labeled = []
    for i in selected:
        d = np.sqrt(np.sum((pos_candidates[i:i+1] - pos_labeled)**2, axis=1)).min()
        dist_to_labeled.append(d)
    dist_to_labeled = np.array(dist_to_labeled)
    print(f"  [Step2] Selected dist to nearest labeled: min={dist_to_labeled.min():.6f}, "
          f"max={dist_to_labeled.max():.6f}, mean={dist_to_labeled.mean():.6f}")

    # Max adj_weight to labeled for selected stations
    print(f"  [Step2] Selected max adj to labeled: min={max_adj_to_labeled[selected].min():.4f}, "
          f"max={max_adj_to_labeled[selected].max():.4f}, mean={max_adj_to_labeled[selected].mean():.4f}")

    # Print selected indices for reproducibility and comparison
    print(f"  [Step2] Selected indices (first 20): {selected[:20].tolist()}")
    if len(selected) > 20:
        print(f"  [Step2] Selected indices (last 20): {selected[-20:].tolist()}")

    return selected


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

    # Select unlabeled stations using FPS (skip first 106 which are V2 labeled stations)
    start_idx = 106  # Skip all 106 V2 labeled stations
    total_unlabeled_available = 2000 - start_idx  # 1894 candidates

    # Get Map, SM, UF, Loc for FPS selection
    usable_idx = labeled['usable_station_indices'].squeeze().astype(int) if 'usable_station_indices' in labeled else np.arange(nNodes_labeled)
    Map_labeled_v2 = unlabeled['Map'][:106][usable_idx]
    SM_labeled_v2 = unlabeled['SimilarityMat'][:106][usable_idx]
    Map_all_candidates = unlabeled['Map'][start_idx:]
    SM_all_candidates = unlabeled['SimilarityMat'][start_idx:]

    # UrbanFeature and NodeLocation for filtering and spatial FPS
    UF_labeled_raw = unlabeled.get('UrbanFeature', None)
    UF_all_candidates = None
    if UF_labeled_raw is not None:
        UF_labeled_fps = UF_labeled_raw[:106][usable_idx]
        UF_all_candidates = UF_labeled_raw[start_idx:]

    Loc_labeled_v2 = unlabeled['NodeLocation'][:106][usable_idx]
    Loc_all_candidates = unlabeled['NodeLocation'][start_idx:]

    n_unlabeled = min(n_unlabeled, total_unlabeled_available)

    # FPS station selection
    # USE_FPS=0: fixed slice, USE_FPS=1: pure spatial FPS, USE_FPS=2: score-weighted FPS
    use_fps = int(os.environ.get('USE_FPS', 0))
    if use_fps and n_unlabeled < total_unlabeled_available:
        fps_mode = "score-weighted" if use_fps == 2 else "pure spatial"
        print(f"  Selecting {n_unlabeled} from {total_unlabeled_available} candidates using FPS ({fps_mode})...")
        fps_indices = select_unlabeled_fps(
            Map_labeled_v2, SM_labeled_v2,
            Map_all_candidates, SM_all_candidates,
            n_unlabeled,
            UF_labeled=UF_labeled_fps, UF_all_unlabeled=UF_all_candidates,
            Loc_labeled=Loc_labeled_v2, Loc_all_unlabeled=Loc_all_candidates,
            use_score=(use_fps == 2))
        # Convert to global indices in the 2000-station array
        global_indices = start_idx + fps_indices
    else:
        print(f"  Using all {n_unlabeled} unlabeled stations (no FPS)")
        fps_indices = np.arange(n_unlabeled)
        global_indices = np.arange(start_idx, start_idx + n_unlabeled)

    # Visualize FPS selection
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        loc_labeled = labeled['NodeLocation'] if 'NodeLocation' in labeled else None
        loc_all_u = unlabeled['NodeLocation']
        loc_not_selected = np.delete(np.arange(start_idx, 2000), fps_indices)

        if loc_labeled is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            # All candidates (gray, small)
            ax.scatter(loc_all_u[loc_not_selected, 1], loc_all_u[loc_not_selected, 0],
                       c='lightgray', s=8, alpha=0.4, label=f'Not selected ({len(loc_not_selected)})')
            # FPS selected (orange)
            ax.scatter(loc_all_u[global_indices, 1], loc_all_u[global_indices, 0],
                       c='orange', s=20, alpha=0.7, label=f'FPS selected ({n_unlabeled})')
            # Labeled (red triangles)
            ax.scatter(loc_labeled[:, 1], loc_labeled[:, 0],
                       c='red', s=80, marker='^', zorder=5, label=f'Labeled ({nNodes_labeled})')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'FPS Station Selection: {n_unlabeled} unlabeled from {total_unlabeled_available} candidates')
            ax.legend(fontsize=9)
            plt.tight_layout()

            # Save to output_dir (passed via env or use default)
            fps_fig_dir = os.environ.get('OUTPUT_DIR', '')
            if not fps_fig_dir:
                fps_fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
            os.makedirs(fps_fig_dir, exist_ok=True)
            fps_fig_path = os.path.join(fps_fig_dir, f'fps_selection_{n_unlabeled}u.png')
            plt.savefig(fps_fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [FPS] Selection map saved: {fps_fig_path}")
    except Exception as e:
        print(f"  [FPS] Visualization skipped: {e}")

    # WRF unlabeled (63 dims, no merge)
    WRFMat_u = unlabeled['WRFMat']
    if WRFMat_u.shape[0] != 6624:
        WRFMat_u = np.transpose(WRFMat_u, (2, 1, 0))
    WRFMat_u = WRFMat_u[:T_2018, :, global_indices]  # FPS selected stations
    cfd_unlabeled = np.transpose(WRFMat_u, (0, 2, 1))  # (T, nU, 63)

    # CLMS unlabeled
    CLMSMat_u = unlabeled['CLMSMat']
    if CLMSMat_u.shape[0] != 6624:
        CLMSMat_u = np.transpose(CLMSMat_u, (2, 1, 0))
    CLMSMat_u = CLMSMat_u[:T_2018, :, global_indices]
    clms_unlabeled = np.transpose(CLMSMat_u, (0, 2, 1))  # (T, nU, 3)

    # UrbanFeature unlabeled
    # Cols 0-15: height/fraction (NaN → 0), Col 16: distance to lake (NaN → median)
    UF_unlabeled = unlabeled.get('UrbanFeature', None)
    if UF_unlabeled is not None:
        UF_unlabeled = UF_unlabeled[global_indices, :]
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
        try:
            UFM_u = np.nan_to_num(UFM_u[:, :, :, global_indices], nan=0.0)
        except Exception as e:
            print(f"  [Warning] UrbanFeatureMat failed for {n_unlabeled} stations: {e}")
            UFM_u = None

    if UFM_u is not None:
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
    norm_mode = os.environ.get('NORM_MODE', 'per_station').lower()
    cfd_labeled_norm, cfd_off, cfd_scl = utils.MinMax(cfd_labeled.copy())
    clms_labeled_norm, clms_off, clms_scl = utils.MinMax(clms_labeled.copy())

    if norm_mode == 'global':
        # Global normalization: all stations share one min/max → cross-station comparable
        _tgt_global_min = targets_labeled.min()
        _tgt_global_max = targets_labeled.max()
        _tgt_global_scl = _tgt_global_max - _tgt_global_min
        if _tgt_global_scl == 0:
            _tgt_global_scl = 1.0
        targets_norm = (targets_labeled - _tgt_global_min) / _tgt_global_scl
        tgt_off = np.full(nNodes_labeled, _tgt_global_min)
        tgt_scl = np.full(nNodes_labeled, _tgt_global_scl)
        print(f"  Target normalization: GLOBAL (min={_tgt_global_min:.2f}, max={_tgt_global_max:.2f})")
    else:
        # Per-station normalization (default, backward compatible)
        targets_norm, tgt_off, tgt_scl = utils.MinMax(targets_labeled.copy())
        _tgt_global_min = targets_labeled.min()
        _tgt_global_max = targets_labeled.max()
        _tgt_global_scl = _tgt_global_max - _tgt_global_min
        if _tgt_global_scl == 0:
            _tgt_global_scl = 1.0
        print(f"  Target normalization: PER-STATION")

    # Global-normalized targets (always computed, for GSR)
    targets_global_norm = (targets_labeled - _tgt_global_min) / _tgt_global_scl

    # WRF T2 in °C (channel 0, convert K→°C) with same global scale as targets
    # For residual GSR: Δ = target - WRF_T2, both in same normalization
    wrf_t2_labeled_celsius = cfd_labeled[:, :, 0] - 273.15  # (T, nL) in °C
    wrf_t2_unlabeled_celsius = cfd_unlabeled[:, :, 0] - 273.15  # (T, nU) in °C
    # [Debug] Verify WRF channel 0 is temperature (should be reasonable °C range)
    assert -50 < wrf_t2_labeled_celsius.min() and wrf_t2_labeled_celsius.max() < 60, \
        f"WRF channel 0 may not be T2: range [{wrf_t2_labeled_celsius.min():.1f}, {wrf_t2_labeled_celsius.max():.1f}]°C"
    wrf_t2_labeled_gnorm = (wrf_t2_labeled_celsius - _tgt_global_min) / _tgt_global_scl
    wrf_t2_unlabeled_gnorm = (wrf_t2_unlabeled_celsius - _tgt_global_min) / _tgt_global_scl
    print(f"  WRF T2 (°C): labeled [{wrf_t2_labeled_celsius.min():.1f}, {wrf_t2_labeled_celsius.max():.1f}], "
          f"unlabeled [{wrf_t2_unlabeled_celsius.min():.1f}, {wrf_t2_unlabeled_celsius.max():.1f}]")
    # [Debug] Residual check: target - WRF_T2 should be small
    _delta_check = targets_global_norm - wrf_t2_labeled_gnorm
    print(f"  [Debug] Residual (target-WRF_T2) gnorm: mean={_delta_check.mean():.4f}, "
          f"std={_delta_check.std():.4f}, range=[{_delta_check.min():.4f}, {_delta_check.max():.4f}]")

    # [BugFix] Normalize unlabeled using LABELED parameters (not independently)
    # This ensures same physical value → same normalized value for both labeled & unlabeled nodes
    cfd_unlabeled_norm = (cfd_unlabeled - cfd_off) / (cfd_scl + 1e-8)
    cfd_unlabeled_norm = np.clip(cfd_unlabeled_norm, 0.0, 1.0)
    clms_unlabeled_norm = (clms_unlabeled - clms_off) / (clms_scl + 1e-8)
    clms_unlabeled_norm = np.clip(clms_unlabeled_norm, 0.0, 1.0)
    print(f"  [BugFix] Unlabeled WRF/CLMS normalized using labeled parameters (not independently)")
    # [Debug] Verify normalization consistency
    print(f"  [Debug] WRF norm range: labeled=[{cfd_labeled_norm.min():.3f}, {cfd_labeled_norm.max():.3f}], "
          f"unlabeled=[{cfd_unlabeled_norm.min():.3f}, {cfd_unlabeled_norm.max():.3f}]")

    # ======================== Build unified graph ========================
    print("Building unified graph (labeled + unlabeled)...")

    # Get Map and SimilarityMat for graph construction (using FPS-selected stations)
    Map_labeled = Map_labeled_v2  # already loaded for FPS
    SM_labeled = SM_labeled_v2

    Map_unlabeled = unlabeled['Map'][global_indices]
    SM_unlabeled = unlabeled['SimilarityMat'][global_indices]

    nNodes_total = nNodes_labeled + n_unlabeled

    # --- Unified graph computation (all nodes same distance normalization) ---
    # Step 3: Build graph with adaptive threshold (target density 30-80%)
    Map_all = np.vstack([Map_labeled, Map_unlabeled])
    SM_all = np.vstack([SM_labeled, SM_unlabeled])
    thres = dataParam['thres']
    Adj = build_adj_matrix(Map_all, SM_all, thres)
    np.fill_diagonal(Adj, 0)
    assert np.allclose(Adj, Adj.T)

    # Adaptive threshold: if density > 80%, increase threshold by 0.05 until 30-80%
    max_possible_edges = nNodes_total * (nNodes_total - 1) / 2
    current_edges = int(np.sum(Adj > 0) // 2)
    current_density = current_edges / max_possible_edges * 100
    original_thres = thres

    while current_density > 80.0 and thres < 0.95:
        thres += 0.05
        Adj[Adj < thres] = 0.0
        Adj = (Adj + Adj.T) / 2  # keep symmetric
        np.fill_diagonal(Adj, 0)
        current_edges = int(np.sum(Adj > 0) // 2)
        current_density = current_edges / max_possible_edges * 100
        # Stop if density drops below 30%
        if current_density < 30.0:
            print(f"  [Step3] Density dropped below 30% at thres={thres:.2f}, reverting to {thres-0.05:.2f}")
            thres -= 0.05
            # Rebuild at previous threshold
            Adj = build_adj_matrix(Map_all, SM_all, thres)
            np.fill_diagonal(Adj, 0)
            break

    if thres != original_thres:
        print(f"  [Step3] Adaptive threshold: {original_thres} -> {thres:.2f} (density: {current_density:.1f}%)")
    else:
        print(f"  [Step3] Threshold {thres} OK, density={current_density:.1f}% (within 30-80%)")

    # # --- Old: Block-wise graph construction (separate normalization per block) ---
    # thres_ll = dataParam.get('thres_ll', dataParam['thres'])
    # default_thres_uu = 0.4 if n_unlabeled > 500 else 0.4
    # default_thres_lu = 0.4 if n_unlabeled > 500 else 0.4
    # thres_uu = dataParam.get('thres_uu', default_thres_uu)
    # thres_lu = dataParam.get('thres_lu', default_thres_lu)
    # Adj_ll = build_adj_matrix(Map_labeled, SM_labeled, thres_ll)
    # Adj_uu = build_adj_matrix(Map_unlabeled, SM_unlabeled, thres_uu)
    # Map_cross = np.vstack([Map_labeled, Map_unlabeled])
    # SM_cross = np.vstack([SM_labeled, SM_unlabeled])
    # Adj_cross_full = build_adj_matrix(Map_cross, SM_cross, thres_lu)
    # Adj_lu = Adj_cross_full[:nNodes_labeled, nNodes_labeled:]
    # Adj = np.zeros((nNodes_total, nNodes_total))
    # Adj[:nNodes_labeled, :nNodes_labeled] = Adj_ll
    # Adj[nNodes_labeled:, nNodes_labeled:] = Adj_uu
    # Adj[:nNodes_labeled, nNodes_labeled:] = Adj_lu
    # Adj[nNodes_labeled:, :nNodes_labeled] = Adj_lu.T
    # np.fill_diagonal(Adj, 0)
    # assert np.allclose(Adj, Adj.T)

    # Edge statistics (before edge mode filtering)
    n_ll = int(np.sum(Adj[:nNodes_labeled, :nNodes_labeled] > 0) // 2)
    n_uu = int(np.sum(Adj[nNodes_labeled:, nNodes_labeled:] > 0) // 2)
    n_lu = int(np.sum(Adj[:nNodes_labeled, nNodes_labeled:] > 0))
    total_edges = int(np.sum(Adj > 0) // 2)
    density = total_edges / (nNodes_total * (nNodes_total - 1) / 2) * 100
    print(f"  Graph: {nNodes_total} nodes, {total_edges*2} edges, density={density:.1f}%")
    print(f"  L-L={n_ll}, U-U={n_uu}, L-U={n_lu} (unified thres={thres})")

    # Edge mode: optionally remove or discount UU edges
    edge_mode = os.environ.get('EDGE_MODE', 'all').lower()
    uu_discount = float(os.environ.get('UU_DISCOUNT', '0.1'))
    if edge_mode == 'no_uu':
        # Remove all U-U edges, keep only L-L and L-U
        Adj[nNodes_labeled:, nNodes_labeled:] = 0.0
        total_after = int(np.sum(Adj > 0) // 2)
        density_after = total_after / (nNodes_total * (nNodes_total - 1) / 2) * 100
        print(f"  [EDGE_MODE=no_uu] Removed {n_uu} U-U edges")
        print(f"  After: L-L={n_ll}, U-U=0, L-U={n_lu}, total={total_after}, density={density_after:.1f}%")
    elif edge_mode == 'discount_uu':
        # Discount U-U edges by a factor, keep L-L and L-U unchanged
        Adj[nNodes_labeled:, nNodes_labeled:] *= uu_discount
        n_uu_after = int(np.sum(Adj[nNodes_labeled:, nNodes_labeled:] > 0) // 2)
        total_after = int(np.sum(Adj > 0) // 2)
        density_after = total_after / (nNodes_total * (nNodes_total - 1) / 2) * 100
        print(f"  [EDGE_MODE=discount_uu] U-U edges multiplied by {uu_discount}")
        print(f"  After: L-L={n_ll}, U-U={n_uu_after}, L-U={n_lu}, total={total_after}, density={density_after:.1f}%")
    elif edge_mode in ('block_no_uu', 'block_discount_uu'):
        # Block-wise thresholds: rebuild LL with lower threshold, keep LU, handle UU
        thres_ll = float(os.environ.get('THRES_LL', '0.1'))
        thres_lu = float(os.environ.get('THRES_LU', '0.35'))
        # Rebuild LL block with lower threshold
        Adj_ll_new = build_adj_matrix(Map_labeled, SM_labeled, thres_ll)
        np.fill_diagonal(Adj_ll_new, 0)
        Adj[:nNodes_labeled, :nNodes_labeled] = Adj_ll_new
        # Rebuild LU block with its own threshold
        Map_cross = np.vstack([Map_labeled, Map_unlabeled])
        SM_cross = np.vstack([SM_labeled, SM_unlabeled])
        Adj_cross_full = build_adj_matrix(Map_cross, SM_cross, thres_lu)
        np.fill_diagonal(Adj_cross_full, 0)
        Adj[:nNodes_labeled, nNodes_labeled:] = Adj_cross_full[:nNodes_labeled, nNodes_labeled:]
        Adj[nNodes_labeled:, :nNodes_labeled] = Adj_cross_full[nNodes_labeled:, :nNodes_labeled]
        # Handle UU
        if edge_mode == 'block_no_uu':
            Adj[nNodes_labeled:, nNodes_labeled:] = 0.0
        elif edge_mode == 'block_discount_uu':
            Adj[nNodes_labeled:, nNodes_labeled:] *= uu_discount
        # Stats
        n_ll_new = int(np.sum(Adj[:nNodes_labeled, :nNodes_labeled] > 0) // 2)
        n_uu_new = int(np.sum(Adj[nNodes_labeled:, nNodes_labeled:] > 0) // 2)
        n_lu_new = int(np.sum(Adj[:nNodes_labeled, nNodes_labeled:] > 0))
        total_new = int(np.sum(Adj > 0) // 2)
        density_new = total_new / (nNodes_total * (nNodes_total - 1) / 2) * 100
        print(f"  [EDGE_MODE={edge_mode}] Block-wise thresholds: LL={thres_ll}, LU={thres_lu}, UU={'removed' if edge_mode=='block_no_uu' else f'x{uu_discount}'}")
        print(f"  After: L-L={n_ll_new} (was {n_ll}), U-U={n_uu_new} (was {n_uu}), L-U={n_lu_new} (was {n_lu}), total={total_new}, density={density_new:.1f}%")

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

        # Targets (only labeled have real targets) — per-station normalized
        targets_all = np.concatenate([
            targets_norm[n],
            np.zeros(n_unlabeled)
        ]).reshape(-1, 1)

        # Targets globally normalized (same scale for all stations, for GSR interpolation)
        targets_global = np.concatenate([
            targets_global_norm[n],
            np.zeros(n_unlabeled)
        ]).reshape(-1, 1)

        # WRF T2 in °C, globally normalized with same scale as targets (for residual GSR)
        wrf_t2_gnorm_all = np.concatenate([
            wrf_t2_labeled_gnorm[n],
            wrf_t2_unlabeled_gnorm[n]
        ]).reshape(-1, 1)

        # Residual target: Δ = target_global - WRF_T2_global (for residual prediction mode)
        y_residual = targets_global - wrf_t2_gnorm_all  # (nNodes, 1)

        _dataset.append(Data(
            x=torch.FloatTensor(features_all),
            y=torch.FloatTensor(targets_all),
            y_global=torch.FloatTensor(targets_global),
            y_residual=torch.FloatTensor(y_residual),
            wrf_t2=torch.FloatTensor(wrf_t2_gnorm_all),
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

    # ======================== Train/Valid Split ========================
    eval_mode = os.environ.get('EVAL_MODE', 'spatial').lower()

    if eval_mode == 'spatial':
        # --- Spatial split: leave 8 stations out for validation ---
        # Select 8 most spatially spread labeled stations as validation
        labeled_locs = labeled['NodeLocation']
        valid_station_idx, train_station_idx = select_validation_stations_fps(labeled_locs, n_valid=8)

        print(f"  [Spatial Split] Eval mode: SPATIAL (leave-{len(valid_station_idx)}-stations-out)")
        print(f"  [Spatial Split] Train stations: {len(train_station_idx)}, Valid stations: {len(valid_station_idx)}")
        print(f"  [Spatial Split] Valid station indices: {valid_station_idx.tolist()}")
        print(f"  [Spatial Split] All {T_2018 - 2*_window} timesteps used for both train and valid")

        # Visualize spatial split
        try:
            _output_dir = os.environ.get('OUTPUT_DIR', '')
            if _output_dir:
                try:
                    unlabeled_locs = unlabeled['NodeLocation'][global_indices]
                except:
                    unlabeled_locs = None
                visualize_spatial_split(labeled_locs, train_station_idx, valid_station_idx,
                                       unlabeled_locations=unlabeled_locs,
                                       save_path=os.path.join(_output_dir, 'spatial_split.png'))
        except Exception as e:
            print(f"  [Spatial Split] Visualization skipped: {e}")

        # Create train label_mask: only train stations are labeled
        train_label_mask = np.zeros(nNodes_total, dtype=bool)
        train_label_mask[train_station_idx] = True  # only 50 train stations

        # Create valid label_mask: only valid stations
        valid_label_mask = np.zeros(nNodes_total, dtype=bool)
        valid_label_mask[valid_station_idx] = True  # only 8 valid stations

        # Build train dataset (all timesteps, train_label_mask)
        trainSet = []
        for i, data in enumerate(_dataset):
            d = Data(x=data.x, y=data.y,
                     label_mask=torch.BoolTensor(train_label_mask),
                     edge_index=data.edge_index, edge_attr=data.edge_attr)
            for attr in ['wrf_t2', 'y_global', 'y_residual']:
                if hasattr(data, attr):
                    setattr(d, attr, getattr(data, attr))
            trainSet.append(d)

        # Build valid dataset (all timesteps, valid_label_mask)
        validSet = []
        for i, data in enumerate(_dataset):
            d = Data(x=data.x, y=data.y,
                     label_mask=torch.BoolTensor(valid_label_mask),
                     edge_index=data.edge_index, edge_attr=data.edge_attr)
            for attr in ['wrf_t2', 'y_global', 'y_residual']:
                if hasattr(data, attr):
                    setattr(d, attr, getattr(data, attr))
            validSet.append(d)

        _shuffle_train = dataParam.get('shuffle_train', not predMode)
        trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=_shuffle_train)
        valid_batch_size = min(_batchSize, len(validSet))
        validLoader = DataLoader(validSet, batch_size=valid_batch_size, shuffle=False)

        train_indices = list(range(len(trainSet)))
        valid_indices = list(range(len(validSet)))

    else:
        # --- Temporal split (original): 75/25 time split, all stations ---
        print(f"  [Temporal Split] Eval mode: TEMPORAL (75/25 time split)")
        _generator = torch.Generator().manual_seed(19)
        _trainLength = int(len(_dataset) * nTrn)
        _validLength = len(_dataset) - _trainLength
        trainSet, validSet = torch.utils.data.random_split(
            _dataset, [_trainLength, _validLength], _generator)
        _shuffle_train = dataParam.get('shuffle_train', not predMode)
        trainLoader = DataLoader(trainSet, batch_size=_batchSize, shuffle=_shuffle_train)
        valid_batch_size = min(_batchSize, len(validSet))
        validLoader = DataLoader(validSet, batch_size=valid_batch_size, shuffle=False)

        train_indices = trainSet.indices
        valid_indices = validSet.indices
        valid_station_idx = None
        train_station_idx = None

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
        'trainIdx': train_indices,
        'validIdx': valid_indices,
        'AdjMatrix': Adj,
        'label_mask': label_mask,  # original full label_mask (all 58 labeled)
        'tgt_off': tgt_off,
        'tgt_scl': tgt_scl,
        'eval_mode': eval_mode,
        'valid_station_idx': valid_station_idx,
        'train_station_idx': train_station_idx,
    }
    return trainLoader, validLoader, metadata, validSet

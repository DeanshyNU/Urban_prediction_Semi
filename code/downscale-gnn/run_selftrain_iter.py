"""
Iterative self-training: each round, re-rank remaining unlabeled candidates
with the latest model's snapshots, select K=100 best, retrain from scratch
on a graph that has only labeled + selected_so_far stations.

Round 0   : pure supervised on small graph (50 train + 8 valid only)
            → strongest "no aux used" baseline
Round k   : 1) Build SCORING subgraph (labeled + remaining candidates)
            2) Apply M_(k-1) snapshots → ε̂, σ, confidence
            3) Filter (top τ% conf) + greedy select K=100 NEW candidates
            4) Auto-stop if confident candidates < K
            5) Build TRAINING subgraph (labeled + selected_so_far)
            6) Train M_k from scratch on training subgraph
            7) Cyclic LR collect 5 snapshots S_k for next round
"""
import os
import copy
import pickle
from datetime import datetime

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import subgraph
import wandb

import data_semi, network_semi, utils

from run_selftrain_ead import (
    precompute_alpha_beta,
    train_ead_epoch,
    eval_ead,
    predict_eps_all,
    per_station_embedding,
    greedy_select,
    aggregate_snapshots,
)


# =========================================================================
# Setup
# =========================================================================
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
conv_type = os.environ.get('CONV_TYPE', 'graphconv').lower()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get('SLURM_JOB_ID', '')

EAD_ALPHA = int(os.environ.get('EAD_ALPHA', '1'))
EAD_BETA  = int(os.environ.get('EAD_BETA',  '0'))
EAD_LAP   = int(os.environ.get('EAD_LAP',   '1'))
LAMBDA_LAP = float(os.environ.get('LAMBDA_LAP', '0.1')) if EAD_LAP else 0.0
USE_GNN = int(os.environ.get('USE_GNN', '1'))
N_GNN   = int(os.environ.get('N_GNN', '3'))

ITER_N_UNLABELED   = int(os.environ.get('ITER_N_UNLABELED', '1799'))
ITER_N_ROUNDS_MAX  = int(os.environ.get('ITER_N_ROUNDS_MAX', '8'))
ITER_K_PER_ROUND   = int(os.environ.get('ITER_K_PER_ROUND', '100'))
ITER_BASE_EPOCHS   = int(os.environ.get('ITER_BASE_EPOCHS', '500'))
ITER_ROUND_EPOCHS  = int(os.environ.get('ITER_ROUND_EPOCHS', '500'))
ITER_M_SNAPSHOTS   = int(os.environ.get('ITER_M_SNAPSHOTS', '5'))
ITER_T_CYCLE       = int(os.environ.get('ITER_T_CYCLE', '40'))
ITER_LR_MAX        = float(os.environ.get('ITER_LR_MAX', '1e-3'))
ITER_LR_MIN        = float(os.environ.get('ITER_LR_MIN', '1e-5'))
ITER_TAU_QUANTILE  = float(os.environ.get('ITER_TAU_QUANTILE', '0.5'))
ITER_ALPHA_DIV     = float(os.environ.get('ITER_ALPHA_DIV', '1.0'))
ITER_BETA_REL      = float(os.environ.get('ITER_BETA_REL', '1.0'))
ITER_K_NN_REL      = int(os.environ.get('ITER_K_NN_REL', '5'))
ITER_LAMBDA_PSEUDO = float(os.environ.get('ITER_LAMBDA_PSEUDO', '0.5'))

iter_tag = f"st_iter_K{ITER_K_PER_ROUND}_R{ITER_N_ROUNDS_MAX}_N{ITER_N_UNLABELED}"
output_dir = os.path.join(project_root, 'log',
                          f'job{job_id}_{iter_tag}_{timestamp}' if job_id else f'{iter_tag}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
os.environ['OUTPUT_DIR'] = output_dir


# =========================================================================
# Subgraph utilities
# =========================================================================
def build_subgraph_dataset(full_dataset, active_indices, num_total_nodes,
                            alpha_per_sample, beta_hat,
                            pseudo_eps_all=None, pseudo_sigma_all=None,
                            pseudo_select_mask=None):
    """
    Subset every Data object in `full_dataset` to only `active_indices` nodes.
    Edges relabeled to [0, len(active)-1).

    Args:
        full_dataset       : iterable of PyG Data with full N=num_total_nodes nodes
        active_indices     : list/array of original node IDs to KEEP
        num_total_nodes    : full graph size (e.g. 1857)
        alpha_per_sample   : (T,) — α_t per timestep
        beta_hat           : (N_total,) — β̂ per station (same for all t)
        pseudo_eps_all     : (T, N_total) or None
        pseudo_sigma_all   : (T, N_total) or None
        pseudo_select_mask : (N_total,) bool or None
    Returns:
        new_dataset : list of subsetted Data
    """
    active_sorted = sorted([int(x) for x in active_indices])
    active_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
    active_mask[torch.LongTensor(active_sorted)] = True

    # Subset edges once (assumes shared edge structure across timesteps)
    sample_d = full_dataset[0]
    sub_ei, sub_ea = subgraph(active_mask, sample_d.edge_index, sample_d.edge_attr,
                              relabel_nodes=True, num_nodes=num_total_nodes)

    # Static beta_hat subset
    beta_hat_full_t = torch.FloatTensor(beta_hat).unsqueeze(-1)
    beta_hat_sub = beta_hat_full_t[active_mask]

    # Static pseudo_select_mask subset (same across all t)
    if pseudo_select_mask is not None:
        select_full_t = torch.from_numpy(pseudo_select_mask).bool().unsqueeze(-1)
        select_sub = select_full_t[active_mask]
    else:
        select_sub = None

    new_dataset = []
    for t, d in enumerate(full_dataset):
        new = Data()
        # Per-node tensors
        for attr in ['x', 'y', 'y_residual', 'wrf_t2']:
            if hasattr(d, attr) and getattr(d, attr) is not None:
                new[attr] = getattr(d, attr)[active_mask]
        if hasattr(d, 'label_mask') and d.label_mask is not None:
            new.label_mask = d.label_mask[active_mask]
        # Edges (shared)
        new.edge_index = sub_ei
        new.edge_attr  = sub_ea
        # Per-graph scalar
        new.alpha_t = torch.FloatTensor([alpha_per_sample[t]])
        # Static
        new.beta_hat = beta_hat_sub.clone()
        # Pseudo (per-t)
        if pseudo_eps_all is not None:
            pe = torch.FloatTensor(pseudo_eps_all[t]).unsqueeze(-1)[active_mask]
            ps = torch.FloatTensor(pseudo_sigma_all[t]).unsqueeze(-1)[active_mask]
            new.pseudo_eps   = pe
            new.pseudo_sigma = ps
            new.pseudo_select_mask = select_sub.clone()
        new_dataset.append(new)

    return new_dataset


def map_local_to_full(local_array_per_node, scoring_node_ids, num_total_nodes,
                       default_value=0.0, time_dim=False):
    """
    Map array indexed by scoring-graph local positions back to original
    full-graph node IDs.

    Args:
        local_array_per_node : either (n_scoring,) or (T, n_scoring)
        scoring_node_ids     : sorted original IDs of scoring graph nodes
        num_total_nodes      : full graph size
        default_value        : value at non-scoring positions
        time_dim             : if True, input has shape (T, n_scoring)
    Returns:
        full_array : (num_total_nodes,) or (T, num_total_nodes)
    """
    scoring_node_ids = np.array(scoring_node_ids)
    if time_dim:
        T = local_array_per_node.shape[0]
        out = np.full((T, num_total_nodes), default_value, dtype=np.float32)
        out[:, scoring_node_ids] = local_array_per_node
    else:
        out = np.full(num_total_nodes, default_value, dtype=np.float32)
        out[scoring_node_ids] = local_array_per_node
    return out


# =========================================================================
# Main
# =========================================================================
def main():
    log_path = os.path.join(output_dir, f'selftrain_iter_log')
    with open(log_path, 'w') as f:
        f.write(f"=== Iterative self-training ===\n")
        f.write(f"N_UNLABELED={ITER_N_UNLABELED}, K_PER_ROUND={ITER_K_PER_ROUND}, "
                f"MAX_ROUNDS={ITER_N_ROUNDS_MAX}, "
                f"BASE_EP={ITER_BASE_EPOCHS}, ROUND_EP={ITER_ROUND_EPOCHS}, "
                f"M={ITER_M_SNAPSHOTS}, T_CYCLE={ITER_T_CYCLE}, "
                f"τ={ITER_TAU_QUANTILE}, λ_pseudo={ITER_LAMBDA_PSEUDO}\n")

    # ----- Load FULL data once -----
    print("\n" + "#"*72)
    print(f"# Loading FULL dataset (N_UNLABELED={ITER_N_UNLABELED})")
    print("#"*72)
    dataParam = {
        'geoMethod': 'average', 'nCompPCA': 40, 'window': 2,
        'poolSize': int(os.environ.get('POOL_SIZE', '12')),
        'batchSize': 512, 'thres': 0.1,
        'geoFeatures': 'full', 'n_unlabeled': ITER_N_UNLABELED,
    }
    os.environ.setdefault('EVAL_MODE', 'spatial')
    os.environ.setdefault('USE_FPS', '2')
    os.environ.setdefault('EDGE_MODE', 'no_uu')
    os.environ.setdefault('NORM_MODE', 'global')

    trainL_full, validL_full, meta, _ = data_semi.dataGen(dataParam, path)
    full_train_ds = trainL_full.dataset
    full_valid_ds = validL_full.dataset
    nNodes_total   = meta['nNodes']
    nNodes_labeled = meta['nNodes_labeled']
    train_idx = np.array(meta['train_station_idx'])
    valid_idx = np.array(meta['valid_station_idx'])
    tgt_scl_celsius = float(meta.get('tgt_global_scl', 1.0))

    print(f"  Total nodes: {nNodes_total} ({nNodes_labeled} labeled, "
          f"{nNodes_total - nNodes_labeled} unlabeled candidate pool)")

    alpha_per_sample, beta_hat, _ = precompute_alpha_beta(trainL_full, meta)

    all_unlabeled_set = set(range(nNodes_labeled, nNodes_total))
    labeled_idx = np.concatenate([train_idx, valid_idx])
    selected_so_far = set()  # original node IDs

    modelParam = {
        'HLD': 128, 'nMLP': 2, 'nGNN': N_GNN, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'iDim': meta['iDim'], 'oDim': meta['oDim'],
        'BN': False, 'Dropout': True, 'conv_type': conv_type,
        'use_gnn': bool(USE_GNN),
    }

    modelName = "selftrain_iter"
    wandb.init(
        entity="urban_prediction", project="Semi-supervised GNN",
        name=f"{modelName}_job{job_id}" if job_id else modelName,
        config={**modelParam,
                'ITER_N_UNLABELED': ITER_N_UNLABELED,
                'ITER_K_PER_ROUND': ITER_K_PER_ROUND,
                'ITER_N_ROUNDS_MAX': ITER_N_ROUNDS_MAX,
                'ITER_BASE_EPOCHS': ITER_BASE_EPOCHS,
                'ITER_ROUND_EPOCHS': ITER_ROUND_EPOCHS,
                'ITER_LAMBDA_PSEUDO': ITER_LAMBDA_PSEUDO,
                'ITER_TAU_QUANTILE': ITER_TAU_QUANTILE,
                'ITER_ALPHA_DIV': ITER_ALPHA_DIV,
                'ITER_BETA_REL': ITER_BETA_REL,
                'EAD_ALPHA': EAD_ALPHA, 'LAMBDA_LAP': LAMBDA_LAP,
                'variant': 'iterative'},
    )

    # =========================================================================
    # ROUND 0 — clean baseline (no aux at all)
    # =========================================================================
    print("\n" + "#"*72)
    print("# ROUND 0 — pure supervised baseline (50 train + 8 valid, no aux)")
    print("#"*72)
    active_r0 = labeled_idx.tolist()
    train_ds_r0 = build_subgraph_dataset(full_train_ds, active_r0, nNodes_total,
                                          alpha_per_sample, beta_hat)
    valid_ds_r0 = build_subgraph_dataset(full_valid_ds, active_r0, nNodes_total,
                                          alpha_per_sample, beta_hat)
    n_r0 = len(active_r0)
    train_loader_r0 = PyGDataLoader(train_ds_r0, batch_size=512, shuffle=True)
    valid_loader_r0 = PyGDataLoader(valid_ds_r0, batch_size=512, shuffle=False)

    sample = train_ds_r0[0]
    edge_src_r0 = sample.edge_index[0].to(device)
    edge_dst_r0 = sample.edge_index[1].to(device)
    edge_w_r0   = sample.edge_attr.to(device)
    print(f"  Round 0 graph: {n_r0} nodes, {sample.edge_index.shape[1]} edges")

    gnn = network_semi.GNN(modelParam).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)

    best_r0_rmse = float('inf')
    best_r0_state = None
    for epoch in range(ITER_BASE_EPOCHS):
        stats, train_rmse = train_ead_epoch(
            train_loader_r0, gnn, opt, sched, device,
            n_r0, edge_src_r0, edge_dst_r0, edge_w_r0, use_pseudo=False)
        em = eval_ead(valid_loader_r0, gnn, device, n_r0, nNodes_labeled,
                      tgt_scl=tgt_scl_celsius)
        v_rmse = em['rmse_norm']
        if v_rmse < best_r0_rmse:
            best_r0_rmse = v_rmse
            best_r0_state = copy.deepcopy(gnn.state_dict())
        if epoch % 25 == 0 or epoch < 3:
            line = (f"R0 ep{epoch:4d}: train_RMSE={train_rmse[0]:.4f}  "
                    f"valid_RMSE={v_rmse:.4f} ({em['rmse_C']:.3f}°C)")
            print(line)
            with open(log_path, 'a') as f: f.write(line + '\n')
        wandb.log({'round': 0, 'global_epoch': epoch,
                   'r0/train_rmse': train_rmse[0], 'r0/valid_rmse': v_rmse,
                   'r0/best_rmse': best_r0_rmse})

    print(f"\n[Round 0 done] best_valid_RMSE = {best_r0_rmse:.4f}")
    with open(log_path, 'a') as f: f.write(f"[R0 done] best={best_r0_rmse:.4f}\n")
    gnn.load_state_dict(best_r0_state)

    # ----- Round 0 snapshots (still on small graph) -----
    print(f"\n  Snapshot collection ({ITER_M_SNAPSHOTS} × {ITER_T_CYCLE} ep)")
    for pg in opt.param_groups: pg['lr'] = ITER_LR_MAX
    snap_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=ITER_T_CYCLE, T_mult=1, eta_min=ITER_LR_MIN)
    snapshot_states = []
    for ep in range(ITER_M_SNAPSHOTS * ITER_T_CYCLE):
        train_ead_epoch(train_loader_r0, gnn, opt, snap_sched, device,
                        n_r0, edge_src_r0, edge_dst_r0, edge_w_r0, use_pseudo=False)
        if (ep + 1) % ITER_T_CYCLE == 0:
            snapshot_states.append(copy.deepcopy(gnn.state_dict()))
            ev = eval_ead(valid_loader_r0, gnn, device, n_r0, nNodes_labeled,
                          tgt_scl=tgt_scl_celsius)
            line = f"  [Snapshot {len(snapshot_states)}/{ITER_M_SNAPSHOTS}] valid_RMSE={ev['rmse_norm']:.4f}"
            print(line)
            with open(log_path, 'a') as f: f.write(line + '\n')

    # Track best across rounds
    overall_best = {'round': 0, 'rmse': best_r0_rmse, 'state': best_r0_state}

    # Pre-extract y_residual once (for pseudo quality checks)
    print("\n  Pre-extracting y_residual for quality checks...")
    all_y_residual = np.zeros((len(full_train_ds), nNodes_total), dtype=np.float32)
    for t, d in enumerate(full_train_ds):
        all_y_residual[t] = d.y_residual.squeeze(-1).cpu().numpy()
    true_eps_full = all_y_residual - alpha_per_sample[:, None] - beta_hat[None, :]

    # =========================================================================
    # ROUND 1..N — iterative selection + retrain
    # =========================================================================
    round_results = []
    snapshot_train_n_active = n_r0   # n_active for the graph the snapshots came from

    for round_k in range(1, ITER_N_ROUNDS_MAX + 1):
        print("\n" + "#"*72)
        print(f"# ROUND {round_k}")
        print("#"*72)

        # ---- Step A: Build SCORING subgraph (labeled + remaining) ----
        remaining = sorted(all_unlabeled_set - selected_so_far)
        if len(remaining) < ITER_K_PER_ROUND:
            line = f"  [STOP] only {len(remaining)} candidates left; ending."
            print(line)
            with open(log_path, 'a') as f: f.write(line + '\n')
            break

        scoring_nodes = sorted(labeled_idx.tolist() + remaining)
        n_scoring = len(scoring_nodes)
        print(f"  Scoring graph: {n_scoring} nodes "
              f"(50+8+{len(remaining)} remaining)")

        # NB: snapshots came from a DIFFERENT graph size (snapshot_train_n_active).
        # When applied to scoring graph (n_scoring), inference reshapes by n_scoring.
        scoring_train_ds = build_subgraph_dataset(
            full_train_ds, scoring_nodes, nNodes_total, alpha_per_sample, beta_hat)

        # ---- Step B: Snapshot ensemble inference ----
        print(f"  Inferring with {len(snapshot_states)} snapshots...")
        snap_preds = []
        for state in snapshot_states:
            gnn_inf = network_semi.GNN(modelParam).to(device)
            gnn_inf.load_state_dict(state)
            eps_pred = predict_eps_all(gnn_inf, scoring_train_ds, device, n_scoring)
            snap_preds.append(eps_pred)

        ens_mean_loc, ens_std_loc, conf_loc = aggregate_snapshots(snap_preds)

        # Embeddings (last snapshot)
        gnn_emb = network_semi.GNN(modelParam).to(device)
        gnn_emb.load_state_dict(snapshot_states[-1])
        emb_loc = per_station_embedding(gnn_emb, scoring_train_ds, device, n_scoring)

        # ---- Step C: Map local → full coordinates ----
        ens_mean_full = map_local_to_full(ens_mean_loc, scoring_nodes, nNodes_total,
                                           default_value=0.0, time_dim=True)
        ens_std_full  = map_local_to_full(ens_std_loc, scoring_nodes, nNodes_total,
                                           default_value=0.0, time_dim=True)
        conf_full = map_local_to_full(conf_loc, scoring_nodes, nNodes_total,
                                       default_value=-1e9, time_dim=False)
        # Embeddings: (n_scoring, HLD) → (nNodes_total, HLD)
        emb_full = np.zeros((nNodes_total, emb_loc.shape[1]), dtype=np.float32)
        emb_full[np.array(scoring_nodes)] = emb_loc

        # ---- Step D: Quality diagnostics ----
        quality_warnings = []

        # Pseudo-label quality on V (held-out, ground truth ε known)
        pseudo_rmse_v = float(np.sqrt(
            ((ens_mean_full[:, valid_idx] - true_eps_full[:, valid_idx]) ** 2).mean(axis=0)
        ).mean())
        line = f"  [Quality] pseudo ε̂ RMSE on V = {pseudo_rmse_v:.4f}  (R0 best = {best_r0_rmse:.4f})"
        print(line)
        with open(log_path, 'a') as f: f.write(line + '\n')
        if pseudo_rmse_v > 1.5 * best_r0_rmse:
            quality_warnings.append(f"Pseudo RMSE on V ({pseudo_rmse_v:.4f}) > 1.5× R0 best")

        # Confidence breakdown
        conf_train_mean = conf_full[train_idx].mean()
        conf_valid_mean = conf_full[valid_idx].mean()
        conf_rem_mean   = conf_full[np.array(remaining)].mean()
        print(f"  [Confidence] train={conf_train_mean:+.4f}, "
              f"valid={conf_valid_mean:+.4f}, remaining={conf_rem_mean:+.4f}")

        # Over-smoothing proxy: spatial std at remaining vs at labeled (in pseudo space)
        ps_std_lab = ens_mean_full[:, labeled_idx].std(axis=1).mean()
        ps_std_rem = ens_mean_full[:, np.array(remaining)].std(axis=1).mean()
        smooth_ratio = ps_std_rem / (ps_std_lab + 1e-8)
        print(f"  [Over-smoothing] pseudo-spatial-std remaining/labeled = {smooth_ratio:.3f}")
        if smooth_ratio < 0.5:
            quality_warnings.append(f"Spatial std ratio (rem/lab) = {smooth_ratio:.2f} < 0.5")

        # Snapshot diversity
        snap_std_diag = ens_std_loc.mean()
        print(f"  [Snapshot diag] mean ensemble σ = {snap_std_diag:.4f}")

        # ---- Step E: Greedy selection ----
        new_picks = greedy_select(
            emb_all=emb_full,
            candidate_idx=np.array(remaining),
            valid_idx=valid_idx,
            confidence_per_station=conf_full,
            K=ITER_K_PER_ROUND,
            alpha=ITER_ALPHA_DIV, beta=ITER_BETA_REL,
            k_nn_rel=ITER_K_NN_REL,
            tau_quantile=ITER_TAU_QUANTILE,
            ablation='full',
        )
        if len(new_picks) < ITER_K_PER_ROUND:
            line = (f"  [AUTO-STOP] selection returned only {len(new_picks)}/{ITER_K_PER_ROUND}. "
                    f"Confidence filter dropped too many. Ending.")
            print(line)
            with open(log_path, 'a') as f: f.write(line + '\n')
            if len(new_picks) > 0:
                selected_so_far.update(new_picks)
            break

        selected_so_far.update(new_picks)
        line = (f"  [Selection] picked {len(new_picks)} new; "
                f"|S_total| = {len(selected_so_far)}")
        print(line)
        with open(log_path, 'a') as f: f.write(line + '\n')

        # ---- Step F: Build TRAINING subgraph (labeled + selected_so_far) ----
        training_nodes = sorted(labeled_idx.tolist() + list(selected_so_far))
        n_train_round = len(training_nodes)

        select_mask_full = np.zeros(nNodes_total, dtype=bool)
        for s in selected_so_far:
            select_mask_full[s] = True

        train_ds_round = build_subgraph_dataset(
            full_train_ds, training_nodes, nNodes_total,
            alpha_per_sample, beta_hat,
            pseudo_eps_all=ens_mean_full,
            pseudo_sigma_all=ens_std_full,
            pseudo_select_mask=select_mask_full)
        valid_ds_round = build_subgraph_dataset(
            full_valid_ds, training_nodes, nNodes_total,
            alpha_per_sample, beta_hat)
        train_loader_round = PyGDataLoader(train_ds_round, batch_size=512, shuffle=True)
        valid_loader_round = PyGDataLoader(valid_ds_round, batch_size=512, shuffle=False)

        sample = train_ds_round[0]
        edge_src_r = sample.edge_index[0].to(device)
        edge_dst_r = sample.edge_index[1].to(device)
        edge_w_r   = sample.edge_attr.to(device)
        print(f"  Training graph: {n_train_round} nodes, "
              f"{sample.edge_index.shape[1]} edges, |S_pseudo| = {len(selected_so_far)}")

        # ---- Step G: Train M_k from scratch ----
        print(f"\n  Training R{round_k} model from scratch ({ITER_ROUND_EPOCHS} ep)...")
        gnn = network_semi.GNN(modelParam).to(device)
        opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)

        best_r_rmse = float('inf')
        best_r_state = None
        for epoch in range(ITER_ROUND_EPOCHS):
            stats, train_rmse = train_ead_epoch(
                train_loader_round, gnn, opt, sched, device,
                n_train_round, edge_src_r, edge_dst_r, edge_w_r,
                use_pseudo=True, lambda_pseudo=ITER_LAMBDA_PSEUDO)
            em = eval_ead(valid_loader_round, gnn, device, n_train_round, nNodes_labeled,
                          tgt_scl=tgt_scl_celsius)
            v_rmse = em['rmse_norm']
            if v_rmse < best_r_rmse:
                best_r_rmse = v_rmse
                best_r_state = copy.deepcopy(gnn.state_dict())
            if epoch % 25 == 0 or epoch < 3:
                line = (f"R{round_k} ep{epoch:4d}: train={train_rmse[0]:.4f} "
                        f"valid={v_rmse:.4f}  sup={stats['sup_eps']:.4e} "
                        f"pseudo={stats['pseudo']:.4e}  lap={stats['lap']:.4e}")
                print(line)
                with open(log_path, 'a') as f: f.write(line + '\n')
            wandb.log({
                'round': round_k, 'global_epoch': epoch,
                f'r{round_k}/train_rmse': train_rmse[0],
                f'r{round_k}/valid_rmse': v_rmse,
                f'r{round_k}/best_rmse':  best_r_rmse,
                f'r{round_k}/sup_loss':   stats['sup_eps'],
                f'r{round_k}/pseudo_loss': stats['pseudo'],
                f'r{round_k}/lap_loss':   stats['lap'],
                'meta/n_selected': len(selected_so_far),
                'meta/best_rmse_so_far': min(overall_best['rmse'], best_r_rmse),
            })

        line = f"\n[Round {round_k} done] best_valid_RMSE = {best_r_rmse:.4f}"
        print(line)
        with open(log_path, 'a') as f: f.write(line + '\n')

        if best_r_rmse < overall_best['rmse']:
            overall_best = {'round': round_k, 'rmse': best_r_rmse, 'state': best_r_state}

        round_results.append({
            'round': round_k,
            'best_rmse': best_r_rmse,
            'n_selected_total': len(selected_so_far),
            'n_picked_this_round': len(new_picks),
            'pseudo_rmse_v': pseudo_rmse_v,
            'smooth_ratio': float(smooth_ratio),
            'mean_ensemble_sigma': float(snap_std_diag),
            'warnings': quality_warnings,
        })

        # ---- Step H: Snapshot collection for next round ----
        print(f"  Snapshot collection for R{round_k+1}...")
        gnn.load_state_dict(best_r_state)
        for pg in opt.param_groups: pg['lr'] = ITER_LR_MAX
        snap_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=ITER_T_CYCLE, T_mult=1, eta_min=ITER_LR_MIN)
        snapshot_states = []
        for ep in range(ITER_M_SNAPSHOTS * ITER_T_CYCLE):
            train_ead_epoch(train_loader_round, gnn, opt, snap_sched, device,
                            n_train_round, edge_src_r, edge_dst_r, edge_w_r,
                            use_pseudo=True, lambda_pseudo=ITER_LAMBDA_PSEUDO)
            if (ep + 1) % ITER_T_CYCLE == 0:
                snapshot_states.append(copy.deepcopy(gnn.state_dict()))
                ev = eval_ead(valid_loader_round, gnn, device, n_train_round, nNodes_labeled,
                              tgt_scl=tgt_scl_celsius)
                print(f"    [Snapshot {len(snapshot_states)}/{ITER_M_SNAPSHOTS}] "
                      f"valid_RMSE={ev['rmse_norm']:.4f}")
        snapshot_train_n_active = n_train_round

        # Save artifacts after each round
        torch.save({
            'overall_best_round': overall_best['round'],
            'overall_best_rmse': overall_best['rmse'],
            'gnn_state_dict': overall_best['state'],
            'selected_so_far': sorted(selected_so_far),
            'round_results': round_results,
        }, os.path.join(output_dir, f'{modelName}.pt'))
        with open(os.path.join(output_dir, 'selection.pkl'), 'wb') as f:
            pickle.dump({
                'selected_so_far': sorted(selected_so_far),
                'round_results': round_results,
                'r0_best_rmse': best_r0_rmse,
                'overall_best_round': overall_best['round'],
                'overall_best_rmse': overall_best['rmse'],
            }, f)

    # =========================================================================
    # FINAL summary
    # =========================================================================
    print("\n" + "="*72)
    print("  FINAL RESULTS (iterative variant)")
    print("="*72)
    print(f"  Total pseudo-labels selected : {len(selected_so_far)}")
    print(f"  Rounds completed             : {len(round_results)}")
    print(f"  Round 0 best (clean baseline): {best_r0_rmse:.4f}")
    print(f"  Overall best                  : R{overall_best['round']} = {overall_best['rmse']:.4f}")
    print(f"  Reference B1 (13204)          : 0.0428")
    print(f"\n  Per-round summary:")
    print(f"  {'Round':>6} {'#PL':>6} {'best_RMSE':>10} {'pseudo_v':>10} {'smooth':>8} {'σ_ens':>8}")
    print(f"  {'R0':>6} {'0':>6} {best_r0_rmse:>10.4f} {'-':>10} {'-':>8} {'-':>8}")
    for r in round_results:
        warn = '⚠' if r['warnings'] else ' '
        print(f"  {'R'+str(r['round']):>6} {r['n_selected_total']:>6} "
              f"{r['best_rmse']:>10.4f} {r['pseudo_rmse_v']:>10.4f} "
              f"{r['smooth_ratio']:>8.3f} {r['mean_ensemble_sigma']:>8.4f} {warn}")
    print("="*72)
    with open(log_path, 'a') as f:
        f.write(f"\n=== FINAL ===\n")
        f.write(f"R0_best={best_r0_rmse:.4f}, overall_best=R{overall_best['round']}={overall_best['rmse']:.4f}\n")
        f.write(f"Total pseudo-labels: {len(selected_so_far)}\n")
        for r in round_results:
            f.write(f"  R{r['round']}: rmse={r['best_rmse']:.4f}, |S|={r['n_selected_total']}, "
                    f"pseudo_v={r['pseudo_rmse_v']:.4f}, smooth={r['smooth_ratio']:.3f}\n")
            for w in r['warnings']:
                f.write(f"    ⚠ {w}\n")


if __name__ == '__main__':
    main()

"""
Iterative Self-Training V2 — addresses 13375's failure modes.

KEY CHANGES vs V1:
  A1. **Target = best_prev_state's prediction** (single forward, not snapshot avg)
      → directly uses the converged R0/Rk-1 model that achieves best valid_RMSE
      → avoids "snapshot ensemble worse than base model" problem (V1: snapshot avg
        on V was 0.0484 vs base 0.0427)

  A2. **σ = small-LR snapshot perturbations** (LR=ITER_LR_MAX small, e.g. 5e-4)
      → snapshots stay NEAR best state, so σ is informative but small
      → only used for confidence weighting, not for target

  B.  **Warm-start each round** from best_prev_state (not from-scratch random init)
      → preserves convergence at 0.0427; round only fine-tunes to incorporate pseudo-labels
      → avoids "from-scratch retraining can't recover R0's quality" problem

ADDITIONAL DEBUG:
  - Per-round: print best_prev_state's V RMSE (sanity check the target source)
  - Per-snapshot: print V RMSE individually (verify σ-snapshots stay near best)
  - Warm-start: print initial vs final RMSE (verify warm-start helps)
  - Compare target vs current model on V at multiple points
  - Track 'lookback ratio': how often selected nodes overlap with previous rounds
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
)
from run_selftrain_iter import (
    build_subgraph_dataset,
    map_local_to_full,
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

# V2-specific config
ITER_N_UNLABELED   = int(os.environ.get('ITER_N_UNLABELED', '1799'))
ITER_N_ROUNDS_MAX  = int(os.environ.get('ITER_N_ROUNDS_MAX', '8'))
ITER_K_PER_ROUND   = int(os.environ.get('ITER_K_PER_ROUND', '50'))      # smaller K
ITER_BASE_EPOCHS   = int(os.environ.get('ITER_BASE_EPOCHS', '500'))
ITER_ROUND_EPOCHS  = int(os.environ.get('ITER_ROUND_EPOCHS', '200'))    # warm-start needs less
ITER_M_SNAPSHOTS   = int(os.environ.get('ITER_M_SNAPSHOTS', '5'))
ITER_T_CYCLE       = int(os.environ.get('ITER_T_CYCLE', '20'))         # smaller cycle
ITER_LR_MAX        = float(os.environ.get('ITER_LR_MAX', '5e-4'))      # ↓ smaller (was 1e-3)
ITER_LR_MIN        = float(os.environ.get('ITER_LR_MIN', '1e-5'))
ITER_WARMSTART_LR  = float(os.environ.get('ITER_WARMSTART_LR', '1e-4'))  # for round k≥1 warm-start
ITER_TAU_QUANTILE  = float(os.environ.get('ITER_TAU_QUANTILE', '0.5'))
ITER_ALPHA_DIV     = float(os.environ.get('ITER_ALPHA_DIV', '1.0'))
ITER_BETA_REL      = float(os.environ.get('ITER_BETA_REL', '1.0'))
ITER_K_NN_REL      = int(os.environ.get('ITER_K_NN_REL', '5'))
ITER_LAMBDA_PSEUDO = float(os.environ.get('ITER_LAMBDA_PSEUDO', '0.3'))   # ↓ smaller (was 0.5)

iter_tag = f"st_iter_v2_K{ITER_K_PER_ROUND}_R{ITER_N_ROUNDS_MAX}_N{ITER_N_UNLABELED}"
output_dir = os.path.join(project_root, 'log',
                          f'job{job_id}_{iter_tag}_{timestamp}' if job_id else f'{iter_tag}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
os.environ['OUTPUT_DIR'] = output_dir


# =========================================================================
# Helper: collect σ-snapshots near best_state
# =========================================================================
def collect_sigma_snapshots(model, train_loader, device, n_active,
                             edge_src, edge_dst, edge_w,
                             use_pseudo, lambda_pseudo,
                             m_snapshots, t_cycle, lr_max, lr_min,
                             tag=""):
    """
    Starting from current model state (assumed near best), do small-LR cyclic
    perturbations and save M snapshots. Snapshots stay close to best because LR is small.

    Returns: list of state_dict
    """
    print(f"\n  [σ-Snapshots {tag}] cyclic LR max={lr_max:.0e} min={lr_min:.0e}, "
          f"{m_snapshots} × {t_cycle} ep")
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    snap_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=t_cycle, T_mult=1, eta_min=lr_min)

    snapshot_states = []
    for ep in range(m_snapshots * t_cycle):
        train_ead_epoch(train_loader, model, opt, snap_sched, device,
                        n_active, edge_src, edge_dst, edge_w,
                        use_pseudo=use_pseudo, lambda_pseudo=lambda_pseudo)
        if (ep + 1) % t_cycle == 0:
            snapshot_states.append(copy.deepcopy(model.state_dict()))
    return snapshot_states


# =========================================================================
# Main
# =========================================================================
def main():
    log_path = os.path.join(output_dir, f'selftrain_iter_v2_log')
    with open(log_path, 'w') as f:
        f.write(f"=== Iterative self-training V2 ===\n")
        f.write(f"N_UNLABELED={ITER_N_UNLABELED}, K_PER_ROUND={ITER_K_PER_ROUND}, "
                f"MAX_ROUNDS={ITER_N_ROUNDS_MAX}\n")
        f.write(f"BASE_EP={ITER_BASE_EPOCHS}, ROUND_EP={ITER_ROUND_EPOCHS} (warm-start), "
                f"M_σ={ITER_M_SNAPSHOTS}, T_CYCLE={ITER_T_CYCLE}\n")
        f.write(f"LR_MAX (σ-snapshots)={ITER_LR_MAX:.0e}, "
                f"LR (warm-start)={ITER_WARMSTART_LR:.0e}\n")
        f.write(f"τ={ITER_TAU_QUANTILE}, λ_pseudo={ITER_LAMBDA_PSEUDO}\n")

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
          f"{nNodes_total - nNodes_labeled} unlabeled)")

    alpha_per_sample, beta_hat, _ = precompute_alpha_beta(trainL_full, meta)

    all_unlabeled_set = set(range(nNodes_labeled, nNodes_total))
    labeled_idx = np.concatenate([train_idx, valid_idx])
    selected_so_far = set()

    modelParam = {
        'HLD': 128, 'nMLP': 2, 'nGNN': N_GNN, 'nGAT': 1, 'nHeads': 1, 'K': 1,
        'iDim': meta['iDim'], 'oDim': meta['oDim'],
        'BN': False, 'Dropout': True, 'conv_type': conv_type,
        'use_gnn': bool(USE_GNN),
    }

    modelName = "selftrain_iter_v2"
    wandb.init(
        entity="urban_prediction", project="Semi-supervised GNN",
        name=f"{modelName}_job{job_id}" if job_id else modelName,
        config={**modelParam,
                'ITER_N_UNLABELED': ITER_N_UNLABELED,
                'ITER_K_PER_ROUND': ITER_K_PER_ROUND,
                'ITER_N_ROUNDS_MAX': ITER_N_ROUNDS_MAX,
                'ITER_BASE_EPOCHS': ITER_BASE_EPOCHS,
                'ITER_ROUND_EPOCHS': ITER_ROUND_EPOCHS,
                'ITER_LR_MAX': ITER_LR_MAX,
                'ITER_WARMSTART_LR': ITER_WARMSTART_LR,
                'ITER_LAMBDA_PSEUDO': ITER_LAMBDA_PSEUDO,
                'ITER_TAU_QUANTILE': ITER_TAU_QUANTILE,
                'EAD_ALPHA': EAD_ALPHA, 'LAMBDA_LAP': LAMBDA_LAP,
                'variant': 'iterative_v2'},
    )

    # =========================================================================
    # ROUND 0 — clean baseline
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

    # Track best across all rounds
    overall_best = {'round': 0, 'rmse': best_r0_rmse, 'state': best_r0_state}
    best_prev_rmse = best_r0_rmse
    best_prev_state = best_r0_state
    snapshot_train_loader = train_loader_r0   # which loader to perturb on for σ
    snapshot_n_active = n_r0
    snapshot_edge = (edge_src_r0, edge_dst_r0, edge_w_r0)
    snapshot_use_pseudo = False
    snapshot_lambda = 0.0

    # Pre-extract y_residual once for quality checks
    print("\n  Pre-extracting y_residual for quality checks...")
    all_y_residual = np.zeros((len(full_train_ds), nNodes_total), dtype=np.float32)
    for t, d in enumerate(full_train_ds):
        all_y_residual[t] = d.y_residual.squeeze(-1).cpu().numpy()
    true_eps_full = all_y_residual - alpha_per_sample[:, None] - beta_hat[None, :]

    round_results = []

    # =========================================================================
    # ROUND 1..N
    # =========================================================================
    for round_k in range(1, ITER_N_ROUNDS_MAX + 1):
        print("\n" + "#"*72)
        print(f"# ROUND {round_k} (V2: target=best_prev, warm-start, smaller perturbations)")
        print("#"*72)

        # ---- Step A: Build SCORING subgraph ----
        remaining = sorted(all_unlabeled_set - selected_so_far)
        if len(remaining) < ITER_K_PER_ROUND:
            print(f"  [STOP] only {len(remaining)} candidates left.")
            break

        scoring_nodes = sorted(labeled_idx.tolist() + remaining)
        n_scoring = len(scoring_nodes)
        scoring_train_ds = build_subgraph_dataset(
            full_train_ds, scoring_nodes, nNodes_total, alpha_per_sample, beta_hat)
        scoring_valid_ds = build_subgraph_dataset(
            full_valid_ds, scoring_nodes, nNodes_total, alpha_per_sample, beta_hat)
        scoring_valid_loader = PyGDataLoader(scoring_valid_ds, batch_size=512, shuffle=False)
        print(f"  Scoring graph: {n_scoring} nodes")

        # ---- Step B (V2): TARGET = best_prev_state's prediction (single forward) ----
        print(f"\n  [V2] Loading best_R{round_k-1} state for pseudo-label TARGET")
        gnn.load_state_dict(best_prev_state)

        # SANITY: best_prev_state should still achieve best_prev_rmse on small valid graph
        # Note: scoring_valid_ds is on a different (larger) graph — direct comparison is approximate
        em_target_on_scoring = eval_ead(scoring_valid_loader, gnn, device,
                                         n_scoring, nNodes_labeled, tgt_scl=tgt_scl_celsius)
        target_v_rmse = em_target_on_scoring['rmse_norm']
        print(f"  [V2 SANITY] best_prev_state's V RMSE on scoring graph = {target_v_rmse:.4f}  "
              f"(small-graph best was {best_prev_rmse:.4f})")
        if abs(target_v_rmse - best_prev_rmse) > 0.01:
            print(f"  [V2 ⚠] Scoring V RMSE deviates from best_prev_rmse by {abs(target_v_rmse - best_prev_rmse):.4f}; "
                  f"likely cross-graph transfer effect")

        # Generate target on scoring graph
        target_eps = predict_eps_all(gnn, scoring_train_ds, device, n_scoring)
        print(f"  [V2] target_eps shape={target_eps.shape}, "
              f"mean={target_eps.mean():+.4f}, std={target_eps.std():.4f}")

        # ---- Step C (V2): σ from small-LR perturbations ----
        # Re-load best_prev_state (since scoring graph eval may have left it in eval mode)
        gnn.load_state_dict(best_prev_state)
        # Build a small training loader for σ-perturbation: use the labeled-only data (R0's train_loader)
        # because that's the graph the best_prev_state was trained on
        sigma_snap_states = collect_sigma_snapshots(
            gnn, snapshot_train_loader, device, snapshot_n_active,
            *snapshot_edge,
            use_pseudo=snapshot_use_pseudo, lambda_pseudo=snapshot_lambda,
            m_snapshots=ITER_M_SNAPSHOTS, t_cycle=ITER_T_CYCLE,
            lr_max=ITER_LR_MAX, lr_min=ITER_LR_MIN,
            tag=f"R{round_k}")

        # Diagnostic: print each σ-snapshot's V RMSE (should all be near best_prev_rmse)
        snap_rmses = []
        for i, st in enumerate(sigma_snap_states):
            gnn.load_state_dict(st)
            em = eval_ead(valid_loader_r0, gnn, device, n_r0, nNodes_labeled,
                          tgt_scl=tgt_scl_celsius)
            snap_rmses.append(em['rmse_norm'])
        print(f"  [σ-snapshots V RMSE] {[f'{r:.4f}' for r in snap_rmses]}, "
              f"mean={np.mean(snap_rmses):.4f}, std={np.std(snap_rmses):.4f}")
        print(f"  [Compare] best_prev_state V RMSE = {best_prev_rmse:.4f}  "
              f"(σ-snapshots should be close)")
        if np.mean(snap_rmses) - best_prev_rmse > 0.005:
            print(f"  [⚠] σ-snapshots avg drifted {np.mean(snap_rmses) - best_prev_rmse:.4f} above best_prev — "
                  f"reduce ITER_LR_MAX")

        # Apply σ-snapshots on scoring graph
        snap_preds_scoring = []
        for st in sigma_snap_states:
            gnn.load_state_dict(st)
            ep_pred = predict_eps_all(gnn, scoring_train_ds, device, n_scoring)
            snap_preds_scoring.append(ep_pred)
        snap_stack = np.stack(snap_preds_scoring, axis=0)   # (M, T, n_scoring)
        ensemble_std_loc = snap_stack.std(axis=0)            # (T, n_scoring)
        conf_loc = -ensemble_std_loc.mean(axis=0)            # (n_scoring,)
        print(f"  [σ-stats on scoring] mean={ensemble_std_loc.mean():.4f}, "
              f"std={ensemble_std_loc.std():.4f}, max={ensemble_std_loc.max():.4f}")

        # ---- Map local → full coords ----
        target_eps_full = map_local_to_full(target_eps, scoring_nodes, nNodes_total,
                                             default_value=0.0, time_dim=True)
        ensemble_std_full = map_local_to_full(ensemble_std_loc, scoring_nodes, nNodes_total,
                                               default_value=0.0, time_dim=True)
        conf_full = map_local_to_full(conf_loc, scoring_nodes, nNodes_total,
                                       default_value=-1e9, time_dim=False)

        # Embeddings: use best_prev_state on scoring graph
        gnn.load_state_dict(best_prev_state)
        emb_loc = per_station_embedding(gnn, scoring_train_ds, device, n_scoring)
        emb_full = np.zeros((nNodes_total, emb_loc.shape[1]), dtype=np.float32)
        emb_full[np.array(scoring_nodes)] = emb_loc

        # ---- Step D: Quality diagnostics ----
        quality_warnings = []
        # Pseudo target quality on V (using best_prev_state's predictions, which we just took)
        target_rmse_v = float(np.sqrt(
            ((target_eps_full[:, valid_idx] - true_eps_full[:, valid_idx]) ** 2).mean(axis=0)
        ).mean())
        print(f"\n  [V2 Quality] best_prev_state's pseudo-target RMSE on V = {target_rmse_v:.4f}")
        print(f"               R0 best_rmse = {best_r0_rmse:.4f}, prev_round best = {best_prev_rmse:.4f}")
        if target_rmse_v > 1.5 * best_r0_rmse:
            quality_warnings.append(f"Pseudo target RMSE on V > 1.5× R0 best ({target_rmse_v:.4f})")

        # Confidence breakdown
        print(f"  [Confidence] train={conf_full[train_idx].mean():+.4f}, "
              f"valid={conf_full[valid_idx].mean():+.4f}, "
              f"remaining={conf_full[np.array(remaining)].mean():+.4f}")

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
            print(f"  [AUTO-STOP] picked only {len(new_picks)}/{ITER_K_PER_ROUND}.")
            if len(new_picks) > 0:
                selected_so_far.update(new_picks)
            break

        selected_so_far.update(new_picks)
        print(f"  [Selection] picked {len(new_picks)} new; |S_total| = {len(selected_so_far)}")

        # ---- Step F: Build TRAINING subgraph (labeled + selected_so_far) ----
        training_nodes = sorted(labeled_idx.tolist() + list(selected_so_far))
        n_train_round = len(training_nodes)
        select_mask_full = np.zeros(nNodes_total, dtype=bool)
        for s in selected_so_far:
            select_mask_full[s] = True

        train_ds_round = build_subgraph_dataset(
            full_train_ds, training_nodes, nNodes_total,
            alpha_per_sample, beta_hat,
            pseudo_eps_all=target_eps_full,
            pseudo_sigma_all=ensemble_std_full,
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

        # ---- Step G (V2): WARM-START from best_prev_state, train ROUND_EPOCHS ----
        print(f"\n  [V2] Warm-start from R{round_k-1} best_state (LR={ITER_WARMSTART_LR:.0e})")
        # Note: graph changed, so `best_prev_state` may produce different predictions on training graph.
        # We still load it as a starting point — encoder/processor/decoder weights transfer cleanly.
        gnn.load_state_dict(best_prev_state)

        # Initial RMSE (warm-start sanity)
        em_init = eval_ead(valid_loader_round, gnn, device, n_train_round, nNodes_labeled,
                            tgt_scl=tgt_scl_celsius)
        print(f"  [V2 SANITY] Initial valid_RMSE on training graph (after load) = {em_init['rmse_norm']:.4f}  "
              f"(warm-start should keep this low)")

        opt = torch.optim.Adam(gnn.parameters(), lr=ITER_WARMSTART_LR)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)

        best_r_rmse = em_init['rmse_norm']
        best_r_state = copy.deepcopy(gnn.state_dict())
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

        # Diagnostic: did warm-start help (vs starting from nothing)?
        delta_init = best_r_rmse - em_init['rmse_norm']
        print(f"\n[Round {round_k} done] best_valid_RMSE = {best_r_rmse:.4f} "
              f"(Δ from initial = {delta_init:+.4f})")
        with open(log_path, 'a') as f:
            f.write(f"[R{round_k} done] best={best_r_rmse:.4f}, init={em_init['rmse_norm']:.4f}, "
                    f"Δ={delta_init:+.4f}\n")

        if best_r_rmse < overall_best['rmse']:
            overall_best = {'round': round_k, 'rmse': best_r_rmse, 'state': best_r_state}

        round_results.append({
            'round': round_k,
            'best_rmse': best_r_rmse,
            'init_rmse': float(em_init['rmse_norm']),
            'n_selected_total': len(selected_so_far),
            'n_picked_this_round': len(new_picks),
            'target_rmse_v': target_rmse_v,
            'sigma_snap_avg': float(np.mean(snap_rmses)),
            'sigma_snap_std': float(np.std(snap_rmses)),
            'warnings': quality_warnings,
        })

        # ---- Update best_prev for next round (B: warm-start chain) ----
        best_prev_rmse = best_r_rmse
        best_prev_state = best_r_state
        # IMPORTANT: σ-snapshots in next round will be done on THIS round's training loader
        snapshot_train_loader = train_loader_round
        snapshot_n_active = n_train_round
        snapshot_edge = (edge_src_r, edge_dst_r, edge_w_r)
        snapshot_use_pseudo = True
        snapshot_lambda = ITER_LAMBDA_PSEUDO

        # Save artifacts
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
    print("  FINAL RESULTS (V2: best_prev_state target + warm-start)")
    print("="*72)
    print(f"  Total pseudo-labels: {len(selected_so_far)}")
    print(f"  Rounds completed:    {len(round_results)}")
    print(f"  Round 0 best:        {best_r0_rmse:.4f}")
    print(f"  Overall best:        R{overall_best['round']} = {overall_best['rmse']:.4f}")
    print(f"  Reference B1 (13204): 0.0428")
    print(f"\n  Per-round summary:")
    print(f"  {'Round':>6} {'#PL':>5} {'init':>8} {'best':>8} {'Δ':>7} {'tgt_v':>8} {'σ_avg':>8} {'σ_std':>8}")
    print(f"  {'R0':>6} {'0':>5} {'-':>8} {best_r0_rmse:>8.4f} {'-':>7} {'-':>8} {'-':>8} {'-':>8}")
    for r in round_results:
        warn = ' ⚠' if r['warnings'] else ''
        delta = r['best_rmse'] - r['init_rmse']
        print(f"  {'R'+str(r['round']):>6} {r['n_selected_total']:>5} "
              f"{r['init_rmse']:>8.4f} {r['best_rmse']:>8.4f} {delta:>+7.4f} "
              f"{r['target_rmse_v']:>8.4f} {r['sigma_snap_avg']:>8.4f} {r['sigma_snap_std']:>8.4f}{warn}")
    print("="*72)
    with open(log_path, 'a') as f:
        f.write(f"\n=== FINAL ===\n")
        f.write(f"R0_best={best_r0_rmse:.4f}, overall=R{overall_best['round']}={overall_best['rmse']:.4f}\n")
        f.write(f"Total pseudo-labels: {len(selected_so_far)}\n")
        for r in round_results:
            f.write(f"  R{r['round']}: best={r['best_rmse']:.4f}, init={r['init_rmse']:.4f}, "
                    f"|S|={r['n_selected_total']}, target_v={r['target_rmse_v']:.4f}, "
                    f"σ_avg={r['sigma_snap_avg']:.4f}\n")


if __name__ == '__main__':
    main()

"""
V1 baseline runner — extended from original_code/run.py.
All knobs are env vars so a single python entry serves 6 baselines:
  V1_VAL_MODE     : random / sequential / spatial   (default: spatial)
  V1_N_UNLABELED  : 0 (sup, default) or 400 (semi)
  V1_DATA_DIR     : path to .mat files (default: data/)
  V1_OUTPUT_DIR   : override output dir (default: log_V1/{jobid}_{method})
  V1_EPOCHS       : default 5000
  V1_BATCH        : default 512 (sup) / 128 (semi)
  V1_WANDB        : 1 (default) / 0 (disable)
  V1_TGT_SCL_C    : approximate Celsius scale to convert normalized → °C (default 30)
  V1_CONV_TYPE    : graphconv (default) / sageconv / gat — GNN 层类型
  V1_DATASET      : V1 (default,GNN_N1_StationMat) / V2 (Labeled_Finalized_new) — 数据源
  V1_EAD_ALPHA    : 0 (默认 off) / 1 — 启用时间锚 α_t (Yu et al. EAD)
  V1_EAD_BETA     : 0 (默认 off) / 1 — 启用空间锚 β_i kriged 到 valid+unlabeled
  V1_LAMBDA_LAP   : 0.0 (默认 off) / >0 — 启用 Laplacian 正则,λ × Σ w_ij(ŷ_i−ŷ_j)²
                    (与 EAD 互斥,不同时启用)

Methodology: GraphConv ×3, Adam lr=1e-3, ExponentialLR γ=0.9992,
Huber loss, no early stop, save best by valid RMSE.

All logging goes to wandb (project=V1_baseline, run_name={jobid}_{method}).
6 metrics per epoch: RMSE / MBE / MAE × normalized / Celsius.
Local files in output dir: only model checkpoint (.pt), param.pkl, hist.png.
"""
import os
import sys
from datetime import datetime
import numpy as np
import torch
import pickle

# Resolve local code_V1/ modules first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data, network, utils

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# ----------- env knobs -----------
V1_DATA_DIR    = os.environ.get('V1_DATA_DIR', '/home/hhz6461/Urban_prediction_Semi/data')
V1_VAL_MODE    = os.environ.get('V1_VAL_MODE', 'spatial').lower()
V1_N_UNLABELED = int(os.environ.get('V1_N_UNLABELED', '0'))
V1_EPOCHS      = int(os.environ.get('V1_EPOCHS', '5000'))
V1_WANDB       = int(os.environ.get('V1_WANDB', '1'))
V1_TGT_SCL_C   = float(os.environ.get('V1_TGT_SCL_C', '30.0'))
V1_CONV_TYPE   = os.environ.get('V1_CONV_TYPE', 'graphconv').lower()
assert V1_CONV_TYPE in ('graphconv', 'sageconv', 'gat'), f"bad V1_CONV_TYPE={V1_CONV_TYPE}"
V1_DATASET     = os.environ.get('V1_DATASET', 'V2').upper()    # 默认 V2(主推 dataset)
assert V1_DATASET in ('V1', 'V2'), f"bad V1_DATASET={V1_DATASET}"
V1_EAD_ALPHA   = int(os.environ.get('V1_EAD_ALPHA', '0'))      # EAD α 时间锚开关
V1_EAD_BETA    = int(os.environ.get('V1_EAD_BETA',  '0'))      # EAD β 空间锚开关
V1_EAD_BETA_MODE = os.environ.get('V1_EAD_BETA_MODE', 'kriging').lower()   # 'kriging' (V0 默认) | 'mlp' (EAD-Plus)
assert V1_EAD_BETA_MODE in ('kriging', 'mlp'), f"bad V1_EAD_BETA_MODE={V1_EAD_BETA_MODE}"
V1_LAMBDA_LAP  = float(os.environ.get('V1_LAMBDA_LAP', '0.0')) # Laplacian 正则权重(0 = off)
V1_CONSISTENCY     = int(os.environ.get('V1_CONSISTENCY', '0'))           # 1 启用 multi-view consistency(GRAND-style)
V1_CONS_K_VIEWS    = int(os.environ.get('V1_CONS_K_VIEWS', '3'))          # 多 view 数
V1_CONS_LAMBDA     = float(os.environ.get('V1_CONS_LAMBDA', '0.1'))       # consistency loss 权重
V1_ADV_MASK        = int(os.environ.get('V1_ADV_MASK', '0'))              # 1 启用 adversarial near-valid mask
V1_ADV_MASK_K      = int(os.environ.get('V1_ADV_MASK_K', '20'))           # 选 K 个离 valid 最近的 unlabeled 当 mask 候选
V1_ADV_MASK_P      = float(os.environ.get('V1_ADV_MASK_P', '0.5'))        # 每 batch mask 候选的比例
V1_DIST_ALIGN      = int(os.environ.get('V1_DIST_ALIGN', '0'))            # 1 启用 distribution alignment(MMD,半 CycleGAN)
V1_DA_LAMBDA       = float(os.environ.get('V1_DA_LAMBDA', '0.1'))         # MMD loss 权重
V1_MASK_RECON      = int(os.environ.get('V1_MASK_RECON', '0'))            # 1 启用 mask reconstruct multi-task aux
V1_MR_LAMBDA       = float(os.environ.get('V1_MR_LAMBDA', '0.1'))         # recon loss 权重
V1_MR_K            = int(os.environ.get('V1_MR_K', '30'))                 # 每 batch 随机 mask K 个 unlabeled 节点
V1_MR_TARGET       = os.environ.get('V1_MR_TARGET', 'uf').lower()         # 重建目标:uf / aux / clms / uf_geo
assert V1_MR_TARGET in ('uf', 'aux', 'clms', 'uf_geo'), f"bad V1_MR_TARGET={V1_MR_TARGET}"
# Pretrain mode(STD-MAE-style modality-aware masked feature reconstruction)
V1_PRETRAIN_MODE   = int(os.environ.get('V1_PRETRAIN_MODE', '0'))         # 1 = run pretrain stage only (no T labels used)
V1_PT_MASK_RATIO   = float(os.environ.get('V1_PT_MASK_RATIO', '0.25'))    # fraction of maskable dynamic features to mask
V1_PRETRAIN_INIT   = os.environ.get('V1_PRETRAIN_INIT', '')               # path to pretrain ckpt to load before finetune (empty = no pretrain init)
# ===== Experiment E: Temporal Laplacian on consecutive timesteps =====
V1_TEMPLAP_LAMBDA  = float(os.environ.get('V1_TEMPLAP_LAMBDA', '0.0'))    # >0 启用 temp lap on baseline
# ===== Experiment D: Dual Laplacian (spatial + feature similarity) =====
V1_FEATLAP_LAMBDA  = float(os.environ.get('V1_FEATLAP_LAMBDA', '0.0'))    # >0 启用 feature-similarity lap (与 spatial Lap 共存)
V1_FEATLAP_K       = int(os.environ.get('V1_FEATLAP_K', '10'))            # feature kNN graph k
# ===== Experiment B: Cross-Modality Prediction SSL =====
V1_CMP_SSL         = int(os.environ.get('V1_CMP_SSL', '0'))               # 1 启用 cross-modality prediction aux task (target=CLMS)
V1_CMP_LAMBDA      = float(os.environ.get('V1_CMP_LAMBDA', '0.1'))        # aux loss weight
V1_CMP_MASK_RATIO  = float(os.environ.get('V1_CMP_MASK_RATIO', '0.25'))   # 节点 mask 比例
# EAD 和 Lap 可以叠加:EAD 改 target(模型预测 ε),Lap 加约束(平滑模型输出 = 平滑 ε)
# 数学上独立维度,对应 V0 的 "residual Laplacian" 概念
V1_GEO_POOL_SIZE = int(os.environ.get('V1_GEO_POOL_SIZE', '12'))   # GeoEmbed 池化尺寸,
                                                                    # default 12 → 12*12*7=1008 维;
                                                                    # 8 → 448; 6 → 252; 4 → 112
assert V1_GEO_POOL_SIZE in (4, 6, 8, 10, 12), f"bad V1_GEO_POOL_SIZE={V1_GEO_POOL_SIZE}"
V1_GEO_DIM_REDUCE = os.environ.get('V1_GEO_DIM_REDUCE', 'pool').lower()   # 'pool' (default,空间池化) | 'pca' (data-driven 主成分)
assert V1_GEO_DIM_REDUCE in ('pool', 'pca'), f"bad V1_GEO_DIM_REDUCE={V1_GEO_DIM_REDUCE}"
V1_GEO_PCA_DIM    = int(os.environ.get('V1_GEO_PCA_DIM', '256'))   # PCA 时保留的主成分数(only used if dim_reduce=pca)
V1_ENCODER_TYPE   = os.environ.get('V1_ENCODER_TYPE', 'flat').lower()  # 'flat' | 'per_modality' | 'modaware'(2D CNN on GeoEmb,Experiment A)
assert V1_ENCODER_TYPE in ('flat', 'per_modality', 'modaware'), f"bad V1_ENCODER_TYPE={V1_ENCODER_TYPE}"
V1_MODAL_HID      = int(os.environ.get('V1_MODAL_HID', '32'))     # per-modality 每模态隐藏维(only used if encoder_type=per_modality)
# ===== Self-Train env (default off) =====
V1_SELFTRAIN          = int(os.environ.get('V1_SELFTRAIN', '0'))
V1_ST_PSEUDO_SOURCE   = os.environ.get('V1_ST_PSEUDO_SOURCE', 'self').lower()      # 'self' | 'kriging' | 'hybrid'
V1_ST_CONFIDENCE      = os.environ.get('V1_ST_CONFIDENCE', 'neighbor').lower()      # 'neighbor' | 'conformal' | 'kriging_struct'
V1_ST_K_PER_ROUND     = int(os.environ.get('V1_ST_K_PER_ROUND', '40'))
V1_ST_N_ROUNDS        = int(os.environ.get('V1_ST_N_ROUNDS', '5'))
V1_ST_ROUND_EPOCHS    = int(os.environ.get('V1_ST_ROUND_EPOCHS', '200'))
V1_ST_WARMSTART_LR    = float(os.environ.get('V1_ST_WARMSTART_LR', '1e-4'))
V1_ST_LAMBDA_PSEUDO   = float(os.environ.get('V1_ST_LAMBDA_PSEUDO', '0.5'))
V1_ST_TAU_QUANTILE    = float(os.environ.get('V1_ST_TAU_QUANTILE', '0.5'))
V1_ST_ALPHA_DIV       = float(os.environ.get('V1_ST_ALPHA_DIV', '1.0'))
V1_ST_BETA_REL        = float(os.environ.get('V1_ST_BETA_REL', '1.0'))
V1_ST_HYBRID_ALPHA    = float(os.environ.get('V1_ST_HYBRID_ALPHA', '0.5'))   # for hybrid: pseudo = α × self + (1-α) × kriging
V1_ST_R0_CKPT         = os.environ.get('V1_ST_R0_CKPT',
    '/home/hhz6461/Urban_prediction_Semi/log_V1/13860_V2_semi_spatial/V2_semi_spatial.pt')
if V1_SELFTRAIN:
    assert V1_ST_PSEUDO_SOURCE in ('self', 'kriging', 'hybrid'), f"bad V1_ST_PSEUDO_SOURCE={V1_ST_PSEUDO_SOURCE}"
    assert V1_ST_CONFIDENCE in ('neighbor', 'conformal', 'kriging_struct'), \
        f"bad V1_ST_CONFIDENCE={V1_ST_CONFIDENCE}"
job_id         = os.environ.get('SLURM_JOB_ID', 'local')

method_tag = 'sup' if V1_N_UNLABELED == 0 else 'semi'
# method_full 加 conv_type 后缀(若非默认 graphconv);保持 .sh 文件名一致
conv_suffix = '' if V1_CONV_TYPE == 'graphconv' else f'_{V1_CONV_TYPE}'
# EAD 后缀:_eadA / _eadAB / _eadB,默认 baseline 无后缀
if V1_EAD_ALPHA and V1_EAD_BETA:
    ead_suffix = '_eadABplus' if V1_EAD_BETA_MODE == 'mlp' else '_eadAB'
elif V1_EAD_ALPHA:
    ead_suffix = '_eadA'
elif V1_EAD_BETA:
    ead_suffix = '_eadBplus' if V1_EAD_BETA_MODE == 'mlp' else '_eadB'
else:
    ead_suffix = ''
# Laplacian 后缀:_lap(若启用,与 EAD 互斥)
lap_suffix = '_lap' if V1_LAMBDA_LAP > 0 else ''
# Consistency 后缀(独立)
cons_suffix = '_cons' if V1_CONSISTENCY else ''
# Adversarial mask 后缀
advmask_suffix = '_advmask' if V1_ADV_MASK else ''
# Distribution alignment 后缀
da_suffix = '_da' if V1_DIST_ALIGN else ''
# Mask reconstruct 后缀:_mr 或 _mr_{target} 若 target ≠ 默认 uf
if V1_MASK_RECON:
    mr_suffix = '_mr' if V1_MR_TARGET == 'uf' else f'_mr_{V1_MR_TARGET}'
else:
    mr_suffix = ''
# Pretrain 后缀: _pretrain (stage 1) 或 _ftpt (finetune-from-pretrain stage 2)
if V1_PRETRAIN_MODE:
    pt_suffix = '_pretrain'
elif V1_PRETRAIN_INIT:
    pt_suffix = '_ftpt'
else:
    pt_suffix = ''
# Experiment E/D/B 后缀
templap_suffix = '_templap' if V1_TEMPLAP_LAMBDA > 0 else ''
featlap_suffix = '_featlap' if V1_FEATLAP_LAMBDA > 0 else ''
cmp_suffix     = '_cmp' if V1_CMP_SSL else ''
# Modaware encoder suffix(Experiment A)
if V1_ENCODER_TYPE == 'modaware':
    enc_a_suffix = '_modaware'
else:
    enc_a_suffix = ''
# GeoEmbed 后缀:
#   pool 模式 + 默认 pool=12 → 无后缀
#   pool 模式 + pool != 12   → _g{n}
#   pca 模式                 → _pca{dim}(覆盖 pool 后缀,因为 PCA 时 pool=12 也只是中间步骤)
if V1_GEO_DIM_REDUCE == 'pca':
    geo_suffix = f'_pca{V1_GEO_PCA_DIM}'
elif V1_GEO_POOL_SIZE != 12:
    geo_suffix = f'_g{V1_GEO_POOL_SIZE}'
else:
    geo_suffix = ''
# Encoder 后缀:non-default 时显示
# Encoder 后缀:per_modality 时含 modal_hid(若非默认 32)
if V1_ENCODER_TYPE == 'per_modality':
    enc_suffix = f'_pmod{V1_MODAL_HID}' if V1_MODAL_HID != 32 else '_pmod'
else:
    enc_suffix = ''
# Self-train 后缀:_st_{pseudo}_{conf} (短)
if V1_SELFTRAIN:
    _ps_short = {'self': 'self', 'kriging': 'krig', 'hybrid': 'hyb'}[V1_ST_PSEUDO_SOURCE]
    _cf_short = {'neighbor': 'neigh', 'conformal': 'conf', 'kriging_struct': 'kstruct'}[V1_ST_CONFIDENCE]
    st_suffix = f'_st_{_ps_short}_{_cf_short}'
else:
    st_suffix = ''
ds_prefix   = V1_DATASET   # V1 / V2
method_full = f'{ds_prefix}_{method_tag}_{V1_VAL_MODE}{conv_suffix}{ead_suffix}{lap_suffix}{cons_suffix}{advmask_suffix}{da_suffix}{mr_suffix}{pt_suffix}{templap_suffix}{featlap_suffix}{cmp_suffix}{enc_a_suffix}{geo_suffix}{enc_suffix}{st_suffix}'
run_name = f'{job_id}_{method_full}'

default_out = f'/home/hhz6461/Urban_prediction_Semi/log_V1/{run_name}'
V1_OUTPUT_DIR = os.environ.get('V1_OUTPUT_DIR', default_out)
os.makedirs(V1_OUTPUT_DIR, exist_ok=True)
os.environ['V1_OUTPUT_DIR'] = V1_OUTPUT_DIR     # for data.py spatial split viz
os.chdir(V1_OUTPUT_DIR)
print(f"[run.py] output_dir = {V1_OUTPUT_DIR}")
print(f"[run.py] data_dir   = {V1_DATA_DIR}")
print(f"[run.py] run_name   = {run_name}")
print(f"[run.py] val_mode   = {V1_VAL_MODE}, n_unlabeled = {V1_N_UNLABELED}, "
      f"epochs = {V1_EPOCHS}, tgt_scl_C = {V1_TGT_SCL_C}")


# 主训练入口:读 env vars → 构数据 → 建模型 → wandb init → 5000 epoch 训练 + log 6 metrics
def main():
    """主训练入口。

    1) 用 env 决定的参数构造 dataParam,调 `data.dataGen()` 拿 train/valid loader;
    2) 构造 modelParam,创建 GNN 模型 + Adam + ExponentialLR + HuberLoss;
    3) `wandb.init()`(若 V1_WANDB=1);
    4) 训练 V1_EPOCHS 轮:每轮 train + test,算 6 metrics(RMSE/MBE/MAE × norm/Celsius),
       wandb log,best 时存 checkpoint;
    5) 结束 wandb summary。
    """
    # ------- Dataset -------
    dataParam = {
        'geoMethod':   'average',
        'nCompPCA':    40,
        'window':      2,
        'poolSize':    V1_GEO_POOL_SIZE,    # env-controlled,GeoEmbed 维度 = 7×poolSize²
        'batchSize':   int(os.environ.get('V1_BATCH', '512' if V1_N_UNLABELED == 0 else '128')),
        'thres':       0.1,
        'geoFeatures': 'full',
    }
    if V1_GEO_DIM_REDUCE == 'pca':
        print(f"[run.py] GeoEmbed dim reduce = PCA {V1_GEO_PCA_DIM} (pool 12→1008→PCA→{V1_GEO_PCA_DIM})")
    else:
        print(f"[run.py] GeoEmbed pool size = {V1_GEO_POOL_SIZE} → GeoEmbed dim = {7 * V1_GEO_POOL_SIZE**2}")
    trainLoader, validLoader, metadata, _ = data.dataGen(dataParam, V1_DATA_DIR)
    # V2 模式有 metadata['tgt_scl'](精确 °C 反归一化标度);V1 模式 fallback 到 env V1_TGT_SCL_C
    tgt_scl_C = float(metadata.get('tgt_scl', V1_TGT_SCL_C))
    print(f"[run.py] tgt_scl_C resolved = {tgt_scl_C:.4f} (source={'metadata' if 'tgt_scl' in metadata else 'env V1_TGT_SCL_C'})")

    # ------- Model -------
    modelParam = {
        'HLD':       128,
        'nMLP':      2,
        'nGNN':      3,
        'nGAT':      1,
        'nHeads':    1,
        'K':         1,
        'iDim':      metadata['iDim'],
        'oDim':      metadata['oDim'],
        'BN':        False,
        'Dropout':   False,
        'conv_type': V1_CONV_TYPE,
        'encoder_type': V1_ENCODER_TYPE,
        'modal_hid': V1_MODAL_HID,    # ModalEncoder 每模态的隐藏维(only used if encoder_type=per_modality)
    }
    modelName = method_full
    model = network.GNN(modelParam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    lossFn = torch.nn.HuberLoss().to(device)

    EPOCH, bestLoss, chkptPath, hist = network.loadCheckPoint(modelName, model, opt, device, load=False)

    # ---- Load pretrained encoder weights if specified (Stage 2 of two-stage pipeline) ----
    if V1_PRETRAIN_INIT and not V1_PRETRAIN_MODE:
        assert os.path.exists(V1_PRETRAIN_INIT), f"[ERR/PT] pretrain ckpt not found: {V1_PRETRAIN_INIT}"
        ckpt_pt = torch.load(V1_PRETRAIN_INIT, map_location=device, weights_only=False)
        state_pt = ckpt_pt.get('model_state_dict', ckpt_pt)
        # Load only encoder + processor weights (decoder + recon_heads ignored)
        own_state = model.state_dict()
        loaded_keys = []
        for k, v in state_pt.items():
            if (k.startswith('encoder.') or k.startswith('processor.')) and k in own_state:
                own_state[k] = v
                loaded_keys.append(k)
        model.load_state_dict(own_state)
        print(f"[run.py] Loaded {len(loaded_keys)} pretrained encoder/processor weights from {V1_PRETRAIN_INIT}")

    # ---- wandb init (all logging goes here) ----
    wandb_run = None
    if V1_WANDB:
        try:
            import wandb
            # wandb project:V1 dataset → V1_baseline(legacy);V2 dataset → urban
            wandb_project = 'V1_baseline' if V1_DATASET == 'V1' else 'urban'
            wandb_run = wandb.init(
                entity="urban_prediction",
                project=wandb_project,
                name=run_name,
                config={
                    'job_id':         job_id,
                    'method':         method_full,
                    'val_mode':       V1_VAL_MODE,
                    'n_unlabeled':    V1_N_UNLABELED,
                    'iDim':           modelParam['iDim'],
                    'nNodes':         metadata['nNodes'],
                    'nNodes_labeled': metadata.get('nNodes_labeled', '-'),
                    'batch_size':     dataParam['batchSize'],
                    'epochs':         V1_EPOCHS,
                    'lr':             1e-3,
                    'scheduler_gamma': 0.9992,
                    'loss':           'HuberLoss',
                    'optimizer':      'Adam',
                    'graph_type':     ('V1_AJM' if V1_DATASET == 'V1' else 'V2_simgraph') if V1_N_UNLABELED == 0 else 'kNN_k10',
                    'tgt_scl_C':      tgt_scl_C,    # 实际用的(V2 来自 metadata,V1 来自 env)
                    'conv_type':      V1_CONV_TYPE,
                    'dataset':        V1_DATASET,
                    'ead_alpha':      bool(V1_EAD_ALPHA),
                    'ead_beta':       bool(V1_EAD_BETA),
                    'ead_active':     bool(V1_EAD_ALPHA or V1_EAD_BETA),
                    'lambda_lap':     V1_LAMBDA_LAP,
                    'lap_active':     V1_LAMBDA_LAP > 0,
                    'geo_pool_size':  V1_GEO_POOL_SIZE,
                    'geo_dim_reduce': V1_GEO_DIM_REDUCE,
                    'geo_pca_dim':    V1_GEO_PCA_DIM if V1_GEO_DIM_REDUCE == 'pca' else None,
                    'adv_mask':          bool(V1_ADV_MASK),
                    'adv_mask_K':        V1_ADV_MASK_K if V1_ADV_MASK else None,
                    'adv_mask_p':        V1_ADV_MASK_P if V1_ADV_MASK else None,
                    'consistency':       bool(V1_CONSISTENCY),
                    'cons_k_views':      V1_CONS_K_VIEWS if V1_CONSISTENCY else None,
                    'cons_lambda':       V1_CONS_LAMBDA if V1_CONSISTENCY else None,
                    'ead_beta_mode':     V1_EAD_BETA_MODE if V1_EAD_BETA else None,
                    'self_train':     bool(V1_SELFTRAIN),
                    'st_pseudo_source': V1_ST_PSEUDO_SOURCE if V1_SELFTRAIN else None,
                    'st_confidence':    V1_ST_CONFIDENCE if V1_SELFTRAIN else None,
                    'st_k_per_round':   V1_ST_K_PER_ROUND if V1_SELFTRAIN else None,
                    'st_n_rounds':      V1_ST_N_ROUNDS if V1_SELFTRAIN else None,
                    'st_round_epochs':  V1_ST_ROUND_EPOCHS if V1_SELFTRAIN else None,
                    'st_lambda_pseudo': V1_ST_LAMBDA_PSEUDO if V1_SELFTRAIN else None,
                    'st_warmstart_lr':  V1_ST_WARMSTART_LR if V1_SELFTRAIN else None,
                },
            )
            print(f"[run.py] wandb run: {wandb_run.url}")
        except Exception as e:
            print(f"[run.py] wandb init failed: {e}; continuing without wandb")
            wandb_run = None

    # Save metadata for later inspection
    with open(f'./{modelName}_param.pkl', 'wb') as f:
        pickle.dump(modelParam, f)
        pickle.dump(dataParam, f)
        pickle.dump(metadata, f)

    # ---- 决定 train/test 函数:EAD / Lap 正交可叠加 ----
    ead_active = bool(V1_EAD_ALPHA or V1_EAD_BETA)
    lap_active = V1_LAMBDA_LAP > 0

    # 预算 Lap 用的图边列表(若启用)
    if lap_active:
        Adj = metadata['AdjMatrix']
        _e_src, _e_dst = np.nonzero(Adj)
        edge_w_t = torch.FloatTensor(Adj[_e_src, _e_dst]).to(device)
        edge_src_t = torch.LongTensor(_e_src).to(device)
        edge_dst_t = torch.LongTensor(_e_dst).to(device)
        n_edges_total = len(_e_src)

    if ead_active and lap_active:
        # EAD + Lap:模型输出 ε,Lap 平滑 ε(等价 V0 residual Laplacian)
        print(f"[run.py] EAD+Lap active(α={V1_EAD_ALPHA}, β={V1_EAD_BETA}, λ_lap={V1_LAMBDA_LAP}),"
              f" 平滑 ε,n_edges={n_edges_total}")
        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return network.train_ead_lap(loader, model, lossFn, opt, scheduler, device, nNodes,
                                         edge_src_t, edge_dst_t, edge_w_t, V1_LAMBDA_LAP)
        test_fn = network.test_ead    # 验证不加 Lap
    elif ead_active:
        train_fn = network.train_ead
        test_fn  = network.test_ead
        print(f"[run.py] EAD active(α={V1_EAD_ALPHA}, β={V1_EAD_BETA}),"
              f"using train_ead/test_ead")
    elif lap_active:
        print(f"[run.py] Lap active(λ={V1_LAMBDA_LAP}),"
              f"using train_lap/test (n_edges={n_edges_total})")
        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return network.train_lap(loader, model, lossFn, opt, scheduler, device, nNodes,
                                     edge_src_t, edge_dst_t, edge_w_t, V1_LAMBDA_LAP)
        test_fn = network.test
    else:
        train_fn = network.train
        test_fn  = network.test
        print(f"[run.py] baseline mode (no EAD, no Lap),using vanilla train/test")

    # === Distribution alignment override(独立 dispatch,加 MMD loss)===
    # 与 EAD/Lap/Consistency/AdvMask 互斥(都改 train_fn);可与 self-train 叠(self-train 走外层)
    if V1_DIST_ALIGN:
        if ead_active or lap_active or V1_CONSISTENCY or V1_ADV_MASK:
            print(f"[WARN/run.py] V1_DIST_ALIGN=1 但 EAD/Lap/Consistency/AdvMask 也启用 → "
                  f"dist_align 优先,其它忽略")
        import dist_align
        n_labeled_meta = metadata.get('nNodes_labeled', None)
        assert n_labeled_meta is not None, "[ERR/DA] need metadata['nNodes_labeled']"
        print(f"[run.py] Distribution Alignment active(λ_mmd={V1_DA_LAMBDA}, n_labeled={n_labeled_meta}),"
              f"using train_dist_align")
        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return dist_align.train_dist_align(loader, model, lossFn, opt, scheduler, device, nNodes,
                                                n_labeled=n_labeled_meta,
                                                lambda_mmd=V1_DA_LAMBDA)
        test_fn = network.test    # 验证不加 MMD

    # === Adversarial near-valid mask override(独立于 EAD/Lap,如启用替换 train_fn)===
    # 注意:目前 adv_mask 只支持 baseline 路径(无 EAD/Lap),与 consistency 互斥(都改 train_fn)
    if V1_ADV_MASK:
        if ead_active or lap_active or V1_CONSISTENCY:
            print(f"[WARN/run.py] V1_ADV_MASK=1 但 EAD/Lap/Consistency 也启用 → adv_mask 优先,其它忽略")
        import adv_mask as advmask_mod
        # 一次性算 mask 候选(用 R0 / 当前模型的 emb)
        valid_idx = metadata.get('valid_station_idx', [])
        assert len(valid_idx) > 0, "[ERR/AdvMask] need metadata['valid_station_idx']"
        mask_idx = advmask_mod.compute_near_valid_mask_idx(
            model, trainLoader, device, metadata['nNodes'], valid_idx,
            K=V1_ADV_MASK_K,
            n_labeled=metadata.get('nNodes_labeled', None))
        print(f"[run.py] AdvMask active(K={V1_ADV_MASK_K}, p={V1_ADV_MASK_P}),"
              f"mask_idx 算好({len(mask_idx)} 个 unlabeled 站)")
        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return advmask_mod.train_advmask(loader, model, lossFn, opt, scheduler, device, nNodes,
                                              mask_idx=mask_idx,
                                              p_mask_per_batch=V1_ADV_MASK_P)
        test_fn = network.test    # 验证不 mask,直接 baseline test

    # === Consistency override(独立于 EAD/Lap,如果启用就替换 train_fn)===
    # 注意:目前 consistency 只支持 baseline 路径(无 EAD/Lap)。
    # 与 EAD/Lap 叠加需要 train_consistency_ead 等组合函数,后续再加。
    if V1_CONSISTENCY:
        if ead_active or lap_active:
            print(f"[WARN/run.py] V1_CONSISTENCY=1 但 EAD / Lap 也启用 → 当前 consistency 只支持 baseline,"
                  f"忽略 EAD/Lap 走 consistency 分支")
        import consistency
        print(f"[run.py] Consistency active(K_views={V1_CONS_K_VIEWS}, λ_cons={V1_CONS_LAMBDA}),"
              f"using train_consistency/test")
        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return consistency.train_consistency(loader, model, lossFn, opt, scheduler, device, nNodes,
                                                  K_views=V1_CONS_K_VIEWS,
                                                  lambda_cons=V1_CONS_LAMBDA)
        test_fn = network.test  # 验证不加 consistency,直接 baseline test

    # === Mask Reconstruct override(LAST,multi-task aux head)===
    # 与 EAD/Lap/Consistency/AdvMask/DA 互斥(都改 train_fn);本身就是给 unlabeled 加监督信号
    # 重建 features 而不预测 T → 不需要 EAD α/β
    if V1_MASK_RECON:
        if ead_active or lap_active or V1_CONSISTENCY or V1_ADV_MASK or V1_DIST_ALIGN:
            print(f"[WARN/run.py] V1_MASK_RECON=1 但其它 train_fn override 也启用 → "
                  f"mask_recon 优先,其它忽略")
        import mask_recon as mr_mod
        n_labeled_meta = metadata.get('nNodes_labeled', None)
        assert n_labeled_meta is not None, "[ERR/MR] need metadata['nNodes_labeled']"
        # 算 recon slice(V2 schema:UF [322,339))
        geo_dim_actual = (7 * V1_GEO_POOL_SIZE**2) if V1_GEO_DIM_REDUCE == 'pool' else V1_GEO_PCA_DIM
        f_start, f_end = mr_mod.get_recon_feat_slice(V1_MR_TARGET, geo_dim=geo_dim_actual)
        recon_dim = f_end - f_start
        # 创建 ReconHead,加入 opt(共享 lr/scheduler)
        recon_head = mr_mod.ReconHead(hidden_dim=modelParam['HLD'], recon_dim=recon_dim).to(device)
        opt.add_param_group({'params': recon_head.parameters()})
        print(f"[run.py] Mask Reconstruct active(λ={V1_MR_LAMBDA}, K={V1_MR_K}, "
              f"target={V1_MR_TARGET} → slice[{f_start},{f_end})={recon_dim}d, "
              f"n_labeled={n_labeled_meta}),using train_mask_recon")
        n_total_meta = metadata['nNodes']
        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return mr_mod.train_mask_recon(loader, model, recon_head, lossFn, opt, scheduler, device,
                                            n_total=n_total_meta, n_labeled=n_labeled_meta,
                                            lambda_recon=V1_MR_LAMBDA,
                                            mask_K=V1_MR_K,
                                            recon_feat_slice=(f_start, f_end))
        test_fn = network.test    # 验证不 mask,baseline test

    # =============================================================
    # Experiment E: Temporal Laplacian on baseline (V1_TEMPLAP_LAMBDA > 0)
    # Forces trainLoader shuffle=False so adjacent batch samples are consecutive in time
    # =============================================================
    if V1_TEMPLAP_LAMBDA > 0 and not V1_PRETRAIN_MODE:
        if ead_active or lap_active or V1_CONSISTENCY or V1_ADV_MASK or V1_DIST_ALIGN or V1_MASK_RECON:
            print(f"[WARN/run.py] V1_TEMPLAP_LAMBDA>0 但其它 train_fn override 也启用 → templap 优先")
        # 重建 trainLoader 强制 shuffle=False(让 batch 内样本时序相邻)
        from torch_geometric.loader import DataLoader as PyGDataLoader
        _trainSet = trainLoader.dataset
        _bs = dataParam['batchSize']
        trainLoader = PyGDataLoader(_trainSet, batch_size=_bs, shuffle=False)
        print(f"[run.py] Temporal Lap active(λ_temp={V1_TEMPLAP_LAMBDA}),"
              f"forced shuffle=False on trainLoader for temporal adjacency")
        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return network.train_templap(loader, model, lossFn, opt, scheduler, device, nNodes,
                                          lambda_temp=V1_TEMPLAP_LAMBDA)
        test_fn = network.test

    # =============================================================
    # Experiment D: Dual Laplacian (spatial + feature similarity)
    # V1_FEATLAP_LAMBDA > 0 启用;与 V1_LAMBDA_LAP > 0 (spatial) 共存
    # =============================================================
    if V1_FEATLAP_LAMBDA > 0 and not V1_PRETRAIN_MODE and not V1_TEMPLAP_LAMBDA > 0:
        if V1_CONSISTENCY or V1_ADV_MASK or V1_DIST_ALIGN or V1_MASK_RECON:
            print(f"[WARN/run.py] V1_FEATLAP_LAMBDA>0 但其它 train_fn override 也启用 → featlap 优先")
        # 构建 feature similarity kNN graph(在 raw 1347-D static features 上做)
        # 用第一个 timestep 的所有 458 节点 features 当 "代表性 features"(static + WRF@t=0)
        # 注:严格说应该用 time-averaged features,但 first-timestep 已足够区分 station identity
        _first_sample = trainLoader.dataset[0]
        _features_all = _first_sample.x.cpu().numpy()      # (n_total, 1347)
        print(f"[run.py] Building feature-sim kNN graph: features shape={_features_all.shape}, "
              f"k={V1_FEATLAP_K}")
        FeatAdj = data.build_feature_sim_knn_adj(_features_all, k=V1_FEATLAP_K)
        _ef_src, _ef_dst = np.nonzero(FeatAdj)
        edge_w_f_t = torch.FloatTensor(FeatAdj[_ef_src, _ef_dst]).to(device)
        edge_src_f_t = torch.LongTensor(_ef_src).to(device)
        edge_dst_f_t = torch.LongTensor(_ef_dst).to(device)
        n_edges_f = len(_ef_src)

        # Spatial Lap edges(已有)
        if lap_active:
            edge_src_s_t = edge_src_t
            edge_dst_s_t = edge_dst_t
            edge_w_s_t   = edge_w_t
            _lambda_s    = V1_LAMBDA_LAP
        else:
            # baseline 上 (无 spatial Lap): 用 spatial graph 但 λ_s=0
            _Adj_s = metadata['AdjMatrix']
            _es_src, _es_dst = np.nonzero(_Adj_s)
            edge_src_s_t = torch.LongTensor(_es_src).to(device)
            edge_dst_s_t = torch.LongTensor(_es_dst).to(device)
            edge_w_s_t   = torch.FloatTensor(_Adj_s[_es_src, _es_dst]).to(device)
            _lambda_s    = 0.0

        print(f"[run.py] Dual Lap active(λ_s={_lambda_s}, λ_f={V1_FEATLAP_LAMBDA}),"
              f"n_edges spatial={edge_src_s_t.shape[0]}, n_edges feature={n_edges_f}")

        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return network.train_duallap(loader, model, lossFn, opt, scheduler, device, nNodes,
                                          edge_src_s_t, edge_dst_s_t, edge_w_s_t,
                                          edge_src_f_t, edge_dst_f_t, edge_w_f_t,
                                          lambda_s=_lambda_s,
                                          lambda_f=V1_FEATLAP_LAMBDA)
        test_fn = network.test

    # =============================================================
    # Experiment B: Cross-Modality Prediction SSL (target=CLMS)
    # V1_CMP_SSL=1 启用;multi-task with aux CLMS reconstruction head
    # =============================================================
    if V1_CMP_SSL and not V1_PRETRAIN_MODE and not V1_TEMPLAP_LAMBDA > 0 and not V1_FEATLAP_LAMBDA > 0:
        if ead_active or lap_active or V1_CONSISTENCY or V1_ADV_MASK or V1_DIST_ALIGN or V1_MASK_RECON:
            print(f"[WARN/run.py] V1_CMP_SSL=1 但其它 train_fn override 也启用 → CMP 优先")
        import cmp_ssl
        cmp_head = cmp_ssl.CMPHead(hidden_dim=modelParam['HLD'], recon_dim=cmp_ssl.CLMS_DIM).to(device)
        opt.add_param_group({'params': cmp_head.parameters()})
        print(f"[run.py] CMP-SSL active(λ={V1_CMP_LAMBDA}, mask_ratio={V1_CMP_MASK_RATIO}, "
              f"target=CLMS[{cmp_ssl.CLMS_START},{cmp_ssl.CLMS_END})={cmp_ssl.CLMS_DIM}d)")
        _n_total_meta = metadata['nNodes']
        def train_fn(loader, model, lossFn, opt, scheduler, device, nNodes):
            return cmp_ssl.train_cmp_ssl(loader, model, cmp_head, lossFn, opt, scheduler, device,
                                          n_total=_n_total_meta,
                                          lambda_cmp=V1_CMP_LAMBDA,
                                          mask_ratio=V1_CMP_MASK_RATIO)
        test_fn = network.test    # 验证 normal baseline test(no mask)

    # ---- Local log file (faithful original_code 风格,每 epoch 1 行)----
    log_path = f'./{modelName}_log'
    with open(log_path, 'w') as f:
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()), file=f)
        print(f"=== {method_full} job={job_id} dataset={V1_DATASET} conv={V1_CONV_TYPE} "
              f"ead=α{V1_EAD_ALPHA}β{V1_EAD_BETA} λ_lap={V1_LAMBDA_LAP} ===", file=f)
        print(f"iDim={modelParam['iDim']}, nNodes={metadata['nNodes']} "
              f"({metadata.get('nNodes_labeled', '-')} labeled + "
              f"{metadata.get('nNodes_unlabeled', 0)} unlabeled)", file=f)
        print(f"tgt_scl_C={tgt_scl_C:.4f}", file=f)
        print(model, file=f)

    # =============================================================
    # Pretrain dispatch (env: V1_PRETRAIN_MODE=1) —— Stage 1 only
    # Saves encoder+processor weights to ckpt for later finetune.
    # No T labels used; only modality-aware masked feature reconstruction.
    # =============================================================
    if V1_PRETRAIN_MODE:
        import pretrain
        print(f"\n[run.py] === PRETRAIN MODE (Stage 1: masked feature reconstruction) ===")
        print(f"[run.py] mask_ratio = {V1_PT_MASK_RATIO}")
        recon_heads = {
            'wrf_dyn': pretrain.ReconHead(hidden_dim=modelParam['HLD'], recon_dim=270).to(device),
            'clms':    pretrain.ReconHead(hidden_dim=modelParam['HLD'], recon_dim=3).to(device),
        }
        # Optimizer includes recon heads
        all_params = list(model.parameters()) + list(recon_heads['wrf_dyn'].parameters()) + \
                     list(recon_heads['clms'].parameters())
        opt = torch.optim.Adam(all_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
        n_params_pt = sum(p.numel() for p in all_params)
        print(f"[run.py] pretrain total params: {n_params_pt} (model + recon heads)")

        log_path = f'./{modelName}_log'
        with open(log_path, 'w') as f:
            print(f"=== {method_full} job={job_id} PRETRAIN mode ===", file=f)
            print(f"mask_ratio={V1_PT_MASK_RATIO}, n_total={metadata['nNodes']}", file=f)

        best_pt_loss = float('inf')
        for epoch in range(EPOCH, V1_EPOCHS):
            pt_loss, _, _, _ = pretrain.train_pretrain(
                trainLoader, model, recon_heads, opt, scheduler, device,
                metadata['nNodes'], mask_ratio=V1_PT_MASK_RATIO)

            if epoch < 5 or epoch % 100 == 0:
                print(f"[PT ep {epoch:4d}] loss={pt_loss:.4e}, lr={scheduler.get_last_lr()[0]:.6e}")

            with open(log_path, 'a') as f:
                print(f"Epoch {epoch}: pretrain_loss={pt_loss:.4e}; LR={scheduler.get_last_lr()[0]:.6e}", file=f)

            if wandb_run is not None:
                wandb_run.log({'epoch': epoch, 'pretrain/loss': pt_loss,
                                'lr': scheduler.get_last_lr()[0]})

            if pt_loss < best_pt_loss:
                best_pt_loss = pt_loss
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'recon_head_wrf':   recon_heads['wrf_dyn'].state_dict(),
                    'recon_head_clms':  recon_heads['clms'].state_dict(),
                    'pretrain_loss':    pt_loss,
                }, chkptPath)
                with open(log_path, 'a') as f:
                    print(f"Pretrain ckpt saved. best_pretrain_loss = {pt_loss:.6e}", file=f)

        print(f"\n[FINAL] {method_full}: best pretrain loss = {best_pt_loss:.6e}")
        print(f"[FINAL] checkpoint saved to: {chkptPath}")
        if wandb_run is not None:
            wandb_run.summary['final_best_pretrain_loss'] = best_pt_loss
            wandb_run.finish()
        return    # Pretrain stage finished; do NOT enter standard sup training

    # =============================================================
    # Self-Train dispatch (env: V1_SELFTRAIN=1) —— 接管训练循环
    # =============================================================
    if V1_SELFTRAIN:
        import selftrain
        print(f"\n[run.py] === Self-Train MODE ===")
        print(f"[run.py] pseudo_source={V1_ST_PSEUDO_SOURCE}, confidence={V1_ST_CONFIDENCE}")
        print(f"[run.py] R0 ckpt = {V1_ST_R0_CKPT}")
        # Load R0 checkpoint —— 兼容多种格式
        if os.path.exists(V1_ST_R0_CKPT):
            ckpt = torch.load(V1_ST_R0_CKPT, map_location=device, weights_only=False)
            if isinstance(ckpt, dict):
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    print(f"[run.py] R0 ckpt loaded from key 'model_state_dict' "
                          f"(saved at epoch={ckpt.get('epoch', '?')}, bestLoss={ckpt.get('bestLoss', '?')})")
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                    print(f"[run.py] R0 ckpt loaded from key 'state_dict'")
                else:
                    # raw state_dict
                    model.load_state_dict(ckpt)
                    print(f"[run.py] R0 ckpt loaded as raw state_dict")
            else:
                raise ValueError(f"unsupported ckpt format: {type(ckpt)}")
        else:
            print(f"[WARN/run.py] R0 ckpt NOT FOUND at {V1_ST_R0_CKPT}, "
                  f"will start from random init (NOT recommended)")
        # st_config dict
        st_config = dict(
            pseudo_source     = V1_ST_PSEUDO_SOURCE,
            confidence_type   = V1_ST_CONFIDENCE,
            n_rounds          = V1_ST_N_ROUNDS,
            k_per_round       = V1_ST_K_PER_ROUND,
            round_epochs      = V1_ST_ROUND_EPOCHS,
            warmstart_lr      = V1_ST_WARMSTART_LR,
            lambda_pseudo     = V1_ST_LAMBDA_PSEUDO,
            tau_quantile      = V1_ST_TAU_QUANTILE,
            alpha_div         = V1_ST_ALPHA_DIV,
            beta_rel          = V1_ST_BETA_REL,
            hybrid_alpha_self = V1_ST_HYBRID_ALPHA,
        )
        # Open log file
        log_path = f'./{modelName}_log'
        with open(log_path, 'w') as f:
            print(f"=== {method_full} job={job_id} self-train mode ===", file=f)
            print(f"R0 ckpt: {V1_ST_R0_CKPT}", file=f)
            print(f"config: {st_config}", file=f)
        log_f = open(log_path, 'a')
        # Get trainSet / validSet from loader
        trainSet = trainLoader.dataset
        # validSet 之前已经从 dataGen 返回(第 4 个返回值)
        # 但当前 main() 没存它,我们再调一次也不行;直接用 validLoader.dataset
        validSet_local = validLoader.dataset
        # Run self-train main
        best_v_rmse, history = selftrain.selftrain_main(
            model=model, lossFn=lossFn,
            trainLoader=trainLoader, validLoader=validLoader,
            trainSet=trainSet, validSet=validSet_local,
            metadata=metadata, modelParam=modelParam, dataParam=dataParam,
            device=device, st_config=st_config,
            modelName=modelName, output_dir='.',  # already cwd-changed earlier
            tgt_scl_C=tgt_scl_C,
            wandb_run=wandb_run, log_file=log_f,
        )
        log_f.close()
        # Final summary
        print(f"\n[FINAL] {method_full}: Best valid RMSE = {best_v_rmse:.6f} "
              f"(≈ {best_v_rmse * tgt_scl_C:.3f}°C, scl={tgt_scl_C:.2f})")
        if wandb_run is not None:
            try:
                wandb_run.summary['final_best_valid_rmse_norm'] = best_v_rmse
                wandb_run.summary['final_best_valid_rmse_C']    = best_v_rmse * tgt_scl_C
                wandb_run.summary['st_n_rounds_completed']      = len(history) - 1
                wandb_run.finish()
            except Exception as e:
                print(f"[WARN] wandb finish error: {e}")
        return    # 提前结束 main(),不进标准训练循环

    print(f"[run.py] Training {modelName} for {V1_EPOCHS} epochs, iDim={modelParam['iDim']}, "
          f"nNodes={metadata['nNodes']}")

    # ===== DEBUG: model + train loop sanity =====
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DEBUG/model] params: total={n_params}, trainable={n_trainable}")
    print(f"[DEBUG/model] device={device}, modelName={modelName}, output={V1_OUTPUT_DIR}")
    print(f"[DEBUG/model] train_batches={len(trainLoader)}, valid_batches={len(validLoader)}, "
          f"batch_size={dataParam['batchSize']}")

    for epoch in range(EPOCH, V1_EPOCHS):
        trainLoss, trainRMSE, train_truth, train_pred = train_fn(
            trainLoader, model, lossFn, opt, scheduler, device, metadata['nNodes'])
        validLoss, validRMSE, valid_truth, valid_pred = test_fn(
            validLoader, model, lossFn, device, metadata['nNodes'])

        # ---- 6 metrics: RMSE / MBE / MAE × normalized / Celsius ----
        m_train = utils.compute_6_metrics(train_truth, train_pred, scl=tgt_scl_C)
        m_valid = utils.compute_6_metrics(valid_truth, valid_pred, scl=tgt_scl_C)

        if epoch < 5 or epoch % 100 == 0:
            print(f"[ep {epoch:4d}] tr_rmse={m_train['rmse_norm']:.4f} ({m_train['rmse_C']:.3f}°C)  "
                  f"v_rmse={m_valid['rmse_norm']:.4f} ({m_valid['rmse_C']:.3f}°C)  "
                  f"best={bestLoss:.4f}")
        if np.isnan(trainLoss) or np.isnan(m_train['rmse_norm']):
            print(f"[ep {epoch}] NaN detected, stopping.")
            break

        # ---- Local log:每 epoch 1 行,faithful original_code 风格 ----
        with open(log_path, 'a') as f:
            print(f"Epoch {epoch}: loss {trainLoss:.4e}/{validLoss:.4e}; "
                  f"RMSE {trainRMSE[0]:.4f}/{validRMSE[0]:.4f}; "
                  f"LR {scheduler.get_last_lr()[0]:.6e}", file=f)
            print(f"RMSE std: {trainRMSE[1]:.4f}/{validRMSE[1]:.4f}; "
                  f"min: {trainRMSE[2]:.4f}/{validRMSE[2]:.4f}; "
                  f"max: {trainRMSE[3]:.4f}/{validRMSE[3]:.4f}", file=f)

        if wandb_run is not None:
            wandb_run.log({
                'epoch':            epoch,
                'lr':               scheduler.get_last_lr()[0],
                'train/loss':       trainLoss,
                'valid/loss':       validLoss,
                # ---- 6-metric suite (training) ----
                'train/rmse_norm':  m_train['rmse_norm'],
                'train/mbe_norm':   m_train['mbe_norm'],
                'train/mae_norm':   m_train['mae_norm'],
                'train/rmse_C':     m_train['rmse_C'],
                'train/mbe_C':      m_train['mbe_C'],
                'train/mae_C':      m_train['mae_C'],
                # ---- 6-metric suite (validation) ----
                'valid/rmse_norm':  m_valid['rmse_norm'],
                'valid/mbe_norm':   m_valid['mbe_norm'],
                'valid/mae_norm':   m_valid['mae_norm'],
                'valid/rmse_C':     m_valid['rmse_C'],
                'valid/mbe_C':      m_valid['mbe_C'],
                'valid/mae_C':      m_valid['mae_C'],
                # ---- best tracker ----
                'best_valid_rmse':  bestLoss,
            })
            # Lap 启用时额外 log sup/lap 分量(覆盖 train_lap 单独 / EAD+Lap 联合两种)
            if lap_active:
                src_fn = network.train_ead_lap if ead_active else network.train_lap
                if hasattr(src_fn, 'last_sup'):
                    wandb_run.log({
                        'epoch':          epoch,
                        'train/sup_loss': src_fn.last_sup,
                        'train/lap_loss': src_fn.last_lap,
                    })

        if m_valid['rmse_norm'] < bestLoss:
            bestLoss = m_valid['rmse_norm']
            with open(log_path, 'a') as f:
                print(f"Model saved. best_valid_RMSE = {bestLoss:.6f}", file=f)
            if wandb_run is not None:
                wandb_run.log({'best_valid_rmse_updated': bestLoss, 'best_epoch': epoch})
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict':   opt.state_dict(),
                'bestLoss':         bestLoss,
            }, chkptPath)

        # Local plot of training history (kept; small file, useful for quick visual)
        hist.append([trainLoss, validLoss, m_train['rmse_norm'], m_valid['rmse_norm']])
        utils.plotHist(hist, modelName)

    final_msg = (f"\n[FINAL] {method_full}: Best valid RMSE = {bestLoss:.6f} "
                 f"(≈ {bestLoss * tgt_scl_C:.3f}°C, scl={tgt_scl_C:.2f})")
    print(final_msg)
    with open(log_path, 'a') as f:
        print(final_msg, file=f)
    if wandb_run is not None:
        wandb_run.summary['final_best_valid_rmse_norm'] = bestLoss
        wandb_run.summary['final_best_valid_rmse_C'] = bestLoss * tgt_scl_C
        wandb_run.finish()


if __name__ == "__main__":
    main()

# Urban Prediction Semi —— 方法 Pipeline + 实验记录

每个框架内一次性给齐:**概念 → 数据/模型/loss 细节 → 该框架下所有 job 的结果**。
末尾有 V2 spatial 综合排行榜 + 关键观察。

状态约定:
- **DONE** — 完整跑完,日志里有明确的 `Best valid RMSE`
- **CANCEL-PART** — 被 SLURM cancelled,但从日志里恢复了 partial best valid RMSE
- **CANCEL** — cancelled 且无 RMSE 记录
- **ERROR** — Python traceback / 崩溃
- **BUGGY (target leak)** — 修复前的 V1 baseline,feature 含 target → 数字不可信

---

## 0. 全局统一约定(所有 V2 框架共享)

下面这些设定贯穿 V2 所有方法,后续框架不另作说明就默认沿用。

### 0.1 数据 schema(V2 spatial,labeled 58 站)

每个 (station, timestep) 的特征向量 `iDim = 1343`:

```
[ WRF window         (315) ]   = 5 × 63 channels  (window=2 → t-2..t+2)
[ CLMS              (3)   ]   = 当前时刻 climate 3 项
[ UrbanFeature      (17)  ]   = 16 形态学 + 1 距离 lake;NaN 填 0/median
[ GeoEmbed          (1008)]   = UrbanFeatureMat (7-ch × 12×12 avg pool)
```

实现:[data.py:225-245](code/downscale-gnn/data.py#L225-L245)([data_semi.py:608-820](code/downscale-gnn/data_semi.py#L608-L820) 对 unlabeled 镜像)。


### 0.2 归一化

| 字段 | 方式 | 入口 |
|---|---|---|
| WRF/CLMS/UF/GeoEmbed | **per-feature global**(对最后一维分别 min-max,跨所有 station 跨所有 t)| `utils.MinMax()` [utils.py:113](code/downscale-gnn/utils.py#L113) |
| Target(`norm_mode='global'`,V2 现行)| **真正 global**(单一 min/max)| [data.py:228-237](code/downscale-gnn/data.py#L228-L237) |
| Target(`norm_mode='per_station'`)| **per-station global-over-time** | 同上 |
| Unlabeled WRF/CLMS | 用 **labeled 的** off/scl 归一化(BugFix)| [data_semi.py:608-611](code/downscale-gnn/data_semi.py#L608-L611) |

### 0.3 图构建(V2 通用)

```
W_ij = SimilarityMat[i,j] × exp(-Distance[i,j] / max_dist)
A_ij = max(W_ij, W_ji)
A[A < thres] = 0;  fill_diagonal(A, 0)
```
[data.py:130-146](code/downscale-gnn/data.py#L130-L146)。

`thres` 默认 0.1;V2 semi 多了 **adaptive threshold**:density > 80% 就把 thres 加 0.05 重算,直到落在 30–80% 区间 [data_semi.py:642-667](code/downscale-gnn/data_semi.py#L642-L667)。所以 12999 实际 thres = 0.35。

`EDGE_MODE`(env)在 adj 构好之后再处理:
- `all`(默认)— 不动
- `no_uu` — `Adj[L:, L:] = 0`
- `discount_uu` — `Adj[L:, L:] *= UU_DISCOUNT`(默认 0.1)
- `block_no_uu` / `block_discount_uu` — LL/LU 用各自 thres 重建

### 0.4 模型骨架(V2 通用)

`network.GNN` / `network_semi.GNN`([network_semi.py:25-96](code/downscale-gnn/network_semi.py#L25-L96)):

```
encoder:    [Linear(iDim → 128), PReLU] × 2
processor:  [GraphConv(128 → 128, aggr='mean'), PReLU] × 3
decoder:    [Linear(128 → 128), PReLU, Linear(128 → oDim)]
```

- `oDim = 1`(普通)或 `oDim = 2`(heteroscedastic = (μ, log σ²))
- `Dropout` flag 存在但**未实际调用** —— dead config
- `conv_type` 可切 graphconv / sageconv / appnp / gat;默认 graphconv

### 0.5 训练 hyperparameters(V2 sup vs semi 不完全对称)

| 项 | sup [run.py](code/downscale-gnn/run.py) | semi [run_semi.py](code/downscale-gnn/run_semi.py) |
|---|---|---|
| Optimizer | Adam,lr=1e-3,无 weight decay | 同 |
| Scheduler | `ExponentialLR(γ=0.9992)` | 同 |
| Loss(supervised 部分)| `nn.HuberLoss()` | 同 |
| Batch size | 512 | 128 |
| Epochs | 5000 | 5000 |
| Early stopping | patience=300 | **disabled** |
| Gradient clipping | 无 | `clip_grad_norm_(max_norm=1.0)` |
| 6-metric eval | RMSE/MBE/MAE × normalized/Celsius(用 `tgt_global_scl` 反归一化)| 同 |

### 0.6 Validation 切分(V2 通用)

`EVAL_MODE` env(默认 spatial):

- **spatial**:FPS 选 8 个最分散 labeled 站作 valid([data.py:17-33](code/downscale-gnn/data.py#L17-L33)),全部 timesteps 都用
- **temporal**:按 `TEMPORAL_FRAC`(默认 0.8)取**最后** 20% timesteps 作 valid(seq 切分)
- 旧 random split 模式已不在 V2 sup/semi 主流程中使用

---

## 1. 框架 1:V2 Supervised baseline

### 1.1 概念
最朴素的 GNN downscaling:只用 58 个 labeled 站,GraphConv 做空间消息传递,Huber loss 直接拟合温度。

### 1.2 入口与数据
- 入口:[run.py](code/downscale-gnn/run.py) + [data.py](code/downscale-gnn/data.py)
- 节点:58 labeled,无 unlabeled
- 图:single-block similarity threshold(thres=0.1 固定,**不触发** adaptive)
- 特征:1343 维(同 §0.1)
- 归一化:`NORM_MODE=global`

### 1.3 Loss
$$
\mathcal{L} = \mathrm{Huber}(\hat{y}_{\mathrm{labeled}},\, y_{\mathrm{labeled}})
$$
只对 `label_mask` 内的节点算,Huber δ=1.0(PyTorch 默认)。

### 1.4 训练流程
- 5000 epoch,early stop patience=300
- 每 epoch 跑全图(58 节点 × 3668 timestep),valid 用 spatial mask

### 1.5 该框架下的 jobs

| jobid | 变体 | 区别于 baseline | val_mode | best_v_rmse | 状态 |
|---|---|---|---|---|---|
| **12998** | V2 sup baseline | — | spatial | **0.0528** | DONE |
| 13101 | V2 MLP supervised | `USE_GNN=0` → encoder + decoder 直连,跳过 processor | spatial | 0.0578 | DONE |
| 13342 | V2 sup dirpool | GeoEmbed 用 `POOL_TYPE=directional, POOL_DIRS=4` 而非 12×12 avg | spatial | **0.0503** | DONE |
| 13378 | V2 sup temporal | `EVAL_MODE=temporal` | sequential | 0.0598 | DONE |
| 13376 | V2 sup temporal | 同 13378 | sequential | — | ERROR |

---

## 2. 框架 2:V2 Naive Semi-Supervised

### 2.1 概念
在 baseline 图上加 400 个 FPS-score 选出的 unlabeled 节点,只让它们参与 message passing,**不给它们任何 target loss**。

### 2.2 入口与数据
- 入口:[run_semi.py](code/downscale-gnn/run_semi.py) + [data_semi.py](code/downscale-gnn/data_semi.py)([SEMI_MODE='basic'](code/downscale-gnn/run_semi.py#L987))
- 节点:58 labeled + 400 unlabeled = 458
- 图:unified similarity-threshold,有 adaptive,**EDGE_MODE 可调**
- 特征:同 sup 1343 维;unlabeled 用同样的 schema,target 占位 0

### 2.3 Unlabeled FPS 选择([data_semi.py:86-243](code/downscale-gnn/data_semi.py#L86-L243))
3 步:
1. **过滤**:UrbanFeature 与 labeled 中心距离 top 5% outlier 排除
2. **score-weighted FPS**(`USE_FPS=2`):每步选 `argmax(score × diversity)`
   - score = 与最近 labeled 节点的图相似度(高 → 与 labeled 像)
   - diversity = max-min 距离(标准 FPS)
3. **adaptive threshold** 控制密度

### 2.4 Loss
**和 supervised 完全相同** —— 只有 labeled 的监督 loss,unlabeled 没有 target signal:
$$
\mathcal{L} = \mathrm{Huber}(\hat{y}_{\mathrm{labeled}},\, y_{\mathrm{labeled}})
$$
唯一变化:forward pass 时 unlabeled 节点会**通过图**影响 labeled 节点的 hidden state。

### 2.5 该框架下的 jobs

| jobid | EDGE_MODE | 备注 | val_mode | best_v_rmse | 状态 |
|---|---|---|---|---|---|
| 12999 | all | naive baseline,U-U 边全保留 → 66107 条 | spatial | 0.0656(被 U-U 噪声淹没)| CANCEL-PART(@ ep 1167)|
| 13000 | no_uu | U-U 边删干净 | spatial | ~0.053 | CANCEL-PART |
| 13322 | no_uu + LU_TOPK | 每个 unlabeled 只保留 top-K labeled 邻居 | spatial | 0.0606 | DONE |
| 13377 | no_uu | `EVAL_MODE=temporal` | sequential | 0.0608 | DONE |

---

## 3. 框架 3:V2 Laplacian Regularization

### 3.1 概念
用图 Laplacian 给 unlabeled 节点一个**直接的训练信号**:邻居预测应该平滑。

### 3.2 入口与变体
入口 [run_semi.py](code/downscale-gnn/run_semi.py) + `SEMI_MODE` env 切到下列之一,训练函数在 [network_semi.py:512-790](code/downscale-gnn/network_semi.py#L512-L790)。

### 3.3 三种 Laplacian 变体的 Loss

#### (a) 标准 Laplacian(`SEMI_MODE='laplacian'`)
$$
\mathcal{L} = \mathrm{Huber}(\hat{y}_L, y_L) + \lambda_{\mathrm{lap}} \cdot \frac{1}{|E|}\sum_{(i,j) \in E} w_{ij} (\hat{y}_i - \hat{y}_j)^2
$$
默认 `LAMBDA_LAP=0.1`。`w_ij` = 边权,求和遍历 batch 里**每个 graph** 的所有边。

#### (b) Residual Laplacian(`SEMI_MODE='residual_laplacian'`)
$$
r_i = \hat{y}_i - \mathrm{WRF\_T2}_i, \qquad
\mathcal{L} = \mathrm{Huber}(\hat{y}_L, y_L) + \lambda_{\mathrm{lap}} \cdot \frac{1}{|E|}\sum_{(i,j)} w_{ij} (r_i - r_j)^2
$$
**物理动机**:UHI correction (= 模型对 WRF 的修正量) 在空间上应该平滑,不是绝对温度本身。`WRF_T2` 是 WRF channel 0(°C → global-norm)。

#### (c) Adaptive Laplacian(`SEMI_MODE='adaptive_laplacian'`)
$$
\alpha_{ij} = \mathrm{EdgeWeightNet}([h_i, h_j]) \in (0,1) \quad\text{(2 层 MLP + sigmoid)}
$$
$$
\mathcal{L} = \mathrm{Huber}(\hat{y}_L, y_L) + \lambda_{\mathrm{lap}} \cdot \mathbb{E}_{ij}\big[\alpha_{ij} w_{ij}(\hat{y}_i - \hat{y}_j)^2\big] + 0.01 \cdot \mathcal{H}(\alpha)
$$
- `EdgeWeightNet` 学每条边该不该平滑(城/郊边界 → α 接近 0,均匀区 → α 接近 1)
- 第三项是 entropy regularizer,鼓励 α 不要全部停在 0.5

### 3.4 Message-Weight 变体(`SEMI_MODE='msg_weight'`)
不改 loss,只在 forward 之前把"以 unlabeled 为源"的边权乘 `unlabeled_discount=0.5`:
```python
edge_attr_mod[src_is_unlabeled] *= unlabeled_discount
```
本质上是 graph attention 的硬编码版本。

### 3.5 该框架下的 jobs(全部 spatial)

| jobid | 变体 | EDGE_MODE | 额外参数 | best_v_rmse | 状态 |
|---|---|---|---|---|---|
| 13001 | Laplacian | no_uu | λ=0.1 | 0.047 | CANCEL-PART |
| 13002 | Laplacian | block_no_uu | LL/LU 各自 thres | 0.047 | CANCEL-PART |
| 13041 | Laplacian | all | 不删 U-U | 0.049 | CANCEL-PART |
| 13020 | Residual Lap | no_uu | λ=0.1 | **0.044** | CANCEL-PART |
| 13092 | Residual Lap + LU_TOPK | no_uu | 每个 U 只连 top-K labeled | **0.044** | CANCEL-PART |
| 13323 | Lap + LU_TOPK | no_uu | — | 0.0475 | CANCEL-PART |
| 13354 | Lap(no_aux)| no_uu | `n_unlabeled=0` 退化 | 0.0437 | CANCEL-PART |
| 13311/19/38/44/50 | lap_no_aux 重试 | — | — | — | ERROR(批量失败,只 13354 部分成功)|
| 13102 | MLP+Residual Lap | no_uu | `USE_GNN=0` | 0.046 | CANCEL-PART |
| 13103 | GNN1+Residual Lap | no_uu | `N_GNN=1` | **0.044** | CANCEL-PART |

**解读**:residual-target + Laplacian + no_uu graph 是 V2 semi-only 这一档里持续最强的组合,~0.044。

---

## 4. 框架 4:V2 Graph Signal Reconstruction (GSR)

### 4.1 概念
**离线**用图谱方法把 labeled target 反推到 unlabeled 节点上,得到 pseudo-labels;然后给 unlabeled 节点也加上 supervised loss。

### 4.2 GSR 数学公式([run_semi.py:121-152](code/downscale-gnn/run_semi.py#L121-L152))
$$
\Delta^* = \arg\min_{\Delta} \|M(\Delta - \Delta_{\mathrm{obs}})\|^2 + \mu \cdot \Delta^\top L \Delta
$$
其中 $L = D - W$ 是图 Laplacian,$M = \mathrm{diag}(\mathbb{1}_{\mathrm{labeled}})$。闭式解:
$$
\Delta^* = (M + \mu L)^{-1} M \Delta_{\mathrm{obs}}
$$
预计算 $A^{-1}M$ 一次,所有 timestep 共用。

### 4.3 三种 GSR 子变体

#### (a) Plain GSR(`SEMI_MODE='graph_signal_recon'`)
直接对 target 做信号重建:`pseudo_label[t] = A_inv_M @ targets[t]`。

#### (b) Residual GSR(`SEMI_MODE='residual_gsr'`)
对 Δ = target − WRF_T2 做信号重建,然后加回 WRF:
```
pseudo_label[t] = WRF_T2[t] + A_inv_M @ (targets[t] - WRF_T2[t])
```

#### (c) Conformal-weighted GSR(`SEMI_MODE='gsr_conformal'`)
[run_semi.py:320-442](code/downscale-gnn/run_semi.py#L320-L442) 算了 LOO 残差 → 每个 unlabeled 节点一个 weight:
$$
w_u = \frac{1}{1 + 10 \cdot \mathrm{interval\_width}_u}
$$
然后 pseudo loss 改成 weighted:
$$
\mathcal{L}_{\mathrm{pseudo}} = \mathbb{E}_u\big[w_u \cdot (\hat{y}_u - \mathrm{pseudo}_u)^2\big]
$$

### 4.4 训练 Loss
$$
\mathcal{L} = \underbrace{\mathrm{Huber}(\hat{y}_L, y_L)}_{\text{sup}} + \lambda_{\mathrm{gsr}} \cdot \mathrm{Huber}(\hat{y}_U, \mathrm{pseudo}_U)
$$
默认 `lambda_gsr=0.1`。

### 4.5 该框架下的 jobs(全部 spatial,EDGE_MODE=no_uu)

| jobid | 变体 | best_v_rmse | 状态 |
|---|---|---|---|
| 13005 | GSR | 0.048 | CANCEL-PART |
| 13006 | Residual GSR | 0.049 | CANCEL-PART |
| 13017 | GraphMix(GSR + mixup augmentation)| 0.048 | CANCEL-PART |

---

## 5. 框架 5:V2 Label Propagation

### 5.1 概念
经典的图上 label propagation:对每一时刻 $t$ 迭代地把 labeled target 推向 unlabeled,然后用结果当 pseudo-label。

### 5.2 算法([run_semi.py:12-42](code/downscale-gnn/run_semi.py#L12-L42))
```
P = D^{-1} A           # row-normalized transition
Y_0 = labeled targets (其它 0)
for k in 1..50:
    Y_{k+1} = α P Y_k + (1-α) Y_0
return Y_K
```
默认 `alpha=0.5, n_iters=50`。

### 5.3 训练 Loss
和 GSR 一样:`sup_huber(L) + λ * pseudo_huber(U)`。

### 5.4 该框架下的 jobs
没有完整 done 的 LP-only 实验入排行榜 —— 这个 mode 主要被 GSR / Lap 取代。

---

## 6. 框架 6:V2 Empirically-Anchored Decomposition (EAD)

### 6.1 概念
把 station 残差(温度 − WRF_T2)分解成时间锚 + 空间锚 + 噪声:
$$
\Delta_{i,t} = \alpha_t + \beta_i + \varepsilon_{i,t}
$$
- $\alpha_t$:全 station 平均(逐 t),离线算
- $\beta_i$:per-station 平均(逐 i),training 站直接观测,unlabeled 站用 **Kriging 插值**
- $\varepsilon_{i,t}$:剩下的"真"残差,**这才是 GNN 学的对象**

### 6.2 入口与预计算
入口:[run_ead.py](code/downscale-gnn/run_ead.py)。`precompute_alpha_beta` [line 150](code/downscale-gnn/run_ead.py#L150):
```python
α_t = mean over stations of Δ at time t            # (T,)
β_train = mean over time of (Δ − α_t) at each train station   # (n_train,)
β_hat   = kriging_beta(Adj, β_train, train_idx)              # (nNodes,)
```
Kriging 公式([line 71-90](code/downscale-gnn/run_ead.py#L71-L90)):
$$
\hat{\beta}_u = \frac{\sum_i W_{ui} \beta_i^{\mathrm{train}}}{\sum_i W_{ui} + \epsilon}
$$
即用图边权对邻居 β 加权平均。

### 6.3 Pseudo-ε 与不确定性
`kriging_pseudo_epsilon` ([line 93-148](code/downscale-gnn/run_ead.py#L93-L148)):
$$
\tilde{\varepsilon}_{u,t} = \mathrm{kriging}(\varepsilon_{train, t}, u);
\quad \sigma_u^2 = \mathrm{var\ of\ kriging\ residuals}
$$
σ² 归一化到 mean=1。

### 6.4 Loss([run_ead.py:282-425](code/downscale-gnn/run_ead.py#L282-L425))

模型预测 $\hat{\varepsilon}$,完整 loss:

$$
\begin{aligned}
\mathcal{L} =\;
& \underbrace{\mathrm{Huber}(\hat{\varepsilon}_L, \varepsilon^{\mathrm{target}}_L)}_{\text{sup\_eps}} \\
& + \lambda_{\mathrm{lap}} \cdot \tfrac{1}{|E|}\sum w_{ij}(\hat{\varepsilon}_i - \hat{\varepsilon}_j)^2  &\text{(EAD\_LAP)} \\
& + \lambda_{\mathrm{zm}} \cdot \overline{(\hat{\varepsilon}_{\cdot, t}.\mathrm{mean})^2}  &\text{(EAD\_ZERO\_MEAN)} \\
& + \lambda_{\mathrm{psd}} \cdot \tfrac{1}{|U|}\sum_u \tfrac{1}{\sigma_u^2 + 10^{-3}} (\hat{\varepsilon}_u - \tilde{\varepsilon}_u)^2  &\text{(EAD\_PSEUDO)} \\
& + \lambda_{\mathrm{rec}} \cdot \mathrm{Huber}(\hat{\varepsilon}_{\mathrm{masked}}, \varepsilon^{\mathrm{target}}_{\mathrm{masked}})  &\text{(EAD\_MASK\_RECON)}
\end{aligned}
$$

其中 $\varepsilon^{\mathrm{target}}_{i,t} = y^{\mathrm{residual}}_{i,t} - \alpha_t - \hat{\beta}_i$。
最终预测:$\hat{\Delta} = \alpha_t + \hat{\beta}_i + \hat{\varepsilon}_{i,t}$。

### 6.5 该框架下的 ablation 矩阵(B1–B9,全部 spatial,EDGE_MODE=no_uu)

| jobid | tag | EAD_ALPHA | EAD_BETA | EAD_ZM | EAD_PSEUDO | EAD_MASK | 其它 | best_v_rmse | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| 13204 | B1 | 1 | 0 | 0 | 0 | 0 | 只 anchor α_t | — | CANCEL |
| 13205 | B2 | 1 | 1 | 0 | 0 | 0 | + β kriging | — | CANCEL |
| 13206 | B3 | 1 | 1 | 1 | 0 | 0 | + zero-mean | — | CANCEL |
| 13207 | B4 | — | — | — | — | — | `USE_GNN=0` MLP only | — | CANCEL |
| 13214–13216 | B2/B3 retries | — | — | — | — | — | — | — | CANCEL |
| 13217 | B6 | 1 | 0 | 0 | **1** | 0 | + kriging pseudo-ε,λ_psd=0.01 | — | INCOMPLETE |
| 13340 | B7 | 1 | 0 | 0 | 0 | **1** | + mask reconstruct,K=10,λ_rec=1.0 | — | CANCEL |
| 13341 | B8 | 1 | 0 | 0 | 0 | 0 | `CONV_TYPE=gat` 4-head | — | CANCEL |
| 13342 | B9a | — | — | — | — | — | dirpool sup-only(无 EAD)| **0.0503** | DONE |
| 13343 | B9b | 1 | 0 | 0 | 0 | 0 | EAD + dirpool | — | INCOMPLETE |
| 13371 | B0 | 1 | 0 | 0 | 0 | 0 | `n_unlabeled=0` 退化 EAD | — | CANCEL |
| 13380 | B1_cnn | 1 | 0 | 0 | 0 | 0 | `POOL_TYPE=raw` + CNN encoder | — | CANCEL |
| 13400 | B1_cnn_raw | — | — | — | — | — | raw 401×401 UFM via buffer | — | ERROR |

**解读**:几乎所有 EAD job CANCEL/INCOMPLETE,只有 13342 (B9a dirpool sup) 跑出 0.0503。

---

## 7. 框架 7:V2 Self-Training(Iterative)

### 7.1 总体概念
- **R0**:训一个 base 模型(类似 EAD B1)
- **R1..N**:每轮基于 R_{k-1} 的预测,选 K 个最自信的 unlabeled 节点 → 给它们打 pseudo-target → 加进训练 → 训 R_k
- 最终用最后一轮模型预测 valid 8 站

### 7.2 V1 (`run_selftrain_iter.py`)

**Confidence:Snapshot Ensemble**
- 每轮训练后期 snapshot 模型 M 次(用 cyclic LR);M=5,T_cycle=40 epoch
- 对每个 unlabeled u:`pseudo_u = mean(snapshots)`,`σ_u = std(snapshots)`
- Confidence = `−σ_u`

**Selection** —— `greedy_select` ([run_selftrain_ead.py:343](code/downscale-gnn/run_selftrain_ead.py#L343)):
$$
\mathrm{score}(u) = \alpha \cdot \mathrm{div}(u, S) + \beta \cdot \mathrm{rel}(u, V)
$$
- 先按 confidence 阈值 τ_quantile 过滤(默认 0.5,即取前 50%)
- `div(u, S)` = max-min 距离(已选集 S)→ 多样性
- `rel(u, V)` = 与 valid 8 站的 kNN 平均(k=5)→ 与最终任务相关
- 默认 α=1, β=1

**Loss(每轮训练)**
$$
\mathcal{L} = \underbrace{\mathrm{Huber}(\hat{\varepsilon}_L, \varepsilon_L)}_{\mathrm{sup}} + \lambda_{\mathrm{lap}} \mathcal{L}_{\mathrm{lap}}(\hat{\varepsilon}) + \lambda_{\mathrm{psd}} \cdot \mathrm{Huber}(\hat{\varepsilon}_{S},\, \mathrm{pseudo}_{S})
$$
其中 $S$ = 至今选过的 pseudo 节点。

**该 V1 下的 jobs(全部 spatial,EDGE_MODE=no_uu,n_unl=1799)**

| jobid | K | M | T_cycle | λ_psd | best_v_rmse | 状态 |
|---|---|---|---|---|---|---|
| 13361 | 100 | 5 | 40 | — | — | ERROR |
| 13374 | 100 | 5 | 40 | 0.5 | — | ERROR |
| 13375 | 100 | 5 | 40 | 0.5 | 0.0445 | DONE |

### 7.3 V2 (`run_selftrain_iter_v2.py`)

V1 的失败模式修复:
- **A1**:pseudo-target 用 best_prev_state 的 prediction(不是 snapshot mean)
- **A2**:σ 用小 LR 的 snapshot perturbation(LR_MAX=5e-4 比 V1 大幅小)
- **B**:每轮 **warm-start** from best_prev_state(不是 from-scratch)
- 更小 K=50,更短 round=200 epoch,λ_psd=0.3

| jobid | K | round_ep | λ_psd | best_v_rmse | 状态 |
|---|---|---|---|---|---|
| 13379 | 50 | 200 | 0.3 | **0.0436** | DONE |

### 7.4 V3 (`run_selftrain_v3.py`)

V3 进一步修复:
- **固定图**:所有 round 都用同一个 1857 节点的图(实际跑出来 258 节点),只改 `pseudo_select_mask`
- **Confidence 两条路**(env `ITER_CONFIDENCE`)

#### (a) Heteroscedastic head(`het`)
模型 `oDim=2`,输出 (μ, log σ²),用 **β-NLL loss**(Seitzer 2022):
$$
\mathrm{NLL}_{\mathrm{β}} = \mathbb{E}\Big[\sigma^{\beta} \cdot \big(\tfrac{(y-\mu)^2}{2\sigma^2} + \tfrac{1}{2}\log\sigma^2\big)\Big]_{\text{stop-grad on }\sigma^{\beta}}
$$
默认 β=0.5,`log_var.clamp(-7, 4)`。Confidence = `−σ`。
[run_selftrain_v3.py:118-150](code/downscale-gnn/run_selftrain_v3.py#L118-L150)

#### (b) Conformal(`conformal`)
开始时做 **5-fold cross-validation**:把 50 train 站分 5 fold,每 fold hold-out 10 站,train 完用模型预测 hold-out → 得到 OOF residuals(每个 train 站一个绝对残差)。
- 对每个 candidate u:用 emb-space kNN(k=8)找最近的 k 个 train 站,加权平均它们的 OOF residual → σ_u
- bandwidth 自动取 emb 距离的 median × 0.5
- 整个 conformal **不接触 valid 8 站的 target**

#### Round 训练(共用)
$$
\mathcal{L} = \mathrm{sup}(\hat{\varepsilon}_L) + \lambda_{\mathrm{lap}} \mathcal{L}_{\mathrm{lap}} + \lambda_{\mathrm{psd}} \cdot \mathrm{sup}(\hat{\varepsilon}_S, \mathrm{pseudo}_S)
$$
- R0:500 epoch,lr=1e-3(标准)
- R1+:200 epoch,**warm-start** lr=1e-4(比 V2 小一个数量级)

**V3 该框架下的 jobs(全部 spatial,EDGE_MODE=no_uu,n_unl=1799)**

| jobid | confidence | β-NLL | K | round_ep | warm_lr | λ_psd | best_v_rmse(R0)| 状态 |
|---|---|---|---|---|---|---|---|---|
| 13414 | het | ❌(普通 NLL,σ-clamp)| 30 | 200 | 1e-4 | 0.3 | 0.0445 | ERROR(σ collapse)|
| 13415 | conformal | — | 30 | 200 | 1e-4 | 0.3 | **0.0433** | DONE |
| 13452 | het | **✓ β-NLL** | 30 | 200 | 3e-4 | 1.0 | 0.0440 | ERROR(σ 仍 collapse,std=0)|
| 13451 | conformal **5-fold** | — | 30 | 200 | 3e-4 | 1.0 | **0.0428** | DONE(R1–R5 Δ=0)|

---

## 8. 框架 8:V2 Progressive Self-Training(V3 的前身)

[run_progressive_st.py](code/downscale-gnn/run_progressive_st.py):
- 不分 round,持续训练
- 每 N epoch 重算所有 unlabeled 的 confidence,top-K 加进 pseudo-mask
- 比 iterative 版"软"

| jobid | from_scratch | best_v_rmse | 状态 |
|---|---|---|---|
| 13027 | yes | 0.047 | CANCEL-PART |
| 13028 | no(预训练初始化)| 0.047 | CANCEL-PART |

---

## 9. 框架 9:V1 Baselines

### 9.1 概念
和 V2 完全独立的入口,用 V1 数据集(68 labeled 站,Chicago AoT 2018,2948 timestep)。

### 9.2 数据 schema
[v1_data_shared.py](scripts/baseline/v1_data_shared.py),`iDim=1298`:
```
[ WRF window      (270) ]   = 5 × 54 channels  (V1: 45 thermal + 9 |Wind|)
[ raw_geo @ t     (20)  ]   = 16 UF + 3 CLMS + 1 dist   (CLMS dynamic, others static)
[ GeoEmbed        (1008)]   = FeaturePatch_401 7-ch × 12×12 avg pool
```

### 9.3 归一化
- WRF/raw_geo/target:**StationMat 预归一化好的 [0,1]**(legacy V1,看起来像 per-station)
- GeoEmbed:per-feature 跨 station 归一化
- Unlabeled(从 V2 raw 转过来):用 V2 自己的 min/max 归到 [0,1]

### 9.4 V1 supervised([run_v1_supervised_multival.py](scripts/baseline/run_v1_supervised_multival.py))
- 节点:68
- 图:V1 自带的 `GNN_N1_AJM.mat` = `|sim × dist|` thres=0.1 → 3836 边
- Loss:`HuberLoss` 直接拟合 pre-normalized target
- 训练:Adam lr=1e-3,γ=0.9992,batch=128,5000 epoch,tracking best valid

### 9.5 V1 semi([run_v1_semi_multival.py](scripts/baseline/run_v1_semi_multival.py))
- 节点:68 labeled + 400 unlabeled = 468
- 图:**k-NN k=10 在 (lat, lon) 上重建**(放弃 V1 的 `GNN_N1_AJM.mat`)→ 5389 边
- 边权 = `1 - normalized_distance`
- Loss:naive semi(同 §2.4),只有 labeled 算 loss
- batch=64

### 9.6 三种 val_mode
- **random**:`torch.randperm(generator(seed=19))` 切 75/25,batch shuffle
- **sequential**:前 80% timestep train,后 20% valid
- **spatial**:FPS(seed=42)选 10 个 valid station,index = [6,11,16,27,31,34,37,38,61,65]

### 9.7 该框架下的 jobs(最终干净版本)

| jobid | method | val_mode | best_v_rmse | 状态 |
|---|---|---|---|---|
| 13445 | sup | random | **0.0209** | DONE |
| 13448 | semi | random | 0.0220 | DONE |
| 13446 | sup | sequential | **0.0394** | DONE |
| 13449 | semi | sequential | 0.0405 | DONE |
| 13447 | sup | spatial | 0.0381 | DONE |
| 13450 | semi | spatial | **0.0330** | DONE |

**Caveat**:V1 用 per-station legacy norm,绝对数字不可与 V2 直接比;sup vs semi 内部 Δ 仍有意义。

### 9.8 V1 旧/中间版本(留作追溯,非当前结果)

| jobid | method | val_mode | best_v_rmse | 状态 | 标注原因 |
|---|---|---|---|---|---|
| 13393–13395 | V1 sup(早期)| random/seq/spatial | — | ERROR | 修复前的 bug |
| 13396 | V1 semi(74 维,3348 节点/68 T —— shape 算错)| random | 0.0491 | DONE | feature shape 错 |
| 13397 | V1 semi 早期 | sequential | 0.0484 | DONE | feature shape 错 |
| 13398 | V1 semi 早期 | spatial | 0.0633 | DONE | feature shape 错 |
| 13405–13407 | V1 sup(早期)| random/seq/spatial | — | ERROR | 修复前的 bug |
| 13408 | V1 semi(74 维,shape 修好)| random | 0.0486 | DONE | 已被替代 |
| 13409 | V1 semi(74 维)| sequential | 0.0427 | DONE | 已被替代 |
| 13410 | V1 semi(74 维)| spatial | 0.0593 | DONE | 已被替代 |
| 13411 | V1 sup(含 target leak)| random | 0.0121 | **BUGGY (target leak)** | feature col 0 = target |
| 13412 | V1 sup(含 target leak)| sequential | 0.0093 | **BUGGY (target leak)** | feature col 0 = target |
| 13413 | V1 sup(含 target leak)| spatial | 0.0132 | **BUGGY (target leak)** | feature col 0 = target |
| 13420 | V1 sup(74 维,target+4 aux 已 drop)| random | 0.0238 | DONE | 已被替代 |
| 13421 | V1 semi(74 维,aux dropped)| random | 0.0323 | DONE | 已被替代 |
| 13422 | V1 semi(74 维,aux dropped)| sequential | 0.0421 | DONE | 已被替代 |
| 13423 | V1 semi(74 维,aux dropped)| spatial | 0.0394 | DONE | 已被替代 |
| 13436 | V1 sup(1298 维,中间版)| sequential | 0.0389 | DONE | 已被替代 |
| 13437 | V1 sup(1298 维,中间版)| spatial | 0.0359 | CANCEL-PART | 已被替代 |

---

## 10. 框架 10:其它(简略)

### 10.1 CNN-on-raw-UFM
不预先 avg-pool,把原始 401×401 UFM 当 buffer,在 forward 阶段 CNN encode → 替代 1008 维 GeoEmbed。
- 13380(CNN with 12×12 pool 输入):CANCEL
- 13400(raw 401×401):ERROR

### 10.2 contrastive / mean teacher / VAT / temporal ensemble / sim-reg-match
代码文件存在于 [code/downscale-gnn/run_*.py](code/downscale-gnn/) 但当前没跑出可入排行榜的结果,跳过详述。

### 10.3 GraphMix
GSR + Mixup 数据增强:把两个时刻的 targets/features 按 λ ~ Beta(α,α) 混合后再做 GSR pseudo-label。
- 13017:0.048 → 见 §4.5

---

## 综合排行榜:V2 spatial val RMSE 总排名

| 排名 | jobid | method | 框架 | best_v_rmse | 状态 |
|---|---|---|---|---|---|
| 1 | 13451 | ST V3 conformal(5-fold)| 7 | **0.0428** | DONE |
| 2 | 13354 | Lap(no aux,n_unl=0)| 3 | 0.0437 | CANCEL-PART |
| 3 | 13379 | st_iter_v2 K=50 R=8 | 7 | 0.0436 | DONE |
| 4 | 13415 | ST V3 conformal(旧版)| 7 | 0.0433 | DONE |
| 5 | 13020 | Residual Lap | 3 | 0.044 | CANCEL-PART |
| 5 | 13092 | Residual Lap + LU_TOPK | 3 | 0.044 | CANCEL-PART |
| 5 | 13103 | GNN1 + Lap residual | 3 | 0.044 | CANCEL-PART |
| 8 | 13375 | st_iter K=100 | 7 | 0.0445 | DONE |
| 9 | 13102 | MLP + Lap residual | 3 | 0.046 | CANCEL-PART |
| 10 | 13001 | Laplacian | 3 | 0.047 | CANCEL-PART |
| 11 | 13005 | GSR | 4 | 0.048 | CANCEL-PART |
| 12 | 13041 | Lap(all edges)| 3 | 0.049 | CANCEL-PART |
| 13 | 13342 | V2 sup dirpool | 1/6 | 0.0503 | DONE |
| 14 | 12998 | V2 sup(baseline)| 1 | 0.0528 | DONE |
| 15 | 13101 | V2 MLP sup | 1 | 0.0578 | DONE |
| 16 | 13322 | Naive LU_TOPK | 2 | 0.0606 | DONE |
| 17 | 12999 | V2 naive semi(baseline)| 2 | 0.0656 | CANCEL-PART |

---

## 关键观察

1. **V1 vs V2 绝对 RMSE 不可直接比较**:V1 用 StationMat 预归一化(per-station 风格),V2 用 GLOBAL norm。0.0381(V1 sup spatial)和 0.0528(V2 sup spatial)不算同一个任务。
2. **V2 当前最优 ≈ 0.0428–0.044**(ST V3 conformal / residual lap 家族),全是 semi-supervised 或 self-train 方法,且都是 `EDGE_MODE=no_uu`。
3. **V2 naive semi(保留 U-U 边)= 0.0656 直接崩** —— 比 sup 差 +0.0128;切换到 no_uu 之后能恢复到 ≈ sup 水平(0.053)。
4. **V1 spatial 是唯一一个 naive semi 真的打败 sup 的设置**(0.0330 vs 0.0381)。random/sequential temporal 上 semi 略差(Δ ≈ +0.001),和 V2 的 trend 一致 —— SSL 在 OOD spatial 任务上有 free lunch,在 i.i.d. temporal 任务上中性偏负。
5. **V3 self-train R0 base ≈ 0.0428** 已经具备竞争力,但 pseudo-label 迭代轮 R1–R5 完全没带来增益 —— 当前 K=30/round + λ_pseudo=1.0 + warmstart_lr=3e-4 + EDGE_MODE=no_uu 这套设置过于保守,推不动指标。
6. **β-NLL 没修好 het 的 σ-collapse**(13452 仍然 σ std=0.0000)。Heteroscedastic head 在这个数据上目前是死路(50 个 train 站太少,σ-head 学不出来)。

---

## 附录 A:Confidence/Selection 选择策略对比(self-train 系列)

| 维度 | V1 Snapshot | V2 Snapshot+best_prev | V3 Het (β-NLL) | V3 Conformal (5-fold) |
|---|---|---|---|---|
| pseudo target | snapshots mean | best_prev prediction | μ from het head | best_prev prediction |
| σ source | snapshots std | small-LR snapshots std | learned log σ² | OOF residual + kNN-σ |
| 接触 valid? | 否 | 否 | 否 | **否**(改用 train OOF)|
| greedy selection | conf+div+rel | conf+div+rel | conf+div+rel | conf+div+rel |
| 图 | 每轮重建 | 每轮重建 | 固定 | 固定 |
| warm-start | 从头 | from best_prev | from best_prev | from best_prev |

## 附录 B:Loss 一览总表

| 框架 | sup loss | reg/aux loss | 最终 loss |
|---|---|---|---|
| 1. V2 sup | Huber(L) | — | Huber(L) |
| 2. V2 naive semi | Huber(L) | — | Huber(L) |
| 3a. Lap | Huber(L) | λ·Σwij(ŷi−ŷj)² | Huber(L) + 0.1·Lap |
| 3b. Residual Lap | Huber(L) | λ·Σwij(ri−rj)² , r=ŷ−WRF_T2 | Huber(L) + 0.1·ResLap |
| 3c. Adaptive Lap | Huber(L) | λ·Σ αij wij(ŷi−ŷj)² + 0.01·H(α) | sum |
| 3d. Msg Weight | Huber(L) | edge_weight 修改 | Huber(L) |
| 4a. GSR | Huber(L) | λ·Huber(U, pseudo_GSR) | Huber(L) + 0.1·GSR |
| 4b. Residual GSR | Huber(L) | λ·Huber(U, WRF+ΔGSR) | Huber(L) + 0.1·rGSR |
| 4c. Conformal GSR | Huber(L) | λ·Σ wu·(ŷu−pseudo_u)² | Huber(L) + 0.1·cGSR |
| 5. Label Prop | Huber(L) | λ·Huber(U, LP_pseudo) | Huber(L) + 0.1·LP |
| 6. EAD | Huber(ε_L) | Lap(ε̂) + ZM(ε̂) + Pseudo(ε̂_U, σ⁻²) + MaskRec | sum 5 项 |
| 7. ST V1/V2 | Huber(ε_L) | Lap + λ_psd·Huber(ε̂_S, pseudo_S) | sum |
| 7. ST V3 het | β-NLL(μ_L, σ²_L, y_L) | Lap + λ_psd·β-NLL(μ_S, σ²_S, pseudo_S) | sum |
| 7. ST V3 conformal | Huber(ε_L) | Lap + λ_psd·Huber(ε̂_S, pseudo_S),σ 来自 OOF | sum |
| 9. V1 sup/semi | Huber(L) | — | Huber(L) |

约定:
- `L` = labeled training nodes
- `U` = unlabeled nodes
- `S` = self-training 选出的 pseudo nodes(S ⊂ U)
- `ε` = EAD 残差;`r` = ŷ − WRF_T2;`ŷ` = 模型直接输出
- 所有 loss 默认 reduction=mean

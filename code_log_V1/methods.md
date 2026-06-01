# 方法论 —— 模型 / Loss / 训练 protocol

## 0. Problem Framework / Paper Framing(2026-05-11)

PPT / paper 一定要强调的 3 个 framing 点(独立于 methods,**这是 problem setup 层面的差异化**):

### 0.1 Spatial 任务,kriging 是自然 baseline

- 我们做的是 **spatial** 外推(spatial OOD:8 valid 站点未参与 loss)
- 不是 temporal(没用 sequential / future prediction setup)
- **Kriging(地统计学经典空间插值)是这类任务的天然 baseline** —— 不是我们临时加的,而是这类问题历史上的 standard
- **关键洞察**:Kriging benchmark = 0.0437 反超模型 baseline 0.0453,**揭示空间结构在该任务上极强**
- **核心创新**:把 kriging 的"空间 prior"以 self-train pseudo source 形式融进 GNN(`Hybrid pseudo = 0.5 model + 0.5 kriging`),让 model 学 features 修正 + 空间先验
- **Story line**:"Kriging 是地理学的经典手段,GNN 是 ML 的现代手段,我们做的是 bridge"

### 0.2 这个 problem 框架近期 ML 界无人主攻

- Urban temperature spatial downscaling **with SSL(semi-supervised)on small graph**(<500 节点)
- 文献 landscape:
  - **传统**:气象 / 大气科学界用 statistical downscaling(线性回归 / kriging)居多
  - **现代 ML**:大气下推主要做 **CNN-based super-resolution**(全场 raster → raster,不是 station-wise)
  - **图 SSL ML**:主流 Cora / Citeseer 等 ~2-5K 节点节点分类,我们 458 节点的回归任务**非主流**
- **机会**:**这块 niche 没有近期工作,我们的 systematic ablation + EAD 设计 = 真贡献**
- **Story line**:"在两个 community 都未充分探索的交叉点工作"

### 0.3 核心研究问题:**如何给 400 个辅助站点"额外监督信号"?**

**问题骨架**:
- 58 labeled stations:有 T 真值 → 主任务 sup loss 来源
- 400 unlabeled stations:**有 features 没 T** → **如何让它们也对模型有贡献?**
- naive semi GCN 已经通过 message passing 利用了 unlabeled features → 但**这只是 passive(被动)使用**
- **真正的研究问题**:能否给 unlabeled 一个 active 监督信号?

**我们测试的 5 种 active signal 策略**(故事弧):

| 策略 | 怎么给 unlabeled 信号 | 实证结果 |
|---|---|---|
| **Naive semi(被动 MP)** | features 通过 GNN 流到 labeled | baseline 0.0453(参考)|
| **Self-train(自蒸馏)** | 给 K 个 unlabeled 直接 pseudo target = 模型自己预测 | 0.0450(Δ≈0,**自蒸馏天花板**)|
| **Self-train(外部 kriging-pseudo)** | 给 K 个 unlabeled pseudo target = 空间插值 | 0.0429(**真涨点 -0.0024**)|
| **Consistency** | 强迫多 augmentation view 下预测一致 | 0.0449(Δ≈0,**小图增强信号弱**)|
| **Adversarial mask** | mask 一部分 unlabeled features 训练鲁棒性 | 0.0455(Δ≈0,**不增加可学信号**)|
| **Distribution Alignment**(13914 待测)| MMD 强制 unlabeled emb 分布对齐到 labeled emb | TBD |
| **Mask reconstruct(未做)| 加 aux head 重建 masked features | future |

**关键 insight**:
- 只有 **"外部信号"(kriging-pseudo)** 真涨点
- **"模型自己产生的信号"(self-distillation / consistency / adv mask)Δ≈0**
- 物理意义:**在小数据(50 train) + naive semi MP 已经吸收了所有"模型自己能榨出的信号"**;要更多就必须**引入外部 prior**(地理统计的 kriging)

**Story line**(可直接进 PPT):
> "How to give 400 unlabeled stations additional supervision beyond naive semi MP?
> We systematically tested 5 strategies. **The winning strategy is to inject 'external' prior knowledge** (geostatistical kriging) **as pseudo targets**, not to rely on model's own predictions (which yields Δ≈0 by self-distillation ceiling)."

这是 paper 的 **central methodological contribution**(SSL 部分),与 EAD(物理 prior 部分)正交。

### 0.4 Transductive 而非 Inductive(重要技术 framing)

| 维度 | Transductive(我们)| Inductive |
|---|---|---|
| 训练时图节点 | 全部 458 个**都看到**(labeled + unlabeled features)| 只看 labeled |
| 训练时 unlabeled targets | ✗ 不用 | ✗ 没有 |
| 测试时 valid features | ✓ **训练时就已在 graph 中**(forward 过)| 训练后才出现 |
| 测试时新 station | ✗ 不支持(必须在 458 之内)| ✓ 可扩展到新 station |
| 经典代表 | Kipf-Welling GCN(2017),GraphSAGE | DeepGCN(异构归纳)|

**我们的 setup 是 transductive**:
- 训练时 valid 8 站的 features 已经在 graph(参与 MP)
- 只是 target y 在训练时被 mask 掉(label_mask=False)
- 测试时直接 forward 这同一组 8 站
- → **valid 站从未被"加入图",一直就在图里,只是 target 被隐藏**

**为什么这框架值得 highlight**:
- Kriging-pseudo 在 transductive 下天然合适(用 train 真值 → unlabeled,**所有节点都已确定**)
- Self-train 选择 + diversity / relevance 都用 valid emb(features 可用,target 不可用)
- **如果是 inductive,kriging-pseudo / valid emb relevance 都不能用** → 我们的核心方法依赖 transductive 假设
- **Paper 一定要 explicit 说明这点**,避免审稿人误以为是 inductive

**Story line**:"我们用 transductive setting,**这与气象观测网部署一致**(站点位置预先已知,只是部分时刻数据缺失)"

---

## 1. 模型骨架(全部方法共用)

[code_V1/network.py](../code_V1/network.py) 中的 `GNN`:

```
encoder:    [Linear(iDim → 128), PReLU] × 2
processor:  [GraphConv(128 → 128, aggr='mean'), PReLU] × 3
decoder:    [Linear(128 → 128), PReLU, Linear(128 → 1), PReLU]
```

- iDim = 1302(WRF window 270 + station_aux 4 + raw_geo 20 + GeoEmbed 1008;sup ≡ semi schema,faithful original_code)
- oDim = 1
- decoder 最后一层有 PReLU(faithful to original_code,V2 时代曾经把这里去掉)
- 无 Dropout、无 BN

## 2. 训练 protocol(全部方法共用)

| 项 | 设定 |
|---|---|
| Optimizer | `Adam(lr=1e-3)`,无 weight decay |
| Scheduler | `ExponentialLR(γ=0.9992)` |
| Loss | `nn.HuberLoss()`(δ=1.0,PyTorch 默认)|
| Batch size | sup=512(同 original_code),semi=128 |
| Epochs | **sup=5000,semi=4000**(2026-05-10 起,n_unl scan 显示 semi best 出现在 ep 500-800,4000 已是 5x 安全边际,节省 GPU)|
| Early stop | **无**(跑满 epoch,save best) |
| Grad clip | **无**(同 original_code) |
| Best model | by valid RMSE |

实现位置:[code_V1/run.py](../code_V1/run.py),[code_V1/network.py:train()/test()](../code_V1/network.py)。

## 3. label_mask 机制

### 3.1 它是什么 —— 不是用来区分 val_mode 的

`label_mask` 是一个 per-node 的 BoolTensor,**作用单一:决定哪些节点的预测进入 loss / RMSE 计算**。

它不是为了切 train/valid split 用的(那个由 dataset 切分 + DataLoader 完成)。它解决的是另一个问题 ——
**图上的所有节点都会被 forward 一次,但不是所有节点都"该"算 loss**:
- semi 模式:unlabeled 节点没有 target,把它们也加进 loss 是错的(target=0 占位)
- spatial 模式:held-out 的 valid station 训练时绝对不能算进 loss(否则就 leak 了)

### 3.2 它如何在 4 种组合里起作用

| 模式 | 总节点 | train 阶段 mask=True 的节点 | valid 阶段 mask=True 的节点 |
|---|---|---|---|
| Sup random/sequential | 68(都 labeled)| 全部 68 | 全部 68 |
| Sup spatial            | 68(都 labeled)| 58 个 train station | 10 个 valid station |
| Semi random/sequential | 68 + 400 = 468 | 前 68 (labeled) | 前 68 (labeled) |
| Semi spatial           | 68 + 400 = 468 | 58 个 train station | 10 个 valid station |

注意 train 和 valid **不一定共用同一个 mask**:spatial 模式下两边显式不同(train 用 train_mask,valid 用 valid_mask);其它三种模式 mask 在 train 和 valid 一样,因为切分由"哪些 timestep 进 train、哪些进 valid"决定。

### 3.3 它如何在代码里实现

数据侧 [code_V1/data.py](../code_V1/data.py):
- spatial:用 `fps_select_stations()` 把 68 labeled 切成 58 train + 10 valid → 两套 mask
- 非 spatial:`label_mask` 标记前 nL 个节点为 True(其它都 False),train/valid 共用
- 每个 PyG `Data` 对象都带上对应的 `label_mask` 字段

训练侧 [code_V1/network.py](../code_V1/network.py):
```python
def _split_for_loss(yHat, batch, nNodes):
    if hasattr(batch, 'label_mask') and batch.label_mask is not None:
        mask = batch.label_mask
        yHat_l = yHat[mask]; y_l = batch.y[mask]
        # only labeled positions enter loss / RMSE
        ...
    else:
        # original_code's all-nodes behavior (sup random with no mask attached)
        ...
```

### 3.4 关键不变量

1. **forward 永远在全图上跑** —— 不管 mask 怎么切,messages 都正常传 → unlabeled 仍能给 labeled 节点贡献 hidden state
2. **loss 永远只在 mask=True 的位置算** —— 没有 target 的节点不污染梯度
3. **RMSE 跟 loss 用同一组节点** —— 报告的 RMSE 是 mask 内的平均

## 4. Baseline 1:V1 Supervised

### 4.1 概念
最朴素 GNN downscaling:68 个 labeled 站,GraphConv 跑空间 message passing,Huber 拟合温度。

### 4.2 Loss
$$
\mathcal{L} = \mathrm{Huber}(\hat{y}_{\mathrm{labeled}},\, y_{\mathrm{labeled}})
$$

- random/sequential 模式:对全部 68 节点算
- spatial 模式:训练时只对 58 train station 算,valid 时只对 10 valid station 算 RMSE

### 4.3 入口
- env:`V1_N_UNLABELED=0`,图用 V1 自带 sim×dist
- 脚本:[scripts_V1/V1_sup_random.sh](../scripts_V1/V1_sup_random.sh)、`V1_sup_sequential.sh`、`V1_sup_spatial.sh`

## 5. Baseline 2:V1 Naive Semi-Supervised

### 5.1 概念
68 labeled + 400 unlabeled 在同一个 k-NN 图上做 message passing,**unlabeled 不算 loss**,只通过图传消息影响 labeled 节点的 hidden state。

### 5.2 Loss
$$
\mathcal{L} = \mathrm{Huber}(\hat{y}_{\mathrm{labeled}},\, y_{\mathrm{labeled}})
$$

**和 sup 完全相同**。差异只在:
- 图变成 468 节点的 k-NN k=10
- 前向时 unlabeled 节点的 hidden state 也会传到 labeled 节点

### 5.3 入口
- env:`V1_N_UNLABELED=400, V1_KNN_K=10, V1_FPS_SEED=0`
- 脚本:[scripts_V1/V1_semi_random.sh](../scripts_V1/V1_semi_random.sh)、`V1_semi_sequential.sh`、`V1_semi_spatial.sh`

## 6. EAD:Empirically-Anchored Decomposition(env: V1_EAD_ALPHA / V1_EAD_BETA)

把 `Δ = T − WRF_T2` 拆成 3 部分:
$$\Delta_{i,t} = \alpha_t + \beta_i + \varepsilon_{i,t}$$

**模型只学 ε̂**(残差),loss target = `y - wrf_t2 - α - β`,推理时 reconstruct T 然后报 RMSE。

| env 组合 | 后缀 | 说明 |
|---|---|---|
| `V1_EAD_ALPHA=0 V1_EAD_BETA=0` | (无)| **默认 baseline**,完全等同原 train/test |
| `V1_EAD_ALPHA=1 V1_EAD_BETA=0` | `_eadA` | 只 α 时间锚 |
| `V1_EAD_ALPHA=1 V1_EAD_BETA=1` | `_eadAB` | α + β 双锚(β 通过 kriging 传到 valid + unlabeled)|
| `V1_EAD_ALPHA=0 V1_EAD_BETA=1` | `_eadB` | 仅 β(罕见)|

### 6.1 关键实现细节

| 点 | 实现 |
|---|---|
| 空间 | 全部在 **target 归一化空间** ([0,1] with `tgt_scl=34.57`) |
| α_t 计算 | 严格只用 train labeled 站(spatial 模式排除 valid 8 站,防 leak)|
| β_i 计算 | train 站直接 `mean_t (Δ - α_t)`;valid + unlabeled 用 `kriging_beta()`(图加权平均)|
| WRF_T2 | 在 `_normf(wrf)` **之前** 抠 channel 0,转 °C 再用 `tgt_min/scl` 归一 |
| RMSE 报告 | **T 空间** `T_hat = wrf_t2 + α + β + ε̂`(与 baseline 同尺度可比)|
| 数学 | RMSE(T) ≡ RMSE(Δ) 在 normalized 空间内严格相等(WRF_T2 抵消)|

### 6.2 代码入口
- [code_V1/data.py](../code_V1/data.py):`kriging_beta()` / `compute_ead_anchors()` / `_dataGen_V2` 内 attach `data.alpha_t / beta_hat / wrf_t2`
- [code_V1/network.py](../code_V1/network.py):`train_ead()` / `test_ead()`
- [code_V1/run.py](../code_V1/run.py):env dispatch,EAD 启用时切换 train/test 函数

### 6.3 默认 off 兼容性
不设 `V1_EAD_ALPHA/BETA` → method_full 无 `_eadX` 后缀 → dispatch 走原 `train()/test()` → **完全等同 baseline**,现有所有非 EAD 实验不受影响。

### 6.4 物理含义 / 局限 / 改进方向

**直觉(α / β 是什么)**

- α_t:**全空间平均后的时间序列** —— "这个时刻 WRF 整体偏多少",反映 WRF 的时间系统偏差(凌晨偏冷、午后偏热等)
- β_i:**全时段平均后的站点画像** —— "这个站平均比 WRF 高/低多少",反映静态微气候(UHI、湖效应、地形)
- ε_i,t:**站×时交互 + 高频残差** —— α/β 一维平均拍不掉的部分,GNN 学这块

EAD 是**物理-ML 混合建模 / delta learning** 的标准做法:WRF 把方差大头(diurnal/synoptic, ~95%)解决,GNN 专注学剩余 5%,在小数据 + 强物理先验场景特别合适。

**局限**

1. **α / β 是纯可加分解** —— 看不到"密集城区夜间 UHI 比白天强"这类**站×时交互**(这部分被丢给 ε 全靠 GNN 容量学)
2. **α_t 全空间平均** —— 假设 WRF 时间偏差的形状对所有站一样,但湖边 vs 内陆其实不同
3. **β_i kriging 不用 aux** —— UHI 边界可以很锐利,kriging 的"图上平滑"假设对 unlabeled 站可能比 noise 还噪;理论上 aux(land cover、density)才是真决定 β 的物理量

**改进方向(优先级由高到低)**

| 思路 | 形式 | 复杂度 |
|---|---|---|
| **β_i = MLP(aux_i)** 替代 kriging | 让 β 从静态物理特征回归出来 | 低,改 `compute_ead_anchors` 即可 |
| **加交互项 γ_i,t** | `γ_i,t = MLP([aux_i, sin/cos(t)])`,unlabeled 自动可算 | 中,EAD 多挂一个 head |
| α_t 按 station 类型分组 | 按 land cover 聚类后 per-cluster α | 中 |
| 完全参数化 bias = MLP(...) | 放弃显式分解 | = baseline,无新增 |

短期不做。等 13860 / EAD-A / EAD-AB / Lap 出来后,根据 EAD 是否给增益判断:**EAD-AB > EAD-A > baseline 三阶递进** → γ 交互项有强动机;否则需重新评估方向。

### 6.5 为什么不做 input-augmentation consistency

图像分类 consistency 假设 `aug(x)` 不改变 label。**回归问题不适用** —— 几何/数值/时间 augmentation 都会改 target:

- WRF 加噪 → 真实 T 也变 → 假设破产
- 站点空间扰动 → target 直接换了
- 时间 shift → 不同时刻完全是另一个 target

**Physics-informed augmentation**(如 WRF 加 δ → 期望 T 跟随 δ)严格上不是 consistency 而是 **equivariance regularization**。在 EAD 已经显式建模 `T = WRF + α + β + ε` 的前提下,equivariance 几乎是冗余信息(模型本就会学到 T 跟随 WRF),边际增益小,不优先。

**真正可行的 consistency 方向**:**structural perturbation**(DropEdge / 节点 dropout / 子图采样)—— 扰动模型而非输入,target 不变,假设成立。如未来要做 consistency,从这条路开始,不从 input augmentation 开始。

### 6.5b EAD-Plus 实施(2026-05-10,V1_EAD_BETA_MODE=mlp)

[code_V1/data.py](../code_V1/data.py) 加 `_fit_beta_mlp()`:
- aux = UF(17) + GeoEmb(1008,可被 PCA 降)+ lat/lon(2)= **1027 维**
- MLP:`Linear(1027→64) → ReLU → Linear(64→1)`,Adam lr=1e-3 + weight_decay=1e-3,500 epoch
- 只在 train 50 站上 fit,然后 forward 全 458 节点得 β_hat
- env 开关 `V1_EAD_BETA_MODE=mlp`(默认 'kriging' 不变,完全向后兼容)
- 后缀:`_eadABplus` / `_eadBplus`
- **动机**:13876 EAD-AB(kriging)实证 β kriging 边际为负(α 红利从 -0.0054 → -0.0005),MLP 用 aux 直接学 β,可能修复

## 7. Laplacian Regularization(env: V1_LAMBDA_LAP)

在 sup loss 之上加图 Laplacian 平滑约束:
$$\mathcal{L} = \mathrm{Huber}(\text{output}_L, \text{target}_L) + \lambda_{\mathrm{lap}} \cdot \frac{1}{|E|}\sum_{(i,j) \in E} w_{ij} (\text{output}_i - \text{output}_j)^2$$

**默认 `V1_LAMBDA_LAP=0` (off)**;启用时设 > 0(V0 经验值 0.1)。

### 7.1 与 EAD 正交可叠加

Lap 是平滑 **模型输出**(不分 EAD on/off),所以:

| EAD | Lap | 模型输出 | Lap 平滑的对象 | 等价于 |
|---|---|---|---|---|
| ✗ | ✓ | ŷ = T | 平滑 T | "plain Lap on temperature" |
| ✓ | ✓ | ŷ = ε | **平滑 ε** | **V0 的 "residual Laplacian"**(物理最合理:WRF 已处理大尺度,只需平滑 UHI correction)|
| ✓ | ✗ | ŷ = ε | 不平滑 | 纯 EAD(α/β 锚消除大部分 variance,残差不强制平滑)|
| ✗ | ✗ | ŷ = T | 不平滑 | baseline |

**3 个 train 函数自动 dispatch**:
- `train_lap`:无 EAD + Lap → 平滑 T
- `train_ead`:有 EAD + 无 Lap → 不平滑
- `train_ead_lap`:有 EAD + 有 Lap → 平滑 ε

method_full 后缀**叠加**:`_eadAB_lap`(EAD α+β + Lap)、`_eadA_lap`、`_lap`、`_eadAB`、`_eadA` 等都合法。

### 7.2 实现位置

| 函数 / dispatch | 位置 |
|---|---|
| `train_lap()` / `train_ead_lap()` | [code_V1/network.py](../code_V1/network.py) |
| run.py 4-way 分支 | [code_V1/run.py](../code_V1/run.py):`(ead_active, lap_active)` 4 种组合 |
| 边列表预算 | run.py 训练循环前从 `metadata['AdjMatrix']` 抠 `(edge_src, edge_dst, edge_w)`,广播到每个 batch graph |

### 7.3 wandb 额外字段
启用 Lap 时,每 epoch 多 log:
- `train/sup_loss`(Huber 部分,EAD 时是 ε 空间,无 EAD 时是 T 空间)
- `train/lap_loss`(λ × mean_edges 部分)

### 7.4 调试输出
- 无 EAD:`[DEBUG/train_lap]` sup/lap/λ/total + yHat range + |ŷ_i−ŷ_j| max + edge_w 范围 + n_edges
- EAD + Lap:`[DEBUG/train_ead_lap]` sup(ε)/lap(ε) + ε range + T_hat range
- 都 assert NaN/Inf

## 7.5 Multi-View Consistency(env: V1_CONSISTENCY)

GRAND-style 图增强 consistency,**回归任务版**(MSE 替代 KL/CE):

```
每 batch:
  对 K=3 个 augmentation views 跑 forward
  view k 增强强度递增:
    k=0(弱): p_edge=0.20, p_node=0.05, edge_noise_std=0.05
    k=1(中): p_edge=0.35, p_node=0.10, edge_noise_std=0.10
    k=2(强): p_edge=0.50, p_node=0.15, edge_noise_std=0.15

  L_sup = Huber(yHat[view 0][label_mask], y[label_mask])
  L_cons = mean((yHat[view k] - yHat_mean.detach()) ** 2)
  L_total = L_sup + λ_cons × L_cons
```

**关键 LEAK 防护**:
- DropNode 不丢 train_mask=True 节点(防破坏 sup loss)
- 增强只动图结构 / node features 置 0,**不变 target T** → consistency 假设成立
- Cons loss 在所有节点算(包括 valid 的 features 路径,target 不参与)

实现:[code_V1/consistency.py](../code_V1/consistency.py)`graph_augment` + `train_consistency`,run.py `V1_CONSISTENCY=1` 启用,后缀 `_cons`。

**与 EAD/Lap 关系**:目前 consistency 只支持 baseline 路径(无 EAD/Lap)。叠加需要 `train_consistency_ead_lap` 等组合函数,后续按需加。

## 7.6 Adversarial Near-Valid Mask(env: V1_ADV_MASK)

**思路**:transductive learning 思想,**让训练时模型熟悉"valid 邻近区域信息缺失"的场景**。

```
Step 1(训练前一次):用当前模型 forward 提取每站 emb(128 维 hidden state)
                  valid_proto = mean(emb[valid 8 站])
                  对每个 unlabeled u:dist(emb[u], valid_proto)
                  取 top K=20 最近的 unlabeled → mask 候选集合 M

Step 2(每个 batch):
                  在 M 中随机选 p_mask=50% 的子集 → x[这些] = 0
                  正常 forward + sup loss(只在 50 train 算)
                  ❌ 没有 reconstruction loss / 没有新 head
```

**关键**:
- ✅ Mask 仅在 **unlabeled** 范围(idx ≥ nL=58),**绝不 mask labeled / valid**(代码 assert)
- ✅ 用 valid features / emb(no leak)选 mask 候选,**target 不接触**
- ✅ 训练时 backbone 学到"这些 valid-similar 区域信息缺失也能从邻居 MP 推断好" → valid 时鲁棒

**与 Mask Reconstruct 区别**:
- Adversarial mask = **input augmentation**(只 mask + 主任务,~150 行)
- Mask reconstruct = **multi-task aux**(mask + 加 reconstruction head + recon loss,~250 行)

实现:[code_V1/adv_mask.py](../code_V1/adv_mask.py)`compute_near_valid_mask_idx` + `train_advmask`,run.py `V1_ADV_MASK=1` 启用,后缀 `_advmask`。

**Env 超参**:
- `V1_ADV_MASK_K`(默认 20):mask 候选数量
- `V1_ADV_MASK_P`(默认 0.5):每 batch mask 候选的比例

**与其它正交**:目前与 EAD / Lap / Consistency 互斥(都改 train_fn);可叠加 self-train 因为 self-train 走独立分支。后续若需叠加 EAD,加 `train_advmask_ead` 组合函数。

## 7.7 Distribution Alignment(半 CycleGAN,env: V1_DIST_ALIGN)

**思路**:让 unlabeled 节点的 emb 分布 ≈ labeled(train+valid)节点的 emb 分布,通过 MMD loss 实现。**无对抗训练 / 无 discriminator,只加一个 MMD 正则项**,所以叫"半 CycleGAN"(完整 CycleGAN 需要 2G + 2D + cycle consistency,工程量 ~500 行)。

```
每 batch:
  yhat, hidden = forward_with_hidden(model, x, edges)
  emb_lbl = hidden[:, :n_labeled, :].mean(over batch)    # (n_labeled=58, HLD)
  emb_unl = hidden[:, n_labeled:, :].mean(over batch)    # (n_unlabeled=400, HLD)
  
  mmd = MMD(emb_lbl, emb_unl)                            # Multi-scale RBF kernel,median heuristic
  loss = sup_loss + λ_mmd × mmd
```

- **MMD = Maximum Mean Discrepancy**:无 discriminator 的分布距离度量
- Multi-scale RBF 核(median heuristic + [0.5×, 1×, 2×] median)
- λ_mmd 默认 0.1
- emb 用 processor 层输出(decoder 前的 hidden state)

**LEAK 规则**:
- ✅ 用 train+valid+unlabeled 的 features → embedding
- ❌ 不用 valid target(sup loss 仍只在 50 train 算)

**与 EAD / Lap / Consistency / AdvMask 互斥**(都改 train_fn),与 self-train 可叠(self-train 走外层)。

实现:[code_V1/dist_align.py](../code_V1/dist_align.py)`_gaussian_kernel`、`mmd_loss`、`train_dist_align`。env `V1_DIST_ALIGN=1`,`V1_DA_LAMBDA=0.1`,后缀 `_da`。

## 7.8 Mask Reconstruct(multi-task aux head,env: V1_MASK_RECON)

**核心动机**(2026-05-11 与用户讨论后补)。前面所有 semi-supervised 信号都有局限:
- **Laplacian**:间接约束 unlabeled,但只是"两节点相邻 → 表示相近"的平滑先验,**unlabeled 自己没有 target**;
- **Self-Train**:有 target 但被 **Pseudo Quality Ceiling** 限制(伪标签噪声 ≈ kriging 噪声);
- **Consistency**:多 view 一致性,但不告诉模型"应该是什么",只告诉"应该稳"。

**Mask Reconstruct 在 unlabeled 节点上提供 ground-truth target,且没有 quality ceiling:**因为重建对象是 features 自己(WRF / UF / etc),features 本来就是 ground truth。

**算法:**
```
每 batch:
  1. 从 unlabeled (idx ≥ n_labeled) 随机选 K 个节点
  2. 保留它们的 UF features (slice [322,339), 17 维静态) 当 recon_target
  3. 把这 K 个节点的 features 置 0(整行 mask)
  4. forward 模型 → (yhat, hidden)
  5. sup_loss = HuberLoss(yhat[label_mask], y[label_mask])     # 主任务,只在 train 节点
  6. recon_pred = ReconHead(hidden[mask_batched])               # 重建被 mask 的 features
  7. recon_loss = MSE(recon_pred, recon_target)
  8. total = sup_loss + λ × recon_loss
```

**为什么不会伤害 labeled 训练?**
- Mask 仅作用在 idx ≥ n_labeled(unlabeled 范围),labeled features 完全没动;
- sup_loss 在 label_mask(train 节点)上算,label_mask ∩ mask_batched = ∅;
- ReconHead 是独立 module,参数加入 opt 但 sup gradient 不流向 ReconHead 权重;
- 唯一 coupling:hidden representation 由 sup 和 recon 联合塑造 → 这是好事(让 hidden 同时编码"T 预测能力"和"feature 重建能力"),λ=0.1 控制 recon 不主导。

**为什么选 UF (17 维静态)而不是 WRF (315 维时序)或 GeoEmbed (1008 维)?**
- UF 是站点的物理属性(建筑高度、土地利用、距水/路距离等),**静态、低维、信号密度高**,从邻居 + WRF 即可推断,**重建难度刚好"中等"**;
- WRF 时序变化太快,重建本质要学 1D 时序模式,**噪声大、信号弱**;
- GeoEmbed 1008 维过高,17 维 target 被 1008 维稀释,**recon_loss 上 noise floor 高,梯度信号弱**;
- `V1_MR_TARGET=uf` 默认,可选 `aux`(4 维时间)/`clms`(3 维)/`uf_geo`(UF + GeoEmb 1025 维)做 ablation。

**ReconHead 架构:**Linear(HLD=128 → 64) → PReLU → Linear(64 → recon_dim=17)。<10K 参数,主模型 305K,**aux head 仅占 3%**,不会喧宾夺主。

**LEAK 规则:**
- ✅ Mask 只作用在 idx ≥ n_labeled(unlabeled),labeled/valid features 不动;
- ✅ Recon target 是 features 自己(no leak,features 训练时本来就有);
- ✅ Valid target 完全不参与(sup 和 recon 都不碰)。

**与其它方法的关系:**
| 方法 | unlabeled 信号源 | quality ceiling? | target |
|---|---|---|---|
| Lap | 邻居平滑 | — | 无 explicit target |
| Self-train | kriging / 自蒸馏 | **有**(kriging 噪声) | T(伪标签)|
| Consistency | 多 view 稳定性 | — | 无 explicit target |
| Adv Mask | mask 邻 valid → robustness | — | T(还是 sup loss)|
| **Mask Recon** | **features 自重建** | **无**(feature 是 ground truth)| **UF features**(17 维)|

**与 Adv Mask 的对比:**Adv mask 也 mask 一部分节点,但 mask 完之后仍只算 T 的 sup loss(在 labeled 上),所以本质是 **data augmentation / regularization**;Mask Recon 给 unlabeled 节点真正的 target(features),本质是 **multi-task learning**。Mask Recon 是 Adv Mask 的真正"升级版"。

**Hyperparameters:** λ=0.1(默认),K=30 unlabeled/batch,target=uf。13916 实验跑这个组合,基线 13860=0.0453;预期若有效 →[0.040, 0.045]。

**与 EAD / Lap / Cons / AdvMask / DA 互斥**(都改 train_fn);MR 优先级最高(放 dispatch 链最后)。**可与 self-train 叠**(self-train 走外层),但 13916 先单独跑看效果。

实现:[code_V1/mask_recon.py](../code_V1/mask_recon.py)`ReconHead`、`train_mask_recon`、`get_recon_feat_slice`。env `V1_MASK_RECON=1`,`V1_MR_LAMBDA=0.1`,`V1_MR_K=30`,`V1_MR_TARGET=uf`,后缀 `_mr`(`_mr_aux`/`_mr_clms`/`_mr_uf_geo` for 非默认 target)。

## 7.9 Modality-Aware Masked Pretraining(env: V1_PRETRAIN_MODE / V1_PRETRAIN_INIT,2026-05-13)

**动机**:基于深入分析(详见 [research_directions_analysis.md](research_directions_analysis.md)),所有 augmentation-based SSL 方法(Consistency / Mean Teacher / DA / Contrastive)都因小图 458 节点上"invariance 假设错误"失败。**Mask-based reconstruction pretext task** 是唯一**不依赖 invariance** 的路径 — 它要求"从 context 恢复信号",而非"对扰动 invariant"。

参考 STD-MAE([arXiv 2312.00516](https://arxiv.org/html/2312.00516v3)),但适配我们的**异构多模态 features**(5 个 modality)。

### 跟 13916 Mask Reconstruct 失败的本质差异

| 维度 | 13916(失败,+0.013)| 7.9(本次,14916)|
|---|---|---|
| 训练范式 | **多任务**(sup + recon 同时,gradient 竞争) | **两阶段**(pretrain only recon,then finetune only sup)|
| Mask 对象 | UF static(邻居站 UF 类似,**trivial 重建**) | **WRF non-Tair 动态(54 ch × 5 window = 270)+ CLMS 3 dim**(真有时序信号)|
| Static features 角色 | 被 mask | **保持可见当 context**(UF 17 + GeoEmb 1008 + station_aux 4 = 1029 维静态 context)|
| WRF Tair anchor | 被 mask 一起 | **永远可见**(5 window × 9 = 45 维 baseline anchor)|

### Mask 策略(V2 schema)

```
V2 iDim = 1347 拆分:
  WRF Tair(anchor,永不 mask):     5 window × 9 ch = 45 dim
  WRF non-Tair(可 mask,dynamic):   5 window × 54 ch = 270 dim
  station_aux(永不 mask,time idx): 4 dim
  CLMS(可 mask,dynamic):           3 dim
  UF(永不 mask,static context):    17 dim
  GeoEmb(永不 mask,static context): 1008 dim

Maskable total: 270 + 3 = 273 dim
Mask 25% per node ≈ 68 dims/node mask 掉(置 0)
Visible context per node: 1074 dim (79.7%)
```

### Two-Stage Pipeline

**Stage 1: Pretrain(env: `V1_PRETRAIN_MODE=1`)**
- 全 458 nodes 都参与(50 train + 8 valid + 400 unlabeled)
- 每 batch 随机 mask 25% maskable 位置 → 通过 GNN encoder + processor → per-modality decoder 重建
- 两个 decoder:`dec_wrf_dyn`(128 → 64 → 270),`dec_clms`(128 → 64 → 3)
- Loss = MSE on masked positions only(per-modality sum)
- 无 T label 使用
- 输出:encoder + processor 权重 ckpt(decoder + recon_heads 丢弃)

**Stage 2: Finetune(env: `V1_PRETRAIN_INIT=<path>`)**
- 加载 pretrain ckpt 中 encoder + processor 权重(initialized state)
- 加上原 GNN decoder + EAD + Lap 等标准 train_fn
- 正常 sup loss 训练 50 train 上
- 期望效果:**预训练的 encoder 学到更好的 representation**(尤其对 unlabeled 站点的 dynamic features 有更好的内部表示)→ fine-tune 收敛更好或 plateau 更低

### 防错 checklist(基于 skill discipline)

| 已验证 | 状态 |
|---|---|
| Pretrain sanity test(n_unl=10, 3 ep)| ✓ mask coverage 0.25 准确,loss 0.22→0.05 收敛 |
| Finetune sanity test(加载 ckpt + EAD+Lap)| ✓ 18 个权重正确加载,RMSE 正常下降 |
| 默认模式无影响(env 全空时 method name 不变)| ✓ `V2_semi_spatial_eadA_lap` 不变 |
| WRF Tair anchor 不被 mask | ✓ maskable_cols 不含 channel 0-8 of each window |
| Static UF / GeoEmb 不被 mask | ✓ maskable_cols 上界 322 |

### Env variables

```
V1_PRETRAIN_MODE=1           # 启用 Stage 1 pretrain(忽略 sup loss / EAD / Lap)
V1_PT_MASK_RATIO=0.25        # mask ratio of maskable positions per node
V1_PRETRAIN_INIT=<ckpt path> # Stage 2 finetune 时加载 pretrain 权重
```

后缀:`_pretrain`(Stage 1) / `_ftpt`(Stage 2 finetune from pretrain)

### 风险评估(诚实)

| 失败模式 | 概率 |
|---|---|
| 重蹈 13916 覆辙(虽然两阶段,但 representation 不通用)| 30% |
| Pretrain 收敛但 fine-tune 没改进(plateau 跟 13897 一样)| 30% |
| 改进 marginal(< 0.002)| 20% |
| 显著改进(0.002 ~ 0.005)| 15% |
| 大成功(> 0.005)| 5% |

预期 fine-tune RMSE vs 13897 baseline = 0.039186:
- 成功:[0.034, 0.038],Δ ∈ [-0.005, -0.001]
- 失败:[0.039, 0.042]

实现:[code_V1/pretrain.py](../code_V1/pretrain.py)`ReconHead`、`apply_mask`、`train_pretrain`、`get_maskable_ranges`。

## 8. GNN 层类型(env: V1_CONV_TYPE)

[code_V1/network.py](../code_V1/network.py) 的 `GNN` 现在支持 3 种 conv:

| `V1_CONV_TYPE` | conv 层 | 用 edge_attr? |
|---|---|---|
| `graphconv`(默认,faithful original_code)| `GraphConv(aggr='mean')` | ✓ |
| `sageconv` | `SAGEConv(aggr='mean')` | ✗(SAGE 不带边权)|
| `gat` | `GATConv(heads=4, concat=True)`,每 head 输出 HLD/4 维 | ✗(GAT 学注意力)|

forward 里按层类型 dispatch:GraphConv 三参数(`x, edge_index, edge_attr`),其它两参数。

## 9. 数据源(env: V1_DATASET)

| `V1_DATASET` | 数据 | 站数 | T | iDim |
|---|---|---|---|---|
| `V1`(默认)| GNN_N1_StationMat.mat + V1-aligned V2 unlabeled | 68 (+400) | 2948 | **1302** |
| `V2` | Labeled_Finalized_new.mat + V2 raw unlabeled | 58 (+N) | 3672 | **1347** |

V2 schema:WRF window 5×63 + station_aux 4(hour/month/year/station_id,代码 broadcast) + CLMS_t 3 + UF 17 + GeoEmbed 1008 = **1347**。详见 [data.md](data.md) §7.2。

## 9.4 实施进度日志(2026-05-09 to 2026-05-10)

简要记录所有已实现的 method,按提交顺序排:

| 日期 | 方法 | 文件 | jobid | 状态 |
|---|---|---|---|---|
| 05-09 | EAD α / α+β / Lap | data.py + network.py + run.py | 13875 / 13876 / 13877 | 完成,有结果 |
| 05-09 | GeoEmbed pool=6 | data.py + run.py | 13878 | 完成,有结果 |
| 05-09 | GeoEmbed PCA=256 | data.py + run.py | 13879 | 完成,有结果 |
| 05-09 | Self-train(self+neighbor / self+conformal / kriging) | selftrain.py + run.py | 13880-13882 | 13880 **bug 重提为 13887** |
| 05-09 | Self-train hybrid pseudo | selftrain.py | 13883 | 完成 |
| 05-09 | Per-modality encoder | network.py + run.py | 13884 | 完成 |
| 05-10 | EAD-Plus(β=MLP)| data.py | 13885 | 完成 |
| 05-10 | Multi-view consistency | consistency.py | 13886 | 完成 |
| 05-10 | Self-train fix(shuffle order bug)| selftrain.py | 13887 | bug 修复后重提 |
| 05-10 | Adversarial near-valid mask | adv_mask.py | 13890 | 完成 |

**关键 bug 修复**:

- **2026-05-10 `compute_self_pseudo` shuffle bug**:用 `trainLoader(shuffle=True)` 拿预测,但 `inject_pseudo_into_dataset` 按 trainSet 自然顺序写入 → pseudo 错位 → 13880 灾难性退化(0.0453→0.1401)。修复:`compute_self_pseudo` 强制创建 sequential loader(shuffle=False)。13880 取消,重提为 13887;13881/83 启动时已自动用修复版(实测 13881 第一 batch psd_loss=2.95e-5 ≈ 0,确认正常)。

- **2026-05-11 EAD + Self-train 集成 bug**:`selftrain.train_one_round` **完全没集成 EAD**,内部 loss = Huber(yhat, y) 直接,从未用 α/β/WRF 锚点。即使 env `V1_EAD_ALPHA=1`,EAD 在 self-train 路径下被静默忽略。13898 实证 v_rmse = 0.0428 完全 = 13883 Hybrid ST(无 EAD),证实 EAD 没生效。**修复(共 3 处)**:
  1. 新增 `train_one_round_ead`:模型输出 ε,loss = Huber(ε_hat, eps_target = y − WRF − α − β),支持 lap loss on ε
  2. `compute_self_pseudo` 加 `ead_active` 参数:EAD 模式下 reconstruct T_pseudo = WRF + α + β + ε̂(让所有 pseudo 统一在 T 空间)
  3. `selftrain_main` 加 dispatch:检测 ead_active 后,train 用 `train_one_round_ead`,test 用 `network.test_ead`
  4. R0 ckpt 修正:EAD + ST 实验应用 13875(EAD α best ckpt),不是 13860(无 EAD baseline)
  - 13898/13899 取消,重提为 13908/13909

### 9.4.1 为什么自蒸馏在我们任务上 Δ≈0 —— 根本原因(2026-05-11 实证后总结)

V1 实证(13881 / 13887)和 V0(13451 R1-R5)都给出 Δ≈0,原因**不是 labeled 数据小**,而是 3 个深层假设不满足:

| 文献自蒸馏涨点的前提 | 我们的情况 | 影响 |
|---|---|---|
| **大 overparameterized 网络**(数百万参数)| GNN 305K 参数 | 自蒸馏的"隐式正则"价值受限 |
| **分类任务**(soft label 有 "dark knowledge")| **回归任务**,只有 1 个数字 | **没有 soft 分布信息可平滑** —— 这是最致命的 |
| **未充分训练的 R0**(还有学习空间)| 13860 R0 best @ ep 770 **完全收敛** | **梯度 ≈ 0**(模型自己预测自己,无 loss)|
| 无其它 unlabeled 利用机制 | **naive semi 已经用 unlabeled MP** | 自蒸馏的"新信息"已被 MP 吸收 |

**精确结论**:自蒸馏在大过参分类网络 + 未充分训练 + 无 SSL 机制时涨点(BAN 2018, Mobahi 2020)。**我们 4 个前提全部反向**,所以 Δ≈0 是必然,**不是 bug 或 labeled 不够**。

→ Future:**任何"自蒸馏改进版"(Mean Teacher EMA / 多轮迭代等)在我们任务上大概率也 Δ≈0**;真要 self-train 涨点,**必须用外部 pseudo source**(我们的 kriging-pseudo 已实证有效)。

### 9.4.1c 关键 framing 洞察(2026-05-11 实证 + 讨论后总结)

**EAD 和"利用 unlabeled"几乎正交**:

| EAD 组件 | 用了 unlabeled 什么? |
|---|---|
| α_t 计算 | ✗ 只用 train 50 labeled 真值 |
| β_train 计算 | ✗ 同上 |
| β_hat at unlabeled | △ 仅用 unlabeled 的**位置**(adj kriging),不用 features |
| WRF_T2 at unlabeled | ✓ 进 GNN graph(naive semi 固有,不是 EAD 设计)|
| Loss | ✗ 只在 50 train labeled 算 |

→ **EAD 本质是"物理-aware supervised learning"**,12% 红利来自**物理 prior 拆分**,**与 SSL 维度独立**

**这给 paper 一个清晰 framing**:
- **维度 1:Physical prior**(EAD 残差分解)→ 单项 -12%
- **维度 2:SSL framework**(Naive semi + Self-train + Hybrid pseudo)→ 单项 -5~-6%
- **维度叠加**(EAD + ST):预期 -15~20%
- 两维度**正交独立改进**,合并最大化

注意:这意味着 **EAD 在 sup baseline 上也大概率给 -12%**(还未实测,13865 V2_sup_spatial = 0.0505 + EAD α 预期 ≈ 0.045)。这是个**还没做的对照**,值得补一个 V2_sup_spatial_eadA 实验来证明 EAD 与 SSL 维度正交。

### 9.4.2b 文档化假设:**无 metric self-train 必坍缩**(基于 V0 + 理论)

V1 阶段 self-train 所有实验都用了 3-metric 框架(防坍缩)。**"无 metric 直接做 self-train 会怎么样"V1 没实测**,但**V0 实测 + 理论分析都证明会坍缩**,所以**不需要 V1 重做**:

| 来源 | 证据 |
|---|---|
| **V0 实测** | `run_self_training.py` 的 8 个变体(`FILTER_MODE=none`,基础 + MC Dropout + graph_uncertainty + neighbor_error + conformal naive)**全部坍缩,未进 V0 排行榜** |
| **V0 progressive ST**(单一 confidence 阈值,无 div+rel)| 13027 / 13028 软坍缩至 0.047 ≈ naive semi |
| **理论分析** | 无 confidence 筛 → 选中错误 pseudo → 误差被自我强化 → 模型预测偏移 → 下轮 pseudo 错得更多 → 雪球式坍缩 |
| **V1 间接证据** | 13880(shuffle bug 导致 pseudo 错位,等价"随机 pseudo")**显示了完全相同的坍缩动态**:R0=0.0453 → R1 ep1=0.0617 → R1 ep200=0.1401(3x baseline)|

**结论文档化**:**3-metric 框架(confidence + diversity + relevance)是 self-train 的必要安全网**,**不必 V1 重做这个 ablation**(资源花在主线组合上更值)。如果 paper 审稿人要求,**V0 数据 + 理论分析 + 13880 间接证据** 已经足以论证。

### 9.4.3 待做的 self-train ablation(优先级低,主线饱和后做)

3-metric 框架(confidence + diversity + relevance)**不是 V1 创新**(V0 13451 已有,源自 active learning 文献)。我们用它是**防坍缩的安全选择**(V0 实证:无 metric 直接崩,单 metric 软崩)。**但每个 metric 个体贡献从未实证**。

待做 ablation(每个 1 个 sbatch,共 ~3-4 个):

| 实验 | 配置 | 预期 |
|---|---|---|
| 随机选 K(无 metric)| `V1_ST_TAU_QUANTILE=1.0, α_div=0, β_rel=0` | 大概率坍缩(V0 L1 已证)|
| confidence only | `α_div=0, β_rel=0` | 接近现状或微差(扎堆风险)|
| confidence + diversity | `β_rel=0` | 几乎 = 现状(rel 贡献小)|
| 完整 3-metric(已做)| 全默认 | 0.0428(参考)|
| τ_quantile 严格(0.3)| `V1_ST_TAU_QUANTILE=0.3` | 选更严谨,可能 ±0.001 |

**做这个的真实价值不在涨点,在 paper 严谨度**(读者会问"3 metric 每个都必要吗")。**优先级低**,主线(EAD+ST 组合)饱和后再补。

### 9.4.4 FPS pool 与 diversity 的隐性冗余(2026-05-12 补)

**观察**:我们的 unlabeled 池本身是 `FPS(2000 → 400, seed=0)` 一次性 max-coverage 选出来的 → **400 个候选已经预先 spread 开**。

```
Step 1(数据构造,一次性): FPS 从 2000 → 400  ← 已经做了一次全局 diversity
Step 2(self-train 每轮):  3-metric 从 400 → 40  ← diversity 是在已 spread 的池里"再选分散"
```

→ **Step 2 里的 diversity 大半是冗余的** — FPS 在 step 1 把"挑远的"任务先做掉了,3-metric 里的 α_div 实际能调整的空间很小。

**3-metric 里在 FPS 池上的真实有效性**:

| Metric | 在 FPS 池里是否仍有意义? | 原因 |
|---|---|---|
| **Diversity** | ⚠ **大半冗余** | FPS 已 max-spread,小范围 fine-tune |
| **Relevance**(靠近 valid) | ✓ 仍有效 | 400 个里有的离 valid 近,有的远 |
| **Confidence**(模型不确定度) | ✓ 仍有效 | 与空间位置无关,看 hidden 状态 |

**对实验现象的解释**:13911(`α_self=0.3`,偏 kriging)= 0.039877 = 13875 EAD α 单独 → 调 hybrid 比例无差异。**根因之一**:diversity 在 FPS 池里几乎不起作用 → 选哪些 pseudo 集合稳定 → 不同混合比下选到同一批节点 → 结果稳定。

**对 paper 框架的影响**(避免 over-claim):
- 不能写 "3-metric framework 是 ST 涨点关键" — 在 FPS 池上 diversity 本就被代偿
- 应写 "3-metric framework 是**防坍缩的安全网**,具体涨点来自 **外部 pseudo source(kriging)**,不是 selection sophistication"
- 这进一步印证 **Pseudo Quality Ceiling**:**选择再聪明也突破不了 pseudo source 的上限**

**潜在 future ablation**:
- 把 unlabeled 池从 400(FPS)扩到 2000(全集)→ diversity 才重新有真实作用空间
- 或干脆**只用 confidence + relevance 两个 metric**,删 diversity → 预计与现状几乎一致(因为本来就没起作用)

### 9.4.2 重要方向(用户特别强调,避免忘)

**两个核心未充分挖掘的方向**(2026-05-11 留备录):

#### 方向 A:**充分利用 unlabeled / aux 站点**(目前还没做的)

我们有 400 个 unlabeled 站点,目前用法:
- ✅ Naive semi MP(被动 message passing)
- ✅ Self-train pseudo label(40 个 × 5 round = 200 站参与 loss)
- ❌ 还可以做:
  1. **Mask + Reconstruct as multi-task aux**(unlabeled 重建 features 当 aux head loss)
  2. **节点级 contrastive**(aug + 对比损失,只在 unlabeled 上算)
  3. **Heterogeneous graph**(labeled vs unlabeled 用不同 encoder)
  4. **Iterative pseudo refinement**(每 N epoch 重算 pseudo,类似 progressive)
  5. **Spatial autoencoder pretraining**(用 unlabeled 做无监督预训练)

#### 方向 B:**充分利用 valid 站点的 feature / location(target 仍不能用)**

Valid 8 站,目前用法:
- ✅ Naive semi MP(features 进 graph 已被用)
- ✅ Self-train relevance metric(用 valid emb 选 pseudo 候选)
- ❌ 还可以做:
  1. **Distribution alignment**:unlabeled emb 向 valid emb prototype 对齐(MMD / DANN-style)
  2. **Adversarial near-valid mask**(已做,Δ≈0,弱)
  3. **Cross-attention with valid as memory**:valid features 当固定记忆,unlabeled / train attend to them
  4. **Iterative inference at test time**:用 valid prediction 当 soft pseudo 反馈给 unlabeled,迭代收敛
  5. **Test-time augmentation**:推理时多次 augmented forward,平均预测 valid
  6. **Active sample re-weighting**:train 样本按"对 valid 预测贡献"加权

→ 这两个方向都有潜力,但我们之前实验的几个尝试(adv mask、distribution alignment 没做、TTA 没做)**信号不强或没测**。建议在 EAD α + 组合主线之外**逐个验证**,优先级中。

## 9.5 Self-Training Roadmap(下一阶段,尚未实现)

### 9.5.1 V0 经验汇总(避坑指南)

V0 三层 self-train 全部尝试过,结论:

| 层级 | 设计 | 结果 |
|---|---|---|
| **L1**:裸 self-train(`run_self_training.py`) | `FILTER_MODE` 切 4 种 confidence,所有 unlabeled 同时入 | **全部坍缩 / 不上榜** |
| **L2**:Progressive top-K(`run_progressive_st.py`) | 单一 confidence 阈值,无 diversity | **软坍缩**(0.047,几乎 = naive semi)|
| **L3**:3-metric greedy + round + warm-start(`run_selftrain_iter*.py` / `_v3.py`) | confidence + diversity + relevance,round-by-round,warm-start from best | **唯一稳的设计**(0.0428–0.0445)|

### 9.5.2 Confidence 4 选 2

V0 测过 4 种 confidence,结论:

| confidence | V0 在哪测过 | 是否 L3 框架内公平测过 | V1 是否继续 |
|---|---|---|---|
| **邻居 error**(labeled 邻居真实 error 的 kNN 加权)| L1 + L2 | **从未**(L1/L2 框架已知必崩,推不到 L3)| ✓ **保留**,首选 |
| **Conformal 5-fold OOF**(emb-kNN to train OOF residual)| L3 V3(13451)| 测过,但 R0=0.0428 后 **R1–R5 Δ=0**,迭代红利没拿到 | ✓ **保留**,二选 |
| Ensemble snapshot(cyclic LR snapshot std)| L3 V1/V2(13379=0.0436)| 测过,**性价比一般** | ✗ 不再做 |
| Het β-NLL(oDim=2 学 σ²)| L3 V3(13414/13452 ERROR)| 测过,**σ-collapse 必死**(50 train 站撑不起 σ-head)| ✗ 不再做 |

**邻居 error 的"未测"特别重要** —— V0 把它和 L1/L2 弱框架绑死,**从未在 L3 (3-metric + warm-start) 框架下公平测过**,完全可能是被框架而非 confidence 本身拖垮。V1 阶段值得复活。

### 9.5.3 V1 self-train 设计原则

| 原则 | 含义 |
|---|---|
| **模块化** | self-train 作为独立 plugin,不依赖 EAD / Lap |
| **隔离测试** | 先在**纯 V2_semi baseline(13860)**上单独测 self-train 收益,不混合 EAD/Lap |
| **后续叠加** | self-train 跑通后再与 EAD / Lap 组合,验证正交性 |
| **R0 base = naive semi(13860)** | 不用 sup R0(图拓扑变化破坏 warm-start;confidence 计算需要 unlabeled embedding)|
| **不重测已知失败路径** | 不再做 L1 plain self-train、Ensemble、Het |

### 9.5.4 EAD / Lap 与 self-train 的关系(更严谨表述)

V0 排行榜第 1(13451 = 0.0428)是 **EAD + Lap + Conformal self-train**,但 R1–R5 Δ=0 → **迭代轮没贡献,功劳全在 R0 base**。

| 组件 | 对 self-train 鲁棒性的影响 |
|---|---|
| **EAD residual** | 几乎正交(只让 R0 起点更好,不改迭代轮稳定性;ε 空间和 T 空间的 pseudo error 1:1 传导)|
| **Lap regularization** | **双刃剑**:pseudo 准时把信号传邻居(好);pseudo 错时**把错误扩散一片 unlabeled 邻居形成错误共识**(糟,尤其图中 unlabeled:labeled=7:1 时)|

→ V1 阶段先**关闭 EAD/Lap 干净测 self-train**,避免组件混淆。

### 9.5.5 V1 实施目标矩阵

| backbone | + 邻居 error ST | + Conformal ST |
|---|---|---|
| pure baseline(= 13860) | **TODO 优先** | **TODO** |
| + EAD | 后续 | 后续 |
| + Lap | 后续 | 后续 |
| + EAD + Lap | 后续(对应 V0 13451) | 后续 |

### 9.5.6 实现核心组件(建议优先级)

| 组件 | 必须? | 备注 |
|---|---|---|
| round-by-round + warm-start from best | ✓ | V0 from-scratch 已证不稳 |
| 3-metric greedy_select(conf + div + rel) | ✓ | V0 单 conf 已证扎堆软坍缩 |
| Confidence:邻居 error(kNN on adj_matrix)| ✓ 第一步 | 工程量小,首选验证 L3 框架 |
| Confidence:Conformal 5-fold OOF(emb-kNN)| ✓ 第二步 | 替换 confidence 子模块即可 |
| Pseudo-label loss(Huber on pseudo nodes)| ✓ | 核心 |
| EAD residual / Lap | ✗ | 先关闭,跑通后再叠加 |

### 9.5.7 V0 vs V1 self-train 设计差异(关键区分)

| 项 | V0 | V1 计划 |
|---|---|---|
| backbone | EAD + Lap(强先验)| 纯 baseline(隔离 self-train 效应)|
| confidence 优先 | Conformal(13451 是最优)| 先邻居 error,再 Conformal |
| 主要待证明问题 | "self-train 在强先验上能否再锦上添花" → 答:Δ=0 | "self-train 在干净 baseline 上能否独立带增益" |

### 9.5.8 Kriging 方法 + sanity benchmark 重大发现(2026-05-09)

**Kriging 实现**(IDW = Inverse Distance Weighted,经典空间插值):

对每个目标站 v(unlabeled / valid),用 k 个最近 train labeled 站做加权平均:

```
1. d_l = sqrt((v.lat - l.lat)² + (v.lon - l.lon)²)    for l in train  (欧氏 lat/lon 距离)
2. nearest_k = argsort(d_l)[:k]                         (取 k 最近,默认 k=10)
3. w_l = 1 / (d_l + 1e-6)  for l in nearest_k          (距离倒数权重)
4. w_l /= Σ w_l                                         (归一化)
5. ŷ_kriging[v, t] = Σ w_l × y_true[l, t]              (按 t 逐时刻加权平均)
```

**关键性质**:
- 不用任何模型,不用任何特征(WRF / aux / GeoEmbed 都不用)
- 只依赖 train labeled 的 ground truth + 站点 (lat, lon)
- 完全 deterministic,可在训练之外离线计算

跑 [code_V1/sanity_kriging.py](../code_V1/sanity_kriging.py)(IDW kriging 用 50 train 预测 8 valid,k=10):

```
Model 13860 baseline valid RMSE = 0.0453 (1.57°C)
Kriging                valid RMSE = 0.0437 (1.51°C)   ← 反超模型!
Δ (kriging − model)             = -0.0016
```

**含义**:
1. **简单空间插值在 V2 spatial 任务上比 GNN 模型更准** —— 模型没充分利用 WRF + aux 高维特征
2. **Kriging-pseudo 实验 B 真有涨点希望** —— 不再只是"独立信号但 noisy"的辩护逻辑
3. 修正之前的判断:不是 "kriging 太 noisy 没法当 pseudo",而是 "kriging 应该当主 pseudo 之一"

**实验优先级(修正后)**:

| 优先级 | 实验 | 期望 |
|---|---|---|
| ⭐⭐⭐ | **B:Kriging-pseudo + 简单结构 confidence** | **可能 Δ < 0,真涨点(kriging > 模型已实证)** |
| ⭐⭐ | A1:self + neighbor_error | Δ ≈ 0(完成 ablation)|
| ⭐ | A2:self + Conformal | Δ ≈ 0(对照 A1)|

### 9.5.8b 🚫 V0 已试过失败的方向 —— 不再重复

下表记录 V0 已经做过且**结果明确**的尝试,V1 阶段不重做(避免浪费 GPU + 重蹈覆辙):

| 失败方向 | V0 jobid / script | 失败模式 | V1 阶段是否重做 |
|---|---|---|---|
| Plain self-train(无筛选)| self_training/basic.sh | 直接坍缩 | ❌ 不做(V0 已实证)|
| MC Dropout confidence | self_training/mc_dropout.sh | GNN 无 Dropout → σ≈0,等同 none | ❌ 不做(架构上不可行)|
| graph_uncertainty | self_training/graph_uncertainty.sh | 实际是 Lap 伪装,非真 uncertainty | ❌ 不做(信号无意义)|
| **Het β-NLL head**(σ²-output) | st_v3 13414 / 13452 | **σ collapse 必死**(50 train 撑不起),β-NLL 救不回 | ❌ **绝对不做** |
| Snapshot Ensemble from-scratch | st_iter 13361 / 13374 | ERROR,from-scratch 不稳 | ❌ 不做 V1(必须 warm-start);V0 已证 V2 warm-start 仅给 0.0436,增益小 |
| Conformal R1-R5 在 EAD+Lap 上 | st_v3 13451 | R0=0.0428 OK,**R1-R5 Δ=0**(自蒸馏在强 backbone 上饱和) | ⚠ **不在 EAD+Lap backbone 上做**;V1 改在**纯 baseline 上做 Conformal**,看自蒸馏在没强 backbone 时是否能贡献 |
| **neighbor_error 在 L3 框架** | (V0 未做)| 只在 L1/L2 测过(都崩),L3 框架未公平测 | ✅ **V1 重做**(主推),A1 实验 |
| **kriging-pseudo + 结构 confidence** | (V0 未做)| V0 只用 self-pseudo,没试过外部 pseudo | ✅ **V1 重做**(主推),B 实验 |

**V1 self-train 的 3 个实验**(仅这 3 个值得做):
- **A1**:self + neighbor_error(在 L3 框架,V0 没公平测过)
- **A2**:self + Conformal(在纯 baseline 上,V0 只在 EAD+Lap 上测)
- **B** :kriging-pseudo + structural confidence(完全新方向)

**所有 self-train 实验都要满足**:
- 起点 = 13860 best ckpt(`log_V1/13860_V2_semi_spatial/V2_semi_spatial.pt`)
- warm-start lr=1e-4(不允许 from-scratch,V0 已证不稳)
- 3-metric greedy_select(confidence + diversity + relevance)
- λ_psd=0.5,K_per_round=40,N_rounds=5
- 不叠加 EAD / Lap(隔离 self-train 自身效应)

**🔑 LEAK 规则**(严格执行):
- ✅ **可以用** valid 站的 feature / embedding(forward 出的中间量)→ relevance、diversity 用 valid emb 没问题
- ❌ **不可以用** valid 站的 target(y)→ Conformal 5-fold OOF 必须只在 train 50 内做,不能 hold-out valid
- 理由:naive semi 本来就让 model 通过 MP 看到 valid features,transductive SSL 允许;**只有 target 才是真 leak**

### 9.5.8c 实施记录(2026-05-10)

- **`code_V1/selftrain.py`** 完整实现(~580 行):
  - 4 个 confidence 函数:`compute_neighbor_error_confidence`、`compute_conformal_confidence`(简化版,用 R0 in-sample residual 当 OOF 近似,完整 5-fold 工程量大,后续补)、`compute_kriging_struct_confidence`
  - 2 个 pseudo 来源:`compute_self_pseudo`(模型当前预测)、`compute_kriging_pseudo`(IDW 从 train labeled 真值)
  - `greedy_select_3metric`:τ-quantile 过滤 + 贪心 div+rel 选 K
  - `train_one_round`:Huber(train_mask) + λ × Huber(pseudo_mask),复用 `network.test()` 验证
  - `selftrain_main`:orchestrator,加载 R0 ckpt → N 轮 → 跨轮 best ckpt 持久化
  - `extract_embeddings`、`inject_pseudo_into_dataset`、`ensure_selftrain_masks` 辅助函数
  - `[DEBUG/ST_*]` 调试输出贯穿(confidence stats、selection list、pseudo stats、loss 分项)
- **`code_V1/run.py`** 加 `V1_SELFTRAIN=1` dispatch + 11 个 ST env(详见 docstring)+ method_full 后缀 `_st_{pseudo}_{conf}`
- **`code_V1/data.py`** metadata 加 `train_station_idx` / `valid_station_idx` / `locations` / `targets_norm_full`(self-train 需要,无 leak 影响)
- **3 个 sbatch 脚本**:V2_self_neighbor.sh、V2_self_conformal.sh、V2_self_kriging.sh
- **Sanity 通过**:n_unl=20 + K=4 + N=2 + ep=2 微型测试,3 个路径全 Exit 0,机制全部走通
- **Submitted jobs**:13880 (A1) / 13881 (A2) / 13882 (B)

### 9.5.11 后续 Self-train 可做的改进方向(2026-05-11 整理)

**🔑 核心洞察:Pseudo Quality Ceiling**

```
Pseudo source quality 限制了 ST 的天花板:
  if model_current > pseudo_source_quality:
      ST 无效或退化(pseudo 把 model 往后拉)
  else:
      ST 有效(pseudo 提供 above-current 信号)

实证:
  - Naive baseline(0.0453) + Kriging-pseudo(0.0437 kriging quality) → 涨点 -0.0024 ✓
  - EAD α(0.0399) + Kriging-pseudo(0.0437)→ 不涨(13908 实证)✗
```

→ **要 ST 在 EAD 上继续涨点,必须找到 quality > 0.0399 的 pseudo source**。

**改进方向分 3 类**:

#### 类 A:**内部超参微调**(便宜,期望 ±0.001)

| 改进 | env | 期望 |
|---|---|---|
| α_self 扫(0.3 / 0.7) | `V1_ST_HYBRID_ALPHA` | 找最优混合比 |
| K_per_round 扫(20 / 60 / 80) | `V1_ST_K_PER_ROUND` | 选 pseudo 节点数 |
| N_rounds 增加(8 / 10) | `V1_ST_N_ROUNDS` | 更多轮 |
| τ_quantile 严(0.3) | `V1_ST_TAU_QUANTILE` | 严选 |
| λ_pseudo 扫(0.3 / 0.7) | `V1_ST_LAMBDA_PSEUDO` | pseudo 权重 |
| **3-metric ablation**(只 conf / no rel 等)| `V1_ST_ALPHA_DIV` / `V1_ST_BETA_REL` | 个体贡献 |

#### 类 B:**改 pseudo source**(中等成本,中等期望 ±0.001-0.005)

| 改进 | 工程量 | 期望 |
|---|---|---|
| **ε 空间 kriging**(直接 IDW kriging ε_train,不经过 T 空间转换)| ~30 行 | 微弱,**Kriging quality ceiling 不变** |
| **EMA teacher pseudo**(Mean Teacher 风格,teacher=student EMA)| ~150 行 | 半破自蒸馏,可能 -0.001 |
| **真 5-fold Conformal**(替代当前 in-sample 简化版) | ~200 行 | 严格性提升,效果未知 |
| **Pseudo refresh during round**(每 N epoch 重算,不固定一轮 ) | ~80 行 | progressive 风格 |

#### 类 C:**根本性改 pseudo source**(高成本,高期望 -0.002~-0.005)

要**突破 kriging quality ceiling(0.0437)**,必须用比 kriging 更准的 pseudo:

| 候选 | 工程量 | 信息来源 |
|---|---|---|
| **Ensemble teacher pseudo**(多个 13875-style EAD α 模型 ensemble 平均当 pseudo)| 高 | 模型 ensemble 通常 ~0.5-1% 涨点 |
| **EAD γ 交互项 + ε kriging**(把 γ 加上去再 kriging)| 中-高 | γ 物理 prior 升级 |
| **多源融合 pseudo**(WRF + reanalysis + 卫星 LST)| **数据依赖,可能拿不到** | 真新物理信息 |
| **SLUCM 等城市冠层模型校正后当 pseudo** | 极高(物理建模)| 真物理模型 |

#### 优先级(基于 13908 / 13910 / 13912 结果待定)

```
等 13910 / 13912 结果(若都 ≈ 0.0399):
  确认 "ST 在 EAD 上饱和"
                ↓
按此优先级试:
  ⭐⭐ 1. EMA teacher pseudo(半破自蒸馏)
  ⭐⭐ 2. ε 空间 kriging(简单变体)
  ⭐  3. Ensemble teacher pseudo(工程量大但稳)
  ⭐  4. 类 A 内部超参扫(性价比低,但 paper 严谨)
  ✗  5. 类 C 多源数据(数据获取难)
```

### 9.5.10 Hybrid Pseudo(实施于 13883)

在 self-train 框架内加 `pseudo_source='hybrid'`:

```
pseudo[u, t] = α_self × model_pred[u, t] + (1 - α_self) × kriging_pseudo[u, t]
```

- α=1.0 退化到自蒸馏(等价 13880/13887)
- α=0.0 退化到纯 kriging(等价 13882)
- α=0.5(13883 默认):**两端各占一半**,模型 anchor + 外部信号都拿到

env:`V1_ST_PSEUDO_SOURCE=hybrid`,`V1_ST_HYBRID_ALPHA=0.5`,后缀 `_hyb_kstruct`(confidence 用 kriging_struct)。

### 9.5.9 实验 B 的精确设计(修正版)

由于 kriging 单独已经比模型准,**不必做"hybrid pseudo (0.7 model + 0.3 kriging)"** —— 直接用 kriging 当主 pseudo 即可:

```
# 每轮 self-train:
  1. 选 K 个 unlabeled(用结构性 confidence:邻居 labeled 边权和)
  2. 给它们的 pseudo target = kriging from train labeled (IDW from k=10 nearest)
  3. label_mask 翻 True
  4. warm-start lr=1e-4,继续训 200 ep
  loss = Huber(50 train) + 0.5 × Huber(K kriging-pseudo)
```

**Kriging confidence(代替 conformal / neighbor error)**:

```
conf(u) = Σ_{l ∈ k_nearest_train_to_u} weight(u, l)
```

直观:周围 train 站越多越近 → kriging 越准 → confidence 越高。这个 confidence **不依赖模型**,完全结构性,与 kriging-pseudo 一致。

## 9.6 GeoEmbed 维度控制(env: V1_GEO_POOL_SIZE)

### 9.6.1 为什么保留 GeoEmbed,不直接 drop

GeoEmbed 来自 `UrbanFeatureMat (401, 401, 7, N)` —— **每个站点 401×401 高分辨率 7 通道城市形态学栅格**(建筑高度、密度、不透水面、植被等)。这是 V2 数据**最独特、最值钱的部分**:

- **高空间分辨率(米级)**:能捕捉 street-level 微气候(街道走向、建筑投影、单个公园边界)
- **物理直接相关**:7 个通道直接编码影响城市热岛的物理因子(albedo、heat capacity、roughness)
- **不可替代**:WRF / CLMS 是公里级网格,UF 只是 17 维聚合统计 → 没 GeoEmbed 就没办法做精细到街区的 downscaling
- **本任务的核心创新点**:之所以做"城市温度空间下推",前提是有这样的细粒度物理数据;扔了等于放弃任务

**关于 GeoEmbed 与 UF 的关系**:两者数据源同(UrbanFeatureMat),但**信息粒度不同**:
- **UF (17 维)**:全栅格的**空间聚合统计**(站点周围全区域的均值、方差等),粗粒度
- **GeoEmbed (7×p² 维)**:**保留 p×p 网格的空间分布**(12×12 → 7 通道在网格内不同位置的池化),细粒度

所以 GeoEmbed **不是 UF 的简单冗余**,而是"UF 的空间结构版本"。即使 UF 已经包含了均值,GeoEmbed 还能反映"建筑密度往哪边偏"这种**站点周围的非均匀分布**,后者对 street-level 局部预测至关重要。

**所以 ablation 方向不是 drop,而是降维**(保留细粒度但减少冗余像素)。

### 9.6.1b 经验观察:当前 pipeline 没把 GeoEmbed 价值挖出来(2026-05-09)

**关键证据**:Kriging benchmark = 0.0437 反超模型 baseline = 0.0453。kriging 完全没用 GeoEmbed(也没用 WRF / UF 等任何特征,只用 train labeled 真值的空间插值),却比有 1300+ 维特征的 GNN 模型还准。

**这说明**:
- 不是"GeoEmbed 没价值",而是**当前架构没把它的价值挖出来**
- 主要怀疑:1008 维 dominate 第一层 Linear 容量,WRF / UF 的真信号被稀释
- 验证方向:降维 ablation(g6/g8 + PCA)+ 未来 per-modality 编码架构

### 9.6.2 实现:`V1_GEO_POOL_SIZE` env

UFM 通过 `AdaptiveAvgPool2d((p, p))` 池化为 `(N, 7, p, p)` → flatten 为 `(N, 7×p²)` 维 GeoEmbed:

| `V1_GEO_POOL_SIZE` | GeoEmbed 维度 | 总 iDim(V2 spatial)| 后缀 | 备注 |
|---|---|---|---|---|
| **12**(默认)| 1008 | 1347 | 无 | baseline,13860 用的 |
| 10 | 700 | 1039 | `_g10` | 中等降维 |
| **8** | **448** | **787** | `_g8` | **推荐先试**(降 ~55%)|
| **6** | **252** | **591** | `_g6` | **更激进**(降 ~75%)|
| 4 | 112 | 451 | `_g4` | 最激进,可能丢信号 |

### 9.6.3 假设与预期

**假设**:GeoEmbed 1008 维里大部分是空间相邻像素的冗余信息,**降维不会丢核心信号,反而能让模型聚焦其它特征**。

**预期** v_rmse 变化(在 13860=0.0453 基础上):

| 池化 | 预期 | 解读 |
|---|---|---|
| 12 (1008) | 0.0453(参考) | 当前 |
| 8 (448) | -0.002 ~ +0.002 | 大概率持平,可能略好(信号没丢 + 噪声减少)|
| 6 (252) | -0.001 ~ +0.005 | 中等概率持平,可能略差(开始丢信号)|
| 4 (112) | +0.003 ~ +0.010 | 大概率显著变差(过度压缩)|

如果 8 和 6 都比 baseline 好 → **降维有用,GeoEmbed 是部分冗余**。
如果 8 持平、6 变差 → **8 是甜蜜点,12 太冗余但 6 太激进**。
如果都变差 → **GeoEmbed 1008 维是必要的,模型确实在用全分辨率**。

### 9.6.4 未来方向:per-modality 编码架构(尚未实现)

当前架构问题:

```
[WRF 315 | CLMS 3 | UF 17 | GeoEmbed 1008 | aux 4]  ← 直接 concat
              ↓
       Linear(1347 → 128)  ← 第一层 172K 参数(占模型 57%),
                              其中 ~133K 处理 GeoEmbed,WRF 这个真信号
                              可能被"稀释"在 noise 里
```

**问题**:
1. 不同模态尺度差异大(WRF window 315 vs GeoEmbed 1008)→ 第一层 Linear 容量被低信号特征 dominate
2. 失去模态结构(WRF 是时间窗、GeoEmbed 是空间池化,flatten 后丢了结构)
3. 高维特征 noise 拖累低维真信号

**改进方案 1:Per-modality MLP**(推荐先做,工程量低)

```python
class ModalEncoder(nn.Module):
    def __init__(self, dims=[315, 3, 17, 1008, 4], hid=32):
        self.encoders = nn.ModuleList([nn.Linear(d, hid) for d in dims])
        self.fusion = nn.Linear(len(dims) * hid, 128)

    def forward(self, x):
        # x = (N, sum(dims)=1347), 按 dims 拆分
        parts = []
        offset = 0
        for enc, d in zip(self.encoders, dims):
            parts.append(F.prelu(enc(x[:, offset:offset+d])))
            offset += d
        return self.fusion(torch.cat(parts, dim=-1))
```

每个模态独立编码到 32 维 → 各占 33% 容量 → concat 后 fusion 到 128。第一层参数从 172K → 63K(减 1/3),且 GeoEmbed 不再 dominate。

**改进方案 2:CNN for GeoEmbed**(中等工程)

把 GeoEmbed 还原成 (N, 7, 12, 12) 走 2D Conv → 32 维。利用 GeoEmbed 的空间结构。

**改进方案 3:Transformer fusion**(高工程)

每个模态当一个 token,cross-attention 学模态间关系。

**实施进度**:
- ✅ **方案 1 Per-modality MLP 已实施**(env `V1_ENCODER_TYPE=per_modality`,后缀 `_pmod`,jobid 13884)。第一层参数 172K→43K,可与 13860 baseline 直接对比。
- ❌ 方案 2 CNN-on-GeoEmbed:未实施(等 13884 结果决定)
- ❌ 方案 3 Transformer fusion:未实施

## 11. Future Work / 论文升级方向

### 11.1 投稿 venue 定位

| 投稿目标 | 可行性 | 备注 |
|---|---|---|
| **Urban / Atmospheric / Geoscience 期刊**(Urban Climate / JGR Atmospheres / Bull AMS / IEEE TGRS)| ✓ **大概率接收** | 当前工作完全够用 |
| **AI for Earth/Climate 工作坊**(NeurIPS Climate Change AI / ICLR Earth System ML)| ✓ 可行 | 应用 + 严谨 empirical 价值高 |
| **ML 主会**(NeurIPS / ICML / ICLR)| ✗ **困难** | Novelty 不够,主要是组合现有方法 |

### 11.2 当前工作的原创性盘点

| 方法 | 原创度 |
|---|---|
| **EAD 残差分解**(α + β + ε,物理可解释)| ⭐⭐ 中等(delta learning 概念 + 任务特定分解)|
| **Hybrid pseudo**(self + kriging 混合)| ⭐ 弱(组合)|
| 其它(Lap / consistency / mask / per-modality / 等)| ✗ 都是文献借鉴 |

### 11.3 要冲 ML 主会还需要补的方向

| 补充方向 | 工程量 | 提升评级 |
|---|---|---|
| **理论分析**:证明 EAD 残差分解在某些假设下最优(类比 Beucler 2021 物理-ML reasoning)| 高 | ⭐⭐⭐ |
| **多城市 generalization**:Chicago → Beijing / Tokyo / Paris + transfer learning ablation | 极高(数据获取难)| ⭐⭐⭐ |
| **新架构**:把 EAD 从 loss-level 改成 architecture-level(物理 prior 层而非 loss 项)| 极高 | ⭐⭐⭐ |
| **Causal inference framing**:把 EAD 解释为 do-calculus 下的 confounder removal | 中 | ⭐⭐ |
| **新 benchmark**:发布 urban T downscaling 公开 benchmark | 中(长期有价值) | ⭐⭐ |

### 11.4 推荐两阶段路线

```
Stage 1(当前):Urban 期刊
  - V1 工作扎实,直接投 Urban Climate / JGR Atmospheres
  - 强调:严谨 ablation + 物理可解释 + 14%+ 改进
                ↓
Stage 2(后续):ML 主会 "deeper version"
  - 加 1-2 个上面的强 novelty 项
  - 不是同一篇,是"加 substance 后的延伸"
```

### 11.5 ML novelty 评分(诚实估计)

| 维度 | 当前 | 加 Stage 2 后 |
|---|---|---|
| Novelty | 5/10 | 6.5-7.5/10 |
| Significance | 7/10 | 8/10 |
| Technical | 8/10 | 8.5/10 |
| Clarity | 写作决定 | 同 |

ML 主会平均接收门槛 ≈ 6.5+。**当前 5.5-6/10,门槛附近偏低**;加 Stage 2 项后 ≈ 7+/10,有戏。

## 10. 输出 / wandb 约定

- 每个 job 输出目录:`log_V1/{jobid}_{method_full}/`(如 `log_V1/12345_V1_sup_random/`)
- SLURM stdout/stderr:`slurm_output_V1/result/{jobid}_V1_{method}_{val}_result.txt` / `_error.txt`
- 文件夹内:`{method_full}_log` / `{method_full}_hist.png` / `{method_full}.pt` / `{method_full}_param.pkl`,spatial 模式下还有 `spatial_split.png`
- wandb:`urban_prediction/V1_baseline`,run name = `{jobid}_{method_full}`
  - 关闭可设 `V1_WANDB=0`

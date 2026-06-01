# V1 / V2 实验结果

每跑完一个 job 就更新对应行。状态约定:**DONE / CANCEL-PART / RUNNING / PD / TODO**。

**wandb 项目分布**:
- `urban_prediction/V1_baseline` — V1 baseline jobs(13848-13853)
- `urban_prediction/urban` — V2 jobs(13854-13865 + 后续)

⚠ **历史命名 bug**:13852 / 13853 因 `run.py` 和 `data.py` 的 `V1_DATASET` 默认值短期不一致(已修),wandb 上 run name 仍显示 "V2_..." 但**实际数据是 V1**,数字可信。本地 `log_V1/` 目录已重命名为 `13852_V1_semi_sequential/` 和 `13853_V1_semi_spatial/`。

---

## 📊 当前状态(2026-05-13 最新)

### 🏆 V2 spatial 最强:**13897 EAD α + Lap = 0.039186 (1.36°C, −14% vs baseline)**

### 🟢 RUNNING / 待回填
- **14916** Modality-Aware Masked Pretraining(Stage 1)— 2026-05-13 提交,2000 ep,~3-4h
- Stage 2 (eadA_lap_ftpt) 待 14916 完成后自动提交

### 📉 已实证失败方向(11 个,**不再投入**)
1. **GNN layer swap**(GAT/SAGE 在 semi setting):13958-13961 全失败(edge weight 信号丢)
2. **EAD β**(任何形式):13876 / 13885 / 13900 全失败
3. **Augmentation-based SSL**(Consistency / Mean Teacher / DA):13886 / 13902 / 13915 / 13914
4. **Mask Reconstruct multi-task**:13916(recon distract 主任务)
5. **Adversarial Mask**:13890(input mask 无 target)
6. **Self-distillation Self-Train**:13881 / 13887(Δ=0 自蒸馏天花板)
7. **Self-Train on EAD**:13908-13912(Pseudo Quality Ceiling 完全饱和)
8. **GeoEmbed 降维**:13878 / 13879(信号被砍)
9. **Per-modality encoder**:13884 / 13901(模型容量不是瓶颈)

### 🎯 当前研究方向(2026-05-13,详 [methods.md §7.9](methods.md) + [research_directions_analysis.md](research_directions_analysis.md))

**Stage 1**(14916,本周):Modality-Aware Masked Pretraining
- 利用 1.7M (station, time) unlabeled samples
- Mask dynamic features (WRF non-Tair + CLMS),保 static + Tair anchor 当 context
- 两阶段 pretrain → finetune,**不踩 augmentation invariance 失败的坑**
- 风险评估:30% 显著成功,30% marginal,40% 失败

**潜在 Phase 2**(等 14916 结果):
- 14916 成功 → Time-extended Masking + Physics PDE loss
- 14916 失败 → JEPA-style embedding prediction / Multi-source Ensemble Pseudo
- 独立可并行:**Temporal Laplacian on unlabeled**(~50 行,1 sbatch)

---

## 表 1:V1 baseline(`V1_baseline` 项目)

| jobid | 脚本 | val_mode | n_unl | iDim | best_v_rmse | ≈ °C(scl≈30 粗估)| 状态 |
|---|---|---|---|---|---|---|---|
| 13848 | V1_sup_random      | random     | 0   | 1302 | **0.0296** | 0.89 | DONE ✓ |
| 13849 | V1_sup_sequential  | sequential | 0   | 1302 | **0.0395** | 1.19 | DONE ✓ |
| 13850 | V1_sup_spatial     | spatial    | 0   | 1302 | **0.0339** | 1.02 | DONE ✓ |
| 13851 | V1_semi_random     | random     | 400 | 1302 | **0.0255** | 0.76 | DONE ✓ |
| 13852 | V1_semi_sequential | sequential | 400 | 1302 | **0.0415** | 1.25 | DONE ✓(wandb 误名 V2)|
| 13853 | V1_semi_spatial    | spatial    | 400 | 1302 | **0.0322** | 0.97 | CANCEL-PART(收敛后停,wandb 误名 V2)|

### V1 sup vs semi 对照(°C 粗估)

| val_mode | V1 sup RMSE | V1 semi RMSE | Δ |
|---|---|---|---|
| random | 0.0296 | **0.0255** | −0.0041(semi 好 ↓)|
| sequential | 0.0395 | 0.0415 | +0.0020(semi 略差 ↑)|
| spatial | 0.0339 | **0.0322** | −0.0017(semi 略好 ↓)|

**初步观察**:V1 random / spatial 上 semi 给改进,sequential 反而略差。

---

## 表 2:V2 baseline(`urban` 项目,精确 °C)

| jobid | 脚本 | val_mode | n_unl | iDim | best_v_rmse | best °C(scl=34.57)| 状态 |
|---|---|---|---|---|---|---|---|
| **13861** | V2_sup_random      | random     | 0   | 1347 | **0.0426** | **1.47** | **DONE**(best @ ep 4050,跑完 5000)|
| **13862** | V2_sup_sequential  | sequential | 0   | 1347 | **0.0583** | **2.02** | **DONE**(best @ ep 524 后严重 overfit)|
| **13865** | V2_sup_spatial     | spatial    | 0   | 1347 | **0.0505** | **1.74** | **DONE**(best @ ep 1583,ep 3460 overfit)|
| 13854 | V2_semi_random     | random     | 400 | 1347 | **0.0373** | 1.29 | CANCEL-PART(收敛后停)|
| 13855 | V2_semi_sequential | sequential | 400 | 1347 | **0.0601** | 2.08 | CANCEL-PART(收敛后停,过拟合)|
| **13860** | **V2_semi_spatial**    | **spatial**    | **400** | **1347** | **0.0453** | **1.57** | **CANCEL-PART(ep 770 best,ep 1419 overfit;V2 spatial baseline,self-train 起点)** |

## 表 3:V2 sup spatial GNN 层 ablation(`urban` 项目)

| jobid | conv_type | iDim | best_v_rmse | best °C | 状态 |
|---|---|---|---|---|---|
| **13865** | graphconv (= V2_sup_spatial baseline) | 1347 | **0.0505** | **1.74** | **DONE**(graphconv 在 sup spatial 上**最差**,gat/sageconv 都比它好)|
| **13863** | gat (4 heads)                         | 1347 | **0.0471** | **1.63** | CANCEL-PART(ep 1735 best,ep 3582 取消)|
| **13864** | sageconv                              | 1347 | **0.0477** | **1.65** | DONE(best @ ep 1600)|

## 表 4:V2 semi spatial unlabeled 数量扫描(`urban` 项目)

| jobid | n_unl | nNodes | iDim | best_v_rmse | best °C | 状态 |
|---|---|---|---|---|---|---|
| 13857 | 200  | 258  | 1347 | 0.0507 | 1.75 | CANCEL-PART(ep 524 best,ep 1939 overfit)|
| **13860** | **400(=baseline V2_semi_spatial)**| 458 | 1347 | **0.0453** | **1.57** | **CANCEL-PART(基准!)** |
| 13858 | 600  | 658  | 1347 | 0.0499 | 1.72 | CANCEL-PART(ep 669 best,ep 1841 overfit)|
| **13859** | **800**  | 858  | 1347 | **0.0482** | **1.67** | **CANCEL-PART(n_unl scan 最佳)** |
| 13856 | 1000 | 1058 | 1347 | 0.0486 | 1.68 | CANCEL-PART(ep 518 best,ep 1576 overfit)|

**n_unl scan 结论**(更新):**13860(n_unl=400)= 0.0453 实际是这个 scan 的最佳**,优于 13859(n_unl=800)= 0.0482。原因可能是 800/1000 unlabeled 引入了空间分布更广但**质量参差**的节点 → 噪声 > 增益。**n_unl=400 是 V2 spatial 的 sweet spot,确认作为 self-train 起点 baseline**。

## 表 5:V1 vs V2 sup/semi spatial 对照

| dataset | sup RMSE | sup °C | semi RMSE | semi °C | Δ |
|---|---|---|---|---|---|
| V1 (13850/13853) | 0.0339 | ~1.02(scl≈30 估)| 0.0322 | ~0.97 | −0.0017 |
| V2 (13865 / 13860) | **0.0505** | **1.74** | **0.0453** | **1.57** | **−0.0052** |

**V2 spatial 观察(13865 已出)**:
- gat (13863) = 0.0471(1.63°C),sageconv (13864) = 0.0477(1.65°C),两者接近
- semi spatial (13860) = 0.0453(1.57°C),比 gat / sageconv sup 都好(差 ~5%)
- → **V2 spatial 上 semi GraphConv 是当前最强**,未来用 GAT/SAGE + semi 才有可能进一步提升

**V2 sup 各 val_mode 对比**:
- random (13861) = 0.0426 → **V2 上最易任务**(随机切分,样本分布相同)
- spatial(13865 graphconv = **0.0505** / 13864 sageconv 0.0477 / 13863 gat 0.0471)→ **最难**(空间外推,且 graphconv 反而最差,GAT/SAGE 略好)
- sequential (13862) = 0.0583 → 时间外推也难,严重过拟合

V1 已显示 spatial 上 **semi 微弱帮助**(同 V0 的 V1 spatial 现象一致)。V2 待跑完确认。

## 🏅 全局排行榜(V2 spatial,2026-05-13 更新)

baseline = 13860 V2_semi_spatial = **0.0453 (1.57°C)**

**所有实验按 best v_rmse 排序**(✗ = 失败,✓ = 涨点):

| 排名 | jobid | 方法 | best | °C | Δ vs baseline | 解读 |
|---|---|---|---|---|---|---|
| 🏆 | **13897** | **EAD α + Lap λ=0.1** | **0.039186** | **1.36** | **−0.0061 (−14%)** ⭐⭐ | **V2 spatial 最强**(组件叠加成功)|
| 🥇 | **13912** | EAD + Lap + Kriging ST (4-way) | 0.039264 | 1.36 | −0.0060 | 与 13897 持平,ST 在 EAD+Lap 之上无贡献(Pseudo Ceiling)|
| 🥈 | **13909** | EAD + Lap + Hybrid ST (4-way) | 0.039541 | 1.37 | −0.0058 | 同上 |
| 🥉 | 13958 | SAGE + EAD α + Lap | 0.039332 | 1.36 | ≈ 0 vs 13897 | SAGE 在 semi 上无优势(edge weight 丢失)|
| 5 | 13910 | EAD α + Kriging ST (3-way) | 0.039619 | 1.36 | ≈ 0 vs 13875 | Pseudo Ceiling 边缘 |
| 6 | **13875** | **EAD α**(物理时间锚)| **0.0399** | **1.38** | **−0.0054 (−12%)** | 最强单一组件 |
| 7 | 13908 / 13911 | EAD + Hybrid ST (3-way) | 0.039877 | 1.38 | = 13875 完全饱和 | Pseudo Quality Ceiling 实证 |
| 8 | 13959 | GAT + EAD α + Lap | 0.040681 | 1.41 | +0.0015 ✗ | GAT 在小图 overfit |
| 9 | 13961 | SAGE + EAD α | 0.040532 | 1.40 | +0.0006 ✗ | 同 13958 根因 |
| 10 | 13877 | Lap λ=0.1 | 0.0422 | 1.46 | −0.0031 (−7%) ✓ | 图正则有效 |
| 11 | 13900 | EAD α + β-MLP UF only | 0.0427 | 1.48 | −0.0026 (−6%) | β 路线仍逊 EAD α 单独 |
| 12 | **13883** | **Hybrid ST**(baseline 之上)| **0.0428** | **1.48** | **−0.0025 (−6%)** ✓ | 外部+自蒸馏混合最佳 |
| 13 | **13882** | Kriging-pseudo ST(baseline 之上)| **0.0429** | **1.48** | **−0.0024 (−5%)** ✓ | 外部 pseudo 打破自蒸馏天花板 |
| 14 | 13876 | EAD α + β kriging | 0.0448 | 1.55 | −0.0005 | β kriging 吃掉 α 红利 |
| 15 | 13886 | Multi-view consistency | 0.0449 | 1.55 | ≈ 0 | 小图增强信号弱 |
| 16 | 13881 | self + conformal ST | 0.0450 | 1.56 | ≈ 0 | 自蒸馏天花板 |
| 17 | 13887 | self + neighbor ST | 0.0450 | 1.56 | ≈ 0 | 同上 |
| 18 | 13860 | **baseline** | **0.0453** | **1.57** | **0** | 参考 |
| 19 | 13890 | Adversarial near-valid mask | 0.0455 | 1.57 | +0.0002 | 无效 |
| 20 | 13960 | SAGE + naive semi | 0.045977 | 1.59 | +0.0007 ✗ | 略差 |
| 21 | 13878 | GeoEmbed g6 (252) | 0.0478 | 1.65 | +0.0025 ✗ | 降维有害 |
| 22 | 13915 | Consistency 中档 | 0.0474 | 1.64 | +0.0021 ✗ | 中档增强也救不了 |
| 23 | 13914 | Distribution Alignment (MMD) | 0.0484 | 1.67 | +0.0031 ✗ | 对齐不是瓶颈 |
| 24 | 13879 | GeoEmbed PCA256 | 0.0490 | 1.69 | +0.0037 ✗ | 同 13878 |
| 25 | 13884 | Per-modality encoder | 0.0498 | 1.72 | +0.0045 ✗ | 容量不是瓶颈 |
| 26 | 13901 | Per-modality modal_hid=64 | 0.0521 | 1.80 | +0.0068 ✗ | 同 13884 |
| 27 | **13885** | EAD α + β-MLP full | 0.0580 | 2.00 | +0.0127 ✗✗ | β-MLP overfit |
| 28 | **13916** | Mask Reconstruct (multi-task) | 0.0583 | 2.02 | +0.0130 ✗✗ | recon distract 主任务 |
| 29 | 13902 | Consistency 强档 | 0.068 | 2.35 | +0.023 ✗ | 过激坍缩 |
| - | **14916** | **Modality-Aware Pretrain (Stage 1)** | DONE | - | - | 2000 ep,final loss=4.91e-4(↓ 450×);ckpt 保存供 Stage 2 用 |
| - | **14925** | **EAD + Lap + Pretrained (Stage 2)** ⭐ | PENDING | - | - | 加载 14916 ckpt + 4000 ep,**对比 13897=0.039186** |

---

## 7 个核心结论

1. **EAD α 是 V1 最大单一红利(-12%)** ⭐⭐
   一行 env 启用,模型本来漏掉的"WRF 全局时间偏差"立刻补上。**物理 prior 的力量**。

2. **外部 pseudo self-train 真有效(-5~-6%)** ⭐
   - Kriging-pseudo(13882)= **0.0429**:实证 kriging benchmark(0.0437)的预测,外部信号确实推进模型
   - Hybrid(13883)= **0.0428**:0.5 model + 0.5 kriging 略好,模型 anchor + 外部信号互补
   - 这两个**实证打破了"自蒸馏天花板"** —— 13881 / 13887 self-distillation Δ≈0 才是天花板,外部信息能突破

3. **β 锚整个方向走不通** ❌
   - EAD α+β kriging(13876)= 0.0448,β 吃掉了 α 的红利 −0.0049
   - EAD α+β-MLP(13885)= **0.0580,反而比 β-kriging 还差 0.0132**
   - **结论**:50 train 站太少,MLP 严重 overfit β;kriging 在 unlabeled 上太 noisy
   - **直接结论**:**只用 α,放弃 β**;γ 交互项也别试(同样数据不足)

4. **GeoEmbed 全 1008 维必需,不是冗余** ❌
   三个降维 / 降参数实验全部变差:g6(+0.0025)、pca256(+0.0037)、per-modality(+0.0045)
   推翻了"GeoEmbed dominate 第一层是噪声"的假说;**模型其实在用这 1008 维信息**

5. **Self-distillation Δ≈0 实证(2 次确认)**
   13881 self+conformal = 0.0450、13887 self+neighbor 修复版 = 0.0450,都几乎 = baseline。
   **与 V0 13451 R1-R5 Δ=0 一致**,自蒸馏在 naive semi 之上没新信息

6. **小图 SSL 全线失败**(Δ≈0 或更差,5 个负发现)
   - Consistency K=3 (13886) = 0.0449(≈0);scale=1.3 中档 (13915) = 0.0474(+0.0021);scale=1.8 强 (13902) = 0.0680(灾难)→ **GRAND-style 任何强度都救不了 458 节点小图**
   - Adv mask (13890) = 0.0455(≈0)→ input mask 无 target 没用
   - **Mask Reconstruct (13916) = 0.0583(+0.0130 ✗✗)** → 设计上想突破 Pseudo Quality Ceiling,但 multi-task aux 严重 distraction 主任务;**重建 feature 任务太难 → 把 hidden 拉离了 T 预测的最佳方向**
   - Distribution Alignment MMD (13914) = 0.0484(+0.0031 ✗)→ unlabeled emb 分布对齐不是瓶颈
   - **教训**:小图(458 节点)上,**任何不依赖 explicit label 的 SSL 信号都失败** —— Lap 是例外,因为它本质是图正则而非 SSL

7. **Lap 是 cheap win(-7%)**
   λ=0.1 一行 env,稳定贡献 -0.0031,可与 EAD α 叠加(下一步组合)

---

## ~~下一步(2026-05-11 提交,3 个组合 sbatch)~~ [历史,已全部完成]

| jobid | 脚本 | 组合 | 预期 |
|---|---|---|---|
| **13897** | V2_semi_spatial_eadA_lap | EAD α + Lap λ=0.1 | ≈ 0.038(-0.0085 vs baseline)|
| **13898** | V2_semi_spatial_eadA_hybST | EAD α + Hybrid ST(0.5/0.5)| ≈ 0.037(-0.0079)|
| **13899** | V2_semi_spatial_eadA_lap_hybST | EAD α + Lap + Hybrid 三向 | ≈ 0.036?(-0.0110 乐观)|

❌ 不做(已实证无效):
   - β 锚(任何形式 — kriging 边际为负,MLP 严重 overfit)
   - GeoEmbed 降维 / 降参数(g6 / pca256 / pmod 全变差)
   - 单纯 self-distillation(Δ≈0,根因见 methods.md §9.4.1)
   - Adv mask / Consistency 单独(增强信号在小图上太弱)

🤔 中性(等组合结果再决定):
   - β-MLP 用小子集 aux(如只 UF 17 维)
   - Hybrid α_self 扫(0.3 / 0.7)
   - Per-modality 加大 modal_hid(32→64)
   - Consistency 更激进增强(p_edge=0.7)


---

## ~~2026-05-11 队列管理决策日志~~ [历史]

**为给 4-way 主线(13909 + 13912)+ 关键对照(13908)让位,做了以下 cancel**:

| 取消项 | 原因 | 后续 |
|---|---|---|
| 13902 cons_strong | scale=1.8 过激坍缩到 0.0680 | TODO:scale=1.2-1.4 中间档,优先级低 |
| 13903 TTA | 推理增强,期望小,不在主线 | **待后续重跑**(~10 min) |
| 13904 DA (MMD) | 新探索方向,可能 ±0,不在主线 | **待后续重跑**(~6h,future work) |
| 13911 α_self=0.3 | hybrid 微调,期望 ±0.001 | TODO:主线饱和后再调 |
| 13910 EAD α + Kriging(3-way) | 与 13912(4-way+kriging)信息重叠 | 可选重跑(为完整 ablation)|

**保留**:
- 🟢 **13908**(3-way EAD + Hybrid ST,**无 Lap**)— 关键对照,**不取消**
- 🥇 **13909**(4-way:EAD + Lap + Hybrid ST)— 主候选
- 🥇 **13912**(4-way:EAD + Lap + Pure Kriging ST)— 主候选

---

## ~~预测表(2026-05-11,待结果回填)~~ [历史,大部分预测已被推翻 — Pseudo Quality Ceiling 实证 + 3 个 SSL 方法全失败]

**目的**:PPT 故事弧提前定型,结果出来直接填进 "actual" 列即可。**预测范围基于线性叠加 + V0 / V1 已有实证**。

### 关键 pending 实验预测

| jobid | 实验 | 预测 best v_rmse | 预期 Δ vs baseline (0.0453) | 故事(若预测正确)| 实际 |
|---|---|---|---|---|---|
| 🥇 13909 | EAD α + Lap + Hybrid ST(4-way)| **0.036-0.039** | -0.0085 ~ -0.0095 | "**4-way 组合是 V1 最强,正交叠加成立**" | TBD |
| 🥇 13912 | EAD α + Lap + Pure Kriging ST(4-way)| **0.036-0.039** | -0.0085 ~ -0.0095 | 同 13909 或微差,**hybrid 中 self 部分非必要** | TBD |
| 🥈 13908 | EAD α + Hybrid ST(3-way,无 Lap)| **0.037-0.040** | -0.0050 ~ -0.0080 | "Lap 也是必要组件"(若 < 13909) | **0.039877 ≈ 0**(场景 C 完美饱和,见下方)|
| 🥈 13910 | EAD α + Pure Kriging ST(3-way,无 Lap)| **0.037-0.040** | -0.0050 ~ -0.0080 | 同 13908,**对照 13912 看 Lap 在 ST 上的贡献** | TBD |
| ⭐ 13911 | EAD α + Hybrid α_self=0.3 | 0.039-0.043 | -0.0023 ~ -0.0063 | "Hybrid 偏 kriging 不一定更好"(α 调参敏感度低)| TBD |
| ⭐ 13902 | Consistency 加强版 | 0.044-0.046 | ≈0 | "小图增强信号弱,加强也救不了" | **0.0680(更糟,过激)** ❌ |
| ⭐ 13913(原 13903)| TTA on 13860 | 0.0440-0.0450 | -0.0003 ~ -0.0013 | "TTA 给小幅 cheap win,可作推理 trick" | TBD |
| 🌟 13914(原 13904)| Distribution Alignment(MMD)| **0.040-0.046** | -0.005 ~ +0.001 | **未知大!**新方向。若 < 0.045 → "对齐 unlabeled 分布有用,后续值得 CycleGAN" | **0.0484(+0.0031)✗** 失败,MMD 不是瓶颈 |
| 🌟 13916 | Mask Reconstruct(multi-task aux,target=UF 17d)| **0.040-0.046** | -0.005 ~ +0.001 | **未知大!**真正给 unlabeled ground-truth target(features),理论上无 Pseudo Quality Ceiling。若 < 0.045 → "feature 自重建是合法 SSL 信号"(对比 self-train kriging) | **0.0583(+0.0130)✗✗** 严重伤害,recon distraction |

### 三种可能场景(故事弧分支)

#### 场景 A:乐观,13909 / 13912 ≤ 0.038 ⭐⭐⭐

→ "**正交叠加成立,V1 最强 0.038 = -16% vs baseline**"
→ TL;DR headline:Physical prior + SSL 双维度组合验证有效
→ 适合 paper 主线 narrative

#### 场景 B:中性,13909 / 13912 ≈ 0.039-0.041

→ "**部分 subadditive**,组合略优于 13897(-14%)但未到线性预期"
→ 解释:Lap 与 Kriging-pseudo 都做空间平滑,有冗余
→ 仍可发表,但故事偏向"组合饱和"

#### 场景 C:悲观,13909 / 13912 ≥ 0.0392

→ "**ST 在 EAD+Lap 之上完全饱和(同 V0 13451 R1-R5 Δ=0)**"
→ V1 最强仍是 13897 EAD α + Lap = 0.0392
→ 重 framing:**Negative result 也是 contribution**(实证自蒸馏/弱 SSL 天花板)

### 高优先级看的 4 个对照

| 对照 | 答案告诉我们 |
|---|---|
| **13912 vs 13909** | Hybrid 中 self 那 50% 在 4-way 上是否必要 |
| **13909 vs 13908** | Lap 在 ST 上是否过度平滑(若 13908 ≥ 13909 → Lap 反而拖累)|
| **13909 vs 13897** | ST 在 EAD+Lap 之上是否有边际(若 13909 ≥ 13897 → 自蒸馏天花板再现)|
| **13904 vs 13860 / 13897** | Distribution Alignment 是否值得做完整 CycleGAN |

### PPT 直接用的 "Expected Final Story"(80% 概率版本)

```
基于线性叠加 + V0/V1 实证经验,**最可能场景**:

  13909/13912 ≈ 0.037-0.038(组合略好于 13897)
  其它 PD job 多为 Δ≈0(失败案例,有教育意义)

故事:
  1. 物理 prior(EAD α)单项 -12% ⭐(核心创新)
  2. 加 Lap → -14%(图正则补充)
  3. 加 ST(kriging-pseudo)→ -16%(SSL framework 补充)
  4. β / GeoEmbed 降维 / 自蒸馏 / consistency 都失败,给出 negative results
  5. Future work:理论分析 / 多城市 / Causal framing → ML 主会版本
```

## 表 EAD / Lap:V2 spatial 在 13860 baseline 上的方法对照(`urban` 项目)

baseline = 13860(V2_semi_spatial,plain Huber),与之唯一差异是 loss 形式。EAD 与 Lap **互斥**(代码 assert)。

| jobid | 脚本 | 方法 / config | iDim | best_v_rmse | best °C | 状态 |
|---|---|---|---|---|---|---|
| **13860** | V2_semi_spatial         | baseline(无 EAD / 无 Lap)| 1347 | **0.0453** | **1.57** | **DONE(参考)** |
| ~~13870~~ | ~~V2_semi_spatial_eadA~~ | ~~EAD α~~ | ~~1347~~ | — | — | **CANCELLED**(epoch=5000,改为 4000 重提)|
| ~~13871~~ | ~~V2_semi_spatial_eadAB~~ | ~~EAD α + β~~ | ~~1347~~ | — | — | **CANCELLED**(epoch=5000,改为 4000 重提)|
| ~~13872~~ | ~~V2_semi_spatial_lap~~ | ~~Lap λ=0.1~~ | ~~1347~~ | — | — | **CANCELLED**(epoch=5000,改为 4000 重提)|
| **13875** | V2_semi_spatial_eadA    | EAD α 时间锚 (epoch=4000)   | 1347 |  |  | **PD**(2026-05-10 提交)|
| **13876** | V2_semi_spatial_eadAB   | EAD α + β 锚 (epoch=4000)   | 1347 |  |  | **PD**(2026-05-10 提交)|
| **13877** | V2_semi_spatial_lap     | Lap λ=0.1 (epoch=4000)     | 1347 | **0.0422** | **1.46** | **CANCEL-PART**(ep 873 best,ep 1395 cancel,Δ=-0.0031 vs baseline)|
| **13885** | V2_semi_spatial_eadABplus | EAD α + β-MLP(替代 β kriging)| 1347 | **0.0580** | **2.00** | **DONE**,**Δ=+0.0127 大幅变差!**(β-MLP overfit 50 站,比 β-kriging 还差)|
| **13886** | V2_semi_spatial_cons    | Multi-view consistency(K=3, λ=0.1)| 1347 | **0.0449** | **1.55** | DONE,Δ=-0.0004(几乎 = baseline,小图增强信号弱)|
| **13890** | V2_semi_spatial_advmask | Adversarial near-valid mask(K=20, p=0.5)| 1347 | **0.0455** | **1.57** | CANCEL-PART(ep 652 best,Δ=+0.0002 ≈ baseline)|
| **13897** | V2_semi_spatial_eadA_lap | **EAD α + Lap λ=0.1**(两强叠加)| 1347 | **0.0392** | **1.36** | **CANCEL-PART(ep 161 best 锁定,V1 当前最强!)** ⭐⭐ |
| ~~13898~~ | ~~V2_semi_spatial_eadA_hyb~~ | ~~EAD α + Hybrid ST~~ | ~~1347~~ | ~~0.0428~~ | ~~1.48~~ | **CANCELLED**(2 个 bug:① selftrain 没集成 EAD;② R0 ckpt 用错)|
| ~~13899~~ | ~~V2_3way~~ | ~~EAD α + Lap + Hybrid ST~~ | ~~1347~~ | — | — | **CANCELLED**(同 bug)|
| **13908** | V2_eadA_hyb(修复版)| EAD α + Hybrid ST(无 Lap)| 1347 | **0.039877** | **1.378** | **DONE,Δ≈0 完美饱和**(R1-R5 全 0.039877,与 13875 持平)→ 证实 **ST 在 EAD 上 Pseudo Quality Ceiling 假说**(kriging 0.0437 < EAD 0.0399)|
| **13909** | V2_3way(修复版)| EAD α + Lap + Hybrid ST 三向 | 1347 |  |  | **PD**(预期 -0.0110)|
| **13910** | V2_eadA_krig | EAD α + **纯 Kriging ST**(α_self=0,对照 13908 hybrid)| 1347 |  |  | **PD**(验证 hybrid 中 self 部分必要性)|
| **13911** | V2_eadA_h03 | EAD α + Hybrid α_self=**0.3**(偏 kriging,对照 13908 的 0.5)| 1347 |  |  | **PD**(扫 hybrid 混合比)|
| **13912** | V2_eAL_krig | EAD α + Lap + **纯 Kriging ST**(对照 13909 hybrid)| 1347 |  |  | **PD**(三向叠加换 pseudo source)|
| **13900** | V2_semi_spatial_eadABplus_uf | EAD α + β-MLP **UF only**(17 维)| 1347 | **0.0427** | **1.48** | **RUNNING(ep 1718,best @ ep 10,Δ=−0.0026)**;比 13885 full 救回 +0.0153 ✓,但仍逊 EAD α 单独 -0.0028 → **β 锚整体走不通** |
| **13901** | V2_semi_spatial_pmod64 | Per-modality `modal_hid=64` | 1347 | **0.0521** | **1.80** | **RUNNING(ep 1767,best @ ep 702,Δ=+0.0068 变差)**;比 13884 pmod32(0.0498)还差 +0.0023 → **per-modality 架构本身有问题,不是 hidden 维度** |
| **13902** | V2_semi_spatial_cons_strong | Consistency 加强(K=5, λ=0.3, **scale=1.8 过激**)| 1347 | **0.0680** | **2.35** | **CANCEL-PART**(ep 96 best,ep 532 已坍缩到 0.116;过激坍缩,见下方注释)|

**13902 坍缩诊断**:scale=1.8 让 p_edge 最强 = 0.90 / p_node 最强 = 0.27 → **90% 边丢图断,GNN 退化为 MLP** → consistency loss 把模型拉向乱平均 → 越拉越烂。**TODO(未来若想救)**:试 `V1_CONS_AUG_SCALE=1.2-1.4` 中间档 + λ_cons=0.2,期望 ±0.001(但小图 458 节点本质不适合 consistency,优先级低)。
| ~~13903~~ | ~~V2_tta_13860~~ | ~~TTA on 13860 ckpt~~ | — | — | — | CANCELLED → 重提为 13913 |
| ~~13904~~ | ~~V2_semi_spatial_da~~ | ~~Distribution Alignment (MMD)~~ | ~~1347~~ | — | — | CANCELLED → 重提为 13914 |
| **13913** | V2_tta_13860(重提)| TTA on 13860 ckpt(scan K + scan p)| — | — | — | **PD**(~10 min 推理)|
| **13914** | V2_semi_spatial_da | Distribution Alignment(MMD,半 CycleGAN,λ_mmd=0.1)| 1347 | **0.0484** | **1.67** | **CANCEL-PART**(best @ ep 676,跑到 ep 1551 无更新 875 ep,Δ=**+0.0031 ✗**)|
| **13915** | V2_semi_spatial_cons_mid | Consistency 中档(K=5, λ=0.2, scale=1.3,救 13902 过激)| 1347 | **0.0474** | **1.64** | **CANCEL-PART**(best @ ep 606,跑到 ep 1307 无更新 700 ep,Δ=**+0.0021 ✗**)|
| **13916** | V2_semi_spatial_mrecon | Mask Reconstruct multi-task aux(λ=0.1, K=30, target=UF 17d)| 1347 | **0.0583** | **2.02** | **CANCEL-PART**(best @ ep 604,跑到 ep 1186 无更新 580 ep,**Δ=+0.0130 ✗✗ 严重伤害**)|
| **13958** | V2_semi_spatial_eadA_lap_sage | **SAGE + EAD α + Lap**(对照 13897 GraphConv)| 1347 | **0.039332** | **1.36** | **DONE**(跑完 4000 ep,vs 13897=0.039186,**Δ=+0.00015 ≈ tie**;SAGE 在 semi+EAD+Lap 下无优势)|
| **13959** | V2_semi_spatial_eadA_lap_gat | **GAT + EAD α + Lap**(对照 13897)| 1347 | **0.040681** | **1.41** | **DONE**(跑完 4000 ep,vs 13897=0.039186,**Δ=+0.0015 ✗** 略差,GAT 在 50 train 上 overfit)|
| **13960** | V2_semi_spatial_sage | **SAGE + naive semi**(对照 13860)| 1347 | **0.045977** | **1.59** | **DONE**(跑完 4000 ep,vs 13860=0.0453,**Δ=+0.0007 ✗** 略差;sup 时 SAGE 优势(13864=0.0477<13865=0.0505)在 semi 下消失)|
| **13961** | V2_semi_spatial_eadA_sage | **SAGE + EAD α**(对照 13875)| 1347 | **0.040532** | **1.40** | **DONE**(跑完 4000 ep,vs 13875=0.0399,**Δ=+0.0006 ✗** 略差)|
| **14916** | V2_semi_spatial_pretrain | **Modality-Aware Masked Pretraining(Stage 1)**(WRF non-Tair + CLMS,mask ratio=0.25,2000 ep)| 1347 | (无 valid RMSE,纯 pretrain)| — | **RUNNING**(2026-05-13 提交,~3-4h);输出 encoder ckpt 供 Stage 2 finetune 用 |

## 表 6:对齐 sanity 检查结果(已通过)

- V1 t=0 = 2018-05-01 02:00,V2 t=0 = 00:00 → V1 t=n ↔ V2 t=(n+2),offset=2
- 实测温度相关性 best offset ∈ {2, 3} 噪声水平,offset=2 可信

## 表 Per-Modality:V2_semi_spatial 上 encoder 架构 ablation(`urban` 项目)

baseline = 13860(flat encoder Linear×2,iDim=1347→128,~190K 参数)。

**动机**:Flat encoder 第一层 Linear(1347 → 128) ≈ 172K 参数,**75% 容量花在 GeoEmbed 1008 维**。Per-modality 让每个模态(WRF / aux / CLMS / UF / GeoEmbed)独立 Linear→32 维 → fusion 到 128,第一层参数 ≈ 43K(降 75%),WRF 真信号不被 GeoEmbed 噪声稀释。

| jobid | 脚本 | encoder | 模型总参数 | best_v_rmse | best °C | 状态 |
|---|---|---|---|---|---|---|
| 13860 | V2_semi_spatial         | flat (Linear×2) | ~305K | 0.0453 | 1.57 | DONE(参考)|
| **13884** | V2_semi_spatial_pmod    | per-modality | **197K** | **0.0498** | **1.72** | **DONE,Δ=+0.0045 ✗ 变差**(降参数 underfit)|

**实现**:`network.py` 加 `ModalEncoder` 类,`V1_ENCODER_TYPE=per_modality` 启用。每模态 Linear→32 → concat → fusion 2 层 Linear→128。

预期:
- 若 -0.002 ~ -0.005:GeoEmbed 容量稀释假说成立,架构改进有效
- 若 ≈ 0:模型容量不是瓶颈,问题在数据 / 物理 prior
- 若 > 0:per-modality 信息融合不够 expressive,需要更复杂架构

## 表 GeoEmbed:V2_semi_spatial 上 GeoEmbed 维度 ablation(`urban` 项目)

baseline = 13860(`V1_GEO_POOL_SIZE=12`,GeoEmbed=1008,iDim=1347,best=0.0453)。

**动机**:Kriging benchmark(0.0437)反超模型(0.0453)→ 怀疑 GeoEmbed 1008 维有冗余/噪声 → **降维但不删除**(GeoEmbed 是 V2 数据最独特的高分辨率城市形态特征,用于 street-level downscaling 不可替代)。

| jobid | 脚本 | 降维方式 | GeoEmbed 维 | iDim | best_v_rmse | best °C | 状态 |
|---|---|---|---|---|---|---|---|
| **13860** | V2_semi_spatial         | pool 12 → 1008 | 1008 | 1347 | **0.0453** | **1.57** | DONE(参考)|
| ~~13873~~ | ~~V2_semi_spatial_g6~~ | ~~pool 6 → 252~~ | ~~252~~ | ~~591~~ | — | — | **CANCELLED**(epoch=5000,改为 4000 重提)|
| ~~13874~~ | ~~V2_semi_spatial_pca256~~ | ~~PCA → 256~~ | ~~256~~ | ~~595~~ | — | — | **CANCELLED**(epoch=5000,改为 4000 重提)|
| **13878** | V2_semi_spatial_g6      | pool 6  → 252 (epoch=4000) | 252  | 591  | **0.0478** | **1.65** | **DONE**(best @ ep 711,Δ=+0.0025 ✗ 变差)|
| **13879** | V2_semi_spatial_pca256  | PCA → 256 (epoch=4000)     | 256 | 595 | **0.0490** | **1.69** | **DONE**(best @ ep 872,Δ=+0.0037 ✗ 变差)|

**对照点**:g6 (252 维,空间块平均) vs pca256 (256 维,数据驱动主成分)—— 维度近,可直接比较两种降维策略。

PCA 实现细节:fit set = 50 train labeled + 400 unlabeled = 450 样本(排除 valid 8 防 leak;unlabeled 无 target 进入,无 leak),sklearn PCA。256 维大概率覆盖 >95% 方差。

预期(具体见 [methods.md §9.6.3](methods.md#963-假设与预期)):
- g6:中等概率持平,可能 -0.001 ~ +0.005(空间池化丢方向性信号风险)
- pca256:**有可能优于 g6**(数据驱动 → 保留有用方差),但也可能因 PCA 失去空间结构而持平

## 表 Self-Train:V2_semi_spatial 上 self-train 三个变体(`urban` 项目)

baseline = 13860(V2_semi_spatial,best=0.0453,作 R0 起点 ckpt)。

**框架**:固定图(58 labeled + 400 unlabeled,k-NN k=10)+ 每轮 K=40 unlabeled 翻 label_mask=True + warm-start lr=1e-4 + 5 轮 × 200 epoch。Loss = Huber(50 train) + 0.5 × Huber(K pseudo)。**不叠加 EAD / Lap**(隔离 self-train 自身效应)。

| jobid | 脚本 | pseudo 来源 | confidence | best_v_rmse | best °C | 状态 |
|---|---|---|---|---|---|---|
| 13860 | V2_semi_spatial | — | — (R0 baseline) | 0.0453 | 1.57 | DONE(参考)|
| ~~13880~~ | ~~V2_self_neighbor (A1)~~ | self | 邻居 error | — | — | **CANCELLED**(shuffle order bug,见下)|
| **13887** | V2_self_neighbor (修复版)| self | 邻居 error | **0.0450** | **1.56** | **DONE,Δ=-0.0003 ≈ 0**(自蒸馏)|
| **13881** | V2_self_conformal (A2) | self | Conformal(简化版)| **0.0450** | **1.56** | **DONE,Δ=-0.0003 ≈ 0**(自蒸馏)|
| **13882** | V2_self_kriging (B) ⭐ | **kriging**(外部!)| kriging_struct | **0.0429** | **1.48** | **DONE,Δ=-0.0024 ✓**(外部 pseudo 真涨点)|
| **13883** | V2_self_hybrid (B+) ⭐⭐ | **hybrid 0.5/0.5** | kriging_struct | **0.0428** | **1.48** | **DONE,Δ=-0.0025 ✓**(混合最佳)|

**3 个实验定位**:
- **A1 (self + neighbor_error)**:V0 在 L1/L2 弱框架测过都软坍缩,**未在 L3 公平测**;期望 Δ≈0(自蒸馏天花板)
- **A2 (self + Conformal 简化)**:复刻 V0 13451 路线但**在纯 baseline**(无 EAD/Lap),V0 R1-R5 Δ=0;期望 Δ≈0
- **B (kriging-pseudo)** ⭐:**外部 pseudo 打破自蒸馏**;sanity benchmark 已证 kriging RMSE=0.0437 略好于模型 0.0453,**最有可能涨点**

**Sanity 通过**(2026-05-10,n_unl=20 + K=4 + N=2 + ep=2 微型测试):
- A1 R0=0.0920 → final=0.0877(Δ=-0.0043)
- A2 R0=0.0920 → final=0.0725(Δ=-0.0194)
- B  R0=0.0920 → final=0.0466(Δ=-0.0454)⭐

**🐛 Bug 修复(2026-05-10)**:**13880 跑出灾难性退化(R0=0.0453 → R1 ep 1=0.0617 → R1 ep 200=0.1401)** 暴露 `compute_self_pseudo` 的 shuffle order bug —— `trainLoader(shuffle=True)` 给出乱序 batch,但 `inject_pseudo_into_dataset` 按 trainSet 自然顺序写入,**pseudo 错位到不对应的时间步**。修复:`compute_self_pseudo` 强制创建 sequential loader(shuffle=False)。

**bug 影响范围**:
- ❌ 13880 self+neighbor:已 cancel,**13887 重提(用修复后代码)**
- ⚠ 13881 self+conformal、13883 self+hybrid:启动时会读修好的 selftrain.py,自动用修复版
- ✓ 13882 kriging:不受影响(kriging 不用 model forward,直接从 train 真值算)

(R0=0.0920 高于 13860 实际 0.0453,是因为 sanity 用 n_unl=20 改了图结构;生产 sbatch n_unl=400 R0 应回 0.0453)

## 表 7:Kriging accuracy sanity(2026-05-09 跑,for self-train 实验 B 决策)

入口:[code_V1/sanity_kriging.py](../code_V1/sanity_kriging.py),IDW kriging,50 train → 8 valid,k=10。

| 对比项 | RMSE(归一化)| °C | 含义 |
|---|---|---|---|
| 13860 模型 baseline | 0.0453 | 1.57 | 1300+ 维特征 + GNN 学习 |
| **Kriging IDW (k=10)** | **0.0437** | **1.51** | **仅用空间距离 + labeled 真值 → 反超模型** |

**Per-station kriging RMSE 范围**:0.0327(站 5,最佳)~ 0.0534(站 1,最差),std=0.0077。

**结论**:
1. 简单空间插值在 V2 spatial 任务上 **优于深度 GNN**(Δ=−0.0016)
2. **Kriging-pseudo self-train 实验 B 优先级最高** —— 真有可能涨点
3. 修正之前判断:kriging 不是 "noisy 信号当 anchor 用",而是 "高质量 pseudo 直接用"

---

## 全部 jobid → 脚本对照表

| jobid | 脚本 | 实际 dataset | val_mode | n_unl | conv |
|---|---|---|---|---|---|
| 13848 | V1_sup_random | V1 | random | 0 | graphconv |
| 13849 | V1_sup_sequential | V1 | sequential | 0 | graphconv |
| 13850 | V1_sup_spatial | V1 | spatial | 0 | graphconv |
| 13851 | V1_semi_random | V1 | random | 400 | graphconv |
| 13852 | V1_semi_sequential | **V1**(wandb 误名 V2)| sequential | 400 | graphconv |
| 13853 | V1_semi_spatial | **V1**(wandb 误名 V2)| spatial | 400 | graphconv |
| 13854 | V2_semi_random | V2 | random | 400 | graphconv |
| 13855 | V2_semi_sequential | V2 | sequential | 400 | graphconv |
| 13856 | V2_semi_spatial_1000u | V2 | spatial | 1000 | graphconv |
| 13857 | V2_semi_spatial_200u | V2 | spatial | 200 | graphconv |
| 13858 | V2_semi_spatial_600u | V2 | spatial | 600 | graphconv |
| 13859 | V2_semi_spatial_800u | V2 | spatial | 800 | graphconv |
| 13860 | V2_semi_spatial | V2 | spatial | 400 | graphconv |
| 13861 | V2_sup_random | V2 | random | 0 | graphconv |
| 13862 | V2_sup_sequential | V2 | sequential | 0 | graphconv |
| 13863 | V2_sup_spatial_gat | V2 | spatial | 0 | gat (4h) |
| 13864 | V2_sup_spatial_sageconv | V2 | spatial | 0 | sageconv |
| 13865 | V2_sup_spatial | V2 | spatial | 0 | graphconv |
| ~~13870~~ | ~~V2_semi_spatial_eadA~~ | V2 | spatial | 400 | **CANCELLED**(epoch=5000)|
| ~~13871~~ | ~~V2_semi_spatial_eadAB~~ | V2 | spatial | 400 | **CANCELLED** |
| ~~13872~~ | ~~V2_semi_spatial_lap~~ | V2 | spatial | 400 | **CANCELLED** |
| ~~13873~~ | ~~V2_semi_spatial_g6~~ | V2 | spatial | 400 | **CANCELLED** |
| ~~13874~~ | ~~V2_semi_spatial_pca256~~ | V2 | spatial | 400 | **CANCELLED** |
| 13875 | V2_semi_spatial_eadA (4000ep) | V2 | spatial | 400 | graphconv + EAD α |
| 13876 | V2_semi_spatial_eadAB (4000ep) | V2 | spatial | 400 | graphconv + EAD α+β |
| 13877 | V2_semi_spatial_lap (4000ep) | V2 | spatial | 400 | graphconv + Lap λ=0.1 |
| 13878 | V2_semi_spatial_g6 (4000ep) | V2 | spatial | 400 | graphconv,GeoEmbed pool=6 |
| 13879 | V2_semi_spatial_pca256 (4000ep) | V2 | spatial | 400 | graphconv,GeoEmbed PCA=256 |
| 13880 | V2_semi_spatial_st_self_neigh | V2 | spatial | 400 | graphconv + ST(self+邻居 error) |
| 13881 | V2_semi_spatial_st_self_conf  | V2 | spatial | 400 | graphconv + ST(self+conformal 简化) |
| 13882 | V2_semi_spatial_st_krig_kstruct | V2 | spatial | 400 | graphconv + ST(kriging+structural)⭐ |
| 13883 | V2_semi_spatial_st_hyb_kstruct | V2 | spatial | 400 | graphconv + ST(hybrid 0.5/0.5 + structural)|
| 13884 | V2_semi_spatial_pmod | V2 | spatial | 400 | graphconv + per-modality encoder(80K vs 190K 参数)|
| 13885 | V2_semi_spatial_eadABplus | V2 | spatial | 400 | graphconv + EAD α + β-MLP(取代 β kriging)|
| 13886 | V2_semi_spatial_cons | V2 | spatial | 400 | graphconv + multi-view consistency(K=3 GRAND-style)|
| ~~13880~~ | ~~V2_semi_spatial_st_self_neigh~~ | V2 | spatial | 400 | **CANCELLED**(shuffle bug,详见表 Self-Train)|
| 13887 | V2_semi_spatial_st_self_neigh (fixed) | V2 | spatial | 400 | graphconv + ST(self+邻居,bug 修复后重提) |
| 13890 | V2_semi_spatial_advmask | V2 | spatial | 400 | graphconv + adversarial near-valid mask(K=20, p=0.5)|
| 13897 | V2_eadA_lap | V2 | spatial | 400 | **EAD α + Lap λ=0.1**(V1 最强 0.0392)⭐⭐ |
| ~~13898~~ | ~~V2_eadA_hyb~~ | V2 | spatial | 400 | **CANCELLED**(EAD+ST 集成 bug)→ 重提 13908 |
| ~~13899~~ | ~~V2_3way~~ | V2 | spatial | 400 | **CANCELLED**(同 bug)→ 重提 13909 |
| 13900 | V2_eadABplus_uf | V2 | spatial | 400 | EAD α + β-MLP UF only(17 维)|
| 13901 | V2_pmod64 | V2 | spatial | 400 | per-modality `modal_hid=64` |
| ~~13902~~ | ~~V2_cons_strong~~ | V2 | spatial | 400 | **CANCEL-PART**(scale=1.8 过激坍缩 0.0680;TODO scale=1.2-1.4)|
| ~~13903~~ | ~~V2_tta_13860~~ | — | — | — | **CANCELLED** → 重提 13913 |
| ~~13904~~ | ~~V2_semi_spatial_da~~ | V2 | spatial | 400 | **CANCELLED** → 重提 13914 |
| 13908 | V2_eadA_hyb(修复版)| V2 | spatial | 400 | **EAD α + Hybrid ST**(3-way,修复后 R0=13875 ckpt)|
| 13909 | V2_3way(修复版)| V2 | spatial | 400 | **EAD α + Lap + Hybrid ST**(4-way 主候选)⭐ |
| 13910 | V2_eadA_krig | V2 | spatial | 400 | EAD α + 纯 Kriging ST(3-way) |
| 13911 | V2_eadA_h03 | V2 | spatial | 400 | EAD α + Hybrid α_self=0.3 |
| 13912 | V2_eAL_krig | V2 | spatial | 400 | **EAD α + Lap + Kriging ST**(4-way 主候选)⭐ |
| 13913 | V2_tta_13860(重提)| — | — | — | TTA on 13860 ckpt |
| 13914 | V2_semi_spatial_da(重提)| V2 | spatial | 400 | Distribution Alignment(MMD)|
| 13915 | V2_semi_spatial_cons_mid | V2 | spatial | 400 | Consistency 中间档(scale=1.3, λ=0.2)|
| 13916 | V2_semi_spatial_mrecon | V2 | spatial | 400 | **Mask Reconstruct multi-task**(λ=0.1, K=30, target=UF 17d)⭐ |
| 13958 | V2_semi_spatial_eadA_lap_sage | V2 | spatial | 400 | **SAGE + EAD α + Lap**(对照 GraphConv 13897)|
| 13959 | V2_semi_spatial_eadA_lap_gat | V2 | spatial | 400 | **GAT + EAD α + Lap**(对照 GraphConv 13897)|
| 13960 | V2_semi_spatial_sage | V2 | spatial | 400 | **SAGE + naive semi**(对照 GraphConv 13860)|
| 13961 | V2_semi_spatial_eadA_sage | V2 | spatial | 400 | **SAGE + EAD α**(对照 GraphConv 13875)|
| 14916 | V2_semi_spatial_pretrain | V2 | spatial | 400 | **Stage 1: Modality-Aware Masked Pretraining**(WRF non-Tair + CLMS,mask ratio=0.25,无 T label)⭐ |
| 14925 | V2_semi_spatial_eadA_lap_ftpt | V2 | spatial | 400 | **Stage 2: Finetune from 14916 pretrain**(EAD α + Lap λ=0.1)⭐ |

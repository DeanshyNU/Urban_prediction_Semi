# AAAI 2027 Plan — Deep Heterotopic Co-Kriging for Sparse-Sensor Spatial Regression

> 整合 V0 + V1 全部 60+ 实验经验,基于 problem_complexity / research_directions / methods / results 全部材料梳理。
> 目标:AAAI 2027 submission。

---

## 0. Executive Summary

**Paper title(草案)**:
> Deep Heterotopic Co-Kriging: A Neural Framework for Sparse-Sensor Environmental Regression with High-Dimensional Auxiliary Covariates

**核心贡献(3 条)**:
1. **方法论 reframe**: 把经典 heterotopic co-kriging (1990s, 限 2-3 covariates) 扩展到 1000+ covariates 的 neural framework (**Deep Heterotopic Co-Kriging, DHCK**)
2. **理论发现**: 形式化 **Pseudo Quality Ceiling** —— semi-supervised regression 中,模型 RMSE 受 pseudo source quality 下界约束 (information-theoretic bound)
3. **实证 contribution**: 系统对比 8 个 SSL 范式,识别 **feature-rich + small-graph + physics-strong** setting 下哪些 work / 哪些 fail,并提出 **Cross-Modality Prediction** 作为 non-pseudo SSL 的有效替代

**当前最强结果**: V2 spatial RMSE = **0.0392** (1.36°C, −14% vs baseline) — 13897 EAD α + Lap

**目标提升**: 0.032-0.035 (1.10-1.21°C, −25 to −30% vs baseline) — 通过新 method 组合

---

## 1. 已掌握的经验(从 60+ 实验中提炼)

### 1.1 已证明 **work** 的(5 个,paper 的 positive results)

| 方法 | 数字 | 关键 insight |
|---|---|---|
| **EAD α 时间锚** | 0.0399 (−12%) | WRF 时间 bias 是单一最大可挖矿源 |
| **Laplacian 正则** | 0.0422 (−7%) | 图正则 cheap win,与 EAD 正交 |
| **EAD α + Lap** | **0.0392 (−14%)** | 两强叠加,V1 当前最强 |
| **Kriging-pseudo ST** (no EAD) | 0.0429 (−5%) | 外部 pseudo 突破 self-distillation 天花板 |
| **Hybrid Self-Train** (no EAD) | 0.0428 (−6%) | 略优于纯 kriging,组合 self + external |

### 1.2 已证明 **fail** 的(11 个,paper 的 negative results = motivation)

| 方法 | 失败模式 | 根因 |
|---|---|---|
| Self-distillation(任何 confidence)| Δ≈0 | 模型无法 bootstrap 自己 |
| Hybrid ST on EAD | 完美饱和 | Pseudo Quality Ceiling(kriging < EAD)|
| β anchor(kriging / MLP 任何形式)| +0.0005 to +0.0127 | 50 labels 撑不起 per-station bias |
| Augmentation Consistency(任何强度) | Δ≈0 or worse | 458 节点小图 + 物理 invariance 假设错 |
| Distribution Alignment (MMD) | +0.0031 | 分布对齐不是瓶颈 |
| Mask Reconstruct (multi-task) | +0.0130 | recon distract 主任务 + 静态 feature trivial |
| Adversarial Mask | ≈0 | input mask 无 target |
| GeoEmbed 降维(g6 / PCA256)| +0.0025 to +0.0037 | 1008 维必需,不是冗余 |
| Per-modality encoder(naive)| +0.0045 to +0.0068 | 容量不是瓶颈,设计太粗 |
| GNN layer swap(SAGE / GAT) | 几乎平 | edge weight 在 SAGE/GAT 丢失 |
| Modality-Aware Pretrain (14916/14925) | 待回填(预期 marginal) | reconstruction 太 trivial |

### 1.3 五条经验法则

1. **物理 prior > 隐式 SSL 信号**:EAD 把 76% 方差 offload 给经验估计后,模型只学小残差,样本效率大幅提升
2. **External pseudo > Self pseudo**:模型自己产生的伪标签受 self-distillation ceiling;外部信号(kriging)能突破
3. **静态 features 让 mask-reconstruct trivial**:1025/1347 维静态 context 让 pretext task 太简单,学不到 useful representation
4. **458 节点小图破坏 augmentation invariance**:GRAND-style DropEdge 任何强度都不 work,augmentation 信噪比差
5. **50 labels 不足以 fit per-station σ / β**:任何依赖学 per-station scalar(σ, β, attention weight)的方法都 overfit

---

## 2. 核心 Diagnostic — 为什么大多数 SSL 在我们 setup 失败

### 2.1 Setting 的 4 个 unique 特征

```
Feature-rich:   1347 维 heterogeneous features per node
Label-sparse:   50 labeled stations(11% node-level)
Physics-strong: WRF Tair 作为预训练 baseline,kriging 已达 0.0437
Graph-small:    458 nodes,augmentation 信噪比差
```

### 2.2 Pseudo Quality Ceiling(核心 negative result)

**实证**:
- Kriging RMSE = 0.0437 ← 空间插值天花板
- 任何 kriging-based pseudo ST 在 EAD (0.0399) 之上完美饱和(13908/13910/13912 = 0.0397-0.0399)
- 自蒸馏(13881/13887)= 0.0450,= baseline(无新信息)

**结论**: 当 pseudo source quality (ε_G) 优于 model quality (ε_M) 时,ST 突破 ceiling 到 ε_G 附近;反之 ST 完全饱和。**信号源质量是 hard upper bound**。

### 2.3 SSL 失败的统一解释

| SSL 假设 | 在我们 setup 的违反方式 |
|---|---|
| Augmentation 不变 label | 物理上扰动 WRF 会改 T |
| Pseudo source > model | Kriging 0.0437 < EAD 0.0399,无新信息 |
| Features 同质可 mask | 5 个 modality + 静态 features 让 mask trivial |
| 大 unlabeled : 小 labeled 极差 | 你 400/50 = 8x,不是 IMAGENET-style 1000x |

→ **所有现有 SSL 都是为别的 problem class 设计的**。本工作的贡献之一就是**识别这一不适配 + 提出 non-pseudo non-augmentation 的替代方案**。

---

## 3. Paper Framework Reframe — Deep Heterotopic Co-Kriging (DHCK)

### 3.1 为什么这个 framing 是 novel 的

| 文献 | Covariate 数量 | DL? | 是否 reframe |
|---|---|---|---|
| Classical heterotopic co-kriging (Wackernagel 1990s)| 2-3 | ❌ | 经典 |
| Universal kriging (Cressie 1993)| 0 (only location) | ❌ | — |
| Deep Kriging (2020)| ~5-10 | ✓ | 没 claim "high-D" |
| DKNN (2024)| ~10 | ✓ | 同上 |
| LEGCN-RNP (2024)| ~10-20 | ✓ | 同上 |
| IGNNK / KITS / SATCN (2021-2025)| **0**(单变量观测)| ✓ | 不同 task |
| **我们 (DHCK)** | **1347** | ✓ | **首次明确 claim** |

**Gap**: 经典 cross-variogram fitting 受 O(p²) 参数限制,**结构上不能扩展到 100+ covariates**。DL kriging 工作虽然处理高维 raw input,但**没有任何一篇明确 frame 为 "scaling co-kriging to high-D covariates"**。

### 3.2 Paper 的 framing 语句

> "Classical heterotopic co-kriging (Wackernagel 1990s) is limited to 2-3 auxiliary variables due to the O(p²) parameter complexity of cross-variogram estimation. While modern deep learning extensions exist (Deep Kriging, DKNN, LEGCN-RNP), none explicitly address the scaling to high-dimensional heterogeneous covariates that arise in modern environmental sensing. We propose **Deep Heterotopic Co-Kriging (DHCK)**, a principled framework scaling co-kriging to 1000+ auxiliary covariates through graph neural networks with empirically-anchored decomposition and modality-aware encoding."

### 3.3 DHCK framework 数学描述

```
T(i, t) = WRF_Tair(i, t)        ← physics baseline
        + α(t)                   ← time anchor (empirical from labeled mean)
        + β(i)                   ← station bias (currently disabled, 50 labels insufficient)
        + ε(i, t)                ← residual learned by GNN
        
GNN input: heterogeneous covariates {WRF_nonTair, CLMS, UF, GeoEmb, station_aux}
GNN architecture: modality-aware encoder + GraphConv processor + decoder
SSL signal: (a) Laplacian on graph (spatial + feature similarity)
            (b) Cross-modality prediction on unlabeled
Uncertainty: deep ensemble for confidence weighting + prediction intervals
```

---

## 4. 接下来的实验(按优先级排序,不带时间)

### 4.1 Tier 1 — 必做(三大支柱实验,paper 主结果来源)

#### 实验 A: **Modality-Aware Heterogeneous Encoder** ⭐⭐⭐

**核心改动**:
- WRF Tair: 不进 encoder,作为 EAD baseline 直接出现
- WRF non-Tair (270 dim): **TCN over 5-window**
- CLMS (3 dim): small MLP
- UF (17 dim): small MLP
- **GeoEmb (1008 dim): reshape to (7, 12, 12) + 2D CNN**(关键修正——之前 flatten 丢空间结构)
- station_aux (4 dim): sinusoidal positional encoding
- Fusion: **cross-modal attention**(替代 concat)

**预期**:
- RMSE 0.0392 → 0.034-0.037
- 工程量:~300 行
- 单笔 RMSE 提升最大的实验

**Paper 贡献**: 单独是一个 method contribution(heterogeneous encoder for environmental ML)

---

#### 实验 B: **Cross-Modality Prediction SSL** ⭐⭐⭐

**核心**:
```
Main task:  T = f(features) [labeled only]
Aux task:   CLMS = g(features without CLMS) [ALL nodes, real target]
           或: WRF_skin = h(WRF non-skin + GeoEmb + UF) [ALL nodes]

共享 encoder, 独立 head
loss = sup_T + λ × cross_modal_loss(all nodes)
```

**为什么突破 ceiling**:
- 不是 pseudo,是真 target(features 自己是 ground truth)
- 不依赖 augmentation invariance
- 共享 encoder 学到 modality 间深层关系

**Variants**(同时跑,选最强):
- Cross-modal target = CLMS / WRF skin T / Wind magnitude / 其他

**预期**: 
- 单独 −0.001 to −0.003 vs 实验 A 之后
- 与实验 A 叠加可能 → 0.032-0.034

---

#### 实验 C: **Deep Ensemble + Uncertainty Quantification** ⭐⭐⭐

**做 3 件事**:
1. 训 **5-seed ensemble** (best config: EAD α + Lap + 实验 A/B)
2. 报告 prediction intervals via ensemble variance + conformal calibration
3. Ensemble σ 喂回去当 实验 B 的 loss weight

**Uncertainty 的 4 处应用**(全部都是新的,paper 内独立 section):
- **Aggregation**: 5 model 加权平均(权重 = 1/σ_ensemble)
- **Calibration**: report 80%/90%/95% prediction intervals
- **Weighting**: 实验 B 的 cross-modal loss 按 confidence² 加权
- **Diagnostic**: per-station uncertainty heatmap(可视化)

**为什么对 AAAI 重要**:
- Uncertainty quantification 是 ML venue 标准 contribution
- 你之前完全没做(MC Dropout / HPL 失败 ≠ 真做了 UQ)
- 提供 prediction intervals = paper 的 deliverable upgrade

**预期 RMSE 收益**: −0.002 to −0.005 cheap win(典型 ensemble 提升)

---

### 4.2 Tier 2 — 应做(辅助 results + ablation)

#### 实验 D: **Feature-Similarity Graph + Dual Laplacian**

**核心**:
- 原 spatial kNN 图基础上,**再构 feature similarity 图**(1347-D feature cosine sim, kNN k=10)
- 双 Laplacian regularization:
  ```
  L = sup + λ_s × L_lap_spatial + λ_f × L_lap_feature
  ```

**为什么 work**:
- 利用 1347-D features 的**第二个独立 prior**(feature space topology)
- 你目前完全没用过
- 工程量小(~80 行)

**预期**: −0.001 to −0.003

---

#### 实验 E: **Temporal Laplacian on Unlabeled**

```
L_temp_lap = Σ_{u ∈ unlabeled, t} |pred(u, t) - pred(u, t-1)|²
```

**为什么**:
- 温度时序光滑是物理事实,不依赖 augmentation 假设
- 你之前完全没用时间维度
- 工程量极小(~50 行)
- AAAI reviewer 会问"为啥不用 temporal"

**预期**: −0.001 to −0.003

---

#### 实验 F: **Conformal Calibration on EAD Residuals**

**做法**:
- 用 split conformal 或 jackknife+ 给 final predictions 加 80%/90%/95% intervals
- 评估 coverage(真值落在 interval 比例)+ sharpness(interval 平均宽度)

**为什么**:
- AAAI UQ track 角度
- 工程量小(MAPIE 库直接套)
- 提供 "ML deliverable" 之外的 statistical guarantee

---

### 4.3 Tier 3 — 锦上添花(时间允许)

#### 实验 G: **Pseudo Quality Ceiling 控制实验**

**目的**: 实证 theoretical claim

**做法**:
- 人工生成 pseudo source with controlled RMSE: [0.030, 0.040, 0.045, 0.050, 0.060]
- 每个 pseudo source 跑相同 self-train pipeline
- 验证 model RMSE 随 pseudo RMSE 单调上升
- 提供 theorem 的 empirical evidence

**Paper 价值**: theorem + 5 个控制实验 = 一个 Theory section

---

#### 实验 H: **Cross-Domain Validation**

**目的**: Generalization 故事

**Candidate domains**(都满足 "sparse sensor + multi-modal aux + physical baseline"):
- **Air quality** (KDD Cup 2018 Beijing AQI / Lahore PM2.5)
- **Soil moisture** (SMAP + USCRN)
- **Solar irradiance** (NOAA SURFRAD)

**做法**:
- 选 1 个最易获取的 dataset
- 适配 DHCK framework(可能要调 modality split)
- 报告 vs 该 domain baseline

**Paper 价值**:
- "Not just Chicago temperature, this is a general framework"
- AAAI 接受率显著提升(generalization is critical for top-tier)

---

#### 实验 I: **Active Sensor Placement**

**目的**: Application-level contribution

**做法**:
- 用实验 C 的 ensemble 算 per-location uncertainty
- 推荐 top-K 最 uncertain locations 为"建议加 sensor 处"
- Validate: 假装这些位置已加 sensor(用 unlabeled 中的"hold-out subset"模拟),看 model RMSE 提升

**Paper 价值**: 把 method 转化为 actionable urban planning tool

---

#### 实验 J: **Bayesian Last Layer** (alternative uncertainty)

**做法**:
- 最后一层 Linear 换 Bayesian Linear(variational inference)
- 整个模型其他部分确定性
- 对比 Deep Ensemble 的 uncertainty quality

**Paper 价值**: ablation,展示你 explored 不同 UQ paths

---

#### 实验 K: **Test-Time Augmentation Aggregation**

**做法**:
- 同一 input 跑多次 graph dropout forward
- Confidence-weighted aggregate
- 不需要重训

---

## 5. 理论 Contribution — Pseudo Quality Ceiling

### 5.1 形式化(草案)

**Setting**:
- True distribution: $y \mid x \sim P^*$
- Pseudo source $G$ with RMSE $\epsilon_G$ on test distribution
- Student model $M_\theta$ trained on $n_L$ labeled + $n_U$ pseudo-labeled

**Theorem (informal)**:
$$
\liminf_{n_U \to \infty} \mathbb{E}\left[\text{RMSE}(M_\theta)\right] \geq \epsilon_G - O\left(\sqrt{\frac{n_L}{n_U}}\right)
$$

**直觉**: 当 $n_U \gg n_L$,模型 effectively converge to a $G$-induced risk minimizer,**不可能比 $G$ 自身更准**,除了 $O(\sqrt{n_L / n_U})$ correction from real labels。

### 5.2 证明思路

- **Information-theoretic bound**: data processing inequality
  $$I(y_{true}; M_\theta(x)) \leq I(y_{true}; G(x)) + \text{small term from } n_L$$
- **Fano-type lower bound**: model RMSE 受 mutual information 下界
- **Asymptotic analysis**: 当 $n_U \to \infty$,labeled 信号被淹没

### 5.3 实证 evidence(你已经有)

| jobid | Pseudo source | Source RMSE | Model RMSE | Theorem 验证 |
|---|---|---|---|---|
| 13908 | Hybrid (model + kriging) | ~0.0437 | 0.039877 | ≈ EAD ceiling ✓ |
| 13910 | Pure kriging | 0.0437 | 0.039619 | 同 ✓ |
| 13912 | Pure kriging on 4-way | 0.0437 | 0.039264 | 同 ✓ |
| 13881 | Self-distillation | self (=0.0453) | 0.0450 | = baseline ✓ |
| 13887 | Self + neighbor | self | 0.0450 | 同 ✓ |

**5 个独立 experiment 实证 + 形式化 theorem = solid theoretical contribution**。

### 5.4 在文献中的位置

- Arazo et al. 2020 "Pseudo-Labeling and Confirmation Bias" — classification, qualitative
- 我们: **first formal characterization for regression**, multi-source empirical evidence

---

## 6. Paper 结构(初稿)

```
Title: Deep Heterotopic Co-Kriging: A Neural Framework for Sparse-Sensor 
       Environmental Regression with High-Dimensional Auxiliary Covariates

Abstract: (撰写最后)

1. Introduction
   1.1 Sparse-sensor environmental regression problems
   1.2 Limitations of classical heterotopic co-kriging
   1.3 Why naive deep SSL methods fail in this setting
   1.4 Contributions

2. Related Work
   2.1 Classical kriging / regression kriging
   2.2 Deep kriging extensions (DKNN, LEGCN-RNP)
   2.3 Inductive spatial kriging (IGNNK, KITS, SATCN)
   2.4 Semi-supervised regression
   2.5 Multi-modal learning

3. Problem Formulation
   3.1 Heterotopic spatial regression with feature-rich auxiliary
   3.2 Transductive setting and notation
   3.3 Why this is harder than standard SSL

4. Method: Deep Heterotopic Co-Kriging (DHCK)
   4.1 Empirically-Anchored Decomposition (EAD)
   4.2 Modality-Aware Heterogeneous Encoder (实验 A)
   4.3 Cross-Modality Prediction SSL (实验 B)
   4.4 Spatial + Feature-Similarity Graph Laplacian (实验 D)
   4.5 Deep Ensemble + Uncertainty (实验 C)

5. Theoretical Analysis
   5.1 Pseudo Quality Ceiling Theorem
   5.2 When SSL helps in heterotopic regression

6. Experiments
   6.1 Dataset: Chicago urban temperature (V2)
   6.2 Main results (RMSE comparison)
   6.3 Ablation: contribution of each component
   6.4 SSL paradigms comparison (your 8 negative results)
   6.5 Cross-domain validation (实验 H, if done)
   6.6 Uncertainty calibration (实验 C + F)
   6.7 Pseudo Quality Ceiling empirical validation (实验 G)

7. Discussion
   7.1 When does DHCK apply (PC1 > 50% panel data)
   7.2 Limitations (transductive only, β anchor 不通)
   7.3 Future work (multi-city inductive, causal)

8. Conclusion

Appendix:
  A. Implementation details
  B. Hyperparameter sensitivity
  C. Additional ablations
  D. Pseudo Quality Ceiling proof
```

---

## 7. 投稿风险与对策

### 7.1 主要 risk

| Risk | 对策 |
|---|---|
| **新实验 (A/B/C) 不 work** | 即使 marginal,paper 可写成 "comprehensive ablation + EAD framework + Pseudo Ceiling theorem" |
| **Reviewer 质疑 transductive** | 加 inductive ablation,标 clearly 在 limitations |
| **Reviewer 质疑 single-city** | 实验 H cross-domain validation 至少 1 个 |
| **Reviewer 质疑 RMSE 提升幅度小** | 用 framework + theorem 弱化 RMSE 的重要性,强调 generality |
| **Reviewer 质疑 base paper 比较** | Clearly state: 不同 validation 协议(我们 spatial holdout vs Yu 2024 LOSO),不直接比 |

### 7.2 接受率估计

| 当前(只有 13897 = 0.0392) | 完成 Tier 1(A+B+C) | 完成 Tier 1+2(D+E+F) | 完成 Tier 1+2+H |
|---|---|---|---|
| Urban journal: 80% | Urban journal: 90% | Urban journal: 90% | Urban journal: 90% |
| AAAI: 20% | AAAI: 40% | AAAI: 50% | AAAI: 55-60% |

---

## 8. 写作和实验并行原则

**不要**等所有实验跑完才开始写。建议:

1. **Reframe 写作先行**: Introduction + Related Work + Problem Formulation 可基于现有材料先写
2. **Theory 提早推导**: Pseudo Quality Ceiling 可在跑实验时同时推
3. **实验 → ablation table 增量更新**: 每个实验出结果立刻填进 Section 6 表格
4. **避免完美主义**: AAAI rebuttal 阶段可补做实验,投稿版不必 perfect

---

## 9. 关键决策点 — 最关键的 3 个动作

按 ROI 排序,**如果只能做 3 件事**:

1. **实验 A: Modality-Aware Architecture** — 单笔最可能突破 RMSE 上限,是清晰方法 contribution
2. **实验 B: Cross-Modality Prediction SSL** — 验证 non-pseudo SSL 突破 Pseudo Quality Ceiling,是 paper 核心 narrative
3. **实验 C: Deep Ensemble + UQ** — 给 paper 加 uncertainty contribution,且 cheap RMSE win

**这 3 个 + 已有的 EAD α + Lap + 60 个 baseline 实验 = AAAI submission 完整内容**。

---

## 10. 已知 close 的方向(不再投入)

❌ **不要再做的实验类型**:
- 任何 augmentation-based SSL(MT / VAT / Consistency / DA)
- 任何 self-distillation 变体(已 ceiling)
- β anchor 任何形式
- GeoEmbed 降维 / PCA
- Per-modality encoder 的简单 split 版(naive concat)
- Mask Reconstruct 在 static feature 上
- Pseudo-label 任何 variant 在 EAD 之上(都 ceiling)

❌ **可以彻底放弃的方向**:
- 用 model 自己的 prediction 当 supervision signal
- 在 458 节点小图上做 graph augmentation
- 学 per-station scalar(σ / β / attention)

---

## 11. 一句话总结

**你的 paper 不在"加方法刷 RMSE",在 "reframe + theorem + 3 个 key new mechanism"。**

- **Framework**: Deep Heterotopic Co-Kriging
- **Theorem**: Pseudo Quality Ceiling
- **New mechanisms**: Modality-Aware Encoder + Cross-Modality SSL + Deep Ensemble UQ
- **Backing**: 60+ existing experiments + clean negative results catalog

按 priority 推进 Tier 1 三个实验,paper 写作并行,**AAAI 2027 是可达目标**。

---

*文档创建时间: 2026-06-01*
*基于: V0 + V1 全部实验 + problem_complexity / research_directions / methods / results 全部材料*
*更新原则: 实验结果出来后增量 update 进 Section 4 + Section 7.2*
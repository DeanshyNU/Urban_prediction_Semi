# 我们这个任务的复杂性 / 特殊性 — 完整剖析

> 这份文档总结**我们的问题为什么不是标准 ML 设置**,以及**为什么大多数现有 SSL / GNN / kriging 方法不适用**。供 paper 写作 / PPT framing / 跟 reviewer 解释 niche 用。

---

## 0. 问题精确定义

**任务**:基于稀疏气象站观测 + 数值天气预报(NWP)+ 多模态环境特征,推断城市内未观测位置的小时温度。

**设置**:
- 58 个 labeled 气象站(分 50 train + 8 valid,FPS spatial split)
- 400 个 unlabeled 辅助点(FPS 选自 2000 候选)
- 3672 时间步(2018-05 至 2018-09,hourly)
- 每个 (station, time) 样本含 **1347 维异构 features**
- **Transductive setting**:训练时整张图(458 节点)所有 features 都参与,只有 valid 的 T label 不进 loss

**核心矛盾**:**1347 维 features × 1.7M 样本 vs 50 站 × 3672 = 18 万 labels**(features 极富,labels 极少)。

---

## 1. 七个具体的复杂性 / 特殊性维度

### 1.1 **Feature-rich auxiliary stations**(vs 文献的 feature-scarce 假设)⭐⭐⭐

**什么**:400 个 unlabeled 不只有坐标,它们**每一个都带完整 1347 维 features**(WRF + GIS + 卫星 embedding)。

**为什么是 niche**:文献的 sparse sensor 工作(KITS / SATCN / IGNNK)假设 unobserved 节点是**空壳**(只有 location),要靠 virtual node feature fusion 或 mask token 给它们 features。**我们方向反过来:features 富余,缺的是 labels**。

**对方法选择的影响**:
- KITS 风格(给空 node 造 features)→ **完全不适用**
- BERT/MAE 风格(学 representation)→ **features 已经是 representation,学不到新东西**(14916/14925 实证)
- 真正需要的是:**如何更好地"用"现有 features 而非"学"它们**

**实验证据**:13916(Mask Reconstruct UF static)失败 +0.013;14925(STD-MAE style)失败 +0.00086 — **两次确认 reconstruction-based SSL 无效**。

---

### 1.2 **Heterogeneous multi-modal features**(5 种性质不同的 modality)⭐⭐⭐

**什么**:1347 维不是一个统一的 representation,而是 **5 种本质不同**的 modality 拼接:

| Modality | 维度 | 时变? | 物理角色 |
|---|---|---|---|
| **WRF Tair**(NWP 气温预测)| 45(5 window × 9 ch)| ✓ | **物理 baseline**(85% 方差贡献)|
| **WRF non-Tair**(风/湿/辐射)| 270 | ✓ | dynamic modifier |
| **station_aux**(时间索引)| 4 | ✓ | 时间锚 |
| **CLMS**(植被 LAI/NDVI/Fraction)| 3 | ✓(daily)| 静态调节(蒸腾)|
| **UF**(城市形态 17 维)| 17 | ❌ | **静态 modulator**(建筑密度 SHAP=0.665)|
| **GeoEmb**(卫星 1008 维)| 1008 | ❌ | 静态 fine-grained |

**为什么是 niche**:
- 文献的 SSL 假设 features **同质**(图像 patches / 文本 tokens / 单变量时序)
- 我们的 5 种 modality **scale、autocorrelation、信息密度、时变性**都不同
- "Mask 一个 patch 然后重建"在同质数据上 work,在异构数据上**重建难度差异极大**:
  - Mask UF → trivial(邻居复制)
  - Mask WRF non-Tair → trivial(从 Tair 推)
  - Mask GeoEmb → trivial(静态)
  - **没有 modality 真正能产生"难的" pretext task**

**对方法选择的影响**:**generic SSL (BERT/MAE/MoCo) 完全不考虑 modality role,直接套用必然失败**。需要 **modality-aware 设计**(我们 14916 尝试了,失败,见 [methods.md §7.9](methods.md))。

**实验证据**:13884 / 13901 Per-modality encoder 失败(+0.0045/+0.0068)—— 直接 modality split 也不是答案。

---

### 1.3 **NWP 物理 baseline 已经吃掉 85% 方差**(EAD 起作用的真正原因)⭐⭐

**什么**:WRF Tair 作为 forecast 已经是个很好的 baseline,**真值 T 跟 WRF Tair 的 RMSE = 0.0437**(kriging 标准),**80%+ 方差由 WRF Tair 解释**。

**为什么是 niche**:
- 文献的 spatial regression 通常没有物理 baseline,模型从零学整个映射
- 我们有 WRF → 模型只学**残差** `T - WRF`(即 EAD α + β + ε)
- 这让"标签信息"变得比 raw T 还稀疏(残差 magnitude < T magnitude)

**对方法选择的影响**:
- **任何不利用 WRF baseline 的方法都是浪费数据**
- EAD 框架(`T = WRF + α + β + ε`)对应**经典 regression kriging + external drift**,1990s 起标准做法
- DL 真正要学的不是"哪里热",是"WRF 在哪里 wrong by how much"

**实验证据**:13875 EAD α(只用 WRF temporal bias 校正)= 0.0399,Δ = **−12%** vs 13860 baseline。**单一最大改进点**。

---

### 1.4 **小图(458 节点)+ Augmentation-based SSL 失败**⭐⭐

**什么**:我们图只有 458 节点(50 + 8 + 400),**任何 augmentation-based SSL(consistency / Mean Teacher / contrastive)需要**:
- DropEdge / DropNode → 直接破坏物理 spatial smoothness
- 强假设"模型对 augmentation 不变" → **温度场对扰动本来就 SHOULD 不变**,这个假设跟物理对立

**为什么是 niche**:文献小图 SSL 几乎不存在,GNN SSL paper 都在 citation network / OGB(几千到几十万节点)上做。我们 458 节点**信息密度太低**,augmentation 制造的 view 信噪比差。

**对方法选择的影响**:**整条 augmentation 路线完全 close** —— Mean Teacher、Contrastive、VAT、ICT 全部不可行。

**实验证据**:
- 13886 Consistency K=3 → Δ ≈ 0
- 13915 Consistency 中档 → +0.0021 ✗
- 13902 Consistency 强档 → +0.023 ✗(过激坍缩)
- 13914 Distribution Alignment(MMD)→ +0.0031 ✗
- 13890 Adversarial Mask → +0.0002

→ **5 个独立失败,统一在"augmentation 假设错"这个 root cause**。

---

### 1.5 **Pseudo Quality Ceiling**:任何 pseudo label 路线被 kriging 上限锁死 ⭐⭐

**什么**:Self-train 给 unlabeled 赋 pseudo label(kriging / 模型自己 / hybrid),然后当 supervised label 一起训练。但:

```
ST_max_improvement  ≤  pseudo_source_RMSE

Kriging 在 train 50 站上的 RMSE = 0.0437(空间插值上限)
→ 任何用 kriging 当 pseudo 的 ST 最多到 0.0437 附近
→ 一旦模型 RMSE 已经 < 0.0437(EAD 后 0.0399),ST 没东西可教
```

**为什么是 niche**:
- 文献 ST 工作通常假设 pseudo source 跟 ground truth 接近(分类问题)
- 回归问题里 pseudo source 自身有 floor,**ST 不能突破这个 floor**
- 这个现象在文献里**没被显式命名 / 量化过**

**对方法选择的影响**:
- 任何 pseudo-label 类方法(ST 变体 / Mean Teacher EMA pseudo / FixMatch-style)都受这个 ceiling 约束
- 真正突破要么**用更好的 pseudo source**(不存在),要么**改 pseudo-label 范式**(改用 representation 等其它信号 — 但我们 1.1 节实证此路也不通)

**实验证据**:
- 13882 Kriging-ST(baseline 之上)= 0.0429,Δ = −5%(逼近 0.0437 ceiling)
- 13908 EAD + Hybrid ST = 0.039877 完美饱和(等于 13875 EAD α 单独 = 0.0399)
- 13910 EAD + Kriging-ST = 0.039619,Δ vs 13875 ≈ 0(噪声内)

→ **Pseudo Quality Ceiling 实证 2 次** —— 模型 < kriging quality 时 ST 完全失效。

---

### 1.6 **Transductive 但带 leak 嫌疑的 spatial split**⭐

**什么**:
- 训练时所有 458 节点 features 都进图(包括 valid)
- 只有 valid 的 T label 不进 loss
- 但 valid 的 features + location 在 kNN 构图、α(t)算、Lap edge weight 等多处被用到

**为什么是 niche**:
- 严格 inductive 要求训练对 valid 完全无感(features 也不能看)
- 我们的 transductive 是**"看 feature 不看 label"** —— 这是个**软 leak**,文献里很少有 paper 显式讨论这个边界
- AAAI 审稿可能问:"严格 inductive 下数字是多少?"

**对方法选择的影响**:
- 现在的 EAD / Lap / ST 都假设 transductive(看 valid features)
- 任何升级到严格 inductive 需要大改 pipeline(~150 行)
- Paper 必须明确 disclose,否则被攻击 "cheating"

---

### 1.7 **Static features 在 SSL 中变成"shortcut"陷阱**⭐

**什么**:UF (17) + GeoEmb (1008) = **1025 维静态 features**,占输入 76%。

**为什么是 niche**:
- 文献 SSL 多在**全动态数据**(时序 / 视频 / 文本)
- 我们一大半 features 静态 → 任何 reconstruction task 模型都可以**"从 UF/GeoEmb lookup 答案"**
- Static features 提供"上下文"是好事,但**让 mask reconstruction trivial 是坏事**

**对方法选择的影响**:
- Mask Reconstruct 系列(MAE / GraphMAE / STD-MAE)必失败 — 静态 context 太富
- 真正可行的 pretext task 必须 **mask 包括静态 features**(但又会破坏 conditioning)
- 这是 "feature richness" 跟 "task difficulty" 的根本张力

**实验证据**:14916 pretrain loss 4.91e-4(450× 缩减,极快收敛)→ **任务太简单的标志**;14925 fine-tune +0.00086 → **学到的 representation 没用**。

---

## 2. 综合:这种复杂性需要什么样的方法?

基于上面 7 个维度,**我们的 problem class 需要的方法应该具备**:

| 必备特征 | 排除哪些方法 | 仍可考虑哪些方法 |
|---|---|---|
| ✓ 利用 WRF 物理 baseline(EAD-style 分解) | 任何不用 WRF residual learning 的 vanilla GNN | EAD α + Lap + kriging-ST(我们的 13897 已实现)|
| ✓ Modality-aware feature handling | Uniform encoder treating 1347 dim equally | Per-modality encoder(但 13884/13901 失败),attention over modality(未试)|
| ✓ 不依赖 augmentation | Mean Teacher / Consistency / VAT / Contrastive | Mask-based(但 14925 也失败)/ Auxiliary task with aligned objective |
| ✓ 不依赖 generic pretrain | STD-MAE / GraphMAE / JEPA / MoCo | 单阶段 / 任务对齐的 multi-task |
| ✓ 不依赖突破 kriging ceiling 的 pseudo | Self-train 任何变体在 EAD 之上 | 替代 supervised signal(causal / heteroscedastic / temporal Lap)|
| ✓ 处理 feature-rich setting | Standard representation learning | 问题分解 / 显式 priors / hierarchical Bayesian |
| ✓ 利用 1.7M (station, time) 数据 | Snapshot-based methods | Temporal regularization / cross-time consistency(只要不靠 augmentation)|

---

## 3. 这个 problem class 对应的文献位置

```
                          ┌─────────────────────────────────────┐
                          │     Classical geostatistics         │
                          │     (Cressie, Hengl, Goovaerts)     │
                          │  - Regression kriging               │
                          │  - External drift kriging           │
                          │  - 1990s-2000s, 非 DL               │
                          └────────────┬────────────────────────┘
                                       │ 
                                       │ ← 我们的 EAD 是这个方向的 DL 化
                                       │
                                       ▼
   ┌────────────────────────────┐    ┌──────────────────────────────┐
   │  Sparse sensor inference   │    │  Multi-modal SSL              │
   │  (KITS, SATCN, IGNNK)      │    │  (BERT, MAE, GraphMAE, CITab) │
   │  - Feature-scarce          │    │  - 同质 features 或 image     │
   │  - Single channel          │    │  - 大数据 + 大 model           │
   │  - Inductive               │    │  - 不考虑 physical baseline    │
   └────────────┬───────────────┘    └────────────┬─────────────────┘
                │                                  │
                │ 我们 ← feature-rich              │ 我们 ← 异构 + small label
                │                                  │
                └────────────┬─────────────────────┘
                             │
                             ▼
                ┌──────────────────────────────┐
                │   ⭐ 我们的 niche             │
                │  Feature-rich + NWP-anchored │
                │  + heterogeneous + small lbl │
                │  + transductive              │
                │                              │
                │   ★ 没有 paper 完全在这里    │
                └──────────────────────────────┘
```

---

## 4. 为什么这是 paper(尤其 AAAI)的卖点

如果我们能**显式刻画这个 problem class**,并在它上面取得有意义结果(无论正向负向),paper 有这些角度:

| 角度 | 卖点 | 风险 |
|---|---|---|
| **新 problem 框架** | "Feature-rich transductive sparse spatial regression"(没人系统命名)| AAAI 审稿可能问 "为什么不是单一应用问题" |
| **系统 negative findings** | 7 个 SSL 方法实证不 work,**揭示 augmentation / pseudo / representation learning 在 feature-rich setup 下的本质局限**| 需要扎实归因,不只是"不 work" |
| **Variance decomposition framework** | EAD = 经典 regression kriging 的 DL 化,−12% 单项最大 | EAD 不是 physics-informed 不能 over-claim |
| **Pseudo Quality Ceiling 现象** | 形式化 + 量化 |  现象简单,需要更多理论分析 |

---

## 5. 当前路线(2026-05 末)

**已 close 的方向**(实证不可行,见 [results.md](results.md) 失败汇总):
- ✗ Augmentation-based SSL(Consistency / DA / Mean Teacher / Contrastive)
- ✗ Pseudo-label SSL 突破 ceiling(Self-distillation / Hybrid ST)
- ✗ Generic representation learning(STD-MAE / GraphMAE / JEPA pretrain + finetune)
- ✗ GNN layer swap(SAGE / GAT 在 semi 上 edge weight 丢失)
- ✗ EAD β(任何形式)/ GeoEmb 降维 / Per-modality encoder

**仍可探索的方向**:
- ⭕ **Causal-style 城市 features 响应分解**(替代 empirical EAD β,AAAI causal track)
- ⭕ **Multi-resolution hierarchical graph**(k=5/20/50 三尺度并行)
- ⭕ **Temporal Laplacian on unlabeled**(无 augmentation 的时序约束)
- ⭕ **Auxiliary task 跟 T 预测信息论对齐**(如 WRF skill modeling)
- ⭕ **Heteroscedastic / per-station uncertainty 建模**

**长期方向**(数据扩展):
- 🔸 NYC Mesonet + WRF + GIS 第二数据集(为 AAAI multi-dataset 需求)
- 🔸 严格 inductive ablation(防 transductive cheating 质疑)

---

## 6. 一句话总结

> **我们的问题不是文献里任何一类标准任务的简单组合,而是 7 个维度同时叠加产生的 unique problem class。它的复杂性不在数据规模或模型容量,而在"features 过于丰富 + labels 过于稀疏 + 物理 baseline 已经很强 + 图过于小"四者的同时存在。这种 setup 下,大多数现代 SSL 方法的根本假设都不成立 —— 这既是挑战,也是 paper 的真正卖点。**

---

## 相关文档

- 详细方法清单和实验记录:[methods.md](methods.md)
- 完整实验结果:[results.md](results.md)
- 数据 schema:[data.md](data.md)
- 矿脉 framework + future directions(部分 stale):[research_directions_analysis.md](research_directions_analysis.md)

---

*文档创建时间: 2026-05-31*
*基于 2026-04 起到 2026-05 的 60+ V1 实验(13848-14925)归纳*

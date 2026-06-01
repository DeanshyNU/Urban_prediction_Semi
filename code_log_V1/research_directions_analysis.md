> ⚠ **STALE NOTICE(2026-05-13 晚补)**:本文档大部分仍有效(4 矿脉 framework、failed methods 总结、paper list 可参考),但**具体方法推荐已过时**:
>   - 推荐的 Mean Teacher / Contrastive **依赖 augmentation,而 augmentation 在 458 节点小图已实证失败**(13886/13902/13915)→ 这两个方法**实际不可行**
>   - 推荐的 MMP(Masked Message Passing)被判定**仅 engineering,非 framework shift**,不适合 AAAI
>   - "Generic anchored decomposition" 被识别为 **over-claim EAD ≠ physics-informed**
>
> **当前真正实施的方向**:Modality-Aware Masked Pretraining(jobid **14916** 在跑)
>   - 详见 [methods.md §7.9](methods.md) 和 [pretrain.py](../code_V1/pretrain.py)
>   - 核心不同:**reconstruction-based pretext task**,**不依赖 invariance 假设**(避开 augmentation 失败的根因)

# 研究方向深度分析(2026-05-13)

> 这份文档总结了当前工作之后**值得深入挖掘的研究方向**,基于:
> - 现有所有 V1 实验结果(13848-13961)
> - 用户提出的三大问题(temporal / unlabeled signal / valid 使用)
> - 2024-2025 相关 paper 的搜索综述
>
> **目的**:作为未来 1-3 个月研究路线的参考,以及 paper 的 Discussion / Future Work 章节素材。

---

## 0. 四矿脉 framework — 当前研究的真正未挖空间

```
                  ┌──────────────────────────────────────────────────────┐
                  │  Level 1: paper-level 大轴(改变 framework)         │
                  └──────────────────────────────────────────────────────┘

  ⛰ 矿脉 A: TEMPORAL              "我们几乎没用时间维度"
                 ↓ unlabeled 站点 3672 时序 → 当前完全压平,等于丢了 99% 的 temporal 信号

  ⛰ 矿脉 B: UNLABELED 信号本质     "现在没有真正涨点的 SSL 信号"
                 ↓ 当前 8 个 SSL 方法只有 Lap(图正则,严格说不算 SSL)+ kriging-pseudo 微弱
                 ↓ 自蒸馏 / consistency / DA / mask recon / advmask 全失败
                 ↓ → 这是 framework-level 的开放问题,不是参数问题

  ⛰ 矿脉 C: VALID 使用方式         "transductive vs inductive 的边界还没探明"
                 ↓ 当前 valid 的 feature + location 在训练时都被"看"(虽不计 loss)
                 ↓ → 涉及 framework 严格性 + 多城市 inductive 升级路径

                  ┌──────────────────────────────────────────────────────┐
                  │  Level 2: 内部优化(在已有 framework 里调)          │
                  └──────────────────────────────────────────────────────┘

  🔧 矿脉 D: SELF-TRAIN 结构优化   K, N_rounds, λ_pseudo, hybrid_α 等参数 sweep
                                  Soft pseudo all-400, kriging confidence weighting
                                  → 量级 ±0.001,paper supplement 价值
```

### 三个 Level-1 矿脉的相互关系

```
  矿脉 B 失败的原因         →  可能因为只在空间上找信号(矿脉 A 没挖)
  矿脉 C 的真正测试         →  必须先解决矿脉 A 和 B(否则 inductive 数字会更差)
  当前最强 13897 EAD+Lap   →  本质是 矿脉 A 微弱版 (EAD α) + 矿脉 B 微弱版 (Lap)
```

→ **三个矿脉相互独立但叠加**:任何一个突破都能让 paper 升级,**两个突破就是顶会级别**。

---

## 1. 矿脉 A — Temporal 利用(架构 + 输入两条路)

### 1.1 输入层面(便宜但增量)

| 改动 | 思路 | 实施 |
|---|---|---|
| WRF window 2 → 5 | 给模型 ±5 小时上下文 | 1 行代码,1 sbatch |
| Periodic encoding | sin/cos(hour), sin/cos(month) → 模型能学昼夜/季节周期性 | 替换 station_aux 4 维 |
| WRF residual time series 作为额外 channel | 模型显式看到 "WRF 残差最近几小时的轨迹" | ~20 行 |

### 1.2 架构层面(真正的研究增量,3 种范式)

#### 范式 1:SATCN/UniSTOK/KITS 风格(spatio-temporal kriging 系)
- **TCN + GNN 双 aggregate**:每个 (u, t) 既聚合空间邻居 (u', t),也聚合时间邻居 (u, t')
- 跟我们问题最直接相关
- Reference:
  - **SATCN** ([arXiv 2109.12144](https://ar5iv.labs.arxiv.org/html/2109.12144)) — masking strategy + 空间聚合 + 时间卷积
  - **KITS** ([arXiv 2311.02565](https://arxiv.org/html/2311.02565v2)) — virtual node + increment training
  - **UniSTOK** / **STA-GANN** / **DarkFarseer** — 更近期变体

#### 范式 2:Diffusion model 做 downscaling(随便想想的可能性)
- 现实情况:**probabilistic generation**,主要用于 grid-to-grid 概率 downscaling 或 ensemble 生成
- 你的任务是**deterministic point estimate**(每 (u, t) 一个数),diffusion 是"杀鸡用牛刀"
- 真正有用的场景:**给每个预测加置信区间** — 这就值得 diffusion;但当前只看 RMSE 不值得切换架构
- **不推荐 Phase 1 做**;**Phase 2 加 uncertainty head 时再考虑**
- Reference:
  - [Generative diffusion-based downscaling for climate (arXiv 2404.17752)](https://arxiv.org/html/2404.17752v1)
  - [Latent Diffusion Model for COSMO_CLM (EGUsphere 2024)](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-2646/)
  - [Spatial Downscaling of SST Using Diffusion (MDPI 2024)](https://www.mdpi.com/2072-4292/16/20/3843)

#### 范式 3:Temporal Transformer 头(轻量级,我最推荐)
- 在 GNN 后加 1 层 Transformer over time:每站的 hidden 序列 [h(u, t-T), ..., h(u, t+T)] 自注意力
- **完全模块化,只加 ~100 行代码**,可与现有 SAGE / GAT / EAD / Lap 全部兼容
- 比 ST-GCN 工程量小,比 diffusion 务实

### 1.3 矿脉 A 推荐路线

| 排名 | 实验 | 工程量 | 预期 |
|---|---|---|---|
| ⭐⭐⭐ | Temporal Transformer head 加在 GNN 后 | ~100 行 | 真正动温度时序 |
| ⭐⭐ | Temporal Laplacian: `|ε(u,t) − ε(u,t+1)|²` 正则 unlabeled 时序平滑 | ~50 行 | 独立新方向 |
| ⭐ | WRF window 2 → 5 | 1 行 | 快速 baseline 验证 |
| 🔹 | Full ST-GCN architecture rewrite | ~500 行 | 长期,大改 |
| 🔹 | Diffusion uncertainty head(Phase 2) | ~200 行 | 论文亮点项 |

---

## 2. 矿脉 B — Unlabeled 信号本质 + EAD 起作用机制 + SSL 框架优化

### 2.1 EAD 起作用的真正机制 — Variance Decomposition

```
T = WRF + α(t) + β(i) + ε
       ↑         ↑         ↑
       ~85% var  ~10% var   ~5% var   ← 估计的方差分解
```

模型本来要 "从零学全部 100% 方差",EAD 把 95% 的方差(WRF + α + β)**预先剥离**,模型只学 ε 的 5% → **样本效率显著提升 + 防过拟合**。

→ 这就是经典 **hierarchical / mixed-effects modeling** 思想:**让简单的 baseline(物理 + 时间锚)吃掉容易学的部分,神经网络专攻难的残差**。

### 2.2 EAD leak 严格检查 — 在 transductive 下完全无 leak ✓

```
α(t) = mean_{i ∈ train_50_stations} (T(i,t) − WRF(i,t))
```

| 检查项 | 状态 |
|---|---|
| α 用了 valid 站点的 T? | ❌ 从未碰过(只用 50 train)|
| α 在 valid 时刻 t 用了 train(t)? | ✓ 用了,但**这是 regression kriging 的标准做法**(同 t 跨站)|
| β 用了 valid? | ❌(只用 train residuals)|
| **严格 transductive**? | ✓ 无 leak |
| **严格 inductive**? | ⚠ α 在 inductive 下需要重写为只用 t' < t 的数据(temporal split)|

**核心**:**在 transductive setting 下完全无 leak**;但如果未来改 inductive,**α 必须改成 only past data**(causal computation)。这点要在 paper supplement 标清楚。

### 2.3 优化 self-train 的逻辑(不是参数,是结构)

搜下来发现 4 个值得 try 的 SSL 范式 — **逻辑层创新**而非参数:

#### ① Soft pseudo with iterative refinement(类 Mean Teacher)
```
当前:  每轮选 40 个 → 固定 pseudo → 训 200 epoch → 下一轮重选
新做法: pseudo = EMA of model predictions(每 epoch 更新)
        loss = Huber(train) + λ × Huber(pseudo, EMA_pred)
        pseudo 不固定 → 跟模型共同进化
```
Reference:
- Mean Teacher ([Tarvainen & Valpola, 2017])
- [Interpolation Consistency Training (IJCAI 2019)](https://www.ijcai.org/Proceedings/2019/0504.pdf)

#### ② Interpolation Consistency Training (ICT) 回归版
```
取两个 unlabeled (u, v) → 算 features 的 mixup:
   x_mix = α × x_u + (1−α) × x_v
预期: model(x_mix) ≈ α × model(x_u) + (1−α) × model(x_v)
→ 这是**线性插值一致性**正则,完全无 target leak
→ 是 manifold smoothness 的强假设,温度场满足
```

#### ③ Co-training(双视角)
```
View 1: 只用 WRF + station_aux  → model_W
View 2: 只用 UF + GeoEmb        → model_S
互相为对方生成 pseudo:用 model_W 的高置信预测训 model_S,反之亦然
→ 利用了 features 多模态的特性,从两个独立信号源互相教
```

#### ④ Virtual Adversarial Training (VAT)
```
找 adversarial 方向 r* 使 model(x + r*) 偏离 model(x) 最大
loss = sup + λ × KL(model(x), model(x + r*))   // 强制模型对 adversarial 鲁棒
→ 不依赖 pseudo source 质量
```

### 2.4 矿脉 B 推荐路线

| 排名 | 实验 | 工程量 | 预期 |
|---|---|---|---|
| ⭐⭐⭐ | **Mean Teacher / EMA soft pseudo** | ~100 行 | 真正突破 Pseudo Quality Ceiling 的可能 |
| ⭐⭐ | **ICT 回归版**(mixup unlabeled 一致性) | ~50 行 | 无 target leak,manifold smoothness |
| ⭐⭐ | **Co-training**(WRF 视角 vs GIS 视角) | ~150 行 | 利用多模态独立性 |
| ⭐ | VAT(adversarial 一致性) | ~80 行 | 鲁棒性正则,与上面叠加可能 |
| 🔹 | 继续 Mask Recon 调参 | — | **不推荐**:已实证伤害主任务 |

**核心:从 "model 自我训" 转向 "model 与不同视角 / 不同 perturbation 一致"**。这才是真正可能突破 ceiling 的方向。

---

## 3. 矿脉 C — Valid / Transductive 内部优化 + Inductive 升级

### 3.1 SATCN-style Masked Message Passing(MMP)— 黄金答案

[SATCN (Wu et al. 2021)](https://ar5iv.labs.arxiv.org/html/2109.12144) 用的就是用户在 Q3 提到的想法:

```
当前我们(naive semi):
  unlabeled 节点既接收消息也发送消息 → 用 features 帮邻居,自己也被邻居影响

SATCN(更激进):
  unlabeled 节点【只接收 不发送】 → 训练时它们贡献"feature 让我自己被聚合"
                                       但不影响其它节点的预测
```

**核心机制 `Masked Message Passing (MMP)`**:
- 在 GNN forward 里,unlabeled 节点的 outgoing edge weight = 0
- 它们的 features 进入自己的 prediction(对 labeled 训练有间接帮助)
- 但**它们不污染 labeled 节点的 hidden**

### 3.2 MMP 在我们 setup 下的 4 个变体

| 变体 | 机制 | 预期 |
|---|---|---|
| **MMP-all**:全部 400 unlabeled 静默 | 防 noisy unlabeled features 污染 train | 可能涨 +0.001-0.003 |
| **MMP-confidence**:低 kriging_struct confidence 的 unlabeled 静默 | 留高质量 unlabeled,屏蔽差的 | 更精细 |
| **Asymmetric kNN**:labeled → labeled 边强,unlabeled → labeled 边弱 | 减弱 unlabeled influence | 软版本 |
| **Adaptive masking**:每 epoch 根据梯度决定屏蔽哪些 | dynamic curriculum | 复杂但 powerful |

### 3.3 其它非"图成员"的 transductive 技巧

1. **基于 valid 相似度加权 train**:训练时对"像 valid"的 train 站 sample 给更高权重 → 类似 importance reweighting
2. **Sample / 子图 mixup**:`x_mix = α × x_train + (1−α) × x_unlabeled`(目标 mixup 也比例插值)
3. **Curriculum unmasking**:训练初期 mask 大部分 unlabeled(噪声),后期逐步放开
4. **Attention-based valid steering**:Bias attention 向 "valid-like features"(我们的 relevance scoring 是雏形)

### 3.4 真正的 Inductive 升级路径(矿脉 C 终极)

#### 严格 inductive 改写需要修的 6 处
| 步骤 | 改动 |
|---|---|
| Target normalization | scl 只用 train 50 算(不能用全部 58 labeled)|
| UF NaN 填充 | median 只用 train 算 |
| kNN 图 | 训练图只含 train + unlabeled(不含 valid)|
| Spatial FPS | 不能用 valid 坐标 |
| EAD α(t) | 改为 causal:只用 t' ≤ t 的 train 数据(temporal causal split)|
| Kriging-pseudo | 推理时再 krige 到 valid |

**工程量**:~150-200 行,**1 周可做**。
**预期数字**:**比 transductive 差 0.005-0.010**(loss of leak benefit)。

#### Multi-City Inductive(顶会路径)
- Chicago train,NYC test → 真正 zero-shot 部署
- 需要找到第二个城市的同 schema 数据(WRF + UF + GeoEmbed + AoT stations)
- **NYC mesonet** 或 **欧洲 EUMETNET** 是候选数据源

### 3.5 矿脉 C 推荐路线

| 排名 | 实验 | 工程量 | 预期 |
|---|---|---|---|
| ⭐⭐⭐ | **SATCN-style MMP (Masked Message Passing)** | ~80 行 | **−2 to −5%** ⭐(最 promising 新方向)|
| ⭐⭐ | Semi-inductive baseline(只改图,不改 stats)| ~30 行 | sanity check,数字落差量化 leak 影响 |
| ⭐⭐ | 严格 inductive 完整 pipeline | ~200 行 | paper supplement 价值 |
| ⭐ | Multi-city / NYC 迁移 | 数据 + 1 周 | **顶会路径** ⭐⭐⭐ |

---

## 4. 总优先级表(基于深入分析)

| 排名 | 实验 | 矿脉 | 工程量 | 预期收益 |
|---|---|---|---|---|
| ⭐⭐⭐ | **SATCN-style Masked Message Passing(MMP)** | C | ~80 行 | **−2 to −5%** ⭐(最 promising)|
| ⭐⭐⭐ | **Mean Teacher + Soft pseudo(EMA)** | B | ~100 行 | 突破 Pseudo Quality Ceiling 的真正可能 |
| ⭐⭐⭐ | **GNN layer swap** (SAGEConv / GAT) | 现有 framework | 1 sbatch | 已提交 13958-13961 |
| ⭐⭐ | **Temporal Transformer head** | A | ~100 行 | 真正动温度时序 |
| ⭐⭐ | **ICT(Interpolation Consistency)** | B | ~50 行 | 无 target leak 的新正则 |
| ⭐⭐ | **Co-training**(WRF vs GIS 双视角) | B | ~150 行 | 利用多模态独立性 |
| ⭐⭐ | **Multi-city / Inductive(NYC 等)** | C | 找数据 + 1 周 | **顶会路径** |
| ⭐ | **WRF window 2 → 5** | A | 1 行 | 快速验证 baseline |
| ⭐ | **Temporal Laplacian on unlabeled** | A | ~50 行 | 独立新方向 |
| ⭐ | **Semi-inductive sanity** | C | ~30 行 | 量化 leak 影响 |

---

## 5. 推荐 2 个月路线图

```
Week 1:
  - 等 13958-13961 (SAGE/GAT) 结果,确定 GNN backbone
  - 同时开始实现 SATCN-style MMP

Week 2:
  - Mean Teacher / EMA 实现
  - ICT 回归版实现

Week 3-4:
  - Temporal Transformer head 实现 + 测试
  - 把 SAGE/GAT + MMP + Mean Teacher + EAD + Lap 全部组合跑一遍
  - 找最强组合作为新 main result

Week 5-6:
  - 严格 inductive pipeline 完整 (~200 行)
  - 报告 inductive vs transductive 数字差
  - 作为 paper supplement 防"transductive 是 cheating"的审稿质问

Week 7-8:
  - 第二城市数据探索(NYC mesonet 等)
  - 开始 multi-city / inductive 实验设计
  - 升级 ML venue 路径打开
```

---

## 6. 相关 paper 列表(2024-2025 最新)

### Spatio-Temporal Kriging / Inductive
- [SATCN: Spatial Aggregation and Temporal Convolution Networks (arXiv 2109.12144)](https://ar5iv.labs.arxiv.org/html/2109.12144) — **MMP 来源**
- [KITS: Inductive Spatio-Temporal Kriging (AAAI 2024)](https://arxiv.org/html/2311.02565v2) — virtual node + increment
- [UniSTOK: Uniform Inductive Spatio-Temporal Kriging (arXiv 2603.05301)](https://arxiv.org/html/2603.05301)
- [STA-GANN: Valid and Generalizable Spatio-Temporal Kriging (arXiv 2508.16161)](https://arxiv.org/html/2508.16161)
- [DarkFarseer: Inductive STK via Hidden Style Enhancement (arXiv 2501.02808)](https://arxiv.org/html/2501.02808v1)
- [Physics-Guided Increment Training for AQI Inference (arXiv 2503.09646)](https://arxiv.org/html/2503.09646v1)

### STGNN / Time Series GNN
- [Spatio-Temporal GNN for Urban Computing Survey (TKDE 2024)](https://arxiv.org/abs/2303.14483)
- [Awesome GNN4TS (TPAMI 2024)](https://github.com/KimMeen/Awesome-GNN4TS)
- [Deep Learning for Spatiotemporal Forecasting in Earth System Science](https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2391952)
- [UrbanGraph: Physics-Informed STGNN (arXiv 2510.00457)](https://arxiv.org/pdf/2510.00457)

### SSL Frameworks (Classification / Regression)
- [Interpolation Consistency Training (IJCAI 2019)](https://www.ijcai.org/Proceedings/2019/0504.pdf) — **ICT 来源**
- [Self-Supervised GNN: A Unified Review (PMC 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9902037/)
- [Semi-Supervised Contrastive Regression with Ordinal Rankings (2024)](https://openreview.net/forum?id=ij3svnPLzG)
- [SGCL: Semi-supervised Graph Contrastive Learning (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124009055)

### Diffusion Downscaling
- [Generative Diffusion Downscaling for Climate (arXiv 2404.17752)](https://arxiv.org/html/2404.17752v1)
- [Probabilistic Spatial Interpolation with Diffusion Models (AMS AIES 2026)](https://journals.ametsoc.org/view/journals/aies/5/1/AIES-D-25-0049.1.xml)
- [Satellite-Guided Diffusion for Arbitrary Resolution Meteorology (arXiv 2502.07814)](https://arxiv.org/html/2502.07814v1)
- [SST Downscaling with Diffusion (MDPI 2024)](https://www.mdpi.com/2072-4292/16/20/3843)
- [Latent Diffusion for COSMO_CLM (EGUsphere 2024)](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-2646/)

---

## 7. PPT / paper 上要明确避免的几个 framing 误区

| 不要说 | 改成 |
|---|---|
| "Our self-train is novel" | "We adopt a 3-metric (confidence + diversity + relevance) self-training scheme [from V0 13451, originally in active learning literature], and report a Pseudo Quality Ceiling phenomenon" |
| "Mask reconstruction works" | "Mask reconstruction fails under feature-rich transductive setting (+0.013), but is exactly what KITS uses successfully for feature-scarce inductive — suggesting feature-scarcity is the critical condition" |
| "First DL approach" | "GNN-based transductive regression kriging with feature-rich auxiliary stations — bridges classical geostatistics + modern GNN-SSL" |
| "All 8 SSL methods compared" | "We systematically evaluate 8 SSL paradigms on a small (458-node) urban graph and find that only methods with explicit anchors (EAD physical prior, Laplacian graph regularization, external kriging pseudo) provide measurable gains — implicit SSL signals universally fail" |

---

## 8. 写在最后:对当前 paper 的诚实评价

| 维度 | 当前(13897 = 0.0392 best) | 加 SATCN-MMP + Mean Teacher 后预期 |
|---|---|---|
| Best valid RMSE | 0.0392 | 0.034-0.036 |
| Novelty | 5/10 | 6.5-7/10 |
| Significance | 7/10 | 8/10 |
| Technical | 8/10 | 8.5/10 |
| ML venue 接收概率 | 30%(Urban journal 80%)| **50-60%(ML venue 可期)** |

**当前 sweet spot**:
- **Urban journal:直接投,80% 接受率**
- **顶会路径:加 MMP + Mean Teacher + inductive sanity 后,有戏**
- **不要再花时间调 self-train 参数**(ceiling 卡死)
- **真正下一步:实现 MMP 和 Mean Teacher,这是单笔 ROI 最高的两个新尝试**

---

*文档创建时间: 2026-05-13*
*基于讨论: 用户三大问题(temporal / unlabeled signal / valid 使用)+ 当前所有 V1 实验结果*

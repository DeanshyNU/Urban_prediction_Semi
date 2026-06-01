# 后续方法 roadmap(明天起做)

当前完成:V1 baseline ×6 + V2 baseline ×6 + V2 GNN 层 ablation ×2 + V2 n_unl 扫描 ×4 = 18 jobs。
下一步:从 baseline 之上加入半监督 / 自监督方法。

所有新方法默认:
- `V1_DATASET=V2`(主推 dataset)
- `V1_VAL_MODE=spatial`(主战场)
- 模型架构 / 优化器 / loss 基底沿用 baseline,只加额外 loss 项 / 流程
- wandb 都进 `urban` 项目

---

## 1. Pseudo-Label / Self-Training

**核心**:base 模型预测 unlabeled → 选置信高的 → 当 pseudo-label → retrain。

| 子变体 | 思路 | env / 实现要点 |
|---|---|---|
| Naive PL | 预测 → 全选 → 加进 loss | 一轮 retrain |
| Confidence-thresholded PL | top-τ% 置信度过滤 | snapshot ensemble σ 或 dropout MC |
| Iterative ST | 多轮迭代,每轮加 K 个最自信 | warm-start each round |
| Conformal-calibrated ST | 5-fold OOF 残差做 σ | kNN-weighted 校准 |

**可能的脚本**:`V2_st_naive_spatial.sh`,`V2_st_iter_spatial.sh`,`V2_st_conformal_spatial.sh`

---

## 2. Consistency Regularization

**核心**:同一 input 的两个 augmented views 输出应一致 → unlabeled 也能算 loss。

| 子变体 | 思路 |
|---|---|
| Π-model | dropout 两次 forward,MSE consistency |
| Mean Teacher | EMA model 当 teacher,student 学 teacher 输出 |
| VAT (Virtual Adversarial Training) | 对 input 加 worst-case perturbation,要求一致 |
| FixMatch | 弱增强预测 → 高置信度选出 → 强增强后预测要匹配 |

**Augmentation 候选**:WRF 加噪,GeoEmbed dropout,edge dropout,node dropout

**可能的脚本**:`V2_consist_pi_spatial.sh`,`V2_meanteacher_spatial.sh`,`V2_vat_spatial.sh`

---

## 3. Graph Regularization

**核心**:邻居节点预测应相近 → unlabeled 通过 Laplacian 直接进 loss。

$$\mathcal{L} = \mathrm{Huber}(\hat{y}_L, y_L) + \lambda_{\mathrm{lap}} \cdot \tfrac{1}{|E|}\sum w_{ij} (\hat{y}_i - \hat{y}_j)^2$$

| 子变体 | 思路 |
|---|---|
| Plain Lap | 直接对 ŷ 平滑 |
| Residual Lap | 平滑 ŷ - WRF_T2(只对 UHI correction 平滑)|
| Adaptive Lap | 学每条边的 α_ij ∈ (0,1) 决定该不该平滑(城/郊边界 α→0) |

**可能的脚本**:`V2_lap_spatial.sh`(λ=0.1),`V2_residlap_spatial.sh`,`V2_adaptlap_spatial.sh`

---

## 4. Mask-Reconstruct

**核心**:训练时随机 mask 掉 K 个 train labeled 站,强制模型从邻居重建它们 → 学到鲁棒的局部插值能力。

```
每 batch 从 50 个 train 站里随机选 K=10 个 mask 掉
→ 用剩 40 个 + 400 unlabeled forward
→ loss 包括:正常 sup loss + 重建 loss(mask 位置的 prediction vs true label)
```

**特点**:不引入 pseudo-label,纯 self-supervised augmentation。
**可能的脚本**:`V2_mask_recon_spatial.sh`(MASK_K=10, λ_recon=1.0)

---

## 5. Multi-task / Auxiliary Supervision

**核心**:除了温度 target,加额外预测头 → 共享 backbone 学到更通用的表征。

可能的辅助任务:
- 预测 CLMS(已有,3 维)
- 预测 UF morph 中某些列(static,需要造 paired data)
- 预测 WRF_T2 偏差(残差预测)
- 预测 GeoEmbed 子集(self-distill 风格)

**可能的脚本**:`V2_multitask_spatial.sh`(主 head 温度 + 副 head CLMS,共享前 5 层)

---

## 6. Pretraining + Fine-tune

**核心**:先用大量 unlabeled 做无监督 pretrain,再用 labeled 做小学习率 fine-tune。

| pretrain 任务候选 | 思路 |
|---|---|
| Masked autoencoder | 随机 mask WRF/UF channel,重建被 mask 的 |
| Contrastive (graph) | 同 station 不同 t 当 positive,跨 station 当 negative |
| Temporal forecasting | 用 t-2..t-1 预测 t,辅助任务,然后下游接温度 |

**可能的脚本**:`V2_pretrain_mae_spatial.sh`(2 stage:`pretrain.pt` → fine-tune)

---

## 时间安排(明天起)

按"实现简单度"排序优先级:

1. **Graph Regularization (Lap)** — 改动最小,半天 + 几 hour 训练
2. **Mask-Reconstruct** — batch 内随机 mask,不需要新 loss head
3. **Self-Training (iterative)** — 已有 V0 实现可参考,1 天
4. **Consistency Regularization (Mean Teacher)** — 需要 EMA model,1 天
5. **Multi-task** — 改 model 输出层,1-2 天
6. **Pretraining** — 2 stage 训练 pipeline,2-3 天

**先动 1 + 2,跑通再说 3-6**。

# 数据 — 来源、特征、归一化、图

## 1. 数据文件(都在 `code/data/`)

| 文件 | 用途 | 内容 |
|---|---|---|
| `GNN_N1_StationMat.mat` | V1 labeled | 68 station × T=2948 × 79 维(col 0=target,1-54=WRF,55-58=station-aux,59-78=raw_geo)|
| `GNN_N1_AJM.mat` | V1 labeled-only graph | dist + similarity 矩阵 |
| `FeaturePatch_401.mat` | V1 labeled UFM | 8 channel × 401×401(drop 第 5 ch → 7 channel)|
| `Unlabeled_Finalized.mat` | V2 raw,作为 V1 的 unlabeled 来源 | 2000 station × T=6624 × WRF 63 ch + UF 17 + CLMS 3 + UFM(2000, 7, 401, 401)|

## 2. 特征拼装(`iDim = 1302`)

V1 labeled 每个 (station, t) 拼成 1302 维向量:

```
[ WRF window         (270) ]   = 5 × 54 channels  (window=2 → t-2..t+2)
[ station_aux        (4)   ]   = raw cols 55-58:hour / month / year / station_id
[ raw geo            (20)  ]   = raw cols 59-78(16 UF morph + 3 CLMS + 1 dist)
[ geoEmbed           (1008)]   = 7-channel × 12×12 avg pool from FeaturePatch_401
```

实现位置:[code_V1/data.py](../code_V1/data.py) 中 `dataGen()` 主体。

### 2.1 station_aux 4 列详解

诊断 `GNN_N1_StationMat.mat` 后确认这 4 列的物理含义:

| col | 数值 | 跨 t 是否变化 | 跨 station 是否相同 | 含义 |
|---|---|---|---|---|
| 55 | [0, 1] | 变 | 同 | **hour / 23**(0.087×23≈2点)|
| 56 | [0.417, 0.667] | 变 | 同 | **month / 12**(May=5/12 到 Aug=8/12)|
| 57 | {2018} | 不变(本轮 V1 全 2018)| 同 | **year**(原值)|
| 58 | [1, 98] | 不变 | **不同** | **station_id**(原值整数)|

**全部保留,与 `original_code/run.py` 严格一致**。

### 2.2 V2 unlabeled 4 列 station_aux 怎么造?

| col | V2 unlabeled 怎么填 |
|---|---|
| 55 hour  | **直接 broadcast V1 在同 t 的值**(所有 station 共享同一 hour)|
| 56 month | 同上,broadcast V1 |
| 57 year  | 同上,broadcast V1(本轮始终 2018)|
| 58 station_id | **每个 V2 unlabeled 一个独立 ID**,从 100 开始(100, 101, ..., 99 + n_unlabeled),避开 V1 的 1-98 范围 |

实现见 [code_V1/data.py](../code_V1/data.py) 中 `for n in range(...)` 循环里构造 `station_aux_u`。

**注意**:col 57 (year) 取值 2018,col 58 (station_id) 范围 1-499,这两列**没有归一化**(faithful original_code)。Linear 第一层会有这两列 magnitude 远大于其它特征(其它都在 [0, 1.5])的情况,但 Adam 自适应学习率能 handle。

## 3. 归一化

| 字段 | 方式 |
|---|---|
| WRF / raw_geo / target | StationMat 文件中**预归一化**好的 [0,1](legacy V1,具体公式未保留)|
| GeoEmbed | per-channel 跨 station min-max([data.py:18-25](../code_V1/data.py#L18-L25))|
| Unlabeled WRF | V2 raw Kelvin → per-feature 跨 (n × T) min-max 归到 [0,1] |
| Unlabeled raw_geo | per-feature min-max 归到 [0,1] |
| Unlabeled GeoEmbed | per-channel min-max 归到 [0,1] |

### 3.1 V1 target 归一化的不确定性 ⚠

**V1 target 在 StationMat 里是预归一化的 [0.065, 0.865]**(实测 range,不是严格 [0, 1]),原始 °C → 归一化的 (min, max) **没保留**。

后果:
- ✅ **normalized 空间的 RMSE 是精确的**(0/1 单位下的 MSE),sup vs semi 内部对比有意义
- ❌ **°C 反归一化只能粗估**:我们假设 `scl ≈ 30K`(Chicago 夏天典型温差),实际可能 25-40K → 报告的 °C 误差是 ±25% 量级的近似值
- ❌ **V1 vs V2 的 °C 数字不可直接比**

**Caveat**:V1 unlabeled 用的是 V2 raw 数据自己 min-max 归到 [0,1],与 V1 labeled 的 legacy [0,1] **scale 不严格对齐** → message passing 时跨 labeled-unlabeled 边的特征 magnitude 略有不同,模型须自行适配。

### 3.2 V2 归一化(精确,有据可查)

[code_V1/data.py](../code_V1/data.py) 的 `load_v2_labeled()`:

| 字段 | 方式 | 实现 |
|---|---|---|
| WRF (63 ch) | per-feature global(每 channel 跨 N×T 一个 min-max)| `_normf()` 末维 min-max |
| CLMS (3) | 同上 | 同 |
| UF (17) | 同上 | 同 |
| GeoEmbed (1008) | 同上 | 同 |
| **Target (`AoT_filled`)** | **GLOBAL 归一化**(单一 (min, max) 跨所有 58 站 × 3672 t)| `(targets - tgt_min) / (tgt_max - tgt_min)` |

实测 `tgt_min=4.35°C, tgt_max=38.93°C → tgt_scl=34.57°C`,写到 `metadata['tgt_scl']`,反归一化时 `RMSE_°C = RMSE_norm × 34.57`(精确)。

**aux 4 列不归一化**(faithful original_code):
- `hour /23`,`month /12` 已经天然在 [0, 1]
- `year` 原值 2018,`station_id` 原值整数 → Adam adaptive lr handle magnitude 差异

### 3.3 V1 vs V2 误差精度对比

| | V1 | V2 |
|---|---|---|
| target 来源 | 预归一化 [0.065, 0.865] | raw °C [4.35, 38.93] |
| 反归一化 scl | **未知**,粗估 30K | 精确 34.57K |
| RMSE_norm 精度 | ✓ 可信(同 norm 下对比) | ✓ 可信 |
| RMSE_°C 精度 | ⚠ ±25% 量级近似 | ✓ 精确 |
| paper 主表用哪个? | 仅作 cross-dataset 验证 | **应该用 V2 报 °C 数字** |

**结论**:V1 sup vs V1 semi 相对比较意义没问题(谁好谁坏排名可信),但 V1 的绝对 °C 误差不要直接写 "1.5°C" 这种值,应注明 "scl assumed ~30K"。

## 4. 图构建

### 4.1 Sup 模式(`V1_N_UNLABELED=0`)
用 V1 自带的 sim×dist 图(原 `original_code` 做法):

```
distW = exp(-dist)
distW = (distW - min) / (max - min)
A = |similarity * distW|
A[A < 0.1] = 0
```

68 节点,3836 条边,无 EDGE_MODE 概念。

### 4.2 Semi 模式(`V1_N_UNLABELED > 0`)
**k-NN k=10** 在合并 (lat, lon) 上重建:

```
for each node i:
    keep i 的最近 k 个邻居 j
    A[i, j] = 1 - dist[i,j] / max(dist)
A = max(A, Aᵀ)   # symmetric
```

n_unlabeled=400 时 → 468 节点,density ≈ 17–18%。

## 5. Unlabeled 节点选择(FPS)

[code_V1/data.py](../code_V1/data.py) `load_unlabeled_v1_aligned()`:

```python
1. 从 Unlabeled_Finalized.mat 加载 2000 station 的 WRF/CLMS/UF/Loc/UFM
2. truncate timestep:V2 t=[V1_OFFSET, V1_OFFSET+T_end) 对齐 V1 t=[0, T_end)
   (见 §6 时间对齐)
3. FPS over (lat, lon) seed=0 → 选 n_select=400 站
4. WRF V2 (63ch) → V1-aligned (54ch):前 45 thermal 不变 + |Wind| = √(WindX² + WindY²) × 9
5. station_aux 4 列:
   - hour/month/year:**broadcast V1 在同 t 的值**(全 station 共享)
   - station_id:V2 每个独立 ID,100..99+n_unlabeled
6. raw_geo: 16 UF morph + 3 CLMS + 1 dist = 20 维
7. geoEmbed: V2 UFM 已经 7-ch,12×12 avg pool → 1008
```

## 6. 时间步对齐(V1 vs V2)

### 6.1 数据时间范围

| 数据 | 时间范围 | 步长 | 总 hours |
|---|---|---|---|
| **V1 labeled** | 2018-05-01 02:00 → 2018-08-31 21:00 | 1h | **2948** |
| **V2 raw Period 1** | 2018-05-01 00:00 → 2018-09-30 23:00 | 1h | 3672 |
| **V2 raw Period 2** | 2019-05-01 00:00 → 2019-08-31 23:00 | 1h | 2952 |
| **V2 raw 总计** | 2 个夏天 | 1h | **6624** |

### 6.2 V1 起点为什么是 02:00 而不是 00:00?

诊断 `GNN_N1_StationMat.mat` 的 cols 55-57 后:

- **col 55 (hour/23) 在 t=0 = 0.087** → 0.087 × 23 ≈ 2 → 2:00am
- **5 月时长**:V1 col 56 显示 5 月共 742 hours,**比 5月满月(744)少 2 小时**
- **8 月时长**:V1 显示 8 月共 742 hours,也少 2 小时
- **6 月、7 月**:正好 720 / 744 完整

所以 V1 是**裁掉了 5月1日 00:00 + 01:00 和 8月31日 22:00 + 23:00**,首尾各 2 小时。猜测原因:WRF window=2 需要 t-2 / t+2,边界两步留作 padding 后被丢弃。

V2 是 raw 数据,没有这种 trim,从 5月1日 00:00 起算。

### 6.3 偏移计算 + 修正

```
V1 t=0 = 2018-05-01 02:00 (=  V2 t=2)
V1 t=n = 2018-05-01 02:00 + n hours  (= V2 t=n+2)
```

`V1_TIMESTEP_OFFSET = 2`。在 [code_V1/data.py:71](../code_V1/data.py#L71) 的 `load_unlabeled_v1_aligned()` 中:

```python
wrf  = wrf[:, :, V1_OFFSET : V1_OFFSET + T_end]   # V2 t=[2, 2950)
clms = clms[:, :, V1_OFFSET : V1_OFFSET + T_end]
```

这样 V2 unlabeled 在 index=n 的数据 = 物理时间 2018-05-01 02:00 + n hours = V1 在 index=n 的物理时间。**完全对齐**。

### 6.4 V1 不覆盖 2019

V1 `col 57 year` 全是 2018(我们已确认),只用 V2 Period 1 的前 2948 小时(其中 2 小时 offset),**Period 2 (2019) 完全没用**。如果未来想用 V2 Period 2 数据,需要 V1 数据扩展到 2019,目前没有。

### 6.5 Sanity 检查的输出

semi 模式启动时,会自动打印:

```
[V2 truncate] using V2 t=2..2949 to align with V1 t=0..2947 (offset=2)
```

如果偏移错了 / V2 数据不够长,这一行会立刻报错。

---

## 7. V2 数据路径(env: V1_DATASET=V2)

### 7.1 来源
- **Labeled_Finalized_new.mat**(58 站,T=3672 = Period 1 only,2018-05-01 → 2018-09-30)
- 同 `Unlabeled_Finalized.mat`(2000 站)用于 unlabeled

### 7.2 V2 schema(`iDim = 1347`)

```
[ WRF window     (315) ]   = 5 × 63 channels(V2 raw,WindX/Y 不合并)
[ station_aux    (4)   ]   = hour/23 + month/12 + year + station_id
[ CLMS at t      (3)   ]
[ UrbanFeature   (17)  ]   = per-station static
[ GeoEmbed       (1008)]   = UrbanFeatureMat 7-ch × 12×12 avg pool
```

**station_aux 4 列怎么造**:
- hour/month/year:基于 V2 t=0 = `2018-05-01 00:00` 用 Python `datetime+timedelta(hours=t)` 算出,broadcast 到所有 station
- station_id:V2 labeled 0..57(per-station 唯一);V2 unlabeled 100..(99+n_unl)
- 实现见 [code_V1/data.py](../code_V1/data.py) 的 `v2_compute_time_features()` + `_dataGen_V2()`

**关键差异 vs V1**:
- V2 没有时间偏移问题(V2 labeled 起 2018-05-01 00:00 = V2 unlabeled 起 0:00,**不需要 OFFSET**)
- V2 target 是 raw °C(`AoT_filled` 范围 [4.35, 38.93]),代码里全局归一化到 [0,1],`tgt_scl ≈ 34.57` 写到 metadata,反归一化时直接用(精确 °C,不再用粗估的 30K)
- WRF 不合并 |Wind|(63ch);V1 合并(54ch)→ 这是 V1/V2 之间唯一不可对齐的 schema 差异

### 7.3 V2 图

| 模式 | 实现 |
|---|---|
| sup(n_unl=0)| `build_v2_adj()`:`Similarity[i,j] = max(0, corrcoef(SimilarityMat[i], SimilarityMat[j]))`,`Adj = Similarity × exp(-dist/max_dist)`,thres=0.1。**Faithful Yu et al. (2024)** |
| semi(n_unl>0)| 复用 `build_knn_adj()` k=10(同 V1 semi)|

### 7.4 V2 unlabeled 怎么造

[code_V1/data.py](../code_V1/data.py) `load_v2_unlabeled()`:
- 加载同一个 `Unlabeled_Finalized.mat`,但**不做 V1-align**(WRF 保持 63ch,**不**合并 WindX/Y)
- **不**做时间偏移(V2 labeled 起点和 V2 unlabeled 起点都是 2018-05-01 00:00)
- truncate 到 V2 labeled 的 T_end=3672
- FPS 选 n_select 站(同 V1 路径的 FPS 算法)

### 7.5 V2 NaN 处理
- V2 labeled UF 有 9 个 NaN:cols 0-15 填 0,col 16(dist to lake)填 median
- V2 labeled UFM 有 ~16M NaN(图像 padding 区域),全填 0
- V2 unlabeled 同理

### 7.6 V2 spatial valid
默认 `V1_N_VALID_STATIONS=8`(58 站太少,留 50 train + 8 valid),FPS seed=42。

### 7.7 V1 vs V2 schema 速查表

| 项 | V1 | V2 |
|---|---|---|
| labeled .mat | GNN_N1_StationMat | Labeled_Finalized_new |
| 站数 | 68 | 58 |
| T | 2948 | 3672 |
| iDim | 1302 | **1347**(加了 4 列 aux 后)|
| WRF channels | 54(|Wind| merged)| 63(WindX/Y separate)|
| station_aux | 4 列保留(从 .mat 读) | 4 列保留(代码计算)|
| 时间偏移 | V1 t=0 = V2 unlabeled t=2 | 无(t=0 对齐)|
| 图(sup)| `|sim×dist|` 自带 AJM | corrcoef(SimilarityMat) × exp(-dist) |
| target | StationMat 预归一化 [0,1] | raw °C → 全局 [0,1],scl=34.57 |
| station_id 范围 | labeled 1..98(原值)| labeled 0..57 + unlabeled 100..N |

### 7.8 V2 时间特征生成

```python
from datetime import datetime, timedelta
base = datetime(2018, 5, 1, 0, 0)
for t in range(T):
    dt = base + timedelta(hours=t)
    aux[t, 0] = dt.hour / 23.0   # hour normalized
    aux[t, 1] = dt.month / 12.0  # month normalized
    aux[t, 2] = float(dt.year)   # year (raw 2018)
```

V2 t=0 → hour=0, V1 t=0 → hour=2(因为 V1 起 02:00),所以即使数值上 V1 station_aux@t=0 与 V2 station_aux@t=0 不同,**它们对应的物理时间也不同**(V1 t=0 = May 1 02:00 vs V2 t=0 = May 1 00:00)→ 各自内部一致。

## 6. Validation 切分(三种)

| `V1_VAL_MODE` | 切分逻辑 | 实现 |
|---|---|---|
| `random` | `torch.randperm(seed=19)` 切 75/25(同 original_code)| `random_split()` |
| `sequential` | 前 80% timestep train,后 20% valid | 直接索引切 |
| `spatial` | FPS(seed=42)选 10 个最分散的 valid station,index 固定 = `[6, 11, 16, 27, 31, 34, 37, 38, 61, 65]` | 用 `label_mask` 切 |

Spatial 模式下,所有 timestep 都 train + valid 都用,只是 `label_mask` 不同(train_mask 包 58 train station,valid_mask 包 10 valid station)。

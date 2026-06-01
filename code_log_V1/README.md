# code_log_V1/ —— code_V1 + scripts_V1 的 pipeline 文档

仅记录 6 个 baseline 的实现。运行时的实际输出去 [log_V1/](../log_V1/);拆分前的杂乱旧 log 在 [log_V0.md](../log_V0.md)。

| 文件 | 内容 |
|---|---|
| [data.md](data.md) | 数据来源、特征 schema、归一化、图、unlabeled 选择 |
| [methods.md](methods.md) | 模型 / loss / 训练 protocol / 6 baseline 配置差异 |
| [results.md](results.md) | 6 baseline 结果表(空表头,跑完填) |
| [roadmap.md](roadmap.md) | 后续要做的 6 类方法 placeholder(Pseudo-Label / Consistency / Graph Reg / Mask-Recon / Multi-task / Pretrain)|

## 目录约定

```
code_V1/                  # original_code/ 的工作副本(可改)
├── data.py               # env 控制 val_mode + n_unlabeled
├── network.py            # train/test 加 label_mask 支持
├── run.py                # env-driven entry,调 wandb,输出去 log_V1/{jobid}_{method}/
└── utils.py              # plotHist / RMSE,未改

scripts_V1/               # 6 个 SLURM 脚本
├── V1_sup_random.sh
├── V1_sup_sequential.sh
├── V1_sup_spatial.sh
├── V1_semi_random.sh
├── V1_semi_sequential.sh
└── V1_semi_spatial.sh

slurm_output_V1/          # SLURM stdout/stderr,文件名 {jobid}_{method}_result.txt
├── result/
└── error/

log_V1/                   # 每个 job 一个子目录:{jobid}_{method}/
                          # 内含 *_log / *_hist.png / *.pt / spatial_split.png 等
```

## 6 个 baseline

| 脚本 | env | iDim | 节点数 | 图 |
|---|---|---|---|---|
| V1_sup_random.sh      | `V1_VAL_MODE=random V1_N_UNLABELED=0`     | 1302 | 68  | V1 sim×dist |
| V1_sup_sequential.sh  | `V1_VAL_MODE=sequential V1_N_UNLABELED=0` | 1302 | 68  | V1 sim×dist |
| V1_sup_spatial.sh     | `V1_VAL_MODE=spatial V1_N_UNLABELED=0`    | 1302 | 68  | V1 sim×dist |
| V1_semi_random.sh     | `V1_VAL_MODE=random V1_N_UNLABELED=400`     | 1302 | 468 | k-NN k=10 |
| V1_semi_sequential.sh | `V1_VAL_MODE=sequential V1_N_UNLABELED=400` | 1302 | 468 | k-NN k=10 |
| V1_semi_spatial.sh    | `V1_VAL_MODE=spatial V1_N_UNLABELED=400`    | 1302 | 468 | k-NN k=10 |

## 提交

主 6 个 baseline:
```bash
for f in scripts_V1/V1_{sup,semi}_{random,sequential,spatial}.sh; do sbatch "$f"; done
```

额外对照实验(GAT/SAGE,n_unl 扫描,V2 dataset):
```bash
for f in scripts_V1/V1_sup_spatial_{gat,sageconv}.sh \
         scripts_V1/V1_semi_spatial_{200u,600u,800u,1000u}.sh \
         scripts_V1/V2_{sup,semi}_spatial.sh; do
    sbatch "$f"
done
```

## 额外的对照脚本(新增 8 个)

| 脚本 | 关键 env | 目的 |
|---|---|---|
| V1_sup_spatial_gat.sh | `V1_CONV_TYPE=gat` | sup spatial 用 GAT 4-head 替换 GraphConv |
| V1_sup_spatial_sageconv.sh | `V1_CONV_TYPE=sageconv` | sup spatial 用 SAGEConv 替换 |
| V1_semi_spatial_{200,600,800,1000}u.sh | `V1_N_UNLABELED=N` | semi spatial 不同 unlabeled 数量扫描 |
| V2_sup_spatial.sh | `V1_DATASET=V2 V1_N_UNLABELED=0` | V2 labeled (58 站) sup spatial |
| V2_semi_spatial.sh | `V1_DATASET=V2 V1_N_UNLABELED=400` | V2 labeled + V2 unlabeled semi |

详见 [methods.md](methods.md) §6-§7 + [data.md](data.md) §7。

## wandb

每个 job 自动登录 wandb:
- entity: `urban_prediction`
- project: `V1_baseline`
- run name: `{jobid}_{method}`(同 SLURM 输出文件名)

可以通过 `V1_WANDB=0` 关闭。

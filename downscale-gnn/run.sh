#!/bin/bash
#SBATCH -A p32685        ## 你的账户
#SBATCH -p gengpu        ## GPU 分区
#SBATCH --gres=gpu:a100:1 ## 请求一块 A100 GPU
#SBATCH -N 1             ## 请求一个节点
#SBATCH -n 1             ## 请求一个任务
#SBATCH -t 48:00:00      ## 最大运行时间 48 小时
#SBATCH --mem=80G      
#SBATCH --job-name=supervised
#SBATCH -o supervised_%j_%x.out
#SBATCH -e supervised_%j_%x.err

# 加载 Conda 环境
source /home/hhz6461/anaconda3/etc/profile.d/conda.sh
conda activate fixmatch             # 激活你的 Conda 环境

# 运行 Python 脚本
python run.py

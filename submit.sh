#!/bin/bash

#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --gpus=4
#SBATCH --no-requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=chenhang20@mails.tsinghua.edu.cn

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

module load cuda/10.0

srun python train.py -c $1 --gpus 4 --nodes 1 2>&1 | tee /data/home/scv1134/run/chenhang/nichang-avdomain/egs/frcnn2_para/$2.log
# srun python eval.py --test=local/data/tt --conf_dir=/data/home/scv1134/run/chenhang/nichang-avdomain/egs/frcnn2_para/exp/frcnn2_64_64_3_adamw_1e-1_blocks8_pretrain/conf.yml

# usage: 
#   sbatch submit.sh local/lrs2_conf_128_128_3_adamw_1e-1_blocks16.yml
#   sbatch --qos=gpugpu submit.sh local/lrs2_conf_128_128_3_adamw_1e-1_blocks16.yml
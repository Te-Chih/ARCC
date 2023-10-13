#!/bin/bash
echo Running on ACT-GPU
echo Directory is $PWD
echo ${SLURM_JOB_NODELIST}
# 注意这里换成自己的运行环境
source anaconda3/etc/profile.d/conda.sh
conda activate ND_paper
echo start on $(date)

cfg_path='and/config/Aminer-18/SCL/cfg.yml'
filename='main.py'
# 注意需要改成你自己的路径
CUDA_VISIBLE_DEVICES=${gpu} nohup python and/main/${filename} \
--run_model='run' \
--cfg_path=${cfg_path} \
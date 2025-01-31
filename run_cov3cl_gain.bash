#!/bin/bash
#SBATCH --job-name="MARCEL-benchmark"
#SBATCH --output="result.%j.%N.out"
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bccg-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 48:00:00

cd /data1/dannie/projects/Auto3D

PORT=$((10000 + (RANDOM % 10001)))


python train_4d_moe_cov.py --seed 8888 --dataset CoV-3CL --target sars_cov_two_cl_protease_active \
  --modeldss:conf_encoder Equiformer --modeldss:topo_encoder GIN --batch_size 16 \
  --port $PORT --optimizer AdamW --scheduler LambdaLR --learning_rate 0.001 \
  --num_epochs 100 --weight_decay 0.001 --gpus 0 --patience 400 --gig True --sag True --upc True\
  --num_experts 4 --num_activated 2 --upcycling_epochs 5 --gumbel_tau 1 --z_beta 0.00001 \
  --wandb_project PaiNN-MoE-CoV-3CL



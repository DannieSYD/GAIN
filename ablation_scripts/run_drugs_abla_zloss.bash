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

#python train_4d_moe.py --seed 8888 --dataset Drugs --target ip --num_epochs 1000\
# --modeldss:conf_encoder PaiNN --modeldss:topo_encoder GIN --batch_size 32 \
# --port $PORT --optimizer AdamW --scheduler LambdaLR --learning_rate 0.0002 \
# --weight_decay 0.001 --gpus 0 --patience 400 --gig True --sag True --upc True\
# --num_experts 8 --num_activated 2 --upcycling_epochs 100 --gumbel_tau 0.1 --z_beta 0.001 \
# --additional_notes sampled_drugs_train_4d_moe_gig_vec_norm \
# --wandb_project PaiNN-MoE-Drugs

python train_4d_moe.py --seed 8888 --dataset Drugs --target ea --num_epochs 1000\
 --modeldss:conf_encoder PaiNN --modeldss:topo_encoder GIN --batch_size 32 \
 --port $PORT --optimizer AdamW --scheduler LambdaLR --learning_rate 0.0002 \
 --weight_decay 0.001 --gpus 0 --patience 400 --gig True --sag True --upc True\
 --num_experts 8 --num_activated 2 --upcycling_epochs 50 --gumbel_tau 0.1 --z_beta 1e-6 \
 --additional_notes sampled_drugs_train_4d_moe_gig_vec_norm \
 --wandb_project MoE-Drugs-Ablation

python train_4d_moe.py --seed 8888 --dataset Drugs --target ea --num_epochs 1000\
 --modeldss:conf_encoder PaiNN --modeldss:topo_encoder GIN --batch_size 32 \
 --port $PORT --optimizer AdamW --scheduler LambdaLR --learning_rate 0.0002 \
 --weight_decay 0.001 --gpus 0 --patience 400 --gig True --sag True --upc True\
 --num_experts 8 --num_activated 2 --upcycling_epochs 50 --gumbel_tau 0.1 --z_beta 1e-5 \
 --additional_notes sampled_drugs_train_4d_moe_gig_vec_norm \
 --wandb_project MoE-Drugs-Ablation

python train_4d_moe.py --seed 8888 --dataset Drugs --target ea --num_epochs 1000\
 --modeldss:conf_encoder PaiNN --modeldss:topo_encoder GIN --batch_size 32 \
 --port $PORT --optimizer AdamW --scheduler LambdaLR --learning_rate 0.0002 \
 --weight_decay 0.001 --gpus 0 --patience 400 --gig True --sag True --upc True\
 --num_experts 8 --num_activated 2 --upcycling_epochs 50 --gumbel_tau 0.1 --z_beta 1e-4 \
 --additional_notes sampled_drugs_train_4d_moe_gig_vec_norm \
 --wandb_project MoE-Drugs-Ablation

python train_4d_moe.py --seed 8888 --dataset Drugs --target ea --num_epochs 1000\
 --modeldss:conf_encoder PaiNN --modeldss:topo_encoder GIN --batch_size 32 \
 --port $PORT --optimizer AdamW --scheduler LambdaLR --learning_rate 0.0002 \
 --weight_decay 0.001 --gpus 0 --patience 400 --gig True --sag True --upc True\
 --num_experts 8 --num_activated 2 --upcycling_epochs 50 --gumbel_tau 0.1 --z_beta 1e-3 \
 --additional_notes sampled_drugs_train_4d_moe_gig_vec_norm \
 --wandb_project MoE-Drugs-Ablation


#!/bin/bash


PORT=$((10000 + (RANDOM % 10001)))


python train_4d_moe_cov.py --seed 8888 --dataset CoV2 --target sars_cov_two_active \
  --modeldss:conf_encoder PaiNN --modeldss:topo_encoder GIN --batch_size 12 \
  --port $PORT --optimizer AdamW --scheduler LambdaLR --learning_rate 0.0002 \
  --num_epochs 12000 --weight_decay 0.001 --gpus 0 --patience 800 --gig True --sag True --upc True\
  --num_experts 8 --num_activated 2 --upcycling_epochs 50 --gumbel_tau 1 --z_beta 0.001 \
  --wandb_project MoE-CoV2


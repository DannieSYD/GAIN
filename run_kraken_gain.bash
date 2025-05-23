#!/bin/bash


PORT=$((10000 + (RANDOM % 10001)))


python train_4d_moe.py --seed 8889 --dataset Kraken --target sterimol_burL --num_epochs 2000\
  --modeldss:conf_encoder ClofNet --modeldss:topo_encoder GIN --batch_size 16 \
  --port $PORT --optimizer AdamW --scheduler LambdaLR --learning_rate 0.0002 \
  --weight_decay 0.001 --gpus 0 --patience 400 --gig True --sag True --upc True\
  --num_experts 8 --num_activated 2 --upcycling_epochs 100 --gumbel_tau 1.5 --z_beta 0.001 \
  --additional_notes sampled_drugs_train_4d_moe_gig_vec_norm \
  --wandb_project ClofNet-MoE-Kraken

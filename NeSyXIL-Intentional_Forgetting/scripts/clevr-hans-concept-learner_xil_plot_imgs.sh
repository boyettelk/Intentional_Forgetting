#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="concept-learner-xil-$NUM"
DATASET=clevr-hans-state

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

CUDA_VISIBLE_DEVICES=$DEVICE python train_IF_concept_learner_xil.py --data-dir $DATA --dataset $DATASET \
--epochs 30 --name $MODEL --lr 0.001 --l2_grads 1000 --batch-size 64 --num-workers 1 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 \
--seed 0 --mode plot \
--fp-ckpt runs/IF_latest/concept-learner-xil-0-IF_latest_seed0/model_epoch8_bestvalloss_2.1815.pth
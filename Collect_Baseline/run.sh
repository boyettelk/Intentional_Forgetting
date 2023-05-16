#!/bin/bash

#training params
MODELS=('Resnet' 'CNN')
EPOCHS=50
BATCHSIZE=64

#train each model, generate accuracy/loss plots
for model in "${MODELS[@]}"
do
    python train_files/train.py --epochs $EPOCHS --batchsize $BATCHSIZE --model $model
    python train_files/task_analysis.py --epochs $EPOCHS --batchsize $BATCHSIZE --model $model
done
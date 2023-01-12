#!/bin/bash

python -u train_net.py  \
    --GPU=5 \
    --DATASET=JGPMulti \
    --RUN_NAME="JGP-Dynamic-ResNet18-32-64-64-128-StepLR-NewJGP" \
    --DATASET_PATH="./datasets/JGP/multi-class"\
    --RESULTS_PATH="./results/JGP_multi-class"\
    --SAVE_IMAGES=True \
    --DILATE=0 \
    --EPOCHS=50 \
    --LEARNING_RATE=0.001 \
    --DELTA_CLS_LOSS=1 \
    --BATCH_SIZE=32 \
    --WEIGHTED_SEG_LOSS=True \
    --WEIGHTED_SEG_LOSS_P=2 \
    --WEIGHTED_SEG_LOSS_MAX=1 \
    --DYN_BALANCED_LOSS=False \
    --GRADIENT_ADJUSTMENT=True \
    --FREQUENCY_SAMPLING=False \
    --NUM_SEGMENTED=1143 \
    --ON_DEMAND_READ True \
    --OUTPUT_CLASS 9 \
    --WEIGHT_DECAY 0.001 \
    --MOMENTUM 0.9
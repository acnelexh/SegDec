#!/bin/bash

python -u train_net.py  \
    --GPU=0 \
    --DATASET=JGP \
    --RUN_NAME="check" \
    --DATASET_PATH="./datasets/JGP"\
    --RESULTS_PATH="./results/JGP"\
    --SAVE_IMAGES=True \
    --DILATE=0 \
    --EPOCHS=50 \
    --LEARNING_RATE=0.001 \
    --DELTA_CLS_LOSS=0.01 \
    --BATCH_SIZE=32 \
    --WEIGHTED_SEG_LOSS=True \
    --WEIGHTED_SEG_LOSS_P=2 \
    --WEIGHTED_SEG_LOSS_MAX=1 \
    --DYN_BALANCED_LOSS=True \
    --GRADIENT_ADJUSTMENT=True \
    --FREQUENCY_SAMPLING=True \
    --NUM_SEGMENTED=939 \
    --ON_DEMAND_READ True 
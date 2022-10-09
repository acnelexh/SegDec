#!/bin/bash

python -u train_net.py  \
    --GPU=0 \
    --DATASET=JGPMulti \
    --RUN_NAME="multiclass" \
    --DATASET_PATH="./datasets/JGP/multi-class"\
    --RESULTS_PATH="./results/JGP_multi-class"\
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
    --FREQUENCY_SAMPLING=False \
    --NUM_SEGMENTED=939 \
    --ON_DEMAND_READ True \
    --OUTPUT_CLASS 9
#!/bin/bash
# scripts/train_controlnet.sh
# Train MultiDiffSense ControlNet model
set -e

CONFIG="configs/controlnet_train.yaml"
BATCH_SIZE=8
LR=1e-5
MAX_EPOCHS=350

echo "=== MultiDiffSense ControlNet Training ==="
echo "Config: $CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Max epochs: $MAX_EPOCHS"
echo ""

python multidiffsense/controlnet/train.py \
    --config "$CONFIG" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_epochs $MAX_EPOCHS \
    --sd_locked

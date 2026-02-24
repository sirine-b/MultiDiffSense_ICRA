#!/bin/bash
# scripts/train_pix2pix.sh
# Train Pix2Pix baseline for a given modality
set -e

MODALITY=${1:-TacTip}
PGAN_DIR="external/pytorch-CycleGAN-and-pix2pix"
DATAROOT="$PGAN_DIR/datasets/depth_to_sensor_${MODALITY}"

if [ ! -d "$PGAN_DIR" ]; then
    echo "ERROR: pytorch-CycleGAN-and-pix2pix not found!"
    echo "Clone it: git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix $PGAN_DIR"
    exit 1
fi

echo "=== Pix2Pix Baseline Training ==="
echo "Modality: $MODALITY"
echo "Dataroot: $DATAROOT"
echo ""

cd "$PGAN_DIR"
python train.py \
    --dataroot "datasets/depth_to_sensor_${MODALITY}" \
    --name "depth_to_sensor_${MODALITY}" \
    --model pix2pix \
    --direction AtoB \
    --n_epochs 200 \
    --n_epochs_decay 100 \
    --batch_size 8 \
    --gpu_ids 0

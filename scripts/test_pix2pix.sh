#!/bin/bash
# scripts/test_pix2pix.sh
# Test Pix2Pix baseline
set -e

MODALITY=${1:-TacTip}
PGAN_DIR="external/pytorch-CycleGAN-and-pix2pix"

echo "=== Pix2Pix Baseline Testing ==="
echo "Modality: $MODALITY"

cd "$PGAN_DIR"
python ../../multidiffsense/baseline_cgan/test.py \
    --dataroot "datasets/depth_to_sensor_${MODALITY}" \
    --name "depth_to_sensor_${MODALITY}" \
    --model pix2pix \
    --direction AtoB \
    --phase test \
    --gpu_ids 0

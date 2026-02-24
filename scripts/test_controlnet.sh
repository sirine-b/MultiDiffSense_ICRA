#!/bin/bash
# scripts/test_controlnet.sh
# Test MultiDiffSense ControlNet across all modalities
set -e

CONFIG="configs/controlnet_train.yaml"
CHECKPOINT=${1:-"results/lightning_logs/version_0/checkpoints/best.ckpt"}

echo "=== MultiDiffSense ControlNet Testing ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

for MODALITY in TacTip ViTac ViTacTip; do
    echo "--- Testing modality: $MODALITY ---"

    # Seen objects
    python multidiffsense/controlnet/test.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --modality "$MODALITY" \
        --seen_objects \
        --output_dir "results/test_seen_${MODALITY}"

    # Unseen objects
    python multidiffsense/controlnet/test.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --modality "$MODALITY" \
        --output_dir "results/test_unseen_${MODALITY}"
done

echo ""
echo "=== Ablation: No Prompt ==="
python multidiffsense/controlnet/test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --modality ViTacTip \
    --no_prompt \
    --output_dir "results/ablation_no_prompt"

echo "=== Ablation: No Source ==="
python multidiffsense/controlnet/test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --modality ViTacTip \
    --no_source \
    --output_dir "results/ablation_no_source"

echo "=== All tests complete ==="

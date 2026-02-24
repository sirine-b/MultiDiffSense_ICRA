#!/bin/bash
# scripts/prepare_model.sh
# Download Stable Diffusion v1.5 and create ControlNet initialisation checkpoint
set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

# Step 1: Download SD1.5 checkpoint (if not already present)
SD_CKPT="$MODELS_DIR/v1-5-pruned.ckpt"
if [ ! -f "$SD_CKPT" ]; then
    echo "Downloading Stable Diffusion v1.5 checkpoint..."
    echo "Please download v1-5-pruned.ckpt from https://huggingface.co/runwayml/stable-diffusion-v1-5"
    echo "and place it in $MODELS_DIR/"
    echo ""
    echo "Or use:"
    echo "curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt -o models/v1-5-pruned.ckpt"
    exit 1
else
    echo "SD1.5 checkpoint found: $SD_CKPT"
fi

# Step 2: Copy the ControlNet architecture config
if [ ! -f "$MODELS_DIR/cldm_v15.yaml" ]; then
    cp configs/cldm_v15.yaml "$MODELS_DIR/cldm_v15.yaml"
    echo "Copied cldm_v15.yaml to $MODELS_DIR/"
fi

# Step 3: Create ControlNet initialisation weights
CTRL_CKPT="$MODELS_DIR/control_sd15_ini.ckpt"
if [ ! -f "$CTRL_CKPT" ]; then
    echo "Creating ControlNet initialisation weights..."
    python tool_add_control.py "$SD_CKPT" "$CTRL_CKPT"
    echo "Created: $CTRL_CKPT"
else
    echo "ControlNet init checkpoint already exists: $CTRL_CKPT"
fi

echo ""
echo "Model preparation complete!"
echo "  SD1.5:      $SD_CKPT"
echo "  ControlNet: $CTRL_CKPT"

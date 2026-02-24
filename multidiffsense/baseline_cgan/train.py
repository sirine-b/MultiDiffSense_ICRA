"""
MultiDiffSense — cGAN (Pix2Pix) Baseline Training Wrapper

This is a thin wrapper that calls the pytorch-CycleGAN-and-pix2pix training script.
The framework must be cloned into external/pytorch-CycleGAN-and-pix2pix/.

Usage:
    python multidiffsense/baseline_cgan/train.py \
        --dataroot datasets/depth_to_sensor_TacTip \
        --name depth_to_sensor_experiment \
        --n_epochs 200 \
        --n_epochs_decay 100

    Or use the shell script:
        bash scripts/train_pix2pix.sh
"""

import os
import subprocess
import sys


def main():
    pix2pix_dir = os.path.join(os.path.dirname(__file__), "../../external/pytorch-CycleGAN-and-pix2pix")

    if not os.path.isdir(pix2pix_dir):
        print("ERROR: pytorch-CycleGAN-and-pix2pix not found!")
        print("Please clone it first:")
        print("  git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix external/pytorch-CycleGAN-and-pix2pix")
        sys.exit(1)

    # Forward all CLI arguments to the Pix2Pix training script
    cmd = [
        sys.executable,
        os.path.join(pix2pix_dir, "train.py"),
        "--model", "pix2pix",
        "--direction", "AtoB",
        "--dataset_mode", "aligned",
    ] + sys.argv[1:]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=pix2pix_dir, check=True)


if __name__ == "__main__":
    main()

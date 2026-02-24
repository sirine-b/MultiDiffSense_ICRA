"""
MultiDiffSense — Dataset Converter (ControlNet → Pix2Pix format)

Converts the ControlNet-format dataset (JSONL prompts + image directories) into
the Pix2Pix format (trainA/trainB, valA/valB, testA/testB paired directories).

Usage:
    python multidiffsense/baseline_cgan/dataset_converter.py \
        --controlnet_dataset data/processed \
        --output_path external/pytorch-CycleGAN-and-pix2pix/datasets/depth_to_sensor \
        --modality TacTip
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ControlNet → Pix2Pix dataset")
    parser.add_argument("--controlnet_dataset", type=str, required=True,
                        help="Path to processed ControlNet dataset")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for Pix2Pix dataset")
    parser.add_argument("--modality", type=str, default="TacTip",
                        choices=["TacTip", "ViTac", "ViTacTip"])
    return parser.parse_args()


def convert_split(controlnet_dir, output_dir, split, modality):
    """Convert a single split (train/val/test) from ControlNet to Pix2Pix format."""
    prompt_path = os.path.join(controlnet_dir, split, f"prompt_{modality}.json")
    if not os.path.exists(prompt_path):
        # Fall back to combined prompt file
        prompt_path = os.path.join(controlnet_dir, split, "prompt.json")
    if not os.path.exists(prompt_path):
        print(f"  Warning: {prompt_path} not found. Skipping {split}.")
        return 0

    # Read and filter entries for the target modality
    source_paths, target_paths = [], []
    with open(prompt_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            target = entry.get("target", "")
            if modality in target or modality.lower() in target.lower():
                source_paths.append(entry["source"])
                target_paths.append(entry["target"])

    if not source_paths:
        print(f"  No {modality} entries found in {split}.")
        return 0

    # Sort by filename for reproducibility
    pairs = sorted(zip(source_paths, target_paths),
                   key=lambda x: os.path.basename(x[0]))

    # Create output directories
    out_A = os.path.join(output_dir, f"{split}A")  # Depth maps
    out_B = os.path.join(output_dir, f"{split}B")  # Tactile images
    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)

    for i, (src, tgt) in enumerate(pairs):
        src_full = os.path.join(controlnet_dir, src)
        tgt_full = os.path.join(controlnet_dir, tgt)
        if os.path.exists(src_full) and os.path.exists(tgt_full):
            shutil.copy2(src_full, os.path.join(out_A, f"{i:04d}.png"))
            shutil.copy2(tgt_full, os.path.join(out_B, f"{i:04d}.png"))

    return len(pairs)


def main():
    args = parse_args()
    print(f"Converting {args.modality} dataset: {args.controlnet_dataset} → {args.output_path}")

    for split in ["train", "val", "test"]:
        n = convert_split(args.controlnet_dataset, args.output_path, split, args.modality)
        print(f"  {split}: {n} image pairs")

    print(f"\nDone. Pix2Pix dataset at {args.output_path}")


if __name__ == "__main__":
    main()

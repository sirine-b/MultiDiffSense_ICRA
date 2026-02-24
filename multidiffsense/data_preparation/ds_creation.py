"""
MultiDiffSense — Dataset Creation (Mega Dataset Assembly)

Combines per-object, per-sensor datasets into a single unified dataset
with merged source/, target/ directories and a combined prompt.json.

Input structure:
    <tactile_dir>/<obj_id>/<sensor_type>/source/
    <tactile_dir>/<obj_id>/<sensor_type>/target/
    <tactile_dir>/<obj_id>/<sensor_type>/prompt.json

Output structure:
    <output_dir>/source/
    <output_dir>/target/
    <output_dir>/prompt.json

Usage:
    python -m multidiffsense.data_preparation.ds_creation \
        --tactile_dir data/example/tactile \
        --output_dir datasets/training \
        --object_ids 1 3 6 8 18 \
        --sensors TacTip ViTac ViTacTip
"""

import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_dataset_dirs(dir_name):
    """Create the mega dataset directory structure."""
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, "source"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "target"), exist_ok=True)


def viridis_map(image):
    """Apply viridis colourmap to a grayscale image."""
    if len(image.shape) != 2:
        raise ValueError("Expected a 2D grayscale image for colormapping.")
    normed = (image - np.min(image)) / (np.ptp(image) + 1e-8)
    colormapped = plt.cm.viridis(normed)
    return (colormapped[:, :, :3] * 255).astype("uint8")


def mega_dataset_creation(tactile_dir, output_dir, object_ids, sensors,
                          prompt_style="short"):
    """Combine per-object datasets into one mega dataset.

    Args:
        tactile_dir:  Root tactile dir: <tactile_dir>/<obj_id>/<sensor>/
        output_dir:   Output directory for the combined dataset
        object_ids:   List of object ID strings to include
        sensors:      List of sensor type strings
        prompt_style: "short" or "long" -- selects which prompt file to merge
    """
    create_dataset_dirs(output_dir)
    all_prompts = []
    prompt_filename = "prompt_long.json" if prompt_style == "long" else "prompt.json"

    for obj_id in object_ids:
        for sensor in sensors:
            dataset_path = os.path.join(tactile_dir, str(obj_id), sensor)
            if not os.path.isdir(dataset_path):
                print(f"Skipping {obj_id}/{sensor} (not found)")
                continue

            print(f"Processing {obj_id}/{sensor}...")

            # Copy source and target images
            for subfolder in ["source", "target"]:
                sub_path = os.path.join(dataset_path, subfolder)
                if not os.path.isdir(sub_path):
                    continue
                for image_name in os.listdir(sub_path):
                    if not image_name.endswith(".png"):
                        continue
                    img = plt.imread(os.path.join(sub_path, image_name))
                    if img.dtype != "uint8":
                        img = (img * 255).astype("uint8")
                    out_path = os.path.join(output_dir, subfolder, image_name)
                    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                print(f"  Copied {subfolder}/ images from {obj_id}/{sensor}")

            # Collect prompt entries
            prompt_path = os.path.join(dataset_path, prompt_filename)
            if os.path.exists(prompt_path):
                with open(prompt_path, "r") as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            all_prompts.append(json.loads(line))

    # Write merged prompt file
    merged_path = os.path.join(output_dir, prompt_filename)
    with open(merged_path, "w") as fout:
        for entry in all_prompts:
            fout.write(json.dumps(entry) + "\n")

    print(f"\nMerged dataset: {len(all_prompts)} samples -> {output_dir}")
    print(f"  {prompt_filename}: {merged_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create mega dataset from per-object datasets")
    parser.add_argument("--tactile_dir", type=str, required=True,
                        help="Root tactile dir: <tactile_dir>/<obj_id>/<sensor>/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged dataset")
    parser.add_argument("--object_ids", nargs="+", required=True,
                        help="Object IDs to include (e.g., 1 3 6 8 18)")
    parser.add_argument("--sensors", nargs="+",
                        default=["TacTip", "ViTac", "ViTacTip"],
                        help="Sensor types to include")
    parser.add_argument("--prompt_style", type=str, default="short",
                        choices=["short", "long"],
                        help="Which prompt file to merge (short=prompt.json, long=prompt_long.json)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mega_dataset_creation(args.tactile_dir, args.output_dir,
                          args.object_ids, args.sensors,
                          prompt_style=args.prompt_style)
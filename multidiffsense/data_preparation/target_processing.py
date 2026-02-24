"""
MultiDiffSense — Target (Tactile Image) Processing

Renames and resizes raw tactile sensor images to a consistent format:
  - Renames: frame_N_0.png → <obj_id>_<sensor_type>_<N>.png
  - Resizes: all images to 512×512

Directory structure expected:
    <tactile_dir>/<obj_id>/<sensor_type>/target/   — raw tactile images

Usage:
    python -m multidiffsense.data_preparation.target_processing \
        --tactile_dir data/example/tactile \
        --obj_id 1 \
        --sensor_type ViTac
"""

import argparse
import os

import cv2


def rename_target_images(target_dir, sensor_type, obj_id):
    """Rename target images to a consistent <obj_id>_<sensor>_<frame>.png format."""
    renamed = 0
    for filename in sorted(os.listdir(target_dir)):
        if not filename.endswith(".png"):
            continue
        if filename == "init_0.png":
            continue  # Skip reference/init image
        # Skip files already renamed (contain sensor_type in name)
        if sensor_type in filename:
            continue

        base = filename.replace(".png", "")
        parts = base.split("_")
        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, f"{obj_id}_{sensor_type}_{parts[1]}.png")
        print(f"  Renaming {filename} → {os.path.basename(new_path)}")
        os.rename(old_path, new_path)
        renamed += 1
    if renamed == 0:
        print("  No files to rename (already renamed or empty).")


def resize_target_images(target_dir, size=(512, 512)):
    """Resize all PNG images in target_dir to the given size."""
    for filename in sorted(os.listdir(target_dir)):
        if not filename.endswith(".png"):
            continue
        img_path = os.path.join(target_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resized = cv2.resize(img, size)
            cv2.imwrite(img_path, resized)
        else:
            print(f"  WARNING: Could not read {img_path}")


def get_frame_indices(target_dir, obj_id, sensor_type):
    """Discover available frame indices from actual files in the target directory.

    Returns:
        Sorted list of integer frame indices found in the directory.
    """
    indices = []
    prefix = f"{obj_id}_{sensor_type}_"
    for filename in os.listdir(target_dir):
        if not filename.endswith(".png"):
            continue
        if filename == "init_0.png":
            continue
        base = filename.replace(".png", "")
        if base.startswith(prefix):
            try:
                idx = int(base[len(prefix):])
                indices.append(idx)
            except ValueError:
                continue
    return sorted(indices)


def target_processing(tactile_dir, obj_id, sensor_type):
    """Full target processing: rename + resize.

    Args:
        tactile_dir:  Root tactile dir: <tactile_dir>/<obj_id>/<sensor_type>/target/
        obj_id:       Object ID (string)
        sensor_type:  One of TacTip, ViTac, ViTacTip
    """
    target_dir = os.path.join(tactile_dir, str(obj_id), sensor_type, "target")
    print(f"Processing targets in {target_dir}")

    rename_target_images(target_dir, sensor_type, obj_id)
    print("  Renaming complete.")

    resize_target_images(target_dir)
    print("  Resizing complete.")

    # Report discovered frames
    indices = get_frame_indices(target_dir, obj_id, sensor_type)
    print(f"  Found {len(indices)} target frames.")
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description="Process tactile target images")
    parser.add_argument("--tactile_dir", type=str, required=True,
                        help="Root tactile dir: <tactile_dir>/<obj_id>/<sensor>/target/")
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--sensor_type", type=str, default="ViTac",
                        choices=["TacTip", "ViTac", "ViTacTip"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_processing(args.tactile_dir, args.obj_id, args.sensor_type)

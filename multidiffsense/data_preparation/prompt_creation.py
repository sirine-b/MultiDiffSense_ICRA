"""
MultiDiffSense — Prompt Creation

Generates prompt.json (JSONL) files from CSV pose data.
Each line contains source/target paths and a structured text prompt
encoding sensor context and contact pose.

Iterates over actual target images in the directory (not CSV row count)
to ensure prompt entries match available data.

Output written to:
    <tactile_dir>/<obj_id>/<sensor_type>/prompt.json

Usage:
    python -m multidiffsense.data_preparation.prompt_creation \
        --csv_dir data/example/csv \
        --tactile_dir data/example/tactile \
        --obj_id 1 \
        --sensor_type ViTac
"""

import argparse
import json
import os

import pandas as pd

from multidiffsense.data_preparation.source_processing import object_names
from multidiffsense.data_preparation.target_processing import get_frame_indices


def create_pose_dict(csv_path):
    """Read CSV and extract per-frame pose values (x, y, z, yaw).

    Returns:
        dict mapping frame index → (x, y, z, yaw) rounded to 2 d.p.
    """
    df = pd.read_csv(csv_path)
    pose_dict = {}
    for i in range(len(df["pose_3"])):
        pose_dict[i] = (
            round(df["pose_1"][i], 2),
            round(df["pose_2"][i], 2),
            round(df["pose_3"][i], 2),
            round(df["pose_6"][i], 2),
        )
    return pose_dict


def create_prompt_json(pose_dict, frame_indices, obj_id, sensor_type, output_dir):
    """Create a JSONL prompt file for one object + sensor combination.

    Only creates entries for frames that actually exist in the target directory.

    Each line contains:
      - source: relative path to depth map
      - target: relative path to tactile image
      - prompt: dict with sensor_context and object_pose
    """
    # Sensor-specific context text
    sensor_context = {
        "ViTacTip": f"captured by a high-resolution visuo-tactile sensor {sensor_type}.",
        "ViTac": f"captured by a high-resolution vision only sensor {sensor_type}.",
        "TacTip": f"captured by a high-resolution tactile only sensor {sensor_type}.",
    }
    sensor_text = sensor_context.get(sensor_type, f"captured by {sensor_type} sensor.")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "prompt.json")
    count = 0

    with open(out_path, "w") as f:
        for idx in frame_indices:
            if idx not in pose_dict:
                print(f"  WARNING: Frame {idx} not in CSV, skipping prompt entry.")
                continue
            x, y, z, yaw = pose_dict[idx]
            entry = {
                "source": f"source/{obj_id}_{idx}.png",
                "target": f"target/{obj_id}_{sensor_type}_{idx}.png",
                "prompt": {
                    "sensor_context": sensor_text,
                    "object_pose": {"x": x, "y": y, "z": z, "yaw": yaw},
                },
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"  Prompt JSON: {count} entries → {out_path}")
    return out_path


def prompt_processing(csv_dir, tactile_dir, obj_id, sensor_type):
    """End-to-end prompt creation for one object + sensor.

    Args:
        csv_dir:      Directory containing <obj_id>.csv files
        tactile_dir:  Root tactile dir: <tactile_dir>/<obj_id>/<sensor_type>/
        obj_id:       Object ID (string)
        sensor_type:  One of TacTip, ViTac, ViTacTip
    """
    csv_path = os.path.join(csv_dir, f"{obj_id}.csv")
    target_dir = os.path.join(tactile_dir, str(obj_id), sensor_type, "target")
    output_dir = os.path.join(tactile_dir, str(obj_id), sensor_type)

    # Discover actual frames from target directory
    frame_indices = get_frame_indices(target_dir, obj_id, sensor_type)
    if not frame_indices:
        print(f"  WARNING: No target images found in {target_dir}")
        return

    pose_dict = create_pose_dict(csv_path)
    print(f"  Pose dict: {len(pose_dict)} rows from CSV, {len(frame_indices)} target frames")
    create_prompt_json(pose_dict, frame_indices, obj_id, sensor_type, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Create prompt JSONL from CSV poses")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing <obj_id>.csv files")
    parser.add_argument("--tactile_dir", type=str, required=True,
                        help="Root tactile dir: <tactile_dir>/<obj_id>/<sensor>/")
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--sensor_type", type=str, default="ViTac",
                        choices=["TacTip", "ViTac", "ViTacTip"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompt_processing(args.csv_dir, args.tactile_dir, args.obj_id, args.sensor_type)

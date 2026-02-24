"""
MultiDiffSense -- Prompt Creation

Generates prompt.json (JSONL) files from CSV pose data.
Supports two prompt styles for ablation studies:

  SHORT (default): sensor context + object pose
    "captured by a high-resolution vision only sensor ViTac.
     Contact pose: x=0.12, y=-0.34, z=1.50, yaw=15.00"

  LONG: object description + contact description + sensor context +
        style tags + negatives + object pose
    "A edge-shaped object with distinct geometric features.
     Medium contact on the object surface with moderate indentation.
     Captured by a high-resolution vision only sensor ViTac.
     High quality, detailed texture, realistic tactile response,
     sharp sensor reading.
     Contact pose: x=0.12, y=-0.34, z=1.50, yaw=15.00"

Usage:
    # Short prompts (default)
    python -m multidiffsense.data_preparation.prompt_creation \
        --csv_dir data/example/csv \
        --tactile_dir data/example/tactile \
        --obj_id 1 --sensor_type ViTac

    # Long prompts (for ablation 2)
    python -m multidiffsense.data_preparation.prompt_creation \
        --csv_dir data/example/csv \
        --tactile_dir data/example/tactile \
        --obj_id 1 --sensor_type ViTac \
        --prompt_style long
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from multidiffsense.data_preparation.source_processing import object_names
from multidiffsense.data_preparation.target_processing import get_frame_indices


# ── Object shape descriptions (for long prompts) ─────────────────────

object_descriptions = {
    1: "A edge-shaped object with distinct geometric features",
    2: "A flat slab with a smooth planar surface",
    3: "A pacman-shaped object with a characteristic notch",
    4: "A dot-shaped object with a small raised point",
    5: "A cylinder with a smooth circular cross-section",
    6: "A hollow cylinder with an open central cavity",
    7: "A ring-shaped object with a circular opening",
    8: "A sphere with a smooth curved surface",
    9: "A moon-shaped object with a curved crescent profile",
    10: "A dot-in object with a recessed indentation",
    11: "A curved surface with smooth continuous curvature",
    12: "A wave-shaped object with undulating surface features",
    13: "A dots pattern with multiple small raised points",
    14: "A cross lines pattern with intersecting linear features",
    15: "A parallel lines pattern with evenly spaced ridges",
    16: "A cone with a tapering pointed geometry",
    17: "A cylinder viewed from the side with a curved profile",
    18: "A cuboid with flat faces and sharp edges",
    19: "A hexagonal object with six-sided symmetry",
    20: "A triangular object with three vertices and edges",
    21: "A random-shaped object with irregular surface geometry",
}


def get_contact_description(z_mm):
    """Generate a contact description based on indentation depth."""
    z_abs = abs(z_mm)
    if z_abs < 0.5:
        return "Light contact on the object surface with minimal indentation"
    elif z_abs < 1.5:
        return "Medium contact on the object surface with moderate indentation"
    else:
        return "Deep contact on the object surface with significant indentation"


def get_sensor_context(sensor_type):
    """Sensor-specific context string."""
    contexts = {
        "ViTacTip": f"Captured by a high-resolution visuo-tactile sensor {sensor_type}",
        "ViTac": f"Captured by a high-resolution vision only sensor {sensor_type}",
        "TacTip": f"Captured by a high-resolution tactile only sensor {sensor_type}",
    }
    return contexts.get(sensor_type, f"Captured by {sensor_type} sensor")


STYLE_TAGS = ("High quality, detailed texture, realistic tactile response, "
              "sharp sensor reading")
NEGATIVES = "Blurry, low quality, artifacts, noise, distortion"


# ── Prompt builders ───────────────────────────────────────────────────

def build_short_prompt(obj_id, sensor_type, x, y, z, yaw):
    """Short prompt: sensor context + pose dict (default / ablation baseline)."""
    sensor_text = get_sensor_context(sensor_type) + "."
    return {
        "sensor_context": sensor_text,
        "object_pose": {"x": x, "y": y, "z": z, "yaw": yaw},
    }


def build_long_prompt(obj_id, sensor_type, x, y, z, yaw):
    """Long prompt: full description with all fields (ablation 2)."""
    obj_desc = object_descriptions.get(
        int(obj_id), f"An object (id={obj_id}) with surface features")
    contact_desc = get_contact_description(z)
    sensor_text = get_sensor_context(sensor_type)

    return {
        "object_description": obj_desc,
        "contact_description": contact_desc,
        "sensor_context": sensor_text,
        "style_tags": STYLE_TAGS,
        "negatives": NEGATIVES,
        "object_pose": {"x": x, "y": y, "z": z, "yaw": yaw},
    }


# ── Core functions ────────────────────────────────────────────────────

def create_pose_dict(csv_path):
    """Read CSV and extract per-frame pose values (x, y, z, yaw)."""
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


def create_prompt_json(pose_dict, frame_indices, obj_id, sensor_type,
                       output_dir, prompt_style="short"):
    """Create a JSONL prompt file for one object + sensor combination.

    Args:
        prompt_style: "short" (sensor_context + pose) or
                      "long" (full description with all fields)
    """
    builder = build_long_prompt if prompt_style == "long" else build_short_prompt

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_long" if prompt_style == "long" else ""
    out_path = os.path.join(output_dir, f"prompt{suffix}.json")
    count = 0

    with open(out_path, "w") as f:
        for idx in frame_indices:
            if idx not in pose_dict:
                print(f"  WARNING: Frame {idx} not in CSV, skipping.")
                continue
            x, y, z, yaw = pose_dict[idx]
            entry = {
                "source": f"source/{obj_id}_{idx}.png",
                "target": f"target/{obj_id}_{sensor_type}_{idx}.png",
                "prompt": builder(obj_id, sensor_type, x, y, z, yaw),
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"  Prompt JSON ({prompt_style}): {count} entries -> {out_path}")
    return out_path


def prompt_processing(csv_dir, tactile_dir, obj_id, sensor_type,
                      prompt_style="short"):
    """End-to-end prompt creation for one object + sensor."""
    csv_path = os.path.join(csv_dir, f"{obj_id}.csv")
    target_dir = os.path.join(tactile_dir, str(obj_id), sensor_type, "target")
    output_dir = os.path.join(tactile_dir, str(obj_id), sensor_type)

    frame_indices = get_frame_indices(target_dir, obj_id, sensor_type)
    if not frame_indices:
        print(f"  WARNING: No target images found in {target_dir}")
        return

    pose_dict = create_pose_dict(csv_path)
    print(f"  Pose dict: {len(pose_dict)} rows, {len(frame_indices)} target frames")
    create_prompt_json(pose_dict, frame_indices, obj_id, sensor_type,
                       output_dir, prompt_style=prompt_style)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create prompt JSONL from CSV poses")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing <obj_id>.csv files")
    parser.add_argument("--tactile_dir", type=str, required=True,
                        help="Root tactile dir")
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--sensor_type", type=str, default="ViTac",
                        choices=["TacTip", "ViTac", "ViTacTip"])
    parser.add_argument("--prompt_style", type=str, default="short",
                        choices=["short", "long"],
                        help="short: sensor context + pose (default). "
                             "long: full description with all fields (ablation 2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompt_processing(args.csv_dir, args.tactile_dir, args.obj_id,
                      args.sensor_type, prompt_style=args.prompt_style)
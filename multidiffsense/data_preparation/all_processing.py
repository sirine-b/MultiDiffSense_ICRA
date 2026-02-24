"""
MultiDiffSense — All Processing (Orchestrator)

Runs the complete per-object data preparation pipeline across all sensor types.

PROCESSING ORDER:
  1. ViTac FIRST  — target processing + source processing + prompt creation
  2. TacTip       — target processing + copy source from ViTac + prompt creation
  3. ViTacTip     — target processing + copy source from ViTac + prompt creation

WHY ViTac FIRST:
  Source (depth map) generation requires extracting the object from the tactile
  image to determine its bounding box and centre position. ViTac images are
  vision-only with no pin markers on the sensor surface, making the object
  boundary much clearer and easier to segment than TacTip (pin markers) or
  ViTacTip (hybrid markers). The source depth maps are identical across all
  three sensors since they represent the same object at the same pose — only
  the tactile response differs. Therefore, source images generated from ViTac
  are copied directly to TacTip and ViTacTip.

Usage:
    python -m multidiffsense.data_preparation.all_processing \
        --stl_dir data/example/stl \
        --csv_dir data/example/csv \
        --tactile_dir data/example/tactile \
        --obj_ids 1 3 6 8 18
"""

import argparse
import os
import shutil

from multidiffsense.data_preparation.target_processing import target_processing
from multidiffsense.data_preparation.source_processing import source_processing
from multidiffsense.data_preparation.prompt_creation import prompt_processing


SENSOR_ORDER = ["ViTac", "TacTip", "ViTacTip"]


def copy_source_images(tactile_dir, obj_id, from_sensor, to_sensor):
    """Copy source (depth map) images from one sensor directory to another.

    Since the depth maps are identical across sensors (same object, same pose),
    we generate them once from ViTac and copy to the others.
    """
    src_dir = os.path.join(tactile_dir, str(obj_id), from_sensor, "source")
    dst_dir = os.path.join(tactile_dir, str(obj_id), to_sensor, "source")
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"  WARNING: Source directory not found: {src_dir}")
        return 0

    count = 0
    for filename in sorted(os.listdir(src_dir)):
        if not filename.endswith(".png"):
            continue
        shutil.copy2(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))
        count += 1

    print(f"  Copied {count} source images: {from_sensor} -> {to_sensor}")
    return count


def all_processing(stl_dir, csv_dir, tactile_dir, obj_id, prompt_style="short"):
    """Run the full pipeline for one object across all sensor types.

    Processing order:
      1. ViTac:    target -> source (from STL) -> prompt
      2. TacTip:   target -> copy source from ViTac -> prompt
      3. ViTacTip: target -> copy source from ViTac -> prompt
    """
    print(f"\n{'='*60}")
    print(f"Processing object {obj_id} (prompt_style={prompt_style})")
    print(f"{'='*60}")

    for sensor in SENSOR_ORDER:
        print(f"\n{'-'*40}")
        print(f"Sensor: {sensor}")
        print(f"{'-'*40}")

        # Step 1: Target processing (rename + resize) -- all sensors
        print(f"\n[1] Target processing ({sensor})...")
        target_processing(tactile_dir, obj_id, sensor)

        # Step 2: Source processing -- ViTac only; copy for others
        if sensor == "ViTac":
            print(f"\n[2] Source processing ({sensor} -- generating from STL)...")
            source_processing(stl_dir, csv_dir, tactile_dir, obj_id, sensor)
        else:
            print(f"\n[2] Copying source images from ViTac -> {sensor}...")
            copy_source_images(tactile_dir, obj_id, "ViTac", sensor)

        # Step 3: Prompt creation -- all sensors
        print(f"\n[3] Prompt creation ({sensor}, style={prompt_style})...")
        prompt_processing(csv_dir, tactile_dir, obj_id, sensor,
                          prompt_style=prompt_style)

    print(f"\nObject {obj_id} complete -- all 3 sensors processed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full data preparation pipeline for specified objects"
    )
    parser.add_argument("--stl_dir", type=str, required=True,
                        help="Directory containing <obj_id>.stl files")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing <obj_id>.csv files")
    parser.add_argument("--tactile_dir", type=str, required=True,
                        help="Root tactile dir: <tactile_dir>/<obj_id>/<sensor>/target/")
    parser.add_argument("--obj_ids", nargs="+", required=True,
                        help="Object IDs to process (e.g., 1 3 6 8 18)")
    parser.add_argument("--prompt_style", type=str, default="short",
                        choices=["short", "long"],
                        help="short: sensor context + pose. "
                             "long: full description (ablation 2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for obj_id in args.obj_ids:
        all_processing(args.stl_dir, args.csv_dir, args.tactile_dir,
                       obj_id, prompt_style=args.prompt_style)
    print(f"\nAll objects processed: {args.obj_ids}")
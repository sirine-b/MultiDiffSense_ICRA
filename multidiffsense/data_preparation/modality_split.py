"""
MultiDiffSense — Modality Split

Splits a prompt.json file into per-sensor-modality files
(prompt_TacTip.json, prompt_ViTac.json, prompt_ViTacTip.json)
for per-modality evaluation.

Usage:
    python -m multidiffsense.data_preparation.modality_split \
        --prompt_path datasets/training/test/prompt.json \
        --output_dir datasets/training/test
"""

import argparse
import json
import os


def modality_split(prompt_path, output_dir):
    """Split prompt.json into per-sensor files based on target path.

    Sensor type is inferred from the target path, e.g.:
        target/1_ViTacTip_0.png → ViTacTip
    """
    modality_prompts = {
        "TacTip": [],
        "ViTac": [],
        "ViTacTip": [],
    }

    with open(prompt_path, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            target_path = entry["target"]
            basename = os.path.basename(target_path)

            # Find which sensor type is in the filename
            matched = False
            # Check ViTacTip before ViTac to avoid partial match
            for sensor in ["ViTacTip", "TacTip", "ViTac"]:
                if sensor in basename:
                    modality_prompts[sensor].append(entry)
                    matched = True
                    break

            if not matched:
                print(f"  WARNING: Could not infer sensor from {target_path}")

    os.makedirs(output_dir, exist_ok=True)

    for sensor, entries in modality_prompts.items():
        if not entries:
            continue
        out_path = os.path.join(output_dir, f"prompt_{sensor}.json")
        with open(out_path, "w") as fout:
            for entry in entries:
                fout.write(json.dumps(entry) + "\n")
        print(f"  {sensor}: {len(entries)} samples → {out_path}")

    print("Modality split complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Split prompt.json by sensor modality")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to prompt.json to split")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write prompt_<sensor>.json files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    modality_split(args.prompt_path, args.output_dir)

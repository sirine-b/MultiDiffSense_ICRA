"""
MultiDiffSense — Dataset Split

Splits the merged prompt.json into train/val/test subsets (70/15/15).
Groups by source image so all sensor modalities for the same contact
stay in the same split.

Usage:
    python -m multidiffsense.data_preparation.dataset_split \
        --base_dir datasets/training \
        --seed 16
"""

import argparse
import json
import os
import random
from collections import defaultdict


def dataset_split(base_dir, seed=16, train_ratio=0.7, val_ratio=0.15,
                  prompt_style="short"):
    """Split prompt file into train/val/test by grouping on source image.

    Args:
        base_dir: Directory containing the merged prompt file
        seed: Random seed for reproducibility
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (remainder goes to test)
        prompt_style: "short" or "long" -- selects which prompt file to split
    """
    suffix = "_long" if prompt_style == "long" else ""
    prompt_filename = f"prompt{suffix}.json"
    prompt_path = os.path.join(base_dir, prompt_filename)

    with open(prompt_path, "r") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    # Group by source image so aligned modalities stay together
    groups = defaultdict(list)
    for entry in prompts:
        groups[entry["source"]].append(entry)

    # Shuffle groups
    group_keys = list(groups.keys())
    random.seed(seed)
    random.shuffle(group_keys)

    # Split
    n = len(group_keys)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": group_keys[:train_end],
        "val": group_keys[train_end:val_end],
        "test": group_keys[val_end:],
    }

    # Write each split
    for split_name, keys in splits.items():
        split_dir = os.path.join(base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        out_path = os.path.join(split_dir, prompt_filename)
        count = 0
        with open(out_path, "w") as f:
            for key in keys:
                for entry in groups[key]:
                    f.write(json.dumps(entry) + "\n")
                    count += 1

        print(f"  {split_name}: {count} samples ({len(keys)} groups) -> {out_path}")

    print("Splits created!")


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Directory containing prompt.json")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--prompt_style", type=str, default="short",
                        choices=["short", "long"],
                        help="Which prompt file to split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_split(args.base_dir, seed=args.seed,
                  train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                  prompt_style=args.prompt_style)
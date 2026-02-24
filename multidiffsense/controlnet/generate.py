"""
MultiDiffSense — Inference / Generation Script

Generate tactile sensor images from depth maps without ground truth targets.

Usage:
    python multidiffsense/controlnet/generate.py \
        --config configs/controlnet_train.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --source_dir data/new_objects/ \
        --prompt_json data/new_objects/prompts.json \
        --output_dir results/generated
"""

import argparse
import os
import sys
import time

# Ensure repo root is on Python path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from cldm.model import create_model, load_state_dict
from multidiffsense.controlnet.data_loader import MultiDiffSenseDataset


def normalise(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def parse_args():
    parser = argparse.ArgumentParser(description="MultiDiffSense Generation")
    parser.add_argument("--config", type=str, default="configs/controlnet_train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root directory containing source/ images")
    parser.add_argument("--prompt_json", type=str, required=True,
                        help="JSONL file with source paths and prompts")
    parser.add_argument("--output_dir", type=str, default="results/generated")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = create_model(cfg["model"]["controlnet_config"]).cpu()
    model.load_state_dict(load_state_dict(args.checkpoint, location="cpu"))
    model = model.to(device).eval()

    # Dataset (no target needed)
    dataset = MultiDiffSenseDataset(
        dataset_dir=args.dataset_dir,
        prompt_json=args.prompt_json,
        no_target=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Output directories
    images_dir = os.path.join(args.output_dir, "images")
    grid_dir = os.path.join(args.output_dir, "grid")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    inference_times = []

    for idx, batch in enumerate(tqdm(loader)):
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        start_time = time.time()
        with torch.no_grad():
            log = model.log_images(batch, sample=True, N=1, n_row=1)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        control_img = log["control"]
        gen_img = log["samples"]

        # Save
        grid = make_grid(
            torch.cat([normalise(control_img), normalise(gen_img)], dim=0), nrow=2
        )
        save_image(grid, os.path.join(grid_dir, f"{idx:04d}.png"))
        save_image(normalise(gen_img), os.path.join(images_dir, f"{idx:04d}.png"))

        prompt = batch.get("txt", [""])[0] if isinstance(batch.get("txt"), list) else batch.get("txt")
        print(f"[{idx:04d}] Prompt: {prompt} | {inference_time:.3f}s")

    print(f"\nInference Time — Avg: {sum(inference_times)/len(inference_times):.3f}s, "
          f"Min: {min(inference_times):.3f}s, Max: {max(inference_times):.3f}s")
    print(f"Generated {len(inference_times)} images → {args.output_dir}")


if __name__ == "__main__":
    main()
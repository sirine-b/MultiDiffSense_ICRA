"""
MultiDiffSense -- Inference / Generation Script

Generate tactile sensor images from depth maps using a trained checkpoint.
The pre-trained model is downloaded automatically from Hugging Face on first run
if no local checkpoint is provided.

Usage:
    # Single depth map + text prompt:
    python multidiffsense/controlnet/generate.py \
        --source_image path/to/depth_map.png \
        --prompt '{"sensor_context": "captured by a high-resolution vision only sensor ViTac.", "object_pose": {"x": 0.12, "y": -0.34, "z": 1.5, "yaw": 15.0}}'

    # Batch from prompt file:
    python multidiffsense/controlnet/generate.py \
        --dataset_dir datasets \
        --prompt_json datasets/test/prompt_ViTacTip.json

    # Use a specific local checkpoint:
    python multidiffsense/controlnet/generate.py \
        --checkpoint path/to/checkpoint.ckpt \
        --config configs/controlnet_train.yaml \
        --dataset_dir datasets \
        --prompt_json datasets/test/prompt_ViTacTip.json
"""

import argparse
import os
import sys
import time

# Ensure repo root is on Python path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from cldm.model import create_model, load_state_dict
from multidiffsense.controlnet.data_loader import MultiDiffSenseDataset

HF_REPO = "sirine16/MultiDiffSense"
HF_CKPT = "multidiffsense.ckpt"


def normalise(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def download_from_hf(local_dir="models"):
    """Download checkpoint from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    os.makedirs(local_dir, exist_ok=True)
    ckpt = hf_hub_download(repo_id=HF_REPO, filename=HF_CKPT, local_dir=local_dir)
    print(f"Downloaded checkpoint: {ckpt}")
    return ckpt


def load_model(checkpoint, controlnet_config, device="cuda"):
    """Load model from checkpoint and config."""
    model = create_model(controlnet_config).cpu()
    model.load_state_dict(load_state_dict(checkpoint, location="cpu"))
    model = model.to(device).eval()
    return model


def resolve_model(args, device="cuda"):
    """Resolve checkpoint and config paths, downloading from HF if needed."""
    checkpoint = args.checkpoint
    controlnet_config = None

    # If a YAML config is provided, read controlnet_config from it
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        controlnet_config = cfg["model"]["controlnet_config"]

    # If no checkpoint provided, download from HF
    if checkpoint is None or not os.path.exists(checkpoint):
        print("No local checkpoint found. Downloading from Hugging Face...")
        checkpoint = download_from_hf()

    # Fallback config path
    if controlnet_config is None:
        controlnet_config = os.path.join(_repo_root, "configs", "cldm_v15.yaml")

    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {controlnet_config}")
    return load_model(checkpoint, controlnet_config, device=device)


def generate_batch(model, dataset_dir, prompt_json, output_dir, device="cuda"):
    """Generate tactile images for all entries in a prompt file."""
    dataset = MultiDiffSenseDataset(
        dataset_dir=dataset_dir,
        prompt_json=prompt_json,
        no_target=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    images_dir = os.path.join(output_dir, "images")
    grid_dir = os.path.join(output_dir, "grid")
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

        grid = make_grid(
            torch.cat([normalise(control_img), normalise(gen_img)], dim=0), nrow=2
        )
        save_image(grid, os.path.join(grid_dir, f"{idx:04d}.png"))
        save_image(normalise(gen_img), os.path.join(images_dir, f"{idx:04d}.png"))

        prompt = batch.get("txt", [""])[0] if isinstance(batch.get("txt"), list) else batch.get("txt")
        print(f"[{idx:04d}] Prompt: {prompt} | {inference_time:.3f}s")

    print(f"\nInference Time -- Avg: {sum(inference_times)/len(inference_times):.3f}s, "
          f"Min: {min(inference_times):.3f}s, Max: {max(inference_times):.3f}s")
    print(f"Generated {len(inference_times)} images -> {output_dir}")


def generate_single(model, source_image_path, prompt, output_dir, device="cuda"):
    """Generate a tactile image from a single depth map + text prompt."""
    # Load and preprocess source image
    source = cv2.imread(source_image_path)
    if source is None:
        raise FileNotFoundError(f"Cannot read: {source_image_path}")
    print(f"After imread: {source.shape}, dtype: {source.dtype}")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    print(f"After cvtColor: {source.shape}")
    source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_LINEAR)
    print(f"After resize: {source.shape}")
    source = source.astype(np.float32) / 255.0

    # Build batch dict matching what model.log_images() expects
    hint = torch.from_numpy(source).float().unsqueeze(0).to(device) 
    print(f"Hint tensor shape: {hint.shape}")  # must be [1, 3, 512, 512]
 
    # Blank target placeholder (log_images expects 'jpg' key but it is
    # not used during sampling)
    jpg = torch.zeros_like(hint)

    batch = {
        "jpg": jpg,
        "hint": hint,
        "txt": [prompt],
    }

    start_time = time.time()
    with torch.no_grad():
        log = model.log_images(batch, sample=True, N=1, n_row=1)
    inference_time = time.time() - start_time

    control_img = log["control"]
    gen_img = log["samples"]

    images_dir = os.path.join(output_dir, "images")
    grid_dir = os.path.join(output_dir, "grid")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    grid = make_grid(
        torch.cat([normalise(control_img), normalise(gen_img)], dim=0), nrow=2
    )
    save_image(grid, os.path.join(grid_dir, "generated_grid.png"))
    save_image(normalise(gen_img), os.path.join(images_dir, "generated.png"))

    print(f"Prompt: {prompt}")
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="MultiDiffSense Generation")
    # Model
    parser.add_argument("--config", type=str, default="configs/controlnet_train.yaml",
                        help="Path to training config YAML (optional if using HF)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .ckpt file (downloads from HF if not provided)")

    # Batch mode
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Root directory containing source/ images")
    parser.add_argument("--prompt_json", type=str, default=None,
                        help="JSONL file with source paths and prompts")

    # Single image mode
    parser.add_argument("--source_image", type=str, default=None,
                        help="Path to a single depth map image")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt as a JSON string matching prompt.json format, "
                             "e.g. '{\"sensor_context\": \"...\", \"object_pose\": {...}}'")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/generated")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = resolve_model(args, device=device)

    if args.source_image and args.prompt:
        # Single image mode
        # Parse prompt the same way data_loader.py does: if the prompt is a
        # JSON dict, str() it to match the format the model was trained on.
        import json as _json
        try:
            prompt_parsed = _json.loads(args.prompt)
            prompt_str = str(prompt_parsed)
        except (_json.JSONDecodeError, TypeError):
            prompt_str = args.prompt

        generate_single(model, args.source_image, prompt_str,
                         args.output_dir, device=device)

    elif args.dataset_dir and args.prompt_json:
        # Batch mode
        generate_batch(model, args.dataset_dir, args.prompt_json,
                        args.output_dir, device=device)

    else:
        print("Provide either:")
        print("  --source_image + --prompt     (single depth map)")
        print("  --dataset_dir + --prompt_json (batch from prompt file)")
        sys.exit(1)


if __name__ == "__main__":
    main()
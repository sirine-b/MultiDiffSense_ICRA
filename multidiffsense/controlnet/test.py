"""
MultiDiffSense — ControlNet Testing Script

Evaluates a trained ControlNet model on seen or unseen objects with
quantitative metrics: SSIM, PSNR, MSE, LPIPS, FID.

Usage:
    python multidiffsense/controlnet/test.py \
        --config configs/controlnet_train.yaml \
        --checkpoint path/to/epoch=119-step=78840.ckpt \
        --modality ViTacTip \
        --output_dir results/test_vitactip

    # Ablation: remove text prompt
    python multidiffsense/controlnet/test.py ... --no_prompt

    # Ablation: remove depth map conditioning
    python multidiffsense/controlnet/test.py ... --no_source
"""

import argparse
import csv
import os
import sys
import time

# Ensure repo root is on Python path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cldm.model import create_model, load_state_dict
from multidiffsense.controlnet.data_loader import MultiDiffSenseDataset
from multidiffsense.evaluation.metrics import (
    compute_fid,
    compute_lpips,
    compute_mse,
    compute_psnr,
    compute_ssim,
    normalise_for_save,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MultiDiffSense ControlNet Testing")
    parser.add_argument("--config", type=str, default="configs/controlnet_train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained ControlNet checkpoint")
    parser.add_argument("--modality", type=str, default="ViTacTip",
                        choices=["TacTip", "ViTac", "ViTacTip"],
                        help="Sensor modality to test")
    parser.add_argument("--seen_objects", action="store_true",
                        help="Test on seen objects (from train/val/test split)")
    parser.add_argument("--output_dir", type=str, default="results/test")
    parser.add_argument("--batch_size", type=int, default=1)
    # Ablation flags
    parser.add_argument("--no_prompt", action="store_true",
                        help="Ablation: remove text prompts")
    parser.add_argument("--no_source", action="store_true",
                        help="Ablation: remove depth map conditioning")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model ---
    print(f"Loading model from {args.checkpoint}...")
    model = create_model(cfg["model"]["controlnet_config"]).cpu()
    model.load_state_dict(load_state_dict(args.checkpoint, location="cpu"))
    model = model.to(device).eval()

    # --- Dataset ---
    dataset_dir = cfg["data"]["dataset_dir"]
    if args.seen_objects:
        prompt_json = os.path.join(
            dataset_dir, "test", f"prompt_{args.modality}.json"
        )
    else:
        # Unseen objects — separate directory
        unseen_dir = cfg["data"].get("unseen_test_dir", "data/unseen/")
        dataset_dir = unseen_dir
        prompt_json = os.path.join(unseen_dir, f"prompt_{args.modality}.json")

    test_dataset = MultiDiffSenseDataset(
        dataset_dir=dataset_dir,
        prompt_json=prompt_json,
        no_prompt=args.no_prompt,
        no_source=args.no_source,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # --- Output directories ---
    tar_dir = os.path.join(args.output_dir, "target")
    gen_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    # --- Testing loop ---
    from tqdm import tqdm

    all_ssim, all_psnr, all_mse, all_lpips = [], [], [], []
    inference_times = []

    for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
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
        target_img = log["reconstruction"]
        gen_img = log["samples"]

        g = normalise_for_save(gen_img.squeeze(0))
        t = normalise_for_save(target_img.squeeze(0))

        # Compute metrics
        ssim_val = compute_ssim(g, t)
        psnr_val = compute_psnr(g, t)
        mse_val = compute_mse(g, t)
        lpips_val = compute_lpips(g, t, device=device)

        all_ssim.append(ssim_val)
        all_psnr.append(psnr_val)
        all_mse.append(mse_val)
        all_lpips.append(lpips_val)

        # Save outputs
        grid = make_grid(
            torch.cat([
                normalise_for_save(control_img),
                normalise_for_save(target_img),
                normalise_for_save(gen_img),
            ], dim=0),
            nrow=3,
        )
        save_image(grid, os.path.join(args.output_dir, f"{idx:04d}_grid.png"))
        save_image(normalise_for_save(target_img), os.path.join(tar_dir, f"{idx:04d}.png"))
        save_image(normalise_for_save(gen_img), os.path.join(gen_dir, f"{idx:04d}.png"))

        prompt = batch.get("txt", [""])[0] if isinstance(batch.get("txt"), list) else batch.get("txt")
        print(
            f"[{idx:04d}] SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}, "
            f"MSE: {mse_val:.4e}, LPIPS: {lpips_val:.4f} | {inference_time:.3f}s"
        )

    # --- Aggregate metrics ---
    avg_ssim = np.mean(all_ssim)
    avg_psnr = np.mean(all_psnr)
    avg_mse = np.mean(all_mse)
    avg_lpips = np.mean(all_lpips)
    fid_val = compute_fid(gen_dir, tar_dir)

    print(f"\n{'='*50}")
    print(f"Results — {args.modality} ({'seen' if args.seen_objects else 'unseen'} objects)")
    if args.no_prompt:
        print("  [ABLATION: no text prompt]")
    if args.no_source:
        print("  [ABLATION: no depth map conditioning]")
    print(f"{'='*50}")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average PSNR:  {avg_psnr:.2f} dB")
    print(f"Average MSE:   {avg_mse:.4e}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"FID:           {fid_val:.4f}")
    print(f"\nInference Time — Avg: {np.mean(inference_times):.3f}s, "
          f"Min: {min(inference_times):.3f}s, Max: {max(inference_times):.3f}s")

    # --- Save CSV ---
    csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Index", "SSIM", "PSNR", "MSE", "LPIPS"])
        for i, (s, p, m, l) in enumerate(zip(all_ssim, all_psnr, all_mse, all_lpips)):
            writer.writerow([i, f"{s:.4f}", f"{p:.2f}", f"{m:.4e}", f"{l:.4f}"])
        writer.writerow([])
        writer.writerow(["Average SSIM", f"{avg_ssim:.4f}"])
        writer.writerow(["Average PSNR", f"{avg_psnr:.2f}"])
        writer.writerow(["Average MSE", f"{avg_mse:.4e}"])
        writer.writerow(["Average LPIPS", f"{avg_lpips:.4f}"])
        writer.writerow(["FID", f"{fid_val:.4f}"])

    print(f"\nMetrics saved to {csv_path}")


if __name__ == "__main__":
    main()
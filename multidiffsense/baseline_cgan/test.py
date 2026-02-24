"""
MultiDiffSense — cGAN (Pix2Pix) Baseline Testing

Tests a trained Pix2Pix model and computes the same metrics as the ControlNet test
(SSIM, PSNR, MSE, LPIPS, FID) for fair comparison.

Requires: pytorch-CycleGAN-and-pix2pix cloned into external/

Usage:
    python multidiffsense/baseline_cgan/test.py \
        --dataroot external/pytorch-CycleGAN-and-pix2pix/datasets/depth_to_sensor_TacTip \
        --name depth_to_sensor_experiment \
        --modality TacTip \
        --phase test
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# Add pix2pix to path
PGAN_DIR = os.path.join(os.path.dirname(__file__), "../../external/pytorch-CycleGAN-and-pix2pix")
sys.path.insert(0, PGAN_DIR)

from multidiffsense.evaluation.metrics import (
    compute_fid,
    compute_lpips,
    compute_mse,
    compute_psnr,
    compute_ssim,
    normalise_for_save,
)


def main():
    from data import create_dataset
    from models import create_model
    from options.test_options import TestOptions

    device = "cuda" if torch.cuda.is_available() else "cpu"

    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.dataset_mode = "aligned"
    opt.eval = True

    modality = getattr(opt, "modality", "TacTip")
    out_root = f"./pix2pix_test_{modality}"
    tar_dir = os.path.join(out_root, "target")
    gen_dir = os.path.join(out_root, "generated")
    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    all_ssim, all_psnr, all_mse, all_lpips = [], [], [], []
    inference_times = []

    for idx, data in enumerate(tqdm(dataset, desc="Testing Pix2Pix")):
        model.set_input(data)
        start_time = time.time()
        with torch.no_grad():
            model.test()
            visuals = model.get_current_visuals()
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        real_B = visuals["real_B"]
        fake_B = visuals["fake_B"]

        g = normalise_for_save(fake_B.squeeze(0))
        t = normalise_for_save(real_B.squeeze(0))

        ssim_val = compute_ssim(g, t)
        psnr_val = compute_psnr(g, t)
        mse_val = compute_mse(g, t)
        lpips_val = compute_lpips(g, t, device=device)

        all_ssim.append(ssim_val)
        all_psnr.append(psnr_val)
        all_mse.append(mse_val)
        all_lpips.append(lpips_val)

        save_image(normalise_for_save(real_B), os.path.join(tar_dir, f"{idx:04d}.png"))
        save_image(normalise_for_save(fake_B), os.path.join(gen_dir, f"{idx:04d}.png"))

    avg_ssim = np.mean(all_ssim)
    avg_psnr = np.mean(all_psnr)
    avg_mse = np.mean(all_mse)
    avg_lpips = np.mean(all_lpips)
    fid_val = compute_fid(gen_dir, tar_dir)

    print(f"\nPix2Pix Baseline — {modality}")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average PSNR:  {avg_psnr:.2f} dB")
    print(f"Average MSE:   {avg_mse:.4e}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"FID:           {fid_val:.4f}")

    csv_path = os.path.join(out_root, "test_metrics.csv")
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

    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()

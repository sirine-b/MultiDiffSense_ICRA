"""
MultiDiffSense — Evaluation Metrics

Computes SSIM, PSNR, MSE, LPIPS, and FID for comparing generated
tactile images against ground truth.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

# Lazy-loaded LPIPS model (shared across calls)
_lpips_fn = None


def normalise_for_save(t):
    """Normalise tensor to [0, 1] for saving as image."""
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def compute_ssim(pred, tgt):
    """Compute SSIM between two tensors (converted to grayscale)."""
    pred_img = TF.to_pil_image(pred.cpu()).convert("L")
    tgt_img = TF.to_pil_image(tgt.cpu()).convert("L")
    return _ssim(np.array(pred_img), np.array(tgt_img), data_range=255)


def compute_psnr(pred, tgt):
    """Compute PSNR between two tensors (converted to grayscale)."""
    pred_img = TF.to_pil_image(pred.cpu()).convert("L")
    tgt_img = TF.to_pil_image(tgt.cpu()).convert("L")
    return _psnr(np.array(tgt_img), np.array(pred_img), data_range=255)


def compute_mse(pred, tgt):
    """Compute Mean Squared Error between two tensors."""
    return torch.mean((pred - tgt) ** 2).item()


def compute_lpips(pred, tgt, device="cuda"):
    """Compute LPIPS (AlexNet) between two tensors."""
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="alex").to(device)

    pred = pred.unsqueeze(0).to(device) * 2 - 1  # [0,1] → [-1,1]
    tgt = tgt.unsqueeze(0).to(device) * 2 - 1
    return _lpips_fn(pred, tgt).item()


def compute_fid(gen_dir, tar_dir):
    """Compute FID between directories of generated and target images."""
    from torch_fidelity import calculate_metrics

    metrics = calculate_metrics(
        input1=gen_dir,
        input2=tar_dir,
        metrics=["fid"],
        fid=True,
        isc=False,
        verbose=True,
    )
    return metrics["frechet_inception_distance"]

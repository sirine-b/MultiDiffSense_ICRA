"""
MultiDiffSense -- ControlNet Training Script

Trains a ControlNet model conditioned on depth maps with text prompts
to generate multi-modal tactile sensor images.

Usage:
    # Full model (default)
    python multidiffsense/controlnet/train.py \
        --config configs/controlnet_train.yaml \
        --batch_size 8 --lr 1e-5 --max_epochs 350 --sd_locked

    # Ablation 1a: train WITHOUT text prompt (source only)
    python multidiffsense/controlnet/train.py \
        --config configs/controlnet_train.yaml \
        --no_prompt --output_suffix _no_prompt

    # Ablation 1b: train WITHOUT source depth map (prompt only)
    python multidiffsense/controlnet/train.py \
        --config configs/controlnet_train.yaml \
        --no_source --output_suffix _no_source

    # Ablation 2: train with long prompts (use a different prompt.json)
    python multidiffsense/controlnet/train.py \
        --config configs/controlnet_train_long_prompt.yaml
"""

import argparse
import json
import os
import signal
import sys

# Ensure repo root is on Python path (needed when running as a script
# rather than as a module, so that `cldm` and `multidiffsense` imports work)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset, random_split

from cldm.logger import ImageLogger
from cldm.loss_plotter import LossPlotter
from cldm.model import create_model, load_state_dict
from multidiffsense.controlnet.data_loader import MultiDiffSenseDataset

# Global reference for signal handler
_loss_plotter = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully -- save loss plot before exiting."""
    print("\nReceived interrupt signal. Saving current progress...")
    if _loss_plotter:
        _loss_plotter.save_current_plot(" (Interrupted)")
        _loss_plotter._save_history()
    print("Progress saved. Exiting...")
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="MultiDiffSense ControlNet Training")
    parser.add_argument("--config", type=str, default="configs/controlnet_train.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--sd_locked", action="store_true", default=None)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from a Lightning checkpoint")

    # Ablation flags
    parser.add_argument("--no_prompt", action="store_true",
                        help="Ablation: train with empty prompts (source-only conditioning)")
    parser.add_argument("--no_source", action="store_true",
                        help="Ablation: train with blank source images (prompt-only conditioning)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to output_dir for ablation runs "
                             "(e.g. _no_prompt, _no_source, _long_prompt)")
    return parser.parse_args()


def main():
    global _loss_plotter

    args = parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI args
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    learning_rate = args.lr or cfg["training"]["learning_rate"]
    max_epochs = args.max_epochs or cfg["training"]["max_epochs"]
    sd_locked = args.sd_locked if args.sd_locked is not None else cfg["training"]["sd_locked"]
    only_mid_control = cfg["training"].get("only_mid_control", False)
    output_dir = cfg["training"]["output_dir"] + args.output_suffix
    image_log_freq = cfg["training"].get("image_log_frequency", 300)
    num_workers = cfg["training"].get("num_workers", 4)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Print ablation mode
    if args.no_prompt:
        print("=" * 60)
        print("ABLATION MODE: no_prompt (training with empty text prompts)")
        print("=" * 60)
    if args.no_source:
        print("=" * 60)
        print("ABLATION MODE: no_source (training with blank depth maps)")
        print("=" * 60)

    # --- Model ---
    print("Loading model...")
    model = create_model(cfg["model"]["controlnet_config"]).cpu()
    model.load_state_dict(load_state_dict(cfg["model"]["resume_path"], location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # --- Datasets ---
    dataset_dir = cfg["data"]["dataset_dir"]

    train_dataset = MultiDiffSenseDataset(
        dataset_dir=dataset_dir,
        prompt_json=cfg["data"]["train_prompt"],
        no_prompt=args.no_prompt,
        no_source=args.no_source,
    )
    val_dataset = MultiDiffSenseDataset(
        dataset_dir=dataset_dir,
        prompt_json=cfg["data"]["val_prompt"],
        no_prompt=args.no_prompt,
        no_source=args.no_source,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Output directory: {output_dir}")

    # --- Callbacks ---
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="lightning_logs")
    image_logger = ImageLogger(batch_frequency=image_log_freq)
    _loss_plotter = LossPlotter()
    early_stop = EarlyStopping(
        monitor=cfg["training"].get("early_stop_monitor", "val/loss"),
        patience=cfg["training"].get("early_stop_patience", 10),
        mode="min",
        verbose=True,
    )

    # --- Trainer ---
    # Auto-detect: use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        print("WARNING: No GPU found -- training on CPU (this will be very slow)")
        accelerator = "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices="auto",
        precision=cfg["training"].get("precision", 32),
        logger=tb_logger,
        callbacks=[image_logger, _loss_plotter, early_stop],
        max_epochs=max_epochs,
        log_every_n_steps=cfg["training"].get("log_every_n_steps", 200),
        val_check_interval=cfg["training"].get("val_check_interval", 1.0),
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # --- Train ---
    try:
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=args.resume_from,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        if _loss_plotter:
            _loss_plotter.save_current_plot(" (User Interrupted)")
        print("Final plot saved. Exiting...")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if _loss_plotter:
            _loss_plotter.save_current_plot(" (Error)")
        raise


if __name__ == "__main__":
    main()
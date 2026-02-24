"""
MultiDiffSense Dataset Loader for ControlNet training and evaluation.

Loads (source, target, prompt) triplets from JSONL prompt files where:
  - source: depth map image (control signal, normalised to [0,1])
  - target: tactile sensor image (generation target, normalised to [-1,1])
  - prompt: text description of object, contact, and sensor type

Prompt JSONL format (one JSON object per line):
  {"source": "train/depth/001.png", "target": "train/tactile/001.png", "prompt": "..."}
"""

import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class MultiDiffSenseDataset(Dataset):
    """Dataset for MultiDiffSense ControlNet training and evaluation.

    The dataset directory should look like:

        datasets/
        ├── source/          ← all depth map images (shared across splits)
        ├── target/          ← all tactile images (shared across splits)
        ├── prompt.json      ← full prompt file (all samples)
        ├── train/
        │   └── prompt.json  ← train split prompt entries only
        ├── val/
        │   └── prompt.json
        └── test/
            └── prompt.json

    The prompt.json paths are relative to dataset_dir, e.g.:
        {"source": "source/1_0.png", "target": "target/1_ViTac_0.png", ...}

    So dataset_dir + source path = full path to the image.

    Args:
        dataset_dir: Root directory containing source/ and target/ subdirs.
        prompt_json: Path to JSONL file with source/target/prompt entries.
        no_prompt: If True, replace all prompts with empty string (ablation).
        no_source: If True, replace source with blank image (ablation).
        no_target: If True, skip loading target (inference-only mode).
        target_size: Tuple (W, H) to resize all images to.
    """

    def __init__(
        self,
        dataset_dir: str,
        prompt_json: str = None,
        no_prompt: bool = False,
        no_source: bool = False,
        no_target: bool = False,
        target_size: tuple = (512, 512),
    ):
        self.data = []
        self.no_prompt = no_prompt
        self.no_source = no_source
        self.no_target = no_target
        self.dataset_dir = dataset_dir
        self.target_size = target_size

        assert prompt_json is not None, "prompt_json path is required"

        with open(prompt_json, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        print(f"[Dataset] Loaded {len(self.data)} samples from {prompt_json}")
        print(f"[Dataset] dataset_dir={self.dataset_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item.get("source", None)
        target_filename = item.get("target", None)
        prompt = "" if self.no_prompt else str(item.get("prompt", ""))

        # --- Load source (depth map) ---
        if not self.no_source and source_filename is not None:
            src_path = os.path.join(self.dataset_dir, source_filename)
            source = cv2.imread(src_path)
            if source is None:
                raise FileNotFoundError(f"Source image not found: {src_path}")
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        else:
            source = np.zeros(
                (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
            )

        source = cv2.resize(source, self.target_size, interpolation=cv2.INTER_LINEAR)
        source = source.astype(np.float32) / 255.0  # Normalise to [0, 1]

        # --- Load target (tactile image) ---
        if not self.no_target and target_filename is not None:
            tgt_path = os.path.join(self.dataset_dir, target_filename)
            target = cv2.imread(tgt_path)
            if target is None:
                raise FileNotFoundError(f"Target image not found: {tgt_path}")
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            target = cv2.resize(
                target, self.target_size, interpolation=cv2.INTER_LINEAR
            )
            target = (target.astype(np.float32) / 127.5) - 1.0  # Normalise to [-1, 1]
        else:
            target = np.zeros_like(source, dtype=np.float32)

        return dict(hint=source, jpg=target, txt=prompt)
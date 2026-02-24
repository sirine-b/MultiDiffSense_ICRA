# Baseline Comparison: Pix2Pix (cGAN)

This directory contains scripts to train and evaluate a Pix2Pix model as a conditional GAN baseline against MultiDiffSense.

## Setup

The Pix2Pix baseline uses the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) framework. Clone it first:

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix external/pytorch-CycleGAN-and-pix2pix
cd external/pytorch-CycleGAN-and-pix2pix
pip install -r requirements.txt
```

## Dataset Conversion

Convert the ControlNet-format dataset to Pix2Pix format:

```bash
python multidiffsense/baseline_cgan/dataset_converter.py \
    --controlnet_dataset data/processed \
    --output_path external/pytorch-CycleGAN-and-pix2pix/datasets/depth_to_sensor_TacTip \
    --modality TacTip
```

This creates paired directories (`trainA`/`trainB`, `valA`/`valB`, `testA`/`testB`) where:
- **A** = depth maps (condition)
- **B** = tactile sensor images (target)

## Training

```bash
cd external/pytorch-CycleGAN-and-pix2pix
python train.py \
    --dataroot datasets/depth_to_sensor_TacTip \
    --name depth_to_sensor_experiment \
    --model pix2pix \
    --direction AtoB \
    --n_epochs 200 \
    --n_epochs_decay 100
```

## Testing

```bash
python multidiffsense/baseline_cgan/test.py \
    --dataroot external/pytorch-CycleGAN-and-pix2pix/datasets/depth_to_sensor_TacTip \
    --name depth_to_sensor_experiment \
    --phase test
```

The test script computes the same metrics (SSIM, PSNR, MSE, LPIPS, FID) as the ControlNet test for fair comparison.

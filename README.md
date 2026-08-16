# KLA PS01 — AI-Based Restoration of Degraded Images

SEMICON India Hackathon 2026 — Track 1 (KLA)

This repository is a starter implementation for paired `.npy` semiconductor-image restoration.

## Structure
- `train.py` — training
- `inference.py` — standalone inference
- `evaluate.py` — PSNR/SSIM evaluation
- `models/model.py` — residual U-Net baseline
- `data/dataset.py` — paired `.npy` loader
- `configs/config.yaml` — configuration

## Expected dataset
```text
dataset/
├── train/
│   ├── NoisyLR/
│   └── GT/
└── Test_NoisyLR/
```
Training files are paired by identical filename.

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py --noisy_dir /path/to/train/NoisyLR --gt_dir /path/to/train/GT --output_dir ./weights
```

## Inference
```bash
python inference.py --input_dir /path/to/Test_NoisyLR --output_dir ./outputs --weights ./weights/final_model.pt
```

## Evaluate
```bash
python evaluate.py --pred_dir ./outputs --gt_dir /path/to/GT --csv_path ./results/metrics.csv
```

**Important:** final benchmark dimensions, normalization, output format and metrics must be verified against the official KLA instructions. Do not claim measured results until they have actually been generated.

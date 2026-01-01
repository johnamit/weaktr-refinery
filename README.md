<h1>
  WeakTR-Refinery
  <a href="https://drive.google.com/file/d/1ab02W1ty949CBcTFcbvVeddcFmWR3-iL/view?usp=drive_link">
    <img align="right"
      src="https://img.shields.io/badge/Read%20Paper-PDF-black?style=for-the-badge&labelColor=0057FF&logo=adobeacrobat&logoColor=white"
      alt="Read Paper"/>
  </a>
</h1>

A weakly-supervised semantic segmentation framework that leverages Vision Transformers (ViT) and Class Activation Maps (CAMs) to generate pseudo-masks for training segmentation models, requiring only image-level labels instead of expensive pixel-level annotations.

## Overview

This project implements a multi-stage pipeline for weakly-supervised segmentation:

1. **ViT Classification** — Train a Vision Transformer classifier on image-level labels
2. **FineCAM Generation** — Extract refined Class Activation Maps from the trained ViT
3. **CNN Decoder Training** — Train a decoder to refine the CAMs into better pseudo-masks
4. **Mask Refinement** — Generate high-quality pseudo-masks using the trained decoder
5. **Segmentation Training** — Train a segmentation model (LR-ASPP MobileNetV3) using pseudo-masks
6. **Evaluation** — Evaluate on ground truth test set

The framework supports mixed supervision, allowing you to combine pseudo-masks with varying proportions of ground truth labels.



## Prerequisites

* **Python** 3.12+
* **PyTorch** 2.6+
* **CUDA** 12.1+ (for GPU acceleration)

**Tested on:** NVIDIA RTX 3090 (24GB) • Ryzen 7 7800X3D • 32GB RAM



## Project Structure

```
weaktr-refinery/
├── configs/
│   └── default.py              # Default configuration dataclass
├── data/                       # Dataset directory (not tracked)
├── models/                     # Trained model weights (LFS tracked)
├── scripts/
│   ├── main.py                 # Entry point / orchestrator
│   ├── split_data.py           # Train/test split generation
│   ├── train_ViT_classification.py
│   ├── build_fine_cams.py      # FineCAM extraction
│   ├── train_CNN_decoder.py    # Decoder training
│   ├── build_masks.py          # Pseudo-mask generation
│   ├── train_supervised_gt.py  # Fully supervised baseline
│   ├── train_supervised_ws.py  # Weakly supervised training
│   ├── evaluation.py           # Evaluation metrics
│   └── utils/
│       ├── utils.py            # Helper functions
│       └── environment.yml     # Conda environment
└── README.md
```



## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/johnamit/weaktr-refinery.git
   cd weaktr-refinery
   ```

2. **Create the conda environment**
   ```bash
   conda env create -f scripts/utils/environment.yml
   ```

3. **Activate the environment**
   ```bash
   conda activate weaktr_refinery
   ```

4. **Prepare your dataset**
   
   Place the [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) in the data folder:
   ```
   data/
   ├── images/           # Training images
   ├── annotations/      # Ground truth masks (for evaluation)
   │   └── trimaps/
   └── Split/            # Auto-generated train/test splits
   ```



## Usage

### Training

#### Default Configuration (DeiT-Tiny + Medium Decoder)

```bash
cd scripts
python main.py --train
```

#### Custom Configuration

```bash
python main.py --train \
    --vit_model tiny \
    --decoder_size medium \
    --seed 42 \
    --loss_threshold 0.35 \
    --epochs 4
```

#### Full Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train` | flag | - | Enable training mode |
| `--vit_model` | str | `tiny` | ViT model size: `tiny`, `small` |
| `--decoder_size` | str | `medium` | Decoder size: `small`, `medium`, `large` |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--loss_threshold` | float | `0.35` | Threshold for mask refinement |
| `--epochs` | int | `4` | Epochs for segmentation training |
| `--gt_ratio` | float | `0.0` | Proportion of ground truth masks (0.0 = fully weakly-supervised) |
| `--use_finecam_only` | flag | - | Use raw FineCAMs without decoder refinement |
| `--train_full_gt` | flag | - | Also train a fully-supervised baseline |

#### Recommended Configurations

| ViT Model | Decoder | Loss Threshold | Epochs | FineCAM Only |
|-----------|---------|----------------|--------|--------------|
| tiny | medium | 0.35 | 4 | No |
| tiny | large | 0.25 | 10 | No |
| small | medium | 0.05 | 10 | Yes |
| small | large | 0.061 | 10 | No |



### Inference / Evaluation

To evaluate a trained model without retraining:

```bash
python main.py --vit_model tiny --decoder_size medium
```

This will:
- Load the trained segmentation model
- Evaluate on the test set
- Report IoU, Dice, Accuracy, Precision, and Recall metrics



## Results

### Weakly-Supervised Segmentation Performance

Segmentation results (%) for different ViT encoders and decoder sizes using the best-performing binarization threshold θ. The fully supervised model is provided as a reference.

| ViT Model | Decoder Size | θ | IoU | Dice | Accuracy | Precision | Recall |
|-----------|--------------|------|-------|-------|----------|-----------|--------|
| DeiT-Tiny | Small | 0.25 | 75.00 | 85.71 | **79.35** | 83.50 | 88.05 |
| DeiT-Tiny | Medium | 0.25 | 74.00 | 85.06 | 78.99 | **85.10** | 85.02 |
| DeiT-Tiny | Large | 0.25 | 72.42 | 84.01 | 78.45 | 87.88 | 80.46 |
| ViT-Small | Small | 0.15 | **75.26** | **85.89** | 77.84 | 77.82 | 95.82 |
| ViT-Small | Medium | 0.15 | 75.06 | 85.75 | 77.37 | 76.95 | 96.83 |
| ViT-Small | Large | 0.15 | 75.00 | 85.71 | 77.17 | 76.55 | **97.37** |
| **Fully Supervised** | - | - | **93.16** | **96.46** | **95.05** | **97.16** | **95.80** |

### Mixed Supervision Results

Segmentation results (%) under mixed supervision with varied proportions of ground truth masks.

| ViT Model | Decoder Size | GT % | IoU | Dice | Accuracy | Precision | Recall |
|-----------|--------------|------|-------|-------|----------|-----------|--------|
| DeiT-Tiny | Small | 10% | 78.29 | 87.83 | 82.45 | 85.78 | 89.97 |
| DeiT-Tiny | Small | 20% | 78.55 | 87.99 | 82.64 | 58.72 | 90.38 |
| DeiT-Tiny | Small | 30% | 82.08 | 90.16 | 85.88 | 88.49 | 91.89 |
| DeiT-Tiny | Small | 40% | 85.47 | 92.17 | 88.88 | 91.35 | 93.00 |
| ViT-Small | Small | 10% | 77.50 | 87.32 | 80.41 | 80.14 | 95.92 |
| ViT-Small | Small | 20% | 77.50 | 87.32 | 80.40 | 80.14 | 95.92 |
| ViT-Small | Small | 30% | 80.72 | 89.33 | 83.85 | 83.41 | 96.15 |
| ViT-Small | Small | 40% | 84.66 | 91.69 | 87.75 | 87.68 | 96.09 |



## Citation

If you use this code in your research, please cite:

```bibtex
@misc{weaktr-refinery,
  author = {Amit John},
  title = {WeakTR-Refinery: Weakly-Supervised Segmentation with Vision Transformers},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/johnamit/weaktr-refinery}
}
```



## License

This project is released under the MIT License.

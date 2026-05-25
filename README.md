# Multimodal Medical Image Fusion (CT + MRI)

> **Goal**: Fuse paired CT and MRI brain scans into a single high-quality grayscale image that preserves the structural detail of CT (bones, calcifications) and the soft-tissue contrast of MRI.

---

## Overview

This repository contains the full pipeline — preprocessing, model training, and inference — for a **MAE-based CT-MRI image fusion** system. The core architecture uses two frozen MAE ViT-Small encoders (one per modality), an Adaptive Pooling Fusion Module (APFM) at every encoder layer, and a trainable MAE ViT-Small decoder with symmetric skip connections.

Multiple experimental variants of the fusion loss and APFM design are provided, each in its own script inside `MAE-diff-main/`.

---

## Repository Structure

```
Multimodal_Image_Fusion/
│
├── preprocessing/                  # Data preparation
│   ├── preprocessing.py            # NIfTI → PNG slices (train/val/test split)
│   ├── train/                      # {patient_id}/CT/*.png + MR/*.png
│   ├── val/
│   └── test/
│
├── MAE-diff-main/                  # Main training & inference code
│   ├── mae_vit_small_patch16_ct/   # Pre-trained MAE encoder weights (CT)
│   │   └── encoder.pth
│   ├── mae_vit_small_patch16_mri/  # Pre-trained MAE encoder weights (MRI)
│   │   └── encoder.pth
│   │
│   ├── exp1_fusion_apfm.py             # Exp 1 — Dual-gate APFM (v2)
│   ├── exp1_fusion_dynloss.py          # Exp 2 — Dynamic modality loss
│   ├── exp1_fusion_dynloss_apfm.py     # Exp 3 — Dynamic loss + dual-gate APFM
│   ├── exp1_fusion_dynloss_apfm_2.py   # Exp 4 — Variant of Exp 3
│   ├── exp1_fusion_dynloss_texture.py  # Exp 5 — Dynamic loss + local texture loss
│   ├── exp1_updatedLoss3_fusion.py     # Exp 6 — Loss v3 (full)
│   ├── exp1_updatedLoss3_fusion_ablation.py  # Exp 7 — Loss v3 with ablation support
│   └── final_output_process.py         # Post-processing utilities
│
├── baseline_implementation/        # Baseline autoencoder comparison
│   ├── evaluate.py
│   └── metrics.csv
│
├── autoencoder_unsupervised_loss/  # Autoencoder baseline experiment
│   ├── implementation.ipynb
│   └── result.txt                  # Baseline results
│
└── SAM-Med2D_/                     # SAM-Med2D segmentation utilities
    ├── DataLoader.py
    ├── extract_features.py
    ├── extract_features_mri.py
    └── ...
```

---

## Architecture

```
CT  image (1×256×256)  ──► Frozen MAE ViT-Small Encoder (12 blocks)
                                    │  (per-block features)
                                    ▼
                              APFM × 12   ◄── MRI features (per block)
                                    │
                              Fused tokens (last layer)
                                    │
                        Trainable MAE ViT-Small Decoder
                          (8 blocks + symmetric skips)
                                    │
                            Fused image (1×256×256)

MRI image (1×256×256)  ──► Frozen MAE ViT-Small Encoder (12 blocks)
```

**Key components:**

| Component | Details |
|---|---|
| **Encoders** | MAE ViT-Small, `embed_dim=384`, `depth=12`, `patch_size=16`, `img_size=256`, frozen after loading |
| **APFM** | Adaptive Pooling Fusion Module — channel + spatial gates per encoder layer |
| **Decoder** | MAE ViT-Small, `decoder_embed_dim=512`, `depth=8`, `num_heads=16`, with skip connections |
| **Input** | Grayscale PNG, resized to 256×256, normalized to [0, 1] |

---

## Experimental Variants

| Script | APFM | Loss Components |
|---|---|---|
| `exp1_fusion_apfm.py` | Dual-gate (independent CT + MRI gates, residual) | L_int + L_ssim + L_msssim + L_grad + L_modal |
| `exp1_fusion_dynloss.py` | Single-gate (zero-sum) | L_int + L_ssim + L_msssim + L_grad + L_modal (dynamic) |
| `exp1_fusion_dynloss_apfm.py` | Dual-gate | Same as dynloss + gate balance penalty |
| `exp1_fusion_dynloss_apfm_2.py` | Dual-gate (variant) | Same |
| `exp1_fusion_dynloss_texture.py` | Single-gate | + **L_texture** (local std map, 5×5 kernel) |
| `exp1_updatedLoss3_fusion.py` | Single-gate | L_int + L_ssim + L_msssim + L_grad + L_texture + L_freq + L_percep + L_modal |
| `exp1_updatedLoss3_fusion_ablation.py` | Single-gate | Same as v3 but any subset can be disabled via `--loss_terms` |

---

## Setup

### Requirements

```bash
pip install torch torchvision timm pytorch-msssim piq pillow numpy nibabel opencv-python scikit-learn tqdm
```

> Python ≥ 3.9, PyTorch ≥ 2.0 recommended.

### Pre-trained Encoder Weights

Place the MAE ViT-Small checkpoints at:
```
MAE-diff-main/mae_vit_small_patch16_ct/encoder.pth
MAE-diff-main/mae_vit_small_patch16_mri/encoder.pth
```

These are domain-specific encoders pre-trained separately on CT and MRI data.

---

## Data Preparation

Raw data: paired CT + MRI NIfTI volumes with brain masks, one folder per patient.

```bash
python preprocessing/preprocessing.py
```

This script:
1. Loads CT, MRI, and mask volumes per patient.
2. Normalises and masks each modality independently.
3. Filters blank/low-gradient slices.
4. Resizes accepted slices to **256×256** and saves as PNG.
5. Splits patients into **train (80%) / val (10%) / test (10%)**.

**Output layout:**
```
preprocessing/
├── train/{patient_id}/CT/*.png
│                     /MR/*.png
├── val/{patient_id}/CT/*.png
│                   /MR/*.png
└── test/{patient_id}/CT/*.png
                     /MR/*.png
```

> ⚠️ Edit `DATA_ROOT` and `OUTPUT_ROOT` at the top of `preprocessing.py` to match your paths.

---

## Training

All experiment scripts share the same CLI. Run from the project root:

```bash
# Train only (uses defaults for all paths)
python MAE-diff-main/exp1_fusion_apfm.py --mode train

# Train + immediately run inference on the test set
python MAE-diff-main/exp1_fusion_apfm.py --mode both

# Resume from a checkpoint
python MAE-diff-main/exp1_fusion_apfm.py --mode train --resume MAE-diff-main/fusion_output_apfm/checkpoints/latest.pth
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | `train`, `infer`, or `both` |
| `--ct_ckpt` | `MAE-diff-main/mae_vit_small_patch16_ct/encoder.pth` | CT encoder weights |
| `--mri_ckpt` | `MAE-diff-main/mae_vit_small_patch16_mri/encoder.pth` | MRI encoder weights |
| `--ct_root` | `CT_dataset` | Training CT dataset root |
| `--mri_root` | `MRI_dataset` | Training MRI dataset root |
| `--test_dir` | `preprocessing/test` | Inference test set |
| `--out_dir` | `MAE-diff-main/fusion_output_<variant>` | All outputs written here |
| `--epochs` | `100` | Max training epochs (early stopping at patience=10) |
| `--batch` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate (AdamW + cosine annealing) |
| `--num_workers` | `4` | DataLoader workers |

### Ablation Study (Loss v3 only)

```bash
# Remove texture and perceptual terms
python MAE-diff-main/exp1_updatedLoss3_fusion_ablation.py --mode both \
    --loss_terms int ssim msssim grad freq modal
```

Valid terms: `int`, `ssim`, `msssim`, `grad`, `texture`, `freq`, `percep`, `modal`

---

## Inference

```bash
python MAE-diff-main/exp1_fusion_apfm.py --mode infer \
    --ckpt MAE-diff-main/fusion_output_apfm/checkpoints/best_model.pth \
    --test_dir preprocessing/test
```

**Output layout** (all under `--out_dir/inference/`):
```
inference/
├── fused/{patient_id}/*.png    ← fused output images
├── grids/{patient_id}/*.png    ← side-by-side: CT | MRI | Fused + metrics
├── metrics.csv                 ← per-image metrics
└── summary.txt                 ← aggregated metrics summary
```

---

## Metrics

All metrics are computed against both individual modalities and their mean `(CT + MRI) / 2`:

| Metric | Description |
|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) — higher is better |
| **SSIM** | Structural Similarity Index — higher is better |
| **MS-SSIM** | Multi-Scale SSIM — higher is better |
| **VIF** | Visual Information Fidelity — higher is better |
| **MSE / MAE** | Pixel-level error — lower is better |
| **SD** | Standard deviation of output (texture richness) |
| **EN** | Shannon entropy (information content) |
| **OVERALL** | Composite score = `(PSNR/50 + SSIM + MS-SSIM + VIF) / 4` |

### Baseline Results (Autoencoder)

```
SSIM    : 0.8303 ± 0.0200
MS-SSIM : 0.9152 ± 0.0226
PSNR    : 21.12  ± 1.48 dB
VIF     : 0.5784 ± 0.0724
SD      : 0.0973 ± 0.0203
EN      : 5.14   ± 0.59
```

---

## Loss Functions

### FusionLoss (all variants share a common core)

| Term | Formula | Purpose |
|---|---|---|
| **L_int** | MAE vs mean(CT, MRI) | Pixel-level fidelity |
| **L_ssim** | 1 − SSIM | Structural similarity |
| **L_msssim** | 1 − MS-SSIM | Multi-scale structure |
| **L_grad** | Sobel gradient MAE (max of CT/MRI) + CT-edge bonus | Preserve sharp edges |
| **L_modal** | Dynamic per-modality MAE (variance-weighted) | Balanced CT/MRI contribution |
| **L_texture** | Local std-map MAE (5×5 kernel) | Prevent blurry outputs |
| **L_freq** | FFT magnitude MAE | Frequency-domain fidelity |
| **L_percep** | VGG-16 feature MAE | Perceptual quality |

---

## Notes

- All scripts expect the **current working directory to be the project root** (`Multimodal_Image_Fusion/`) when using default paths.
- The encoders are **always frozen**; only APFM layers and the decoder are trained (~trainable params depend on variant).
- A **gate balance penalty** (`0.1 × MSE(gate, 0.5)`) is added during training to prevent APFM collapsing to one modality.
- **Early stopping** triggers after 10 epochs without PSNR improvement on the validation set.

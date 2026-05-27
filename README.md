# Multi-Modal Medical Image Fusion using Generative Models

## CS671 Course Project – IIT Mandi

This project focuses on **unsupervised multi-modal medical image fusion** for combining **CT and MRI images** using transformer-based generative architectures. The framework explores multiple fusion strategies including baseline autoencoders, MAE-ViT encoders, Adaptive Patch Fusion Modules (APFM), dynamic loss tuning, and SAM-Med2D fine-tuning.

The objective is to generate fused medical images that preserve both structural and functional information from multiple imaging modalities while improving visual quality and detail retention.

---

# Group Details

| Name | Roll Number |
|---|---|
| Ishika Agarwal | B23399 |
| Aditi Gupta | B23307 |
| Zainab | B23236 |
| Mehak | B23157 |
| Siddhi Pogakwar | B23415 |
| Anamika | B23428 |
| Himanshi | B24310 |

---

# Project Structure

```bash
.
├── autoencoder_unsupervised/
├── baseline_implementation/
├── MAE-diff-main/
├── preprocessing/
├── SAM-Med2D_/
├── .gitignore
```

---

# Folder Descriptions

## 1. autoencoder_unsupervised/

Contains the **basic autoencoder-based implementation** for image fusion using unsupervised reconstruction losses.

### Includes:
- Training checkpoints
- Fused image outputs
- Notebook implementation
- Experimental results

### Key Idea:
A simple encoder-decoder architecture is trained to fuse CT and MRI images without supervised labels.

---

## 2. baseline_implementation/

Contains the baseline inference pipeline performed on the preprocessed dataset using the Mask Diffuser MAE framework.

### Files:
- `evaluate.py` → Evaluation script
- `metrics.csv` → Stores quantitative evaluation metrics

### Purpose:
Provides baseline comparison results for fusion quality evaluation.

---

## 3. MAE-diff-main/

Main implementation folder containing the proposed architecture and experimental variants.

### Components

#### mae_vit_small_patch16_*
Contains pretrained/trained MAE-ViT encoders used for feature extraction.

---

### Experimental Implementations

| File | Description |
|---|---|
| `exp1_fusion_apfm.py` | Original APFM-based fusion implementation |
| `exp1_fusion_dynloss.py` | Dynamic loss tuning implementation |
| `dynloss_apfm.py` | Combined Dual-Gate APFM + Dynamic Loss implementation |
| `final_output_process.py` | Post-processing and final output generation |

---

### APFM Variants

#### Original APFM
Adaptive Patch Fusion Module used for modality-aware feature fusion.

#### Dual-Gate APFM (`apfm2`)
Enhanced APFM variant with dual gating mechanisms for improved feature selection and fusion quality.

---

### Dynamic Loss (`dynloss`)
A tuned adaptive loss function designed to improve:
- Structural preservation
- Texture retention
- Fusion consistency

---

### Combined Model
`dynloss_apfm` integrates:
- Dual-Gate APFM
- Dynamic Loss Optimization

This serves as the final proposed architecture.

---

## 4. preprocessing/

Contains:
- Preprocessing scripts
- Train/Validation/Test datasets

### Files

| File | Purpose |
|---|---|
| `preprocessing.py` | Data preprocessing pipeline |

### Operations Performed
- CT/MRI normalization
- Image resizing
- Patch preparation
- Dataset organization

---

## 5. SAM-Med2D_/

Contains experiments related to fine-tuning SAM-Med2D for medical image fusion tasks.

### Goal
To explore segmentation-aware feature representations for improved CT-MRI fusion.

---

# Methodology

## Pipeline Overview

1. Preprocess CT and MRI datasets
2. Extract modality-specific features using MAE-ViT encoders
3. Fuse features using APFM/Dual-Gate APFM
4. Optimize using dynamic unsupervised losses
5. Reconstruct fused images using decoder architecture
6. Evaluate fusion quality using quantitative metrics

---

# Features

- Unsupervised image fusion
- Vision Transformer (ViT) based feature extraction
- Adaptive Patch Fusion Module (APFM)
- Dual-Gate APFM architecture
- Dynamic loss optimization
- Baseline and ablation studies
- SAM-Med2D fine-tuning experiments
- Quantitative metric evaluation

---

# Evaluation Metrics

The generated fused images are evaluated using:
- Entropy (EN)
- Structural Similarity (SSIM)
- Multi-Scale Structural Similarity (MSSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- Standard Deviation (SD)

Metrics are stored in:

```bash
baseline_implementation/metrics.csv
```

---

# Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Common libraries used:
- Python
- PyTorch
- NumPy
- OpenCV
- torchvision
- matplotlib
- pandas

---

# Running the Project

## Preprocessing

```bash
python preprocessing/preprocessing.py
```

---

## Run Baseline Evaluation

```bash
python baseline_implementation/evaluate.py
```

---

## Run Fusion Experiments

Example:

```bash
python MAE-diff-main/exp1_fusion_apfm.py
```

or

```bash
python MAE-diff-main/dynloss_apfm.py
```

---

# Ablation Studies

The project includes experiments comparing:
- Original APFM
- Dual-Gate APFM
- Static vs Dynamic Loss
- Different fusion strategies
- Encoder configurations

These studies help analyze the contribution of each module to overall fusion performance.

---

# Results

The proposed framework successfully improves:
- Structural detail preservation
- Modality information retention
- Fusion consistency
- Visual quality of fused medical images

---

# Future Work

- Real-time medical image fusion
- Multi-modal transformer scaling
- Clinical evaluation with radiologists
- Integration with downstream diagnostic systems
- Improved segmentation-guided fusion

---

# Course Information

**Course:** CS671 – Deep Learning  
**Institute:** IIT Mandi

---

# Hashtags

`#MedicalImageFusion` `#DeepLearning` `#VisionTransformer` `#MAE` `#ComputerVision` `#MedicalImaging` `#IITMandi` `#CS671`

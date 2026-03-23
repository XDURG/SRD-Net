# SRD-Net: Multimodal Ultrasound Learning for Postoperative Scar-vs-Recurrence Assessment

## Overview
SRD-Net is a publication-ready PyTorch implementation of a multimodal deep learning framework for differentiating postoperative scar tissue from recurrence using paired B-mode ultrasound and color Doppler flow imaging (CDFI) acquired in the same imaging plane. The repository implements a dual-encoder Swin Transformer architecture with multi-scale multimodal fusion, a boundary-aware U-shaped decoder for lesion segmentation, and a mask-guided classifier for recurrence prediction.

## Key Methodological Contributions
- **Dual-modality feature extraction** using two independent Swin Transformer encoders for B-mode and CDFI images.
- **Hierarchical multimodal fusion** at four scales using feature concatenation followed by learnable fusion layers.
- **Boundary-aware lesion segmentation** with a U-shaped decoder, skip connections, and edge-enhanced attention refinement.
- **Mask-guided recurrence classification** that pools features only inside the predicted lesion region.
- **Multi-task optimization** combining Dice loss, focal Tversky loss, and asymmetric focal loss.

## Repository Structure
```text
SRD-Net/
├── configs/
│   └── config.yaml
├── data/
│   ├── README.md
│   └── manifests/
│       ├── train_manifest.csv
│       ├── val_manifest.csv
│       └── test_manifest.csv
├── datasets/
│   ├── __init__.py
│   └── paired_ultrasound.py
├── models/
│   ├── __init__.py
│   ├── losses.py
│   └── srd_net.py
├── src/
│   ├── __init__.py
│   ├── inference.py
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── io.py
│   ├── metrics.py
│   ├── reproducibility.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## Expected Dataset Organization
This repository intentionally excludes binary imaging assets and private patient data. Replace the provided text-only manifest templates with your own dataset manifests during experimentation.

```text
data/
├── manifests/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   └── test_manifest.csv
└── your_dataset/
    └── patients/
        └── case_xxxx/
            ├── bmode.png
            ├── cdfi.png
            └── mask.png
```

Each manifest must contain the following columns:

| column | description |
| --- | --- |
| `sample_id` | Unique identifier for one paired examination. |
| `bmode_path` | Relative or absolute path to the B-mode ultrasound image. |
| `cdfi_path` | Relative or absolute path to the CDFI image from the same plane. |
| `mask_path` | Relative or absolute path to the binary lesion annotation mask. |
| `label` | Binary recurrence label (`0` = scar, `1` = recurrence). |

## Model Summary
### 1. Dual Swin Transformer Encoders
The model instantiates two separate Swin Transformer backbones, one per modality. Features from the four hierarchical stages are extracted independently for B-mode and CDFI streams.

### 2. Multi-Scale Multimodal Fusion
At each Swin stage, modality-specific features are concatenated and passed through fusion layers composed of pointwise and spatial convolutions.

### 3. Boundary-Aware Segmentation Decoder
The fused features are processed through a U-shaped decoder with skip connections. Each decoder block applies a boundary-aware attention module that emphasizes edge-sensitive activations before generating the lesion mask.

### 4. Mask-Guided Classification Head
The final decoder feature map is multiplied by the predicted segmentation probability map. Region-restricted average pooling yields the lesion descriptor, which is passed to an MLP to estimate recurrence probability.

## Training Objective
The training loss is defined as:

- **Segmentation loss**: Dice loss + focal Tversky loss
- **Classification loss**: asymmetric focal loss
- **Total loss**: weighted sum of segmentation and classification terms

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility
A fixed random seed is specified in `configs/config.yaml` and enforced through Python, NumPy, and PyTorch seeding utilities.

## Training
```bash
python -m src.train --config configs/config.yaml
```

## Inference
```bash
python -m src.inference \
  --config configs/config.yaml \
  --checkpoint outputs/srd_net/checkpoints/best_model.pt
```

Inference exports:
- lesion probability masks as PNG files
- recurrence probabilities in `predictions.csv`

## Notes for Manuscript Integration
- Replace the shipped manifest templates with institutional dataset manifests that point to your local or secured image storage.
- Update `configs/config.yaml` with training hyperparameters specific to the experimental protocol reported in the paper.
- Consider reporting Dice score, accuracy, sensitivity, specificity, and precision, all of which are directly supported by the training pipeline.

## Citation
If you use this code in academic work, please cite your manuscript describing SRD-Net.

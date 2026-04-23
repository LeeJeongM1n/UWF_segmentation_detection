# UWF Segmentation & Detection Project

This repository contains a pipeline for preprocessing and analyzing
Ultra-Widefield (UWF) retinal images.

## Objective

1.  Remove imaging artifacts from UWF retinal images\
2.  Extract valid retinal regions via segmentation\
3.  Localize optic disc and macula using heatmap regression

------------------------------------------------------------------------

## Overview

UWF retinal images often include: - Peripheral artifacts\
- Uneven illumination\
- Non-retinal background

This project addresses these challenges using a segmentation-based
pipeline.

------------------------------------------------------------------------

## Method

### 1. Segmentation (Preprocessing)

-   Model: U-Net (segmentation_models_pytorch)
-   Removes non-retinal regions and artifacts
-   Extracts valid retinal region

### 2. Heatmap-based Localization

-   Model outputs 2-channel heatmap:
    -   Channel 1: optic disc
    -   Channel 2: macula
-   Sigmoid activation applied to obtain probability maps
-   Test-Time Augmentation (TTA):
    -   Original / Horizontal flip / Vertical flip
-   Final prediction obtained by averaging TTA outputs

### 3. Post-processing

-   Thresholding on heatmaps
-   Connected component analysis
-   Blob peak detection
-   Selection of largest valid region
-   Extraction of center coordinates (disc / macula)

------------------------------------------------------------------------

## Pipeline

    Input UWF Image
            ↓
    Segmentation (Artifact Removal)
            ↓
    2-channel Heatmap Prediction (Disc / Macula)
            ↓
    TTA Averaging
            ↓
    Connected Component + Peak Detection
            ↓
    Disc & Macula Center Localization

------------------------------------------------------------------------

## Repository Structure (Current)

> This repository is under active organization.

Includes: - Segmentation model training / inference scripts
- Heatmap regression inference scripts
- Data preprocessing utilities
- Evaluation scripts (Dice / IoU)
- Visualization tools (overlay, ETDRS, zones)
- Experimental scripts

------------------------------------------------------------------------

## Key Script

### inference_tta_mean.py

-   Performs inference using U-Net model
-   Generates disc and macula heatmaps
-   Applies flip-based TTA (horizontal / vertical)
-   Uses blob peak + connected components for localization
-   Outputs:
    -   Heatmap images
    -   Optional overlays
    -   Center coordinates (CSV)
    -   Evaluation metrics (Dice / IoU, if GT provided)

------------------------------------------------------------------------

## Getting Started

### Clone

    git clone https://github.com/LeeJeongM1n/UWF_segmentation_detection.git
    cd UWF_segmentation_detection

### Environment

    conda create -n uwf python=3.9
    conda activate uwf
    pip install -r requirements.txt

------------------------------------------------------------------------

## Usage (Example)

    python inference_tta_mean.py \
        --input_dir ./images \
        --ckpt ./model.pth \
        --out_dir ./results \
        --img_size 512 \
        --tta

Optional:

    --gt_dir ./gt_heatmaps

------------------------------------------------------------------------

## Output

-   Heatmap (disc / macula)
-   Binary mask (largest connected component)
-   Center coordinates (CSV)
-   Overlay visualization
-   Optional:
    -   Dice / IoU metrics
    -   ETDRS grid visualization
    -   Zone-based visualization

------------------------------------------------------------------------

## Notes

-   Localization is performed via segmentation heatmaps
-   Designed for medical image analysis robustness

------------------------------------------------------------------------

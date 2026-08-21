# UWF Retinal Segmentation & Heatmap-based Landmark Localization

This repository provides training and inference pipelines for **Ultra-Widefield (UWF) retinal image analysis**.

The pipeline combines:

- Retinal region segmentation
- Optic disc and fovea localization using heatmap regression
- Flip-based Test-Time Augmentation (TTA)
- Connected-component-based landmark localization
- Automated quality-control (QC) analysis
- CSV-based result export

The current implementation uses **segmentation and heatmap regression** for anatomical landmark localization.

---

## Overview

Ultra-widefield retinal images may contain non-retinal background, peripheral imaging artifacts, uneven illumination, retinal distortion, and substantial anatomical variation.

In particular, reliable localization of the fovea can be difficult using conventional bounding-box-based object detection.

The current pipeline therefore separates the problem into two components:

1. **Retinal ROI segmentation**
2. **Optic disc and fovea heatmap regression**

The predicted heatmaps are subsequently refined using Test-Time Augmentation and connected-component analysis to determine the final landmark coordinates.

---

## Pipeline

```text
Input UWF Image
        │
        ├───────────────────────────────┐
        │                               │
        ▼                               ▼
Retinal ROI Segmentation       Landmark Heatmap Regression
DeepLabV3+                     U-Net
ResNet50 encoder               ResNet50 encoder
        │                               │
        │                       ┌───────┴───────┐
        │                       │               │
        │                 Optic Disc         Fovea
        │                   Heatmap          Heatmap
        │                       │               │
        │                       └───────┬───────┘
        │                               │
        └───────────────┬───────────────┘
                        ▼
                 Flip-based TTA
            Original / HFlip / VFlip
                        │
                        ▼
                 Heatmap Averaging
                        │
                        ▼
            Connected-component Analysis
                        │
                        ▼
              Disc / Fovea Localization
                        │
                        ▼
                 QC Measurements
                        │
                        ▼
                CSV / Visualization
```

---

## Repository Structure

```text
UWF_segmentation_detection/
│
├── inference/
│   └── uwf_inference.py
│
├── training/
│   ├── heatmap/
│   │   ├── create_landmark_heatmaps.py
│   │   └── train_landmark_model.py
│   │
│   └── segmentation/
│       └── train_roi_segmentation.py
│
├── weights/
│   ├── uwf_landmark.pth
│   └── uwf_roi_segmentation.ckpt
│
├── datasets/
├── sample_data/
│
├── run_inference.sh
├── README.md
├── .gitignore
└── .gitattributes
```

---

# Models

## 1. Retinal ROI Segmentation

The retinal segmentation model identifies the valid retinal region in each UWF image.

This step is used to distinguish the retinal field from non-retinal background and peripheral imaging artifacts.

### Architecture

```text
DeepLabV3+
└── ResNet50 encoder
```

The pretrained checkpoint is provided at:

```text
weights/uwf_roi_segmentation.ckpt
```

---

## 2. Optic Disc and Fovea Heatmap Regression

An independent heatmap regression model is used to localize the optic disc and fovea.

### Architecture

```text
U-Net
└── ResNet50 encoder
```

The model receives an RGB retinal image and predicts two continuous heatmaps.

```text
Input
└── RGB UWF image

Output
├── Channel 0: Optic disc heatmap
└── Channel 1: Fovea heatmap
```

The pretrained landmark model is provided at:

```text
weights/uwf_landmark.pth
```

---

# Training

Training code for both retinal segmentation and landmark heatmap regression is included in this repository.

## 1. Landmark Heatmap Generation

Script:

```text
training/heatmap/create_landmark_heatmaps.py
```

This script generates the ground-truth heatmaps used to train the landmark regression model.

Landmark annotations are converted into continuous Gaussian heatmaps for:

```text
Optic disc
Fovea
```

These heatmaps are subsequently used as regression targets.

### Workflow

```text
Landmark Annotation
        │
        ▼
Coordinate Extraction
        │
        ▼
Gaussian Heatmap Generation
        │
        ├── Optic Disc Heatmap
        └── Fovea Heatmap
        │
        ▼
2-channel Training Target
```

---

## 2. Landmark Heatmap Regression Training

Script:

```text
training/heatmap/train_landmark_model.py
```

The landmark model is implemented using `segmentation_models_pytorch`.

### Model configuration

```text
Architecture : U-Net
Encoder      : ResNet50
Input        : 3-channel RGB
Output       : 2-channel heatmap
```

The two output channels correspond to:

```text
Channel 0 → Optic disc
Channel 1 → Fovea
```

The model predicts continuous heatmaps rather than bounding boxes.
Predicted logits are converted to heatmap probabilities using sigmoid activation and optimized against the ground-truth Gaussian heatmaps using mean squared error (MSE).


The checkpoint with the lowest validation loss is saved as the best-performing model.

---

## 3. Retinal ROI Segmentation Training

Script:

```text
training/segmentation/train_roi_segmentation.py
```

This script trains the retinal ROI segmentation model used in the final inference pipeline.

### Model

```text
DeepLabV3+
└── ResNet50 encoder
```

The resulting checkpoint can be used by `uwf_inference.py` to identify the valid retinal region.

---

# Inference

The complete inference pipeline is implemented in:

```text
inference/uwf_inference.py
```

This is the **main inference script** of the repository.

It combines the retinal segmentation model and landmark heatmap regression model into a single inference workflow.

### Main steps

```text
1. Load UWF image

2. Retinal ROI segmentation
   └── DeepLabV3+ / ResNet50

3. Landmark heatmap prediction
   └── U-Net / ResNet50

4. Test-Time Augmentation
   ├── Original
   ├── Horizontal flip
   └── Vertical flip

5. Heatmap averaging

6. Connected-component analysis

7. Optic disc / fovea center extraction

8. Coordinate mapping

9. QC measurements

10. CSV and visualization output
```

---

## Test-Time Augmentation

Landmark inference uses flip-based Test-Time Augmentation.

Predictions are generated from:

```text
Original Image
Horizontal Flip
Vertical Flip
```

The flipped predictions are transformed back to the original orientation and averaged.
This provides a more stable heatmap estimate than relying on a single forward pass.

---

## Landmark Extraction

The averaged heatmaps are post-processed to determine the final landmark coordinates.

The procedure includes:

- Heatmap thresholding
- Connected-component analysis
- Valid component selection
- Landmark center extraction

The final coordinates are mapped back to the corresponding image coordinate system for subsequent QC analysis.

---

# Pretrained Weights

Pretrained weights for both models are included under:

```text
weights/
├── uwf_landmark.pth
└── uwf_roi_segmentation.ckpt
```

The large model files are managed using **Git LFS**.

After cloning the repository, make sure Git LFS is installed and retrieve the model files using:

```bash
git lfs install
git lfs pull
```

---

# Getting Started

## 1. Clone Repository

```bash
git clone https://github.com/LeeJeongM1n/UWF_segmentation_detection.git
cd UWF_segmentation_detection
```

## 2. Download Git LFS Files

```bash
git lfs install
git lfs pull
```

This downloads the pretrained model weights stored through Git LFS.

---

# Running Inference

The recommended entry point is:

```bash
bash run_inference.sh
```

`run_inference.sh` calls:

```text
inference/uwf_inference.py
```

and provides the paths and parameters required for the unified inference pipeline.

Before running inference, configure the paths in `run_inference.sh` according to the local dataset and output directories.

---

# Input

The inference pipeline accepts a **directory containing UWF retinal images** as input.

Users do not need to manually prepare a CSV file. Instead, specify the input image directory using `IMAGE_DIR` in `run_inference.sh`.

For example:

```bash
IMAGE_DIR="/path/to/UWF_images"
```

The script automatically searches the specified directory and its subdirectories for supported image files.

Supported image formats include:

```text
.jpg
.jpeg
.png
.bmp
.tif
.tiff
```

For example, the input directory may have the following structure:

```text
UWF_images/
├── image_001.jpg
├── image_002.jpg
├── image_003.jpg
└── subfolder/
    ├── image_004.jpg
    └── image_005.png
```

When `run_inference.sh` is executed, all supported images found under `IMAGE_DIR` are automatically collected and processed.

```bash
bash run_inference.sh
```

Internally, `run_inference.sh` generates an image manifest for compatibility with the inference pipeline and passes it to `inference/uwf_inference.py`. Therefore, users only need to specify the image directory and do not need to create or manage the CSV file manually.

---
# Output

The inference pipeline generates image-level QC results and landmark localization outputs.

Depending on the configured options, outputs may include:

```text
Output Directory
│
├── QC result CSV
├── QC summary CSV
└── QC visualizations
```

The output information can include:

- Retinal ROI information
- Optic disc coordinates
- Fovea coordinates
- Landmark confidence
- TTA consistency
- QC measurements
- Image-level QC results

## Example QC Visualization

An example of the QC visualization generated by the inference pipeline is shown below.

<p align="center">
  <img src="assets/example_qc_output.jpg" width="850">
</p>

The visualization provides an image-level overview of the predicted retinal ROI and anatomical landmark localization results.

---


---

# Notes

- The current repository focuses on **retinal segmentation and heatmap-based landmark localization**.
- Previous YOLO-based landmark detection code is not part of the current pipeline.
- Retinal ROI segmentation and landmark localization use separate models.
- Landmark localization is performed using continuous 2-channel heatmap regression.
- `uwf_inference.py` is the unified inference script and the recommended entry point for inference.
- `run_inference.sh` provides a convenient launcher for the complete inference pipeline.
- Training code for both ROI segmentation and landmark heatmap regression is included.
- Pretrained model weights are distributed using Git LFS.

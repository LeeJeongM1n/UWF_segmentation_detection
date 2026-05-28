# UWF Segmentation & Heatmap-based Detection Project

This repository contains preprocessing, inference, evaluation, and visualization pipelines for Ultra-Widefield (UWF) retinal image analysis.

The project focuses on robust optic disc and macula localization using segmentation-guided heatmap regression.

---

## Objective

1. Remove imaging artifacts from UWF retinal images
2. Extract valid retinal regions via segmentation
3. Localize optic disc and macula using heatmap regression
4. Improve localization robustness compared to conventional YOLO object detection approaches

---

## Motivation

Conventional YOLO-based object detection showed limited robustness for macula localization in UWF retinal images.

In particular, YOLO-based approaches frequently produced unstable or inaccurate macula detection results due to:

* Ultra-wide retinal distortion
* Peripheral artifacts
* Large anatomical variation
* Low-contrast macular regions

To address these limitations, this repository adopts:

* Segmentation-guided preprocessing
* Heatmap-based localization
* Connected-component-based center extraction

instead of direct bounding-box regression.


---

## Overview

UWF retinal images often include:

* Peripheral artifacts
* Uneven illumination
* Non-retinal background
* Retinal distortion near image boundaries

This project addresses these challenges using a segmentation-guided pipeline.

---

## Method

### 1. Retinal Segmentation (Preprocessing)

* Model: U-Net (`segmentation_models_pytorch`)
* Removes non-retinal regions and artifacts
* Extracts valid retinal ROI region
* Used to constrain downstream localization

### 2. Heatmap-based Localization

* Model outputs 2-channel heatmap:

  * Channel 1: optic disc
  * Channel 2: macula
* Sigmoid activation applied to obtain probability maps
* Test-Time Augmentation (TTA):

  * Original
  * Horizontal flip
  * Vertical flip
* Final prediction obtained by averaging TTA outputs

### 3. Post-processing

* Thresholding on heatmaps
* Connected component analysis
* Blob peak detection
* Selection of largest valid region
* Extraction of center coordinates (disc / macula)

---

## Heatmap Generation

Ground-truth regression heatmaps used for detection inference were generated from YOLO-format annotation files.

The original YOLO annotation pipeline is organized under:

```text
YOLOver/
```

The YOLO label annotations (`.txt`) were converted into Gaussian-based regression heatmaps for heatmap localization experiments.

This repository represents a transition from:

```text
YOLO object detection
        ↓
Heatmap-based localization
```

to improve anatomical localization robustness in UWF retinal images.

---

## Pipeline

```text
Input UWF Image
        ↓
Retinal Segmentation
        ↓
Retinal ROI Extraction
        ↓
2-channel Heatmap Prediction
        ↓
TTA Averaging
        ↓
Connected Component + Blob Peak Detection
        ↓
Disc & Macula Center Localization
```

---

## Repository Structure

```text
UWF_segmentation_detection/

├── cropCircle/
│   └── ROI alignment & circular crop utilities
│
├── datasets/
│   └── Dataset preparation utilities
│
├── detection_inference/
│   └── Heatmap-based localization inference
│
├── inference/
│   └── Main inference pipelines
│
├── postprocess/
│   └── Connected component & localization refinement
│
├── sample_data/
│   └── Example images and outputs
│
├── weights/
│   └── Pretrained segmentation & detection weights
│
├── YOLOver/
│   ├── createHeatmap/
│   │   └── YOLO label → regression heatmap generation
│   ├── inference/
│   │   └── Previous YOLO-based detection experiments
│   ├── Data_preprocess.py
│   └── YOLO_datasetSplit.py
│
├── calculateZoneArea.py
├── disc_macula_heatmap_detection.py
├── inference_tta_mean.py
└── README.md
```

---

## Key Script

### inference_tta_mean.py

> This repository currently focuses on inference and evaluation pipelines using pretrained segmentation and heatmap regression models.

* Performs inference using pretrained U-Net models
* Generates disc and macula heatmaps
* Applies flip-based TTA (horizontal / vertical)
* Uses blob peak + connected components for localization

Outputs:

* Heatmap images
* Binary masks
* Overlay visualization
* Center coordinate CSV
* Dice / IoU metrics (optional)

---

## Getting Started

### Clone

```bash
git clone https://github.com/LeeJeongM1n/UWF_segmentation_detection.git
cd UWF_segmentation_detection
```

### Environment

```bash
conda create -n uwf python=3.9
conda activate uwf
pip install -r requirements.txt
```

---

## Pretrained Weights

Pretrained [weights](https://drive.google.com/drive/folders/1NhLqfVaqU6NPpZBci2CtOenpAWpvqO_J?usp=drive_link) for retinal segmentation and heatmap-based localization are available via Google Drive.

After downloading, place the files under:

```text
weights/
```

---

## Usage Example

```bash
python inference_tta_mean.py \
    --input_dir ./images \
    --ckpt ./weights/model.pth \
    --out_dir ./results \
    --img_size 512 \
    --tta
```

Optional:

```bash
--gt_dir ./gt_heatmaps
```

---

## Output

* Disc / macula heatmaps
* Binary masks
* Overlay visualization
* Center coordinate CSV

Optional outputs:

* Dice / IoU metrics
* ETDRS visualization
* Zone-based visualization

---

## Notes

* Localization is performed using segmentation-guided heatmap regression
* This repository emphasizes inference and evaluation pipelines
* Training pipelines are not included in the current public release
* Designed for robust medical image analysis in UWF retinal images

---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMP U-Net heatmap regression inference (+ optional metrics)

- Load trained checkpoint
- Run inference on input images
- Save:
  1) predicted heatmap (uint8 png)
  2) overlay image (input + heatmap)
  3) binary heatmap (thresholded)

If --gt_dir is provided:
  - Match GT heatmap by stem
  - Binarize GT and pred by threshold
  - Compute per-image + overall:
      Dice / IoU / Accuracy / Precision / Recall
  - Save metrics.csv
"""

import argparse
import os
from pathlib import Path
import glob
import csv

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# -------------------------
# Constants
# -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# -------------------------
# Utils
# -------------------------
def denorm_img(img_tensor):
    """ [3,H,W] tensor -> uint8 RGB """
    img = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def heatmap_to_color(hm01):
    hm_u8 = (np.clip(hm01, 0, 1) * 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)


def overlay(rgb, heat_rgb, alpha=0.45):
    return np.clip(
        (1 - alpha) * rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32),
        0, 255
    ).astype(np.uint8)


def load_heatmap_01(path: str) -> np.ndarray:
    """
    Load GT heatmap (image). Returns float32 [H,W] in [0,1].
    - If GT is grayscale 0~255, normalize to 0~1.
    """
    hm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if hm is None:
        raise FileNotFoundError(f"Failed to read GT heatmap: {path}")
    hm = hm.astype(np.float32)
    if hm.max() > 1.0:
        hm = hm / 255.0
    return np.clip(hm, 0.0, 1.0)


def compute_confusion(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """
    pred_bin, gt_bin: uint8/bool arrays of same shape, values {0,1}
    returns TP, FP, TN, FN as floats
    """
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    return float(tp), float(fp), float(tn), float(fn)


def metrics_from_confusion(tp, fp, tn, fn, eps=1e-7):
    """
    Returns Dice, IoU, Acc, Precision, Recall
    """
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)
    acc  = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    return dice, iou, acc, prec, rec


def list_images(input_dir: Path):
    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(str(input_dir / f"*{ext}")))
    return sorted(img_paths)


def build_stem_map(dir_path: Path):
    """
    Map stem -> filepath for files in dir_path (all IMG_EXTS).
    """
    m = {}
    for ext in IMG_EXTS:
        for p in glob.glob(str(dir_path / f"*{ext}")):
            p = Path(p)
            m.setdefault(p.stem, str(p))
    return m


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="directory containing input images")
    parser.add_argument("--gt_dir", default=None, type=str,
                        help="(optional) directory containing GT heatmaps/masks matched by filename stem")
    parser.add_argument("--ckpt", default="/mnt/richul_FM/YOLO/MACULA/best_model.pth", type=str,
                        help="trained model checkpoint (.pth)")
    parser.add_argument("--out_dir", default="./infer_results", type=str)

    parser.add_argument("--test_case", type=str, required=True, help="prefix for best checkpoint filename")
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--encoder", default="efficientnet-b0", type=str)
    parser.add_argument("--threshold", default=0.1, type=float)
    parser.add_argument("--alpha", default=0.45, type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) / args.test_case
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_dir = out_dir / "heatmap"
    overlay_dir = out_dir / "overlay"
    binary_dir  = out_dir / "binary"
    heatmap_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    binary_dir.mkdir(exist_ok=True)

    # -------------------------
    # Model
    # -------------------------
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[Load] checkpoint loaded: {args.ckpt}")

    # -------------------------
    # Transform
    # -------------------------
    transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    img_paths = list_images(input_dir)
    print(f"[Infer] {len(img_paths)} images found in {input_dir}")

    use_gt = args.gt_dir is not None
    gt_map = None
    if use_gt:
        gt_dir = Path(args.gt_dir)
        if not gt_dir.exists():
            raise FileNotFoundError(f"--gt_dir not found: {gt_dir}")
        gt_map = build_stem_map(gt_dir)
        print(f"[GT] {len(gt_map)} GT files indexed from {gt_dir}")

    # Metrics accumulation
    rows = []
    sum_tp = sum_fp = sum_tn = sum_fn = 0.0
    counted = 0
    missing_gt = 0

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Inference"):
            img_path = str(img_path)
            name = Path(img_path).stem

            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            aug = transform(image=rgb)
            img_t = aug["image"].unsqueeze(0).to(device)

            pred = torch.sigmoid(model(img_t))[0, 0].detach().cpu().numpy()  # [H,W], 0~1

            # save predicted heatmap (uint8)
            heat_u8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(heatmap_dir / f"{name}_heatmap.png"), heat_u8)

            # save binary
            pred_bin = (pred >= args.threshold).astype(np.uint8)
            cv2.imwrite(str(binary_dir / f"{name}_binary.png"), pred_bin * 255)

            # save overlay
            img_denorm = denorm_img(img_t[0])  # uint8 RGB in resized space
            heat_color = heatmap_to_color(pred)
            over = overlay(img_denorm, heat_color, alpha=args.alpha)
            cv2.imwrite(str(overlay_dir / f"{name}_overlay.png"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

            if use_gt:
                gt_path = gt_map.get(name) if gt_map is not None else None
                if gt_path is None:
                    missing_gt += 1
                    continue

                gt = load_heatmap_01(gt_path)  # [H,W] in [0,1]

                # IMPORTANT:
                # We compare in resized space. If GT is not same size, resize to match pred.
                if gt.shape != pred.shape:
                    gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)

                gt_bin = (gt >= args.threshold).astype(np.uint8)

                tp, fp, tn, fn = compute_confusion(pred_bin, gt_bin)
                dice, iou, acc, prec, rec = metrics_from_confusion(tp, fp, tn, fn)

                rows.append({
                    "stem": name,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "dice": dice, "iou": iou, "accuracy": acc, "precision": prec, "recall": rec,
                    "img_path": img_path,
                    "gt_path": gt_path,
                })

                sum_tp += tp; sum_fp += fp; sum_tn += tn; sum_fn += fn
                counted += 1

    print(f"[Done] results saved to: {out_dir}")

    if use_gt:
        metrics_csv = out_dir / "metrics.csv"

        # save per-image metrics
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["stem", "dice", "iou", "accuracy", "precision", "recall",
                          "tp", "fp", "tn", "fn", "img_path", "gt_path"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fieldnames})

        # overall (micro) metrics from summed confusion
        odice, oiou, oacc, oprec, orec = metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)

        # also print macro avg (mean of per-image)
        if counted > 0:
            mdice = float(np.mean([r["dice"] for r in rows]))
            miou  = float(np.mean([r["iou"] for r in rows]))
            macc  = float(np.mean([r["accuracy"] for r in rows]))
            mprec = float(np.mean([r["precision"] for r in rows]))
            mrec  = float(np.mean([r["recall"] for r in rows]))
        else:
            mdice = miou = macc = mprec = mrec = 0.0

        print(f"[GT] matched: {counted}, missing_gt: {missing_gt}")
        print(f"[Metrics saved] {metrics_csv}")

        print("\n[Overall metrics: MICRO (from summed TP/FP/TN/FN)]")
        print(f"  Dice      : {odice:.6f}")
        print(f"  IoU       : {oiou:.6f}")
        print(f"  Accuracy  : {oacc:.6f}")
        print(f"  Precision : {oprec:.6f}")
        print(f"  Recall    : {orec:.6f}")

        print("\n[Overall metrics: MACRO (mean of per-image)]")
        print(f"  Dice      : {mdice:.6f}")
        print(f"  IoU       : {miou:.6f}")
        print(f"  Accuracy  : {macc:.6f}")
        print(f"  Precision : {mprec:.6f}")
        print(f"  Recall    : {mrec:.6f}")


if __name__ == "__main__":
    main()

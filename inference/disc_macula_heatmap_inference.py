#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import csv
from pathlib import Path
from typing import Tuple

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


def overlay(rgb, heat_rgb, alpha=0.45):
    return np.clip(
        (1 - alpha) * rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32),
        0, 255
    ).astype(np.uint8)


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


def load_both_heatmap_rg01(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load BOTH heatmap PNG.
    Expected encoding (OpenCV BGR uint8):
      G(channel 1) = disc  (0~255)
      R(channel 2) = macula(0~255)
      B ignored
    Return:
      disc01, mac01 float32 in [0,1], shape (H,W)
    """
    hm = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if hm is None:
        raise FileNotFoundError(f"Failed to read BOTH heatmap: {path}")

    disc = hm[..., 1].astype(np.float32) / 255.0  # G -> disc
    mac  = hm[..., 2].astype(np.float32) / 255.0  # R -> macula
    disc = np.clip(disc, 0.0, 1.0)
    mac  = np.clip(mac,  0.0, 1.0)
    return disc, mac


def both_pred_to_bgr_u8(disc01: np.ndarray, mac01: np.ndarray) -> np.ndarray:
    """Return BGR uint8 where G=disc, R=mac, B=0."""
    g = (np.clip(disc01, 0, 1) * 255).astype(np.uint8)  # disc -> G
    r = (np.clip(mac01,  0, 1) * 255).astype(np.uint8)  # mac  -> R
    b = np.zeros_like(r, dtype=np.uint8)
    return np.stack([b, g, r], axis=-1)



def both_bin_to_bgr_u8(disc_bin: np.ndarray, mac_bin: np.ndarray) -> np.ndarray:
    """Return BGR uint8 binary image where G=disc_bin, R=mac_bin, B=0."""
    g = (disc_bin.astype(np.uint8) * 255)  # disc -> G
    r = (mac_bin.astype(np.uint8)  * 255)  # mac  -> R
    b = np.zeros_like(r, dtype=np.uint8)
    return np.stack([b, g, r], axis=-1)



def heatmap_both_to_color_rgb(disc01: np.ndarray, mac01: np.ndarray) -> np.ndarray:
    """
    For visualization only:
      disc -> JET colormap
      mac  -> HOT colormap
      mix  -> 50/50 blend
    Return RGB uint8.
    """
    d_u8 = (np.clip(disc01, 0, 1) * 255).astype(np.uint8)
    m_u8 = (np.clip(mac01,  0, 1) * 255).astype(np.uint8)

    d_col = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)  # BGR
    m_col = cv2.applyColorMap(m_u8, cv2.COLORMAP_HOT)  # BGR
    mix_bgr = cv2.addWeighted(d_col, 0.5, m_col, 0.5, 0.0)
    return cv2.cvtColor(mix_bgr, cv2.COLOR_BGR2RGB)


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


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="directory containing input images")
    parser.add_argument("--gt_dir", default=None, type=str,
                        help="(optional) directory containing GT BOTH heatmaps matched by filename stem")
    parser.add_argument("--ckpt", required=True, type=str, help="trained model checkpoint (.pth)")
    parser.add_argument("--out_dir", default="./infer_results", type=str)

    parser.add_argument("--test_case", type=str, required=True, help="subfolder name under out_dir")
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
    # Model (2-class)
    # -------------------------
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=2,          # ✅ 2-class
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

    # -------------------------
    # GT indexing (optional)
    # -------------------------
    use_gt = args.gt_dir is not None
    gt_map = None
    if use_gt:
        gt_dir = Path(args.gt_dir)
        if not gt_dir.exists():
            raise FileNotFoundError(f"--gt_dir not found: {gt_dir}")
        gt_map = build_stem_map(gt_dir)
        print(f"[GT] {len(gt_map)} GT files indexed from {gt_dir}")

    # -------------------------
    # Metrics accumulation
    # -------------------------
    rows = []

    # per-class micro confusion sums
    sum_tp_d = sum_fp_d = sum_tn_d = sum_fn_d = 0.0
    sum_tp_m = sum_fp_m = sum_tn_m = sum_fn_m = 0.0

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

            # pred: (2,H,W) in [0,1]
            pred2 = torch.sigmoid(model(img_t))[0].detach().cpu().numpy()
            pred_disc = pred2[0]
            pred_mac  = pred2[1]

            # ---- save predicted heatmap (ONE PNG, BGR: R=disc, G=mac, B=0)
            both_hm_bgr = both_pred_to_bgr_u8(pred_disc, pred_mac)
            cv2.imwrite(str(heatmap_dir / f"{name}_heatmap.png"), both_hm_bgr)

            # ---- save binary (ONE PNG, same encoding)
            pred_disc_bin = (pred_disc >= args.threshold).astype(np.uint8)
            pred_mac_bin  = (pred_mac  >= args.threshold).astype(np.uint8)
            both_bin_bgr = both_bin_to_bgr_u8(pred_disc_bin, pred_mac_bin)
            cv2.imwrite(str(binary_dir / f"{name}_binary.png"), both_bin_bgr)

            # ---- save overlay
            img_denorm = denorm_img(img_t[0])  # uint8 RGB in resized space
            heat_color = heatmap_both_to_color_rgb(pred_disc, pred_mac)  # RGB uint8
            over = overlay(img_denorm, heat_color, alpha=args.alpha)
            cv2.imwrite(str(overlay_dir / f"{name}_overlay.png"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

            # ---- metrics (optional)
            if use_gt:
                gt_path = gt_map.get(name) if gt_map is not None else None
                if gt_path is None:
                    missing_gt += 1
                    continue

                gt_disc, gt_mac = load_both_heatmap_rg01(gt_path)

                # Compare in resized space. If GT is not same size, resize to match pred.
                if gt_disc.shape != pred_disc.shape:
                    gt_disc = cv2.resize(gt_disc, (pred_disc.shape[1], pred_disc.shape[0]), interpolation=cv2.INTER_LINEAR)
                if gt_mac.shape != pred_mac.shape:
                    gt_mac = cv2.resize(gt_mac, (pred_mac.shape[1], pred_mac.shape[0]), interpolation=cv2.INTER_LINEAR)

                gt_disc_bin = (gt_disc >= args.threshold).astype(np.uint8)
                gt_mac_bin  = (gt_mac  >= args.threshold).astype(np.uint8)

                # disc metrics
                tp_d, fp_d, tn_d, fn_d = compute_confusion(pred_disc_bin, gt_disc_bin)
                dice_d, iou_d, acc_d, prec_d, rec_d = metrics_from_confusion(tp_d, fp_d, tn_d, fn_d)

                # mac metrics
                tp_m, fp_m, tn_m, fn_m = compute_confusion(pred_mac_bin, gt_mac_bin)
                dice_m, iou_m, acc_m, prec_m, rec_m = metrics_from_confusion(tp_m, fp_m, tn_m, fn_m)

                # accumulate micro sums
                sum_tp_d += tp_d; sum_fp_d += fp_d; sum_tn_d += tn_d; sum_fn_d += fn_d
                sum_tp_m += tp_m; sum_fp_m += fp_m; sum_tn_m += tn_m; sum_fn_m += fn_m

                rows.append({
                    "stem": name,
                    "dice_disc": dice_d, "iou_disc": iou_d, "acc_disc": acc_d, "prec_disc": prec_d, "rec_disc": rec_d,
                    "dice_mac":  dice_m, "iou_mac":  iou_m, "acc_mac":  acc_m, "prec_mac":  prec_m, "rec_mac":  rec_m,
                    "tp_disc": tp_d, "fp_disc": fp_d, "tn_disc": tn_d, "fn_disc": fn_d,
                    "tp_mac":  tp_m, "fp_mac":  fp_m, "tn_mac":  tn_m, "fn_mac":  fn_m,
                    "img_path": img_path,
                    "gt_path": gt_path,
                })

                counted += 1

    print(f"[Done] results saved to: {out_dir}")

    # -------------------------
    # Save metrics.csv (optional)
    # -------------------------
    if use_gt:
        metrics_csv = out_dir / "metrics.csv"

        # per-image metrics
        fieldnames = [
            "stem",
            "dice_disc", "iou_disc", "acc_disc", "prec_disc", "rec_disc",
            "dice_mac",  "iou_mac",  "acc_mac",  "prec_mac",  "rec_mac",
            "tp_disc", "fp_disc", "tn_disc", "fn_disc",
            "tp_mac",  "fp_mac",  "tn_mac",  "fn_mac",
            "img_path", "gt_path"
        ]
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        # overall MICRO per-class
        odice_d, oiou_d, oacc_d, oprec_d, orec_d = metrics_from_confusion(sum_tp_d, sum_fp_d, sum_tn_d, sum_fn_d)
        odice_m, oiou_m, oacc_m, oprec_m, orec_m = metrics_from_confusion(sum_tp_m, sum_fp_m, sum_tn_m, sum_fn_m)

        # overall MICRO across both classes (sum confusions)
        sum_tp = sum_tp_d + sum_tp_m
        sum_fp = sum_fp_d + sum_fp_m
        sum_tn = sum_tn_d + sum_tn_m
        sum_fn = sum_fn_d + sum_fn_m
        odice_micro, oiou_micro, oacc_micro, oprec_micro, orec_micro = metrics_from_confusion(sum_tp, sum_fp, sum_tn, sum_fn)

        # overall MACRO across classes (average of per-class micro)
        odice_macro = 0.5 * (odice_d + odice_m)
        oiou_macro  = 0.5 * (oiou_d  + oiou_m)
        oacc_macro  = 0.5 * (oacc_d  + oacc_m)
        oprec_macro = 0.5 * (oprec_d + oprec_m)
        orec_macro  = 0.5 * (orec_d  + orec_m)

        print(f"[GT] matched: {counted}, missing_gt: {missing_gt}")
        print(f"[Metrics saved] {metrics_csv}")

        print("\n[Overall metrics: MICRO (per-class, from summed TP/FP/TN/FN)]")
        print(
            f"  DISC  "
            f"Dice={odice_d:.6f}  IoU={oiou_d:.6f}  "
            f"Acc={oacc_d:.6f}  Prec={oprec_d:.6f}  Rec={orec_d:.6f}"
        )
        print(
            f"  MAC   "
            f"Dice={odice_m:.6f}  IoU={oiou_m:.6f}  "
            f"Acc={oacc_m:.6f}  Prec={oprec_m:.6f}  Rec={orec_m:.6f}"
        )


        print("\n[Overall metrics: MICRO (both classes combined)]")
        print(
            f"  Dice={odice_micro:.6f}  IoU={oiou_micro:.6f}  "
            f"Acc={oacc_micro:.6f}  Prec={oprec_micro:.6f}  Rec={orec_micro:.6f}"
        )


        print("\n[Overall metrics: MACRO (avg of per-class micro)]")
        print(
            f"  Dice={odice_macro:.6f}  IoU={oiou_macro:.6f}  "
            f"Acc={oacc_macro:.6f}  Prec={oprec_macro:.6f}  Rec={orec_macro:.6f}"
        )



if __name__ == "__main__":
    main()

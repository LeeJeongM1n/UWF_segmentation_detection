#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import cv2
import numpy as np


# ---------------------------
# Utils
# ---------------------------
def list_images(images_dir: Path, recursive: bool = False):
    exts = {".png", ".jpg", ".jpeg"}
    if recursive:
        return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def read_yolo_centers(label_path: Path, want_class_ids=(0, 1)):
    """
    YOLO label format per line:
      cls cx cy w h   (cx,cy,w,h normalized)
    Return:
      dict[int, (cx_norm, cy_norm)]
    """
    found = {}
    if not label_path.exists():
        return found

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="latin-1").splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
        except ValueError:
            continue

        if cls in want_class_ids and cls not in found:
            found[cls] = (cx, cy)

        if all(cid in found for cid in want_class_ids):
            break
    return found


def gaussian_heatmap(H: int, W: int, cx_px: float, cy_px: float, sigma_px: float):
    """
    Return float32 heatmap in [0,1], shape (H,W)
    """
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist2 = (xx - cx_px) ** 2 + (yy - cy_px) ** 2
    hm = np.exp(-dist2 / (2.0 * (sigma_px ** 2))).astype(np.float32)
    return hm


def overlay_heatmap(bgr_img: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45,
                    colormap: int = cv2.COLORMAP_JET):
    """
    bgr_img: uint8 (H,W,3)
    heatmap01: float32 (H,W) in [0,1]
    """
    hm_u8 = np.clip(heatmap01 * 255.0, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, colormap)  # BGR
    out = cv2.addWeighted(bgr_img, 1.0 - alpha, hm_color, alpha, 0)
    return out


def ensure_output_dirs(class_root: Path, test_case: str):
    """
    class_root/{overlay|heatmap}/{test_case}/(+ no_label)
    """
    overlay_dir = class_root / "overlay" / test_case
    overlay_nl = overlay_dir / "no_label"
    heatmap_dir = class_root / "heatmap" / test_case
    heatmap_nl = heatmap_dir / "no_label"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_nl.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    heatmap_nl.mkdir(parents=True, exist_ok=True)

    return overlay_dir, overlay_nl, heatmap_dir, heatmap_nl


def save_pair(bgr: np.ndarray, hm01: np.ndarray, rel: Path, overlay_dir: Path, heatmap_dir: Path, alpha: float):
    """
    overlay: 컬러 overlay png
    heatmap: grayscale 0~255 png
    rel: images_dir 기준 상대경로
    """
    out_overlay = (overlay_dir / rel).with_suffix(".png")
    out_overlay.parent.mkdir(parents=True, exist_ok=True)
    ov = overlay_heatmap(bgr, hm01, alpha=alpha)
    cv2.imwrite(str(out_overlay), ov)

    out_hm = (heatmap_dir / rel).with_suffix(".png")
    out_hm.parent.mkdir(parents=True, exist_ok=True)
    hm_u8 = np.clip(hm01 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_hm), hm_u8)


# ---------------------------
# Core
# ---------------------------
def process_one_dataset(dataset_dir: Path,
                        sigma_disc: float,
                        sigma_macula: float,
                        alpha: float,
                        test_case: str,
                        recursive_images: bool = False,
                        require_labels: bool = False):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        print(f"[SKIP] no images/: {images_dir}")
        return

    if not labels_dir.is_dir():
        if require_labels:
            print(f"[SKIP] no labels/: {labels_dir}")
            return
        print(f"[WARN] no labels/: {labels_dir} (all will be treated as no_label)")

    # output roots
    disc_root = dataset_dir / "heatmap" / "disc"
    mac_root = dataset_dir / "heatmap" / "macula"

    disc_overlay, disc_overlay_nl, disc_hm, disc_hm_nl = ensure_output_dirs(disc_root, test_case)
    mac_overlay, mac_overlay_nl, mac_hm, mac_hm_nl = ensure_output_dirs(mac_root, test_case)

    img_paths = list_images(images_dir, recursive=recursive_images)
    if not img_paths:
        print(f"[SKIP] no images found in {images_dir}")
        return

    total = 0
    miss_lbl = 0
    disc_missing = 0
    mac_missing = 0
    failed_read = 0

    for img_path in img_paths:
        total += 1
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            failed_read += 1
            print(f"  [WARN] cannot read: {img_path}")
            continue

        H, W = bgr.shape[:2]
        rel = img_path.relative_to(images_dir)
        label_path = (labels_dir / rel).with_suffix(".txt")

        if not label_path.exists():
            miss_lbl += 1

        centers = read_yolo_centers(label_path, want_class_ids=(0, 1))

        # disc (cls=0)
        if 0 in centers:
            cx0 = float(centers[0][0]) * W
            cy0 = float(centers[0][1]) * H
            hm0 = gaussian_heatmap(H, W, cx0, cy0, sigma_disc)
            save_pair(bgr, hm0, rel, disc_overlay, disc_hm, alpha)
        else:
            disc_missing += 1
            hm0 = np.zeros((H, W), dtype=np.float32)
            save_pair(bgr, hm0, rel, disc_overlay_nl, disc_hm_nl, alpha)

        # macula (cls=1)
        if 1 in centers:
            cx1 = float(centers[1][0]) * W
            cy1 = float(centers[1][1]) * H
            hm1 = gaussian_heatmap(H, W, cx1, cy1, sigma_macula)
            save_pair(bgr, hm1, rel, mac_overlay, mac_hm, alpha)
        else:
            mac_missing += 1
            hm1 = np.zeros((H, W), dtype=np.float32)
            save_pair(bgr, hm1, rel, mac_overlay_nl, mac_hm_nl, alpha)

        if total % 500 == 0:
            print(f"  [INFO] {dataset_dir.name}: processed {total}/{len(img_paths)}")

    print(f"========== {dataset_dir.name} | test_case={test_case} ==========")
    print(f"images processed        : {total}")
    print(f"failed image read       : {failed_read}")
    print(f"missing label files     : {miss_lbl}")
    print(f"no disc (cls=0) labels  : {disc_missing}")
    print(f"no macula (cls=1) labels: {mac_missing}")
    print(f"disc out root           : {disc_root}")
    print(f"macula out root         : {mac_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det_root", type=str, default="/mnt/richul_FM/UWF_seg_det/datasets/Det",
                   help="Det root containing {Dataset_name}/images and {Dataset_name}/labels")

    p.add_argument("--test_case", type=str, required=True,
                   help="Output run tag folder name under overlay/ and heatmap/ (e.g., exp01, epoch0, v1)")

    p.add_argument("--sigma_disc", type=float, default=50.0,
                   help="Gaussian sigma for disc (class 0), in pixels")
    p.add_argument("--sigma_macula", type=float, default=200.0,
                   help="Gaussian sigma for macula (class 1), in pixels")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha (0~1)")

    p.add_argument("--recursive_images", action="store_true",
                   help="If images/ contains subfolders, enable recursive search.")
    p.add_argument("--require_labels", action="store_true",
                   help="If set, skip datasets without labels/ folder.")
    p.add_argument("--only_dataset", type=str, default=None,
                   help="Process only this Dataset_name under det_root (optional).")

    args = p.parse_args()

    det_root = Path(args.det_root)
    if not det_root.is_dir():
        raise FileNotFoundError(f"det_root not found: {det_root}")

    # datasets 선택
    if args.only_dataset:
        ds_dirs = [det_root / args.only_dataset]
        if not ds_dirs[0].is_dir():
            raise FileNotFoundError(f"only_dataset not found: {ds_dirs[0]}")
        print(f"[MODE] single dataset: {args.only_dataset}")
    else:
        ds_dirs = sorted([d for d in det_root.iterdir() if d.is_dir()])
        print(f"[MODE] batch datasets: {len(ds_dirs)} folders")

    print(f"[DET_ROOT]  {det_root}")
    print(f"[TEST_CASE] {args.test_case}")

    for ds in ds_dirs:
        process_one_dataset(
            dataset_dir=ds,
            sigma_disc=args.sigma_disc,
            sigma_macula=args.sigma_macula,
            alpha=args.alpha,
            test_case=args.test_case,
            recursive_images=args.recursive_images,
            require_labels=args.require_labels,
        )

    print("[DONE] heatmap generation finished.")


if __name__ == "__main__":
    main()

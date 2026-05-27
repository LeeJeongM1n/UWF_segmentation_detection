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


def ensure_output_dirs(both_root: Path, run_name: str):
    """
    both_root/{overlay|heatmap}/{run_name}/(+ no_label)
    """
    overlay_dir = both_root / "overlay" / run_name
    overlay_nl = overlay_dir / "no_label"
    heatmap_dir = both_root / "heatmap" / run_name
    heatmap_nl = heatmap_dir / "no_label"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_nl.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    heatmap_nl.mkdir(parents=True, exist_ok=True)

    return overlay_dir, overlay_nl, heatmap_dir, heatmap_nl


def make_both_heatmap_bgr(hm_disc01: np.ndarray, hm_mac01: np.ndarray) -> np.ndarray:
    """
    Save ONE PNG that contains both heatmaps (OpenCV uses BGR order):
      - disc  -> Green channel (G)
      - macula-> Red channel (R)
      - Blue channel (B) is 0
    """
    # OpenCV channel order: [B, G, R]
    g = np.clip(hm_disc01 * 255.0, 0, 255).astype(np.uint8)   # disc -> G
    r = np.clip(hm_mac01 * 255.0, 0, 255).astype(np.uint8)    # macula -> R
    b = np.zeros_like(r, dtype=np.uint8)
    return np.stack([b, g, r], axis=-1)  # BGR



def overlay_both(bgr_img: np.ndarray, both_hm_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay both-heatmap(BGR) onto the original image.
    """
    out = cv2.addWeighted(bgr_img, 1.0 - alpha, both_hm_bgr, alpha, 0)
    return out


def save_both_png(bgr: np.ndarray,
                  hm_disc01: np.ndarray,
                  hm_mac01: np.ndarray,
                  rel: Path,
                  overlay_dir: Path,
                  heatmap_dir: Path,
                  alpha: float):
    """
    heatmap: 3-channel PNG (BGR) holding both disc/macula
    overlay: original + both heatmap overlay
    """
    both_hm_bgr = make_both_heatmap_bgr(hm_disc01, hm_mac01)

    out_hm = (heatmap_dir / rel).with_suffix(".png")
    out_hm.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_hm), both_hm_bgr)

    out_overlay = (overlay_dir / rel).with_suffix(".png")
    out_overlay.parent.mkdir(parents=True, exist_ok=True)
    ov = overlay_both(bgr, both_hm_bgr, alpha=alpha)
    cv2.imwrite(str(out_overlay), ov)


# ---------------------------
# Core
# ---------------------------
def process_one_dataset(dataset_dir: Path,
                        split_name: str,
                        sigma: float,
                        alpha: float,
                        test_case: str,
                        recursive_images: bool = False,
                        require_labels: bool = False):
    """
    dataset_dir/
      images/{train|valid|test}/...
      labels/{train|valid|test}/...
    """
    images_dir = dataset_dir / "images" / split_name
    labels_dir = dataset_dir / "labels" / split_name


    if not images_dir.is_dir():
        print(f"[SKIP] no images/: {images_dir}")
        return

    if not labels_dir.is_dir():
        if require_labels:
            print(f"[SKIP] no labels/: {labels_dir}")
            return
        print(f"[WARN] no labels/: {labels_dir} (all will be treated as no_label)")

    both_root = dataset_dir / "heatmap"
    run_name = f"{test_case}_{split_name}"
    both_overlay, both_overlay_nl, both_hm, both_hm_nl = ensure_output_dirs(both_root, run_name)

    img_paths = list_images(images_dir, recursive=recursive_images)
    if not img_paths:
        print(f"[SKIP] no images found in {images_dir}")
        return

    total = 0
    miss_lbl_file = 0
    not_both_cnt = 0
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

        # default: empty maps
        hm0 = np.zeros((H, W), dtype=np.float32)  # disc
        hm1 = np.zeros((H, W), dtype=np.float32)  # macula

        if not label_path.exists():
            miss_lbl_file += 1
            # 라벨 파일 자체가 없으면 no_label로 저장
            save_both_png(bgr, hm0, hm1, rel, both_overlay_nl, both_hm_nl, alpha)
            continue

        centers = read_yolo_centers(label_path, want_class_ids=(0, 1))

        has_disc = 0 in centers
        has_mac = 1 in centers

        if has_disc:
            cx0 = float(centers[0][0]) * W
            cy0 = float(centers[0][1]) * H
            hm0 = gaussian_heatmap(H, W, cx0, cy0, sigma)

        if has_mac:
            cx1 = float(centers[1][0]) * W
            cy1 = float(centers[1][1]) * H
            hm1 = gaussian_heatmap(H, W, cx1, cy1, sigma)

        # "both" 의미를 엄격히: disc+macula 둘 다 있을 때만 정상 폴더에 저장
        if has_disc and has_mac:
            save_both_png(bgr, hm0, hm1, rel, both_overlay, both_hm, alpha)
        else:
            not_both_cnt += 1
            save_both_png(bgr, hm0, hm1, rel, both_overlay_nl, both_hm_nl, alpha)

        if total % 500 == 0:
            print(f"  [INFO] {dataset_dir.name}/{split_name}: processed {total}/{len(img_paths)}")

    print(f"========== {dataset_dir.name} | split={split_name} | run={run_name} ==========")
    print(f"images processed        : {total}")
    print(f"failed image read       : {failed_read}")
    print(f"missing label files     : {miss_lbl_file}")
    print(f"label exists but not both(0&1) : {not_both_cnt}")
    print(f"both out root           : {both_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--det_root", type=str, required=True,
        help="Path containing images/{train,valid,test} and labels/{train,valid,test}"
    )
    p.add_argument("--test_case", type=str, required=True)
    p.add_argument("--sigma", type=float, default=20.0, help="Gaussian sigma in pixels")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha (0~1)")
    p.add_argument("--recursive_images", action="store_true")
    p.add_argument("--require_labels", action="store_true")

    args = p.parse_args()

    print(f"[RUNNING FILE] {__file__}")

    dataset_dir = Path(args.det_root)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"det_root not found: {dataset_dir}")

    print(f"[DATASET]   {dataset_dir}")
    print(f"[TEST_CASE] {args.test_case}")

    for split_name in ["train", "valid", "test"]:
    # split_name=""
        process_one_dataset(
            dataset_dir=dataset_dir,
            split_name=split_name,
            sigma=args.sigma,
            alpha=args.alpha,
            test_case=args.test_case,
            recursive_images=args.recursive_images,
            require_labels=args.require_labels,
        )

    print("[DONE] both-heatmap PNG generation finished.")

if __name__ == "__main__":
    main()


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
    YOLO label format:
      cls cx cy w h  (all normalized 0~1)

    Return:
      dict[int, (cx_norm, cy_norm)]  for cls in want_class_ids
      - 동일 클래스가 여러 줄이면 첫 줄만 사용
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
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist2 = (xx - cx_px) ** 2 + (yy - cy_px) ** 2
    hm = np.exp(-dist2 / (2.0 * (sigma_px ** 2))).astype(np.float32)
    return hm


def overlay_heatmap(bgr_img: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45,
                    colormap: int = cv2.COLORMAP_JET):
    hm_u8 = np.clip(heatmap01 * 255.0, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, colormap)  # BGR
    out = cv2.addWeighted(bgr_img, 1.0 - alpha, hm_color, alpha, 0)
    return out


def ensure_output_dirs(heatmap_root: Path, cls_name: str, run_tag: str):
    """
    heatmap_root/
      {cls_name}/
        overlay/{run_tag}/
        overlay/{run_tag}/no_label/
        heatmap/{run_tag}/
        heatmap/{run_tag}/no_label/
    """
    class_root = heatmap_root / cls_name

    overlay_dir = class_root / "overlay" / run_tag
    overlay_nl = overlay_dir / "no_label"
    heat_dir = class_root / "heatmap" / run_tag
    heat_nl = heat_dir / "no_label"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_nl.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)
    heat_nl.mkdir(parents=True, exist_ok=True)

    return overlay_dir, overlay_nl, heat_dir, heat_nl


def save_pair(bgr: np.ndarray, hm01: np.ndarray, rel: Path, overlay_dir: Path, heat_dir: Path, alpha: float):
    """
    rel: split_images_dir 기준 상대경로 (train/ 아래 구조 유지)
    """
    out_overlay = (overlay_dir / rel).with_suffix(".png")
    out_overlay.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_overlay), overlay_heatmap(bgr, hm01, alpha=alpha))

    out_hm = (heat_dir / rel).with_suffix(".png")
    out_hm.parent.mkdir(parents=True, exist_ok=True)
    hm_u8 = np.clip(hm01 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_hm), hm_u8)


# ---------------------------
# Core (split 기반)
# ---------------------------
def process_split(det_root: Path, split: str, sigma: float, alpha: float,
                  recursive_images: bool, require_labels: bool):
    """
    det_root/
      images/{split}/...
      labels/{split}/...
      heatmap/...

    run_tag = f"sigma{sigma}_{split}"
    """
    images_dir = det_root / "images" / split
    labels_dir = det_root / "labels" / split
    heatmap_root = det_root / "heatmap"

    if not images_dir.is_dir():
        print(f"[SKIP] images/{split} not found: {images_dir}")
        return

    if not labels_dir.is_dir():
        if require_labels:
            print(f"[SKIP] labels/{split} not found: {labels_dir}")
            return
        print(f"[WARN] labels/{split} not found: {labels_dir} (all treated as no_label)")

    run_tag = f"sigma{sigma:g}_{split}"  # 50.0 -> 50, 12.5 -> 12.5

    # output dirs (disc/macula 각각)
    disc_overlay, disc_overlay_nl, disc_heat, disc_heat_nl = ensure_output_dirs(
        heatmap_root, "disc", run_tag
    )
    mac_overlay, mac_overlay_nl, mac_heat, mac_heat_nl = ensure_output_dirs(
        heatmap_root, "macula", run_tag
    )

    img_paths = list_images(images_dir, recursive=recursive_images)
    if not img_paths:
        print(f"[SKIP] no images under: {images_dir}")
        return

    total = 0
    miss_lbl_file = 0
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

        # split images_dir 기준 상대경로 (하위폴더 구조 그대로 유지)
        rel = img_path.relative_to(images_dir)

        # labels도 동일 상대경로 + .txt
        label_path = (labels_dir / rel).with_suffix(".txt")
        if not label_path.exists():
            miss_lbl_file += 1

        centers = read_yolo_centers(label_path, want_class_ids=(0, 1))

        # disc (cls=0)
        if 0 in centers:
            cx0 = float(centers[0][0]) * W
            cy0 = float(centers[0][1]) * H
            hm0 = gaussian_heatmap(H, W, cx0, cy0, sigma)
            save_pair(bgr, hm0, rel, disc_overlay, disc_heat, alpha)
        else:
            disc_missing += 1
            hm0 = np.zeros((H, W), dtype=np.float32)
            save_pair(bgr, hm0, rel, disc_overlay_nl, disc_heat_nl, alpha)

        # macula (cls=1)
        if 1 in centers:
            cx1 = float(centers[1][0]) * W
            cy1 = float(centers[1][1]) * H
            hm1 = gaussian_heatmap(H, W, cx1, cy1, sigma)
            save_pair(bgr, hm1, rel, mac_overlay, mac_heat, alpha)
        else:
            mac_missing += 1
            hm1 = np.zeros((H, W), dtype=np.float32)
            save_pair(bgr, hm1, rel, mac_overlay_nl, mac_heat_nl, alpha)

        if total % 500 == 0:
            print(f"  [INFO] {split}: processed {total}/{len(img_paths)}")

    print(f"========== SPLIT={split} | run_tag={run_tag} ==========")
    print(f"images processed         : {total}")
    print(f"failed image read        : {failed_read}")
    print(f"missing label files      : {miss_lbl_file}")
    print(f"no disc labels (cls=0)   : {disc_missing}")
    print(f"no macula labels (cls=1) : {mac_missing}")
    print(f"output heatmap root      : {heatmap_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det_root", type=str, required=True,
                   help="Folder containing images/{train,test}, labels/{train,test}, heatmap/")
    p.add_argument("--sigma", type=float, required=True,
                   help="Gaussian sigma in pixels (same for disc & macula)")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha (0~1)")

    p.add_argument("--recursive_images", action="store_true",
                   help="If images/{split} contains subfolders, enable recursive search.")
    p.add_argument("--require_labels", action="store_true",
                   help="If set, skip split when labels/{split} folder does not exist.")

    # 처리할 split 선택 (기본: train+test 둘 다)
    p.add_argument("--splits", type=str, default="train,test",
                   help="Comma-separated splits to process. e.g., 'train' or 'test' or 'train,test'")

    args = p.parse_args()

    det_root = Path(args.det_root)
    if not det_root.is_dir():
        raise FileNotFoundError(f"det_root not found: {det_root}")

    # 기본 구조 체크(강제는 아니지만 안내)
    if not (det_root / "images").is_dir():
        print(f"[WARN] images/ not found under det_root: {det_root}")
    if not (det_root / "labels").is_dir():
        print(f"[WARN] labels/ not found under det_root: {det_root}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No splits specified. Use --splits train,test")

    print(f"[DET_ROOT] {det_root}")
    print(f"[SIGMA]    {args.sigma}")
    print(f"[SPLITS]   {splits}")
    print(f"[RECURSIVE_IMAGES] {args.recursive_images}")

    for split in splits:
        process_split(
            det_root=det_root,
            split=split,
            sigma=args.sigma,
            alpha=args.alpha,
            recursive_images=args.recursive_images,
            require_labels=args.require_labels,
        )

    print("[DONE] heatmap generation finished.")


if __name__ == "__main__":
    main()

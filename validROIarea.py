#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg"}


@dataclass
class DatasetPair:
    name: str
    img_dir: str
    mask_dir: str


def list_images(img_dir: Path) -> List[Path]:
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def default_mask_path(mask_dir: Path, img_path: Path) -> Path:
    # image.png -> image_mask.png
    return mask_dir / f"{img_path.stem}_mask.png"


def read_mask_as_bool(mask_path: Path) -> np.ndarray:
    """
    Read mask image (grayscale) and return boolean valid mask.
    Rule: any non-zero pixel is valid retinal area.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
    return (m > 0)


def compute_proportion(mask_bool: np.ndarray, target_hw: Optional[Tuple[int, int]] = None) -> float:
    """
    proportion = valid_pixels / (H*W)
    If target_hw provided, resize mask to that size with nearest interpolation (to preserve binary).
    target_hw = (H, W)
    """
    if target_hw is not None:
        H, W = target_hw
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = (mask_u8 > 0)

    H, W = mask_bool.shape[:2]
    valid = int(mask_bool.sum())
    total = int(H * W)
    return valid / total if total > 0 else float("nan")


def summarize(arr: np.ndarray) -> Dict[str, float]:
    arr = arr.astype(np.float64)
    return {
        "N": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr, ddof=0)) if arr.size else float("nan"),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
    }


def main(
    datasets: List[DatasetPair],
    out_csv: str = "retina_area_proportions.csv",
    out_npy: str = "retina_area_proportions.npy",
    fixed_hw: Optional[Tuple[int, int]] = (614, 780),  # (H, W) 기본값: 614x780
    strict: bool = False,  # True면 마스크 누락 시 즉시 에러
):
    rows = []
    all_props = []
    per_ds_props: Dict[str, List[float]] = {d.name: [] for d in datasets}

    for ds in datasets:
        img_dir = Path(ds.img_dir)
        mask_dir = Path(ds.mask_dir)

        if not img_dir.is_dir():
            raise NotADirectoryError(f"Image dir not found: {img_dir}")
        if not mask_dir.is_dir():
            raise NotADirectoryError(f"Mask dir not found: {mask_dir}")

        imgs = list_images(img_dir)
        if len(imgs) == 0:
            print(f"[WARN] No images in: {img_dir}")
            continue

        print(f"\n== Dataset: {ds.name} ==")
        print(f"  images: {img_dir}")
        print(f"  masks : {mask_dir}")
        print(f"  count : {len(imgs)}")

        for img_path in imgs:
            mask_path = default_mask_path(mask_dir, img_path)

            if not mask_path.exists():
                msg = f"[MISS] mask not found for {img_path.name} -> {mask_path.name}"
                if strict:
                    raise FileNotFoundError(msg)
                else:
                    # 누락은 스킵
                    # 필요하면 proportion을 NaN으로 기록하도록 바꿔도 됨
                    print(msg)
                    continue

            try:
                mask_bool = read_mask_as_bool(mask_path)
                prop = compute_proportion(mask_bool, target_hw=fixed_hw if fixed_hw else None)
            except Exception as e:
                msg = f"[ERR] {img_path.name}: {e}"
                if strict:
                    raise
                print(msg)
                continue

            rows.append({
                "dataset": ds.name,
                "image_name": img_path.name,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "H": int((fixed_hw[0] if fixed_hw else mask_bool.shape[0])),
                "W": int((fixed_hw[1] if fixed_hw else mask_bool.shape[1])),
                "proportion": float(prop),
            })
            all_props.append(prop)
            per_ds_props[ds.name].append(prop)

    # 저장
    out_csv = Path(out_csv)
    out_npy = Path(out_npy)

    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        np.save(out_npy, np.array(all_props, dtype=np.float32))
        print(f"\n[SAVED] CSV: {out_csv.resolve()}")
        print(f"[SAVED] NPY: {out_npy.resolve()}")
    else:
        print("\n[WARN] No rows to save (all skipped).")

    # 통계 출력
    all_arr = np.array(all_props, dtype=np.float64)
    print("\n===== Overall (combined 3 datasets) =====")
    s = summarize(all_arr)
    print(f"N    : {s['N']}")
    print(f"mean : {s['mean']:.6f}")
    print(f"std  : {s['std']:.6f}")
    print(f"min  : {s['min']:.6f}")
    print(f"max  : {s['max']:.6f}")

    print("\n===== Per-dataset =====")
    for ds in datasets:
        arr = np.array(per_ds_props.get(ds.name, []), dtype=np.float64)
        ss = summarize(arr)
        print(f"\n[{ds.name}]")
        print(f"  N    : {ss['N']}")
        print(f"  mean : {ss['mean']:.6f}")
        print(f"  std  : {ss['std']:.6f}")
        print(f"  min  : {ss['min']:.6f}")
        print(f"  max  : {ss['max']:.6f}")


if __name__ == "__main__":
    datasets = [
        DatasetPair(
            name="Internal",
            img_dir="/mnt/richul_FM/UWF_seg_det/datasets/YOLO/new_split/images/test",
            mask_dir="/mnt/richul_FM/UWF_seg_det/datasets/YOLO/seg_output/test_inference/internal",
        ),
        DatasetPair(
            name="OUWFD",
            img_dir="/mnt/richul_FM/UWF_seg_det/datasets/Det/OUWFD/images",
            mask_dir="/mnt/richul_FM/UWF_seg_det/datasets/YOLO/seg_output/test_inference/OUWFD",
        ),
        DatasetPair(
            name="MSHF",
            img_dir="/mnt/richul_FM/UWF_seg_det/datasets/Det/MSHF/images",
            mask_dir="/mnt/richul_FM/UWF_seg_det/datasets/YOLO/seg_output/test_inference/MSHF",
        ),
    ]

    # fixed_hw=(614,780) -> 모든 마스크를 614x780으로 맞춰 계산 (원본 기준이 이 크기라면 유지 추천)
    main(
        datasets=datasets,
        out_csv="retina_area_proportions.csv",
        out_npy="retina_area_proportions.npy",
        fixed_hw=(614, 780),
        strict=False,
    )

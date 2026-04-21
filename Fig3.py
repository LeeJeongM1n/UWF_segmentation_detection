#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    fs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    fs.sort()
    return fs


def imread_color(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {p}")
    return img


def resize_to_wh(img: np.ndarray, wh: Tuple[int, int], interp: int) -> np.ndarray:
    w, h = wh
    return cv2.resize(img, (w, h), interpolation=interp)


def fit_to_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    nw, nh = int(round(w * s)), int(round(h * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)



def binarize_u8(gray_u8: np.ndarray, thr: int) -> np.ndarray:
    return (gray_u8 >= thr).astype(np.uint8)  # 0/1


def overlay_binary(base_bgr: np.ndarray, mask01: np.ndarray, color_bgr, alpha: float) -> np.ndarray:
    out = base_bgr.copy()
    m = mask01.astype(bool)
    if not m.any():
        return out
    color = np.zeros_like(out, dtype=np.uint8)
    color[:] = color_bgr
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + color[m].astype(np.float32) * alpha).astype(np.uint8)
    return out


def find_by_stem(folder: Path, stem: str) -> Optional[Path]:
    # 가장 흔한 패턴 우선
    candidates = [
        folder / f"{stem}.png",
        folder / f"{stem}.jpg",
        folder / f"{stem}_mask.png",
        folder / f"{stem}_roi.png",
        folder / f"{stem}_heatmap.png",
        folder / f"{stem}_hm.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def make_panel(A, B, C, D, cell_wh: Tuple[int, int]) -> np.ndarray:
    cw, ch = cell_wh

    def prep(x):
        interp = cv2.INTER_AREA if x.shape[1] > cw else cv2.INTER_LINEAR
        return resize_to_wh(x, (cw, ch), interp)

    top = np.concatenate([prep(A), prep(B)], axis=1)
    bot = np.concatenate([prep(C), prep(D)], axis=1)
    return np.concatenate([top, bot], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--heatmap_dir", required=True, help="heatmap PNG dir (combined or 2 separate PNGs)")
    ap.add_argument("--output_dir", required=True)

    # heatmap PNG가 combined(BGR 3ch)일 때: 어떤 채널이 disc/mac인지 지정
    # OpenCV는 BGR: 0=B, 1=G, 2=R
    ap.add_argument("--disc_ch", type=int, default=1, help="disc channel index in BGR heatmap PNG. default=1 (G)")
    ap.add_argument("--mac_ch", type=int, default=2, help="macula channel index in BGR heatmap PNG. default=2 (R)")

    ap.add_argument("--mask_thr", type=float, default=1, help="ROI binary threshold (u8). default>0")
    ap.add_argument("--hm_thr", type=int, default=128, help="heatmap binary threshold (u8). default=128")

    ap.add_argument("--alpha_mask", type=float, default=0.25)
    ap.add_argument("--alpha_disc", type=float, default=0.35)
    ap.add_argument("--alpha_mac", type=float, default=0.35)

    ap.add_argument("--max_side", type=int, default=1600)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    hm_dir = Path(args.heatmap_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(in_dir)
    if not imgs:
        raise RuntimeError(f"No images in {in_dir}")

    for img_path in imgs:
        stem = img_path.stem

        # (A)
        A = imread_color(img_path)
        H, W = A.shape[:2]

        # (B) ROI image or ROI mask image
        b_path = find_by_stem(mask_dir, stem)
        if b_path is None:
            print(f"[SKIP] (B) not found for {img_path.name}")
            continue
        B_raw = imread_color(b_path)

        # overlay용 ROI binary 마스크
        B_gray = cv2.cvtColor(B_raw, cv2.COLOR_BGR2GRAY)
        B_gray = resize_to_wh(B_gray, (W, H), cv2.INTER_NEAREST)
        roi_bin = binarize_u8(B_gray, args.mask_thr)

        # (C) heatmap PNG (combined)
        c_path = find_by_stem(hm_dir, stem)
        if c_path is None:
            print(f"[SKIP] (C) heatmap png not found for {img_path.name}")
            continue
        C_vis = imread_color(c_path)
        C_vis = resize_to_wh(C_vis, (W, H), cv2.INTER_LINEAR)

        # channel split -> binary
        disc_u8 = C_vis[:, :, args.disc_ch]
        mac_u8  = C_vis[:, :, args.mac_ch]
        disc_bin = binarize_u8(disc_u8, args.hm_thr)
        mac_bin  = binarize_u8(mac_u8,  args.hm_thr)

        # (D) overlay on original
        D = A.copy()
        D = overlay_binary(D, roi_bin, (255, 255, 255), args.alpha_mask)  # ROI white
        D = overlay_binary(D, disc_bin, (0, 255, 0), args.alpha_disc)     # disc green
        D = overlay_binary(D, mac_bin,  (0, 0, 255), args.alpha_mac)      # mac red

        # panel compose
        A_s = fit_to_max_side(A, args.max_side)
        B_s = fit_to_max_side(B_raw, args.max_side)
        C_s = fit_to_max_side(C_vis, args.max_side)
        D_s = fit_to_max_side(D, args.max_side)

        cell_h = min(A_s.shape[0], B_s.shape[0], C_s.shape[0], D_s.shape[0])
        cell_w = min(A_s.shape[1], B_s.shape[1], C_s.shape[1], D_s.shape[1])

        panel = make_panel(A_s, B_s, C_s, D_s, cell_wh=(cell_w, cell_h))

        out_path = out_dir / f"{stem}_fig3.png"
        cv2.imwrite(str(out_path), panel)
        print(f"[SAVE] {out_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()

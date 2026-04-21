#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp


# =========================================================
# utils
# =========================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_uint8_255(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)


def overlay_heatmap_on_bgr(
    bgr: np.ndarray,
    hm01: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay single heatmap (0..1) onto BGR image with given BGR color."""
    if bgr is None:
        return bgr
    hm01 = np.clip(hm01, 0.0, 1.0).astype(np.float32)
    overlay = bgr.copy().astype(np.float32)
    mask = hm01[..., None]
    col = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay = overlay * (1.0 - alpha * mask) + col * (alpha * mask)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def overlay_two_heatmaps_rgb(
    hm_disc01: np.ndarray,
    hm_mac01: np.ndarray,
) -> np.ndarray:
    """Return BGR image: G=disc, R=mac (both 0..1)."""
    g = to_uint8_255(hm_disc01)
    r = to_uint8_255(hm_mac01)
    b = np.zeros_like(r, dtype=np.uint8)
    return np.stack([b, g, r], axis=-1)


# =========================================================
# blob candidates from connected components
# =========================================================
def find_blob_candidates(hm: np.ndarray, thr: float):
    """
    Return list of candidates for each connected component blob.
    Each candidate: dict with keys: label, area, peak(x,y,v), sum, mean, centroid(xc,yc)
    """
    bin_map = (hm >= thr).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_map, connectivity=8)
    if num <= 1:
        return []

    cands = []
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        mask = (labels == lbl)
        vals = hm[mask]
        if vals.size == 0:
            continue

        # peak inside blob
        yy, xx = np.where(mask)
        k = int(np.argmax(hm[yy, xx]))
        y_peak = int(yy[k]); x_peak = int(xx[k]); v_peak = float(hm[y_peak, x_peak])

        v_sum = float(vals.sum())
        v_mean = float(vals.mean())
        xc, yc = centroids[lbl]  # (x,y) float

        cands.append({
            "label": lbl,
            "area": area,
            "x": x_peak, "y": y_peak, "v": v_peak,
            "sum": v_sum, "mean": v_mean,
            "xc": float(xc), "yc": float(yc),
        })
    return cands


def select_macula_largest_area(
    hm_mac: np.ndarray,
    thr: float,
) -> Optional[Tuple[int, int, float]]:
    """Select macula blob ONLY by largest connected-component area.

    - Binarize heatmap by (hm >= thr)
    - Find connected components
    - Pick the component with the largest area
    - Return the peak (x,y,v) inside that component
    """
    cands = find_blob_candidates(hm_mac, thr)
    if not cands:
        return None
    best = max(cands, key=lambda c: int(c.get("area", 0)))
    return int(best["x"]), int(best["y"]), float(best["v"])


def spatial_spread(points: List[Tuple[int, int]]) -> float:
    """RMS distance to centroid (px)."""
    if len(points) <= 1:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    cx = float(xs.mean())
    cy = float(ys.mean())
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return float(np.sqrt(d2.mean()))


def find_blob_peak(hm: np.ndarray, thr: float) -> Optional[Tuple[int, int, float]]:
    """Return peak (x,y,v) among all blobs (hm>=thr). If none, return None."""
    cands = find_blob_candidates(hm, thr)
    if not cands:
        return None
    best = max(cands, key=lambda c: float(c["v"]))
    return int(best["x"]), int(best["y"]), float(best["v"])


def pick_largest_cc_and_center(bin01: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """
    Given binary (0/1) mask, keep only the largest connected component.
    Return (largest_cc_mask, center(xc,yc)) where center is centroid of CC.
    """
    if bin01 is None:
        return bin01, None
    bin01 = bin01.astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if num <= 1:
        return bin01 * 0, None
    # find largest label by area
    best_lbl = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for lbl in range(2, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_lbl = lbl
    out = (labels == best_lbl).astype(np.uint8)
    xc, yc = centroids[best_lbl]
    return out, (float(xc), float(yc))


def draw_points_on_bgr(
    bgr: np.ndarray,
    pts: List[Tuple[int, int, float]],
    color: Tuple[int, int, int],
    radius: int = 4,
    thickness: int = -1,
) -> np.ndarray:
    if bgr is None:
        return bgr
    out = bgr.copy()
    for p in pts:
        if p is None:
            continue
        x, y = int(p[0]), int(p[1])
        cv2.circle(out, (x, y), radius, color, thickness)
    return out


def draw_text(
    bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color=(0, 0, 255),
    scale: float = 0.5,
    thickness: int = 1,
):
    if bgr is None:
        return bgr
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return bgr


def draw_mac_topk_all_blobs_on_bgr(
    bgr: np.ndarray,
    mac_cands: list,
    img_size: int = 512,
    topk: int = 3,
):
    """
    Debug draw:
      - Draw TOP-K macula blob peaks among ALL blobs
        Candidates come from find_blob_candidates(hm, thr), so they are already thr-filtered blobs.
    """
    if bgr is None:
        return bgr

    font = cv2.FONT_HERSHEY_SIMPLEX

    # sort by peak value
    mac_cands_sorted = sorted(mac_cands, key=lambda c: float(c["v"]), reverse=True)
    mac_topk = mac_cands_sorted[:topk]

    # draw topk peaks
    for i, c in enumerate(mac_topk):
        x, y = int(c["x"]), int(c["y"])
        v = float(c["v"])
        area = int(c["area"])
        # marker
        cv2.circle(bgr, (x, y), 6, (0, 0, 255), -1)
        cv2.circle(bgr, (x, y), 10, (255, 255, 255), 2)
        # label
        txt = f"#{i+1} v={v:.3f} area={area}"
        cv2.putText(bgr, txt, (x + 8, y - 8), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return bgr


# =========================================================
# circle crop utils (existing)
# =========================================================
def crop_circle_bbox(
    img: np.ndarray,
    center_xy: Tuple[float, float],
    radius: float,
    pad_value=(0, 0, 0),
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    원형으로 바로 crop이 안되기 때문에 사각형 crop 먼저 한 후
    circle_mask_to_black_bgr함수에서 원형 영역 외부는 검은색으로 처리
    """
    h, w = img.shape[:2]
    cx, cy = center_xy

    side = int(math.ceil(2.0 * radius))
    if side < 2:
        side = 2

    x0 = int(round(cx - radius))
    y0 = int(round(cy - radius))
    x1 = x0 + side
    y1 = y0 + side

    # pad if out of bounds
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img_pad = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
        x0 += pad_left
        y0 += pad_top
        x1 += pad_left
        y1 += pad_top
        crop = img_pad[y0:y1, x0:x1].copy()
        center_in_crop = (cx - (x0 - pad_left), cy - (y0 - pad_top))
    else:
        crop = img[y0:y1, x0:x1].copy()
        center_in_crop = (cx - x0, cy - y0)

    return crop, center_in_crop


def circle_mask_to_black_bgr(
    crop_bgr: np.ndarray,
    center_in_crop: Tuple[float, float],
    radius: float,
    outside_value=(0, 0, 0),
) -> np.ndarray:
    """Outside circle set to outside_value."""
    h, w = crop_bgr.shape[:2]
    cx, cy = center_in_crop
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(radius)), 1, -1)
    out = crop_bgr.copy()
    out[mask == 0] = outside_value
    return out


# =========================================================
# model
# =========================================================
def load_model(ckpt_path: str, device: torch.device):
    # NOTE: Adjust model creation to match your training setup.
    # The existing file uses SMP create_model; keep as is.
    model = smp.create_model(
        arch="unet",
        encoder_name="resnet50",
        in_channels=3,
        classes=2,  # disc, macula heatmaps
    )
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        # remove possible "model." prefix
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("model.", "", 1) if k.startswith("model.") else k
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    elif isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model


# =========================================================
# mapping / TTA
# =========================================================
def apply_tta(img: np.ndarray, t: int) -> np.ndarray:
    """TTA index: 0 orig, 1 hflip, 2 vflip"""
    if t == 0:
        return img
    if t == 1:
        return cv2.flip(img, 1)
    if t == 2:
        return cv2.flip(img, 0)
    raise ValueError(t)


def undo_tta_map(hm: np.ndarray, t: int) -> np.ndarray:
    """Undo TTA on heatmap (H,W)."""
    if t == 0:
        return hm
    if t == 1:
        return cv2.flip(hm, 1)
    if t == 2:
        return cv2.flip(hm, 0)
    raise ValueError(t)


def safe_softmax_channels(x: torch.Tensor) -> torch.Tensor:
    # If model outputs logits; keep as is.
    return x


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--blob_thr", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--test_case", required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--save_heatmap", action="store_true")
    parser.add_argument("--save_binary", action="store_true")
    parser.add_argument("--save_tta_debug", action="store_true")

    # evaluation
    parser.add_argument("--gt_dir", type=str, default=None, help="Optional GT heatmap dir for eval")

    # circle crop saving
    parser.add_argument("--save_circle_crop", action="store_true")
    parser.add_argument("--d_factors", type=float, nargs="*", default=[0.6, 0.7, 0.8])

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)/ args.test_case
    # out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    overlay_dir = out_dir / "overlay"
    heatmap_dir = out_dir / "heatmap"
    binary_dir = out_dir / "binary"
    tta_debug_dir = out_dir / "tta_debug"
    csv_dir = out_dir / "csv"

    if args.save_overlay:
        ensure_dir(overlay_dir)
    if args.save_heatmap:
        ensure_dir(heatmap_dir)
    if args.save_binary:
        ensure_dir(binary_dir)
    if args.save_tta_debug:
        ensure_dir(tta_debug_dir)
    ensure_dir(csv_dir)

    # circle crop dirs
    circle_crop_root = out_dir / "circle_crop"
    if args.save_circle_crop:
        ensure_dir(circle_crop_root)

    model = load_model(args.ckpt, device=device)

    # transforms
    # Keep the same normalization values as your original file
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = A.Compose(
        [
            A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )

    rows: List[Dict[str, Any]] = []
    centers_rows: List[Dict[str, Any]] = []

    tag_mapped = {0: "T0_orig_mapped", 1: "T1_hflip_mapped", 2: "T2_vflip_mapped"}
    tag_rawdbg = {0: "T0_orig_in_rawpred", 1: "T1_hflip_in_rawpred", 2: "T2_vflip_in_rawpred"}

    img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
    if len(img_paths) == 0:
        print(f"[WARN] No images found in: {img_dir}")
        return

    for idx, p in enumerate(img_paths):
        t0 = time.time()
        name = p.stem

        bgr0 = read_image_bgr(str(p))
        H0, W0 = bgr0.shape[:2]

        # forward TTA
        preds = []
        raw_preds = []
        for t in range(3):
            bgr_t = apply_tta(bgr0, t)
            rgb_t = bgr_to_rgb(bgr_t)

            aug = transform(image=rgb_t)
            x = aug["image"].unsqueeze(0).to(device)  # 1,3,H,W
            with torch.no_grad():
                y = model(x)  # 1,2,h,w
                y = safe_softmax_channels(y)
                y = y.squeeze(0).detach().cpu().numpy()  # 2,h,w

            # resize back to original H0,W0
            disc = cv2.resize(y[0], (W0, H0), interpolation=cv2.INTER_CUBIC)
            mac = cv2.resize(y[1], (W0, H0), interpolation=cv2.INTER_CUBIC)

            # undo tta map to align with original orientation
            disc = undo_tta_map(disc, t)
            mac = undo_tta_map(mac, t)

            preds.append(np.stack([disc, mac], axis=0))
            raw_preds.append(y)

        preds = np.stack(preds, axis=0)  # T,2,H0,W0
        mean_map = preds.mean(axis=0)    # 2,H0,W0

        # per-TTA peaks (disc by peak-v, macula by largest-area blob)
        disc_pts: List[Optional[Tuple[int, int, float]]] = []
        mac_pts: List[Optional[Tuple[int, int, float]]] = []
        disc_global_pts: List[Tuple[int, int, float]] = []
        mac_global_pts: List[Tuple[int, int, float]] = []

        for t in range(preds.shape[0]):
            d = find_blob_peak(preds[t, 0], args.blob_thr)
            m = select_macula_largest_area(
                hm_mac=preds[t, 1],
                thr=args.blob_thr,
            )

            disc_pts.append(d)
            mac_pts.append(m)
            if d is not None:
                disc_global_pts.append((int(d[0]), int(d[1]), float(d[2])))
            if m is not None:
                mac_global_pts.append((int(m[0]), int(m[1]), float(m[2])))

        # mean final heatmaps (0..1 normalize for visualization)
        disc_mean = normalize01(mean_map[0])
        mac_mean = normalize01(mean_map[1])

        # binaries from mean heatmap
        disc_bin = (disc_mean >= args.threshold).astype(np.uint8)
        mac_bin = (mac_mean >= args.threshold).astype(np.uint8)

        # pick largest CC and compute centers
        disc_cc, disc_center = pick_largest_cc_and_center(disc_bin)
        mac_cc, mac_center = pick_largest_cc_and_center(mac_bin)

        # record centers
        centers_rows.append({
            "image": name,
            "disc_center_x": None if disc_center is None else disc_center[0],
            "disc_center_y": None if disc_center is None else disc_center[1],
            "mac_center_x": None if mac_center is None else mac_center[0],
            "mac_center_y": None if mac_center is None else mac_center[1],
        })

        # optional eval if gt_dir is provided
        if args.gt_dir is not None:
            # expected GT heatmap: {gt_dir}/{stem}.png or similar; keep your existing convention below if different.
            gt_path = Path(args.gt_dir) / f"{name}.png"
            if gt_path.exists():
                gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
                if gt is not None:
                    # assume GT is 2-channel packed as BGR: G=disc, R=mac (same as overlay_two_heatmaps_rgb)
                    if gt.ndim == 3 and gt.shape[2] >= 3:
                        gt_disc = gt[..., 1].astype(np.float32) / 255.0
                        gt_mac = gt[..., 2].astype(np.float32) / 255.0
                    else:
                        # fallback: single channel not supported for 2-class eval
                        gt_disc = None
                        gt_mac = None

                    if gt_disc is not None and gt_mac is not None:
                        gt_disc = cv2.resize(gt_disc, (W0, H0), interpolation=cv2.INTER_CUBIC)
                        gt_mac = cv2.resize(gt_mac, (W0, H0), interpolation=cv2.INTER_CUBIC)

                        gt_disc_bin = (gt_disc >= args.threshold).astype(np.uint8)
                        gt_mac_bin = (gt_mac >= args.threshold).astype(np.uint8)

                        # metrics on binaries (simple Dice/IoU)
                        def dice_iou(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
                            a = a.astype(bool)
                            b = b.astype(bool)
                            inter = float(np.logical_and(a, b).sum())
                            sa = float(a.sum())
                            sb = float(b.sum())
                            union = float(np.logical_or(a, b).sum())
                            dice = (2.0 * inter) / (sa + sb + 1e-8)
                            iou = inter / (union + 1e-8)
                            return dice, iou

                        d_dice, d_iou = dice_iou(disc_cc, gt_disc_bin)
                        m_dice, m_iou = dice_iou(mac_cc, gt_mac_bin)

                        rows.append({
                            "image": name,
                            "disc_dice": d_dice,
                            "disc_iou": d_iou,
                            "mac_dice": m_dice,
                            "mac_iou": m_iou,
                        })

        # save outputs
        if args.save_heatmap:
            hm_bgr = overlay_two_heatmaps_rgb(disc_mean, mac_mean)
            cv2.imwrite(str(heatmap_dir / f"{name}_heatmap.png"), hm_bgr)

        if args.save_binary:
            bin_bgr = np.zeros((H0, W0, 3), dtype=np.uint8)
            bin_bgr[..., 1] = (disc_cc * 255).astype(np.uint8)
            bin_bgr[..., 2] = (mac_cc * 255).astype(np.uint8)
            cv2.imwrite(str(binary_dir / f"{name}_binary.png"), bin_bgr)

        if args.save_overlay:
            over = bgr0.copy()
            over = overlay_heatmap_on_bgr(over, disc_mean, color=(0, 255, 0), alpha=0.35)
            over = overlay_heatmap_on_bgr(over, mac_mean, color=(0, 0, 255), alpha=0.35)

            # draw mean CC centers
            if disc_center is not None:
                cv2.circle(over, (int(round(disc_center[0])), int(round(disc_center[1]))), 6, (0, 255, 0), -1)
            if mac_center is not None:
                cv2.circle(over, (int(round(mac_center[0])), int(round(mac_center[1]))), 6, (0, 0, 255), -1)

            cv2.imwrite(str(overlay_dir / f"{name}_overlay.png"), over)

        if args.save_tta_debug:
            # per-TTA overlay with points
            for t in range(preds.shape[0]):
                over_bgr = bgr0.copy()
                dis_t = normalize01(preds[t, 0])
                mac_t = normalize01(preds[t, 1])

                over_bgr = overlay_heatmap_on_bgr(over_bgr, dis_t, color=(0, 255, 0), alpha=0.25)
                over_bgr = overlay_heatmap_on_bgr(over_bgr, mac_t, color=(0, 0, 255), alpha=0.25)

                # disc peak
                if disc_pts[t] is not None:
                    x, y, v = disc_pts[t]
                    cv2.circle(over_bgr, (int(x), int(y)), 7, (0, 255, 0), -1)
                    draw_text(over_bgr, f"D v={v:.3f}", (int(x) + 8, int(y) - 8), color=(0, 255, 0))

                # macula chosen (largest-area blob peak)
                if mac_pts[t] is not None:
                    x, y, v = mac_pts[t]
                    cv2.circle(over_bgr, (int(x), int(y)), 7, (0, 0, 255), -1)
                    draw_text(over_bgr, f"M v={v:.3f}", (int(x) + 8, int(y) - 8), color=(0, 0, 255))

                # additionally draw top-k macula blobs (by peak-v) among ALL blobs for inspection
                mac_cands = find_blob_candidates(preds[t, 1], args.blob_thr)
                over_bgr = draw_mac_topk_all_blobs_on_bgr(
                    over_bgr,
                    mac_cands=mac_cands,
                    img_size=args.img_size,
                    topk=5,
                )

                cv2.imwrite(str(tta_debug_dir / f"{name}_{tag_mapped[t]}_overlay.png"), over_bgr)

        # optional circle crop saving (existing logic)
        if args.save_circle_crop and (disc_center is not None) and (mac_center is not None):
            # example: circle crop centered at disc->mac line midpoint with radius factors
            disc_xy = np.array([disc_center[0], disc_center[1]], dtype=np.float32)
            mac_xy = np.array([mac_center[0], mac_center[1]], dtype=np.float32)

            # distance disc-mac
            dvec = mac_xy - disc_xy
            dist = float(np.linalg.norm(dvec) + 1e-6)

            for d_factor in args.d_factors:
                tag = f"d{d_factor:.2f}"
                out_sub = circle_crop_root / tag
                ensure_dir(out_sub)

                # radius from factor
                radius = float(dist * d_factor)
                center_xy = tuple(((disc_xy + mac_xy) * 0.5).tolist())

                box, center_in = crop_circle_bbox(bgr0, center_xy=center_xy, radius=radius, pad_value=(0, 0, 0))
                circ = circle_mask_to_black_bgr(box, center_in, radius, outside_value=(0, 0, 0))
                crop_out_path = out_sub / f"{name}_{tag}.png"
                cv2.imwrite(str(crop_out_path), circ)

        dt = time.time() - t0
        if (idx + 1) % 50 == 0 or idx == 0 or (idx + 1) == len(img_paths):
            print(f"[{idx+1}/{len(img_paths)}] {name} done ({dt:.3f}s)")

    # save CSVs
    # metrics
    if len(rows) > 0:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(csv_dir / "metrics.csv", index=False)
        print("Saved metrics to metrics.csv")

    # centers
    if len(centers_rows) > 0:
        import pandas as pd
        dfc = pd.DataFrame(centers_rows)
        dfc.to_csv(csv_dir / "internal_testDataset.csv", index=False)
        print("Saved center Information to internal_testDataset.csv")


if __name__ == "__main__":
    main()

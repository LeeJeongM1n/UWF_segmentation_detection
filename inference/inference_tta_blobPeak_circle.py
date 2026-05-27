#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TTA inference (disc/macula heatmap regression) with blob-peak spatial uncertainty.

Outputs:
  - overlay/{stem}_overlay_peaks.png                     (mean overlay + all D/M markers, mapped-to-orig)
  - overlay/{stem}_T0_orig_mapped_overlay.png            (preds[0] mapped-to-orig on original image)
  - overlay/{stem}_T1_hflip_mapped_overlay.png           (preds[1] mapped-to-orig on original image)
  - overlay/{stem}_T2_vflip_mapped_overlay.png           (preds[2] mapped-to-orig on original image)
  - heatmap/{stem}_heatmap.png                           (BGR, G=disc, R=mac) from mean_map
  - binary/{stem}_binary.png                             (BGR, G=disc_bin, R=mac_bin) from mean_map
  - tta_debug/{stem}_T0_orig_in_rawpred.png              (orig input-view + raw pred)
  - tta_debug/{stem}_T1_hflip_in_rawpred.png             (hflip input-view + raw pred)
  - tta_debug/{stem}_T2_vflip_in_rawpred.png             (vflip input-view + raw pred)
  - pred_npz/{stem}.npz (optional)
  - tta_metrics.csv

Key points:
  - preds[t] is ALWAYS mapped back to ORIGINAL coordinate system (unflipped)
  - raw_preds[t] stays in the COORDINATE SYSTEM of the corresponding input view
"""

import argparse
import glob
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

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
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# -------------------------
# Utils
# -------------------------
def denorm_img(img_tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] tensor -> uint8 RGB"""
    img = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay(rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(
        (1 - alpha) * rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)


def list_images(input_dir: Path):
    img_paths: List[str] = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(str(input_dir / f"*{ext}")))
    return sorted(img_paths)


def both_pred_to_bgr_u8(disc01: np.ndarray, mac01: np.ndarray) -> np.ndarray:
    """Return BGR uint8 where G=disc, R=mac, B=0."""
    g = (np.clip(disc01, 0, 1) * 255).astype(np.uint8)
    r = (np.clip(mac01, 0, 1) * 255).astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)
    return np.stack([b, g, r], axis=-1)


def both_bin_to_bgr_u8(disc_bin: np.ndarray, mac_bin: np.ndarray) -> np.ndarray:
    """Return BGR uint8 where G=disc_bin, R=mac_bin, B=0."""
    g = (disc_bin.astype(np.uint8) * 255)
    r = (mac_bin.astype(np.uint8) * 255)
    b = np.zeros_like(r, dtype=np.uint8)
    return np.stack([b, g, r], axis=-1)


def heatmap_both_to_color_rgb(disc01: np.ndarray, mac01: np.ndarray) -> np.ndarray:
    """Visualization only: disc->JET, mac->HOT, mix 50/50. Return RGB uint8."""
    d_u8 = (np.clip(disc01, 0, 1) * 255).astype(np.uint8)
    m_u8 = (np.clip(mac01, 0, 1) * 255).astype(np.uint8)

    d_col = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)  # BGR
    m_col = cv2.applyColorMap(m_u8, cv2.COLORMAP_HOT)  # BGR
    mix = cv2.addWeighted(d_col, 0.5, m_col, 0.5, 0)
    return cv2.cvtColor(mix, cv2.COLOR_BGR2RGB)


# -------------------------
# TTA inference
# -------------------------
def _infer_one(model: torch.nn.Module, img_t: torch.Tensor) -> np.ndarray:
    """Return (2,H,W) sigmoid prediction as numpy float32."""
    return torch.sigmoid(model(img_t))[0].detach().cpu().numpy().astype(np.float32)


def tta_predict_flip3(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    use_tta: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flip-based TTA: orig, hflip, vflip.

    Returns:
      preds:     (T,2,H,W)  -> ALWAYS mapped back to original coordinate (unflipped)
      mean_map:  (2,H,W)    -> mean over preds (original coordinate)
      var_map:   (2,H,W)
      raw_preds: (T,2,H,W)  -> raw predictions in each input-view coordinate
                              (t=1 is hflip coord, t=2 is vflip coord)
    """
    preds_unflip: List[np.ndarray] = []
    preds_raw: List[np.ndarray] = []

    # T0: orig
    p1 = _infer_one(model, img_t)
    preds_raw.append(p1)
    preds_unflip.append(p1)

    if use_tta:
        # T1: hflip input
        img_h = torch.flip(img_t, dims=[3])     # flip W
        p2_raw = _infer_one(model, img_h)       # raw in hflip coord
        p2_unflip = np.flip(p2_raw, axis=2)     # unflip W -> original coord
        preds_raw.append(p2_raw)
        preds_unflip.append(p2_unflip)

        # T2: vflip input
        img_v = torch.flip(img_t, dims=[2])     # flip H
        p3_raw = _infer_one(model, img_v)       # raw in vflip coord
        p3_unflip = np.flip(p3_raw, axis=1)     # unflip H -> original coord
        preds_raw.append(p3_raw)
        preds_unflip.append(p3_unflip)

    preds = np.stack(preds_unflip, axis=0).astype(np.float32)
    raw_preds = np.stack(preds_raw, axis=0).astype(np.float32)

    mean_map = preds.mean(axis=0).astype(np.float32)
    var_map = preds.var(axis=0).astype(np.float32)
    return preds, mean_map, var_map, raw_preds


# -------------------------
# Blob-peak + spatial spread
# -------------------------
def find_blob_peak(hm: np.ndarray, thr: float) -> Optional[Tuple[int, int, float]]:
    """
    Find blob-peak (x,y,val) in heatmap hm.

    - Build binary mask hm>=thr and find connected components.
    - Select component that contains global argmax. If argmax is not inside any blob,
      select the largest blob.
    - Return argmax position within selected blob.
    - If no blob exists -> return None
    """
    bin_map = (hm >= thr).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_map, connectivity=8)

    if num <= 1:
        return None

    gy, gx = np.unravel_index(int(np.argmax(hm)), hm.shape)
    lbl = int(labels[gy, gx])

    if lbl == 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        lbl = 1 + int(np.argmax(areas))

    mask = (labels == lbl)
    masked = hm * mask.astype(hm.dtype)
    y, x = np.unravel_index(int(np.argmax(masked)), hm.shape)
    return int(x), int(y), float(hm[y, x])


# for macula bob candidates
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


def _in_center_circle(x: float, y: float, cx: float, cy: float, radius: float) -> bool:
    return float((x - cx) ** 2 + (y - cy) ** 2) <= float(radius ** 2)


def select_macula_in_center_circle(
    hm_mac: np.ndarray,
    thr: float,
    center_xy: Tuple[float, float],
    radius: float,
) -> Optional[Tuple[int, int, float]]:
    """Select macula blob by peak value, but ONLY if the blob-peak lies inside a center circle."""
    cands = find_blob_candidates(hm_mac, thr)
    if not cands:
        return None

    cx, cy = center_xy
    in_c = [c for c in cands if _in_center_circle(c["x"], c["y"], cx, cy, radius)]
    if not in_c:
        return None

    best = max(in_c, key=lambda c: float(c["v"]))
    return int(best["x"]), int(best["y"]), float(best["v"])



def spatial_spread(points: List[Tuple[int, int]]) -> float:
    """RMS distance to centroid (px)."""
    if len(points) <= 1:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    mx, my = xs.mean(), ys.mean()
    return float(np.sqrt(((xs - mx) ** 2 + (ys - my) ** 2).mean()))


def draw_tta_peaks_on_overlay(
    overlay_rgb: np.ndarray,
    disc_pts,
    mac_pts,
    disc_global_pts,
    mac_global_pts,
) -> np.ndarray:
    """Draw D0(val), D1(val)... and M0(val)... on overlay."""
    out = cv2.cvtColor(overlay_rgb.copy(), cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # blob peaks
    for i, pt in enumerate(disc_pts):
        if pt is None:
            continue
        x, y, v = pt
        cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), 2)
        cv2.putText(out, f"D{i}({v:.2f})", (int(x) + 8, int(y) - 6), font, 0.5, (0, 255, 0), 1)

    for i, pt in enumerate(mac_pts):
        if pt is None:
            continue
        x, y, v = pt
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), 2)
        cv2.putText(out, f"M{i}({v:.2f})", (int(x) + 8, int(y) + 16), font, 0.5, (0, 0, 255), 1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_mac_topk_all_blobs_with_center_circle_on_bgr(
    bgr: np.ndarray,
    mac_cands: list,
    center_xy: Tuple[float, float],
    img_size: int = 512,
    topk: int = 3,
):
    """
    Debug draw:
      - Draw the 0.8*img_size center circle boundary
      - Draw TOP-K macula blob peaks among ALL blobs (NO center-circle filtering)
        Candidates come from find_blob_candidates(hm, thr), so they are already thr-filtered blobs.
    """
    if bgr is None:
        return bgr

    font = cv2.FONT_HERSHEY_SIMPLEX

    # draw center circle boundary
    cx, cy = map(int, map(round, center_xy))
    radius = int(round(img_size * 0.4))
    cv2.circle(bgr, (cx, cy), radius, (255, 255, 255), 2)

    if not mac_cands:
        return bgr

    # sort ALL blobs by peak value and take top-k
    mac_cands = sorted(mac_cands, key=lambda c: float(c["v"]), reverse=True)[:topk]

    for i, c in enumerate(mac_cands):
        x, y, v = int(c["x"]), int(c["y"]), float(c["v"])
        cv2.circle(bgr, (x, y), 6, (0, 0, 255), 2)
        cv2.putText(
            bgr, f"M{i}({v:.2f})",
            (x + 8, y + 12),
            font, 0.5, (0, 0, 255), 1
        )
    return bgr



def mean_pairwise_distance(points: List[Tuple[int, int]]) -> float:
    """Average pairwise Euclidean distance (px). For 3 points -> mean of 3 pairs."""
    if len(points) <= 1:
        return 0.0
    pts = [(float(x), float(y)) for (x, y) in points]
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            dists.append(float(np.hypot(dx, dy)))
    return float(np.mean(dists)) if dists else 0.0



def get_global_max_point(hm: np.ndarray):
    y, x = np.unravel_index(np.argmax(hm), hm.shape)
    v = float(hm[y, x])
    return int(x), int(y), v


def fmt_xys(pts) -> str:
    # pts: List[Optional[Tuple[x,y,v]]]
    return "|".join(
        f"{x},{y}" for p in pts if p is not None for (x, y, _) in [p]
    )


def fmt_vals(pts) -> str:
    return "|".join(
        f"{v:.6f}" for p in pts if p is not None for (_, _, v) in [p]
    )


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", default="./infer_results")
    parser.add_argument("--test_case", required=True)

    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--encoder", default="efficientnet-b0")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--blob_thr", default=None, type=float)
    parser.add_argument("--alpha", default=0.45, type=float)

    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta_conf_thr", default=0.5, type=float)
    parser.add_argument("--tta_peak_spread_thr", default=15.0, type=float)

    parser.add_argument(
        "--save_pred_npz",
        action="store_true",
        help="save preds(T,2,H,W), raw_preds(T,2,H,W), mean_map(2,H,W), var_map(2,H,W) as npz per image",
    )

    args = parser.parse_args()
    if args.blob_thr is None:
        args.blob_thr = args.threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_dir) / args.test_case
    out_root.mkdir(parents=True, exist_ok=True)

    overlay_dir = out_root / "overlay"
    heatmap_dir = out_root / "heatmap"
    binary_dir = out_root / "binary"
    tta_dbg_dir = out_root / "tta_debug"

    overlay_dir.mkdir(exist_ok=True)
    heatmap_dir.mkdir(exist_ok=True)
    binary_dir.mkdir(exist_ok=True)
    tta_dbg_dir.mkdir(exist_ok=True)

    npz_dir = out_root / "pred_npz"
    if args.save_pred_npz:
        npz_dir.mkdir(exist_ok=True)

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=2,
        activation=None,
    )
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    transform = A.Compose(
        [
            A.Resize(args.img_size, args.img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    rows: List[Dict[str, Any]] = []

    # center circle constraint for macula selection
    cx = float(args.img_size) * 0.5
    cy = float(args.img_size) * 0.5
    radius = float(args.img_size) * 0.4  # diameter = 0.8 * img_size

    tag_mapped = {0: "T0_orig_mapped", 1: "T1_hflip_mapped", 2: "T2_vflip_mapped"}
    tag_rawdbg = {0: "T0_orig_in_rawpred", 1: "T1_hflip_in_rawpred", 2: "T2_vflip_in_rawpred"}

    with torch.no_grad():
        for img_path in tqdm(list_images(Path(args.input_dir)), desc="Inference"):
            img_path = str(img_path)
            name = Path(img_path).stem

            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_t = transform(image=rgb)["image"].unsqueeze(0).to(device)

            preds, mean_map, var_map, raw_preds = tta_predict_flip3(model, img_t, use_tta=args.tta)

            # original image (denormed) once
            img_denorm = denorm_img(img_t[0])

            # -------------------------
            # 1) per-TTA blob peaks in ORIGINAL coordinate (from preds[t])
            # -------------------------
            disc_pts: List[Optional[Tuple[int, int, float]]] = []
            mac_pts: List[Optional[Tuple[int, int, float]]] = []
            disc_global_pts: List[Tuple[int, int, float]] = []
            mac_global_pts: List[Tuple[int, int, float]] = []

            for t in range(preds.shape[0]):
                d = find_blob_peak(preds[t, 0], args.blob_thr)
                m = select_macula_in_center_circle(
                    hm_mac=preds[t, 1],
                    thr=args.blob_thr,
                    center_xy=(cx, cy),
                    radius=radius,
                )

                disc_pts.append(d)
                mac_pts.append(m)

                disc_global_pts.append(get_global_max_point(preds[t, 0]))
                mac_global_pts.append(get_global_max_point(preds[t, 1]))

                # print prediction max
                disc_global_max = float(np.max(preds[t, 0]))
                mac_global_max = float(np.max(preds[t, 1]))
                print(
                    f"[{name}] TTA-{t} | "
                    f"Disc global max={disc_global_max:.4f}, "
                    f"Macula global max={mac_global_max:.4f}"
                )

            # -------------------------
            # 2) TTA stability/confidence (None-safe)
            # -------------------------
            disc_xy = [(x, y) for p in disc_pts if p is not None for (x, y, _) in [p]]
            mac_xy = [(x, y) for p in mac_pts if p is not None for (x, y, _) in [p]]

            disc_spread = spatial_spread(disc_xy)
            mac_spread = spatial_spread(mac_xy)

            disc_vals = [v for p in disc_pts if p is not None for (_, _, v) in [p]]
            mac_vals = [v for p in mac_pts if p is not None for (_, _, v) in [p]]

            disc_conf = float(np.mean(disc_vals)) if len(disc_vals) > 0 else 0.0
            mac_conf = float(np.mean(mac_vals)) if len(mac_vals) > 0 else 0.0

            disc_status = "Good"
            if disc_conf < args.tta_conf_thr:
                disc_status = "Low Confidence"
            elif disc_spread > args.tta_peak_spread_thr:
                disc_status = "Unstable (Spatial Spread)"

            mac_status = "Good"
            if mac_conf < args.tta_conf_thr:
                mac_status = "Low Confidence"
            elif mac_spread > args.tta_peak_spread_thr:
                mac_status = "Unstable (Spatial Spread)"

            # -------------------------
            # 3) Save mean heatmap/binary (original coord)
            # -------------------------
            heatmap_path = heatmap_dir / f"{name}_heatmap.png"
            # cv2.imwrite(str(heatmap_path), both_pred_to_bgr_u8(mean_map[0], mean_map[1]))
            H, W = mean_map.shape[1], mean_map.shape[2]
            mask = np.zeros((H, W), dtype=np.uint8)

            # center circle: diameter=0.8*img_size => radius=0.4*img_size
            cx_i, cy_i = int(round(cx)), int(round(cy))
            r_i = int(round(args.img_size * 0.4))
            cv2.circle(mask, (cx_i, cy_i), r_i, 1, -1)  # 1 inside, 0 outside

            disc_in = mean_map[0] * mask
            mac_in  = mean_map[1] * mask
            cv2.imwrite(str(heatmap_path), both_pred_to_bgr_u8(disc_in, mac_in))



            binary_path = binary_dir / f"{name}_binary.png"
            disc_bin = (mean_map[0] >= args.threshold).astype(np.uint8)
            mac_bin = (mean_map[1] >= args.threshold).astype(np.uint8)
            cv2.imwrite(str(binary_path), both_bin_to_bgr_u8(disc_bin, mac_bin))

            # -------------------------
            # 4) Save overlay (mean_map) on ORIGINAL image  (mapped-to-orig)  [기존 동작 유지]
            # -------------------------
            heat_rgb = heatmap_both_to_color_rgb(mean_map[0], mean_map[1])
            over = overlay(img_denorm, heat_rgb, args.alpha)
            over = draw_tta_peaks_on_overlay(over, disc_pts, mac_pts, disc_global_pts, mac_global_pts)

            overlay_path = overlay_dir / f"{name}_overlay_peaks.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

            # -------------------------
            # 5) Save per-TTA overlays mapped-to-orig (preds[t]) on ORIGINAL image
            # -------------------------
            for t in range(preds.shape[0]):
                heat_rgb_t = heatmap_both_to_color_rgb(preds[t, 0], preds[t, 1])
                over_t = overlay(img_denorm, heat_rgb_t, args.alpha)  # always original image
                over_t = draw_tta_peaks_on_overlay(
                    over_t,
                    disc_pts=[disc_pts[t]],
                    mac_pts=[mac_pts[t]],
                    disc_global_pts=[],
                    mac_global_pts=[],
                )
                out_path = overlay_dir / f"{name}_{tag_mapped.get(t, f'T{t}')}_overlay.png"
                # cv2.imwrite(str(out_path), cv2.cvtColor(over_t, cv2.COLOR_RGB2BGR))

            # -------------------------
            # 6) tta_debug: show RAW preds on each INPUT view (flip-input visualization)
            #    - view: orig/hflip/vflip image
            #    - heat: raw_preds[t] (same coordinate as view)
            #    - peaks: computed on raw_preds[t] (same coordinate)
            # -------------------------
            for t in range(raw_preds.shape[0]):
                if t == 0:
                    view_rgb = img_denorm
                elif t == 1:
                    view_rgb = img_denorm[:, ::-1, :]  # hflip view
                elif t == 2:
                    view_rgb = img_denorm[::-1, :, :]  # vflip view
                else:
                    view_rgb = img_denorm

                d_raw = find_blob_peak(raw_preds[t, 0], args.blob_thr)
                # m_raw = find_blob_peak(raw_preds[t, 1], args.blob_thr)
                # pick top 3 macula candidates
                mac_cands = find_blob_candidates(raw_preds[t, 1], args.blob_thr)


                heat_raw_rgb = heatmap_both_to_color_rgb(raw_preds[t, 0], raw_preds[t, 1])
                over_raw = overlay(view_rgb, heat_raw_rgb, args.alpha)
                # over_raw = draw_tta_peaks_on_overlay(
                #     over_raw,
                #     disc_pts=[d_raw],
                #     mac_pts=[m_raw],
                #     disc_global_pts=[],
                #     mac_global_pts=[],
                # )

                dbg_path = tta_dbg_dir / f"{name}_{tag_rawdbg.get(t, f'T{t}')}.png"
                # cv2.imwrite(str(dbg_path), cv2.cvtColor(over_raw, cv2.COLOR_RGB2BGR))

                over_bgr = cv2.cvtColor(over_raw, cv2.COLOR_RGB2BGR)
                # disc ( only pick 1 )
                if d_raw is not None:
                    x, y, v = d_raw
                    cv2.circle(over_bgr, (int(x), int(y)), 6, (0, 255, 0), 2)
                    cv2.putText(over_bgr, f"D({v:.2f})", (int(x)+8, int(y)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # macula (top-k : blob_peak max value within center circle; NO dist prior)
                over_bgr = draw_mac_topk_all_blobs_with_center_circle_on_bgr(
                    over_bgr,
                    mac_cands=mac_cands,
                    center_xy=(cx, cy),
                    img_size=args.img_size,
                    topk=5,
                )

                cv2.imwrite(str(dbg_path), over_bgr)

            # -------------------------
            # 7) Save npz (optional)
            # -------------------------
            npz_path = ""
            if args.save_pred_npz:
                npz_path = str(npz_dir / f"{name}.npz")
                np.savez_compressed(
                    npz_path,
                    preds=preds.astype(np.float32),         # mapped-to-orig
                    raw_preds=raw_preds.astype(np.float32), # raw in each input-view coord
                    mean_map=mean_map.astype(np.float32),
                    var_map=var_map.astype(np.float32),
                    stem=name,
                    img_path=img_path,
                    tta=int(bool(args.tta)),
                    threshold=float(args.threshold),
                    blob_thr=float(args.blob_thr),
                )

            # -------------------------
            # 8) CSV row (None-safe)
            # -------------------------
            # requested 4 metrics per image
            disc_xy_pts = [(x, y) for p in disc_pts if p is not None for (x, y, _) in [p]]
            mac_xy_pts = [(x, y) for p in mac_pts if p is not None for (x, y, _) in [p]]

            rows.append(
                {
                    "stem": name,
                    "macula_max_pred_T0": float(np.max(preds[0, 1])),
                    "macula_tta_mean_dist": mean_pairwise_distance(mac_xy_pts),
                    "disc_max_pred_T0": float(np.max(preds[0, 0])),
                    "disc_tta_mean_dist": mean_pairwise_distance(disc_xy_pts),
                }
            )

    csv_path = out_root / "tta_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"[Done] saved results to {out_root}")
    print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()

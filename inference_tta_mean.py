#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import csv
import math
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# -------------------------
# Constants
# -------------------------
IMG_EXTS = (".png", ".jpg")


# -------------------------
# Utils
# -------------------------
def denorm_img(img_tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] tensor -> uint8 RGB"""
    img = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_DEFAULT_STD, dtype=np.float32) + np.array(IMAGENET_DEFAULT_MEAN, dtype=np.float32)
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
      preds:     (T,2,H,W) 
      mean_map:  (2,H,W)   
      var_map:   (2,H,W)
      raw_preds: (T,2,H,W)  -> raw predictions in each input-view coordinate
                              (t=0 : original, t=1 : hflip, t=2 : vflip)
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
    # in_c = [c for c in cands if _in_center_circle(c["x"], c["y"], cx, cy, radius)]
    in_c = cands
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
    # cx, cy = map(int, map(round, center_xy))
    # radius = int(round(img_size * 0.4))
    # cv2.circle(bgr, (cx, cy), radius, (255, 255, 255), 2)

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


def pick_largest_cc_and_center(bin_mask01: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
    """Pick largest connected component (excluding background) and return (cc_mask01, (cx,cy)).
    - bin_mask01: uint8 or bool mask with values {0,1}
    - cc_mask01: uint8 mask {0,1} for the selected component
    - center: centroid (x,y) in pixel coordinates (float)
    """
    if bin_mask01 is None:
        return None, None
    bm = (bin_mask01 > 0).astype(np.uint8)
    if bm.max() == 0:
        return None, None

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bm, connectivity=8)
    if num <= 1:
        return None, None

    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    cc_mask = (labels == k).astype(np.uint8)

    cx, cy = centroids[k]  # (x,y)
    return cc_mask, (float(cx), float(cy))


def draw_final_centers_on_overlay(
    overlay_rgb: np.ndarray,
    disc_center: Optional[Tuple[float, float]],
    mac_center: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Draw final centers (DF, MF) on overlay."""
    out = cv2.cvtColor(overlay_rgb.copy(), cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if disc_center is not None:
        x, y = disc_center
        cv2.circle(out, (int(round(x)), int(round(y))), 7, (0, 255, 0), -1)
        cv2.putText(out, "DF", (int(round(x)) + 8, int(round(y)) - 8), font, 0.6, (0, 255, 0), 2)

    if mac_center is not None:
        x, y = mac_center
        cv2.circle(out, (int(round(x)), int(round(y))), 7, (0, 0, 255), -1)
        cv2.putText(out, "MF", (int(round(x)) + 8, int(round(y)) + 18), font, 0.6, (0, 0, 255), 2)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)



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
# GT Dice & IOU Score Metrics
# -------------------------

def find_gt_heatmap_path(gt_dir: Path, stem: str) -> Optional[Path]:
    """
    Try common GT heatmap naming patterns inside gt_dir.
    Returns the first existing path, else None.
    """
    candidates = [
        gt_dir / f"{stem}_heatmap.png",
        gt_dir / f"{stem}.png",
        gt_dir / f"{stem}_heatmap.jpg",
        gt_dir / f"{stem}.jpg",
        gt_dir / f"{stem}_heatmap.jpeg",
        gt_dir / f"{stem}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_bgr_heatmap_as_disc_mac01(gt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read heatmap GT image (BGR) where G=disc, R=mac, values are 0..255.
    Return disc01, mac01 in float32 [0,1].
    """
    bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read GT heatmap: {gt_path}")
    disc01 = (bgr[:, :, 1].astype(np.float32) / 255.0)  # G
    mac01  = (bgr[:, :, 2].astype(np.float32) / 255.0)  # R
    return disc01, mac01


def dice_iou_from_binary(pred01: np.ndarray, gt01: np.ndarray) -> Tuple[float, float]:
    """
    pred01/gt01: {0,1} uint8/bool arrays.
    Returns (dice, iou). Empty-handling:
      - if both empty: dice=1, iou=1
      - if one empty:  dice=0, iou=0
    """
    p = (pred01 > 0).astype(np.uint8)
    g = (gt01 > 0).astype(np.uint8)

    p_sum = int(p.sum())
    g_sum = int(g.sum())
    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0, 0.0

    inter = int((p & g).sum())
    union = int((p | g).sum())
    dice = (2.0 * inter) / (p_sum + g_sum) if (p_sum + g_sum) > 0 else 0.0
    iou  = (inter / union) if union > 0 else 0.0
    return float(dice), float(iou)

# predicted disc / macula center -> resize & match to original image size 
def _map_center_to_orig(center_xy, orig_w, orig_h, resized_w, resized_h):
    """(x,y) in resized(img_size) -> (x,y) in original image"""
    if center_xy is None:
        return None
    x, y = center_xy
    x0 = x * (orig_w / float(resized_w))
    y0 = y * (orig_h / float(resized_h))
    return (int(round(x0)), int(round(y0)))

def _fmt_center(c):
    return "" if c is None else f"{c[0]},{c[1]}"


# -------------------------
# Visualize - ETDRS 7-field
# -------------------------

def draw_etdrs_7field_outline(
    bgr_img: np.ndarray,
    disc_xy: Tuple[int, int],
    mac_xy: Tuple[int, int],
    color=(255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    bgr_img: 원본 BGR 이미지
    disc_xy, mac_xy: 원본 좌표계 (x,y) 픽셀
    - 반지름 R = ||macula - disc|| (원본 픽셀 거리)
    - 7개 원의 중심은 ETDRSGenerator 로직과 동일
    """
    out = bgr_img.copy()

    disc = np.array(disc_xy, dtype=np.float32)
    mac = np.array(mac_xy, dtype=np.float32)

    vec = mac - disc
    R = float(np.linalg.norm(vec))
    if R < 1.0:
        return out

    u = vec / R
    v = np.array([-u[1], u[0]], dtype=np.float32) # superior

    centers = {}
    centers[1] = disc
    centers[2] = mac
    centers[3] = mac + (u * R)
    centers[4] = mac + (u * 0.5 * R) + (v * 0.866 * R)
    centers[5] = mac + (u * 0.5 * R) - (v * 0.866 * R)
    centers[6] = disc + (v * R)
    centers[7] = disc - (v * R)

    R_i = int(round(R))
    for k in range(1, 8):
        cx, cy = centers[k]
        cv2.circle(out, (int(round(cx)), int(round(cy))), R_i, color, thickness)

    return out

# -------------------------
# Visualize - Zone
# -------------------------

def draw_ddf_zones(
    bgr_img: np.ndarray,
    disc_xy: Tuple[int, int],
    fovea_xy: Tuple[int, int],
    color=(0, 255, 255),      # 노랑(BGR)
    thickness: int = 1,
    draw_circles = True,
    disc_circle=None,  # (cx,cy,r) in orig coords
    mac_circle=None,   # (cx,cy,r) in orig coords
) -> np.ndarray:
    """
    Zone 1/2/3 + 2/3 quadrant(2S/2T/2I/2N, 3S/3T/3I/3N) 시각화.
    - DDF = ||Disc - Fovea||
    - Zone1: 원(반지름 1.5*DDF, 중심 fovea)
    - Zone2: 고리(1.5*DDF ~ 3.0*DDF) + 대각선 2개(45°,135°)
    - Zone3: 3.0*DDF 바깥(그림은 외곽선 + 라벨)
    """
    out = bgr_img.copy()
    H, W = out.shape[:2]

    disc = np.array(disc_xy, dtype=np.float32)
    fov  = np.array(fovea_xy, dtype=np.float32)

    vec_df = fov - disc
    DDF = float(np.linalg.norm(vec_df))
    if DDF < 1.0:
        return out

    # u: temporal 방향(Disc->Fovea), v: superior 방향(영상 좌표계 기준)
    u = vec_df / DDF                              # DDF axis 
    v = np.array([-u[1], u[0]], dtype=np.float32) # vertical axis
    # debug horizontal axis 
    dx = float(fov[0] - disc[0])
    dy = float(fov[1] - disc[1])
    ang = math.degrees(math.atan2(dy, dx))
    print(f"[ZONES] dx={dx:.2f}, dy={dy:.2f}")
    print(f"[ZONES] disc={disc_xy}, fov={fovea_xy}, DDF={DDF:.2f}, angle={ang:.2f}")

    
    r1 = 1.5 * DDF
    r2 = 3.0 * DDF

    # --- circles (Zone1 boundary, Zone2 boundary)
    cv2.circle(out, (int(round(fov[0])), int(round(fov[1]))), int(round(r1)), color, thickness)
    cv2.circle(out, (int(round(fov[0])), int(round(fov[1]))), int(round(r2)), color, thickness)


    # --- diagonal lines: u(DDF axis=disc->fovea) 기준 2개 직선(±45°)을 그리고, Zone1 내부는 지움 ----
    # 교점: macula bbox center (= fovea_xy)
    c = math.sqrt(0.5)  # cos45 = sin45
    dir_45  = (u * c + v * c)        # 45°
    dir_135 = (-u * c + v * c)       # 135°

    line_layer = out.copy()
    L = max(W, H) * 2

    def draw_full_line(img, center, direction, col, th):
        p0 = center - direction * L
        p1 = center + direction * L

        pt0 = (int(round(p0[0])), int(round(p0[1])))
        pt1 = (int(round(p1[0])), int(round(p1[1])))

        # 직선을 "왜곡 없이" 이미지 사각형 안으로 클리핑
        ok, c0, c1 = cv2.clipLine((0, 0, W, H), pt0, pt1)
        if not ok:
            return

        cv2.line(img, c0, c1, col, th, cv2.LINE_AA)


    # debug: horizontal axis (red)
    # cv2.line(
    #     out,
    #     (int(round(disc[0])), int(round(disc[1]))),
    #     (int(round(fov[0])),  int(round(fov[1]))),
    #     (0, 0, 255), 2, cv2.LINE_AA
    # )

    # diagonal line 2개: 교점은 fov(=macula bbox center)
    draw_full_line(line_layer, fov, dir_45,  color, thickness)
    draw_full_line(line_layer, fov, dir_135, color, thickness)

    # --- Zone1 내부에서는 대각선을 그리지 않음 ---
    r1_i = int(round(r1))
    gap = max(2, thickness + 2)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (int(round(fov[0])), int(round(fov[1]))), r1_i + gap, 1, -1)
    line_layer[mask == 1] = out[mask == 1]
    out = line_layer

    # --- optional: disc/fovea overlay 
    if draw_circles:
        # disc circle (green)
        if disc_circle is not None:
            cx, cy, r = disc_circle
            cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
            if r > 0:
                cv2.circle(out, (cx, cy), r, (0, 255, 0), thickness, cv2.LINE_AA)

        # mac circle (red)
        if mac_circle is not None:
            cx, cy, r = mac_circle
            cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
            if r > 0:
                cv2.circle(out, (cx, cy), r, (0, 0, 255), thickness, cv2.LINE_AA)

    return out


def circle_from_binmask(bin01: np.ndarray):
    if bin01 is None:
        return None
    m = (bin01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return (cx, cy, r)


# -------------------------
# Visaulize - summary Images
# -------------------------
def normalize_angle_leq_90(angle_deg: float) -> float:
    while angle_deg <= -90.0:
        angle_deg += 180.0
    while angle_deg > 90.0:
        angle_deg -= 180.0
    return angle_deg


def rotate_image_and_points(
    img: np.ndarray,
    pts_xy: np.ndarray,          # (N,2) [x,y]
    angle_deg: float,
    pivot_xy: Tuple[float, float],
    border_value=(0, 0, 0),
    interp: int = cv2.INTER_LINEAR,
):
    h, w = img.shape[:2]
    px, py = float(pivot_xy[0]), float(pivot_xy[1])

    M = cv2.getRotationMatrix2D((px, py), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    ones = np.ones((pts_xy.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts_xy.astype(np.float32), ones], axis=1)
    pts_rot = (M.astype(np.float32) @ pts_h.T).T
    return rotated, pts_rot


def crop_min_box_for_circle_with_padding(
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
    if side % 2 == 0:
        side += 1

    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = x0 + side
    y1 = y0 + side

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)

    if any(p > 0 for p in (pad_l, pad_t, pad_r, pad_b)):
        img = cv2.copyMakeBorder(
            img, pad_t, pad_b, pad_l, pad_r,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )
        x0 += pad_l; x1 += pad_l
        y0 += pad_t; y1 += pad_t
        cx += pad_l
        cy += pad_t

    crop = img[y0:y1, x0:x1].copy()
    center_in_crop = (cx - x0, cy - y0)
    return crop, center_in_crop


def circle_mask_to_black_bgr(
    crop_bgr: np.ndarray,
    center_in_crop: Tuple[float, float],
    radius: float,
    outside_value=(0, 0, 0),
) -> np.ndarray:
    """Outside circle -> black."""
    h, w = crop_bgr.shape[:2]
    cx, cy = center_in_crop

    yy, xx = np.ogrid[:h, :w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

    out = crop_bgr.copy()
    out[~inside] = np.array(outside_value, dtype=out.dtype)
    return out


def _tag_from_factor(k: float) -> str:
    # 3.0 -> d3, 3.5 -> d3p5
    if float(k).is_integer():
        return f"d{int(k)}"
    s = f"{k:g}".replace(".", "p")
    return f"d{s}"


def _resize_to_height(bgr: np.ndarray, target_h: int) -> np.ndarray:
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if h == target_h:
        return bgr
    new_w = int(round(w * (target_h / float(h))))
    return cv2.resize(bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _hconcat_with_gaps(panels: List[np.ndarray], gap: int = 16, gap_color=(255, 255, 255)) -> np.ndarray:
    # assumes all same height
    panels = [p for p in panels if p is not None]
    if not panels:
        return None
    h = panels[0].shape[0]
    sep = np.full((h, gap, 3), gap_color, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = cv2.hconcat([out, sep, p])
    return out



# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", default="./infer_results")
    parser.add_argument("--gt_dir", default=None, type=str,
                    help="Optional. If set, compare predicted heatmap vs GT heatmap in this directory and compute Dice/IoU per class.")

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
    parser.add_argument(
        "--csv_name",
        default="internal_testDataset.csv",
        type=str,
        help="CSV filename to save (columns: image_fname,width,height,macula_center,disc_center)",
    )

    # cricle crop arguments
    parser.add_argument("--d_factors", type=float, nargs="+", default=None,
        help="One or more d-factors for circle crop on ROTATED image (e.g. 3 5 6). If not set, summary is skipped.")
    parser.add_argument("--save_summary", action="store_true", help="If set, save summary image per input: [orig | zones | (rot-crops...) | etdrs].")


    args = parser.parse_args()
    if args.blob_thr is None:
        args.blob_thr = args.threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_dir) / args.test_case
    out_root.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_dir) if args.gt_dir is not None else None

    overlay_dir = out_root / "overlay"
    heatmap_dir = out_root / "heatmap"
    binary_dir = out_root / "binary"
    tta_dbg_dir = out_root / "tta_debug"
    etdrs_dir = out_root / "etdrs"
    zones_dir = out_root / "zones"
    summary_dir = out_root / "summary"
    circle_crop_root = out_root / "circle_crop"

    overlay_dir.mkdir(exist_ok=True)
    heatmap_dir.mkdir(exist_ok=True)
    binary_dir.mkdir(exist_ok=True)
    tta_dbg_dir.mkdir(exist_ok=True)
    etdrs_dir.mkdir(exist_ok=True)
    zones_dir.mkdir(exist_ok=True)

    if args.save_summary and (args.d_factors is not None):
        summary_dir.mkdir(exist_ok=True)

    if args.d_factors is not None:
        circle_crop_root.mkdir(exist_ok=True)

        # d-factor별 폴더 미리 생성
        for k in args.d_factors:
            tag = _tag_from_factor(float(k))
            (circle_crop_root / tag).mkdir(exist_ok=True)

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
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )

    rows: List[Dict[str, Any]] = []
    centers_rows: List[Dict[str, Any]] = []


    # center circle constraint for macula selection
    cx = float(args.img_size) * 0.5
    cy = float(args.img_size) * 0.5
    radius = float(args.img_size) * 0.4  # diameter = 0.8 * img_size

    tag_mapped = {0: "T0_orig_mapped", 1: "T1_hflip_mapped", 2: "T2_vflip_mapped"}
    tag_rawdbg = {0: "T0_orig_in_rawpred", 1: "T1_hflip_in_rawpred", 2: "T2_vflip_in_rawpred"}

    disc_dices, disc_ious = [], []
    mac_dices,  mac_ious  = [], []
    n_gt_found = 0
    n_gt_missing = 0


    with torch.no_grad():
        for img_path in tqdm(list_images(Path(args.input_dir)), desc="Inference"):
            img_path = str(img_path)
            name = Path(img_path).stem

            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            orig_h, orig_w = bgr.shape[:2]

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
            # 3) Build FINAL masks from TTA-mean and save heatmap/binary (original coord)
            #    - TTA > threshold
            # -------------------------
            H, W = mean_map.shape[1], mean_map.shape[2]

            # center circle mask for macula constraint
            # circle = np.zeros((H, W), dtype=np.uint8)
            # cx_i, cy_i = int(round(cx)), int(round(cy))
            # r_i = int(round(args.img_size * 0.4))  # diameter = 0.8 * img_size
            # cv2.circle(circle, (cx_i, cy_i), r_i, 1, -1)  # 1 inside, 0 outside

            # final mean heatmaps (already in original coord)
            disc_mean = mean_map[0]
            mac_mean = mean_map[1]

            # final binary masks from mean heatmap
            disc_bin = (disc_mean > args.threshold).astype(np.uint8)
            mac_bin = (mac_mean > args.threshold).astype(np.uint8)

            # apply center circle constraint only to macula final mask
            # mac_bin = (mac_bin * circle).astype(np.uint8)

            # pick largest CC and compute centers
            disc_cc, disc_center = pick_largest_cc_and_center(disc_bin)
            mac_cc, mac_center = pick_largest_cc_and_center(mac_bin)

            # keep only the selected CC (single blob) in heatmap/binary ----
            disc_cc01 = disc_cc if disc_cc is not None else np.zeros_like(disc_bin, dtype=np.uint8)
            mac_cc01  = mac_cc  if mac_cc  is not None else np.zeros_like(mac_bin, dtype=np.uint8)

            # final heatmaps: only values inside the selected CC remain
            disc_final_hm = disc_mean * disc_cc01.astype(np.float32)
            mac_final_hm  = mac_mean  * mac_cc01.astype(np.float32)

            # final binary masks: selected CC 
            disc_final_bin = disc_cc01
            mac_final_bin  = mac_cc01

            # Save FINAL heatmap (single-blob only)
            heatmap_path = heatmap_dir / f"{name}_heatmap.png"
            cv2.imwrite(str(heatmap_path), both_pred_to_bgr_u8(disc_final_hm, mac_final_hm))

            # Save FINAL binary (single-blob only)
            binary_path = binary_dir / f"{name}_binary.png"
            # cv2.imwrite(str(binary_path), both_bin_to_bgr_u8(disc_final_bin, mac_final_bin))


            # centers are computed in args.img_size coord -> map to original (orig_w, orig_h)
            disc_center_orig = _map_center_to_orig(disc_center, orig_w, orig_h, args.img_size, args.img_size)
            mac_center_orig  = _map_center_to_orig(mac_center,  orig_w, orig_h, args.img_size, args.img_size)
            centers_rows.append({
                "image_fname": Path(img_path).name,
                "width": orig_w,
                "height": orig_h,
                "macula_center": _fmt_center(mac_center_orig),
                "disc_center": _fmt_center(disc_center_orig),
            })

            # Visualize 
            # --- save ETDRS 7-field outline  ---
            if (disc_center_orig is not None) and (mac_center_orig is not None):
                etdrs_bgr = draw_etdrs_7field_outline(
                    bgr_img=bgr,                      
                    disc_xy=disc_center_orig,          
                    mac_xy=mac_center_orig,           
                    color=(255, 255, 255),  # white
                    thickness=1,
                )
                etdrs_path = etdrs_dir / f"{name}_etdrs.png"
                # cv2.imwrite(str(etdrs_path), etdrs_bgr)

            # --- save DDF Zone ---
            disc_final_up = cv2.resize(disc_final_bin.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mac_final_up  = cv2.resize(mac_final_bin.astype(np.uint8),  (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            disc_circle = circle_from_binmask(disc_final_up)
            mac_circle  = circle_from_binmask(mac_final_up)


            if (disc_center_orig is not None) and (mac_center_orig is not None):
                zones_bgr = draw_ddf_zones(
                    bgr_img=bgr,                 
                    disc_xy=disc_center_orig,    
                    fovea_xy=mac_center_orig,   
                    color=(0, 255, 255),        # yellow
                    thickness=1,
                    draw_circles=True,
                    disc_circle=disc_circle,  # green circle
                    mac_circle=mac_circle,    # red circle
                )
                # cv2.imwrite(str(zones_dir / f"{name}_zones.png"), zones_bgr)



            if args.save_summary and (args.d_factors is not None):
                if (disc_center_orig is not None) and (mac_center_orig is not None):
                    zones_panel = zones_bgr if 'zones_bgr' in locals() else None
                    etdrs_panel = etdrs_bgr if 'etdrs_bgr' in locals() else None

                    # 1) compute rotation about macula (pivot), using disc->mac vector(horizontal axis)
                    dx = float(disc_center_orig[0] - mac_center_orig[0])
                    dy = float(disc_center_orig[1] - mac_center_orig[1])
                    angle_deg = math.degrees(math.atan2(dy, dx))
                    angle_deg = normalize_angle_leq_90(angle_deg)

                    pts = np.array(
                        [
                            [float(disc_center_orig[0]), float(disc_center_orig[1])],
                            [float(mac_center_orig[0]),  float(mac_center_orig[1])],
                        ],
                        dtype=np.float32
                    )
                    pivot = (float(mac_center_orig[0]), float(mac_center_orig[1]))

                    # 2) rotate original image and points
                    orig_rot, pts_rot = rotate_image_and_points(
                        bgr, pts, angle_deg, pivot_xy=pivot,
                        border_value=(0, 0, 0),
                        interp=cv2.INTER_LINEAR
                    )
                    disc_r = (float(pts_rot[0, 0]), float(pts_rot[0, 1]))
                    mac_r  = (float(pts_rot[1, 0]), float(pts_rot[1, 1]))

                    # 3) circle crop by distance * d_factor
                    dist = float(math.hypot(mac_r[0] - disc_r[0], mac_r[1] - disc_r[1]))
                    if dist >= 1.0:
                        crop_panels = []
                        for k in args.d_factors:
                            tag = _tag_from_factor(float(k))
                            radius = (float(k) * dist) / 2.0

                            box, center_in = crop_min_box_for_circle_with_padding(
                                orig_rot, center_xy=mac_r, radius=radius, pad_value=(0, 0, 0)
                            )
                            circ = circle_mask_to_black_bgr(box, center_in, radius, outside_value=(0, 0, 0))
                            crop_out_path = circle_crop_root / tag / f"{name}_{tag}.png"
                            # cv2.imwrite(str(crop_out_path), circ)
                            crop_panels.append(circ)

                        # 4) build summary (resize all images to same height then concat)
                        target_h = 320  # hconcat 위해 원본, zone, cropped image, ETDRS 이미지 높이 통일
                        panels = [bgr, zones_panel] + crop_panels + [etdrs_panel]
                        panels = [_resize_to_height(p, target_h) for p in panels if p is not None]
                        summary = _hconcat_with_gaps(panels, gap=20, gap_color=(255, 255, 255))

                        if summary is not None:
                            summary_path = summary_dir / f"{name}_summary.png"
                            # cv2.imwrite(str(summary_path), summary)




            # -------------------------
            # (Optional) GT metrics (Dice/IoU) on heatmap region
            # - only if --gt_dir is provided
            # -------------------------
            disc_dice = disc_iou = mac_dice = mac_iou = ""
            gt_path_str = ""

            if gt_dir is not None:
                gt_path = find_gt_heatmap_path(gt_dir, name)
                if gt_path is None:
                    print(f"[WARN] GT heatmap not found for stem={name} in {gt_dir}")
                else:
                    gt_path_str = str(gt_path)
                    gt_disc01, gt_mac01 = read_bgr_heatmap_as_disc_mac01(gt_path)

                    # Ensure GT is same H,W as prediction (pred : args.img_size x args.img_size -> (H,W))
                    if gt_disc01.shape != disc_final_bin.shape:
                        gt_disc01 = cv2.resize(gt_disc01, (W, H), interpolation=cv2.INTER_LINEAR)
                        gt_mac01  = cv2.resize(gt_mac01,  (W, H), interpolation=cv2.INTER_LINEAR)

                    #  GT (Binary)
                    gt_disc_bin = (gt_disc01 > args.threshold).astype(np.uint8)
                    gt_mac_bin  = (gt_mac01  > args.threshold).astype(np.uint8)
                    # gt_mac_bin = (gt_mac_bin * circle).astype(np.uint8)

                    # GT도 가장 큰 CC만으로 비교
                    gt_disc_cc, _ = pick_largest_cc_and_center(gt_disc_bin)
                    gt_mac_cc,  _ = pick_largest_cc_and_center(gt_mac_bin)
                    gt_disc_bin = gt_disc_cc if gt_disc_cc is not None else np.zeros_like(gt_disc_bin, dtype=np.uint8)
                    gt_mac_bin  = gt_mac_cc  if gt_mac_cc  is not None else np.zeros_like(gt_mac_bin,  dtype=np.uint8)

                    disc_dice, disc_iou = dice_iou_from_binary(disc_final_bin, gt_disc_bin)
                    mac_dice,  mac_iou  = dice_iou_from_binary(mac_final_bin,  gt_mac_bin)

                    disc_dices.append(disc_dice)
                    disc_ious.append(disc_iou)
                    mac_dices.append(mac_dice)
                    mac_ious.append(mac_iou)

            

            # -------------------------
            # 4) Save overlay (mean_map) on ORIGINAL image  (mapped-to-orig)
            # -------------------------
            heat_rgb = heatmap_both_to_color_rgb(mean_map[0], mean_map[1])
            over = overlay(img_denorm, heat_rgb, args.alpha)
            over = draw_tta_peaks_on_overlay(over, disc_pts, mac_pts, disc_global_pts, mac_global_pts)
            over = draw_final_centers_on_overlay(over, disc_center=disc_center, mac_center=mac_center)

            overlay_path = overlay_dir / f"{name}_overlay_peaks.png"
            # cv2.imwrite(str(overlay_path), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
            # -------------------------
            # 6) tta_debug: show raw preds on each INPUT view (flip-input visualization)
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
                # disc (pick only 1)
                if d_raw is not None:
                    x, y, v = d_raw
                    cv2.circle(over_bgr, (int(x), int(y)), 6, (0, 255, 0), 2)
                    cv2.putText(over_bgr, f"D({v:.2f})", (int(x)+8, int(y)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # macula (top-k : blob_peak max value within center circle)
                over_bgr = draw_mac_topk_all_blobs_with_center_circle_on_bgr(
                    over_bgr,
                    mac_cands=mac_cands,
                    center_xy=(cx, cy),
                    img_size=args.img_size,
                    topk=5,
                )

                # cv2.imwrite(str(dbg_path), over_bgr)

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
            disc_xy_pts = [(x, y) for p in disc_pts if p is not None for (x, y, _) in [p]]
            mac_xy_pts = [(x, y) for p in mac_pts if p is not None for (x, y, _) in [p]]

            rows.append(
                {
                    "stem": name,
                    "macula_max_pred_T0": float(np.max(preds[0, 1])),
                    "macula_tta_mean_dist": mean_pairwise_distance(mac_xy_pts),
                    "disc_max_pred_T0": float(np.max(preds[0, 0])),
                    "disc_tta_mean_dist": mean_pairwise_distance(disc_xy_pts),


                    # optional GT metrics
                    "gt_path": gt_path_str,
                    "disc_dice": disc_dice,
                    "disc_iou": disc_iou,
                    "mac_dice": mac_dice,
                    "mac_iou": mac_iou,
                }
            )

    csv_path = out_root / "tta_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    centers_csv_path = out_root / args.csv_name
    if centers_rows:
        fieldnames = ["image_fname", "width", "height", "macula_center", "disc_center"]
        with open(centers_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(centers_rows)

    print(f"[Centers CSV] {centers_csv_path}")


    if gt_dir is not None:
        if len(disc_dices) == 0:
            print(f"[GT METRICS] No matched GT files found in: {gt_dir}")
            print(f"             missing={n_gt_missing}")
        else:
            disc_dice_mean = float(np.mean(disc_dices))
            disc_iou_mean  = float(np.mean(disc_ious))
            mac_dice_mean  = float(np.mean(mac_dices))
            mac_iou_mean   = float(np.mean(mac_ious))

            print("\n========== GT METRICS (dataset mean) ==========")
            print(f"GT dir: {gt_dir}")
            print(f"Matched GT: {n_gt_found} | Missing GT: {n_gt_missing}")
            print(f"DISC  mean Dice: {disc_dice_mean:.6f} | mean IoU: {disc_iou_mean:.6f}")
            print(f"MAC   mean Dice: {mac_dice_mean:.6f} | mean IoU: {mac_iou_mean:.6f}")
            print("==============================================\n")

    print(f"[Done] saved results to {out_root}")
    print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()

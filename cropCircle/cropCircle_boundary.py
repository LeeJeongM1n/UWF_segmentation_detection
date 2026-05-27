#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import math
import argparse
from typing import Tuple, List

import cv2
import numpy as np


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_png(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure BGR uint8 (drop alpha if exists)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        mn, mx = float(img_f.min()), float(img_f.max())
        if mx > mn:
            img_f = (img_f - mn) / (mx - mn)
        img = (img_f * 255.0).clip(0, 255).astype(np.uint8)
    return img


def argmax_center(map2d: np.ndarray) -> Tuple[float, float]:
    idx = int(np.argmax(map2d))
    y, x = np.unravel_index(idx, map2d.shape)
    return float(x), float(y)


def get_centers_from_heatmap_bgr(
    heat_bgr: np.ndarray,
    disc_channel: str = "G",   # disc=G
    macula_channel: str = "R", # macula=R
    blur_ksize: int = 11,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ch_map = {"B": 0, "G": 1, "R": 2}
    if disc_channel not in ch_map or macula_channel not in ch_map:
        raise ValueError("disc_channel/macula_channel must be one of B,G,R")

    disc_2d = heat_bgr[:, :, ch_map[disc_channel]].astype(np.float32)
    mac_2d = heat_bgr[:, :, ch_map[macula_channel]].astype(np.float32)

    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        disc_2d = cv2.GaussianBlur(disc_2d, (blur_ksize, blur_ksize), 0)
        mac_2d = cv2.GaussianBlur(mac_2d, (blur_ksize, blur_ksize), 0)

    disc_c = argmax_center(disc_2d)
    mac_c = argmax_center(mac_2d)
    return disc_c, mac_c


def normalize_angle_leq_90(angle_deg: float) -> float:
    while angle_deg <= -90.0:
        angle_deg += 180.0
    while angle_deg > 90.0:
        angle_deg -= 180.0
    return angle_deg


def rotate_image_and_points(
    img: np.ndarray,
    pts_xy: np.ndarray,
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
    """Outside circle -> black"""
    h, w = crop_bgr.shape[:2]
    cx, cy = center_in_crop

    yy, xx = np.ogrid[:h, :w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

    out = crop_bgr.copy()
    out[~inside] = np.array(outside_value, dtype=out.dtype)
    return out


def draw_centers_and_line(
    img_bgr: np.ndarray,
    disc_xy: Tuple[float, float],
    mac_xy: Tuple[float, float],
    title: str,
) -> np.ndarray:
    out = img_bgr.copy()
    d = (int(round(disc_xy[0])), int(round(disc_xy[1])))
    m = (int(round(mac_xy[0])), int(round(mac_xy[1])))

    cv2.line(out, d, m, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, d, 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(out, m, 5, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(out, "disc(G)", (d[0] + 6, d[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "macula(R)", (m[0] + 6, m[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    dx = mac_xy[0] - disc_xy[0]
    dy = mac_xy[1] - disc_xy[1]
    theta = math.degrees(math.atan2(dy, dx))

    cv2.putText(out, title, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"atan2(dy,dx)={theta:.1f} deg", (6, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_mask_binary(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    return (m > 0).astype(np.uint8)


def overlay_mask_on_image(
    img_bgr: np.ndarray,
    mask_bin: np.ndarray,
    color=(255, 0, 255),   # BGR
    alpha=0.35,
) -> np.ndarray:
    """
    img_bgr : (H,W,3) uint8
    mask_bin: (h,w) binary (0/1 or 0/255) or (h,w,3/4)
    - if mask size != image size, resize mask to image size with NEAREST
    """
    overlay = img_bgr.copy()

    # mask -> 2D
    if mask_bin.ndim == 3:
        if mask_bin.shape[2] == 4:
            mask_bin = cv2.cvtColor(mask_bin, cv2.COLOR_BGRA2GRAY)
        else:
            mask_bin = cv2.cvtColor(mask_bin, cv2.COLOR_BGR2GRAY)

    H, W = overlay.shape[:2]
    h, w = mask_bin.shape[:2]

    # resize mask if needed
    if (h != H) or (w != W):
        mask_bin = cv2.resize(mask_bin, (W, H), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask_bin > 0

    overlay_f = overlay.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)

    overlay_f[mask_bool] = (1 - alpha) * overlay_f[mask_bool] + alpha * color_arr
    overlay = overlay_f.clip(0, 255).astype(np.uint8)

    return overlay


def _tag_from_factor(k: float) -> str:
    """
    folder/tag name for factor k
    - 3.0 -> d3
    - 3.5 -> d3p5   (dot -> p)
    """
    if float(k).is_integer():
        return f"d{int(k)}"
    s = f"{k:g}"          # e.g. "3.5"
    s = s.replace(".", "p")
    return f"d{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap_dir", type=str, default="/mnt/richul_FM/UWF_seg_det/datasets/Det/inference_output/2class/sigma20/heatmap")
    ap.add_argument("--orig_dir", type=str, default="/mnt/richul_FM/UWF_seg_det/datasets/Det/YOLO/new_split/images/test")
    ap.add_argument("--out_dir", type=str, default="/mnt/richul_FM/UWF_seg_det/datasets/Det/inference_output/2class/sigma20/crop_circles_black")
    ap.add_argument("--mask_dir", type=str, default="/mnt/richul_FM/UWF_seg_det/datasets/Seg/YOLO/masks")

    ap.add_argument("--disc_channel", default="G", choices=["B", "G", "R"])
    ap.add_argument("--macula_channel", default="R", choices=["B", "G", "R"])
    ap.add_argument("--blur_ksize", default=11, type=int)
    ap.add_argument("--ext", default="png", type=str)
    ap.add_argument("--d_factors", type=float, nargs="+", default=[3.0, 5.0, 6.0], help="One or more diameter factors (e.g. 3 5 6). diameter = d_factor * dist(disc,macula).")

    ap.add_argument("--check_vis", action="store_true")
    ap.add_argument("--check_vis_after_rot", action="store_true")

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # build configs (d factor)
    configs = []
    for k in args.d_factors:
        tag = _tag_from_factor(k)
        out_k = os.path.join(args.out_dir, tag)
        ensure_dir(out_k)
        configs.append((tag, float(k), out_k))

    vis_dir = os.path.join(args.out_dir, "_check_vis")  
    if args.check_vis:
        ensure_dir(vis_dir)
    vis_rot_dir = os.path.join(args.out_dir, "_check_vis_after_rot")
    if args.check_vis_after_rot:
        ensure_dir(vis_rot_dir)

    heatmap_paths = sorted(glob.glob(os.path.join(args.heatmap_dir, f"*.png")))
    if len(heatmap_paths) == 0:
        raise RuntimeError(f"No heatmap files found in: {args.heatmap_dir}")

    angles_signed = []
    angles_abs = []


    for hp in heatmap_paths:
        stem = os.path.splitext(os.path.basename(hp))[0]
        op = os.path.join(args.orig_dir, f"{stem}.{args.ext}")
        if not os.path.exists(op):
            print(f"[SKIP] original not found: {op}")
            continue

        heat = to_bgr_uint8(read_png(hp))
        orig = to_bgr_uint8(read_png(op))

        mask_path = os.path.join(args.mask_dir, f"{stem}_mask.png")
        if not os.path.exists(mask_path):
            print(f"[SKIP] mask not found: {mask_path}")
            continue

        mask = read_mask_binary(mask_path)  # 0/1
        H, W = orig.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh != H) or (mw != W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # 1) centers
        disc_c, mac_c = get_centers_from_heatmap_bgr(
            heat,
            disc_channel=args.disc_channel,
            macula_channel=args.macula_channel,
            blur_ksize=args.blur_ksize,
        )

        # distance (disc-macula) 
        dist_pre = math.hypot(disc_c[0] - mac_c[0], disc_c[1] - mac_c[1])

        # -------------- check_vis (PRE) --------------
        if args.check_vis:
            heat_vis = draw_centers_and_line(heat, disc_c, mac_c, title=f"{stem} | HEATMAP (pre)")
            orig_vis = draw_centers_and_line(orig, disc_c, mac_c, title=f"{stem} | ORIG (pre)")

            orig_vis = overlay_mask_on_image(orig_vis, mask, color=(255, 255, 255), alpha=0.35)

            # draw planned circles for all d_factors before crop (for debugging)
            cx, cy = int(round(mac_c[0])), int(round(mac_c[1]))
            for tag, k, _ in configs:
                radius = (k * dist_pre) / 2.0
                cv2.circle(orig_vis, (cx, cy), int(round(radius)), (0, 0, 255), 2)
                cv2.putText(orig_vis, tag, (cx + int(round(radius)) + 5, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(vis_dir, f"{stem}_heat_pre.png"), heat_vis)
            cv2.imwrite(os.path.join(vis_dir, f"{stem}_orig_pre.png"), orig_vis)

        # 2) rotation angle (macula pivot)
        dx = disc_c[0] - mac_c[0]
        dy = disc_c[1] - mac_c[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        angle_deg = normalize_angle_leq_90(angle_deg)
        # --- collect angle stats (absolute) ---
        angles_signed.append(float(angle_deg))
        angles_abs.append(float(abs(angle_deg)))

        # 3) rotate
        pts = np.array([disc_c, mac_c], dtype=np.float32)
        pivot = mac_c
        orig_rot, pts_rot = rotate_image_and_points(orig, pts, angle_deg, pivot, border_value=(0, 0, 0))
        heat_rot, _       = rotate_image_and_points(heat, pts, angle_deg, pivot, border_value=(0, 0, 0))
        mask_rot, _       = rotate_image_and_points(mask, pts, angle_deg, pivot, border_value=0, interp=cv2.INTER_NEAREST)

        disc_r = (float(pts_rot[0, 0]), float(pts_rot[0, 1]))
        mac_r  = (float(pts_rot[1, 0]), float(pts_rot[1, 1]))

        if args.check_vis_after_rot:
            heat_vis2 = draw_centers_and_line(heat_rot, disc_r, mac_r, title=f"{stem} | HEATMAP (rot)")
            orig_vis2 = draw_centers_and_line(orig_rot, disc_r, mac_r, title=f"{stem} | ORIG (rot)")
            cv2.imwrite(os.path.join(vis_rot_dir, f"{stem}_heat_rot.png"), heat_vis2)
            cv2.imwrite(os.path.join(vis_rot_dir, f"{stem}_orig_rot.png"), orig_vis2)

        # 4) dist (rot)
        dy_after = mac_r[1] - disc_r[1]
        dx_after = mac_r[0] - disc_r[0]
        dist = math.hypot(dx_after, dy_after)
        if dist < 1.0:
            print(f"[SKIP] too small dist for {stem}: dist={dist:.3f}")
            continue

        # 5) crop circles for all d_factors
        for tag, k, out_dir in configs:
            diameter = k * dist
            radius = diameter / 2.0

            mask_box, mask_center_in = crop_min_box_for_circle_with_padding(
                mask_rot, mac_r, radius, pad_value=0
            )

            H, W = mask_rot.shape[:2]
            cx, cy = float(mac_r[0]), float(mac_r[1])

            yy, xx = np.ogrid[:H, :W]
            inside_full = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

            # crop 영역이 모두 mask GT 내부에 속해있는지 판단 (원의 일부 영역이 GT를 벗어나면 해당 이미지는 crop하지 않고 pass)
            if not np.all(mask_rot[inside_full] > 0):
                print(f"[SKIP-{tag}] circle exceeds GT mask for {stem}")
                continue

            orig_box, orig_center_in = crop_min_box_for_circle_with_padding(
                orig_rot, mac_r, radius, pad_value=(0, 0, 0)
            )
            heat_box, heat_center_in = crop_min_box_for_circle_with_padding(
                heat_rot, mac_r, radius, pad_value=(0, 0, 0)
            )

            orig_circ = circle_mask_to_black_bgr(orig_box, orig_center_in, radius, outside_value=(0, 0, 0))
            heat_circ = circle_mask_to_black_bgr(heat_box, heat_center_in, radius, outside_value=(0, 0, 0))

            out_orig = os.path.join(out_dir, f"{stem}_{tag}_orig.png")
            out_heat = os.path.join(out_dir, f"{stem}_{tag}_heatmap.png")
            cv2.imwrite(out_orig, orig_circ)
            cv2.imwrite(out_heat, heat_circ)

        print(f"[OK] {stem} | angle={angle_deg:.2f}deg | dy_after={dy_after:.3f}px | dist={dist:.2f}px")


    # ===============================
    # Save angle list for this run
    # ===============================
    angles_abs = np.asarray(angles_abs, dtype=np.float32)
    angles_signed = np.asarray(angles_signed, dtype=np.float32)
    npy_dir = os.path.join(args.out_dir, "npy")
    ensure_dir(npy_dir)

    np.save(os.path.join(npy_dir, "angles_abs.npy"), angles_abs)
    np.save(os.path.join(npy_dir, "angles_signed.npy"), angles_signed)

    print(
        f"[ANGLE-SAVE] Saved {len(angles_abs)} angles to "
        f"{os.path.join(npy_dir, 'angles_abs.npy')}"
    )

    print("Done.")


if __name__ == "__main__":
    main()

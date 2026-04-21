#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# -------------------------
# Parsing / I/O
# -------------------------
def parse_center_xy(s: str):
    """
    internal_testDataset.csv stores centers as "x,y" (string).
    Returns (x,y) as float, or None if empty/invalid.
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return None
    parts = s.split(",")
    if len(parts) != 2:
        return None
    try:
        x = float(parts[0].strip())
        y = float(parts[1].strip())
        return (x, y)
    except Exception:
        return None


def read_mask_as01(mask_path: Path) -> np.ndarray:
    """
    Read mask png and return binary mask {0,1} uint8.
    Foreground is assumed to be non-zero.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > 0).astype(np.uint8)


# -------------------------
# Zone definition (reproducible from centers)
# -------------------------
def build_zone_label_map(H: int, W: int, disc_xy, fovea_xy):
    """
    Zone definition:
      - DDF = ||fovea - disc||
      - Zone1: r < 1.5*DDF centered at fovea
      - Zone2: 1.5*DDF <= r < 3.0*DDF, split by +/-45deg around u-axis
      - Zone3: r >= 3.0*DDF, split likewise

    Labels:
      1 = Zone1
      21/22/23/24 = Zone2T/S/N/I
      31/32/33/34 = Zone3T/S/N/I
      0 = invalid (DDF too small)
    """
    disc = np.array(disc_xy, dtype=np.float32)
    fov  = np.array(fovea_xy, dtype=np.float32)

    vec_df = fov - disc
    DDF = float(np.linalg.norm(vec_df))
    if DDF < 1.0:
        return np.zeros((H, W), np.uint8), DDF, 0.0, 0.0

    # u: disc->fovea, v: superior (90deg CCW)
    u = vec_df / DDF
    v = np.array([-u[1], u[0]], dtype=np.float32)

    r1 = 1.5 * DDF
    r2 = 3.0 * DDF

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - float(fov[0])
    dy = yy - float(fov[1])

    rr = np.sqrt(dx * dx + dy * dy)

    a = dx * float(u[0]) + dy * float(u[1])  # along u (T/N)
    b = dx * float(v[0]) + dy * float(v[1])  # along v (S/I)

    zone_id = np.zeros((H, W), dtype=np.uint8)

    zone1 = (rr < r1)
    zone2 = (rr >= r1) & (rr < r2)
    zone3 = (rr >= r2)

    # +/-45deg boundary: compare |a| and |b|
    along_u = (np.abs(a) >= np.abs(b))  # wedges near u-axis (T/N)
    along_v = ~along_u                  # wedges near v-axis (S/I)

    T = (a < 0)
    N = (a >= 0)
    S = (b >= 0)
    I = (b < 0)

    zone_id[zone1] = 1

    # Zone2
    zone_id[zone2 & along_u & T] = 21  # Zone2T
    zone_id[zone2 & along_v & S] = 22  # Zone2S
    zone_id[zone2 & along_u & N] = 23  # Zone2N
    zone_id[zone2 & along_v & I] = 24  # Zone2I

    # Zone3
    zone_id[zone3 & along_u & T] = 31  # Zone3T
    zone_id[zone3 & along_v & S] = 32  # Zone3S
    zone_id[zone3 & along_u & N] = 33  # Zone3N
    zone_id[zone3 & along_v & I] = 34  # Zone3I

    return zone_id, DDF, r1, r2


def compute_A_B_ratio(zone_id: np.ndarray, gt01: np.ndarray):
    """
    For each zone:
      A = area(GT ∩ Zone)  (pixel count)
      B = area(Zone)       (pixel count)
      ratio = A/B
    """
    gt = (gt01 > 0)

    id2name = {
        1:  "Zone1",
        21: "Zone2T", 22: "Zone2S", 23: "Zone2N", 24: "Zone2I",
        31: "Zone3T", 32: "Zone3S", 33: "Zone3N", 34: "Zone3I",
    }

    out = {}
    for zid, name in id2name.items():
        z = (zone_id == zid)
        B = int(z.sum())
        A = int((z & gt).sum())
        ratio = float(A / B) if B > 0 else 0.0
        out[name] = (A, B, ratio)
    return out


# -------------------------
# Visualization overlay
# -------------------------
def _put_text_with_outline(img, text, org, font_scale=0.7, color=(0, 255, 255), thickness=1):
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def overlay_gt_hits_on_zonevis(
    vis_bgr: np.ndarray,
    zone_id: np.ndarray,
    gt01: np.ndarray,
    stats: dict,
    stem: str,
    DDF: float,
    alpha: float = 0.45,
    draw_contours: bool = True
) -> np.ndarray:
    """
    Overlay (GT ∩ Zone) masks on top of zone visualization image.
    Also writes A/B/ratio texts.
    - vis_bgr: zone visualization image (BGR)
    - zone_id, gt01: arrays defined on some (H,W). If sizes differ from vis, they must be resized before calling.
    """
    out = vis_bgr.copy()
    overlay = vis_bgr.copy()
    # SAME_COLOR = (255, 255, 255)

    ## Zone id -> name + color (BGR).
    zinfo = [
        (1,  "Zone1",  (0, 255, 255)),
        (21, "Zone2T", (255, 0, 0)),
        (22, "Zone2S", (0, 255, 0)),
        (23, "Zone2N", (0, 0, 255)),
        (24, "Zone2I", (255, 255, 0)),
        (31, "Zone3T", (255, 0, 255)),
        (32, "Zone3S", (0, 255, 255)),
        (33, "Zone3N", (255, 128, 0)),
        (34, "Zone3I", (0, 128, 255)),
    ]

    gt = (gt01 > 0)
    for zid, zname, color in zinfo:
        hit = (zone_id == zid) & gt  # GT ∩ Zone
        if not hit.any():
            continue

        overlay[hit] = color
        # overlay[hit] = SAME_COLOR

        if draw_contours:
            hit_u8 = (hit.astype(np.uint8) * 255)
            cnts, _ = cv2.findContours(hit_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(out, cnts, -1, color, 2)

    # alpha blend
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    # Text block
    x0, y0 = 15, 30
    dy = 26
    _put_text_with_outline(out, f"{stem}  DDF={DDF:.2f}", (x0, y0), font_scale=0.8, color=(255, 255, 255))

    header = "Zone: A=GT&Zone, B=Zone, A/B"
    _put_text_with_outline(out, header, (x0, y0 + dy), font_scale=0.7, color=(255, 255, 255))

    order = ["Zone1", "Zone2T", "Zone2S", "Zone2N", "Zone2I", "Zone3T", "Zone3S", "Zone3N", "Zone3I"]
    for i, z in enumerate(order):
        A, B, ratio = stats.get(z, (0, 0, 0.0))
        line = f"{z}: {A}/{B} ({ratio:.4f})"
        _put_text_with_outline(out, line, (x0, y0 + (i + 2) * dy), font_scale=0.75)

    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers_csv", type=str, required=True,
                    help="internal_testDataset.csv (image_fname,width,height,macula_center,disc_center).")
    ap.add_argument("--maskGT_dir", type=str, required=True,
                    help="Directory containing {stem}_mask.png")
    ap.add_argument("--out_csv", type=str, default="zone_stats.csv",
                    help="Output CSV path.")
    ap.add_argument("--mask_suffix", type=str, default="_mask.png",
                    help="Mask filename suffix. Default: _mask.png")
    ap.add_argument("--allow_resize_mask", action="store_true",
                    help="If mask size differs from (width,height) in CSV, resize mask to (width,height) with INTER_NEAREST.")

    # Optional: read zone visualization images and write annotated outputs
    ap.add_argument("--zoneVis_dir", type=str, default=None,
                    help="Optional. Directory containing zone-visualized images for each stem.")
    ap.add_argument("--zoneVis_suffix", type=str, default="_ddf_zones.png",
                    help="Filename suffix for zone visualization images. Default: _ddf_zones.png")
    ap.add_argument("--out_vis_dir", type=str, default=None,
                    help="If set, write annotated zone visualization images here.")
    ap.add_argument("--overlay_alpha", type=float, default=0.45,
                    help="Alpha for overlaying (GT ∩ Zone) on zone visualization image.")
    ap.add_argument("--no_contours", action="store_true",
                    help="If set, do not draw contours of (GT ∩ Zone) areas.")

    args = ap.parse_args()

    centers_csv = Path(args.centers_csv)
    mask_dir = Path(args.maskGT_dir)
    out_csv = Path(args.out_csv)

    zoneVis_dir = Path(args.zoneVis_dir) if args.zoneVis_dir else None
    out_vis_dir = Path(args.out_vis_dir) if args.out_vis_dir else None
    if out_vis_dir is not None:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(centers_csv)

    required_cols = ["image_fname", "width", "height", "macula_center", "disc_center"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in centers_csv: {missing}\nAvailable: {list(df.columns)}")

    rows_out = []
    n_missing_mask = 0
    n_missing_center = 0
    n_missing_vis = 0
    n_written_vis = 0

    for _, r in df.iterrows():
        img_fname = str(r["image_fname"])
        stem = Path(img_fname).stem

        W = int(r["width"])
        H = int(r["height"])

        mac = parse_center_xy(r["macula_center"])
        disc = parse_center_xy(r["disc_center"])
        if mac is None or disc is None:
            n_missing_center += 1
            continue

        mask_path = mask_dir / f"{stem}{args.mask_suffix}"
        if not mask_path.exists():
            n_missing_mask += 1
            continue

        gt01 = read_mask_as01(mask_path)

        if (gt01.shape[0] != H) or (gt01.shape[1] != W):
            if args.allow_resize_mask:
                gt01 = cv2.resize(gt01, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                raise RuntimeError(
                    f"Mask size mismatch for {stem}: mask(W,H)=({gt01.shape[1]},{gt01.shape[0]}) vs csv(W,H)=({W},{H}). "
                    f"Use --allow_resize_mask to auto-fix."
                )

        zone_id, DDF, r1, r2 = build_zone_label_map(H, W, disc_xy=disc, fovea_xy=mac)
        stats = compute_A_B_ratio(zone_id, gt01)

        out = {
            "stem": stem,
            "image_fname": img_fname,
            "mask_path": str(mask_path),
            "width": W,
            "height": H,
            "disc_center": f"{disc[0]:.4f},{disc[1]:.4f}",
            "macula_center": f"{mac[0]:.4f},{mac[1]:.4f}",
            "DDF": float(DDF),
            "r1": float(r1),
            "r2": float(r2),
        }

        for zname, (A, B, ratio) in stats.items():
            out[f"{zname}_A_gt_in_zone"] = A
            out[f"{zname}_B_zone_area"] = B
            out[f"{zname}_ratio_A_div_B"] = ratio

        rows_out.append(out)

        # ---- Optional: overlay (GT ∩ Zone) on zone visualization image and save
        if zoneVis_dir is not None and out_vis_dir is not None:
            vis_path = zoneVis_dir / f"{stem}{args.zoneVis_suffix}"
            if not vis_path.exists():
                n_missing_vis += 1
            else:
                vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
                if vis is None:
                    n_missing_vis += 1
                else:
                    vh, vw = vis.shape[:2]

                    # zone_id / gt01 are on (H,W). If vis size differs, resize both with NEAREST.
                    zone_id_vis = zone_id
                    gt01_vis = gt01
                    if (vh != H) or (vw != W):
                        zone_id_vis = cv2.resize(zone_id, (vw, vh), interpolation=cv2.INTER_NEAREST)
                        gt01_vis = cv2.resize(gt01, (vw, vh), interpolation=cv2.INTER_NEAREST)

                    vis_anno = overlay_gt_hits_on_zonevis(
                        vis_bgr=vis,
                        zone_id=zone_id_vis,
                        gt01=gt01_vis,
                        stats=stats,
                        stem=stem,
                        DDF=DDF,
                        alpha=float(args.overlay_alpha),
                        draw_contours=(not args.no_contours),
                    )

                    out_path = out_vis_dir / f"{stem}_zoneStats_overlay.png"
                    cv2.imwrite(str(out_path), vis_anno)
                    n_written_vis += 1

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)

    print(f"[Done] saved CSV: {out_csv}")
    print(f"  N={len(out_df)} | missing_center={n_missing_center} | missing_mask={n_missing_mask}")
    if zoneVis_dir is not None and out_vis_dir is not None:
        print(f"  zoneVis_dir={zoneVis_dir}")
        print(f"  wrote overlay vis: {n_written_vis} | missing vis: {n_missing_vis}")


if __name__ == "__main__":
    main()

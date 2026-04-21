#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2


def parse_center_xy(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return None
    parts = s.split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        return None


def read_mask_as01(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > 0).astype(np.uint8)


def etdrs_7_centers_and_radius(disc_xy, mac_xy):
    disc = np.array(disc_xy, np.float32)
    mac  = np.array(mac_xy,  np.float32)
    vec = mac - disc
    R = float(np.linalg.norm(vec))
    if R < 1.0:
        return None, None

    u = vec / R
    v = np.array([-u[1], u[0]], np.float32)  # superior

    centers = [
        disc,
        mac,
        mac + (u * R),
        mac + (u * 0.5 * R) + (v * 0.866 * R),
        mac + (u * 0.5 * R) - (v * 0.866 * R),
        disc + (v * R),
        disc - (v * R),
    ]
    return centers, R


def circle_mask(H, W, cx, cy, r):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)


def circle_fully_inside_image(cx, cy, r, W, H):
    return (cx - r >= 0) and (cy - r >= 0) and (cx + r < W) and (cy + r < H)


def completeness_for_row(row, mask_dir: Path, mask_suffix: str,
                         eps=1e-6, allow_resize_mask=False,
                         require_circle_inside_image=False):
    stem = Path(str(row["image_fname"])).stem
    W = int(row["width"])
    H = int(row["height"])

    disc = parse_center_xy(row["disc_center"])
    mac  = parse_center_xy(row["macula_center"])
    if disc is None or mac is None:
        return None  # skip

    mask_path = mask_dir / f"{stem}{mask_suffix}"
    if not mask_path.exists():
        return None  # skip

    gt01 = read_mask_as01(mask_path)

    if gt01.shape[:2] != (H, W):
        if not allow_resize_mask:
            return None
        gt01 = cv2.resize(gt01, (W, H), interpolation=cv2.INTER_NEAREST)

    centers, R = etdrs_7_centers_and_radius(disc, mac)
    if centers is None:
        return None

    gt = (gt01 > 0)

    ratios = []
    for c in centers:
        cx, cy = float(c[0]), float(c[1])

        if require_circle_inside_image:
            if not circle_fully_inside_image(cx, cy, R, W, H):
                return False

        cm = circle_mask(H, W, cx, cy, R)
        B = int(cm.sum())
        if B == 0:
            return False
        A = int((cm & gt).sum())
        ratios.append(A / B)

    return all(abs(r - 1.0) <= eps for r in ratios)


def compute_group(csv_path: Path, mask_dir: Path, mask_suffix: str, **kwargs):
    df = pd.read_csv(csv_path)
    need = ["image_fname", "width", "height", "disc_center", "macula_center"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Missing column {c} in {csv_path}")

    total = 0
    n_complete = 0
    n_missing_mask = 0
    n_missing_center = 0

    for _, row in df.iterrows():
        # quick center check
        disc = parse_center_xy(row["disc_center"])
        mac  = parse_center_xy(row["macula_center"])
        if disc is None or mac is None:
            n_missing_center += 1
            continue

        stem = Path(str(row["image_fname"])).stem
        mask_path = mask_dir / f"{stem}{mask_suffix}"
        if not mask_path.exists():
            print("[DEBUG repr] mask_dir =", repr(str(mask_dir)))
            print("[DEBUG repr] mask_path =", repr(str(mask_path)))
            print("[DEBUG] mask_dir exists? ", mask_dir.exists())

            n_missing_mask += 1

            # DEBUG: print first few missing cases
            if n_missing_mask <= 5:
                print(f"[MISS] csv={csv_path.name} stem={stem}")
                print(f"       expected: {mask_path}")

                # show candidates that start with the stem
                cand = sorted(mask_dir.glob(f"{stem}*"))
                if cand:
                    print("       candidates:")
                    for p in cand[:5]:
                        print("        -", p.name)
                else:
                    # show a few random masks in that folder
                    any_masks = sorted(mask_dir.glob("*.png"))[:5]
                    print("       sample masks in dir:")
                    for p in any_masks:
                        print("        -", p.name)

            continue


        ok = completeness_for_row(row, mask_dir, mask_suffix, **kwargs)
        if ok is None:
            continue
        total += 1
        if ok:
            n_complete += 1

    return n_complete, total, n_missing_mask, n_missing_center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="3 CSVs in order: [Internal, External1, External2]")
    ap.add_argument("--maskGT_dirs", nargs="+", required=True,
                    help="3 mask dirs corresponding to the CSVs")
    ap.add_argument("--mask_suffixes", nargs="+", required=True,
                    help="3 mask suffixes corresponding to the CSVs, e.g. _mask.png _Retina_O.png _Retina_O.png")

    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--allow_resize_mask", action="store_true")
    ap.add_argument("--require_circle_inside_image", action="store_true")
    args = ap.parse_args()

    if len(args.csvs) != 3 or len(args.maskGT_dirs) != 3 or len(args.mask_suffixes) != 3:
        raise RuntimeError("Need exactly 3 CSVs, 3 maskGT_dirs, and 3 mask_suffixes (Internal, External1, External2).")

    csvs = [Path(p) for p in args.csvs]
    mdirs = [Path(p) for p in args.maskGT_dirs]
    msufs = list(args.mask_suffixes)

    def pct(n, d): return 100.0 * n / d if d else 0.0

    # Internal
    n_i, tot_i, miss_i, missc_i = compute_group(
        csvs[0], mdirs[0], msufs[0],
        eps=args.eps,
        allow_resize_mask=args.allow_resize_mask,
        require_circle_inside_image=args.require_circle_inside_image
    )

    # External1
    n_e1, tot_e1, miss_e1, missc_e1 = compute_group(
        csvs[1], mdirs[1], msufs[1],
        eps=args.eps,
        allow_resize_mask=args.allow_resize_mask,
        require_circle_inside_image=args.require_circle_inside_image
    )

    # External2
    n_e2, tot_e2, miss_e2, missc_e2 = compute_group(
        csvs[2], mdirs[2], msufs[2],
        eps=args.eps,
        allow_resize_mask=args.allow_resize_mask,
        require_circle_inside_image=args.require_circle_inside_image
    )

    n_e = n_e1 + n_e2
    tot_e = tot_e1 + tot_e2

    print("==============================================")
    print(" ETDRS Completeness (All 7 fields visible)")
    print("==============================================")
    print(f"[Internal]  {n_i}/{tot_i} = {pct(n_i, tot_i):.2f}% "
          f"(missing_mask={miss_i}, missing_center={missc_i}, suffix={msufs[0]})")
    print(f"[External]  {n_e}/{tot_e} = {pct(n_e, tot_e):.2f}% "
          f"(= {n_e1}/{tot_e1} + {n_e2}/{tot_e2})")
    print(f"           External1 missing_mask={miss_e1}, missing_center={missc_e1}, suffix={msufs[1]}")
    print(f"           External2 missing_mask={miss_e2}, missing_center={missc_e2}, suffix={msufs[2]}")
    print("==============================================")


if __name__ == "__main__":
    main()

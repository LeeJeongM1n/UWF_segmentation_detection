#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import cv2

def is_normalized_yolo(vals):
    """
    vals: [x,y,w,h]
    - 모두 0~1 범위(약간의 오차 허용)면 normalized로 판단
    - 하나라도 1보다 크면 pixel 좌표로 판단
    """
    eps = 1e-6
    if any(v > 1.0 + eps for v in vals):
        return False
    # 음수나 말도 안되는 값 방지(아주 약간의 음수는 클램프 처리 가능)
    return True

def clamp01(x):
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def process_one_dataset(ds_dir: Path, out_root: Path, out_w: int, out_h: int,
                        keep_empty_label: bool, img_exts=(".png", ".jpg")):
    images_dir = ds_dir / "images"
    labels_dir = ds_dir / "labels"

    if not images_dir.is_dir():
        print(f"[SKIP] images dir not found: {images_dir}")
        return

    ds_name = ds_dir.name
    out_img_dir = out_root / ds_name / "images"
    out_lbl_dir = out_root / ds_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for ext in img_exts:
        img_paths.extend(sorted(images_dir.glob(f"*{ext}")))

    if not img_paths:
        print(f"[SKIP] no images in: {images_dir}")
        return

    print(f"[DATASET] {ds_name} | images: {len(img_paths)}")

    for img_path in img_paths:
        stem = img_path.stem
        lbl_in = labels_dir / f"{stem}.txt"
        lbl_out = out_lbl_dir / f"{stem}.txt"

        # ---- read image ----
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [WARN] cannot read image: {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]

        # ---- resize and save image ----
        resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out_img_path = out_img_dir / img_path.name
        ok = cv2.imwrite(str(out_img_path), resized)
        if not ok:
            print(f"  [WARN] failed to write image: {out_img_path}")

        # ---- handle label ----
        if not lbl_in.exists():
            # 라벨 파일이 없으면: 옵션에 따라 빈 파일 생성 or 스킵
            if keep_empty_label:
                lbl_out.write_text("", encoding="utf-8")
            continue

        txt = lbl_in.read_text(encoding="utf-8").strip()
        if txt == "":
            if keep_empty_label:
                lbl_out.write_text("", encoding="utf-8")
            continue

        out_lines = []
        bad_line = False

        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"  [WARN] malformed label line (len<5): {lbl_in} | '{line}'")
                bad_line = True
                continue

            try:
                cls = parts[0]
                x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            except Exception:
                print(f"  [WARN] cannot parse label line: {lbl_in} | '{line}'")
                bad_line = True
                continue

            # normalized인지(pixel인지) 판단
            if is_normalized_yolo([x, y, w, h]):
                xn, yn, wn, hn = x, y, w, h
            else:
                # pixel 좌표로 가정 → 원본 크기 기준으로 0~1 정규화
                # (YOLO pixel 포맷이라고 가정: center x/y, width/height in pixels)
                if orig_w <= 0 or orig_h <= 0:
                    print(f"  [WARN] invalid image size for normalization: {img_path}")
                    bad_line = True
                    continue
                xn = x / orig_w
                yn = y / orig_h
                wn = w / orig_w
                hn = h / orig_h

            xn = clamp01(xn); yn = clamp01(yn); wn = clamp01(wn); hn = clamp01(hn)

            out_lines.append(f"{cls} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}")

        # 라벨 저장
        # bad_line이 있어도 파싱 가능한 라인들은 저장
        lbl_out.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=str, default="/mnt/richul_FM/YOLO/external_datasets",
                    help="source root containing {Dataset_name}/images,labels,heatmap")
    ap.add_argument("--dst_root", type=str, default="/mnt/richul_FM/UWF_seg_det/datasets/Det",
                    help="destination root: {Dataset_name}/images and {Dataset_name}/labels will be created")
    ap.add_argument("--out_w", type=int, default=780)
    ap.add_argument("--out_h", type=int, default=614)
    ap.add_argument("--keep_empty_label", action="store_true",
                    help="if set, create empty txt when label missing/empty (recommended for YOLO training)")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not src_root.is_dir():
        raise FileNotFoundError(f"src_root not found: {src_root}")

    ds_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])
    if not ds_dirs:
        print(f"[DONE] no dataset folders under: {src_root}")
        return

    for ds_dir in ds_dirs:
        process_one_dataset(
            ds_dir=ds_dir,
            out_root=dst_root,
            out_w=args.out_w,
            out_h=args.out_h,
            keep_empty_label=args.keep_empty_label
        )

    print("[DONE] all datasets processed.")

if __name__ == "__main__":
    main()

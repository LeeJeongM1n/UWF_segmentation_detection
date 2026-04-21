import shutil
from pathlib import Path
import cv2
import pandas as pd


# -------------------------
# Paths
# -------------------------
xlsx_path = Path("/mnt/richul_FM/UWF_seg_det/datasets/OUWFD/Ground_Truth.xlsx")
src_img_dir = Path("/mnt/richul_FM/UWF_seg_det/datasets/OUWFD/images")
dst_dir = Path("/mnt/richul_FM/UWF_seg_det/datasets/Det/OUWFD/poor_quality_images")

dst_dir.mkdir(parents=True, exist_ok=True)

# resize target (W, H)
RESIZE_W = 780
RESIZE_H = 614

# -------------------------
# Load Excel
# -------------------------
df = pd.read_excel(xlsx_path)

# 필수 컬럼 체크
required_cols = {"Image ID ", "Overall quality"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in Excel: {missing}")


# -------------------------
# Filter poor quality rows
# -------------------------
poor_df = df[df["Overall quality"] == 0]

print(f"[INFO] Total rows: {len(df)}")
print(f"[INFO] Poor quality rows (Overall quality == 0): {len(poor_df)}")


# -------------------------
# Copy images
# -------------------------
copied = 0
missing_img = 0

for _, row in poor_df.iterrows():
    image_id = str(row["Image ID "])
    img_name = f"{image_id}"

    src_img_path = src_img_dir / img_name
    dst_img_path = dst_dir / img_name

    if not src_img_path.exists():
        print(f"[WARN] Image not found: {src_img_path}")
        missing_img += 1
        continue

    img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Failed to read image: {src_img_path}")
        read_fail += 1
        continue
    img_resized = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)

    # shutil.copy(src_img_path, dst_img_path)
    cv2.imwrite(str(dst_img_path), img_resized)
    copied += 1


# -------------------------
# Summary
# -------------------------
print("\n[RESULT]")
print(f"  Copied images : {copied}")
print(f"  Missing images: {missing_img}")
print(f"  Saved to      : {dst_dir}")

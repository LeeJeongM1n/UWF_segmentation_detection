import os
import random
import shutil

src_dir = "/mnt/richul_FM/UWF_seg_det/datasets/Det/inference_output/tta_OUWFD/tta_blob_spread2/tta/overlay"
dst_dir = "/mnt/richul_FM/UWF_seg_det/datasets/Det/inference_output/tta_OUWFD/tta_blob_spread2/tta/sample"
skip_dir = "/mnt/richul_FM/UWF_seg_det/datasets/Det/inference_output/tta_poorQuality/OUWFD/overlay"

NUM_SAMPLES = 10
VALID_EXT = (".png", ".jpg", ".jpeg")

os.makedirs(dst_dir, exist_ok=True)

# source 이미지 목록
src_files = [
    f for f in os.listdir(src_dir)
    if f.lower().endswith(VALID_EXT)
]

# skip 대상 파일명 set
skip_files = set(os.listdir(skip_dir))

random.shuffle(src_files)

selected = []

for fname in src_files:
    if fname in skip_files:
        continue

    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    shutil.copy(src_path, dst_path)
    selected.append(fname)

    if len(selected) >= NUM_SAMPLES:
        break

if len(selected) < NUM_SAMPLES:
    print(f"Warning: {NUM_SAMPLES}개를 채우지 못했습니다. (선택됨: {len(selected)})")
else:
    print(f"완료: {len(selected)}개 이미지 샘플링 완료")

print("선택된 파일 목록:")
for f in selected:
    print(f" - {f}")

#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Folder -> temporary CSV -> ROI segmentation + disc/fovea QC
#
# Output:
# - 전체 결과 CSV
# - 전체 요약 CSV
# - 처리 순서상 첫 N장 QC 결과 이미지
#
# uwf_inference.py는 기존 CSV 입력 방식을 그대로 사용합니다.
# 이 shell script가 IMAGE_DIR 내부의 이미지 목록을 자동으로 CSV로 만듭니다.
# ============================================================

# ------------------------------------------------------------
# 1. PATHS: 이 부분만 수정
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"

# ------------------------------------------------------------
# 입력 이미지 폴더
# ------------------------------------------------------------
# 이 폴더 내부의 이미지를 자동으로 찾아 inference합니다.
# 하위 폴더까지 재귀적으로 검색합니다.

# GitHub sample data를 사용하려면:
IMAGE_DIR="${REPO_DIR}/sample_data/sample10"

# 기존 uwf_inference.py가 요구하는 CSV column 이름
IMAGE_COLUMN="image_path"

# Shell에서 절대경로를 가진 임시 CSV를 생성하므로
# IMAGE_ROOT는 비워둡니다.
IMAGE_ROOT=""

# ------------------------------------------------------------
# Model checkpoints
# ------------------------------------------------------------

LANDMARK_CKPT="${REPO_DIR}/weights/uwf_landmark.pth"
ROI_CKPT="${REPO_DIR}/weights/uwf_roi_segmentation.ckpt"

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR="${REPO_DIR}/output/UWF_QC_ALL"

# shell이 자동 생성할 임시 CSV
TEMP_CSV="${OUTPUT_DIR}/inference_input.csv"

# ------------------------------------------------------------
# 2. SAMPLE
# ------------------------------------------------------------

# 0 = IMAGE_DIR에서 발견된 모든 이미지 처리
SAMPLE_N="0"

# SAMPLE_N > 0일 때 사용할 방식
SAMPLE_MODE="first"
RANDOM_SEED="42"

# 처리 순서상 첫 N장 QC 이미지 저장
# 0이면 QC 이미지를 저장하지 않음
QC_IMAGE_COUNT="10"
QC_IMAGE_DIR_NAME="qc_first10"

# ------------------------------------------------------------
# 3. MODEL SETTINGS
# ------------------------------------------------------------

CUDA_DEVICE="0"
IMG_SIZE="512"

# Landmark: SMP U-Net + ResNet50
LANDMARK_ARCHITECTURE="unet"
LANDMARK_ENCODER="resnet50"

LANDMARK_THRESHOLD="0.5"
LANDMARK_BLOB_THRESHOLD="0.5"

TTA_CONF_THRESHOLD="0.5"
TTA_SPREAD_THRESHOLD="15"

# ROI: SMP DeepLabV3Plus + ResNet50
ROI_ARCHITECTURE="deeplabv3plus"
ROI_ENCODER="resnet50"
ROI_ENCODER_CANDIDATES="resnet50"

ROI_THRESHOLD="0.5"

# ROI model이 1-channel이면 이 값은 무시됨
# 2-channel(background/foreground)이면 foreground channel
ROI_FOREGROUND_CHANNEL="1"

# 0: ROI TTA 사용 안 함
# 1: ROI에도 original/hflip/vflip 평균 적용
ROI_TTA="0"

# 1이면 기존 결과 덮어쓰기
OVERWRITE="1"

# 이미 conda activate 후 실행한다면 빈 문자열 유지
CONDA_ENV=""

# ------------------------------------------------------------
# 4. DO NOT EDIT BELOW
# ------------------------------------------------------------

PY_SCRIPT="${REPO_DIR}/inference/uwf_inference.py"

# ------------------------------------------------------------
# Conda
# ------------------------------------------------------------

if [[ -n "${CONDA_ENV}" ]]; then
    if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        echo "[ERROR] conda.sh not found."
        exit 1
    fi

    conda activate "${CONDA_ENV}"
fi

# ------------------------------------------------------------
# Path validation
# ------------------------------------------------------------

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "[ERROR] Repository not found: ${REPO_DIR}"
    exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "[ERROR] Python script not found: ${PY_SCRIPT}"
    exit 1
fi

if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "[ERROR] Image directory not found: ${IMAGE_DIR}"
    exit 1
fi

if [[ ! -f "${LANDMARK_CKPT}" ]]; then
    echo "[ERROR] Landmark checkpoint not found: ${LANDMARK_CKPT}"
    exit 1
fi

if [[ -n "${ROI_CKPT}" ]] && [[ ! -f "${ROI_CKPT}" ]]; then
    echo "[ERROR] ROI checkpoint not found: ${ROI_CKPT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------
# IMAGE_DIR -> temporary CSV
# ------------------------------------------------------------

echo "[INFO] Searching images in: ${IMAGE_DIR}"

# CSV header
printf '%s\n' "${IMAGE_COLUMN}" > "${TEMP_CSV}"

# 하위 폴더까지 재귀 검색
find "${IMAGE_DIR}" -type f \
    \( \
        -iname "*.jpg" \
        -o -iname "*.jpeg" \
        -o -iname "*.png" \
        -o -iname "*.bmp" \
        -o -iname "*.tif" \
        -o -iname "*.tiff" \
    \) \
    -print0 \
    | sort -z \
    | while IFS= read -r -d '' img_path; do
        printf '%s\n' "${img_path}" >> "${TEMP_CSV}"
    done

# CSV header 제외한 이미지 수
IMAGE_COUNT=$(( $(wc -l < "${TEMP_CSV}") - 1 ))

if [[ "${IMAGE_COUNT}" -le 0 ]]; then
    echo "[ERROR] No supported image files found in: ${IMAGE_DIR}"
    rm -f "${TEMP_CSV}"
    exit 1
fi

echo "[INFO] Found ${IMAGE_COUNT} image(s)."
echo "[INFO] Temporary CSV created: ${TEMP_CSV}"

# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

# ------------------------------------------------------------
# Build arguments
# ------------------------------------------------------------

ARGS=(
    python "${PY_SCRIPT}"

    --csv_path "${TEMP_CSV}"
    --image_column "${IMAGE_COLUMN}"

    --repo_dir "${REPO_DIR}"

    --landmark_ckpt "${LANDMARK_CKPT}"

    --output_dir "${OUTPUT_DIR}"

    --qc_image_count "${QC_IMAGE_COUNT}"
    --qc_image_dir_name "${QC_IMAGE_DIR_NAME}"

    --sample_n "${SAMPLE_N}"
    --sample_mode "${SAMPLE_MODE}"
    --random_seed "${RANDOM_SEED}"

    --img_size "${IMG_SIZE}"

    --landmark_architecture "${LANDMARK_ARCHITECTURE}"
    --landmark_encoder "${LANDMARK_ENCODER}"

    --landmark_threshold "${LANDMARK_THRESHOLD}"
    --landmark_blob_threshold "${LANDMARK_BLOB_THRESHOLD}"

    --tta_conf_threshold "${TTA_CONF_THRESHOLD}"
    --tta_spread_threshold "${TTA_SPREAD_THRESHOLD}"

    --roi_architecture "${ROI_ARCHITECTURE}"
    --roi_encoder "${ROI_ENCODER}"
    --roi_encoder_candidates "${ROI_ENCODER_CANDIDATES}"

    --roi_threshold "${ROI_THRESHOLD}"
    --roi_foreground_channel "${ROI_FOREGROUND_CHANNEL}"
)

# 현재는 TEMP_CSV에 절대경로를 기록하므로 보통 실행되지 않음
if [[ -n "${IMAGE_ROOT}" ]]; then
    ARGS+=(--image_root "${IMAGE_ROOT}")
fi

if [[ -n "${ROI_CKPT}" ]]; then
    ARGS+=(--roi_ckpt "${ROI_CKPT}")
fi

if [[ "${ROI_TTA}" == "1" ]]; then
    ARGS+=(--roi_tta)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

echo "============================================================"
echo "[UWF Folder Inference]"
echo "IMAGE_DIR       : ${IMAGE_DIR}"
echo "IMAGE_COUNT     : ${IMAGE_COUNT}"
echo "TEMP_CSV        : ${TEMP_CSV}"
echo "REPO_DIR        : ${REPO_DIR}"
echo "LANDMARK_CKPT   : ${LANDMARK_CKPT}"
echo "ROI_CKPT        : ${ROI_CKPT:-AUTO}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "SAMPLE_N        : ${SAMPLE_N} (0 = all images)"
echo "QC_IMAGE_COUNT  : ${QC_IMAGE_COUNT}"
echo "GPU             : ${CUDA_DEVICE}"
echo "============================================================"

# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

"${ARGS[@]}"
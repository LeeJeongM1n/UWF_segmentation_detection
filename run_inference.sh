#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CSV -> ROI segmentation + disc/fovea landmark QC
# Output:
# - 전체 결과 CSV
# - 전체 요약 CSV
# - 처리 순서상 첫 N장 QC 결과 이미지
# ============================================================

# ------------------------------------------------------------
# 1. PATHS: 이 부분만 수정
# ------------------------------------------------------------

# df_before2022를 저장한 CSV
CSV_PATH="/home/hero/Documents/2026_UWFCLIP/01_UWF_only/uwf_before2022.csv"
IMAGE_COLUMN="image_path"

# image_path가 상대경로일 때만 지정합니다.
# image_path가 절대경로면 빈 문자열로 두세요.
IMAGE_ROOT=""

REPO_DIR="/home/hero/Documents/2026_UWFCLIP/UWF_segmentation_detection"

# 현재 정상 작동한 landmark checkpoint
LANDMARK_CKPT="${REPO_DIR}/weights/uwf_landmark.pth"

# ROI checkpoint:
# - 정확한 파일명을 알면 여기에 직접 입력
# - 빈 문자열이면 weights/ 안에서 roi, seg, mask, retina 이름을 자동 탐색
ROI_CKPT="${REPO_DIR}/weights/uwf_roi_segmentation.ckpt"
# 예:
# ROI_CKPT="${REPO_DIR}/weights/roi_best_model.pth"

# 결과 CSV와 첫 10장 QC 이미지가 이 폴더에 생성됩니다.
OUTPUT_DIR="/home/hero/Documents/2026_UWFCLIP/UWF_QC_ALL"

# ------------------------------------------------------------
# 2. SAMPLE
# ------------------------------------------------------------

# 전체 CSV 실행
SAMPLE_N="0"
SAMPLE_MODE="first"        # SAMPLE_N=0일 때 원본 CSV 순서 유지
RANDOM_SEED="42"

# 처리 순서상 첫 10장만 결과 이미지를 저장합니다.
# 0으로 설정하면 QC 이미지도 전혀 저장하지 않습니다.
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

# ROI model이 1-channel이면 이 값은 무시됩니다.
# 2-channel(background/foreground)이면 foreground channel을 1로 사용
ROI_FOREGROUND_CHANNEL="1"

# 0: ROI TTA 사용 안 함
# 1: ROI에도 original/hflip/vflip 평균 적용
ROI_TTA="0"

# 1이면 기존 결과 CSV 덮어쓰기
OVERWRITE="1"

# Conda 환경 이름.
# 이미 환경을 activate한 뒤 실행한다면 빈 문자열로 둡니다.
CONDA_ENV=""

# ------------------------------------------------------------
# 4. DO NOT EDIT BELOW
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/uwf_qc_csv_only.py"

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

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "[ERROR] Python script not found: ${PY_SCRIPT}"
    exit 1
fi

if [[ ! -f "${CSV_PATH}" ]]; then
    echo "[ERROR] CSV not found: ${CSV_PATH}"
    exit 1
fi

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "[ERROR] Repository not found: ${REPO_DIR}"
    exit 1
fi

if [[ ! -f "${LANDMARK_CKPT}" ]]; then
    echo "[ERROR] Landmark checkpoint not found: ${LANDMARK_CKPT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

ARGS=(
    python "${PY_SCRIPT}"
    --csv_path "${CSV_PATH}"
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

echo "============================================================"
echo "[CSV-only UWF QC]"
echo "CSV_PATH       : ${CSV_PATH}"
echo "IMAGE_COLUMN   : ${IMAGE_COLUMN}"
echo "REPO_DIR       : ${REPO_DIR}"
echo "LANDMARK_CKPT  : ${LANDMARK_CKPT}"
echo "ROI_CKPT       : ${ROI_CKPT:-AUTO}"
echo "OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "SAMPLE_N       : ${SAMPLE_N} (0 = all rows)"
echo "QC_IMAGE_COUNT : ${QC_IMAGE_COUNT}"
echo "GPU            : ${CUDA_DEVICE}"
echo "============================================================"

"${ARGS[@]}"
